//! Hand-rolled OpenXR API-layer dispatch.
//!
//! - M1: negotiate, advance the API-layer chain, pass-through, banner.
//! - M3: capture the Vulkan graphics device (`xrCreateSession`), the eye
//!   swapchain `VkImage`s (`xrCreateSwapchain` + `xrEnumerateSwapchainImages`),
//!   and per-eye FOV/sub-rect (`xrEndFrame`); derive PPD + optical center via
//!   `ffr-core` and publish `FoveationDesc`s into the shared registry.

use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::{transmute, transmute_copy, zeroed};
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;

use ffr_core::wire::{FalloffParams, FoveationDesc, FFR_SHARED_MAGIC, FFR_SHARED_VERSION};
use once_cell::sync::Lazy;
use openxr_sys::{
    pfn, BaseInStructure, CompositionLayerProjection, CompositionLayerProjectionView, FrameEndInfo,
    GraphicsBindingVulkanKHR, Instance, InstanceCreateInfo, InstanceProperties, Result as XrResult,
    Session, SessionCreateInfo, StructureType, Swapchain, SwapchainCreateInfo,
    SwapchainImageBaseHeader, SwapchainImageVulkanKHR, CURRENT_API_VERSION,
};
use tracing::info;

use crate::xr_sys::*;

struct InstanceData {
    next_gipa: pfn::GetInstanceProcAddr,
    destroy_instance: Option<pfn::DestroyInstance>,
    create_session: Option<pfn::CreateSession>,
    destroy_session: Option<pfn::DestroySession>,
    create_swapchain: Option<pfn::CreateSwapchain>,
    destroy_swapchain: Option<pfn::DestroySwapchain>,
    enumerate_swapchain_images: Option<pfn::EnumerateSwapchainImages>,
    end_frame: Option<pfn::EndFrame>,
}

struct SessionData {
    instance: u64,
    vk_device: u64,
}

struct SwapchainData {
    instance: u64,
    vk_device: u64,
    images: Vec<u64>,
}

static INSTANCES: Lazy<RwLock<HashMap<u64, InstanceData>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static SESSIONS: Lazy<RwLock<HashMap<u64, SessionData>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static SWAPCHAINS: Lazy<RwLock<HashMap<u64, SwapchainData>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static GENERATION: AtomicU64 = AtomicU64::new(0);
static LOGGED_PUBLISH: AtomicBool = AtomicBool::new(false);
/// Monotonic heartbeat counter (M2 cross-dylib channel proof).
static HEARTBEAT: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Loader negotiation
// ---------------------------------------------------------------------------

/// Exported entry point named in the API-layer manifest.
///
/// # Safety
/// `loader_info` / `api_layer_request` must be null or valid pointers the
/// loader owns for the call (the standard loader contract).
#[no_mangle]
pub unsafe extern "system" fn xrNegotiateLoaderApiLayerInterface(
    loader_info: *const XrNegotiateLoaderInfo,
    _layer_name: *const c_char,
    api_layer_request: *mut XrNegotiateApiLayerRequest,
) -> XrResult {
    crate::logging::init();
    if loader_info.is_null() || api_layer_request.is_null() {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    }

    let li = &*loader_info;
    if li.struct_type != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO
        || XR_CURRENT_LOADER_API_LAYER_VERSION < li.min_interface_version
        || XR_CURRENT_LOADER_API_LAYER_VERSION > li.max_interface_version
    {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    }

    let req = &mut *api_layer_request;
    if req.struct_type != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    }
    req.layer_interface_version = XR_CURRENT_LOADER_API_LAYER_VERSION;
    req.layer_api_version = CURRENT_API_VERSION;
    req.get_instance_proc_addr = Some(xr_get_instance_proc_addr);
    req.create_api_layer_instance = Some(xr_create_api_layer_instance);

    info!(
        "FFR OpenXR layer ({} v{}) negotiated loader interface v{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        XR_CURRENT_LOADER_API_LAYER_VERSION
    );
    XrResult::SUCCESS
}

// ---------------------------------------------------------------------------
// Layer-instance creation
// ---------------------------------------------------------------------------

unsafe extern "system" fn xr_create_api_layer_instance(
    info: *const InstanceCreateInfo,
    api_layer_info: *const XrApiLayerCreateInfo,
    instance: *mut Instance,
) -> XrResult {
    crate::logging::init();
    if api_layer_info.is_null() {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    }
    let next_info = (*api_layer_info).next_info;
    if next_info.is_null() {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    }
    let Some(next_gipa) = (*next_info).next_get_instance_proc_addr else {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    };
    let Some(next_create) = (*next_info).next_create_api_layer_instance else {
        return XrResult::ERROR_INITIALIZATION_FAILED;
    };

    let mut copy = *api_layer_info;
    copy.next_info = (*next_info).next;

    let res = next_create(info, &copy, instance);
    if res != XrResult::SUCCESS {
        return res;
    }

    let inst = *instance;
    let data = InstanceData {
        next_gipa,
        destroy_instance: resolve(next_gipa, inst, c"xrDestroyInstance"),
        create_session: resolve(next_gipa, inst, c"xrCreateSession"),
        destroy_session: resolve(next_gipa, inst, c"xrDestroySession"),
        create_swapchain: resolve(next_gipa, inst, c"xrCreateSwapchain"),
        destroy_swapchain: resolve(next_gipa, inst, c"xrDestroySwapchain"),
        enumerate_swapchain_images: resolve(next_gipa, inst, c"xrEnumerateSwapchainImages"),
        end_frame: resolve(next_gipa, inst, c"xrEndFrame"),
    };
    let get_props = resolve::<pfn::GetInstanceProperties>(next_gipa, inst, c"xrGetInstanceProperties");
    INSTANCES.write().unwrap().insert(inst.into_raw(), data);

    info!("xrCreateInstance ok; runtime={}", runtime_name(get_props, inst));

    // M2 heartbeat (cross-dylib channel proof); M3 publishes real FoveationDescs
    // from xrEndFrame.
    let counter = HEARTBEAT.fetch_add(1, Ordering::Relaxed) + 1;
    ffr_registry::set_heartbeat(counter, 42.5);
    res
}

// ---------------------------------------------------------------------------
// proc-addr routing
// ---------------------------------------------------------------------------

unsafe extern "system" fn xr_get_instance_proc_addr(
    instance: Instance,
    name: *const c_char,
    function: *mut Option<pfn::VoidFunction>,
) -> XrResult {
    if name.is_null() || function.is_null() {
        return XrResult::ERROR_VALIDATION_FAILURE;
    }
    macro_rules! set_fn {
        ($f:expr) => {{
            *function = Some(transmute::<*const (), pfn::VoidFunction>($f as *const ()));
            return XrResult::SUCCESS;
        }};
    }
    match CStr::from_ptr(name).to_bytes() {
        b"xrDestroyInstance" => set_fn!(xr_destroy_instance),
        b"xrCreateSession" => set_fn!(xr_create_session),
        b"xrDestroySession" => set_fn!(xr_destroy_session),
        b"xrCreateSwapchain" => set_fn!(xr_create_swapchain),
        b"xrDestroySwapchain" => set_fn!(xr_destroy_swapchain),
        b"xrEnumerateSwapchainImages" => set_fn!(xr_enumerate_swapchain_images),
        b"xrEndFrame" => set_fn!(xr_end_frame),
        _ => {}
    }

    let next = INSTANCES.read().unwrap().get(&instance.into_raw()).map(|d| d.next_gipa);
    match next {
        Some(gipa) => gipa(instance, name, function),
        None => {
            *function = None;
            XrResult::ERROR_FUNCTION_UNSUPPORTED
        }
    }
}

unsafe extern "system" fn xr_destroy_instance(instance: Instance) -> XrResult {
    let raw = instance.into_raw();
    // Drop any sessions/swapchains belonging to this instance.
    SWAPCHAINS.write().unwrap().retain(|_, s| s.instance != raw);
    SESSIONS.write().unwrap().retain(|_, s| s.instance != raw);
    let data = INSTANCES.write().unwrap().remove(&raw);
    if let Some(d) = data {
        if let Some(destroy) = d.destroy_instance {
            return destroy(instance);
        }
    }
    XrResult::SUCCESS
}

// ---------------------------------------------------------------------------
// M3 hooks
// ---------------------------------------------------------------------------

unsafe extern "system" fn xr_create_session(
    instance: Instance,
    create_info: *const SessionCreateInfo,
    session: *mut Session,
) -> XrResult {
    let down = INSTANCES.read().unwrap().get(&instance.into_raw()).and_then(|d| d.create_session);
    let Some(down) = down else {
        return XrResult::ERROR_HANDLE_INVALID;
    };
    let res = down(instance, create_info, session);
    if res != XrResult::SUCCESS {
        return res;
    }
    let vk_device = if create_info.is_null() {
        0
    } else {
        find_vk_device((*create_info).next)
    };
    SESSIONS.write().unwrap().insert(
        (*session).into_raw(),
        SessionData { instance: instance.into_raw(), vk_device },
    );
    info!("xrCreateSession ok; vk_device=0x{vk_device:x}");
    res
}

unsafe extern "system" fn xr_destroy_session(session: Session) -> XrResult {
    let raw = session.into_raw();
    let instance = SESSIONS.write().unwrap().remove(&raw).map(|s| s.instance);
    let down = instance
        .and_then(|i| INSTANCES.read().unwrap().get(&i).and_then(|d| d.destroy_session));
    match down {
        Some(d) => d(session),
        None => XrResult::SUCCESS,
    }
}

unsafe extern "system" fn xr_create_swapchain(
    session: Session,
    create_info: *const SwapchainCreateInfo,
    swapchain: *mut Swapchain,
) -> XrResult {
    let (instance_raw, vk_device) = SESSIONS
        .read()
        .unwrap()
        .get(&session.into_raw())
        .map(|s| (s.instance, s.vk_device))
        .unwrap_or((0, 0));
    let down = INSTANCES.read().unwrap().get(&instance_raw).and_then(|d| d.create_swapchain);
    let Some(down) = down else {
        return XrResult::ERROR_HANDLE_INVALID;
    };
    let res = down(session, create_info, swapchain);
    if res != XrResult::SUCCESS {
        return res;
    }
    SWAPCHAINS.write().unwrap().insert(
        (*swapchain).into_raw(),
        SwapchainData { instance: instance_raw, vk_device, images: Vec::new() },
    );
    res
}

unsafe extern "system" fn xr_destroy_swapchain(swapchain: Swapchain) -> XrResult {
    let raw = swapchain.into_raw();
    let removed = SWAPCHAINS.write().unwrap().remove(&raw);
    let mut down = None;
    if let Some(s) = removed {
        for img in &s.images {
            ffr_registry::remove_image(s.vk_device, *img);
        }
        down = INSTANCES.read().unwrap().get(&s.instance).and_then(|d| d.destroy_swapchain);
    }
    match down {
        Some(d) => d(swapchain),
        None => XrResult::SUCCESS,
    }
}

unsafe extern "system" fn xr_enumerate_swapchain_images(
    swapchain: Swapchain,
    capacity: u32,
    count_output: *mut u32,
    images: *mut SwapchainImageBaseHeader,
) -> XrResult {
    let instance_raw =
        SWAPCHAINS.read().unwrap().get(&swapchain.into_raw()).map(|s| s.instance).unwrap_or(0);
    let down =
        INSTANCES.read().unwrap().get(&instance_raw).and_then(|d| d.enumerate_swapchain_images);
    let Some(down) = down else {
        return XrResult::ERROR_HANDLE_INVALID;
    };
    let res = down(swapchain, capacity, count_output, images);
    if res == XrResult::SUCCESS && capacity > 0 && !images.is_null() && !count_output.is_null() {
        let n = *count_output as isize;
        if n > 0 && (*images).ty == StructureType::SWAPCHAIN_IMAGE_VULKAN_KHR {
            let arr = images as *const SwapchainImageVulkanKHR;
            let mut vk_images = Vec::with_capacity(n as usize);
            for i in 0..n {
                vk_images.push((*arr.offset(i)).image);
            }
            if let Some(s) = SWAPCHAINS.write().unwrap().get_mut(&swapchain.into_raw()) {
                s.images = vk_images;
            }
        }
    }
    res
}

unsafe extern "system" fn xr_end_frame(
    session: Session,
    frame_end_info: *const FrameEndInfo,
) -> XrResult {
    if !frame_end_info.is_null() {
        publish_from_frame(&*frame_end_info);
    }
    let down = SESSIONS
        .read()
        .unwrap()
        .get(&session.into_raw())
        .and_then(|s| INSTANCES.read().unwrap().get(&s.instance).and_then(|d| d.end_frame));
    match down {
        Some(d) => d(session, frame_end_info),
        None => XrResult::ERROR_HANDLE_INVALID,
    }
}

// ---------------------------------------------------------------------------
// Publishing
// ---------------------------------------------------------------------------

unsafe fn publish_from_frame(info: &FrameEndInfo) {
    if info.layers.is_null() {
        return;
    }
    for li in 0..info.layer_count as isize {
        let base = *info.layers.offset(li);
        if base.is_null() {
            continue;
        }
        if (*base).ty != StructureType::COMPOSITION_LAYER_PROJECTION {
            continue;
        }
        let proj = base as *const CompositionLayerProjection;
        for vi in 0..(*proj).view_count as isize {
            publish_view(vi as u32, &*(*proj).views.offset(vi));
        }
    }
}

unsafe fn publish_view(eye: u32, view: &CompositionLayerProjectionView) {
    let sub = view.sub_image;
    let (vk_device, images) = match SWAPCHAINS.read().unwrap().get(&sub.swapchain.into_raw()) {
        Some(s) => (s.vk_device, s.images.clone()),
        None => return,
    };
    if vk_device == 0 || images.is_empty() {
        return;
    }

    let rect = sub.image_rect;
    let rect_x = rect.offset.x;
    let rect_y = rect.offset.y;
    let rect_w = rect.extent.width.max(1) as u32;
    let rect_h = rect.extent.height.max(1) as u32;

    let f = view.fov;
    let fov = ffr_core::math::Fov::new(f.angle_left, f.angle_right, f.angle_up, f.angle_down);
    let center = fov.optical_center();
    let ppd = fov.ppd(rect_w, rect_h);
    let center_px_x = rect_x as f32 + center.u * rect_w as f32;
    let center_px_y = rect_y as f32 + center.v * rect_h as f32;

    if !LOGGED_PUBLISH.swap(true, Ordering::Relaxed) {
        info!(
            "publishing foveation: eye {eye} {rect_w}x{rect_h} \
             optical-center=({center_px_x:.0},{center_px_y:.0}) [u={:.3},v={:.3}] \
             ppd_center={:.1}h/{:.1}v across {} image(s) on vk_device=0x{vk_device:x}",
            center.u,
            center.v,
            ppd.center_h,
            ppd.center_v,
            images.len()
        );
    }

    let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
    for image in images {
        let desc = FoveationDesc {
            magic: FFR_SHARED_MAGIC,
            version: FFR_SHARED_VERSION,
            vk_device,
            vk_image: image,
            image_array_index: sub.image_array_index,
            eye,
            rect_x,
            rect_y,
            rect_w,
            rect_h,
            center_px_x,
            center_px_y,
            ppd_center_h: ppd.center_h,
            ppd_center_v: ppd.center_v,
            falloff: FalloffParams::default(),
            generation,
        };
        ffr_registry::publish(&desc);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the Vulkan `VkDevice` handle in a session-create `next` chain.
unsafe fn find_vk_device(mut next: *const c_void) -> u64 {
    while !next.is_null() {
        let base = next as *const BaseInStructure;
        if (*base).ty == StructureType::GRAPHICS_BINDING_VULKAN_KHR {
            let b = next as *const GraphicsBindingVulkanKHR;
            return (*b).device as u64;
        }
        next = (*base).next as *const c_void;
    }
    0
}

/// Resolve a down-chain function by name into a concrete pfn type.
unsafe fn resolve<T: Copy>(
    gipa: pfn::GetInstanceProcAddr,
    inst: Instance,
    name: &CStr,
) -> Option<T> {
    let mut f: Option<pfn::VoidFunction> = None;
    if gipa(inst, name.as_ptr(), &mut f) == XrResult::SUCCESS {
        f.map(|vf| transmute_copy::<pfn::VoidFunction, T>(&vf))
    } else {
        None
    }
}

unsafe fn runtime_name(get_props: Option<pfn::GetInstanceProperties>, inst: Instance) -> String {
    let Some(get) = get_props else {
        return "<unknown>".to_string();
    };
    let mut props: InstanceProperties = zeroed();
    props.ty = StructureType::INSTANCE_PROPERTIES;
    if get(inst, &mut props) == XrResult::SUCCESS {
        CStr::from_ptr(props.runtime_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    } else {
        "<unknown>".to_string()
    }
}
