//! Hand-rolled Vulkan layer dispatch.
//!
//! M1 scope: negotiate the loader interface, advance the instance/device
//! dispatch chains, forward every call straight through, and log a banner at
//! `vkCreateInstance` / `vkCreateDevice`. The interception points for VRS
//! (device feature enable, pipeline recreation, `vkCmdBeginRendering`) are added
//! to this same dispatch in later milestones.

use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::{transmute, transmute_copy};
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};

use ash::vk::{self, Handle};
use ffr_core::rate::ShadingRate;
use once_cell::sync::Lazy;
use tracing::{info, warn};

use crate::vk_sys::*;

const FSR_EXT_NAME: &CStr = c"VK_KHR_fragment_shading_rate";
/// Device extensions `VK_KHR_fragment_shading_rate` depends on, with the core
/// version each was promoted to. We must add any not-yet-core dependency to the
/// enabled list when we enable FSR, or the loader/driver rejects the device (and
/// our v1→v2 render-pass promotion calls a null `vkCreateRenderPass2`). The
/// renderpass2 dependency is what Vulkan-1.0/1.1 apps (xrgears) are missing.
const FSR_DEP_EXTS: &[(&CStr, u32)] = &[
    (c"VK_KHR_create_renderpass2", VK_API_VERSION_1_2),
    (c"VK_KHR_multiview", VK_API_VERSION_1_1),
    (c"VK_KHR_maintenance2", VK_API_VERSION_1_1),
];
const VK_API_VERSION_1_1: u32 = vk::make_api_version(0, 1, 1, 0);
const VK_API_VERSION_1_2: u32 = vk::make_api_version(0, 1, 2, 0);

/// Kill switch: when `FFR_VRS_DISABLE` is set the layer stays in the call chain
/// but injects nothing (for A/B comparison and emergency bypass).
static KILL: Lazy<bool> = Lazy::new(|| std::env::var_os("FFR_VRS_DISABLE").is_some());

/// When `FFR_VRS_DEBUG` is set, dump each eye's shading-rate map as a PPM image
/// (false-colored by rate) so the foveation pattern can be inspected.
static DEBUG: Lazy<bool> = Lazy::new(|| std::env::var_os("FFR_VRS_DEBUG").is_some());

/// When `FFR_VRS_OVERLAY` is set, the layer draws a translucent false-color of
/// the applied shading rate over each eye render, so foveation is visible in any
/// app without modifying it.
static OVERLAY: Lazy<bool> = Lazy::new(|| std::env::var_os("FFR_VRS_OVERLAY").is_some());

const FULLSCREEN_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));
const RATE_OVERLAY_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rate_overlay.frag.spv"));

/// Down-chain instance state, keyed by `VkInstance` handle.
struct InstanceData {
    next_gipa: PfnGetInstanceProcAddr,
    destroy_instance: vk::PFN_vkDestroyInstance,
    enumerate_physical_devices: vk::PFN_vkEnumeratePhysicalDevices,
    get_physical_device_properties: vk::PFN_vkGetPhysicalDeviceProperties,
    /// ash wrapper over the down-chain instance functions (for VRS cap queries).
    instance: ash::Instance,
    /// The API version the app requested (`VkApplicationInfo.apiVersion`, or 1.0).
    app_api: u32,
}

/// Fragment-shading-rate capabilities resolved at device creation.
#[derive(Clone, Copy)]
struct VrsCaps {
    /// Attachment texel size (framebuffer pixels per shading-rate-map texel).
    texel_size: vk::Extent2D,
}

/// A lazily-created shading-rate attachment image holding one eye's radial
/// foveation map. Cached per eye; rebuilt if the eye's render area / optical
/// center / falloff changes.
struct VrsImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// The render-area parameters this map was built for (change detection).
    area_w: u32,
    area_h: u32,
    center_x: f32,
    center_y: f32,
    falloff: ffr_core::wire::FalloffParams,
}

impl VrsImage {
    fn matches(&self, area_w: u32, area_h: u32, cx: f32, cy: f32, f: &ffr_core::wire::FalloffParams) -> bool {
        self.area_w == area_w
            && self.area_h == area_h
            && (self.center_x - cx).abs() < 1.0
            && (self.center_y - cy).abs() < 1.0
            && self.falloff == *f
    }
}

/// Down-chain device state, keyed by `VkDevice` handle.
struct DeviceData {
    next_gdpa: PfnGetDeviceProcAddr,
    /// ash wrapper over the down-chain device functions (for our own GPU work).
    device: ash::Device,
    /// `vkCreateRenderPass2` entry point, resolved under the right name for the
    /// device's version: the core symbol on Vulkan ≥1.2, else the
    /// `VK_KHR_create_renderpass2` alias `vkCreateRenderPass2KHR`. ash's wrapper
    /// only knows the core symbol, which 1.0/1.1 devices don't expose.
    create_rp2: Option<vk::PFN_vkCreateRenderPass2>,
    /// `Some` if attachment-based VRS is available + enabled on this device.
    vrs: Option<VrsCaps>,
    /// A graphics queue family the app created (for our one-shot init submits).
    queue_family: u32,
    /// Device memory properties (for allocating the shading-rate image).
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Per-eye shading-rate images, created on first injection for each eye.
    vrs_images: Mutex<HashMap<u32, VrsImage>>,
    /// Cached overlay pipeline (debug viz), keyed by color format + extent.
    overlay: Mutex<Option<OverlayPipeline>>,
}

/// The layer's own full-screen shading-rate overlay pipeline.
struct OverlayPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    color_format: i32,
    area_w: u32,
    area_h: u32,
}

static INSTANCES: Lazy<RwLock<HashMap<u64, InstanceData>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static PHYS_TO_INSTANCE: Lazy<RwLock<HashMap<u64, u64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static DEVICES: Lazy<RwLock<HashMap<u64, DeviceData>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
/// `VkImageView` handle → its underlying `VkImage` handle.
static VIEW_TO_IMAGE: Lazy<RwLock<HashMap<u64, u64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
/// `VkCommandBuffer` handle → the `VkDevice` it was allocated from.
static CB_TO_DEVICE: Lazy<RwLock<HashMap<u64, u64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Per-command-buffer state for the eye render currently being recorded, used
/// to draw the overlay at `vkCmdEndRendering`.
struct RenderState {
    color_image: u64,
    color_view: u64,
    color_format: i32,
    render_area: vk::Rect2D,
    sr_view: u64,
    texel: vk::Extent2D,
    multiview: bool,
}
static CB_RENDER: Lazy<RwLock<HashMap<u64, RenderState>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static LOGGED_RECOGNIZE: AtomicBool = AtomicBool::new(false);
static LOGGED_INJECT: AtomicBool = AtomicBool::new(false);
static LOGGED_OVERLAY: AtomicBool = AtomicBool::new(false);
static LOGGED_LEGACY: AtomicBool = AtomicBool::new(false);

/// A render pass we augmented with a fragment-shading-rate attachment (legacy
/// / non-dynamic-rendering path), keyed by the returned `VkRenderPass`.
#[derive(Clone, Copy)]
struct LegacyRp {
    texel: vk::Extent2D,
    /// Array layers the shading-rate attachment needs (1, or #views for multiview).
    layers: u32,
}
static LEGACY_RP: Lazy<RwLock<HashMap<u64, LegacyRp>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// A framebuffer of an augmented render pass: carries its own shading-rate
/// image (filled lazily at begin-render-pass from the eye image's descriptor).
struct LegacyFb {
    color_image: u64,
    sr_image: vk::Image,
    sr_memory: vk::DeviceMemory,
    sr_view: vk::ImageView,
    cols: u32,
    rows: u32,
    area: vk::Extent2D,
    texel: vk::Extent2D,
    /// Array layers (1, or #views for multiview — one foveation map per eye).
    layers: u32,
    // The foveation the map was last filled from (refill only when it changes —
    // NOT every frame, since the descriptor generation bumps each frame).
    filled: bool,
    fill_center_x: f32,
    fill_center_y: f32,
    fill_falloff: ffr_core::wire::FalloffParams,
}
static LEGACY_FB: Lazy<Mutex<HashMap<u64, LegacyFb>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Cast a layer entry-point fn item into the loader's `PFN_vkVoidFunction`.
macro_rules! vk_fn {
    ($f:expr) => {
        Some(unsafe { transmute::<*const (), unsafe extern "system" fn()>($f as *const ()) })
    };
}

/// Reinterpret a `PFN_vkVoidFunction` returned by a down-chain proc-addr as a
/// concrete function-pointer type. All Vulkan function pointers (and the
/// niche-optimized `Option<fn()>`) are pointer-sized, so this is layout-safe.
/// The caller must ensure `f` is `Some` before calling the result.
#[inline]
unsafe fn pfn<T: Copy>(f: vk::PFN_vkVoidFunction) -> T {
    transmute_copy::<vk::PFN_vkVoidFunction, T>(&f)
}

// ---------------------------------------------------------------------------
// Loader negotiation
// ---------------------------------------------------------------------------

/// Exported entry point the loader calls first (named in the layer manifest).
///
/// # Safety
/// `p` must be null or point to a valid `VkNegotiateLayerInterface` the loader
/// owns for the duration of the call (the standard loader contract).
#[no_mangle]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p: *mut VkNegotiateLayerInterface,
) -> vk::Result {
    crate::logging::init();
    if !p.is_null() {
        let iface = &mut *p;
        // Speak at most version 2; clamp to what the loader offers.
        iface.loader_layer_interface_version = iface.loader_layer_interface_version.min(2);
        iface.pfn_get_instance_proc_addr = Some(get_instance_proc_addr);
        iface.pfn_get_device_proc_addr = Some(get_device_proc_addr);
        iface.pfn_get_physical_device_proc_addr = None;
        info!(
            "FFR Vulkan layer ({} v{}) negotiated loader interface v{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            iface.loader_layer_interface_version
        );
    }
    vk::Result::SUCCESS
}

// ---------------------------------------------------------------------------
// proc-addr routing
// ---------------------------------------------------------------------------

pub unsafe extern "system" fn get_instance_proc_addr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    crate::logging::init();
    if p_name.is_null() {
        return None;
    }
    match CStr::from_ptr(p_name).to_bytes() {
        b"vkGetInstanceProcAddr" => return vk_fn!(get_instance_proc_addr),
        b"vkCreateInstance" => return vk_fn!(create_instance),
        b"vkDestroyInstance" => return vk_fn!(destroy_instance),
        b"vkEnumeratePhysicalDevices" => return vk_fn!(enumerate_physical_devices),
        b"vkCreateDevice" => return vk_fn!(create_device),
        b"vkGetDeviceProcAddr" => return vk_fn!(get_device_proc_addr),
        _ => {}
    }
    // Not ours — forward to the next layer's instance proc-addr.
    let next = INSTANCES.read().unwrap().get(&instance.as_raw()).map(|d| d.next_gipa);
    match next {
        Some(next_gipa) => next_gipa(instance, p_name),
        None => None,
    }
}

pub unsafe extern "system" fn get_device_proc_addr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }
    match CStr::from_ptr(p_name).to_bytes() {
        b"vkGetDeviceProcAddr" => return vk_fn!(get_device_proc_addr),
        b"vkDestroyDevice" => return vk_fn!(destroy_device),
        b"vkCreateImageView" => return vk_fn!(create_image_view),
        b"vkDestroyImageView" => return vk_fn!(destroy_image_view),
        b"vkAllocateCommandBuffers" => return vk_fn!(allocate_command_buffers),
        b"vkCmdBeginRendering" => return vk_fn!(cmd_begin_rendering),
        b"vkCmdEndRendering" => return vk_fn!(cmd_end_rendering),
        b"vkCreateGraphicsPipelines" => return vk_fn!(create_graphics_pipelines),
        b"vkCreateRenderPass" => return vk_fn!(create_render_pass),
        b"vkCreateRenderPass2" => return vk_fn!(create_render_pass2),
        b"vkDestroyRenderPass" => return vk_fn!(destroy_render_pass),
        b"vkCreateFramebuffer" => return vk_fn!(create_framebuffer),
        b"vkDestroyFramebuffer" => return vk_fn!(destroy_framebuffer),
        b"vkCmdBeginRenderPass" => return vk_fn!(cmd_begin_render_pass),
        b"vkCmdBeginRenderPass2" => return vk_fn!(cmd_begin_render_pass2),
        _ => {}
    }
    let next = DEVICES.read().unwrap().get(&device.as_raw()).map(|d| d.next_gdpa);
    match next {
        Some(next_gdpa) => next_gdpa(device, p_name),
        None => None,
    }
}

// ---------------------------------------------------------------------------
// Instance lifecycle
// ---------------------------------------------------------------------------

unsafe extern "system" fn create_instance(
    p_create_info: *const vk::InstanceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    crate::logging::init();

    let layer_ci = find_instance_chain_info(p_create_info);
    if layer_ci.is_null() {
        warn!("vkCreateInstance: no layer chain info found; cannot dispatch");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let link = (*layer_ci).u.p_layer_info;
    let next_gipa = (*link).pfn_next_get_instance_proc_addr;
    // Advance the chain so the next layer consumes its own link.
    (*layer_ci).u.p_layer_info = (*link).p_next;

    let next_create = next_gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr());
    if next_create.is_none() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let create: vk::PFN_vkCreateInstance = pfn(next_create);
    let res = create(p_create_info, p_allocator, p_instance);
    if res != vk::Result::SUCCESS {
        return res;
    }

    let instance = *p_instance;
    let load = |name: &CStr| next_gipa(instance, name.as_ptr());
    // ash wrapper whose functions resolve through the *next* layer's GIPA, so
    // our own calls go straight down the chain (below us) to the driver.
    let static_fn = ash::StaticFn { get_instance_proc_addr: next_gipa };
    let ash_instance = ash::Instance::load(&static_fn, instance);
    let data = InstanceData {
        next_gipa,
        destroy_instance: pfn(load(c"vkDestroyInstance")),
        enumerate_physical_devices: pfn(load(c"vkEnumeratePhysicalDevices")),
        get_physical_device_properties: pfn(load(c"vkGetPhysicalDeviceProperties")),
        instance: ash_instance,
        app_api: app_api_version_raw(p_create_info),
    };
    INSTANCES.write().unwrap().insert(instance.as_raw(), data);

    info!(
        "vkCreateInstance ok: app={:?} api={}",
        app_name(p_create_info),
        app_api_version(p_create_info)
    );

    // M2: read back the heartbeat the OpenXR layer published, proving both
    // cdylibs share the one libffr_shared.so state.
    match ffr_registry::get_heartbeat() {
        Some((counter, ppd)) => info!(
            "read heartbeat #{counter} (ppd={ppd}) from ffr-shared — cross-layer channel OK"
        ),
        None => {
            if ffr_registry::is_available() {
                info!("ffr-shared loaded but no heartbeat yet (OpenXR layer not run in this process)");
            } else {
                warn!("ffr-shared not found; cross-layer channel unavailable");
            }
        }
    }
    res
}

unsafe extern "system" fn destroy_instance(
    instance: vk::Instance,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let data = INSTANCES.write().unwrap().remove(&instance.as_raw());
    PHYS_TO_INSTANCE
        .write()
        .unwrap()
        .retain(|_, inst| *inst != instance.as_raw());
    if let Some(d) = data {
        (d.destroy_instance)(instance, p_allocator);
    }
}

unsafe extern "system" fn enumerate_physical_devices(
    instance: vk::Instance,
    p_count: *mut u32,
    p_phys: *mut vk::PhysicalDevice,
) -> vk::Result {
    let down = INSTANCES
        .read()
        .unwrap()
        .get(&instance.as_raw())
        .map(|d| d.enumerate_physical_devices);
    let Some(down) = down else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let res = down(instance, p_count, p_phys);
    if !p_phys.is_null() && (res == vk::Result::SUCCESS || res == vk::Result::INCOMPLETE) {
        let n = *p_count as usize;
        let mut map = PHYS_TO_INSTANCE.write().unwrap();
        for i in 0..n {
            let pd = *p_phys.add(i);
            map.insert(pd.as_raw(), instance.as_raw());
        }
    }
    res
}

// ---------------------------------------------------------------------------
// Device lifecycle
// ---------------------------------------------------------------------------

unsafe extern "system" fn create_device(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_device: *mut vk::Device,
) -> vk::Result {
    let layer_ci = find_device_chain_info(p_create_info);
    if layer_ci.is_null() {
        warn!("vkCreateDevice: no layer chain info found; cannot dispatch");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let link = (*layer_ci).u.p_layer_info;
    let next_gipa = (*link).pfn_next_get_instance_proc_addr;
    let next_gdpa = (*link).pfn_next_get_device_proc_addr;
    (*layer_ci).u.p_layer_info = (*link).p_next;

    let instance_raw = PHYS_TO_INSTANCE
        .read()
        .unwrap()
        .get(&physical_device.as_raw())
        .copied();
    let instance = instance_raw
        .map(vk::Instance::from_raw)
        .unwrap_or(vk::Instance::null());

    // Decide whether to enable attachment-based VRS on this device.
    let caps = instance_raw.and_then(|ir| query_vrs(ir, physical_device));
    let already_enabled = device_has_extension(p_create_info, FSR_EXT_NAME);
    let do_enable = caps.is_some() && !already_enabled;
    let device_api = effective_device_api(instance_raw, physical_device);

    // Build a (possibly) modified create info that adds the extension + feature.
    // These locals must outlive the down-call below.
    let mut ext_ptrs: Vec<*const c_char> = Vec::new();
    let mut fsr_feature = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default();
    let mut modified = *p_create_info;
    if do_enable {
        let existing = std::slice::from_raw_parts(
            (*p_create_info).pp_enabled_extension_names,
            (*p_create_info).enabled_extension_count as usize,
        );
        ext_ptrs.extend_from_slice(existing);
        ext_ptrs.push(FSR_EXT_NAME.as_ptr());
        // Add any FSR dependency that is not yet core for this device's effective
        // API version (min of the app's instance version and the GPU's), and that
        // the app hasn't already enabled. Without this, sub-1.2 apps crash when we
        // promote their render passes to renderpass2.
        for (ext, core_since) in FSR_DEP_EXTS {
            if device_api < *core_since && !device_has_extension(p_create_info, ext) {
                ext_ptrs.push(ext.as_ptr());
            }
        }
        modified.pp_enabled_extension_names = ext_ptrs.as_ptr();
        modified.enabled_extension_count = ext_ptrs.len() as u32;

        fsr_feature.attachment_fragment_shading_rate = vk::TRUE;
        fsr_feature.p_next = (*p_create_info).p_next as *mut c_void;
        modified.p_next = (&fsr_feature as *const vk::PhysicalDeviceFragmentShadingRateFeaturesKHR)
            .cast::<c_void>();
    }
    let ci_ptr: *const vk::DeviceCreateInfo = if do_enable { &modified } else { p_create_info };

    let next_create = next_gipa(instance, c"vkCreateDevice".as_ptr());
    if next_create.is_none() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let create: vk::PFN_vkCreateDevice = pfn(next_create);
    let res = create(physical_device, ci_ptr, p_allocator, p_device);
    if res != vk::Result::SUCCESS {
        return res;
    }

    let device = *p_device;
    let queue_family = first_queue_family(p_create_info);
    // ash device whose functions resolve through the next layer's device
    // proc-addr (so our own calls go straight down to the driver).
    let gdpa_ptr = next_gdpa as *const c_void;
    let inst_fn = ash::InstanceFnV1_0::load(|name| {
        if name.to_bytes() == c"vkGetDeviceProcAddr".to_bytes() {
            gdpa_ptr
        } else {
            ptr::null()
        }
    });
    let ash_device = ash::Device::load(&inst_fn, device);

    // Resolve vkCreateRenderPass2 under the name this device version exposes it:
    // core on ≥1.2, else the KHR alias we enabled above.
    let rp2_name = if device_api >= VK_API_VERSION_1_2 {
        c"vkCreateRenderPass2"
    } else {
        c"vkCreateRenderPass2KHR"
    };
    let rp2_fp = next_gdpa(device, rp2_name.as_ptr());
    let create_rp2: Option<vk::PFN_vkCreateRenderPass2> =
        rp2_fp.map(|_| pfn::<vk::PFN_vkCreateRenderPass2>(rp2_fp));

    let mem_props = {
        let map = INSTANCES.read().unwrap();
        match instance_raw.and_then(|ir| map.get(&ir)) {
            Some(inst) => inst.instance.get_physical_device_memory_properties(physical_device),
            None => vk::PhysicalDeviceMemoryProperties::default(),
        }
    };

    DEVICES.write().unwrap().insert(
        device.as_raw(),
        DeviceData {
            next_gdpa,
            device: ash_device,
            create_rp2,
            vrs: caps,
            queue_family,
            mem_props,
            vrs_images: Mutex::new(HashMap::new()),
            overlay: Mutex::new(None),
        },
    );

    match caps {
        Some(c) => info!(
            "vkCreateDevice ok on GPU {}: attachment VRS {} (texel {}x{})",
            gpu_name(instance_raw, physical_device),
            if do_enable { "enabled by layer" } else { "already enabled by app" },
            c.texel_size.width,
            c.texel_size.height
        ),
        None => info!(
            "vkCreateDevice ok on GPU {}: VRS unavailable; passing through",
            gpu_name(instance_raw, physical_device)
        ),
    }
    res
}

unsafe extern "system" fn destroy_device(
    device: vk::Device,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let data = DEVICES.write().unwrap().remove(&device.as_raw());
    if let Some(d) = data {
        let images: Vec<VrsImage> =
            d.vrs_images.lock().unwrap().drain().map(|(_, v)| v).collect();
        for img in &images {
            destroy_vrs_image(&d, img);
        }
        if let Some(o) = d.overlay.lock().unwrap().take() {
            d.device.destroy_pipeline(o.pipeline, None);
            d.device.destroy_pipeline_layout(o.layout, None);
        }
        // Down-chain destroy via the ash wrapper (honors the app's allocator).
        d.device.destroy_device(p_allocator.as_ref());
    }
}

/// Query attachment-VRS support for a physical device and return the chosen
/// attachment texel size, or `None` if unsupported.
unsafe fn query_vrs(instance_raw: u64, phys: vk::PhysicalDevice) -> Option<VrsCaps> {
    let map = INSTANCES.read().unwrap();
    let inst = &map.get(&instance_raw)?.instance;

    let exts = inst.enumerate_device_extension_properties(phys).ok()?;
    let has_ext = exts.iter().any(|e| {
        CStr::from_ptr(e.extension_name.as_ptr()).to_bytes() == FSR_EXT_NAME.to_bytes()
    });
    if !has_ext {
        return None;
    }

    let mut feature = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut feature);
    inst.get_physical_device_features2(phys, &mut features2);
    if feature.attachment_fragment_shading_rate != vk::TRUE {
        return None;
    }

    let mut props = vk::PhysicalDeviceFragmentShadingRatePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
    inst.get_physical_device_properties2(phys, &mut props2);

    let texel_size = props.max_fragment_shading_rate_attachment_texel_size;
    if texel_size.width == 0 || texel_size.height == 0 {
        return None;
    }
    Some(VrsCaps { texel_size })
}

/// Whether a `VkDeviceCreateInfo` already enables the named extension.
unsafe fn device_has_extension(p_ci: *const vk::DeviceCreateInfo, name: &CStr) -> bool {
    let names = std::slice::from_raw_parts(
        (*p_ci).pp_enabled_extension_names,
        (*p_ci).enabled_extension_count as usize,
    );
    names
        .iter()
        .any(|&p| !p.is_null() && CStr::from_ptr(p).to_bytes() == name.to_bytes())
}

/// The first queue family index requested in a device create info.
unsafe fn first_queue_family(p_ci: *const vk::DeviceCreateInfo) -> u32 {
    if (*p_ci).queue_create_info_count == 0 || (*p_ci).p_queue_create_infos.is_null() {
        return 0;
    }
    (*(*p_ci).p_queue_create_infos).queue_family_index
}

// ---------------------------------------------------------------------------
// M4b: image identification
// ---------------------------------------------------------------------------

unsafe extern "system" fn create_image_view(
    device: vk::Device,
    p_create_info: *const vk::ImageViewCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_view: *mut vk::ImageView,
) -> vk::Result {
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device.as_raw()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    match d.device.create_image_view(&*p_create_info, p_allocator.as_ref()) {
        Ok(view) => {
            *p_view = view;
            VIEW_TO_IMAGE
                .write()
                .unwrap()
                .insert(view.as_raw(), (*p_create_info).image.as_raw());
            vk::Result::SUCCESS
        }
        Err(e) => e,
    }
}

unsafe extern "system" fn destroy_image_view(
    device: vk::Device,
    view: vk::ImageView,
    p_allocator: *const vk::AllocationCallbacks,
) {
    VIEW_TO_IMAGE.write().unwrap().remove(&view.as_raw());
    if let Some(d) = DEVICES.read().unwrap().get(&device.as_raw()) {
        d.device.destroy_image_view(view, p_allocator.as_ref());
    }
}

unsafe extern "system" fn allocate_command_buffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo,
    p_command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device.as_raw()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    match d.device.allocate_command_buffers(&*p_allocate_info) {
        Ok(cbs) => {
            let mut cb_map = CB_TO_DEVICE.write().unwrap();
            for (i, cb) in cbs.iter().enumerate() {
                *p_command_buffers.add(i) = *cb;
                cb_map.insert(cb.as_raw(), device.as_raw());
            }
            vk::Result::SUCCESS
        }
        Err(e) => e,
    }
}

unsafe extern "system" fn cmd_begin_rendering(
    command_buffer: vk::CommandBuffer,
    p_rendering_info: *const vk::RenderingInfo,
) {
    let device_raw = CB_TO_DEVICE.read().unwrap().get(&command_buffer.as_raw()).copied();
    let Some(device_raw) = device_raw else {
        return;
    };
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device_raw) else {
        return;
    };

    let matched = if p_rendering_info.is_null() {
        None
    } else {
        matched_desc(device_raw, p_rendering_info)
    };

    if let Some((desc, color_view)) = matched {
        if d.vrs.is_some() && !*KILL {
            let area = (*p_rendering_info).render_area;
            if let Some((sr_view, texel)) = ensure_vrs_image(d, &desc, area) {
                let info = &*p_rendering_info;
                let mut attach = vk::RenderingFragmentShadingRateAttachmentInfoKHR::default()
                    .image_view(sr_view)
                    .image_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
                    .shading_rate_attachment_texel_size(texel);
                attach.p_next = info.p_next as *mut c_void;
                let mut modified = *info;
                modified.p_next =
                    (&attach as *const vk::RenderingFragmentShadingRateAttachmentInfoKHR)
                        .cast::<c_void>();
                d.device.cmd_begin_rendering(command_buffer, &modified);

                if *OVERLAY {
                    CB_RENDER.write().unwrap().insert(
                        command_buffer.as_raw(),
                        RenderState {
                            color_image: desc.vk_image,
                            color_view: color_view.as_raw(),
                            color_format: desc.vk_format,
                            render_area: area,
                            sr_view: sr_view.as_raw(),
                            texel,
                            multiview: info.view_mask != 0,
                        },
                    );
                }
                return;
            }
        }
    }

    d.device.cmd_begin_rendering(command_buffer, &*p_rendering_info);
}

/// The first OXR-tagged eye descriptor (and its color view) among this render's
/// color attachments.
unsafe fn matched_desc(
    device_raw: u64,
    p_info: *const vk::RenderingInfo,
) -> Option<(ffr_core::wire::FoveationDesc, vk::ImageView)> {
    let info = &*p_info;
    if info.p_color_attachments.is_null() {
        return None;
    }
    let attachments =
        std::slice::from_raw_parts(info.p_color_attachments, info.color_attachment_count as usize);
    for att in attachments {
        if att.image_view.is_null() {
            continue;
        }
        let image = VIEW_TO_IMAGE.read().unwrap().get(&att.image_view.as_raw()).copied();
        let Some(image) = image else { continue };
        let descs = ffr_registry::lookup(device_raw, image);
        if let Some(d) = descs.first() {
            if !LOGGED_RECOGNIZE.swap(true, Ordering::Relaxed) {
                info!(
                    "recognized eye image 0x{image:x}: eye {} {}x{} optical-center=({:.0},{:.0})",
                    d.eye, d.rect_w, d.rect_h, d.center_px_x, d.center_px_y
                );
            }
            return Some((*d, att.image_view));
        }
    }
    None
}

unsafe extern "system" fn cmd_end_rendering(command_buffer: vk::CommandBuffer) {
    let device_raw = CB_TO_DEVICE.read().unwrap().get(&command_buffer.as_raw()).copied();
    let Some(device_raw) = device_raw else {
        return;
    };
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device_raw) else {
        return;
    };
    // Forward the app's end-rendering first.
    d.device.cmd_end_rendering(command_buffer);

    // Then, if this was a tagged eye render and the overlay is enabled, draw it
    // in a separate pass that loads the rendered image and blends on top.
    let state = CB_RENDER.write().unwrap().remove(&command_buffer.as_raw());
    if let Some(state) = state {
        if *OVERLAY && !state.multiview {
            draw_overlay(d, command_buffer, &state);
        }
    }
}

/// Draw the shading-rate overlay into the eye image (own render pass).
unsafe fn draw_overlay(d: &DeviceData, cb: vk::CommandBuffer, state: &RenderState) {
    let Some(pipeline) = ensure_overlay_pipeline(d, state.color_format, state.render_area) else {
        return;
    };

    // Wait for the app's color writes before we load + blend.
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let b = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(vk::Image::from_raw(state.color_image))
        .subresource_range(range);
    d.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[b],
    );

    let color_att = vk::RenderingAttachmentInfo::default()
        .image_view(vk::ImageView::from_raw(state.color_view))
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE);
    let mut sr_att = vk::RenderingFragmentShadingRateAttachmentInfoKHR::default()
        .image_view(vk::ImageView::from_raw(state.sr_view))
        .image_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
        .shading_rate_attachment_texel_size(state.texel);
    let color_atts = [color_att];
    let mut rendering = vk::RenderingInfo::default()
        .render_area(state.render_area)
        .layer_count(1)
        .color_attachments(&color_atts);
    rendering.p_next =
        (&mut sr_att as *mut vk::RenderingFragmentShadingRateAttachmentInfoKHR).cast::<c_void>();

    d.device.cmd_begin_rendering(cb, &rendering);
    d.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
    d.device.cmd_draw(cb, 3, 1, 0, 0);
    d.device.cmd_end_rendering(cb);

    if !LOGGED_OVERLAY.swap(true, Ordering::Relaxed) {
        info!("drawing shading-rate overlay on eye render ({}x{})", state.render_area.extent.width, state.render_area.extent.height);
    }
}

/// Get/create the overlay pipeline for a given color format + extent.
unsafe fn ensure_overlay_pipeline(
    d: &DeviceData,
    color_format: i32,
    area: vk::Rect2D,
) -> Option<vk::Pipeline> {
    let mut slot = d.overlay.lock().unwrap();
    if let Some(o) = slot.as_ref() {
        if o.color_format == color_format
            && o.area_w == area.extent.width
            && o.area_h == area.extent.height
        {
            return Some(o.pipeline);
        }
        d.device.destroy_pipeline(o.pipeline, None);
        d.device.destroy_pipeline_layout(o.layout, None);
        *slot = None;
    }

    let o = build_overlay_pipeline(d, color_format, area)?;
    let pipeline = o.pipeline;
    *slot = Some(o);
    Some(pipeline)
}

unsafe fn build_overlay_pipeline(
    d: &DeviceData,
    color_format: i32,
    area: vk::Rect2D,
) -> Option<OverlayPipeline> {
    let dev = &d.device;
    let layout = dev
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        .ok()?;
    let vert = load_module(dev, FULLSCREEN_VERT)?;
    let frag = load_module(dev, RATE_OVERLAY_FRAG)?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // Static viewport/scissor so we never touch the app's dynamic state.
    let viewports = [vk::Viewport {
        x: area.offset.x as f32,
        y: area.offset.y as f32,
        width: area.extent.width as f32,
        height: area.extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [area];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
    let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
    let color_formats = [vk::Format::from_raw(color_format)];
    let mut rendering =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
    let mut fsr = vk::PipelineFragmentShadingRateStateCreateInfoKHR::default()
        .fragment_size(vk::Extent2D { width: 1, height: 1 })
        .combiner_ops([
            vk::FragmentShadingRateCombinerOpKHR::KEEP,
            vk::FragmentShadingRateCombinerOpKHR::REPLACE,
        ]);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(layout)
        .push_next(&mut rendering)
        .push_next(&mut fsr);

    let pipeline = dev
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        .ok()
        .and_then(|p| p.into_iter().next());
    dev.destroy_shader_module(vert, None);
    dev.destroy_shader_module(frag, None);
    let pipeline = match pipeline {
        Some(p) => p,
        None => {
            dev.destroy_pipeline_layout(layout, None);
            return None;
        }
    };

    Some(OverlayPipeline {
        pipeline,
        layout,
        color_format,
        area_w: area.extent.width,
        area_h: area.extent.height,
    })
}

unsafe fn load_module(dev: &ash::Device, spv: &[u8]) -> Option<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).ok()?;
    dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).ok()
}

// ---------------------------------------------------------------------------
// M7: legacy render-pass injection (classic VkRenderPass, non-multiview)
// ---------------------------------------------------------------------------

const FSR_LAYOUT: vk::ImageLayout = vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;

/// Call `vkCreateRenderPass2` via the entry point resolved for this device's
/// version (core or KHR alias), falling back to ash's core wrapper if we somehow
/// have no resolved pointer (only reachable on ≥1.2 devices).
unsafe fn device_create_rp2(
    d: &DeviceData,
    ci: &vk::RenderPassCreateInfo2,
    p_allocator: *const vk::AllocationCallbacks,
) -> Result<vk::RenderPass, vk::Result> {
    if let Some(f) = d.create_rp2 {
        let mut rp = vk::RenderPass::null();
        let r = f(d.device.handle(), ci, p_allocator, &mut rp);
        return if r == vk::Result::SUCCESS { Ok(rp) } else { Err(r) };
    }
    d.device.create_render_pass2(ci, p_allocator.as_ref())
}

/// Augment a render pass with a fragment-shading-rate attachment so legacy
/// (non-dynamic-rendering) apps get foveation. Returns the augmented handle to
/// the app transparently.
unsafe extern "system" fn create_render_pass2(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo2,
    p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device.as_raw()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let ci = &*p_create_info;
    let layers = layers_from_masks(
        std::slice::from_raw_parts(ci.p_subpasses, ci.subpass_count as usize)
            .iter()
            .map(|s| s.view_mask),
    );
    let augment = d.vrs.is_some()
        && !*KILL
        && ci.attachment_count >= 1
        && has_color_subpass(ci);
    if !augment {
        return match device_create_rp2(d, ci, p_allocator) {
            Ok(rp) => {
                *p_render_pass = rp;
                vk::Result::SUCCESS
            }
            Err(e) => e,
        };
    }

    let texel = d.vrs.unwrap().texel_size;
    let fsr_index = ci.attachment_count;
    let orig_atts = std::slice::from_raw_parts(ci.p_attachments, ci.attachment_count as usize);
    let mut atts: Vec<vk::AttachmentDescription2> = orig_atts.to_vec();
    atts.push(
        vk::AttachmentDescription2::default()
            .format(vk::Format::R8_UINT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(FSR_LAYOUT)
            .final_layout(FSR_LAYOUT),
    );

    let orig_subs = std::slice::from_raw_parts(ci.p_subpasses, ci.subpass_count as usize);
    let n = orig_subs.len();
    let mut fsr_refs: Vec<vk::AttachmentReference2> = Vec::with_capacity(n);
    for _ in 0..n {
        fsr_refs.push(
            vk::AttachmentReference2::default()
                .attachment(fsr_index)
                .layout(FSR_LAYOUT)
                .aspect_mask(vk::ImageAspectFlags::COLOR),
        );
    }
    let mut fsr_infos: Vec<vk::FragmentShadingRateAttachmentInfoKHR> = Vec::with_capacity(n);
    for (i, sub) in orig_subs.iter().enumerate() {
        let mut info = vk::FragmentShadingRateAttachmentInfoKHR::default()
            .shading_rate_attachment_texel_size(texel);
        info.p_fragment_shading_rate_attachment = &fsr_refs[i];
        info.p_next = sub.p_next; // prepend ours before any existing chain
        fsr_infos.push(info);
    }
    let mut subs: Vec<vk::SubpassDescription2> = Vec::with_capacity(n);
    for (i, sub) in orig_subs.iter().enumerate() {
        let mut s = *sub;
        s.p_next = (&fsr_infos[i] as *const vk::FragmentShadingRateAttachmentInfoKHR).cast();
        subs.push(s);
    }

    let mut aug = *ci;
    aug.attachment_count = atts.len() as u32;
    aug.p_attachments = atts.as_ptr();
    aug.subpass_count = subs.len() as u32;
    aug.p_subpasses = subs.as_ptr();

    let rp = match device_create_rp2(d, &aug, p_allocator) {
        Ok(rp) => rp,
        Err(e) => return e,
    };
    *p_render_pass = rp;
    LEGACY_RP.write().unwrap().insert(rp.as_raw(), LegacyRp { texel, layers });
    tracing::debug!(
        "augmented legacy render pass (FSR attachment at index {fsr_index}, {layers} layer(s))"
    );
    vk::Result::SUCCESS
}

unsafe extern "system" fn destroy_render_pass(
    device: vk::Device,
    render_pass: vk::RenderPass,
    p_allocator: *const vk::AllocationCallbacks,
) {
    LEGACY_RP.write().unwrap().remove(&render_pass.as_raw());
    if let Some(d) = DEVICES.read().unwrap().get(&device.as_raw()) {
        d.device.destroy_render_pass(render_pass, p_allocator.as_ref());
    }
}

unsafe extern "system" fn create_framebuffer(
    device: vk::Device,
    p_create_info: *const vk::FramebufferCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_framebuffer: *mut vk::Framebuffer,
) -> vk::Result {
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device.as_raw()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let ci = &*p_create_info;
    let passthrough = || match d.device.create_framebuffer(ci, p_allocator.as_ref()) {
        Ok(fb) => {
            *p_framebuffer = fb;
            vk::Result::SUCCESS
        }
        Err(e) => e,
    };

    let rp = LEGACY_RP.read().unwrap().get(&ci.render_pass.as_raw()).copied();
    let Some(rp) = rp else { return passthrough() };
    if ci.flags.contains(vk::FramebufferCreateFlags::IMAGELESS) {
        return passthrough(); // imageless framebuffers unsupported here
    }

    let atts = std::slice::from_raw_parts(ci.p_attachments, ci.attachment_count as usize);
    let color_view = atts.first().copied().unwrap_or(vk::ImageView::null());
    let color_image = VIEW_TO_IMAGE.read().unwrap().get(&color_view.as_raw()).copied().unwrap_or(0);
    let cols = ci.width.div_ceil(rp.texel.width.max(1));
    let rows = ci.height.div_ceil(rp.texel.height.max(1));
    let Some((sr_image, sr_memory, sr_view)) = alloc_sr_image(d, cols, rows, rp.layers) else {
        return passthrough();
    };

    let mut aug_atts: Vec<vk::ImageView> = atts.to_vec();
    aug_atts.push(sr_view);
    let mut aug = *ci;
    aug.attachment_count = aug_atts.len() as u32;
    aug.p_attachments = aug_atts.as_ptr();
    let fb = match d.device.create_framebuffer(&aug, p_allocator.as_ref()) {
        Ok(fb) => fb,
        Err(e) => {
            d.device.destroy_image_view(sr_view, None);
            d.device.destroy_image(sr_image, None);
            d.device.free_memory(sr_memory, None);
            return e;
        }
    };
    *p_framebuffer = fb;
    LEGACY_FB.lock().unwrap().insert(
        fb.as_raw(),
        LegacyFb {
            color_image,
            sr_image,
            sr_memory,
            sr_view,
            cols,
            rows,
            area: vk::Extent2D { width: ci.width, height: ci.height },
            texel: rp.texel,
            layers: rp.layers,
            filled: false,
            fill_center_x: 0.0,
            fill_center_y: 0.0,
            fill_falloff: ffr_core::wire::FalloffParams::default(),
        },
    );
    vk::Result::SUCCESS
}

unsafe extern "system" fn destroy_framebuffer(
    device: vk::Device,
    framebuffer: vk::Framebuffer,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let fb = LEGACY_FB.lock().unwrap().remove(&framebuffer.as_raw());
    if let Some(d) = DEVICES.read().unwrap().get(&device.as_raw()) {
        if let Some(fb) = fb {
            d.device.destroy_image_view(fb.sr_view, None);
            d.device.destroy_image(fb.sr_image, None);
            d.device.free_memory(fb.sr_memory, None);
        }
        d.device.destroy_framebuffer(framebuffer, p_allocator.as_ref());
    }
}

unsafe extern "system" fn cmd_begin_render_pass2(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo,
    p_subpass_begin_info: *const vk::SubpassBeginInfo,
) {
    let device_raw = CB_TO_DEVICE.read().unwrap().get(&command_buffer.as_raw()).copied();
    let Some(device_raw) = device_raw else {
        return;
    };
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device_raw) else {
        return;
    };
    let fb_raw = (*p_render_pass_begin).framebuffer.as_raw();
    ensure_legacy_fill(d, device_raw, fb_raw);
    d.device.cmd_begin_render_pass2(command_buffer, &*p_render_pass_begin, &*p_subpass_begin_info);
}

/// Fill an augmented framebuffer's shading-rate image from the eye image's
/// latest descriptor (or a uniform 1x1 if none yet). One-time per generation.
unsafe fn ensure_legacy_fill(d: &DeviceData, device_raw: u64, fb_raw: u64) {
    let mut fbs = LEGACY_FB.lock().unwrap();
    let Some(fb) = fbs.get_mut(&fb_raw) else {
        return;
    };
    let descs = ffr_registry::lookup(device_raw, fb.color_image);
    let best = descs.iter().copied().max_by_key(|x| x.generation);
    // No-descriptor uses a sentinel center (-1) so a real one always differs.
    // Change detection keys on the freshest descriptor: in multiview both eyes'
    // descriptors bump generation together, so they refill in lockstep.
    let (cx, cy, falloff) = match &best {
        Some(d) => (d.center_px_x, d.center_px_y, d.falloff),
        None => (-1.0, -1.0, ffr_core::wire::FalloffParams::default()),
    };
    if fb.filled
        && (fb.fill_center_x - cx).abs() < 1.0
        && (fb.fill_center_y - cy).abs() < 1.0
        && fb.fill_falloff == falloff
    {
        return;
    }

    // Read the Copy fields we need before the per-layer upload loop borrows `d`.
    let (layers, area, texel, cols, rows) =
        (fb.layers, fb.area, fb.texel, fb.cols, fb.rows);
    let mut all_ok = true;
    for layer in 0..layers {
        // Per-view: the descriptor whose array index matches this layer; fall
        // back to the freshest one (e.g. single-descriptor double-wide images).
        let chosen = descs
            .iter()
            .find(|x| x.image_array_index == layer)
            .copied()
            .or(best);
        let map = match &chosen {
            Some(desc) => ffr_core::foveation::FoveationMap::generate(
                area.width,
                area.height,
                (desc.center_px_x, desc.center_px_y),
                texel.width.max(1),
                (area.width as f32 / 2.0, area.height as f32 / 2.0),
                &desc.falloff,
            ),
            None => ffr_core::foveation::FoveationMap {
                cols,
                rows,
                texel_size: texel.width,
                bytes: vec![0u8; (cols * rows) as usize],
            },
        };
        if upload_map(d, fb.sr_image, &map, layer).is_none() {
            all_ok = false;
        }
    }
    if all_ok {
        fb.filled = true;
        fb.fill_center_x = cx;
        fb.fill_center_y = cy;
        fb.fill_falloff = falloff;
        if best.is_some() && !LOGGED_LEGACY.swap(true, Ordering::Relaxed) {
            info!(
                "legacy render-pass VRS active: filled shading-rate map for eye image \
                 0x{:x} ({}x{} tiles, {} layer(s))",
                fb.color_image, fb.cols, fb.rows, layers
            );
        }
    }
}

unsafe fn has_color_subpass(ci: &vk::RenderPassCreateInfo2) -> bool {
    if ci.p_subpasses.is_null() {
        return false;
    }
    std::slice::from_raw_parts(ci.p_subpasses, ci.subpass_count as usize)
        .iter()
        .any(|s| s.color_attachment_count > 0)
}

/// Allocate an (unfilled) R8_UINT shading-rate image + view.
unsafe fn alloc_sr_image(
    d: &DeviceData,
    cols: u32,
    rows: u32,
    layers: u32,
) -> Option<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let dev = &d.device;
    let image = dev
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8_UINT)
                .extent(vk::Extent3D { width: cols, height: rows, depth: 1 })
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .ok()?;
    let req = dev.get_image_memory_requirements(image);
    let memory = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(
                memory_type(&d.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL),
            ),
            None,
        )
        .ok()?;
    dev.bind_image_memory(image, memory, 0).ok()?;
    let view_type = if layers > 1 {
        vk::ImageViewType::TYPE_2D_ARRAY
    } else {
        vk::ImageViewType::TYPE_2D
    };
    let view = dev
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(view_type)
                .format(vk::Format::R8_UINT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: layers,
                }),
            None,
        )
        .ok()?;
    Some((image, memory, view))
}

/// Number of array layers a shading-rate attachment needs to cover all views in
/// a multiview render pass: the highest set view-mask bit + 1 (1 if no multiview).
fn layers_from_masks(masks: impl Iterator<Item = u32>) -> u32 {
    let mut layers = 1u32;
    for m in masks {
        if m != 0 {
            layers = layers.max(32 - m.leading_zeros());
        }
    }
    layers
}

/// The per-subpass view masks, dependency view offsets, and correlation masks of
/// a v1 render pass that opts into multiview via `VkRenderPassMultiviewCreateInfo`.
/// Returned as owned vectors so they outlive the borrowed create-info chain.
struct MultiviewV1 {
    view_masks: Vec<u32>,
    view_offsets: Vec<i32>,
    correlation_masks: Vec<u32>,
}

unsafe fn find_multiview_v1(ci: &vk::RenderPassCreateInfo) -> Option<MultiviewV1> {
    let mut next = ci.p_next;
    while !next.is_null() {
        let base = next as *const vk::BaseInStructure;
        if (*base).s_type == vk::StructureType::RENDER_PASS_MULTIVIEW_CREATE_INFO {
            let mv = &*(next as *const vk::RenderPassMultiviewCreateInfo);
            let masks = slice_or_empty(mv.p_view_masks, mv.subpass_count);
            if !masks.iter().any(|&m| m != 0) {
                return None; // present but inert
            }
            return Some(MultiviewV1 {
                view_masks: masks.to_vec(),
                view_offsets: slice_or_empty(mv.p_view_offsets, mv.dependency_count).to_vec(),
                correlation_masks: slice_or_empty(mv.p_correlation_masks, mv.correlation_mask_count)
                    .to_vec(),
            });
        }
        next = (*base).p_next as *const c_void;
    }
    None
}

/// Slice from a pointer/count, treating a null pointer as empty.
unsafe fn slice_or_empty<'a, T>(ptr: *const T, count: u32) -> &'a [T] {
    if ptr.is_null() || count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, count as usize)
    }
}

/// The image aspect to use for an attachment reference of the given format.
fn aspect_for(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// `vkCreateRenderPass` (v1) — promote to renderpass2 and augment with the
/// shading-rate attachment (needed for Vulkan 1.0 apps like xrgears).
unsafe extern "system" fn create_render_pass(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device.as_raw()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let ci = &*p_create_info;
    let v1_atts = std::slice::from_raw_parts(ci.p_attachments, ci.attachment_count as usize);
    let v1_subs = std::slice::from_raw_parts(ci.p_subpasses, ci.subpass_count as usize);
    let mv = find_multiview_v1(ci);
    let layers = layers_from_masks(mv.iter().flat_map(|m| m.view_masks.iter().copied()));
    let augment = d.vrs.is_some()
        && !*KILL
        && ci.attachment_count >= 1
        && v1_subs.iter().any(|s| s.color_attachment_count > 0);
    if !augment {
        return match d.device.create_render_pass(ci, p_allocator.as_ref()) {
            Ok(rp) => {
                *p_render_pass = rp;
                vk::Result::SUCCESS
            }
            Err(e) => e,
        };
    }
    let texel = d.vrs.unwrap().texel_size;

    // Attachments v1 -> v2, plus the appended FSR attachment.
    let mut atts2: Vec<vk::AttachmentDescription2> = v1_atts
        .iter()
        .map(|a| {
            vk::AttachmentDescription2::default()
                .flags(a.flags)
                .format(a.format)
                .samples(a.samples)
                .load_op(a.load_op)
                .store_op(a.store_op)
                .stencil_load_op(a.stencil_load_op)
                .stencil_store_op(a.stencil_store_op)
                .initial_layout(a.initial_layout)
                .final_layout(a.final_layout)
        })
        .collect();
    let fsr_index = atts2.len() as u32;
    atts2.push(
        vk::AttachmentDescription2::default()
            .format(vk::Format::R8_UINT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(FSR_LAYOUT)
            .final_layout(FSR_LAYOUT),
    );

    let n = v1_subs.len();
    let translate = |r: &vk::AttachmentReference| {
        let aspect = if r.attachment == vk::ATTACHMENT_UNUSED {
            vk::ImageAspectFlags::COLOR
        } else {
            aspect_for(v1_atts[r.attachment as usize].format)
        };
        vk::AttachmentReference2::default()
            .attachment(r.attachment)
            .layout(r.layout)
            .aspect_mask(aspect)
    };
    // Per-subpass reference storage (heap-stable across the outer Vec growth).
    let mut input_refs: Vec<Vec<vk::AttachmentReference2>> = Vec::with_capacity(n);
    let mut color_refs: Vec<Vec<vk::AttachmentReference2>> = Vec::with_capacity(n);
    let mut resolve_refs: Vec<Vec<vk::AttachmentReference2>> = Vec::with_capacity(n);
    let mut depth_refs: Vec<vk::AttachmentReference2> = Vec::with_capacity(n);
    let mut has_depth: Vec<bool> = Vec::with_capacity(n);
    let mut fsr_refs: Vec<vk::AttachmentReference2> = Vec::with_capacity(n);
    for sub in v1_subs {
        let inputs = std::slice::from_raw_parts(
            sub.p_input_attachments,
            sub.input_attachment_count as usize,
        )
        .iter()
        .map(&translate)
        .collect();
        let colors = std::slice::from_raw_parts(
            sub.p_color_attachments,
            sub.color_attachment_count as usize,
        )
        .iter()
        .map(&translate)
        .collect();
        let resolves: Vec<vk::AttachmentReference2> = if sub.p_resolve_attachments.is_null() {
            Vec::new()
        } else {
            std::slice::from_raw_parts(sub.p_resolve_attachments, sub.color_attachment_count as usize)
                .iter()
                .map(&translate)
                .collect()
        };
        input_refs.push(inputs);
        color_refs.push(colors);
        resolve_refs.push(resolves);
        if sub.p_depth_stencil_attachment.is_null() {
            depth_refs.push(vk::AttachmentReference2::default());
            has_depth.push(false);
        } else {
            depth_refs.push(translate(&*sub.p_depth_stencil_attachment));
            has_depth.push(true);
        }
        fsr_refs.push(
            vk::AttachmentReference2::default()
                .attachment(fsr_index)
                .layout(FSR_LAYOUT)
                .aspect_mask(vk::ImageAspectFlags::COLOR),
        );
    }
    let mut fsr_infos: Vec<vk::FragmentShadingRateAttachmentInfoKHR> = Vec::with_capacity(n);
    for fsr_ref in &fsr_refs {
        let mut info = vk::FragmentShadingRateAttachmentInfoKHR::default()
            .shading_rate_attachment_texel_size(texel);
        info.p_fragment_shading_rate_attachment = fsr_ref;
        fsr_infos.push(info);
    }
    let mut subs2: Vec<vk::SubpassDescription2> = Vec::with_capacity(n);
    for (i, sub) in v1_subs.iter().enumerate() {
        let view_mask = mv.as_ref().and_then(|m| m.view_masks.get(i).copied()).unwrap_or(0);
        let mut s = vk::SubpassDescription2::default()
            .flags(sub.flags)
            .pipeline_bind_point(sub.pipeline_bind_point)
            .view_mask(view_mask)
            .input_attachments(&input_refs[i])
            .color_attachments(&color_refs[i]);
        if !resolve_refs[i].is_empty() {
            s = s.resolve_attachments(&resolve_refs[i]);
        }
        if has_depth[i] {
            s = s.depth_stencil_attachment(&depth_refs[i]);
        }
        if sub.preserve_attachment_count > 0 {
            s = s.preserve_attachments(std::slice::from_raw_parts(
                sub.p_preserve_attachments,
                sub.preserve_attachment_count as usize,
            ));
        }
        s.p_next = (&fsr_infos[i] as *const vk::FragmentShadingRateAttachmentInfoKHR).cast();
        subs2.push(s);
    }
    let v1_deps =
        std::slice::from_raw_parts(ci.p_dependencies, ci.dependency_count as usize);
    let deps2: Vec<vk::SubpassDependency2> = v1_deps
        .iter()
        .enumerate()
        .map(|(i, dep)| {
            // Multiview view offsets travel as a parallel array on the multiview
            // struct (one per dependency), not on the v1 dependency itself.
            let view_offset =
                mv.as_ref().and_then(|m| m.view_offsets.get(i).copied()).unwrap_or(0);
            vk::SubpassDependency2::default()
                .src_subpass(dep.src_subpass)
                .dst_subpass(dep.dst_subpass)
                .src_stage_mask(dep.src_stage_mask)
                .dst_stage_mask(dep.dst_stage_mask)
                .src_access_mask(dep.src_access_mask)
                .dst_access_mask(dep.dst_access_mask)
                .dependency_flags(dep.dependency_flags)
                .view_offset(view_offset)
        })
        .collect();

    let mut info2 = vk::RenderPassCreateInfo2::default()
        .attachments(&atts2)
        .subpasses(&subs2)
        .dependencies(&deps2);
    if let Some(m) = &mv {
        info2 = info2.correlated_view_masks(&m.correlation_masks);
    }
    info2.flags = ci.flags;
    let rp = match device_create_rp2(d, &info2, p_allocator) {
        Ok(rp) => rp,
        Err(e) => return e,
    };
    *p_render_pass = rp;
    LEGACY_RP.write().unwrap().insert(rp.as_raw(), LegacyRp { texel, layers });
    tracing::debug!("promoted + augmented legacy v1 render pass ({layers} layer(s))");
    vk::Result::SUCCESS
}

unsafe extern "system" fn cmd_begin_render_pass(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo,
    contents: vk::SubpassContents,
) {
    let device_raw = CB_TO_DEVICE.read().unwrap().get(&command_buffer.as_raw()).copied();
    let Some(device_raw) = device_raw else {
        return;
    };
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device_raw) else {
        return;
    };
    ensure_legacy_fill(d, device_raw, (*p_render_pass_begin).framebuffer.as_raw());
    d.device.cmd_begin_render_pass(command_buffer, &*p_render_pass_begin, contents);
}

/// Recreate graphics pipelines with a fragment-shading-rate state that REPLACEs
/// the pipeline rate with the attachment rate, so they consume the SR map.
unsafe extern "system" fn create_graphics_pipelines(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::GraphicsPipelineCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    let map = DEVICES.read().unwrap();
    let Some(d) = map.get(&device.as_raw()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };

    let n = create_info_count as usize;
    let infos = std::slice::from_raw_parts(p_create_infos, n);

    let result = if d.vrs.is_some() && !*KILL {
        // Reserve so the Vec never reallocates (we store pointers into it).
        let mut fsr_states: Vec<vk::PipelineFragmentShadingRateStateCreateInfoKHR> =
            Vec::with_capacity(n);
        for info in infos {
            let mut fsr = vk::PipelineFragmentShadingRateStateCreateInfoKHR::default()
                .fragment_size(vk::Extent2D { width: 1, height: 1 })
                .combiner_ops([
                    vk::FragmentShadingRateCombinerOpKHR::KEEP,
                    vk::FragmentShadingRateCombinerOpKHR::REPLACE,
                ]);
            fsr.p_next = info.p_next;
            fsr_states.push(fsr);
        }
        let mut modified: Vec<vk::GraphicsPipelineCreateInfo> = Vec::with_capacity(n);
        for (i, info) in infos.iter().enumerate() {
            let mut mi = *info;
            mi.p_next = (&fsr_states[i]
                as *const vk::PipelineFragmentShadingRateStateCreateInfoKHR)
                .cast::<c_void>();
            modified.push(mi);
        }
        d.device
            .create_graphics_pipelines(pipeline_cache, &modified, p_allocator.as_ref())
    } else {
        d.device
            .create_graphics_pipelines(pipeline_cache, infos, p_allocator.as_ref())
    };

    match result {
        Ok(pipes) => {
            for (i, p) in pipes.iter().enumerate() {
                *p_pipelines.add(i) = *p;
            }
            vk::Result::SUCCESS
        }
        Err((pipes, e)) => {
            for (i, p) in pipes.iter().enumerate() {
                *p_pipelines.add(i) = *p;
            }
            e
        }
    }
}

/// Ensure the eye's shading-rate image exists and matches its current
/// foveation, returning its view + the attachment texel size.
unsafe fn ensure_vrs_image(
    d: &DeviceData,
    desc: &ffr_core::wire::FoveationDesc,
    area: vk::Rect2D,
) -> Option<(vk::ImageView, vk::Extent2D)> {
    let caps = d.vrs?;
    let texel = caps.texel_size;
    let area_w = area.extent.width;
    let area_h = area.extent.height;
    // Optical center relative to the render area origin.
    let center_x = desc.center_px_x - area.offset.x as f32;
    let center_y = desc.center_px_y - area.offset.y as f32;

    let mut slot = d.vrs_images.lock().unwrap();
    if let Some(img) = slot.get(&desc.eye) {
        if img.matches(area_w, area_h, center_x, center_y, &desc.falloff) {
            return Some((img.view, texel));
        }
        destroy_vrs_image(d, img);
        slot.remove(&desc.eye);
    }

    let img = create_vrs_image(d, desc, area_w, area_h, center_x, center_y, texel)?;
    let view = img.view;
    slot.insert(desc.eye, img);
    Some((view, texel))
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_vrs_image(
    d: &DeviceData,
    desc: &ffr_core::wire::FoveationDesc,
    area_w: u32,
    area_h: u32,
    center_x: f32,
    center_y: f32,
    texel: vk::Extent2D,
) -> Option<VrsImage> {
    // Build the radial foveation map: full rate at the offset optical center,
    // coarsening toward the edges. Radii are fractions of the half-extent.
    let map = ffr_core::foveation::FoveationMap::generate(
        area_w,
        area_h,
        (center_x, center_y),
        texel.width.max(1),
        (area_w as f32 / 2.0, area_h as f32 / 2.0),
        &desc.falloff,
    );

    let dev = &d.device;
    let image = dev
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8_UINT)
                .extent(vk::Extent3D { width: map.cols, height: map.rows, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .ok()?;
    let req = dev.get_image_memory_requirements(image);
    let mem_type =
        memory_type(&d.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let memory = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type),
            None,
        )
        .ok()?;
    dev.bind_image_memory(image, memory, 0).ok()?;
    let view = dev
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8_UINT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
        .ok()?;

    upload_map(d, image, &map, 0)?;

    if *DEBUG {
        dump_foveation_ppm(desc.eye, &map);
    }

    if !LOGGED_INJECT.swap(true, Ordering::Relaxed) {
        info!(
            "built foveation map for eye {}: {}x{} tiles (texel {}x{}), \
             center at tile ({},{}) — VRS active",
            desc.eye,
            map.cols,
            map.rows,
            texel.width,
            texel.height,
            (center_x / texel.width as f32) as u32,
            (center_y / texel.height as f32) as u32,
        );
    }
    Some(VrsImage {
        image,
        memory,
        view,
        area_w,
        area_h,
        center_x,
        center_y,
        falloff: desc.falloff,
    })
}

/// Upload the foveation-map bytes into the `R8_UINT` image (staging buffer →
/// copy) and transition it to `FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL`.
unsafe fn upload_map(
    d: &DeviceData,
    image: vk::Image,
    map: &ffr_core::foveation::FoveationMap,
    base_layer: u32,
) -> Option<()> {
    let dev = &d.device;
    let size = map.bytes.len() as u64;

    let staging = dev
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .ok()?;
    let req = dev.get_buffer_memory_requirements(staging);
    let mem = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(
                memory_type(
                    &d.mem_props,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                ),
            ),
            None,
        )
        .ok()?;
    dev.bind_buffer_memory(staging, mem, 0).ok()?;
    let ptr = dev.map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty()).ok()?;
    std::ptr::copy_nonoverlapping(map.bytes.as_ptr(), ptr as *mut u8, map.bytes.len());
    dev.unmap_memory(mem);

    let queue = dev.get_device_queue(d.queue_family, 0);
    let pool = dev
        .create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(d.queue_family),
            None,
        )
        .ok()?;
    let cb = dev
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1),
        )
        .ok()?[0];
    dev.begin_command_buffer(
        cb,
        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )
    .ok()?;

    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: base_layer,
        layer_count: 1,
    };
    barrier(dev, cb, image, range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
    dev.cmd_copy_buffer_to_image(
        cb,
        staging,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: base_layer,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D { width: map.cols, height: map.rows, depth: 1 })],
    );
    barrier(dev, cb, image, range, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
        vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR);
    dev.end_command_buffer(cb).ok()?;

    let cbs = [cb];
    dev.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], vk::Fence::null())
        .ok()?;
    dev.queue_wait_idle(queue).ok()?;

    dev.destroy_command_pool(pool, None);
    dev.destroy_buffer(staging, None);
    dev.free_memory(mem, None);
    Some(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn barrier(
    dev: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) {
    let b = vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    dev.cmd_pipeline_barrier(cb, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[b]);
}

unsafe fn destroy_vrs_image(d: &DeviceData, img: &VrsImage) {
    d.device.destroy_image_view(img.view, None);
    d.device.destroy_image(img.image, None);
    d.device.free_memory(img.memory, None);
}

/// Write the shading-rate map to a PPM image, false-colored by rate:
/// green (1x1, full) → yellow (2x2) → red (4x4). Lets you *see* the foveation
/// pattern and confirm it tracks the offset optical center.
fn dump_foveation_ppm(eye: u32, map: &ffr_core::foveation::FoveationMap) {
    let mut rgb = Vec::with_capacity(map.bytes.len() * 3);
    for &b in &map.bytes {
        let (r, g, bl) = match ShadingRate::decode(b).coverage() {
            1 => (0u8, 255, 0),
            2 => (160, 255, 0),
            4 => (255, 255, 0),
            8 => (255, 140, 0),
            _ => (255, 0, 0),
        };
        rgb.extend_from_slice(&[r, g, bl]);
    }
    let dir = crate::logging::log_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = format!("{dir}/foveation-eye{eye}.ppm");
    let mut data = format!("P6\n{} {}\n255\n", map.cols, map.rows).into_bytes();
    data.extend_from_slice(&rgb);
    if std::fs::write(&path, &data).is_ok() {
        info!("wrote foveation debug image: {path}");
    }
}

/// Pick a memory type index satisfying `props` from `req_bits`.
fn memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    req_bits: u32,
    props: vk::MemoryPropertyFlags,
) -> u32 {
    (0..mem_props.memory_type_count)
        .find(|&i| {
            (req_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(props)
        })
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk a `pNext` chain (every Vulkan struct starts with `{sType, pNext}`) to
/// find the loader's instance dispatch-chain link.
unsafe fn find_instance_chain_info(
    p_ci: *const vk::InstanceCreateInfo,
) -> *mut VkLayerInstanceCreateInfo {
    let mut p = (*p_ci).p_next as *const VkLayerInstanceCreateInfo;
    while !p.is_null() {
        if (*p).s_type == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
            && (*p).function == VK_LAYER_LINK_INFO
        {
            return p as *mut VkLayerInstanceCreateInfo;
        }
        p = (*p).p_next as *const VkLayerInstanceCreateInfo;
    }
    ptr::null_mut()
}

unsafe fn find_device_chain_info(
    p_ci: *const vk::DeviceCreateInfo,
) -> *mut VkLayerDeviceCreateInfo {
    let mut p = (*p_ci).p_next as *const VkLayerDeviceCreateInfo;
    while !p.is_null() {
        if (*p).s_type == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
            && (*p).function == VK_LAYER_LINK_INFO
        {
            return p as *mut VkLayerDeviceCreateInfo;
        }
        p = (*p).p_next as *const VkLayerDeviceCreateInfo;
    }
    ptr::null_mut()
}

unsafe fn app_name(p_ci: *const vk::InstanceCreateInfo) -> String {
    let app_info = (*p_ci).p_application_info;
    if app_info.is_null() || (*app_info).p_application_name.is_null() {
        return "<none>".to_string();
    }
    CStr::from_ptr((*app_info).p_application_name)
        .to_string_lossy()
        .into_owned()
}

/// The raw `VkApplicationInfo.apiVersion`, or 1.0 if the app supplied none
/// (per spec, a zero/absent app api version means Vulkan 1.0).
unsafe fn app_api_version_raw(p_ci: *const vk::InstanceCreateInfo) -> u32 {
    let app_info = (*p_ci).p_application_info;
    let v = if app_info.is_null() { 0 } else { (*app_info).api_version };
    if v == 0 {
        vk::make_api_version(0, 1, 0, 0)
    } else {
        v
    }
}

unsafe fn app_api_version(p_ci: *const vk::InstanceCreateInfo) -> String {
    let app_info = (*p_ci).p_application_info;
    let v = if app_info.is_null() {
        0
    } else {
        (*app_info).api_version
    };
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

/// The effective device API version — `min(app instance version, GPU version)`.
/// Extension-dependency rules (e.g. "FSR needs renderpass2 unless core ≥1.2")
/// are evaluated against this, matching how the loader/validation judge them.
unsafe fn effective_device_api(instance_raw: Option<u64>, phys: vk::PhysicalDevice) -> u32 {
    let map = INSTANCES.read().unwrap();
    let Some(inst) = instance_raw.and_then(|ir| map.get(&ir)) else {
        return vk::make_api_version(0, 1, 0, 0);
    };
    let props = inst.instance.get_physical_device_properties(phys);
    inst.app_api.min(props.api_version)
}

unsafe fn gpu_name(instance_raw: Option<u64>, phys: vk::PhysicalDevice) -> String {
    let Some(inst) = instance_raw else {
        return "<unknown>".to_string();
    };
    let get_props = INSTANCES
        .read()
        .unwrap()
        .get(&inst)
        .map(|d| d.get_physical_device_properties);
    let Some(get_props) = get_props else {
        return "<unknown>".to_string();
    };
    let mut props = vk::PhysicalDeviceProperties::default();
    get_props(phys, &mut props);
    CStr::from_ptr(props.device_name.as_ptr())
        .to_string_lossy()
        .into_owned()
}
