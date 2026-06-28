//! Headless multiview validation harness for the FFR Vulkan layer (M8).
//!
//! This drives the exact code path xrgears exercises — a Vulkan-1.0-style
//! `vkCreateRenderPass` carrying `VkRenderPassMultiviewCreateInfo` (viewMask
//! 0b11), a 2-layer array color target, and a layered framebuffer — but with no
//! OpenXR and no window, so it can run under the Khronos validation layer in CI
//! or a dev shell. The FFR layer (loaded implicitly via `ENABLE_FFR_VRS=1`)
//! promotes the render pass to renderpass2, appends a 2-layer fragment-shading-
//! rate attachment, and fills one shading-rate map per view. Any spec violation
//! in that augmentation surfaces as a validation error, which this harness
//! counts and turns into a non-zero exit code.
//!
//! Two modes:
//!   * default — load the FFR layer (explicitly, above any validation layer) and
//!     submit a *minimal* multiview pass; the layer injects the layered FSR
//!     attachment and the driver executes it. Confirms the real layer output runs.
//!   * `MV_CHECK_SELF_AUGMENT=1` — no FFR layer; the harness itself builds the
//!     fully-augmented structure the layer would produce (appended 2-layer
//!     R8_UINT shading-rate attachment, layered framebuffer, multiview view mask)
//!     and runs it under the Khronos validation layer. This is the authoritative,
//!     driver-independent spec check of the layered-FSR multiview structure. A
//!     validation layer cannot sit above the live FFR layer because the objects
//!     FFR injects would be invisible to it, so the two checks are kept separate.
//!
//! Run (from a dev shell, with the nix-built layer on `XDG_DATA_DIRS`):
//!   ENABLE_FFR_VRS=1 cargo run --bin mv_check                 # layer + driver
//!   MV_CHECK_SELF_AUGMENT=1 cargo run --bin mv_check          # structure + validation

use ash::vk;
use std::error::Error;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicU32, Ordering};

static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;
const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const VIEWS: u32 = 2;
const VIEW_MASK: u32 = 0b11; // both views

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = CStr::from_ptr((*data).p_message).to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        eprintln!("[VALIDATION ERROR] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        eprintln!("[validation warning] {msg}");
    }
    vk::FALSE
}

fn memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    want: vk::MemoryPropertyFlags,
) -> u32 {
    (0..props.memory_type_count)
        .find(|&i| {
            (type_bits & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(want)
        })
        .unwrap_or(0)
}

fn main() {
    match run() {
        Ok(()) => {
            let errors = VALIDATION_ERRORS.load(Ordering::Relaxed);
            if errors == 0 {
                println!("mv_check: PASS (multiview render pass + layered FSR attachment, no validation errors)");
            } else {
                eprintln!("mv_check: FAIL ({errors} validation error(s))");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("mv_check: ERROR {e}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let self_augment = std::env::var_os("MV_CHECK_SELF_AUGMENT").is_some();
    let entry = unsafe { ash::Entry::load()? };

    let ffr = c"VK_LAYER_FFR_VRS_foveation";
    let want_validation = c"VK_LAYER_KHRONOS_validation";
    let available = unsafe { entry.enumerate_instance_layer_properties()? };
    let has = |name: &CStr| {
        available.iter().any(|p| unsafe { CStr::from_ptr(p.layer_name.as_ptr()) } == name)
    };
    let have_validation = has(want_validation);

    // Self-augment mode validates the augmented structure directly — no FFR
    // layer (its injected objects would be invisible to a validation layer above
    // it). Default mode loads FFR to exercise its real injection on the driver.
    let mut layers: Vec<*const i8> = Vec::new();
    if self_augment {
        println!("mv_check: self-augment mode (harness builds the layered FSR structure)");
        if !have_validation {
            eprintln!("mv_check: WARNING validation layer not found — structural check only");
        } else {
            layers.push(want_validation.as_ptr());
        }
    } else {
        println!("mv_check: layer mode (FFR injects; driver executes)");
        if !has(ffr) {
            return Err(
                "FFR layer not discoverable (is XDG_DATA_DIRS pointing at the built layer?)".into(),
            );
        }
        layers.push(ffr.as_ptr());
    }
    let inst_exts = [vk::EXT_DEBUG_UTILS_NAME.as_ptr()];

    let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&inst_exts),
            None,
        )?
    };

    let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let messenger = unsafe {
        debug_utils.create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback)),
            None,
        )?
    };

    let physical_device = unsafe { instance.enumerate_physical_devices()? }
        .into_iter()
        .next()
        .ok_or("no Vulkan physical device")?;
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let queue_family = unsafe {
        instance
            .get_physical_device_queue_family_properties(physical_device)
            .into_iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("no graphics queue")? as u32
    };

    // Enable multiview. In layer mode the FFR layer adds fragment-shading-rate +
    // its renderpass2 dependency; in self-augment mode the harness must enable
    // them itself (it is standing in for the layer).
    let mut multiview = vk::PhysicalDeviceMultiviewFeatures::default().multiview(true);
    let mut fsr_feature =
        vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default().attachment_fragment_shading_rate(true);
    let dev_exts = [
        c"VK_KHR_fragment_shading_rate".as_ptr(),
        c"VK_KHR_create_renderpass2".as_ptr(),
    ];
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let mut dev_ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .push_next(&mut multiview);
    if self_augment {
        dev_ci = dev_ci.enabled_extension_names(&dev_exts).push_next(&mut fsr_feature);
    }
    let device = unsafe { instance.create_device(physical_device, &dev_ci, None)? };
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    unsafe {
        exercise_multiview(&instance, &device, &mem_props, queue, queue_family, self_augment)?
    };

    unsafe {
        device.device_wait_idle()?;
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    Ok(())
}

const FSR_LAYOUT: vk::ImageLayout = vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
const TEXEL: u32 = 16;

unsafe fn alloc_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    format: vk::Format,
    extent: vk::Extent2D,
    usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn Error>> {
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(VIEWS)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let req = device.get_image_memory_requirements(image);
    let mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(
            memory_type(mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL),
        ),
        None,
    )?;
    device.bind_image_memory(image, mem, 0)?;
    let view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: VIEWS,
            }),
        None,
    )?;
    Ok((image, mem, view))
}

/// Build a 2-layer color target and a multiview render pass + layered
/// framebuffer, then record a clear pass.
///
/// In layer mode the render pass is a *minimal* v1 multiview pass (xrgears
/// style) and the FFR layer injects the shading-rate attachment. In self-augment
/// mode the harness builds the augmented renderpass2 itself — appended 2-layer
/// R8_UINT shading-rate attachment, layered framebuffer, view mask preserved —
/// exactly as the layer would, so the validation layer checks that structure.
unsafe fn exercise_multiview(
    instance: &ash::Instance,
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    queue_family: u32,
    self_augment: bool,
) -> Result<(), Box<dyn Error>> {
    let extent = vk::Extent2D { width: WIDTH, height: HEIGHT };
    let (color, color_mem, color_view) = alloc_image(
        device,
        mem_props,
        COLOR_FORMAT,
        extent,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    )?;

    // Optional 2-layer shading-rate attachment (self-augment mode only).
    let sr = if self_augment {
        let sr_extent =
            vk::Extent2D { width: WIDTH.div_ceil(TEXEL), height: HEIGHT.div_ceil(TEXEL) };
        Some(alloc_image(
            device,
            mem_props,
            vk::Format::R8_UINT,
            sr_extent,
            vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        )?)
    } else {
        None
    };

    let view_masks = [VIEW_MASK];
    let correlation_masks = [VIEW_MASK];
    let fb_atts: Vec<vk::ImageView>;

    let render_pass = if let Some((_, _, sr_view)) = sr {
        // Augmented renderpass2 with the layered FSR attachment (the layer's output).
        let attachments = [
            vk::AttachmentDescription2::default()
                .format(COLOR_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentDescription2::default()
                .format(vk::Format::R8_UINT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(FSR_LAYOUT)
                .final_layout(FSR_LAYOUT),
        ];
        let color_ref = [vk::AttachmentReference2::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::COLOR)];
        let fsr_ref = vk::AttachmentReference2::default()
            .attachment(1)
            .layout(FSR_LAYOUT)
            .aspect_mask(vk::ImageAspectFlags::COLOR);
        let mut fsr_info = vk::FragmentShadingRateAttachmentInfoKHR::default()
            .shading_rate_attachment_texel_size(vk::Extent2D { width: TEXEL, height: TEXEL });
        fsr_info.p_fragment_shading_rate_attachment = &fsr_ref;
        let subpasses = [vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .view_mask(VIEW_MASK)
            .color_attachments(&color_ref)
            .push_next(&mut fsr_info)];
        let ci = vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .correlated_view_masks(&correlation_masks);
        let rp2 = ash::khr::create_renderpass2::Device::new(instance, device);
        fb_atts = vec![color_view, sr_view];
        rp2.create_render_pass2(&ci, None)?
    } else {
        // Minimal v1 multiview pass — the FFR layer augments it.
        let attachments = [vk::AttachmentDescription::default()
            .format(COLOR_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let color_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref)];
        let mut multiview_ci = vk::RenderPassMultiviewCreateInfo::default()
            .view_masks(&view_masks)
            .correlation_masks(&correlation_masks);
        fb_atts = vec![color_view];
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .push_next(&mut multiview_ci),
            None,
        )?
    };

    // Layered framebuffer (layers=1 for multiview — the view mask spans layers).
    let framebuffer = device.create_framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&fb_atts)
            .width(WIDTH)
            .height(HEIGHT)
            .layers(1),
        None,
    )?;

    let pool = device.create_command_pool(
        &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
        None,
    )?;
    let cb = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1),
    )?[0];
    device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
    // The FSR attachment loads from FSR_LAYOUT, so transition the SR image first.
    if let Some((sr_image, _, _)) = sr {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(FSR_LAYOUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(sr_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: VIEWS,
            });
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
    let clears = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
    device.cmd_begin_render_pass(
        cb,
        &vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clears),
        vk::SubpassContents::INLINE,
    );
    device.cmd_end_render_pass(cb);
    device.end_command_buffer(cb)?;

    let cbs = [cb];
    device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;

    device.destroy_command_pool(pool, None);
    device.destroy_framebuffer(framebuffer, None);
    device.destroy_render_pass(render_pass, None);
    if let Some((sr_image, sr_mem, sr_view)) = sr {
        device.destroy_image_view(sr_view, None);
        device.destroy_image(sr_image, None);
        device.free_memory(sr_mem, None);
    }
    device.destroy_image_view(color_view, None);
    device.destroy_image(color, None);
    device.free_memory(color_mem, None);
    Ok(())
}
