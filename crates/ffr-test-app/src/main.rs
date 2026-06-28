//! `ffr-test-app`: a minimal OpenXR + Vulkan **dynamic-rendering** stereo app
//! that draws 8 cubes arranged in a circle in the X/Y plane. It exists to drive
//! the FFR layers under a runtime (Monado): the OpenXR layer reads the per-eye
//! FOV and tags the swapchain images, and the Vulkan layer injects variable-rate
//! shading into these `vkCmdBeginRendering` passes.

use std::error::Error;

use ash::vk::{self, Handle};
use glam::{Mat4, Quat, Vec3};
use openxr as xr;

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const BLEND: xr::EnvironmentBlendMode = xr::EnvironmentBlendMode::OPAQUE;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const CUBE_COUNT: usize = 8;
const RING_RADIUS: f32 = 1.2;
const RING_DEPTH: f32 = -2.5; // metres in front of the local origin (-Z forward)
const CUBE_SCALE: f32 = 0.5;

type VkGetInstanceProcAddr = unsafe extern "system" fn(
    *const std::ffi::c_void,
    *const std::os::raw::c_char,
) -> Option<unsafe extern "system" fn()>;

#[rustfmt::skip]
const CUBE_VERTS: [[f32; 3]; 8] = [
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5],
];
#[rustfmt::skip]
const CUBE_INDICES: [u16; 36] = [
    0, 1, 2, 2, 3, 0, // -Z
    4, 5, 6, 6, 7, 4, // +Z
    0, 4, 7, 7, 3, 0, // -X
    1, 5, 6, 6, 2, 1, // +X
    3, 2, 6, 6, 7, 3, // +Y
    0, 1, 5, 5, 4, 0, // -Y
];

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    mvp: [f32; 16],
    tint: [f32; 4],
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ffr-test-app error: {e}");
        std::process::exit(1);
    }
    println!("ffr-test-app: done.");
}

fn run() -> Result<(), Box<dyn Error>> {
    println!("ffr-test-app: loading OpenXR loader…");
    let entry = unsafe { xr::Entry::load()? };
    let mut exts = xr::ExtensionSet::default();
    exts.khr_vulkan_enable2 = true;
    let xr_instance = entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "ffr-test-app",
            application_version: 1,
            engine_name: "ffr",
            engine_version: 0,
            api_version: xr::sys::CURRENT_API_VERSION,
        },
        &exts,
        &[],
    )?;
    if let Ok(p) = xr_instance.properties() {
        println!("Runtime: {} (v{:?})", p.runtime_name, p.runtime_version);
    }

    let system = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    if let Ok(sp) = xr_instance.system_properties(system) {
        println!("System: {}", sp.system_name);
    }
    let view_cfg = xr_instance.enumerate_view_configuration_views(system, VIEW_TYPE)?;
    let resolution = vk::Extent2D {
        width: view_cfg[0].recommended_image_rect_width,
        height: view_cfg[0].recommended_image_rect_height,
    };
    println!("Per-eye resolution: {}x{}", resolution.width, resolution.height);

    // Debug mode: overlay a translucent false-color of the actual applied
    // shading rate (gl_ShadingRateEXT) on top of the scene so foveation is
    // visible even when the scene fills the frame.
    let debug_rate = std::env::var_os("FFR_TEST_DEBUG_RATE").is_some();
    if debug_rate {
        println!("FFR_TEST_DEBUG_RATE: overlaying shading-rate false-color on the scene");
    }

    let _reqs = xr_instance.graphics_requirements::<xr::Vulkan>(system)?;
    let vk = VkCtx::new(&xr_instance, system, debug_rate)?;

    let (session, mut frame_wait, mut frame_stream) = unsafe {
        xr_instance.create_session::<xr::Vulkan>(
            system,
            &xr::vulkan::SessionCreateInfo {
                instance: vk.instance.handle().as_raw() as _,
                physical_device: vk.physical_device.as_raw() as _,
                device: vk.device.handle().as_raw() as _,
                queue_family_index: vk.queue_family_index,
                queue_index: 0,
            },
        )?
    };

    let color_format = pick_color_format(&session)?;
    let stage = session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;
    let renderer = Renderer::new(&vk, color_format, resolution, debug_rate)?;
    let mut eyes: Vec<Eye> = (0..2)
        .map(|_| Eye::new(&vk, &session, color_format, resolution))
        .collect::<Result<_, _>>()?;

    let mut event_storage = xr::EventDataBuffer::new();
    let mut running = false;
    let mut frame: u32 = 0;
    // Render-time accumulation for the GPU A/B (the app waits idle after each
    // eye submit, so this reflects per-eye GPU render time + submit overhead).
    let mut render_time = std::time::Duration::ZERO;
    let mut render_samples: u64 = 0;

    'main: loop {
        while let Some(event) = xr_instance.poll_event(&mut event_storage)? {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        session.begin(VIEW_TYPE)?;
                        running = true;
                        println!("session READY — drawing 8 cubes in a ring…");
                    }
                    xr::SessionState::STOPPING => {
                        session.end()?;
                        running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => break 'main,
                    _ => {}
                },
                InstanceLossPending(_) => break 'main,
                _ => {}
            }
        }
        if !running {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        let frame_state = frame_wait.wait()?;
        frame_stream.begin()?;
        if !frame_state.should_render {
            frame_stream.end(frame_state.predicted_display_time, BLEND, &[])?;
            continue;
        }

        // Self-terminate after a bounded number of frames (0 = run until closed).
        let max_frames: u32 = std::env::var("FFR_TEST_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(900);
        if max_frames != 0 && frame >= max_frames {
            session.request_exit()?;
        }

        let (_flags, views) =
            session.locate_views(VIEW_TYPE, frame_state.predicted_display_time, &stage)?;
        let spin = frame as f32 * 0.02;

        for (i, eye) in eyes.iter_mut().enumerate() {
            let idx = eye.swapchain.acquire_image()? as usize;
            eye.swapchain.wait_image(xr::Duration::INFINITE)?;
            let proj = projection_from_fov(views[i].fov, 0.05, 100.0);
            let view = view_from_pose(views[i].pose);
            let t0 = std::time::Instant::now();
            renderer.render(&vk, eye.images[idx], eye.views[idx], proj * view, spin)?;
            // Ignore warm-up frames (pipeline/shader caching, first-use allocs).
            if frame > 30 {
                render_time += t0.elapsed();
                render_samples += 1;
            }
            eye.swapchain.release_image()?;
        }

        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: resolution.width as i32,
                height: resolution.height as i32,
            },
        };
        let proj_views: Vec<xr::CompositionLayerProjectionView<xr::Vulkan>> = views
            .iter()
            .zip(eyes.iter())
            .map(|(view, eye)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&eye.swapchain)
                            .image_array_index(0)
                            .image_rect(rect),
                    )
            })
            .collect();
        frame_stream.end(
            frame_state.predicted_display_time,
            BLEND,
            &[&xr::CompositionLayerProjection::new()
                .space(&stage)
                .views(&proj_views)],
        )?;
        frame += 1;
    }

    unsafe { vk.device.device_wait_idle().ok() };

    if render_samples > 0 {
        let avg_ms = render_time.as_secs_f64() * 1000.0 / render_samples as f64;
        let vrs = if std::env::var_os("FFR_VRS_DISABLE").is_some() {
            "FFR_VRS_DISABLE set (no injection)"
        } else {
            "VRS injected if layer active"
        };
        println!(
            "avg eye-render time: {avg_ms:.3} ms over {render_samples} samples [{vrs}]"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

/// Vulkan projection matrix from an OpenXR asymmetric FOV (Y-down clip, 0..1 Z).
fn projection_from_fov(fov: xr::Fovf, near: f32, far: f32) -> Mat4 {
    let l = fov.angle_left.tan();
    let r = fov.angle_right.tan();
    let u = fov.angle_up.tan();
    let d = fov.angle_down.tan();
    let w = r - l;
    let h = d - u; // Vulkan: positive Y is down
    let mut m = [0.0f32; 16];
    m[0] = 2.0 / w;
    m[8] = (r + l) / w;
    m[5] = 2.0 / h;
    m[9] = (u + d) / h;
    m[10] = -far / (far - near);
    m[14] = -(far * near) / (far - near);
    m[11] = -1.0;
    Mat4::from_cols_array(&m)
}

fn view_from_pose(pose: xr::Posef) -> Mat4 {
    let q = pose.orientation;
    let p = pose.position;
    let rot = Quat::from_xyzw(q.x, q.y, q.z, q.w);
    let trans = Vec3::new(p.x, p.y, p.z);
    Mat4::from_rotation_translation(rot, trans).inverse()
}

fn cube_transform(i: usize, spin: f32) -> (Mat4, [f32; 4]) {
    let a = i as f32 / CUBE_COUNT as f32 * std::f32::consts::TAU;
    let pos = Vec3::new(RING_RADIUS * a.cos(), RING_RADIUS * a.sin(), RING_DEPTH);
    let model = Mat4::from_translation(pos)
        * Mat4::from_rotation_y(spin + a)
        * Mat4::from_scale(Vec3::splat(CUBE_SCALE));
    let tint = [
        0.5 + 0.5 * a.cos(),
        0.5 + 0.5 * (a + 2.094).cos(),
        0.5 + 0.5 * (a + 4.188).cos(),
        1.0,
    ];
    (model, tint)
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

struct Renderer {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// Optional full-screen shading-rate overlay pipeline (debug mode).
    overlay_pipeline: Option<vk::Pipeline>,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    depth_image: vk::Image,
    depth_view: vk::ImageView,
    resolution: vk::Extent2D,
}

impl Renderer {
    fn new(
        vk: &VkCtx,
        color_format: vk::Format,
        resolution: vk::Extent2D,
        debug_rate: bool,
    ) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let cube_vert =
                load_shader(&vk.device, include_bytes!(concat!(env!("OUT_DIR"), "/cube.vert.spv")))?;
            let cube_frag =
                load_shader(&vk.device, include_bytes!(concat!(env!("OUT_DIR"), "/cube.frag.spv")))?;

            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<PushConstants>() as u32)];
            let pipeline_layout = vk.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_range),
                None,
            )?;

            let bindings = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(12)
                .input_rate(vk::VertexInputRate::VERTEX)];
            let attrs = [vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)];
            let pipeline = build_pipeline(
                &vk.device, pipeline_layout, cube_vert, cube_frag, color_format, &bindings, &attrs,
                true, false,
            )?;
            vk.device.destroy_shader_module(cube_vert, None);
            vk.device.destroy_shader_module(cube_frag, None);

            // Debug overlay: a full-screen triangle (no vertex input, no depth)
            // drawn over the scene with alpha blending, reading the applied
            // shading rate and false-coloring it.
            let overlay_pipeline = if debug_rate {
                let fs_vert = load_shader(
                    &vk.device,
                    include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv")),
                )?;
                let rate_frag = load_shader(
                    &vk.device,
                    include_bytes!(concat!(env!("OUT_DIR"), "/rate_vis.frag.spv")),
                )?;
                let p = build_pipeline(
                    &vk.device, pipeline_layout, fs_vert, rate_frag, color_format, &[], &[], false,
                    true,
                )?;
                vk.device.destroy_shader_module(fs_vert, None);
                vk.device.destroy_shader_module(rate_frag, None);
                Some(p)
            } else {
                None
            };

            let vertex_buffer =
                vk.create_buffer(bytes_of(&CUBE_VERTS), vk::BufferUsageFlags::VERTEX_BUFFER)?;
            let index_buffer =
                vk.create_buffer(bytes_of(&CUBE_INDICES), vk::BufferUsageFlags::INDEX_BUFFER)?;
            let (depth_image, depth_view) = vk.create_depth(resolution)?;

            Ok(Renderer {
                pipeline_layout,
                pipeline,
                overlay_pipeline,
                vertex_buffer,
                index_buffer,
                depth_image,
                depth_view,
                resolution,
            })
        }
    }

    /// Record + submit a command buffer that draws the cube ring into one eye.
    fn render(
        &self,
        vk: &VkCtx,
        color_image: vk::Image,
        color_view: vk::ImageView,
        view_proj: Mat4,
        spin: f32,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            let cb = vk.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(vk.cmd_pool)
                    .command_buffer_count(1),
            )?[0];
            vk.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // Dynamic rendering does not transition layouts for us.
            let barriers = [
                image_barrier(
                    color_image,
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                image_barrier(
                    self.depth_image,
                    vk::ImageAspectFlags::DEPTH,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            ];
            vk.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            self.draw(vk, cb, color_view, view_proj, spin);

            vk.device.end_command_buffer(cb)?;
            let cbs = [cb];
            vk.device.queue_submit(
                vk.queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                vk::Fence::null(),
            )?;
            vk.device.queue_wait_idle(vk.queue)?;
            vk.device.free_command_buffers(vk.cmd_pool, &cbs);
        }
        Ok(())
    }

    unsafe fn draw(
        &self,
        vk: &VkCtx,
        cb: vk::CommandBuffer,
        color_view: vk::ImageView,
        view_proj: Mat4,
        spin: f32,
    ) {
        let extent = self.resolution;
        let color_att = vk::RenderingAttachmentInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.01, 0.01, 0.03, 1.0] },
            });
        let depth_att = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            });
        let color_atts = [color_att];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(&color_atts)
            .depth_attachment(&depth_att);

        vk.device.cmd_begin_rendering(cb, &rendering);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        vk.device.cmd_set_viewport(cb, 0, &[viewport]);
        vk.device
            .cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);

        // Scene: the cube ring.
        vk.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        vk.device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer], &[0]);
        vk.device.cmd_bind_index_buffer(cb, self.index_buffer, 0, vk::IndexType::UINT16);

        for i in 0..CUBE_COUNT {
            let (model, tint) = cube_transform(i, spin);
            let pc = PushConstants { mvp: (view_proj * model).to_cols_array(), tint };
            vk.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytes_of_val(&pc),
            );
            vk.device.cmd_draw_indexed(cb, CUBE_INDICES.len() as u32, 1, 0, 0, 0);
        }

        // Debug: alpha-blended shading-rate overlay on top of the whole scene.
        if let Some(overlay) = self.overlay_pipeline {
            vk.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, overlay);
            vk.device.cmd_draw(cb, 3, 1, 0, 0);
        }
        vk.device.cmd_end_rendering(cb);
    }
}

// ---------------------------------------------------------------------------
// Vulkan context
// ---------------------------------------------------------------------------

struct Eye {
    swapchain: xr::Swapchain<xr::Vulkan>,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

impl Eye {
    fn new(
        vk: &VkCtx,
        session: &xr::Session<xr::Vulkan>,
        format: vk::Format,
        resolution: vk::Extent2D,
    ) -> Result<Self, Box<dyn Error>> {
        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED,
            format: format.as_raw() as u32,
            sample_count: 1,
            width: resolution.width,
            height: resolution.height,
            face_count: 1,
            array_size: 1,
            mip_count: 1,
        })?;
        let images: Vec<vk::Image> = swapchain
            .enumerate_images()?
            .into_iter()
            .map(vk::Image::from_raw)
            .collect();
        let views = images
            .iter()
            .map(|&img| vk.image_view(img, format))
            .collect::<Result<_, _>>()?;
        Ok(Eye { swapchain, images, views })
    }
}

struct VkCtx {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    cmd_pool: vk::CommandPool,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl VkCtx {
    fn new(
        xr_instance: &xr::Instance,
        system: xr::SystemId,
        debug_rate: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { ash::Entry::load()? };
        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let instance = unsafe {
            let raw = xr_instance
                .create_vulkan_instance(
                    system,
                    std::mem::transmute::<ash::vk::PFN_vkGetInstanceProcAddr, VkGetInstanceProcAddr>(
                        entry.static_fn().get_instance_proc_addr,
                    ),
                    &vk::InstanceCreateInfo::default().application_info(&app_info) as *const _
                        as *const _,
                )?
                .map_err(vk::Result::from_raw)?;
            ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(raw as _))
        };
        let physical_device = vk::PhysicalDevice::from_raw(unsafe {
            xr_instance.vulkan_graphics_device(system, instance.handle().as_raw() as _)? as _
        });
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(i, _)| i as u32)
                .ok_or("no graphics queue family")?
        };

        let mut vk13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        // In debug mode the app's own pipeline reads gl_ShadingRateEXT, so it must
        // enable the extension + feature itself (the layer also enables it, but
        // for the app's pipeline the app must request it at device creation).
        let mut fsr =
            vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default()
                .attachment_fragment_shading_rate(true);
        let fsr_ext = [c"VK_KHR_fragment_shading_rate".as_ptr()];
        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let mut create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(&queue_info).push_next(&mut vk13);
        if debug_rate {
            create_info = create_info.enabled_extension_names(&fsr_ext).push_next(&mut fsr);
        }
        let device = unsafe {
            let raw = xr_instance
                .create_vulkan_device(
                    system,
                    std::mem::transmute::<ash::vk::PFN_vkGetInstanceProcAddr, VkGetInstanceProcAddr>(
                        entry.static_fn().get_instance_proc_addr,
                    ),
                    physical_device.as_raw() as _,
                    &create_info as *const _ as *const _,
                )?
                .map_err(vk::Result::from_raw)?;
            ash::Device::load(instance.fp_v1_0(), vk::Device::from_raw(raw as _))
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            cmd_pool,
            mem_props,
        })
    }

    fn memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem_props.memory_type_count)
            .find(|&i| {
                (type_bits & (1 << i)) != 0
                    && self.mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(props)
            })
            .unwrap_or(0)
    }

    fn image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
    ) -> Result<vk::ImageView, Box<dyn Error>> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range());
        Ok(unsafe { self.device.create_image_view(&info, None)? })
    }

    fn create_buffer(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<vk::Buffer, Box<dyn Error>> {
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(data.len() as u64)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(
                    self.memory_type(
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ),
                ),
                None,
            )?;
            self.device.bind_buffer_memory(buffer, mem, 0)?;
            let ptr = self.device.map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(mem);
            Ok(buffer)
        }
    }

    fn create_depth(
        &self,
        extent: vk::Extent2D,
    ) -> Result<(vk::Image, vk::ImageView), Box<dyn Error>> {
        unsafe {
            let image = self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(DEPTH_FORMAT)
                    .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = self.device.get_image_memory_requirements(image);
            let mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(
                    self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                ),
                None,
            )?;
            self.device.bind_image_memory(image, mem, 0)?;
            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(DEPTH_FORMAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?;
            Ok((image, view))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn load_shader(
    device: &ash::Device,
    bytes: &[u8],
) -> Result<vk::ShaderModule, Box<dyn Error>> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    Ok(device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?)
}

/// Build a dynamic-rendering graphics pipeline. With empty `bindings`/`attrs`
/// and `depth_test = false` it's a full-screen pass.
#[allow(clippy::too_many_arguments)]
unsafe fn build_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    color_format: vk::Format,
    bindings: &[vk::VertexInputBindingDescription],
    attrs: &[vk::VertexInputAttributeDescription],
    depth_test: bool,
    blend: bool,
) -> Result<vk::Pipeline, Box<dyn Error>> {
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
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(bindings)
        .vertex_attribute_descriptions(attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state =
        vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_test)
        .depth_write_enable(depth_test)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attachment = if blend {
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
    } else {
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
    };
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(DEPTH_FORMAT);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut rendering);
    Ok(device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        .map_err(|(_, e)| e)?[0])
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn image_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

fn bytes_of<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn bytes_of_val<T>(t: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(t as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn pick_color_format(session: &xr::Session<xr::Vulkan>) -> Result<vk::Format, Box<dyn Error>> {
    let formats = session.enumerate_swapchain_formats()?;
    for preferred in [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB] {
        if formats.contains(&(preferred.as_raw() as u32)) {
            return Ok(preferred);
        }
    }
    formats
        .first()
        .map(|&f| vk::Format::from_raw(f as i32))
        .ok_or_else(|| "no swapchain formats".into())
}
