//! `ffr-vulkan-layer`: Vulkan layer (consumer side).
//!
//! M1: real `vkNegotiateLoaderLayerInterfaceVersion` + hand-rolled pass-through
//! dispatch (see [`layer`]) with a file-logged banner at instance/device
//! creation. Later milestones extend the same dispatch to enable
//! `VK_KHR_fragment_shading_rate` at `vkCreateDevice`, recreate graphics
//! pipelines with the shading-rate combiner, and inject a shading-rate
//! attachment at `vkCmdBeginRendering` for registry-tagged eye images.
#![allow(non_snake_case)] // crate/lib name must be `VkLayer_FFR_VRS` for the loader

mod layer;
mod logging;
mod vk_sys;

pub use layer::vkNegotiateLoaderLayerInterfaceVersion;
