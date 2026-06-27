//! `ffr-openxr-layer`: OpenXR API layer (producer side).
//!
//! M1: real `xrNegotiateLoaderApiLayerInterface` + hand-rolled
//! `xrCreateApiLayerInstance` / `xrGetInstanceProcAddr` pass-through (see
//! [`layer`]) with a file-logged banner. Later milestones add hooks to derive
//! PPD + optical center from `XrView` fov, capture eye swapchain `VkImage`
//! handles, and publish `FoveationDesc`s into the shared registry.
#![allow(non_snake_case)] // crate/lib name must be `XrApiLayer_FFR_VRS` for the loader

mod layer;
mod logging;
mod xr_sys;

pub use layer::xrNegotiateLoaderApiLayerInterface;
