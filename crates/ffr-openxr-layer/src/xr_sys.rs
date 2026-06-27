//! OpenXR *loader ↔ API-layer* interface structs from `loader_interfaces.h`.
//! These are loader-specific and absent from `openxr-sys`, so we declare them
//! here with matching `#[repr(C)]` layout, reusing `openxr-sys` for base types.

use std::os::raw::{c_char, c_void};

use openxr_sys::{pfn, Instance, InstanceCreateInfo, Result as XrResult, Version};

pub const XR_API_LAYER_MAX_SETTINGS_PATH_SIZE: usize = 512;
pub const XR_MAX_API_LAYER_NAME_SIZE: usize = 256;

/// Current loader ↔ API-layer interface version we implement.
pub const XR_CURRENT_LOADER_API_LAYER_VERSION: u32 = 1;

// `XrLoaderInterfaceStructs` discriminants we care about.
pub const XR_LOADER_INTERFACE_STRUCT_LOADER_INFO: i32 = 1;
pub const XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST: i32 = 2;

/// `PFN_xrCreateApiLayerInstance` — not in `openxr-sys` (loader-only).
pub type PfnCreateApiLayerInstance = unsafe extern "system" fn(
    info: *const InstanceCreateInfo,
    api_layer_info: *const XrApiLayerCreateInfo,
    instance: *mut Instance,
) -> XrResult;

/// `XrNegotiateLoaderInfo` — passed by the loader for us to inspect.
#[repr(C)]
pub struct XrNegotiateLoaderInfo {
    pub struct_type: i32,
    pub struct_version: u32,
    pub struct_size: usize,
    pub min_interface_version: u32,
    pub max_interface_version: u32,
    pub min_api_version: Version,
    pub max_api_version: Version,
}

/// `XrNegotiateApiLayerRequest` — we fill this in during negotiation.
#[repr(C)]
pub struct XrNegotiateApiLayerRequest {
    pub struct_type: i32,
    pub struct_version: u32,
    pub struct_size: usize,
    pub layer_interface_version: u32,
    pub layer_api_version: Version,
    pub get_instance_proc_addr: Option<pfn::GetInstanceProcAddr>,
    pub create_api_layer_instance: Option<PfnCreateApiLayerInstance>,
}

/// `XrApiLayerNextInfo` — one link in the API-layer call-down chain.
#[repr(C)]
pub struct XrApiLayerNextInfo {
    pub struct_type: i32,
    pub struct_version: u32,
    pub struct_size: usize,
    pub layer_name: [c_char; XR_MAX_API_LAYER_NAME_SIZE],
    pub next_get_instance_proc_addr: Option<pfn::GetInstanceProcAddr>,
    pub next_create_api_layer_instance: Option<PfnCreateApiLayerInstance>,
    pub next: *mut XrApiLayerNextInfo,
}

/// `XrApiLayerCreateInfo` — handed to `xrCreateApiLayerInstance`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XrApiLayerCreateInfo {
    pub struct_type: i32,
    pub next: *mut c_void,
    pub loader_instance: *mut c_void,
    pub settings_file_path: [c_char; XR_API_LAYER_MAX_SETTINGS_PATH_SIZE],
    pub next_info: *mut XrApiLayerNextInfo,
}
