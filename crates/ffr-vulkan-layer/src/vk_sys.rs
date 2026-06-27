//! The Vulkan *loader ↔ layer* interface structs from `vk_layer.h`. These are
//! not part of the core API and are absent from `ash`, so we declare them here
//! with matching `#[repr(C)]` layout.

use std::os::raw::c_void;

use ash::vk;

/// `VkStructureType` chained into `VkInstanceCreateInfo::pNext` to carry the
/// layer dispatch chain. The loader's `vk_layer.h` defines this as `47` (it
/// reuses the small enum value because the loader struct can never collide with
/// a core struct inside an instance-create `pNext`).
pub const VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO: vk::StructureType =
    vk::StructureType::from_raw(47);
/// `VkStructureType` chained into `VkDeviceCreateInfo::pNext` (loader value `48`).
pub const VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO: vk::StructureType =
    vk::StructureType::from_raw(48);

/// `VkLayerFunction::VK_LAYER_LINK_INFO` — selects the dispatch-chain link in the
/// `VkLayer*CreateInfo` union.
pub const VK_LAYER_LINK_INFO: i32 = 0;

pub type PfnGetInstanceProcAddr = vk::PFN_vkGetInstanceProcAddr;
pub type PfnGetDeviceProcAddr = vk::PFN_vkGetDeviceProcAddr;
pub type PfnGetPhysicalDeviceProcAddr =
    unsafe extern "system" fn(vk::Instance, *const std::os::raw::c_char) -> vk::PFN_vkVoidFunction;

/// One link in the instance dispatch chain.
#[repr(C)]
pub struct VkLayerInstanceLink {
    pub p_next: *mut VkLayerInstanceLink,
    pub pfn_next_get_instance_proc_addr: PfnGetInstanceProcAddr,
    pub pfn_next_get_physical_device_proc_addr: Option<PfnGetPhysicalDeviceProcAddr>,
}

#[repr(C)]
pub union VkLayerInstanceCreateInfoUnion {
    pub p_layer_info: *mut VkLayerInstanceLink,
    pub pfn_set_instance_loader_data: *mut c_void,
}

/// `VkLayerInstanceCreateInfo` — found in `VkInstanceCreateInfo::pNext`.
#[repr(C)]
pub struct VkLayerInstanceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: i32,
    pub u: VkLayerInstanceCreateInfoUnion,
}

/// One link in the device dispatch chain.
#[repr(C)]
pub struct VkLayerDeviceLink {
    pub p_next: *mut VkLayerDeviceLink,
    pub pfn_next_get_instance_proc_addr: PfnGetInstanceProcAddr,
    pub pfn_next_get_device_proc_addr: PfnGetDeviceProcAddr,
}

#[repr(C)]
pub union VkLayerDeviceCreateInfoUnion {
    pub p_layer_info: *mut VkLayerDeviceLink,
    pub pfn_set_device_loader_data: *mut c_void,
}

/// `VkLayerDeviceCreateInfo` — found in `VkDeviceCreateInfo::pNext`.
#[repr(C)]
pub struct VkLayerDeviceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: i32,
    pub u: VkLayerDeviceCreateInfoUnion,
}

/// `VkNegotiateLayerInterface` — the struct the loader hands to
/// `vkNegotiateLoaderLayerInterfaceVersion` for us to fill in.
#[repr(C)]
pub struct VkNegotiateLayerInterface {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub loader_layer_interface_version: u32,
    pub pfn_get_instance_proc_addr: Option<PfnGetInstanceProcAddr>,
    pub pfn_get_device_proc_addr: Option<PfnGetDeviceProcAddr>,
    pub pfn_get_physical_device_proc_addr: Option<PfnGetPhysicalDeviceProcAddr>,
}
