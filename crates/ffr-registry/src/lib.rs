//! Client for the shared registry that lives in `libffr_shared.so`.
//!
//! Both the OpenXR and Vulkan layers statically link this crate, but it holds
//! no registry state of its own — it `dlopen`s the single `libffr_shared.so`
//! (deduped by soname, loaded once per process) and calls its C ABI. That is
//! what makes the state genuinely shared across the two cdylibs.
//!
//! The library is found via the dynamic loader's normal search (we rely on
//! `LD_LIBRARY_PATH` / the layers' RPATH pointing at the install `lib/` dir).

use std::sync::OnceLock;

use ffr_core::wire::FoveationDesc;
use libloading::Library;

type FnVersion = unsafe extern "C" fn() -> u32;
type FnPublish = unsafe extern "C" fn(*const FoveationDesc);
type FnLookup = unsafe extern "C" fn(u64, u64, *mut FoveationDesc, u32) -> u32;
type FnRemoveImage = unsafe extern "C" fn(u64, u64);
type FnRemoveDevice = unsafe extern "C" fn(u64);
type FnSetHeartbeat = unsafe extern "C" fn(u64, f32);
type FnGetHeartbeat = unsafe extern "C" fn(*mut u64, *mut f32) -> u32;

struct Api {
    // Keep the library mapped for the process lifetime; the fn pointers below
    // borrow nothing once copied out, but remain valid only while it is loaded.
    _lib: Library,
    version: FnVersion,
    publish: FnPublish,
    lookup: FnLookup,
    remove_image: FnRemoveImage,
    remove_device: FnRemoveDevice,
    set_heartbeat: FnSetHeartbeat,
    get_heartbeat: FnGetHeartbeat,
}

static API: OnceLock<Option<Api>> = OnceLock::new();

fn api() -> Option<&'static Api> {
    API.get_or_init(|| unsafe { load() }).as_ref()
}

unsafe fn load() -> Option<Api> {
    let lib = Library::new("libffr_shared.so").ok()?;
    macro_rules! sym {
        ($t:ty, $name:expr) => {
            *lib.get::<$t>($name).ok()?
        };
    }
    let api = Api {
        version: sym!(FnVersion, b"ffr_shared_version\0"),
        publish: sym!(FnPublish, b"ffr_shared_publish\0"),
        lookup: sym!(FnLookup, b"ffr_shared_lookup\0"),
        remove_image: sym!(FnRemoveImage, b"ffr_shared_remove_image\0"),
        remove_device: sym!(FnRemoveDevice, b"ffr_shared_remove_device\0"),
        set_heartbeat: sym!(FnSetHeartbeat, b"ffr_shared_set_heartbeat\0"),
        get_heartbeat: sym!(FnGetHeartbeat, b"ffr_shared_get_heartbeat\0"),
        _lib: lib,
    };
    Some(api)
}

/// Whether `libffr_shared.so` was found and its symbols resolved.
pub fn is_available() -> bool {
    api().is_some()
}

/// The shared library's ABI version, if loaded.
pub fn version() -> Option<u32> {
    api().map(|a| unsafe { (a.version)() })
}

/// Publish a heartbeat (diagnostic: proves the cross-dylib channel).
pub fn set_heartbeat(counter: u64, ppd: f32) {
    if let Some(a) = api() {
        unsafe { (a.set_heartbeat)(counter, ppd) }
    }
}

/// Read the heartbeat, if one has been published.
pub fn get_heartbeat() -> Option<(u64, f32)> {
    let a = api()?;
    let mut counter = 0u64;
    let mut ppd = 0f32;
    if unsafe { (a.get_heartbeat)(&mut counter, &mut ppd) } != 0 {
        Some((counter, ppd))
    } else {
        None
    }
}

/// Publish a foveation descriptor.
pub fn publish(desc: &FoveationDesc) {
    if let Some(a) = api() {
        unsafe { (a.publish)(desc) }
    }
}

/// Look up descriptors for a `(device, image)` (0, 1, or 2 — double-wide).
pub fn lookup(device: u64, image: u64) -> Vec<FoveationDesc> {
    let Some(a) = api() else {
        return Vec::new();
    };
    // Safety: FoveationDesc is plain POD; a zeroed buffer is a valid scratch.
    let mut buf: [FoveationDesc; 2] = unsafe { std::mem::zeroed() };
    let n = unsafe { (a.lookup)(device, image, buf.as_mut_ptr(), 2) } as usize;
    buf[..n.min(2)].to_vec()
}

/// Remove all descriptors for a `(device, image)`.
pub fn remove_image(device: u64, image: u64) {
    if let Some(a) = api() {
        unsafe { (a.remove_image)(device, image) }
    }
}

/// Remove all descriptors for a device.
pub fn remove_device(device: u64) {
    if let Some(a) = api() {
        unsafe { (a.remove_device)(device) }
    }
}
