//! `ffr-shared`: the single process-global registry shared by the OpenXR and
//! Vulkan layers, exposed over a tiny C ABI.
//!
//! Why a standalone cdylib: if each layer statically linked a crate containing
//! a `static REGISTRY`, every cdylib would get its *own* copy and they'd never
//! see each other's data. By living in exactly one dynamic object (soname-
//! deduped, loaded once), this registry is genuinely shared in-process.
//!
//! Only `#[repr(C)]` POD (`FoveationDesc`) crosses the boundary, so the backing
//! store can later be swapped for POSIX shared memory without ABI changes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use ffr_core::wire::{FoveationDesc, FFR_SHARED_VERSION};
use once_cell::sync::Lazy;

/// Keyed by `(vk_device, vk_image)`. Up to two descriptors per image
/// (double-wide swapchains pack both eyes into one image).
type Registry = HashMap<(u64, u64), Vec<FoveationDesc>>;

static REGISTRY: Lazy<Mutex<Registry>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// ABI/layout version. Consumers compare against their own `FFR_SHARED_VERSION`.
#[no_mangle]
pub extern "C" fn ffr_shared_version() -> u32 {
    FFR_SHARED_VERSION
}

// --- Heartbeat: a tiny diagnostic channel proving cross-dylib sharing ---
// The OpenXR layer writes a (counter, ppd); the Vulkan layer reads it back.
// Both bind to this single `.so`, so they observe the same values.

static HB_SET: AtomicBool = AtomicBool::new(false);
static HB_COUNTER: AtomicU64 = AtomicU64::new(0);
static HB_PPD_BITS: AtomicU32 = AtomicU32::new(0);

/// Publish a heartbeat (a monotonic counter + a sample PPD value).
#[no_mangle]
pub extern "C" fn ffr_shared_set_heartbeat(counter: u64, ppd: f32) {
    HB_COUNTER.store(counter, Ordering::Relaxed);
    HB_PPD_BITS.store(ppd.to_bits(), Ordering::Relaxed);
    HB_SET.store(true, Ordering::Release);
}

/// Read the heartbeat. Returns 1 and fills `out_counter`/`out_ppd` if one has
/// been published, else 0.
///
/// # Safety
/// `out_counter` and `out_ppd` must be valid, writable pointers.
#[no_mangle]
pub unsafe extern "C" fn ffr_shared_get_heartbeat(
    out_counter: *mut u64,
    out_ppd: *mut f32,
) -> u32 {
    if !HB_SET.load(Ordering::Acquire) {
        return 0;
    }
    if !out_counter.is_null() {
        *out_counter = HB_COUNTER.load(Ordering::Relaxed);
    }
    if !out_ppd.is_null() {
        *out_ppd = f32::from_bits(HB_PPD_BITS.load(Ordering::Relaxed));
    }
    1
}

/// Publish (insert or replace) a foveation descriptor for its `(device, image)`
/// — replacing any existing entry for the same `eye`.
///
/// # Safety
/// `desc` must point to a valid, initialized `FoveationDesc`.
#[no_mangle]
pub unsafe extern "C" fn ffr_shared_publish(desc: *const FoveationDesc) {
    if desc.is_null() {
        return;
    }
    let desc = unsafe { *desc };
    if !desc.is_valid() {
        return;
    }
    let mut reg = match REGISTRY.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let entry = reg.entry((desc.vk_device, desc.vk_image)).or_default();
    if let Some(slot) = entry.iter_mut().find(|d| d.eye == desc.eye) {
        *slot = desc;
    } else {
        entry.push(desc);
    }
}

/// Look up descriptors for a `(device, image)`. Writes up to `max` descriptors
/// into `out` and returns the number written (0 means "no foveation — pass the
/// image through unmodified").
///
/// # Safety
/// `out` must point to space for at least `max` `FoveationDesc` values.
#[no_mangle]
pub unsafe extern "C" fn ffr_shared_lookup(
    vk_device: u64,
    vk_image: u64,
    out: *mut FoveationDesc,
    max: u32,
) -> u32 {
    if out.is_null() || max == 0 {
        return 0;
    }
    let reg = match REGISTRY.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(entry) = reg.get(&(vk_device, vk_image)) else {
        return 0;
    };
    let n = entry.len().min(max as usize);
    for (i, d) in entry.iter().take(n).enumerate() {
        unsafe { *out.add(i) = *d };
    }
    n as u32
}

/// Remove all descriptors for a `(device, image)` (e.g. on swapchain destroy).
#[no_mangle]
pub extern "C" fn ffr_shared_remove_image(vk_device: u64, vk_image: u64) {
    let mut reg = match REGISTRY.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    reg.remove(&(vk_device, vk_image));
}

/// Remove every entry for a device (e.g. on `vkDestroyDevice` / session end).
#[no_mangle]
pub extern "C" fn ffr_shared_remove_device(vk_device: u64) {
    let mut reg = match REGISTRY.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    reg.retain(|(dev, _), _| *dev != vk_device);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffr_core::wire::{FalloffParams, FFR_SHARED_MAGIC};

    fn desc(device: u64, image: u64, eye: u32) -> FoveationDesc {
        FoveationDesc {
            magic: FFR_SHARED_MAGIC,
            version: FFR_SHARED_VERSION,
            vk_device: device,
            vk_image: image,
            image_array_index: eye,
            eye,
            rect_x: 0,
            rect_y: 0,
            rect_w: 100,
            rect_h: 100,
            center_px_x: 50.0,
            center_px_y: 50.0,
            ppd_center_h: 20.0,
            ppd_center_v: 20.0,
            falloff: FalloffParams::default(),
            generation: 1,
        }
    }

    #[test]
    fn publish_then_lookup_roundtrips() {
        let dev = 0xAAAA;
        let img = 0xBBBB;
        unsafe { ffr_shared_publish(&desc(dev, img, 0)) };
        unsafe { ffr_shared_publish(&desc(dev, img, 1)) };
        let mut out = [desc(0, 0, 0); 2];
        let n = unsafe { ffr_shared_lookup(dev, img, out.as_mut_ptr(), 2) };
        assert_eq!(n, 2);
        ffr_shared_remove_image(dev, img);
        let n2 = unsafe { ffr_shared_lookup(dev, img, out.as_mut_ptr(), 2) };
        assert_eq!(n2, 0);
    }

    #[test]
    fn republish_same_eye_replaces() {
        let (dev, img) = (1, 2);
        let mut d = desc(dev, img, 0);
        unsafe { ffr_shared_publish(&d) };
        d.generation = 99;
        unsafe { ffr_shared_publish(&d) };
        let mut out = [desc(0, 0, 0); 2];
        let n = unsafe { ffr_shared_lookup(dev, img, out.as_mut_ptr(), 2) };
        assert_eq!(n, 1);
        assert_eq!(out[0].generation, 99);
        ffr_shared_remove_device(dev);
    }
}
