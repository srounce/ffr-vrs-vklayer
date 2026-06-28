//! `#[repr(C)]` POD wire types shared across the `ffr-shared` C ABI boundary.
//!
//! Only plain-old-data crosses the boundary (no `String`/`Vec`), so the same
//! layout can later back a POSIX shared-memory segment unchanged. Every record
//! carries `magic`/`version` so a consumer can reject a mismatched producer.

use crate::rate::ShadingRate;

/// Identifies a valid `FoveationDesc` ("FFR1" little-endian).
pub const FFR_SHARED_MAGIC: u32 = 0x3152_4646;
/// Bump whenever the wire layout changes.
pub const FFR_SHARED_VERSION: u32 = 2;

/// Compact rate identifier stored in POD structs (avoids enum-transmute UB).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateCode {
    R1x1 = 0,
    R1x2 = 1,
    R2x1 = 2,
    R2x2 = 3,
    R2x4 = 4,
    R4x2 = 5,
    R4x4 = 6,
}

impl RateCode {
    pub fn to_rate(self) -> ShadingRate {
        match self {
            RateCode::R1x1 => ShadingRate::RATE_1X1,
            RateCode::R1x2 => ShadingRate::RATE_1X2,
            RateCode::R2x1 => ShadingRate::RATE_2X1,
            RateCode::R2x2 => ShadingRate::RATE_2X2,
            RateCode::R2x4 => ShadingRate::RATE_2X4,
            RateCode::R4x2 => ShadingRate::RATE_4X2,
            RateCode::R4x4 => ShadingRate::RATE_4X4,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => RateCode::R1x1,
            1 => RateCode::R1x2,
            2 => RateCode::R2x1,
            3 => RateCode::R2x2,
            4 => RateCode::R2x4,
            5 => RateCode::R4x2,
            6 => RateCode::R4x4,
            _ => return None,
        })
    }

    /// Parse a config string like `"2x2"`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "1x1" => RateCode::R1x1,
            "1x2" => RateCode::R1x2,
            "2x1" => RateCode::R2x1,
            "2x2" => RateCode::R2x2,
            "2x4" => RateCode::R2x4,
            "4x2" => RateCode::R4x2,
            "4x4" => RateCode::R4x4,
            _ => return None,
        })
    }
}

/// Which axis the ring radii are measured against.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadiusBasis {
    Horizontal = 0,
    Vertical = 1,
    Diagonal = 2,
}

/// Static radial-falloff parameters. Radii are fractions of the chosen
/// half-FOV basis, measured from the optical center — so one set of values is
/// correct across arbitrary FOV/resolution and on asymmetric eyes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FalloffParams {
    pub inner_radius: f32,
    pub mid_radius: f32,
    pub outer_radius: f32,
    /// Rate codes (`RateCode as u8`) for each band.
    pub rate_inner: u8,
    pub rate_mid: u8,
    pub rate_outer: u8,
    pub rate_edge: u8,
    /// `RadiusBasis as u8`.
    pub radius_basis: u8,
    pub _pad: [u8; 3],
    /// Scale applied to the vertical radius (panels are wider than tall).
    pub vertical_scale: f32,
}

impl Default for FalloffParams {
    fn default() -> Self {
        Self {
            inner_radius: 0.30,
            mid_radius: 0.55,
            outer_radius: 0.80,
            rate_inner: RateCode::R1x1 as u8,
            rate_mid: RateCode::R2x2 as u8,
            rate_outer: RateCode::R4x4 as u8,
            rate_edge: RateCode::R4x4 as u8,
            radius_basis: RadiusBasis::Diagonal as u8,
            _pad: [0; 3],
            vertical_scale: 1.0,
        }
    }
}

/// One published foveation descriptor for a specific eye view of a specific
/// `VkImage`. Double-wide swapchains publish two descriptors per image.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoveationDesc {
    pub magic: u32,
    pub version: u32,
    pub vk_device: u64,
    pub vk_image: u64,
    pub image_array_index: u32,
    pub eye: u32,
    /// The eye image's `VkFormat` (raw `i32`), for building an overlay pipeline.
    pub vk_format: i32,
    pub _pad2: u32,
    /// Sub-rectangle of the image this eye renders into (handles double-wide).
    pub rect_x: i32,
    pub rect_y: i32,
    pub rect_w: u32,
    pub rect_h: u32,
    /// Optical center in pixels, relative to the full image (rect offset added).
    pub center_px_x: f32,
    pub center_px_y: f32,
    pub ppd_center_h: f32,
    pub ppd_center_v: f32,
    pub falloff: FalloffParams,
    /// Monotonic counter so consumers can cheaply detect updates.
    pub generation: u64,
}

impl FoveationDesc {
    pub fn is_valid(&self) -> bool {
        self.magic == FFR_SHARED_MAGIC && self.version == FFR_SHARED_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falloff_is_pod_sized() {
        // No surprises in size; layout is stable for the ABI / future shm use.
        assert_eq!(std::mem::align_of::<FalloffParams>(), 4);
    }

    #[test]
    fn rate_code_string_roundtrip() {
        assert_eq!(RateCode::parse("4x4"), Some(RateCode::R4x4));
        assert_eq!(
            RateCode::parse("2x2").unwrap().to_rate(),
            ShadingRate::RATE_2X2
        );
        assert_eq!(RateCode::parse("nope"), None);
    }

    #[test]
    fn magic_validates() {
        let d = FoveationDesc {
            magic: FFR_SHARED_MAGIC,
            version: FFR_SHARED_VERSION,
            vk_device: 1,
            vk_image: 2,
            image_array_index: 0,
            eye: 0,
            vk_format: 0,
            _pad2: 0,
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
        };
        assert!(d.is_valid());
    }
}
