//! PPD and optical-center math derived purely from the per-eye field of view.
//!
//! Mirrors OpenXR's `XrFovf`: four **signed** half-angles in radians, where
//! `left`/`down` are typically negative and `right`/`up` positive. A symmetric
//! frustum yields an optical center of exactly (0.5, 0.5); an asymmetric/canted
//! frustum (e.g. a wide-FOV headset with canted panels) yields an offset center.

use std::f32::consts::PI;

/// Per-eye field of view: signed half-angles in radians (OpenXR `XrFovf`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fov {
    pub left: f32,
    pub right: f32,
    pub up: f32,
    pub down: f32,
}

impl Fov {
    pub fn new(left: f32, right: f32, up: f32, down: f32) -> Self {
        Self {
            left,
            right,
            up,
            down,
        }
    }

    /// A symmetric frustum with the given half-FOV (radians) on each axis.
    pub fn symmetric(half_h: f32, half_v: f32) -> Self {
        Self {
            left: -half_h,
            right: half_h,
            up: half_v,
            down: -half_v,
        }
    }

    #[inline]
    pub fn tan_left(&self) -> f32 {
        self.left.tan()
    }
    #[inline]
    pub fn tan_right(&self) -> f32 {
        self.right.tan()
    }
    #[inline]
    pub fn tan_up(&self) -> f32 {
        self.up.tan()
    }
    #[inline]
    pub fn tan_down(&self) -> f32 {
        self.down.tan()
    }

    /// Horizontal tangent extent (`tanR - tanL`), always positive for a valid frustum.
    #[inline]
    pub fn tan_width(&self) -> f32 {
        self.tan_right() - self.tan_left()
    }

    /// Vertical tangent extent (`tanU - tanD`), always positive for a valid frustum.
    #[inline]
    pub fn tan_height(&self) -> f32 {
        self.tan_up() - self.tan_down()
    }

    /// Total horizontal FOV in degrees.
    pub fn h_fov_deg(&self) -> f32 {
        (self.right - self.left) * 180.0 / PI
    }

    /// Total vertical FOV in degrees.
    pub fn v_fov_deg(&self) -> f32 {
        (self.up - self.down) * 180.0 / PI
    }

    /// On-screen location of the optical axis (view-space forward `(0,0,-1)`),
    /// in normalized texture coords with origin top-left, `y` growing downward.
    /// (0.5, 0.5) for a symmetric frustum; offset otherwise.
    pub fn optical_center(&self) -> OpticalCenter {
        let u = -self.tan_left() / self.tan_width();
        // v flips because pixel y grows downward while `up` is positive.
        let v = self.tan_up() / self.tan_height();
        OpticalCenter { u, v }
    }

    /// Pixels-per-degree for the given render resolution. Reports the on-axis
    /// (local, at the optical center) density and the whole-FOV average, per
    /// axis. NOTE: for a rectilinear projection the on-axis density is the
    /// *minimum* across the FOV — pixels stretch toward the periphery, so the
    /// edges are denser. That peripheral density is exactly what foveation
    /// reclaims. `center_*` is the right factor for converting small angular
    /// offsets near the axis into pixels (ring placement).
    pub fn ppd(&self, width_px: u32, height_px: u32) -> Ppd {
        let w = width_px as f32;
        let h = height_px as f32;
        let deg = PI / 180.0;
        Ppd {
            center_h: (w / self.tan_width()) * deg,
            center_v: (h / self.tan_height()) * deg,
            avg_h: w / self.h_fov_deg(),
            avg_v: h / self.v_fov_deg(),
        }
    }
}

/// Optical-axis location within the render target, normalized to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticalCenter {
    pub u: f32,
    pub v: f32,
}

/// Pixels-per-degree, peak (center) and whole-FOV average, per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ppd {
    pub center_h: f32,
    pub center_v: f32,
    pub avg_h: f32,
    pub avg_v: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg(d: f32) -> f32 {
        d * PI / 180.0
    }

    #[test]
    fn symmetric_center_is_middle() {
        let fov = Fov::symmetric(deg(50.0), deg(45.0));
        let c = fov.optical_center();
        assert!((c.u - 0.5).abs() < 1e-6, "u={}", c.u);
        assert!((c.v - 0.5).abs() < 1e-6, "v={}", c.v);
    }

    #[test]
    fn asymmetric_canted_center_is_offset() {
        // Pimax-like: wider toward the temporal side, so the optical axis sits
        // off-center. left=-60deg, right=45deg.
        let fov = Fov::new(deg(-60.0), deg(45.0), deg(45.0), deg(-45.0));
        let c = fov.optical_center();
        assert!(c.u > 0.5, "expected offset center u>0.5, got {}", c.u);
        // Vertically symmetric here, so v stays centered.
        assert!((c.v - 0.5).abs() < 1e-6, "v={}", c.v);
    }

    #[test]
    fn onaxis_ppd_below_average_for_rectilinear() {
        // Rectilinear projection: the on-axis density is the minimum; the
        // periphery is denser, so on-axis < whole-FOV average.
        let fov = Fov::symmetric(deg(55.0), deg(50.0));
        let ppd = fov.ppd(2000, 2000);
        assert!(
            ppd.center_h < ppd.avg_h,
            "center_h={} avg_h={}",
            ppd.center_h,
            ppd.avg_h
        );
        assert!(
            ppd.center_v < ppd.avg_v,
            "center_v={} avg_v={}",
            ppd.center_v,
            ppd.avg_v
        );
    }

    #[test]
    fn ppd_scales_with_resolution() {
        let fov = Fov::symmetric(deg(50.0), deg(50.0));
        let lo = fov.ppd(1000, 1000);
        let hi = fov.ppd(2000, 2000);
        assert!((hi.avg_h - 2.0 * lo.avg_h).abs() < 1e-3);
    }
}
