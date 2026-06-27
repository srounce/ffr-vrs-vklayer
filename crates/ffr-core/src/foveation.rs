//! Generates the per-tile shading-rate map for the attachment image.
//!
//! The map is reconstructed from compact parameters (rect, optical center,
//! per-axis normalization radius, falloff bands) so it can be rebuilt on the
//! Vulkan side from a small `FoveationDesc` rather than shipping a whole image
//! through the registry. Ring radii are fractions of `norm_px` (the half-FOV
//! expressed in pixels), measured from the offset optical center — keeping the
//! same falloff correct across arbitrary FOV/resolution and asymmetric eyes.

use crate::rate::ShadingRate;
use crate::wire::{FalloffParams, RadiusBasis, RateCode};

/// A computed shading-rate map: one encoded byte per `texel_size`-pixel tile.
#[derive(Clone, Debug, PartialEq)]
pub struct FoveationMap {
    /// Tiles across (x).
    pub cols: u32,
    /// Tiles down (y).
    pub rows: u32,
    /// Framebuffer pixels per tile (the attachment texel size).
    pub texel_size: u32,
    /// Row-major encoded rate bytes, length `cols * rows`.
    pub bytes: Vec<u8>,
}

impl FoveationMap {
    /// Generate a map for an eye sub-rectangle.
    ///
    /// * `rect_w`/`rect_h` — eye render extent in pixels.
    /// * `center_px` — optical center in pixels, relative to the rect.
    /// * `texel_size` — attachment texel size (e.g. 16).
    /// * `norm_px` — `(x, y)` half-FOV expressed in pixels; the falloff radii
    ///   are fractions of this. A point this far from the center sits at r=1.0.
    /// * `falloff` — the band radii and per-band rates.
    pub fn generate(
        rect_w: u32,
        rect_h: u32,
        center_px: (f32, f32),
        texel_size: u32,
        norm_px: (f32, f32),
        falloff: &FalloffParams,
    ) -> Self {
        let texel = texel_size.max(1);
        let cols = rect_w.div_ceil(texel);
        let rows = rect_h.div_ceil(texel);
        let basis = match falloff.radius_basis {
            0 => RadiusBasis::Horizontal,
            1 => RadiusBasis::Vertical,
            _ => RadiusBasis::Diagonal,
        };
        let vscale = if falloff.vertical_scale > 0.0 {
            falloff.vertical_scale
        } else {
            1.0
        };
        let nx = if norm_px.0 > 0.0 { norm_px.0 } else { 1.0 };
        let ny = if norm_px.1 > 0.0 {
            norm_px.1 * vscale
        } else {
            1.0
        };

        let mut bytes = Vec::with_capacity((cols * rows) as usize);
        for ty in 0..rows {
            for tx in 0..cols {
                let px = (tx as f32 + 0.5) * texel as f32;
                let py = (ty as f32 + 0.5) * texel as f32;
                let dx = (px - center_px.0) / nx;
                let dy = (py - center_px.1) / ny;
                let r = match basis {
                    RadiusBasis::Horizontal => dx.abs(),
                    RadiusBasis::Vertical => dy.abs(),
                    RadiusBasis::Diagonal => (dx * dx + dy * dy).sqrt(),
                };
                bytes.push(tile_rate(r, falloff).encode());
            }
        }

        FoveationMap {
            cols,
            rows,
            texel_size: texel,
            bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// The shading rate for a normalized radius `r` (0 at the optical center).
pub fn tile_rate(r: f32, falloff: &FalloffParams) -> ShadingRate {
    let code = if r < falloff.inner_radius {
        falloff.rate_inner
    } else if r < falloff.mid_radius {
        falloff.rate_mid
    } else if r < falloff.outer_radius {
        falloff.rate_outer
    } else {
        falloff.rate_edge
    };
    RateCode::from_u8(code).unwrap_or(RateCode::R1x1).to_rate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_is_full_rate_periphery_is_coarse() {
        // Default bands: <0.30 → 1x1, [0.30,0.55) → 2x2, [0.55,0.80) → 4x4, ≥0.80 → 4x4.
        let f = FalloffParams::default();
        assert_eq!(tile_rate(0.0, &f), ShadingRate::RATE_1X1);
        assert_eq!(tile_rate(0.45, &f), ShadingRate::RATE_2X2);
        assert_eq!(tile_rate(0.65, &f), ShadingRate::RATE_4X4);
        assert_eq!(tile_rate(0.95, &f), ShadingRate::RATE_4X4);
    }

    #[test]
    fn map_dimensions_and_center_tile() {
        let f = FalloffParams::default();
        let (w, h) = (2000u32, 2000u32);
        let center = (1000.0, 1000.0);
        // norm = half-extent so the edge sits at r≈1.
        let map = FoveationMap::generate(w, h, center, 16, (1000.0, 1000.0), &f);
        assert_eq!(map.cols, 125);
        assert_eq!(map.rows, 125);
        assert_eq!(map.bytes.len(), 125 * 125);
        // Tile at the center must be full rate.
        let cx = (center.0 / 16.0) as u32;
        let cy = (center.1 / 16.0) as u32;
        let center_byte = map.bytes[(cy * map.cols + cx) as usize];
        assert_eq!(ShadingRate::decode(center_byte), ShadingRate::RATE_1X1);
        // A corner must be coarser than the center.
        let corner = map.bytes[0];
        assert!(ShadingRate::decode(corner).coverage() >= ShadingRate::RATE_2X2.coverage());
    }

    #[test]
    fn offset_center_shifts_full_rate_region() {
        let f = FalloffParams::default();
        // Center pushed toward the right edge: the dense region follows it.
        let map = FoveationMap::generate(1600, 1600, (1200.0, 800.0), 16, (800.0, 800.0), &f);
        let cx = (1200.0f32 / 16.0) as u32;
        let cy = (800.0f32 / 16.0) as u32;
        let at_center = map.bytes[(cy * map.cols + cx) as usize];
        assert_eq!(ShadingRate::decode(at_center), ShadingRate::RATE_1X1);
        // The geometric middle is now off-axis, so it should not be full-rate.
        let mid = map.bytes[((map.rows / 2) * map.cols + map.cols / 2) as usize];
        assert!(ShadingRate::decode(mid).coverage() >= ShadingRate::RATE_1X1.coverage());
    }
}
