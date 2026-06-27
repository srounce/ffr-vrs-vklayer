//! Fragment shading rates and their `R8_UINT` attachment encoding.
//!
//! `VK_KHR_fragment_shading_rate` encodes a per-tile rate in a single byte:
//! `(log2(width) << 2) | log2(height)`. Width/height are the coarse fragment
//! size in pixels and must be powers of two in `{1, 2, 4}`.

/// A fragment shading rate: coarse fragment size `width` x `height` (in pixels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShadingRate {
    pub width: u8,
    pub height: u8,
}

impl ShadingRate {
    pub const RATE_1X1: ShadingRate = ShadingRate {
        width: 1,
        height: 1,
    };
    pub const RATE_1X2: ShadingRate = ShadingRate {
        width: 1,
        height: 2,
    };
    pub const RATE_2X1: ShadingRate = ShadingRate {
        width: 2,
        height: 1,
    };
    pub const RATE_2X2: ShadingRate = ShadingRate {
        width: 2,
        height: 2,
    };
    pub const RATE_2X4: ShadingRate = ShadingRate {
        width: 2,
        height: 4,
    };
    pub const RATE_4X2: ShadingRate = ShadingRate {
        width: 4,
        height: 2,
    };
    pub const RATE_4X4: ShadingRate = ShadingRate {
        width: 4,
        height: 4,
    };

    pub const fn new(width: u8, height: u8) -> Self {
        Self { width, height }
    }

    /// Encode to the `R8_UINT` attachment byte: `(log2(w) << 2) | log2(h)`.
    pub fn encode(self) -> u8 {
        (log2_u8(self.width) << 2) | log2_u8(self.height)
    }

    /// Decode an attachment byte back into a rate.
    pub fn decode(byte: u8) -> Self {
        let w = 1u8 << ((byte >> 2) & 0b11);
        let h = 1u8 << (byte & 0b11);
        Self {
            width: w,
            height: h,
        }
    }

    /// The number of pixels covered by one coarse fragment (the shading-cost
    /// reduction factor vs 1x1).
    pub fn coverage(self) -> u32 {
        self.width as u32 * self.height as u32
    }
}

#[inline]
fn log2_u8(v: u8) -> u8 {
    match v {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_known_values() {
        assert_eq!(ShadingRate::RATE_1X1.encode(), 0x00);
        assert_eq!(ShadingRate::RATE_1X2.encode(), 0x01);
        assert_eq!(ShadingRate::RATE_2X1.encode(), 0x04);
        assert_eq!(ShadingRate::RATE_2X2.encode(), 0x05);
        assert_eq!(ShadingRate::RATE_2X4.encode(), 0x06);
        assert_eq!(ShadingRate::RATE_4X2.encode(), 0x09);
        assert_eq!(ShadingRate::RATE_4X4.encode(), 0x0A);
    }

    #[test]
    fn encode_decode_roundtrip() {
        for r in [
            ShadingRate::RATE_1X1,
            ShadingRate::RATE_1X2,
            ShadingRate::RATE_2X1,
            ShadingRate::RATE_2X2,
            ShadingRate::RATE_2X4,
            ShadingRate::RATE_4X2,
            ShadingRate::RATE_4X4,
        ] {
            assert_eq!(ShadingRate::decode(r.encode()), r);
        }
    }

    #[test]
    fn coverage_factors() {
        assert_eq!(ShadingRate::RATE_1X1.coverage(), 1);
        assert_eq!(ShadingRate::RATE_2X2.coverage(), 4);
        assert_eq!(ShadingRate::RATE_4X4.coverage(), 16);
    }
}
