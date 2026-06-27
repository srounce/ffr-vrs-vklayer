//! `ffr-core`: pure, stateless foundation for the FFR/VRS layers.
//!
//! Design tenet: **fully HMD-agnostic**. Everything here is derived generically
//! from an OpenXR-style field-of-view (`XrFovf`-equivalent angles) plus the
//! runtime-recommended resolution. No headset is special-cased. This crate has
//! no graphics/XR dependencies so it can be exhaustively unit-tested without
//! hardware.

pub mod config;
pub mod foveation;
pub mod math;
pub mod rate;
pub mod wire;

pub use config::{Config, HmdOverride, Profile};
pub use foveation::{tile_rate, FoveationMap};
pub use math::{Fov, OpticalCenter, Ppd};
pub use rate::ShadingRate;
pub use wire::{FalloffParams, FoveationDesc, RateCode, FFR_SHARED_MAGIC, FFR_SHARED_VERSION};
