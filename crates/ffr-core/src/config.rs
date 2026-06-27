//! TOML configuration: a single default profile that works for any headset,
//! plus optional per-HMD overrides for hand-tuning. Overrides are NEVER
//! required for correctness — they only adjust aggressiveness.

use serde::Deserialize;

use crate::wire::{FalloffParams, RadiusBasis, RateCode};

/// Top-level config file.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub default: Profile,
    /// Optional per-HMD overrides, matched on a substring of the system name.
    #[serde(default)]
    pub hmd: Vec<HmdOverride>,
}

impl Config {
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Resolve the falloff for a headset by name: the default profile, with the
    /// first matching `[[hmd]]` override applied on top.
    pub fn resolve_for(&self, system_name: &str) -> FalloffParams {
        let mut profile = self.default.clone();
        if let Some(o) = self
            .hmd
            .iter()
            .find(|o| system_name.contains(&o.match_system_name))
        {
            o.apply_to(&mut profile);
        }
        profile.to_params()
    }
}

/// A complete falloff profile (string-typed for human-friendly TOML).
#[derive(Clone, Debug, Deserialize)]
pub struct Profile {
    pub inner_radius: f32,
    pub mid_radius: f32,
    pub outer_radius: f32,
    pub rate_inner: String,
    pub rate_mid: String,
    pub rate_outer: String,
    pub rate_edge: String,
    pub radius_basis: String,
    pub vertical_scale: f32,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            inner_radius: 0.30,
            mid_radius: 0.55,
            outer_radius: 0.80,
            rate_inner: "1x1".into(),
            rate_mid: "2x2".into(),
            rate_outer: "4x4".into(),
            rate_edge: "4x4".into(),
            radius_basis: "diagonal".into(),
            vertical_scale: 1.0,
        }
    }
}

impl Profile {
    pub fn to_params(&self) -> FalloffParams {
        let rate = |s: &str, fallback: RateCode| RateCode::parse(s).unwrap_or(fallback) as u8;
        let basis = match self.radius_basis.as_str() {
            "horizontal" => RadiusBasis::Horizontal,
            "vertical" => RadiusBasis::Vertical,
            _ => RadiusBasis::Diagonal,
        };
        FalloffParams {
            inner_radius: self.inner_radius,
            mid_radius: self.mid_radius,
            outer_radius: self.outer_radius,
            rate_inner: rate(&self.rate_inner, RateCode::R1x1),
            rate_mid: rate(&self.rate_mid, RateCode::R2x2),
            rate_outer: rate(&self.rate_outer, RateCode::R4x4),
            rate_edge: rate(&self.rate_edge, RateCode::R4x4),
            radius_basis: basis as u8,
            _pad: [0; 3],
            vertical_scale: self.vertical_scale,
        }
    }
}

/// Optional per-HMD override. Every field is optional; unset fields fall back
/// to the default profile.
#[derive(Clone, Debug, Deserialize)]
pub struct HmdOverride {
    pub match_system_name: String,
    pub inner_radius: Option<f32>,
    pub mid_radius: Option<f32>,
    pub outer_radius: Option<f32>,
    pub rate_inner: Option<String>,
    pub rate_mid: Option<String>,
    pub rate_outer: Option<String>,
    pub rate_edge: Option<String>,
    pub radius_basis: Option<String>,
    pub vertical_scale: Option<f32>,
}

impl HmdOverride {
    fn apply_to(&self, p: &mut Profile) {
        if let Some(v) = self.inner_radius {
            p.inner_radius = v;
        }
        if let Some(v) = self.mid_radius {
            p.mid_radius = v;
        }
        if let Some(v) = self.outer_radius {
            p.outer_radius = v;
        }
        if let Some(v) = &self.rate_inner {
            p.rate_inner = v.clone();
        }
        if let Some(v) = &self.rate_mid {
            p.rate_mid = v.clone();
        }
        if let Some(v) = &self.rate_outer {
            p.rate_outer = v.clone();
        }
        if let Some(v) = &self.rate_edge {
            p.rate_edge = v.clone();
        }
        if let Some(v) = &self.radius_basis {
            p.radius_basis = v.clone();
        }
        if let Some(v) = self.vertical_scale {
            p.vertical_scale = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses_and_resolves() {
        let cfg = Config::default();
        let p = cfg.resolve_for("Anything XR");
        assert_eq!(p, FalloffParams::default());
    }

    #[test]
    fn override_applies_by_substring() {
        let toml = r#"
            [default]
            inner_radius = 0.30
            mid_radius = 0.55
            outer_radius = 0.80
            rate_inner = "1x1"
            rate_mid = "2x2"
            rate_outer = "4x4"
            rate_edge = "4x4"
            radius_basis = "diagonal"
            vertical_scale = 1.0

            [[hmd]]
            match_system_name = "Pimax"
            outer_radius = 0.85
            rate_mid = "2x4"
        "#;
        let cfg = Config::from_toml(toml).unwrap();
        let generic = cfg.resolve_for("Valve Index");
        assert_eq!(generic.outer_radius, 0.80);
        let pimax = cfg.resolve_for("Pimax 5K XR");
        assert_eq!(pimax.outer_radius, 0.85);
        assert_eq!(pimax.rate_mid, RateCode::R2x4 as u8);
        // Unset override fields keep the default.
        assert_eq!(pimax.inner_radius, 0.30);
    }
}
