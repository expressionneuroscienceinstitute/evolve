//! # Physics Engine: Data Ingestion Module
//!
//! This module is responsible for fetching, parsing, and preparing external physics data
//! for use in the simulation. It is designed to be used at build time to compile
//! data directly into the engine.

#![cfg(feature = "data-ingestion")]

use crate::nuclear_physics::{DecayMode, NuclearDecayData};
use serde::{de, Deserialize, Deserializer};
use std::str::FromStr;

/// Represents the top-level structure of the NuDat 3 JSON export.
#[derive(Debug, Deserialize)]
pub struct NuDatData {
    pub nuclides: Vec<NuDatNuclide>,
}

/// Represents a single nuclide entry from the NuDat database.
/// Field names match the NuDat 3 JSON output.
#[derive(Debug, Deserialize)]
pub struct NuDatNuclide {
    #[serde(deserialize_with = "from_string")]
    pub z: u32,
    #[serde(deserialize_with = "from_string")]
    pub n: u32,
    #[serde(deserialize_with = "from_string")]
    pub a: u32,
    pub symbol: String,
    #[serde(alias = "mass_number", deserialize_with = "from_string")]
    pub atomic_mass: f64,
    pub half_life: String,
    pub decay_modes: Vec<DecayInfo>,
}

#[derive(Debug, Deserialize)]
pub struct DecayInfo {
    pub mode: String,
    #[serde(deserialize_with = "from_string_option")]
    pub percentage: Option<f64>,
}

/// Custom deserializer for values that might be strings.
fn from_string<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr + Deserialize<'de>,
    T::Err: std::fmt::Display,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrT<T> {
        String(String),
        T(T),
    }

    match StringOrT::<T>::deserialize(deserializer)? {
        StringOrT::String(s) => s.parse::<T>().map_err(de::Error::custom),
        StringOrT::T(i) => Ok(i),
    }
}

/// Custom deserializer for optional values that might be strings.
fn from_string_option<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr + Deserialize<'de>,
    T::Err: std::fmt::Display,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrT<T> {
        String(String),
        T(T),
    }

    match Option::<StringOrT<T>>::deserialize(deserializer)? {
        Some(StringOrT::String(s)) => s.parse::<T>().map(Some).map_err(de::Error::custom),
        Some(StringOrT::T(i)) => Ok(Some(i)),
        None => Ok(None),
    }
}

/// Fetches and parses NuDat 3 data from the specified file path.
pub fn load_nudat_data(path: &str) -> Result<Vec<NuDatNuclide>, anyhow::Error> {
    let json_str = std::fs::read_to_string(path)?;
    // The actual data is nested inside a single-element array in the JSON file.
    let data: Vec<Vec<NuDatNuclide>> = serde_json::from_str(&json_str)?;
    Ok(data.into_iter().flatten().collect())
}

/// Converts a `NuDatNuclide` into the engine's `NuclearDecayData`.
pub fn convert_to_decay_data(nuclide: &NuDatNuclide) -> Option<((u32, u32), NuclearDecayData)> {
    let z = nuclide.z;
    let a = nuclide.a;

    let (half_life_seconds, is_stable) = parse_half_life(&nuclide.half_life);

    if is_stable {
        return Some((
            (z, a),
            NuclearDecayData {
                half_life_seconds: f64::INFINITY,
                primary_mode: DecayMode::Stable,
                decay_energy: 0.0,
                branching_ratio: 1.0,
            },
        ));
    }

    // Find the primary decay mode (highest branching ratio)
    let primary_decay = nuclide
        .decay_modes
        .iter()
        .filter_map(|d| d.percentage.map(|p| (p, &d.mode)))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((branching_ratio, mode_str)) = primary_decay {
        let primary_mode = parse_decay_mode(mode_str)?;
        let decay_data = NuclearDecayData {
            half_life_seconds,
            primary_mode,
            decay_energy: 0.0, // Note: Q-value is not directly used from this part of NuDat
            branching_ratio: branching_ratio / 100.0,
        };
        Some(((z, a), decay_data))
    } else {
        None
    }
}

/// Parses the half-life string from NuDat into seconds.
fn parse_half_life(hl_str: &str) -> (f64, bool) {
    if hl_str == "STABLE" {
        return (f64::INFINITY, true);
    }
    let parts: Vec<&str> = hl_str.split_whitespace().collect();
    if parts.is_empty() {
        return (0.0, false);
    }
    let value: f64 = parts[0].parse().unwrap_or(0.0);
    let unit = parts.get(1).cloned().unwrap_or("");

    let multiplier = match unit {
        "ys" => 1e-24, // yoctoseconds
        "zs" => 1e-21, // zeptoseconds
        "as" => 1e-18, // attoseconds
        "fs" => 1e-15, // femtoseconds
        "ps" => 1e-12, // picoseconds
        "ns" => 1e-9,  // nanoseconds
        "us" => 1e-6,  // microseconds
        "ms" => 1e-3,  // milliseconds
        "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        "d" => 86400.0,
        "y" => 3.15576e7, // 365.25 days
        "ky" => 3.15576e10,
        "My" => 3.15576e13,
        "Gy" => 3.15576e16,
        "Ty" => 3.15576e19,
        "Py" => 3.15576e22,
        "Ey" => 3.15576e25,
        _ => 0.0,
    };
    (value * multiplier, false)
}

/// Parses the decay mode string into a `DecayMode` enum.
fn parse_decay_mode(mode: &str) -> Option<DecayMode> {
    match mode {
        "A" => Some(DecayMode::Alpha),
        "B-" => Some(DecayMode::BetaMinus),
        "B+" => Some(DecayMode::BetaPlus),
        "EC" => Some(DecayMode::ElectronCapture),
        "SF" => Some(DecayMode::SpontaneousFission),
        _ => None, // Ignore more complex/rare decay modes for now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_sample_nudat() {
        // A small sample of the NuDat JSON format for testing
        let sample_json = r#"
        {
            "nuclides": [
                {
                    "z": "1",
                    "n": "0",
                    "a": "1",
                    "symbol": "H",
                    "mass_number": "1",
                    "atomic_mass": "1.0078250322",
                    "half_life": "stable",
                    "decay_modes": []
                },
                {
                    "z": "1",
                    "n": "2",
                    "a": "3",
                    "symbol": "H",
                    "mass_number": "3",
                    "atomic_mass": "3.016049281",
                    "half_life": "12.32 y",
                    "decay_modes": [
                        {
                            "mode": "B-",
                            "percentage": "100"
                        }
                    ]
                }
            ]
        }
        "#;

        let data: NuDatData = serde_json::from_str(sample_json).expect("Failed to parse sample JSON");
        assert_eq!(data.nuclides.len(), 2);

        let hydrogen = &data.nuclides[0];
        assert_eq!(hydrogen.z, 1);
        assert_eq!(hydrogen.symbol, "H");
        assert_eq!(hydrogen.half_life, "stable");

        let tritium = &data.nuclides[1];
        assert_eq!(tritium.z, 1);
        assert_eq!(tritium.n, 2);
        assert_eq!(tritium.half_life, "12.32 y");
        assert_eq!(tritium.decay_modes.len(), 1);
        assert_eq!(tritium.decay_modes[0].mode, "B-");
        assert_eq!(tritium.decay_modes[0].percentage, Some(100.0));
    }
} 