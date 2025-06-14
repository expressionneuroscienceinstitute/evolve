//! Scientific constants (CODATA 2023)
//! 
//! This crate centralises peer-reviewed fundamental constants so all
//! simulation modules share a single authoritative source.
//! Values are in SI units and double-precision.
//! Reference: Mohr, P.J., Newell, D.B. & Taylor, B.N. (2023), CODATA.

use once_cell::sync::Lazy;

/// Speed of light in vacuum (m/s)
pub const C: f64 = 299_792_458.0;

/// Gravitational constant (m^3 kg^-1 s^-2) — 2023 value
pub const G: f64 = 6.674_30e-11;

/// Planck constant (J·s)
pub const H: f64 = 6.626_070_15e-34;

/// Reduced Planck constant (ħ)
pub const HBAR: f64 = H / (2.0 * std::f64::consts::PI);

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380_649e-23;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602_176_634e-19;

/// Solar mass (kg)
pub const SOLAR_MASS: f64 = 1.988_47e30;

/// Atomic mass unit (kg)
pub const AMU: f64 = 1.660_539_066_60e-27;

/// Avogadro constant (1/mol)
pub const N_A: f64 = 6.022_140_76e23;

/// Lazy-loaded map for any constant by symbol (for scripting)
pub static CONSTANTS: Lazy<std::collections::HashMap<&'static str, f64>> = Lazy::new(|| {
    use std::collections::HashMap;
    let mut m = HashMap::new();
    m.insert("c", C);
    m.insert("G", G);
    m.insert("h", H);
    m.insert("hbar", HBAR);
    m.insert("k_B", K_B);
    m.insert("e", E_CHARGE);
    m.insert("M_sun", SOLAR_MASS);
    m.insert("amu", AMU);
    m.insert("N_A", N_A);
    m
});