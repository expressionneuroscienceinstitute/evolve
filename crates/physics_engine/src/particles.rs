//! Fundamental particle data tables and helpers (stub)

use anyhow::Result;

/// Initialise particle constants (placeholder)
pub fn init_particles() -> Result<()> {
    // TODO: Populate with PDG constants and Standard Model parameters
    Ok(())
}

//! Standard-Model particle catalogue (PDG 2024)

use std::collections::HashMap;
use once_cell::sync::Lazy;
use nalgebra::Vector3;

use crate::constants::*;
use crate::FundamentalParticle;
use crate::ParticleType;

/// Intrinsic properties needed by the simulator
#[derive(Debug, Clone, Copy)]
pub struct ParticleProperties {
    pub mass_kg: f64,
    pub charge_c: f64,
    pub spin: f64,          // ħ units
    pub width: Option<f64>, // Decay width (Γ) in s⁻¹; None for stable / massless
}

/// Lookup table keyed by `ParticleType`
pub static PARTICLE_DATA: Lazy<HashMap<ParticleType, ParticleProperties>> = Lazy::new(|| {
    use ParticleType::*;
    let mut m = HashMap::new();

    // Helper macro to shorten inserts
    macro_rules! ins { ($t:ident, $mass:expr, $q:expr, $spin:expr, $width:expr) => { m.insert($t, ParticleProperties { mass_kg: $mass, charge_c: $q, spin: $spin, width: $width }); } }

    // Quarks (constituent masses; PDG ranges → representative values)
    ins!(Up,    2.16e6_f64 * 1.78266192e-36,  +2.0/3.0 * ELEMENTARY_CHARGE, 0.5, None);
    ins!(Down,  4.67e6_f64 * 1.78266192e-36,  -1.0/3.0 * ELEMENTARY_CHARGE, 0.5, None);
    ins!(Charm, 1.27e9_f64 * 1.78266192e-36,  +2.0/3.0 * ELEMENTARY_CHARGE, 0.5, Some(1.3e-12_f64.recip()));
    ins!(Strange, 93.0e6_f64 * 1.78266192e-36, -1.0/3.0 * ELEMENTARY_CHARGE, 0.5, None);
    ins!(Top,   172.69e9_f64 * 1.78266192e-36, +2.0/3.0 * ELEMENTARY_CHARGE, 0.5, Some(5.0e-25_f64.recip()));
    ins!(Bottom, 4.18e9_f64 * 1.78266192e-36,  -1.0/3.0 * ELEMENTARY_CHARGE, 0.5, Some(1.6e-12_f64.recip()));

    // Leptons
    ins!(Electron, ELECTRON_MASS, -ELEMENTARY_CHARGE, 0.5, None);
    ins!(ElectronNeutrino, 0.0, 0.0, 0.5, None);
    ins!(Muon, 105.6583745e6 * 1.78266192e-36, -ELEMENTARY_CHARGE, 0.5, Some((2.1969811e-6).recip()));
    ins!(MuonNeutrino, 0.0, 0.0, 0.5, None);
    ins!(Tau, 1.77686e9 * 1.78266192e-36, -ELEMENTARY_CHARGE, 0.5, Some((2.903e-13).recip()));
    ins!(TauNeutrino, 0.0, 0.0, 0.5, None);

    // Gauge bosons & scalar
    ins!(Photon, 0.0, 0.0, 1.0, None);
    ins!(Gluon, 0.0, 0.0, 1.0, None);
    ins!(WBoson, 80.379e9 * 1.78266192e-36, ELEMENTARY_CHARGE, 1.0, Some((3.2e-25).recip()));
    ins!(ZBoson, 91.1876e9 * 1.78266192e-36, 0.0, 1.0, Some((2.6e-25).recip()));
    ins!(Higgs, 125.25e9 * 1.78266192e-36, 0.0, 0.0, Some((1.6e-22).recip()));

    // Composite baryons (approx masses)
    ins!(Proton, PROTON_MASS, ELEMENTARY_CHARGE, 0.5, None);
    ins!(Neutron, NEUTRON_MASS, 0.0, 0.5, Some((880.2).recip()));

    m
});

/// Convenience accessor
pub fn get_properties(pt: ParticleType) -> ParticleProperties {
    *PARTICLE_DATA.get(&pt).expect("Particle data missing")
}

/// Build a `FundamentalParticle` with zero momentum at origin for quick tests
pub fn spawn_rest(pt: ParticleType) -> FundamentalParticle {
    let props = get_properties(pt);
    FundamentalParticle {
        particle_type: pt,
        position: Vector3::zeros(),
        momentum: Vector3::zeros(),
        spin: Vector3::zeros(),
        color_charge: None,
        electric_charge: props.charge_c,
        mass: props.mass_kg,
        energy: 0.0,
        creation_time: 0.0,
        decay_time: None,
        quantum_state: crate::QuantumState::new(),
        interaction_history: Vec::new(),
    }
}