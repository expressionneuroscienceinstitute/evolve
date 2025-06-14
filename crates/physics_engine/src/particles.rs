//! Standard-Model particle catalogue (PDG 2024)

use std::collections::HashMap;
use once_cell::sync::Lazy;
use nalgebra::Vector3;
use anyhow::Result;

use crate::constants::*;
use crate::FundamentalParticle;
use crate::ParticleType;

/// Initialise particle constants (placeholder)
pub fn init_particles() -> Result<()> {
    // TODO: Populate with PDG constants and Standard Model parameters
    Ok(())
}

/// Intrinsic properties needed by the simulator
#[derive(Debug, Clone, Copy)]
pub struct ParticleProperties {
    pub mass_kg: f64,
    pub charge_c: f64,
    pub spin: f64,          // ħ units
    pub width: Option<f64>, // Decay width (Γ) in s⁻¹; None for stable / massless
    pub has_color: bool,    // Quarks & gluons carry color charge
}

/// Lookup table keyed by `ParticleType`
pub static PARTICLE_DATA: Lazy<HashMap<ParticleType, ParticleProperties>> = Lazy::new(|| {
    use ParticleType::*;
    let mut m = HashMap::new();

    // Helper macro to shorten inserts
    macro_rules! ins { ($t:ident, $mass:expr, $q:expr, $spin:expr, $width:expr, $color:expr) => { m.insert($t, ParticleProperties { mass_kg: $mass, charge_c: $q, spin: $spin, width: $width, has_color: $color }); } }

    // Quarks (constituent masses; PDG ranges → representative values)
    ins!(Up,    2.16e6_f64 * 1.78266192e-36,  (2.0/3.0) * ELEMENTARY_CHARGE, 0.5, None, true);
    ins!(Down,  4.67e6_f64 * 1.78266192e-36,  (-1.0/3.0) * ELEMENTARY_CHARGE, 0.5, None, true);
    ins!(Charm, 1.27e9_f64 * 1.78266192e-36,  (2.0/3.0) * ELEMENTARY_CHARGE, 0.5, Some(1.3e-12_f64.recip()), true);
    ins!(Strange, 93.0e6_f64 * 1.78266192e-36, (-1.0/3.0) * ELEMENTARY_CHARGE, 0.5, None, true);
    ins!(Top,   172.69e9_f64 * 1.78266192e-36, (2.0/3.0) * ELEMENTARY_CHARGE, 0.5, Some(5.0e-25_f64.recip()), true);
    ins!(Bottom, 4.18e9_f64 * 1.78266192e-36,  (-1.0/3.0) * ELEMENTARY_CHARGE, 0.5, Some(1.6e-12_f64.recip()), true);

    // Leptons
    ins!(Electron, ELECTRON_MASS, -ELEMENTARY_CHARGE, 0.5, None, false);
    ins!(Positron, ELECTRON_MASS, ELEMENTARY_CHARGE, 0.5, None, false);
    ins!(ElectronNeutrino, 0.0, 0.0, 0.5, None, false);
    ins!(Muon, 105.6583745e6 * 1.78266192e-36, -ELEMENTARY_CHARGE, 0.5, Some(2.1969811e-6_f64.recip()), false);
    ins!(MuonNeutrino, 0.0, 0.0, 0.5, None, false);
    ins!(Tau, 1.77686e9 * 1.78266192e-36, -ELEMENTARY_CHARGE, 0.5, Some(2.903e-13_f64.recip()), false);
    ins!(TauNeutrino, 0.0, 0.0, 0.5, None, false);

    // Gauge bosons & scalar
    ins!(Photon, 0.0, 0.0, 1.0, None, false);
    ins!(Gluon, 0.0, 0.0, 1.0, None, true);
    ins!(WBoson, 80.379e9 * 1.78266192e-36, ELEMENTARY_CHARGE, 1.0, Some(3.2e-25_f64.recip()), false);
    ins!(ZBoson, 91.1876e9 * 1.78266192e-36, 0.0, 1.0, Some(2.6e-25_f64.recip()), false);
    ins!(Higgs, 125.25e9 * 1.78266192e-36, 0.0, 0.0, Some(1.6e-22_f64.recip()), false);

    // Composite baryons (approx masses)
    ins!(Proton, PROTON_MASS, ELEMENTARY_CHARGE, 0.5, None, false);
    ins!(Neutron, NEUTRON_MASS, 0.0, 0.5, Some(880.2_f64.recip()), false);

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

/// CKM matrix (absolute values) – PDG 2024
pub static CKM_MATRIX: [[f64; 3]; 3] = [
    [0.97420, 0.2243, 0.00394],
    [0.218,   0.997,  0.0422 ],
    [0.0081,  0.041,  0.99910],
];

/// Dominant branching ratios for selected unstable particles (<channel list>, BR)
pub static BRANCHING_RATIOS: Lazy<HashMap<ParticleType, Vec<(Vec<ParticleType>, f64)>>> = Lazy::new(|| {
    use ParticleType::*;
    let mut h = HashMap::new();

    // μ⁻ → e⁻ ν̄_e ν_μ (≈ 100 %)
    h.insert(Muon, vec![ (vec![Electron], 1.0) ]);

    // τ⁻ major modes (approx)
    h.insert(Tau, vec![ (vec![Muon], 0.1741), (vec![Electron], 0.1783) ]);

    // W boson → l ν (10.8 % each) or qq' (~67 %) – simplified
    h.insert(WBoson, vec![ (vec![Electron], 0.108), (vec![Muon], 0.106), (vec![Tau], 0.112) ]);

    // Z boson → hadrons (69 %), l⁺l⁻ (3.4 % each) – simplified leptonic
    h.insert(ZBoson, vec![ (vec![Electron], 0.033), (vec![Muon], 0.033), (vec![Tau], 0.033) ]);

    // Top quark → W + b (~100 %)
    h.insert(Top, vec![ (vec![WBoson], 1.0) ]);

    h
});

/// Electroweak and QCD global parameters (PDG 2024)
pub mod sm_params {
    /// Weak mixing angle \(\sin^2 \theta_W\) in \(\overline{MS}\) at \(M_Z\)
    pub const SIN2_THETA_W: f64 = 0.23122; // ±0.00003

    /// CKM CP-violating phase δ (radians) – global fit
    pub const DELTA_CP: f64 = 1.20; // ±0.08

    /// Jarlskog invariant |J| = 3.04 × 10⁻⁵
    pub const JARLSKOG_J: f64 = 3.04e-5;

    /// Strong coupling α_s(M_Z) (5-flavour) – world average
    pub const ALPHA_S_MZ: f64 = 0.1179; // ±0.0009
}

/// One-loop running of α_s using Λ^(5)_MS = 211 MeV derived from PDG average
pub fn alpha_s_one_loop(q2_gev2: f64) -> f64 {
    // Λ^(5)_MS ≈ 0.211 GeV
    let lambda2 = 0.211f64.powi(2);
    let b0 = (33.0 - 2.0 * 5.0) / (12.0 * std::f64::consts::PI);
    1.0 / (b0 * (q2_gev2 / lambda2).ln())
}

/// Convenience: α_s(Q) with Q in GeV (one-loop)
pub fn alpha_s(q_gev: f64) -> f64 { alpha_s_one_loop(q_gev * q_gev) }