//! Shared types and traits for physics simulation
//!
//! This crate contains the common types and traits used by both the physics engine
//! and FFI integration crates to avoid circular dependencies.

use nalgebra::Vector3;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

/// Fundamental particle types from the Standard Model and beyond
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleType {
    // Quarks
    Up,
    Down,
    Charm,
    Strange,
    Top,
    Bottom,

    // Leptons
    Electron,
    ElectronNeutrino,
    Muon,
    MuonNeutrino,
    Tau,
    TauNeutrino,

    // Antiparticles
    Positron,
    AntiProton,
    AntiNeutron,
    AntiMuon,
    AntiTau,
    ElectronAntiNeutrino,
    MuonAntiNeutrino,
    TauAntiNeutrino,


    // Gauge bosons
    Photon,
    WPlus,
    WMinus,
    Z,
    Gluon,

    // Scalar bosons
    Higgs,

    // Composite particles
    Proton,
    Neutron,

    // Light mesons (π, K, η)
    PionPlus,
    PionMinus,
    PionZero,
    KaonPlus,
    KaonMinus,
    KaonZero,
    Eta,

    // Baryons (Λ, Σ, Ξ, Ω)
    Lambda,
    SigmaPlus,
    SigmaMinus,
    SigmaZero,
    XiMinus,
    XiZero,
    OmegaMinus,

    // Heavy quarkonium states
    JPsi,
    Upsilon,
    
    // Atomic nuclei (examples)
    Deuteron,
    Triton,
    Alpha,
    Carbon12,
    Oxygen16,
    Iron56,
    Uranium235,
    Uranium238,

    // Atoms
    HydrogenAtom,
    HeliumAtom,
    CarbonAtom,
    OxygenAtom,
    IronAtom,

    // Molecules
    H2,
    H2O,
    CO2,
    CH4,
    NH3,

    // Dark matter candidate
    DarkMatter,
    
    Other(u32),
}

/// Individual fundamental particle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundamentalParticle {
    pub particle_type: ParticleType,
    pub position: Vector3<f64>,
    pub momentum: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub spin: Vector3<Complex<f64>>,
    pub color_charge: Option<ColorCharge>,
    pub electric_charge: f64,
    pub mass: f64,
    pub energy: f64,
    pub creation_time: f64,
    pub decay_time: Option<f64>,
    pub quantum_state: QuantumState,
    pub interaction_history: Vec<InteractionEvent>,
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumState {
    pub wave_function: Vec<Complex<f64>>,
    pub entanglement_partners: Vec<usize>,
    pub decoherence_time: f64,
    pub measurement_basis: MeasurementBasis,
    pub superposition_amplitudes: HashMap<String, Complex<f64>>,
}

/// Basis for quantum measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MeasurementBasis {
    #[default]
    Position,
    Momentum,
    Energy,
    Spin,
}

/// Color charge for strong force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorCharge {
    Red,
    Green,
    Blue,
    AntiRed,
    AntiGreen,
    AntiBlue,
    ColorSinglet,
}

/// Represents a particle interaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub timestamp: f64,
    pub interaction_type: InteractionType,
    pub particles_in: Vec<FundamentalParticle>,
    pub particles_out: Vec<FundamentalParticle>,
    pub energy_exchanged: f64,
    pub momentum_transfer: Vector3<f64>,
    pub cross_section: f64,
}

/// Types of particle interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    Elastic,
    Inelastic,
    Annihilation,
    Decay,
    Fusion,
    Fission,
    Absorption,
    Emission,
    ElectromagneticScattering,
    WeakDecay,
    StrongInteraction,
    GravitationalAttraction,
    PairProduction,
}

/// Trait for particle transport simulation
pub trait ParticleTransport {
    /// Transport a particle through a material
    fn transport_particle(
        &mut self,
        particle: &FundamentalParticle,
        material: &str,
        step_length_cm: f64,
    ) -> Result<Vec<InteractionEvent>>;
}

/// Trait for nuclear cross-section calculation
pub trait NuclearCrossSections {
    /// Calculate nuclear cross-section
    fn calculate_cross_section(
        &self,
        isotope: u32,
        energy_ev: f64,
        temperature_k: f64,
    ) -> Result<f64>;
}