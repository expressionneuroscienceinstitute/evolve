//! Common types for FFI integration

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Fundamental particle representation
#[derive(Debug, Clone)]
pub struct FundamentalParticle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub mass: f64,
    pub charge: f64,
    pub energy: f64,
    pub particle_type: ParticleType,
}

/// Types of fundamental particles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleType {
    Electron,
    Positron,
    Muon,
    AntiMuon,
    Tau,
    AntiTau,
    ElectronNeutrino,
    ElectronAntiNeutrino,
    MuonNeutrino,
    MuonAntiNeutrino,
    TauNeutrino,
    TauAntiNeutrino,
    Photon,
    Proton,
    Neutron,
    Alpha,
    Deuteron,
    Triton,
    He3,
    PiPlus,
    PiMinus,
    PiZero,
    Kaon,
    AntiKaon,
    Unknown,
}

/// Interaction event between particles
#[derive(Debug, Clone)]
pub struct InteractionEvent {
    pub timestamp: f64,
    pub interaction_type: InteractionType,
    pub participants: Vec<usize>,
    pub energy_exchanged: f64,
    pub momentum_transfer: Vector3<f64>,
    pub products: Vec<ParticleType>,
    pub cross_section: f64,
}

/// Types of particle interactions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InteractionType {
    ElectromagneticScattering,
    WeakDecay,
    StrongInteraction,
    PairProduction,
    Annihilation,
    PhotonInteraction,
    NuclearFission,
    NuclearFusion,
} 