//! EVOLVE – physics_engine::particle_types
//!
//! This module now serves primarily as a thin façade that re-exports the canonical
//! Standard-Model type definitions provided by the workspace-wide `physics_types`
//! crate.  All simulation crates should use these shared definitions to guarantee
//! type equality across crate boundaries.
//!
//! In addition, we retain a handful of helper enums / structs which are _specific_ to
//! the high-level physics engine and do **not** yet exist in `physics_types`.
//! This avoids the large amount of duplicate code that previously caused
//! mismatched-type compilation errors.

use nalgebra::{Vector3, Complex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Re-export canonical shared types from `physics_types` so that callers can
//    continue to reference them through `crate::particle_types::*` if they wish.
// ---------------------------------------------------------------------------
pub use physics_types::{
    ParticleType,
    ColorCharge,
    QuantumState,
    MeasurementBasis,
    InteractionEvent,
    InteractionType,
    FundamentalParticle,
};

// ---------------------------------------------------------------------------
// 2. Engine-specific helper types (not present in the shared crate)
// ---------------------------------------------------------------------------

/// Six-flavour quark enumeration used for nucleon composition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuarkType {
    Up,
    Down,
    Charm,
    Strange,
    Top,
    Bottom,
}

/// Nucleon identity helper (proton or neutron).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NucleonType {
    Proton,
    Neutron,
}

/// Enumeration of the various quantum fields used by the simulation lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    // Lepton fields
    ElectronField,
    MuonField,
    TauField,
    ElectronNeutrinoField,
    MuonNeutrinoField,
    TauNeutrinoField,

    // Quark fields
    UpQuarkField,
    DownQuarkField,
    CharmQuarkField,
    StrangeQuarkField,
    TopQuarkField,
    BottomQuarkField,

    // Gauge boson fields
    PhotonField,
    WBosonField,
    ZBosonField,
    GluonField,

    // Scalar field
    HiggsField,

    // Beyond-Standard-Model candidate field
    DarkMatterField,
}

/// Simple set of hydrogenic quantum numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNumbers {
    pub n: u32,  // Principal
    pub l: u32,  // Orbital angular momentum
    pub m_l: i32, // Magnetic
    pub m_s: f64, // Spin magnetic
}

/// Boundary-condition options for lattice fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Periodic,
    Absorbing,
    Reflecting,
}

/// Compact representation of a quantum field on a 3-D lattice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumField {
    pub field_type: FieldType,
    pub field_values: Vec<Vec<Vec<Complex<f64>>>>,
    pub field_derivatives: Vec<Vec<Vec<Vector3<Complex<f64>>>>>,
    pub vacuum_expectation_value: Complex<f64>,
    pub coupling_constants: HashMap<FieldType, f64>,
    pub lattice_spacing: f64,
    pub boundary_conditions: BoundaryConditions,
}

// Aliases that the engine expects ------------------------------------------------

pub type GluonField = Vec<Vector3<Complex<f64>>>;
pub type NuclearShellState = HashMap<String, f64>;
pub type ElectronicState  = HashMap<String, Complex<f64>>;

/// Minimal classical state descriptor used by some legacy routines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub acceleration: Vector3<f64>,
    pub mass: f64,
    pub charge: f64,
    pub temperature: f64,
    pub entropy: f64,
} 