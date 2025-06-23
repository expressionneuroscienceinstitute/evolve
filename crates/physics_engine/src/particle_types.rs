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

impl QuantumField {
    /// Create a new quantum field with vacuum fluctuations
    pub fn new(field_type: FieldType, size: (usize, usize, usize), lattice_spacing: f64) -> Self {
        let mut field = Self {
            field_type,
            field_values: vec![vec![vec![Complex::new(0.0, 0.0); size.2]; size.1]; size.0],
            field_derivatives: vec![vec![vec![Vector3::zeros(); size.2]; size.1]; size.0],
            vacuum_expectation_value: Complex::new(0.0, 0.0),
            coupling_constants: HashMap::new(),
            lattice_spacing,
            boundary_conditions: BoundaryConditions::Periodic,
        };
        
        // Initialize with quantum vacuum fluctuations
        field.initialize_vacuum_fluctuations();
        
        field
    }
    
    /// Initialize quantum field with vacuum fluctuations
    pub fn initialize_vacuum_fluctuations(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Calculate quantum fluctuation amplitude based on field properties
        let fluctuation_amplitude = self.calculate_quantum_fluctuation_amplitude();
        
        // Initialize field values with quantum fluctuations
        for i in 0..self.field_values.len() {
            for j in 0..self.field_values[i].len() {
                for k in 0..self.field_values[i][j].len() {
                    let real_part = rng.gen_range(-fluctuation_amplitude..fluctuation_amplitude);
                    let imag_part = rng.gen_range(-fluctuation_amplitude..fluctuation_amplitude);
                    self.field_values[i][j][k] = Complex::new(real_part, imag_part) + self.vacuum_expectation_value;
                }
            }
        }
    }
    
    /// Calculate quantum fluctuation amplitude based on field properties
    fn calculate_quantum_fluctuation_amplitude(&self) -> f64 {
        // Heisenberg uncertainty principle: ΔEΔt ≥ ℏ/2
        // For quantum fields: Δφ ≈ ℏ/(m*c*Δx)
        let planck_constant = 1.054571817e-34; // Reduced Planck constant
        let speed_of_light = 299792458.0;
        
        // Get field mass based on type (simplified)
        let field_mass = match self.field_type {
            FieldType::ElectronField => 9.1093837015e-31,
            FieldType::PhotonField => 0.0, // Massless
            FieldType::HiggsField => 2.246e-25,
            _ => 1.0e-30, // Default mass
        };
        
        if field_mass > 0.0 {
            // Massive field fluctuations
            planck_constant / (field_mass * speed_of_light * self.lattice_spacing)
        } else {
            // Massless field fluctuations (like photons)
            planck_constant / (speed_of_light * self.lattice_spacing)
        }
    }
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