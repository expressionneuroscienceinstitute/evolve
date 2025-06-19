//! Comprehensive particle type definitions for the EVOLVE universe simulation
//! 
//! This module consolidates all particle-related data structures from the Standard Model
//! and beyond, providing a clean foundation for particle physics simulations.
//! 
//! Based on peer-reviewed particle physics references:
//! - PDG (Particle Data Group) 2023 Review
//! - CODATA 2022 fundamental constants
//! - Standard Model particle classifications

use nalgebra::{Vector3, Complex};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Fundamental particle types in the Standard Model and beyond
/// Based on PDG 2023 particle classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleType {
    // Quarks (6 flavors)
    Up, Down, Charm, Strange, Top, Bottom,
    
    // Leptons (6 types + antiparticles)
    Electron, ElectronNeutrino, ElectronAntiNeutrino, 
    Muon, MuonNeutrino, MuonAntiNeutrino,
    Tau, TauNeutrino, TauAntiNeutrino,
    
    // Antiparticles
    Positron,
    
    // Gauge bosons
    Photon, WBoson, WBosonMinus, ZBoson, Gluon,
    
    // Scalar bosons
    Higgs,
    
    // Composite particles (hadrons)
    Proton, Neutron, 
    
    // Light mesons (π, K, η)
    PionPlus, PionMinus, PionZero,
    KaonPlus, KaonMinus, KaonZero,
    Eta,
    
    // Baryons (Λ, Σ, Ξ, Ω)
    Lambda, SigmaPlus, SigmaMinus, SigmaZero,
    XiMinus, XiZero, OmegaMinus,
    
    // Heavy quarkonium states
    JPsi, Upsilon,
    
    // Atomic nuclei (by mass number)
    Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen, Fluorine, 
    Silicon, Phosphorus, Sulfur, Chlorine, Bromine, Iodine, Iron,
    
    // Atoms
    HydrogenAtom, HeliumAtom, CarbonAtom, OxygenAtom, IronAtom,
    
    // Molecules
    H2, H2O, CO2, CH4, NH3,
    
    // Dark matter candidate
    DarkMatter,
}

/// Color charge for strong force interactions
/// Based on QCD (Quantum Chromodynamics) theory
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorCharge {
    Red, Green, Blue,
    AntiRed, AntiGreen, AntiBlue,
    ColorSinglet,
}

/// Quark types for hadron composition
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuarkType {
    Up, Down, Charm, Strange, Top, Bottom,
}

/// Nucleon types (protons and neutrons)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NucleonType {
    Proton, Neutron,
}

/// Quantum field types for field theory calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    // Lepton fields
    ElectronField, MuonField, TauField,
    ElectronNeutrinoField, MuonNeutrinoField, TauNeutrinoField,
    
    // Quark fields
    UpQuarkField, DownQuarkField, CharmQuarkField, 
    StrangeQuarkField, TopQuarkField, BottomQuarkField,
    
    // Gauge boson fields
    PhotonField, WBosonField, ZBosonField, GluonField,
    
    // Scalar fields
    HiggsField,
    
    // Beyond Standard Model
    DarkMatterField,
}

/// Quantum numbers for particle states
/// Based on standard quantum mechanical principles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNumbers {
    pub n: u32,      // Principal quantum number
    pub l: u32,      // Orbital angular momentum
    pub m_l: i32,    // Magnetic quantum number
    pub m_s: f64,    // Spin magnetic quantum number
}

/// Measurement basis for quantum states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementBasis {
    Position,
    Momentum,
    Energy,
    Spin,
}

impl Default for MeasurementBasis {
    fn default() -> Self { Self::Position }
}

/// Boundary conditions for quantum fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Periodic, 
    Absorbing, 
    Reflecting,
}

/// Comprehensive quantum state representation
/// Includes wave function, entanglement, and measurement information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumState {
    pub wave_function: Vec<Complex<f64>>,
    pub entanglement_partners: Vec<usize>,
    pub decoherence_time: f64,
    pub measurement_basis: MeasurementBasis,
    pub superposition_amplitudes: HashMap<String, Complex<f64>>,
    
    // Quantum number fields
    pub principal_quantum_number: u32,
    pub orbital_angular_momentum: u32,
    pub magnetic_quantum_number: i32,
    pub spin_quantum_number: f64,
    pub energy_level: f64,
    pub occupation_probability: f64,
}

impl QuantumState {
    /// Create a new quantum state with default values
    pub fn new() -> Self {
        Self {
            wave_function: Vec::new(),
            entanglement_partners: Vec::new(),
            decoherence_time: f64::INFINITY,
            measurement_basis: MeasurementBasis::default(),
            superposition_amplitudes: HashMap::new(),
            principal_quantum_number: 1,
            orbital_angular_momentum: 0,
            magnetic_quantum_number: 0,
            spin_quantum_number: 0.5,
            energy_level: 0.0,
            occupation_probability: 1.0,
        }
    }
    
    /// Check if this state is entangled with other particles
    pub fn is_entangled(&self) -> bool {
        !self.entanglement_partners.is_empty()
    }
    
    /// Get the number of entangled partners
    pub fn entanglement_count(&self) -> usize {
        self.entanglement_partners.len()
    }
}

/// Quantum field representation on a 3D lattice
/// Used for field theory calculations and interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumField {
    pub field_type: FieldType,
    pub field_values: Vec<Vec<Vec<Complex<f64>>>>, // 3D grid
    pub field_derivatives: Vec<Vec<Vec<Vector3<Complex<f64>>>>>,
    pub vacuum_expectation_value: Complex<f64>,
    pub coupling_constants: HashMap<FieldType, f64>,
    pub lattice_spacing: f64,
    pub boundary_conditions: BoundaryConditions,
}

impl QuantumField {
    /// Create a new quantum field with specified type
    /// 
    /// # Arguments
    /// * `field_type` - The type of quantum field to create
    /// * `grid_size` - Size of the 3D lattice grid
    /// * `spacing` - Lattice spacing in meters
    pub fn new(field_type: FieldType, grid_size: usize, spacing: f64) -> Self {
        let field_values = vec![vec![vec![Complex::new(0.0, 0.0); grid_size]; grid_size]; grid_size];
        let field_derivatives = vec![vec![vec![Vector3::zeros(); grid_size]; grid_size]; grid_size];
        
        Self {
            field_type,
            field_values,
            field_derivatives,
            vacuum_expectation_value: Complex::new(0.0, 0.0),
            coupling_constants: HashMap::new(),
            lattice_spacing: spacing,
            boundary_conditions: BoundaryConditions::Periodic,
        }
    }
    
    /// Get the field value at a specific grid point
    pub fn get_field_value(&self, x: usize, y: usize, z: usize) -> Option<Complex<f64>> {
        self.field_values.get(x)?.get(y)?.get(z).copied()
    }
    
    /// Set the field value at a specific grid point
    pub fn set_field_value(&mut self, x: usize, y: usize, z: usize, value: Complex<f64>) -> Result<(), &'static str> {
        if let Some(layer) = self.field_values.get_mut(x) {
            if let Some(row) = layer.get_mut(y) {
                if let Some(cell) = row.get_mut(z) {
                    *cell = value;
                    return Ok(());
                }
            }
        }
        Err("Grid coordinates out of bounds")
    }
}

/// Individual fundamental particle with complete physical state
/// Based on relativistic quantum mechanics and particle physics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundamentalParticle {
    pub particle_type: ParticleType,
    pub position: Vector3<f64>,           // Position in 3D space (m)
    pub momentum: Vector3<f64>,           // Relativistic momentum (kg⋅m/s)
    pub spin: Vector3<Complex<f64>>,      // Spin angular momentum
    pub color_charge: Option<ColorCharge>, // For quarks and gluons
    pub electric_charge: f64,             // Electric charge (C)
    pub mass: f64,                        // Rest mass (kg)
    pub energy: f64,                      // Total energy E = √((pc)² + (mc²)²)
    pub creation_time: f64,               // Time of creation (s)
    pub decay_time: Option<f64>,          // Time until decay (s)
    pub quantum_state: QuantumState,      // Full quantum state
    pub interaction_history: Vec<crate::InteractionEvent>, // Interaction record
    pub velocity: Vector3<f64>,           // Classical velocity (m/s)
    pub charge: f64,                      // Duplicate of electric_charge for compatibility
}

impl FundamentalParticle {
    /// Create a new fundamental particle with minimal required information
    pub fn new(particle_type: ParticleType, mass: f64, position: Vector3<f64>) -> Self {
        let electric_charge = Self::get_standard_charge(particle_type);
        
        Self {
            particle_type,
            position,
            momentum: Vector3::zeros(),
            spin: Vector3::zeros(),
            color_charge: Self::get_standard_color_charge(particle_type),
            electric_charge,
            mass,
            energy: mass * crate::constants::SPEED_OF_LIGHT.powi(2), // Rest energy
            creation_time: 0.0,
            decay_time: Self::get_standard_lifetime(particle_type),
            quantum_state: QuantumState::new(),
            interaction_history: Vec::new(),
            velocity: Vector3::zeros(),
            charge: electric_charge,
        }
    }
    
    /// Get standard electric charge for a particle type (in units of elementary charge)
    fn get_standard_charge(particle_type: ParticleType) -> f64 {
        use crate::constants::ELEMENTARY_CHARGE;
        
        match particle_type {
            // Quarks
            ParticleType::Up | ParticleType::Charm | ParticleType::Top => 2.0/3.0 * ELEMENTARY_CHARGE,
            ParticleType::Down | ParticleType::Strange | ParticleType::Bottom => -1.0/3.0 * ELEMENTARY_CHARGE,
            
            // Charged leptons
            ParticleType::Electron | ParticleType::Muon | ParticleType::Tau => -ELEMENTARY_CHARGE,
            ParticleType::Positron => ELEMENTARY_CHARGE,
            
            // Neutrinos
            ParticleType::ElectronNeutrino | ParticleType::MuonNeutrino | ParticleType::TauNeutrino |
            ParticleType::ElectronAntiNeutrino | ParticleType::MuonAntiNeutrino | ParticleType::TauAntiNeutrino => 0.0,
            
            // Gauge bosons
            ParticleType::Photon | ParticleType::ZBoson | ParticleType::Gluon => 0.0,
            ParticleType::WBoson => ELEMENTARY_CHARGE,
            ParticleType::WBosonMinus => -ELEMENTARY_CHARGE,
            
            // Composite particles
            ParticleType::Proton => ELEMENTARY_CHARGE,
            ParticleType::Neutron => 0.0,
            
            // Everything else defaults to neutral
            _ => 0.0,
        }
    }
    
    /// Get standard color charge for a particle type
    fn get_standard_color_charge(particle_type: ParticleType) -> Option<ColorCharge> {
        match particle_type {
            ParticleType::Up | ParticleType::Down | ParticleType::Charm | 
            ParticleType::Strange | ParticleType::Top | ParticleType::Bottom => {
                // Quarks have color charge (randomly assigned for now)
                Some(ColorCharge::Red) // In reality, would be assigned dynamically
            },
            ParticleType::Gluon => Some(ColorCharge::ColorSinglet),
            _ => None, // Most particles are color-neutral
        }
    }
    
    /// Get standard lifetime for unstable particles
    fn get_standard_lifetime(particle_type: ParticleType) -> Option<f64> {
        match particle_type {
            // Stable particles
            ParticleType::Electron | ParticleType::Proton | ParticleType::ElectronNeutrino |
            ParticleType::MuonNeutrino | ParticleType::TauNeutrino => None,
            
            // Unstable particles (approximate lifetimes in seconds)
            ParticleType::Neutron => Some(879.4), // Free neutron lifetime
            ParticleType::Muon => Some(2.197e-6),
            ParticleType::Tau => Some(2.906e-13),
            ParticleType::PionPlus | ParticleType::PionMinus => Some(2.603e-8),
            ParticleType::PionZero => Some(8.52e-17),
            
            // Very short-lived particles
            ParticleType::WBoson | ParticleType::WBosonMinus => Some(3.0e-25),
            ParticleType::ZBoson => Some(2.6e-25),
            
            _ => None, // Unknown or effectively stable
        }
    }
    
    /// Calculate relativistic energy E = √((pc)² + (mc²)²)
    pub fn calculate_relativistic_energy(&self) -> f64 {
        let c = crate::constants::SPEED_OF_LIGHT;
        let momentum_magnitude = self.momentum.norm();
        let momentum_energy = momentum_magnitude * c;
        let rest_energy = self.mass * c * c;
        
        (momentum_energy * momentum_energy + rest_energy * rest_energy).sqrt()
    }
    
    /// Update energy based on current momentum and mass
    pub fn update_energy(&mut self) {
        self.energy = self.calculate_relativistic_energy();
    }
    
    /// Calculate velocity from relativistic momentum: v = pc²/E
    pub fn calculate_velocity(&self) -> Vector3<f64> {
        if self.energy > 0.0 && self.momentum.norm() > 0.0 {
            let c_squared = crate::constants::SPEED_OF_LIGHT.powi(2);
            self.momentum * (c_squared / self.energy)
        } else {
            Vector3::zeros()
        }
    }
    
    /// Update velocity based on current momentum and energy
    pub fn update_velocity(&mut self) {
        self.velocity = self.calculate_velocity();
    }
}

/// Type aliases for common composite structures
pub type GluonField = Vec<Vector3<Complex<f64>>>;
pub type NuclearShellState = HashMap<String, f64>;
pub type ElectronicState = HashMap<String, Complex<f64>>;

/// Interaction types for particle physics processes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InteractionType {
    ElectromagneticScattering,
    WeakDecay,
    StrongInteraction,
    GravitationalAttraction,
    NuclearFusion,
    NuclearFission,
    Annihilation,
    PairProduction,
}

/// Particle physics state representation for simulations
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