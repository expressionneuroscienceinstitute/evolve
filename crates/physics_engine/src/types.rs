//! Common Physics Types
//! 
//! This module contains fundamental type definitions used throughout the physics engine.
//! Extracted from lib.rs to reduce code duplication and improve modularity.
//!
//! ## Organization
//! - Basic particle and quantum types
//! - Environment and material types  
//! - Computational utility types
//! - Type aliases for complex data structures
//!
//! All types maintain full scientific rigor with proper documentation and units.

use nalgebra::{Vector3, Complex};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Type aliases for complex nested data structures
/// These improve code readability and reduce repetition
pub type GluonField = Vec<Vector3<Complex<f64>>>;
pub type NuclearShellState = HashMap<String, f64>;
pub type ElectronicState = HashMap<String, Complex<f64>>;
pub type MolecularOrbital = AtomicOrbital;
pub type VibrationalMode = Vector3<f64>;
pub type PotentialEnergySurface = Vec<Vec<Vec<f64>>>;
pub type ReactionCoordinate = Vector3<f64>;

/// General physics state for particles and systems
/// Used in molecular dynamics and other simulations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsState {
    /// Position in 3D space (meters)
    pub position: Vector3<f64>,
    /// Velocity vector (m/s)
    pub velocity: Vector3<f64>,
    /// Acceleration vector (m/s²)
    pub acceleration: Vector3<f64>,
    /// Net force acting on the state (N)
    pub force: Vector3<f64>,
    /// Rest mass (kg)
    pub mass: f64,
    /// Electric charge (Coulombs)
    pub charge: f64,
    /// Temperature (Kelvin)
    pub temperature: f64,
    /// Entropy (J/K)
    pub entropy: f64,
    /// Auxiliary identifier for particle/atom type (for MD cell lists, etc.)
    pub type_id: u32,
}

/// Interaction event recording for analysis
/// Tracks all particle interactions for statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Time when interaction occurred (seconds)
    pub timestamp: f64,
    /// Type of interaction that occurred
    pub interaction_type: InteractionType,
    /// Indices of particles involved
    pub participants: Vec<usize>,
    /// Energy exchanged in interaction (Joules)
    pub energy_exchanged: f64,
    /// Momentum transfer vector (kg⋅m/s)
    pub momentum_transfer: Vector3<f64>,
    /// Products created by interaction
    pub products: Vec<ParticleType>,
    /// Interaction cross-section (m²)
    pub cross_section: f64,
}

/// Types of fundamental interactions
/// Based on Standard Model of particle physics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InteractionType {
    /// Electromagnetic interactions (photon exchange)
    ElectromagneticScattering,
    /// Weak nuclear force (W/Z boson exchange)
    WeakDecay,
    /// Strong nuclear force (gluon exchange)
    StrongInteraction,
    /// Gravitational attraction
    GravitationalAttraction,
    /// Nuclear fusion processes
    NuclearFusion,
    /// Nuclear fission processes
    NuclearFission,
    /// Particle-antiparticle annihilation
    Annihilation,
    /// High-energy pair production
    PairProduction,
}

/// Environmental conditions profile
/// Computed from fundamental particle physics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentProfile {
    /// Liquid water concentration (fraction)
    pub liquid_water: f64,
    /// Atmospheric oxygen content (fraction)
    pub atmos_oxygen: f64,
    /// Atmospheric pressure (Pascals)
    pub atmos_pressure: f64,
    /// Temperature in Celsius
    pub temp_celsius: f64,
    /// Radiation level (Sieverts/hour)
    pub radiation: f64,
    /// Energy flux density (W/m²)
    pub energy_flux: f64,
    /// Shelter availability index (0-1)
    pub shelter_index: f64,
    /// Hazard rate (events/time)
    pub hazard_rate: f64,
}

impl Default for EnvironmentProfile {
    fn default() -> Self {
        Self {
            liquid_water: 0.0,
            atmos_oxygen: 0.21,  // Earth-like atmosphere
            atmos_pressure: 101325.0,  // 1 atmosphere in Pascals
            temp_celsius: 15.0,  // Temperate Earth-like
            radiation: 0.0001,  // Background radiation
            energy_flux: 1361.0,  // Solar constant W/m²
            shelter_index: 0.5,
            hazard_rate: 0.01,
        }
    }
}

impl EnvironmentProfile {
    /// Compute environment from fundamental physics
    /// 
    /// Derives macroscopic environmental properties from the underlying
    /// particle physics simulation state.
    /// 
    /// # Arguments
    /// * `particles` - Current particle distribution
    /// * `atoms` - Atomic composition
    /// * `molecules` - Molecular composition  
    /// * `temperature` - System temperature (K)
    /// 
    /// # Returns
    /// Environment profile derived from physics
    pub fn from_fundamental_physics(
        particles: &[FundamentalParticle],
        atoms: &[Atom],
        molecules: &[Molecule],
        temperature: f64,
    ) -> Self {
        // Derive temperature from kinetic energy
        let temp_celsius = temperature - 273.15;
        
        // Calculate radiation from high-energy particles
        let radiation_particles = particles.iter()
            .filter(|p| p.energy > 1e-13) // MeV scale
            .count();
        let radiation = radiation_particles as f64 * 1e-6; // Rough conversion
        
        // Estimate atmospheric composition from molecules
        let total_molecules = molecules.len() as f64;
        let (atmos_oxygen, liquid_water) = if total_molecules > 0.0 {
            let oxygen_molecules = molecules.iter()
                .filter(|m| m.atoms.iter().any(|a| a.nucleus.atomic_number == 8))
                .count() as f64;
            let oxygen_fraction = (oxygen_molecules / total_molecules).min(1.0);
            
            let water_molecules = molecules.iter()
                .filter(|m| is_water_molecule(m))
                .count() as f64;
            let water_fraction = (water_molecules / total_molecules).min(1.0);
            
            (oxygen_fraction, water_fraction)
        } else {
            (0.0, 0.0)
        };
        
        // Estimate pressure from particle density and temperature
        let particle_density = particles.len() as f64 / 1e9; // Rough volume estimate
        let atmos_pressure = particle_density * 1.38e-23 * temperature; // Ideal gas law approximation
        
        Self {
            temp_celsius,
            radiation,
            atmos_oxygen,
            liquid_water,
            atmos_pressure,
            ..Default::default()
        }
    }
}

/// Geological stratum layer for planetary modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumLayer {
    /// Layer thickness in meters
    pub thickness_m: f64,
    /// Type of material in this layer
    pub material_type: MaterialType,
    /// Bulk density (kg/m³)
    pub bulk_density: f64,
    /// Elemental composition
    pub elements: ElementTable,
}

/// Types of geological materials
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaterialType {
    /// Gaseous phase
    Gas,
    /// Loose rocky material
    Regolith,
    /// Surface soil layer
    Topsoil,
    /// Deeper soil layer
    Subsoil,
    /// Sedimentary rock formations
    SedimentaryRock,
    /// Igneous rock formations
    IgneousRock,
    /// Metamorphic rock formations
    MetamorphicRock,
    /// Mineral ore deposits
    OreVein,
    /// Frozen water
    Ice,
    /// Molten rock
    Magma,
}

/// Elemental abundance table
/// Tracks all 118 elements with their abundances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementTable {
    /// Abundances in parts per million (ppm)
    #[serde(with = "serde_arrays")]
    pub abundances: [u32; 118],
}

impl Default for ElementTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementTable {
    /// Create new empty element table
    pub fn new() -> Self {
        Self {
            abundances: [0; 118],
        }
    }
    
    /// Set abundance for element Z
    pub fn set_abundance(&mut self, z: usize, ppm: u32) {
        if z < 118 {
            self.abundances[z] = ppm;
        }
    }
    
    /// Get abundance for element Z
    pub fn get_abundance(&self, z: usize) -> u32 {
        if z < 118 {
            self.abundances[z]
        } else {
            0
        }
    }
    
    /// Create element table from particle distribution
    pub fn from_particles(particles: &[FundamentalParticle]) -> Self {
        let mut table = Self::new();
        
        // Count particles by atomic number
        for particle in particles {
            let z = match particle.particle_type {
                ParticleType::Hydrogen | ParticleType::HydrogenAtom => 1,
                ParticleType::Helium | ParticleType::HeliumAtom => 2,
                ParticleType::Carbon | ParticleType::CarbonAtom => 6,
                ParticleType::Oxygen | ParticleType::OxygenAtom => 8,
                ParticleType::Iron | ParticleType::IronAtom => 26,
                _ => continue,
            };
            
            if z <= 118 {
                table.abundances[z - 1] = table.abundances[z - 1].saturating_add(1);
            }
        }
        
        table
    }
}

/// Boundary conditions for field equations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundaryConditions {
    /// Periodic boundary conditions
    Periodic,
    /// Absorbing boundary conditions
    Absorbing,
    /// Reflecting boundary conditions
    Reflecting,
}

/// Quantum measurement basis
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MeasurementBasis {
    /// Position basis measurement
    Position,
    /// Momentum basis measurement
    Momentum,
    /// Energy basis measurement
    Energy,
    /// Spin basis measurement
    Spin,
}

impl Default for MeasurementBasis {
    fn default() -> Self {
        Self::Position
    }
}

/// Nuclear decay channel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayChannel {
    /// Products of the decay
    pub products: Vec<ParticleType>,
    /// Branching ratio (probability)
    pub branching_ratio: f64,
    /// Decay constant (1/s)
    pub decay_constant: f64,
}

/// Nuclear fusion reaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionReaction {
    /// Indices of reactant particles
    pub reactant_indices: Vec<usize>,
    /// Mass number of product nucleus
    pub product_mass_number: u32,
    /// Atomic number of product nucleus
    pub product_atomic_number: u32,
    /// Energy released (J)
    pub q_value: f64,
    /// Reaction cross-section (m²)
    pub cross_section: f64,
    /// Whether catalysis is required
    pub requires_catalysis: bool,
}

impl Default for FusionReaction {
    fn default() -> Self {
        Self {
            reactant_indices: Vec::new(),
            product_mass_number: 0,
            product_atomic_number: 0,
            q_value: 0.0,
            cross_section: 0.0,
            requires_catalysis: false,
        }
    }
}

// Import forward declarations for types used in this module
use super::{ParticleType, FundamentalParticle, Atom, Molecule, AtomicOrbital};

/// Helper function to identify water molecules
fn is_water_molecule(molecule: &Molecule) -> bool {
    // Check if molecule has 1 oxygen and 2 hydrogen atoms
    let mut oxygen_count = 0;
    let mut hydrogen_count = 0;
    
    for atom in &molecule.atoms {
        match atom.nucleus.atomic_number {
            1 => hydrogen_count += 1,
            8 => oxygen_count += 1,
            _ => return false,
        }
    }
    
    oxygen_count == 1 && hydrogen_count == 2
} 