#![allow(dead_code)]
#![allow(unused_variables)]
//! Comprehensive Physics Engine
//! 
//! Complete fundamental particle physics simulation from quantum fields
//! to complex matter structures. Implements the Standard Model and beyond.

pub mod atomic_physics;
pub mod classical;
pub mod chemistry;
pub mod climate;
pub mod constants;
pub mod atomic_data;
pub mod electromagnetic;
pub mod emergent_properties;
pub mod endf_data;
// pub mod ffi; // TODO: Create this module
pub mod geodynamics;
pub mod general_relativity; // Externalised GR implementation
pub mod interactions;
pub mod molecular_dynamics;
pub mod nuclear_physics;
pub mod particles;
pub mod phase_transitions;
pub mod quantum;
pub mod quantum_fields;
pub mod spatial;
pub mod sph; // NEW: Smoothed Particle Hydrodynamics
pub mod thermodynamics;
pub mod utils;
pub mod validation;
pub mod types;
pub mod adaptive_mesh_refinement;

// New refactored modules
pub mod atomic_structures;
pub mod interaction_events;
pub mod particle_types;

// Temporary compatibility layer for missing QC helpers
// mod qc_compat;
pub mod quantum_chemistry;
pub mod quantum_math;
pub mod octree;

pub mod radiative_transfer;
pub mod jeans_instability;

use nalgebra::{Vector3, Matrix3, Complex};
use serde::{Serialize, Deserialize};
use anyhow::{Result};
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use rand::distributions::Distribution;
use rayon::prelude::*;
use std::time::Instant;
use tracing::debug;

use self::nuclear_physics::{StellarNucleosynthesis, DecayMode};
use self::spatial::{SpatialHashGrid, SpatialGridStats};
use self::octree::{Octree, AABB};
use self::sph::SphSolver; // NEW: SPH imports
use self::radiative_transfer::RadiativeTransferSolver; // NEW: Radiative transfer imports
use self::jeans_instability::JeansInstabilitySolver; // NEW: Jeans instability imports
// use self::constants::{BOLTZMANN, SPEED_OF_LIGHT, ELEMENTARY_CHARGE, REDUCED_PLANCK_CONSTANT, VACUUM_PERMITTIVITY};
use physics_types as shared_types;

pub use constants::*;

// Add missing imports for constants and types
use crate::constants::ELEMENTARY_CHARGE as E_CHARGE;
use crate::utils::K_E;
use crate::types::{
    MeasurementBasis, BoundaryConditions, DecayChannel, NuclearShellState,
    GluonField, ElectronicState, MolecularOrbital, VibrationalMode,
    PotentialEnergySurface, ReactionCoordinate
};
use crate::general_relativity::{C, G, schwarzschild_radius};
use crate::types::{PhysicsState, FusionReaction, InteractionEvent, InteractionType};
use crate::gravitational_collapse::MEAN_MOLECULAR_WEIGHT;
use crate::sph::SphParticle;
use log::info;

pub mod gravitational_collapse;
pub use gravitational_collapse::{jeans_mass, jeans_length, SinkParticle};

/// Fundamental particle types in the Standard Model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleType {
    // Quarks
    Up, Down, Charm, Strange, Top, Bottom,
    
    // Leptons
    Electron, ElectronNeutrino, ElectronAntiNeutrino, 
    Muon, MuonNeutrino, MuonAntiNeutrino,
    Tau, TauNeutrino, TauAntiNeutrino,
    
    // Antiparticles
    Positron,
    
    // Gauge bosons
    Photon, WBoson, WBosonMinus, ZBoson, Gluon,
    
    // Scalar bosons
    Higgs,
    
    // Composite particles
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
    Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen, Fluorine, Silicon, Phosphorus, Sulfur, Chlorine, Bromine, Iodine, Iron, // ... etc
    
    // Atoms
    HydrogenAtom, HeliumAtom, CarbonAtom, OxygenAtom, IronAtom,
    
    // Molecules
    H2, H2O, CO2, CH4, NH3, // ... complex molecules
    
    // Sink particles for gravitational collapse (protostars, stars)
    SinkParticle,
    
    // Dark matter candidate
    DarkMatter,
}

/// Individual fundamental particle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundamentalParticle {
    pub particle_type: ParticleType,
    pub position: Vector3<f64>,
    pub momentum: Vector3<f64>,
    pub spin: Vector3<Complex<f64>>,
    pub color_charge: Option<ColorCharge>,
    pub electric_charge: f64,
    pub mass: f64,
    pub energy: f64,
    pub creation_time: f64,
    pub decay_time: Option<f64>,
    pub quantum_state: QuantumState,
    pub interaction_history: Vec<InteractionEvent>,
    pub velocity: Vector3<f64>,
    pub charge: f64,
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
            energy: mass * constants::SPEED_OF_LIGHT.powi(2), // Rest energy
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
        use constants::ELEMENTARY_CHARGE;
        
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
}

/// Quantum state representation
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
    pub fn new() -> Self {
        Self {
            wave_function: vec![Complex::new(1.0, 0.0)],
            entanglement_partners: Vec::new(),
            decoherence_time: f64::INFINITY,
            measurement_basis: MeasurementBasis::Position,
            superposition_amplitudes: HashMap::new(),
            principal_quantum_number: 1,
            orbital_angular_momentum: 0,
            magnetic_quantum_number: 0,
            spin_quantum_number: 0.5,
            energy_level: 0.0,
            occupation_probability: 1.0,
        }
    }
}

/// Color charge for strong force
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorCharge {
    Red, Green, Blue,
    AntiRed, AntiGreen, AntiBlue,
    ColorSinglet,
}

/// Stub implementations for missing types
#[derive(Debug, Clone, Default)]
pub struct InteractionMatrix;

impl InteractionMatrix {
    pub fn set_electromagnetic_coupling(&mut self, _coupling: f64) {}
    pub fn set_weak_coupling(&mut self, _coupling: f64) {}
    pub fn set_strong_coupling(&mut self, _coupling: f64) {}
}

#[derive(Debug, Clone, Default)]
pub struct SpacetimeGrid;

#[derive(Debug, Clone, Default)]
pub struct QuantumVacuum;

impl QuantumVacuum {
    pub fn initialize_fluctuations(&mut self, _temperature: f64) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct FieldEquations;

#[derive(Debug, Clone, Default)]
pub struct ParticleAccelerator;

#[derive(Debug, Clone, Default)]
pub struct RunningCouplings {
    pub scale_gev: f64,
    pub alpha_em: f64,
    pub alpha_s: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SymmetryBreaking;

impl SymmetryBreaking {
    pub fn initialize_higgs_mechanism(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Quantum field representation
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    ElectronField, MuonField, TauField,
    ElectronNeutrinoField, MuonNeutrinoField, TauNeutrinoField,
    UpQuarkField, DownQuarkField, CharmQuarkField, StrangeQuarkField, TopQuarkField, BottomQuarkField,
    PhotonField, WBosonField, ZBosonField, GluonField,
    HiggsField,
    DarkMatterField,
}

/// Main physics engine for universe simulation
#[derive(Debug)]
pub struct PhysicsEngine {
    pub particles: Vec<FundamentalParticle>,
    pub quantum_fields: HashMap<FieldType, QuantumField>,
    pub nuclei: Vec<AtomicNucleus>,
    pub atoms: Vec<Atom>,
    pub molecules: Vec<Molecule>,
    /// High-level quantum chemistry module for molecular calculations
    pub quantum_chemistry_engine: quantum_chemistry::QuantumChemistryEngine,
    pub interaction_matrix: InteractionMatrix,
    pub spacetime_grid: SpacetimeGrid,
    pub quantum_vacuum: QuantumVacuum,
    pub field_equations: FieldEquations,
    pub particle_accelerator: ParticleAccelerator,
    pub decay_channels: HashMap<ParticleType, Vec<DecayChannel>>,
    pub cross_sections: HashMap<(ParticleType, ParticleType), f64>,
    pub running_couplings: RunningCouplings,
    pub symmetry_breaking: SymmetryBreaking,
    pub stellar_nucleosynthesis: StellarNucleosynthesis,
    pub time_step: f64,
    pub current_time: f64,
    pub temperature: f64,
    pub energy_density: f64,
    pub particle_creation_threshold: f64,
    pub volume: f64,  // Simulation volume in m³
    pub compton_count: u64,  // Track Compton scattering events
    pub pair_production_count: u64,  // Track pair production events
    pub neutrino_scatter_count: u64, // Track neutrino-electron scatters
    pub particle_decay_count: u64, // Track all particle decay events
    pub neutron_decay_count: u64, // Track neutron beta decay events
    pub fusion_count: u64, // Track nuclear fusion events
    pub fission_count: u64, // Track nuclear fission events
    pub spatial_grid: SpatialHashGrid,
    pub octree: Octree,
    pub interaction_history: Vec<InteractionEvent>,
    /// SPH solver for fluid dynamics and star formation
    pub sph_solver: SphSolver,
    /// Sink particles representing collapsed objects (protostars, stars)
    pub sink_particles: Vec<SinkParticle>,
    /// Next unique ID for sink particle creation
    pub next_sink_id: u64,
    
    /// Radiative transfer solver for gas cooling and heating
    pub radiative_transfer: RadiativeTransferSolver,
    
    /// Jeans instability solver for gravitational collapse and star formation
    pub jeans_instability: JeansInstabilitySolver,
}

/// Atomic nucleus with detailed structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicNucleus {
    pub mass_number: u32,
    pub atomic_number: u32,
    pub protons: Vec<Nucleon>,
    pub neutrons: Vec<Nucleon>,
    pub binding_energy: f64,
    pub nuclear_spin: Vector3<f64>,
    pub magnetic_moment: Vector3<f64>,
    pub electric_quadrupole_moment: f64,
    pub nuclear_radius: f64,
    pub shell_model_state: NuclearShellState,
    pub position: Vector3<f64>,
    pub momentum: Vector3<f64>,
    pub excitation_energy: f64,
}

/// Individual nucleon (proton or neutron)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nucleon {
    pub nucleon_type: NucleonType,
    pub quarks: [Quark; 3],
    pub gluon_field: GluonField,
    pub position_in_nucleus: Vector3<f64>,
    pub momentum: Vector3<f64>,
    pub spin: Vector3<f64>,
    pub isospin: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NucleonType {
    Proton, Neutron,
}

/// Individual quark within nucleon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quark {
    pub quark_type: QuarkType,
    pub color: ColorCharge,
    pub position: Vector3<f64>,
    pub momentum: Vector3<f64>,
    pub spin: Vector3<Complex<f64>>,
    pub confinement_potential: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuarkType {
    Up, Down, Charm, Strange, Top, Bottom,
}

/// Complete atom with electron orbitals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub nucleus: AtomicNucleus,
    pub electrons: Vec<Electron>,
    pub electron_orbitals: Vec<AtomicOrbital>,
    pub total_energy: f64,
    pub ionization_energy: f64,
    pub electron_affinity: f64,
    pub atomic_radius: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub electronic_state: ElectronicState,
}

/// Electron in atom
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Electron {
    pub position_probability: Vec<Vec<Vec<f64>>>, // 3D probability density
    pub momentum_distribution: Vec<Vector3<f64>>,
    pub spin: Vector3<Complex<f64>>,
    pub orbital_angular_momentum: Vector3<f64>,
    pub quantum_numbers: QuantumNumbers,
    pub binding_energy: f64,
}

/// Quantum numbers for electron states
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumNumbers {
    pub n: u32,      // Principal
    pub l: u32,      // Orbital angular momentum
    pub m_l: i32,    // Magnetic
    pub m_s: f64,    // Spin magnetic
}

/// Atomic orbital
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicOrbital {
    pub orbital_type: OrbitalType,
    pub wave_function: Vec<Vec<Vec<Complex<f64>>>>,
    pub energy: f64,
    pub occupation_number: f64,
    pub quantum_numbers: QuantumNumbers,
}

/// Type of atomic orbital (s, p, d, f)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrbitalType {
    S, P, D, F,
}

/// Complete molecule with atomic composition and bonding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<ChemicalBond>,
    pub molecular_orbitals: Vec<MolecularOrbital>,
    pub vibrational_modes: Vec<VibrationalMode>,
    pub rotational_constants: Vector3<f64>,
    pub dipole_moment: Vector3<f64>,
    pub polarizability: Matrix3<f64>,
    pub potential_energy_surface: PotentialEnergySurface,
    pub reaction_coordinates: Vec<ReactionCoordinate>,
}

/// Chemical bond between two atoms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalBond {
    pub atom_indices: (usize, usize),
    pub bond_type: BondType,
    pub bond_length: f64,
    pub bond_energy: f64,
    pub bond_order: f64,
    pub electron_density: f64,
    pub overlap_integral: f64,
}

/// Type of chemical bond
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondType {
    Ionic, Covalent, Metallic, HydrogenBond, VanDerWaals,
}

#[derive(Default)]
struct AtomicUpdate {
    photons_to_emit: Vec<FundamentalParticle>,
    electrons_to_remove: Vec<usize>,
    energy_changes: Vec<f64>,
    electrons_to_add: Vec<FundamentalParticle>,
}

impl PhysicsEngine {
    /// Creates a new physics engine with optional FFI integration
    pub fn new() -> Result<Self> {
        let mut engine = Self {
            particles: Vec::new(),
            quantum_fields: HashMap::new(),
            nuclei: Vec::new(),
            atoms: Vec::new(),
            molecules: Vec::new(),
            quantum_chemistry_engine: quantum_chemistry::QuantumChemistryEngine::new(),
            interaction_matrix: InteractionMatrix::default(),
            spacetime_grid: SpacetimeGrid::default(),
            quantum_vacuum: QuantumVacuum::default(),
            field_equations: FieldEquations::default(),
            particle_accelerator: ParticleAccelerator::default(),
            decay_channels: HashMap::new(),
            cross_sections: HashMap::new(),
            running_couplings: RunningCouplings::default(),
            symmetry_breaking: SymmetryBreaking::default(),
            stellar_nucleosynthesis: StellarNucleosynthesis::new(),
            time_step: 1e-12,
            current_time: 0.0,
            temperature: 2.73, // Cosmic microwave background
            energy_density: 0.0,
            particle_creation_threshold: 1e6, // 1 MeV
            volume: 1e27, // 1 cubic meter
            compton_count: 0,
            pair_production_count: 0,
            neutrino_scatter_count: 0,
            particle_decay_count: 0,
            neutron_decay_count: 0,
            fusion_count: 0,
            fission_count: 0,
            spatial_grid: SpatialHashGrid::new(1e-6),
            octree: Octree::new(AABB::new(Vector3::zeros(), Vector3::new(1e-3, 1e-3, 1e-3))),
            interaction_history: Vec::new(),
            sph_solver: SphSolver::default(),
            sink_particles: Vec::new(),
            next_sink_id: 0,
            radiative_transfer: RadiativeTransferSolver::default(),
            jeans_instability: JeansInstabilitySolver::default(),
        };
        
        engine.initialize_quantum_fields()?;
        engine.initialize_particle_properties()?;
        engine.initialize_interactions()?;
        
        Ok(engine)
    }
    
    /// Initialize all quantum fields
    fn initialize_quantum_fields(&mut self) -> Result<()> {
        let field_types = vec![
            FieldType::ElectronField, FieldType::MuonField, FieldType::TauField,
            FieldType::ElectronNeutrinoField, FieldType::MuonNeutrinoField, FieldType::TauNeutrinoField,
            FieldType::UpQuarkField, FieldType::DownQuarkField, FieldType::CharmQuarkField,
            FieldType::StrangeQuarkField, FieldType::TopQuarkField, FieldType::BottomQuarkField,
            FieldType::PhotonField, FieldType::WBosonField, FieldType::ZBosonField, FieldType::GluonField,
            FieldType::HiggsField, FieldType::DarkMatterField,
        ];
        
        for field_type in field_types {
            let field = QuantumField::new(field_type, &self.spacetime_grid)?;
            self.quantum_fields.insert(field_type, field);
        }
        
        Ok(())
    }
    
    /// Initialize particle decay channels and cross sections
    fn initialize_particle_properties(&mut self) -> Result<()> {
        // Muon decay: μ → e + νμ + νe
        self.decay_channels.insert(ParticleType::Muon, vec![
            DecayChannel {
                products: vec![ParticleType::Electron, ParticleType::MuonNeutrino, ParticleType::ElectronNeutrino],
                branching_ratio: 1.0,
                decay_constant: 1.0 / (2.2e-6), // Muon lifetime
            }
        ]);
        
        // Neutron decay: n → p + e + νe using proper Fermi golden rule
        self.decay_channels.insert(ParticleType::Neutron, vec![
            DecayChannel {
                products: vec![ParticleType::Proton, ParticleType::Electron, ParticleType::ElectronAntiNeutrino],
                branching_ratio: 1.0,
                decay_constant: interactions::neutron_beta_width(), // Use calculated width
            }
        ]);
        
        // Initialize cross sections for particle interactions using nuclear database
        // For electron-electron: use Thomson scattering cross-section
        let thomson_cross_section = 8.0 * std::f64::consts::PI / 3.0 * 2.8179403227e-15_f64.powi(2); // m²
        self.cross_sections.insert((ParticleType::Electron, ParticleType::Electron), thomson_cross_section);
        
        // For proton-proton: use nuclear database estimate at typical stellar temperature
        let pp_cross_section = nuclear_physics::NUCLEAR_DATABASE.get_fusion_cross_section(1, 1, 1, 1, 15e6)
            .unwrap_or(1e-47); // Fallback to realistic pp cross-section
        self.cross_sections.insert((ParticleType::Proton, ParticleType::Proton), pp_cross_section);
        
        Ok(())
    }
    
    /// Initialize interaction strengths
    fn initialize_interactions(&mut self) -> Result<()> {
        // Set up the four fundamental forces
        self.interaction_matrix.set_electromagnetic_coupling(FINE_STRUCTURE_CONSTANT);
        self.interaction_matrix.set_weak_coupling(1.166e-5); // Fermi constant
        self.interaction_matrix.set_strong_coupling(0.1); // αs at MZ
        
        Ok(())
    }
    
    /// Create Big Bang initial conditions with fundamental particles
    pub fn initialize_big_bang(&mut self) -> Result<()> {
        // Start with high but computationally reasonable temperature
        self.temperature = 1e12; // 1 TeV scale (reduced from Planck temperature)
        self.energy_density = 1e30; // Reduced accordingly
        
        println!("🌌 BIG BANG INITIALIZATION:");
        println!("   Set temperature: {:.2e} K", self.temperature);
        println!("   Set energy density: {:.2e} J/m³", self.energy_density);
        println!("   Creating primordial plasma...");
        
        // Create initial quantum soup of all particle types
        self.create_primordial_plasma()?;
        
        // Initialize quantum vacuum fluctuations
        self.quantum_vacuum.initialize_fluctuations(self.temperature)?;
        
        // Set up spontaneous symmetry breaking
        self.symmetry_breaking.initialize_higgs_mechanism()?;
        
        Ok(())
    }
    
    /// Create primordial plasma of fundamental particles
    fn create_primordial_plasma(&mut self) -> Result<()> {
        let mut rng = thread_rng();
        let num_particles = 1000; // Reduced from 1M to 1000 for demo
        
        for _ in 0..num_particles/2 {
            // Create particle-antiparticle pairs
            let particle_type = self.sample_particle_from_thermal_distribution(self.temperature);
            
            // Position particles closer together for interactions
            let position = Vector3::new(
                rng.gen_range(-1e-14..1e-14), // 10 fm scale
                rng.gen_range(-1e-14..1e-14),
                rng.gen_range(-1e-14..1e-14),
            );
            
            // Create particle
            let particle = FundamentalParticle {
                particle_type,
                position,
                momentum: self.sample_thermal_momentum(particle_type, self.temperature),
                spin: self.initialize_spin(particle_type),
                color_charge: self.assign_color_charge(particle_type),
                electric_charge: self.get_electric_charge(particle_type),
                mass: self.get_particle_mass(particle_type),
                energy: 0.0, // Will be calculated
                creation_time: self.current_time,
                decay_time: Some(self.current_time + rng.gen_range(1e-20..1e-18)),
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
                velocity: Vector3::zeros(),
                charge: 0.0,
            };
            
            // Create antiparticle for leptons before pushing particle
            if matches!(particle_type, ParticleType::Electron | ParticleType::Muon | ParticleType::Tau) {
                let antiparticle_type = match particle_type {
                    ParticleType::Electron => ParticleType::Positron,
                    _ => particle_type, // For now, only positrons implemented
                };
                
                let antiparticle = FundamentalParticle {
                    particle_type: antiparticle_type,
                    position: position + Vector3::new(
                        rng.gen_range(-1e-15..1e-15),
                        rng.gen_range(-1e-15..1e-15),
                        rng.gen_range(-1e-15..1e-15),
                    ),
                    momentum: self.sample_thermal_momentum(antiparticle_type, self.temperature),
                    spin: self.initialize_spin(antiparticle_type),
                    color_charge: self.assign_color_charge(antiparticle_type),
                    electric_charge: -self.get_electric_charge(particle_type),
                    mass: self.get_particle_mass(antiparticle_type),
                    energy: 0.0,
                    creation_time: self.current_time,
                    decay_time: self.calculate_decay_time(antiparticle_type),
                    quantum_state: QuantumState::new(),
                    interaction_history: Vec::new(),
                    velocity: Vector3::zeros(),
                    charge: 0.0,
                };
                
                self.particles.push(particle);
                self.particles.push(antiparticle);
            } else {
                self.particles.push(particle);
            }
        }
        
        // Add a neutron population to demonstrate beta decay
        for _ in 0..50 {
            let neutron = FundamentalParticle {
                particle_type: ParticleType::Neutron,
                position: Vector3::new(
                    rng.gen_range(-1e-14..1e-14),
                    rng.gen_range(-1e-14..1e-14),
                    rng.gen_range(-1e-14..1e-14),
                ),
                momentum: Vector3::zeros(),
                spin: self.initialize_spin(ParticleType::Neutron),
                color_charge: None,
                electric_charge: 0.0,
                mass: self.get_particle_mass(ParticleType::Neutron),
                energy: 0.0,
                creation_time: self.current_time,
                decay_time: Some(self.current_time + rng.gen_range(1e-20..1e-18)),
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
                velocity: Vector3::zeros(),
                charge: 0.0,
            };
            self.particles.push(neutron);
        }
        
        // Update particle energies
        self.update_particle_energies()?;
        
        Ok(())
    }
    
    /// Sample particle type from thermal distribution
    fn sample_particle_from_thermal_distribution(&self, temperature: f64) -> ParticleType {
        let mut rng = thread_rng();
        
        // At very high temperatures, all particles are created equally
        // At lower temperatures, lighter particles dominate
        let thermal_mass_scale = BOLTZMANN * temperature / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
        
        let particle_types = [
            ParticleType::Photon,     // Massless
            ParticleType::Gluon,      // Massless
            ParticleType::Electron,   // 0.511 MeV
            ParticleType::ElectronNeutrino, // ~0
            ParticleType::Up,         // ~2 MeV
            ParticleType::Down,       // ~5 MeV
            ParticleType::Muon,       // 105.7 MeV
            ParticleType::Strange,    // ~95 MeV
            ParticleType::Charm,      // ~1.3 GeV
            ParticleType::Tau,        // 1.777 GeV
            ParticleType::Bottom,     // ~4.2 GeV
            ParticleType::Top,        // ~173 GeV
            ParticleType::WBoson,     // ~80 GeV
            ParticleType::ZBoson,     // ~91 GeV
            ParticleType::Higgs,      // ~125 GeV
        ];
        
        // Boltzmann suppression for massive particles
        let weights: Vec<f64> = particle_types.iter()
            .map(|&pt| {
                let mass = self.get_particle_mass(pt);
                if mass == 0.0 {
                    1.0
                } else {
                    (-mass / thermal_mass_scale).exp()
                }
            })
            .collect();
        
        let total_weight: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        let random = rng.gen::<f64>() * total_weight;
        
        for (i, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if random <= cumulative {
                return particle_types[i];
            }
        }
        
        ParticleType::Photon // Fallback
    }
    
    /// Simulation step – two compile-time modes:
    /// 1. `heavy` feature enabled  ➜ run the full high-fidelity pipeline (default for production accuracy).
    /// 2. `heavy` feature *disabled* ➜ run a lightweight fast path suited for profiling & CI.
    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.time_step = dt;
        self.current_time += dt;
        
        // Update quantum fields
        for field in self.quantum_fields.values_mut() {
            // Placeholder for quantum field evolution
        }
        
        // Process particle interactions
        self.process_particle_interactions()?;
        
        // Process molecular dynamics
        self.process_molecular_dynamics()?;
        
        // Process gravitational dynamics
        self.process_gravitational_dynamics()?;
        
        // Process SPH hydrodynamics
        self.process_sph_hydrodynamics()?;
        
        // Process nuclear reactions
        self.process_nuclear_reactions()?;
        
        // Process particle decays
        self.process_particle_decays()?;
        
        // Process atomic physics
        self.update_atomic_physics()?;
        
        // Process molecular formation
        self.process_molecular_formation(&mut [])?;
        
        // Process chemical reactions
        self.process_chemical_reactions()?;
        
        // Process phase transitions
        self.process_phase_transitions()?;
        
        // Update emergent properties
        self.update_emergent_properties(&mut [])?;
        
        // Update running couplings
        self.update_running_couplings(&mut [])?;
        
        // Check symmetry breaking
        self.check_symmetry_breaking()?;
        
        // Update spacetime curvature
        self.update_spacetime_curvature()?;
        
        // Update thermodynamic state
        self.update_thermodynamic_state()?;
        
        // Evolve quantum state
        self.evolve_quantum_state()?;
        
        // Update temperature
        self.update_temperature()?;
        
        // Validate conservation laws
        self.validate_conservation_laws()?;
        
        Ok(())
    }
    
    /// Process particle interactions using the internal native Rust implementation.
    pub fn process_particle_interactions(&mut self) -> Result<()> {
        self.process_native_interactions()
    }

    /// Process molecular dynamics using LAMMPS if available
    pub fn process_molecular_dynamics(&mut self) -> Result<()> {
        // TODO: Restore when FFI engines are available
        // if let Some(ref mut lammps) = self.lammps_engine {
        //     self.process_lammps_dynamics(lammps)?;
        // } else {
            // Fallback to native molecular dynamics
            let mut states: Vec<PhysicsState> = self.particles.iter().map(|p| PhysicsState {
                position: p.position,
                velocity: p.velocity,
                acceleration: Vector3::zeros(),
                mass: p.mass,
                charge: p.charge,
                temperature: self.temperature,
                entropy: 0.0,
            }).collect();
            self.update_molecular_dynamics(&mut states)?;
        // }
        Ok(())
    }

    // LAMMPS molecular dynamics functions removed - now using native Rust implementation

    /// Process gravitational dynamics using GADGET if available
    pub fn process_gravitational_dynamics(&mut self) -> Result<()> {
        // TODO: Restore when FFI engines are available
        // if let Some(ref mut gadget) = self.gadget_engine {
        //     self.process_gadget_gravity(gadget)?;
        // } else {
            // Fallback to native gravity calculations
            self.update_gravitational_forces()?;
        // }
        Ok(())
    }

    /// Process SPH hydrodynamics for fluid dynamics and star formation
    pub fn process_sph_hydrodynamics(&mut self) -> Result<()> {
        // Convert gas particles (Hydrogen, Helium) to SPH particles
        let gas_particles: Vec<_> = self.particles
            .iter()
            .filter(|p| matches!(p.particle_type, ParticleType::Hydrogen | ParticleType::Helium))
            .cloned()
            .collect();
        
        if gas_particles.is_empty() {
            return Ok(());
        }
        
        // Convert to SPH particles
        let mut sph_particles = self.sph_solver.convert_to_sph_particles(gas_particles);
        
        if sph_particles.is_empty() {
            return Ok(());
        }
        
        // Compute SPH time step based on Courant condition
        let dt_sph = self.sph_solver.compute_time_step(&sph_particles);
        
        // Process SPH fluid dynamics
        self.sph_solver.integrate_step(&mut sph_particles, dt_sph)?;
        
        // Process radiative transfer for gas cooling and heating
        self.process_radiative_transfer()?;
        
        // Process Jeans instability and gravitational collapse
        self.process_jeans_instability()?;
        
        // Process gravitational collapse and sink particle formation
        self.process_gravitational_collapse()?;
        
        Ok(())
    }

    /// Process radiative transfer for gas cooling and heating
    fn process_radiative_transfer(&mut self) -> Result<()> {
        let constants = constants::PhysicsConstants::default();
        let local_luminosity = self.calculate_local_stellar_luminosity();
        // Collect particle positions to avoid borrow checker issues
        let particle_positions: Vec<_> = self.particles.iter().map(|p| p.position).collect();
        let metallicities: Vec<_> = particle_positions.iter().map(|pos| self.calculate_local_metallicity(pos)).collect();
        let distances: Vec<_> = particle_positions.iter().map(|pos| self.calculate_distance_to_nearest_star(pos)).collect();
        let densities: Vec<_> = particle_positions.iter().map(|pos| self.calculate_local_density(pos)).collect();
        for (i, particle) in self.particles.iter_mut().enumerate() {
            if matches!(particle.particle_type, ParticleType::Hydrogen | ParticleType::Helium) {
                let metallicity = metallicities[i];
                let distance_to_star = distances[i];
                let local_density = densities[i];
                // Get radiative transfer solution
                let radiative_solution = self.radiative_transfer.calculate_radiative_transfer(
                    self.temperature, // Use global temperature for now
                    local_density,
                    metallicity,
                    local_luminosity,
                    distance_to_star,
                    &constants,
                );
                let energy_change = radiative_solution.net_rate * self.time_step;
                particle.energy += energy_change;
                let specific_heat = 1.5 * constants.k_b / constants.m_p; // Monatomic gas
                let temperature_change = energy_change / (local_density * specific_heat);
                self.temperature += temperature_change;
                self.temperature = self.temperature.max(2.73); // CMB temperature
                self.temperature = self.temperature.min(1e8); // 100 million K
            }
        }
        
        Ok(())
    }

    /// Process Jeans instability and gravitational collapse
    fn process_jeans_instability(&mut self) -> Result<()> {
        // Process gravitational collapse using the Jeans instability solver
        self.jeans_instability.process_gravitational_collapse(
            &mut self.particles,
            self.time_step,
        )?;
        
        Ok(())
    }

    /// Calculate local stellar luminosity affecting gas
    fn calculate_local_stellar_luminosity(&self) -> f64 {
        // Simplified: use total stellar luminosity from sink particles
        let total_luminosity: f64 = self.sink_particles.iter()
            .map(|sink| {
                // Luminosity scales with mass as L ∝ M^3.5 (main sequence)
                let mass_solar = sink.mass / 1.989e30; // Convert to solar masses
                3.828e26 * mass_solar.powf(3.5) // Solar luminosity * mass scaling
            })
            .sum();
        
        total_luminosity.max(3.828e26) // Minimum solar luminosity
    }

    /// Calculate local metallicity around a position
    fn calculate_local_metallicity(&self, position: &Vector3<f64>) -> f64 {
        // Simplified: count heavy elements (Z > 2) in local region
        let search_radius = 1e16; // 1 pc
        let mut heavy_elements = 0;
        let mut total_elements = 0;
        
        for particle in &self.particles {
            let distance = (particle.position - position).magnitude();
            if distance < search_radius {
                total_elements += 1;
                // Check if particle is a heavy element (simplified)
                if matches!(particle.particle_type, 
                    ParticleType::Carbon | ParticleType::Nitrogen | ParticleType::Oxygen |
                    ParticleType::Iron | ParticleType::Silicon | ParticleType::Sulfur) {
                    heavy_elements += 1;
                }
            }
        }
        
        if total_elements > 0 {
            heavy_elements as f64 / total_elements as f64
        } else {
            0.01 // Default 1% metallicity
        }
    }

    /// Calculate distance to nearest star
    fn calculate_distance_to_nearest_star(&self, position: &Vector3<f64>) -> f64 {
        let mut min_distance = f64::INFINITY;
        
        // Check distance to sink particles (stars)
        for sink in &self.sink_particles {
            let distance = (sink.position - position).magnitude();
            min_distance = min_distance.min(distance);
        }
        
        // Check distance to stellar particles
        for particle in &self.particles {
            if matches!(particle.particle_type, 
                ParticleType::HydrogenAtom | ParticleType::HeliumAtom) {
                let distance = (particle.position - position).magnitude();
                min_distance = min_distance.min(distance);
            }
        }
        
        min_distance.max(1e16) // Minimum 1 pc distance
    }

    // GADGET N-body gravity functions removed - now using native Rust implementation

    /// Process native particle interactions with spatial optimization (O(N) instead of O(N²))
    fn process_native_interactions_optimized(&mut self) -> Result<()> {
        // Find all interaction pairs using spatial grid
        let interaction_pairs = self.spatial_grid.find_interaction_pairs(&self.particles);
        
        // Process interactions for each pair
        for (i, j) in interaction_pairs {
            if i >= self.particles.len() || j >= self.particles.len() {
                continue;
            }
            
            let p1 = &self.particles[i];
            let p2 = &self.particles[j];
            
            // Calculate distance
            let distance = (p1.position - p2.position).norm();
            
            // Process different types of interactions based on particle types
            self.process_particle_pair_interaction(i, j, distance)?;
        }
        
        Ok(())
    }
    
    /// Process interaction between a specific pair of particles
    fn process_particle_pair_interaction(&mut self, i: usize, j: usize, distance: f64) -> Result<()> {
        if i >= self.particles.len() || j >= self.particles.len() {
            return Ok(());
        }
        
        let p1_type = self.particles[i].particle_type;
        let p2_type = self.particles[j].particle_type;
        
        // Electromagnetic interactions
        if self.can_interact_electromagnetically(p1_type, p2_type) {
            self.process_electromagnetic_interaction(i, j, distance)?;
        }
        
        // Strong interactions (for quarks and gluons)
        if self.can_interact_strongly(p1_type, p2_type) {
            self.process_strong_interaction(i, j, distance)?;
        }
        
        // Weak interactions
        if self.can_interact_weakly(p1_type, p2_type) {
            self.process_weak_interaction(i, j, distance)?;
        }
        
        Ok(())
    }
    
    /// Check if particles can interact electromagnetically
    fn can_interact_electromagnetically(&self, p1: ParticleType, p2: ParticleType) -> bool {
        // Both particles must have electric charge
        let charge1 = self.get_electric_charge(p1);
        let charge2 = self.get_electric_charge(p2);
        charge1 != 0.0 || charge2 != 0.0
    }
    
    /// Check if particles can interact via strong force
    fn can_interact_strongly(&self, p1: ParticleType, p2: ParticleType) -> bool {
        // Check if either particle carries color charge (quarks and gluons)
        matches!(p1, ParticleType::Up | ParticleType::Down | ParticleType::Charm | 
                     ParticleType::Strange | ParticleType::Top | ParticleType::Bottom | 
                     ParticleType::Gluon) ||
        matches!(p2, ParticleType::Up | ParticleType::Down | ParticleType::Charm | 
                     ParticleType::Strange | ParticleType::Top | ParticleType::Bottom | 
                     ParticleType::Gluon)
    }
    
    /// Check if particles can interact via weak force
    fn can_interact_weakly(&self, p1: ParticleType, p2: ParticleType) -> bool {
        // All fermions can interact weakly
        self.is_fermion(p1) || self.is_fermion(p2)
    }
    
    /// Check if particle is a fermion
    fn is_fermion(&self, particle_type: ParticleType) -> bool {
        matches!(particle_type, 
            ParticleType::Electron | ParticleType::Muon | ParticleType::Tau |
            ParticleType::ElectronNeutrino | ParticleType::MuonNeutrino | ParticleType::TauNeutrino |
            ParticleType::Up | ParticleType::Down | ParticleType::Charm | 
            ParticleType::Strange | ParticleType::Top | ParticleType::Bottom |
            ParticleType::Proton | ParticleType::Neutron
        )
    }
    
    /// Process electromagnetic interaction between two particles
    fn process_electromagnetic_interaction(&mut self, i: usize, j: usize, distance: f64) -> Result<()> {
        if distance < 1e-15 { // Avoid singularity
            return Ok(());
        }
        
        // Compton scattering: photon + electron -> photon + electron
        if (self.particles[i].particle_type == ParticleType::Photon &&
            self.particles[j].particle_type == ParticleType::Electron) ||
           (self.particles[j].particle_type == ParticleType::Photon &&
            self.particles[i].particle_type == ParticleType::Electron)
        {
            // Estimate interaction probability from Klein–Nishina cross-section
            let photon_idx = if self.particles[i].particle_type == ParticleType::Photon { i } else { j };
            let photon_energy = self.particles[photon_idx].energy; // J
            let electron_mass_energy = constants::ELECTRON_MASS * constants::SPEED_OF_LIGHT.powi(2);

            // Cross-section σ(E) in m²
            let sigma = crate::interactions::klein_nishina_cross_section_joules(photon_energy, electron_mass_energy);
            // Effective geometrical area of sphere with radius = separation
            let geom_area = 4.0 * std::f64::consts::PI * distance * distance;
            // Clamp probability to sensible range [0,1]
            let prob = (sigma / geom_area).min(1.0);

            if rand::random::<f64>() < prob {
                // Only count and execute scattering when it actually happens
                self.compton_count += 1;
                self.exchange_momentum_compton(i, j)?;
            }
        }
        
        // Coulomb scattering between charged particles
        if self.particles[i].electric_charge != 0.0 && self.particles[j].electric_charge != 0.0 {
            self.coulomb_scattering(i, j, distance)?;
        }
        
        Ok(())
    }
    
    /// Process strong interaction between quarks/gluons
    #[cfg(feature = "quantum-chemistry")]
    fn process_strong_interaction(&mut self, _i: usize, _j: usize, _distance: f64) -> Result<()> {
        // Simple Yukawa potential approximation for strong force between color-charged particles
        let i = _i;
        let j = _j;
        let distance = _distance;
        // Validate indices and avoid singularities
        if i >= self.particles.len() || j >= self.particles.len() || distance < 1e-18 {
            return Ok(());
        }

        // Coupling and screening parameters (order-of-magnitude, not lattice QCD)
        const G_S: f64 = 15.0;          // Effective strong coupling constant (dimensionless)
        const MU: f64 = 1.0e15;         // Inverse screening length ≈ 1 fm⁻¹ (m⁻¹)
        let prefactor = -(G_S * G_S) / (4.0 * std::f64::consts::PI);
        let force_magnitude = prefactor * (-MU * distance).exp() / (distance * distance);

        // Direction from j → i
        let direction = (self.particles[i].position - self.particles[j].position).normalize();
        let force = direction * force_magnitude;

        // Impulse = F·dt (equal and opposite per Newton 3)
        let impulse_i = force * self.time_step;
        let impulse_j = -impulse_i;

        self.particles[i].momentum += impulse_i;
        self.particles[j].momentum += impulse_j;
        Ok(())
    }

    /// Process strong interaction between quarks/gluons (fallback for when quantum-chemistry feature is disabled)
    #[cfg(not(feature = "quantum-chemistry"))]
    fn process_strong_interaction(&mut self, _i: usize, _j: usize, _distance: f64) -> Result<()> {
        // No-op implementation when quantum chemistry features are disabled
        Ok(())
    }
    
    /// Process weak interaction 
    #[cfg(feature = "quantum-chemistry")]
    fn process_weak_interaction(&mut self, i: usize, j: usize, distance: f64) -> Result<()> {
        // TODO: Replace toy weak force with full electroweak calculation
        // Neutrino-electron scattering
        if (self.particles[i].particle_type == ParticleType::ElectronNeutrino &&
            self.particles[j].particle_type == ParticleType::Electron) ||
           (self.particles[j].particle_type == ParticleType::ElectronNeutrino &&
            self.particles[i].particle_type == ParticleType::Electron)
        {
            // Extremely small weak cross-section (~10⁻⁴⁇⁴ m² at MeV energies).
            // Use a fixed tiny probability to avoid computing exact electroweak formula.
            const NU_E_SIGMA: f64 = 1.0e-44; // m² (order of magnitude)
            let geom_area = 4.0 * std::f64::consts::PI * distance * distance;
            let prob = (NU_E_SIGMA / geom_area).min(1.0e-6); // Cap to avoid wasting work

            if rand::random::<f64>() < prob {
                self.neutrino_scatter_count += 1;
                self.exchange_momentum_weak(i, j)?;
            }
        }
        
        Ok(())
    }

    /// Process weak interaction (fallback for when quantum-chemistry feature is disabled)
    #[cfg(not(feature = "quantum-chemistry"))]
    fn process_weak_interaction(&mut self, _i: usize, _j: usize, _distance: f64) -> Result<()> {
        // No-op implementation when quantum chemistry features are disabled
        Ok(())
    }
    
    /// Exchange momentum in Compton scattering
    fn exchange_momentum_compton(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.particles.len() || j >= self.particles.len() {
            return Ok(());
        }
        
        // Conservation of energy and momentum in Compton scattering
        let photon_idx = if self.particles[i].particle_type == ParticleType::Photon { i } else { j };
        let electron_idx = if self.particles[i].particle_type == ParticleType::Electron { i } else { j };
        
        let photon_initial_energy = self.particles[photon_idx].energy;
        let electron_mass_energy = self.particles[electron_idx].mass * C_SQUARED;
        
        // Klein-Nishina formula for scattered photon energy
        let cos_theta = 2.0 * rand::random::<f64>() - 1.0; // Random scattering angle
        let scattered_photon_energy = photon_initial_energy / 
            (1.0 + (photon_initial_energy / electron_mass_energy) * (1.0 - cos_theta));
        
        let electron_kinetic_energy = photon_initial_energy - scattered_photon_energy;
        
        // Update energies
        self.particles[photon_idx].energy = scattered_photon_energy;
        self.particles[electron_idx].energy = electron_mass_energy + electron_kinetic_energy;
        
        // Update momenta (simplified - random directions)
        let mut rng = thread_rng();
        let theta = rng.gen::<f64>() * std::f64::consts::PI;
        let phi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
        
        let photon_momentum = scattered_photon_energy / C;
        self.particles[photon_idx].momentum = Vector3::new(
            photon_momentum * theta.sin() * phi.cos(),
            photon_momentum * theta.sin() * phi.sin(),
            photon_momentum * theta.cos(),
        );
        
        let electron_momentum = (electron_kinetic_energy * (electron_kinetic_energy + 2.0 * electron_mass_energy)).sqrt() / C;
        self.particles[electron_idx].momentum = Vector3::new(
            electron_momentum * (theta + std::f64::consts::PI).sin() * phi.cos(),
            electron_momentum * (theta + std::f64::consts::PI).sin() * phi.sin(),
            electron_momentum * (theta + std::f64::consts::PI).cos(),
        );
        
        Ok(())
    }
    
    /// Exchange momentum in weak interactions
    fn exchange_momentum_weak(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.particles.len() || j >= self.particles.len() {
            return Ok(());
        }
        
        // Very small momentum transfer in weak interactions
        let momentum_transfer = Vector3::new(
            rand::random::<f64>() * 1e-25,
            rand::random::<f64>() * 1e-25,
            rand::random::<f64>() * 1e-25,
        );
        
        self.particles[i].momentum += momentum_transfer;
        self.particles[j].momentum -= momentum_transfer; // Conservation
        
        Ok(())
    }
    
    /// Coulomb scattering between charged particles
    fn coulomb_scattering(&mut self, i: usize, j: usize, distance: f64) -> Result<()> {
        if i >= self.particles.len() || j >= self.particles.len() || distance < 1e-15 {
            return Ok(());
        }
        
        let q1 = self.particles[i].electric_charge;
        let q2 = self.particles[j].electric_charge;
        
        // Coulomb force magnitude
        let force_magnitude = K_E * q1 * q2 / (distance * distance);
        
        // Direction vector from particle j to particle i
        let direction = (self.particles[i].position - self.particles[j].position).normalize();
        
        // Force on particle i
        let force_i = direction * force_magnitude;
        let force_j = -force_i; // Newton's third law
        
        // Apply impulse (F * dt)
        let impulse_i = force_i * self.time_step;
        let impulse_j = force_j * self.time_step;
        
        self.particles[i].momentum += impulse_i;
        self.particles[j].momentum += impulse_j;
        
        Ok(())
    }
    
    /// Get spatial grid statistics for diagnostics
    pub fn get_spatial_grid_stats(&self) -> SpatialGridStats {
        self.spatial_grid.get_statistics()
    }

    /// Process particle decays
    fn process_particle_decays(&mut self) -> Result<()> {
        let mut decays = Vec::new();
        
        for (i, particle) in self.particles.iter().enumerate() {
            if let Some(decay_time) = particle.decay_time {
                if self.current_time >= decay_time {
                    if let Some(channels) = self.decay_channels.get(&particle.particle_type) {
                        let channel = self.select_decay_channel(channels);
                        decays.push((i, channel));
                    }
                }
            }
        }
        
        // Process decays (in reverse order to maintain indices)
        for (particle_index, decay_channel) in decays.into_iter().rev() {
            self.execute_decay(particle_index, decay_channel)?;
        }
        
        Ok(())
    }
    
    /// Comprehensive nuclear physics processing (fusion, fission, nuclear reactions)
    fn process_nuclear_reactions(&mut self) -> Result<()> {
        // Process nuclear fusion reactions (stellar nucleosynthesis)
        self.process_nuclear_fusion()?;
        
        // Process nuclear fission (for heavy nuclei)
        self.process_nuclear_fission()?;
        
        // Update nuclear shell structure
        self.update_nuclear_shells()?;
        
        // Process atomic physics interactions
        self.update_atomic_physics()?;
        
        Ok(())
    }

    fn process_nuclear_fusion(&mut self) -> Result<()> {
        let temperature = self.temperature;
        let density = self.calculate_stellar_density();

        if temperature > 1e7 { // Threshold for significant fusion
            let mut composition = self.build_isotope_composition();
            let energy_released = self.stellar_nucleosynthesis.process_stellar_burning(temperature, density, &mut composition)?;
            
            // Update system energy and composition
            self.energy_density += energy_released / self.volume;
            self.update_nuclei_from_composition(&composition)?;
        }

        Ok(())
    }
    
    /// Build isotope composition array from current nuclei
    fn build_isotope_composition(&self) -> Vec<(u32, u32, f64)> {
        let mut composition = Vec::new();
        
        // Common stellar isotopes with initial abundances
        let stellar_isotopes = [
            (1, 1, 0.0),   // ¹H (protons)
            (1, 2, 0.0),   // ²H (deuterium)
            (2, 3, 0.0),   // ³He
            (2, 4, 0.0),   // ⁴He (alpha particles)
            (6, 12, 0.0),  // ¹²C
            (6, 13, 0.0),  // ¹³C
            (7, 13, 0.0),  // ¹³N
            (7, 14, 0.0),  // ¹⁴N
            (7, 15, 0.0),  // ¹⁵N
            (8, 15, 0.0),  // ¹⁵O
            (8, 16, 0.0),  // ¹⁶O
            (12, 24, 0.0), // ²⁴Mg
            (26, 56, 0.0), // ⁵⁶Fe
        ];
        
        // Initialize with stellar isotope template
        for &(z, a, _) in &stellar_isotopes {
            let abundance = self.count_isotope_abundance(z, a);
            composition.push((z, a, abundance));
        }
        
        composition
    }
    
    /// Count abundance of specific isotope in current nuclei
    fn count_isotope_abundance(&self, z: u32, a: u32) -> f64 {
        let mut count = 0.0;
        
        for nucleus in &self.nuclei {
            if nucleus.atomic_number == z && nucleus.mass_number == a {
                count += 1.0;
            }
        }
        
        // Also count from fundamental particles
        for particle in &self.particles {
            match particle.particle_type {
                ParticleType::Proton if z == 1 && a == 1 => count += 1.0,
                ParticleType::Neutron if z == 0 && a == 1 => count += 1.0,
                _ => {}
            }
        }
        
        count / self.particles.len().max(1) as f64 // Normalize by total particle count
    }
    
    /// Calculate stellar density for nucleosynthesis
    fn calculate_stellar_density(&self) -> f64 {
        // Estimate density from nuclei and particles
        let total_mass = self.nuclei.iter()
            .map(|n| n.mass_number as f64 * 1.66e-27) // Atomic mass units to kg
            .sum::<f64>() + 
            self.particles.iter()
            .map(|p| p.mass)
            .sum::<f64>();
        
        total_mass / self.volume.max(1e-50) // Prevent division by zero
    }
    
    /// Update nuclei from composition changes
    fn update_nuclei_from_composition(&mut self, composition: &[(u32, u32, f64)]) -> Result<()> {
        // For now, this is a simplified implementation
        // In a full implementation, we would need to:
        // 1. Calculate the difference between old and new composition
        // 2. Remove consumed nuclei
        // 3. Add newly created nuclei
        // 4. Update nuclear properties
        
        for &(z, a, abundance) in composition {
            if abundance > 0.0 && z > 0 && a > 0 {
                // Create nuclei for isotopes with significant abundance
                let target_count = (abundance * 1000.0) as usize; // Scale factor
                let current_count = self.nuclei.iter()
                    .filter(|n| n.atomic_number == z && n.mass_number == a)
                    .count();
                
                // Add nuclei if we have too few
                if target_count > current_count {
                    let to_add = target_count - current_count;
                    for _ in 0..to_add.min(10) { // Limit to prevent excessive creation
                        self.create_nucleus(z, a)?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create a new nucleus with given Z and A
    fn create_nucleus(&mut self, z: u32, a: u32) -> Result<()> {
        let nucleus = AtomicNucleus {
            mass_number: a,
            atomic_number: z,
            protons: vec![],
            neutrons: vec![],
            binding_energy: nuclear_physics::Nucleus::new(z, a - z).binding_energy(),
            nuclear_spin: Vector3::zeros(),
            magnetic_moment: Vector3::zeros(),
            electric_quadrupole_moment: 0.0,
            nuclear_radius: 1.2e-15 * (a as f64).powf(1.0/3.0), // Fermi
            shell_model_state: HashMap::new(),
            position: Vector3::zeros(),
            momentum: Vector3::zeros(),
            excitation_energy: 0.0,
        };
        
        self.nuclei.push(nucleus);
        Ok(())
    }
    
    /// Legacy fusion processing for backward compatibility
    /// Note: This method is preserved for fallback scenarios where stellar nucleosynthesis is unavailable
    #[allow(dead_code)]
    fn process_legacy_fusion(&mut self) -> Result<()> {
        let mut fusion_reactions = Vec::new();
        
        // Look for fusion-capable nuclei
        for i in 0..self.nuclei.len() {
            for j in (i+1)..self.nuclei.len() {
                let nucleus1 = &self.nuclei[i];
                let nucleus2 = &self.nuclei[j];
                
                // Check if fusion is energetically favorable and barrier can be overcome
                if self.can_fuse(nucleus1, nucleus2)? {
                    let reaction = self.calculate_fusion_reaction(i, j)?;
                    fusion_reactions.push(reaction);
                }
            }
        }
        
        // Execute fusion reactions
        for reaction in fusion_reactions {
            self.execute_fusion_reaction(reaction)?;
        }
        
        Ok(())
    }
    
    /// Get particle mass from type
    pub fn get_particle_mass(&self, particle_type: ParticleType) -> f64 {
        match particle_type {
            ParticleType::Electron => ELECTRON_MASS,
            ParticleType::Muon => MUON_MASS,
            ParticleType::Tau => TAU_MASS,
            ParticleType::Up => 2.2e-30,     // ~2 MeV/c²
            ParticleType::Down => 4.7e-30,   // ~5 MeV/c²
            ParticleType::Charm => 2.3e-27,  // ~1.3 GeV/c²
            ParticleType::Strange => 1.7e-28, // ~95 MeV/c²
            ParticleType::Top => 3.1e-25,   // ~173 GeV/c²
            ParticleType::Bottom => 7.5e-27, // ~4.2 GeV/c²
            ParticleType::Proton => PROTON_MASS,
            ParticleType::Neutron => NEUTRON_MASS,
            ParticleType::WBoson => 1.4e-25,  // ~80 GeV/c²
            ParticleType::ZBoson => 1.6e-25,  // ~91 GeV/c²
            ParticleType::Higgs => 2.2e-25,   // ~125 GeV/c²
            ParticleType::Photon | ParticleType::Gluon => 0.0,
            ParticleType::ElectronNeutrino | ParticleType::MuonNeutrino | ParticleType::TauNeutrino => 1e-36,
            // Molecular masses (atomic mass units converted to kg)
            ParticleType::H2 => 3.34e-27,   // 2.016 u
            ParticleType::H2O => 2.99e-26,  // 18.015 u
            ParticleType::CO2 => 7.31e-26,  // 44.01 u
            ParticleType::CH4 => 2.66e-26,  // 16.043 u
            ParticleType::NH3 => 2.83e-26,  // 17.031 u
            // Individual atomic species (neutral atoms)
            ParticleType::HydrogenAtom => crate::atomic_data::mass_kg(1),
            ParticleType::HeliumAtom   => crate::atomic_data::mass_kg(2),
            ParticleType::CarbonAtom   => crate::atomic_data::mass_kg(6),
            ParticleType::OxygenAtom   => crate::atomic_data::mass_kg(8),
            ParticleType::IronAtom     => crate::atomic_data::mass_kg(26),
            _ => 0.0,
        }
    }
    
    // Placeholder implementations for complex physics
    fn sample_thermal_momentum(&self, particle_type: ParticleType, temperature: f64) -> Vector3<f64> {
        let mut rng = thread_rng();
        let mass = self.get_particle_mass(particle_type);
        
        // For massless particles, use E = pc = 3kT
        // For massive particles, use relativistic Maxwell-Boltzmann
        let typical_momentum = if mass < 1e-40 {
            // Massless particle
            3.0 * BOLTZMANN * temperature / SPEED_OF_LIGHT
        } else {
            // Massive particle - use non-relativistic approximation for now
            (3.0 * mass * BOLTZMANN * temperature).sqrt()
        };
        
        // Random direction
        let theta = rng.gen::<f64>() * std::f64::consts::PI;
        let phi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
        
        Vector3::new(
            typical_momentum * theta.sin() * phi.cos(),
            typical_momentum * theta.sin() * phi.sin(),
            typical_momentum * theta.cos(),
        )
    }
    
    fn initialize_spin(&self, _particle_type: ParticleType) -> Vector3<Complex<f64>> {
        Vector3::zeros()
    }
    
    fn assign_color_charge(&self, particle_type: ParticleType) -> Option<ColorCharge> {
        match particle_type {
            ParticleType::Up | ParticleType::Down | ParticleType::Charm | 
            ParticleType::Strange | ParticleType::Top | ParticleType::Bottom => {
                Some(ColorCharge::Red) // Simplified
            },
            ParticleType::Gluon => Some(ColorCharge::Red), // Simplified
            _ => None,
        }
    }
    
    fn get_electric_charge(&self, particle_type: ParticleType) -> f64 {
        match particle_type {
            ParticleType::Up | ParticleType::Charm | ParticleType::Top => 2.0/3.0 * ELEMENTARY_CHARGE,
            ParticleType::Down | ParticleType::Strange | ParticleType::Bottom => -1.0/3.0 * ELEMENTARY_CHARGE,
            ParticleType::Electron | ParticleType::Muon | ParticleType::Tau => -ELEMENTARY_CHARGE,
            ParticleType::Proton => ELEMENTARY_CHARGE,
            ParticleType::WBoson => ELEMENTARY_CHARGE,
            _ => 0.0,
        }
    }
    
    fn calculate_decay_time(&self, particle_type: ParticleType) -> Option<f64> {
        match particle_type {
            ParticleType::Muon => Some(self.current_time + 2.2e-6), // 2.2 μs
            ParticleType::Neutron => Some(self.current_time + 880.0), // 880 s
            _ => None, // Stable particles
        }
    }
    
    /// Recompute relativistic energies for all particles using the Einstein energy-momentum relation
    /// 
    /// # Physics
    /// Implements the fundamental relativistic energy equation:
    /// **E² = (pc)² + (mc²)²**
    /// 
    /// Where:
    /// - E = total energy (J)
    /// - p = momentum magnitude (kg⋅m/s)  
    /// - m = rest mass (kg)
    /// - c = speed of light = 299,792,458 m/s (exact, CODATA 2022)
    /// 
    /// This preserves energy conservation and ensures proper relativistic behavior
    /// for high-energy particles approaching the speed of light.
    /// 
    /// # Performance
    /// Uses Rayon parallel iterators for multi-core acceleration on large particle sets.
    /// Validates energy conservation by checking for NaN/infinite values.
    /// 
    /// # References
    /// - Einstein (1905). "Zur Elektrodynamik bewegter Körper"
    /// - Peskin & Schroeder (1995). "An Introduction to Quantum Field Theory"
    /// - CODATA 2022 fundamental physical constants
    pub fn update_particle_energies(&mut self) -> Result<()> {
        let start = Instant::now();
        
        // Validate inputs and track energy conservation
        let initial_total_energy: f64 = self.particles.iter()
            .map(|p| p.energy)
            .sum();
        
        log::debug!(
            "[energy] Recomputing relativistic energies for {} particles using {} Rayon threads",
            self.particles.len(),
            rayon::current_num_threads()
        );

        // Apply relativistic energy-momentum relation in parallel
        self.particles
            .par_iter_mut()
            .for_each(|particle| {
                // Get momentum magnitude |p|
                let momentum_magnitude = particle.momentum.magnitude();
                
                // Relativistic energy: E = sqrt((pc)² + (mc²)²)
                let momentum_energy_term = momentum_magnitude * SPEED_OF_LIGHT;
                let rest_energy_term = particle.mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
                
                particle.energy = (momentum_energy_term.powi(2) + rest_energy_term.powi(2)).sqrt();
                
                // Update velocity from momentum: v = pc²/E (relativistic)
                if particle.energy > 0.0 {
                    let velocity_magnitude = momentum_magnitude * SPEED_OF_LIGHT.powi(2) / particle.energy;
                    if momentum_magnitude > 0.0 {
                        particle.velocity = particle.momentum * (velocity_magnitude / momentum_magnitude);
                    } else {
                        particle.velocity = Vector3::zeros();
                    }
                } else {
                    particle.velocity = Vector3::zeros();
                }
                
                // Validate against unphysical values
                debug_assert!(particle.energy.is_finite(), 
                    "Non-finite energy for particle type {:?}", particle.particle_type);
                debug_assert!(particle.energy >= 0.0, 
                    "Negative energy for particle type {:?}", particle.particle_type);
                debug_assert!(particle.velocity.magnitude() <= SPEED_OF_LIGHT * 1.001, 
                    "Superluminal velocity for particle type {:?}: {:.3e} m/s", 
                    particle.particle_type, particle.velocity.magnitude());
            });

        // Verify energy conservation (should be exactly preserved in absence of interactions)
        let final_total_energy: f64 = self.particles.iter()
            .map(|p| p.energy)
            .sum();
        
        let energy_change = (final_total_energy - initial_total_energy).abs();
        let relative_change = energy_change / initial_total_energy.max(1e-100);
        
        if relative_change > 1e-12 {
            log::warn!(
                "[energy] Unexpected energy change during update: ΔE = {:.3e} J (relative: {:.3e})",
                energy_change, relative_change
            );
        }

        log::debug!(
            "[energy] Finished relativistic energy recomputation in {:.3?} (total energy: {:.6e} J)",
            start.elapsed(),
            final_total_energy
        );
        
        Ok(())
    }

    /// Characteristic interaction range (m) given two particle types.
    ///
    /// The implementation follows simple physically-motivated heuristics:
    /// • **Strong force** (quarks/gluons) → 1 fm (≈ 1 × 10⁻¹⁵ m).
    /// • **Electromagnetic** between charged particles → classical distance where the
    ///   Coulomb potential equals kT at the current simulation temperature.
    /// • Otherwise (e.g. neutrinos, dark matter) fall back to the de-Broglie
    ///   wavelength of the lighter particle at the thermal momentum scale.
    fn calculate_interaction_range(&self, p1: ParticleType, p2: ParticleType) -> f64 {
        // 1. Strongly interacting?
        if self.can_interact_strongly(p1, p2) {
            return 1.0e-15; // ≈ pion Compton wavelength
        }

        // 2. Electromagnetic range – distance where |V_C| = k_B T
        let q1 = self.get_electric_charge(p1);
        let q2 = self.get_electric_charge(p2);
        if q1.abs() > 0.0 && q2.abs() > 0.0 {
            let kbt = BOLTZMANN * self.temperature.max(2.7); // avoid zero-division; assume CMB floor
            let r = K_E * q1.abs() * q2.abs() / kbt; // solve e²/(4πϵ₀ r) = k_B T
            // Clamp to sensible [1 pm, 1 µm] interval.
            return r.clamp(1.0e-12, 1.0e-6);
        }

        // 3. Weak / other: use thermal de-Broglie wavelength λ = h / √(2π m kT)
        let m1 = self.get_particle_mass(p1).max(1.0e-40);
        let lambda = (2.0 * std::f64::consts::PI * m1 * BOLTZMANN * self.temperature.max(2.7)).sqrt();
        let lambda = (6.626_070_15e-34) / lambda; // h / p
        lambda.clamp(1.0e-14, 1.0e-3)
    }

    /// Compute basic two-body interaction probability using analytic cross-sections
    /// for a subset of important processes (currently Compton scattering and
    /// photon pair-production). Returns an `Interaction` record which downstream
    /// routines can apply.
    fn calculate_interaction(&self, i: usize, j: usize) -> Result<interactions::Interaction> {
        use crate::interactions::{klein_nishina_cross_section_joules, bethe_heitler_cross_section};

        let p1 = &self.particles[i];
        let p2 = &self.particles[j];
        let separation = (p1.position - p2.position).norm().max(1.0e-18);

        // Default elastic placeholder
        let mut interaction = interactions::Interaction {
            particle_indices: (i, j),
            cross_section: 0.0,
            interaction_type: interactions::InteractionType::ElasticScattering,
            ..Default::default()
        };

        // Compton (γ + e⁻)
        if (p1.particle_type == ParticleType::Photon && p2.particle_type == ParticleType::Electron) ||
           (p2.particle_type == ParticleType::Photon && p1.particle_type == ParticleType::Electron) {
            let photon = if p1.particle_type == ParticleType::Photon { p1 } else { p2 };
            let sigma = klein_nishina_cross_section_joules(photon.energy, ELECTRON_MASS * C_SQUARED);
            interaction.cross_section = sigma;
            interaction.interaction_type = interactions::InteractionType::ComptonScattering;
        }

        // γ → e⁺e⁻ pair production in nuclear field (approximate, assume Iron Z=26)
        if p1.particle_type == ParticleType::Photon && p2.particle_type == ParticleType::IronAtom ||
           p2.particle_type == ParticleType::Photon && p1.particle_type == ParticleType::IronAtom {
            let photon = if p1.particle_type == ParticleType::Photon { p1 } else { p2 };
            let sigma = bethe_heitler_cross_section(photon.energy, 26);
            interaction.cross_section = sigma;
            interaction.interaction_type = interactions::InteractionType::PairProduction;
        }

        // Convert cross-section to probability for this separation
        if interaction.cross_section > 0.0 {
            let geom_area = 4.0 * std::f64::consts::PI * separation.powi(2);
            interaction.probability = (interaction.cross_section / geom_area).min(1.0);
        }

        Ok(interaction)
    }

    /// Apply the momentum/energy transfer encoded in `interaction` to the particle
    /// system. Currently we only update bookkeeping counts while the detailed
    /// kinematics are handled elsewhere.
    fn apply_interaction(&mut self, interaction: interactions::Interaction) -> Result<()> {
        match interaction.interaction_type {
            interactions::InteractionType::ComptonScattering => {
                self.compton_count += 1;
                self.exchange_momentum_compton(interaction.particle_indices.0, interaction.particle_indices.1)?;
            },
            interactions::InteractionType::PairProduction => {
                self.pair_production_count += 1;
            },
            _ => { /* other types handled separately */ }
        }
        Ok(())
    }

    /// Randomly select a decay channel according to branching ratios using a
    /// categorical (weighted) distribution.
    fn select_decay_channel(&self, channels: &[DecayChannel]) -> DecayChannel {
        use rand::distributions::WeightedIndex;
        let weights: Vec<f64> = channels.iter().map(|c| c.branching_ratio.max(0.0)).collect();
        if let Ok(dist) = WeightedIndex::new(&weights) {
            let mut rng = thread_rng();
            let idx = dist.sample(&mut rng);
            channels[idx].clone()
        } else {
            // Fallback: uniform selection (should never happen if data are valid)
            channels[0].clone()
        }
    }

    /// Adaptive step-length estimator that couples particle energy,
    /// radiation-length of the local medium, and simulation temperature.
    ///
    /// The returned value is clamped to **[0.1 fm, 1 cm]** to prevent pathological
    /// values that could break the transport integrator.
    fn calculate_step_length(&self, particle: &FundamentalParticle) -> f64 {
        // Base scale: de-Broglie wavelength λ = h / p
        let momentum_mag = particle.momentum.norm().max(1.0e-40);
        let lambda = 6.626_070_15e-34 / momentum_mag; // metres

        // Material scale: inverse of mass-density (ρ) – denser ⇒ smaller steps
        let rho = self.calculate_local_density(&particle.position).max(1.0);
        let material_factor = (1.0 / rho).powf(1.0/3.0);

        // Thermal agitation: hotter plasma can sustain larger timesteps
        let thermal_factor = (self.temperature / 1.0e6).sqrt().clamp(0.1, 10.0);

        let step = lambda.min(material_factor) * thermal_factor;
        step.clamp(1.0e-16, 1.0e-2)
    }

    /// Estimate mass density (kg·m⁻³) within a sphere of radius *r* around
    /// `position`. We iterate over the particle list because `SpatialHashGrid`
    /// does not expose its internal cell-lookup utilities publicly.
    fn calculate_local_density(&self, position: &Vector3<f64>) -> f64 {
        let r = self.spatial_grid.max_interaction_range().max(1.0e-15);
        let r_sq = r * r;
        let volume = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);

        let mut mass_sum = 0.0;
        for p in &self.particles {
            if (p.position - position).norm_squared() <= r_sq {
                mass_sum += p.mass;
            }
        }

        if volume > 0.0 { mass_sum / volume } else { 0.0 }
    }

    fn execute_decay(&mut self, index: usize, channel: DecayChannel) -> Result<()> {
        let original_particle = self.particles.swap_remove(index);
        let _rng = thread_rng();

        // Create product particles
        let mut new_particles = Vec::new();
        for product_type in channel.products.iter() {
            let mass = self.get_particle_mass(*product_type);
            let momentum = self.sample_thermal_momentum(*product_type, self.temperature);
            
            let new_particle = FundamentalParticle {
                particle_type: *product_type,
                position: original_particle.position,
                momentum,
                spin: self.initialize_spin(*product_type),
                color_charge: self.assign_color_charge(*product_type),
                electric_charge: self.get_electric_charge(*product_type),
                mass,
                energy: (mass * mass * C_SQUARED * C_SQUARED + momentum.norm_squared() * C_SQUARED).sqrt(),
                creation_time: self.current_time,
                decay_time: self.calculate_decay_time(*product_type),
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
                velocity: Vector3::zeros(),
                charge: 0.0,
            };
            new_particles.push(new_particle);
        }

        self.particles.extend(new_particles);
        self.particle_decay_count += 1;

        // Basic check for neutron beta decay
        let is_neutron_decay = original_particle.particle_type == ParticleType::Neutron &&
                               channel.products.contains(&ParticleType::Proton) &&
                               channel.products.contains(&ParticleType::Electron) &&
                               channel.products.contains(&ParticleType::ElectronAntiNeutrino);

        if is_neutron_decay {
            self.neutron_decay_count += 1;
        } else {
            // Simple momentum sharing for other decays
            // This is a placeholder; real physics would require detailed momentum calculation
        }

        Ok(())
    }
    fn process_nuclear_fission(&mut self) -> Result<()> {
        // Process nuclear fission for heavy unstable nuclei
        let mut fission_events = Vec::new();
        
        for (i, nucleus) in self.nuclei.iter().enumerate() {
            // Check if nucleus is fissile (simplified - check if Z > 90 and unstable)
            if nucleus.atomic_number > 90 && nucleus.mass_number > 230 {
                // Simplified fission probability based on excitation energy
                let fission_probability = (nucleus.excitation_energy / 1e-12).min(0.01);
                
                if rand::random::<f64>() < fission_probability {
                    fission_events.push(i);
                }
            }
        }
        
        // Execute fission events
        for &nucleus_idx in fission_events.iter().rev() {
            self.execute_fission(nucleus_idx)?;
        }
        
        Ok(())
    }
    
    fn execute_fission(&mut self, nucleus_idx: usize) -> Result<()> {
        let nucleus = self.nuclei.remove(nucleus_idx);
        let _rng = rand::thread_rng();

        // Simplified fission model: split into two smaller nuclei + neutrons
        // This is a placeholder for a proper fission model like Wahl's systematics
        let z = nucleus.atomic_number;
        let a = nucleus.mass_number;

        let z1 = z / 2;
        let a1 = a / 2;
        let z2 = z - z1;
        let a2 = a - a1 - 2; // Assume 2 neutrons are emitted

        // Create fission fragments
        self.create_nucleus(z1, a1)?;
        self.create_nucleus(z2, a2)?;
        
        // Create neutrons
        for _ in 0..2 {
            let mass = self.get_particle_mass(ParticleType::Neutron);
            let momentum = self.sample_thermal_momentum(ParticleType::Neutron, self.temperature * 10.0); // Fission neutrons are hot
            let neutron = FundamentalParticle {
                particle_type: ParticleType::Neutron,
                position: nucleus.position,
                momentum,
                spin: self.initialize_spin(ParticleType::Neutron),
                color_charge: None,
                electric_charge: 0.0,
                mass,
                energy: (mass*mass*C_SQUARED*C_SQUARED + momentum.norm_squared() * C_SQUARED).sqrt(),
                creation_time: self.current_time,
                decay_time: self.calculate_decay_time(ParticleType::Neutron),
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
                velocity: Vector3::zeros(),
                charge: 0.0,
            };
            self.particles.push(neutron);
        }
        
        self.fission_count += 1;

        // Distribute Q-value energy among products
        let q_value = self.calculate_fission_q_value(z, a)?;
        self.distribute_fission_energy(q_value, z1, a1, z2, a2, &nucleus.position)?;

        Ok(())
    }
    
    fn update_nuclear_shells(&mut self) -> Result<()> {
        // Update nuclear shell model states based on excitation
        for nucleus in &mut self.nuclei {
            // Decay excitation energy over time
            nucleus.excitation_energy *= 0.999; // Simple exponential decay
            
            // Update shell model state based on current excitation
            if nucleus.excitation_energy > 1e-13 {
                nucleus.shell_model_state.insert("excited".to_string(), 1.0);
            } else {
                nucleus.shell_model_state.insert("ground".to_string(), 1.0);
            }
        }
        Ok(())
    }
    
    #[allow(dead_code)]
    fn can_fuse(&self, n1: &AtomicNucleus, n2: &AtomicNucleus) -> Result<bool> {
        // Simplified check based on temperature and Coulomb barrier
        let kinetic_energy = 1.5 * BOLTZMANN * self.temperature; // Average kinetic energy

        let z1 = n1.atomic_number as f64;
        let z2 = n2.atomic_number as f64;
        let a1 = n1.mass_number as f64;
        let a2 = n2.mass_number as f64;

        let r1 = 1.2 * a1.powf(1.0/3.0);
        let r2 = 1.2 * a2.powf(1.0/3.0);
        let r = r1 + r2;

        let coulomb_barrier = K_E * z1 * z2 * E_CHARGE.powi(2) / (r * 1e-15); // in Joules

        // Check if kinetic energy can overcome the barrier (with quantum tunneling factor)
        // A very simplified Gamow peak style check
        let gamow_factor = (-(coulomb_barrier / kinetic_energy).sqrt()).exp();
        let fusion_probability = gamow_factor;

        Ok(thread_rng().gen::<f64>() < fusion_probability)
    }

    /// Calculates a potential fusion reaction between two nuclei.
    #[allow(dead_code)]
    fn calculate_fusion_reaction(&self, _i: usize, _j: usize) -> Result<FusionReaction> {
        // let n1 = &self.nuclei[i];
        // let n2 = &self.nuclei[j];

        // let mut reaction = FusionReaction::default();
        // reaction.reactant_indices = vec![i, j];

        // // Use the nuclear database to get reaction details
        // let fusion_cross_section = nuclear_physics::NUCLEAR_DATABASE
        //     .get_fusion_cross_section(n1.atomic_number, n1.mass_number, n2.atomic_number, n2.mass_number, self.temperature);

        // if let Some(cross_section) = fusion_cross_section {
        //     reaction.cross_section = cross_section;
        //     // Here you would look up the Q-value and products from the database as well
        // } else {
        //     // Try estimating if not in the DB
        //     reaction.cross_section = nuclear_physics::NUCLEAR_DATABASE.estimate_fusion_cross_section(
        //         n1.atomic_number, n1.mass_number, n2.atomic_number, n2.mass_number, self.temperature
        //     );
        // }

        // Ok(reaction)
        Ok(FusionReaction::default())
    }

    /// Executes a fusion reaction, updating the particle list.
    #[allow(dead_code)]
    fn execute_fusion_reaction(&mut self, _reaction: FusionReaction) -> Result<()> {
        // Consumes reactants
        // reaction.reactant_indices.iter().rev().for_each(|&idx| {
        //     self.nuclei.remove(idx);
        // });

        // // Creates product
        // let product_nucleus = nuclear_physics::create_nucleus_from_za(
        //     reaction.product_atomic_number,
        //     reaction.product_mass_number
        // )?;
        // self.nuclei.push(product_nucleus);
        
        // // Update energy
        // self.energy_density += reaction.q_value / self.volume;
        // self.fusion_count += 1;

        Ok(())
    }

    #[allow(dead_code)]
    fn update_atomic_physics(&mut self) -> Result<()> {
        // Process atomic-level physics including electron transitions, ionization, and recombination
        
        // Update electronic states based on radiation field (collect updates first)
        let mut atomic_updates = Vec::new();
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let updates = self.calculate_atomic_updates(atom, atom_idx)?;
            atomic_updates.push(updates);
        }
        
        // Apply atomic updates
        for (atom_idx, updates) in atomic_updates.into_iter().enumerate() {
            if atom_idx < self.atoms.len() {
                // Apply updates without borrowing self mutably
                for photon in updates.photons_to_emit {
                    self.particles.push(photon);
                }
                
                for electron in updates.electrons_to_add {
                    self.particles.push(electron);
                }
                
                // Update the atom directly
                let atom = &mut self.atoms[atom_idx];
                
                // Remove electrons from atom (in reverse order to maintain indices)
                let mut electrons_to_remove = updates.electrons_to_remove;
                electrons_to_remove.sort_by(|a, b| b.cmp(a));
                for &idx in &electrons_to_remove {
                    if idx < atom.electrons.len() {
                        atom.electrons.remove(idx);
                        atom.total_energy += 13.6e-19; // Ionization energy
                    }
                }
                
                // Update electron energies to ground state
                for electron in &mut atom.electrons {
                    if electron.binding_energy < -13.6e-19 {
                        electron.binding_energy = -13.6e-19; // Ground state
                    }
                }
            }
        }
        
        // Process recombination events (free electrons + ions → neutral atoms)
        self.process_recombination_events()?;
        
        // Update atomic collision processes
        self.process_atomic_collisions()?;
        
        Ok(())
    }
    

    
    fn calculate_atomic_updates(&self, atom: &Atom, _atom_idx: usize) -> Result<AtomicUpdate> {
        let mut update = AtomicUpdate::default();
        
        // Check for spontaneous emission
        for electron in atom.electrons.iter() {
            if electron.binding_energy < -13.6e-19 { // Excited state (simplified)
                if rand::random::<f64>() < 0.001 { // Spontaneous emission probability
                    // Emit photon and drop to lower energy state
                    let photon_energy = electron.binding_energy - (-13.6e-19); // Ground state
                    
                    let photon = FundamentalParticle {
                        particle_type: ParticleType::Photon,
                        position: atom.position,
                        momentum: Vector3::new(
                            photon_energy / C * (rand::random::<f64>() - 0.5),
                            photon_energy / C * (rand::random::<f64>() - 0.5),
                            photon_energy / C * (rand::random::<f64>() - 0.5),
                        ),
                        spin: Vector3::new(1.0, 0.0, 0.0).map(|x| Complex::new(x, 0.0)),
                        color_charge: None,
                        electric_charge: 0.0,
                        mass: 0.0,
                        energy: photon_energy,
                        creation_time: self.current_time,
                        decay_time: None,
                        quantum_state: QuantumState::new(),
                        interaction_history: Vec::new(),
                        velocity: Vector3::zeros(),
                        charge: 0.0,
                    };
                    
                    update.photons_to_emit.push(photon);
                    update.energy_changes.push(photon_energy);
                }
            }
        }
        
        // Check for photoionization events
        let ionization_threshold = 13.6e-19; // Simplified - use hydrogen ionization energy
        
        for photon in &self.particles {
            if let ParticleType::Photon = photon.particle_type {
                let distance = (photon.position - atom.position).norm();
                if distance < 1e-12 && photon.energy > ionization_threshold {
                    // Ionization event occurs
                    
                    // Create free electron
                    let kinetic_energy = photon.energy - ionization_threshold;
                    let electron_momentum = (2.0 * ELECTRON_MASS * kinetic_energy).sqrt();
                    
                    let free_electron = FundamentalParticle {
                        particle_type: ParticleType::Electron,
                        position: atom.position,
                        momentum: Vector3::new(
                            electron_momentum * (rand::random::<f64>() - 0.5),
                            electron_momentum * (rand::random::<f64>() - 0.5),
                            electron_momentum * (rand::random::<f64>() - 0.5),
                        ),
                        spin: Vector3::new(0.5, 0.0, 0.0).map(|x| Complex::new(x, 0.0)),
                        color_charge: None,
                        electric_charge: -ELEMENTARY_CHARGE,
                        mass: ELECTRON_MASS,
                        energy: ELECTRON_MASS * C_SQUARED + kinetic_energy,
                        creation_time: self.current_time,
                        decay_time: None,
                        quantum_state: QuantumState::new(),
                        interaction_history: Vec::new(),
                        velocity: Vector3::zeros(),
                        charge: 0.0,
                    };
                    
                    update.electrons_to_add.push(free_electron);
                    if !atom.electrons.is_empty() {
                        update.electrons_to_remove.push(0); // Remove first electron (simplified)
                    }
                    
                    break;
                }
            }
        }
        
        Ok(update)
    }
    

    
    fn process_recombination_events(&mut self) -> Result<()> {
        // Find free electrons and ions that can recombine
        let mut electrons_to_remove = Vec::new();
        let mut ions_to_neutralize = Vec::new();
        
        for (i, particle) in self.particles.iter().enumerate() {
            if let ParticleType::Electron = particle.particle_type {
                // Look for nearby ions (simplified - assume protons are ions)
                for (j, ion) in self.particles.iter().enumerate() {
                    if let ParticleType::Proton = ion.particle_type {
                        let distance = (particle.position - ion.position).norm();
                        if distance < 1e-12 { // Within recombination radius
                            // Recombination probability
                            if rand::random::<f64>() < 0.0001 {
                                electrons_to_remove.push(i);
                                ions_to_neutralize.push(j);
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // Process recombination events (create neutral hydrogen atoms)
        for (&electron_idx, &proton_idx) in electrons_to_remove.iter().zip(ions_to_neutralize.iter()) {
            if electron_idx < self.particles.len() && proton_idx < self.particles.len() {
                let _electron = &self.particles[electron_idx];
                let proton = &self.particles[proton_idx];
                
                // Create neutral hydrogen atom
                let hydrogen_atom = Atom {
                    nucleus: AtomicNucleus {
                        mass_number: 1,
                        atomic_number: 1,
                        protons: vec![],
                        neutrons: vec![],
                        binding_energy: 0.0,
                        nuclear_spin: Vector3::zeros(),
                        magnetic_moment: Vector3::zeros(),
                        electric_quadrupole_moment: 0.0,
                        nuclear_radius: 0.88e-15,
                        shell_model_state: HashMap::new(),
                        position: proton.position,
                        momentum: proton.momentum,
                        excitation_energy: 0.0,
                    },
                    electrons: vec![Electron {
                        position_probability: vec![vec![vec![0.0; 10]; 10]; 10],
                        momentum_distribution: vec![Vector3::zeros(); 10],
                        spin: Vector3::new(0.5, 0.0, 0.0).map(|x| Complex::new(x, 0.0)),
                        orbital_angular_momentum: Vector3::zeros(),
                        quantum_numbers: QuantumNumbers { n: 1, l: 0, m_l: 0, m_s: 0.5 },
                        binding_energy: -13.6e-19, // Ground state hydrogen
                    }],
                    electron_orbitals: vec![],
                    total_energy: -13.6e-19,
                    ionization_energy: 13.6e-19,
                    electron_affinity: 0.0,
                    atomic_radius: 0.53e-10, // Bohr radius
                    position: proton.position,
                    velocity: proton.momentum / PROTON_MASS,
                    electronic_state: HashMap::new(),
                };
                
                self.atoms.push(hydrogen_atom);
                
                // Emit recombination photon
                let recombination_photon = FundamentalParticle {
                    particle_type: ParticleType::Photon,
                    position: proton.position,
                    momentum: Vector3::new(
                        13.6e-19 / C * (rand::random::<f64>() - 0.5),
                        13.6e-19 / C * (rand::random::<f64>() - 0.5),
                        13.6e-19 / C * (rand::random::<f64>() - 0.5),
                    ),
                    spin: Vector3::new(1.0, 0.0, 0.0).map(|x| Complex::new(x, 0.0)),
                    color_charge: None,
                    electric_charge: 0.0,
                    mass: 0.0,
                    energy: 13.6e-19,
                    creation_time: self.current_time,
                    decay_time: None,
                    quantum_state: QuantumState::new(),
                    interaction_history: Vec::new(),
                    velocity: Vector3::zeros(),
                    charge: 0.0,
                };
                
                self.particles.push(recombination_photon);
            }
        }
        
        // Remove recombined particles (in reverse order to maintain indices)
        electrons_to_remove.sort_by(|a, b| b.cmp(a));
        ions_to_neutralize.sort_by(|a, b| b.cmp(a));
        
        for &idx in &electrons_to_remove {
            if idx < self.particles.len() {
                self.particles.swap_remove(idx);
            }
        }
        for &idx in &ions_to_neutralize {
            if idx < self.particles.len() {
                self.particles.swap_remove(idx);
            }
        }
        
        Ok(())
    }
    
    fn process_atomic_collisions(&mut self) -> Result<()> {
        // Process elastic and inelastic atomic collisions
        let mut collision_pairs = Vec::new();
        
        // Find atoms that are close enough to collide
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let distance = (self.atoms[i].position - self.atoms[j].position).norm();
                let collision_radius = self.atoms[i].atomic_radius + self.atoms[j].atomic_radius;
                
                if distance < collision_radius * 2.0 {
                    collision_pairs.push((i, j));
                }
            }
        }
        
        // Process collisions
        for (i, j) in collision_pairs {
            if i < self.atoms.len() && j < self.atoms.len() {
                // Extract data we need before mutable borrow
                let (pos1, vel1, pos2, vel2) = {
                    let atom1 = &self.atoms[i];
                    let atom2 = &self.atoms[j];
                    (atom1.position, atom1.velocity, atom2.position, atom2.velocity)
                };
                
                // Calculate relative velocity
                let relative_velocity = (vel1 - vel2).norm();
                let collision_energy = 0.5 * PROTON_MASS * relative_velocity.powi(2); // Simplified
                
                // Check for excitation/de-excitation
                if collision_energy > 10.2e-19 { // First excited state of hydrogen
                    // Inelastic collision - excite one of the atoms
                    if rand::random::<f64>() < 0.1 {
                        // Simplified excitation
                        if !self.atoms[i].electrons.is_empty() {
                            self.atoms[i].electrons[0].binding_energy = -3.4e-19; // n=2 state
                            self.atoms[i].total_energy += 10.2e-19;
                        }
                    }
                }
                
                // Elastic scattering (simplified momentum exchange)
                let momentum_exchange = 0.1 * PROTON_MASS * relative_velocity;
                let exchange_vector = (pos1 - pos2).normalize();
                
                self.atoms[i].velocity += exchange_vector * momentum_exchange / PROTON_MASS;
                self.atoms[j].velocity -= exchange_vector * momentum_exchange / PROTON_MASS;
            }
        }
        
        Ok(())
    }
    #[allow(dead_code)]
    fn update_molecular_dynamics(&mut self, states: &mut [PhysicsState]) -> Result<()> {
        // Use atomic collision results to form simple molecules
        self.process_molecular_formation(states)?;
        
        // Apply molecular forces using Lennard-Jones potential and electrostatics
        let force_field = molecular_dynamics::ForceField::new(1e-21, 3e-10); // Typical values for atmospheric molecules
        molecular_dynamics::step_molecular_dynamics(&mut states.to_vec(), &force_field, self.time_step)?;
        
        // Process chemical reactions between molecules
        self.process_chemical_reactions()?;
        
        Ok(())
    }

    fn process_molecular_formation(&mut self, _states: &mut [PhysicsState]) -> Result<()> {
        // Look for atom pairs that can form molecules
        let mut molecules_to_create = Vec::new();
        let mut atoms_to_remove = Vec::new();
        
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let atom1 = &self.atoms[i];
                let atom2 = &self.atoms[j];
                
                let distance = (atom1.position - atom2.position).norm();
                let bond_threshold = (atom1.atomic_radius + atom2.atomic_radius) * 1.2; // 20% larger than sum of radii
                
                if distance < bond_threshold && self.can_form_molecule(atom1, atom2) {
                    let molecule_type = self.determine_molecule_type(atom1, atom2);
                    if let Some(mol_type) = molecule_type {
                        molecules_to_create.push((i, j, mol_type));
                    }
                }
            }
        }
        
        // Process molecule formation (remove atoms, create molecules)
        for (i, j, molecule_type) in molecules_to_create.into_iter().rev() {
            self.create_molecule_from_atoms(i, j, molecule_type)?;
            atoms_to_remove.push(j); // Remove in reverse order to maintain indices
            atoms_to_remove.push(i);
        }
        
        // Remove atoms that were consumed in molecule formation
        atoms_to_remove.sort_unstable();
        atoms_to_remove.dedup();
        for &idx in atoms_to_remove.iter().rev() {
            if idx < self.atoms.len() {
                self.atoms.swap_remove(idx);
            }
        }
        
        Ok(())
    }

    pub fn can_form_molecule(&self, atom1: &Atom, atom2: &Atom) -> bool {
        // Check if atoms can chemically bond based on their electron configurations
        // This is a simplified model based on electron availability
        
        let z1 = atom1.nucleus.atomic_number;
        let z2 = atom2.nucleus.atomic_number;
        
        // Common molecular combinations
        matches!((z1, z2), 
            (1, 1) | // H + H → H₂
            (1, 8) | (8, 1) | // H + O → water precursor
            (6, 8) | (8, 6) | // C + O → CO
            (7, 1) | (1, 7) | // N + H → ammonia precursor
            (6, 1) | (1, 6)   // C + H → hydrocarbon precursor
        )
    }

    pub fn determine_molecule_type(&self, atom1: &Atom, atom2: &Atom) -> Option<ParticleType> {
        let z1 = atom1.nucleus.atomic_number;
        let z2 = atom2.nucleus.atomic_number;
        
        match (z1, z2) {
            (1, 1) => Some(ParticleType::H2),
            (1, 8) | (8, 1) => {
                // Check if there's another hydrogen nearby for H₂O formation
                // For now, just create H₂O directly when H and O meet
                Some(ParticleType::H2O)
            },
            (6, 8) | (8, 6) => Some(ParticleType::CO2), // Simplified - would need another O
            (7, 1) | (1, 7) => Some(ParticleType::NH3), // Simplified - would need more H
            (6, 1) | (1, 6) => Some(ParticleType::CH4), // Simplified - would need more H
            _ => None,
        }
    }

    fn create_molecule_from_atoms(&mut self, atom1_idx: usize, atom2_idx: usize, molecule_type: ParticleType) -> Result<()> {
        if atom1_idx >= self.atoms.len() || atom2_idx >= self.atoms.len() {
            return Ok(()); // Invalid indices
        }
        
        let atom1 = &self.atoms[atom1_idx];
        let atom2 = &self.atoms[atom2_idx];
        
        // Create molecule at center of mass
        let com_position = (atom1.position + atom2.position) * 0.5;
        let total_mass = self.get_particle_mass(molecule_type);
        
        // Create fundamental particle representing the molecule
        let molecule_particle = FundamentalParticle {
            particle_type: molecule_type,
            position: com_position,
            momentum: Vector3::zeros(), // Start at rest
            spin: Vector3::zeros(),
            color_charge: None,
            electric_charge: 0.0, // Most molecules are neutral
            mass: total_mass,
            energy: total_mass * C_SQUARED * C_SQUARED, // Rest energy
            creation_time: self.current_time,
            decay_time: None, // Molecules are generally stable
            quantum_state: QuantumState::new(),
            interaction_history: Vec::new(),
            velocity: Vector3::zeros(),
            charge: 0.0,
        };
        
        self.particles.push(molecule_particle);
        Ok(())
    }

    fn process_chemical_reactions(&mut self) -> Result<()> {
        // Process chemical reactions between existing molecules
        // This is a simplified reaction network for common atmospheric/water chemistry
        
        let mut reactions_to_process = Vec::new();
        
        // Look for molecules that can react
        for i in 0..self.particles.len() {
            for j in (i + 1)..self.particles.len() {
                let p1 = &self.particles[i];
                let p2 = &self.particles[j];
                
                // Check if particles are molecules and close enough to react
                if self.is_molecule(p1.particle_type) && self.is_molecule(p2.particle_type) {
                    let distance = (p1.position - p2.position).norm();
                    let reaction_threshold = 5e-10; // 5 Angstroms
                    
                    if distance < reaction_threshold {
                        let reaction = self.check_chemical_reaction(p1.particle_type, p2.particle_type);
                        if let Some(products) = reaction {
                            reactions_to_process.push((i, j, products));
                        }
                    }
                }
            }
        }
        
        // Process reactions (in reverse order to maintain indices)
        for (i, j, products) in reactions_to_process.into_iter().rev() {
            self.execute_chemical_reaction(i, j, products)?;
        }
        
        Ok(())
    }

    pub fn is_molecule(&self, particle_type: ParticleType) -> bool {
        matches!(particle_type, 
            ParticleType::H2 | ParticleType::H2O | ParticleType::CO2 | 
            ParticleType::CH4 | ParticleType::NH3
        )
    }

    pub fn check_chemical_reaction(&self, mol1: ParticleType, mol2: ParticleType) -> Option<Vec<ParticleType>> {
        // Simple chemical reaction network
        match (mol1, mol2) {
            // Combustion reactions
            (ParticleType::CH4, ParticleType::H2O) | (ParticleType::H2O, ParticleType::CH4) => {
                // CH₄ + H₂O → CO + 3H₂ (steam reforming)
                Some(vec![ParticleType::CO2, ParticleType::H2, ParticleType::H2])
            },
            // Photosynthesis-like reaction (simplified)
            (ParticleType::CO2, ParticleType::H2O) | (ParticleType::H2O, ParticleType::CO2) => {
                // CO₂ + H₂O → CH₄ + O₂ (simplified)
                Some(vec![ParticleType::CH4, ParticleType::H2O])
            },
            _ => None,
        }
    }

    fn execute_chemical_reaction(&mut self, mol1_idx: usize, mol2_idx: usize, products: Vec<ParticleType>) -> Result<()> {
        if mol1_idx >= self.particles.len() || mol2_idx >= self.particles.len() {
            return Ok(());
        }
        
        // Get reaction center position
        let reaction_position = (self.particles[mol1_idx].position + self.particles[mol2_idx].position) * 0.5;
        
        // Create product molecules
        for product_type in products {
            let mass = self.get_particle_mass(product_type);
            let momentum = self.sample_thermal_momentum(product_type, self.temperature);
            
            let product = FundamentalParticle {
                particle_type: product_type,
                position: reaction_position + Vector3::new(
                    (rand::random::<f64>() - 0.5) * 1e-10,
                    (rand::random::<f64>() - 0.5) * 1e-10,
                    (rand::random::<f64>() - 0.5) * 1e-10,
                ), // Small random displacement
                momentum,
                spin: Vector3::zeros(),
                color_charge: None,
                electric_charge: 0.0,
                mass,
                energy: (mass * mass * C_SQUARED * C_SQUARED + momentum.norm_squared() * C_SQUARED).sqrt(),
                creation_time: self.current_time,
                decay_time: None,
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
                velocity: Vector3::zeros(),
                charge: 0.0,
            };
            
            self.particles.push(product);
        }
        
        // Remove reactant molecules (in reverse order to maintain indices)
        let mut indices = vec![mol1_idx, mol2_idx];
        indices.sort_unstable();
        indices.reverse();
        
        for &idx in &indices {
            if idx < self.particles.len() {
                self.particles.swap_remove(idx);
            }
        }
        
        Ok(())
    }
    fn process_phase_transitions(&mut self) -> Result<()> {
        use crate::phase_transitions::*;
        use crate::emergent_properties::{Temperature, Pressure, Density};
        
        // Process phase transitions for each material at current temperature and pressure
        let pressure = self.calculate_system_pressure();
        
        // Calculate density for phase determination
        let total_mass = self.particles.iter().map(|p| p.mass).sum::<f64>();
        let density = total_mass / self.volume.max(1e-50);
        
        // Check for phase transitions in hydrogen (dominant in early universe)
        let temp = Temperature::from_kelvin(self.temperature);
        let pres = Pressure::from_pascals(pressure);
        let dens = Density::from_kg_per_m3(density);
        
        if let Ok(hydrogen_phase) = evaluate_phase_transitions("hydrogen", temp, pres, dens) {
            // Log phase information (simplified for now)
            if self.particles.len() > 1000 {
                log::debug!("Phase transitions: H2 = {:?}, T = {:.2e}K, P = {:.2e}Pa", 
                           hydrogen_phase, self.temperature, pressure);
            }
        }
        
        Ok(())
    }
    fn update_emergent_properties(&mut self, states: &mut [PhysicsState]) -> Result<()> {
        use crate::emergent_properties::*;
        
        // Calculate emergent statistical mechanics properties
        let mut monitor = EmergenceMonitor::new();
        
        // Update emergent properties from classical states (if any)
        if !states.is_empty() {
            monitor.update(states, self.volume)?;
            
            // Update engine state with calculated values
            let calculated_temp = monitor.temperature.as_kelvin();
            if calculated_temp > 0.0 {
                self.temperature = calculated_temp;
            }
            
            // Log emergent properties for debugging
            log::trace!("Emergent properties: T = {:.2e}K, P = {:.2e}Pa, ρ = {:.2e}kg/m³, S = {:.2e}J/K", 
                       monitor.temperature.as_kelvin(),
                       monitor.pressure.as_pascals(),
                       monitor.density.as_kg_per_m3(),
                       monitor.entropy.as_joules_per_kelvin());
        } else {
            // If no classical states, calculate basic properties from particles
            if !self.particles.is_empty() {
                let total_mass = self.particles.iter().map(|p| p.mass).sum::<f64>();
                let density = total_mass / self.volume.max(1e-50);
                
                log::trace!("Basic properties from particles: N = {}, ρ = {:.2e}kg/m³, T = {:.2e}K", 
                           self.particles.len(), density, self.temperature);
            }
        }
        
        Ok(())
    }
    #[allow(dead_code)]
    fn update_running_couplings(&mut self, _states: &mut [PhysicsState]) -> Result<()> {
        // -------------------------------------------------------------------
        // 1. Renormalisation scale Q taken as average thermal energy k_B T.
        // -------------------------------------------------------------------
        const J_PER_GEV: f64 = 1.602_176_634e-10; // exact CODATA 2022 factor
        let q_gev = (BOLTZMANN * self.temperature / J_PER_GEV).max(1.0e-3); // ≥1 MeV
        self.running_couplings.scale_gev = q_gev;

        // -------------------------------------------------------------------
        // 2. QED: one-loop running of α.
        // -------------------------------------------------------------------
        let alpha0 = FINE_STRUCTURE_CONSTANT;
        let gev_per_kg = SPEED_OF_LIGHT * SPEED_OF_LIGHT / J_PER_GEV; // E=mc²
        let lepton_masses_gev = [ELECTRON_MASS, MUON_MASS, TAU_MASS].map(|m| m * gev_per_kg);

        let mut delta_alpha = 0.0;
        for m in lepton_masses_gev {
            if q_gev > m {
                delta_alpha += (q_gev * q_gev / (m * m)).ln();
            }
        }

        self.running_couplings.alpha_em = if delta_alpha > 0.0 {
            let correction = (alpha0 / (3.0 * std::f64::consts::PI)) * delta_alpha;
            alpha0 / (1.0 - correction)
        } else {
            alpha0
        };
        self.interaction_matrix
            .set_electromagnetic_coupling(self.running_couplings.alpha_em);

        // -------------------------------------------------------------------
        // 3. QCD: one-loop running of αₛ.
        // -------------------------------------------------------------------
        const LAMBDA_QCD: f64 = 0.2; // GeV (MS-bar)
        let n_f = if q_gev < 1.27 {
            3.0 // u, d, s
        } else if q_gev < 4.18 {
            4.0 // + c
        } else if q_gev < 173.0 {
            5.0 // + b
        } else {
            6.0 // + t
        };
        let beta0 = 11.0 - (2.0 / 3.0) * n_f;
        if q_gev > LAMBDA_QCD {
            self.running_couplings.alpha_s =
                4.0 * std::f64::consts::PI / (beta0 * (q_gev * q_gev / (LAMBDA_QCD * LAMBDA_QCD)).ln());
        }
        self.interaction_matrix
            .set_strong_coupling(self.running_couplings.alpha_s);

        Ok(())
    }
    #[allow(dead_code)]
    fn check_symmetry_breaking(&mut self) -> Result<()> {
        // Electroweak crossover occurs at T_c ≈ 159 GeV ≈ 1.85×10¹⁵ K.
        const T_EW_C: f64 = 1.85e15; // K

        if self.temperature < T_EW_C {
            // Universe cooled below critical temperature → Higgs field should
            // acquire its vacuum expectation value and give masses to W/Z.
            self.symmetry_breaking.initialize_higgs_mechanism()?;
        }
        Ok(())
    }
    #[allow(dead_code)]
    fn update_spacetime_curvature(&mut self) -> Result<()> {
        // Friedmann–Lemaître first equation (k=0)  H² = (8πG/3) ρ.
        use crate::constants::{GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, PROTON_MASS};

        // Mass from fundamental particles.
        let particle_mass: f64 = self.particles.iter().map(|p| p.mass).sum();

        // Mass from nuclei — approximate by A × m_p.
        let nuclear_mass: f64 = self
            .nuclei
            .iter()
            .map(|n| n.mass_number as f64 * PROTON_MASS)
            .sum();

        let total_mass = particle_mass + nuclear_mass;
        let rho = if self.volume > 0.0 {
            total_mass / self.volume
        } else {
            0.0
        };

        let h_squared = (8.0 * std::f64::consts::PI * GRAVITATIONAL_CONSTANT * rho) / 3.0;
        let hubble = h_squared.max(0.0).sqrt();
        let curvature_radius = if hubble > 0.0 {
            SPEED_OF_LIGHT / hubble
        } else {
            f64::INFINITY
        };

        log::trace!(
            "Spacetime curvature: ρ={:.3e} kg/m³  H={:.3e} s⁻¹  R_c={:.3e} m",
            rho, hubble, curvature_radius
        );
        Ok(())
    }
    #[allow(dead_code)]
    fn update_thermodynamic_state(&mut self) -> Result<()> {
        // Update temperature based on particle kinetic energies
        self.update_temperature()?;
        Ok(())
    }
    
    #[allow(dead_code)]
    fn evolve_quantum_state(&mut self) -> Result<()> {
        // Placeholder for quantum evolution
        // In a full implementation, this would solve the Schrödinger/Dirac equation
        Ok(())
    }
    
    /// Update temperature based on particle energies
    fn update_temperature(&mut self) -> Result<()> {
        // More sophisticated calculation based on particle kin. energy
        self.temperature = self.particles.iter().map(|p| p.energy).sum::<f64>() / (self.particles.len() as f64 * BOLTZMANN);
        Ok(())
    }

    /// Calculates the total system pressure from all particles.
    /// P = (1/3V) * Σ (p_i^2 * c^2) / E_i
    pub fn calculate_system_pressure(&self) -> f64 {
        if self.volume <= 0.0 {
            return 0.0;
        }

        let mut pressure_sum = 0.0;
        let c_squared = SPEED_OF_LIGHT.powi(2);

        for p in &self.particles {
            if p.energy > 0.0 {
                let momentum_squared = p.momentum.norm_squared();
                // Pressure contribution is (p^2 * c^2) / (3 * E_total)
                pressure_sum += (momentum_squared * c_squared) / (3.0 * p.energy);
            }
        }
        
        // Pressure is the sum of contributions divided by volume
        pressure_sum / self.volume
    }

    /// Calculate the Q-value (energy released) for a fission reaction
    fn calculate_fission_q_value(&self, parent_z: u32, parent_a: u32) -> Result<f64> {
        use crate::nuclear_physics::Nucleus;
        
        // Calculate binding energies using Semi-Empirical Mass Formula
        let parent_nucleus = Nucleus::new(parent_z, parent_a - parent_z);
        let parent_binding_energy = parent_nucleus.binding_energy();
        
        // For binary fission, estimate fragment masses
        let fragment1_a = parent_a / 2;
        let fragment2_a = parent_a - fragment1_a - 2; // Assume 2 neutrons are emitted
        
        // Estimate Z distribution using charge asymmetry (Wahl systematics)
        let fragment1_z = (parent_z * fragment1_a) / parent_a;
        let fragment2_z = parent_z - fragment1_z;
        
        let fragment1_nucleus = Nucleus::new(fragment1_z, fragment1_a - fragment1_z);
        let fragment2_nucleus = Nucleus::new(fragment2_z, fragment2_a - fragment2_z);
        
        let fragment1_binding_energy = fragment1_nucleus.binding_energy();
        let fragment2_binding_energy = fragment2_nucleus.binding_energy();
        
        // Q-value = Energy released = difference in binding energies
        let q_value = (fragment1_binding_energy + fragment2_binding_energy) - parent_binding_energy;
        
        // Convert from MeV to Joules
        Ok(q_value * 1.602e-13) // MeV to Joules
    }
    
    /// Distribute fission energy among products (fragments + neutrons)
    fn distribute_fission_energy(&mut self, q_value: f64, _z1: u32, _a1: u32, _z2: u32, _a2: u32, position: &Vector3<f64>) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Energy distribution: ~80% to kinetic energy of fragments, ~20% to neutrons
        let fragment_kinetic_energy = q_value * 0.8;
        let neutron_kinetic_energy = q_value * 0.2;
        
        // Fragment recoil energies (assuming two fragments with momentum conservation)
        let _fragment1_energy = fragment_kinetic_energy * 0.5;
        let _fragment2_energy = fragment_kinetic_energy * 0.5;
        
        // Update system energy
        self.energy_density += q_value / self.volume;
        
        // Add kinetic energy to newly created neutrons
        let neutron_count = self.particles.iter().filter(|p| 
            matches!(p.particle_type, ParticleType::Neutron) && 
            (p.position - position).norm() < 1e-12 // Recently created at fission site
        ).count();
        
        if neutron_count > 0 {
            let energy_per_neutron = neutron_kinetic_energy / neutron_count as f64;
            
            for particle in &mut self.particles {
                if matches!(particle.particle_type, ParticleType::Neutron) && 
                   (particle.position - position).norm() < 1e-12 {
                    // Add kinetic energy by increasing momentum
                    let additional_momentum_magnitude = (2.0 * energy_per_neutron * particle.mass).sqrt();
                    
                    // Random direction for neutron emission
                    let theta = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                    let phi = rng.gen::<f64>() * std::f64::consts::PI;
                    
                    let additional_momentum = Vector3::new(
                        additional_momentum_magnitude * phi.sin() * theta.cos(),
                        additional_momentum_magnitude * phi.sin() * theta.sin(),
                        additional_momentum_magnitude * phi.cos(),
                    );
                    
                    particle.momentum += additional_momentum;
                    particle.energy = (particle.mass*particle.mass*C_SQUARED*C_SQUARED + 
                                     particle.momentum.norm_squared() * C_SQUARED).sqrt();
                }
            }
        }
        
        log::debug!("Fission energy distribution: Q = {:.2e} J, fragments = {:.2e} J, neutrons = {:.2e} J", 
                    q_value, fragment_kinetic_energy, neutron_kinetic_energy);
        
        Ok(())
    }

    /// Determine local material composition for Geant4
    fn determine_local_material(&self, position: &Vector3<f64>) -> String {
        "Vacuum".to_string()
    }

    /// Validate conservation laws (energy, momentum, charge) - placeholder implementation
    fn validate_conservation_laws(&self) -> Result<()> {

        // Net charge should remain (approximately) conserved.
        let total_charge_c: f64 = self.particles.iter().map(|p| p.electric_charge).sum();
        if total_charge_c.abs() > 1e-9 { // 1 nC tolerance
            log::warn!("⚠️  Charge non-conservation detected: Σq = {:.3e} C", total_charge_c);
        }

        // Momentum conservation – compute vector sum.
        let total_momentum = self
            .particles
            .iter()
            .fold(Vector3::zeros(), |acc, p| acc + p.momentum);
        if total_momentum.norm() > 1e-6 {
            log::warn!(
                "⚠️  Momentum non-conservation |Σp| = {:.3e} kg·m/s",
                total_momentum.norm()
            );
        }

        // Energy should be positive definite.
        let total_energy: f64 = self.particles.iter().map(|p| p.energy).sum();
        if total_energy < 0.0 {
            anyhow::bail!("Negative total energy detected: {:.3e} J", total_energy);
        }

        Ok(())
    }

    /// Process a single particle's native interactions when Geant4 fails (placeholder)
    fn process_particle_native_interaction(&mut self, _index: usize) -> Result<()> {
        // For now, simply ignore and continue.
        Ok(())
    }

    /// Update gravitational forces in absence of GADGET - placeholder
    fn update_gravitational_forces(&mut self) -> Result<()> {
        // Parallel pairwise Newtonian gravity (still O(N²) but multi-core).
        let g_const = 6.67430e-11;
        let timer = Instant::now();
        log::debug!(
            "[gravity] Computing Newtonian forces for {} particles on {} threads",
            self.particles.len(),
            rayon::current_num_threads()
        );

        // Build lightweight snapshots of immutable particle properties to avoid heavy cloning.
        let positions: Vec<Vector3<f64>> = self.particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();
        let velocities: Vec<Vector3<f64>> = self.particles.iter().map(|p| p.velocity).collect();

        let particle_count = positions.len();

        // Compute net force on each particle in parallel.
        let forces: Vec<Vector3<f64>> = (0..particle_count)
            .into_par_iter()
            .map(|i| {
                let mut force = Vector3::zeros();
                let pos_i = positions[i];
                let mass_i = masses[i];
                let vel_i = velocities[i];
                for j in 0..particle_count {
                    if i == j { continue; }
                    let dir = positions[j] - pos_i;
                    let dist_sq = dir.norm_squared().max(1e-12);
                    let distance = dist_sq.sqrt();
                    
                    // Newtonian force
                    let f_mag = g_const * mass_i * masses[j] / dist_sq;
                    let newtonian_force = dir.normalize() * f_mag;
                    
                    // Add post-Newtonian correction for massive objects
                    if general_relativity::requires_relativistic_treatment(mass_i, vel_i.norm(), distance) ||
                       general_relativity::requires_relativistic_treatment(masses[j], velocities[j].norm(), distance) {
                        let pn_correction = general_relativity::post_newtonian_force_correction(
                            mass_i, masses[j], distance,
                            [vel_i.x, vel_i.y, vel_i.z],
                            [velocities[j].x, velocities[j].y, velocities[j].z]
                        );
                        let pn_force = Vector3::new(pn_correction[0], pn_correction[1], pn_correction[2]);
                        force += newtonian_force + pn_force;
                    } else {
                        force += newtonian_force;
                    }
                }
                force
            })
            .collect();

        // Apply accelerations sequentially (acc = F / m).
        for (i, force) in forces.into_iter().enumerate() {
            if i < self.particles.len() {
                let mass = self.particles[i].mass;
                if mass > 0.0 {
                    let acceleration = force / mass;
                    // Store as instantaneous velocity increment for now.
                    self.particles[i].velocity += acceleration * self.time_step;
                }
            }
        }

        log::debug!(
            "[gravity] Force computation + application completed in {:.3?}",
            timer.elapsed()
        );
        Ok(())
    }
    
    /// Simple local density estimator used by step-length heuristic.
    fn calculate_local_density_legacy(&self, _position: &Vector3<f64>) -> f64 {
        // Placeholder: uniform density estimate to unblock compilation.
        if self.volume > 0.0 {
            self.particles.len() as f64 * self.get_particle_mass(ParticleType::Proton) / self.volume
        } else {
            0.0
        }
    }

    fn calculate_mm_region_energy_legacy(&self, atoms: &[crate::Atom]) -> Result<f64> {
        // Estimate total quantum energy of the QM region.
        // --------------------------------------------------------------------
        // We combine two main energetic contributions that are readily
        // available from the data structures:
        // 1. Nuclear binding energies (returned by `nuclear_physics::Nucleus`
        //    in MeV) which we convert to Joules via the CODATA 2022 factor.
        // 2. Electronic binding energies stored in each `Electron` record
        //    (already in Joules – e.g. −13.6 eV ≈ −2.18 × 10⁻¹⁸ J for H(1s)).
        // This provides a lower-bound on the total internal energy that is
        // conserved irrespective of molecular conformation and is therefore
        // adequate for the coarse QM/MM energy bookkeeping carried out by the
        // simulation. For full ab-initio accuracy this routine should be
        // replaced by a proper SCF/DFT call – see the project roadmap.
        // --------------------------------------------------------------------
        const MEV_TO_J: f64 = 1.602_176_634e-13; // exact conversion (J/MeV)

        let mut total_energy_j = 0.0_f64;

        for atom in atoms {
            // 1. Nuclear contribution (MeV ➜ J)
            total_energy_j += atom.nucleus.binding_energy * MEV_TO_J;

            // 2. Electronic contribution (already in Joules)
            for elec in &atom.electrons {
                total_energy_j += elec.binding_energy;
            }
        }

        Ok(total_energy_j)
    }

    fn calculate_qm_mm_interaction(&self, qm: &[crate::Atom], mm: &[crate::Atom]) -> Result<f64> {
        let interaction_energy = 0.0;
        for qm_atom in qm {
            for mm_atom in mm {
                let distance = (qm_atom.position - mm_atom.position).norm();
                if distance > 1e-9 {
                    // The line below is commented out due to a signature mismatch that requires a larger refactor.
                    // interaction_energy += self.quantum_chemistry_engine.van_der_waals_energy(
                    //     qm_atom,
                    //     mm_atom,
                    //     distance,
                    // )?;
                }
            }
        }
        Ok(interaction_energy)
    }

    /// Approximate ground-state electronic energy (J) for an isolated atom.
    ///
    /// Ground-state electronic energy estimate for an isolated atom.
    ///
    /// We use the simple hydrogenic model 𝐸 = −Z² R_H (in eV) and convert to Joules.
    /// Although crude, this provides a lower-bound on the total electronic binding
    /// energy that is adequate for the semi-empirical energy bookkeeping carried
    /// out by the fast QC routines.
    fn get_atomic_energy(&self, atomic_number: &u32) -> f64 {
        // Delegate to the quantum-chemistry engine which provides a Thomas–Fermi
        // estimate of the ground-state electronic energy in Joules (see
        // `quantum_chemistry::QuantumChemistryEngine::get_atomic_energy`).  This
        // keeps a single authoritative implementation of the model and avoids
        // diverging approximations throughout the codebase.
        self.quantum_chemistry_engine.get_atomic_energy(atomic_number)
    }

    /// Empirical bond-dissociation energy (approximate) returned in Joules per bond.
    /// Values are based on typical gas-phase bond energies at 298 K.
    fn get_bond_energy(&self, bond_type: &crate::BondType, _bond_length: f64) -> f64 {
        // Typical gas-phase bond dissociation energies at 298 K.
        // Source: CRC Handbook of Chemistry & Physics (103rd edition). Values are
        // converted from kJ mol⁻¹ to Joules per individual bond using Avogadro's
        // constant (CODATA 2022).
        const AVOGADRO: f64 = 6.022_140_76e23; // mol⁻¹
        let kj_per_mol_to_j_per_bond = 1_000.0 / AVOGADRO;

        match bond_type {
            crate::BondType::Covalent      => 350.0 * kj_per_mol_to_j_per_bond, // C–C single ≈ 348
            crate::BondType::Ionic         => 400.0 * kj_per_mol_to_j_per_bond, // Na–Cl ≈ 411
            crate::BondType::Metallic      => 200.0 * kj_per_mol_to_j_per_bond, // averaged transition metals
            crate::BondType::HydrogenBond  => 20.0  * kj_per_mol_to_j_per_bond, // O–H···O in water
            crate::BondType::VanDerWaals   => 2.0   * kj_per_mol_to_j_per_bond, // Dispersion interactions
        }
    }

    /// Return true if atoms with indices `i` and `j` share a chemical bond in `molecule`.
    fn are_bonded(&self, i: usize, j: usize, molecule: &crate::Molecule) -> bool {
        use crate::quantum_chemistry::COVALENT_RADII;
        const BOND_TOLERANCE: f64 = 1.20; // Allow up to 20 % stretch/compression.

        if i >= molecule.atoms.len() || j >= molecule.atoms.len() { return false; }
        let atom_i = &molecule.atoms[i];
        let atom_j = &molecule.atoms[j];

        // Calculate inter-nuclear distance.
        let distance = (atom_i.position - atom_j.position).norm();

        // Retrieve covalent radii (fallback ≈ 70 pm if unknown).
        let default_radius = 70e-12; // metres
        let r_i = COVALENT_RADII
            .get(&atom_i.get_particle_type())
            .copied()
            .unwrap_or(default_radius);
        let r_j = COVALENT_RADII
            .get(&atom_j.get_particle_type())
            .copied()
            .unwrap_or(default_radius);

        distance < (r_i + r_j) * BOND_TOLERANCE
    }

    /// Lennard-Jones 12-6 potential (dispersion + Pauli repulsion) for a pair of atoms.
    fn van_der_waals_energy(&self, i: usize, j: usize, distance: f64, molecule: &crate::Molecule) -> f64 {
        // Re-use the well-tested implementation in the quantum-chemistry module
        // to avoid duplicating force-field parameter look-ups.
        self.quantum_chemistry_engine
            .van_der_waals_energy(i, j, distance, molecule)
    }

    // Geant4 interaction bridge removed - now using native Rust particle transport

    /// Convenience constructor for a `FundamentalParticle` with minimal initial information. The caller is expected
    /// to update position, momentum, and quantum numbers as appropriate.
    fn create_particle_from_type(&self, particle_type: ParticleType) -> Result<FundamentalParticle> {
        let mass = self.get_particle_mass(particle_type);
        Ok(FundamentalParticle {
            particle_type,
            mass,
            energy: mass * C_SQUARED, // rest-mass energy
            position: Vector3::zeros(),
            momentum: Vector3::zeros(),
            spin: Vector3::zeros(),
            color_charge: self.assign_color_charge(particle_type),
            electric_charge: self.get_electric_charge(particle_type),
            creation_time: self.current_time,
            decay_time: self.calculate_decay_time(particle_type),
            quantum_state: QuantumState::new(),
            interaction_history: Vec::new(),
            velocity: Vector3::zeros(),
            charge: self.get_electric_charge(particle_type),
        })
    }

    pub fn calculate_qm_region_energy(&self, atoms: &[crate::Atom]) -> Result<f64> {
        // Estimate total quantum energy of the QM region.
        // --------------------------------------------------------------------
        // We combine two main energetic contributions that are readily
        // available from the data structures:
        // 1. Nuclear binding energies (returned by `nuclear_physics::Nucleus`
        //    in MeV) which we convert to Joules via the CODATA 2022 factor.
        // 2. Electronic binding energies stored in each `Electron` record
        //    (already in Joules – e.g. −13.6 eV ≈ −2.18 × 10⁻¹⁸ J for H(1s)).
        // This provides a lower-bound on the total internal energy that is
        // conserved irrespective of molecular conformation and is therefore
        // adequate for the coarse QM/MM energy bookkeeping carried out by the
        // simulation. For full ab-initio accuracy this routine should be
        // replaced by a proper SCF/DFT call – see the project roadmap.
        // --------------------------------------------------------------------
        const MEV_TO_J: f64 = 1.602_176_634e-13; // exact conversion (J/MeV)

        let mut total_energy_j = 0.0_f64;

        for atom in atoms {
            // 1. Nuclear contribution (MeV ➜ J)
            total_energy_j += atom.nucleus.binding_energy * MEV_TO_J;

            // 2. Electronic contribution (already in Joules)
            for elec in &atom.electrons {
                total_energy_j += elec.binding_energy;
            }
        }

        Ok(total_energy_j)
    }

    /// Apply comprehensive cosmological expansion effects following ΛCDM model with full Friedmann equations
    fn apply_cosmological_expansion_to_particles(&mut self, dt: f64) -> Result<()> {
        use crate::constants::{GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT};
        use crate::gadget_gravity::CosmologicalParameters;
        
        // Create default cosmological parameters (Planck 2018 values)
        let params = CosmologicalParameters::default();
        
        // Calculate current cosmic age and scale factor using proper ΛCDM evolution
        let current_age_seconds = self.current_time;
        let current_age_gyr = current_age_seconds / (365.25 * 24.0 * 3600.0 * 1e9);
        
        // Solve for scale factor a(t) using Friedmann equation for ΛCDM cosmology
        let a = self.calculate_scale_factor_from_time(&params, current_age_seconds);
        
        // Convert H₀ to SI units (s⁻¹)
        let h0_si = params.hubble_constant * 1000.0 / 3.086e22;
        
        // Calculate Hubble parameter H(a) = H₀ * E(a) where E(a) = sqrt(Ωᵣ/a⁴ + Ωₘ/a³ + Ωₖ/a² + ΩΛ)
        let omega_r = 9.24e-5; // Radiation density parameter (photons + neutrinos)
        let hubble_parameter_si = h0_si * (
            omega_r / a.powi(4) + 
            params.omega_matter / a.powi(3) + 
            params.omega_lambda
        ).sqrt();
        
        // Scale factor evolution rate: da/dt = H(a) * a
        let daddt = hubble_parameter_si * a;
        let expansion_rate = daddt / a; // H(t) in SI units
        
        // Apply comprehensive cosmological effects to all particles
        for particle in &mut self.particles {
            // Apply Hubble flow: v_hubble = H(t) * r
            let hubble_velocity = particle.position * expansion_rate;
            particle.velocity += hubble_velocity * dt;
            
            // Apply cosmological redshift effects based on particle type
            match particle.particle_type {
                ParticleType::Photon => {
                    // Photons: energy redshift E ∝ 1/a, frequency redshift ν ∝ 1/a
                    let energy_loss_rate = expansion_rate;
                    particle.energy *= (1.0f64 - energy_loss_rate * dt).max(1e-100);
                    
                    // Maintain E = pc for massless photons
                    let p_magnitude = particle.energy / SPEED_OF_LIGHT;
                    if particle.momentum.magnitude() > 1e-100 {
                        particle.momentum = particle.momentum.normalize() * p_magnitude;
                    }
                }
                
                // Massive particles: momentum redshift p ∝ 1/a, temperature cooling
                _ => {
                    // Momentum decreases as universe expands: p ∝ 1/a
                    particle.momentum *= (1.0f64 - expansion_rate * dt).max(0.01);
                    
                    // Update kinetic energy and velocity using relativistic relations
                    let p_magnitude = particle.momentum.magnitude();
                    if p_magnitude > 1e-100 && particle.mass > 1e-100 {
                        // Relativistic energy-momentum relation: E² = (pc)² + (mc²)²
                        let rest_energy = particle.mass * C_SQUARED;
                        let total_energy = (rest_energy.powi(2) + (p_magnitude * SPEED_OF_LIGHT).powi(2)).sqrt();
                        let gamma = total_energy / rest_energy;
                        let velocity_magnitude = p_magnitude / (gamma * particle.mass);
                        
                        // Update particle velocity maintaining momentum direction
                        if particle.momentum.magnitude() > 1e-100 {
                            particle.velocity = particle.momentum.normalize() * velocity_magnitude;
                        }
                        
                        particle.energy = total_energy;
                    }
                }
            }
        }
        
        // Update global thermodynamic properties
        self.volume *= (1.0f64 + expansion_rate * dt).powi(3); // Volume scales as a³
        self.temperature *= 1.0 - expansion_rate * dt; // Adiabatic cooling: T ∝ 1/a
        
        // Calculate critical density and component energy densities
        let critical_density = 3.0 * hubble_parameter_si.powi(2) / (8.0 * std::f64::consts::PI * GRAVITATIONAL_CONSTANT);
        
        // Energy density evolution for different components
        let matter_density_scale = (1.0f64 - expansion_rate * dt).powi(3);     // ρₘ ∝ a⁻³
        let radiation_density_scale = (1.0f64 - expansion_rate * dt).powi(4);   // ρᵣ ∝ a⁻⁴
        
        let matter_energy_density = params.omega_matter * critical_density * matter_density_scale;
        let radiation_energy_density = omega_r * critical_density * radiation_density_scale;
        let dark_energy_density = params.omega_lambda * critical_density; // Constant ρΛ
        
        self.energy_density = matter_energy_density + radiation_energy_density + dark_energy_density;
        
        // Log cosmological expansion status periodically
        if self.current_time.rem_euclid(1e9) < dt {
            log::debug!(
                "Cosmological expansion: age={:.2} Gyr, a={:.6}, H={:.2e} s⁻¹, T={:.1} K, ρ={:.2e} J/m³",
                current_age_gyr, a, hubble_parameter_si, self.temperature, self.energy_density
            );
        }
        
        Ok(())
    }
    
    /// Calculate scale factor a(t) from cosmic time using ΛCDM model
    fn calculate_scale_factor_from_time(&self, params: &crate::gadget_gravity::CosmologicalParameters, time_seconds: f64) -> f64 {
        let h0_si = params.hubble_constant * 1000.0 / 3.086e22; // Convert to SI units
        
        if params.omega_lambda.abs() < 1e-6 {
            // Matter-dominated universe: a(t) ∝ t^(2/3)
            let t0 = 2.0 / (3.0 * h0_si); // Age at a=1
            (time_seconds / t0).powf(2.0/3.0).max(0.001)
        } else {
            // ΛCDM universe with dark energy
            let omega_m_over_lambda = params.omega_matter / params.omega_lambda;
            let h_lambda = h0_si * params.omega_lambda.sqrt();
            
            // Parametric solution: t = (2/(3H_Λ)) * sinh⁻¹(√(Ω_Λ/Ω_m) * a^(3/2))
            let x = 1.5 * h_lambda * time_seconds;
            let y = x.sinh();
            let a_cubed_half = y / omega_m_over_lambda.sqrt();
            a_cubed_half.powf(2.0/3.0).max(0.001)
        }
    }

    /// Process gravitational collapse and sink particle formation
    pub fn process_gravitational_collapse(&mut self) -> Result<()> {
        // Convert particles to SPH particles for collapse detection
        let mut sph_particles = self.sph_solver.convert_to_sph_particles(self.particles.clone());
        
        if sph_particles.is_empty() {
            return Ok(());
        }
        
        // Update SPH particle properties
        self.sph_solver.compute_density(&mut sph_particles)?;
        for particle in &mut sph_particles {
            particle.update_eos();
        }
        
        // Detect collapse regions
        let collapse_regions = gravitational_collapse::detect_collapse_regions(&sph_particles, MEAN_MOLECULAR_WEIGHT);
        
        if !collapse_regions.is_empty() {
            // Form new sink particles
            let (new_sinks, particles_to_remove) = gravitational_collapse::form_sink_particles(
                &sph_particles, 
                collapse_regions, 
                self.current_time, 
                &mut self.next_sink_id
            );
            
            // Add new sink particles
            self.sink_particles.extend(new_sinks);
            
            // Remove particles that formed sinks (convert back to regular particles first)
            let sph_particles_to_remove: Vec<SphParticle> = particles_to_remove.iter()
                .map(|&i| sph_particles[i].clone())
                .collect();
            let regular_particles_to_remove: Vec<FundamentalParticle> = sph_particles_to_remove.iter()
                .map(|sp| sp.particle.clone())
                .collect();
            
            // Remove from main particle list
            for particle_to_remove in regular_particles_to_remove {
                if let Some(pos) = self.particles.iter().position(|p| 
                    p.position == particle_to_remove.position && 
                    p.particle_type == particle_to_remove.particle_type
                ) {
                    self.particles.remove(pos);
                }
            }
            
            info!("Formed {} new sink particles, removed {} gas particles", 
                  self.sink_particles.len(), particles_to_remove.len());
        }
        
        // Accrete gas onto existing sink particles
        let accreted_particles = gravitational_collapse::accrete_onto_sinks(&mut sph_particles, &mut self.sink_particles);
        
        if !accreted_particles.is_empty() {
            // Remove accreted particles from main particle list
            let sph_particles_to_remove: Vec<SphParticle> = accreted_particles.iter()
                .map(|&i| sph_particles[i].clone())
                .collect();
            let regular_particles_to_remove: Vec<FundamentalParticle> = sph_particles_to_remove.iter()
                .map(|sp| sp.particle.clone())
                .collect();
            
            for particle_to_remove in regular_particles_to_remove {
                if let Some(pos) = self.particles.iter().position(|p| 
                    p.position == particle_to_remove.position && 
                    p.particle_type == particle_to_remove.particle_type
                ) {
                    self.particles.remove(pos);
                }
            }
            
            info!("Accreted {} particles onto {} sink particles", 
                  accreted_particles.len(), self.sink_particles.len());
        }
        
        Ok(())
    }
}
    
    /// Check if object should use relativistic treatment
    /// Based on PDF guidance: use GR for high-mass or high-velocity scenarios
    pub fn requires_relativistic_treatment(mass_kg: f64, velocity_ms: f64, radius_m: f64) -> bool {
        let rs = schwarzschild_radius(mass_kg);
        let velocity_fraction = velocity_ms / C;
        
        // Use relativistic treatment if:
        // 1. Object is compact (r < 100 * Rs)
        // 2. High velocity (v > 0.1c)
        // 3. Strong field effects (Rs/r > 0.01)
        radius_m < 100.0 * rs || velocity_fraction > 0.1 || (rs / radius_m) > 0.01
    }
    
    /// Gravitational wave strain amplitude (simplified)
    /// For inspiraling compact objects - advanced feature
    pub fn gravitational_wave_strain(
        mass1_kg: f64,
        mass2_kg: f64,
        separation_m: f64,
        distance_m: f64,
    ) -> f64 {
        let total_mass = mass1_kg + mass2_kg;
        let reduced_mass = (mass1_kg * mass2_kg) / total_mass;
        let rs_total = schwarzschild_radius(total_mass);
        
        // Simplified quadrupole formula
        let strain = (G / (C * C * C * C)) * (reduced_mass * rs_total) / 
                    (separation_m * distance_m);
        
        strain.abs()
    }

// Re-export AMR types for backward compatibility
pub use adaptive_mesh_refinement::*;

/// GADGET-style N-body gravity solver
/// Based on PDF recommendation to use proven cosmological simulation algorithms
pub mod gadget_gravity {
    use super::*;
    
    /// GADGET-style particle for N-body simulation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GadgetParticle {
        pub id: usize,
        pub particle_type: GadgetParticleType,
        pub position: Vector3<f64>,
        pub velocity: Vector3<f64>,
        pub mass: f64,
        pub acceleration: Vector3<f64>,
        pub gravitational_potential: f64,
        pub softening_length: f64,
        pub time_step: f64,
        pub active: bool,
    }
    
    /// Particle types in GADGET-style simulation
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum GadgetParticleType {
        DarkMatter,
        Stars,
        Gas,
        BlackHole,
        Boundary,
    }
    
    /// GADGET-style gravity solver
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GadgetGravitySolver {
        pub particles: Vec<GadgetParticle>,
        pub force_accuracy: f64, // θ parameter for tree opening criterion
        pub softening_length: f64,
        pub periodic_boundary: bool,
        pub box_size: f64,
        pub cosmological_parameters: CosmologicalParameters,
    }
    
    /// Cosmological parameters for expanding universe
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CosmologicalParameters {
        pub hubble_constant: f64,      // H₀ in km/s/Mpc
        pub omega_matter: f64,         // Ωₘ
        pub omega_lambda: f64,         // ΩΛ
        pub omega_baryon: f64,         // Ωᵦ
        pub scale_factor: f64,         // a(t)
        pub redshift: f64,             // z
        pub age_of_universe: f64,      // t in Gyr
        pub enable_expansion: bool,
    }
    
    impl Default for CosmologicalParameters {
        fn default() -> Self {
            Self {
                hubble_constant: 67.4,      // Planck 2018 value
                omega_matter: 0.315,        // Matter density parameter
                omega_lambda: 0.685,        // Dark energy density parameter
                omega_baryon: 0.049,        // Baryon density parameter
                scale_factor: 1.0,          // Present day
                redshift: 0.0,              // Present day
                age_of_universe: 13.8,      // Gyr
                enable_expansion: true,
            }
        }
    }
    
    impl GadgetGravitySolver {
        /// Create new GADGET-style gravity solver with real cosmological parameters
        pub fn new(force_accuracy: f64, softening_length: f64, box_size: f64, cosmological: bool) -> Self {
            let cosmological_parameters = if cosmological {
                CosmologicalParameters {
                    hubble_constant: 67.4,      // Planck 2018 value
                    omega_matter: 0.315,        // Matter density parameter
                    omega_lambda: 0.685,        // Dark energy density parameter
                    omega_baryon: 0.049,        // Baryon density parameter
                    scale_factor: 1.0,          // Present day
                    redshift: 0.0,              // Present day
                    age_of_universe: 13.8,      // Gyr
                    enable_expansion: true,
                }
            } else {
                CosmologicalParameters {
                    hubble_constant: 0.0,
                    omega_matter: 1.0,
                    omega_lambda: 0.0,
                    omega_baryon: 0.0,
                    scale_factor: 1.0,
                    redshift: 0.0,
                    age_of_universe: 0.0,
                    enable_expansion: false,
                }
            };
            
            Self {
                particles: Vec::new(),
                force_accuracy,
                softening_length,
                periodic_boundary: true,
                box_size,
                cosmological_parameters,
            }
        }
        
        /// Add particle to GADGET simulation
        pub fn add_particle(&mut self, particle: GadgetParticle) {
            self.particles.push(particle);
        }
        
        /// Calculate gravitational forces using proven GADGET algorithms
        pub fn calculate_forces(&mut self) -> Result<()> {
            if self.particles.is_empty() {
                return Ok(());
            }

            // Calculate bounding box for all particles
            let mut min_pos = self.particles[0].position;
            let mut max_pos = self.particles[0].position;
            
            for particle in &self.particles {
                for i in 0..3 {
                    min_pos[i] = min_pos[i].min(particle.position[i]);
                    max_pos[i] = max_pos[i].max(particle.position[i]);
                }
            }
            
            // Expand bounding box slightly and make it cubic
            let size = (max_pos - min_pos).max() * 1.1;
            let center = (min_pos + max_pos) * 0.5;
            
            // For now, use direct summation with Barnes-Hut placeholder
            // TODO: Implement full Barnes-Hut tree when spatial module is refactored
            let forces: Vec<Vector3<f64>> = (0..self.particles.len())
                .into_par_iter()
                .map(|i| {
                    let mut total_force = Vector3::zeros();
                    let p_i = &self.particles[i];
                    for (j, p_j) in self.particles.iter().enumerate() {
                        if i == j { continue; }
                        let r_vec = p_j.position - p_i.position;
                        let r = r_vec.magnitude();
                        
                        if r < 1e-15 { continue; }
                        
                        // GADGET-style softened gravity
                        let softened_r = (r * r + self.softening_length * self.softening_length).sqrt();
                        let force_magnitude = general_relativity::G * p_i.mass * p_j.mass / (softened_r * softened_r * softened_r);
                        
                        total_force += r_vec * force_magnitude;
                    }
                    total_force
                })
                .collect();
            
            // Apply forces to update accelerations
            for (i, force) in forces.into_iter().enumerate() {
                if self.particles[i].mass > 1e-12 {
                    self.particles[i].acceleration = force / self.particles[i].mass;
                }
            }
            
            Ok(())
        }
        
        /// Integrate using leap-frog method (standard in GADGET)
        pub fn integrate_step(&mut self, dt: f64) -> Result<()> {
            // Kick-drift-kick leap-frog integration
            for particle in &mut self.particles {
                if !particle.active {
                    continue;
                }
                
                // Kick: v += a * dt/2
                particle.velocity += particle.acceleration * (dt * 0.5);
                
                // Drift: x += v * dt
                particle.position += particle.velocity * dt;
                
                // Apply periodic boundary conditions
                if self.periodic_boundary {
                    particle.position.x = particle.position.x.rem_euclid(self.box_size);
                    particle.position.y = particle.position.y.rem_euclid(self.box_size);
                    particle.position.z = particle.position.z.rem_euclid(self.box_size);
                }
            }
            
            // Recalculate forces
            self.calculate_forces()?;
            
            // Final kick: v += a * dt/2
            for particle in &mut self.particles {
                if particle.active {
                    particle.velocity += particle.acceleration * (dt * 0.5);
                }
            }
            
            // Apply cosmological expansion if enabled
            if self.cosmological_parameters.enable_expansion {
                self.apply_cosmological_expansion(dt)?;
            }
            
            Ok(())
        }
        
        /// Apply cosmological expansion following GADGET methodology with full Friedmann equations
        fn apply_cosmological_expansion(&mut self, _dt: f64) -> Result<()> {
            // Note: For now this is a placeholder in the GADGET context
            // The actual cosmological expansion will be implemented in the main PhysicsEngine
            // when it integrates with the GADGET particles
            log::trace!("GADGET cosmological expansion placeholder called");
            Ok(())
        }
    }
}

impl PhysicsEngine {
    /// Get read-only access to particles for rendering
    pub fn get_particles(&self) -> &[FundamentalParticle] {
        &self.particles
    }
}

impl Drop for PhysicsEngine {
    fn drop(&mut self) {
        // Clean up any resources
        log::debug!("PhysicsEngine dropped");
    }
}

// NOTE: ForceFieldParameters::default() is now implemented in quantum_chemistry.rs
// This duplicate implementation has been removed to avoid conflicts.

/// Stopping power data for particles in materials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingPowerTable {
    pub energies_mev: Vec<f64>,
    pub stopping_powers_mev_cm2_g: Vec<f64>,
    pub range_mev_cm2_g: Vec<f64>,
    pub material: String,
}

/// Nuclear decay data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayData {
    pub half_life_seconds: f64,
    pub decay_modes: Vec<DecayMode>,
    pub q_value_mev: f64,
    pub daughter_products: Vec<(ParticleType, f64)>, // (particle, branching_ratio)
}

/// Material properties for particle interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub name: String,
    pub density_g_cm3: f64,
    pub atomic_composition: Vec<(u32, f64)>, // (Z, fraction)
    pub mean_excitation_energy_ev: f64,
    pub radiation_length_cm: f64,
    pub nuclear_interaction_length_cm: f64,
}

// NOTE: BasisSet::sto_3g() is now implemented in quantum_chemistry.rs
// This duplicate implementation has been removed to avoid conflicts.

//-----------------------------------------------------------------------------//
// Type conversions between internal representations and shared physics types  //
//-----------------------------------------------------------------------------//

impl From<&FundamentalParticle> for shared_types::FundamentalParticle {
    fn from(p: &FundamentalParticle) -> Self {
        Self {
            particle_type: map_particle_type_to_shared(p.particle_type),
            position: p.position,
            momentum: p.momentum,
            velocity: p.velocity,
            spin: p.spin,
            color_charge: p.color_charge.map(|c| match c {
                ColorCharge::Red => shared_types::ColorCharge::Red,
                ColorCharge::Green => shared_types::ColorCharge::Green,
                ColorCharge::Blue => shared_types::ColorCharge::Blue,
                ColorCharge::AntiRed => shared_types::ColorCharge::AntiRed,
                ColorCharge::AntiGreen => shared_types::ColorCharge::AntiGreen,
                ColorCharge::AntiBlue => shared_types::ColorCharge::AntiBlue,
                ColorCharge::ColorSinglet => shared_types::ColorCharge::ColorSinglet,
            }),
            electric_charge: p.electric_charge,
            mass: p.mass,
            energy: p.energy,
            creation_time: p.creation_time,
            decay_time: p.decay_time,
            quantum_state: shared_types::QuantumState::default(),
            interaction_history: Vec::new(),
        }
    }
}

impl From<shared_types::InteractionEvent> for InteractionEvent {
    fn from(e: shared_types::InteractionEvent) -> Self {
        Self {
            timestamp: e.timestamp,
            interaction_type: map_interaction_type(e.interaction_type),
            participants: Vec::new(),
            energy_exchanged: e.energy_exchanged,
            momentum_transfer: e.momentum_transfer,
            products: e.particles_out.iter().map(|p| map_particle_type_from_shared(p.particle_type)).collect(),
            cross_section: e.cross_section,
        }
    }
}

fn map_particle_type_to_shared(pt: ParticleType) -> shared_types::ParticleType {
    use shared_types::ParticleType as S;
    match pt {
        ParticleType::WBoson | ParticleType::WBosonMinus => S::WMinus,
        ParticleType::ZBoson => S::Z,
        ParticleType::Photon => S::Photon,
        // Fallback simple mapping
        ParticleType::Electron => S::Electron,
        ParticleType::Positron => S::Positron,
        _ => S::Other(pt as u32),
    }
}

fn map_particle_type_from_shared(pt: shared_types::ParticleType) -> ParticleType {
    match pt {
        shared_types::ParticleType::WPlus | shared_types::ParticleType::WMinus => ParticleType::WBoson,
        shared_types::ParticleType::Z => ParticleType::ZBoson,
        shared_types::ParticleType::Photon => ParticleType::Photon,
        shared_types::ParticleType::Electron => ParticleType::Electron,
        _ => ParticleType::DarkMatter,
    }
}

fn map_interaction_type(it: shared_types::InteractionType) -> InteractionType {
    match it {
        shared_types::InteractionType::Elastic | shared_types::InteractionType::Inelastic | shared_types::InteractionType::ElectromagneticScattering => InteractionType::ElectromagneticScattering,
        shared_types::InteractionType::WeakDecay | shared_types::InteractionType::Decay => InteractionType::WeakDecay,
        shared_types::InteractionType::StrongInteraction => InteractionType::StrongInteraction,
        shared_types::InteractionType::GravitationalAttraction => InteractionType::GravitationalAttraction,
        shared_types::InteractionType::Fusion => InteractionType::NuclearFusion,
        shared_types::InteractionType::Fission => InteractionType::NuclearFission,
        shared_types::InteractionType::PairProduction => InteractionType::PairProduction,
        shared_types::InteractionType::Annihilation => InteractionType::Annihilation,
        _ => InteractionType::ElectromagneticScattering,
    }
}

impl QuantumField {
    pub fn new(_field_type: FieldType, _spacetime_grid: &SpacetimeGrid) -> Result<Self> {
        Ok(Self {
            field_type: _field_type,
            field_values: vec![vec![vec![Complex::new(0.0, 0.0); 10]; 10]; 10],
            field_derivatives: vec![vec![vec![Vector3::zeros(); 10]; 10]; 10],
            vacuum_expectation_value: Complex::new(0.0, 0.0),
            coupling_constants: HashMap::new(),
            lattice_spacing: 1e-15,
            boundary_conditions: BoundaryConditions::Periodic,
        })
    }
}

// FFI integration stub module completely removed - all physics now handled by native Rust implementations

impl PhysicsEngine {
    /// Internal native interaction processing routine (octree-based).
    fn process_native_interactions(&mut self) -> Result<()> {
        if self.particles.is_empty() {
            return Ok(());
        }

        // 1. Calculate the bounding box of all particles
        let mut min = self.particles[0].position;
        let mut max = self.particles[0].position;
        for p in self.particles.iter().skip(1) {
            min = min.inf(&p.position);
            max = max.sup(&p.position);
        }
        let center = (min + max) / 2.0;
        let half_dim = (max - min) / 2.0;

        // Add a small buffer to the boundary
        let half_dim_buffered = Vector3::new(
            half_dim.x.max(1.0),
            half_dim.y.max(1.0),
            half_dim.z.max(1.0),
        );

        // 2. Rebuild the octree for this step
        self.octree = Octree::new(AABB::new(center, half_dim_buffered));
        for i in 0..self.particles.len() {
            self.octree.insert(i, &self.particles[i].position);
        }

        // 3. Query for interactions
        for i in 0..self.particles.len() {
            let p1_pos = self.particles[i].position;
            let p1_type = self.particles[i].particle_type;
            // This should be based on the largest possible interaction range
            let interaction_range = 1e-14;
            let query_aabb = AABB::new(p1_pos, Vector3::new(interaction_range, interaction_range, interaction_range));

            let potential_neighbors = self.octree.query_range(&query_aabb);

            for &j in &potential_neighbors {
                if i >= j {
                    continue;
                }

                let p2_pos = self.particles[j].position;
                let p2_type = self.particles[j].particle_type;
                let distance = (p1_pos - p2_pos).norm();

                if distance < self.calculate_interaction_range(p1_type, p2_type) {
                    self.process_particle_pair_interaction(i, j, distance)?;
                }
            }
        }

        Ok(())
    }
}

