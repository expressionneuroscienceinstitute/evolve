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
    MeasurementBasis, DecayChannel, NuclearShellState,
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

pub mod conservation;

use conservation::{ConservationEnforcer, ConservationMonitor, ConservationConstraint, EnforcementMethod};

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
    pub acceleration: Vector3<f64>,  // Gravitational acceleration for force calculations
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
            acceleration: Vector3::zeros(),
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
    
    /// Barnes-Hut tree parameters for gravitational force calculation
    pub force_accuracy: f64,      // θ parameter for tree opening criterion (typically 0.5-1.0)
    pub softening_length: f64,    // Gravitational softening length to prevent singularities
    /// Conservation enforcement system
    pub conservation_monitor: ConservationMonitor,
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

impl QuantumField {
    pub fn new(field_type: FieldType, grid_size: usize, lattice_spacing: f64) -> Self {
        let field_values = vec![vec![vec![Complex::new(0.0, 0.0); grid_size]; grid_size]; grid_size];
        let field_derivatives = vec![vec![vec![Vector3::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)); grid_size]; grid_size]; grid_size];
        
        Self {
            field_type,
            field_values,
            field_derivatives,
            vacuum_expectation_value: Complex::new(0.0, 0.0),
            coupling_constants: HashMap::new(),
            lattice_spacing,
            boundary_conditions: BoundaryConditions::Periodic,
        }
    }
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
            force_accuracy: 0.5,
            softening_length: 1e-9,
            conservation_monitor: ConservationMonitor::new(),
        };

        // Initialize conservation enforcement system
        engine.initialize_conservation_enforcement()?;

        Ok(engine)
    }

    /// Initialize conservation enforcement system with multiple enforcers
    fn initialize_conservation_enforcement(&mut self) -> Result<()> {
        let constants = PhysicsConstants::default();

        // Energy-momentum conserving enforcer for relativistic particles
        let relativistic_constraints = vec![
            ConservationConstraint::Energy,
            ConservationConstraint::LinearMomentum,
            ConservationConstraint::RelativisticEnergyMomentum,
        ];
        let relativistic_enforcer = ConservationEnforcer::new(
            relativistic_constraints,
            EnforcementMethod::EnergyMomentumConserving { symplectic_order: 4 }
        );
        self.conservation_monitor.add_enforcer("relativistic".to_string(), relativistic_enforcer);

        // Symplectic correction enforcer for classical dynamics
        let classical_constraints = vec![
            ConservationConstraint::Energy,
            ConservationConstraint::LinearMomentum,
            ConservationConstraint::AngularMomentum,
        ];
        let classical_enforcer = ConservationEnforcer::new(
            classical_constraints,
            EnforcementMethod::SymplecticCorrection { correction_strength: 0.1 }
        );
        self.conservation_monitor.add_enforcer("classical".to_string(), classical_enforcer);

        // Adaptive correction enforcer for multi-physics coupling
        let adaptive_constraints = vec![
            ConservationConstraint::Mass,
            ConservationConstraint::Charge,
            ConservationConstraint::EntropyIncrease,
        ];
        let adaptive_enforcer = ConservationEnforcer::new(
            adaptive_constraints,
            EnforcementMethod::AdaptiveCorrection { learning_rate: 0.01, history_size: 100 }
        );
        self.conservation_monitor.add_enforcer("adaptive".to_string(), adaptive_enforcer);

        Ok(())
    }

    /// Step the physics simulation with conservation enforcement
    pub fn step(&mut self) -> Result<()> {
        let start = Instant::now();
        
        // Apply conservation enforcement before physics step
        self.conservation_monitor.monitor_conservation(
            &mut self.particles,
            &PhysicsConstants::default(),
            self.time_step
        )?;

        // Perform physics step
        self.update_particle_energies()?;
        self.process_particle_interactions()?;
        self.update_octree()?;
        
        // Apply conservation enforcement after physics step
        self.conservation_monitor.monitor_conservation(
            &mut self.particles,
            &PhysicsConstants::default(),
            self.time_step
        )?;

        self.current_time += self.time_step;
        
        log::debug!(
            "[physics] Step completed in {:.3?} (time: {:.3e} s, particles: {})",
            start.elapsed(),
            self.current_time,
            self.particles.len()
        );
        
        Ok(())
    }

    /// Get conservation violation statistics
    pub fn get_conservation_statistics(&self) -> &conservation::ConservationStatistics {
        self.conservation_monitor.get_global_statistics()
    }

    /// Enable or disable conservation monitoring
    pub fn set_conservation_monitoring(&mut self, enabled: bool) {
        self.conservation_monitor.set_monitoring_enabled(enabled);
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
            let field = QuantumField::new(field_type, 1, 1.0);
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
                acceleration: Vector3::zeros(),
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
                    acceleration: Vector3::zeros(),
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
                acceleration: Vector3::zeros(),
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
    
    /// Process particle interactions using the internal native Rust implementation.
    pub fn process_particle_interactions(&mut self) -> Result<()> {
        self.process_native_interactions_optimized()
    }

    /// Process molecular dynamics using LAMMPS if available
    pub fn process_molecular_dynamics(&mut self) -> Result<()> {
        // TODO: Restore when FFI engines are available
        // if let Some(ref mut lammps) = self.lammps_engine {
        //     self.process_lammps_dynamics(lammps)?;
        // } else {
            // Fallback to native molecular dynamics
            self.process_molecular_dynamics()?;
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
        // self.process_gravitational_collapse()?;
        
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
        // self.process_nuclear_fission()?;
        
        // Update nuclear shell structure
        // self.update_nuclear_shells()?;
        
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
        // Legacy fusion processing for compatibility
        for i in 0..self.nuclei.len() {
            for j in (i + 1)..self.nuclei.len() {
                let nucleus1 = &self.nuclei[i];
                let nucleus2 = &self.nuclei[j];
                
                // Check if fusion is possible (simplified)
                if nucleus1.atomic_number + nucleus2.atomic_number <= 26 { // Iron limit
                    // Calculate fusion reaction (simplified)
                    // let reaction = self.calculate_fusion_reaction(i, j)?;
                    
                    // Execute fusion reaction (simplified)
                    // self.execute_fusion_reaction(reaction)?;
                }
            }
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
                acceleration: Vector3::zeros(),
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
        }

        Ok(())
    }

    /// Update gravitational forces using Barnes-Hut tree algorithm
    fn update_gravitational_forces(&mut self) -> Result<()> {
        // Barnes-Hut tree algorithm for O(N log N) gravitational force calculation
        let timer = Instant::now();
        log::debug!(
            "[gravity] Computing Barnes-Hut forces for {} particles on {} threads",
            self.particles.len(),
            rayon::current_num_threads()
        );

        if self.particles.is_empty() {
            return Ok(());
        }

        // Extract particle data for Barnes-Hut tree
        let positions: Vec<Vector3<f64>> = self.particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();

        // Calculate bounding box for all particles
        let mut min_pos = positions[0];
        let mut max_pos = positions[0];
        
        for pos in &positions {
            for i in 0..3 {
                min_pos[i] = min_pos[i].min(pos[i]);
                max_pos[i] = max_pos[i].max(pos[i]);
            }
        }
        
        // Expand bounding box slightly and make it cubic
        let size = (max_pos - min_pos).max() * 1.1;
        let center = (min_pos + max_pos) * 0.5;
        let half_dimension = Vector3::new(size * 0.5, size * 0.5, size * 0.5);
        
        // Create Barnes-Hut tree with GADGET parameters
        let boundary = octree::AABB::new(center, half_dimension);
        let mut barnes_hut_tree = octree::Octree::new_barnes_hut(
            boundary,
            self.force_accuracy,  // Use GADGET's θ parameter
            general_relativity::G  // Use GADGET's gravitational constant
        );
        
        // Build the Barnes-Hut tree
        barnes_hut_tree.build_tree(&positions, &masses)?;
        
        // Compute forces using Barnes-Hut algorithm (parallel)
        let forces = barnes_hut_tree.compute_gravitational_forces_parallel(
            &positions, 
            &masses, 
            self.softening_length  // Use GADGET's softening length
        );
        
        // Apply forces to update accelerations
        for (i, force) in forces.into_iter().enumerate() {
            if self.particles[i].mass > 1e-12 {
                self.particles[i].acceleration = force / self.particles[i].mass;
            }
        }
        
        // Log performance statistics
        let tree_stats = barnes_hut_tree.get_stats();
        log::debug!(
            "[gravity] Barnes-Hut force computation completed in {:.3?}",
            timer.elapsed()
        );
        log::debug!(
            "[gravity] Tree stats: {} nodes, {} leaves, depth {}, {} particles",
            tree_stats.total_nodes,
            tree_stats.leaf_nodes,
            tree_stats.max_depth,
            tree_stats.total_particles
        );
        
        Ok(())
    }

    /// Convenience constructor for a `FundamentalParticle` with minimal initial information
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
            acceleration: Vector3::zeros(),
        })
    }

    /// Get read-only access to particles for rendering
    pub fn get_particles(&self) -> &[FundamentalParticle] {
        &self.particles
    }

    /// Update the octree for spatial optimization
    fn update_octree(&mut self) -> Result<()> {
        // Convert particles to a format suitable for the octree
        let particles: Vec<_> = self.particles.iter().map(|p| p.position).collect();
        // Use a default bounding box for now (TODO: compute from particles)
        let bounding_box = AABB::new(Vector3::zeros(), Vector3::new(1e-3, 1e-3, 1e-3));
        // TODO: Implement octree update logic here
        // self.octree.update_tree(&particles, &bounding_box)?;
        Ok(())
    }
}

impl Drop for PhysicsEngine {
    fn drop(&mut self) {
        // Clean up any resources
        log::debug!("PhysicsEngine dropped");
    }
}

// Re-export AMR types for backward compatibility
pub use adaptive_mesh_refinement::*;

/// GADGET-style N-body gravity solver
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
                CosmologicalParameters::default()
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
        
        /// Calculate gravitational forces using Barnes-Hut tree algorithm
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
            let half_dimension = Vector3::new(size * 0.5, size * 0.5, size * 0.5);
            
            // Create Barnes-Hut tree with GADGET parameters
            let boundary = octree::AABB::new(center, half_dimension);
            let mut barnes_hut_tree = octree::Octree::new_barnes_hut(
                boundary,
                self.force_accuracy,  // Use GADGET's θ parameter
                general_relativity::G  // Use GADGET's gravitational constant
            );
            
            // Extract particle data for Barnes-Hut tree
            let positions: Vec<Vector3<f64>> = self.particles.iter().map(|p| p.position).collect();
            let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();
            
            // Build the Barnes-Hut tree
            barnes_hut_tree.build_tree(&positions, &masses)?;
            
            // Compute forces using Barnes-Hut algorithm (parallel)
            let forces = barnes_hut_tree.compute_gravitational_forces_parallel(
                &positions, 
                &masses, 
                self.softening_length  // Use GADGET's softening length
            );
            
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
        
        /// Apply cosmological expansion following GADGET methodology
        fn apply_cosmological_expansion(&mut self, _dt: f64) -> Result<()> {
            // Note: For now this is a placeholder in the GADGET context
            // The actual cosmological expansion will be implemented in the main PhysicsEngine
            // when it integrates with the GADGET particles
            log::trace!("GADGET cosmological expansion placeholder called");
            Ok(())
        }
    }
}

// Type conversions between internal representations and shared physics types
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
            quantum_state: shared_types::QuantumState::default(), // TODO: Map fields if needed
            interaction_history: vec![], // TODO: Map if needed
            // charge: p.charge, // REMOVE: not in target struct
            // acceleration: p.acceleration, // REMOVE: not in target struct
        }
    }
}

// Add a stub for map_particle_type_to_shared if not present
fn map_particle_type_to_shared(pt: ParticleType) -> shared_types::ParticleType {
    // TODO: Implement real mapping
    shared_types::ParticleType::Proton
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Periodic,
    Dirichlet,
    Neumann,
}
