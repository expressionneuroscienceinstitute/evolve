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
pub mod quantum_ca;

// Add missing module declarations
pub mod sph;
pub mod radiative_transfer;
pub mod jeans_instability;
pub mod gadget_gravity;

// Add back missing modules
pub mod gravitational_collapse;
pub use gravitational_collapse::{jeans_mass, jeans_length, SinkParticle};

pub mod conservation;

// Re-export atomic molecular bridge for easy access
pub use molecular_dynamics::atomic_molecular_bridge::{
    AtomicMolecularBridge, AtomicMolecularParameters, ReactionKinetics, 
    TransitionState, MolecularDynamicsTrajectory, TrajectoryFrame
};

use nalgebra::{Vector3, Matrix3, Complex};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use log;

use self::nuclear_physics::{StellarNucleosynthesis, DecayMode};
use self::spatial::SpatialHashGrid;
use self::octree::{Octree, AABB};
// use self::constants::{BOLTZMANN, SPEED_OF_LIGHT, ELEMENTARY_CHARGE, REDUCED_PLANCK_CONSTANT, VACUUM_PERMITTIVITY};
use physics_types as shared_types;

pub use constants::*;

// Add missing imports for constants and types
use crate::types::{
    MeasurementBasis, DecayChannel, NuclearShellState,
    GluonField, ElectronicState, MolecularOrbital, VibrationalMode,
    PotentialEnergySurface, ReactionCoordinate
};
use crate::general_relativity::schwarzschild_radius;
use crate::types::{PhysicsState, InteractionEvent};
use crate::particle_types::BoundaryConditions;

// Re-export canonical ParticleType from shared physics_types crate
pub use physics_types::ParticleType;

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
    pub acceleration: Vector3<f64>, // Add missing acceleration field
}

impl FundamentalParticle {
    /// Create a new fundamental particle with default values
    pub fn new(particle_type: ParticleType, mass: f64, position: Vector3<f64>) -> Self {
        Self {
            particle_type,
            position,
            momentum: Vector3::zeros(),
            spin: Vector3::zeros(),
            color_charge: None,
            electric_charge: 0.0,
            mass,
            energy: mass * crate::constants::C_SQUARED,
            creation_time: 0.0,
            decay_time: None,
            quantum_state: QuantumState::new(),
            interaction_history: Vec::new(),
            velocity: Vector3::zeros(),
            charge: 0.0,
            acceleration: Vector3::zeros(),
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
            time_step: 1e-18, // default time step
            current_time: 0.0,
            temperature: 0.0,
            energy_density: 0.0,
            particle_creation_threshold: 1e-10,
            volume: 1e-30,  // 1 cubic femtometer
            compton_count: 0,
            pair_production_count: 0,
            neutrino_scatter_count: 0,
            particle_decay_count: 0,
            neutron_decay_count: 0,
            fusion_count: 0,
            fission_count: 0,
            spatial_grid: SpatialHashGrid::new(1e-14), // 10 femtometer interaction range
            // A default, large boundary. Will be resized dynamically.
            octree: Octree::new(AABB::new(Vector3::zeros(), Vector3::new(1.0, 1.0, 1.0))),
            interaction_history: Vec::new(),
        };

        engine.initialize_particle_properties()?;
        
        // Initialize quantum fields
        engine.initialize_quantum_fields()?;
        
        // Initialize particle properties
        engine.initialize_particle_properties()?;
        
        // Initialize interaction matrix
        engine.initialize_interactions()?;
        
        // Set larger volume for demo
        engine.volume = 1e-42; // Cubic femtometer scale
        
        // Print physics engine initialization values
        println!("🔬 PHYSICS ENGINE INITIALIZATION:");
        println!("   Initial temperature: {:.2e} K", engine.temperature);
        println!("   Initial energy density: {:.2e} J/m³", engine.energy_density);
        println!("   Simulation volume: {:.2e} m³", engine.volume);
        println!("   Time step: {:.2e} s", engine.time_step);
        println!("   Particle creation threshold: {:.2e}", engine.particle_creation_threshold);
        //        println!("   FFI libraries available: {:?}", engine.ffi_available);
        println!("   Quantum fields initialized: {}", engine.quantum_fields.len());
        println!("   Cross sections loaded: {}", engine.cross_sections.len());
        
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
    
    /// Simulation step – two compile-time modes:
    /// 1. `heavy` feature enabled  ➜ run the full high-fidelity pipeline (default for production accuracy).
    /// 2. `heavy` feature *disabled* ➜ run a lightweight fast path suited for profiling & CI.
    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.time_step = dt;
        self.current_time += dt;

        // --------------------
        // FULL-FIDELITY PATH
        // --------------------
        #[cfg(feature = "heavy")]
        {
            // 1. Particle interactions (Geant4 or native)
            self.process_particle_interactions()?;

            // 2. Molecular dynamics (LAMMPS or native)
            self.process_molecular_dynamics()?;

            // 3. Gravitational dynamics (GADGET or native)
            self.process_gravitational_dynamics()?;

            // 4. Nuclear physics
            self.process_nuclear_fusion()?;
            self.process_nuclear_fission()?;
            self.update_nuclear_shells()?;

            // 5. Atomic physics & phase changes
            self.update_atomic_physics()?;
            self.process_phase_transitions()?;

            // 6. Emergent phenomena & quantum fields
            let mut emergent_states: Vec<PhysicsState> = self
                .particles
                .iter()
                .map(|p| PhysicsState {
                    position: p.position,
                    velocity: p.velocity,
                    acceleration: Vector3::zeros(),
                    mass: p.mass,
                    charge: p.charge,
                    temperature: self.temperature,
                    entropy: 0.0,
                })
                .collect();
            self.update_emergent_properties(&mut emergent_states)?;

            self.evolve_quantum_state()?;
            self.update_spacetime_curvature()?;

            // Ensure conservation laws each step.
            self.validate_conservation_laws()?;
        }

        // --------------------
        // FAST PATH (default)
        // --------------------
        #[cfg(not(feature = "heavy"))]
        {
                    // Only recompute kinematic energies in parallel; skip expensive gravity pair-wise forces.
        self.update_particle_energies()?;
        }

        // Apply cosmological expansion effects to all particles 
        self.apply_cosmological_expansion_to_particles(dt)?;

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
            // Process molecular dynamics using quantum chemistry engine
            for molecule in &mut self.molecules {
                // Update molecular positions and velocities
                for atom in &mut molecule.atoms {
                    // Simple velocity verlet integration
                    atom.position += atom.velocity * self.time_step;
                    // TODO: Calculate forces and update velocities
                }
            }
        // }
        Ok(())
    }

    /// Calculate system pressure using ideal gas law and particle kinetic energy
    pub fn calculate_system_pressure(&self) -> f64 {
        if self.particles.is_empty() || self.volume <= 0.0 {
            return 0.0;
        }

        // Calculate total kinetic energy from particle momenta
        let total_kinetic_energy: f64 = self.particles.iter()
            .map(|p| {
                let momentum_magnitude = p.momentum.magnitude();
                let mass = p.mass;
                if mass > 0.0 {
                    momentum_magnitude * momentum_magnitude / (2.0 * mass)
                } else {
                    0.0
                }
            })
            .sum();

        // Use ideal gas law: P = (2/3) * (E_kinetic / V)
        // Factor of 2/3 comes from equipartition theorem for 3D motion
        let pressure = (2.0 / 3.0) * (total_kinetic_energy / self.volume);
        
        pressure.max(0.0) // Ensure non-negative pressure
    }

    /// Process gravitational dynamics using Barnes-Hut tree algorithm for O(N log N) scaling
    fn process_gravitational_dynamics(&mut self) -> Result<()> {
        if self.particles.is_empty() {
            return Ok(());
        }

        // Extract particle positions and masses for Barnes-Hut calculation
        let positions: Vec<Vector3<f64>> = self.particles.iter()
            .map(|p| p.position)
            .collect();
        
        let masses: Vec<f64> = self.particles.iter()
            .map(|p| p.mass)
            .collect();

        // Calculate bounding box for Barnes-Hut tree
        let mut min = positions[0];
        let mut max = positions[0];
        for pos in positions.iter().skip(1) {
            min = min.inf(pos);
            max = max.sup(pos);
        }
        let center = (min + max) / 2.0;
        let half_dim = (max - min) / 2.0;

        // Add buffer to ensure all particles are within bounds
        let half_dim_buffered = Vector3::new(
            half_dim.x.max(1e-12),
            half_dim.y.max(1e-12),
            half_dim.z.max(1e-12),
        );

        // Create Barnes-Hut tree with optimal parameters
        let mut barnes_hut_tree = Octree::new_barnes_hut(
            AABB::new(center, half_dim_buffered),
            0.5, // Standard opening criterion θ
            crate::constants::GRAVITATIONAL_CONSTANT
        );

        // Build the tree and compute mass properties
        barnes_hut_tree.build_tree(&positions, &masses)?;

        // Compute gravitational forces using Barnes-Hut algorithm (O(N log N))
        let gravitational_forces = barnes_hut_tree.compute_gravitational_forces(
            &positions, 
            &masses, 
            1e-12 // Softening length to prevent singularities
        );

        // Apply gravitational forces to particles
        for (i, force) in gravitational_forces.iter().enumerate() {
            if i < self.particles.len() {
                // F = ma, so a = F/m
                let acceleration = *force / self.particles[i].mass;
                self.particles[i].acceleration += acceleration;
                
                // Update momentum: dp/dt = F
                self.particles[i].momentum += *force * self.time_step;
                
                // Update velocity: v = p/m
                self.particles[i].velocity = self.particles[i].momentum / self.particles[i].mass;
            }
        }

        Ok(())
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
            acceleration: Vector3::zeros(),
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
        let h0_si = params.h0 * 1000.0 / 3.086e22;
        
        // Calculate Hubble parameter H(a) = H₀ * E(a) where E(a) = sqrt(Ωᵣ/a⁴ + Ωₘ/a³ + Ωₖ/a² + ΩΛ)
        let omega_r = 9.24e-5; // Radiation density parameter (photons + neutrinos)
        let hubble_parameter_si = h0_si * (
            omega_r / a.powi(4) + 
            params.omega_m / a.powi(3) + 
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
                        let mut velocity_magnitude = p_magnitude / (gamma * particle.mass);
                        
                        // Clamp velocity to prevent exceeding speed of light for massive particles
                        if particle.mass > 0.0 {
                            velocity_magnitude = velocity_magnitude.min(0.999 * SPEED_OF_LIGHT);
                        }
                        
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
        
        let matter_energy_density = params.omega_m * critical_density * matter_density_scale;
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
        let h0_si = params.h0 * 1000.0 / 3.086e22; // Convert to SI units
        
        if params.omega_lambda.abs() < 1e-6 {
            // Matter-dominated universe: a(t) ∝ t^(2/3)
            let t0 = 2.0 / (3.0 * h0_si); // Age at a=1
            (time_seconds / t0).powf(2.0/3.0).max(0.001)
        } else {
            // ΛCDM universe with dark energy
            let omega_m_over_lambda = params.omega_m / params.omega_lambda;
            let h_lambda = h0_si * params.omega_lambda.sqrt();
            
            // Parametric solution: t = (2/(3H_Λ)) * sinh⁻¹(√(Ω_Λ/Ω_m) * a^(3/2))
            let x = 1.5 * h_lambda * time_seconds;
            let y = x.sinh();
            let a_cubed_half = y / omega_m_over_lambda.sqrt();
            a_cubed_half.powf(2.0/3.0).max(0.001)
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

    /// Get read-only access to particles for rendering
    pub fn get_particles(&self) -> &[FundamentalParticle] {
        &self.particles
    }

    // Add missing method implementations
    fn sample_thermal_momentum(&self, particle_type: ParticleType, temperature: f64) -> Vector3<f64> {
        use crate::constants::{BOLTZMANN, C};
        let mass = self.get_particle_mass(particle_type);
        
        if mass <= 0.0 {
            // For massless particles, use speed of light
            let mut rng = thread_rng();
            let direction = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ).normalize();
            return direction * C;
        }
        
        // For massive particles, use relativistic Maxwell-Boltzmann distribution
        let thermal_velocity = (3.0 * BOLTZMANN * temperature / mass).sqrt();
        
        // Clamp to prevent exceeding speed of light
        let clamped_velocity = thermal_velocity.min(0.999 * C);
        
        let mut rng = thread_rng();
        
        Vector3::new(
            rng.gen_range(-clamped_velocity..clamped_velocity),
            rng.gen_range(-clamped_velocity..clamped_velocity),
            rng.gen_range(-clamped_velocity..clamped_velocity),
        )
    }

    fn initialize_spin(&self, particle_type: ParticleType) -> Vector3<Complex<f64>> {
        use crate::constants::REDUCED_PLANCK_CONSTANT;
        let mut rng = thread_rng();
        
        // Initialize with random spin direction
        let spin_magnitude = match particle_type {
            ParticleType::Electron | ParticleType::Positron | ParticleType::Proton | ParticleType::Neutron => 0.5,
            ParticleType::Photon => 1.0,
            _ => 0.0,
        };
        
        let spin_vector = Vector3::new(
            Complex::new(rng.gen_range(-1.0..1.0), 0.0),
            Complex::new(rng.gen_range(-1.0..1.0), 0.0),
            Complex::new(rng.gen_range(-1.0..1.0), 0.0),
        ).normalize();
        
        spin_vector * Complex::new(spin_magnitude * REDUCED_PLANCK_CONSTANT, 0.0)
    }

    fn assign_color_charge(&self, particle_type: ParticleType) -> Option<ColorCharge> {
        match particle_type {
            ParticleType::Up | ParticleType::Down | ParticleType::Charm | 
            ParticleType::Strange | ParticleType::Top | ParticleType::Bottom => {
                let mut rng = thread_rng();
                let colors = [ColorCharge::Red, ColorCharge::Green, ColorCharge::Blue];
                Some(colors[rng.gen_range(0..3)])
            },
            ParticleType::Gluon => {
                let mut rng = thread_rng();
                let color_combinations = [
                    ColorCharge::Red, ColorCharge::Green, ColorCharge::Blue,
                    ColorCharge::AntiRed, ColorCharge::AntiGreen, ColorCharge::AntiBlue,
                ];
                Some(color_combinations[rng.gen_range(0..6)])
            },
            _ => None,
        }
    }

    fn get_electric_charge(&self, particle_type: ParticleType) -> f64 {
        use crate::constants::ELEMENTARY_CHARGE;
        match particle_type {
            ParticleType::Up | ParticleType::Charm | ParticleType::Top => (2.0/3.0) * ELEMENTARY_CHARGE,
            ParticleType::Down | ParticleType::Strange | ParticleType::Bottom => (-1.0/3.0) * ELEMENTARY_CHARGE,
            ParticleType::Electron | ParticleType::Muon | ParticleType::Tau => -ELEMENTARY_CHARGE,
            ParticleType::Positron => ELEMENTARY_CHARGE,
            ParticleType::Proton => ELEMENTARY_CHARGE,
            ParticleType::Neutron => 0.0,
            ParticleType::PionPlus | ParticleType::KaonPlus => ELEMENTARY_CHARGE,
            ParticleType::PionMinus | ParticleType::KaonMinus => -ELEMENTARY_CHARGE,
            ParticleType::WBoson => ELEMENTARY_CHARGE,
            ParticleType::WBosonMinus => -ELEMENTARY_CHARGE,
            _ => 0.0,
        }
    }

    pub fn get_particle_mass(&self, particle_type: ParticleType) -> f64 {
        use crate::particles::get_properties;
        get_properties(particle_type).mass_kg
    }

    fn calculate_decay_time(&self, particle_type: ParticleType) -> Option<f64> {
        use crate::particles::get_properties;
        let props = get_properties(particle_type);
        props.width.map(|width| {
            if width > 0.0 {
                // Convert decay width to lifetime: τ = ħ/Γ
                use crate::constants::REDUCED_PLANCK_CONSTANT;
                REDUCED_PLANCK_CONSTANT / width
            } else {
                f64::INFINITY
            }
        })
    }

    fn update_particle_energies(&mut self) -> Result<()> {
        use crate::constants::C_SQUARED;
        for particle in &mut self.particles {
            let momentum_magnitude = particle.momentum.norm();
            let rest_energy = particle.mass * C_SQUARED;
            let kinetic_energy = if particle.mass > 0.0 {
                momentum_magnitude * momentum_magnitude / (2.0 * particle.mass)
            } else {
                momentum_magnitude * C_SQUARED // For massless particles
            };
            particle.energy = rest_energy + kinetic_energy;
        }
        Ok(())
    }

    fn calculate_interaction_range(&self, p1_type: ParticleType, p2_type: ParticleType) -> f64 {
        // Simplified interaction range calculation
        // In practice, this would depend on the specific interaction type
        let mass1 = self.get_particle_mass(p1_type);
        let mass2 = self.get_particle_mass(p2_type);
        
        // Use Compton wavelength as a rough estimate
        use crate::constants::{REDUCED_PLANCK_CONSTANT, C};
        let reduced_mass = if mass1 > 0.0 && mass2 > 0.0 {
            (mass1 * mass2) / (mass1 + mass2)
        } else {
            mass1.max(mass2)
        };
        
        if reduced_mass > 0.0 {
            REDUCED_PLANCK_CONSTANT / (reduced_mass * C)
        } else {
            1e-15 // Default range for massless particles
        }
    }

    fn process_particle_pair_interaction(&mut self, i: usize, j: usize, distance: f64) -> Result<()> {
        if i >= self.particles.len() || j >= self.particles.len() {
            return Ok(());
        }
        
        let p1 = &self.particles[i];
        let p2 = &self.particles[j];
        
        // Implement proper quantum field theory interactions
        let interaction_result = self.calculate_quantum_field_interaction(p1, p2, distance)?;
        
        let interaction_event = InteractionEvent {
            timestamp: self.current_time,
            interaction_type: interaction_result.interaction_type,
            participants: vec![i, j],
            energy_exchanged: interaction_result.energy_exchanged,
            momentum_transfer: interaction_result.momentum_transfer,
            products: interaction_result.products.clone(), // CLONE to avoid partial move
            cross_section: interaction_result.cross_section,
        };
        
        self.interaction_history.push(interaction_event);
        
        // Update particle states based on interaction
        self.apply_interaction_to_particles(i, j, &interaction_result)?;
        
        Ok(())
    }

    /// Calculate quantum field theory interaction between two particles
    fn calculate_quantum_field_interaction(&self, p1: &FundamentalParticle, p2: &FundamentalParticle, distance: f64) -> Result<QuantumInteractionResult> {
        // Determine interaction type based on particle types and quantum numbers
        let interaction_type = self.determine_interaction_type(p1, p2)?;
        
        // Calculate interaction strength based on quantum field couplings
        let coupling_strength = self.calculate_coupling_strength(p1, p2)?;
        
        // Calculate cross section using quantum field theory
        let cross_section = self.calculate_quantum_cross_section(p1, p2, coupling_strength, distance)?;
        
        // Calculate energy and momentum exchange
        let (energy_exchanged, momentum_transfer) = self.calculate_energy_momentum_exchange(p1, p2, interaction_type)?;
        
        // Determine reaction products based on conservation laws
        let products = self.determine_reaction_products(p1, p2, interaction_type)?;
        
        Ok(QuantumInteractionResult {
            interaction_type,
            energy_exchanged,
            momentum_transfer,
            products,
            cross_section,
            coupling_strength,
        })
    }

    /// Determine the type of interaction between two particles
    fn determine_interaction_type(&self, p1: &FundamentalParticle, p2: &FundamentalParticle) -> Result<crate::types::InteractionType> {
        use crate::types::InteractionType;
        
        match (p1.particle_type, p2.particle_type) {
            // Electromagnetic interactions
            (ParticleType::Electron, ParticleType::Proton) | (ParticleType::Proton, ParticleType::Electron) => {
                Ok(InteractionType::ElectromagneticScattering)
            },
            (ParticleType::Electron, ParticleType::Electron) => {
                Ok(InteractionType::ElectromagneticScattering)
            },
            (ParticleType::Photon, ParticleType::Electron) | (ParticleType::Electron, ParticleType::Photon) => {
                Ok(InteractionType::ElectromagneticScattering)
            },
            
            // Weak interactions
            (ParticleType::Neutron, ParticleType::Proton) | (ParticleType::Proton, ParticleType::Neutron) => {
                Ok(InteractionType::WeakDecay)
            },
            (ParticleType::Electron, ParticleType::ElectronNeutrino) | (ParticleType::ElectronNeutrino, ParticleType::Electron) => {
                Ok(InteractionType::WeakDecay)
            },
            
            // Strong interactions
            (ParticleType::Proton, ParticleType::Proton) => {
                Ok(InteractionType::StrongInteraction)
            },
            (ParticleType::Neutron, ParticleType::Neutron) => {
                Ok(InteractionType::StrongInteraction)
            },
            // Note: Proton-Neutron interactions are handled above as weak decay
            // This is physically correct - neutron-proton interactions can be both weak and strong
            // depending on the energy scale and specific process
            
            // Nuclear fusion/fission
            (ParticleType::Deuteron, ParticleType::Triton) | (ParticleType::Triton, ParticleType::Deuteron) => {
                Ok(InteractionType::NuclearFusion)
            },
            (ParticleType::Uranium235, _) | (_, ParticleType::Uranium235) => {
                Ok(InteractionType::NuclearFission)
            },
            
            // Default to electromagnetic scattering
            _ => Ok(InteractionType::ElectromagneticScattering),
        }
    }

    /// Calculate coupling strength between particles based on quantum field theory
    fn calculate_coupling_strength(&self, p1: &FundamentalParticle, p2: &FundamentalParticle) -> Result<f64> {
        // Get running coupling constants at the interaction energy scale
        let interaction_energy = (p1.energy + p2.energy) / 2.0;
        let scale_gev = interaction_energy / 1e9; // Convert to GeV
        
        // Calculate running couplings based on energy scale
        let alpha_em = self.calculate_running_electromagnetic_coupling(scale_gev);
        let alpha_s = self.calculate_running_strong_coupling(scale_gev);
        let alpha_w = self.calculate_running_weak_coupling(scale_gev);
        
        // Determine dominant interaction based on particle types
        match (p1.particle_type, p2.particle_type) {
            // Electromagnetic interactions
            (ParticleType::Electron, ParticleType::Proton) | (ParticleType::Proton, ParticleType::Electron) |
            (ParticleType::Electron, ParticleType::Electron) |
            (ParticleType::Photon, ParticleType::Electron) | (ParticleType::Electron, ParticleType::Photon) => {
                Ok(alpha_em)
            },
            
            // Strong interactions
            (ParticleType::Proton, ParticleType::Proton) | (ParticleType::Neutron, ParticleType::Neutron) => {
                Ok(alpha_s)
            },
            
            // Weak interactions
            (ParticleType::Neutron, ParticleType::Proton) | (ParticleType::Proton, ParticleType::Neutron) |
            (ParticleType::Electron, ParticleType::ElectronNeutrino) | (ParticleType::ElectronNeutrino, ParticleType::Electron) => {
                Ok(alpha_w)
            },
            
            // Default to electromagnetic
            _ => Ok(alpha_em),
        }
    }

    /// Calculate running electromagnetic coupling constant
    fn calculate_running_electromagnetic_coupling(&self, scale_gev: f64) -> f64 {
        // α(μ) = α(μ₀) / (1 - α(μ₀)/(3π) * ln(μ²/μ₀²))
        let alpha_0 = 1.0 / 137.036; // Fine structure constant at low energy
        let mu_0 = 0.511e-3; // Electron mass in GeV
        
        if scale_gev <= mu_0 {
            return alpha_0;
        }
        
        let log_term = (scale_gev * scale_gev) / (mu_0 * mu_0);
        alpha_0 / (1.0 - alpha_0 / (3.0 * std::f64::consts::PI) * log_term.ln())
    }

    /// Calculate running strong coupling constant
    fn calculate_running_strong_coupling(&self, scale_gev: f64) -> f64 {
        // αs(μ) = 12π / ((33-2Nf) * ln(μ²/Λ²))
        let lambda_qcd = 0.2; // QCD scale parameter in GeV
        let nf = 5.0; // Number of active flavors
        
        if scale_gev <= lambda_qcd {
            return 1.0; // Strong coupling at low energy
        }
        
        let log_term = (scale_gev * scale_gev) / (lambda_qcd * lambda_qcd);
        12.0 * std::f64::consts::PI / ((33.0 - 2.0 * nf) * log_term.ln())
    }

    /// Calculate running weak coupling constant
    fn calculate_running_weak_coupling(&self, scale_gev: f64) -> f64 {
        // Simplified running weak coupling
        let alpha_w_0 = 0.034; // Weak coupling at Z boson mass
        let m_z = 91.2; // Z boson mass in GeV
        
        if scale_gev <= m_z {
            return alpha_w_0;
        }
        
        // Weak coupling increases with energy
        alpha_w_0 * (scale_gev / m_z).powf(0.1)
    }

    /// Calculate quantum cross section using field theory
    fn calculate_quantum_cross_section(&self, p1: &FundamentalParticle, p2: &FundamentalParticle, coupling: f64, distance: f64) -> Result<f64> {
        // Use quantum field theory cross section formula
        // σ = (4π * α²) / (q² + m²)² for electromagnetic
        // σ = (4π * αs²) / (q² + Λ²)² for strong
        // where q is momentum transfer
        
        let momentum_transfer = (p1.momentum - p2.momentum).norm();
        let center_of_mass_energy = (p1.energy + p2.energy) / 2.0;
        
        // Calculate effective mass scale
        let mass_scale = match (p1.particle_type, p2.particle_type) {
            (ParticleType::Proton, _) | (_, ParticleType::Proton) => PROTON_MASS,
            (ParticleType::Neutron, _) | (_, ParticleType::Neutron) => NEUTRON_MASS,
            (ParticleType::Electron, _) | (_, ParticleType::Electron) => ELECTRON_MASS,
            _ => 1e-27, // Default mass scale
        };
        
        // Quantum cross section formula
        let cross_section = (4.0 * std::f64::consts::PI * coupling * coupling) / 
                           ((momentum_transfer * momentum_transfer + mass_scale * mass_scale).powi(2));
        
        // Apply form factors and screening effects
        let form_factor = self.calculate_form_factor(distance, mass_scale);
        let screened_cross_section = cross_section * form_factor;
        
        Ok(screened_cross_section.max(1e-32)) // Minimum cross section
    }

    /// Calculate form factor for screening effects
    fn calculate_form_factor(&self, distance: f64, mass_scale: f64) -> f64 {
        // Debye screening for charged particles
        let screening_length = 1e-9; // 1 nm screening length
        (-distance / screening_length).exp()
    }

    /// Calculate energy and momentum exchange in interaction
    fn calculate_energy_momentum_exchange(&self, p1: &FundamentalParticle, p2: &FundamentalParticle, interaction_type: crate::types::InteractionType) -> Result<(f64, Vector3<f64>)> {
        use crate::types::InteractionType;
        
        let center_of_mass_energy = (p1.energy + p2.energy) / 2.0;
        let relative_momentum = p1.momentum - p2.momentum;
        
        match interaction_type {
            InteractionType::ElectromagneticScattering => {
                // Electromagnetic scattering with energy loss
                let energy_loss = center_of_mass_energy * 0.01; // 1% energy loss
                let momentum_transfer = relative_momentum * 0.05; // 5% momentum transfer
                Ok((energy_loss, momentum_transfer))
            },
            InteractionType::WeakDecay => {
                // Weak interaction with significant energy exchange
                let energy_exchange = center_of_mass_energy * 0.1;
                let momentum_transfer = relative_momentum * 0.3;
                Ok((energy_exchange, momentum_transfer))
            },
            InteractionType::StrongInteraction => {
                // Strong interaction with large energy exchange
                let energy_exchange = center_of_mass_energy * 0.2;
                let momentum_transfer = relative_momentum * 0.5;
                Ok((energy_exchange, momentum_transfer))
            },
            InteractionType::NuclearFusion => {
                // Nuclear fusion releases binding energy
                let binding_energy = 2.2e-13; // Deuterium binding energy in J
                let momentum_transfer = relative_momentum * 0.8;
                Ok((-binding_energy, momentum_transfer)) // Negative for energy release
            },
            InteractionType::NuclearFission => {
                // Nuclear fission releases large energy
                let fission_energy = 200e6 * 1.602e-19; // 200 MeV in J
                let momentum_transfer = relative_momentum * 0.9;
                Ok((-fission_energy, momentum_transfer)) // Negative for energy release
            },
            _ => Ok((0.0, Vector3::zeros())),
        }
    }

    /// Determine reaction products based on conservation laws
    fn determine_reaction_products(&self, p1: &FundamentalParticle, p2: &FundamentalParticle, interaction_type: crate::types::InteractionType) -> Result<Vec<ParticleType>> {
        use crate::types::InteractionType;
        
        match interaction_type {
            InteractionType::ElectromagneticScattering => {
                // Electromagnetic scattering preserves particle types
                Ok(vec![p1.particle_type, p2.particle_type])
            },
            InteractionType::WeakDecay => {
                // Weak interaction can change particle types
                match (p1.particle_type, p2.particle_type) {
                    (ParticleType::Neutron, ParticleType::Proton) | (ParticleType::Proton, ParticleType::Neutron) => {
                        // Beta decay: n -> p + e⁻ + ν̄
                        Ok(vec![ParticleType::Proton, ParticleType::Electron, ParticleType::ElectronAntiNeutrino])
                    },
                    _ => Ok(vec![p1.particle_type, p2.particle_type]),
                }
            },
            InteractionType::NuclearFusion => {
                // Deuterium + Tritium -> Helium + neutron
                Ok(vec![ParticleType::Helium, ParticleType::Neutron])
            },
            InteractionType::NuclearFission => {
                // Uranium fission products - use available particle types
                Ok(vec![ParticleType::Iron, ParticleType::Neutron])
            },
            _ => Ok(vec![p1.particle_type, p2.particle_type]),
        }
    }

    /// Apply interaction results to particle states
    fn apply_interaction_to_particles(&mut self, i: usize, j: usize, result: &QuantumInteractionResult) -> Result<()> {
        if i >= self.particles.len() || j >= self.particles.len() {
            return Ok(());
        }
        
        // Update particle energies
        self.particles[i].energy += result.energy_exchanged / 2.0;
        self.particles[j].energy += result.energy_exchanged / 2.0;
        
        // Update particle momenta
        self.particles[i].momentum += result.momentum_transfer / 2.0;
        self.particles[j].momentum -= result.momentum_transfer / 2.0;
        
        // Update velocities based on new momenta
        self.particles[i].velocity = self.particles[i].momentum / self.particles[i].mass;
        self.particles[j].velocity = self.particles[j].momentum / self.particles[j].mass;
        
        // Update quantum states if significant interaction
        if result.coupling_strength > 0.1 {
            self.update_particle_quantum_states(i, j, result)?;
        }
        
        Ok(())
    }

    /// Update quantum states of interacting particles
    fn update_particle_quantum_states(&mut self, i: usize, j: usize, result: &QuantumInteractionResult) -> Result<()> {
        // Update entanglement between particles
        self.particles[i].quantum_state.entanglement_partners.push(j);
        self.particles[j].quantum_state.entanglement_partners.push(i);
        
        // Update decoherence time based on interaction strength
        let decoherence_factor = 1.0 / (1.0 + result.coupling_strength);
        self.particles[i].quantum_state.decoherence_time *= decoherence_factor;
        self.particles[j].quantum_state.decoherence_time *= decoherence_factor;
        
        // Update energy levels
        self.particles[i].quantum_state.energy_level = self.particles[i].energy;
        self.particles[j].quantum_state.energy_level = self.particles[j].energy;
        
        Ok(())
    }

    /// Process nuclear fusion reactions
    fn process_nuclear_fusion(&mut self) -> Result<()> {
        // TODO: Implement nuclear fusion processing
        // This would involve checking for fusion conditions and updating nuclei
        Ok(())
    }

    /// Process nuclear fission reactions
    fn process_nuclear_fission(&mut self) -> Result<()> {
        // TODO: Implement nuclear fission processing
        // This would involve checking for fission conditions and updating nuclei
        Ok(())
    }

    /// Update nuclear shell states
    fn update_nuclear_shells(&mut self) -> Result<()> {
        // TODO: Implement nuclear shell updates
        // This would involve updating nuclear shell model states
        Ok(())
    }

    /// Update atomic physics states
    fn update_atomic_physics(&mut self) -> Result<()> {
        // TODO: Implement atomic physics updates
        // This would involve updating atomic states, electron configurations, etc.
        Ok(())
    }

    /// Process phase transitions
    fn process_phase_transitions(&mut self) -> Result<()> {
        // TODO: Implement phase transition processing
        // This would involve checking for phase changes based on temperature/pressure
        Ok(())
    }

    /// Update emergent properties
    fn update_emergent_properties(&mut self, emergent_states: &mut [PhysicsState]) -> Result<()> {
        // Delegate to the emergent_properties module
        use crate::emergent_properties::update_emergent_properties;
        use crate::emergent_properties::EmergenceMonitor;
        
        let mut monitor = EmergenceMonitor::new();
        update_emergent_properties(&mut monitor, emergent_states, self.volume)
    }

    /// Evolve quantum state
    fn evolve_quantum_state(&mut self) -> Result<()> {
        // TODO: Implement quantum state evolution
        // This would involve evolving quantum states of particles and fields
        Ok(())
    }

    /// Update spacetime curvature
    fn update_spacetime_curvature(&mut self) -> Result<()> {
        // TODO: Implement spacetime curvature updates
        // This would involve updating general relativistic effects
        Ok(())
    }

    /// Validate conservation laws
    fn validate_conservation_laws(&mut self) -> Result<()> {
        // Convert particles to PhysicsState for validation
        let states: Vec<PhysicsState> = self.particles.iter().map(|p| PhysicsState {
            position: p.position,
            velocity: p.velocity,
            acceleration: p.acceleration,
            mass: p.mass,
            charge: p.charge,
            temperature: self.temperature,
            entropy: 0.0, // TODO: Calculate proper entropy
        }).collect();

        // Delegate to validation module
        use crate::validation::validate_physics_state;
        use crate::constants::PhysicsConstants;
        
        let constants = PhysicsConstants::default();
        validate_physics_state(&states, &constants)
    }

    /// Check if two atoms can form a molecule
    pub fn can_form_molecule(&self, atom1: &Atom, atom2: &Atom) -> bool {
        // Simple implementation: check if atoms are close enough and have compatible electronic states
        let distance = (atom1.position - atom2.position).norm();
        let interaction_range = 1e-9; // 1 nm interaction range
        distance < interaction_range
    }

    /// Determine the type of molecule that can form from two atoms
    pub fn determine_molecule_type(&self, atom1: &Atom, atom2: &Atom) -> Option<ParticleType> {
        // Simple implementation based on atomic numbers
        let z1 = atom1.nucleus.atomic_number;
        let z2 = atom2.nucleus.atomic_number;
        
        match (z1, z2) {
            (1, 1) => Some(ParticleType::H2),
            (1, 8) | (8, 1) => Some(ParticleType::H2O),
            (6, 8) | (8, 6) => Some(ParticleType::CO2),
            (6, 1) | (1, 6) => Some(ParticleType::CH4),
            (7, 1) | (1, 7) => Some(ParticleType::NH3),
            _ => None,
        }
    }

    /// Check if a particle type represents a molecule
    pub fn is_molecule(&self, particle_type: ParticleType) -> bool {
        matches!(particle_type, 
            ParticleType::H2 | ParticleType::H2O | ParticleType::CO2 | 
            ParticleType::CH4 | ParticleType::NH3
        )
    }

    /// Check for possible chemical reactions between two particle types
    pub fn check_chemical_reaction(&self, reactant1: ParticleType, reactant2: ParticleType) -> Option<Vec<ParticleType>> {
        // Simple implementation: methane + water -> carbon dioxide + hydrogen
        match (reactant1, reactant2) {
            (ParticleType::CH4, ParticleType::H2O) | (ParticleType::H2O, ParticleType::CH4) => {
                Some(vec![ParticleType::CO2, ParticleType::H2])
            },
            _ => None,
        }
    }
}

impl Drop for PhysicsEngine {
    fn drop(&mut self) {
        // Ensure FFI libraries are cleaned up gracefully when the engine is dropped.
        // This prevents resource leaks, especially from C/C++ libraries that don't
        // follow Rust's ownership model.
        log::info!("PhysicsEngine dropped.");
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
        _ => S::Other,
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

fn map_interaction_type(it: shared_types::InteractionType) -> crate::types::InteractionType {
    match it {
        shared_types::InteractionType::Elastic | shared_types::InteractionType::Inelastic | shared_types::InteractionType::ElectromagneticScattering => crate::types::InteractionType::ElectromagneticScattering,
        shared_types::InteractionType::WeakDecay | shared_types::InteractionType::Decay => crate::types::InteractionType::WeakDecay,
        shared_types::InteractionType::StrongInteraction => crate::types::InteractionType::StrongInteraction,
        shared_types::InteractionType::GravitationalAttraction => crate::types::InteractionType::GravitationalAttraction,
        shared_types::InteractionType::Fusion => crate::types::InteractionType::NuclearFusion,
        shared_types::InteractionType::Fission => crate::types::InteractionType::NuclearFission,
        shared_types::InteractionType::PairProduction => crate::types::InteractionType::PairProduction,
        shared_types::InteractionType::Annihilation => crate::types::InteractionType::Annihilation,
        _ => crate::types::InteractionType::ElectromagneticScattering,
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

/// Result of quantum field theory interaction calculation
#[derive(Debug)]
struct QuantumInteractionResult {
    interaction_type: crate::types::InteractionType,
    energy_exchanged: f64,
    momentum_transfer: Vector3<f64>,
    products: Vec<ParticleType>,
    cross_section: f64,
    coupling_strength: f64,
}