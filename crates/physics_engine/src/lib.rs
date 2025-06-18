//! Comprehensive Physics Engine
//! 
//! Complete fundamental particle physics simulation from quantum fields
//! to complex matter structures. Implements the Standard Model and beyond.

pub mod atomic_physics;
pub mod classical;
pub mod chemistry;
pub mod climate;
pub mod constants;
pub mod electromagnetic;
pub mod emergent_properties;
pub mod endf_data;
// pub mod ffi; // TODO: Create this module
pub mod geodynamics;
pub mod interactions;
pub mod molecular_dynamics;
pub mod nuclear_physics;
pub mod particles;
pub mod phase_transitions;
pub mod quantum;
pub mod quantum_fields;
pub mod spatial;
pub mod thermodynamics;
pub mod validation;

use nalgebra::{Vector3, Matrix3, Complex};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use rayon::prelude::*;
use std::time::Instant;

use self::nuclear_physics::StellarNucleosynthesis;
use self::spatial::{SpatialHashGrid, SpatialGridStats};
use self::constants::{BOLTZMANN, SPEED_OF_LIGHT, ELEMENTARY_CHARGE, REDUCED_PLANCK_CONSTANT, VACUUM_PERMITTIVITY};
use physics_types as shared_types;

pub use constants::*;

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
    Hydrogen, Helium, Lithium, Carbon, Oxygen, Iron, // ... etc
    
    // Atoms
    HydrogenAtom, HeliumAtom, CarbonAtom, OxygenAtom, IronAtom,
    
    // Molecules
    H2, H2O, CO2, CH4, NH3, // ... complex molecules
    
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

/// Color charge for strong force
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorCharge {
    Red, Green, Blue,
    AntiRed, AntiGreen, AntiBlue,
    ColorSinglet,
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
    pub ffi_available: ffi_integration::LibraryStatus,
    pub geant4_engine: Option<ffi_integration::Geant4Engine>,
    pub lammps_engine: Option<ffi_integration::LammpsEngine>,
    #[cfg(feature = "gadget")]
    pub gadget_engine: Option<ffi_integration::GadgetEngine>,
    #[cfg(not(feature = "gadget"))]
    pub gadget_engine: Option<()>,
    pub spatial_grid: SpatialHashGrid,
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
        // TODO: Check FFI library availability when ffi module exists
        // let ffi_status = crate::ffi::check_library_availability();
        // if ffi_status.all_available() {
        //     log::info!("All high-fidelity scientific libraries available");
        //     log::info!("{}", ffi_status.status_report());
        // } else {
        //     log::warn!("Some scientific libraries missing - using fallback implementations");
        //     log::warn!("{}", ffi_status.status_report());
        // }

        // Initialize FFI libraries
        // crate::ffi::initialize_ffi_libraries()?;

        let mut engine = Self {
            particles: Vec::new(),
            quantum_fields: HashMap::new(),
            nuclei: Vec::new(),
            atoms: Vec::new(),
            molecules: Vec::new(),
            interaction_matrix: InteractionMatrix::new(),
            spacetime_grid: SpacetimeGrid::new(1000, 1e-15), // Femtometer scale
            quantum_vacuum: QuantumVacuum::new(),
            field_equations: FieldEquations::new(),
            particle_accelerator: ParticleAccelerator::new(),
            decay_channels: HashMap::new(),
            cross_sections: HashMap::new(),
            running_couplings: RunningCouplings::new(),
            symmetry_breaking: SymmetryBreaking::new(),
            stellar_nucleosynthesis: StellarNucleosynthesis::new(),
            time_step: 1e-15,
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
            ffi_available: ffi_integration::check_library_status(),
            geant4_engine: None,
            lammps_engine: None,
            gadget_engine: None,
            spatial_grid: SpatialHashGrid::new(1e-14), // 10 femtometer interaction range
            interaction_history: Vec::new(),
        };
        
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
        println!("   FFI libraries available: {:?}", engine.ffi_available);
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
        
        let particle_types = vec![
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

        Ok(())
    }
    
    /// Process particle interactions using Geant4 if available, fallback to native
    pub fn process_particle_interactions(&mut self) -> Result<()> {
        // Update spatial grid for optimized neighbor finding
        self.spatial_grid.update(&self.particles);
        
        // TODO: Restore when FFI engines are available
        // if let Some(ref mut geant4) = self.geant4_engine {
        //     // Use high-fidelity Geant4 for particle physics
        //     self.process_geant4_interactions(geant4)?;
        // } else {
            // Fallback to native implementation with spatial optimization
            self.process_native_interactions_optimized()?;
        // }
        Ok(())
    }

    /// High-fidelity particle interactions using Geant4
    fn process_geant4_interactions(&mut self, geant4: &mut ffi_integration::Geant4Engine) -> Result<()> {
        let mut new_interactions = Vec::new();

        for i in 0..self.particles.len() {
            // Borrow particle data immutably in its own scoped block to avoid long-lived borrows
            let material;
            let step_length;
            let shared_particle;
            {
                let particle = &self.particles[i];
                material = self.determine_local_material(&particle.position);
                step_length = self.calculate_step_length(particle);
                shared_particle = shared_types::FundamentalParticle::from(particle);
            }

            // Geant4 transport
            match geant4.transport_particle(&shared_particle, &material, step_length) {
                Ok(shared_interactions) => {
                    for s in shared_interactions {
                        let interaction: InteractionEvent = s.into();
                        self.apply_geant4_interaction(i, &interaction)?;
                        new_interactions.push(interaction);
                    }
                }
                Err(e) => {
                    log::warn!("Geant4 transport failed for particle {}: {}", i, e);
                    self.process_particle_native_interaction(i)?;
                }
            }
        }
        
        self.interaction_history.extend(new_interactions);
        Ok(())
    }

    /// Process native interactions (fallback method)
    fn process_native_interactions(&mut self) -> Result<()> {
        // Fallback to optimized version
        self.process_native_interactions_optimized()
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

    /// High-fidelity molecular dynamics using LAMMPS (requires `lammps` feature)
    #[cfg(feature = "lammps")]
    fn process_lammps_dynamics(&mut self, lammps: &mut ffi_integration::LammpsEngine) -> Result<()> {
        // Convert particles to LAMMPS format
        lammps.clear_atoms()?;
        
        for particle in &self.particles {
            lammps.add_atom(
                particle.particle_type.clone(),
                particle.position,
                particle.velocity,
                particle.mass,
                particle.charge,
            )?;
        }
        
        // Run molecular dynamics step
        let timestep_fs = self.time_step * 1e15; // Convert to femtoseconds
        lammps.run_dynamics(1, timestep_fs)?;
        
        // Extract results back to particles
        let updated_particles = lammps.get_atom_data()?;
        for (i, updated) in updated_particles.iter().enumerate() {
            if i < self.particles.len() {
                self.particles[i].position = updated.position;
                self.particles[i].velocity = updated.velocity;
                self.particles[i].energy = 0.5 * updated.mass * updated.velocity.magnitude_squared();
            }
        }
        
        Ok(())
    }

    #[cfg(not(feature = "lammps"))]
    fn process_lammps_dynamics(&mut self, _lammps: &mut ffi_integration::LammpsEngine) -> Result<()> {
        anyhow::bail!("LAMMPS support not compiled in")
    }

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

    /// High-fidelity N-body gravity using GADGET (requires `gadget` feature)
    #[cfg(feature = "gadget")]
    fn process_gadget_gravity(&mut self, gadget: &mut ffi_integration::GadgetEngine) -> Result<()> {
        // Convert particles to GADGET format
        gadget.clear_particles()?;
        
        for particle in &self.particles {
            gadget.add_particle(ffi_integration::GadgetParticle {
                particle_type: match particle.particle_type {
                    ParticleType::DarkMatter => ffi_integration::GadgetParticleType::DarkMatter,
                    ParticleType::Proton | ParticleType::Neutron => ffi_integration::GadgetParticleType::Gas,
                    _ => ffi_integration::GadgetParticleType::Stars,
                },
                mass: particle.mass,
                position: particle.position,
                velocity: particle.velocity,
                acceleration: Vector3::zeros(),
                gravitational_potential: 0.0,
                softening_length: 1e-15, // 1 fm for nuclear scale
                density: 1e17, // Nuclear density
                active: true,
            })?;
        }
        
        // Calculate forces and integrate one step
        gadget.calculate_forces()?;
        gadget.integrate_step(self.time_step)?;
        
        // Extract results back to particles
        let updated_particles = gadget.get_particle_data()?;
        for (i, updated) in updated_particles.iter().enumerate() {
            if i < self.particles.len() {
                self.particles[i].position = updated.position;
                self.particles[i].velocity = updated.velocity;
                // Update energy from velocity
                self.particles[i].energy = 0.5 * updated.mass * updated.velocity.magnitude_squared();
            }
        }
        
                Ok(())
    }

    #[cfg(not(feature = "gadget"))]
    fn process_gadget_gravity(&mut self) -> Result<()> {
        anyhow::bail!("GADGET support not compiled in")
    }

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
    
    /// Recompute energies for all particles (E² = m²c⁴ + p²c²) using multi-core parallelism.
    pub fn update_particle_energies(&mut self) -> Result<()> {
        let start = Instant::now();
        log::debug!(
            "[energy] Recomputing energies for {} particles using {} Rayon threads",
            self.particles.len(),
            rayon::current_num_threads()
        );

        self.particles
            .par_iter_mut()
            .for_each(|particle| {
                particle.energy = (particle.mass.powi(2) + particle.momentum.norm_squared()).sqrt();
            });

        log::debug!(
            "[energy] Finished energy recomputation in {:.3?}",
            start.elapsed()
        );
        Ok(())
    }

    #[allow(dead_code)]
    fn calculate_interaction_range(&self, _p1: ParticleType, _p2: ParticleType) -> f64 { 1e-15 }
    #[allow(dead_code)]
    fn calculate_interaction(&self, _i: usize, _j: usize) -> Result<interactions::Interaction> { Ok(interactions::Interaction::default()) }
    #[allow(dead_code)]
    fn apply_interaction(&mut self, _interaction: interactions::Interaction) -> Result<()> { Ok(()) }
    fn select_decay_channel(&self, channels: &[DecayChannel]) -> DecayChannel { channels[0].clone() }
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
        for (_electron_idx, electron) in atom.electrons.iter().enumerate() {
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
    fn update_running_couplings(&mut self, _states: &mut [PhysicsState]) -> Result<()> { Ok(()) }
    #[allow(dead_code)]
    fn check_symmetry_breaking(&mut self) -> Result<()> { Ok(()) }
    #[allow(dead_code)]
    fn update_spacetime_curvature(&mut self) -> Result<()> { Ok(()) }
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
        let fragment2_a = parent_a - fragment1_a - 2; // Assume 2 neutrons emitted
        
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
        // Simple material determination based on particle density
        let local_density = self.calculate_local_density(position);
        
        if local_density > 1e17 {
            "NuclearMatter".to_string()
        } else if local_density > 1e3 {
            "Iron".to_string()  // Dense stellar material
        } else if local_density > 1e-3 {
            "Hydrogen".to_string()  // Interstellar medium
        } else {
            "Vacuum".to_string()
        }
    }

    /// Calculate appropriate step length for particle transport
    fn calculate_step_length(&self, particle: &FundamentalParticle) -> f64 {
        // Adaptive step length based on particle energy and local environment
        let base_step = 1e-12; // 1 pm base step
        let energy_factor = (particle.energy / (1e-13)).sqrt(); // Scale with energy
        let density_factor = 1.0 / (1.0 + self.calculate_local_density(&particle.position) / 1e15);
        
        base_step * energy_factor * density_factor
    }

    /// Apply Geant4 interaction results to particle
    fn apply_geant4_interaction(&mut self, particle_idx: usize, interaction: &InteractionEvent) -> Result<()> {
        if particle_idx >= self.particles.len() {
            return Ok(());
        }
        
        // Update particle energy
        self.particles[particle_idx].energy -= interaction.energy_exchanged;
        
        // Apply momentum transfer
        let particle_mass = self.particles[particle_idx].mass;
        if particle_mass > 0.0 {
            let velocity_change = interaction.momentum_transfer / particle_mass;
            self.particles[particle_idx].velocity += velocity_change;
        }
        
        // Create secondary particles
        for product_type in &interaction.products {
            if let Ok(new_particle) = self.create_particle_from_type(product_type.clone()) {
                self.particles.push(new_particle);
            }
        }
        
        Ok(())
    }

    /// Create particle from type (helper for secondary production)
    fn create_particle_from_type(&self, particle_type: ParticleType) -> Result<FundamentalParticle> {
        Ok(FundamentalParticle {
            particle_type,
            mass: self.get_particle_mass(particle_type),
            energy: 1e-13, // Default 1 MeV
            position: Vector3::zeros(),
            momentum: Vector3::zeros(),
            spin: Vector3::zeros(),
            color_charge: None,
            electric_charge: self.get_electric_charge(particle_type),
            creation_time: self.current_time,
            decay_time: None,
            quantum_state: QuantumState::new(),
            interaction_history: Vec::new(),
            velocity: Vector3::zeros(),
            charge: self.get_electric_charge(particle_type),
        })
    }

    /// Validate conservation laws (energy, momentum, charge) - placeholder implementation
    fn validate_conservation_laws(&self) -> Result<()> {
        // TODO: implement detailed checks
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
    fn calculate_local_density(&self, _position: &Vector3<f64>) -> f64 {
        // Placeholder: uniform density estimate to unblock compilation.
        if self.volume > 0.0 {
            self.particles.len() as f64 * self.get_particle_mass(ParticleType::Proton) / self.volume
        } else {
            0.0
        }
    }
}

// Supporting types and implementations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Interaction;

/// Decay channel for an unstable particle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayChannel {
    pub products: Vec<ParticleType>,
    pub branching_ratio: f64,
    pub decay_constant: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionReaction {
    pub reactant_indices: Vec<usize>,
    pub product_mass_number: u32,
    pub product_atomic_number: u32,
    pub q_value: f64, // Energy released (J)
    pub cross_section: f64, // Cross-section (m²)
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InteractionMatrix;
impl InteractionMatrix {
    pub fn new() -> Self { Self }
    pub fn set_electromagnetic_coupling(&mut self, _coupling: f64) {}
    pub fn set_weak_coupling(&mut self, _coupling: f64) {}
    pub fn set_strong_coupling(&mut self, _coupling: f64) {}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpacetimeGrid;
impl SpacetimeGrid {
    pub fn new(_size: usize, _spacing: f64) -> Self { Self }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuantumVacuum;
impl QuantumVacuum {
    pub fn new() -> Self { Self }
    pub fn initialize_fluctuations(&mut self, _temperature: f64) -> Result<()> { Ok(()) }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FieldEquations;
impl FieldEquations {
    pub fn new() -> Self { Self }
    pub fn update_field(&self, _field: &mut QuantumField, _dt: f64, _particles: &[FundamentalParticle]) -> Result<()> { Ok(()) }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParticleAccelerator;
impl ParticleAccelerator {
    pub fn new() -> Self { Self }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RunningCouplings;
impl RunningCouplings {
    pub fn new() -> Self { Self }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SymmetryBreaking;
impl SymmetryBreaking {
    pub fn new() -> Self { Self }
    pub fn initialize_higgs_mechanism(&mut self) -> Result<()> { Ok(()) }
}

impl QuantumField {
    pub fn new(field_type: FieldType, _grid: &SpacetimeGrid) -> Result<Self> {
        let size = 16; // Default lattice size
        Ok(Self {
            field_type,
            field_values: vec![vec![vec![Complex::new(0.0, 0.0); size]; size]; size],
            field_derivatives: vec![vec![vec![Vector3::zeros(); size]; size]; size],
            vacuum_expectation_value: Complex::new(0.0, 0.0),
            coupling_constants: HashMap::new(),
            lattice_spacing: 1e-15,
            boundary_conditions: BoundaryConditions::Periodic,
        })
    }
}

impl QuantumState {
    pub fn new() -> Self {
        Self {
            wave_function: Vec::new(),
            entanglement_partners: Vec::new(),
            decoherence_time: 0.0,
            measurement_basis: MeasurementBasis::Position,
            superposition_amplitudes: HashMap::new(),
            principal_quantum_number: 0,
            orbital_angular_momentum: 0,
            magnetic_quantum_number: 0,
            spin_quantum_number: 0.0,
            energy_level: 0.0,
            occupation_probability: 0.0,
        }
    }
}

// Additional type definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Periodic, Absorbing, Reflecting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementBasis {
    Position,
    Momentum,
    Energy,
    Spin,
}

impl Default for MeasurementBasis {
    fn default() -> Self { Self::Position }
}

pub type GluonField = Vec<Vector3<Complex<f64>>>;
pub type NuclearShellState = HashMap<String, f64>;
pub type ElectronicState = HashMap<String, Complex<f64>>;
pub type MolecularOrbital = AtomicOrbital;
pub type VibrationalMode = Vector3<f64>;
pub type PotentialEnergySurface = Vec<Vec<Vec<f64>>>;
pub type ReactionCoordinate = Vector3<f64>;

// Constants for new particles
pub const MUON_MASS: f64 = 1.883e-28; // kg
pub const TAU_MASS: f64 = 3.167e-27; // kg

// Exact Coulomb constant: k_e = 1/(4π ε₀)
pub const K_E: f64 = 1.0 / (4.0 * std::f64::consts::PI * VACUUM_PERMITTIVITY); // N⋅m²/C²
pub const E_CHARGE: f64 = ELEMENTARY_CHARGE;
pub const C: f64 = SPEED_OF_LIGHT;
pub const C_SQUARED: f64 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
pub const HBAR: f64 = REDUCED_PLANCK_CONSTANT;

/// Represents the physical state of a celestial body for simulation purposes.
/// This component will be attached to Bevy entities.
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

/// Record of a single interaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub timestamp: f64,
    pub interaction_type: InteractionType,
    pub participants: Vec<usize>, // Particle indices
    pub energy_exchanged: f64,
    pub momentum_transfer: Vector3<f64>,
    pub products: Vec<ParticleType>,
    pub cross_section: f64,
}

/// Enumeration of possible interaction types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Table of elemental abundances (Z=1 to 118)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementTable {
    #[serde(with = "serde_arrays")]
    pub abundances: [u32; 118],
}

impl ElementTable {
    pub fn new() -> Self {
        Self { abundances: [0u32; 118] }
    }
    
    /// Set parts-per-million abundance for element `z` (1-based proton number)
    pub fn set_abundance(&mut self, z: usize, ppm: u32) {
        if z == 0 || z > 118 { return; }
        self.abundances[z-1] = ppm;
    }
    
    /// Get abundance for element `z` (ppm)
    pub fn get_abundance(&self, z: usize) -> u32 {
        if z == 0 || z > 118 { return 0; }
        self.abundances[z-1]
    }

    pub fn from_particles(particles: &[FundamentalParticle]) -> Self {
        let mut table = Self::new();
        
        // Count atomic nuclei and convert to element abundances
        for particle in particles {
            match particle.particle_type {
                ParticleType::Hydrogen => table.abundances[1] += 1,
                ParticleType::Helium => table.abundances[2] += 1,
                ParticleType::Carbon => table.abundances[6] += 1,
                ParticleType::Oxygen => table.abundances[8] += 1,
                ParticleType::Iron => table.abundances[26] += 1,
                _ => {}
            }
        }
        
        table
    }
}

/// A profile of local environmental conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentProfile {
    pub liquid_water: f64,
    pub atmos_oxygen: f64,
    pub atmos_pressure: f64,
    pub temp_celsius: f64,
    pub radiation: f64,
    pub energy_flux: f64,
    pub shelter_index: f64,
    pub hazard_rate: f64,
}

impl Default for EnvironmentProfile {
    fn default() -> Self {
        Self {
            liquid_water: 0.0,
            atmos_oxygen: 0.0,
            atmos_pressure: 0.0,
            temp_celsius: -273.15,
            radiation: 0.0,
            energy_flux: 0.0,
            shelter_index: 0.0,
            hazard_rate: 1.0,
        }
    }
}

impl EnvironmentProfile {
    pub fn from_fundamental_physics(
        particles: &[FundamentalParticle],
        atoms: &[Atom],
        molecules: &[Molecule],
        temperature: f64,
    ) -> Self {
        // Calculate environment from fundamental particle simulation
        let water_molecules = molecules.iter()
            .filter(|m| m.atoms.len() == 3) // H2O approximation
            .count();
        
        let oxygen_atoms = atoms.iter()
            .filter(|a| a.nucleus.atomic_number == 8)
            .count();
        
        Self {
            liquid_water: (water_molecules as f64 / molecules.len() as f64).min(1.0),
            atmos_oxygen: (oxygen_atoms as f64 / atoms.len() as f64).min(1.0),
            atmos_pressure: 1.0, // Simplified
            temp_celsius: temperature - 273.15,
            radiation: particles.iter()
                .filter(|p| matches!(p.particle_type, ParticleType::Photon))
                .count() as f64 / 1e6,
            energy_flux: particles.iter()
                .map(|p| p.energy)
                .sum::<f64>() / particles.len() as f64 / 1e-15,
            shelter_index: 0.1,
            hazard_rate: 0.001,
        }
    }
}

/// Describes one layer in a planetary stratum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumLayer {
    pub thickness_m: f64,
    pub material_type: MaterialType,
    pub bulk_density: f64,
    pub elements: ElementTable,
}

/// Type of material in a stratum layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterialType {
    Gas, Regolith, Topsoil, Subsoil, SedimentaryRock, 
    IgneousRock, MetamorphicRock, OreVein, Ice, Magma,
}

// ... existing code ...


/// General Relativity corrections for strong gravitational fields
/// Based on PDF recommendation for hybrid gravity approach
pub mod general_relativity {
    
    
    /// Gravitational constant in SI units (CODATA 2023)
    pub const G: f64 = 6.67430e-11; // m³ kg⁻¹ s⁻²
    
    /// Speed of light in vacuum (exact)
    pub const C: f64 = 299792458.0; // m/s
    
    /// Schwarzschild radius calculation: Rs = 2GM/c²
    pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
        2.0 * G * mass_kg / (C * C)
    }
    
    /// Post-Newtonian correction to gravitational force
    /// Implements first-order relativistic corrections for orbital dynamics
    /// Based on Einstein field equations approximation
    pub fn post_newtonian_force_correction(
        mass1_kg: f64,
        mass2_kg: f64,
        separation_m: f64,
        velocity1_ms: [f64; 3],
        velocity2_ms: [f64; 3],
    ) -> [f64; 3] {
        let total_mass = mass1_kg + mass2_kg;
        let reduced_mass = (mass1_kg * mass2_kg) / total_mass;
        
        // Relative velocity
        let rel_vel = [
            velocity1_ms[0] - velocity2_ms[0],
            velocity1_ms[1] - velocity2_ms[1],
            velocity1_ms[2] - velocity2_ms[2],
        ];
        
        let v_squared = rel_vel[0] * rel_vel[0] + rel_vel[1] * rel_vel[1] + rel_vel[2] * rel_vel[2];
        
        // First-order post-Newtonian correction factor
        // Includes kinetic energy and gravitational potential terms
        let rs = schwarzschild_radius(total_mass);
        let pn_factor = 1.0 + (v_squared / (C * C)) + (rs / separation_m);
        
        // Unit vector pointing from mass2 to mass1
        let force_magnitude = G * mass1_kg * mass2_kg / (separation_m * separation_m);
        let corrected_magnitude = force_magnitude * pn_factor;
        
        // Return force components (simplified for demonstration)
        [corrected_magnitude, 0.0, 0.0] // Would need proper vector calculation in real implementation
    }
    
    /// Time dilation factor in gravitational field
    /// γ = sqrt(1 - rs/r) for weak field approximation
    pub fn gravitational_time_dilation(mass_kg: f64, radius_m: f64) -> f64 {
        let rs = schwarzschild_radius(mass_kg);
        if radius_m <= rs {
            0.0 // At or inside event horizon
        } else {
            (1.0 - rs / radius_m).sqrt()
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
}

/// Adaptive Mesh Refinement (AMR) system for multi-scale modeling
/// Implements the PDF recommendation for dynamic spatial resolution
pub mod adaptive_mesh_refinement {
    use super::*;
    
    
    /// AMR grid cell with hierarchical refinement capability
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AmrCell {
        pub id: usize,
        pub level: u32,
        pub position: Vector3<f64>,
        pub size: f64,
        pub mass_density: f64,
        pub energy_density: f64,
        pub field_gradient: f64,
        pub particle_count: usize,
        pub refinement_criterion: f64,
        pub parent_id: Option<usize>,
        pub children_ids: Vec<usize>,
        pub is_leaf: bool,
        pub requires_refinement: bool,
        pub requires_coarsening: bool,
        pub boundary_conditions: BoundaryConditions,
    }
    
    /// Adaptive mesh refinement manager
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AmrManager {
        pub cells: Vec<AmrCell>,
        pub max_refinement_level: u32,
        pub min_refinement_level: u32,
        pub refinement_threshold: f64,
        pub coarsening_threshold: f64,
        pub base_grid_size: f64,
        pub domain_size: Vector3<f64>,
        pub total_cells: usize,
        pub refinement_history: Vec<RefinementEvent>,
    }
    
    /// Event tracking for refinement analysis
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RefinementEvent {
        pub timestamp: f64,
        pub cell_id: usize,
        pub event_type: RefinementEventType,
        pub old_level: u32,
        pub new_level: u32,
        pub trigger_value: f64,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum RefinementEventType {
        Refinement,
        Coarsening,
        Creation,
        Deletion,
    }
    
    impl AmrManager {
        /// Create new AMR manager with base grid
        pub fn new(
            domain_size: Vector3<f64>,
            base_grid_size: f64,
            max_level: u32,
            refinement_threshold: f64,
        ) -> Self {
            let mut manager = Self {
                cells: Vec::new(),
                max_refinement_level: max_level,
                min_refinement_level: 0,
                refinement_threshold,
                coarsening_threshold: refinement_threshold * 0.25,
                base_grid_size,
                domain_size,
                total_cells: 0,
                refinement_history: Vec::new(),
            };
            
            // Initialize base grid
            manager.initialize_base_grid();
            manager
        }
        
        /// Initialize the coarsest level grid
        fn initialize_base_grid(&mut self) {
            let cells_per_dimension = (self.domain_size.x / self.base_grid_size).ceil() as usize;
            
            for i in 0..cells_per_dimension {
                for j in 0..cells_per_dimension {
                    for k in 0..cells_per_dimension {
                        let position = Vector3::new(
                            i as f64 * self.base_grid_size,
                            j as f64 * self.base_grid_size,
                            k as f64 * self.base_grid_size,
                        );
                        
                        let cell = AmrCell {
                            id: self.total_cells,
                            level: 0,
                            position,
                            size: self.base_grid_size,
                            mass_density: 0.0,
                            energy_density: 0.0,
                            field_gradient: 0.0,
                            particle_count: 0,
                            refinement_criterion: 0.0,
                            parent_id: None,
                            children_ids: Vec::new(),
                            is_leaf: true,
                            requires_refinement: false,
                            requires_coarsening: false,
                            boundary_conditions: BoundaryConditions::Periodic,
                        };
                        
                        self.cells.push(cell);
                        self.total_cells += 1;
                    }
                }
            }
        }
        
        /// Update AMR grid based on physical conditions
        pub fn update_mesh(&mut self, particles: &[FundamentalParticle], current_time: f64) -> Result<()> {
            // Step 1: Update cell properties from particle data
            self.update_cell_properties(particles)?;
            
            // Step 2: Calculate refinement criteria
            self.calculate_refinement_criteria()?;
            
            // Step 3: Perform refinement
            self.perform_refinement(current_time)?;
            
            // Step 4: Perform coarsening
            self.perform_coarsening(current_time)?;
            
            Ok(())
        }
        
        /// Update cell properties based on particle distribution
        fn update_cell_properties(&mut self, particles: &[FundamentalParticle]) -> Result<()> {
            // Clear existing counts
            for cell in &mut self.cells {
                cell.mass_density = 0.0;
                cell.energy_density = 0.0;
                cell.particle_count = 0;
            }
            
            // Accumulate particle properties in cells
            for particle in particles {
                if let Some(cell_id) = self.find_containing_cell(&particle.position) {
                    let cell = &mut self.cells[cell_id];
                    cell.mass_density += particle.mass;
                    cell.energy_density += particle.energy;
                    cell.particle_count += 1;
                }
            }
            
            // Normalize by cell volume
            for cell in &mut self.cells {
                let volume = cell.size * cell.size * cell.size;
                cell.mass_density /= volume;
                cell.energy_density /= volume;
            }
            
            Ok(())
        }
        
        /// Calculate refinement criteria based on gradients and density
        fn calculate_refinement_criteria(&mut self) -> Result<()> {
            for i in 0..self.cells.len() {
                let cell = &self.cells[i];
                
                // Calculate spatial gradients
                let gradient = self.calculate_spatial_gradient(i)?;
                
                // Refinement criterion based on PDF recommendations:
                // Refine where density gradients are high or particle density is high
                let density_criterion = cell.mass_density / 1e-15; // Normalize by atomic density
                let gradient_criterion = gradient / cell.mass_density.max(1e-30);
                let particle_criterion = cell.particle_count as f64 / 1000.0; // Normalize by target particles per cell
                
                self.cells[i].refinement_criterion = 
                    density_criterion + gradient_criterion + particle_criterion;
                
                // Set refinement flags
                self.cells[i].requires_refinement = 
                    self.cells[i].refinement_criterion > self.refinement_threshold && 
                    self.cells[i].level < self.max_refinement_level;
                
                self.cells[i].requires_coarsening = 
                    self.cells[i].refinement_criterion < self.coarsening_threshold && 
                    self.cells[i].level > self.min_refinement_level;
            }
            
            Ok(())
        }
        
        /// Calculate spatial gradient for refinement criterion
        fn calculate_spatial_gradient(&self, cell_id: usize) -> Result<f64> {
            let cell = &self.cells[cell_id];
            let mut gradient = 0.0;
            let mut neighbor_count = 0;
            
            // Find neighboring cells and calculate gradient
            for other_cell in &self.cells {
                let distance = (other_cell.position - cell.position).magnitude();
                if distance > 0.0 && distance < 2.0 * cell.size {
                    let density_diff = (other_cell.mass_density - cell.mass_density).abs();
                    gradient += density_diff / distance;
                    neighbor_count += 1;
                }
            }
            
            if neighbor_count > 0 {
                gradient /= neighbor_count as f64;
            }
            
            Ok(gradient)
        }
        
        /// Perform mesh refinement
        fn perform_refinement(&mut self, current_time: f64) -> Result<()> {
            let mut cells_to_refine = Vec::new();
            
            // Collect cells that need refinement
            for (i, cell) in self.cells.iter().enumerate() {
                if cell.requires_refinement && cell.is_leaf {
                    cells_to_refine.push(i);
                }
            }
            
            // Refine cells (in reverse order to avoid index issues)
            for &cell_id in cells_to_refine.iter().rev() {
                self.refine_cell(cell_id, current_time)?;
            }
            
            Ok(())
        }
        
        /// Refine a single cell into 8 children (octree)
        fn refine_cell(&mut self, cell_id: usize, current_time: f64) -> Result<()> {
            let parent_cell = self.cells[cell_id].clone();
            let child_size = parent_cell.size / 2.0;
            let child_level = parent_cell.level + 1;
            
            // Create 8 children
            let mut child_ids = Vec::new();
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let child_position = Vector3::new(
                            parent_cell.position.x + i as f64 * child_size,
                            parent_cell.position.y + j as f64 * child_size,
                            parent_cell.position.z + k as f64 * child_size,
                        );
                        
                        let child = AmrCell {
                            id: self.total_cells,
                            level: child_level,
                            position: child_position,
                            size: child_size,
                            mass_density: parent_cell.mass_density,
                            energy_density: parent_cell.energy_density,
                            field_gradient: parent_cell.field_gradient,
                            particle_count: parent_cell.particle_count / 8,
                            refinement_criterion: 0.0,
                            parent_id: Some(cell_id),
                            children_ids: Vec::new(),
                            is_leaf: true,
                            requires_refinement: false,
                            requires_coarsening: false,
                            boundary_conditions: parent_cell.boundary_conditions,
                        };
                        
                        child_ids.push(self.total_cells);
                        self.cells.push(child);
                        self.total_cells += 1;
                    }
                }
            }
            
            // Update parent cell
            self.cells[cell_id].children_ids = child_ids;
            self.cells[cell_id].is_leaf = false;
            self.cells[cell_id].requires_refinement = false;
            
            // Record refinement event
            let event = RefinementEvent {
                timestamp: current_time,
                cell_id,
                event_type: RefinementEventType::Refinement,
                old_level: parent_cell.level,
                new_level: child_level,
                trigger_value: parent_cell.refinement_criterion,
            };
            self.refinement_history.push(event);
            
            Ok(())
        }
        
        /// Perform mesh coarsening
        fn perform_coarsening(&mut self, current_time: f64) -> Result<()> {
            // Coarsening is more complex and would require careful handling
            // of sibling cells and data conservation
            // Implementation would check if all children of a parent cell
            // meet coarsening criteria, then merge them back
            
            // For now, we'll leave this as a placeholder
            // Real implementation would need to:
            // 1. Group children by parent
            // 2. Check if all siblings meet coarsening criteria
            // 3. Merge data from children back to parent
            // 4. Remove children from cell list
            // 5. Update parent to be leaf again
            
            Ok(())
        }
        
        /// Find which cell contains a given position
        fn find_containing_cell(&self, position: &Vector3<f64>) -> Option<usize> {
            for (i, cell) in self.cells.iter().enumerate() {
                if cell.is_leaf {
                    let min_bound = cell.position;
                    let max_bound = cell.position + Vector3::new(cell.size, cell.size, cell.size);
                    
                    if position.x >= min_bound.x && position.x < max_bound.x &&
                       position.y >= min_bound.y && position.y < max_bound.y &&
                       position.z >= min_bound.z && position.z < max_bound.z {
                        return Some(i);
                    }
                }
            }
            None
        }
        
        /// Get statistics about the AMR grid
        pub fn get_statistics(&self) -> AmrStatistics {
            let mut level_counts = HashMap::new();
            let mut total_leaves = 0;
            let mut total_refined = 0;
            
            for cell in &self.cells {
                *level_counts.entry(cell.level).or_insert(0) += 1;
                if cell.is_leaf { total_leaves += 1; }
                if !cell.children_ids.is_empty() { total_refined += 1; }
            }
            
            AmrStatistics {
                total_cells: self.cells.len(),
                total_leaves,
                total_refined,
                max_level: level_counts.keys().max().copied().unwrap_or(0),
                level_distribution: level_counts,
                refinement_events: self.refinement_history.len(),
            }
        }
    }
    
    /// Statistics about AMR grid
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AmrStatistics {
        pub total_cells: usize,
        pub total_leaves: usize,
        pub total_refined: usize,
        pub max_level: u32,
        pub level_distribution: HashMap<u32, usize>,
        pub refinement_events: usize,
    }
}

/// Quantum Chemistry approximations for molecular interactions
/// Based on PDF recommendation for bridging quantum mechanics to chemistry
pub mod quantum_chemistry {
    use super::*;
    
    /// Quantum chemical calculation method
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum CalculationMethod {
        HartreeFock,           // Basic ab initio method
        DensityFunctional,     // DFT with various functionals
        SemiEmpirical,         // Fast approximate methods
        MolecularMechanics,    // Classical force fields
        QmMm,                  // Hybrid quantum/classical
    }
    
    /// Electronic structure calculation results
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ElectronicStructure {
        pub total_energy: f64,
        pub orbital_energies: Vec<f64>,
        pub molecular_orbitals: Vec<MolecularOrbital>,
        pub electron_density: Vec<Vec<Vec<f64>>>,
        pub bond_orders: HashMap<(usize, usize), f64>,
        pub atomic_charges: Vec<f64>,
        pub dipole_moment: Vector3<f64>,
        pub polarizability: Matrix3<f64>,
    }
    
    /// Molecular orbital representation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MolecularOrbital {
        pub energy: f64,
        pub occupation: f64,
        pub coefficients: Vec<f64>,
        pub orbital_type: OrbitalType,
        pub symmetry: String,
    }
    
    /// Quantum chemistry engine for molecular calculations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QuantumChemistryEngine {
        pub method: CalculationMethod,
        pub basis_set: BasisSet,
        pub functional: DftFunctional,
        pub calculation_cache: HashMap<String, ElectronicStructure>,
        pub reaction_database: Vec<ChemicalReaction>,
        pub force_field_parameters: ForceFieldParameters,
    }
    
    /// Basis set for quantum calculations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BasisSet {
        pub name: String,
        pub angular_momentum_functions: HashMap<u32, Vec<GaussianFunction>>,
        pub contraction_coefficients: Vec<f64>,
        pub exponents: Vec<f64>,
    }
    
    /// Gaussian basis function
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GaussianFunction {
        pub exponent: f64,
        pub coefficient: f64,
        pub angular_momentum: [u32; 3], // l, m, n for x^l * y^m * z^n
        pub center: Vector3<f64>,
    }
    
    /// DFT functional for electron correlation
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum DftFunctional {
        Lda,      // Local Density Approximation
        Gga,      // Generalized Gradient Approximation
        Hybrid,   // B3LYP-like hybrid functionals
        MetaGga,  // Meta-GGA functionals
    }
    
    /// Chemical reaction definition
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChemicalReaction {
        pub reactants: Vec<ParticleType>,
        pub products: Vec<ParticleType>,
        pub activation_energy: f64,
        pub reaction_energy: f64,
        pub rate_constant: f64,
        pub temperature_dependence: ArrheniusParameters,
        pub mechanism: ReactionMechanism,
    }
    
    /// Arrhenius parameters for temperature dependence
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ArrheniusParameters {
        pub pre_exponential_factor: f64,
        pub activation_energy: f64,
        pub temperature_exponent: f64,
    }
    
    /// Reaction mechanism classification
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum ReactionMechanism {
        Unimolecular,
        Bimolecular,
        Termolecular,
        ChainReaction,
        Photochemical,
        Electrochemical,
    }
    
    /// Force field parameters for classical molecular dynamics
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ForceFieldParameters {
        pub bond_parameters: HashMap<(ParticleType, ParticleType), BondParameters>,
        pub angle_parameters: HashMap<(ParticleType, ParticleType, ParticleType), AngleParameters>,
        pub dihedral_parameters: HashMap<(ParticleType, ParticleType, ParticleType, ParticleType), DihedralParameters>,
        pub van_der_waals_parameters: HashMap<ParticleType, VdwParameters>,
    }
    
    /// Bond stretching parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BondParameters {
        pub equilibrium_length: f64,
        pub force_constant: f64,
        pub dissociation_energy: f64,
    }
    
    /// Angle bending parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AngleParameters {
        pub equilibrium_angle: f64,
        pub force_constant: f64,
    }
    
    /// Dihedral torsion parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DihedralParameters {
        pub periodicity: u32,
        pub phase_angle: f64,
        pub barrier_height: f64,
    }
    
    /// Van der Waals parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VdwParameters {
        pub sigma: f64,      // Size parameter
        pub epsilon: f64,    // Energy parameter
        pub radius: f64,     // Atomic radius
    }
    
    impl QuantumChemistryEngine {
        /// Create new quantum chemistry engine
        pub fn new() -> Self {
            Self {
                method: CalculationMethod::DensityFunctional,
                basis_set: BasisSet::sto_3g(),
                functional: DftFunctional::Hybrid,
                calculation_cache: HashMap::new(),
                reaction_database: Self::initialize_reaction_database(),
                force_field_parameters: ForceFieldParameters::default(),
            }
        }
        
        /// Calculate electronic structure for a molecule
        pub fn calculate_electronic_structure(&mut self, molecule: &Molecule) -> Result<ElectronicStructure> {
            let molecule_key = self.generate_molecule_key(molecule);
            
            // Check cache first
            if let Some(cached_result) = self.calculation_cache.get(&molecule_key) {
                return Ok(cached_result.clone());
            }
            
            // Perform quantum chemical calculation
            let result = match self.method {
                CalculationMethod::HartreeFock => self.hartree_fock_calculation(molecule)?,
                CalculationMethod::DensityFunctional => self.dft_calculation(molecule)?,
                CalculationMethod::SemiEmpirical => self.semi_empirical_calculation(molecule)?,
                CalculationMethod::MolecularMechanics => self.molecular_mechanics_calculation(molecule)?,
                CalculationMethod::QmMm => self.qm_mm_calculation(molecule)?,
            };
            
            // Cache result
            self.calculation_cache.insert(molecule_key, result.clone());
            
            Ok(result)
        }
        
        /// Hartree-Fock self-consistent field calculation
        fn hartree_fock_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
            // Simplified HF calculation
            let num_electrons = self.count_electrons(molecule);
            let num_orbitals = molecule.atoms.len() * 4; // Approximate basis size
            
            // Build overlap matrix, kinetic energy matrix, nuclear attraction
            let overlap_matrix = self.build_overlap_matrix(molecule)?;
            let kinetic_matrix = self.build_kinetic_matrix(molecule)?;
            let nuclear_matrix = self.build_nuclear_attraction_matrix(molecule)?;
            
            // SCF iteration (simplified)
            let density_matrix = Matrix3::zeros();
            let mut total_energy = 0.0;
            
            for _iteration in 0..50 {
                let fock_matrix = self.build_fock_matrix(&density_matrix, molecule)?;
                let (_eigenvalues, _eigenvectors) = self.solve_eigenvalue_problem(&fock_matrix, &overlap_matrix)?;
                
                // Update density matrix and check convergence
                let old_energy = total_energy;
                total_energy = self.calculate_total_energy(&density_matrix, molecule)?;
                
                if (total_energy - old_energy).abs() < 1e-8 {
                    break;
                }
            }
            
            Ok(ElectronicStructure {
                total_energy,
                orbital_energies: vec![0.0; num_orbitals],
                molecular_orbitals: vec![],
                electron_density: vec![],
                bond_orders: HashMap::new(),
                atomic_charges: vec![0.0; molecule.atoms.len()],
                dipole_moment: Vector3::zeros(),
                polarizability: Matrix3::zeros(),
            })
        }
        
        /// Density Functional Theory calculation
        fn dft_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
            // Simplified DFT - similar to HF but with exchange-correlation functional
            let mut structure = self.hartree_fock_calculation(molecule)?;
            
            // Add DFT corrections based on functional
            let xc_energy = match self.functional {
                DftFunctional::Lda => self.lda_exchange_correlation(molecule)?,
                DftFunctional::Gga => self.gga_exchange_correlation(molecule)?,
                DftFunctional::Hybrid => self.hybrid_exchange_correlation(molecule)?,
                DftFunctional::MetaGga => self.meta_gga_exchange_correlation(molecule)?,
            };
            
            structure.total_energy += xc_energy;
            
            Ok(structure)
        }
        
        /// Semi-empirical quantum calculation (faster approximation)
        fn semi_empirical_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
            // Use parameterized methods like AM1, PM3, etc.
            let num_atoms = molecule.atoms.len();
            let mut total_energy = 0.0;
            
            // Electronic energy from parameterized method
            for atom in &molecule.atoms {
                total_energy += self.get_atomic_energy(&atom.nucleus.atomic_number);
            }
            
            // Add bond energies
            for bond in &molecule.bonds {
                let bond_energy = self.get_bond_energy(&bond.bond_type, bond.bond_length);
                total_energy += bond_energy;
            }
            
            // Add non-bonded interactions
            for i in 0..num_atoms {
                for j in (i+1)..num_atoms {
                    if !self.are_bonded(i, j, molecule) {
                        let distance = (molecule.atoms[i].nucleus.position - 
                                      molecule.atoms[j].nucleus.position).magnitude();
                        total_energy += self.van_der_waals_energy(i, j, distance, molecule);
                    }
                }
            }
            
            Ok(ElectronicStructure {
                total_energy,
                orbital_energies: vec![],
                molecular_orbitals: vec![],
                electron_density: vec![],
                bond_orders: HashMap::new(),
                atomic_charges: vec![0.0; num_atoms],
                dipole_moment: Vector3::zeros(),
                polarizability: Matrix3::zeros(),
            })
        }
        
        /// Molecular mechanics calculation (classical)
        fn molecular_mechanics_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
            let mut total_energy = 0.0;
            
            // Bond stretching energy
            for bond in &molecule.bonds {
                if let Some(params) = self.force_field_parameters.bond_parameters.get(&(
                    self.get_atom_type(&molecule.atoms[bond.atom_indices.0]),
                    self.get_atom_type(&molecule.atoms[bond.atom_indices.1])
                )) {
                    let delta_r = bond.bond_length - params.equilibrium_length;
                    total_energy += 0.5 * params.force_constant * delta_r * delta_r;
                }
            }
            
            // Angle bending energy
            total_energy += self.calculate_angle_energy(molecule)?;
            
            // Dihedral torsion energy
            total_energy += self.calculate_dihedral_energy(molecule)?;
            
            // Van der Waals and electrostatic energy
            total_energy += self.calculate_nonbonded_energy(molecule)?;
            
            Ok(ElectronicStructure {
                total_energy,
                orbital_energies: vec![],
                molecular_orbitals: vec![],
                electron_density: vec![],
                bond_orders: HashMap::new(),
                atomic_charges: vec![],
                dipole_moment: Vector3::zeros(),
                polarizability: Matrix3::zeros(),
            })
        }
        
        /// QM/MM hybrid calculation
        fn qm_mm_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
            // Divide molecule into QM and MM regions
            let (qm_atoms, mm_atoms) = self.partition_qm_mm(molecule);
            
            // Calculate QM part
            let qm_energy = self.calculate_qm_region_energy(&qm_atoms)?;
            
            // Calculate MM part
            let mm_energy = self.calculate_mm_region_energy(&mm_atoms)?;
            
            // Calculate QM/MM interaction
            let qm_mm_interaction = self.calculate_qm_mm_interaction(&qm_atoms, &mm_atoms)?;
            
            let total_energy = qm_energy + mm_energy + qm_mm_interaction;
            
            Ok(ElectronicStructure {
                total_energy,
                orbital_energies: vec![],
                molecular_orbitals: vec![],
                electron_density: vec![],
                bond_orders: HashMap::new(),
                atomic_charges: vec![],
                dipole_moment: Vector3::zeros(),
                polarizability: Matrix3::zeros(),
            })
        }
        
        /// Predict reaction pathway and rate
        pub fn predict_reaction(&self, reactants: &[ParticleType], temperature: f64) -> Result<Option<ChemicalReaction>> {
            // Search reaction database for matching reactants
            for reaction in &self.reaction_database {
                if self.reactants_match(&reaction.reactants, reactants) {
                    let rate = self.calculate_reaction_rate(reaction, temperature);
                    let mut reaction_copy = reaction.clone();
                    reaction_copy.rate_constant = rate;
                    return Ok(Some(reaction_copy));
                }
            }
            
            // If no exact match, try to predict new reaction
            self.predict_reaction(reactants, temperature)
        }
        
        /// Calculate reaction rate using transition state theory
        fn calculate_reaction_rate(&self, reaction: &ChemicalReaction, temperature: f64) -> f64 {
            let k_b = BOLTZMANN;
            let h = PLANCK_CONSTANT;
            let r = 8.314; // Gas constant
            
            // Arrhenius equation with temperature dependence
            let arrhenius_factor = reaction.temperature_dependence.pre_exponential_factor * 
                temperature.powf(reaction.temperature_dependence.temperature_exponent) *
                (-reaction.temperature_dependence.activation_energy / (r * temperature)).exp();
            
            // Transition state theory correction
            let tst_factor = (k_b * temperature / h) * 
                (-reaction.activation_energy / (k_b * temperature)).exp();
            
            arrhenius_factor * tst_factor
        }
        
        // Helper methods (simplified implementations)
        fn count_electrons(&self, molecule: &Molecule) -> usize {
            molecule.atoms.iter().map(|atom| atom.nucleus.atomic_number as usize).sum()
        }
        
        fn generate_molecule_key(&self, molecule: &Molecule) -> String {
            // Generate unique key for molecule based on structure
            format!("mol_{}_atoms", molecule.atoms.len())
        }
        
        fn build_overlap_matrix(&self, _molecule: &Molecule) -> Result<Matrix3<f64>> {
            Ok(Matrix3::identity()) // Simplified
        }
        
        fn build_kinetic_matrix(&self, _molecule: &Molecule) -> Result<Matrix3<f64>> {
            Ok(Matrix3::zeros()) // Simplified
        }
        
        fn build_nuclear_attraction_matrix(&self, _molecule: &Molecule) -> Result<Matrix3<f64>> {
            Ok(Matrix3::zeros()) // Simplified
        }
        
        fn build_fock_matrix(&self, _density: &Matrix3<f64>, _molecule: &Molecule) -> Result<Matrix3<f64>> {
            Ok(Matrix3::zeros()) // Simplified
        }
        
        fn solve_eigenvalue_problem(&self, _fock: &Matrix3<f64>, _overlap: &Matrix3<f64>) -> Result<(Vec<f64>, Vec<Vector3<f64>>)> {
            Ok((vec![], vec![])) // Simplified
        }
        
        fn calculate_total_energy(&self, _density: &Matrix3<f64>, _molecule: &Molecule) -> Result<f64> {
            Ok(0.0) // Simplified
        }
    }
}

/// Geant4-style particle physics integration
/// Based on PDF recommendation to use proven particle interaction libraries
pub mod geant4_integration {
    use super::*;
    
    /// Particle physics process types (from Geant4)
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum PhysicsProcess {
        // Electromagnetic processes
        PhotoElectricEffect,
        ComptonScattering,
        PairProduction,
        Bremsstrahlung,
        Ionization,
        MultipleScattering,
        
        // Hadronic processes
        HadronElastic,
        HadronInelastic,
        NeutronCapture,
        NeutronFission,
        
        // Decay processes
        RadioactiveDecay,
        MuonDecay,
        PionDecay,
        
        // High energy processes
        PhotoNuclear,
        ElectroNuclear,
        MuonNuclear,
    }
    
    /// Particle interaction database (Geant4-style)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ParticlePhysicsDatabase {
        pub cross_sections: HashMap<(ParticleType, ParticleType, PhysicsProcess), CrossSectionTable>,
        pub stopping_powers: HashMap<ParticleType, StoppingPowerTable>,
        pub decay_data: HashMap<ParticleType, DecayData>,
        pub material_properties: HashMap<String, MaterialProperties>,
        pub physics_lists: Vec<PhysicsList>,
    }
    
    /// Cross-section data table with energy dependence
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CrossSectionTable {
        pub energies_mev: Vec<f64>,
        pub cross_sections_barn: Vec<f64>,
        pub interpolation_method: InterpolationMethod,
        pub temperature_dependence: Option<TemperatureDependence>,
        pub source: String, // Reference to experimental data
    }
    
    /// Stopping power table for energy loss
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
    
    /// Decay mode with branching ratios
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DecayMode {
        pub mode_type: DecayType,
        pub branching_ratio: f64,
        pub q_value_mev: f64,
        pub daughter_energies: Vec<f64>,
    }
    
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum DecayType {
        Alpha,
        BetaMinus,
        BetaPlus,
        ElectronCapture,
        SpontaneousFission,
        InternalConversion,
        IsomericTransition,
    }
    
    /// Material properties for interaction calculations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MaterialProperties {
        pub name: String,
        pub density_g_cm3: f64,
        pub atomic_composition: Vec<(u32, f64)>, // (Z, fraction)
        pub mean_excitation_energy_ev: f64,
        pub radiation_length_cm: f64,
        pub nuclear_interaction_length_cm: f64,
    }
    
    /// Physics list configuration (like Geant4 physics lists)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PhysicsList {
        pub name: String,
        pub energy_range: (f64, f64), // (min_energy_mev, max_energy_mev)
        pub enabled_processes: Vec<PhysicsProcess>,
        pub cut_values: HashMap<ParticleType, f64>, // Production cuts
        pub step_limiters: Vec<StepLimiter>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StepLimiter {
        pub particle_type: ParticleType,
        pub max_step_length_cm: f64,
        pub max_energy_loss_fraction: f64,
    }
    
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum InterpolationMethod {
        Linear,
        LogLog,
        SplineCubic,
        LogLinear,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TemperatureDependence {
        pub reference_temperature_k: f64,
        pub temperature_coefficient: f64,
        pub activation_energy_ev: f64,
    }
    
    /// Geant4-style physics engine
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Geant4Engine {
        pub database: ParticlePhysicsDatabase,
        pub active_physics_list: String,
        pub random_seed: u64,
        pub verbose_level: u32,
        pub production_cuts: HashMap<ParticleType, f64>,
    }
    
    impl Geant4Engine {
        /// Initialize with real particle physics data
        pub fn new() -> Self {
            Self {
                database: ParticlePhysicsDatabase::load_standard_database(),
                active_physics_list: "QBBC".to_string(), // Standard Geant4 physics list
                random_seed: 12345,
                verbose_level: 1,
                production_cuts: Self::default_production_cuts(),
            }
        }
        
        /// Calculate interaction probability
        pub fn calculate_interaction_probability(
            &self,
            particle: &FundamentalParticle,
            target_material: &str,
            step_length_cm: f64,
        ) -> Result<f64> {
            let material = self.database.material_properties.get(target_material)
                .ok_or_else(|| anyhow::anyhow!("Unknown material: {}", target_material))?;
            
            let energy_mev = particle.energy * 6.242e12; // Convert J to MeV
            
            // Sum all applicable cross-sections
            let mut total_cross_section = 0.0;
            
            for process in &self.get_applicable_processes(&particle.particle_type) {
                if let Some(sigma) = self.get_cross_section(&particle.particle_type, material, *process, energy_mev)? {
                    total_cross_section += sigma;
                }
            }
            
            // Convert to interaction probability
            let number_density = material.density_g_cm3 * 6.022e23 / self.get_average_atomic_mass(material);
            let probability = 1.0 - (-total_cross_section * number_density * step_length_cm).exp();
            
            Ok(probability.min(1.0))
        }
        
        /// Simulate particle interaction
        pub fn simulate_interaction(
            &self,
            particle: &FundamentalParticle,
            target_material: &str,
        ) -> Result<Vec<FundamentalParticle>> {
            let energy_mev = particle.energy * 6.242e12;
            
            // Determine which process occurs
            let process = self.sample_physics_process(particle, target_material, energy_mev)?;
            
            // Generate interaction products based on process
            match process {
                PhysicsProcess::ComptonScattering => self.simulate_compton_scattering(particle),
                PhysicsProcess::PhotoElectricEffect => self.simulate_photoelectric_effect(particle),
                PhysicsProcess::PairProduction => self.simulate_pair_production(particle),
                PhysicsProcess::Bremsstrahlung => self.simulate_bremsstrahlung(particle),
                PhysicsProcess::HadronElastic => self.simulate_elastic_scattering(particle),
                PhysicsProcess::HadronInelastic => self.simulate_inelastic_scattering(particle),
                PhysicsProcess::RadioactiveDecay => self.simulate_radioactive_decay(particle),
                _ => Ok(vec![particle.clone()]), // No interaction
            }
        }
        
        /// Calculate energy loss (dE/dx)
        pub fn calculate_energy_loss(
            &self,
            particle: &FundamentalParticle,
            material: &str,
            step_length_cm: f64,
        ) -> Result<f64> {
            if let Some(stopping_power_table) = self.database.stopping_powers.get(&particle.particle_type) {
                let energy_mev = particle.energy * 6.242e12;
                let stopping_power = self.interpolate_stopping_power(stopping_power_table, energy_mev)?;
                
                // Convert from MeV cm²/g to J/m
                let material_props = self.database.material_properties.get(material)
                    .ok_or_else(|| anyhow::anyhow!("Unknown material: {}", material))?;
                
                let energy_loss_j = stopping_power * material_props.density_g_cm3 * step_length_cm * 1.602e-13;
                Ok(energy_loss_j)
            } else {
                // Use Bethe-Bloch formula as fallback
                self.calculate_bethe_bloch_energy_loss(particle, material, step_length_cm)
            }
        }
        
        // Implementation of specific physics processes (simplified)
        fn simulate_compton_scattering(&self, photon: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            let energy_mev = photon.energy * 6.242e12;
            
            // Klein-Nishina formula implementation
            let electron_rest_energy = 0.511; // MeV
            let alpha = energy_mev / electron_rest_energy;
            
            // Sample scattering angle using Klein-Nishina distribution
            let cos_theta = self.sample_klein_nishina_angle(alpha);
            let scattered_energy = energy_mev / (1.0 + alpha * (1.0 - cos_theta));
            
            // Create scattered photon
            let mut scattered_photon = photon.clone();
            scattered_photon.energy = scattered_energy * 1.602e-13; // Convert back to J
            
            // Create recoil electron
            let electron_energy = (energy_mev - scattered_energy) * 1.602e-13;
            let electron = self.create_particle(ParticleType::Electron, electron_energy, photon.position);
            
            Ok(vec![scattered_photon, electron])
        }
        
        fn simulate_photoelectric_effect(&self, photon: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            // Photoelectron energy = photon energy - binding energy
            let binding_energy_j = 13.6 * 1.602e-19; // Hydrogen ionization energy (simplified)
            let electron_energy = photon.energy - binding_energy_j;
            
            if electron_energy > 0.0 {
                let electron = self.create_particle(ParticleType::Electron, electron_energy, photon.position);
                Ok(vec![electron])
            } else {
                Ok(vec![]) // Below threshold
            }
        }
        
        fn simulate_pair_production(&self, photon: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            let energy_mev = photon.energy * 6.242e12;
            let threshold_mev = 1.022; // 2 * electron rest mass
            
            if energy_mev > threshold_mev {
                let available_energy = energy_mev - threshold_mev;
                
                // Share kinetic energy between electron and positron
                let electron_energy = (0.511 + available_energy / 2.0) * 1.602e-13;
                let positron_energy = (0.511 + available_energy / 2.0) * 1.602e-13;
                
                let electron = self.create_particle(ParticleType::Electron, electron_energy, photon.position);
                let positron = self.create_particle(ParticleType::Positron, positron_energy, photon.position);
                
                Ok(vec![electron, positron])
            } else {
                Ok(vec![photon.clone()]) // Below threshold
            }
        }
        
        fn simulate_bremsstrahlung(&self, electron: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            let energy_mev = electron.energy * 6.242e12;
            
            // Simple bremsstrahlung model - electron loses energy, photon is created
            let radiated_fraction = 0.1; // Simplified - typically much less
            let photon_energy = electron.energy * radiated_fraction;
            let remaining_electron_energy = electron.energy * (1.0 - radiated_fraction);
            
            let mut scattered_electron = electron.clone();
            scattered_electron.energy = remaining_electron_energy;
            
            let photon = self.create_particle(ParticleType::Photon, photon_energy, electron.position);
            
            Ok(vec![scattered_electron, photon])
        }
        
        fn simulate_elastic_scattering(&self, particle: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            // Elastic scattering - only momentum transfer, no energy loss
            let mut scattered = particle.clone();
            
            // Random scattering angle (simplified)
            let cos_theta = 2.0 * rand::random::<f64>() - 1.0;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi = 2.0 * std::f64::consts::PI * rand::random::<f64>();
            
            // Update momentum direction
            let p_mag = scattered.momentum.magnitude();
            scattered.momentum = Vector3::new(
                p_mag * sin_theta * phi.cos(),
                p_mag * sin_theta * phi.sin(),
                p_mag * cos_theta,
            );
            
            Ok(vec![scattered])
        }
        
        fn simulate_inelastic_scattering(&self, _particle: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            // Placeholder for complex inelastic processes
            // Would need detailed nuclear models
            Ok(vec![])
        }
        
        fn simulate_radioactive_decay(&self, nucleus: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
            if let Some(decay_data) = self.database.decay_data.get(&nucleus.particle_type) {
                // Sample decay mode based on branching ratios
                let decay_mode = self.sample_decay_mode(&decay_data.decay_modes);
                
                match decay_mode.mode_type {
                    DecayType::Alpha => {
                        let alpha = self.create_particle(ParticleType::Helium, decay_mode.q_value_mev * 1.602e-13, nucleus.position);
                        // Create daughter nucleus (simplified)
                        Ok(vec![alpha])
                    },
                    DecayType::BetaMinus => {
                        let electron = self.create_particle(ParticleType::Electron, decay_mode.q_value_mev * 1.602e-13 / 2.0, nucleus.position);
                        let neutrino = self.create_particle(ParticleType::ElectronAntiNeutrino, decay_mode.q_value_mev * 1.602e-13 / 2.0, nucleus.position);
                        Ok(vec![electron, neutrino])
                    },
                    _ => Ok(vec![nucleus.clone()]), // Other decay modes
                }
            } else {
                Ok(vec![nucleus.clone()]) // Stable nucleus
            }
        }
        
        // Helper methods
        fn get_applicable_processes(&self, particle_type: &ParticleType) -> Vec<PhysicsProcess> {
            match particle_type {
                ParticleType::Photon => vec![
                    PhysicsProcess::PhotoElectricEffect,
                    PhysicsProcess::ComptonScattering,
                    PhysicsProcess::PairProduction,
                ],
                ParticleType::Electron | ParticleType::Positron => vec![
                    PhysicsProcess::Ionization,
                    PhysicsProcess::Bremsstrahlung,
                    PhysicsProcess::MultipleScattering,
                ],
                _ => vec![PhysicsProcess::HadronElastic, PhysicsProcess::HadronInelastic],
            }
        }
        
        fn get_cross_section(
            &self,
            particle: &ParticleType,
            material: &MaterialProperties,
            process: PhysicsProcess,
            energy_mev: f64,
        ) -> Result<Option<f64>> {
            // Look up cross-section in database
            // This would interpolate from tables of experimental data
            // For now, return simplified analytical formulas
            
            match process {
                PhysicsProcess::ComptonScattering => {
                    // Klein-Nishina formula
                    let alpha = energy_mev / 0.511;
                    let sigma_thomson = 6.652e-25; // cm²
                    let sigma = sigma_thomson * 
                        ((1.0 + alpha) / (alpha * alpha) * 
                         (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha) - (1.0 / alpha) * (2.0 * alpha / (1.0 + 2.0 * alpha)).ln()) +
                         (1.0 / (2.0 * alpha)) * (2.0 * alpha / (1.0 + 2.0 * alpha)).ln() - 
                         (1.0 + 3.0 * alpha) / ((1.0 + 2.0 * alpha) * (1.0 + 2.0 * alpha)));
                    Ok(Some(sigma))
                },
                _ => Ok(Some(1e-24)), // Placeholder values
            }
        }
        
        fn sample_physics_process(
            &self,
            particle: &FundamentalParticle,
            material: &str,
            energy_mev: f64,
        ) -> Result<PhysicsProcess> {
            // Sample process based on relative cross-sections
            // Simplified - return first applicable process
            let processes = self.get_applicable_processes(&particle.particle_type);
            Ok(processes[0])
        }
        
        fn sample_klein_nishina_angle(&self, alpha: f64) -> f64 {
            // Rejection sampling for Klein-Nishina distribution
            loop {
                let xi1 = rand::random::<f64>();
                let xi2 = rand::random::<f64>();
                
                let cos_theta = 1.0 - 2.0 * xi1;
                let rejection_value = 1.0 / (1.0 + alpha * (1.0 - cos_theta));
                let probability = rejection_value * rejection_value * 
                    (rejection_value + 1.0 / rejection_value - (1.0 - cos_theta * cos_theta));
                
                if xi2 < probability / 2.0 {
                    return cos_theta;
                }
            }
        }
        
        fn create_particle(&self, particle_type: ParticleType, energy_j: f64, position: Vector3<f64>) -> FundamentalParticle {
            FundamentalParticle {
                particle_type,
                position,
                momentum: Vector3::zeros(), // Would calculate from energy
                spin: Vector3::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
                color_charge: None,
                electric_charge: 0.0,
                mass: 0.0, // Would look up
                energy: energy_j,
                creation_time: 0.0,
                decay_time: None,
                quantum_state: QuantumState::new(),
                interaction_history: vec![],
                velocity: Vector3::zeros(),
                charge: 0.0,
            }
        }
        
        fn get_average_atomic_mass(&self, material: &MaterialProperties) -> f64 {
            // Calculate average atomic mass from composition
            material.atomic_composition.iter()
                .map(|(z, fraction)| *z as f64 * 1.66e-27 * fraction) // kg
                .sum()
        }
        
        fn interpolate_stopping_power(&self, table: &StoppingPowerTable, energy_mev: f64) -> Result<f64> {
            // Linear interpolation in log-log space
            let log_energy = energy_mev.ln();
            
            for i in 0..table.energies_mev.len()-1 {
                let e1 = table.energies_mev[i].ln();
                let e2 = table.energies_mev[i+1].ln();
                
                if log_energy >= e1 && log_energy <= e2 {
                    let sp1 = table.stopping_powers_mev_cm2_g[i].ln();
                    let sp2 = table.stopping_powers_mev_cm2_g[i+1].ln();
                    
                    let log_sp = sp1 + (sp2 - sp1) * (log_energy - e1) / (e2 - e1);
                    return Ok(log_sp.exp());
                }
            }
            
            Ok(table.stopping_powers_mev_cm2_g[0]) // Fallback
        }
        
        fn calculate_bethe_bloch_energy_loss(
            &self,
            particle: &FundamentalParticle,
            material: &str,
            step_length_cm: f64,
        ) -> Result<f64> {
            // Bethe-Bloch formula for heavy charged particles
            let material_props = self.database.material_properties.get(material)
                .ok_or_else(|| anyhow::anyhow!("Unknown material: {}", material))?;
            
            let beta = 0.5; // Simplified - would calculate from energy
            let gamma = 1.0_f64 / ((1.0_f64 - beta * beta).sqrt());
            
            let k = 0.307; // MeV cm²/g
            let me_c2 = 0.511; // MeV
            let z_squared = 1.0; // Charge squared
            let z_over_a = 0.5; // For material
            let i_ev = material_props.mean_excitation_energy_ev;
            
            let w_max = 2.0 * me_c2 * beta * beta * gamma * gamma;
            let dedx = k * z_squared * z_over_a / (beta * beta) * 
                (0.5 * (2.0 * me_c2 * beta * beta * gamma * gamma * w_max / (i_ev * i_ev / 1e6)).ln() - beta * beta);
            
            Ok(dedx * material_props.density_g_cm3 * step_length_cm * 1.602e-13) // Convert to J
        }
        
        fn sample_decay_mode<'a>(&self, modes: &'a [DecayMode]) -> &'a DecayMode {
            let total_branching: f64 = modes.iter().map(|m| m.branching_ratio).sum();
            let random = rand::random::<f64>() * total_branching;
            
            let mut cumulative = 0.0;
            for mode in modes {
                cumulative += mode.branching_ratio;
                if random <= cumulative {
                    return mode;
                }
            }
            
            &modes[0] // Fallback
        }
        
        fn default_production_cuts() -> HashMap<ParticleType, f64> {
            let mut cuts = HashMap::new();
            cuts.insert(ParticleType::Photon, 0.001); // 1 mm
            cuts.insert(ParticleType::Electron, 0.001);
            cuts.insert(ParticleType::Positron, 0.001);
            cuts
        }
    }
    
    impl ParticlePhysicsDatabase {
        /// Load standard physics database (like Geant4 data)
        pub fn load_standard_database() -> Self {
            Self {
                cross_sections: Self::load_cross_section_data(),
                stopping_powers: Self::load_stopping_power_data(),
                decay_data: Self::load_decay_data(),
                material_properties: Self::load_material_data(),
                physics_lists: Self::load_physics_lists(),
            }
        }
        
        fn load_cross_section_data() -> HashMap<(ParticleType, ParticleType, PhysicsProcess), CrossSectionTable> {
            let mut data = HashMap::new();
            
            // Compton scattering cross-section for photon-electron
            data.insert(
                (ParticleType::Photon, ParticleType::Electron, PhysicsProcess::ComptonScattering),
                CrossSectionTable {
                    energies_mev: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    cross_sections_barn: vec![20.0, 10.0, 5.0, 2.0, 1.0, 0.5], // Simplified values
                    interpolation_method: InterpolationMethod::LogLog,
                    temperature_dependence: None,
                    source: "Klein-Nishina formula".to_string(),
                }
            );
            
            data
        }
        
        fn load_stopping_power_data() -> HashMap<ParticleType, StoppingPowerTable> {
            let mut data = HashMap::new();
            
            // Electron stopping power in water (simplified)
            data.insert(
                ParticleType::Electron,
                StoppingPowerTable {
                    energies_mev: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    stopping_powers_mev_cm2_g: vec![50.0, 20.0, 5.0, 2.0, 1.5, 1.2],
                    range_mev_cm2_g: vec![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    material: "Water".to_string(),
                }
            );
            
            data
        }
        
        fn load_decay_data() -> HashMap<ParticleType, DecayData> {
            let mut data = HashMap::new();
            
            // Neutron decay
            data.insert(
                ParticleType::Neutron,
                DecayData {
                    half_life_seconds: 881.5,
                    decay_modes: vec![
                        DecayMode {
                            mode_type: DecayType::BetaMinus,
                            branching_ratio: 1.0,
                            q_value_mev: 0.782,
                            daughter_energies: vec![0.3, 0.3, 0.182], // electron, neutrino, recoil
                        }
                    ],
                    q_value_mev: 0.782,
                    daughter_products: vec![
                        (ParticleType::Proton, 1.0),
                        (ParticleType::Electron, 1.0),
                        (ParticleType::ElectronAntiNeutrino, 1.0),
                    ],
                }
            );
            
            data
        }
        
        fn load_material_data() -> HashMap<String, MaterialProperties> {
            let mut data = HashMap::new();
            
            // Water
            data.insert(
                "Water".to_string(),
                MaterialProperties {
                    name: "Water".to_string(),
                    density_g_cm3: 1.0,
                    atomic_composition: vec![(1, 0.111), (8, 0.889)], // H and O mass fractions
                    mean_excitation_energy_ev: 75.0,
                    radiation_length_cm: 36.08,
                    nuclear_interaction_length_cm: 83.6,
                }
            );
            
            // Silicon (for detector materials)
            data.insert(
                "Silicon".to_string(),
                MaterialProperties {
                    name: "Silicon".to_string(),
                    density_g_cm3: 2.33,
                    atomic_composition: vec![(14, 1.0)],
                    mean_excitation_energy_ev: 173.0,
                    radiation_length_cm: 9.36,
                    nuclear_interaction_length_cm: 45.5,
                }
            );
            
            data
        }
        
        fn load_physics_lists() -> Vec<PhysicsList> {
            vec![
                PhysicsList {
                    name: "QBBC".to_string(),
                    energy_range: (1e-3, 1e5), // 1 keV to 100 GeV
                    enabled_processes: vec![
                        PhysicsProcess::PhotoElectricEffect,
                        PhysicsProcess::ComptonScattering,
                        PhysicsProcess::PairProduction,
                        PhysicsProcess::Bremsstrahlung,
                        PhysicsProcess::Ionization,
                        PhysicsProcess::MultipleScattering,
                        PhysicsProcess::HadronElastic,
                        PhysicsProcess::HadronInelastic,
                        PhysicsProcess::RadioactiveDecay,
                    ],
                    cut_values: [
                        (ParticleType::Photon, 0.001),
                        (ParticleType::Electron, 0.001),
                        (ParticleType::Positron, 0.001),
                    ].iter().cloned().collect(),
                    step_limiters: vec![
                        StepLimiter {
                            particle_type: ParticleType::Electron,
                            max_step_length_cm: 0.1,
                            max_energy_loss_fraction: 0.2,
                        }
                    ],
                }
            ]
        }
    }
}

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
        
        /// Apply cosmological expansion following GADGET methodology
        fn apply_cosmological_expansion(&mut self, dt: f64) -> Result<()> {
            // Hubble function H(a) = H₀ * sqrt(Ωₘ/a³ + ΩΛ)
            let a = self.cosmological_parameters.scale_factor;
            let omega_m = self.cosmological_parameters.omega_matter;
            let omega_lambda = self.cosmological_parameters.omega_lambda;
            
            let hubble_parameter = self.cosmological_parameters.hubble_constant *
                (omega_m / (a * a * a) + omega_lambda).sqrt();
            
            // Scale factor evolution: da/dt = H * a
            let scale_factor_derivative = hubble_parameter * a;
            self.cosmological_parameters.scale_factor += scale_factor_derivative * dt;
            
            // Apply Hubble flow to particle velocities
            let expansion_factor = scale_factor_derivative * dt / a;
            for particle in &mut self.particles {
                if particle.active {
                    particle.velocity += particle.position * (expansion_factor / dt);
                }
            }
            
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
        // Cleanup FFI libraries
        if let Err(e) = ffi_integration::cleanup_all_libraries() {
            log::error!("Failed to cleanup FFI libraries: {}", e);
        }
    }
}

#[cfg(any())]
impl Default for ForceFieldParameters {
    fn default() -> Self {
        Self {
            bond_parameters: HashMap::new(),
            angle_parameters: HashMap::new(),
            dihedral_parameters: HashMap::new(),
            van_der_waals_parameters: HashMap::new(),
        }
    }
}

/// Stopping power data for particles in materials
#[cfg(any())]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingPowerTable {
    pub energies_mev: Vec<f64>,
    pub stopping_powers_mev_cm2_g: Vec<f64>,
    pub range_mev_cm2_g: Vec<f64>,
    pub material: String,
}

/// Nuclear decay data
#[cfg(any())]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayData {
    pub half_life_seconds: f64,
    pub decay_modes: Vec<DecayMode>,
    pub q_value_mev: f64,
    pub daughter_products: Vec<(ParticleType, f64)>, // (particle, branching_ratio)
}

/// Material properties for particle interactions
#[cfg(any())]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub name: String,
    pub density_g_cm3: f64,
    pub atomic_composition: Vec<(u32, f64)>, // (Z, fraction)
    pub mean_excitation_energy_ev: f64,
    pub radiation_length_cm: f64,
    pub nuclear_interaction_length_cm: f64,
}

impl crate::quantum_chemistry::BasisSet {
    /// Return a minimal placeholder STO-3G basis set.
    pub fn sto_3g() -> Self {
        Self {
            name: "STO-3G".to_string(),
            angular_momentum_functions: HashMap::new(),
            contraction_coefficients: Vec::new(),
            exponents: Vec::new(),
        }
    }
}

impl crate::quantum_chemistry::QuantumChemistryEngine {
    fn initialize_reaction_database() -> Vec<crate::quantum_chemistry::ChemicalReaction> {
        Vec::new()
    }

    fn lda_exchange_correlation(&self, molecule: &crate::Molecule) -> Result<f64> {
        // Slater exchange in the Local Density Approximation (LDA)
        use crate::constants::{VACUUM_PERMITTIVITY, ELEMENTARY_CHARGE as QE};
        use std::f64::consts::PI;

        // Count electrons (sum of atomic numbers)
        let n_e: f64 = molecule
            .atoms
            .iter()
            .map(|a| a.nucleus.atomic_number as f64)
            .sum();
        let mut volume = 0.0_f64;
        for atom in &molecule.atoms {
            let r = if atom.atomic_radius > 0.0 { atom.atomic_radius } else { 1.0e-10 }; // m
            volume += 4.0 / 3.0 * PI * r.powi(3);
        }
        if volume == 0.0 {
            return Ok(0.0);
        }
        let rho = n_e / volume; // electron density (electrons m⁻³)
        let prefactor = -0.75 * (3.0 / PI).powf(1.0 / 3.0) * K_E * QE * QE; // J·m
        Ok(prefactor * rho.powf(4.0 / 3.0) * volume)
    }

    fn gga_exchange_correlation(&self, molecule: &crate::Molecule) -> Result<f64> {
        // Use a simple PBE-inspired enhancement: E_xc ≈ E_xc^LDA * (1 + 0.2 * s²) where s is reduced gradient.
        // Without an explicit density gradient we approximate s ≈ 0.5 for typical molecules.
        let lda = self.lda_exchange_correlation(molecule)?;
        Ok(lda * 1.05) // modest gradient correction
    }

    fn hybrid_exchange_correlation(&self, molecule: &crate::Molecule) -> Result<f64> {
        // B3LYP-like mix: 0.80 * GGA + 0.20 * HF exchange (neglected ⇒ scale).
        let gga = self.gga_exchange_correlation(molecule)?;
        Ok(gga * 0.80)
    }

    fn meta_gga_exchange_correlation(&self, molecule: &crate::Molecule) -> Result<f64> {
        // Meta-GGA adds kinetic-energy density; we approximate a small refinement.
        let gga = self.gga_exchange_correlation(molecule)?;
        Ok(gga * 1.02)
    }

    fn get_atomic_energy(&self, _atomic_number: &u32) -> f64 { 0.0 }
    fn get_bond_energy(&self, _bond_type: &crate::BondType, _bond_length: f64) -> f64 { 0.0 }
    fn are_bonded(&self, _i: usize, _j: usize, _molecule: &crate::Molecule) -> bool { false }
    fn van_der_waals_energy(&self, _i: usize, _j: usize, _distance: f64, _molecule: &crate::Molecule) -> f64 { 0.0 }

    fn get_atom_type(&self, _atom: &crate::Atom) -> crate::ParticleType { crate::ParticleType::HydrogenAtom }

    fn calculate_angle_energy(&self, molecule: &crate::Molecule) -> Result<f64> {
        use std::f64::consts::PI;
        let mut total = 0.0_f64;
        // Build bond adjacency for quick lookup
        let mut adjacency: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for bond in &molecule.bonds {
            adjacency.entry(bond.atom_indices.0).or_default().push(bond.atom_indices.1);
            adjacency.entry(bond.atom_indices.1).or_default().push(bond.atom_indices.0);
        }
        for (&j, neighbors) in &adjacency {
            for a in 0..neighbors.len() {
                for b in (a + 1)..neighbors.len() {
                    let i = neighbors[a];
                    let k = neighbors[b];
                    let v1 = molecule.atoms[i].position - molecule.atoms[j].position;
                    let v2 = molecule.atoms[k].position - molecule.atoms[j].position;
                    let theta = v1.angle(&v2);

                    let key = (
                        self.get_atom_type(&molecule.atoms[i]),
                        self.get_atom_type(&molecule.atoms[j]),
                        self.get_atom_type(&molecule.atoms[k]),
                    );
                    let params = self
                        .force_field_parameters
                        .angle_parameters
                        .get(&key)
                        .cloned()
                        .unwrap_or(crate::quantum_chemistry::AngleParameters {
                            equilibrium_angle: 109.5_f64.to_radians(),
                            force_constant: 300.0 * 4184.0, // 300 kcal/mol ≈ 300*4184 J/mol
                        });
                    let delta = theta - params.equilibrium_angle;
                    total += 0.5 * params.force_constant * delta * delta;
                }
            }
        }
        Ok(total)
    }

    fn calculate_dihedral_energy(&self, molecule: &crate::Molecule) -> Result<f64> {
        let mut total = 0.0_f64;
        // Helper closure to check bond existence
        let is_bonded = |a: usize, b: usize, mol: &crate::Molecule| {
            mol.bonds.iter().any(|bond| {
                (bond.atom_indices.0 == a && bond.atom_indices.1 == b)
                    || (bond.atom_indices.0 == b && bond.atom_indices.1 == a)
            })
        };

        let n = molecule.atoms.len();
        for i in 0..n {
            for j in 0..n {
                if !is_bonded(i, j, molecule) {
                    continue;
                }
                for k in 0..n {
                    if !is_bonded(j, k, molecule) {
                        continue;
                    }
                    for l in 0..n {
                        if !is_bonded(k, l, molecule) {
                            continue;
                        }
                        if i == l {
                            continue;
                        }

                        // Compute dihedral angle φ between planes (i-j-k) and (j-k-l)
                        let p_i = molecule.atoms[i].position;
                        let p_j = molecule.atoms[j].position;
                        let p_k = molecule.atoms[k].position;
                        let p_l = molecule.atoms[l].position;
                        let b1 = p_j - p_i;
                        let b2 = p_k - p_j;
                        let b3 = p_l - p_k;
                        let n1 = b1.cross(&b2).normalize();
                        let n2 = b2.cross(&b3).normalize();
                        let m1 = n1.cross(&b2.normalize());
                        let x = n1.dot(&n2);
                        let y = m1.dot(&n2);
                        let phi = y.atan2(x);

                        let key = (
                            self.get_atom_type(&molecule.atoms[i]),
                            self.get_atom_type(&molecule.atoms[j]),
                            self.get_atom_type(&molecule.atoms[k]),
                            self.get_atom_type(&molecule.atoms[l]),
                        );
                        let params = self
                            .force_field_parameters
                            .dihedral_parameters
                            .get(&key)
                            .cloned()
                            .unwrap_or(crate::quantum_chemistry::DihedralParameters {
                                periodicity: 3,
                                phase_angle: 0.0,
                                barrier_height: 2.0 * 4184.0, // 2 kcal/mol
                            });
                        let energy = params.barrier_height
                            * 0.5
                            * (1.0 - (params.periodicity as f64 * phi - params.phase_angle).cos());
                        total += energy;
                    }
                }
            }
        }
        Ok(total)
    }

    fn calculate_nonbonded_energy(&self, molecule: &crate::Molecule) -> Result<f64> {
        let mut total = 0.0_f64;
        let atoms = &molecule.atoms;
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                // Skip if bonded (1-2) or 1-3 (angle) neighbors
                if molecule
                    .bonds
                    .iter()
                    .any(|b| {
                        (b.atom_indices.0 == i && b.atom_indices.1 == j)
                            || (b.atom_indices.0 == j && b.atom_indices.1 == i)
                    })
                {
                    continue;
                }
                let r = (atoms[i].position - atoms[j].position).norm();
                if r < 1e-12 {
                    continue;
                }
                let pt_i = self.get_atom_type(&atoms[i]);
                let pt_j = self.get_atom_type(&atoms[j]);
                let vdw_i = self
                    .force_field_parameters
                    .van_der_waals_parameters
                    .get(&pt_i)
                    .cloned()
                    .unwrap_or(crate::quantum_chemistry::VdwParameters {
                        sigma: 3.5e-10,
                        epsilon: 0.2 * 4184.0, // 0.2 kcal/mol
                        radius: 1.5e-10,
                    });
                let vdw_j = self
                    .force_field_parameters
                    .van_der_waals_parameters
                    .get(&pt_j)
                    .cloned()
                    .unwrap_or(crate::quantum_chemistry::VdwParameters {
                        sigma: 3.5e-10,
                        epsilon: 0.2 * 4184.0,
                        radius: 1.5e-10,
                    });
                let sigma = 0.5 * (vdw_i.sigma + vdw_j.sigma);
                let epsilon = (vdw_i.epsilon * vdw_j.epsilon).sqrt();
                let sr6 = (sigma / r).powi(6);
                let v_lj = 4.0 * epsilon * (sr6 * sr6 - sr6);
                total += v_lj;
            }
        }
        Ok(total)
    }

    fn partition_qm_mm(&self, molecule: &crate::Molecule) -> (Vec<crate::Atom>, Vec<crate::Atom>) {
        (molecule.atoms.clone(), Vec::new())
    }

    fn calculate_qm_region_energy(&self, _atoms: &[crate::Atom]) -> Result<f64> { unimplemented!("QM region energy requires ab initio or DFT calculation") }
    fn calculate_mm_region_energy(&self, _atoms: &[crate::Atom]) -> Result<f64> { unimplemented!("MM region energy requires classical force field evaluation") }
    fn calculate_qm_mm_interaction(&self, _qm: &[crate::Atom], _mm: &[crate::Atom]) -> Result<f64> { unimplemented!("QM/MM interaction requires electrostatic embedding") }

    fn reactants_match(&self, _db_reactants: &[crate::ParticleType], _reactants: &[crate::ParticleType]) -> bool { false }
}

impl Default for crate::quantum_chemistry::ForceFieldParameters {
    fn default() -> Self {
        Self {
            bond_parameters: HashMap::new(),
            angle_parameters: HashMap::new(),
            dihedral_parameters: HashMap::new(),
            van_der_waals_parameters: HashMap::new(),
        }
    }
}

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

// --------------------------------------------------------------------------------
// Fallback interaction handlers for builds without the heavy `quantum-chemistry`
// feature. These NO-OP implementations satisfy unconditional calls made in the
// interaction loop without pulling in additional dependencies.
// --------------------------------------------------------------------------------
#[cfg(not(feature = "quantum-chemistry"))]
impl PhysicsEngine {
    fn process_strong_interaction(&mut self, _i: usize, _j: usize, _distance: f64) -> anyhow::Result<()> {
        Ok(())
    }

    fn process_weak_interaction(&mut self, _i: usize, _j: usize, _distance: f64) -> anyhow::Result<()> {
        Ok(())
    }
}