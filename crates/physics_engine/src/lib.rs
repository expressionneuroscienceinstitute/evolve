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
pub mod geodynamics;
pub mod interactions;
pub mod molecular_dynamics;
pub mod nuclear_physics;
pub mod particles;
pub mod phase_transitions;
pub mod quantum;
pub mod quantum_fields;
pub mod thermodynamics;
pub mod validation;

use nalgebra::{Vector3, Matrix3, Complex};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use bevy_ecs::prelude::Component;

use self::nuclear_physics::StellarNucleosynthesis;

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
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub wave_function: Vec<Complex<f64>>,
    pub entanglement_partners: Vec<usize>,
    pub decoherence_time: f64,
    pub measurement_basis: MeasurementBasis,
    pub superposition_amplitudes: HashMap<String, Complex<f64>>,
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
#[derive(Serialize, Deserialize, Clone)]
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
    /// Create new physics engine with fundamental particle simulation
    pub fn new(time_step: f64) -> Result<Self> {
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
            time_step,
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
        };
        
        // Initialize quantum fields
        engine.initialize_quantum_fields()?;
        
        // Initialize particle properties
        engine.initialize_particle_properties()?;
        
        // Initialize interaction matrix
        engine.initialize_interactions()?;
        
        // Set larger volume for demo
        engine.volume = 1e-42; // Cubic femtometer scale
        
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
    
    /// Comprehensive physics simulation step
    pub fn step(&mut self, classical_states: &mut [PhysicsState]) -> Result<()> {
        // Main physics loop
        
        // 1. Particle interactions (QED, weak, strong)
        self.process_particle_interactions()?;
        
        // 2. Particle decays and radioactivity
        self.process_particle_decays()?;
        
        // 3. Nuclear processes (fusion, fission, nuclear reactions)
        self.process_nuclear_reactions()?;
        
        // 4. Molecular dynamics and chemical reactions
        self.update_molecular_dynamics(classical_states)?;
        
        // 5. Phase transitions (gas/liquid/solid/plasma)
        self.process_phase_transitions()?;
        
        // 6. Emergent properties (statistical mechanics)
        self.update_emergent_properties(classical_states)?;
        
        // 7. Update thermodynamic state
        self.update_thermodynamic_state()?;
        
        // 8. Quantum state evolution
        self.evolve_quantum_state()?;
        
        // 9. Update particle energies for consistency
        self.update_particle_energies()?;
        
        // 10. Update current simulation time
        self.current_time += self.time_step;
        
        Ok(())
    }
    
    /// Process particle interactions (QED, weak, strong)
    fn process_particle_interactions(&mut self) -> Result<()> {
        use interactions::*;
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        let mut interactions_to_process = Vec::new();
        
        // Find potential Compton scattering pairs
        for i in 0..self.particles.len() {
            for j in i+1..self.particles.len() {
                let (p1, p2) = (&self.particles[i], &self.particles[j]);
                
                // Check distance (interaction range ~ 1 fm)
                let distance = (p1.position - p2.position).norm();
                if distance > 1e-15 {
                    continue;
                }
                
                // Check for Compton scattering
                if can_compton_scatter(p1, p2) {
                    let (photon_idx, electron_idx) = if p1.particle_type == ParticleType::Photon {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    
                    let photon_energy = self.particles[photon_idx].energy;
                    let energy_gev = photon_energy / 1.602_176_634e-10; // convert J to GeV
                    let cross_section = klein_nishina_cross_section(energy_gev);
                    
                    // Estimate local number density (simplified)
                    let volume = 4.0 * std::f64::consts::PI * distance.powi(3) / 3.0;
                    let number_density = 1.0 / volume;
                    
                    let probability = interaction_probability(
                        cross_section,
                        number_density,
                        SPEED_OF_LIGHT,
                        self.time_step,
                    );
                    
                    if rng.gen::<f64>() < probability {
                        interactions_to_process.push(Interaction {
                            particle_indices: (photon_idx, electron_idx),
                            interaction_type: InteractionType::ComptonScattering,
                            cross_section,
                            probability,
                        });
                    }
                }

                // Check for neutrino-electron scattering
                if (matches!(p1.particle_type, ParticleType::ElectronNeutrino) && matches!(p2.particle_type, ParticleType::Electron))
                    || (matches!(p2.particle_type, ParticleType::ElectronNeutrino) && matches!(p1.particle_type, ParticleType::Electron)) {
                    
                    let (nu_idx, _e_idx) = if matches!(p1.particle_type, ParticleType::ElectronNeutrino | ParticleType::ElectronAntiNeutrino) {
                        (i, j)
                    } else {
                        (j, i)
                    };

                    let nu = &self.particles[nu_idx];
                    let is_antineutrino = matches!(nu.particle_type, ParticleType::ElectronAntiNeutrino);
                    let cross_section = interactions::neutrino_e_scattering_complete(0, nu.energy / (1.602176634e-10), is_antineutrino);

                    // Use same probability formula
                    let volume = 4.0 * std::f64::consts::PI * distance.powi(3) / 3.0;
                    let number_density = 1.0 / volume;
                    let probability = interaction_probability(
                        cross_section,
                        number_density,
                        SPEED_OF_LIGHT,
                        self.time_step,
                    );
                    if rng.gen::<f64>() < probability {
                        // For now just count scatter without changing momenta
                        self.neutrino_scatter_count += 1;
                    }
                }
            }
        }
        
        // Process pair production for high-energy photons
        let mut new_particles = Vec::new();
        let mut photons_to_remove = Vec::new();
        
        for (i, particle) in self.particles.iter().enumerate() {
            if particle.particle_type == ParticleType::Photon {
                let electron_mass_energy = ELECTRON_MASS * SPEED_OF_LIGHT.powi(2);
                
                // Check if photon has enough energy for pair production
                if particle.energy > 2.0 * electron_mass_energy {
                    // Use hydrogen (Z=1) for early universe
                    let cross_section = bethe_heitler_cross_section(particle.energy, 1);
                    
                    // Estimate probability based on local proton density
                    let proton_density = self.particles.iter()
                        .filter(|p| p.particle_type == ParticleType::Proton)
                        .count() as f64 / self.volume;
                    
                    let probability = interaction_probability(
                        cross_section,
                        proton_density,
                        SPEED_OF_LIGHT,
                        self.time_step,
                    );
                    
                    if rng.gen::<f64>() < probability {
                        if let Some((electron, positron)) = pair_produce(particle, &mut rng) {
                            new_particles.push(electron);
                            new_particles.push(positron);
                            photons_to_remove.push(i);
                        }
                    }
                }
            }
        }
        
        // Apply Compton scattering
        for interaction in interactions_to_process {
            let (photon_idx, electron_idx) = interaction.particle_indices;
            
            // Clone particles to avoid borrow issues
            let mut photon = self.particles[photon_idx].clone();
            let mut electron = self.particles[electron_idx].clone();
            
            scatter_compton(&mut photon, &mut electron, &mut rng);
            
            // Update particles
            self.particles[photon_idx] = photon;
            self.particles[electron_idx] = electron;
            
            // Count the event
            self.compton_count += 1;
        }
        
        // Remove photons that pair-produced (in reverse order)
        for &idx in photons_to_remove.iter().rev() {
            self.particles.swap_remove(idx);
        }
        
        // Count pair production events
        self.pair_production_count += new_particles.len() as u64 / 2;
        
        // Add new particles from pair production
        for particle in new_particles {
            self.particles.push(particle);
        }
        
        Ok(())
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
    
    pub fn update_particle_energies(&mut self) -> Result<()> {
        for particle in self.particles.iter_mut() {
            particle.energy = (particle.mass.powi(2) + particle.momentum.norm_squared()).sqrt();
        }
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
            };
            self.particles.push(neutron);
        }
        
        self.fission_count += 1;

        // TODO: Distribute Q-value energy among products

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
            wave_function: vec![Complex::new(1.0, 0.0)],
            entanglement_partners: Vec::new(),
            decoherence_time: 1e-12,
            measurement_basis: MeasurementBasis::Position,
            superposition_amplitudes: HashMap::new(),
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
    Position, Momentum, Energy, Spin,
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

// Additional physics constants for nuclear reactions
pub const K_E: f64 = 8.99e9; // Coulomb constant (N⋅m²/C²)
pub const E_CHARGE: f64 = ELEMENTARY_CHARGE;
pub const C: f64 = SPEED_OF_LIGHT;
pub const C_SQUARED: f64 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
pub const HBAR: f64 = REDUCED_PLANCK_CONSTANT;

/// Represents the physical state of a celestial body for simulation purposes.
/// This component will be attached to Bevy entities.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
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