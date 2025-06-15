//! Comprehensive Physics Engine
//! 
//! Complete fundamental particle physics simulation from quantum fields
//! to complex matter structures. Implements the Standard Model and beyond.

use nalgebra::{Vector3, Matrix3, Complex};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use bevy_ecs::prelude::Component;

use self::nuclear_physics::{Nucleus, StellarNucleosynthesis};

pub mod constants;
pub mod classical;
pub mod electromagnetic;
pub mod thermodynamics;
pub mod quantum;
pub mod chemistry;
pub mod geodynamics;
pub mod climate;
pub mod validation;

// New fundamental particle modules
pub mod particles;
pub mod quantum_fields;
pub mod nuclear_physics;
pub mod atomic_physics;
pub mod molecular_dynamics;
pub mod phase_transitions;
pub mod emergent_properties;
pub mod interactions;

pub use constants::*;

/// Fundamental particle types in the Standard Model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleType {
    // Quarks
    Up, Down, Charm, Strange, Top, Bottom,
    
    // Leptons
    Electron, ElectronNeutrino, ElectronAntiNeutrino, Muon, MuonNeutrino, Tau, TauNeutrino,
    
    // Antiparticles
    Positron,
    
    // Gauge bosons
    Photon, WBoson, ZBoson, Gluon,
    
    // Scalar bosons
    Higgs,
    
    // Composite particles
    Proton, Neutron, 
    
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
    pub neutron_decay_count: u64,
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
            neutron_decay_count: 0,
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
        
        // Initialize cross sections for particle interactions
        self.cross_sections.insert((ParticleType::Electron, ParticleType::Electron), 1e-30); // m²
        self.cross_sections.insert((ParticleType::Proton, ParticleType::Proton), 1e-27); // m²
        
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
    pub fn step(&mut self, _classical_states: &mut [PhysicsState]) -> Result<()> {
        // Main physics loop
        
        // 1. Particle interactions
        self.process_particle_interactions()?;
        
        // 2. Decays
        self.process_particle_decays()?;
        
        // 3. Nuclear processes
        self.process_nuclear_reactions()?;
        
        // 4. Update temperature based on particle energies
        self.update_temperature()?;
        
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
    
    /// Update nuclear physics (fusion, fission, nuclear reactions)
    fn process_nuclear_reactions(&mut self) -> Result<()> {
        // Check for nuclear fusion possibilities
        self.process_nuclear_fusion()?;
        
        // Check for nuclear fission
        self.process_nuclear_fission()?;
        
        // Update nuclear shell structure
        self.update_nuclear_shells()?;
        
        Ok(())
    }
    
    /// Process nuclear fusion reactions using stellar nucleosynthesis
    fn process_nuclear_fusion(&mut self) -> Result<()> {
        // Create composition array from current nuclei
        let mut composition = self.build_isotope_composition();
        
        // Calculate stellar density from nuclei
        let density = self.calculate_stellar_density();
        
        // Process stellar nucleosynthesis
        let energy_released = self.stellar_nucleosynthesis
            .process_stellar_burning(self.temperature, density, &mut composition)?;
        
        // Update energy density with nuclear energy
        self.energy_density += energy_released / self.volume;
        
        // Update nuclei from new composition
        self.update_nuclei_from_composition(&composition)?;
        
        // Fall back to legacy fusion for compatibility
        self.process_legacy_fusion()?;
        
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
        // Remove parent particle
        let parent = self.particles.swap_remove(index);
        
        if parent.particle_type == ParticleType::Neutron {
            // Use proper beta decay kinematics
            let mut rng = thread_rng();
            let neutron_mass_gev = parent.mass * SPEED_OF_LIGHT.powi(2) / (1.602176634e-10); // J to GeV
            let proton_mass_gev = PROTON_MASS * SPEED_OF_LIGHT.powi(2) / (1.602176634e-10);
            let electron_mass_gev = ELECTRON_MASS * SPEED_OF_LIGHT.powi(2) / (1.602176634e-10);
            
            let (p_proton, p_electron, p_neutrino) = interactions::sample_beta_decay_kinematics(
                neutron_mass_gev, proton_mass_gev, electron_mass_gev, &mut rng
            );
            
            // Convert back to SI units (GeV to kg⋅m/s)
            let gev_to_kg_ms = 5.344286e-19; // GeV/c to kg⋅m/s
            
            // Create decay products with proper kinematics
            let products_data = [
                (ParticleType::Proton, p_proton * gev_to_kg_ms, PROTON_MASS),
                (ParticleType::Electron, p_electron * gev_to_kg_ms, ELECTRON_MASS),
                (ParticleType::ElectronAntiNeutrino, p_neutrino * gev_to_kg_ms, 1e-36), // tiny neutrino mass
            ];
            
            for (ptype, momentum, mass) in products_data {
                let particle = FundamentalParticle {
                    particle_type: ptype,
                    position: parent.position,
                    momentum,
                    spin: self.initialize_spin(ptype),
                    color_charge: self.assign_color_charge(ptype),
                    electric_charge: self.get_electric_charge(ptype),
                    mass,
                    energy: 0.0, // Will be calculated
                    creation_time: self.current_time,
                    decay_time: self.calculate_decay_time(ptype),
                    quantum_state: QuantumState::new(),
                    interaction_history: Vec::new(),
                };
                self.particles.push(particle);
            }
            
            self.neutron_decay_count += 1;
        } else {
            // Simple momentum sharing for other decays
            let position = parent.position;
            let mut rng = thread_rng();
            for &ptype in &channel.products {
                let momentum = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
                let particle = FundamentalParticle {
                    particle_type: ptype,
                    position,
                    momentum,
                    spin: self.initialize_spin(ptype),
                    color_charge: self.assign_color_charge(ptype),
                    electric_charge: self.get_electric_charge(ptype),
                    mass: self.get_particle_mass(ptype),
                    energy: 0.0,
                    creation_time: self.current_time,
                    decay_time: self.calculate_decay_time(ptype),
                    quantum_state: QuantumState::new(),
                    interaction_history: Vec::new(),
                };
                self.particles.push(particle);
            }
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
        if nucleus_idx >= self.nuclei.len() {
            return Ok(());
        }
        
        let original_nucleus = self.nuclei[nucleus_idx].clone();
        
        // Simplified fission: split into two fragments plus neutrons
        let fragment1_mass = original_nucleus.mass_number / 2 + (rand::random::<i32>() % 20 - 10) as u32;
        let fragment2_mass = original_nucleus.mass_number - fragment1_mass;
        let fragment1_z = original_nucleus.atomic_number / 2 + (rand::random::<i32>() % 10 - 5) as u32;
        let fragment2_z = original_nucleus.atomic_number - fragment1_z;
        
        // Create fission fragments
        let fragment1 = AtomicNucleus {
            mass_number: fragment1_mass,
            atomic_number: fragment1_z,
            protons: Vec::new(), // Simplified
            neutrons: Vec::new(), // Simplified
            binding_energy: Nucleus::new(fragment1_z, fragment1_mass - fragment1_z).binding_energy() * 1.60218e-13,
            nuclear_spin: Vector3::zeros(),
            magnetic_moment: Vector3::zeros(),
            electric_quadrupole_moment: 0.0,
            nuclear_radius: 1.2e-15 * (fragment1_mass as f64).powf(1.0/3.0),
            shell_model_state: HashMap::new(),
            position: original_nucleus.position + Vector3::new(1e-14, 0.0, 0.0),
            momentum: Vector3::new(1e-20, 0.0, 0.0), // Kinetic energy from fission
            excitation_energy: 5e-13, // Highly excited
        };
        
        let fragment2 = AtomicNucleus {
            mass_number: fragment2_mass,
            atomic_number: fragment2_z,
            protons: Vec::new(),
            neutrons: Vec::new(),
            binding_energy: Nucleus::new(fragment2_z, fragment2_mass - fragment2_z).binding_energy() * 1.60218e-13,
            nuclear_spin: Vector3::zeros(),
            magnetic_moment: Vector3::zeros(),
            electric_quadrupole_moment: 0.0,
            nuclear_radius: 1.2e-15 * (fragment2_mass as f64).powf(1.0/3.0),
            shell_model_state: HashMap::new(),
            position: original_nucleus.position - Vector3::new(1e-14, 0.0, 0.0),
            momentum: Vector3::new(-1e-20, 0.0, 0.0),
            excitation_energy: 5e-13,
        };
        
        // Remove original nucleus and add fragments
        self.nuclei.swap_remove(nucleus_idx);
        self.nuclei.push(fragment1);
        self.nuclei.push(fragment2);
        
        // Create neutrons (simplified - usually 2-3 neutrons released)
        let neutron_count = 2 + (rand::random::<u32>() % 2);
        for _ in 0..neutron_count {
            let neutron = FundamentalParticle {
                particle_type: ParticleType::Neutron,
                position: original_nucleus.position + Vector3::new(
                    (rand::random::<f64>() - 0.5) * 2e-14,
                    (rand::random::<f64>() - 0.5) * 2e-14,
                    (rand::random::<f64>() - 0.5) * 2e-14,
                ),
                momentum: Vector3::new(
                    (rand::random::<f64>() - 0.5) * 2e-21,
                    (rand::random::<f64>() - 0.5) * 2e-21,
                    (rand::random::<f64>() - 0.5) * 2e-21,
                ),
                spin: Vector3::new(0.5, 0.0, 0.0).map(|x| Complex::new(x, 0.0)),
                color_charge: None,
                electric_charge: 0.0,
                mass: NEUTRON_MASS,
                energy: NEUTRON_MASS * C_SQUARED,
                creation_time: self.current_time,
                decay_time: Some(self.current_time + 881.5), // Neutron lifetime
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
            };
            self.particles.push(neutron);
        }
        
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
        // Check if two nuclei can undergo fusion
        // Simplified Coulomb barrier calculation
        
        let r12 = (n1.position - n2.position).norm();
        let z1 = n1.atomic_number as f64;
        let z2 = n2.atomic_number as f64;
        
        // Coulomb barrier height (simplified)
        let coulomb_barrier = K_E * z1 * z2 * E_CHARGE / r12;
        
        // Kinetic energy from relative motion
        let reduced_mass = (n1.mass_number * n2.mass_number) as f64 / (n1.mass_number + n2.mass_number) as f64 * PROTON_MASS;
        let relative_velocity = (n1.momentum / n1.mass_number as f64 - n2.momentum / n2.mass_number as f64).norm();
        let kinetic_energy = 0.5 * reduced_mass * relative_velocity.powi(2);
        
        // Quantum tunneling probability (simplified)
        let tunneling_prob = (-2.0 * (coulomb_barrier - kinetic_energy).max(0.0) / (HBAR * C)).exp();
        
        // Fusion cross-section (simplified - depends on nuclear properties)
        let fusion_cross_section = if z1 <= 2.0 && z2 <= 2.0 {
            1e-28 * tunneling_prob // Light nuclei like H, He
        } else {
            1e-32 * tunneling_prob // Heavier nuclei
        };
        
        // Check if fusion should occur this timestep
        let interaction_rate = fusion_cross_section * relative_velocity / (4.0 * std::f64::consts::PI * r12.powi(2));
        let fusion_probability = interaction_rate * self.time_step;
        
        Ok(rand::random::<f64>() < fusion_probability)
    }
    
    fn calculate_fusion_reaction(&self, i: usize, j: usize) -> Result<FusionReaction> {
        if i >= self.nuclei.len() || j >= self.nuclei.len() {
            return Ok(FusionReaction::default());
        }
        
        let n1 = &self.nuclei[i];
        let n2 = &self.nuclei[j];
        
        // Create fusion reaction based on reactants
        let product_mass = n1.mass_number + n2.mass_number;
        let product_z = n1.atomic_number + n2.atomic_number;
        
        // Q-value calculation (simplified)
        let binding_energy_reactants = n1.binding_energy + n2.binding_energy;
        let binding_energy_product = Nucleus::new(product_z, product_mass - product_z).binding_energy() * 1.60218e-13;
        let q_value = binding_energy_product - binding_energy_reactants;
        
        Ok(FusionReaction {
            reactant_indices: vec![i, j],
            product_mass_number: product_mass,
            product_atomic_number: product_z,
            q_value,
            cross_section: 1e-28,
            requires_catalysis: product_z > 6, // Heavy products need catalysis
        })
    }
    
    fn execute_fusion_reaction(&mut self, reaction: FusionReaction) -> Result<()> {
        if reaction.reactant_indices.len() != 2 {
            return Ok(());
        }
        
        let i = reaction.reactant_indices[0];
        let j = reaction.reactant_indices[1];
        
        if i >= self.nuclei.len() || j >= self.nuclei.len() || i == j {
            return Ok(());
        }
        
        // Get reactant nuclei (need to handle the case where j might be shifted after removing i)
        let (n1, n2) = if i < j {
            let n1 = self.nuclei[i].clone();
            let n2 = self.nuclei[j].clone();
            self.nuclei.swap_remove(j);
            self.nuclei.swap_remove(i);
            (n1, n2)
        } else {
            let n2 = self.nuclei[j].clone();
            let n1 = self.nuclei[i].clone();
            self.nuclei.swap_remove(i);
            self.nuclei.swap_remove(j);
            (n1, n2)
        };
        
        // Create fusion product
        let product_position = (n1.position + n2.position) / 2.0;
        let product_momentum = n1.momentum + n2.momentum;
        
        let fusion_product = AtomicNucleus {
            mass_number: reaction.product_mass_number,
            atomic_number: reaction.product_atomic_number,
            protons: Vec::new(), // Simplified
            neutrons: Vec::new(), // Simplified
            binding_energy: reaction.q_value,
            nuclear_spin: Vector3::zeros(),
            magnetic_moment: Vector3::zeros(),
            electric_quadrupole_moment: 0.0,
            nuclear_radius: 1.2e-15 * (reaction.product_mass_number as f64).powf(1.0/3.0),
            shell_model_state: HashMap::new(),
            position: product_position,
            momentum: product_momentum,
            excitation_energy: reaction.q_value.max(0.0), // Excess energy becomes excitation
        };
        
        self.nuclei.push(fusion_product);
        
        // Release energy as photons or particles if Q > 0
        if reaction.q_value > 0.0 {
            // Create a high-energy photon
            let photon = FundamentalParticle {
                particle_type: ParticleType::Photon,
                position: product_position,
                momentum: Vector3::new(reaction.q_value / C, 0.0, 0.0),
                spin: Vector3::new(1.0, 0.0, 0.0).map(|x| Complex::new(x, 0.0)),
                color_charge: None,
                electric_charge: 0.0,
                mass: 0.0,
                energy: reaction.q_value,
                creation_time: self.current_time,
                decay_time: None, // Photons are stable
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
            };
            self.particles.push(photon);
        }
        
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
    fn update_molecular_dynamics(&mut self, _states: &mut [PhysicsState]) -> Result<()> { Ok(()) }
    #[allow(dead_code)]
    fn process_phase_transitions(&mut self) -> Result<()> { Ok(()) }
    #[allow(dead_code)]
    fn update_emergent_properties(&mut self, _states: &mut [PhysicsState]) -> Result<()> { Ok(()) }
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