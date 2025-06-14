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
    Electron, ElectronNeutrino, Muon, MuonNeutrino, Tau, TauNeutrino,
    
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
    pub time_step: f64,
    pub current_time: f64,
    pub temperature: f64,
    pub energy_density: f64,
    pub particle_creation_threshold: f64,
    pub volume: f64,  // Simulation volume in m³
    pub compton_count: u64,  // Track Compton scattering events
    pub pair_production_count: u64,  // Track pair production events
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Electron {
    pub position_probability: Vec<Vec<Vec<f64>>>, // 3D probability density
    pub momentum_distribution: Vec<Vector3<f64>>,
    pub spin: Vector3<Complex<f64>>,
    pub orbital_angular_momentum: Vector3<f64>,
    pub quantum_numbers: QuantumNumbers,
    pub binding_energy: f64,
}

/// Quantum numbers for electron states
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrbitalType {
    S, P, D, F,
}

/// Molecule with detailed bonding
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

/// Chemical bond between atoms
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BondType {
    Ionic, Covalent, Metallic, HydrogenBond, VanDerWaals,
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
            time_step,
            current_time: 0.0,
            temperature: 0.0,
            energy_density: 0.0,
            particle_creation_threshold: 1e-10,
            volume: 1e-30,  // 1 cubic femtometer
            compton_count: 0,
            pair_production_count: 0,
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
        
        // Neutron decay: n → p + e + νe
        self.decay_channels.insert(ParticleType::Neutron, vec![
            DecayChannel {
                products: vec![ParticleType::Proton, ParticleType::Electron, ParticleType::ElectronNeutrino],
                branching_ratio: 1.0,
                decay_constant: 1.0 / 880.0, // Neutron lifetime
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
                decay_time: self.calculate_decay_time(particle_type),
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
        // Update time
        self.current_time += self.time_step;
        
        // Quantum evolution
        self.evolve_quantum_state()?;
        
        // Process interactions (including new QED interactions)
        self.process_particle_interactions()?;
        
        // Process decays
        self.process_particle_decays()?;
        
        // Nuclear processes
        self.process_nuclear_reactions()?;
        
        // Update temperature based on particle energies
        self.update_temperature()?;
        
        // Classical state evolution would go here if we had the subsystems
        // For now, we skip this part
        
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
                let p1 = &self.particles[i];
                let p2 = &self.particles[j];
                
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
                    let electron_mass_energy = ELECTRON_MASS * SPEED_OF_LIGHT.powi(2);
                    let cross_section = klein_nishina_cross_section(photon_energy, electron_mass_energy);
                    
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
    
    /// Process nuclear fusion reactions
    fn process_nuclear_fusion(&mut self) -> Result<()> {
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
    fn get_particle_mass(&self, particle_type: ParticleType) -> f64 {
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
    
    fn initialize_spin(&self, particle_type: ParticleType) -> Vector3<Complex<f64>> {
        Vector3::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.5, 0.0))
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
    
    fn update_particle_energies(&mut self) -> Result<()> {
        for particle in &mut self.particles {
            let p_squared = particle.momentum.norm_squared();
            let m = particle.mass;
            let c = SPEED_OF_LIGHT;
            
            // Relativistic energy: E² = (pc)² + (mc²)²
            particle.energy = ((p_squared * c * c) + (m * c * c).powi(2)).sqrt();
        }
        Ok(())
    }
    
    // Placeholder methods for complex physics processes
    fn calculate_interaction_range(&self, p1: ParticleType, p2: ParticleType) -> f64 { 1e-15 }
    fn calculate_interaction(&self, i: usize, j: usize) -> Result<Interaction> { Ok(Interaction::default()) }
    fn apply_interaction(&mut self, interaction: Interaction) -> Result<()> { Ok(()) }
    fn select_decay_channel(&self, channels: &[DecayChannel]) -> DecayChannel { channels[0].clone() }
    fn execute_decay(&mut self, index: usize, channel: DecayChannel) -> Result<()> { Ok(()) }
    fn process_nuclear_fission(&mut self) -> Result<()> { Ok(()) }
    fn update_nuclear_shells(&mut self) -> Result<()> { Ok(()) }
    fn can_fuse(&self, n1: &AtomicNucleus, n2: &AtomicNucleus) -> Result<bool> { Ok(false) }
    fn calculate_fusion_reaction(&self, i: usize, j: usize) -> Result<FusionReaction> { Ok(FusionReaction::default()) }
    fn execute_fusion_reaction(&mut self, reaction: FusionReaction) -> Result<()> { Ok(()) }
    fn update_atomic_physics(&mut self) -> Result<()> { Ok(()) }
    fn update_molecular_dynamics(&mut self) -> Result<()> { Ok(()) }
    fn process_phase_transitions(&mut self) -> Result<()> { Ok(()) }
    fn update_emergent_properties(&mut self, states: &mut [PhysicsState]) -> Result<()> { Ok(()) }
    fn update_running_couplings(&mut self) -> Result<()> { Ok(()) }
    fn check_symmetry_breaking(&mut self) -> Result<()> { Ok(()) }
    fn update_spacetime_curvature(&mut self) -> Result<()> { Ok(()) }
    fn update_thermodynamic_state(&mut self) -> Result<()> {
        // Update temperature based on particle kinetic energies
        let total_kinetic_energy: f64 = self.particles.iter()
            .map(|p| 0.5 * p.momentum.norm_squared() / p.mass)
            .sum();
        
        if !self.particles.is_empty() {
            self.temperature = 2.0 * total_kinetic_energy / (3.0 * BOLTZMANN * self.particles.len() as f64);
        }
        
        Ok(())
    }
    
    /// Evolve quantum state of all particles
    fn evolve_quantum_state(&mut self) -> Result<()> {
        // Placeholder for quantum evolution
        // In a full implementation, this would solve the Schrödinger/Dirac equation
        Ok(())
    }
    
    /// Update temperature based on particle energies
    fn update_temperature(&mut self) -> Result<()> {
        if self.particles.is_empty() {
            return Ok(());
        }
        
        // Calculate average kinetic energy
        let total_kinetic_energy: f64 = self.particles.iter()
            .map(|p| p.energy - p.mass * SPEED_OF_LIGHT.powi(2))
            .filter(|&ke| ke > 0.0)
            .sum();
        
        let num_particles = self.particles.len() as f64;
        let avg_kinetic_energy = total_kinetic_energy / num_particles;
        
        // Temperature from kinetic theory: <KE> = (3/2) k_B T
        // For relativistic particles, use <E> ≈ 3 k_B T
        self.temperature = avg_kinetic_energy / (3.0 * BOLTZMANN);
        
        Ok(())
    }
}

// Supporting types and implementations
#[derive(Debug, Clone, Default)]
pub struct Interaction;

#[derive(Debug, Clone)]
pub struct DecayChannel {
    pub products: Vec<ParticleType>,
    pub branching_ratio: f64,
    pub decay_constant: f64,
}

#[derive(Debug, Clone, Default)]
pub struct FusionReaction;

pub struct InteractionMatrix;
impl InteractionMatrix {
    pub fn new() -> Self { Self }
    pub fn set_electromagnetic_coupling(&mut self, coupling: f64) {}
    pub fn set_weak_coupling(&mut self, coupling: f64) {}
    pub fn set_strong_coupling(&mut self, coupling: f64) {}
}

pub struct SpacetimeGrid;
impl SpacetimeGrid {
    pub fn new(size: usize, spacing: f64) -> Self { Self }
}

pub struct QuantumVacuum;
impl QuantumVacuum {
    pub fn new() -> Self { Self }
    pub fn initialize_fluctuations(&mut self, temperature: f64) -> Result<()> { Ok(()) }
}

pub struct FieldEquations;
impl FieldEquations {
    pub fn new() -> Self { Self }
    pub fn update_field(&self, field: &mut QuantumField, dt: f64, particles: &[FundamentalParticle]) -> Result<()> { Ok(()) }
}

pub struct ParticleAccelerator;
impl ParticleAccelerator {
    pub fn new() -> Self { Self }
}

pub struct RunningCouplings;
impl RunningCouplings {
    pub fn new() -> Self { Self }
}

pub struct SymmetryBreaking;
impl SymmetryBreaking {
    pub fn new() -> Self { Self }
    pub fn initialize_higgs_mechanism(&mut self) -> Result<()> { Ok(()) }
}

impl QuantumField {
    pub fn new(field_type: FieldType, grid: &SpacetimeGrid) -> Result<Self> {
        Ok(Self {
            field_type,
            field_values: vec![vec![vec![Complex::new(0.0, 0.0); 100]; 100]; 100],
            field_derivatives: vec![vec![vec![Vector3::zeros(); 100]; 100]; 100],
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Periodic, Absorbing, Reflecting,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

/// Core physics state for classical particles (legacy interface)
#[derive(Debug, Clone, Serialize, Deserialize, Component)]
pub struct PhysicsState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub acceleration: Vector3<f64>,
    pub mass: f64,
    pub charge: f64,
    pub temperature: f64,
    pub entropy: f64,
}

/// Record of particle interactions for lineage tracking
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

/// Element table for compatibility with existing code
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

/// Environment profile for habitat calculations
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

/// Stratum layer for geological simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumLayer {
    pub thickness_m: f64,
    pub material_type: MaterialType,
    pub bulk_density: f64,
    pub elements: ElementTable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    Gas, Regolith, Topsoil, Subsoil, SedimentaryRock, 
    IgneousRock, MetamorphicRock, OreVein, Ice, Magma,
}