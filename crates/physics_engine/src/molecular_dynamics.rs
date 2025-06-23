//! Molecular Dynamics Physics Layer
//!
//! Implements atomic bonding and molecular interactions using physics-first principles.
//! - All emergent behavior must arise from quantum/atomic layer (no hard-coded biology)
//! - Follows best practices from CPVR Chapter 17: force fields, Verlet integration, neighbor lists, parallelization
//! - Designed for scientific rigor, extensibility, and performance
//!
//! References:
//! - https://cpvr.rantai.dev/docs/part-iv/chapter-17/
//! - Project RESEARCH_PAPERS.md

pub mod force_fields;
pub mod integration;
pub mod neighbor_list;
pub mod atomic_molecular_bridge;

// Core data structures for molecular dynamics
// (To be implemented: Particle, Molecule, System, etc.)

// Main simulation engine struct and methods will be defined here
// (To be implemented: step, run, energy conservation, etc.)

use anyhow::Result;
use nalgebra::Vector3;
use rand::Rng;

use crate::{PhysicsState, constants::{ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY}};
use crate::molecular_dynamics::force_fields::{LennardJonesParams, apply_lennard_jones_forces};
use crate::molecular_dynamics::integration::{velocity_verlet_step, kinetic_energy, temperature};
use crate::molecular_dynamics::neighbor_list::{CellList, VerletNeighborList};

/// Represents the parameters for a force field, e.g., Lennard-Jones.
#[derive(Debug, Clone, Default)]
pub struct ForceField {
    pub epsilon: f64, // Depth of the potential well
    pub sigma: f64,   // Finite distance at which the inter-particle potential is zero
}

impl ForceField {
    /// Creates a new `ForceField` with default parameters.
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self { epsilon, sigma }
    }

    /// Calculate forces on each particle using the force field.
    fn calculate_forces(&self, particles: &[PhysicsState]) -> Vec<Vector3<f64>> {
        let mut forces = vec![Vector3::zeros(); particles.len()];

        for i in 0..particles.len() {
            for j in (i + 1)..particles.len() {
                let r_ij = particles[j].position - particles[i].position;
                let dist = r_ij.norm();

                if dist < 1e-9 { continue; } // Avoid singularity

                // Lennard-Jones potential force
                let lj_force = self.lennard_jones_force(dist);
                
                // Electrostatic force (Coulomb's Law)
                let electrostatic_force = self.electrostatic_force(
                    particles[i].charge,
                    particles[j].charge,
                    dist,
                );
                
                let force_vec = (lj_force + electrostatic_force) * r_ij.normalize();

                forces[i] += force_vec;
                forces[j] -= force_vec;
            }
        }
        forces
    }
    
    /// Calculates the magnitude of the Lennard-Jones force.
    fn lennard_jones_force(&self, distance: f64) -> f64 {
        let sigma_over_r_6 = (self.sigma / distance).powi(6);
        let sigma_over_r_12 = sigma_over_r_6.powi(2);
        24.0 * self.epsilon / distance * (2.0 * sigma_over_r_12 - sigma_over_r_6)
    }

    /// Calculates the magnitude of the electrostatic force.
    fn electrostatic_force(&self, q1: f64, q2: f64, distance: f64) -> f64 {
        (q1 * ELEMENTARY_CHARGE) * (q2 * ELEMENTARY_CHARGE) / (4.0 * std::f64::consts::PI * VACUUM_PERMITTIVITY * distance.powi(2))
    }
}

/// Struct for managing Molecular Dynamics simulations.
#[derive(Debug, Default)]
pub struct MolecularDynamics {
    pub force_field: ForceField,
    pub temperature: f64,
    pub time_step: f64,
}

impl MolecularDynamics {
    /// Creates a new Molecular Dynamics manager.
    pub fn new(force_field: ForceField, temperature: f64, time_step: f64) -> Self {
        Self {
            force_field,
            temperature,
            time_step,
        }
    }

    /// Update particle states based on molecular forces.
    pub fn update(&self, states: &mut [PhysicsState]) -> Result<()> {
        let forces = self.force_field.calculate_forces(states);
        self.integrate(states, &forces);
        Ok(())
    }

    /// Integrate equations of motion (e.g., Verlet integration).
    fn integrate(&self, particles: &mut [PhysicsState], forces: &[Vector3<f64>]) {
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            let acceleration = f / p.mass;
            p.position += p.velocity * self.time_step + 0.5 * acceleration * self.time_step.powi(2);
            p.velocity += acceleration * self.time_step;
        }
    }
}

/// A convenience function to run a single step of a molecular dynamics simulation.
pub fn step_molecular_dynamics(
    particles: &mut [PhysicsState],
    force_field: &ForceField,
    time_step: f64,
) -> Result<()> {
    // This function is a simplified wrapper. For a full simulation, you would
    // create a `MolecularDynamics` struct and call its `step` method in a loop.
    
    // For this example, we'll follow the Velocity Verlet steps directly.
    let forces = force_field.calculate_forces(particles);
    for (i, p) in particles.iter_mut().enumerate() {
        let acc = forces[i] / p.mass;
        p.velocity += 0.5 * acc * time_step;
        p.position += p.velocity * time_step;
    }

    let new_forces = force_field.calculate_forces(particles);
    for (i, p) in particles.iter_mut().enumerate() {
        let new_acc = new_forces[i] / p.mass;
        p.velocity += 0.5 * new_acc * time_step;
    }

    Ok(())
}

/// Calculates the Lennard-Jones potential between two particles.
pub fn lennard_jones_potential(r: f64, epsilon: f64, sigma: f64) -> f64 {
    if r == 0.0 {
        return f64::INFINITY;
    }

    let sr6 = (sigma / r).powi(6);
    4.0 * epsilon * (sr6.powi(2) - sr6)
}

/// Represents a single particle (atom or molecule) in the simulation.
/// All properties must be physically meaningful and emerge from quantum/atomic layer.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Position in 3D space (meters)
    pub position: [f64; 3],
    /// Velocity in 3D space (meters/second)
    pub velocity: [f64; 3],
    /// Force acting on the particle (Newtons)
    pub force: [f64; 3],
    /// Mass of the particle (kg)
    pub mass: f64,
    /// Unique identifier for the particle type (e.g., atomic number)
    pub type_id: u32,
}

/// Represents the full molecular system being simulated.
/// Contains all particles and global simulation parameters.
#[derive(Debug)]
pub struct System {
    /// All particles in the simulation
    pub particles: Vec<Particle>,
    /// Simulation box size (for periodic boundary conditions, meters)
    pub box_size: [f64; 3],
    /// Time step for integration (seconds)
    pub dt: f64,
    /// Current simulation time (seconds)
    pub time: f64,
}

/// Main molecular dynamics simulation engine
/// 
/// Coordinates all aspects of the simulation including force calculation,
/// time integration, and neighbor list management.
#[derive(Debug)]
pub struct MolecularDynamicsEngine {
    /// The molecular system being simulated
    pub system: System,
    /// Lennard-Jones potential parameters
    pub lj_params: LennardJonesParams,
    /// Neighbor list for efficient force computation
    pub neighbor_list: VerletNeighborList,
    /// Number of steps between neighbor list updates
    pub neighbor_update_frequency: usize,
    /// Current simulation step
    pub step: usize,
}

impl MolecularDynamicsEngine {
    /// Create a new molecular dynamics engine
    pub fn new(
        system: System,
        lj_params: LennardJonesParams,
        neighbor_update_frequency: usize,
    ) -> Self {
        let neighbor_list = VerletNeighborList::new(lj_params.r_cut, lj_params.r_cut * 0.1);
        
        Self {
            system,
            lj_params,
            neighbor_list,
            neighbor_update_frequency,
            step: 0,
        }
    }
    
    /// Initialize a system with particles in a cubic lattice
    pub fn create_cubic_lattice(
        n_particles_per_dim: usize,
        box_size: [f64; 3],
        mass: f64,
        temperature: f64,
        dt: f64,
    ) -> System {
        let total_particles = n_particles_per_dim * n_particles_per_dim * n_particles_per_dim;
        let mut particles = Vec::with_capacity(total_particles);
        
        let spacing = [
            box_size[0] / n_particles_per_dim as f64,
            box_size[1] / n_particles_per_dim as f64,
            box_size[2] / n_particles_per_dim as f64,
        ];
        
        // Boltzmann constant for velocity initialization
        let k_b = 1.380649e-23;
        let velocity_scale = (3.0 * k_b * temperature / mass).sqrt();
        
        let mut rng = rand::thread_rng();

        for i in 0..n_particles_per_dim {
            for j in 0..n_particles_per_dim {
                for k in 0..n_particles_per_dim {
                    let position = [
                        (i as f64 + 0.5) * spacing[0],
                        (j as f64 + 0.5) * spacing[1],
                        (k as f64 + 0.5) * spacing[2],
                    ];
                    
                    // Initialize velocities with Maxwell-Boltzmann distribution
                    let velocity = [
                        velocity_scale * (rng.gen::<f64>() - 0.5),
                        velocity_scale * (rng.gen::<f64>() - 0.5),
                        velocity_scale * (rng.gen::<f64>() - 0.5),
                    ];
                    
                    particles.push(Particle {
                        position,
                        velocity,
                        force: [0.0; 3],
                        mass,
                        type_id: 1, // Default type
                    });
                }
            }
        }
        
        System {
            particles,
            box_size,
            dt,
            time: 0.0,
        }
    }
    
    /// Perform one simulation step
    pub fn step(&mut self) {
        // Update neighbor list if needed
        if self.step % self.neighbor_update_frequency == 0 {
            let mut cell_list = CellList::new(
                self.lj_params.r_cut,
                [
                    [0.0, self.system.box_size[0]],
                    [0.0, self.system.box_size[1]],
                    [0.0, self.system.box_size[2]],
                ]
            );
            let states = particles_to_physics_states(&self.system.particles);
            self.neighbor_list.build(&states, &mut cell_list).unwrap();
        }
        
        // Calculate forces
        apply_lennard_jones_forces(&mut self.system, &self.lj_params);
        
        // Integrate equations of motion
        velocity_verlet_step(&mut self.system);
        
        self.step += 1;
    }
    
    /// Run the simulation for a specified number of steps
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }
    
    /// Get current system properties
    pub fn get_properties(&self) -> SystemProperties {
        SystemProperties {
            step: self.step,
            time: self.system.time,
            temperature: temperature(&self.system),
            kinetic_energy: kinetic_energy(&self.system),
            n_particles: self.system.particles.len(),
            // Count unique neighbor pairs using the neighbor list. Each neighbor appears twice
            // across all per-particle lists, so divide the total by 2 to avoid double-counting.
            neighbor_pairs: self
                .neighbor_list
                .neighbor_lists
                .iter()
                .map(|neighbors| neighbors.len())
                .sum::<usize>()
                / 2,
        }
    }
}

/// System properties for monitoring and analysis
#[derive(Debug, Clone)]
pub struct SystemProperties {
    pub step: usize,
    pub time: f64,
    pub temperature: f64,
    pub kinetic_energy: f64,
    pub n_particles: usize,
    pub neighbor_pairs: usize,
}

fn particles_to_physics_states(particles: &[Particle]) -> Vec<PhysicsState> {
    particles.iter().map(|p| PhysicsState {
        position: nalgebra::Vector3::new(p.position[0], p.position[1], p.position[2]),
        velocity: nalgebra::Vector3::new(p.velocity[0], p.velocity[1], p.velocity[2]),
        acceleration: nalgebra::Vector3::zeros(),
        mass: p.mass,
        charge: 0.0, // Particle struct does not have charge, set to 0.0
        temperature: 300.0, // Default temperature
        entropy: 1e-20, // Default entropy
    }).collect()
}

/// Represents a snapshot of the molecular dynamics system for visualization
/// Contains all necessary data for real-time molecular visualization
#[derive(Debug, Clone)]
pub struct MDSnapshot {
    /// Current simulation step
    pub step: usize,
    /// Current simulation time (seconds)
    pub time: f64,
    /// Particle positions for visualization
    pub particle_positions: Vec<[f64; 3]>,
    /// Particle velocities for trajectory visualization
    pub particle_velocities: Vec<[f64; 3]>,
    /// Particle forces for force field visualization
    pub particle_forces: Vec<[f64; 3]>,
    /// Particle types for color coding
    pub particle_types: Vec<u32>,
    /// Particle masses for physics accuracy
    pub particle_masses: Vec<f64>,
    /// Active bonds between particles (particle indices)
    pub bonds: Vec<(usize, usize, f64)>, // (particle1, particle2, bond_strength)
    /// Neighbor pairs for interaction visualization
    pub neighbor_pairs: Vec<(usize, usize, f64)>, // (particle1, particle2, distance)
    /// System properties for monitoring
    pub properties: SystemProperties,
    /// Quantum-classical regions (particle indices requiring quantum treatment)
    pub quantum_regions: Vec<usize>,
    /// Molecular clusters (groups of bonded atoms)
    pub molecular_clusters: Vec<Vec<usize>>,
    /// Chemical reaction events (bond breaking/forming)
    pub reaction_events: Vec<ReactionEvent>,
}

/// Represents a chemical reaction event for visualization
#[derive(Debug, Clone)]
pub struct ReactionEvent {
    /// Type of reaction (bond formation or breaking)
    pub event_type: ReactionEventType,
    /// Particles involved in the reaction
    pub participants: Vec<usize>,
    /// Time when the reaction occurred
    pub time: f64,
    /// Energy change associated with the reaction
    pub energy_change: f64,
}

/// Types of chemical reaction events
#[derive(Debug, Clone)]
pub enum ReactionEventType {
    BondFormation,
    BondBreaking,
    ElectronTransfer,
    ConformationalChange,
}

impl MolecularDynamicsEngine {
    /// Get a comprehensive snapshot of the current molecular dynamics state
    /// This method provides all data needed for real-time visualization
    pub fn get_snapshot(&self) -> MDSnapshot {
        let particle_positions: Vec<[f64; 3]> = self.system.particles.iter()
            .map(|p| p.position)
            .collect();
            
        let particle_velocities: Vec<[f64; 3]> = self.system.particles.iter()
            .map(|p| p.velocity)
            .collect();
            
        let particle_forces: Vec<[f64; 3]> = self.system.particles.iter()
            .map(|p| p.force)
            .collect();
            
        let particle_types: Vec<u32> = self.system.particles.iter()
            .map(|p| p.type_id)
            .collect();
            
        let particle_masses: Vec<f64> = self.system.particles.iter()
            .map(|p| p.mass)
            .collect();

        // Detect active bonds based on distance and force thresholds
        let bonds = self.detect_bonds();
        
        // Get neighbor pairs from the neighbor list
        let neighbor_pairs = self.get_neighbor_pairs();
        
        // Detect quantum regions (atoms with strong quantum effects)
        let quantum_regions = self.detect_quantum_regions();
        
        // Identify molecular clusters
        let molecular_clusters = self.identify_molecular_clusters(&bonds);
        
        // Detect recent reaction events
        let reaction_events = self.detect_reaction_events();

        MDSnapshot {
            step: self.step,
            time: self.system.time,
            particle_positions,
            particle_velocities,
            particle_forces,
            particle_types,
            particle_masses,
            bonds,
            neighbor_pairs,
            properties: self.get_properties(),
            quantum_regions,
            molecular_clusters,
            reaction_events,
        }
    }
    
    /// Detect active chemical bonds based on distance and interaction strength
    fn detect_bonds(&self) -> Vec<(usize, usize, f64)> {
        let mut bonds = Vec::new();
        
        for i in 0..self.system.particles.len() {
            for j in (i + 1)..self.system.particles.len() {
                let pos_i = self.system.particles[i].position;
                let pos_j = self.system.particles[j].position;
                
                let dx = pos_i[0] - pos_j[0];
                let dy = pos_i[1] - pos_j[1];
                let dz = pos_i[2] - pos_j[2];
                let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                
                // Bond detection based on Lennard-Jones parameters and distance
                let bond_threshold = self.lj_params.sigma * 1.2; // Typical bond length
                
                if distance < bond_threshold {
                    // Calculate bond strength based on Lennard-Jones potential
                    let bond_strength = self.calculate_bond_strength(distance);
                    if bond_strength > 0.1 { // Minimum bond strength threshold
                        bonds.push((i, j, bond_strength));
                    }
                }
            }
        }
        
        bonds
    }
    
    /// Calculate bond strength based on interatomic potential
    fn calculate_bond_strength(&self, distance: f64) -> f64 {
        // Use Lennard-Jones potential derivative to estimate bond strength
        let sigma = self.lj_params.sigma;
        let epsilon = self.lj_params.epsilon;
        
        if distance < sigma * 0.5 || distance > sigma * 2.0 {
            return 0.0; // No bond outside reasonable range
        }
        
        let sr6 = (sigma / distance).powi(6);
        let potential = 4.0 * epsilon * (sr6.powi(2) - sr6);
        
        // Bond strength is related to the depth of the potential well
        (-potential / epsilon).max(0.0).min(1.0)
    }
    
    /// Get neighbor pairs from the neighbor list for visualization
    fn get_neighbor_pairs(&self) -> Vec<(usize, usize, f64)> {
        let mut pairs = Vec::new();
        
        for i in 0..self.system.particles.len() {
            let neighbors = self.neighbor_list.get_neighbors(i);
            for &j in neighbors {
                if i < j { // Avoid duplicate pairs
                    let pos_i = self.system.particles[i].position;
                    let pos_j = self.system.particles[j].position;
                    
                    let dx = pos_i[0] - pos_j[0];
                    let dy = pos_i[1] - pos_j[1];
                    let dz = pos_i[2] - pos_j[2];
                    let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                    
                    pairs.push((i, j, distance));
                }
            }
        }
        
        pairs
    }
    
    /// Detect regions requiring quantum treatment based on electronic structure
    fn detect_quantum_regions(&self) -> Vec<usize> {
        let mut quantum_regions = Vec::new();
        
        for (i, particle) in self.system.particles.iter().enumerate() {
            // Heuristic: atoms with high kinetic energy or in reactive environments
            let kinetic_energy = 0.5 * particle.mass * 
                (particle.velocity[0].powi(2) + particle.velocity[1].powi(2) + particle.velocity[2].powi(2));
            
            // Atoms with high kinetic energy may need quantum treatment
            let thermal_energy = 1.380649e-23 * 300.0; // kT at room temperature
            if kinetic_energy > 10.0 * thermal_energy {
                quantum_regions.push(i);
            }
            
            // Atoms involved in multiple bonds may need quantum treatment
            let bond_count = self.count_bonds_for_particle(i);
            if bond_count >= 3 {
                quantum_regions.push(i);
            }
        }
        
        quantum_regions
    }
    
    /// Count the number of bonds for a specific particle
    fn count_bonds_for_particle(&self, particle_index: usize) -> usize {
        let bonds = self.detect_bonds();
        bonds.iter()
            .filter(|(i, j, _)| *i == particle_index || *j == particle_index)
            .count()
    }
    
    /// Identify molecular clusters (groups of bonded atoms)
    fn identify_molecular_clusters(&self, bonds: &[(usize, usize, f64)]) -> Vec<Vec<usize>> {
        let mut clusters = Vec::new();
        let mut visited = vec![false; self.system.particles.len()];
        
        for i in 0..self.system.particles.len() {
            if !visited[i] {
                let mut cluster = Vec::new();
                self.dfs_cluster(i, bonds, &mut visited, &mut cluster);
                if !cluster.is_empty() {
                    clusters.push(cluster);
                }
            }
        }
        
        clusters
    }
    
    /// Depth-first search to identify connected atoms in a molecular cluster
    fn dfs_cluster(&self, node: usize, bonds: &[(usize, usize, f64)], visited: &mut [bool], cluster: &mut Vec<usize>) {
        visited[node] = true;
        cluster.push(node);
        
        for &(i, j, _) in bonds {
            if i == node && !visited[j] {
                self.dfs_cluster(j, bonds, visited, cluster);
            } else if j == node && !visited[i] {
                self.dfs_cluster(i, bonds, visited, cluster);
            }
        }
    }
    
    /// Detect recent chemical reaction events
    fn detect_reaction_events(&self) -> Vec<ReactionEvent> {
        // This is a simplified implementation
        // In a full system, you would track bond changes over time
        let mut events = Vec::new();
        
        // For demonstration, detect high-energy configurations that might indicate reactions
        for (i, particle) in self.system.particles.iter().enumerate() {
            let force_magnitude = (particle.force[0].powi(2) + 
                                 particle.force[1].powi(2) + 
                                 particle.force[2].powi(2)).sqrt();
            
            // High forces might indicate bond breaking or formation
            if force_magnitude > 1e-10 { // Threshold for reaction detection
                events.push(ReactionEvent {
                    event_type: ReactionEventType::ConformationalChange,
                    participants: vec![i],
                    time: self.system.time,
                    energy_change: force_magnitude * 1e-12, // Approximate energy scale
                });
            }
        }
        
        events
    }
    
    /// Get quantum-enhanced molecular dynamics snapshot with quantum field integration
    pub fn get_quantum_enhanced_snapshot(&self) -> MDSnapshot {
        let mut snapshot = self.get_snapshot();
        
        // Enhanced quantum region detection using quantum field data
        // This would integrate with the quantum field system for more accurate detection
        snapshot.quantum_regions = self.detect_enhanced_quantum_regions();
        
        snapshot
    }
    
    /// Enhanced quantum region detection using quantum field information
    fn detect_enhanced_quantum_regions(&self) -> Vec<usize> {
        // This method would integrate with the quantum field system
        // For now, use the basic heuristic approach
        self.detect_quantum_regions()
    }
}