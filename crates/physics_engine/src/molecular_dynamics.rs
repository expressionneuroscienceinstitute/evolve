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