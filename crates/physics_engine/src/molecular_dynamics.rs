//! # Physics Engine: Molecular Dynamics Module
//!
//! This module provides a framework for running molecular dynamics (MD) simulations.
//! MD is a computational method for analyzing the physical movements of atoms and
//! molecules. The atoms and molecules are allowed to interact for a fixed period of
//! time, giving a view of the dynamical evolution of the system.

use anyhow::Result;
use nalgebra::Vector3;
use crate::particles::Particle;
use crate::interactions::ForceField;

/// A simulator for running molecular dynamics simulations.
/// This implementation uses the Velocity Verlet integration algorithm, which is a
/// common choice for MD due to its good energy conservation properties.
pub struct MolecularDynamics {
    pub particles: Vec<Particle>,
    pub force_field: ForceField,
    pub time_step: f64,
}

impl MolecularDynamics {
    /// Creates a new molecular dynamics simulator.
    pub fn new(particles: Vec<Particle>, force_field: ForceField, time_step: f64) -> Self {
        MolecularDynamics {
            particles,
            force_field,
            time_step,
        }
    }

    /// Performs a single step of the molecular dynamics simulation.
    pub fn step(&mut self) -> Result<()> {
        // 1. Calculate the current forces on all particles.
        let forces = self.force_field.calculate_forces(&self.particles);

        // 2. Update particle positions based on current velocity and acceleration.
        // v(t + dt/2) = v(t) + a(t) * dt / 2
        // x(t + dt) = x(t) + v(t + dt/2) * dt
        for (i, particle) in self.particles.iter_mut().enumerate() {
            let acceleration = forces[i] / particle.mass;
            particle.velocity += 0.5 * acceleration * self.time_step;
            particle.position += particle.velocity * self.time_step;
        }

        // 3. Recalculate forces at the new positions.
        let new_forces = self.force_field.calculate_forces(&self.particles);

        // 4. Update velocities with the new acceleration.
        // v(t + dt) = v(t + dt/2) + a(t + dt) * dt / 2
        for (i, particle) in self.particles.iter_mut().enumerate() {
            let new_acceleration = new_forces[i] / particle.mass;
            particle.velocity += 0.5 * new_acceleration * self.time_step;
        }

        Ok(())
    }
}

/// A convenience function to run a single step of a molecular dynamics simulation.
pub fn step_molecular_dynamics(
    particles: &mut Vec<Particle>,
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