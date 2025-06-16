//! # Physics Engine: Molecular Dynamics Module
//!
//! This module provides a framework for running molecular dynamics (MD) simulations.
//! MD is a computational method for analyzing the physical movements of atoms and
//! molecules. The atoms and molecules are allowed to interact for a fixed period of
//! time, giving a view of the dynamical evolution of the system.

use anyhow::Result;
use nalgebra::Vector3;

use crate::{PhysicsState, constants::{ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY}};

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
    particles: &mut Vec<PhysicsState>,
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