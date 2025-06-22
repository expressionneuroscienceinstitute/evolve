//! Force Fields for Molecular Dynamics
//!
//! Implements interaction potentials for molecular dynamics simulations.
//! All potentials must be physically meaningful and emerge from quantum/atomic layer.
//!
//! References:
//! - CPVR Chapter 17: Force Fields and Interaction Potentials
//! - IMPETUS Tutorial: Lennard-Jones Liquid Model
//! - https://cmlab.engr.uconn.edu/impetus/tutorialfiles/tutorial1.html

use crate::molecular_dynamics::System;

/// Lennard-Jones potential parameters
#[derive(Debug, Clone)]
pub struct LennardJonesParams {
    /// Depth of the potential well (Joules)
    pub epsilon: f64,
    /// Distance at which potential is zero (meters)
    pub sigma: f64,
    /// Cutoff distance for the potential (meters)
    pub r_cut: f64,
}

impl Default for LennardJonesParams {
    fn default() -> Self {
        Self {
            epsilon: 1.0, // Default energy scale
            sigma: 1.0,   // Default length scale
            r_cut: 2.5,   // Standard cutoff for LJ potential
        }
    }
}

/// Calculates the Lennard-Jones potential energy between two particles
/// 
/// V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
/// 
/// This is the standard 12-6 Lennard-Jones potential used in molecular dynamics.
/// The potential is truncated at r_cut for computational efficiency.
pub fn lennard_jones_energy(
    r: f64,
    params: &LennardJonesParams,
) -> f64 {
    if r >= params.r_cut {
        return 0.0;
    }
    
    let sr6 = (params.sigma / r).powi(6);
    let sr12 = sr6.powi(2);
    
    4.0 * params.epsilon * (sr12 - sr6)
}

/// Calculates the force between two particles due to Lennard-Jones potential
/// 
/// F(r) = -dV/dr = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
/// 
/// The force is the negative gradient of the potential energy.
pub fn lennard_jones_force(
    r: f64,
    params: &LennardJonesParams,
) -> f64 {
    if r >= params.r_cut {
        return 0.0;
    }
    
    let sr6 = (params.sigma / r).powi(6);
    let sr12 = sr6.powi(2);
    
    24.0 * params.epsilon / r * (2.0 * sr12 - sr6)
}

/// Applies Lennard-Jones forces to all particle pairs in the system
/// 
/// This function calculates pairwise forces between all particles within
/// the cutoff distance and updates the force vectors accordingly.
pub fn apply_lennard_jones_forces(
    system: &mut System,
    params: &LennardJonesParams,
) {
    let n_particles = system.particles.len();
    
    // Reset all forces to zero
    for particle in &mut system.particles {
        particle.force = [0.0; 3];
    }
    
    // Calculate pairwise forces
    for i in 0..n_particles {
        for j in (i + 1)..n_particles {
            let p1 = &system.particles[i];
            let p2 = &system.particles[j];
            
            // Calculate distance vector
            let mut r_vec = [0.0; 3];
            let mut r_squared = 0.0;
            
            for d in 0..3 {
                let dx = p2.position[d] - p1.position[d];
                r_vec[d] = dx;
                r_squared += dx * dx;
            }
            
            let r = r_squared.sqrt();
            
            if r < params.r_cut && r > 0.0 {
                let force_magnitude = lennard_jones_force(r, params);
                
                // Apply forces to both particles (Newton's 3rd law)
                for d in 0..3 {
                    let force_component = force_magnitude * r_vec[d] / r;
                    system.particles[i].force[d] -= force_component;
                    system.particles[j].force[d] += force_component;
                }
            }
        }
    }
} 