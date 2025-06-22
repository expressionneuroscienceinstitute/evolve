use crate::molecular_dynamics::System;

/// Calculate kinetic energy of the system
/// 
/// K = (1/2) * Σ(m_i * v_i^2)
pub fn kinetic_energy(system: &System) -> f64 {
    system.particles.iter()
        .map(|p| {
            let v_squared = p.velocity[0].powi(2) + p.velocity[1].powi(2) + p.velocity[2].powi(2);
            0.5 * p.mass * v_squared
        })
        .sum()
}

/// Calculate temperature from kinetic energy
/// 
/// T = (2/3) * K / (N * k_B)
/// where N is the number of particles and k_B is Boltzmann's constant
pub fn temperature(system: &System) -> f64 {
    let n_particles = system.particles.len() as f64;
    let k_b = 1.380649e-23; // Boltzmann's constant (J/K)
    
    if n_particles > 0.0 {
        (2.0 / 3.0) * kinetic_energy(system) / (n_particles * k_b)
    } else {
        0.0
    }
}

/// Standard velocity Verlet integration step
/// 
/// This is the standard second-order symplectic integrator for molecular dynamics.
/// It conserves energy well and is time-reversible.
pub fn velocity_verlet_step(system: &mut System) {
    let dt = system.dt;
    
    // First half-step: update velocities
    for particle in &mut system.particles {
        for d in 0..3 {
            particle.velocity[d] += 0.5 * dt * particle.force[d] / particle.mass;
            particle.position[d] += dt * particle.velocity[d];
        }
    }
    
    // Note: Forces should be recalculated here by the calling code
    // This function assumes forces are already updated
    
    // Second half-step: update velocities
    for particle in &mut system.particles {
        for d in 0..3 {
            particle.velocity[d] += 0.5 * dt * particle.force[d] / particle.mass;
        }
    }
    
    system.time += dt;
}

/// Velocity Verlet integration with multiple time steps (Ruth's 4th order method)
/// 
/// This is a higher-order symplectic integrator that provides better
/// energy conservation than the standard velocity Verlet method.
pub fn velocity_verlet_ruth4_step(system: &mut System) {
    let dt = system.dt;
    
    // Ruth's 4th order coefficients
    let w1 = 1.0 / (2.0 - 2.0_f64.powf(1.0/3.0));
    let w0 = 1.0 - 2.0 * w1;
    
    // First substep
    for particle in &mut system.particles {
        for d in 0..3 {
            particle.velocity[d] += w1 * dt * particle.force[d] / particle.mass;
            particle.position[d] += w1 * dt * particle.velocity[d];
        }
    }
    
    // Second substep
    for particle in &mut system.particles {
        for d in 0..3 {
            particle.velocity[d] += w0 * dt * particle.force[d] / particle.mass;
            particle.position[d] += w0 * dt * particle.velocity[d];
        }
    }
    
    // Third substep
    for particle in &mut system.particles {
        for d in 0..3 {
            particle.velocity[d] += w1 * dt * particle.force[d] / particle.mass;
            particle.position[d] += w1 * dt * particle.velocity[d];
        }
    }
    
    system.time += dt;
} 