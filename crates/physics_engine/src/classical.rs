//! Classical Mechanics Solver
//! 
//! Implements leap-frog integration with relativistic corrections when v ≥ 0.1c

use nalgebra::Vector3;
use anyhow::Result;
use crate::{PhysicsState, PhysicsConstants};

/// Classical mechanics solver using leap-frog integration
pub struct ClassicalSolver {
    pub time_step: f64,
    pub adaptive_dt: bool,
    pub max_velocity: f64,
}

impl ClassicalSolver {
    pub fn new(time_step: f64) -> Self {
        Self {
            time_step,
            adaptive_dt: true,
            max_velocity: 0.9 * 299_792_458.0, // 0.9c limit
        }
    }

    /// Advance classical mechanics by one time step using leap-frog integration
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Calculate forces for all particles
        let forces = self.calculate_forces(states, constants)?;
        
        // Leap-frog integration
        for (state, force) in states.iter_mut().zip(forces.iter()) {
            self.integrate_particle(state, force, constants)?;
        }
        
        Ok(())
    }

    /// Calculate all forces acting on particles
    fn calculate_forces(&self, states: &[PhysicsState], constants: &PhysicsConstants)
        -> Result<Vec<Vector3<f64>>> 
    {
        let n = states.len();
        let mut forces = vec![Vector3::zeros(); n];
        
        // Calculate pairwise gravitational forces
        for i in 0..n {
            for j in (i+1)..n {
                let force = self.gravitational_force(&states[i], &states[j], constants);
                forces[i] += force;
                forces[j] -= force; // Newton's third law
            }
        }
        
        Ok(forces)
    }

    /// Calculate gravitational force between two particles
    fn gravitational_force(&self, state1: &PhysicsState, state2: &PhysicsState, 
                          constants: &PhysicsConstants) -> Vector3<f64> 
    {
        let r_vec = state2.position - state1.position;
        let r = r_vec.magnitude();
        
        if r < 1e-10 {
            return Vector3::zeros(); // Avoid singularity
        }
        
        let force_magnitude = constants.g * state1.mass * state2.mass / (r * r);
        force_magnitude * r_vec.normalize()
    }

    /// Integrate single particle using leap-frog method with relativistic corrections
    fn integrate_particle(&self, state: &mut PhysicsState, force: &Vector3<f64>, 
                         constants: &PhysicsConstants) -> Result<()> 
    {
        let dt = self.adaptive_timestep(state, constants);
        
        // Current velocity magnitude
        let v = state.velocity.magnitude();
        
        if constants.is_relativistic(v) {
            // Relativistic integration
            self.integrate_relativistic(state, force, constants, dt)?;
        } else {
            // Classical leap-frog integration
            self.integrate_classical(state, force, dt)?;
        }
        
        // Enforce speed limit
        if state.velocity.magnitude() > self.max_velocity {
            state.velocity = state.velocity.normalize() * self.max_velocity;
        }
        
        Ok(())
    }

    /// Classical leap-frog integration
    fn integrate_classical(&self, state: &mut PhysicsState, force: &Vector3<f64>, dt: f64) 
        -> Result<()> 
    {
        // Calculate acceleration
        let acceleration = *force / state.mass;
        
        // Leap-frog algorithm:
        // v(t + dt/2) = v(t - dt/2) + a(t) * dt
        // x(t + dt) = x(t) + v(t + dt/2) * dt
        
        // Update velocity (assuming we store v at half-step intervals)
        state.velocity += acceleration * dt;
        
        // Update position
        state.position += state.velocity * dt;
        
        // Store acceleration for diagnostics
        state.acceleration = acceleration;
        
        Ok(())
    }

    /// Relativistic integration using 4-momentum
    fn integrate_relativistic(&self, state: &mut PhysicsState, force: &Vector3<f64>, 
                             constants: &PhysicsConstants, dt: f64) -> Result<()> 
    {
        let v = state.velocity.magnitude();
        let gamma = constants.lorentz_factor(v);
        
        // Relativistic momentum p = γmv
        let momentum = gamma * state.mass * state.velocity;
        
        // dp/dt = F (in special relativity)
        let new_momentum = momentum + *force * dt;
        
        // Calculate new velocity from relativistic momentum
        // p = γmv where γ = 1/√(1 - v²/c²)
        // Solving for v: v = p / √(m² + p²/c²)
        let p_magnitude = new_momentum.magnitude();
        let denominator = (state.mass * state.mass + 
                          (p_magnitude * p_magnitude) / (constants.c * constants.c)).sqrt();
        
        if denominator > 1e-30 {
            state.velocity = new_momentum / denominator;
        }
        
        // Update position
        state.position += state.velocity * dt;
        
        // Calculate proper acceleration
        let new_gamma = constants.lorentz_factor(state.velocity.magnitude());
        state.acceleration = *force / (new_gamma * state.mass);
        
        Ok(())
    }

    /// Calculate adaptive timestep based on velocity and acceleration
    fn adaptive_timestep(&self, state: &PhysicsState, constants: &PhysicsConstants) -> f64 {
        if !self.adaptive_dt {
            return self.time_step;
        }
        
        let v = state.velocity.magnitude();
        let a = state.acceleration.magnitude();
        
        // Courant-Friedrichs-Lewy condition for stability
        let dt_velocity = if v > 1e-10 { 0.1 / v } else { self.time_step };
        let dt_acceleration = if a > 1e-10 { (0.1 / a).sqrt() } else { self.time_step };
        
        // Use minimum for stability
        let dt_adaptive = dt_velocity.min(dt_acceleration).min(self.time_step);
        
        // Extra restriction for relativistic particles
        if constants.is_relativistic(v) {
            dt_adaptive.min(1e-12) // Very small timestep for relativistic motion
        } else {
            dt_adaptive
        }
    }

    /// Calculate kinetic energy (classical or relativistic)
    pub fn kinetic_energy(&self, state: &PhysicsState, constants: &PhysicsConstants) -> f64 {
        let v = state.velocity.magnitude();
        
        if constants.is_relativistic(v) {
            // Relativistic kinetic energy: T = (γ - 1)mc²
            let gamma = constants.lorentz_factor(v);
            (gamma - 1.0) * state.mass * constants.c * constants.c
        } else {
            // Classical kinetic energy: T = ½mv²
            0.5 * state.mass * v * v
        }
    }

    /// Calculate total mechanical energy
    pub fn total_energy(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> f64 {
        let mut total = 0.0;
        
        // Kinetic energy
        for state in states {
            total += self.kinetic_energy(state, constants);
        }
        
        // Gravitational potential energy
        for i in 0..states.len() {
            for j in (i+1)..states.len() {
                let r = (states[i].position - states[j].position).magnitude();
                if r > 1e-10 {
                    total -= constants.g * states[i].mass * states[j].mass / r;
                }
            }
        }
        
        total
    }

    /// Calculate total momentum
    pub fn total_momentum(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Vector3<f64> {
        let mut total = Vector3::zeros();
        
        for state in states {
            let v = state.velocity.magnitude();
            if constants.is_relativistic(v) {
                let gamma = constants.lorentz_factor(v);
                total += gamma * state.mass * state.velocity;
            } else {
                total += state.mass * state.velocity;
            }
        }
        
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solver_creation() {
        let solver = ClassicalSolver::new(1e-6);
        assert_relative_eq!(solver.time_step, 1e-6);
        assert!(solver.adaptive_dt);
    }

    #[test]
    fn test_gravitational_force() {
        let solver = ClassicalSolver::new(1e-6);
        let constants = PhysicsConstants::default();
        
        let state1 = PhysicsState {
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            mass: constants.m_earth,
            charge: 0.0,
            temperature: 288.0,
            entropy: 0.0,
        };
        
        let state2 = PhysicsState {
            position: Vector3::new(constants.au, 0.0, 0.0),
            velocity: Vector3::new(0.0, 29780.0, 0.0), // Earth orbital speed
            acceleration: Vector3::zeros(),
            mass: constants.m_sun,
            charge: 0.0,
            temperature: 5778.0,
            entropy: 0.0,
        };
        
        let force = solver.gravitational_force(&state1, &state2, &constants);
        
        // Expected force calculation
        let expected_magnitude = constants.g * constants.m_earth * constants.m_sun 
                                / (constants.au * constants.au);
        
        assert_relative_eq!(force.magnitude(), expected_magnitude, epsilon = 1e-3);
    }

    #[test]
    fn test_energy_conservation() {
        let mut solver = ClassicalSolver::new(1e-3);
        let constants = PhysicsConstants::default();
        
        // Simple two-body system (Earth-Moon)
        let mut states = vec![
            PhysicsState {
                position: Vector3::new(0.0, 0.0, 0.0),
                velocity: Vector3::new(0.0, 1000.0, 0.0),
                acceleration: Vector3::zeros(),
                mass: constants.m_earth,
                charge: 0.0,
                temperature: 288.0,
                entropy: 0.0,
            },
            PhysicsState {
                position: Vector3::new(3.844e8, 0.0, 0.0),
                velocity: Vector3::new(0.0, -1022.0, 0.0),
                acceleration: Vector3::zeros(),
                mass: 7.342e22, // Moon mass
                charge: 0.0,
                temperature: 250.0,
                entropy: 0.0,
            },
        ];
        
        let initial_energy = solver.total_energy(&states, &constants);
        
        // Run for a few steps
        for _ in 0..10 {
            solver.step(&mut states, &constants).unwrap();
        }
        
        let final_energy = solver.total_energy(&states, &constants);
        
        // Energy should be conserved to within numerical precision
        assert_relative_eq!(initial_energy, final_energy, epsilon = 1e-6);
    }

    #[test]
    fn test_relativistic_integration() {
        let mut solver = ClassicalSolver::new(1e-12);
        let constants = PhysicsConstants::default();
        
        let mut state = PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(0.5 * constants.c, 0.0, 0.0), // Relativistic velocity
            acceleration: Vector3::zeros(),
            mass: constants.m_e,
            charge: -constants.e,
            temperature: 1e6,
            entropy: 0.0,
        };
        
        let force = Vector3::new(1e-15, 0.0, 0.0); // Small force
        
        // Test that relativistic integration doesn't exceed c
        solver.integrate_particle(&mut state, &force, &constants).unwrap();
        
        assert!(state.velocity.magnitude() < constants.c);
    }
}