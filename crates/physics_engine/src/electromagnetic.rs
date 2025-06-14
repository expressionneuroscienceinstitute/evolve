//! Electromagnetic Field Solver
//! 
//! Implements simplified FDTD (Finite Difference Time Domain) solver for
//! electric and magnetic fields with Coulomb and Lorentz forces

use nalgebra::Vector3;
use anyhow::Result;
use crate::{PhysicsState, PhysicsConstants};

/// Electromagnetic field solver
pub struct EMSolver {
    pub grid_size: usize,
    pub cell_size: f64,
    pub electric_field: Vec<Vector3<f64>>,
    pub magnetic_field: Vec<Vector3<f64>>,
    pub charge_density: Vec<f64>,
    pub current_density: Vec<Vector3<f64>>,
}

impl EMSolver {
    pub fn new() -> Self {
        let grid_size = 64; // Default grid resolution
        let cell_size = 1e6; // 1 million meters per cell
        let total_cells = grid_size * grid_size * grid_size;
        
        Self {
            grid_size,
            cell_size,
            electric_field: vec![Vector3::zeros(); total_cells],
            magnetic_field: vec![Vector3::zeros(); total_cells],
            charge_density: vec![0.0; total_cells],
            current_density: vec![Vector3::zeros(); total_cells],
        }
    }

    /// Update electromagnetic fields and forces
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Update charge and current densities from particle positions
        self.update_sources(states);
        
        // Solve Maxwell's equations (simplified)
        self.update_fields(constants)?;
        
        // Apply electromagnetic forces to particles
        self.apply_forces(states, constants)?;
        
        Ok(())
    }

    /// Update charge and current densities from particle positions
    fn update_sources(&mut self, states: &[PhysicsState]) {
        // Clear previous sources
        self.charge_density.fill(0.0);
        self.current_density.fill(Vector3::zeros());
        
        for state in states {
            let cell_index = self.position_to_cell_index(&state.position);
            if let Some(index) = cell_index {
                // Add charge density
                self.charge_density[index] += state.charge / (self.cell_size.powi(3));
                
                // Add current density J = ρv
                let j = state.charge * state.velocity / (self.cell_size.powi(3));
                self.current_density[index] += j;
            }
        }
    }

    /// Convert 3D position to 1D cell index
    fn position_to_cell_index(&self, position: &Vector3<f64>) -> Option<usize> {
        let x = (position.x / self.cell_size + self.grid_size as f64 / 2.0) as i32;
        let y = (position.y / self.cell_size + self.grid_size as f64 / 2.0) as i32;
        let z = (position.z / self.cell_size + self.grid_size as f64 / 2.0) as i32;
        
        if x >= 0 && y >= 0 && z >= 0 
            && x < self.grid_size as i32 
            && y < self.grid_size as i32 
            && z < self.grid_size as i32 
        {
            Some((x as usize) * self.grid_size * self.grid_size 
                 + (y as usize) * self.grid_size 
                 + (z as usize))
        } else {
            None
        }
    }

    /// Update electric and magnetic fields using simplified Maxwell equations
    fn update_fields(&mut self, constants: &PhysicsConstants) -> Result<()> {
        // Simplified electrostatic field calculation
        // In a full implementation, this would use FDTD with proper boundary conditions
        
        for i in 0..self.electric_field.len() {
            // Simple Coulomb field from local charge density
            if self.charge_density[i].abs() > 1e-30 {
                let field_magnitude = self.charge_density[i] / (4.0 * std::f64::consts::PI * constants.epsilon_0);
                // For simplicity, assume radial field (would need proper gradient calculation)
                self.electric_field[i] = Vector3::new(field_magnitude, 0.0, 0.0);
            }
        }
        
        // Magnetic field from current (simplified Biot-Savart)
        for i in 0..self.magnetic_field.len() {
            let j = &self.current_density[i];
            if j.magnitude() > 1e-30 {
                let b_magnitude = constants.mu_0 * j.magnitude() / (4.0 * std::f64::consts::PI);
                // Simplified: B perpendicular to J
                self.magnetic_field[i] = Vector3::new(0.0, 0.0, b_magnitude);
            }
        }
        
        Ok(())
    }

    /// Apply electromagnetic forces to charged particles
    fn apply_forces(&self, states: &mut [PhysicsState], _constants: &PhysicsConstants) -> Result<()> {
        for state in states.iter_mut() {
            if state.charge.abs() < 1e-30 {
                continue; // Skip neutral particles
            }
            
            let cell_index = self.position_to_cell_index(&state.position);
            if let Some(index) = cell_index {
                let e_field = &self.electric_field[index];
                let b_field = &self.magnetic_field[index];
                
                // Lorentz force: F = q(E + v × B)
                let electric_force = state.charge * e_field;
                let magnetic_force = state.charge * state.velocity.cross(b_field);
                let total_force = electric_force + magnetic_force;
                
                // Add to acceleration (will be integrated by classical solver)
                state.acceleration += total_force / state.mass;
            }
        }
        
        Ok(())
    }

    /// Calculate direct Coulomb forces between charged particles
    pub fn coulomb_forces(&self, states: &[PhysicsState], constants: &PhysicsConstants) 
        -> Vec<Vector3<f64>> 
    {
        let n = states.len();
        let mut forces = vec![Vector3::zeros(); n];
        
        for i in 0..n {
            for j in (i+1)..n {
                if states[i].charge.abs() < 1e-30 || states[j].charge.abs() < 1e-30 {
                    continue;
                }
                
                let r_vec = states[j].position - states[i].position;
                let r = r_vec.magnitude();
                
                if r < 1e-10 {
                    continue; // Avoid singularity
                }
                
                let force_magnitude = constants.coulomb_force(states[i].charge, states[j].charge, r);
                let direction = r_vec.normalize();
                let force = -force_magnitude * direction;
                
                forces[i] += force;
                forces[j] -= force; // Newton's third law
            }
        }
        
        forces
    }

    /// Calculate electromagnetic field energy
    pub fn field_energy(&self, constants: &PhysicsConstants) -> f64 {
        let mut energy = 0.0;
        let cell_volume = self.cell_size.powi(3);
        
        for i in 0..self.electric_field.len() {
            let e_field = &self.electric_field[i];
            let b_field = &self.magnetic_field[i];
            
            // Energy density: u = ½(ε₀E² + B²/μ₀)
            let e_energy = 0.5 * constants.epsilon_0 * e_field.magnitude_squared();
            let b_energy = 0.5 * b_field.magnitude_squared() / constants.mu_0;
            
            energy += (e_energy + b_energy) * cell_volume;
        }
        
        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_em_solver_creation() {
        let solver = EMSolver::new();
        assert_eq!(solver.grid_size, 64);
        assert_eq!(solver.electric_field.len(), 64 * 64 * 64);
    }

    #[test]
    fn test_position_to_cell_index() {
        let solver = EMSolver::new();
        
        // Test center position
        let center = Vector3::zeros();
        let index = solver.position_to_cell_index(&center);
        assert!(index.is_some());
        
        // Test out of bounds
        let far = Vector3::new(1e12, 1e12, 1e12);
        let index = solver.position_to_cell_index(&far);
        assert!(index.is_none());
    }

    #[test]
    fn test_coulomb_forces() {
        let solver = EMSolver::new();
        let constants = PhysicsConstants::default();
        
        // Two point charges
        let states = vec![
            PhysicsState {
                position: Vector3::new(0.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                acceleration: Vector3::zeros(),
                mass: constants.m_e,
                charge: constants.e,
                temperature: 0.0,
                entropy: 0.0,
            },
            PhysicsState {
                position: Vector3::new(1.0, 0.0, 0.0),
                velocity: Vector3::zeros(),
                acceleration: Vector3::zeros(),
                mass: constants.m_e,
                charge: constants.e,
                temperature: 0.0,
                entropy: 0.0,
            },
        ];
        
        let forces = solver.coulomb_forces(&states, &constants);
        
        // Forces should be repulsive (same charge)
        assert!(forces[0].x < 0.0); // Force on first charge points left
        assert!(forces[1].x > 0.0); // Force on second charge points right
        
        // Magnitudes should be equal
        assert_relative_eq!(forces[0].magnitude(), forces[1].magnitude(), epsilon = 1e-10);
    }
}