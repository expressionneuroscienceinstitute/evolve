//! Electromagnetic Field Solver
//! 
//! Implements simplified FDTD (Finite Difference Time Domain) solver for
//! electric and magnetic fields with Coulomb and Lorentz forces

use nalgebra::Vector3;
use anyhow::{Result, anyhow};
use crate::{PhysicsState, PhysicsConstants};

/// Maximum allowed grid size to prevent memory exhaustion
const MAX_GRID_SIZE: usize = 1024;

/// Minimum cell size to prevent numerical instability
const MIN_CELL_SIZE: f64 = 1e-15;

/// Error types for electromagnetic solver
#[derive(Debug, thiserror::Error)]
pub enum EMError {
    #[error("Grid size {0} exceeds maximum allowed size {}", MAX_GRID_SIZE)]
    GridSizeTooLarge(usize),
    #[error("Cell size {0} is below minimum allowed size {}", MIN_CELL_SIZE)]
    CellSizeTooSmall(f64),
    #[error("Invalid input state: {0}")]
    InvalidState(String),
    #[error("Position calculation overflow")]
    PositionOverflow,
}

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
        Self::with_grid_params(64, 1e6).expect("Default parameters should be valid")
    }

    /// Create EMSolver with validated parameters
    pub fn with_grid_params(grid_size: usize, cell_size: f64) -> Result<Self, EMError> {
        if grid_size > MAX_GRID_SIZE {
            return Err(EMError::GridSizeTooLarge(grid_size));
        }
        
        if cell_size < MIN_CELL_SIZE {
            return Err(EMError::CellSizeTooSmall(cell_size));
        }

        let total_cells = grid_size
            .checked_mul(grid_size)
            .and_then(|intermediate| intermediate.checked_mul(grid_size))
            .ok_or_else(|| EMError::GridSizeTooLarge(grid_size))?;
        
        Ok(Self {
            grid_size,
            cell_size,
            electric_field: vec![Vector3::zeros(); total_cells],
            magnetic_field: vec![Vector3::zeros(); total_cells],
            charge_density: vec![0.0; total_cells],
            current_density: vec![Vector3::zeros(); total_cells],
        })
    }

    /// Update electromagnetic fields and forces
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Validate inputs
        self.validate_states(states)?;
        self.validate_constants(constants)?;
        
        // Update charge and current densities from particle positions
        self.update_sources(states)?;
        
        // Solve Maxwell's equations (simplified)
        self.update_fields(constants)?;
        
        // Apply electromagnetic forces to particles
        self.apply_forces(states, constants)?;
        
        Ok(())
    }

    /// Validate input states for security issues
    fn validate_states(&self, states: &[PhysicsState]) -> Result<(), EMError> {
        for (i, state) in states.iter().enumerate() {
            if !state.position.iter().all(|&x| x.is_finite()) {
                return Err(EMError::InvalidState(format!("Non-finite position in state {}", i)));
            }
            if !state.velocity.iter().all(|&x| x.is_finite()) {
                return Err(EMError::InvalidState(format!("Non-finite velocity in state {}", i)));
            }
            if !state.mass.is_finite() || state.mass <= 0.0 {
                return Err(EMError::InvalidState(format!("Invalid mass in state {}", i)));
            }
            if !state.charge.is_finite() {
                return Err(EMError::InvalidState(format!("Non-finite charge in state {}", i)));
            }
        }
        Ok(())
    }

    /// Validate physics constants
    fn validate_constants(&self, constants: &PhysicsConstants) -> Result<(), EMError> {
        if !constants.epsilon_0.is_finite() || constants.epsilon_0 <= 0.0 {
            return Err(EMError::InvalidState("Invalid epsilon_0".to_string()));
        }
        if !constants.mu_0.is_finite() || constants.mu_0 <= 0.0 {
            return Err(EMError::InvalidState("Invalid mu_0".to_string()));
        }
        Ok(())
    }

    /// Update charge and current densities from particle positions
    fn update_sources(&mut self, states: &[PhysicsState]) -> Result<()> {
        // Clear previous sources
        self.charge_density.fill(0.0);
        self.current_density.fill(Vector3::zeros());
        
        // Validate cell volume to prevent division by zero
        let cell_volume = self.cell_size.powi(3);
        if !cell_volume.is_finite() || cell_volume <= 0.0 {
            return Err(anyhow!("Invalid cell volume: {}", cell_volume));
        }
        
        for state in states {
            let cell_index = self.position_to_cell_index(&state.position)?;
            if let Some(index) = cell_index {
                // Add charge density with overflow protection
                let charge_density_contribution = state.charge / cell_volume;
                if charge_density_contribution.is_finite() {
                    self.charge_density[index] += charge_density_contribution;
                }
                
                // Add current density J = ρv with overflow protection
                let j = state.charge * state.velocity / cell_volume;
                if j.iter().all(|&x| x.is_finite()) {
                    self.current_density[index] += j;
                }
            }
        }
        
        Ok(())
    }

    /// Convert 3D position to 1D cell index with overflow protection
    fn position_to_cell_index(&self, position: &Vector3<f64>) -> Result<Option<usize>, EMError> {
        // Use checked arithmetic to prevent overflow
        let grid_half = self.grid_size as f64 / 2.0;
        
        // Calculate indices with overflow checks
        let x_coord = position.x / self.cell_size + grid_half;
        let y_coord = position.y / self.cell_size + grid_half;
        let z_coord = position.z / self.cell_size + grid_half;
        
        // Check for finite values before casting
        if !x_coord.is_finite() || !y_coord.is_finite() || !z_coord.is_finite() {
            return Ok(None);
        }
        
        // Safe casting with bounds checking
        let x = x_coord as i64;  // Use i64 to avoid i32 overflow
        let y = y_coord as i64;
        let z = z_coord as i64;
        
        if x >= 0 && y >= 0 && z >= 0 
            && x < self.grid_size as i64
            && y < self.grid_size as i64 
            && z < self.grid_size as i64 
        {
            // Safe conversion back to usize after bounds check
            let x_idx = x as usize;
            let y_idx = y as usize;
            let z_idx = z as usize;
            
            // Calculate index with overflow protection
            let index = x_idx
                .checked_mul(self.grid_size)
                .and_then(|intermediate| intermediate.checked_mul(self.grid_size))
                .and_then(|partial| partial.checked_add(y_idx.checked_mul(self.grid_size)?))
                .and_then(|partial| partial.checked_add(z_idx))
                .ok_or(EMError::PositionOverflow)?;
                
            Ok(Some(index))
        } else {
            Ok(None)
        }
    }

    /// Update electric and magnetic fields using proper FDTD method
    fn update_fields(&mut self, constants: &PhysicsConstants) -> Result<()> {
        // Implement proper Finite-Difference Time-Domain (FDTD) solver
        // This solves Maxwell's equations in differential form:
        // ∇ × E = -∂B/∂t (Faraday's law)
        // ∇ × B = μ₀J + μ₀ε₀∂E/∂t (Ampère's law)
        
        let dt = 1e-12; // Time step (1 ps)
        let c = constants.speed_of_light;
        let epsilon_0 = constants.epsilon_0;
        let mu_0 = constants.mu_0;
        
        // Stability condition: dt ≤ dx/(c√3) for 3D FDTD
        let max_dt = self.cell_size / (c * 3.0_f64.sqrt());
        let actual_dt = dt.min(max_dt);
        
        // Create temporary arrays for field updates
        let mut new_electric_field = self.electric_field.clone();
        let mut new_magnetic_field = self.magnetic_field.clone();
        
        // Update electric field: E^(n+1) = E^n + (dt/ε₀)(∇ × B - J)
        for x in 1..self.grid_size-1 {
            for y in 1..self.grid_size-1 {
                for z in 1..self.grid_size-1 {
                    let idx = x + y * self.grid_size + z * self.grid_size * self.grid_size;
                    
                    // Calculate curl of B using finite differences
                    let curl_b_x = (self.magnetic_field[idx + self.grid_size].z - self.magnetic_field[idx - self.grid_size].z) / self.cell_size
                                 - (self.magnetic_field[idx + self.grid_size * self.grid_size].y - self.magnetic_field[idx - self.grid_size * self.grid_size].y) / self.cell_size;
                    
                    let curl_b_y = (self.magnetic_field[idx + self.grid_size * self.grid_size].x - self.magnetic_field[idx - self.grid_size * self.grid_size].x) / self.cell_size
                                 - (self.magnetic_field[idx + 1].z - self.magnetic_field[idx - 1].z) / self.cell_size;
                    
                    let curl_b_z = (self.magnetic_field[idx + 1].y - self.magnetic_field[idx - 1].y) / self.cell_size
                                 - (self.magnetic_field[idx + self.grid_size].x - self.magnetic_field[idx - self.grid_size].x) / self.cell_size;
                    
                    let curl_b = Vector3::new(curl_b_x, curl_b_y, curl_b_z);
                    
                    // Update electric field
                    new_electric_field[idx] = self.electric_field[idx] + 
                        (actual_dt / epsilon_0) * (curl_b - self.current_density[idx]);
                }
            }
        }
        
        // Update magnetic field: B^(n+1) = B^n - dt(∇ × E)
        for x in 1..self.grid_size-1 {
            for y in 1..self.grid_size-1 {
                for z in 1..self.grid_size-1 {
                    let idx = x + y * self.grid_size + z * self.grid_size * self.grid_size;
                    
                    // Calculate curl of E using finite differences
                    let curl_e_x = (new_electric_field[idx + self.grid_size].z - new_electric_field[idx - self.grid_size].z) / self.cell_size
                                 - (new_electric_field[idx + self.grid_size * self.grid_size].y - new_electric_field[idx - self.grid_size * self.grid_size].y) / self.cell_size;
                    
                    let curl_e_y = (new_electric_field[idx + self.grid_size * self.grid_size].x - new_electric_field[idx - self.grid_size * self.grid_size].x) / self.cell_size
                                 - (new_electric_field[idx + 1].z - new_electric_field[idx - 1].z) / self.cell_size;
                    
                    let curl_e_z = (new_electric_field[idx + 1].y - new_electric_field[idx - 1].y) / self.cell_size
                                 - (new_electric_field[idx + self.grid_size].x - new_electric_field[idx - self.grid_size].x) / self.cell_size;
                    
                    let curl_e = Vector3::new(curl_e_x, curl_e_y, curl_e_z);
                    
                    // Update magnetic field
                    new_magnetic_field[idx] = self.magnetic_field[idx] - actual_dt * curl_e;
                }
            }
        }
        
        // Apply boundary conditions (simple absorbing boundary)
        self.apply_absorbing_boundary_conditions(&mut new_electric_field, &mut new_magnetic_field);
        
        // Update fields
        self.electric_field = new_electric_field;
        self.magnetic_field = new_magnetic_field;
        
        Ok(())
    }
    
    /// Apply absorbing boundary conditions to prevent reflections
    fn apply_absorbing_boundary_conditions(
        &self,
        electric_field: &mut [Vector3<f64>],
        magnetic_field: &mut [Vector3<f64>],
    ) {
        let damping_factor = 0.99; // Simple damping at boundaries
        
        // Apply damping at grid boundaries
        for x in 0..self.grid_size {
            for y in 0..self.grid_size {
                // Z boundaries
                let idx_front = x + y * self.grid_size;
                let idx_back = x + y * self.grid_size + (self.grid_size - 1) * self.grid_size * self.grid_size;
                
                electric_field[idx_front] *= damping_factor;
                electric_field[idx_back] *= damping_factor;
                magnetic_field[idx_front] *= damping_factor;
                magnetic_field[idx_back] *= damping_factor;
            }
        }
        
        for x in 0..self.grid_size {
            for z in 0..self.grid_size {
                // Y boundaries
                let idx_bottom = x + z * self.grid_size * self.grid_size;
                let idx_top = x + (self.grid_size - 1) * self.grid_size + z * self.grid_size * self.grid_size;
                
                electric_field[idx_bottom] *= damping_factor;
                electric_field[idx_top] *= damping_factor;
                magnetic_field[idx_bottom] *= damping_factor;
                magnetic_field[idx_top] *= damping_factor;
            }
        }
        
        for y in 0..self.grid_size {
            for z in 0..self.grid_size {
                // X boundaries
                let idx_left = y * self.grid_size + z * self.grid_size * self.grid_size;
                let idx_right = (self.grid_size - 1) + y * self.grid_size + z * self.grid_size * self.grid_size;
                
                electric_field[idx_left] *= damping_factor;
                electric_field[idx_right] *= damping_factor;
                magnetic_field[idx_left] *= damping_factor;
                magnetic_field[idx_right] *= damping_factor;
            }
        }
    }

    /// Apply electromagnetic forces to charged particles
    fn apply_forces(&self, states: &mut [PhysicsState], _constants: &PhysicsConstants) -> Result<()> {
        for state in states.iter_mut() {
            // Skip neutral particles using epsilon comparison
            if state.charge.abs() < 1e-30 {
                continue;
            }
            
            let cell_index = self.position_to_cell_index(&state.position)?;
            if let Some(index) = cell_index {
                let e_field = &self.electric_field[index];
                let b_field = &self.magnetic_field[index];
                
                // Lorentz force: F = q(E + v × B)
                let electric_force = state.charge * e_field;
                let magnetic_force = state.charge * state.velocity.cross(b_field);
                let total_force = electric_force + magnetic_force;
                
                // Validate force before applying
                if total_force.iter().all(|&x| x.is_finite()) && state.mass > 0.0 {
                    // Add to acceleration (will be integrated by classical solver)
                    state.acceleration += total_force / state.mass;
                }
            }
        }
        
        Ok(())
    }

    /// Calculate direct Coulomb forces between charged particles
    pub fn coulomb_forces(&self, states: &[PhysicsState], constants: &PhysicsConstants) 
        -> Result<Vec<Vector3<f64>>, EMError> 
    {
        self.validate_states(states)?;
        
        let n = states.len();
        let mut forces = vec![Vector3::zeros(); n];
        
        for i in 0..n {
            for j in (i+1)..n {
                // Skip neutral particles using epsilon comparison
                if states[i].charge.abs() < 1e-30 || states[j].charge.abs() < 1e-30 {
                    continue;
                }
                
                let r_vec = states[j].position - states[i].position;
                let r = r_vec.magnitude();
                
                // Avoid singularity with safer threshold
                if r < 1e-10 {
                    continue;
                }
                
                let force_magnitude = constants.coulomb_force(states[i].charge, states[j].charge, r);
                
                // Safe normalization to prevent panic
                if r_vec.magnitude() > 0.0 {
                    let direction = r_vec / r;  // Safer than normalize() which can panic
                    let force = -force_magnitude * direction;
                    
                    // Validate force before applying
                    if force.iter().all(|&x| x.is_finite()) {
                        forces[i] += force;
                        forces[j] -= force; // Newton's third law
                    }
                }
            }
        }
        
        Ok(forces)
    }

    /// Calculate electromagnetic field energy
    pub fn field_energy(&self, constants: &PhysicsConstants) -> Result<f64, EMError> {
        self.validate_constants(constants)?;
        
        let mut energy = 0.0;
        let cell_volume = self.cell_size.powi(3);
        
        // Validate cell volume
        if !cell_volume.is_finite() || cell_volume <= 0.0 {
            return Err(EMError::CellSizeTooSmall(self.cell_size));
        }
        
        for i in 0..self.electric_field.len() {
            let e_field = &self.electric_field[i];
            let b_field = &self.magnetic_field[i];
            
            // Energy density: u = ½(ε₀E² + B²/μ₀) with overflow protection
            let e_magnitude_sq = e_field.magnitude_squared();
            let b_magnitude_sq = b_field.magnitude_squared();
            
            if e_magnitude_sq.is_finite() && b_magnitude_sq.is_finite() {
                let e_energy = 0.5 * constants.epsilon_0 * e_magnitude_sq;
                let b_energy = 0.5 * b_magnitude_sq / constants.mu_0;
                
                let local_energy = (e_energy + b_energy) * cell_volume;
                if local_energy.is_finite() {
                    energy += local_energy;
                }
            }
        }
        
        Ok(energy)
    }
}

impl Default for EMSolver {
    fn default() -> Self {
        Self::new()
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
    fn test_grid_size_validation() {
        // Test maximum size exceeded
        let result = EMSolver::with_grid_params(MAX_GRID_SIZE + 1, 1e6);
        assert!(result.is_err());
        
        // Test minimum cell size
        let result = EMSolver::with_grid_params(64, MIN_CELL_SIZE - 1e-20);
        assert!(result.is_err());
    }

    #[test]
    fn test_position_to_cell_index() {
        let solver = EMSolver::new();
        
        // Test center position
        let center = Vector3::zeros();
        let index = solver.position_to_cell_index(&center).unwrap();
        assert!(index.is_some());
        
        // Test out of bounds
        let far = Vector3::new(1e12, 1e12, 1e12);
        let index = solver.position_to_cell_index(&far).unwrap();
        assert!(index.is_none());
        
        // Test infinite position
        let infinite = Vector3::new(f64::INFINITY, 0.0, 0.0);
        let index = solver.position_to_cell_index(&infinite).unwrap();
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
        
        let forces = solver.coulomb_forces(&states, &constants).unwrap();
        
        // Forces should be repulsive (same charge)
        assert!(forces[0].x < 0.0); // Force on first charge points left
        assert!(forces[1].x > 0.0); // Force on second charge points right
        
        // Magnitudes should be equal
        assert_relative_eq!(forces[0].magnitude(), forces[1].magnitude(), epsilon = 1e-10);
    }

    #[test]
    fn test_input_validation() {
        let solver = EMSolver::new();
        
        // Test invalid state with infinite position
        let invalid_states = vec![
            PhysicsState {
                position: Vector3::new(f64::INFINITY, 0.0, 0.0),
                velocity: Vector3::zeros(),
                acceleration: Vector3::zeros(),
                mass: 1.0,
                charge: 1.0,
                temperature: 0.0,
                entropy: 0.0,
            },
        ];
        
        let result = solver.validate_states(&invalid_states);
        assert!(result.is_err());
    }
}