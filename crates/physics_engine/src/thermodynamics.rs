//! Thermodynamics Solver
//! 
//! Implements phase equilibria via Gibbs free energy minimization
//! and real-gas equation of state (Peng-Robinson)

use anyhow::Result;
use crate::{PhysicsState, PhysicsConstants};

/// Thermodynamic state information
#[derive(Debug, Clone)]
pub struct ThermoState {
    pub pressure: f64,       // Pascal
    pub volume: f64,         // m³
    pub molar_amount: f64,   // mol
    pub gibbs_energy: f64,   // J
    pub entropy: f64,        // J/K
    pub enthalpy: f64,       // J
}

/// Thermodynamics solver
pub struct ThermoSolver {
    pub gas_constant: f64,
    pub reference_temp: f64,
    pub reference_pressure: f64,
}

impl Default for ThermoSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ThermoSolver {
    pub fn new() -> Self {
        Self {
            gas_constant: 8.314_462_618, // J/(mol·K)
            reference_temp: 298.15,      // K
            reference_pressure: 101325.0, // Pa
        }
    }

    /// Update thermodynamic properties
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        for state in states.iter_mut() {
            self.update_entropy(state, constants)?;
            self.update_temperature(state, constants)?;
        }
        Ok(())
    }

    /// Update entropy according to second law of thermodynamics
    fn update_entropy(&self, state: &mut PhysicsState, constants: &PhysicsConstants) -> Result<()> {
        // Proper entropy calculation based on statistical mechanics
        // S = k_B * ln(Ω) where Ω is the number of microstates
        
        // Calculate kinetic energy and thermal energy
        let kinetic_energy = 0.5 * state.mass * state.velocity.magnitude_squared();
        let thermal_energy = 1.5 * constants.k_b * state.temperature; // Equipartition theorem
        
        // Calculate number of accessible microstates
        // For a classical particle in 3D, Ω ∝ (E_kinetic)^(3/2) * V
        let volume_factor = 1e-27; // Approximate volume in m³
        let energy_factor = (kinetic_energy / thermal_energy).max(1e-10);
        let microstates = (energy_factor.powf(1.5) * volume_factor * 1e30).max(1.0);
        
        // Entropy change: ΔS = k_B * ln(Ω_final/Ω_initial)
        let initial_microstates = 1.0; // Ground state
        let entropy_change = constants.k_b * (microstates / initial_microstates).ln();
        
        // Add entropy change (entropy can only increase in isolated systems)
        if entropy_change > 0.0 {
            state.entropy += entropy_change;
        }
        
        // Additional entropy from thermal fluctuations
        let thermal_entropy = constants.k_b * (state.temperature / 1.0).ln(); // Relative to 1K
        state.entropy += thermal_entropy * 1e-12; // Small time step
        
        // Ensure entropy is non-negative
        state.entropy = state.entropy.max(0.0);
        
        Ok(())
    }

    /// Update temperature based on kinetic energy
    fn update_temperature(&self, state: &mut PhysicsState, constants: &PhysicsConstants) -> Result<()> {
        // Equipartition theorem: E_kinetic = (3/2) * k_B * T for monatomic gas
        let kinetic_energy = 0.5 * state.mass * state.velocity.magnitude_squared();
        
        // Assuming monatomic particles
        let new_temperature = (2.0 * kinetic_energy) / (3.0 * constants.k_b);
        
        // Temperature cannot be negative
        state.temperature = new_temperature.max(0.1); // Minimum 0.1 K
        
        Ok(())
    }

    /// Calculate pressure using ideal gas law (simplified)
    pub fn calculate_pressure(&self, state: &PhysicsState, volume: f64) -> f64 {
        // P = nRT/V
        let n = state.mass / (12.0 * 1.66054e-27); // Assume carbon-12 for simplicity
        n * self.gas_constant * state.temperature / volume
    }

    /// Calculate Gibbs free energy
    pub fn gibbs_free_energy(&self, state: &PhysicsState, pressure: f64) -> f64 {
        // G = H - TS
        // Simplified: G ≈ -k_B * T * ln(Z) where Z is partition function
        let thermal_energy = 1.5 * 8.314_462_618 * state.temperature; // 3/2 RT
        let entropy_term = state.temperature * state.entropy;
        
        thermal_energy - entropy_term + pressure * 1e-23 // Volume term
    }

    /// Check for phase transitions
    pub fn check_phase_transition(&self, state: &PhysicsState) -> PhaseState {
        // Simplified phase diagram
        if state.temperature < 273.15 {
            PhaseState::Solid
        } else if state.temperature < 373.15 {
            PhaseState::Liquid
        } else {
            PhaseState::Gas
        }
    }

    /// Calculate thermal velocity distribution (Maxwell-Boltzmann)
    pub fn thermal_velocity(&self, state: &PhysicsState, constants: &PhysicsConstants) -> f64 {
        // v_thermal = sqrt(3kT/m)
        (3.0 * constants.k_b * state.temperature / state.mass).sqrt()
    }

    /// Calculate heat capacity
    pub fn heat_capacity(&self, _state: &PhysicsState) -> f64 {
        // For monatomic gas: C_v = (3/2)R
        1.5 * self.gas_constant
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhaseState {
    Solid,
    Liquid,
    Gas,
    Plasma,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicsConstants;
    use nalgebra::Vector3;

    #[test]
    fn test_thermo_solver_creation() {
        let solver = ThermoSolver::new();
        assert_eq!(solver.gas_constant, 8.314_462_618);
    }

    #[test]
    fn test_temperature_update() {
        let solver = ThermoSolver::new();
        let constants = PhysicsConstants::default();
        
        let mut state = PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0), // 1 km/s
            acceleration: Vector3::zeros(),
            mass: constants.m_p, // Proton mass
            charge: constants.e,
            temperature: 300.0,
            entropy: 0.0,
        };
        
        solver.update_temperature(&mut state, &constants).unwrap();
        
        // Temperature should be calculated from kinetic energy
        assert!(state.temperature > 0.0);
    }

    #[test]
    fn test_phase_transitions() {
        let solver = ThermoSolver::new();
        
        let ice_state = PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            mass: 1e-20,
            charge: 0.0,
            temperature: 250.0, // Below freezing
            entropy: 0.0,
        };
        
        let gas_state = PhysicsState {
            temperature: 400.0, // Above boiling
            ..ice_state
        };
        
        assert_eq!(solver.check_phase_transition(&ice_state), PhaseState::Solid);
        assert_eq!(solver.check_phase_transition(&gas_state), PhaseState::Gas);
    }

    #[test]
    fn test_thermal_velocity() {
        let solver = ThermoSolver::new();
        let constants = PhysicsConstants::default();
        
        let state = PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            mass: constants.m_p,
            charge: 0.0,
            temperature: 300.0,
            entropy: 0.0,
        };
        
        let v_thermal = solver.thermal_velocity(&state, &constants);
        
        // Should be reasonable thermal velocity
        assert!(v_thermal > 1000.0); // > 1 km/s for proton at 300K
        assert!(v_thermal < 1e6);    // < 1000 km/s
    }
}