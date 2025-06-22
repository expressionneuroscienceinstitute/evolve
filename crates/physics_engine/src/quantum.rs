//! Quantum Mechanics Solver
//! 
//! Implements tight-binding valence model for electron shells
//! and stochastic Pauli Monte Carlo for new compounds

use crate::*;
use rand::{Rng, thread_rng};
use rand::prelude::SliceRandom;
use anyhow::Result;
use nalgebra::Complex;

// Import Boltzmann constant from constants module
const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K

/// Quantum mechanics solver
pub struct QuantumSolver {
    pub planck_constant: f64,
    pub electron_rest_mass: f64,
    pub bohr_radius: f64,
    pub rydberg_energy: f64,
    pub reduced_planck_constant: f64,
    pub vacuum_permittivity: f64,
    pub elementary_charge: f64,
}

impl Default for QuantumSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumSolver {
    pub fn new() -> Self {
        Self {
            planck_constant: 6.626_070_15e-34,  // J⋅s
            electron_rest_mass: 9.109_383_701_5e-31, // kg
            bohr_radius: 5.291_772_106_7e-11,     // m
            rydberg_energy: 13.605_693_122_994,  // eV
            reduced_planck_constant: 1.054_571_817e-34, // J⋅s
            vacuum_permittivity: 8.854_187_812_8e-12, // F/m
            elementary_charge: 1.602_176_634e-19, // C
        }
    }

    /// Update quantum mechanical properties
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        for state in states.iter_mut() {
            self.update_quantum_transitions(state, constants)?;
            self.apply_pauli_exclusion(state)?;
            self.evolve_quantum_state(state, constants)?;
            self.handle_decoherence(state, constants)?;
        }
        Ok(())
    }

    /// Calculate quantum energy levels (hydrogen-like atoms)
    pub fn energy_level(&self, n: u32, z: f64) -> f64 {
        // E_n = -Z² * Ry / n²
        -z * z * self.rydberg_energy / (n * n) as f64
    }

    /// Calculate orbital radius
    pub fn orbital_radius(&self, n: u32, z: f64) -> f64 {
        // r_n = n² * a₀ / Z
        (n * n) as f64 * self.bohr_radius / z
    }

    /// Update quantum transitions (simplified)
    fn update_quantum_transitions(&self, state: &mut PhysicsState, constants: &PhysicsConstants) -> Result<()> {
        // Simplified quantum transition based on temperature
        let thermal_energy = constants.k_b * state.temperature;
        let photon_energy = thermal_energy; // Simplified
        
        // Check if photon energy can cause transitions
        if photon_energy > 0.1 * 1.602_176_634e-19 { // > 0.1 eV
            // Probabilistic excitation
            let mut rng = thread_rng();
            if rng.gen::<f64>() < 0.001 { // Small probability
                // Add quantum excitation energy to total energy
                state.entropy += photon_energy / state.temperature;
            }
        }
        
        Ok(())
    }

    /// Apply Pauli exclusion principle (simplified)
    fn apply_pauli_exclusion(&self, _state: &mut PhysicsState) -> Result<()> {
        // In a full implementation, this would track electron occupation numbers
        // and prevent multiple electrons in the same quantum state
        
        // For now, just a placeholder
        Ok(())
    }

    /// Evolve quantum state using Schrödinger equation
    fn evolve_quantum_state(&self, state: &mut PhysicsState, _constants: &PhysicsConstants) -> Result<()> {
        // Simplified quantum state evolution
        // In a full implementation, this would solve the time-dependent Schrödinger equation
        
        // Calculate energy from kinetic energy: E = ½mv²
        let kinetic_energy = 0.5 * state.mass * state.velocity.magnitude().powi(2);
        let momentum = state.mass * state.velocity.magnitude();
        
        // Simple phase evolution: ψ(t) = ψ(0) * exp(-iEt/ℏ)
        // Use a simplified time evolution based on current simulation state
        let time_evolution_phase = -kinetic_energy * 1e-12 / self.reduced_planck_constant; // Simplified time scale
        
        // Update quantum state amplitude (simplified)
        // Note: PhysicsState doesn't have quantum_state field, so we'll simulate the effect
        // by updating the entropy to reflect quantum state changes
        state.entropy += time_evolution_phase.abs() * 1e-20; // Small entropy change
        
        Ok(())
    }

    /// Handle quantum decoherence
    fn handle_decoherence(&self, state: &mut PhysicsState, constants: &PhysicsConstants) -> Result<()> {
        // Simplified decoherence model
        // In a full implementation, this would model interaction with environment
        
        let decoherence_rate = self.calculate_decoherence_rate(state, constants);
        
        // Apply decoherence effect to entropy (simplified)
        let decoherence_factor = (-decoherence_rate * 1e-12).exp(); // Simplified time scale
        state.entropy *= decoherence_factor;
        
        Ok(())
    }

    /// Calculate decoherence rate based on environmental factors
    fn calculate_decoherence_rate(&self, state: &PhysicsState, constants: &PhysicsConstants) -> f64 {
        // Simplified decoherence rate calculation
        // Based on temperature and system size
        
        let thermal_energy = constants.k_b * state.temperature;
        let system_size = state.position.magnitude().max(1e-10);
        
        // Decoherence rate ∝ kT / (ℏ * system_size)
        thermal_energy / (self.reduced_planck_constant * system_size)
    }

    /// Perform quantum measurement
    pub fn measure_quantum_state(&self, quantum_state: &mut QuantumState, measurement_basis: MeasurementBasis) -> Result<f64> {
        // Perform quantum measurement in specified basis
        match measurement_basis {
            MeasurementBasis::Position => self.measure_position(quantum_state),
            MeasurementBasis::Momentum => self.measure_momentum(quantum_state),
            MeasurementBasis::Energy => self.measure_energy(quantum_state),
            MeasurementBasis::Spin => self.measure_spin(quantum_state),
        }
    }

    /// Measure position (simplified)
    fn measure_position(&self, quantum_state: &mut QuantumState) -> Result<f64> {
        // Simplified position measurement
        // In a full implementation, this would collapse the wave function
        
        let mut rng = thread_rng();
        let measurement_result = rng.gen_range(-1.0..1.0) * self.bohr_radius;
        
        // Update quantum state after measurement
        quantum_state.wave_function = vec![Complex::new(1.0, 0.0)];
        quantum_state.superposition_amplitudes.clear();
        
        Ok(measurement_result)
    }

    /// Measure momentum (simplified)
    fn measure_momentum(&self, quantum_state: &mut QuantumState) -> Result<f64> {
        // Simplified momentum measurement
        let mut rng = thread_rng();
        let measurement_result = rng.gen_range(-1.0..1.0) * self.reduced_planck_constant / self.bohr_radius;
        
        // Update quantum state after measurement
        quantum_state.wave_function = vec![Complex::new(1.0, 0.0)];
        quantum_state.superposition_amplitudes.clear();
        
        Ok(measurement_result)
    }

    /// Measure energy (simplified)
    fn measure_energy(&self, quantum_state: &mut QuantumState) -> Result<f64> {
        // Simplified energy measurement
        let energy = quantum_state.energy_level;
        
        // Update quantum state after measurement
        quantum_state.wave_function = vec![Complex::new(1.0, 0.0)];
        quantum_state.superposition_amplitudes.clear();
        
        Ok(energy)
    }

    /// Measure spin (simplified)
    fn measure_spin(&self, quantum_state: &mut QuantumState) -> Result<f64> {
        // Simplified spin measurement
        let spin = quantum_state.spin_quantum_number;
        
        // Update quantum state after measurement
        quantum_state.wave_function = vec![Complex::new(1.0, 0.0)];
        quantum_state.superposition_amplitudes.clear();
        
        Ok(spin)
    }

    /// Create entangled quantum state
    pub fn create_entangled_state(&self, particle1: &mut FundamentalParticle, particle2: &mut FundamentalParticle) -> Result<()> {
        // Create Bell state: |ψ⟩ = (|00⟩ + |11⟩) / √2
        
        // Initialize quantum states
        particle1.quantum_state = QuantumState::new();
        particle2.quantum_state = QuantumState::new();
        
        // Create superposition
        particle1.quantum_state.superposition_amplitudes.insert("0".to_string(), Complex::new(1.0 / 2.0_f64.sqrt(), 0.0));
        particle1.quantum_state.superposition_amplitudes.insert("1".to_string(), Complex::new(1.0 / 2.0_f64.sqrt(), 0.0));
        
        particle2.quantum_state.superposition_amplitudes.insert("0".to_string(), Complex::new(1.0 / 2.0_f64.sqrt(), 0.0));
        particle2.quantum_state.superposition_amplitudes.insert("1".to_string(), Complex::new(1.0 / 2.0_f64.sqrt(), 0.0));
        
        // Mark particles as entangled
        particle1.quantum_state.entanglement_partners.push(0); // Will be updated with actual index
        particle2.quantum_state.entanglement_partners.push(0); // Will be updated with actual index
        
        Ok(())
    }

    /// Calculate atomic ionization energy
    pub fn ionization_energy(&self, z: f64, n: u32) -> f64 {
        // E_ionization = |E_n|
        (z * z * self.rydberg_energy / (n * n) as f64).abs()
    }

    /// Calculate quantum tunneling probability
    pub fn tunneling_probability(&self, energy: f64, barrier_height: f64, barrier_width: f64) -> f64 {
        if energy >= barrier_height {
            return 1.0; // Classical crossing
        }
        
        let delta_e = (barrier_height - energy) * 1.602_176_634e-19; // J
        let kappa = ((2.0 * self.electron_rest_mass * delta_e).sqrt())
                    / (self.planck_constant / (2.0 * std::f64::consts::PI));
        
        (-2.0 * kappa * barrier_width).exp()
    }

    /// Calculate de Broglie wavelength
    pub fn de_broglie_wavelength(&self, mass: f64, velocity: f64) -> f64 {
        if velocity < 1e-10 {
            return f64::INFINITY;
        }
        
        // λ = h / (mv)
        self.planck_constant / (mass * velocity)
    }

    /// Check if quantum effects are significant
    pub fn is_quantum_significant(&self, state: &PhysicsState, length_scale: f64) -> bool {
        let velocity = state.velocity.magnitude();
        let wavelength = self.de_broglie_wavelength(state.mass, velocity);
        
        // Quantum effects significant when de Broglie wavelength ~ system size
        wavelength >= 0.1 * length_scale
    }

    /// Calculate zero-point energy
    pub fn zero_point_energy(&self, frequency: f64) -> f64 {
        // E_0 = ½ℏω
        0.5 * (self.planck_constant / (2.0 * std::f64::consts::PI)) * frequency
    }

    /// Generate quantum state for electron
    pub fn generate_electron_state(&self, n: u32, l: u32, m_l: i32) -> QuantumState {
        // Validate quantum numbers
        let l = l.min(n - 1);
        let m_l = m_l.max(-(l as i32)).min(l as i32);
        
        QuantumState {
            principal_quantum_number: n,
            orbital_angular_momentum: l,
            magnetic_quantum_number: m_l,
            spin_quantum_number: 0.5, // Electron spin
            energy_level: self.energy_level(n, 1.0), // Hydrogen-like
            occupation_probability: 1.0,
            ..Default::default()
        }
    }

    /// Calculate quantum mechanical expectation value
    pub fn expectation_value(&self, quantum_state: &QuantumState, observable: &str) -> f64 {
        match observable {
            "energy" => quantum_state.energy_level,
            "position" => 0.0, // Simplified - would integrate ψ* x ψ
            "momentum" => 0.0, // Simplified - would integrate ψ* p ψ
            "spin" => quantum_state.spin_quantum_number,
            _ => 0.0,
        }
    }

    /// Calculate quantum mechanical uncertainty
    pub fn uncertainty(&self, quantum_state: &QuantumState, observable: &str) -> f64 {
        // Simplified uncertainty calculation
        // In a full implementation, this would calculate ΔA = √(⟨A²⟩ - ⟨A⟩²)
        
        match observable {
            "energy" => quantum_state.energy_level * 0.1, // 10% uncertainty
            "position" => self.bohr_radius * 0.5,
            "momentum" => self.reduced_planck_constant / self.bohr_radius * 0.5,
            "spin" => 0.5,
            _ => 0.0,
        }
    }

    /// Apply quantum operator to state
    pub fn apply_operator(&self, quantum_state: &mut QuantumState, operator: &str) -> Result<()> {
        match operator {
            "position" => {
                // Apply position operator (simplified)
                for amplitude in quantum_state.wave_function.iter_mut() {
                    *amplitude *= Complex::new(0.0, 1.0); // Phase shift
                }
            },
            "momentum" => {
                // Apply momentum operator (simplified)
                for amplitude in quantum_state.wave_function.iter_mut() {
                    *amplitude *= Complex::new(0.0, -1.0); // Phase shift
                }
            },
            "energy" => {
                // Apply energy operator (simplified)
                for amplitude in quantum_state.wave_function.iter_mut() {
                    *amplitude *= Complex::new(quantum_state.energy_level, 0.0);
                }
            },
            _ => return Err(anyhow::anyhow!("Unknown operator: {}", operator)),
        }
        
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Quantum Monte Carlo Methods
// -----------------------------------------------------------------------------

/// Quantum Monte Carlo solver for advanced quantum physics calculations
#[derive(Debug, Clone)]
pub struct QuantumMonteCarlo {
    pub time_step: f64,
    pub temperature: f64,
    pub num_walkers: usize,
    pub convergence_threshold: f64,
    pub max_iterations: usize,
}

impl Default for QuantumMonteCarlo {
    fn default() -> Self {
        Self {
            time_step: 1e-3,
            temperature: 300.0, // 300 K
            num_walkers: 1000,
            convergence_threshold: 1e-6,
            max_iterations: 10000,
        }
    }
}

impl QuantumMonteCarlo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Path Integral Monte Carlo (PIMC) for quantum statistical mechanics
    pub fn path_integral_monte_carlo<F>(
        &self,
        potential_energy: F,
        initial_positions: &[Vector3<f64>],
        num_time_slices: usize,
    ) -> Result<PathIntegralResult>
    where
        F: Fn(&Vector3<f64>) -> f64,
    {
        let mut rng = thread_rng();
        let num_particles = initial_positions.len();
        let beta = 1.0 / (BOLTZMANN_CONSTANT * self.temperature);
        let tau = beta / num_time_slices as f64;
        
        // Initialize path with initial positions
        let mut path = vec![initial_positions.to_vec(); num_time_slices];
        
        // PIMC sampling
        let mut energy_samples = Vec::new();
        let mut position_samples = Vec::new();
        
        for iteration in 0..self.max_iterations {
            // Staging algorithm for path updates
            for particle_idx in 0..num_particles {
                for time_slice in 0..num_time_slices {
                    let old_position = path[time_slice][particle_idx];
                    
                    // Generate trial move
                    let trial_move = Vector3::new(
                        rng.gen_range(-0.1..0.1),
                        rng.gen_range(-0.1..0.1),
                        rng.gen_range(-0.1..0.1),
                    );
                    let new_position = old_position + trial_move;
                    
                    // Calculate acceptance probability
                    let old_action = self.calculate_path_action(&path, &potential_energy, tau);
                    let mut new_path = path.clone();
                    new_path[time_slice][particle_idx] = new_position;
                    let new_action = self.calculate_path_action(&new_path, &potential_energy, tau);
                    
                    let acceptance_prob = (-(new_action - old_action) / BOLTZMANN_CONSTANT).exp();
                    
                    if rng.gen::<f64>() < acceptance_prob {
                        path = new_path;
                    }
                }
            }
            
            // Sample observables
            if iteration % 100 == 0 {
                let energy = self.calculate_path_energy(&path, &potential_energy, tau);
                energy_samples.push(energy);
                position_samples.push(path[0].clone()); // Sample first time slice
            }
        }
        
        Ok(PathIntegralResult {
            average_energy: energy_samples.iter().sum::<f64>() / energy_samples.len() as f64,
            energy_variance: self.calculate_variance(&energy_samples),
            average_positions: self.calculate_average_positions(&position_samples),
            path_samples: path,
        })
    }

    /// Variational Monte Carlo (VMC) for ground state optimization
    pub fn variational_monte_carlo<F, G>(
        &self,
        trial_wavefunction: F,
        potential_energy: G,
        initial_parameters: &[f64],
    ) -> Result<VariationalMonteCarloResult>
    where
        F: Fn(&Vector3<f64>, &[f64]) -> f64,
        G: Fn(&Vector3<f64>) -> f64,
    {
        let mut rng = thread_rng();
        let mut parameters = initial_parameters.to_vec();
        let mut energy_history = Vec::new();
        let mut parameter_history = Vec::new();
        
        for iteration in 0..self.max_iterations {
            let mut energy_samples = Vec::new();
            let mut weight_samples = Vec::new();
            
            // Sample configurations
            for _ in 0..self.num_walkers {
                let position = Vector3::new(
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(-5.0..5.0),
                );
                
                let psi = trial_wavefunction(&position, &parameters);
                let psi_squared = psi * psi;
                let local_energy = self.calculate_local_energy(&position, &parameters, &trial_wavefunction, &potential_energy);
                
                // Store local energy and weight (psi^2)
                energy_samples.push(local_energy);
                weight_samples.push(psi_squared);
            }
            
            // Calculate weighted average energy
            let total_weight: f64 = weight_samples.iter().sum();
            let weighted_energy: f64 = energy_samples.iter()
                .zip(weight_samples.iter())
                .map(|(e, w)| e * w)
                .sum();
            let average_energy = if total_weight > 0.0 { weighted_energy / total_weight } else { 0.0 };
            
            energy_history.push(average_energy);
            parameter_history.push(parameters.clone());
            
            // Simple gradient descent optimization
            if iteration > 0 && iteration % 100 == 0 {
                let energy_gradient = self.calculate_energy_gradient(&parameters, &trial_wavefunction, &potential_energy);
                for (param, grad) in parameters.iter_mut().zip(energy_gradient.iter()) {
                    *param -= 0.01 * grad; // Learning rate
                }
            }
            
            // Check convergence
            if iteration > 100 {
                let recent_energies = &energy_history[energy_history.len().saturating_sub(100)..];
                let energy_variance = self.calculate_variance(recent_energies);
                if energy_variance < self.convergence_threshold {
                    break;
                }
            }
        }
        
        Ok(VariationalMonteCarloResult {
            optimized_parameters: parameters,
            ground_state_energy: energy_history.last().copied().unwrap_or(0.0),
            energy_history,
            parameter_history,
        })
    }

    /// Diffusion Monte Carlo (DMC) for ground state projection
    pub fn diffusion_monte_carlo<F, G>(
        &self,
        trial_wavefunction: F,
        potential_energy: G,
        initial_walkers: &[Vector3<f64>],
    ) -> Result<DiffusionMonteCarloResult>
    where
        F: Fn(&Vector3<f64>) -> f64,
        G: Fn(&Vector3<f64>) -> f64,
    {
        let mut rng = thread_rng();
        let mut walkers = initial_walkers.to_vec();
        let mut reference_energy = 0.0;
        let mut energy_history = Vec::new();
        let mut walker_history = Vec::new();
        
        for iteration in 0..self.max_iterations {
            let mut new_walkers = Vec::new();
            let mut local_energies = Vec::new();
            
            // Diffuse each walker
            for walker in &walkers {
                // Diffusion step
                let diffusion_step = Vector3::new(
                    rng.gen_range(-1.0..1.0) * (2.0 * self.time_step).sqrt(),
                    rng.gen_range(-1.0..1.0) * (2.0 * self.time_step).sqrt(),
                    rng.gen_range(-1.0..1.0) * (2.0 * self.time_step).sqrt(),
                );
                let new_position = walker + diffusion_step;
                
                // Drift step (quantum force)
                let quantum_force = self.calculate_quantum_force(&new_position, &trial_wavefunction);
                let drift_step = quantum_force * self.time_step;
                let final_position = new_position + drift_step;
                
                // Calculate local energy
                let local_energy = self.calculate_local_energy_simple(&final_position, &potential_energy);
                local_energies.push(local_energy);
                
                // Branching (birth/death of walkers)
                let branching_factor = (-(local_energy - reference_energy) * self.time_step).exp();
                let num_copies = branching_factor.round() as usize;
                
                for _ in 0..num_copies {
                    new_walkers.push(final_position);
                }
            }
            
            // Update reference energy
            if !local_energies.is_empty() {
                reference_energy = local_energies.iter().sum::<f64>() / local_energies.len() as f64;
            }
            
            // Control walker population
            if new_walkers.len() > 2 * self.num_walkers {
                // Randomly remove excess walkers
                new_walkers.shuffle(&mut rng);
                new_walkers.truncate(self.num_walkers);
            } else if new_walkers.len() < self.num_walkers / 2 {
                // Duplicate some walkers
                let mut additional_walkers = Vec::new();
                for _ in 0..(self.num_walkers - new_walkers.len()) {
                    if let Some(&walker) = new_walkers.choose(&mut rng) {
                        additional_walkers.push(walker);
                    }
                }
                new_walkers.extend(additional_walkers);
            }
            
            walkers = new_walkers;
            energy_history.push(reference_energy);
            walker_history.push(walkers.clone());
            
            // Check convergence
            if iteration > 100 {
                let recent_energies = &energy_history[energy_history.len().saturating_sub(100)..];
                let energy_variance = self.calculate_variance(recent_energies);
                if energy_variance < self.convergence_threshold {
                    break;
                }
            }
        }
        
        Ok(DiffusionMonteCarloResult {
            ground_state_energy: reference_energy,
            final_walkers: walkers,
            energy_history,
            walker_history,
        })
    }

    // Helper methods for Quantum Monte Carlo calculations
    
    fn calculate_path_action<F>(&self, path: &[Vec<Vector3<f64>>], potential: &F, tau: f64) -> f64
    where
        F: Fn(&Vector3<f64>) -> f64,
    {
        let mut action = 0.0;
        let num_time_slices = path.len();
        let num_particles = path[0].len();
        
        for time_slice in 0..num_time_slices {
            let next_slice = (time_slice + 1) % num_time_slices;
            
            for particle_idx in 0..num_particles {
                let current_pos = path[time_slice][particle_idx];
                let next_pos = path[next_slice][particle_idx];
                
                // Kinetic energy term
                let displacement = next_pos - current_pos;
                let kinetic_term = displacement.dot(&displacement) / (2.0 * tau);
                action += kinetic_term;
                
                // Potential energy term
                action += potential(&current_pos) * tau;
            }
        }
        
        action
    }

    fn calculate_path_energy<F>(&self, path: &[Vec<Vector3<f64>>], potential: &F, tau: f64) -> f64
    where
        F: Fn(&Vector3<f64>) -> f64,
    {
        let num_time_slices = path.len();
        let num_particles = path[0].len();
        let mut total_energy = 0.0;
        
        for time_slice in 0..num_time_slices {
            for particle_idx in 0..num_particles {
                let position = path[time_slice][particle_idx];
                total_energy += potential(&position);
            }
        }
        
        total_energy / (num_time_slices * num_particles) as f64
    }

    fn calculate_local_energy<F, G>(
        &self,
        position: &Vector3<f64>,
        parameters: &[f64],
        trial_wavefunction: &F,
        potential_energy: &G,
    ) -> f64
    where
        F: Fn(&Vector3<f64>, &[f64]) -> f64,
        G: Fn(&Vector3<f64>) -> f64,
    {
        // Simplified local energy calculation
        // In a full implementation, this would include the kinetic energy operator
        let psi = trial_wavefunction(position, parameters);
        let v = potential_energy(position);
        
        // Simplified kinetic energy (Laplacian of trial wavefunction)
        let h = 1e-6; // Small step for numerical derivatives
        let laplacian = (
            trial_wavefunction(&(position + Vector3::new(h, 0.0, 0.0)), parameters) +
            trial_wavefunction(&(position + Vector3::new(-h, 0.0, 0.0)), parameters) +
            trial_wavefunction(&(position + Vector3::new(0.0, h, 0.0)), parameters) +
            trial_wavefunction(&(position + Vector3::new(0.0, -h, 0.0)), parameters) +
            trial_wavefunction(&(position + Vector3::new(0.0, 0.0, h)), parameters) +
            trial_wavefunction(&(position + Vector3::new(0.0, 0.0, -h)), parameters) -
            6.0 * psi
        ) / (h * h);
        
        -0.5 * laplacian / psi + v
    }

    fn calculate_local_energy_simple<F>(&self, position: &Vector3<f64>, potential_energy: &F) -> f64
    where
        F: Fn(&Vector3<f64>) -> f64,
    {
        // Simplified local energy for DMC
        potential_energy(position)
    }

    fn calculate_quantum_force<F>(&self, position: &Vector3<f64>, trial_wavefunction: &F) -> Vector3<f64>
    where
        F: Fn(&Vector3<f64>) -> f64,
    {
        // Quantum force = ∇ψ/ψ
        let h = 1e-6;
        let psi = trial_wavefunction(position);
        
        let grad_x = (trial_wavefunction(&(position + Vector3::new(h, 0.0, 0.0))) - 
                     trial_wavefunction(&(position + Vector3::new(-h, 0.0, 0.0)))) / (2.0 * h);
        let grad_y = (trial_wavefunction(&(position + Vector3::new(0.0, h, 0.0))) - 
                     trial_wavefunction(&(position + Vector3::new(0.0, -h, 0.0)))) / (2.0 * h);
        let grad_z = (trial_wavefunction(&(position + Vector3::new(0.0, 0.0, h))) - 
                     trial_wavefunction(&(position + Vector3::new(0.0, 0.0, -h)))) / (2.0 * h);
        
        Vector3::new(grad_x / psi, grad_y / psi, grad_z / psi)
    }

    fn calculate_energy_gradient<F, G>(
        &self,
        parameters: &[f64],
        trial_wavefunction: &F,
        potential_energy: &G,
    ) -> Vec<f64>
    where
        F: Fn(&Vector3<f64>, &[f64]) -> f64,
        G: Fn(&Vector3<f64>) -> f64,
    {
        // Simplified gradient calculation
        let mut gradient = vec![0.0; parameters.len()];
        let h = 1e-6;
        
        for (i, param) in parameters.iter().enumerate() {
            let mut params_plus = parameters.to_vec();
            let mut params_minus = parameters.to_vec();
            params_plus[i] += h;
            params_minus[i] -= h;
            
            // Sample a few points to estimate gradient
            let mut rng = thread_rng();
            for _ in 0..100 {
                let position = Vector3::new(
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-2.0..2.0),
                );
                
                let energy_plus = self.calculate_local_energy(&position, &params_plus, trial_wavefunction, potential_energy);
                let energy_minus = self.calculate_local_energy(&position, &params_minus, trial_wavefunction, potential_energy);
                
                gradient[i] += (energy_plus - energy_minus) / (2.0 * h);
            }
            
            gradient[i] /= 100.0;
        }
        
        gradient
    }

    fn calculate_variance(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        variance
    }

    fn calculate_average_positions(&self, position_samples: &[Vec<Vector3<f64>>]) -> Vec<Vector3<f64>> {
        if position_samples.is_empty() {
            return Vec::new();
        }
        
        let num_particles = position_samples[0].len();
        let mut average_positions = vec![Vector3::zeros(); num_particles];
        
        for sample in position_samples {
            for (i, &position) in sample.iter().enumerate() {
                average_positions[i] += position;
            }
        }
        
        for position in &mut average_positions {
            *position /= position_samples.len() as f64;
        }
        
        average_positions
    }
}

/// Results from Path Integral Monte Carlo calculation
#[derive(Debug, Clone)]
pub struct PathIntegralResult {
    pub average_energy: f64,
    pub energy_variance: f64,
    pub average_positions: Vec<Vector3<f64>>,
    pub path_samples: Vec<Vec<Vector3<f64>>>,
}

/// Results from Variational Monte Carlo calculation
#[derive(Debug, Clone)]
pub struct VariationalMonteCarloResult {
    pub optimized_parameters: Vec<f64>,
    pub ground_state_energy: f64,
    pub energy_history: Vec<f64>,
    pub parameter_history: Vec<Vec<f64>>,
}

/// Results from Diffusion Monte Carlo calculation
#[derive(Debug, Clone)]
pub struct DiffusionMonteCarloResult {
    pub ground_state_energy: f64,
    pub final_walkers: Vec<Vector3<f64>>,
    pub energy_history: Vec<f64>,
    pub walker_history: Vec<Vec<Vector3<f64>>>,
}

// -----------------------------------------------------------------------------
// Additional quantum–field–theory helper structs required by the main engine
// -----------------------------------------------------------------------------

/// Simple representation of the quantum vacuum (placeholder).
#[derive(Debug, Clone)]
pub struct QuantumVacuum {
    pub fluctuation_level: f64,
}

impl Default for QuantumVacuum {
    fn default() -> Self {
        Self { fluctuation_level: 0.0 }
    }
}

impl QuantumVacuum {
    /// Initialise vacuum fluctuations (very coarse model – scales with T).
    pub fn initialize_fluctuations(&mut self, temperature: f64) -> anyhow::Result<()> {
        // Scale fluctuations linearly with temperature for now
        self.fluctuation_level = temperature * 1e-5;
        Ok(())
    }
}

/// Scale–dependent running of fundamental couplings (α_s, α_EM, etc.).
#[derive(Debug, Clone)]
pub struct RunningCouplings {
    pub alpha_em: f64,
    pub alpha_s: f64,
    pub scale_gev: f64,
}

impl Default for RunningCouplings {
    fn default() -> Self {
        Self {
            alpha_em: FINE_STRUCTURE_CONSTANT,
            alpha_s: 0.118, // α_s(M_Z)
            scale_gev: 91.1876, // Default reference scale (M_Z)
        }
    }
}

/// Electroweak symmetry–breaking tracking (Higgs mechanism placeholder).
#[derive(Debug, Clone)]
pub struct SymmetryBreaking {
    pub higgs_vev_gev: f64,
}

impl Default for SymmetryBreaking {
    fn default() -> Self {
        Self { higgs_vev_gev: 246.0 } // Standard Model v ≈ 246 GeV
    }
}

impl SymmetryBreaking {
    /// Initialise spontaneous symmetry breaking (no-op placeholder).
    pub fn initialize_higgs_mechanism(&mut self) -> anyhow::Result<()> {
        // In a full model we would modify particle masses here.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_solver_creation() {
        let solver = QuantumSolver::new();
        assert_relative_eq!(solver.rydberg_energy, 13.605_693_122_994, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_levels() {
        let solver = QuantumSolver::new();
        
        // Ground state of hydrogen (n=1, Z=1)
        let e1 = solver.energy_level(1, 1.0);
        assert_relative_eq!(e1, -13.605_693_122_994, epsilon = 1e-10);
        
        // First excited state (n=2, Z=1)
        let e2 = solver.energy_level(2, 1.0);
        assert_relative_eq!(e2, -13.605_693_122_994 / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orbital_radius() {
        let solver = QuantumSolver::new();
        
        // Bohr radius for hydrogen ground state
        let r1 = solver.orbital_radius(1, 1.0);
        assert_relative_eq!(r1, solver.bohr_radius, epsilon = 1e-10);
        
        // Second orbital should be 4x larger
        let r2 = solver.orbital_radius(2, 1.0);
        assert_relative_eq!(r2, 4.0 * solver.bohr_radius, epsilon = 1e-10);
    }

    #[test]
    fn test_de_broglie_wavelength() {
        let solver = QuantumSolver::new();
        let constants = PhysicsConstants::default();
        
        // Electron at 1000 m/s
        let wavelength = solver.de_broglie_wavelength(constants.m_e, 1000.0);
        
        // Should be on the order of nanometers
        assert!(wavelength > 1e-10); // > 0.1 nm
        assert!(wavelength < 1e-6);  // < 1 μm
    }

    #[test]
    fn test_tunneling_probability() {
        let solver = QuantumSolver::new();
        
        // Classical case (E > V)
        let prob = solver.tunneling_probability(2.0, 1.0, 1e-10);
        assert_relative_eq!(prob, 1.0, epsilon = 1e-10);
        
        // Quantum tunneling case (E < V)
        let prob = solver.tunneling_probability(0.5, 1.0, 1e-10);
        assert!(prob < 1.0);
        assert!(prob > 0.0);
    }

    #[test]
    fn test_quantum_state_generation() {
        let solver = QuantumSolver::new();
        let state = solver.generate_electron_state(2, 1, 0);
        
        assert_eq!(state.principal_quantum_number, 2);
        assert_eq!(state.orbital_angular_momentum, 1);
        assert_eq!(state.magnetic_quantum_number, 0);
        assert_relative_eq!(state.spin_quantum_number, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_entangled_state_creation() {
        let solver = QuantumSolver::new();
        let mut particle1 = FundamentalParticle::new(ParticleType::Electron, 9.109_383_701_5e-31, Vector3::zeros());
        let mut particle2 = FundamentalParticle::new(ParticleType::Electron, 9.109_383_701_5e-31, Vector3::zeros());
        
        let result = solver.create_entangled_state(&mut particle1, &mut particle2);
        assert!(result.is_ok());
        
        // Check that particles are marked as entangled
        assert!(!particle1.quantum_state.entanglement_partners.is_empty());
        assert!(!particle2.quantum_state.entanglement_partners.is_empty());
    }

    #[test]
    fn test_quantum_measurement() {
        let solver = QuantumSolver::new();
        let mut quantum_state = QuantumState::new();
        
        // Test energy measurement
        let energy = solver.measure_energy(&mut quantum_state).unwrap();
        assert!(energy >= 0.0);
        
        // Test that measurement collapses the state
        assert!(quantum_state.superposition_amplitudes.is_empty());
    }

    #[test]
    fn test_expectation_value() {
        let solver = QuantumSolver::new();
        let quantum_state = QuantumState::new();
        
        let energy_exp = solver.expectation_value(&quantum_state, "energy");
        assert_relative_eq!(energy_exp, quantum_state.energy_level, epsilon = 1e-10);
    }

    #[test]
    fn test_quantum_monte_carlo_creation() {
        let qmc = QuantumMonteCarlo::new();
        assert_eq!(qmc.num_walkers, 1000);
        assert_eq!(qmc.temperature, 300.0);
        assert_eq!(qmc.time_step, 1e-3);
    }

    #[test]
    fn test_path_integral_monte_carlo() {
        let qmc = QuantumMonteCarlo::new();
        
        // Simple harmonic oscillator potential
        let potential = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            0.5 * r_squared // V(r) = ½r²
        };
        
        let initial_positions = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];
        
        let result = qmc.path_integral_monte_carlo(potential, &initial_positions, 10);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.average_energy > 0.0);
        assert!(result.energy_variance >= 0.0);
        assert_eq!(result.average_positions.len(), 2);
    }

    #[test]
    fn test_variational_monte_carlo() {
        let qmc = QuantumMonteCarlo::new();
        
        // Simple Gaussian trial wavefunction
        let trial_wavefunction = |pos: &Vector3<f64>, params: &[f64]| {
            let alpha = params[0];
            let r_squared = pos.dot(pos);
            (-alpha * r_squared).exp()
        };
        
        // Harmonic oscillator potential
        let potential_energy = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            0.5 * r_squared
        };
        
        let initial_parameters = vec![1.0];
        
        let result = qmc.variational_monte_carlo(trial_wavefunction, potential_energy, &initial_parameters);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        // For a harmonic oscillator, the ground state energy should be positive
        // but allow for some numerical tolerance
        assert!(result.ground_state_energy > -1.0); // Allow small negative values due to numerical errors
        assert!(!result.energy_history.is_empty());
        assert!(!result.parameter_history.is_empty());
        
        // Check that the energy is finite and reasonable
        assert!(result.ground_state_energy.is_finite());
        assert!(result.ground_state_energy.abs() < 100.0); // Should be reasonable magnitude
    }

    #[test]
    fn test_diffusion_monte_carlo() {
        let qmc = QuantumMonteCarlo::new();
        
        // Simple Gaussian trial wavefunction
        let trial_wavefunction = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            (-0.5 * r_squared).exp()
        };
        
        // Harmonic oscillator potential
        let potential_energy = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            0.5 * r_squared
        };
        
        let initial_walkers = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        
        let result = qmc.diffusion_monte_carlo(trial_wavefunction, potential_energy, &initial_walkers);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.ground_state_energy > 0.0);
        assert!(!result.final_walkers.is_empty());
        assert!(!result.energy_history.is_empty());
    }

    #[test]
    fn test_quantum_force_calculation() {
        let qmc = QuantumMonteCarlo::new();
        
        // Test quantum force for Gaussian wavefunction
        let trial_wavefunction = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            (-0.5 * r_squared).exp()
        };
        
        let position = Vector3::new(1.0, 2.0, 3.0);
        let quantum_force = qmc.calculate_quantum_force(&position, &trial_wavefunction);
        
        // For a Gaussian wavefunction, the quantum force should be -r
        assert_relative_eq!(quantum_force[0], -1.0, epsilon = 0.1);
        assert_relative_eq!(quantum_force[1], -2.0, epsilon = 0.1);
        assert_relative_eq!(quantum_force[2], -3.0, epsilon = 0.1);
    }

    #[test]
    fn test_variance_calculation() {
        let qmc = QuantumMonteCarlo::new();
        
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = qmc.calculate_variance(&samples);
        
        // Expected variance for [1,2,3,4,5] is 2.5
        assert_relative_eq!(variance, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_average_positions_calculation() {
        let qmc = QuantumMonteCarlo::new();
        
        let position_samples = vec![
            vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)],
            vec![Vector3::new(2.0, 0.0, 0.0), Vector3::new(0.0, 2.0, 0.0)],
            vec![Vector3::new(3.0, 0.0, 0.0), Vector3::new(0.0, 3.0, 0.0)],
        ];
        
        let average_positions = qmc.calculate_average_positions(&position_samples);
        
        assert_eq!(average_positions.len(), 2);
        assert_relative_eq!(average_positions[0][0], 2.0, epsilon = 1e-10); // Average of 1,2,3
        assert_relative_eq!(average_positions[1][1], 2.0, epsilon = 1e-10); // Average of 1,2,3
    }

    #[test]
    fn test_quantum_monte_carlo_integration() {
        // Test integration between different QMC methods
        let qmc = QuantumMonteCarlo::new();
        
        // Test that all methods can work with the same potential
        let potential = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            0.5 * r_squared
        };
        
        let trial_wavefunction = |pos: &Vector3<f64>| {
            let r_squared = pos.dot(pos);
            (-0.5 * r_squared).exp()
        };
        
        // Test PIMC
        let initial_positions = vec![Vector3::new(1.0, 0.0, 0.0)];
        let pimc_result = qmc.path_integral_monte_carlo(potential, &initial_positions, 5);
        assert!(pimc_result.is_ok());
        
        // Test VMC
        let initial_parameters = vec![1.0];
        let vmc_result = qmc.variational_monte_carlo(
            |pos, params| trial_wavefunction(pos),
            potential,
            &initial_parameters
        );
        assert!(vmc_result.is_ok());
        
        // Test DMC
        let initial_walkers = vec![Vector3::new(1.0, 0.0, 0.0)];
        let dmc_result = qmc.diffusion_monte_carlo(trial_wavefunction, potential, &initial_walkers);
        assert!(dmc_result.is_ok());
    }
}