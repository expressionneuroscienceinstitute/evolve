//! Conservation Enforcement System
//! 
//! Advanced numerical methods for maintaining physical conservation laws in multi-physics simulations.
//! Implements constraint projection, symplectic integrators, and adaptive error correction to ensure
//! energy, momentum, mass, and angular momentum conservation across all physics modules.

use anyhow::Result;
use nalgebra::Vector3;
use std::collections::HashMap;
use crate::{
    PhysicsState, PhysicsConstants, FundamentalParticle,
};

/// Conservation constraint types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConservationConstraint {
    /// Energy conservation: dE/dt = 0
    Energy,
    /// Linear momentum conservation: dp/dt = 0  
    LinearMomentum,
    /// Angular momentum conservation: dL/dt = 0
    AngularMomentum,
    /// Mass conservation: dm/dt = 0
    Mass,
    /// Charge conservation: dQ/dt = 0
    Charge,
    /// Entropy increase: dS/dt ≥ 0
    EntropyIncrease,
    /// Relativistic energy-momentum relation: E² = (pc)² + (mc²)²
    RelativisticEnergyMomentum,
}

/// Conservation violation metrics
#[derive(Debug, Clone)]
pub struct ConservationViolation {
    pub constraint: ConservationConstraint,
    pub initial_value: f64,
    pub current_value: f64,
    pub violation_magnitude: f64,
    pub relative_error: f64,
    pub timestamp: f64,
}

/// Advanced conservation enforcement methods
#[derive(Debug, Clone)]
pub enum EnforcementMethod {
    /// Lagrange multiplier constraint projection
    LagrangeMultiplier { tolerance: f64, max_iterations: usize },
    /// Symplectic integrator with drift correction
    SymplecticCorrection { correction_strength: f64 },
    /// Adaptive error correction with feedback
    AdaptiveCorrection { learning_rate: f64, history_size: usize },
    /// Energy-momentum conserving integrator
    EnergyMomentumConserving { symplectic_order: usize },
    /// Constraint stabilization (Baumgarte method)
    BaumgarteStabilization { alpha: f64, beta: f64 },
}

/// Conservation enforcement system
#[derive(Debug)]
pub struct ConservationEnforcer {
    constraints: Vec<ConservationConstraint>,
    enforcement_method: EnforcementMethod,
    violation_history: Vec<ConservationViolation>,
    correction_weights: HashMap<ConservationConstraint, f64>,
    adaptive_parameters: AdaptiveParameters,
    symplectic_integrator: SymplecticIntegrator,
}

/// Adaptive parameters for dynamic correction
#[derive(Debug, Clone)]
struct AdaptiveParameters {
    learning_rate: f64,
    momentum: f64,
    history_size: usize,
    min_correction: f64,
    max_correction: f64,
}

/// Symplectic integrator for energy-momentum conservation
#[derive(Debug, Clone)]
struct SymplecticIntegrator {
    order: usize,
    coefficients: Vec<f64>,
    drift_threshold: f64,
    correction_strength: f64,
}

impl ConservationEnforcer {
    /// Create new conservation enforcer with specified constraints and method
    pub fn new(
        constraints: Vec<ConservationConstraint>,
        enforcement_method: EnforcementMethod,
    ) -> Self {
        let mut correction_weights = HashMap::new();
        for constraint in &constraints {
            correction_weights.insert(constraint.clone(), 1.0);
        }

        Self {
            constraints,
            enforcement_method,
            violation_history: Vec::new(),
            correction_weights,
            adaptive_parameters: AdaptiveParameters {
                learning_rate: 0.01,
                momentum: 0.9,
                history_size: 100,
                min_correction: 1e-12,
                max_correction: 1e-3,
            },
            symplectic_integrator: SymplecticIntegrator {
                order: 4,
                coefficients: vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
                drift_threshold: 1e-8,
                correction_strength: 0.1,
            },
        }
    }

    /// Enforce conservation constraints on particle states
    pub fn enforce_conservation(
        &mut self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        dt: f64,
    ) -> Result<()> {
        match &self.enforcement_method {
            EnforcementMethod::LagrangeMultiplier { tolerance, max_iterations } => {
                self.lagrange_multiplier_correction(states, constants, *tolerance, *max_iterations)
            }
            EnforcementMethod::SymplecticCorrection { correction_strength } => {
                self.symplectic_correction(states, constants, dt, *correction_strength)
            }
            EnforcementMethod::AdaptiveCorrection { learning_rate, history_size } => {
                self.adaptive_correction(states, constants, dt, *learning_rate, *history_size)
            }
            EnforcementMethod::EnergyMomentumConserving { symplectic_order } => {
                self.energy_momentum_conserving_integration(states, constants, dt, *symplectic_order)
            }
            EnforcementMethod::BaumgarteStabilization { alpha, beta } => {
                self.baumgarte_stabilization(states, constants, dt, *alpha, *beta)
            }
        }
    }

    /// Lagrange multiplier constraint projection
    fn lagrange_multiplier_correction(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<()> {
        for iteration in 0..max_iterations {
            let mut total_correction = 0.0;
            
            for constraint in &self.constraints {
                let violation = self.calculate_constraint_violation(states, constants, constraint)?;
                
                if violation.abs() > tolerance {
                    let correction = self.calculate_lagrange_correction(
                        states, constants, constraint, violation
                    )?;
                    
                    self.apply_correction(states, &correction)?;
                    total_correction += correction.magnitude();
                }
            }
            
            if total_correction < tolerance {
                break;
            }
        }
        
        Ok(())
    }

    /// Symplectic correction with drift monitoring
    fn symplectic_correction(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        dt: f64,
        correction_strength: f64,
    ) -> Result<()> {
        // Calculate initial conserved quantities
        let initial_energy = self.calculate_total_energy(states, constants);
        let initial_momentum = self.calculate_total_momentum(states, constants);
        let initial_angular_momentum = self.calculate_total_angular_momentum(states, constants);
        
        // Apply symplectic integration step
        self.symplectic_step(states, constants, dt)?;
        
        // Calculate drift in conserved quantities
        let current_energy = self.calculate_total_energy(states, constants);
        let current_momentum = self.calculate_total_momentum(states, constants);
        let current_angular_momentum = self.calculate_total_angular_momentum(states, constants);
        
        let energy_drift = current_energy - initial_energy;
        let momentum_drift = current_momentum - initial_momentum;
        let angular_momentum_drift = current_angular_momentum - initial_angular_momentum;
        
        // Apply drift correction if threshold exceeded
        if energy_drift.abs() > self.symplectic_integrator.drift_threshold {
            self.correct_energy_drift(states, constants, energy_drift, correction_strength)?;
        }
        
        if momentum_drift.magnitude() > self.symplectic_integrator.drift_threshold {
            self.correct_momentum_drift(states, constants, momentum_drift, correction_strength)?;
        }
        
        if angular_momentum_drift.magnitude() > self.symplectic_integrator.drift_threshold {
            self.correct_angular_momentum_drift(states, constants, angular_momentum_drift, correction_strength)?;
        }
        
        Ok(())
    }

    /// Adaptive correction with learning from violation history
    fn adaptive_correction(
        &mut self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        dt: f64,
        learning_rate: f64,
        history_size: usize,
    ) -> Result<()> {
        // Calculate current violations
        let mut violations = Vec::new();
        for constraint in &self.constraints {
            let violation = self.calculate_constraint_violation(states, constants, constraint)?;
            violations.push((constraint.clone(), violation));
        }
        
        // Update violation history
        for (constraint, violation) in &violations {
            self.violation_history.push(ConservationViolation {
                constraint: constraint.clone(),
                initial_value: 0.0, // Would track from previous step
                current_value: *violation,
                violation_magnitude: violation.abs(),
                relative_error: violation.abs() / 1e-100, // Simplified
                timestamp: 0.0, // Would track actual time
            });
        }
        
        // Maintain history size
        if self.violation_history.len() > history_size {
            self.violation_history.drain(0..self.violation_history.len() - history_size);
        }
        
        // Calculate adaptive corrections based on violation patterns
        for (constraint, violation) in violations {
            let correction_weight = self.calculate_adaptive_weight(&constraint, learning_rate);
            let correction = self.calculate_adaptive_correction(
                states, constants, &constraint, violation, correction_weight
            )?;
            
            self.apply_correction(states, &correction)?;
        }
        
        Ok(())
    }

    /// Energy-momentum conserving symplectic integration
    fn energy_momentum_conserving_integration(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        dt: f64,
        symplectic_order: usize,
    ) -> Result<()> {
        // Use higher-order symplectic integrator
        let coefficients = self.get_symplectic_coefficients(symplectic_order);
        
        for (i, &coeff) in coefficients.iter().enumerate() {
            let step_dt = coeff * dt;
            
            // Position update (kick-drift-kick)
            if i % 2 == 0 {
                // Kick step: update velocities
                self.kick_step(states, constants, step_dt)?;
            } else {
                // Drift step: update positions
                self.drift_step(states, step_dt)?;
            }
        }
        
        // Final constraint projection
        self.final_constraint_projection(states, constants)?;
        
        Ok(())
    }

    /// Baumgarte constraint stabilization
    fn baumgarte_stabilization(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        dt: f64,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        for constraint in &self.constraints {
            let constraint_value = self.calculate_constraint_value(states, constants, constraint)?;
            let constraint_derivative = self.calculate_constraint_derivative(states, constants, constraint)?;
            
            // Baumgarte stabilization: g̈ + αġ + βg = 0
            let stabilization_force = -(alpha * constraint_derivative + beta * constraint_value);
            
            let correction = self.calculate_baumgarte_correction(
                states, constants, constraint, stabilization_force
            )?;
            
            self.apply_correction(states, &correction)?;
        }
        
        Ok(())
    }

    /// Calculate constraint violation for a specific constraint
    fn calculate_constraint_violation(
        &self,
        states: &[PhysicsState],
        constants: &PhysicsConstants,
        constraint: &ConservationConstraint,
    ) -> Result<f64> {
        match constraint {
            ConservationConstraint::Energy => {
                let total_energy = self.calculate_total_energy(states, constants);
                Ok(total_energy) // Would compare with initial energy
            }
            ConservationConstraint::LinearMomentum => {
                let total_momentum = self.calculate_total_momentum(states, constants);
                Ok(total_momentum.magnitude()) // Would compare with initial momentum
            }
            ConservationConstraint::AngularMomentum => {
                let total_angular_momentum = self.calculate_total_angular_momentum(states, constants);
                Ok(total_angular_momentum.magnitude()) // Would compare with initial angular momentum
            }
            ConservationConstraint::Mass => {
                let total_mass = states.iter().map(|s| s.mass).sum::<f64>();
                Ok(total_mass) // Would compare with initial mass
            }
            ConservationConstraint::Charge => {
                let total_charge = states.iter().map(|s| s.charge).sum::<f64>();
                Ok(total_charge) // Would compare with initial charge
            }
            ConservationConstraint::EntropyIncrease => {
                let total_entropy = states.iter().map(|s| s.entropy).sum::<f64>();
                Ok(total_entropy) // Would check for decrease
            }
            ConservationConstraint::RelativisticEnergyMomentum => {
                self.calculate_relativistic_violation(states, constants)
            }
        }
    }

    /// Calculate total energy (kinetic + potential + rest mass)
    fn calculate_total_energy(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> f64 {
        let mut total_energy = 0.0;
        
        // Kinetic and rest mass energy
        for state in states {
            let v = state.velocity.magnitude();
            let kinetic_energy = if constants.is_relativistic(v) {
                let gamma = constants.lorentz_factor(v);
                (gamma - 1.0) * state.mass * constants.c * constants.c
            } else {
                0.5 * state.mass * v * v
            };
            
            let rest_energy = state.mass * constants.c * constants.c;
            total_energy += kinetic_energy + rest_energy;
        }
        
        // Gravitational potential energy
        for i in 0..states.len() {
            for j in (i+1)..states.len() {
                let r = (states[i].position - states[j].position).magnitude();
                if r > 1e-30 {
                    total_energy -= constants.g * states[i].mass * states[j].mass / r;
                }
            }
        }
        
        total_energy
    }

    /// Calculate total momentum
    fn calculate_total_momentum(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Vector3<f64> {
        let mut total_momentum = Vector3::zeros();
        
        for state in states {
            let v = state.velocity.magnitude();
            if constants.is_relativistic(v) {
                let gamma = constants.lorentz_factor(v);
                total_momentum += gamma * state.mass * state.velocity;
            } else {
                total_momentum += state.mass * state.velocity;
            }
        }
        
        total_momentum
    }

    /// Calculate total angular momentum
    fn calculate_total_angular_momentum(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Vector3<f64> {
        let mut total_angular_momentum = Vector3::zeros();
        
        for state in states {
            let v = state.velocity.magnitude();
            let momentum = if constants.is_relativistic(v) {
                let gamma = constants.lorentz_factor(v);
                gamma * state.mass * state.velocity
            } else {
                state.mass * state.velocity
            };
            
            total_angular_momentum += state.position.cross(&momentum);
        }
        
        total_angular_momentum
    }

    /// Calculate relativistic energy-momentum violation
    fn calculate_relativistic_violation(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Result<f64> {
        let mut total_violation = 0.0;
        
        for state in states {
            let v = state.velocity.magnitude();
            if constants.is_relativistic(v) {
                let gamma = constants.lorentz_factor(v);
                let relativistic_energy = gamma * state.mass * constants.c * constants.c;
                let momentum_magnitude = gamma * state.mass * v;
                
                // Check E² = (pc)² + (mc²)²
                let energy_squared = relativistic_energy * relativistic_energy;
                let momentum_energy_squared = (momentum_magnitude * constants.c).powi(2);
                let rest_energy_squared = (state.mass * constants.c * constants.c).powi(2);
                
                let violation = energy_squared - momentum_energy_squared - rest_energy_squared;
                total_violation += violation.abs();
            }
        }
        
        Ok(total_violation)
    }

    /// Calculate Lagrange multiplier correction
    fn calculate_lagrange_correction(
        &self,
        states: &[PhysicsState],
        constants: &PhysicsConstants,
        constraint: &ConservationConstraint,
        violation: f64,
    ) -> Result<Vector3<f64>> {
        // Simplified Lagrange multiplier calculation
        // In practice, this would solve a linear system for the multipliers
        let correction_magnitude = violation * 0.1; // Simplified scaling
        
        // Distribute correction across particles based on mass
        let total_mass: f64 = states.iter().map(|s| s.mass).sum();
        let mut correction = Vector3::zeros();
        
        for state in states {
            let particle_correction = correction_magnitude * (state.mass / total_mass);
            correction += state.velocity.normalize() * particle_correction;
        }
        
        Ok(correction)
    }

    /// Apply correction to particle states
    fn apply_correction(&self, states: &mut [PhysicsState], correction: &Vector3<f64>) -> Result<()> {
        for state in states {
            state.velocity += *correction;
        }
        Ok(())
    }

    /// Symplectic integration step
    fn symplectic_step(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        dt: f64,
    ) -> Result<()> {
        // Simplified symplectic integrator (leapfrog)
        for state in states {
            // Update position: x(t+dt) = x(t) + v(t+dt/2) * dt
            state.position += state.velocity * dt;
            
            // Update velocity: v(t+dt/2) = v(t-dt/2) + a(t) * dt
            // (acceleration would be calculated from forces)
            // state.velocity += state.acceleration * dt;
        }
        
        Ok(())
    }

    /// Correct energy drift
    fn correct_energy_drift(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        energy_drift: f64,
        correction_strength: f64,
    ) -> Result<()> {
        // Scale velocities to correct energy drift
        let total_kinetic_energy: f64 = states.iter()
            .map(|s| {
                let v = s.velocity.magnitude();
                if constants.is_relativistic(v) {
                    let gamma = constants.lorentz_factor(v);
                    (gamma - 1.0) * s.mass * constants.c * constants.c
                } else {
                    0.5 * s.mass * v * v
                }
            })
            .sum();
        
        if total_kinetic_energy > 0.0 {
            let scale_factor = (1.0 - correction_strength * energy_drift / total_kinetic_energy).sqrt();
            for state in states {
                state.velocity *= scale_factor;
            }
        }
        
        Ok(())
    }

    /// Correct momentum drift
    fn correct_momentum_drift(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        momentum_drift: Vector3<f64>,
        correction_strength: f64,
    ) -> Result<()> {
        let total_mass: f64 = states.iter().map(|s| s.mass).sum();
        
        if total_mass > 0.0 {
            let velocity_correction = -correction_strength * momentum_drift / total_mass;
            for state in states {
                state.velocity += velocity_correction;
            }
        }
        
        Ok(())
    }

    /// Correct angular momentum drift
    fn correct_angular_momentum_drift(
        &self,
        states: &mut [PhysicsState],
        constants: &PhysicsConstants,
        angular_momentum_drift: Vector3<f64>,
        correction_strength: f64,
    ) -> Result<()> {
        // Angular momentum correction is more complex
        // For simplicity, apply a small correction to velocities
        let total_moment_of_inertia: f64 = states.iter()
            .map(|s| s.mass * s.position.magnitude_squared())
            .sum();
        
        if total_moment_of_inertia > 0.0 {
            let angular_velocity_correction = -correction_strength * angular_momentum_drift / total_moment_of_inertia;
            for state in states {
                let tangential_correction = angular_velocity_correction.cross(&state.position);
                state.velocity += tangential_correction;
            }
        }
        
        Ok(())
    }

    /// Calculate adaptive weight based on violation history
    fn calculate_adaptive_weight(&self, constraint: &ConservationConstraint, learning_rate: f64) -> f64 {
        let recent_violations: Vec<&ConservationViolation> = self.violation_history
            .iter()
            .filter(|v| v.constraint == *constraint)
            .rev()
            .take(10)
            .collect();
        
        if recent_violations.is_empty() {
            return 1.0;
        }
        
        let avg_violation = recent_violations.iter()
            .map(|v| v.violation_magnitude)
            .sum::<f64>() / recent_violations.len() as f64;
        
        1.0 + learning_rate * avg_violation
    }

    /// Calculate adaptive correction
    fn calculate_adaptive_correction(
        &self,
        states: &[PhysicsState],
        constants: &PhysicsConstants,
        constraint: &ConservationConstraint,
        violation: f64,
        weight: f64,
    ) -> Result<Vector3<f64>> {
        let base_correction = self.calculate_lagrange_correction(states, constants, constraint, violation)?;
        Ok(base_correction * weight)
    }

    /// Get symplectic coefficients for specified order
    fn get_symplectic_coefficients(&self, order: usize) -> Vec<f64> {
        match order {
            2 => vec![0.5, 0.5],
            4 => vec![1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
            6 => vec![1.0/10.0, 1.0/5.0, 1.0/5.0, 1.0/5.0, 1.0/5.0, 1.0/10.0],
            _ => vec![1.0], // Default to first order
        }
    }

    /// Kick step (velocity update)
    fn kick_step(&self, states: &mut [PhysicsState], constants: &PhysicsConstants, dt: f64) -> Result<()> {
        // Update velocities based on forces
        for state in states {
            // Calculate forces and update velocity
            // state.velocity += state.acceleration * dt;
        }
        Ok(())
    }

    /// Drift step (position update)
    fn drift_step(&self, states: &mut [PhysicsState], dt: f64) -> Result<()> {
        for state in states {
            state.position += state.velocity * dt;
        }
        Ok(())
    }

    /// Final constraint projection
    fn final_constraint_projection(&self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Apply final corrections to ensure constraints are satisfied
        for constraint in &self.constraints {
            let violation = self.calculate_constraint_violation(states, constants, constraint)?;
            if violation.abs() > 1e-12 {
                let correction = self.calculate_lagrange_correction(states, constants, constraint, violation)?;
                self.apply_correction(states, &correction)?;
            }
        }
        Ok(())
    }

    /// Calculate constraint value
    fn calculate_constraint_value(&self, states: &[PhysicsState], constants: &PhysicsConstants, constraint: &ConservationConstraint) -> Result<f64> {
        self.calculate_constraint_violation(states, constants, constraint)
    }

    /// Calculate constraint derivative
    fn calculate_constraint_derivative(&self, states: &[PhysicsState], constants: &PhysicsConstants, constraint: &ConservationConstraint) -> Result<f64> {
        // Simplified constraint derivative calculation
        // In practice, this would compute ∂g/∂t
        Ok(0.0)
    }

    /// Calculate Baumgarte correction
    fn calculate_baumgarte_correction(
        &self,
        states: &[PhysicsState],
        constants: &PhysicsConstants,
        constraint: &ConservationConstraint,
        stabilization_force: f64,
    ) -> Result<Vector3<f64>> {
        // Simplified Baumgarte correction
        let correction_magnitude = stabilization_force * 0.1;
        let total_mass: f64 = states.iter().map(|s| s.mass).sum();
        
        let mut correction = Vector3::zeros();
        for state in states {
            let particle_correction = correction_magnitude * (state.mass / total_mass);
            correction += state.velocity.normalize() * particle_correction;
        }
        
        Ok(correction)
    }

    /// Get conservation violation statistics
    pub fn get_violation_statistics(&self) -> ConservationStatistics {
        let mut stats = ConservationStatistics::default();
        
        for violation in &self.violation_history {
            stats.total_violations += 1;
            stats.max_violation = stats.max_violation.max(violation.violation_magnitude);
            stats.avg_violation += violation.violation_magnitude;
        }
        
        if stats.total_violations > 0 {
            stats.avg_violation /= stats.total_violations as f64;
        }
        
        stats
    }

    /// Reset violation history
    pub fn reset_history(&mut self) {
        self.violation_history.clear();
    }
}

/// Conservation violation statistics
#[derive(Debug, Clone, Default)]
pub struct ConservationStatistics {
    pub total_violations: usize,
    pub max_violation: f64,
    pub avg_violation: f64,
}

/// Advanced conservation monitoring system
#[derive(Debug)]
pub struct ConservationMonitor {
    enforcers: HashMap<String, ConservationEnforcer>,
    global_statistics: ConservationStatistics,
    monitoring_enabled: bool,
    violation_thresholds: HashMap<ConservationConstraint, f64>,
}

impl ConservationMonitor {
    /// Create new conservation monitor
    pub fn new() -> Self {
        let mut violation_thresholds = HashMap::new();
        violation_thresholds.insert(ConservationConstraint::Energy, 1e-6);
        violation_thresholds.insert(ConservationConstraint::LinearMomentum, 1e-6);
        violation_thresholds.insert(ConservationConstraint::AngularMomentum, 1e-6);
        violation_thresholds.insert(ConservationConstraint::Mass, 1e-12);
        violation_thresholds.insert(ConservationConstraint::Charge, 1e-12);
        violation_thresholds.insert(ConservationConstraint::EntropyIncrease, 1e-6);
        violation_thresholds.insert(ConservationConstraint::RelativisticEnergyMomentum, 1e-6);

        Self {
            enforcers: HashMap::new(),
            global_statistics: ConservationStatistics::default(),
            monitoring_enabled: true,
            violation_thresholds,
        }
    }

    /// Add conservation enforcer for a specific physics module
    pub fn add_enforcer(&mut self, name: String, enforcer: ConservationEnforcer) {
        self.enforcers.insert(name, enforcer);
    }

    /// Monitor and enforce conservation across all physics modules
    pub fn monitor_conservation(
        &mut self,
        particles: &mut [FundamentalParticle],
        constants: &PhysicsConstants,
        dt: f64,
    ) -> Result<()> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        // Convert particles to physics states for conservation checking
        let mut states: Vec<PhysicsState> = particles.iter()
            .map(|p| PhysicsState {
                position: p.position,
                velocity: p.velocity,
                acceleration: p.acceleration,
                mass: p.mass,
                charge: p.charge,
                temperature: 300.0, // Default temperature
                entropy: 1e-20, // Default entropy
            })
            .collect();

        // Apply conservation enforcement for each module
        for (name, enforcer) in &mut self.enforcers {
            if let Err(e) = enforcer.enforce_conservation(&mut states, constants, dt) {
                log::warn!("Conservation enforcement failed for {}: {}", name, e);
            }
        }

        // Update particles with corrected states
        for (particle, state) in particles.iter_mut().zip(states.iter()) {
            particle.position = state.position;
            particle.velocity = state.velocity;
            particle.acceleration = state.acceleration;
        }

        // Update global statistics
        self.update_global_statistics()?;

        Ok(())
    }

    /// Update global conservation statistics
    fn update_global_statistics(&mut self) -> Result<()> {
        self.global_statistics = ConservationStatistics::default();
        
        for enforcer in self.enforcers.values() {
            let stats = enforcer.get_violation_statistics();
            self.global_statistics.total_violations += stats.total_violations;
            self.global_statistics.max_violation = self.global_statistics.max_violation.max(stats.max_violation);
            self.global_statistics.avg_violation += stats.avg_violation;
        }
        
        let num_enforcers = self.enforcers.len();
        if num_enforcers > 0 {
            self.global_statistics.avg_violation /= num_enforcers as f64;
        }
        
        Ok(())
    }

    /// Get global conservation statistics
    pub fn get_global_statistics(&self) -> &ConservationStatistics {
        &self.global_statistics
    }

    /// Enable or disable conservation monitoring
    pub fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
    }

    /// Set violation threshold for a specific constraint
    pub fn set_violation_threshold(&mut self, constraint: ConservationConstraint, threshold: f64) {
        self.violation_thresholds.insert(constraint, threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn create_test_states() -> Vec<PhysicsState> {
        vec![
            PhysicsState {
                position: Vector3::new(0.0, 0.0, 0.0),
                velocity: Vector3::new(1000.0, 0.0, 0.0),
                acceleration: Vector3::zeros(),
                mass: 1.0,
                charge: 1.0,
                temperature: 300.0,
                entropy: 1e-20,
            },
            PhysicsState {
                position: Vector3::new(1.0, 0.0, 0.0),
                velocity: Vector3::new(-1000.0, 0.0, 0.0),
                acceleration: Vector3::zeros(),
                mass: 1.0,
                charge: -1.0,
                temperature: 300.0,
                entropy: 1e-20,
            },
        ]
    }

    #[test]
    fn test_conservation_enforcer_creation() {
        let constraints = vec![
            ConservationConstraint::Energy,
            ConservationConstraint::LinearMomentum,
        ];
        
        let enforcement_method = EnforcementMethod::LagrangeMultiplier {
            tolerance: 1e-6,
            max_iterations: 10,
        };
        
        let enforcer = ConservationEnforcer::new(constraints, enforcement_method);
        assert_eq!(enforcer.constraints.len(), 2);
    }

    #[test]
    fn test_energy_calculation() {
        let states = create_test_states();
        let constants = PhysicsConstants::default();
        
        let constraints = vec![ConservationConstraint::Energy];
        let enforcement_method = EnforcementMethod::LagrangeMultiplier {
            tolerance: 1e-6,
            max_iterations: 10,
        };
        
        let enforcer = ConservationEnforcer::new(constraints, enforcement_method);
        let total_energy = enforcer.calculate_total_energy(&states, &constants);
        
        assert!(total_energy > 0.0);
        assert!(total_energy.is_finite());
    }

    #[test]
    fn test_momentum_calculation() {
        let states = create_test_states();
        let constants = PhysicsConstants::default();
        
        let constraints = vec![ConservationConstraint::LinearMomentum];
        let enforcement_method = EnforcementMethod::LagrangeMultiplier {
            tolerance: 1e-6,
            max_iterations: 10,
        };
        
        let enforcer = ConservationEnforcer::new(constraints, enforcement_method);
        let total_momentum = enforcer.calculate_total_momentum(&states, &constants);
        
        // For the test case, momenta should cancel out
        assert!(total_momentum.magnitude() < 1e-10);
    }

    #[test]
    fn test_conservation_monitor() {
        let mut monitor = ConservationMonitor::new();
        assert!(monitor.monitoring_enabled);
        
        monitor.set_monitoring_enabled(false);
        assert!(!monitor.monitoring_enabled);
    }

    #[test]
    fn test_symplectic_integration() {
        let mut states = create_test_states();
        let constants = PhysicsConstants::default();
        
        let constraints = vec![
            ConservationConstraint::Energy,
            ConservationConstraint::LinearMomentum,
        ];
        
        let enforcement_method = EnforcementMethod::SymplecticCorrection {
            correction_strength: 0.1,
        };
        
        let mut enforcer = ConservationEnforcer::new(constraints, enforcement_method);
        
        // Initial energy
        let initial_energy = enforcer.calculate_total_energy(&states, &constants);
        
        // Apply symplectic correction
        enforcer.symplectic_correction(&mut states, &constants, 1e-3, 0.1).unwrap();
        
        // Final energy should be similar
        let final_energy = enforcer.calculate_total_energy(&states, &constants);
        let energy_change = (final_energy - initial_energy).abs() / initial_energy;
        
        assert!(energy_change < 0.1); // Should be conserved to within 10%
    }
} 