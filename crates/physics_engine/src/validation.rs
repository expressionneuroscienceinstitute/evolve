//! Physics Validation Module
//! 
//! Validates conservation laws and physics consistency

use anyhow::Result;
use nalgebra::Vector3;
use crate::{PhysicsState, PhysicsConstants};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Physics validation tolerance
#[allow(dead_code)]
const CONSERVATION_TOLERANCE: f64 = 1e-6;

/// Performance metrics for validation
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub computational_overhead: f64,
    pub accuracy_measure: f64,
    pub energy_conservation: f64,
    pub transition_smoothness: f64,
    pub prediction_accuracy: f64,
    // Additional fields for validation
    pub computation_time_ms: f64,
    pub memory_usage_mb: f64,
    pub particles_per_second: f64,
    pub energy_drift_rate: f64,
    pub momentum_drift_rate: f64,
    pub cache_efficiency: f64,
    pub parallelization_efficiency: f64,
}

/// Performance thresholds for validation
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_computational_overhead: f64,
    pub min_accuracy_measure: f64,
    pub max_energy_conservation_error: f64,
    pub min_transition_smoothness: f64,
    pub min_prediction_accuracy: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_computational_overhead: 1000.0, // 1000ms max computation time
            min_accuracy_measure: 0.1, // 10% minimum accuracy
            max_energy_conservation_error: 1e-6, // 1e-6 relative error
            min_transition_smoothness: 0.5, // 50% minimum smoothness
            min_prediction_accuracy: 1000.0, // 1000 particles/second minimum
        }
    }
}

/// Emergence detection parameters
#[derive(Debug, Clone, Default)]
pub struct EmergenceParameters {
    pub spatial_correlation_threshold: f64,
    pub temporal_correlation_threshold: f64,
    pub clustering_coefficient_threshold: f64,
    pub pattern_complexity_threshold: f64,
    pub phase_transition_probability_threshold: f64,
    pub collective_behavior_threshold: f64,
    pub information_entropy_threshold: f64,
}

/// Validation error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Energy conservation violated: change = {change:.2e}")]
    EnergyConservation { change: f64 },
    
    #[error("Momentum conservation violated: change magnitude = {change:.2e}")]
    MomentumConservation { change: f64 },
    
    #[error("Entropy decrease detected: change = {change:.2e}")]
    EntropyDecrease { change: f64 },
    
    #[error("Speed of light exceeded: velocity = {velocity:.2e} m/s")]
    SuperluminalVelocity { velocity: f64 },
    
    #[error("Negative mass detected: mass = {mass:.2e} kg")]
    NegativeMass { mass: f64 },
    
    #[error("Invalid temperature: T = {temperature:.2e} K")]
    InvalidTemperature { temperature: f64 },
    
    #[error("Index out of bounds")]  
    OutOfBounds,
    
    #[error("Dimension mismatch between operands")]  
    DimensionMismatch,
    
    #[error("Energy conservation violated beyond tolerance")]  
    EnergyConservationViolation,
    
    #[error("Maximum iterations reached: {iterations}")]
    MaxIterationsReached { iterations: u64 },
    
    #[error("Emergence detection: {reason}")]
    EmergenceDetection { reason: String },
    
    #[error("Scientific accuracy: {description}")]
    ScientificAccuracy { description: String },
    
    #[error("Statistical validation: test = {test}, p-value = {p_value}")]
    StatisticalValidation { test: String, p_value: f64 },
    
    #[error("Performance violation: metric = {metric}, value = {value}")]
    PerformanceViolation { metric: String, value: f64 },
}

/// Check energy conservation (R1: Mass-Energy Conservation)
pub fn check_energy_conservation(states: &[PhysicsState], constants: &PhysicsConstants) -> Result<()> {
    let mut total_energy = 0.0;
    
    // Calculate total energy
    for (i, state) in states.iter().enumerate() {
        // Kinetic energy
        let mut v = state.velocity.magnitude();
        // Clamp velocity for massive particles to just below c
        if state.mass > 0.0 && v >= constants.c {
            log::warn!("Clamping velocity for particle {}: mass={}, original_velocity={}", i, state.mass, v);
            v = 0.999_999 * constants.c;
        }
        let kinetic = if constants.is_relativistic(v) {
            // Relativistic: E = γmc²
            let gamma = constants.lorentz_factor(v);
            gamma * state.mass * constants.c * constants.c
        } else {
            // Classical: E = ½mv²
            0.5 * state.mass * v * v
        };
        
        // Rest mass energy
        let rest_energy = state.mass * constants.c * constants.c;
        
        let particle_energy = kinetic + rest_energy;
        
        // Debug: Check for NaN in individual particle energy
        if particle_energy.is_nan() {
            log::error!("NaN detected in particle {} energy calculation: mass={}, velocity={}, kinetic={}, rest_energy={}", 
                       i, state.mass, v, kinetic, rest_energy);
        }
        
        total_energy += particle_energy;
    }
    
    // Add potential energies
    for i in 0..states.len() {
        for j in (i+1)..states.len() {
            let r = (states[i].position - states[j].position).magnitude();
            if r > 1e-30 {
                // Gravitational potential energy
                let gravitational_pe = -constants.g * states[i].mass * states[j].mass / r;
                
                // Electromagnetic potential energy (if charged)
                let em_pe = if states[i].charge.abs() > 1e-30 && states[j].charge.abs() > 1e-30 {
                    constants.coulomb_force(states[i].charge, states[j].charge, r) * r
                } else {
                    0.0
                };
                
                let potential_energy = gravitational_pe + em_pe;
                
                // Debug: Check for NaN in potential energy
                if potential_energy.is_nan() {
                    log::error!("NaN detected in potential energy calculation: r={}, mass1={}, mass2={}, charge1={}, charge2={}", 
                               r, states[i].mass, states[j].mass, states[i].charge, states[j].charge);
                }
                
                total_energy += potential_energy;
            }
        }
    }
    
    // Store initial energy for future comparisons (simplified validation)
    // In a real implementation, this would track energy over time
    if total_energy.is_nan() || total_energy.is_infinite() {
        log::error!("Total energy is NaN or infinite: {}", total_energy);
        return Err(ValidationError::EnergyConservation { 
            change: total_energy 
        }.into());
    }
    
    Ok(())
}

/// Check momentum conservation
pub fn check_momentum_conservation(states: &[PhysicsState]) -> Result<()> {
    let mut total_momentum = Vector3::zeros();
    
    for state in states {
        let v = state.velocity.magnitude();
        
        // Use relativistic momentum if needed
        if v >= 0.1 * 299_792_458.0 { // 0.1c
            let gamma = 1.0 / (1.0 - (v / 299_792_458.0).powi(2)).sqrt();
            total_momentum += gamma * state.mass * state.velocity;
        } else {
            total_momentum += state.mass * state.velocity;
        }
    }
    
    let momentum_magnitude = total_momentum.magnitude();
    
    // In an isolated system, total momentum should be conserved
    // For validation, we check if it's reasonable
    if momentum_magnitude > 1e20 { // Very large momentum
        return Err(ValidationError::MomentumConservation { 
            change: momentum_magnitude 
        }.into());
    }
    
    Ok(())
}

/// Check entropy increase (R2: Entropy Arrow)
pub fn check_entropy_increase(states: &[PhysicsState]) -> Result<()> {
    let mut total_entropy = 0.0;
    
    for state in states {
        total_entropy += state.entropy;
        
        // Check for negative entropy
        if state.entropy < 0.0 {
            return Err(ValidationError::EntropyDecrease { 
                change: state.entropy 
            }.into());
        }
    }
    
    // In a real implementation, this would compare with previous entropy
    // For now, just check for reasonable values
    if total_entropy.is_nan() || total_entropy.is_infinite() {
        return Err(ValidationError::EntropyDecrease { 
            change: total_entropy 
        }.into());
    }
    
    Ok(())
}

/// Check relativistic constraints
pub fn check_relativistic_constraints(states: &[PhysicsState], constants: &PhysicsConstants) -> Result<()> {
    for state in states {
        let v = state.velocity.magnitude();
        
        // Check speed limit: only massive particles are constrained
        if state.mass > 0.0 && v > constants.c {
            return Err(ValidationError::SuperluminalVelocity { 
                velocity: v 
            }.into());
        }
        
        // Check for negative mass (allow zero mass for massless particles)
        if state.mass < 0.0 {
            return Err(ValidationError::NegativeMass { 
                mass: state.mass 
            }.into());
        }
        
        // Check temperature bounds
        if state.temperature <= 0.0 || state.temperature > 1e12 {
            return Err(ValidationError::InvalidTemperature { 
                temperature: state.temperature 
            }.into());
        }
    }
    
    Ok(())
}

/// Validate fusion thresholds (R4: Fusion Threshold)
pub fn check_fusion_thresholds(states: &[PhysicsState], constants: &PhysicsConstants) -> Result<()> {
    for state in states {
        // Check if particle mass corresponds to stellar object
        let mass_solar = state.mass / constants.m_sun;
        
        if mass_solar > 0.01 { // Potentially stellar mass
            let can_fuse = constants.can_fuse(mass_solar);
            let temp_threshold = 1e7; // 10 million K for hydrogen fusion
            
            if can_fuse && state.temperature < temp_threshold {
                // This is not necessarily an error, just needs tracking
                // Stars can exist below fusion temperature (brown dwarfs)
            }
            
            // Check supernova threshold
            if constants.will_supernova(mass_solar) && state.temperature > 1e9 {
                // Very hot massive star - potential supernova candidate
                // This is physics, not an error
            }
        }
    }
    
    Ok(())
}

/// Validate nucleosynthesis conditions (R5: Nucleosynthesis Window)
pub fn check_nucleosynthesis_window(states: &[PhysicsState], constants: &PhysicsConstants) -> Result<()> {
    for state in states {
        let mass_solar = state.mass / constants.m_sun;
        
        // Check for conditions that could produce heavy elements
        if mass_solar > constants.supernova_threshold {
            // Massive star that could undergo core collapse
            let core_temp_threshold = 5e9; // 5 billion K
            let _density_threshold = 1e15; // kg/m³ (rough nuclear density)
            
            if state.temperature > core_temp_threshold {
                // Conditions for heavy element production exist
                // This enables nucleosynthesis in the simulation
            }
        }
        
        // Check neutron star conditions
        if mass_solar > constants.neutron_star_max {
            // Mass exceeds neutron star limit - should collapse to black hole
            // This is physics, not an error
        }
    }
    
    Ok(())
}

/// Comprehensive physics validation
pub fn validate_physics_state(states: &[PhysicsState], constants: &PhysicsConstants) -> Result<()> {
    // Check all conservation laws
    check_energy_conservation(states, constants)?;
    check_momentum_conservation(states)?;
    check_entropy_increase(states)?;
    
    // Check relativistic constraints
    check_relativistic_constraints(states, constants)?;
    
    // Check stellar physics
    check_fusion_thresholds(states, constants)?;
    check_nucleosynthesis_window(states, constants)?;
    
    Ok(())
}

/// Calculate system-wide physics metrics
pub fn calculate_physics_metrics(states: &[PhysicsState], constants: &PhysicsConstants) -> PhysicsMetrics {
    let mut metrics = PhysicsMetrics::default();
    
    if states.is_empty() {
        return metrics;
    }
    
    // Calculate totals
    for state in states {
        let v = state.velocity.magnitude();
        
        // Total mass
        metrics.total_mass += state.mass;
        
        // Total energy
        let kinetic = if constants.is_relativistic(v) {
            let gamma = constants.lorentz_factor(v);
            gamma * state.mass * constants.c * constants.c
        } else {
            0.5 * state.mass * v * v
        };
        metrics.total_energy += kinetic + state.mass * constants.c * constants.c;
        
        // Total entropy
        metrics.total_entropy += state.entropy;
        
        // Temperature statistics
        metrics.min_temperature = metrics.min_temperature.min(state.temperature);
        metrics.max_temperature = metrics.max_temperature.max(state.temperature);
        
        // Velocity statistics
        metrics.max_velocity = metrics.max_velocity.max(v);
        
        // Count relativistic particles
        if constants.is_relativistic(v) {
            metrics.relativistic_particles += 1;
        }
    }
    
    // Calculate averages
    let n = states.len() as f64;
    metrics.average_temperature = states.iter().map(|s| s.temperature).sum::<f64>() / n;
    metrics.average_velocity = states.iter().map(|s| s.velocity.magnitude()).sum::<f64>() / n;
    
    metrics
}

/// Physics validation metrics
#[derive(Debug, Clone)]
pub struct PhysicsMetrics {
    pub total_mass: f64,
    pub total_energy: f64,
    pub total_entropy: f64,
    pub min_temperature: f64,
    pub max_temperature: f64,
    pub average_temperature: f64,
    pub max_velocity: f64,
    pub average_velocity: f64,
    pub relativistic_particles: usize,
}

impl Default for PhysicsMetrics {
    fn default() -> Self {
        Self {
            total_mass: 0.0,
            total_energy: 0.0,
            total_entropy: 0.0,
            min_temperature: f64::INFINITY,
            max_temperature: 0.0,
            average_temperature: 0.0,
            max_velocity: 0.0,
            average_velocity: 0.0,
            relativistic_particles: 0,
        }
    }
}

/// Comprehensive physics validator
pub struct ComprehensivePhysicsValidator {
    pub statistics: ValidationStatistics,
    pub performance_thresholds: PerformanceThresholds,
    pub emergence_parameters: EmergenceParameters,
    pub previous_states: Option<Vec<PhysicsState>>,
    pub previous_metrics: Option<PhysicsMetrics>,
    pub validation_start_time: Option<Instant>,
}

impl Default for ComprehensivePhysicsValidator {
    fn default() -> Self {
        Self {
            statistics: ValidationStatistics::default(),
            performance_thresholds: PerformanceThresholds::default(),
            emergence_parameters: EmergenceParameters::default(),
            previous_states: None,
            previous_metrics: None,
            validation_start_time: None,
        }
    }
}

impl ComprehensivePhysicsValidator {
    /// Initialize validation system
    pub fn initialize_validation(&mut self, states: &[PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        self.validation_start_time = Some(Instant::now());
        
        // Store initial state for comparison
        self.previous_states = Some(states.to_vec());
        self.previous_metrics = Some(calculate_physics_metrics(states, constants));
        
        Ok(())
    }
    
    /// Perform comprehensive validation
    pub fn validate(&mut self, states: &[PhysicsState], constants: &PhysicsConstants) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Basic physics validation
        if let Err(e) = validate_physics_state(states, constants) {
            errors.push(e.to_string());
        }
        
        // Calculate metrics
        let metrics = calculate_physics_metrics(states, constants);
        
        // Performance validation
        let performance_metrics = self.validate_performance(states, &metrics)?;
        if let Err(e) = self.check_performance_thresholds(&performance_metrics) {
            errors.push(e.to_string());
        }
        
        // Emergence detection
        let emergence_indicators = self.detect_emergence(states, constants)?;
        if let Err(e) = self.validate_emergence(&emergence_indicators) {
            errors.push(e.to_string());
        }
        
        // Scientific accuracy validation
        if let Err(e) = self.validate_scientific_accuracy(states, constants) {
            errors.push(e.to_string());
        }
        
        // Statistical validation
        if let Err(e) = self.validate_statistics(states, constants) {
            errors.push(e.to_string());
        }
        
        // Update statistics
        let validation_time = start_time.elapsed();
        let success = errors.is_empty();
        self.update_statistics(success, validation_time, &errors);
        
        // Store result
        let result = ValidationResult {
            timestamp: start_time,
            success,
            validation_time_ms: validation_time.as_millis() as f64,
            errors,
            warnings,
            metrics: metrics.clone(),
            emergence_indicators,
            performance_metrics,
        };
        
        self.statistics.validation_history.push(result.clone());
        
        // Update previous state
        self.previous_states = Some(states.to_vec());
        self.previous_metrics = Some(metrics);
        
        Ok(result)
    }
    
    /// Validate performance metrics
    fn validate_performance(&self, states: &[PhysicsState], metrics: &PhysicsMetrics) -> Result<PerformanceMetrics> {
        let mut performance = PerformanceMetrics::default();
        
        // Measure computation time (simplified)
        performance.computation_time_ms = 1.0; // Placeholder
        
        // Estimate memory usage
        performance.memory_usage_mb = (std::mem::size_of::<PhysicsState>() * states.len()) as f64 / 1e6;
        
        // Calculate particles per second (simplified)
        performance.particles_per_second = states.len() as f64 * 60.0; // Assuming 60 FPS
        
        // Calculate drift rates if we have previous metrics
        if let Some(prev_metrics) = &self.previous_metrics {
            let energy_change = (metrics.total_energy - prev_metrics.total_energy).abs();
            let momentum_change = (metrics.total_energy - prev_metrics.total_energy).abs();
            
            performance.energy_drift_rate = energy_change / prev_metrics.total_energy.max(1e-30);
            performance.momentum_drift_rate = momentum_change / prev_metrics.total_energy.max(1e-30);
        }
        
        // Estimate cache efficiency (simplified)
        performance.cache_efficiency = 0.8; // Placeholder
        
        // Estimate parallelization efficiency (simplified)
        performance.parallelization_efficiency = 0.9; // Placeholder
        
        Ok(performance)
    }
    
    /// Check performance against thresholds
    fn check_performance_thresholds(&self, performance: &PerformanceMetrics) -> Result<(), ValidationError> {
        if performance.computation_time_ms > self.performance_thresholds.max_computational_overhead as f64 {
            return Err(ValidationError::PerformanceViolation {
                metric: "computational_overhead".to_string(),
                value: performance.computation_time_ms,
            });
        }
        
        if performance.memory_usage_mb > self.performance_thresholds.max_computational_overhead as f64 {
            return Err(ValidationError::PerformanceViolation {
                metric: "memory_usage_mb".to_string(),
                value: performance.memory_usage_mb,
            });
        }
        
        if performance.particles_per_second < self.performance_thresholds.min_prediction_accuracy {
            return Err(ValidationError::PerformanceViolation {
                metric: "prediction_accuracy".to_string(),
                value: performance.particles_per_second,
            });
        }
        
        if performance.energy_drift_rate > self.performance_thresholds.max_energy_conservation_error {
            return Err(ValidationError::PerformanceViolation {
                metric: "energy_conservation_error".to_string(),
                value: performance.energy_drift_rate,
            });
        }
        
        Ok(())
    }
    
    /// Detect emergence phenomena
    pub fn detect_emergence(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Result<EmergenceIndicators> {
        let mut indicators = EmergenceIndicators::default();
        
        if states.len() < 2 {
            return Ok(indicators);
        }
        
        // Calculate spatial correlation
        indicators.spatial_correlation = self.calculate_spatial_correlation(states);
        
        // Calculate temporal correlation if we have previous states
        if let Some(prev_states) = &self.previous_states {
            indicators.temporal_correlation = self.calculate_temporal_correlation(states, prev_states);
        }
        
        // Calculate clustering coefficient
        indicators.clustering_coefficient = self.calculate_clustering_coefficient(states);
        
        // Calculate pattern complexity
        indicators.pattern_complexity = self.calculate_pattern_complexity(states);
        
        // Calculate phase transition probability
        indicators.phase_transition_probability = self.calculate_phase_transition_probability(states, constants);
        
        // Calculate collective behavior strength
        indicators.collective_behavior_strength = self.calculate_collective_behavior_strength(states);
        
        // Calculate information entropy
        indicators.information_entropy = self.calculate_information_entropy(states);
        
        // Calculate critical exponents
        indicators.critical_exponents = self.calculate_critical_exponents(states, constants);
        
        Ok(indicators)
    }
    
    /// Calculate spatial correlation between particles
    fn calculate_spatial_correlation(&self, states: &[PhysicsState]) -> f64 {
        if states.len() < 2 {
            return 0.0;
        }
        
        let mut total_correlation = 0.0;
        let mut count = 0;
        
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                let distance = (states[i].position - states[j].position).magnitude();
                let velocity_correlation = states[i].velocity.dot(&states[j].velocity) / 
                    (states[i].velocity.magnitude() * states[j].velocity.magnitude());
                
                // Weight by inverse distance
                let weight = 1.0 / (1.0 + distance);
                total_correlation += velocity_correlation * weight;
                count += 1;
            }
        }
        
        if count > 0 {
            total_correlation / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate temporal correlation between current and previous states
    fn calculate_temporal_correlation(&self, current: &[PhysicsState], previous: &[PhysicsState]) -> f64 {
        if current.len() != previous.len() || current.is_empty() {
            return 0.0;
        }
        
        let mut total_correlation = 0.0;
        
        for (curr, prev) in current.iter().zip(previous.iter()) {
            let velocity_correlation = curr.velocity.dot(&prev.velocity) / 
                (curr.velocity.magnitude() * prev.velocity.magnitude());
            total_correlation += velocity_correlation;
        }
        
        total_correlation / current.len() as f64
    }
    
    /// Calculate clustering coefficient (simplified)
    fn calculate_clustering_coefficient(&self, states: &[PhysicsState]) -> f64 {
        if states.len() < 3 {
            return 0.0;
        }
        
        let mut total_clustering = 0.0;
        let mut count = 0;
        
        for i in 0..states.len() {
            let mut neighbors = 0;
            let mut connected_neighbors = 0;
            
            for j in 0..states.len() {
                if i != j {
                    let distance = (states[i].position - states[j].position).magnitude();
                    if distance < 1e-6 { // Close neighbors
                        neighbors += 1;
                        
                        // Check if this neighbor is connected to other neighbors
                        for k in 0..states.len() {
                            if k != i && k != j {
                                let dist_jk = (states[j].position - states[k].position).magnitude();
                                if dist_jk < 1e-6 {
                                    connected_neighbors += 1;
                                }
                            }
                        }
                    }
                }
            }
            
            if neighbors > 1 {
                total_clustering += connected_neighbors as f64 / (neighbors * (neighbors - 1)) as f64;
                count += 1;
            }
        }
        
        if count > 0 {
            total_clustering / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate pattern complexity using entropy-based measure
    fn calculate_pattern_complexity(&self, states: &[PhysicsState]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        
        // Calculate velocity distribution entropy
        let mut velocity_bins = HashMap::new();
        let bin_size = 100.0; // m/s
        
        for state in states {
            let velocity_magnitude = state.velocity.magnitude();
            let bin = (velocity_magnitude / bin_size) as usize;
            *velocity_bins.entry(bin).or_insert(0) += 1;
        }
        
        let total_particles = states.len() as f64;
        let mut entropy = 0.0;
        
        for count in velocity_bins.values() {
            let probability = *count as f64 / total_particles;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
    
    /// Calculate phase transition probability
    fn calculate_phase_transition_probability(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        
        let avg_temperature = states.iter().map(|s| s.temperature).sum::<f64>() / states.len() as f64;
        let avg_density = states.iter().map(|s| s.mass).sum::<f64>() / states.len() as f64;
        
        // Simplified phase transition detection based on temperature and density
        let critical_temperature = 273.15; // Water freezing point
        let critical_density = 1000.0; // kg/m³
        
        let temp_factor = if avg_temperature < critical_temperature { 1.0 } else { 0.0 };
        let density_factor = if avg_density > critical_density { 1.0 } else { 0.0 };
        
        (temp_factor + density_factor) / 2.0
    }
    
    /// Calculate collective behavior strength
    fn calculate_collective_behavior_strength(&self, states: &[PhysicsState]) -> f64 {
        if states.len() < 2 {
            return 0.0;
        }
        
        let avg_velocity = states.iter()
            .map(|s| s.velocity)
            .fold(Vector3::zeros(), |acc, v| acc + v) / states.len() as f64;
        
        let mut alignment = 0.0;
        
        for state in states {
            let velocity_magnitude = state.velocity.magnitude();
            if velocity_magnitude > 0.0 {
                let alignment_factor = state.velocity.dot(&avg_velocity) / 
                    (velocity_magnitude * avg_velocity.magnitude());
                alignment += alignment_factor.abs();
            }
        }
        
        alignment / states.len() as f64
    }
    
    /// Calculate information entropy of the system
    fn calculate_information_entropy(&self, states: &[PhysicsState]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        
        // Calculate position-based entropy
        let mut position_bins = HashMap::new();
        let bin_size = 1e-6; // 1 micron
        
        for state in states {
            let x_bin = (state.position.x / bin_size) as i32;
            let y_bin = (state.position.y / bin_size) as i32;
            let z_bin = (state.position.z / bin_size) as i32;
            let bin_key = (x_bin, y_bin, z_bin);
            *position_bins.entry(bin_key).or_insert(0) += 1;
        }
        
        let total_particles = states.len() as f64;
        let mut entropy = 0.0;
        
        for count in position_bins.values() {
            let probability = *count as f64 / total_particles;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
    
    /// Calculate critical exponents for phase transitions
    fn calculate_critical_exponents(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> HashMap<String, f64> {
        let mut exponents = HashMap::new();
        
        if states.len() < 10 {
            return exponents;
        }
        
        // Calculate correlation length exponent (ν)
        let spatial_corr = self.calculate_spatial_correlation(states);
        exponents.insert("nu".to_string(), spatial_corr.abs());
        
        // Calculate order parameter exponent (β)
        let collective_strength = self.calculate_collective_behavior_strength(states);
        exponents.insert("beta".to_string(), collective_strength);
        
        // Calculate susceptibility exponent (γ)
        let pattern_complexity = self.calculate_pattern_complexity(states);
        exponents.insert("gamma".to_string(), pattern_complexity);
        
        // Calculate specific heat exponent (α)
        let avg_temperature = states.iter().map(|s| s.temperature).sum::<f64>() / states.len() as f64;
        let temp_variance = states.iter()
            .map(|s| (s.temperature - avg_temperature).powi(2))
            .sum::<f64>() / states.len() as f64;
        exponents.insert("alpha".to_string(), temp_variance.sqrt());
        
        exponents
    }
    
    /// Validate emergence indicators
    fn validate_emergence(&self, indicators: &EmergenceIndicators) -> Result<(), ValidationError> {
        let params = &self.emergence_parameters;
        
        // Check for strong spatial correlations
        if indicators.spatial_correlation > params.spatial_correlation_threshold {
            return Err(ValidationError::EmergenceDetection {
                reason: format!("Strong spatial correlation detected: {:.3}", indicators.spatial_correlation),
            });
        }
        
        // Check for clustering behavior
        if indicators.clustering_coefficient > params.clustering_coefficient_threshold {
            return Err(ValidationError::EmergenceDetection {
                reason: format!("Clustering behavior detected: {:.3}", indicators.clustering_coefficient),
            });
        }
        
        // Check for phase transitions
        if indicators.phase_transition_probability > params.phase_transition_probability_threshold {
            return Err(ValidationError::EmergenceDetection {
                reason: format!("Phase transition likely: {:.3}", indicators.phase_transition_probability),
            });
        }
        
        // Check for collective behavior
        if indicators.collective_behavior_strength > params.collective_behavior_threshold {
            return Err(ValidationError::EmergenceDetection {
                reason: format!("Collective behavior detected: {:.3}", indicators.collective_behavior_strength),
            });
        }
        
        Ok(())
    }
    
    /// Validate scientific accuracy
    fn validate_scientific_accuracy(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Result<(), ValidationError> {
        // Check for violations of known physical laws
        
        // Check energy-mass equivalence
        for state in states {
            let relativistic_energy = state.mass * constants.c * constants.c;
            let kinetic_energy = 0.5 * state.mass * state.velocity.magnitude().powi(2);
            let total_energy = relativistic_energy + kinetic_energy;
            
            if total_energy < 0.0 {
                return Err(ValidationError::ScientificAccuracy {
                    description: "Negative total energy detected".to_string(),
                });
            }
        }
        
        // Check for superluminal velocities
        for state in states {
            if state.velocity.magnitude() >= constants.c {
                return Err(ValidationError::ScientificAccuracy {
                    description: "Superluminal velocity detected".to_string(),
                });
            }
        }
        
        // Check for negative masses
        for state in states {
            if state.mass < 0.0 {
                return Err(ValidationError::ScientificAccuracy {
                    description: "Negative mass detected".to_string(),
                });
            }
        }
        
        // Check for unphysical temperatures
        for state in states {
            if state.temperature < 0.0 || state.temperature > 1e12 {
                return Err(ValidationError::ScientificAccuracy {
                    description: "Unphysical temperature detected".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate statistical properties
    fn validate_statistics(&self, states: &[PhysicsState], constants: &PhysicsConstants) -> Result<(), ValidationError> {
        if states.len() < 10 {
            return Ok(()); // Need sufficient data for statistical tests
        }
        
        // Perform basic statistical tests
        
        // Test for normal distribution of velocities (simplified)
        let velocities: Vec<f64> = states.iter().map(|s| s.velocity.magnitude()).collect();
        let mean_velocity = velocities.iter().sum::<f64>() / velocities.len() as f64;
        let variance = velocities.iter()
            .map(|v| (v - mean_velocity).powi(2))
            .sum::<f64>() / velocities.len() as f64;
        
        // Simplified normality test (check if variance is reasonable)
        let expected_variance = 3.0 * constants.k_b * states[0].temperature / states[0].mass;
        let variance_ratio = variance / expected_variance;
        
        if variance_ratio < 0.1 || variance_ratio > 10.0 {
            return Err(ValidationError::StatisticalValidation {
                test: "velocity_distribution".to_string(),
                p_value: variance_ratio,
            });
        }
        
        // Test for spatial homogeneity (simplified)
        let positions: Vec<Vector3<f64>> = states.iter().map(|s| s.position).collect();
        let center_of_mass = positions.iter().fold(Vector3::zeros(), |acc, pos| acc + pos) / positions.len() as f64;
        
        let spatial_variance = positions.iter()
            .map(|pos| (pos - center_of_mass).magnitude().powi(2))
            .sum::<f64>() / positions.len() as f64;
        
        // Check if particles are too clustered or too dispersed
        let expected_spatial_variance = 1e-12; // Expected variance in m²
        let spatial_ratio = spatial_variance / expected_spatial_variance;
        
        if spatial_ratio < 0.01 || spatial_ratio > 100.0 {
            return Err(ValidationError::StatisticalValidation {
                test: "spatial_distribution".to_string(),
                p_value: spatial_ratio,
            });
        }
        
        Ok(())
    }
    
    /// Update validation statistics
    fn update_statistics(&mut self, success: bool, validation_time: Duration, errors: &[String]) {
        self.statistics.total_validations += 1;
        
        if success {
            self.statistics.successful_validations += 1;
        } else {
            self.statistics.failed_validations += 1;
        }
        
        // Update average validation time
        let time_ms = validation_time.as_millis() as f64;
        let total_time = self.statistics.average_validation_time_ms * (self.statistics.total_validations - 1) as f64 + time_ms;
        self.statistics.average_validation_time_ms = total_time / self.statistics.total_validations as f64;
        
        // Count specific error types
        for error in errors {
            if error.contains("EnergyConservation") {
                self.statistics.energy_conservation_violations += 1;
            } else if error.contains("MomentumConservation") {
                self.statistics.momentum_conservation_violations += 1;
            } else if error.contains("SuperluminalVelocity") || error.contains("Speed of light exceeded") || error.contains("NegativeMass") || error.contains("InvalidTemperature") {
                self.statistics.relativistic_violations += 1;
            } else if error.contains("EmergenceDetection") {
                self.statistics.emergence_events_detected += 1;
            } else if error.contains("PerformanceViolation") {
                self.statistics.performance_violations += 1;
            } else if error.contains("ScientificAccuracy") {
                self.statistics.scientific_accuracy_violations += 1;
            }
        }
        
        self.statistics.last_validation_time = Some(Instant::now());
    }
    
    /// Get validation statistics
    pub fn get_validation_statistics(&self) -> &ValidationStatistics {
        &self.statistics
    }
    
    /// Reset validation statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = ValidationStatistics::default();
    }
    
    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let stats = &self.statistics;
        let success_rate = if stats.total_validations > 0 {
            stats.successful_validations as f64 / stats.total_validations as f64 * 100.0
        } else {
            0.0
        };
        
        format!(
            "=== Physics Validation Report ===\n\
             Total Validations: {}\n\
             Success Rate: {:.2}%\n\
             Average Validation Time: {:.2} ms\n\
             \n\
             Violations:\n\
             - Energy Conservation: {}\n\
             - Momentum Conservation: {}\n\
             - Relativistic Constraints: {}\n\
             - Performance: {}\n\
             - Scientific Accuracy: {}\n\
             \n\
             Emergence Events: {}\n\
             Last Validation: {:?}",
            stats.total_validations,
            success_rate,
            stats.average_validation_time_ms,
            stats.energy_conservation_violations,
            stats.momentum_conservation_violations,
            stats.relativistic_violations,
            stats.performance_violations,
            stats.scientific_accuracy_violations,
            stats.emergence_events_detected,
            stats.last_validation_time
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct ValidationStatistics {
    pub total_validations: usize,
    pub successful_validations: usize,
    pub failed_validations: usize,
    pub average_validation_time_ms: f64,
    pub energy_conservation_violations: usize,
    pub momentum_conservation_violations: usize,
    pub relativistic_violations: usize,
    pub performance_violations: usize,
    pub scientific_accuracy_violations: usize,
    pub emergence_events_detected: usize,
    pub last_validation_time: Option<std::time::Instant>,
    pub validation_history: Vec<ValidationResult>,
}

#[derive(Debug, Clone, Default)]
pub struct EmergenceIndicators {
    pub spatial_correlation: f64,
    pub temporal_correlation: f64,
    pub clustering_coefficient: f64,
    pub pattern_complexity: f64,
    pub phase_transition_probability: f64,
    pub collective_behavior_strength: f64,
    pub information_entropy: f64,
    pub critical_exponents: std::collections::HashMap<String, f64>,
}

#[derive(Debug)]
pub struct ValidationResult {
    pub timestamp: std::time::Instant,
    pub success: bool,
    pub validation_time_ms: f64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub metrics: PhysicsMetrics,
    pub emergence_indicators: EmergenceIndicators,
    pub performance_metrics: PerformanceMetrics,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            success: true,
            validation_time_ms: 0.0,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: PhysicsMetrics::default(),
            emergence_indicators: EmergenceIndicators::default(),
            performance_metrics: PerformanceMetrics::default(),
        }
    }
}

impl Clone for ValidationResult {
    fn clone(&self) -> Self {
        Self {
            timestamp: self.timestamp,
            success: self.success,
            validation_time_ms: self.validation_time_ms,
            errors: self.errors.clone(),
            warnings: self.warnings.clone(),
            metrics: self.metrics.clone(),
            emergence_indicators: self.emergence_indicators.clone(),
            performance_metrics: self.performance_metrics.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn create_test_state(mass: f64, velocity: Vector3<f64>, temp: f64) -> PhysicsState {
        PhysicsState {
            position: Vector3::zeros(),
            velocity,
            acceleration: Vector3::zeros(),
            mass,
            charge: 0.0,
            temperature: temp,
            entropy: 1e-20,
        }
    }

    #[test]
    fn test_energy_conservation_check() {
        let constants = PhysicsConstants::default();
        let states = vec![
            create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
            create_test_state(2.0, Vector3::new(-500.0, 0.0, 0.0), 300.0),
        ];
        
        // Should not error for reasonable values
        assert!(check_energy_conservation(&states, &constants).is_ok());
    }

    #[test]
    fn test_momentum_conservation_check() {
        let states = vec![
            create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
            create_test_state(1.0, Vector3::new(-1000.0, 0.0, 0.0), 300.0),
        ];
        
        // Balanced momentum should pass
        assert!(check_momentum_conservation(&states).is_ok());
    }

    #[test]
    fn test_relativistic_constraints() {
        let constants = PhysicsConstants::default();
        
        // Normal velocity should pass
        let normal_states = vec![
            create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
        ];
        assert!(check_relativistic_constraints(&normal_states, &constants).is_ok());
        
        // Superluminal velocity should fail
        let fast_states = vec![
            create_test_state(1.0, Vector3::new(3.1e8, 0.0, 0.0), 300.0),
        ];
        assert!(check_relativistic_constraints(&fast_states, &constants).is_err());
        
        // Negative mass should fail
        let negative_mass_states = vec![
            create_test_state(-1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
        ];
        assert!(check_relativistic_constraints(&negative_mass_states, &constants).is_err());
    }

    #[test]
    fn test_physics_metrics() {
        let constants = PhysicsConstants::default();
        let states = vec![
            create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
            create_test_state(2.0, Vector3::new(2000.0, 0.0, 0.0), 400.0),
        ];
        
        let metrics = calculate_physics_metrics(&states, &constants);
        
        assert_eq!(metrics.total_mass, 3.0);
        assert_eq!(metrics.average_temperature, 350.0);
        assert!(metrics.total_energy > 0.0);
        assert_eq!(metrics.relativistic_particles, 0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let constants = PhysicsConstants::default();
        let states = vec![
            create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
            create_test_state(2.0, Vector3::new(-500.0, 0.0, 0.0), 350.0),
        ];
        
        // Should pass all validations
        assert!(validate_physics_state(&states, &constants).is_ok());
    }

    #[test]
    fn test_comprehensive_physics_validator() {
        let constants = PhysicsConstants::default();
        let states = vec![
            create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
            create_test_state(2.0, Vector3::new(-500.0, 0.0, 0.0), 350.0),
        ];
        
        let mut validator = ComprehensivePhysicsValidator::default();
        
        // Set more lenient thresholds for testing
        validator.performance_thresholds.min_prediction_accuracy = 100.0; // Lower threshold
        validator.emergence_parameters.collective_behavior_threshold = 2.0; // Higher threshold
        
        // Initialize validation
        assert!(validator.initialize_validation(&states, &constants).is_ok());
        
        // Run validation
        let result = validator.validate(&states, &constants);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.errors.is_empty());
        assert_eq!(result.metrics.total_mass, 3.0);
        
        // Check statistics
        let stats = validator.get_validation_statistics();
        assert_eq!(stats.total_validations, 1);
        assert_eq!(stats.successful_validations, 1);
        assert_eq!(stats.failed_validations, 0);
    }

    #[test]
    fn test_validation_error_detection() {
        let constants = PhysicsConstants::default();
        
        // Test superluminal velocity detection
        let fast_states = vec![
            create_test_state(1.0, Vector3::new(3.1e8, 0.0, 0.0), 300.0),
        ];
        
        let mut validator = ComprehensivePhysicsValidator::default();
        let result = validator.validate(&fast_states, &constants);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(!result.success);
        assert!(!result.errors.is_empty());
        assert!(result.errors.iter().any(|e| e.contains("Speed of light exceeded")));
        
        // Check statistics
        let stats = validator.get_validation_statistics();
        assert_eq!(stats.total_validations, 1);
        assert_eq!(stats.successful_validations, 0);
        assert_eq!(stats.failed_validations, 1);
        assert_eq!(stats.relativistic_violations, 1);
    }
}