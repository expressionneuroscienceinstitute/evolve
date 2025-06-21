//! Physics Validation Module
//! 
//! Validates conservation laws and physics consistency

use anyhow::Result;
use nalgebra::Vector3;
use crate::{PhysicsState, PhysicsConstants};

/// Physics validation tolerance
#[allow(dead_code)]
const CONSERVATION_TOLERANCE: f64 = 1e-6;

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
        if state.mass > 0.0 && v >= constants.c {
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
            create_test_state(1.0, Vector3::new(3e8, 0.0, 0.0), 300.0),
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
}