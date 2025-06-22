use physics_engine::validation::{
    ComprehensivePhysicsValidator, ValidationStatistics,
    PerformanceThresholds, ValidationError, check_energy_conservation, check_momentum_conservation,
    check_relativistic_constraints, validate_physics_state, calculate_physics_metrics,
};
use physics_engine::{PhysicsState, PhysicsConstants};
use nalgebra::Vector3;
use std::collections::HashMap;

/// Test comprehensive physics validation system
#[test]
fn test_comprehensive_validation_system() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Create test states with various properties
    let states = vec![
        PhysicsState {
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(-500.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 2.0,
            charge: 0.0,
            temperature: 350.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(0.0, 1e-6, 0.0),
            velocity: Vector3::new(0.0, 1000.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 0.5,
            charge: 0.0,
            temperature: 250.0,
            entropy: 1e-20,
        },
    ];
    
    // Initialize validation
    assert!(validator.initialize_validation(&states, &constants).is_ok());
    
    // Perform comprehensive validation
    let result = validator.validate(&states, &constants).unwrap();
    
    // Verify results - these should pass with valid physics states
    assert!(result.success);
    assert!(result.errors.is_empty());
    assert!(result.validation_time_ms > 0.0);
    
    // Check metrics
    assert_eq!(result.metrics.total_mass, 3.5);
    assert!(result.metrics.total_energy > 0.0);
    assert_eq!(result.metrics.average_temperature, 300.0);
    assert!(result.metrics.max_velocity > 0.0);
    
    // Check emergence indicators
    assert!(result.emergence_indicators.spatial_correlation >= -1.0);
    assert!(result.emergence_indicators.spatial_correlation <= 1.0);
    assert!(result.emergence_indicators.pattern_complexity >= 0.0);
    assert!(result.emergence_indicators.information_entropy >= 0.0);
    
    // Check performance metrics
    assert!(result.performance_metrics.memory_usage_mb > 0.0);
    assert!(result.performance_metrics.particles_per_second > 0.0);
    assert!(result.performance_metrics.cache_efficiency > 0.0);
    assert!(result.performance_metrics.parallelization_efficiency > 0.0);
}

/// Test conservation law validation
#[test]
fn test_conservation_law_validation() {
    let constants = PhysicsConstants::default();
    
    // Test energy conservation with balanced system
    let balanced_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(-1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    assert!(check_energy_conservation(&balanced_states, &constants).is_ok());
    assert!(check_momentum_conservation(&balanced_states).is_ok());
    
    // Test with charged particles
    let charged_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 1e-19, // Elementary charge
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(-1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: -1e-19, // Negative elementary charge
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    assert!(check_energy_conservation(&charged_states, &constants).is_ok());
}

/// Test relativistic validation
#[test]
fn test_relativistic_validation() {
    let constants = PhysicsConstants::default();
    
    // Test normal velocities
    let normal_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1e6, 0.0, 0.0), // 1000 km/s
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    assert!(check_relativistic_constraints(&normal_states, &constants).is_ok());
    
    // Test superluminal velocity (should fail)
    let superluminal_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(3e8, 0.0, 0.0), // Speed of light
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    assert!(check_relativistic_constraints(&superluminal_states, &constants).is_err());
    
    // Test negative mass (should fail)
    let negative_mass_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: -1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    assert!(check_relativistic_constraints(&negative_mass_states, &constants).is_err());
}

/// Test emergence validation
#[test]
fn test_emergence_validation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Create states with strong collective behavior
    let mut collective_states = Vec::new();
    for i in 0..10 {
        collective_states.push(PhysicsState {
            position: Vector3::new(i as f64 * 1e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0), // All moving in same direction
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        });
    }
    
    let result = validator.validate(&collective_states, &constants).unwrap();
    
    // Should have higher collective behavior due to aligned velocities
    assert!(result.emergence_indicators.collective_behavior_strength > 0.1);
    
    // Create states with random behavior
    let mut random_states = Vec::new();
    for i in 0..10 {
        random_states.push(PhysicsState {
            position: Vector3::new(i as f64 * 1e-6, 0.0, 0.0),
            velocity: Vector3::new(
                if i % 2 == 0 { 1000.0 } else { -1000.0 },
                if i % 3 == 0 { 500.0 } else { -500.0 },
                0.0
            ),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        });
    }
    
    let result = validator.validate(&random_states, &constants).unwrap();
    
    // Should have lower collective behavior
    assert!(result.emergence_indicators.collective_behavior_strength < 0.8);
}

/// Test performance validation
#[test]
fn test_performance_validation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Create many states to test performance
    let mut many_states = Vec::new();
    for i in 0..1000 {
        many_states.push(PhysicsState {
            position: Vector3::new(i as f64 * 1e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        });
    }
    
    let result = validator.validate(&many_states, &constants).unwrap();
    
    // Should pass performance validation with reasonable states
    assert!(result.success);
    assert!(result.performance_metrics.memory_usage_mb > 0.0);
    assert!(result.performance_metrics.particles_per_second > 0.0);
    assert!(result.performance_metrics.computation_time_ms > 0.0);
    
    // Test with custom performance thresholds that are too strict
    let mut strict_validator = ComprehensivePhysicsValidator::default();
    strict_validator.performance_thresholds.max_computational_overhead = 0.001; // Very strict
    strict_validator.performance_thresholds.min_prediction_accuracy = 1e12; // Very high
    
    let result = strict_validator.validate(&many_states, &constants).unwrap();
    
    // Should fail due to strict thresholds
    assert!(!result.success);
    assert!(!result.errors.is_empty());
}

/// Test validation statistics
#[test]
fn test_validation_statistics() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    let states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(-500.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 2.0,
            charge: 0.0,
            temperature: 350.0,
            entropy: 1e-20,
        },
    ];
    
    validator.initialize_validation(&states, &constants).unwrap();
    
    // Run multiple validations to build statistics
    for _ in 0..10 {
        validator.validate(&states, &constants).unwrap();
    }
    
    let stats = validator.get_validation_statistics();
    assert_eq!(stats.total_validations, 10);
    assert_eq!(stats.successful_validations, 10);
    assert_eq!(stats.failed_validations, 0);
    assert!(stats.average_validation_time_ms > 0.0);
    assert!(stats.last_validation_time.is_some());
    
    // Test with some failures
    let invalid_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(3e8, 0.0, 0.0), // Superluminal
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    validator.initialize_validation(&invalid_states, &constants).unwrap();
    validator.validate(&invalid_states, &constants).unwrap();
    
    let stats = validator.get_validation_statistics();
    assert_eq!(stats.total_validations, 11);
    assert_eq!(stats.successful_validations, 10);
    assert_eq!(stats.failed_validations, 1);
    assert_eq!(stats.relativistic_violations, 1);
}

/// Test validation error handling
#[test]
fn test_validation_error_handling() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Test with various invalid states
    let invalid_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(3e8, 0.0, 0.0), // Superluminal
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: -1.0, // Negative mass
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(2e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: -100.0, // Negative temperature
            entropy: 1e-20,
        },
    ];
    
    let result = validator.validate(&invalid_states, &constants).unwrap();
    
    assert!(!result.success);
    assert!(result.errors.len() >= 3);
    
    // Check specific error types
    let error_types: Vec<String> = result.errors.iter()
        .map(|e| e.to_string())
        .collect();
    
    assert!(error_types.iter().any(|e| e.contains("SuperluminalVelocity")));
    assert!(error_types.iter().any(|e| e.contains("NegativeMass")));
    assert!(error_types.iter().any(|e| e.contains("InvalidTemperature")));
}

/// Test scientific accuracy validation
#[test]
fn test_scientific_accuracy_validation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Test with physically impossible values
    let impossible_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 1e15, // Unphysically high temperature
            entropy: 1e-20,
        },
    ];
    
    let result = validator.validate(&impossible_states, &constants).unwrap();
    
    // Should fail scientific accuracy validation
    assert!(!result.success);
    assert!(!result.errors.is_empty());
    
    // Test with reasonable values
    let reasonable_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0, // Room temperature
            entropy: 1e-20,
        },
    ];
    
    let result = validator.validate(&reasonable_states, &constants).unwrap();
    
    // Should pass scientific accuracy validation
    assert!(result.success);
    assert!(result.errors.is_empty());
}

/// Test comprehensive validation edge cases
#[test]
fn test_comprehensive_validation_edge_cases() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Test with empty state list
    let empty_states: Vec<PhysicsState> = Vec::new();
    let result = validator.validate(&empty_states, &constants).unwrap();
    
    // Should handle empty states gracefully
    assert!(result.success);
    assert_eq!(result.metrics.total_mass, 0.0);
    assert_eq!(result.metrics.total_energy, 0.0);
    
    // Test with single particle
    let single_state = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    let result = validator.validate(&single_state, &constants).unwrap();
    assert!(result.success);
    assert_eq!(result.metrics.total_mass, 1.0);
    
    // Test with extreme values (but still physically possible)
    let extreme_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(0.99 * constants.c, 0.0, 0.0), // Very high but sub-luminal
            acceleration: Vector3::zeros(),
            mass: 1e-30, // Very small mass
            charge: 0.0,
            temperature: 1e-6, // Very low temperature
            entropy: 1e-20,
        },
    ];
    
    let result = validator.validate(&extreme_states, &constants).unwrap();
    assert!(result.success);
}

/// Test emergence parameters sensitivity
#[test]
fn test_emergence_parameters_sensitivity() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Create states with moderate collective behavior
    let states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(800.0, 0.0, 0.0), // Similar but not identical
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    // Test with default parameters
    let result = validator.validate(&states, &constants).unwrap();
    let default_collective = result.emergence_indicators.collective_behavior_strength;
    
    // Test with more sensitive parameters
    validator.emergence_parameters.collective_behavior_threshold = 0.1;
    let result = validator.validate(&states, &constants).unwrap();
    let sensitive_collective = result.emergence_indicators.collective_behavior_strength;
    
    // Values should be consistent
    assert!((default_collective - sensitive_collective).abs() < 0.1);
}

/// Test validation report generation
#[test]
fn test_validation_report_generation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Run some validations to generate statistics
    let states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    for _ in 0..5 {
        validator.validate(&states, &constants).unwrap();
    }
    
    let report = validator.generate_report();
    
    // Check that report contains expected sections
    assert!(report.contains("Validation Statistics"));
    assert!(report.contains("Total Validations"));
    assert!(report.contains("Success Rate"));
    assert!(report.contains("Performance Metrics"));
    
    // Check success rate calculation
    let stats = validator.get_validation_statistics();
    let expected_success_rate = (stats.successful_validations as f64 / stats.total_validations as f64) * 100.0;
    assert!(report.contains(&format!("{:.2}%", expected_success_rate)));
}

/// Test critical exponents calculation
#[test]
fn test_critical_exponents_calculation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Create states for critical exponent calculation
    let mut critical_states = Vec::new();
    for i in 0..20 {
        critical_states.push(PhysicsState {
            position: Vector3::new(i as f64 * 1e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0 + (i as f64 * 10.0), 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0 + (i as f64 * 5.0),
            entropy: 1e-20,
        });
    }
    
    let indicators = validator.detect_emergence(&critical_states, &constants).unwrap();
    
    // Check that critical exponents are calculated
    assert!(!indicators.critical_exponents.is_empty());
    assert!(indicators.critical_exponents.contains_key("nu"));
    assert!(indicators.critical_exponents.contains_key("beta"));
    assert!(indicators.critical_exponents.contains_key("gamma"));
    assert!(indicators.critical_exponents.contains_key("alpha"));
    
    // Check that exponents are reasonable values
    for (_, value) in &indicators.critical_exponents {
        assert!(*value >= 0.0);
        assert!(*value < 1e6); // Should not be unreasonably large
    }
}

/// Test information entropy calculation
#[test]
fn test_information_entropy_calculation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Test with uniform distribution (high entropy)
    let mut uniform_states = Vec::new();
    for i in 0..10 {
        uniform_states.push(PhysicsState {
            position: Vector3::new(i as f64 * 1e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        });
    }
    
    let indicators = validator.detect_emergence(&uniform_states, &constants).unwrap();
    let uniform_entropy = indicators.information_entropy;
    
    // Test with clustered distribution (lower entropy)
    let mut clustered_states = Vec::new();
    for i in 0..10 {
        clustered_states.push(PhysicsState {
            position: Vector3::new(0.0, 0.0, 0.0), // All at same position
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        });
    }
    
    let indicators = validator.detect_emergence(&clustered_states, &constants).unwrap();
    let clustered_entropy = indicators.information_entropy;
    
    // Uniform distribution should have higher entropy than clustered
    assert!(uniform_entropy > clustered_entropy);
    assert!(uniform_entropy > 0.0);
    assert!(clustered_entropy >= 0.0);
}

/// Test pattern complexity calculation
#[test]
fn test_pattern_complexity_calculation() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Test with simple pattern (low complexity)
    let simple_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
        PhysicsState {
            position: Vector3::new(1e-6, 0.0, 0.0),
            velocity: Vector3::new(1000.0, 0.0, 0.0), // Same velocity
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0,
            entropy: 1e-20,
        },
    ];
    
    let indicators = validator.detect_emergence(&simple_states, &constants).unwrap();
    let simple_complexity = indicators.pattern_complexity;
    
    // Test with complex pattern (high complexity)
    let mut complex_states = Vec::new();
    for i in 0..10 {
        complex_states.push(PhysicsState {
            position: Vector3::new(i as f64 * 1e-6, 0.0, 0.0),
            velocity: Vector3::new(
                (i * 100) as f64,
                (i * 200) as f64,
                (i * 300) as f64,
            ),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 300.0 + (i as f64 * 10.0),
            entropy: 1e-20,
        });
    }
    
    let indicators = validator.detect_emergence(&complex_states, &constants).unwrap();
    let complex_complexity = indicators.pattern_complexity;
    
    // Complex pattern should have higher complexity
    assert!(complex_complexity > simple_complexity);
    assert!(simple_complexity >= 0.0);
    assert!(complex_complexity >= 0.0);
}

/// Test phase transition detection
#[test]
fn test_phase_transition_detection() {
    let mut validator = ComprehensivePhysicsValidator::default();
    let constants = PhysicsConstants::default();
    
    // Test with low temperature (potential freezing)
    let low_temp_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 250.0, // Below freezing
            entropy: 1e-20,
        },
    ];
    
    let indicators = validator.detect_emergence(&low_temp_states, &constants).unwrap();
    let low_temp_probability = indicators.phase_transition_probability;
    
    // Test with high temperature (no phase transition)
    let high_temp_states = vec![
        PhysicsState {
            position: Vector3::zeros(),
            velocity: Vector3::new(1000.0, 0.0, 0.0),
            acceleration: Vector3::zeros(),
            mass: 1.0,
            charge: 0.0,
            temperature: 400.0, // Above freezing
            entropy: 1e-20,
        },
    ];
    
    let indicators = validator.detect_emergence(&high_temp_states, &constants).unwrap();
    let high_temp_probability = indicators.phase_transition_probability;
    
    // Low temperature should have higher phase transition probability
    assert!(low_temp_probability > high_temp_probability);
    assert!(low_temp_probability >= 0.0);
    assert!(low_temp_probability <= 1.0);
    assert!(high_temp_probability >= 0.0);
    assert!(high_temp_probability <= 1.0);
} 