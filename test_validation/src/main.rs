use nalgebra::Vector3;
use physics_engine::{PhysicsState, PhysicsConstants, validation::*};

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

fn main() {
    println!("Testing Physics Validation Framework");
    
    let constants = PhysicsConstants::default();
    
    // Test basic validation
    let states = vec![
        create_test_state(1.0, Vector3::new(1000.0, 0.0, 0.0), 300.0),
        create_test_state(2.0, Vector3::new(-500.0, 0.0, 0.0), 350.0),
    ];
    
    println!("Testing energy conservation...");
    match check_energy_conservation(&states, &constants) {
        Ok(total_energy) => println!("✓ Energy conservation check passed, total energy: {}", total_energy),
        Err(e) => println!("✗ Energy conservation check failed: {}", e),
    }
    
    println!("Testing momentum conservation...");
    match check_momentum_conservation(&states) {
        Ok(total_momentum) => println!("✓ Momentum conservation check passed, total momentum: {:?}", total_momentum),
        Err(e) => println!("✗ Momentum conservation check failed: {}", e),
    }
    
    println!("Testing relativistic constraints...");
    match check_relativistic_constraints(&states, &constants) {
        Ok(()) => println!("✓ Relativistic constraints check passed"),
        Err(e) => println!("✗ Relativistic constraints check failed: {}", e),
    }
    
    println!("Testing comprehensive validation...");
    match validate_physics_state(&states, &constants) {
        Ok(()) => println!("✓ Comprehensive validation passed"),
        Err(e) => println!("✗ Comprehensive validation failed: {}", e),
    }
    
    // Test comprehensive validator
    println!("\nTesting ComprehensivePhysicsValidator...");
    let mut validator = ComprehensivePhysicsValidator::default();
    
    match validator.initialize_validation(&states, &constants) {
        Ok(()) => println!("✓ Validator initialization passed"),
        Err(e) => println!("✗ Validator initialization failed: {}", e),
    }
    
    match validator.validate(&states, &constants) {
        Ok(result) => {
            println!("✓ Validator validation passed");
            println!("  Success: {}", result.success);
            println!("  Errors: {}", result.errors.len());
            println!("  Total mass: {}", result.metrics.total_mass);
            println!("  Total energy: {}", result.metrics.total_energy);
        },
        Err(e) => println!("✗ Validator validation failed: {}", e),
    }
    
    // Test error detection
    println!("\nTesting error detection...");
    let fast_states = vec![
        create_test_state(1.0, Vector3::new(3e8, 0.0, 0.0), 300.0), // Superluminal
    ];
    
    let mut validator = ComprehensivePhysicsValidator::default();
    match validator.validate(&fast_states, &constants) {
        Ok(result) => {
            println!("✓ Error detection test completed");
            println!("  Success: {}", result.success);
            println!("  Errors: {}", result.errors.len());
            if !result.errors.is_empty() {
                println!("  Error types: {:?}", result.errors);
            }
        },
        Err(e) => println!("✗ Error detection test failed: {}", e),
    }
    
    println!("\nValidation framework test completed!");
} 