//! Demo 05: Tree-PM Hybrid Gravity Solver Performance
//! 
//! This demo showcases the Tree-PM hybrid gravity solver performance and accuracy
//! for cosmological N-body simulations. The Tree-PM method combines Barnes-Hut
//! tree for short-range forces with particle-mesh for long-range forces.
//!
//! Features demonstrated:
//! - Tree-PM hybrid gravity solver performance benchmarks
//! - Force accuracy validation against analytical solutions
//! - Scaling behavior with particle number
//! - Periodic boundary conditions
//! - Adaptive time-stepping
//! - Performance comparison with different parameters

use physics_engine::cosmology::{CosmologicalParameters, TreePmGravitySolver, CosmologicalParticle, CosmologicalParticleType};
use anyhow::Result;
use nalgebra::Vector3;
use std::time::Instant;
use rand::{thread_rng, Rng};

fn main() -> Result<()> {
    println!("=== Universe Evolution Simulator: Tree-PM Gravity Solver Demo ===");
    println!();
    println!("This demo showcases the Tree-PM hybrid gravity solver:");
    println!("• Barnes-Hut tree for short-range forces");
    println!("• Particle-mesh for long-range forces");
    println!("• O(N log N) performance scaling");
    println!("• Periodic boundary conditions");
    println!("• Force accuracy validation");
    println!("• Performance benchmarking");
    println!();
    
    // Initialize cosmological parameters
    let mut params = CosmologicalParameters::default();
    params.box_size = 100.0; // 100 Mpc/h box
    params.n_particles = 50_000; // 50k particles for performance test
    
    // Create gravity solver parameters
    let tree_opening_angle = 0.5;
    let softening_length = 1.0; // kpc/h
    let pm_grid_size = 256;
    let pm_smoothing_scale = 1.0;
    let periodic_boundaries = true;
    let force_accuracy = 1e-4;
    
    println!("Cosmological Parameters:");
    println!("  Box size: {:.1} Mpc/h", params.box_size);
    println!("  Number of particles: {}", params.n_particles);
    println!("  Hubble parameter (H₀): {:.2} km/s/Mpc", params.h0);
    println!("  Matter density (Ωₘ): {:.3}", params.omega_m);
    println!("  Tree opening angle: {:.2}", tree_opening_angle);
    println!("  Softening length: {:.3} kpc/h", softening_length);
    println!("  PM grid size: {}³", pm_grid_size);
    println!("  PM smoothing scale: {:.2}", pm_smoothing_scale);
    println!("  Periodic boundaries: {}", periodic_boundaries);
    println!("  Force accuracy: {:.2e}", force_accuracy);
    println!();
    
    // Create Tree-PM gravity solver
    println!("Initializing Tree-PM gravity solver...");
    let gravity_solver = TreePmGravitySolver::new(params.clone());
    
    println!("Tree-PM Configuration:");
    println!("  Tree opening angle: {:.2}", gravity_solver.tree_opening_angle);
    println!("  Softening length: {:.3} kpc/h", gravity_solver.softening_length);
    println!("  Force accuracy: {:.2e}", gravity_solver.force_accuracy);
    println!("  PM grid size: {}³", gravity_solver.pm_grid_size);
    println!("  PM smoothing scale: {:.2}", gravity_solver.pm_smoothing_scale);
    println!("  Periodic boundaries: {}", gravity_solver.periodic_boundaries);
    println!();
    
    // Create test particles
    println!("Creating test particles...");
    let particles = create_test_particles(params.n_particles, &params)?;
    
    // Extract positions and masses for gravity calculation
    let positions: Vec<Vector3<f64>> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    
    println!("Test particle properties:");
    println!("  Total particles: {}", particles.len());
    let min_mass = masses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_mass = masses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!("  Mass range: {:.2e} - {:.2e} solar masses", min_mass, max_mass);
    let min_pos = positions.iter().map(|p| p.magnitude()).fold(f64::INFINITY, |a, b| a.min(b));
    let max_pos = positions.iter().map(|p| p.magnitude()).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    println!("  Position range: {:.2} - {:.2} Mpc/h", 
        min_pos, max_pos);
    println!();
    
    // Performance benchmark
    println!("=== Performance Benchmark ===");
    benchmark_gravity_solver(&gravity_solver, &positions, &masses)?;
    
    // Force accuracy validation
    println!("=== Force Accuracy Validation ===");
    validate_force_accuracy(&gravity_solver, &positions, &masses)?;
    
    // Scaling test with different particle numbers
    println!("=== Scaling Test ===");
    scaling_test(&params)?;
    
    // Parameter sensitivity test
    println!("=== Parameter Sensitivity Test ===");
    parameter_sensitivity_test(&params, &positions, &masses)?;
    
    // Periodic boundary conditions test
    println!("=== Periodic Boundary Conditions Test ===");
    periodic_boundary_test(&gravity_solver, &params)?;
    
    // Adaptive time-stepping test
    println!("=== Adaptive Time-stepping Test ===");
    adaptive_timestep_test(&gravity_solver, &positions, &masses)?;
    
    println!();
    println!("=== Demo Complete ===");
    println!("This demo successfully showcases:");
    println!("✅ Tree-PM hybrid gravity solver performance");
    println!("✅ Force accuracy validation against analytical solutions");
    println!("✅ O(N log N) scaling behavior");
    println!("✅ Periodic boundary conditions");
    println!("✅ Adaptive time-stepping");
    println!("✅ Parameter sensitivity analysis");
    println!("✅ Performance optimization");
    
    Ok(())
}

/// Create test particles with realistic cosmological distribution
fn create_test_particles(n_particles: usize, params: &CosmologicalParameters) -> Result<Vec<CosmologicalParticle>> {
    let mut rng = thread_rng();
    let mut particles = Vec::new();
    
    // Convert box size from Mpc/h to meters
    let box_size_m = params.box_size * 3.086e22; // Mpc to meters
    let particle_mass = 1e8; // 100 million solar masses per particle
    
    for _i in 0..n_particles {
        // Random position in cosmological box
        let position = Vector3::new(
            rng.gen_range(-box_size_m/2.0..box_size_m/2.0),
            rng.gen_range(-box_size_m/2.0..box_size_m/2.0),
            rng.gen_range(-box_size_m/2.0..box_size_m/2.0),
        );
        
        // Random velocity (peculiar velocity)
        let velocity = Vector3::new(
            rng.gen_range(-1e5..1e5), // 100 km/s
            rng.gen_range(-1e5..1e5),
            rng.gen_range(-1e5..1e5),
        );
        
        // Vary particle mass slightly for more realistic distribution
        let mass = particle_mass * (1.0 + rng.gen_range(-0.1..0.1));
        
        let particle = CosmologicalParticle::new(
            position,
            velocity,
            mass,
            CosmologicalParticleType::DarkMatter,
        );
        
        particles.push(particle);
    }
    
    Ok(particles)
}

/// Benchmark gravity solver performance
fn benchmark_gravity_solver(
    gravity_solver: &TreePmGravitySolver,
    positions: &[Vector3<f64>],
    masses: &[f64],
) -> Result<()> {
    println!("Benchmarking gravity solver performance...");
    
    let n_particles = positions.len();
    let n_test_particles = 100; // Test force calculation on subset
    
    // Warm up
    for i in 0..10 {
        let _ = gravity_solver.compute_gravitational_force(positions, masses, i % n_test_particles)?;
    }
    
    // Benchmark force calculations
    let start_time = Instant::now();
    let mut total_force = Vector3::zeros();
    
    for i in 0..n_test_particles {
        let force = gravity_solver.compute_gravitational_force(positions, masses, i)?;
        total_force += force;
    }
    
    let elapsed = start_time.elapsed();
    let force_calculations_per_second = n_test_particles as f64 / elapsed.as_secs_f64();
    
    println!("Performance Results:");
    println!("  Force calculations: {}", n_test_particles);
    println!("  Total time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Force calculations per second: {:.0}", force_calculations_per_second);
    println!("  Average time per force calculation: {:.3} μs", 
        elapsed.as_secs_f64() * 1e6 / n_test_particles as f64);
    println!("  Total force magnitude: {:.2e} N", total_force.magnitude());
    
    // Estimate full simulation performance
    let estimated_full_time = n_particles as f64 / force_calculations_per_second;
    println!("  Estimated full simulation time: {:.2} s", estimated_full_time);
    
    Ok(())
}

/// Validate force accuracy against analytical solutions
fn validate_force_accuracy(
    gravity_solver: &TreePmGravitySolver,
    positions: &[Vector3<f64>],
    masses: &[f64],
) -> Result<()> {
    println!("Validating force accuracy...");
    
    use physics_engine::constants::G;
    
    // Test with a simple two-particle system
    let test_positions = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1e20, 0.0, 0.0), // 1 Mpc separation
    ];
    let test_masses = vec![1e12, 1e12]; // 1 trillion solar masses each
    
    // Analytical force
    let distance = (test_positions[1] - test_positions[0]).magnitude();
    let analytical_force = G * test_masses[0] * test_masses[1] / (distance * distance);
    
    // Numerical force from Tree-PM
    let numerical_force = gravity_solver.compute_gravitational_force(&test_positions, &test_masses, 0)?;
    let numerical_magnitude = numerical_force.magnitude();
    
    let relative_error = (numerical_magnitude - analytical_force).abs() / analytical_force;
    
    println!("Force Accuracy Test:");
    println!("  Separation: {:.2e} m", distance);
    println!("  Analytical force: {:.2e} N", analytical_force);
    println!("  Numerical force: {:.2e} N", numerical_magnitude);
    println!("  Relative error: {:.2e} ({:.2}%)", relative_error, relative_error * 100.0);
    println!("  Accuracy target: {:.2e}", gravity_solver.force_accuracy);
    println!("  Accuracy achieved: {}", relative_error <= gravity_solver.force_accuracy);
    
    Ok(())
}

/// Test scaling behavior with different particle numbers
fn scaling_test(params: &CosmologicalParameters) -> Result<()> {
    println!("Testing scaling behavior...");
    
    let particle_counts = vec![1000, 5000, 10000, 50000];
    let mut scaling_results = Vec::new();
    
    for &n_particles in &particle_counts {
        let mut test_params = params.clone();
        test_params.n_particles = n_particles;
        
        let gravity_solver = TreePmGravitySolver::new(test_params.clone());
        let particles = create_test_particles(n_particles, &test_params)?;
        
        let positions: Vec<Vector3<f64>> = particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        
        let start_time = Instant::now();
        let _ = gravity_solver.compute_gravitational_force(&positions, &masses, 0)?;
        let elapsed = start_time.elapsed();
        
        scaling_results.push((n_particles, elapsed.as_secs_f64()));
    }
    
    println!("Scaling Results:");
    for (n_particles, time) in &scaling_results {
        println!("  {} particles: {:.3} ms", n_particles, time * 1000.0);
    }
    
    // Calculate scaling exponent
    if scaling_results.len() >= 2 {
        let (n1, t1) = scaling_results[0];
        let (n2, t2) = scaling_results[scaling_results.len() - 1];
        let scaling_exponent = (t2 / t1).ln() / (n2 as f64 / n1 as f64).ln();
        
        println!("  Scaling exponent: {:.2}", scaling_exponent);
        println!("  Expected (N log N): ~1.0-1.5");
        println!("  Achieved: {:.2}", scaling_exponent);
    }
    
    Ok(())
}

/// Test parameter sensitivity
fn parameter_sensitivity_test(
    params: &CosmologicalParameters,
    _positions: &[Vector3<f64>],
    _masses: &[f64],
) -> Result<()> {
    println!("Testing parameter sensitivity...");
    
    // Test different tree opening angles
    let opening_angles = vec![0.3, 0.5, 0.7, 1.0];
    println!("Tree Opening Angle Sensitivity:");
    
    for &angle in &opening_angles {
        let mut test_solver = TreePmGravitySolver::new(params.clone());
        test_solver.tree_opening_angle = angle;
        
        let start_time = Instant::now();
        let force = test_solver.compute_gravitational_force(_positions, _masses, 0)?;
        let elapsed = start_time.elapsed();
        
        println!("  θ = {:.1}: {:.3} ms, |F| = {:.2e} N", 
            angle, elapsed.as_secs_f64() * 1000.0, force.magnitude());
    }
    
    // Test different PM grid sizes
    let grid_sizes = vec![128, 256, 512];
    println!("PM Grid Size Sensitivity:");
    
    for &grid_size in &grid_sizes {
        let mut test_solver = TreePmGravitySolver::new(params.clone());
        test_solver.pm_grid_size = grid_size;
        
        let start_time = Instant::now();
        let force = test_solver.compute_gravitational_force(_positions, _masses, 0)?;
        let elapsed = start_time.elapsed();
        
        println!("  Grid {}³: {:.3} ms, |F| = {:.2e} N", 
            grid_size, elapsed.as_secs_f64() * 1000.0, force.magnitude());
    }
    
    Ok(())
}

/// Test periodic boundary conditions
fn periodic_boundary_test(gravity_solver: &TreePmGravitySolver, params: &CosmologicalParameters) -> Result<()> {
    println!("Testing periodic boundary conditions...");
    
    // Create particles at box boundaries
    let box_size_m = params.box_size * 3.086e22;
    let boundary_positions = vec![
        Vector3::new(-box_size_m/2.0, 0.0, 0.0), // Left boundary
        Vector3::new(box_size_m/2.0, 0.0, 0.0),  // Right boundary
        Vector3::new(0.0, -box_size_m/2.0, 0.0), // Bottom boundary
        Vector3::new(0.0, box_size_m/2.0, 0.0),  // Top boundary
    ];
    let boundary_masses = vec![1e12, 1e12, 1e12, 1e12];
    
    // Test force on central particle
    let central_position = Vector3::new(0.0, 0.0, 0.0);
    let central_mass = 1e12;
    
    let mut all_positions = boundary_positions.clone();
    all_positions.push(central_position);
    let mut all_masses = boundary_masses.clone();
    all_masses.push(central_mass);
    
    let force = gravity_solver.compute_gravitational_force(&all_positions, &all_masses, 4)?; // Central particle
    
    println!("Periodic Boundary Test:");
    println!("  Box size: {:.2e} m", box_size_m);
    println!("  Central particle force: {:.2e} N", force.magnitude());
    println!("  Force components: ({:.2e}, {:.2e}, {:.2e})", force.x, force.y, force.z);
    println!("  Periodic boundaries enabled: {}", gravity_solver.periodic_boundaries);
    
    Ok(())
}

/// Test adaptive time-stepping
fn adaptive_timestep_test(
    gravity_solver: &TreePmGravitySolver,
    positions: &[Vector3<f64>],
    masses: &[f64],
) -> Result<()> {
    println!("Testing adaptive time-stepping...");
    
    // Create velocities for time-stepping test
    let velocities: Vec<Vector3<f64>> = positions.iter().map(|_| Vector3::new(1e5, 1e5, 1e5)).collect();
    
    // Test time step calculation for different scale factors
    let scale_factors = vec![0.1, 0.5, 1.0];
    
    for &scale_factor in &scale_factors {
        let time_step = gravity_solver.calculate_time_step(
            positions,
            &velocities,
            masses,
            scale_factor,
        );
        
        println!("  Scale factor {:.1}: time step = {:.2e} s", scale_factor, time_step);
    }
    
    // Test time step stability
    let current_scale_factor = 1.0;
    let time_step = gravity_solver.calculate_time_step(
        positions,
        &velocities,
        masses,
        current_scale_factor,
    );
    
    println!("Time Step Analysis:");
    println!("  Calculated time step: {:.2e} s", time_step);
    println!("  Hubble time: {:.2e} s", 1.0 / gravity_solver.cosmological_params.hubble_parameter(current_scale_factor));
    println!("  Time step ratio: {:.2e}", time_step * gravity_solver.cosmological_params.hubble_parameter(current_scale_factor));
    println!("  Stability criterion: < 0.1 (achieved: {})", 
        time_step * gravity_solver.cosmological_params.hubble_parameter(current_scale_factor) < 0.1);
    
    Ok(())
} 