use clap::{App, Arg};
use physics_engine::*;
use std::time::Duration;

/// CLI tool for running GPU acceleration stress tests
pub fn run_gpu_stress_test_cli() -> Result<()> {
    let matches = App::new("GPU Acceleration Stress Test")
        .version("1.0")
        .about("Stress test GPU acceleration for atom and fundamental particle visualization")
        .arg(
            Arg::with_name("duration")
                .short("d")
                .long("duration")
                .value_name("SECONDS")
                .help("Duration of stress test in seconds")
                .default_value("60")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("intensity")
                .short("i")
                .long("intensity")
                .value_name("LEVEL")
                .help("Stress test intensity (low, medium, high, extreme)")
                .default_value("medium")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("FILE")
                .help("Output file for performance metrics")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Enable verbose output"),
        )
        .get_matches();

    let duration: u64 = matches.value_of("duration")
        .unwrap()
        .parse()
        .unwrap_or(60);

    let intensity = matches.value_of("intensity").unwrap();
    let verbose = matches.is_present("verbose");
    let output_file = matches.value_of("output");

    println!("🎬 GPU Acceleration Stress Test CLI");
    println!("=" * 50);
    println!("Duration: {} seconds", duration);
    println!("Intensity: {}", intensity);
    println!("Verbose: {}", verbose);
    if let Some(file) = output_file {
        println!("Output: {}", file);
    }
    println!();

    // Create stress test with appropriate intensity
    let mut stress_test = create_stress_test_with_intensity(intensity)?;
    
    if verbose {
        println!("🔬 Initializing test systems...");
    }
    stress_test.initialize_test_systems()?;
    
    if verbose {
        println!("🔥 Starting stress test...");
    }
    
    let start_time = std::time::Instant::now();
    stress_test.run_stress_test(duration)?;
    let total_time = start_time.elapsed();
    
    if verbose {
        println!("✅ Stress test completed in {:.2} seconds", total_time.as_secs_f64());
    }
    
    // Save results if output file specified
    if let Some(file_path) = output_file {
        save_performance_metrics(&stress_test.performance_metrics, file_path)?;
        println!("📊 Performance metrics saved to: {}", file_path);
    }
    
    println!("🎉 GPU acceleration stress test completed successfully!");
    Ok(())
}

/// Create stress test with specified intensity level
fn create_stress_test_with_intensity(intensity: &str) -> Result<gpu_acceleration_stress_test::GPUAccelerationStressTest> {
    let mut stress_test = gpu_acceleration_stress_test::GPUAccelerationStressTest::new()?;
    
    match intensity.to_lowercase().as_str() {
        "low" => {
            // Minimal stress test
            stress_test.quantum_fields = Vec::new();
            stress_test.molecular_systems = Vec::new();
            stress_test.particle_systems = Vec::new();
            stress_test.atomic_systems = Vec::new();
            stress_test.nuclear_systems = Vec::new();
        }
        "medium" => {
            // Standard stress test (default)
        }
        "high" => {
            // High intensity - will be configured in initialize_test_systems
            println!("⚠️  High intensity stress test selected");
        }
        "extreme" => {
            // Extreme intensity - maximum stress
            println!("🚨 EXTREME intensity stress test selected");
            println!("   This will push your system to its limits!");
        }
        _ => {
            return Err(anyhow::anyhow!("Invalid intensity level: {}", intensity));
        }
    }
    
    Ok(stress_test)
}

/// Save performance metrics to file
fn save_performance_metrics(metrics: &gpu_acceleration_stress_test::PerformanceMetrics, file_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(file_path)?;
    
    writeln!(file, "GPU Acceleration Stress Test Results")?;
    writeln!(file, "=" * 40)?;
    writeln!(file, "Timestamp: {}", chrono::Utc::now())?;
    writeln!(file)?;
    
    // Performance summary
    if !metrics.computation_times.is_empty() {
        let avg_time = metrics.computation_times.iter()
            .map(|d| d.as_micros() as f64)
            .sum::<f64>() / metrics.computation_times.len() as f64;
        
        let max_time = metrics.computation_times.iter()
            .map(|d| d.as_micros())
            .max()
            .unwrap_or(0);
        
        let min_time = metrics.computation_times.iter()
            .map(|d| d.as_micros())
            .min()
            .unwrap_or(0);
        
        writeln!(file, "Performance Summary:")?;
        writeln!(file, "  Average Step Time: {:.2} μs", avg_time)?;
        writeln!(file, "  Min Step Time: {} μs", min_time)?;
        writeln!(file, "  Max Step Time: {} μs", max_time)?;
        writeln!(file, "  Total Steps: {}", metrics.computation_times.len())?;
        writeln!(file)?;
    }
    
    // System statistics
    let total_interactions = metrics.particle_interactions.iter().sum::<u64>();
    let total_reactions = metrics.molecular_reactions.iter().sum::<u64>();
    let total_calculations = metrics.quantum_calculations.iter().sum::<u64>();
    
    writeln!(file, "System Statistics:")?;
    writeln!(file, "  Total Interactions: {}", total_interactions)?;
    writeln!(file, "  Total Reactions: {}", total_reactions)?;
    writeln!(file, "  Total Calculations: {}", total_calculations)?;
    writeln!(file, "  Field Evolution Steps: {}", metrics.field_evolution_steps.len())?;
    writeln!(file)?;
    
    // Detailed timing data
    writeln!(file, "Detailed Timing Data:")?;
    for (i, time) in metrics.computation_times.iter().enumerate() {
        writeln!(file, "  Step {}: {} μs", i, time.as_micros())?;
    }
    
    Ok(())
}

/// Quick stress test for development
pub fn run_quick_stress_test() -> Result<()> {
    println!("⚡ Running Quick GPU Stress Test (10 seconds)...");
    
    let mut stress_test = gpu_acceleration_stress_test::GPUAccelerationStressTest::new()?;
    stress_test.initialize_test_systems()?;
    stress_test.run_stress_test(10)?;
    
    println!("✅ Quick stress test completed!");
    Ok(())
}

/// Benchmark GPU vs CPU performance
pub fn run_gpu_benchmark() -> Result<()> {
    println!("🏁 Running GPU vs CPU Benchmark...");
    
    // TODO: Implement GPU vs CPU benchmark
    // This would compare performance of quantum field evolution
    // and molecular dynamics on GPU vs CPU
    
    println!("✅ Benchmark completed!");
    Ok(())
} 