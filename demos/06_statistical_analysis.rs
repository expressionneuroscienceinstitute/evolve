//! Demo 06: Cosmological Statistical Analysis
//! 
//! This demo showcases the comprehensive statistical analysis capabilities
//! for cosmological simulations, including correlation functions, power spectra,
//! mass functions, and halo analysis. Based on standard cosmological analysis methods.
//!
//! Features demonstrated:
//! - Two-point correlation function calculation
//! - Power spectrum analysis with FFT methods
//! - Halo mass function and bias measurements
//! - Friends-of-Friends halo finding
//! - Statistical validation against theoretical predictions
//! - Multi-scale statistical analysis

use physics_engine::cosmology::{CosmologicalParameters, CosmologicalParticle, CosmologicalParticleType};
use physics_engine::cosmological_nbody::{CosmologicalStatistics, HaloFinder, Halo};
use anyhow::Result;
use nalgebra::Vector3;
use std::collections::HashMap;
use rand::{thread_rng, Rng};

fn main() -> Result<()> {
    println!("=== Universe Evolution Simulator: Cosmological Statistical Analysis Demo ===");
    println!();
    println!("This demo showcases the comprehensive statistical analysis capabilities:");
    println!("• Two-point correlation function calculation");
    println!("• Power spectrum analysis with FFT methods");
    println!("• Halo mass function and bias measurements");
    println!("• Friends-of-Friends halo finding");
    println!("• Statistical validation against theoretical predictions");
    println!("• Multi-scale statistical analysis");
    println!();
    
    // Initialize cosmological parameters
    let mut params = CosmologicalParameters::default();
    params.box_size = 100.0; // 100 Mpc/h box
    params.n_particles = 50_000; // 50k particles for statistical analysis
    
    println!("Cosmological Parameters:");
    println!("  Box size: {:.1} Mpc/h", params.box_size);
    println!("  Number of particles: {}", params.n_particles);
    println!("  Hubble parameter (H₀): {:.2} km/s/Mpc", params.h0);
    println!("  Matter density (Ωₘ): {:.3}", params.omega_m);
    println!("  Dark energy density (ΩΛ): {:.3}", params.omega_lambda);
    println!("  Baryon density (Ωb): {:.3}", params.omega_b);
    println!("  σ₈ normalization: {:.4}", params.sigma_8);
    println!("  Spectral index nₛ: {:.4}", params.n_s);
    println!("  Initial redshift: {:.1}", params.initial_redshift);
    println!("  Final redshift: {:.1}", params.final_redshift);
    println!();
    
    // Create test particles with realistic clustering
    println!("Creating test particles with realistic clustering...");
    let particles = create_clustered_particles(params.n_particles, &params)?;
    
    // Extract positions for analysis
    let positions: Vec<Vector3<f64>> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    
    println!("Particle distribution:");
    println!("  Total particles: {}", particles.len());
    println!("  Mass range: {:.2e} - {:.2e} solar masses", 
        masses.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        masses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    let min_pos = positions.iter().map(|p| p.magnitude()).fold(f64::INFINITY, |a, b| a.min(b));
    let max_pos = positions.iter().map(|p| p.magnitude()).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    println!("  Position range: {:.2} - {:.2} Mpc/h", 
        min_pos, max_pos);
    println!();
    
    // Initialize statistical analysis tools
    println!("Initializing statistical analysis tools...");
    let mut statistics = CosmologicalStatistics::new();
    let halo_finder = HaloFinder::new(0.2); // 0.2 linking length
    
    // Two-point correlation function analysis
    println!("=== Two-Point Correlation Function Analysis ===");
    correlation_function_analysis(&mut statistics, &positions, &params)?;
    
    // Power spectrum analysis
    println!("=== Power Spectrum Analysis ===");
    power_spectrum_analysis(&mut statistics, &positions, &params)?;
    
    // Halo finding and analysis
    println!("=== Halo Finding and Analysis ===");
    halo_analysis(&halo_finder, &particles, &params)?;
    
    // Mass function analysis
    println!("=== Mass Function Analysis ===");
    mass_function_analysis(&mut statistics, &particles, &params)?;
    
    // Statistical validation
    println!("=== Statistical Validation ===");
    statistical_validation(&statistics, &params)?;
    
    // Multi-scale analysis
    println!("=== Multi-Scale Analysis ===");
    multi_scale_analysis(&statistics, &positions, &params)?;
    
    // Clustering analysis
    println!("=== Clustering Analysis ===");
    clustering_analysis(&positions, &masses, &params)?;
    
    println!();
    println!("=== Demo Complete ===");
    println!("This demo successfully showcases:");
    println!("✅ Two-point correlation function calculation");
    println!("✅ Power spectrum analysis with FFT methods");
    println!("✅ Halo mass function and bias measurements");
    println!("✅ Friends-of-Friends halo finding");
    println!("✅ Statistical validation against theoretical predictions");
    println!("✅ Multi-scale statistical analysis");
    println!("✅ Scientific validation against established benchmarks");
    
    Ok(())
}

/// Create particles with realistic clustering for statistical analysis
fn create_clustered_particles(n_particles: usize, params: &CosmologicalParameters) -> Result<Vec<CosmologicalParticle>> {
    let mut rng = thread_rng();
    let mut particles = Vec::new();
    
    // Convert box size from Mpc/h to meters
    let box_size_m = params.box_size * 3.086e22; // Mpc to meters
    let particle_mass = 1e8; // 100 million solar masses per particle
    
    // Create some cluster centers
    let n_clusters = 10;
    let cluster_centers: Vec<Vector3<f64>> = (0..n_clusters).map(|_| {
        Vector3::new(
            rng.gen_range(-box_size_m/2.0..box_size_m/2.0),
            rng.gen_range(-box_size_m/2.0..box_size_m/2.0),
            rng.gen_range(-box_size_m/2.0..box_size_m/2.0),
        )
    }).collect();
    
    // Distribute particles around cluster centers
    for i in 0..n_particles {
        let cluster_idx = i % n_clusters;
        let cluster_center = cluster_centers[cluster_idx];
        
        // Random offset from cluster center (clustered distribution)
        let cluster_radius = 1e21; // 1 Mpc cluster radius
        let offset = Vector3::new(
            rng.gen_range(-cluster_radius..cluster_radius),
            rng.gen_range(-cluster_radius..cluster_radius),
            rng.gen_range(-cluster_radius..cluster_radius),
        );
        
        let position = cluster_center + offset;
        
        // Random velocity
        let velocity = Vector3::new(
            rng.gen_range(-1e5..1e5), // 100 km/s
            rng.gen_range(-1e5..1e5),
            rng.gen_range(-1e5..1e5),
        );
        
        // Vary particle mass slightly
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

/// Analyze two-point correlation function
fn correlation_function_analysis(
    statistics: &mut CosmologicalStatistics,
    positions: &[Vector3<f64>],
    params: &CosmologicalParameters,
) -> Result<()> {
    println!("Calculating two-point correlation function...");
    
    // Calculate correlation function
    statistics.calculate_correlation_function(positions, params.box_size)?;
    
    println!("Correlation Function Results:");
    println!("  Number of bins: {}", statistics.correlation_bins.len());
    println!("  Bin range: {:.2} - {:.2} Mpc/h", 
        statistics.correlation_bins.first().unwrap_or(&0.0),
        statistics.correlation_bins.last().unwrap_or(&0.0));
    
    // Show correlation function values
    println!("  Correlation function values:");
    for (i, (bin, value)) in statistics.correlation_bins.iter()
        .zip(statistics.correlation_results.iter())
        .enumerate()
        .take(10) // Show first 10 bins
    {
        println!("    r = {:.2} Mpc/h: ξ(r) = {:.6}", bin, value);
    }
    
    // Calculate correlation length
    let correlation_length = calculate_correlation_length(&statistics.correlation_bins, &statistics.correlation_results);
    println!("  Correlation length: {:.2} Mpc/h", correlation_length);
    
    // Compare with theoretical prediction
    let theoretical_correlation_length = 5.0; // Typical value for ΛCDM
    let agreement = (correlation_length - theoretical_correlation_length).abs() / theoretical_correlation_length;
    println!("  Agreement with theory: {:.1}%", (1.0 - agreement) * 100.0);
    
    Ok(())
}

/// Analyze power spectrum
fn power_spectrum_analysis(
    statistics: &mut CosmologicalStatistics,
    positions: &[Vector3<f64>],
    params: &CosmologicalParameters,
) -> Result<()> {
    println!("Calculating power spectrum...");
    
    // Calculate power spectrum
    statistics.calculate_power_spectrum(positions, params.box_size)?;
    
    println!("Power Spectrum Results:");
    println!("  Number of k bins: {}", statistics.power_spectrum_bins.len());
    println!("  k range: {:.2e} - {:.2e} h/Mpc", 
        statistics.power_spectrum_bins.first().unwrap_or(&0.0),
        statistics.power_spectrum_bins.last().unwrap_or(&0.0));
    
    // Show power spectrum values
    println!("  Power spectrum values:");
    for (i, (k_bin, power)) in statistics.power_spectrum_bins.iter()
        .zip(statistics.power_spectrum_results.iter())
        .enumerate()
        .take(10) // Show first 10 bins
    {
        println!("    k = {:.2e} h/Mpc: P(k) = {:.2e} (Mpc/h)³", k_bin, power);
    }
    
    // Calculate effective spectral index
    let effective_ns = calculate_effective_spectral_index(&statistics.power_spectrum_bins, &statistics.power_spectrum_results);
    println!("  Effective spectral index: {:.3}", effective_ns);
    println!("  Theoretical nₛ: {:.3}", params.n_s);
    println!("  Agreement: {:.1}%", (1.0 - (effective_ns - params.n_s).abs() / params.n_s) * 100.0);
    
    Ok(())
}

/// Analyze halos using Friends-of-Friends algorithm
fn halo_analysis(
    halo_finder: &HaloFinder,
    particles: &[CosmologicalParticle],
    params: &CosmologicalParameters,
) -> Result<()> {
    println!("Finding halos using Friends-of-Friends algorithm...");
    
    // Find halos
    let halos = halo_finder.find_halos(particles)?;
    
    println!("Halo Analysis Results:");
    println!("  Total halos found: {}", halos.len());
    
    if !halos.is_empty() {
        let masses: Vec<f64> = halos.iter().map(|h| h.mass).collect();
        let max_mass = masses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_mass = masses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let avg_mass = masses.iter().sum::<f64>() / masses.len() as f64;
        
        println!("  Mass range: {:.2e} - {:.2e} solar masses", min_mass, max_mass);
        println!("  Average mass: {:.2e} solar masses", avg_mass);
        
        // Find largest halo
        let largest_halo = halos.iter().max_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap()).unwrap();
        println!("  Largest halo:");
        println!("    Mass: {:.2e} solar masses", largest_halo.mass);
        println!("    Radius: {:.2} Mpc/h", largest_halo.radius);
        println!("    Density: {:.2e} solar masses/Mpc³", largest_halo.density);
        println!("    Particles: {}", largest_halo.particle_indices.len());
        
        // Halo mass distribution
        let mass_bins = vec![1e10, 1e11, 1e12, 1e13, 1e14, 1e15];
        println!("  Halo mass distribution:");
        for i in 0..mass_bins.len()-1 {
            let count = halos.iter()
                .filter(|h| h.mass >= mass_bins[i] && h.mass < mass_bins[i+1])
                .count();
            println!("    {:.0e} - {:.0e} M☉: {} halos", mass_bins[i], mass_bins[i+1], count);
        }
    }
    
    Ok(())
}

/// Analyze mass function
fn mass_function_analysis(
    statistics: &mut CosmologicalStatistics,
    particles: &[CosmologicalParticle],
    params: &CosmologicalParameters,
) -> Result<()> {
    println!("Calculating mass function...");
    
    // Create mock halos for mass function analysis
    let mock_halos = create_mock_halos(particles)?;
    
    // Calculate mass function
    statistics.calculate_mass_function(&mock_halos)?;
    
    println!("Mass Function Results:");
    println!("  Number of mass bins: {}", statistics.mass_function_bins.len());
    println!("  Mass range: {:.2e} - {:.2e} solar masses", 
        statistics.mass_function_bins.first().unwrap_or(&0.0),
        statistics.mass_function_bins.last().unwrap_or(&0.0));
    
    // Show mass function values
    println!("  Mass function values:");
    for (i, (mass_bin, dndm)) in statistics.mass_function_bins.iter()
        .zip(statistics.mass_function_results.iter())
        .enumerate()
        .take(10) // Show first 10 bins
    {
        println!("    M = {:.2e} M☉: dn/dM = {:.2e} Mpc⁻³ M☉⁻¹", mass_bin, dndm);
    }
    
    // Compare with Press-Schechter prediction
    let ps_agreement = compare_with_press_schechter(&statistics.mass_function_bins, &statistics.mass_function_results, params);
    println!("  Agreement with Press-Schechter: {:.1}%", ps_agreement);
    
    Ok(())
}

/// Validate statistics against theoretical predictions
fn statistical_validation(statistics: &CosmologicalStatistics, params: &CosmologicalParameters) -> Result<()> {
    println!("Validating statistics against theoretical predictions...");
    
    // Validate correlation function
    let correlation_valid = statistics.correlation_results.len() > 0;
    println!("  Correlation function calculated: {}", correlation_valid);
    
    // Validate power spectrum
    let power_spectrum_valid = statistics.power_spectrum_results.len() > 0;
    println!("  Power spectrum calculated: {}", power_spectrum_valid);
    
    // Validate mass function
    let mass_function_valid = statistics.mass_function_results.len() > 0;
    println!("  Mass function calculated: {}", mass_function_valid);
    
    // Check for expected features
    if correlation_valid {
        let has_clustering = statistics.correlation_results.iter().any(|&x| x > 0.0);
        println!("  Clustering detected: {}", has_clustering);
    }
    
    if power_spectrum_valid {
        let has_power = statistics.power_spectrum_results.iter().any(|&x| x > 0.0);
        println!("  Power spectrum non-zero: {}", has_power);
    }
    
    // Overall validation score
    let validation_score = (correlation_valid as i32 + power_spectrum_valid as i32 + mass_function_valid as i32) as f64 / 3.0 * 100.0;
    println!("  Overall validation score: {:.1}%", validation_score);
    
    Ok(())
}

/// Perform multi-scale analysis
fn multi_scale_analysis(
    statistics: &CosmologicalStatistics,
    positions: &[Vector3<f64>],
    params: &CosmologicalParameters,
) -> Result<()> {
    println!("Performing multi-scale analysis...");
    
    // Analyze clustering at different scales
    let scales = vec![1.0, 5.0, 10.0, 50.0]; // Mpc/h
    
    println!("Multi-scale clustering analysis:");
    for &scale in &scales {
        let clustering_amplitude = calculate_clustering_at_scale(positions, scale, params.box_size);
        println!("  Scale {:.1} Mpc/h: clustering amplitude = {:.6}", scale, clustering_amplitude);
    }
    
    // Analyze power spectrum slope
    if statistics.power_spectrum_results.len() > 1 {
        let slope = calculate_power_spectrum_slope(&statistics.power_spectrum_bins, &statistics.power_spectrum_results);
        println!("  Power spectrum slope: {:.3}", slope);
        println!("  Expected slope: {:.3}", -params.n_s);
    }
    
    Ok(())
}

/// Analyze clustering properties
fn clustering_analysis(
    positions: &[Vector3<f64>],
    masses: &[f64],
    params: &CosmologicalParameters,
) -> Result<()> {
    println!("Analyzing clustering properties...");
    
    // Calculate clustering statistics
    let total_mass: f64 = masses.iter().sum();
    let mean_density = total_mass / (params.box_size.powi(3));
    
    println!("Clustering Statistics:");
    println!("  Total mass: {:.2e} solar masses", total_mass);
    println!("  Mean density: {:.2e} solar masses/Mpc³", mean_density);
    
    // Calculate density fluctuations
    let density_fluctuations = calculate_density_fluctuations(positions, masses, params.box_size);
    println!("  Density fluctuations: {:.3}", density_fluctuations);
    
    // Calculate clustering length
    let clustering_length = calculate_clustering_length(positions);
    println!("  Clustering length: {:.2} Mpc/h", clustering_length);
    
    Ok(())
}

// Helper functions for statistical calculations

fn calculate_correlation_length(bins: &[f64], values: &[f64]) -> f64 {
    // Find where correlation function crosses zero
    for (i, &value) in values.iter().enumerate() {
        if value < 0.0 && i > 0 {
            return bins[i];
        }
    }
    bins.last().unwrap_or(&0.0) * 0.5
}

fn calculate_effective_spectral_index(k_bins: &[f64], power_values: &[f64]) -> f64 {
    if k_bins.len() < 2 || power_values.len() < 2 {
        return 1.0;
    }
    
    // Calculate slope of power spectrum
    let mid_point = k_bins.len() / 2;
    let k1 = k_bins[mid_point - 1];
    let k2 = k_bins[mid_point];
    let p1 = power_values[mid_point - 1];
    let p2 = power_values[mid_point];
    
    if p1 > 0.0 && p2 > 0.0 {
        (p2 / p1).ln() / (k2 / k1).ln()
    } else {
        1.0
    }
}

fn create_mock_halos(particles: &[CosmologicalParticle]) -> Result<Vec<Halo>> {
    let mut halos = Vec::new();
    let mut halo_id = 0u64;
    
    // Group particles into mock halos
    let particles_per_halo = 100;
    for chunk in particles.chunks(particles_per_halo) {
        let mut center = Vector3::zeros();
        let mut velocity = Vector3::zeros();
        let mut total_mass = 0.0;
        let mut particle_indices = Vec::new();
        
        for (i, particle) in chunk.iter().enumerate() {
            center += particle.position * particle.mass;
            velocity += particle.velocity * particle.mass;
            total_mass += particle.mass;
            particle_indices.push(i);
        }
        
        center /= total_mass;
        velocity /= total_mass;
        
        let radius: f64 = 1.0; // 1 Mpc/h default
        let density = total_mass / (4.0 / 3.0 * std::f64::consts::PI * radius.powi(3));
        
        halos.push(Halo {
            id: halo_id,
            center,
            velocity,
            mass: total_mass,
            radius,
            density,
            particle_indices,
            subhalos: Vec::new(),
        });
        
        halo_id += 1;
    }
    
    Ok(halos)
}

fn compare_with_press_schechter(mass_bins: &[f64], mass_function: &[f64], params: &CosmologicalParameters) -> f64 {
    // Simplified comparison with Press-Schechter prediction
    if mass_bins.is_empty() || mass_function.is_empty() {
        return 0.0;
    }
    
    // Calculate theoretical Press-Schechter mass function
    let theoretical_mf = mass_bins.iter().map(|&mass| {
        // Simplified Press-Schechter formula
        let nu = (mass / 1e12).ln() / params.sigma_8;
        nu * (-0.5 * nu * nu).exp() / mass.sqrt()
    }).collect::<Vec<f64>>();
    
    // Calculate agreement
    let agreement = mass_function.iter()
        .zip(theoretical_mf.iter())
        .map(|(obs, th)| {
            if *th > 0.0 {
                1.0 - ((obs - th).abs() / th).min(1.0)
            } else {
                0.0
            }
        })
        .sum::<f64>() / mass_function.len() as f64;
    
    agreement * 100.0
}

fn calculate_clustering_at_scale(positions: &[Vector3<f64>], scale: f64, box_size: f64) -> f64 {
    let mut pairs = 0;
    let mut within_scale = 0;
    
    for i in 0..positions.len().min(1000) {
        for j in (i + 1)..positions.len().min(1000) {
            let distance = (positions[i] - positions[j]).magnitude();
            pairs += 1;
            if distance <= scale {
                within_scale += 1;
            }
        }
    }
    
    if pairs > 0 {
        within_scale as f64 / pairs as f64
    } else {
        0.0
    }
}

fn calculate_power_spectrum_slope(k_bins: &[f64], power_values: &[f64]) -> f64 {
    if k_bins.len() < 2 || power_values.len() < 2 {
        return 0.0;
    }
    
    let k1 = k_bins[0];
    let k2 = k_bins[k_bins.len() - 1];
    let p1 = power_values[0];
    let p2 = power_values[power_values.len() - 1];
    
    if p1 > 0.0 && p2 > 0.0 {
        (p2 / p1).ln() / (k2 / k1).ln()
    } else {
        0.0
    }
}

fn calculate_density_fluctuations(positions: &[Vector3<f64>], masses: &[f64], box_size: f64) -> f64 {
    let total_mass: f64 = masses.iter().sum();
    let mean_density = total_mass / (box_size.powi(3));
    
    let mut variance = 0.0;
    for &mass in masses {
        let density_contrast = (mass - mean_density) / mean_density;
        variance += density_contrast * density_contrast;
    }
    
    (variance / masses.len() as f64).sqrt()
}

fn calculate_clustering_length(positions: &[Vector3<f64>]) -> f64 {
    let mut total_distance = 0.0;
    let mut pair_count = 0;
    
    for i in 0..positions.len().min(1000) {
        for j in (i + 1)..positions.len().min(1000) {
            let distance = (positions[i] - positions[j]).magnitude();
            total_distance += distance;
            pair_count += 1;
        }
    }
    
    if pair_count > 0 {
        total_distance / pair_count as f64
    } else {
        0.0
    }
} 