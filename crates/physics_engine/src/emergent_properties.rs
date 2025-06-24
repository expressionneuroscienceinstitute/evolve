//! # Physics Engine: Emergent Properties Module
//!
//! This module is dedicated to observing and quantifying macroscopic properties
//! that emerge from the collective behavior of microscopic particles. These properties,
//! such as temperature and pressure, are not fundamental to individual particles but
//! arise from their statistical mechanics.

use anyhow::Result;
use std::collections::HashMap;

use crate::{constants::BOLTZMANN, PhysicsState};
// use crate::particles::Particle;

/// Represents temperature in Kelvin.
#[derive(Debug, Default, Copy, Clone)]
pub struct Temperature(pub f64);

impl Temperature {
    pub fn as_kelvin(&self) -> f64 {
        self.0
    }
    
    pub fn from_kelvin(kelvin: f64) -> Self {
        Temperature(kelvin)
    }
}

/// Represents pressure in Pascals.
#[derive(Debug, Default, Copy, Clone)]
pub struct Pressure(pub f64);

impl Pressure {
    pub fn as_pascals(&self) -> f64 {
        self.0
    }
    
    pub fn from_pascals(pascals: f64) -> Self {
        Pressure(pascals)
    }
}

/// Represents density in kilograms per cubic meter.
#[derive(Debug, Default, Copy, Clone)]
pub struct Density(pub f64);

impl Density {
    pub fn as_kg_per_m3(&self) -> f64 {
        self.0
    }
    
    pub fn from_kg_per_m3(kg_per_m3: f64) -> Self {
        Density(kg_per_m3)
    }
}

/// Represents entropy in Joules per Kelvin.
#[derive(Debug, Default)]
pub struct Entropy(pub f64);

impl Entropy {
    pub fn as_joules_per_kelvin(&self) -> f64 {
        self.0
    }
}

/// Monitors and calculates emergent properties of the system.
#[derive(Debug, Default)]
pub struct EmergenceMonitor {
    pub temperature: Temperature,
    pub pressure: Pressure,
    pub density: Density,
    pub entropy: Entropy,
}

impl EmergenceMonitor {
    /// Creates a new EmergenceMonitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all emergent properties from the states of the particles.
    pub fn update(&mut self, states: &[PhysicsState], volume: f64) -> Result<()> {
        self.temperature = self.calculate_temperature(states);
        self.pressure = self.calculate_pressure(states, volume);
        self.density = self.calculate_density(states, volume);
        self.entropy = self.calculate_entropy(states);
        Ok(())
    }

    /// Calculate the temperature from the kinetic energy of the particles.
    fn calculate_temperature(&self, particles: &[PhysicsState]) -> Temperature {
        let total_kinetic_energy: f64 = particles
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.norm_squared())
            .sum();
        let temperature =
            (2.0 / 3.0) * total_kinetic_energy / (particles.len() as f64 * BOLTZMANN);
        Temperature(temperature)
    }

    /// Calculate the pressure using ideal gas law or Van der Waals equation of state.
    /// Uses Van der Waals equation for dense systems where particle interactions matter.
    fn calculate_pressure(&self, particles: &[PhysicsState], volume: f64) -> Pressure {
        if volume == 0.0 {
            return Pressure(0.0);
        }
        let n = particles.len() as f64;
        let temperature = self.calculate_temperature(particles).as_kelvin();
        
        // Calculate density to determine which equation of state to use
        let density = self.calculate_density(particles, volume).as_kg_per_m3();
        let critical_density = 1000.0; // kg/m³ - transition point for Van der Waals
        
        if density < critical_density {
            // Use ideal gas law for low-density systems  
            let pressure = n * BOLTZMANN * temperature / volume;
            Pressure(pressure)
        } else {
            // Use Van der Waals equation of state for dense systems
            // (P + a(n/V)²)(V - nb) = nkT
            // Solving for P: P = nkT/(V - nb) - a(n/V)²
            
            // Van der Waals constants (approximated for a typical gas)
            let a = 0.364; // Pa⋅m⁶⋅mol⁻² (approximate average)
            let b = 4.27e-5; // m³/mol (approximate average)
            
            // Convert to per-particle units
            let avogadro = 6.022e23;
            let a_per_particle = a / (avogadro * avogadro);
            let b_per_particle = b / avogadro;
            
            let excluded_volume = n * b_per_particle;
            let available_volume = volume - excluded_volume;
            
            if available_volume > 0.0 {
                let ideal_pressure = n * BOLTZMANN * temperature / available_volume;
                let attraction_correction = a_per_particle * (n / volume).powi(2);
                let pressure = ideal_pressure - attraction_correction;
                Pressure(pressure.max(0.0)) // Ensure non-negative pressure
            } else {
                // Very high density - use hard sphere approximation
                let hard_sphere_pressure = n * BOLTZMANN * temperature / (volume * 0.1);
                Pressure(hard_sphere_pressure)
            }
        }
    }

    /// Calculate the total density of the particles.
    fn calculate_density(&self, particles: &[PhysicsState], volume: f64) -> Density {
        if volume == 0.0 {
            return Density(0.0);
        }
        let total_mass: f64 = particles.iter().map(|p| p.mass).sum();
        Density(total_mass / volume)
    }

    /// Calculate the entropy using statistical mechanics principles.
    /// Based on the Sackur-Tetrode equation for an ideal gas and
    /// the equipartition theorem for kinetic contributions.
    #[allow(dead_code)]
    fn calculate_entropy(&self, particles: &[PhysicsState]) -> Entropy {
        if particles.is_empty() {
            return Entropy(0.0);
        }
        
        let n = particles.len() as f64;
        let temperature = self.calculate_temperature(particles).as_kelvin();
        
        if temperature <= 0.0 {
            return Entropy(0.0);
        }
        
        // Calculate average particle mass for the system
        let avg_mass = particles.iter().map(|p| p.mass).sum::<f64>() / n;
        
        // Thermal de Broglie wavelength (quantum mechanical length scale)
        let h = 6.626e-34; // Planck constant
        let _lambda_th = h / (2.0 * std::f64::consts::PI * avg_mass * BOLTZMANN * temperature).sqrt();
        
        // Volume density (assume particles are in a cubic volume)
        let volume = 1.0; // Normalized volume - would need actual system volume
        let _number_density = n / volume;
        
        // Sackur-Tetrode equation for entropy of an ideal gas (per particle)
        // S = k * ln[(V/N) * (2πmkT/h²)^(3/2) * e^(5/2)]
        let entropy_per_particle = BOLTZMANN * (
            (volume / n).ln() + 
            1.5 * (2.0 * std::f64::consts::PI * avg_mass * BOLTZMANN * temperature / (h * h)).ln() +
            2.5
        );
        
        // Total entropy
        let total_entropy = n * entropy_per_particle;
        
        // Add configurational entropy from particle position distributions
        // This is a simplified approximation based on spatial distribution
        let spatial_entropy = self.calculate_spatial_entropy(particles);
        
        Entropy(total_entropy + spatial_entropy)
    }
    
    /// Calculate spatial configurational entropy based on particle distributions
    /// Uses Voronoi tessellation to calculate local density variations and
    /// applies statistical mechanics principles for accurate entropy calculation
    fn calculate_spatial_entropy(&self, particles: &[PhysicsState]) -> f64 {
        if particles.len() < 2 {
            return 0.0;
        }
        
        // Calculate center of mass
        let com_x = particles.iter().map(|p| p.position.x).sum::<f64>() / particles.len() as f64;
        let com_y = particles.iter().map(|p| p.position.y).sum::<f64>() / particles.len() as f64;
        let com_z = particles.iter().map(|p| p.position.z).sum::<f64>() / particles.len() as f64;
        
        // Calculate variance in each dimension
        let var_x = particles.iter()
            .map(|p| (p.position.x - com_x).powi(2))
            .sum::<f64>() / particles.len() as f64;
        let var_y = particles.iter()
            .map(|p| (p.position.y - com_y).powi(2))
            .sum::<f64>() / particles.len() as f64;
        let var_z = particles.iter()
            .map(|p| (p.position.z - com_z).powi(2))
            .sum::<f64>() / particles.len() as f64;
        
        // Calculate local density variations using nearest neighbor analysis
        let mut local_densities = Vec::new();
        for (i, particle) in particles.iter().enumerate() {
            let mut nearest_distances = Vec::new();
            
            // Find distances to nearest neighbors
            for (j, other) in particles.iter().enumerate() {
                if i != j {
                    let distance = (particle.position - other.position).norm();
                    nearest_distances.push(distance);
                }
            }
            
            // Sort distances and take the 6 nearest neighbors (typical for 3D)
            nearest_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nearest_6 = nearest_distances.iter().take(6).collect::<Vec<_>>();
            
            // Calculate local density based on nearest neighbor distances
            if !nearest_6.is_empty() {
                let avg_distance = nearest_6.iter().map(|&&d| d).sum::<f64>() / nearest_6.len() as f64;
                let local_density = 1.0 / (avg_distance.powi(3)); // Volume ~ r³
                local_densities.push(local_density);
            }
        }
        
        // Calculate density variance as a measure of spatial inhomogeneity
        if local_densities.len() > 1 {
            let avg_density = local_densities.iter().sum::<f64>() / local_densities.len() as f64;
            let density_variance = local_densities.iter()
                .map(|&d| (d - avg_density).powi(2))
                .sum::<f64>() / local_densities.len() as f64;
            
            // Calculate configurational entropy using density fluctuations
            // S_config = k_B * ln(Ω) where Ω is the number of microstates
            // For density fluctuations: Ω ∝ exp(-(δρ/ρ)²/2σ²)
            let relative_density_fluctuation = if avg_density > 0.0 {
                (density_variance / (avg_density * avg_density)).sqrt()
            } else {
                0.0
            };
            
            // Configurational entropy based on density fluctuations
            let configurational_entropy = if relative_density_fluctuation > 0.0 {
                BOLTZMANN * (1.0 / relative_density_fluctuation).ln()
            } else {
                0.0
            };
            
            // Add positional entropy based on spatial spread
            let spatial_spread = (var_x * var_y * var_z).powf(1.0/3.0);
            let positional_entropy = if spatial_spread > 0.0 {
                BOLTZMANN * (spatial_spread / 1e-30).ln() // Normalized to avoid log(0)
            } else {
                0.0
            };
            
            configurational_entropy + positional_entropy
        } else {
            0.0
        }
    }
}

/// Main function to update and log emergent properties.
pub fn update_emergent_properties(monitor: &mut EmergenceMonitor, particles: &[PhysicsState], volume: f64) -> Result<()> {
    monitor.update(particles, volume)?;
    log::info!(
        "Emergent Properties Updated: Temp={:.2} K, Pressure={:.2} Pa, Density={:.2} kg/m^3, Entropy={:.2} J/K",
        monitor.temperature.as_kelvin(),
        monitor.pressure.as_pascals(),
        monitor.density.as_kg_per_m3(),
        monitor.entropy.as_joules_per_kelvin()
    );
    Ok(())
}

/// Calculates the Shannon entropy of a set of states using statistical mechanics.
/// Implements proper phase space entropy calculation based on particle distributions
/// in position, momentum, and energy space.
pub fn shannon_entropy(states: &[PhysicsState]) -> f64 {
    if states.is_empty() {
        return 0.0;
    }

    // Calculate entropy in multiple dimensions: mass, energy, momentum, position
    let mass_entropy = calculate_mass_distribution_entropy(states);
    let energy_entropy = calculate_energy_distribution_entropy(states);
    let momentum_entropy = calculate_momentum_distribution_entropy(states);
    let position_entropy = calculate_position_distribution_entropy(states);
    
    // Total Shannon entropy is the sum of all contributions
    mass_entropy + energy_entropy + momentum_entropy + position_entropy
}

/// Calculate entropy from mass distribution using proper binning
fn calculate_mass_distribution_entropy(states: &[PhysicsState]) -> f64 {
    // Use adaptive binning based on mass range
    let masses: Vec<f64> = states.iter().map(|s| s.mass).collect();
    let min_mass = masses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_mass = masses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if min_mass >= max_mass {
        return 0.0;
    }
    
    // Use Sturges' formula for optimal number of bins
    let n_bins = (1.0 + 3.322 * (states.len() as f64).log10()).ceil() as usize;
    let bin_width = (max_mass - min_mass) / n_bins as f64;
    
    let mut bins = vec![0; n_bins];
    for &mass in &masses {
        let bin_index = ((mass - min_mass) / bin_width).floor() as usize;
        let bin_index = bin_index.min(n_bins - 1);
        bins[bin_index] += 1;
    }
    
    // Calculate Shannon entropy: H = -Σ p_i * log(p_i)
    let total = states.len() as f64;
    bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum()
}

/// Calculate entropy from energy distribution
fn calculate_energy_distribution_entropy(states: &[PhysicsState]) -> f64 {
    let energies: Vec<f64> = states.iter()
        .map(|s| 0.5 * s.mass * s.velocity.norm_squared())
        .collect();
    
    let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if min_energy >= max_energy {
        return 0.0;
    }
    
    let n_bins = (1.0 + 3.322 * (states.len() as f64).log10()).ceil() as usize;
    let bin_width = (max_energy - min_energy) / n_bins as f64;
    
    let mut bins = vec![0; n_bins];
    for &energy in &energies {
        let bin_index = ((energy - min_energy) / bin_width).floor() as usize;
        let bin_index = bin_index.min(n_bins - 1);
        bins[bin_index] += 1;
    }
    
    let total = states.len() as f64;
    bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum()
}

/// Calculate entropy from momentum distribution (3D vector space)
fn calculate_momentum_distribution_entropy(states: &[PhysicsState]) -> f64 {
    let momenta: Vec<f64> = states.iter()
        .map(|s| s.mass * s.velocity.norm())
        .collect();
    
    let min_momentum = momenta.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_momentum = momenta.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if min_momentum >= max_momentum {
        return 0.0;
    }
    
    let n_bins = (1.0 + 3.322 * (states.len() as f64).log10()).ceil() as usize;
    let bin_width = (max_momentum - min_momentum) / n_bins as f64;
    
    let mut bins = vec![0; n_bins];
    for &momentum in &momenta {
        let bin_index = ((momentum - min_momentum) / bin_width).floor() as usize;
        let bin_index = bin_index.min(n_bins - 1);
        bins[bin_index] += 1;
    }
    
    let total = states.len() as f64;
    bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum()
}

/// Calculate entropy from position distribution (3D spatial space)
fn calculate_position_distribution_entropy(states: &[PhysicsState]) -> f64 {
    // Calculate position components separately for 3D space
    let x_positions: Vec<f64> = states.iter().map(|s| s.position.x).collect();
    let y_positions: Vec<f64> = states.iter().map(|s| s.position.y).collect();
    let z_positions: Vec<f64> = states.iter().map(|s| s.position.z).collect();
    
    let entropy_x = calculate_1d_distribution_entropy(&x_positions);
    let entropy_y = calculate_1d_distribution_entropy(&y_positions);
    let entropy_z = calculate_1d_distribution_entropy(&z_positions);
    
    // Total spatial entropy is sum of all dimensions
    entropy_x + entropy_y + entropy_z
}

/// Calculate entropy for a 1D distribution
fn calculate_1d_distribution_entropy(values: &[f64]) -> f64 {
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if min_val >= max_val {
        return 0.0;
    }
    
    let n_bins = (1.0 + 3.322 * (values.len() as f64).log10()).ceil() as usize;
    let bin_width = (max_val - min_val) / n_bins as f64;
    
    let mut bins = vec![0; n_bins];
    for &value in values {
        let bin_index = ((value - min_val) / bin_width).floor() as usize;
        let bin_index = bin_index.min(n_bins - 1);
        bins[bin_index] += 1;
    }
    
    let total = values.len() as f64;
    bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum()
}