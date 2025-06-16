//! # Physics Engine: Emergent Properties Module
//!
//! This module is dedicated to observing and quantifying macroscopic properties
//! that emerge from the collective behavior of microscopic particles. These properties,
//! such as temperature and pressure, are not fundamental to individual particles but
//! arise from their statistical mechanics.

use nalgebra::Vector3;
use crate::*;
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
    fn calculate_spatial_entropy(&self, particles: &[PhysicsState]) -> f64 {
        if particles.len() < 2 {
            return 0.0;
        }
        
        // Simple approximation: calculate entropy from position variance
        // This is a rough estimate of the spatial distribution entropy
        
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
        
        // Geometric mean of variances as a measure of spatial spread
        let spatial_spread = (var_x * var_y * var_z).powf(1.0/3.0);
        
        // Convert to entropy contribution (logarithmic relationship)
        // This is an approximation - exact calculation would require full phase space
        if spatial_spread > 0.0 {
            BOLTZMANN * spatial_spread.ln()
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

/// Calculates the Shannon entropy of a set of states.
pub fn shannon_entropy(states: &[PhysicsState]) -> f64 {
    // ... existing code ...
}