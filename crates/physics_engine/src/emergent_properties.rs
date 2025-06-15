//! # Physics Engine: Emergent Properties Module
//!
//! This module is dedicated to observing and quantifying macroscopic properties
//! that emerge from the collective behavior of microscopic particles. These properties,
//! such as temperature and pressure, are not fundamental to individual particles but
//! arise from their statistical mechanics.

use anyhow::Result;
use nalgebra::Vector3;
use crate::particles::Particle;

/// Represents the temperature of a system, an emergent property related to the
/// average kinetic energy of its constituent particles.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Temperature(f64); // In Kelvin

impl Temperature {
    /// Calculates the temperature of a collection of particles.
    /// T = (2/3k) * <E_k>, where <E_k> is the average kinetic energy and k is Boltzmann's constant.
    pub fn from_particles(particles: &[&Particle]) -> Self {
        if particles.is_empty() {
            return Temperature(0.0);
        }

        // Using a mock Boltzmann constant for simulation purposes.
        const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

        let total_kinetic_energy: f64 = particles.iter()
            .map(|p| 0.5 * p.mass * p.velocity.magnitude_squared())
            .sum();
        
        let avg_kinetic_energy = total_kinetic_energy / particles.len() as f64;
        let temperature = (2.0 / (3.0 * BOLTZMANN_CONSTANT)) * avg_kinetic_energy;
        Temperature(temperature)
    }

    pub fn as_kelvin(&self) -> f64 {
        self.0
    }
}

/// Represents the pressure of a system, an emergent property related to the
/// force exerted by particles on a surface.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pressure(f64); // In Pascals

impl Pressure {
    /// Calculates pressure using the ideal gas law: P = (NkT) / V.
    /// This is a simplification and assumes the system behaves like an ideal gas.
    pub fn from_state(num_particles: usize, temperature: Temperature, volume: f64) -> Self {
        if volume <= 0.0 {
            return Pressure(0.0);
        }
        const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;
        let pressure = (num_particles as f64 * BOLTZMANN_CONSTANT * temperature.as_kelvin()) / volume;
        Pressure(pressure)
    }

    pub fn as_pascals(&self) -> f64 {
        self.0
    }
}

/// Monitors and calculates emergent properties of the system.
#[derive(Debug, Default)]
pub struct EmergenceMonitor {
    pub temperature: Temperature,
    pub pressure: Pressure,
}

impl EmergenceMonitor {
    pub fn new() -> Self {
        EmergenceMonitor {
            temperature: Temperature(0.0),
            pressure: Pressure(0.0),
        }
    }

    /// Updates all emergent properties based on the current state of the particle system.
    pub fn update(&mut self, particles: &[&Particle], volume: f64) {
        self.temperature = Temperature::from_particles(particles);
        self.pressure = Pressure::from_state(particles.len(), self.temperature, volume);
    }
}

/// Main function to update and log emergent properties.
pub fn update_emergent_properties(monitor: &mut EmergenceMonitor, particles: &[&Particle], volume: f64) -> Result<()> {
    monitor.update(particles, volume);
    log::info!(
        "Emergent Properties Updated: Temp={:.2} K, Pressure={:.2} Pa",
        monitor.temperature.as_kelvin(),
        monitor.pressure.as_pascals()
    );
    Ok(())
}