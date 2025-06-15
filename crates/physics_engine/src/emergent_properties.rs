//! # Physics Engine: Emergent Properties Module
//!
//! This module is dedicated to observing and quantifying macroscopic properties
//! that emerge from the collective behavior of microscopic particles. These properties,
//! such as temperature and pressure, are not fundamental to individual particles but
//! arise from their statistical mechanics.

use anyhow::Result;

use crate::{constants::BOLTZMANN, PhysicsState};
// use crate::particles::Particle;

/// Represents temperature in Kelvin.
#[derive(Debug, Default)]
pub struct Temperature(pub f64);

impl Temperature {
    pub fn as_kelvin(&self) -> f64 {
        self.0
    }
}

/// Represents pressure in Pascals.
#[derive(Debug, Default)]
pub struct Pressure(pub f64);

impl Pressure {
    pub fn as_pascals(&self) -> f64 {
        self.0
    }
}

/// Represents density in kilograms per cubic meter.
#[derive(Debug, Default)]
pub struct Density(pub f64);

impl Density {
    pub fn as_kg_per_m3(&self) -> f64 {
        self.0
    }
}

/// Represents entropy in Joules per Kelvin.
#[derive(Debug, Default)]
pub struct Entropy(pub f64);

/// Monitors and calculates emergent properties of the system.
#[derive(Debug, Default)]
pub struct EmergenceMonitor {
    pub temperature: Temperature,
    pub pressure: Pressure,
    pub density: Density,
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
        // self.entropy = self.calculate_entropy(states);
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

    /// Calculate the pressure using the ideal gas law for now.
    fn calculate_pressure(&self, particles: &[PhysicsState], volume: f64) -> Pressure {
        if volume == 0.0 {
            return Pressure(0.0);
        }
        let num_particles = particles.len() as f64;
        let temperature = self.calculate_temperature(particles).as_kelvin();
        let pressure = num_particles * BOLTZMANN * temperature / volume;
        Pressure(pressure)
    }

    /// Calculate the total density of the particles.
    fn calculate_density(&self, particles: &[PhysicsState], volume: f64) -> Density {
        if volume == 0.0 {
            return Density(0.0);
        }
        let total_mass: f64 = particles.iter().map(|p| p.mass).sum();
        Density(total_mass / volume)
    }

    /// Calculate the entropy.
    /// This is a placeholder and would require a more complex statistical
    /// mechanics calculation.
    #[allow(dead_code)]
    fn calculate_entropy(&self, _particles: &[PhysicsState]) -> Entropy {
        // A proper implementation would likely involve phase space volume, which is
        // very complex to calculate.
        Entropy(0.0)
    }
}

/// Main function to update and log emergent properties.
pub fn update_emergent_properties(monitor: &mut EmergenceMonitor, particles: &[PhysicsState], volume: f64) -> Result<()> {
    monitor.update(particles, volume)?;
    log::info!(
        "Emergent Properties Updated: Temp={:.2} K, Pressure={:.2} Pa, Density={:.2} kg/m^3",
        monitor.temperature.as_kelvin(),
        monitor.pressure.as_pascals(),
        monitor.density.as_kg_per_m3()
    );
    Ok(())
}