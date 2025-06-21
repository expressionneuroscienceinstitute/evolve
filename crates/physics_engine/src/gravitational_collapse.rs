//! Gravitational Collapse and Jeans Instability
//!
//! Implements detection of gravitational instability (Jeans criterion) and sink particle formation for star formation.
//!
//! References:
//! - Jeans, J. H. (1902). The Stability of a Spherical Nebula. Philosophical Transactions of the Royal Society A.
//! - Binney & Tremaine, Galactic Dynamics (2nd Ed.), Princeton, 2008.
//! - Hubber et al. (2011), SEREN SPH code, A&A 529, A27.
//! - Krumholz, M. R. (2014). Star Formation in Molecular Clouds. Physics Reports.

use crate::constants::{BOLTZMANN, GRAVITATIONAL_CONSTANT};
use crate::sph::SphParticle;
use nalgebra::{Vector3, zero};

/// Physical constants for mean molecular weight (default: atomic hydrogen)
pub const MEAN_MOLECULAR_WEIGHT: f64 = 1.0; // Can be set per simulation
pub const HYDROGEN_MASS: f64 = 1.6735575e-27; // kg

/// Calculate the Jeans mass (kg) for a region with given temperature (K) and density (kg/m^3)
pub fn jeans_mass(temperature: f64, density: f64, mu: f64) -> f64 {
    let num = 5.0 * BOLTZMANN * temperature;
    let denom = GRAVITATIONAL_CONSTANT * mu * HYDROGEN_MASS;
    let factor = (num / denom).powf(1.5);
    let rho_factor = (3.0 / (4.0 * std::f64::consts::PI * density)).sqrt();
    factor * rho_factor
}

/// Calculate the Jeans length (m) for a region with given temperature (K) and density (kg/m^3)
pub fn jeans_length(temperature: f64, density: f64, mu: f64) -> f64 {
    ((15.0 * BOLTZMANN * temperature) / (4.0 * std::f64::consts::PI * GRAVITATIONAL_CONSTANT * mu * HYDROGEN_MASS * density)).sqrt()
}

/// Represents a collapsed object (protostar, sink particle) formed by gravitational instability
#[derive(Debug, Clone)]
pub struct SinkParticle {
    pub mass: f64,                // Total mass (kg)
    pub position: Vector3<f64>,   // Center of mass (m)
    pub velocity: Vector3<f64>,   // Mass-weighted velocity (m/s)
    pub angular_momentum: Vector3<f64>, // Total angular momentum (kg m^2/s)
    pub accretion_radius: f64,    // Accretion radius (m)
    pub creation_time: f64,       // Simulation time of formation (s)
    pub id: u64,                  // Unique identifier
}

impl SinkParticle {
    /// Create a new sink particle from a set of SPH particles
    pub fn from_particles(particles: &[SphParticle], accretion_radius: f64, creation_time: f64, id: u64) -> Self {
        let total_mass: f64 = particles.iter().map(|p| p.particle.mass).sum();
        let position = particles.iter().fold(Vector3::zeros(), |acc, p| acc + p.particle.position * p.particle.mass) / total_mass;
        let velocity = particles.iter().fold(Vector3::zeros(), |acc, p| acc + p.particle.velocity * p.particle.mass) / total_mass;
        let angular_momentum = particles.iter().fold(Vector3::zeros(), |acc, p| {
            acc + (p.particle.position - position).cross(&(p.particle.velocity * p.particle.mass))
        });
        Self {
            mass: total_mass,
            position,
            velocity,
            angular_momentum,
            accretion_radius,
            creation_time,
            id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_jeans_mass_and_length() {
        // Typical molecular cloud: T=10K, n=1e3 cm^-3
        let temperature = 10.0;
        let density = 1e3 * 1.6735575e-27 * 1e6; // 1e3 cm^-3 to m^-3
        let mu = 2.33; // mean molecular weight for molecular hydrogen
        let m_j = jeans_mass(temperature, density, mu);
        let l_j = jeans_length(temperature, density, mu);
        // Should be in the range of a few solar masses and tenths of a parsec
        assert!(m_j > 1.0e30 && m_j < 1.0e33, "Jeans mass out of expected range: {}", m_j);
        assert!(l_j > 1.0e15 && l_j < 1.0e17, "Jeans length out of expected range: {}", l_j);
    }
} 