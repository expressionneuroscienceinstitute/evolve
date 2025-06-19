//! Physics Engine Utility Functions
//! 
//! Common utility functions for physics calculations, conversions, and mathematical operations.
//! This module extracts reusable functionality to reduce code duplication across the physics engine.

use nalgebra::Vector3;
use anyhow::{Result, anyhow};
use crate::{ParticleType, FundamentalParticle, constants::*};
use std::f64::consts::PI;

// Coulomb constant K_e = 1/(4πε₀)
pub const K_E: f64 = 1.0 / (4.0 * PI * VACUUM_PERMITTIVITY);

/// Mathematical and conversion utilities
pub mod math {
    use super::*;
    
    /// Calculate relativistic total energy from momentum and mass
    /// E = sqrt((pc)^2 + (mc^2)^2) where c = speed of light
    pub fn calculate_relativistic_energy(momentum: &Vector3<f64>, mass: f64) -> f64 {
        let momentum_magnitude = momentum.magnitude();
        let rest_energy = mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        let momentum_energy = momentum_magnitude * SPEED_OF_LIGHT;
        
        // Total relativistic energy
        (momentum_energy * momentum_energy + rest_energy * rest_energy).sqrt()
    }
    
    /// Calculate classical kinetic energy: KE = 1/2 * m * v^2
    pub fn calculate_kinetic_energy(velocity: &Vector3<f64>, mass: f64) -> f64 {
        0.5 * mass * velocity.magnitude_squared()
    }
    
    /// Calculate momentum magnitude from velocity and mass: p = mv
    pub fn calculate_momentum_magnitude(velocity: &Vector3<f64>, mass: f64) -> f64 {
        mass * velocity.magnitude()
    }
    
    /// Calculate temperature from average kinetic energy using equipartition theorem
    /// <KE> = (3/2) * k_B * T  =>  T = (2/3) * <KE> / k_B
    pub fn temperature_from_kinetic_energy(avg_kinetic_energy: f64) -> f64 {
        (2.0 / 3.0) * avg_kinetic_energy / BOLTZMANN
    }
    
    /// Calculate thermal velocity for Maxwell-Boltzmann distribution
    /// v_th = sqrt(k_B * T / m)
    pub fn thermal_velocity(temperature: f64, mass: f64) -> f64 {
        (BOLTZMANN * temperature / mass).sqrt()
    }
    
    /// Clamp a value between minimum and maximum bounds
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        if value < min { min }
        else if value > max { max }
        else { value }
    }
    
    /// Linear interpolation between two values
    pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }
    
    /// Convert temperature from Celsius to Kelvin
    pub fn celsius_to_kelvin(celsius: f64) -> f64 {
        celsius + 273.15
    }
    
    /// Convert temperature from Kelvin to Celsius
    pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
        kelvin - 273.15
    }
    
    /// Convert energy from Joules to electron volts
    pub fn joules_to_ev(joules: f64) -> f64 {
        joules / ELEMENTARY_CHARGE
    }
    
    /// Convert energy from electron volts to Joules
    pub fn ev_to_joules(ev: f64) -> f64 {
        ev * ELEMENTARY_CHARGE
    }
    
    /// Convert mass from kg to atomic mass units (u)
    pub fn kg_to_amu(kg: f64) -> f64 {
        kg / ATOMIC_MASS_UNIT
    }
    
    /// Convert mass from atomic mass units (u) to kg
    pub fn amu_to_kg(amu: f64) -> f64 {
        amu * ATOMIC_MASS_UNIT
    }
    
    /// Calculate Gaussian distribution value
    pub fn gaussian(x: f64, mean: f64, std_dev: f64) -> f64 {
        let variance = std_dev * std_dev;
        let coefficient = 1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt());
        let exponent = -0.5 * ((x - mean) * (x - mean)) / variance;
        coefficient * exponent.exp()
    }
    
    /// Calculate distance between two 3D points
    pub fn distance_3d(pos1: &Vector3<f64>, pos2: &Vector3<f64>) -> f64 {
        (pos2 - pos1).magnitude()
    }
    
    /// Normalize a vector (make it unit length)
    pub fn normalize_vector(v: &Vector3<f64>) -> Vector3<f64> {
        let magnitude = v.magnitude();
        if magnitude > 0.0 {
            v / magnitude
        } else {
            Vector3::zeros()
        }
    }
}

/// Physics calculation utilities
pub mod physics {
    use super::*;
    
    /// Calculate Coulomb force between two charged particles
    /// F = k * q1 * q2 / r^2, where k = 1/(4πε₀)
    pub fn coulomb_force(charge1: f64, charge2: f64, distance: f64) -> f64 {
        if distance == 0.0 { return 0.0; }
        K_E * charge1 * charge2 / (distance * distance)
    }
    
    /// Calculate gravitational force between two masses
    /// F = G * m1 * m2 / r^2
    pub fn gravitational_force(mass1: f64, mass2: f64, distance: f64) -> f64 {
        if distance == 0.0 { return 0.0; }
        GRAVITATIONAL_CONSTANT * mass1 * mass2 / (distance * distance)
    }
    
    /// Calculate cross-sectional area for particle interactions
    pub fn calculate_cross_section(particle1: ParticleType, particle2: ParticleType) -> f64 {
        // Simplified cross-section calculation
        // In practice, this would use detailed quantum field theory calculations
        match (particle1, particle2) {
            (ParticleType::Electron, ParticleType::Photon) => 6.65e-29, // Thomson scattering
            (ParticleType::Proton, ParticleType::Proton) => 1e-27,      // Nuclear strong force
            (ParticleType::Electron, ParticleType::Electron) => 1e-31,  // EM scattering
            _ => 1e-32, // Default small cross-section
        }
    }
    
    /// Calculate decay time for unstable particles based on half-life
    pub fn calculate_decay_time(particle_type: ParticleType) -> Option<f64> {
        let half_life = match particle_type {
            ParticleType::Neutron => 881.5, // seconds
            ParticleType::Muon => 2.197e-6,  // seconds
            ParticleType::Tau => 2.9e-13,    // seconds
            ParticleType::PionPlus => 2.6e-8, // seconds
            ParticleType::PionMinus => 2.6e-8,
            _ => return None, // Stable particle
        };
        
        // Convert half-life to mean lifetime: τ = t₁/₂ / ln(2)
        Some(half_life / 2.0_f64.ln())
    }
    
    /// Calculate interaction range based on particle types
    pub fn calculate_interaction_range(p1: ParticleType, p2: ParticleType) -> f64 {
        use ParticleType::*;
        
        match (p1, p2) {
            // Strong force (very short range)
            (Proton, Proton) | (Proton, Neutron) | (Neutron, Neutron) => 1e-15, // 1 fm
            
            // Electromagnetic force (long range, but practical cutoff)
            (Electron, _) | (_, Electron) if has_charge(p1) || has_charge(p2) => 1e-10, // 1 Å
            
            // Weak force (very short range)
            _ if can_weak_interact(p1) && can_weak_interact(p2) => 1e-18, // 0.001 fm
            
            // Default fallback
            _ => 1e-12, // 1 pm
        }
    }
    
    /// Check if particle type has electric charge
    pub fn has_charge(particle_type: ParticleType) -> bool {
        !matches!(particle_type, 
            ParticleType::Photon | ParticleType::ZBoson | ParticleType::Gluon |
            ParticleType::Neutron | ParticleType::ElectronNeutrino |
            ParticleType::MuonNeutrino | ParticleType::TauNeutrino |
            ParticleType::ElectronAntiNeutrino | ParticleType::MuonAntiNeutrino |
            ParticleType::TauAntiNeutrino | ParticleType::PionZero | ParticleType::Eta
        )
    }
    
    /// Check if particle can participate in weak interactions
    pub fn can_weak_interact(particle_type: ParticleType) -> bool {
        // All fermions and weak gauge bosons can participate in weak interactions
        matches!(particle_type,
            ParticleType::Electron | ParticleType::Muon | ParticleType::Tau |
            ParticleType::ElectronNeutrino | ParticleType::MuonNeutrino | ParticleType::TauNeutrino |
            ParticleType::ElectronAntiNeutrino | ParticleType::MuonAntiNeutrino | ParticleType::TauAntiNeutrino |
            ParticleType::Up | ParticleType::Down | ParticleType::Charm | ParticleType::Strange |
            ParticleType::Top | ParticleType::Bottom | ParticleType::Proton | ParticleType::Neutron |
            ParticleType::WBoson | ParticleType::WBosonMinus | ParticleType::ZBoson
        )
    }
    
    /// Calculate step length for particle simulation based on local conditions
    pub fn calculate_step_length(particle: &FundamentalParticle, local_density: f64) -> f64 {
        let base_step = 1e-12; // 1 picometer
        let velocity_factor = particle.velocity.magnitude() / SPEED_OF_LIGHT;
        let density_factor = 1.0 / (1.0 + local_density * 1e30); // Scale density appropriately
        
        base_step * velocity_factor.max(0.1) * density_factor.max(0.01)
    }
    
    /// Calculate local particle density in a region
    pub fn calculate_local_density(position: &Vector3<f64>, particles: &[FundamentalParticle], radius: f64) -> f64 {
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
        let nearby_mass: f64 = particles.iter()
            .filter(|p| (p.position - position).magnitude() <= radius)
            .map(|p| p.mass)
            .sum();
        
        nearby_mass / volume
    }
}

/// Stellar physics utilities
pub mod stellar {
    
    /// Calculate stellar radius using empirical mass-radius relation
    /// R ∝ M^α where α depends on stellar mass regime
    pub fn calculate_stellar_radius(mass_solar: f64) -> f64 {
        const SOLAR_RADIUS: f64 = 6.96e8; // meters
        
        let radius_ratio = if mass_solar < 1.0 {
            // Low-mass stars: R ∝ M^0.8
            mass_solar.powf(0.8)
        } else if mass_solar < 20.0 {
            // Main sequence: R ∝ M^0.57
            mass_solar.powf(0.57)
        } else {
            // Massive stars: R ∝ M^0.5
            mass_solar.powf(0.5)
        };
        
        radius_ratio * SOLAR_RADIUS
    }
    
    /// Calculate stellar luminosity using mass-luminosity relation
    /// L ∝ M^α where α ≈ 3.5 for main sequence stars
    pub fn calculate_stellar_luminosity(mass_solar: f64) -> f64 {
        const SOLAR_LUMINOSITY: f64 = 3.828e26; // watts
        
        let luminosity_ratio = if mass_solar < 0.43 {
            // Very low mass: L ∝ M^2.3
            mass_solar.powf(2.3)
        } else if mass_solar < 2.0 {
            // Low to intermediate mass: L ∝ M^4
            mass_solar.powf(4.0)
        } else if mass_solar < 20.0 {
            // High mass: L ∝ M^3.5
            mass_solar.powf(3.5)
        } else {
            // Very high mass: L ∝ M^3
            mass_solar.powf(3.0)
        };
        
        luminosity_ratio * SOLAR_LUMINOSITY
    }
    
    /// Calculate stellar effective temperature
    /// Using Stefan-Boltzmann law: L = 4πR²σT⁴
    pub fn calculate_stellar_temperature(mass_solar: f64) -> f64 {
        const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W m⁻² K⁻⁴
        
        let luminosity = calculate_stellar_luminosity(mass_solar);
        let radius = calculate_stellar_radius(mass_solar);
        let surface_area = 4.0 * std::f64::consts::PI * radius * radius;
        
        // T⁴ = L / (4πR²σ)
        let t4 = luminosity / (surface_area * STEFAN_BOLTZMANN);
        t4.powf(0.25)
    }
    
    /// Classify stellar type based on temperature (Harvard spectral classification)
    pub fn classify_stellar_type(temperature: f64) -> &'static str {
        match temperature as i32 {
            t if t >= 30000 => "O", // Blue stars
            t if t >= 10000 => "B", // Blue-white stars
            t if t >= 7500 => "A",  // White stars
            t if t >= 6000 => "F",  // Yellow-white stars
            t if t >= 5200 => "G",  // Yellow stars (like Sun)
            t if t >= 3700 => "K",  // Orange stars
            _ => "M",               // Red stars
        }
    }
}

/// Nuclear physics utilities
pub mod nuclear {
    use super::*;
    
    /// Calculate Q-value for nuclear reactions (energy released)
    /// Q = (mass_reactants - mass_products) * c²
    pub fn calculate_q_value(reactant_masses: &[f64], product_masses: &[f64]) -> f64 {
        let total_reactant_mass: f64 = reactant_masses.iter().sum();
        let total_product_mass: f64 = product_masses.iter().sum();
        let mass_difference = total_reactant_mass - total_product_mass;
        
        // Convert mass difference to energy using E = mc²
        mass_difference * SPEED_OF_LIGHT * SPEED_OF_LIGHT
    }
    
    /// Calculate Coulomb barrier height for nuclear fusion
    /// V = k * Z1 * Z2 * e² / (R1 + R2)
    pub fn calculate_coulomb_barrier(z1: u32, z2: u32, r1: f64, r2: f64) -> f64 {
        let charge1 = z1 as f64 * ELEMENTARY_CHARGE;
        let charge2 = z2 as f64 * ELEMENTARY_CHARGE;
        let separation = r1 + r2;
        
        K_E * charge1 * charge2 / separation
    }
    
    /// Calculate nuclear radius using empirical formula
    /// R = r₀ * A^(1/3) where r₀ ≈ 1.2 fm
    pub fn calculate_nuclear_radius(mass_number: u32) -> f64 {
        const R0: f64 = 1.2e-15; // meters (femtometers)
        R0 * (mass_number as f64).powf(1.0/3.0)
    }
    
    /// Calculate binding energy per nucleon (semi-empirical mass formula)
    pub fn calculate_binding_energy_per_nucleon(mass_number: u32, atomic_number: u32) -> f64 {
        let a = mass_number as f64;
        let z = atomic_number as f64;
        let n = a - z; // Number of neutrons
        
        // Semi-empirical mass formula terms (in MeV)
        let volume_term = 15.75 * a;
        let surface_term = -17.8 * a.powf(2.0/3.0);
        let coulomb_term = -0.711 * z * z / a.powf(1.0/3.0);
        let asymmetry_term = -23.7 * (n - z).powi(2) / a;
        
        // Pairing term
        let pairing_term = if atomic_number % 2 == 0 && (mass_number - atomic_number) % 2 == 0 {
            12.0 / a.sqrt() // Even-even
        } else if atomic_number % 2 == 1 && (mass_number - atomic_number) % 2 == 1 {
            -12.0 / a.sqrt() // Odd-odd
        } else {
            0.0 // Even-odd or odd-even
        };
        
        let total_be = volume_term + surface_term + coulomb_term + asymmetry_term + pairing_term;
        total_be / a // Binding energy per nucleon
    }
}

/// Validation and error checking utilities
pub mod validation {
    use super::*;
    
    /// Validate that a physical quantity is finite and non-negative
    pub fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
        if !value.is_finite() {
            return Err(anyhow!("{} must be finite, got: {}", name, value));
        }
        if value < 0.0 {
            return Err(anyhow!("{} must be non-negative, got: {}", name, value));
        }
        Ok(())
    }
    
    /// Validate that a vector contains finite values
    pub fn validate_vector_finite(vector: &Vector3<f64>, name: &str) -> Result<()> {
        for (i, &component) in vector.iter().enumerate() {
            if !component.is_finite() {
                return Err(anyhow!("{} component {} must be finite, got: {}", name, i, component));
            }
        }
        Ok(())
    }
    
    /// Validate particle mass is within reasonable bounds
    pub fn validate_particle_mass(mass: f64) -> Result<()> {
        const MIN_MASS: f64 = 1e-35; // Roughly electron mass / 1000
        const MAX_MASS: f64 = 1e-20;  // Roughly proton mass * 1000
        
        validate_positive_finite(mass, "particle mass")?;
        
        if mass < MIN_MASS {
            return Err(anyhow!("Particle mass too small: {} kg", mass));
        }
        if mass > MAX_MASS {
            return Err(anyhow!("Particle mass too large: {} kg", mass));
        }
        
        Ok(())
    }
    
    /// Validate temperature is physically reasonable
    pub fn validate_temperature(temperature: f64) -> Result<()> {
        validate_positive_finite(temperature, "temperature")?;
        
        const MAX_TEMP: f64 = 1e12; // 1 TK (roughly Planck temperature / 100)
        
        if temperature > MAX_TEMP {
            return Err(anyhow!("Temperature too high: {} K", temperature));
        }
        
        Ok(())
    }
    
    /// Validate energy conservation in interactions
    pub fn validate_energy_conservation(
        initial_energy: f64, 
        final_energy: f64, 
        tolerance: f64
    ) -> Result<()> {
        let energy_difference = (final_energy - initial_energy).abs();
        let relative_error = energy_difference / initial_energy.max(final_energy).max(1e-100);
        
        if relative_error > tolerance {
            return Err(anyhow!(
                "Energy conservation violated: initial={:.3e} J, final={:.3e} J, error={:.3e}",
                initial_energy, final_energy, relative_error
            ));
        }
        
        Ok(())
    }
}

/// Performance and optimization utilities
pub mod performance {
    use super::*;
    
    /// Calculate optimal spatial grid cell size for particle interactions
    pub fn optimal_grid_cell_size(particles: &[FundamentalParticle], interaction_range: f64) -> f64 {
        if particles.is_empty() {
            return interaction_range;
        }
        
        // Use interaction range as base, but adjust based on particle density
        let total_volume = calculate_bounding_box_volume(particles);
        let particle_density = particles.len() as f64 / total_volume;
        let avg_particle_spacing = particle_density.powf(-1.0/3.0);
        
        // Cell size should be between interaction range and average spacing
        interaction_range.min(avg_particle_spacing * 2.0).max(interaction_range * 0.5)
    }
    
    /// Calculate bounding box volume for a set of particles
    fn calculate_bounding_box_volume(particles: &[FundamentalParticle]) -> f64 {
        if particles.is_empty() {
            return 1.0; // Default volume
        }
        
        let mut min_pos = particles[0].position;
        let mut max_pos = particles[0].position;
        
        for particle in particles.iter().skip(1) {
            for i in 0..3 {
                min_pos[i] = min_pos[i].min(particle.position[i]);
                max_pos[i] = max_pos[i].max(particle.position[i]);
            }
        }
        
        let dimensions = max_pos - min_pos;
        dimensions.x.max(1e-10) * dimensions.y.max(1e-10) * dimensions.z.max(1e-10)
    }
    
    /// Estimate memory usage for particle storage
    pub fn estimate_particle_memory_usage(particle_count: usize) -> usize {
        // Rough estimate based on FundamentalParticle struct size
        const BYTES_PER_PARTICLE: usize = 512; // Conservative estimate
        particle_count * BYTES_PER_PARTICLE
    }
    
    /// Get optimal number of threads for parallel computation
    pub fn get_optimal_thread_count() -> usize {
        // Use standard library function available since Rust 1.59
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1) // Fallback to single thread if detection fails
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;
    use crate::{SPEED_OF_LIGHT, ELEMENTARY_CHARGE};
    
    #[test]
    fn test_relativistic_energy() {
        let momentum = Vector3::new(1e-21, 0.0, 0.0); // kg⋅m/s
        let mass = 9.109e-31; // electron mass in kg
        
        let energy = super::math::calculate_relativistic_energy(&momentum, mass);
        
        // Should be greater than rest mass energy
        let rest_energy = mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        assert!(energy > rest_energy);
        assert!(energy.is_finite());
    }
    
    #[test]
    fn test_coulomb_force() {
        let charge1 = ELEMENTARY_CHARGE;  // proton
        let charge2 = -ELEMENTARY_CHARGE; // electron
        let distance = 5.29e-11; // Bohr radius
        
        let force = super::physics::coulomb_force(charge1, charge2, distance);
        
        // Should be negative (attractive) and reasonable magnitude
        assert!(force < 0.0);
        assert!(force.abs() > 1e-10); // Should be significant force
        assert!(force.is_finite());
    }
    
    #[test]
    fn test_stellar_calculations() {
        let solar_mass = 1.0;
        
        let radius = super::stellar::calculate_stellar_radius(solar_mass);
        let luminosity = super::stellar::calculate_stellar_luminosity(solar_mass);
        let temperature = super::stellar::calculate_stellar_temperature(solar_mass);
        
        // Solar values should be close to known constants
        assert!((radius / 6.96e8 - 1.0).abs() < 0.1); // Within 10% of solar radius
        assert!((temperature - 5778.0).abs() < 1000.0); // Within 1000K of solar temperature
        assert!(luminosity > 0.0);
        
        let stellar_type = super::stellar::classify_stellar_type(temperature);
        assert_eq!(stellar_type, "G"); // Sun is G-type
    }
    
    #[test]
    fn test_validation() {
        use super::validation::*;
        
        // Valid inputs should pass
        assert!(validate_positive_finite(1.0, "test").is_ok());
        assert!(validate_particle_mass(9.109e-31).is_ok()); // electron mass
        assert!(validate_temperature(300.0).is_ok()); // room temperature
        
        // Invalid inputs should fail
        assert!(validate_positive_finite(-1.0, "test").is_err());
        assert!(validate_positive_finite(f64::NAN, "test").is_err());
        assert!(validate_particle_mass(0.0).is_err());
        assert!(validate_temperature(-1.0).is_err());
    }
    
    #[test]
    fn test_unit_conversions() {
        use super::math::*;
        
        // Temperature conversions
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((kelvin_to_celsius(273.15)).abs() < 1e-10);
        
        // Energy conversions (test with 1 eV)
        let energy_j = ev_to_joules(1.0);
        let energy_ev = joules_to_ev(energy_j);
        assert!((energy_ev - 1.0).abs() < 1e-10);
        
        // Mass conversions (test with 1 u)
        let mass_kg = amu_to_kg(1.0);
        let mass_amu = kg_to_amu(mass_kg);
        assert!((mass_amu - 1.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod molecular_dynamics_tests {
    // Add any necessary imports and setup for molecular dynamics tests
} 