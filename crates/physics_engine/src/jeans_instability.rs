//! Jeans Instability Physics
//! 
//! Implements the critical physics for gravitational collapse and star formation
//! through Jeans instability analysis. This module provides the mathematical
//! foundation for determining when gas clouds will collapse under their own
//! gravity, overcoming internal pressure support.
//! 
//! The Jeans instability is fundamental to:
//! - Star formation from molecular clouds
//! - Galaxy formation and structure
//! - Cosmological structure formation
//! - Initial mass function determination
//! 
//! Mathematical Foundation:
//! The Jeans mass and length are derived from the balance between gravitational
//! potential energy and thermal kinetic energy:
//! 
//! M_J = (5 k_B T / (G μ m_H))^(3/2) * (3 / (4π ρ))^(1/2)
//! λ_J = √(15 k_B T / (4π G μ m_H ρ))
//! 
//! where:
//! - T: gas temperature (K)
//! - ρ: gas density (kg/m³)
//! - μ: mean molecular weight
//! - m_H: hydrogen mass (kg)
//! - k_B: Boltzmann constant (J/K)
//! - G: gravitational constant (m³/kg/s²)
//! 
//! References:
//! - Binney & Tremaine, "Galactic Dynamics" (2008)
//! - Shu, "The Physics of Astrophysics: Gas Dynamics" (1992)
//! - Larson, "The physics of star formation" (2003)
//! - Krumholz & McKee, "The theory of the initial mass function" (2005)

use anyhow::Result;
use nalgebra::Vector3;
use crate::{PhysicsConstants, ParticleType, FundamentalParticle as Particle};

/// Jeans instability analysis results
#[derive(Debug, Clone)]
pub struct JeansAnalysis {
    /// Jeans mass (kg) - critical mass for gravitational collapse
    pub jeans_mass: f64,
    /// Jeans length (m) - critical length scale for collapse
    pub jeans_length: f64,
    /// Jeans density (kg/m³) - critical density for collapse
    pub jeans_density: f64,
    /// Free-fall time (s) - time for collapse under gravity
    pub free_fall_time: f64,
    /// Sound crossing time (s) - time for pressure to respond
    pub sound_crossing_time: f64,
    /// Instability parameter (M/M_J) - >1 means unstable
    pub instability_parameter: f64,
    /// Collapse mode - how the instability will proceed
    pub collapse_mode: CollapseMode,
}

/// Different modes of gravitational collapse
#[derive(Debug, Clone, PartialEq)]
pub enum CollapseMode {
    /// Stable - pressure support exceeds gravity
    Stable,
    /// Fragmentation - will break into multiple objects
    Fragmentation,
    /// Monolithic collapse - single object formation
    MonolithicCollapse,
    /// Turbulent fragmentation - turbulence-driven breakup
    TurbulentFragmentation,
}

/// Jeans instability solver for gravitational collapse analysis
#[derive(Debug, Clone)]
pub struct JeansInstabilitySolver {
    /// Physics constants for calculations
    constants: PhysicsConstants,
    /// Minimum mass for sink particle formation (solar masses)
    min_sink_mass: f64,
    /// Maximum mass for sink particle formation (solar masses)
    max_sink_mass: f64,
    /// Sink particle accretion radius (m)
    sink_accretion_radius: f64,
}

impl Default for JeansInstabilitySolver {
    fn default() -> Self {
        Self {
            constants: PhysicsConstants::default(),
            min_sink_mass: 0.01, // 0.01 M☉ minimum
            max_sink_mass: 100.0, // 100 M☉ maximum
            sink_accretion_radius: 1e12, // 1 AU accretion radius
        }
    }
}

impl JeansInstabilitySolver {
    /// Create a new Jeans instability solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a solver with custom parameters
    pub fn with_parameters(
        min_sink_mass: f64,
        max_sink_mass: f64,
        sink_accretion_radius: f64,
    ) -> Self {
        Self {
            constants: PhysicsConstants::default(),
            min_sink_mass,
            max_sink_mass,
            sink_accretion_radius,
        }
    }

    /// Calculate Jeans mass for given gas conditions
    /// 
    /// M_J = (5 k_B T / (G μ m_H))^(3/2) * (3 / (4π ρ))^(1/2)
    /// 
    /// # Arguments
    /// * `temperature` - Gas temperature (K)
    /// * `density` - Gas density (kg/m³)
    /// * `mean_molecular_weight` - Mean molecular weight μ
    /// 
    /// # Returns
    /// Jeans mass in kg
    pub fn calculate_jeans_mass(
        &self,
        temperature: f64,
        density: f64,
        mean_molecular_weight: f64,
    ) -> f64 {
        let k_b = self.constants.k_b;
        let g = self.constants.g;
        let m_h = self.constants.m_p; // Use proton mass as hydrogen mass
        
        // Jeans mass formula
        let factor1 = (5.0 * k_b * temperature) / (g * mean_molecular_weight * m_h);
        let factor2 = 3.0 / (4.0 * std::f64::consts::PI * density);
        
        factor1.powf(1.5) * factor2.sqrt()
    }

    /// Calculate Jeans length for given gas conditions
    /// 
    /// λ_J = √(15 k_B T / (4π G μ m_H ρ))
    /// 
    /// # Arguments
    /// * `temperature` - Gas temperature (K)
    /// * `density` - Gas density (kg/m³)
    /// * `mean_molecular_weight` - Mean molecular weight μ
    /// 
    /// # Returns
    /// Jeans length in meters
    pub fn calculate_jeans_length(
        &self,
        temperature: f64,
        density: f64,
        mean_molecular_weight: f64,
    ) -> f64 {
        let k_b = self.constants.k_b;
        let g = self.constants.g;
        let m_h = self.constants.m_p; // Use proton mass as hydrogen mass
        
        // Jeans length formula
        let numerator = 15.0 * k_b * temperature;
        let denominator = 4.0 * std::f64::consts::PI * g * mean_molecular_weight * m_h * density;
        
        (numerator / denominator).sqrt()
    }

    /// Calculate Jeans density for given conditions
    /// 
    /// ρ_J = (15 k_B T) / (4π G μ m_H λ²)
    /// 
    /// # Arguments
    /// * `temperature` - Gas temperature (K)
    /// * `length_scale` - Characteristic length scale (m)
    /// * `mean_molecular_weight` - Mean molecular weight μ
    /// 
    /// # Returns
    /// Jeans density in kg/m³
    pub fn calculate_jeans_density(
        &self,
        temperature: f64,
        length_scale: f64,
        mean_molecular_weight: f64,
    ) -> f64 {
        let k_b = self.constants.k_b;
        let g = self.constants.g;
        let m_h = self.constants.m_p; // Use proton mass as hydrogen mass
        
        // Jeans density formula
        let numerator = 15.0 * k_b * temperature;
        let denominator = 4.0 * std::f64::consts::PI * g * mean_molecular_weight * m_h * length_scale.powi(2);
        
        numerator / denominator
    }

    /// Calculate free-fall time for gravitational collapse
    /// 
    /// t_ff = √(3π / (32 G ρ))
    /// 
    /// # Arguments
    /// * `density` - Gas density (kg/m³)
    /// 
    /// # Returns
    /// Free-fall time in seconds
    pub fn calculate_free_fall_time(&self, density: f64) -> f64 {
        let g = self.constants.g;
        
        // Free-fall time formula
        let factor = 3.0 * std::f64::consts::PI / (32.0 * g * density);
        factor.sqrt()
    }

    /// Calculate sound crossing time for pressure response
    /// 
    /// t_sound = L / c_s
    /// where c_s = √(γ k_B T / (μ m_H))
    /// 
    /// # Arguments
    /// * `length_scale` - Characteristic length scale (m)
    /// * `temperature` - Gas temperature (K)
    /// * `mean_molecular_weight` - Mean molecular weight μ
    /// * `adiabatic_index` - Adiabatic index γ (default 5/3 for monatomic gas)
    /// 
    /// # Returns
    /// Sound crossing time in seconds
    pub fn calculate_sound_crossing_time(
        &self,
        length_scale: f64,
        temperature: f64,
        mean_molecular_weight: f64,
        adiabatic_index: f64,
    ) -> f64 {
        let k_b = self.constants.k_b;
        let m_h = self.constants.m_p;
        
        // Sound speed
        let sound_speed = ((adiabatic_index * k_b * temperature) / (mean_molecular_weight * m_h)).sqrt();
        
        // Sound crossing time
        length_scale / sound_speed
    }

    /// Perform comprehensive Jeans instability analysis
    /// 
    /// # Arguments
    /// * `temperature` - Gas temperature (K)
    /// * `density` - Gas density (kg/m³)
    /// * `mean_molecular_weight` - Mean molecular weight μ
    /// * `mass` - Mass of the gas region (kg)
    /// * `length_scale` - Characteristic length scale (m)
    /// * `turbulent_velocity` - Turbulent velocity dispersion (m/s)
    /// 
    /// # Returns
    /// Complete Jeans analysis results
    pub fn analyze_jeans_instability(
        &self,
        temperature: f64,
        density: f64,
        mean_molecular_weight: f64,
        mass: f64,
        length_scale: f64,
        turbulent_velocity: f64,
    ) -> JeansAnalysis {
        // Calculate basic Jeans parameters
        let jeans_mass = self.calculate_jeans_mass(temperature, density, mean_molecular_weight);
        let jeans_length = self.calculate_jeans_length(temperature, density, mean_molecular_weight);
        let jeans_density = self.calculate_jeans_density(temperature, length_scale, mean_molecular_weight);
        
        // Calculate timescales
        let free_fall_time = self.calculate_free_fall_time(density);
        let sound_crossing_time = self.calculate_sound_crossing_time(
            length_scale,
            temperature,
            mean_molecular_weight,
            5.0 / 3.0, // Adiabatic index for monatomic gas
        );
        
        // Calculate instability parameter
        let instability_parameter = mass / jeans_mass;
        
        // Determine collapse mode
        let collapse_mode = self.determine_collapse_mode(
            instability_parameter,
            density,
            jeans_density,
            turbulent_velocity,
            temperature,
            mean_molecular_weight,
        );
        
        JeansAnalysis {
            jeans_mass,
            jeans_length,
            jeans_density,
            free_fall_time,
            sound_crossing_time,
            instability_parameter,
            collapse_mode,
        }
    }

    /// Determine the mode of gravitational collapse
    fn determine_collapse_mode(
        &self,
        instability_parameter: f64,
        density: f64,
        jeans_density: f64,
        turbulent_velocity: f64,
        temperature: f64,
        mean_molecular_weight: f64,
    ) -> CollapseMode {
        // If M < M_J, the region is stable
        if instability_parameter < 1.0 {
            return CollapseMode::Stable;
        }
        
        // Calculate sound speed for comparison
        let k_b = self.constants.k_b;
        let m_h = self.constants.m_p;
        let sound_speed = ((5.0 / 3.0) * k_b * temperature / (mean_molecular_weight * m_h)).sqrt();
        
        // If turbulent velocity is much larger than sound speed, expect turbulent fragmentation
        if turbulent_velocity > 2.0 * sound_speed {
            return CollapseMode::TurbulentFragmentation;
        }
        
        // If density is much higher than Jeans density, expect fragmentation
        if density > 10.0 * jeans_density {
            return CollapseMode::Fragmentation;
        }
        
        // Otherwise, expect monolithic collapse
        CollapseMode::MonolithicCollapse
    }

    /// Check if a gas region should form a sink particle (protostar)
    /// 
    /// # Arguments
    /// * `analysis` - Jeans instability analysis results
    /// * `mass` - Mass of the gas region (kg)
    /// 
    /// # Returns
    /// True if sink particle formation is appropriate
    pub fn should_form_sink_particle(&self, analysis: &JeansAnalysis, mass: f64) -> bool {
        // Must be unstable
        if analysis.instability_parameter < 1.0 {
            return false;
        }
        
        // Mass must be within sink particle limits
        let mass_solar = mass / self.constants.m_sun;
        if mass_solar < self.min_sink_mass || mass_solar > self.max_sink_mass {
            return false;
        }
        
        // Must be collapsing (not fragmenting)
        matches!(analysis.collapse_mode, CollapseMode::MonolithicCollapse)
    }

    /// Calculate the accretion rate for a sink particle
    /// 
    /// Uses Bondi-Hoyle accretion formula:
    /// Ṁ = 4π G² M² ρ / (c_s³ + v³)^(3/2)
    /// 
    /// # Arguments
    /// * `sink_mass` - Mass of the sink particle (kg)
    /// * `gas_density` - Surrounding gas density (kg/m³)
    /// * `sound_speed` - Gas sound speed (m/s)
    /// * `relative_velocity` - Relative velocity between sink and gas (m/s)
    /// 
    /// # Returns
    /// Accretion rate in kg/s
    pub fn calculate_accretion_rate(
        &self,
        sink_mass: f64,
        gas_density: f64,
        sound_speed: f64,
        relative_velocity: f64,
    ) -> f64 {
        let g = self.constants.g;
        
        // Bondi-Hoyle accretion rate
        let numerator = 4.0 * std::f64::consts::PI * g.powi(2) * sink_mass.powi(2) * gas_density;
        let denominator = (sound_speed.powi(3) + relative_velocity.powi(3)).powf(1.5);
        
        numerator / denominator
    }

    /// Find gas particles within accretion radius of a sink particle
    /// 
    /// # Arguments
    /// * `sink_position` - Position of the sink particle
    /// * `gas_particles` - List of gas particles to check
    /// 
    /// # Returns
    /// Indices of gas particles within accretion radius
    pub fn find_accretable_particles(
        &self,
        sink_position: Vector3<f64>,
        gas_particles: &[Particle],
    ) -> Vec<usize> {
        gas_particles
            .iter()
            .enumerate()
            .filter(|(_, particle)| {
                // Only consider gas particles
                matches!(particle.particle_type, ParticleType::Hydrogen | ParticleType::Helium)
            })
            .filter(|(_, particle)| {
                // Check if within accretion radius
                let distance = (particle.position - sink_position).norm();
                distance <= self.sink_accretion_radius
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// Process gravitational collapse and sink particle formation
    /// 
    /// # Arguments
    /// * `particles` - All particles in the simulation
    /// * `time_step` - Current time step (s)
    /// 
    /// # Returns
    /// Modified particle list with new sink particles
    pub fn process_gravitational_collapse(
        &mut self,
        particles: &mut Vec<Particle>,
        time_step: f64,
    ) -> Result<()> {
        // Group gas particles by proximity for collapse analysis
        let gas_particles: Vec<_> = particles
            .iter()
            .filter(|p| matches!(p.particle_type, ParticleType::Hydrogen | ParticleType::Helium))
            .cloned()
            .collect();
        
        if gas_particles.is_empty() {
            return Ok(());
        }
        
        // Analyze each gas particle for collapse conditions
        let mut new_sink_particles = Vec::new();
        
        for (i, gas_particle) in gas_particles.iter().enumerate() {
            // Estimate local gas properties around this particle
            let local_properties = self.estimate_local_properties(&gas_particles, i);
            
            // Perform Jeans analysis
            let analysis = self.analyze_jeans_instability(
                local_properties.temperature,
                local_properties.density,
                local_properties.mean_molecular_weight,
                local_properties.mass,
                local_properties.length_scale,
                local_properties.turbulent_velocity,
            );
            
            // Check if sink particle formation is appropriate
            if self.should_form_sink_particle(&analysis, local_properties.mass) {
                // Create sink particle
                let sink_particle = self.create_sink_particle(
                    gas_particle.position,
                    local_properties.mass,
                    &analysis,
                );
                new_sink_particles.push(sink_particle);
            }
        }
        
        // Add new sink particles to the simulation
        particles.extend(new_sink_particles);
        
        Ok(())
    }

    /// Estimate local gas properties around a particle
    fn estimate_local_properties(
        &self,
        gas_particles: &[Particle],
        particle_index: usize,
    ) -> LocalGasProperties {
        let particle = &gas_particles[particle_index];
        let search_radius = 1e12; // 1 AU search radius
        
        // Find nearby particles
        let nearby_particles: Vec<_> = gas_particles
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                if *i == particle_index {
                    return false;
                }
                let distance = (p.position - particle.position).norm();
                distance <= search_radius
            })
            .collect();
        
        if nearby_particles.is_empty() {
            // Use particle's own properties
            return LocalGasProperties {
                temperature: 100.0, // Default temperature
                density: 1e-20,     // Default density
                mean_molecular_weight: 2.0, // Default for H₂
                mass: particle.mass,
                length_scale: search_radius,
                turbulent_velocity: 1000.0, // Default turbulent velocity
            };
        }
        
        // Calculate average properties
        let total_mass: f64 = nearby_particles.iter().map(|(_, p)| p.mass).sum();
        let avg_velocity = nearby_particles.iter().map(|(_, p)| p.velocity.norm()).sum::<f64>() / nearby_particles.len() as f64;
        
        // Estimate density from particle spacing
        let volume = (4.0 / 3.0) * std::f64::consts::PI * search_radius.powi(3);
        let density = total_mass / volume;
        
        LocalGasProperties {
            temperature: 100.0, // Default temperature for now
            density,
            mean_molecular_weight: 2.0, // Assume molecular hydrogen
            mass: total_mass,
            length_scale: search_radius,
            turbulent_velocity: avg_velocity,
        }
    }

    /// Create a sink particle from collapsing gas
    fn create_sink_particle(
        &self,
        position: Vector3<f64>,
        mass: f64,
        analysis: &JeansAnalysis,
    ) -> Particle {
        Particle::new(ParticleType::DarkMatter, mass, position)
    }
}

/// Local gas properties for Jeans analysis
#[derive(Debug, Clone)]
struct LocalGasProperties {
    temperature: f64,
    density: f64,
    mean_molecular_weight: f64,
    mass: f64,
    length_scale: f64,
    turbulent_velocity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jeans_mass_calculation() {
        let solver = JeansInstabilitySolver::new();
        
        // Test with typical molecular cloud conditions
        let temperature = 10.0; // 10 K
        let density = 1e-18;    // 1e-18 kg/m³
        let mean_molecular_weight = 2.0; // H₂
        
        let jeans_mass = solver.calculate_jeans_mass(temperature, density, mean_molecular_weight);
        
        // Jeans mass should be positive and reasonable for molecular clouds
        assert!(jeans_mass > 0.0);
        assert!(jeans_mass < 1e30); // Should be less than 1000 M☉
        assert!(jeans_mass.is_finite());
    }

    #[test]
    fn test_jeans_length_calculation() {
        let solver = JeansInstabilitySolver::new();
        
        let temperature = 10.0;
        let density = 1e-18;
        let mean_molecular_weight = 2.0;
        
        let jeans_length = solver.calculate_jeans_length(temperature, density, mean_molecular_weight);
        
        // Jeans length should be positive and reasonable
        assert!(jeans_length > 0.0);
        assert!(jeans_length < 1e16); // Should be less than 1 pc
        assert!(jeans_length.is_finite());
    }

    #[test]
    fn test_free_fall_time() {
        let solver = JeansInstabilitySolver::new();
        
        let density = 1e-18;
        let free_fall_time = solver.calculate_free_fall_time(density);
        
        // Free-fall time should be positive and reasonable
        assert!(free_fall_time > 0.0);
        assert!(free_fall_time < 1e15); // Should be less than 30 Myr
        assert!(free_fall_time.is_finite());
    }

    #[test]
    fn test_comprehensive_jeans_analysis() {
        let solver = JeansInstabilitySolver::new();
        
        let analysis = solver.analyze_jeans_instability(
            10.0,   // temperature
            1e-18,  // density
            2.0,    // mean_molecular_weight
            1e30,   // mass (1000 M☉)
            1e15,   // length_scale (0.03 pc)
            1000.0, // turbulent_velocity
        );
        
        // Analysis should be complete and reasonable
        assert!(analysis.jeans_mass > 0.0);
        assert!(analysis.jeans_length > 0.0);
        assert!(analysis.free_fall_time > 0.0);
        assert!(analysis.sound_crossing_time > 0.0);
        assert!(analysis.instability_parameter > 0.0);
        
        // All values should be finite
        assert!(analysis.jeans_mass.is_finite());
        assert!(analysis.jeans_length.is_finite());
        assert!(analysis.free_fall_time.is_finite());
        assert!(analysis.sound_crossing_time.is_finite());
        assert!(analysis.instability_parameter.is_finite());
    }

    #[test]
    fn test_collapse_mode_determination() {
        let solver = JeansInstabilitySolver::new();
        
        // Test stable case (M < M_J)
        let analysis = solver.analyze_jeans_instability(
            100.0,  // high temperature
            1e-20,  // low density
            2.0,    // mean_molecular_weight
            1e25,   // small mass
            1e15,   // length_scale
            100.0,  // low turbulent velocity
        );
        assert_eq!(analysis.collapse_mode, CollapseMode::Stable);
        
        // Test unstable case (M > M_J)
        let analysis = solver.analyze_jeans_instability(
            10.0,   // low temperature
            1e-16,  // high density
            2.0,    // mean_molecular_weight
            1e30,   // large mass
            1e15,   // length_scale
            100.0,  // low turbulent velocity
        );
        assert_ne!(analysis.collapse_mode, CollapseMode::Stable);
    }

    #[test]
    fn test_sink_particle_formation() {
        let solver = JeansInstabilitySolver::new();
        
        // Create analysis for unstable case
        let analysis = solver.analyze_jeans_instability(
            10.0,   // temperature
            1e-16,  // density
            2.0,    // mean_molecular_weight
            1e30,   // mass (1000 M☉)
            1e15,   // length_scale
            100.0,  // turbulent_velocity
        );
        
        // Should form sink particle for unstable case
        let should_form = solver.should_form_sink_particle(&analysis, 1e30);
        assert!(should_form);
        
        // Should not form sink particle for stable case
        let stable_analysis = solver.analyze_jeans_instability(
            100.0,  // high temperature
            1e-20,  // low density
            2.0,    // mean_molecular_weight
            1e25,   // small mass
            1e15,   // length_scale
            100.0,  // turbulent velocity
        );
        let should_not_form = solver.should_form_sink_particle(&stable_analysis, 1e25);
        assert!(!should_not_form);
    }

    #[test]
    fn test_accretion_rate_calculation() {
        let solver = JeansInstabilitySolver::new();
        
        let accretion_rate = solver.calculate_accretion_rate(
            1e30,   // sink mass (1000 M☉)
            1e-18,  // gas density
            1000.0, // sound speed
            100.0,  // relative velocity
        );
        
        // Accretion rate should be positive and reasonable
        assert!(accretion_rate > 0.0);
        assert!(accretion_rate < 1e25); // Should be less than 1 M☉/yr
        assert!(accretion_rate.is_finite());
    }
} 