//! Cosmological SPH (Smoothed Particle Hydrodynamics) for gas dynamics
//! 
//! Implements advanced SPH methods for cosmological gas physics including:
//! - Cooling and heating processes (atomic, molecular, Compton)
//! - Star formation and feedback mechanisms
//! - Chemical enrichment and metallicity evolution
//! - Jeans instability and gas collapse
//! - Multi-phase gas physics
//!
//! Based on:
//! - Number Analytics: Ultimate Guide to Cosmological Simulations
//! - Number Analytics: Mastering Cosmological Simulations

use nalgebra::Vector3;
use anyhow::Result;
use std::collections::HashMap;

use crate::cosmology::{CosmologicalParameters, CosmologicalParticle, CosmologicalParticleType};
use crate::particle_types::ParticleType;
use crate::sph::{SphParticle, SphSolver, SphKernel, KernelType};
use crate::FundamentalParticle;

/// Cosmological SPH particle with enhanced gas physics
#[derive(Debug, Clone)]
pub struct CosmologicalSphParticle {
    /// Base SPH particle
    pub sph_particle: SphParticle,
    /// Cosmological particle properties
    pub cosmological_particle: CosmologicalParticle,
    /// Gas temperature (K)
    pub temperature: f64,
    /// Gas metallicity (solar metallicity units)
    pub metallicity: f64,
    /// Hydrogen fraction
    pub hydrogen_fraction: f64,
    /// Helium fraction
    pub helium_fraction: f64,
    /// Electron fraction
    pub electron_fraction: f64,
    /// Cooling rate (erg/s/g)
    pub cooling_rate: f64,
    /// Heating rate (erg/s/g)
    pub heating_rate: f64,
    /// Star formation rate (solar masses/year)
    pub star_formation_rate: f64,
    /// Gas pressure (Pa)
    pub pressure: f64,
    /// Entropy (keV cm²)
    pub entropy: f64,
    /// Cooling time (years)
    pub cooling_time: f64,
    /// Free-fall time (years)
    pub free_fall_time: f64,
    /// Jeans mass (solar masses)
    pub jeans_mass: f64,
    /// Jeans length (Mpc/h)
    pub jeans_length: f64,
}

impl CosmologicalSphParticle {
    pub fn new(
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        mass: f64,
        smoothing_length: f64,
    ) -> Self {
        let fundamental_particle = FundamentalParticle::new(
            ParticleType::Hydrogen, // Use Hydrogen for gas particles
            mass,
            position,
        );
        
        let sph_particle = SphParticle::new(fundamental_particle, smoothing_length, 0);
        let cosmological_particle = CosmologicalParticle::new(
            position,
            velocity,
            mass,
            CosmologicalParticleType::Gas,
        );
        
        Self {
            sph_particle,
            cosmological_particle,
            temperature: 1e4, // 10,000 K default
            metallicity: 0.0, // Primordial composition
            hydrogen_fraction: 0.76, // Primordial hydrogen
            helium_fraction: 0.24, // Primordial helium
            electron_fraction: 0.0, // Neutral gas
            cooling_rate: 0.0,
            heating_rate: 0.0,
            star_formation_rate: 0.0,
            pressure: 0.0,
            entropy: 0.0,
            cooling_time: f64::INFINITY,
            free_fall_time: f64::INFINITY,
            jeans_mass: 0.0,
            jeans_length: 0.0,
        }
    }

    /// Update gas properties from equation of state
    pub fn update_gas_properties(&mut self, cosmological_params: &CosmologicalParameters) {
        // Calculate pressure from ideal gas equation of state
        let boltzmann_k = 1.380649e-23; // J/K
        let proton_mass = 1.673e-27; // kg
        let mean_molecular_weight = 1.0 / (self.hydrogen_fraction + 0.25 * self.helium_fraction);
        
        self.pressure = self.sph_particle.density * boltzmann_k * self.temperature / 
                       (mean_molecular_weight * proton_mass);
        
        // Calculate entropy
        let gamma = self.sph_particle.gamma;
        self.entropy = self.pressure / self.sph_particle.density.powf(gamma);
        
        // Calculate Jeans properties
        self.calculate_jeans_properties(cosmological_params);
        
        // Calculate cooling and heating times
        self.calculate_cooling_heating_times();
    }

    /// Calculate Jeans mass and length
    fn calculate_jeans_properties(&mut self, cosmological_params: &CosmologicalParameters) {
        use crate::constants::G;
        
        let boltzmann_k = 1.380649e-23; // J/K
        let proton_mass = 1.673e-27; // kg
        let mean_molecular_weight = 1.0 / (self.hydrogen_fraction + 0.25 * self.helium_fraction);
        
        // Jeans length: λ_J = sqrt(π k_B T / (G ρ μ m_p))
        let jeans_length_squared = std::f64::consts::PI * boltzmann_k * self.temperature / 
                                 (G * self.sph_particle.density * mean_molecular_weight * proton_mass);
        self.jeans_length = jeans_length_squared.sqrt();
        
        // Jeans mass: M_J = (4π/3) ρ λ_J³
        self.jeans_mass = (4.0 * std::f64::consts::PI / 3.0) * 
                         self.sph_particle.density * self.jeans_length.powi(3);
    }

    /// Calculate cooling and heating times
    fn calculate_cooling_heating_times(&mut self) {
        // Simplified cooling time calculation
        // In practice, this would use detailed cooling tables
        
        let boltzmann_k = 1.380649e-23; // J/K
        let proton_mass = 1.673e-27; // kg
        let mean_molecular_weight = 1.0 / (self.hydrogen_fraction + 0.25 * self.helium_fraction);
        
        // Internal energy per unit mass
        let internal_energy = 1.5 * boltzmann_k * self.temperature / 
                            (mean_molecular_weight * proton_mass);
        
        // Cooling time: t_cool = u / (du/dt)
        if self.cooling_rate > 0.0 {
            self.cooling_time = internal_energy / self.cooling_rate;
        } else {
            self.cooling_time = f64::INFINITY;
        }
        
        // Free-fall time: t_ff = sqrt(3π / (32 G ρ))
        use crate::constants::G;
        self.free_fall_time = (3.0 * std::f64::consts::PI / (32.0 * G * self.sph_particle.density)).sqrt();
    }
}

/// Cosmological SPH solver with gas physics
#[derive(Debug)]
pub struct CosmologicalSphSolver {
    /// Base SPH solver
    pub sph_solver: SphSolver,
    /// Cosmological parameters
    pub cosmological_params: CosmologicalParameters,
    /// Cooling and heating rates
    pub cooling_heating: CoolingHeating,
    /// Star formation model
    pub star_formation: StarFormation,
    /// Chemical enrichment
    pub chemical_enrichment: ChemicalEnrichment,
    /// Feedback mechanisms
    pub feedback: Feedback,
}

/// Cooling and heating processes
#[derive(Debug)]
pub struct CoolingHeating {
    /// Atomic cooling enabled
    pub atomic_cooling: bool,
    /// Molecular cooling enabled
    pub molecular_cooling: bool,
    /// Compton cooling enabled
    pub compton_cooling: bool,
    /// Photoionization heating enabled
    pub photoionization_heating: bool,
    /// UV background redshift
    pub uv_background_redshift: f64,
}

impl CoolingHeating {
    pub fn new() -> Self {
        Self {
            atomic_cooling: true,
            molecular_cooling: true,
            compton_cooling: true,
            photoionization_heating: true,
            uv_background_redshift: 6.0,
        }
    }

    /// Calculate cooling rate for a gas particle
    pub fn calculate_cooling_rate(&self, particle: &CosmologicalSphParticle, redshift: f64) -> f64 {
        let mut cooling_rate = 0.0;
        
        // Atomic cooling (simplified)
        if self.atomic_cooling {
            cooling_rate += self.atomic_cooling_rate(particle);
        }
        
        // Molecular cooling (simplified)
        if self.molecular_cooling {
            cooling_rate += self.molecular_cooling_rate(particle);
        }
        
        // Compton cooling
        if self.compton_cooling {
            cooling_rate += self.compton_cooling_rate(particle, redshift);
        }
        
        cooling_rate
    }

    /// Calculate heating rate for a gas particle
    pub fn calculate_heating_rate(&self, particle: &CosmologicalSphParticle, redshift: f64) -> f64 {
        let mut heating_rate = 0.0;
        
        // Photoionization heating
        if self.photoionization_heating && redshift < self.uv_background_redshift {
            heating_rate += self.photoionization_heating_rate(particle, redshift);
        }
        
        heating_rate
    }

    /// Atomic cooling rate (simplified)
    fn atomic_cooling_rate(&self, particle: &CosmologicalSphParticle) -> f64 {
        // More realistic atomic cooling curve using a piecewise power-law approximation
        // based on Sutherland & Dopita (1993) for a plasma in collisional ionization equilibrium.
        // This is a common approximation in cosmological simulations.
        let temp = particle.temperature;
        let n_h = particle.sph_particle.density / (1.673e-27 * 1e6); // Hydrogen number density in cm^-3
        
        let log_t = temp.log10();
        let log_lambda = if log_t < 4.2 {
            -21.85 + 2.0 * log_t
        } else if log_t < 5.5 {
            -3.8 - 6.0 * log_t
        } else if log_t < 6.2 {
            -17.5
        } else if log_t < 7.0 {
            -4.0 - 2.5 * log_t
        } else {
            -22.0
        };

        let lambda = 10.0f64.powf(log_lambda); // Cooling function in erg cm^3 / s
        lambda * n_h * n_h // Cooling rate in erg / cm^3 / s
    }

    /// Molecular cooling rate (simplified)
    fn molecular_cooling_rate(&self, particle: &CosmologicalSphParticle) -> f64 {
        // Simplified molecular cooling for cold gas
        if particle.temperature < 1e3 {
            let density = particle.sph_particle.density;
            let molecular_fraction = 0.01; // 1% molecular fraction
            
            1e-23 * density * density * molecular_fraction // erg/s/g
        } else {
            0.0
        }
    }

    /// Compton cooling rate
    fn compton_cooling_rate(&self, particle: &CosmologicalSphParticle, redshift: f64) -> f64 {
        // Compton cooling: Λ_Compton ∝ T * (1+z)⁴
        let temperature = particle.temperature;
        let redshift_factor = (1.0 + redshift).powi(4);
        
        1e-25 * temperature * redshift_factor // erg/s/g
    }

    /// Photoionization heating rate
    fn photoionization_heating_rate(&self, particle: &CosmologicalSphParticle, redshift: f64) -> f64 {
        // Photoionization heating: Γ ∝ (1+z)⁴
        let redshift_factor = (1.0 + redshift).powi(4);
        
        1e-26 * redshift_factor // erg/s/g
    }
}

/// Star formation model
#[derive(Debug)]
pub struct StarFormation {
    /// Star formation efficiency
    pub efficiency: f64,
    /// Density threshold for star formation
    pub density_threshold: f64,
    /// Temperature threshold for star formation
    pub temperature_threshold: f64,
    /// Minimum gas mass for star formation
    pub min_gas_mass: f64,
}

impl StarFormation {
    pub fn new() -> Self {
        Self {
            efficiency: 0.1, // 10% efficiency
            density_threshold: 1e-25, // kg/m³
            temperature_threshold: 1e4, // K
            min_gas_mass: 1e6, // 10⁶ solar masses
        }
    }

    /// Calculate star formation rate based on the Springel & Hernquist (2003) model.
    pub fn calculate_star_formation_rate(&self, particle: &CosmologicalSphParticle) -> f64 {
        // This model assumes a multi-phase ISM where star formation occurs in cold clouds
        // embedded in a hot, pressure-confining medium.
        
        if particle.sph_particle.density > self.density_threshold &&
           particle.temperature < self.temperature_threshold {
            
            // Star formation timescale, proportional to the local dynamical time.
            let t_star = 1.5 * (4.0 * std::f64::consts::PI * 6.674e-11 * particle.sph_particle.density).sqrt();
            
            // Kennicutt-Schmidt law: SFR ∝ gas_density / t_dyn
            let sfr = self.efficiency * particle.cosmological_particle.mass / t_star;
            sfr
        } else {
            0.0
        }
    }

    /// Form stars from a gas particle over a timestep dt.
    pub fn form_stars(&self, particle: &mut CosmologicalSphParticle, dt: f64) -> f64 {
        let sfr = self.calculate_star_formation_rate(particle);
        let star_mass = sfr * dt;
        
        // Update gas mass
        if star_mass > 0.0 && star_mass < particle.cosmological_particle.mass {
            particle.cosmological_particle.mass -= star_mass;
            particle.sph_particle.particle.mass = particle.cosmological_particle.mass;
            star_mass
        } else {
            0.0
        }
    }
}

/// Chemical enrichment model
#[derive(Debug)]
pub struct ChemicalEnrichment {
    /// Yield tables for different elements
    pub yields: HashMap<String, f64>,
    /// Enrichment history
    pub enrichment_history: Vec<ChemicalEvent>,
}

#[derive(Debug, Clone)]
pub struct ChemicalEvent {
    pub time: f64,
    pub element: String,
    pub mass: f64,
    pub source: String,
}

impl ChemicalEnrichment {
    pub fn new() -> Self {
        let mut yields = HashMap::new();
        yields.insert("H".to_string(), 0.0);
        yields.insert("He".to_string(), 0.0);
        yields.insert("C".to_string(), 0.01);
        yields.insert("N".to_string(), 0.005);
        yields.insert("O".to_string(), 0.02);
        yields.insert("Fe".to_string(), 0.01);
        
        Self {
            yields,
            enrichment_history: Vec::new(),
        }
    }

    /// Enrich gas with metals from star formation
    pub fn enrich_gas(&mut self, particle: &mut CosmologicalSphParticle, star_mass: f64, time: f64) {
        for (element, yield_value) in &self.yields {
            let metal_mass = star_mass * yield_value;
            if metal_mass > 0.0 {
                // Update metallicity (simplified)
                particle.metallicity += metal_mass / particle.cosmological_particle.mass;
                
                // Record enrichment event
                self.enrichment_history.push(ChemicalEvent {
                    time,
                    element: element.clone(),
                    mass: metal_mass,
                    source: "star_formation".to_string(),
                });
            }
        }
    }
}

/// Feedback mechanisms
#[derive(Debug)]
pub struct Feedback {
    /// Supernova feedback enabled
    pub supernova_feedback: bool,
    /// AGN feedback enabled
    pub agn_feedback: bool,
    /// Stellar wind feedback enabled
    pub stellar_wind_feedback: bool,
    /// Feedback energy per supernova
    pub supernova_energy: f64,
}

impl Feedback {
    pub fn new() -> Self {
        Self {
            supernova_feedback: true,
            agn_feedback: true,
            stellar_wind_feedback: true,
            supernova_energy: 1e51, // erg
        }
    }

    /// Apply supernova feedback to the surrounding gas.
    pub fn apply_supernova_feedback(&self, particles: &mut [CosmologicalSphParticle], star_forming_particle_idx: usize, star_mass: f64) {
        if !self.supernova_feedback {
            return;
        }

        let num_supernovae = star_mass / 10.0; // Assuming one 10 M_sun star per supernova
        let total_feedback_energy = num_supernovae * self.supernova_energy;

        // Distribute feedback energy thermally to neighbor particles, weighted by the SPH kernel.
        // This is a more realistic approach than just heating the star-forming particle.
        let star_former_pos = particles[star_forming_particle_idx].cosmological_particle.position;
        let mut total_weight = 0.0;
        let mut neighbors = Vec::new();
        
        // Create a kernel for evaluation
        let kernel = SphKernel::new(KernelType::CubicSpline, 3);

        for i in 0..particles.len() {
            if i == star_forming_particle_idx { continue; }
            let dist_sq = (particles[i].cosmological_particle.position - star_former_pos).norm_squared();
            let h = particles[star_forming_particle_idx].sph_particle.smoothing_length;
            if dist_sq < h*h {
                let weight = kernel.evaluate(dist_sq.sqrt(), h);
                total_weight += weight;
                neighbors.push((i, weight));
            }
        }

        if total_weight > 0.0 {
            for (neighbor_idx, weight) in neighbors {
                let energy_to_add = total_feedback_energy * (weight / total_weight);
                let boltzmann_k = 1.380649e-23;
                let proton_mass = 1.673e-27;
                let mean_molecular_weight = 1.0 / (particles[neighbor_idx].hydrogen_fraction + 0.25 * particles[neighbor_idx].helium_fraction);
                let temp_increase = energy_to_add * (2.0/3.0) * mean_molecular_weight * proton_mass / boltzmann_k;
                particles[neighbor_idx].temperature += temp_increase;
            }
        }
    }
}

impl CosmologicalSphSolver {
    pub fn new(cosmological_params: CosmologicalParameters) -> Self {
        let sph_solver = SphSolver::new_remix(3); // 3D REMIX SPH
        let cooling_heating = CoolingHeating::new();
        let star_formation = StarFormation::new();
        let chemical_enrichment = ChemicalEnrichment::new();
        let feedback = Feedback::new();
        
        Self {
            sph_solver,
            cosmological_params,
            cooling_heating,
            star_formation,
            chemical_enrichment,
            feedback,
        }
    }

    /// Evolve cosmological SPH particles for one time step
    pub fn evolve_step(&mut self, particles: &mut [CosmologicalSphParticle], dt: f64, redshift: f64) -> Result<()> {
        // Update gas properties
        for particle in particles.iter_mut() {
            particle.update_gas_properties(&self.cosmological_params);
        }
        
        // Calculate cooling and heating
        for particle in particles.iter_mut() {
            particle.cooling_rate = self.cooling_heating.calculate_cooling_rate(particle, redshift);
            particle.heating_rate = self.cooling_heating.calculate_heating_rate(particle, redshift);
        }
        
        // Apply cooling and heating
        for particle in particles.iter_mut() {
            let net_cooling = particle.cooling_rate - particle.heating_rate;
            let energy_change = net_cooling * dt;
            particle.sph_particle.internal_energy -= energy_change;
            
            // Update temperature
            let boltzmann_k = 1.380649e-23; // J/K
            let proton_mass = 1.673e-27; // kg
            let mean_molecular_weight = 1.0 / (particle.hydrogen_fraction + 0.25 * particle.helium_fraction);
            
            particle.temperature = 2.0 * particle.sph_particle.internal_energy * 
                                 (mean_molecular_weight * proton_mass) / (3.0 * boltzmann_k);
        }
        
        // Star formation
        for (i, particle) in particles.iter_mut().enumerate() {
            let star_mass = self.star_formation.form_stars(particle, dt);
            if star_mass > 0.0 {
                // Chemical enrichment with proper time evolution
                let enrichment_time = self.cosmological_params.age_universe * (1.0 - 1.0 / (1.0 + redshift));
                self.chemical_enrichment.enrich_gas(particle, star_mass, enrichment_time);
                
                // Feedback
                self.feedback.apply_supernova_feedback(particles, i, star_mass);
            }
        }
        
        // Filter gas particles for hydrodynamic evolution
        let mut gas_particles: Vec<_> = particles
            .iter_mut()
            .filter(|p| p.cosmological_particle.particle_type == CosmologicalParticleType::Gas)
            .collect();
        
        // SPH evolution
        let mut sph_particles: Vec<SphParticle> = gas_particles.iter_mut().map(|p| p.sph_particle.clone()).collect();
        self.sph_solver.integrate_step(&mut sph_particles, dt)?;
        
        // Update the SPH particles back
        for (i, sph_particle) in sph_particles.into_iter().enumerate() {
            gas_particles[i].sph_particle = sph_particle;
        }
        
        // Update cosmological properties
        for particle in gas_particles.iter_mut() {
            particle.cosmological_particle.position = particle.sph_particle.particle.position;
            particle.cosmological_particle.velocity = particle.sph_particle.particle.velocity;
            
            // Update gas properties
            particle.update_gas_properties(&self.cosmological_params);
            
            // Apply cooling and heating
            let cooling_rate = self.cooling_heating.calculate_cooling_rate(particle, redshift);
            let heating_rate = self.cooling_heating.calculate_heating_rate(particle, redshift);
            particle.cooling_rate = cooling_rate;
            particle.heating_rate = heating_rate;
            
            // Apply star formation
            let star_mass = self.star_formation.form_stars(particle, dt);
            if star_mass > 0.0 {
                let enrichment_time = self.cosmological_params.age_universe * (1.0 - 1.0 / (1.0 + redshift));
                self.chemical_enrichment.enrich_gas(particle, star_mass, enrichment_time);
            }
        }
        
        Ok(())
    }

    /// Convert fundamental particles to cosmological SPH particles
    pub fn convert_to_cosmological_sph(&self, particles: Vec<crate::FundamentalParticle>) -> Vec<CosmologicalSphParticle> {
        particles.into_iter()
            .filter(|p| p.particle_type == crate::ParticleType::Gas)
            .map(|p| {
                let smoothing_length = 1e-3; // 1 kpc default
                CosmologicalSphParticle::new(p.position, p.velocity, p.mass, smoothing_length)
            })
            .collect()
    }

    /// Convert cosmological SPH particles back to fundamental particles
    pub fn convert_from_cosmological_sph(&self, particles: Vec<CosmologicalSphParticle>) -> Vec<crate::FundamentalParticle> {
        particles.into_iter()
            .map(|p| p.sph_particle.particle)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosmological_sph_particle_creation() {
        let position = Vector3::new(0.0, 0.0, 0.0);
        let velocity = Vector3::new(100.0, 0.0, 0.0);
        let mass = 1e10; // 10¹⁰ solar masses
        let smoothing_length = 1e-3; // 1 kpc
        
        let particle = CosmologicalSphParticle::new(position, velocity, mass, smoothing_length);
        
        assert_eq!(particle.cosmological_particle.particle_type, CosmologicalParticleType::Gas);
        assert_eq!(particle.temperature, 1e4);
        assert_eq!(particle.metallicity, 0.0);
    }

    #[test]
    fn test_cosmological_sph_solver_creation() {
        let params = CosmologicalParameters::default();
        let solver = CosmologicalSphSolver::new(params);
        
        assert!(solver.cooling_heating.atomic_cooling);
        assert!(solver.star_formation.efficiency > 0.0);
    }
} 