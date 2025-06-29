//! Universe Physical State Tracking
//! 
//! Tracks the emergent physical state of the universe based on simulation conditions
//! rather than predetermined cosmic eras or hard-coded checkpoints.

use crate::storage::{AgentLineage, CelestialBody, CelestialBodyType};
use serde::{Serialize, Deserialize};

const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W⋅m−2⋅K−4
const SPEED_OF_LIGHT: f64 = 299_792_458.0;    // m/s

/// Calculate black-body energy density from temperature.
pub fn calculate_energy_density(temperature: f64) -> f64 {
    (4.0 * STEFAN_BOLTZMANN / SPEED_OF_LIGHT) * temperature.powi(4)
}

/// Emergent universe state based on physical conditions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniverseState {
    /// Current age of universe in billion years
    pub age_gyr: f64,
    
    /// Average temperature across all particles (K)
    pub mean_temperature: f64,
    
    /// Fraction of matter in stars vs free particles
    pub stellar_fraction: f64,
    
    /// Fraction of heavy elements (metallicity)
    pub metallicity: f64,
    
    /// Number of habitable environments
    pub habitable_count: usize,
    
    /// Complexity metric for most advanced life forms
    pub max_complexity: f64,
    
    /// Energy density of the universe (J/m³)
    pub energy_density: f64,
    
    /// Hubble expansion rate
    pub hubble_constant: f64,
    
    // Additional fields needed by the simulation
    /// Average technology level across all civilizations
    pub average_tech_level: f64,
    
    /// Total stellar mass in the universe (kg)
    pub total_stellar_mass: f64,
    
    /// Dark energy density (J/m³)
    pub dark_energy_density: f64,
    
    /// Dark matter density (kg/m³)
    pub dark_matter_density: f64,
    
    /// Cosmic ray flux (particles/m²/s)
    pub cosmic_ray_flux: f64,
    
    /// Gravitational wave strain amplitude
    pub gravitational_wave_strain: f64,
    
    /// Total mass of the universe (kg)
    pub total_mass: f64,
    
    /// Iron abundance (mass fraction)
    pub iron_abundance: f64,
    
    /// Carbon abundance (mass fraction)
    pub carbon_abundance: f64,
    
    /// Oxygen abundance (mass fraction)
    pub oxygen_abundance: f64,
    
    /// Nitrogen abundance (mass fraction)
    pub nitrogen_abundance: f64,
}

impl UniverseState {
    /// Create initial universe state (Big Bang conditions)
    pub fn initial() -> Self {
        Self {
            age_gyr: 0.0,
            mean_temperature: 1e12, // Very hot initially
            stellar_fraction: 0.0,  // No stars yet
            metallicity: 0.0,       // Only H and He
            habitable_count: 0,     // No habitable zones
            max_complexity: 0.0,    // No complex structures
            energy_density: 1e20,   // Extremely dense
            hubble_constant: 100.0, // Fast initial expansion
            average_tech_level: 0.0,
            total_stellar_mass: 0.0,
            dark_energy_density: 0.0,
            dark_matter_density: 0.0,
            cosmic_ray_flux: 0.0,
            gravitational_wave_strain: 0.0,
            total_mass: 1e50, // Initial universe mass
            iron_abundance: 0.0,
            carbon_abundance: 0.0,
            oxygen_abundance: 0.0,
            nitrogen_abundance: 0.0,
        }
    }
    
    /// Update state based on current simulation measurements
    pub fn update_from_simulation(&mut self, 
                                  current_tick: u64, 
                                  tick_span_years: f64,
                                  particles: &[physics_engine::types::PhysicsState],
                                  celestial_bodies: &[CelestialBody],
                                  lineages: &[AgentLineage]) {
        
        // Update age
        self.age_gyr = (current_tick as f64 * tick_span_years) / 1e9;
        
        // Calculate mean temperature from particle physics
        if !particles.is_empty() {
            self.mean_temperature = particles.iter()
                .map(|p| p.temperature)
                .sum::<f64>() / particles.len() as f64;
        }
        
        // Calculate stellar fraction
        let star_count = celestial_bodies.iter()
            .filter(|b| matches!(b.body_type, CelestialBodyType::Star))
            .count();
        let total_matter_objects = celestial_bodies.len().max(1);
        self.stellar_fraction = star_count as f64 / total_matter_objects as f64;
        
        // Calculate metallicity from heavy element abundance
        self.metallicity = self.calculate_heavy_element_fraction(celestial_bodies);
        
        // Count habitable environments
        self.habitable_count = celestial_bodies.iter()
            .filter(|b| self.is_potentially_habitable(b))
            .count();
        
        // Calculate maximum biological/technological complexity
        if !lineages.is_empty() {
            self.max_complexity = lineages.iter()
                .map(|l| l.tech_level + l.sentience_level)
                .fold(0.0, f64::max);
        }
        
        // Update energy density based on particle energies
        if !particles.is_empty() {
            let total_energy: f64 = particles.iter()
                .map(|p| {
                    // Calculate kinetic energy: 0.5 * m * v^2
                    let kinetic = 0.5 * p.mass * p.velocity.magnitude_squared();
                    // Use mass-energy: E = mc^2
                    let rest_mass = p.mass * 299_792_458.0_f64.powi(2);
                    kinetic + rest_mass
                })
                .sum();
            // Assume some volume scaling with age
            let volume = (self.age_gyr + 0.1).powi(3) * 1e50; // Rough expansion
            self.energy_density = total_energy / volume;
        }
        
        // Hubble constant decreases with time due to expansion
        self.hubble_constant = 100.0 / (1.0 + self.age_gyr * 0.1);
    }
    
    /// Calculate heavy element fraction from stellar composition
    fn calculate_heavy_element_fraction(&self, bodies: &[CelestialBody]) -> f64 {
        if bodies.is_empty() {
            return 0.0;
        }
        
        let total_heavy_elements: f64 = bodies.iter()
            .map(|b| {
                // Sum elements heavier than helium (Z > 2)
                let mut heavy_element_mass = 0.0;
                for z in 3..118 { // From lithium (Z=3) to maximum
                    heavy_element_mass += b.composition.get_abundance(z) as f64;
                }
                heavy_element_mass
            })
            .sum();
            
        let total_mass: f64 = bodies.iter()
            .map(|b| b.mass)
            .sum();
            
        if total_mass > 0.0 {
            total_heavy_elements / total_mass
        } else {
            0.0
        }
    }
    
    /// Check if a celestial body could support life based on physics
    fn is_potentially_habitable(&self, body: &CelestialBody) -> bool {
        matches!(body.body_type, CelestialBodyType::Planet) &&
        body.temperature > 273.0 && // Above freezing
        body.temperature < 373.0 && // Below boiling
        body.mass > 0.1 &&          // Minimum mass for atmosphere retention
        body.age > 1e9              // Minimum time for chemical evolution
    }
    
    /// Get a natural description of current universe state
    pub fn description(&self) -> String {
        match self {
            // Very early universe - hot, dense, no structure
            s if s.age_gyr < 0.001 => {
                format!("Universe age {:.3} Myr: Primordial plasma at {:.0}K. No stable structures yet.",
                       s.age_gyr * 1000.0, s.mean_temperature)
            },
            // Biological complexity
            s if s.max_complexity < 10.0 => {
                format!("Universe age {:.2} Gyr: Life complexity {:.1}. Biological evolution developing.",
                       s.age_gyr, s.max_complexity)
            },
            // First structures forming
            s if s.stellar_fraction < 0.01 => {
                format!("Universe age {:.2} Gyr: Gas clouds cooling to {:.0}K. First gravitational collapse beginning.",
                       s.age_gyr, s.mean_temperature)
            },
            // Active star formation
            s if s.stellar_fraction < 0.1 && s.metallicity < 0.01 => {
                format!("Universe age {:.2} Gyr: {:.1}% matter in stars. Population III stars forging first heavy elements.",
                       s.age_gyr, s.stellar_fraction * 100.0)
            },
            // Mature stellar populations
            s if s.metallicity < 0.02 => {
                format!("Universe age {:.2} Gyr: {:.1}% stellar, Z={:.4}. Second-generation stars with rocky cores forming.",
                       s.age_gyr, s.stellar_fraction * 100.0, s.metallicity)
            },
            // Complex chemistry possible
            s if s.habitable_count == 0 => {
                format!("Universe age {:.2} Gyr: {:.1}% stellar, Z={:.4}. Complex chemistry possible but no habitable zones detected.",
                       s.age_gyr, s.stellar_fraction * 100.0, s.metallicity)
            },
            // Life emerging
            s if s.max_complexity < 1.0 => {
                format!("Universe age {:.2} Gyr: {} habitable environments. Basic chemical evolution in progress.",
                       s.age_gyr, s.habitable_count)
            },
            // Intelligence emerging
            s if s.max_complexity < 100.0 => {
                format!("Universe age {:.2} Gyr: Intelligence level {:.1}. Technology and self-modification beginning.",
                       s.age_gyr, s.max_complexity)
            },
            // Advanced civilization
            _ => {
                format!("Universe age {:.2} Gyr: Advanced civilizations (complexity {:.1}) reshaping matter and energy.",
                       self.age_gyr, self.max_complexity)
            },
        }
    }
    
    /// Check if universe conditions allow star formation
    pub fn allows_star_formation(&self) -> bool {
        // Star formation possible when:
        // - Temperature low enough for gas to collapse
        // - Not too late in universe evolution (gas exhausted)
        self.mean_temperature < 50000.0 && 
        self.stellar_fraction < 0.3 &&
        self.age_gyr < 20.0
    }
    
    /// Get star formation rate based on physical conditions
    pub fn star_formation_rate(&self) -> f64 {
        if !self.allows_star_formation() {
            return 0.0;
        }
        
        // Rate depends on available gas and cooling efficiency
        let gas_fraction = 1.0 - self.stellar_fraction;
        let cooling_efficiency = 1.0 / (1.0 + self.mean_temperature / 10000.0);
        let age_factor = (-self.age_gyr / 10.0).exp(); // Exponential decline
        
        0.01 * gas_fraction * cooling_efficiency * age_factor
    }
    
    /// Check if conditions allow planet formation
    pub fn allows_planet_formation(&self) -> bool {
        // Planets need heavy elements from stellar nucleosynthesis
        self.metallicity > 0.001 && self.stellar_fraction > 0.01
    }
    
    /// Check if life emergence is possible
    pub fn allows_life_emergence(&self) -> bool {
        // Life needs complex chemistry and stable environments
        self.metallicity > 0.01 && self.habitable_count > 0
    }
}

/// Record of a significant physical transition in the universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalTransition {
    pub tick: u64,
    pub age_gyr: f64,
    pub transition_type: TransitionType,
    pub description: String,
    pub physical_parameters: Vec<(String, f64)>,
    pub timestamp: f64,
    pub temperature: f64,
    pub energy_density: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    /// First stable atoms formed
    Recombination,
    /// First gravitational collapse
    FirstCollapse,
    /// First stars ignited
    FirstStars,
    /// First heavy elements created
    FirstMetals,
    /// First planets formed
    FirstPlanets,
    /// First life detected
    FirstLife,
    /// First intelligence detected
    FirstIntelligence,
    /// Major technological milestone
    TechBreakthrough,
    /// Cosmic era transition
    CosmicEra,
    /// Temperature-based transition
    Temperature,
    /// Energy density transition
    EnergyDensity,
}

impl PhysicalTransition {
    pub fn new(tick: u64, age_gyr: f64, transition_type: TransitionType, 
               description: String, parameters: Vec<(String, f64)>) -> Self {
        Self {
            tick,
            age_gyr,
            transition_type,
            description,
            physical_parameters: parameters,
            timestamp: tick as f64,
            temperature: 0.0,
            energy_density: 0.0,
        }
    }
    
    pub fn new_with_physics(tick: u64, age_gyr: f64, transition_type: TransitionType, 
                           description: String, parameters: Vec<(String, f64)>,
                           timestamp: f64, temperature: f64, energy_density: f64) -> Self {
        Self {
            tick,
            age_gyr,
            transition_type,
            description,
            physical_parameters: parameters,
            timestamp,
            temperature,
            energy_density,
        }
    }
}

pub type CosmicEra = UniverseState;

/// Determine the current cosmic era based on universe age
pub fn determine_cosmic_era(age_gyr: f64) -> UniverseState {
    let mut state = UniverseState::initial();
    state.age_gyr = age_gyr;
    
    // Set physical parameters based on cosmic age
    match age_gyr {
        // Planck epoch to inflation (< 1e-32 Gyr)
        age if age < 1e-23 => {
            state.mean_temperature = 1e32;
            state.energy_density = 1e113;
            state.hubble_constant = 1e60;
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 0.0;
            state.dark_energy_density = 0.0;
            state.dark_matter_density = 0.0;
            state.cosmic_ray_flux = 0.0;
            state.gravitational_wave_strain = 1e-15;
            state.total_mass = 1e60;
            state.iron_abundance = 0.0;
            state.carbon_abundance = 0.0;
            state.oxygen_abundance = 0.0;
            state.nitrogen_abundance = 0.0;
        },
        // Electroweak epoch (1e-32 to 1e-12 Gyr)  
        age if age < 1e-3 => {
            state.mean_temperature = 1e15;
            state.energy_density = 1e50;
            state.hubble_constant = 1e20;
            // Initialize all new fields
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 0.0;
            state.dark_energy_density = 0.0;
            state.dark_matter_density = 0.0;
            state.cosmic_ray_flux = 0.0;
            state.gravitational_wave_strain = 1e-16;
            state.total_mass = 1e55;
            state.iron_abundance = 0.0;
            state.carbon_abundance = 0.0;
            state.oxygen_abundance = 0.0;
            state.nitrogen_abundance = 0.0;
        },
        // Nucleosynthesis (1e-3 to 0.02 Gyr)
        age if age < 0.02 => {
            state.mean_temperature = 1e9;
            state.energy_density = 1e20;
            state.hubble_constant = 1000.0;
            state.metallicity = 0.0001; // Trace Li, Be
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 0.0;
            state.dark_energy_density = 1e-5;
            state.dark_matter_density = 1e-15;
            state.cosmic_ray_flux = 0.0;
            state.gravitational_wave_strain = 1e-17;
            state.total_mass = 1e54;
            state.iron_abundance = 0.0;
            state.carbon_abundance = 0.0;
            state.oxygen_abundance = 0.0;
            state.nitrogen_abundance = 0.0;
        },
        // Recombination era (0.02 to 0.4 Gyr)
        age if age < 0.4 => {
            state.mean_temperature = 3000.0;
            state.energy_density = 1e-15;
            state.hubble_constant = 200.0;
            state.metallicity = 0.0001;
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 0.0;
            state.dark_energy_density = 1e-8;
            state.dark_matter_density = 1e-18;
            state.cosmic_ray_flux = 0.0;
            state.gravitational_wave_strain = 1e-18;
            state.total_mass = 1e53;
            state.iron_abundance = 0.0;
            state.carbon_abundance = 0.0;
            state.oxygen_abundance = 0.0;
            state.nitrogen_abundance = 0.0;
        },
        // Dark ages (0.4 to 1.0 Gyr)
        age if age < 1.0 => {
            state.mean_temperature = 100.0;
            state.energy_density = 1e-20;
            state.hubble_constant = 150.0;
            state.metallicity = 0.0001;
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 0.0;
            state.dark_energy_density = 1e-9;
            state.dark_matter_density = 1e-20;
            state.cosmic_ray_flux = 0.0;
            state.gravitational_wave_strain = 1e-19;
            state.total_mass = 1e53;
            state.iron_abundance = 0.0;
            state.carbon_abundance = 0.0;
            state.oxygen_abundance = 0.0;
            state.nitrogen_abundance = 0.0;
        },
        // First stars (1.0 to 3.0 Gyr)
        age if age < 3.0 => {
            state.mean_temperature = 50.0;
            state.energy_density = 1e-25;
            state.hubble_constant = 120.0;
            state.stellar_fraction = 0.01;
            state.metallicity = 0.001;
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 1e38;
            state.dark_energy_density = 2e-10;
            state.dark_matter_density = 1e-21;
            state.cosmic_ray_flux = 1e6;
            state.gravitational_wave_strain = 1e-20;
            state.total_mass = 1e53;
            state.iron_abundance = 0.0001;
            state.carbon_abundance = 0.0005;
            state.oxygen_abundance = 0.001;
            state.nitrogen_abundance = 0.0001;
        },
        // Galaxy formation (3.0 to 8.0 Gyr)
        age if age < 8.0 => {
            state.mean_temperature = 20.0;
            state.energy_density = 1e-27;
            state.hubble_constant = 90.0;
            state.stellar_fraction = 0.05;
            state.metallicity = 0.01;
            state.habitable_count = (age_gyr * 10.0) as usize;
            state.average_tech_level = 0.0;
            state.total_stellar_mass = 1e40;
            state.dark_energy_density = 4e-10;
            state.dark_matter_density = 2e-21;
            state.cosmic_ray_flux = 1e8;
            state.gravitational_wave_strain = 5e-21;
            state.total_mass = 1e53;
            state.iron_abundance = 0.0005;
            state.carbon_abundance = 0.001;
            state.oxygen_abundance = 0.005;
            state.nitrogen_abundance = 0.0002;
        },
        // Mature universe (8.0 to 13.8 Gyr)
        age if age < 13.8 => {
            state.mean_temperature = 10.0;
            state.energy_density = 1e-29;
            state.hubble_constant = 70.0;
            state.stellar_fraction = 0.1;
            state.metallicity = 0.02;
            state.habitable_count = ((age_gyr - 8.0) * 100.0) as usize;
            state.max_complexity = (age_gyr - 8.0) * 10.0;
            state.average_tech_level = (age_gyr - 8.0) * 1.0;
            state.total_stellar_mass = 1e41;
            state.dark_energy_density = 5e-10;
            state.dark_matter_density = 2.2e-21;
            state.cosmic_ray_flux = 1e9;
            state.gravitational_wave_strain = 1e-21;
            state.total_mass = 1e53;
            state.iron_abundance = 0.0008;
            state.carbon_abundance = 0.0015;
            state.oxygen_abundance = 0.008;
            state.nitrogen_abundance = 0.0003;
        },
        // Current era (> 13.8 Gyr)
        _ => {
            state.mean_temperature = 2.7; // CMB temperature
            state.energy_density = 1e-30;
            state.hubble_constant = 67.4;
            state.stellar_fraction = 0.15;
            state.metallicity = 0.025;
            state.habitable_count = (age_gyr * 50.0) as usize;
            state.max_complexity = age_gyr * 5.0;
            state.average_tech_level = age_gyr * 2.0;
            state.total_stellar_mass = 1e42; // kg
            state.dark_energy_density = 6e-10; // J/m³
            state.dark_matter_density = 2.3e-21; // kg/m³
            state.cosmic_ray_flux = 1e10; // particles/m²/s
            state.gravitational_wave_strain = 1e-21;
            state.total_mass = 1e53; // kg
            state.iron_abundance = 0.001;
            state.carbon_abundance = 0.002;
            state.oxygen_abundance = 0.01;
            state.nitrogen_abundance = 0.0005;
        }
    }
    
    state
}

/// Cosmological calculations based on Friedmann equations
pub mod cosmology {
    use std::f64::consts::PI;
    
    /// Standard cosmological parameters (Planck 2018)
    pub struct CosmologicalParameters {
        pub h0: f64,           // Hubble constant in km/s/Mpc
        pub omega_m: f64,      // Matter density parameter
        pub omega_lambda: f64, // Dark energy density parameter
        pub omega_r: f64,      // Radiation density parameter
        pub omega_k: f64,      // Curvature parameter
    }
    
    impl Default for CosmologicalParameters {
        fn default() -> Self {
            Self {
                h0: 67.4,           // km/s/Mpc (Planck 2018)
                omega_m: 0.315,     // Total matter
                omega_lambda: 0.685, // Dark energy
                omega_r: 9.24e-5,   // Radiation (photons + neutrinos)
                omega_k: 0.0,       // Flat universe
            }
        }
    }
    
    impl CosmologicalParameters {
        /// Calculate universe age from redshift using Friedmann equation
        pub fn universe_age_from_redshift(&self, redshift: f64) -> f64 {
            // Age = integral from z to infinity of dz' / ((1+z') * H(z'))
            // Using approximate analytical solution for ΛCDM
            let h0_si = self.h0 * 1000.0 / 3.086e22; // Convert to SI units (s^-1)
            
            // Scale factor at given redshift
            let a = 1.0 / (1.0 + redshift);
            
            // Hubble parameter as function of scale factor
            let h_a = h0_si * (self.omega_r / a.powi(4) + self.omega_m / a.powi(3) + 
                              self.omega_k / a.powi(2) + self.omega_lambda).sqrt();
            
            // Approximate age calculation (analytical for matter-dominated)
            if self.omega_lambda.abs() < 1e-6 {
                // Matter-dominated universe
                2.0 / (3.0 * h_a * a.powf(1.5))
            } else {
                // ΛCDM universe - use approximation
                let x = self.omega_lambda / self.omega_m;
                let eta = (2.0 * x.sqrt() * a.powf(1.5)).asinh();
                (2.0 / (3.0 * h0_si * self.omega_lambda.sqrt())) * eta
            }
        }
        
        /// Calculate scale factor from cosmic time
        pub fn scale_factor_from_time(&self, time_seconds: f64) -> f64 {
            let h0_si = self.h0 * 1000.0 / 3.086e22; // Convert to SI units
            
            if self.omega_lambda.abs() < 1e-6 {
                // Matter-dominated: a(t) ∝ t^(2/3)
                let t0 = 2.0 / (3.0 * h0_si); // Age at a=1
                (time_seconds / t0).powf(2.0/3.0)
            } else {
                // ΛCDM universe
                let omega_m_over_lambda = self.omega_m / self.omega_lambda;
                let h_lambda = h0_si * self.omega_lambda.sqrt();
                
                // Solve parametrically: t = (2/(3H_Λ)) * sinh^(-1)(sqrt(Ω_Λ/Ω_m) * a^(3/2))
                let x = 1.5 * h_lambda * time_seconds;
                let y = x.sinh();
                let a_cubed_half = y / omega_m_over_lambda.sqrt();
                a_cubed_half.powf(2.0/3.0)
            }
        }
        
        /// Calculate current age of universe (at z=0)
        pub fn current_universe_age(&self) -> f64 {
            self.universe_age_from_redshift(0.0)
        }
        
        /// Calculate Hubble parameter at given redshift
        pub fn hubble_parameter(&self, redshift: f64) -> f64 {
            let a = 1.0 / (1.0 + redshift);
            self.h0 * (self.omega_r / a.powi(4) + self.omega_m / a.powi(3) + 
                      self.omega_k / a.powi(2) + self.omega_lambda).sqrt()
        }
        
        /// Calculate critical density at given redshift
        pub fn critical_density(&self, redshift: f64) -> f64 {
            let h_z = self.hubble_parameter(redshift) * 1000.0 / 3.086e22; // SI units
            3.0 * h_z.powi(2) / (8.0 * PI * 6.67430e-11) // ρ_c = 3H²/(8πG)
        }
        
        /// Calculate lookback time to given redshift
        pub fn lookback_time(&self, redshift: f64) -> f64 {
            let current_age = self.current_universe_age();
            let age_at_z = self.universe_age_from_redshift(redshift);
            current_age - age_at_z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_universe_state() {
        let state = UniverseState::initial();
        assert_eq!(state.age_gyr, 0.0);
        assert!(state.mean_temperature > 1e10);
        assert_eq!(state.stellar_fraction, 0.0);
        assert_eq!(state.metallicity, 0.0);
    }

    #[test]
    fn test_star_formation_conditions() {
        let mut state = UniverseState::initial();
        
        // Initially too hot for star formation
        assert!(!state.allows_star_formation());
        
        // Cool down - should allow star formation
        state.mean_temperature = 10000.0;
        state.stellar_fraction = 0.01;
        assert!(state.allows_star_formation());
        
        // Too much matter already in stars
        state.stellar_fraction = 0.4;
        assert!(!state.allows_star_formation());
    }

    #[test]
    fn test_physics_driven_descriptions() {
        let mut state = UniverseState::initial();
        state.age_gyr = 13.8;
        state.habitable_count = 10;
        state.max_complexity = 5.5;

        assert!(state.description().contains("Life complexity 5.5"));
    }
}