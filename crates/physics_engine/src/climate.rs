//! Climate & Ocean Solver
//! 
//! Implements 0-D energy-balance + CO₂ cycle + Budyko ice-albedo feedback
//! and ocean mixing box model

use anyhow::Result;
use crate::{PhysicsState, PhysicsConstants};

/// Climate state information
#[derive(Debug, Clone)]
pub struct ClimateState {
    pub global_temperature: f64,    // K
    pub surface_temperature: f64,   // K
    pub ocean_temperature: f64,     // K
    pub ice_coverage: f64,          // fraction (0-1)
    pub albedo: f64,               // fraction (0-1)
    pub co2_concentration: f64,     // ppm
    pub solar_constant: f64,        // W/m²
    pub greenhouse_effect: f64,     // W/m²
}

/// Ocean box model parameters
#[derive(Debug, Clone)]
pub struct OceanBox {
    pub temperature: f64,          // K
    pub salinity: f64,            // psu (practical salinity units)
    pub density: f64,             // kg/m³
    pub circulation_rate: f64,     // m³/s
    pub heat_capacity: f64,        // J/(kg⋅K)
    pub depth: f64,               // m
}

/// Climate & ocean solver
pub struct ClimateSolver {
    pub climate_state: ClimateState,
    pub ocean_boxes: Vec<OceanBox>,
    pub stefan_boltzmann: f64,     // W⋅m⁻²⋅K⁻⁴
    pub earth_radius: f64,         // m
    pub ocean_area: f64,          // m²
    pub land_area: f64,           // m²
}

impl ClimateSolver {
    pub fn new() -> Self {
        Self {
            climate_state: ClimateState {
                global_temperature: 288.0,  // K (15°C)
                surface_temperature: 288.0,
                ocean_temperature: 277.0,   // K (4°C deep ocean)
                ice_coverage: 0.1,          // 10% ice coverage
                albedo: 0.3,               // Earth's albedo
                co2_concentration: 280.0,   // ppm pre-industrial
                solar_constant: 1361.0,     // W/m² solar constant
                greenhouse_effect: 33.0,    // K greenhouse warming
            },
            ocean_boxes: Vec::new(),
            stefan_boltzmann: 5.670374419e-8, // W⋅m⁻²⋅K⁻⁴
            earth_radius: 6.371e6,           // m
            ocean_area: 3.61e14,             // m² (71% of Earth)
            land_area: 1.49e14,              // m² (29% of Earth)
        }
    }

    /// Initialize Earth-like climate system
    pub fn init_earth_like(&mut self) {
        // Create simplified ocean boxes (surface, thermocline, deep)
        self.ocean_boxes = vec![
            OceanBox {
                temperature: 293.0,      // K (20°C surface)
                salinity: 35.0,         // psu
                density: 1025.0,        // kg/m³
                circulation_rate: 1e6,   // m³/s
                heat_capacity: 4186.0,   // J/(kg⋅K) for seawater
                depth: 100.0,           // m mixed layer
            },
            OceanBox {
                temperature: 283.0,      // K (10°C thermocline)
                salinity: 34.5,         // psu
                density: 1026.0,        // kg/m³
                circulation_rate: 1e5,   // m³/s
                heat_capacity: 4186.0,
                depth: 900.0,           // m thermocline
            },
            OceanBox {
                temperature: 277.0,      // K (4°C deep ocean)
                salinity: 34.7,         // psu
                density: 1028.0,        // kg/m³
                circulation_rate: 1e4,   // m³/s
                heat_capacity: 4186.0,
                depth: 3000.0,          // m deep ocean
            },
        ];
    }

    /// Update climate system
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Update CO₂ concentrations from particle chemistry
        self.update_co2_cycle(states)?;
        
        // Calculate energy balance
        self.update_energy_balance(constants)?;
        
        // Update ice-albedo feedback
        self.update_ice_albedo_feedback()?;
        
        // Update ocean circulation
        self.update_ocean_circulation(constants)?;
        
        // Apply climate effects to particles
        self.apply_climate_effects(states)?;
        
        Ok(())
    }

    /// Update CO₂ cycle from atmospheric chemistry
    fn update_co2_cycle(&mut self, states: &[PhysicsState]) -> Result<()> {
        // Simplified CO₂ budget from particle interactions
        let mut co2_change = 0.0;
        
        // Count carbon-containing particles (very simplified)
        let carbon_particles = states.iter()
            .filter(|s| s.mass > 1.99e-26 && s.mass < 2.01e-26) // ~12 amu (carbon)
            .count();
        
        // Estimate atmospheric CO₂ change
        let baseline_carbon = 850e15; // kg C in atmosphere at 280 ppm
        let current_carbon = carbon_particles as f64 * 1.99e-26 * 1e12; // Scale factor
        
        // Convert to ppm
        self.climate_state.co2_concentration = 280.0 * (current_carbon / baseline_carbon);
        
        // Oceanic CO₂ absorption (simplified)
        let ocean_absorption_rate = 0.3; // fraction absorbed per year
        let dt = 1.0 / (365.25 * 24.0 * 3600.0); // 1 second
        
        co2_change -= self.climate_state.co2_concentration * ocean_absorption_rate * dt;
        
        // Limit CO₂ concentration
        self.climate_state.co2_concentration = (self.climate_state.co2_concentration + co2_change)
            .max(180.0) // Minimum for photosynthesis
            .min(10000.0); // Maximum reasonable
        
        Ok(())
    }

    /// Update global energy balance
    fn update_energy_balance(&mut self, constants: &PhysicsConstants) -> Result<()> {
        // Incoming solar radiation
        let solar_area = std::f64::consts::PI * self.earth_radius * self.earth_radius;
        let incoming_solar = self.climate_state.solar_constant * solar_area * (1.0 - self.climate_state.albedo);
        
        // Outgoing thermal radiation (Stefan-Boltzmann)
        let earth_surface_area = 4.0 * std::f64::consts::PI * self.earth_radius * self.earth_radius;
        let outgoing_thermal = self.stefan_boltzmann * earth_surface_area * 
                              self.climate_state.surface_temperature.powi(4);
        
        // Greenhouse effect from CO₂
        let co2_radiative_forcing = self.calculate_co2_forcing();
        let total_greenhouse = self.climate_state.greenhouse_effect + co2_radiative_forcing;
        
        // Energy balance: incoming = outgoing + greenhouse
        let net_energy = incoming_solar - outgoing_thermal + total_greenhouse * earth_surface_area;
        
        // Update temperature based on energy imbalance
        let heat_capacity = 1e22; // J/K approximate for Earth system
        let dt = 1.0; // 1 second timestep
        let temperature_change = net_energy * dt / heat_capacity;
        
        self.climate_state.global_temperature += temperature_change;
        self.climate_state.surface_temperature = self.climate_state.global_temperature + 
                                               total_greenhouse * 0.1; // Greenhouse warming
        
        // Limits
        self.climate_state.global_temperature = self.climate_state.global_temperature
            .max(200.0) // Minimum temperature
            .min(400.0); // Maximum temperature
        
        self.climate_state.surface_temperature = self.climate_state.surface_temperature
            .max(200.0)
            .min(400.0);
        
        Ok(())
    }

    /// Calculate CO₂ radiative forcing
    fn calculate_co2_forcing(&self) -> f64 {
        // Logarithmic relationship: ΔF = 5.35 * ln(C/C₀)
        let reference_co2 = 280.0; // ppm pre-industrial
        let forcing_coefficient = 5.35; // W/m²
        
        if self.climate_state.co2_concentration > 0.0 {
            forcing_coefficient * (self.climate_state.co2_concentration / reference_co2).ln()
        } else {
            0.0
        }
    }

    /// Update ice-albedo feedback (Budyko model)
    fn update_ice_albedo_feedback(&mut self) -> Result<()> {
        // Critical temperature for ice formation
        let ice_temp_threshold = 273.15; // K (0°C)
        let ice_temp_range = 10.0; // K range for ice transition
        
        // Calculate ice coverage based on temperature
        let temp_below_freezing = ice_temp_threshold - self.climate_state.surface_temperature;
        
        if temp_below_freezing > ice_temp_range {
            self.climate_state.ice_coverage = 1.0; // Full ice coverage
        } else if temp_below_freezing > 0.0 {
            self.climate_state.ice_coverage = temp_below_freezing / ice_temp_range;
        } else {
            self.climate_state.ice_coverage = 0.0; // No ice
        }
        
        // Update albedo based on ice coverage
        let ice_albedo = 0.7; // Fresh ice/snow
        let ocean_albedo = 0.06; // Dark ocean
        let land_albedo = 0.2; // Vegetation/soil
        
        let ocean_fraction = self.ocean_area / (self.ocean_area + self.land_area);
        let base_albedo = ocean_fraction * ocean_albedo + (1.0 - ocean_fraction) * land_albedo;
        
        self.climate_state.albedo = base_albedo + self.climate_state.ice_coverage * 
                                   (ice_albedo - base_albedo);
        
        Ok(())
    }

    /// Update ocean circulation and mixing
    fn update_ocean_circulation(&mut self, constants: &PhysicsConstants) -> Result<()> {
        let dt = 1.0; // 1 second timestep
        
        // Thermohaline circulation between boxes
        for i in 0..self.ocean_boxes.len() {
            for j in (i+1)..self.ocean_boxes.len() {
                let temp_diff = self.ocean_boxes[j].temperature - self.ocean_boxes[i].temperature;
                let salt_diff = self.ocean_boxes[j].salinity - self.ocean_boxes[i].salinity;
                
                // Density-driven circulation
                let density_diff = -0.2 * temp_diff + 0.8 * salt_diff; // Simplified density
                let circulation_strength = density_diff * 1e3; // m³/s
                
                // Heat exchange
                let heat_exchange = circulation_strength * self.ocean_boxes[i].heat_capacity * 
                                  self.ocean_boxes[i].density * temp_diff * dt;
                
                let box_i_mass = self.ocean_boxes[i].density * self.ocean_area * self.ocean_boxes[i].depth;
                let box_j_mass = self.ocean_boxes[j].density * self.ocean_area * self.ocean_boxes[j].depth;
                
                self.ocean_boxes[i].temperature += heat_exchange / (box_i_mass * self.ocean_boxes[i].heat_capacity);
                self.ocean_boxes[j].temperature -= heat_exchange / (box_j_mass * self.ocean_boxes[j].heat_capacity);
            }
        }
        
        // Update ocean surface temperature in climate state
        if !self.ocean_boxes.is_empty() {
            self.climate_state.ocean_temperature = self.ocean_boxes[0].temperature;
        }
        
        Ok(())
    }

    /// Apply climate effects to particles
    fn apply_climate_effects(&self, states: &mut [PhysicsState]) -> Result<()> {
        for state in states.iter_mut() {
            // Apply temperature changes to particles near surface
            let surface_height = 6.371e6; // Earth radius
            let height = state.position.magnitude();
            
            if (height - surface_height).abs() < 1e4 { // Within 10 km of surface
                // Gradual temperature adjustment
                let temp_diff = self.climate_state.surface_temperature - state.temperature;
                state.temperature += 0.001 * temp_diff; // Slow adjustment
                
                // Limit particle temperatures
                state.temperature = state.temperature.max(200.0).min(400.0);
            }
        }
        
        Ok(())
    }

    /// Get climate sensitivity (temperature change per CO₂ doubling)
    pub fn climate_sensitivity(&self) -> f64 {
        // Simplified calculation
        let forcing_2x_co2 = self.calculate_co2_forcing() * 2.0; // Double CO₂
        let sensitivity = 3.0; // K/(W/m²) typical climate sensitivity
        forcing_2x_co2 * sensitivity
    }

    /// Check for runaway greenhouse effect
    pub fn is_runaway_greenhouse(&self) -> bool {
        self.climate_state.surface_temperature > 350.0 && // Very hot
        self.climate_state.co2_concentration > 1000.0      // High CO₂
    }

    /// Check for snowball Earth conditions
    pub fn is_snowball_earth(&self) -> bool {
        self.climate_state.ice_coverage > 0.9 && // Near-global ice
        self.climate_state.surface_temperature < 240.0 // Very cold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_climate_solver_creation() {
        let solver = ClimateSolver::new();
        assert!(solver.climate_state.global_temperature > 0.0);
        assert!(solver.climate_state.co2_concentration > 0.0);
    }

    #[test]
    fn test_earth_like_initialization() {
        let mut solver = ClimateSolver::new();
        solver.init_earth_like();
        
        assert_eq!(solver.ocean_boxes.len(), 3);
        assert!(solver.ocean_boxes[0].temperature > solver.ocean_boxes[2].temperature);
    }

    #[test]
    fn test_co2_forcing() {
        let mut solver = ClimateSolver::new();
        
        // Test CO₂ doubling
        solver.climate_state.co2_concentration = 560.0; // 2x pre-industrial
        let forcing = solver.calculate_co2_forcing();
        
        // Should be approximately 3.7 W/m² for CO₂ doubling
        assert!(forcing > 3.0);
        assert!(forcing < 4.0);
    }

    #[test]
    fn test_ice_albedo_feedback() {
        let mut solver = ClimateSolver::new();
        
        // Test cold conditions
        solver.climate_state.surface_temperature = 263.0; // -10°C
        solver.update_ice_albedo_feedback().unwrap();
        
        assert!(solver.climate_state.ice_coverage > 0.0);
        assert!(solver.climate_state.albedo > 0.3); // Higher than default
        
        // Test warm conditions
        solver.climate_state.surface_temperature = 298.0; // 25°C
        solver.update_ice_albedo_feedback().unwrap();
        
        assert!(solver.climate_state.ice_coverage < 0.1);
    }

    #[test]
    fn test_extreme_climate_detection() {
        let mut solver = ClimateSolver::new();
        
        // Test runaway greenhouse
        solver.climate_state.surface_temperature = 360.0;
        solver.climate_state.co2_concentration = 2000.0;
        assert!(solver.is_runaway_greenhouse());
        
        // Test snowball Earth
        solver.climate_state.surface_temperature = 230.0;
        solver.climate_state.ice_coverage = 0.95;
        assert!(solver.is_snowball_earth());
    }
}