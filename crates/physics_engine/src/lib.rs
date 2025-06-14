//! Comprehensive Physics Engine for Universe Simulation
//! 
//! Implements peer-reviewed physics with multiple layers:
//! - Classical Mechanics (Leap-frog integrator)
//! - Electromagnetism (FDTD solver)
//! - Thermodynamics (Gibbs free energy)
//! - Quantum Layer (Tight-binding model)
//! - Chemical Kinetics (Stiff ODE solver)
//! - Geodynamics (Mantle convection)
//! - Climate & Ocean (Energy balance + CO2 cycle)

use nalgebra::{Vector3, Matrix3};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;

#[cfg(feature = "bevy")]
use bevy_ecs::prelude::Component;

pub mod constants;
pub mod classical;
pub mod electromagnetic;
pub mod thermodynamics;
pub mod quantum;
pub mod chemistry;
pub mod geodynamics;
pub mod climate;
pub mod validation;

pub use constants::*;

/// Core physics state for a single particle/entity
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "bevy", derive(Component))]
pub struct PhysicsState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub acceleration: Vector3<f64>,
    pub mass: f64,
    pub charge: f64,
    pub temperature: f64,
    pub entropy: f64,
}

/// Element table storing abundances (ppm) for all 118 elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementTable {
    /// Abundances in parts per million (index = proton number Z)
    #[serde(with = "serde_arrays")]
    pub abundances: [u32; 118],
}

impl ElementTable {
    pub fn new() -> Self {
        Self {
            abundances: [0; 118],
        }
    }

    pub fn earth_baseline() -> Self {
        let mut table = Self::new();
        
        // Set Earth-like abundances (in ppm)
        table.abundances[1] = 140_000;    // Hydrogen
        table.abundances[6] = 200;        // Carbon
        table.abundances[8] = 461_000;    // Oxygen
        table.abundances[14] = 282_000;   // Silicon
        table.abundances[26] = 56_300;    // Iron
        table.abundances[92] = 3;         // Uranium-238
        
        table
    }

    pub fn get_abundance(&self, element: u8) -> u32 {
        if element < 118 {
            self.abundances[element as usize]
        } else {
            0
        }
    }

    pub fn set_abundance(&mut self, element: u8, ppm: u32) {
        if element < 118 {
            self.abundances[element as usize] = ppm;
        }
    }
}

/// Environmental conditions for habitability calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentProfile {
    pub liquid_water: f64,      // Surface liquid H2O fraction
    pub atmos_oxygen: f64,      // O2 percentage of atmosphere
    pub atmos_pressure: f64,    // Relative to Earth sea-level
    pub temp_celsius: f64,      // Mean surface temperature
    pub radiation: f64,         // Cosmic & solar ionizing flux (Sv/year)
    pub energy_flux: f64,       // Stellar insolation (kW/m²)
    pub shelter_index: f64,     // Availability of caves/structures
    pub hazard_rate: f64,       // Events per year (meteors, quakes, storms)
}

impl EnvironmentProfile {
    pub fn earth_baseline() -> Self {
        Self {
            liquid_water: 0.71,
            atmos_oxygen: 0.21,
            atmos_pressure: 1.0,
            temp_celsius: 15.0,
            radiation: 0.002,  // 2 mSv/year background
            energy_flux: 1.361, // Solar constant
            shelter_index: 0.1,
            hazard_rate: 0.001,
        }
    }

    /// Check if environment supports life without protection
    pub fn is_habitable(&self) -> bool {
        self.liquid_water >= 0.2 
            && self.atmos_oxygen > 0.05 
            && self.atmos_oxygen < 0.4
            && self.temp_celsius >= -20.0 
            && self.temp_celsius <= 80.0
            && self.radiation < 5.0  // 5 Sv/year lethal threshold
    }

    /// Calculate survival probability for unprotected organisms
    pub fn survival_probability(&self) -> f64 {
        let water_factor = if self.liquid_water < 0.2 { 0.1 } 
                          else if self.liquid_water > 0.8 { 0.5 }
                          else { 1.0 };
        
        let oxygen_factor = if self.atmos_oxygen < 0.05 { 0.0 }
                           else if self.atmos_oxygen > 0.4 { 0.2 }
                           else { 1.0 };
        
        let temp_factor = if self.temp_celsius < -20.0 || self.temp_celsius > 80.0 { 0.0 }
                         else { 1.0 };
        
        let radiation_factor = if self.radiation > 5.0 { 0.0 }
                              else { (5.0 - self.radiation) / 5.0 };
        
        water_factor * oxygen_factor * temp_factor * radiation_factor
    }
}

/// Geological layer in planetary stratigraphy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumLayer {
    pub thickness_m: f64,
    pub material_type: MaterialType,
    pub bulk_density: f64,      // kg/m³
    pub elements: ElementTable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    Gas,
    Regolith,
    Topsoil,
    Subsoil,
    SedimentaryRock,
    IgneousRock,
    MetamorphicRock,
    OreVein,
    Ice,
    Magma,
}

/// Material properties database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub tensile_strength: f64,  // MPa
    pub hardness: f64,          // Mohs scale
    pub thermal_stability: f64, // Kelvin
    pub electrical_conductivity: f64, // S/m
    pub optical_properties: HashMap<String, f64>,
}

/// Core physics engine that manages all physics calculations
pub struct PhysicsEngine {
    pub constants: PhysicsConstants,
    pub time_step: f64,
    pub classical_solver: classical::ClassicalSolver,
    pub em_solver: electromagnetic::EMSolver,
    pub thermo_solver: thermodynamics::ThermoSolver,
    pub quantum_solver: quantum::QuantumSolver,
    pub chemistry_solver: chemistry::ChemistrySolver,
    pub geo_solver: geodynamics::GeodynamicsSolver,
    pub climate_solver: climate::ClimateSolver,
}

impl PhysicsEngine {
    pub fn new(time_step: f64) -> Result<Self> {
        Ok(Self {
            constants: PhysicsConstants::default(),
            time_step,
            classical_solver: classical::ClassicalSolver::new(time_step),
            em_solver: electromagnetic::EMSolver::new(),
            thermo_solver: thermodynamics::ThermoSolver::new(),
            quantum_solver: quantum::QuantumSolver::new(),
            chemistry_solver: chemistry::ChemistrySolver::new(),
            geo_solver: geodynamics::GeodynamicsSolver::new(),
            climate_solver: climate::ClimateSolver::new(),
        })
    }

    /// Advance all physics by one time step
    pub fn step(&mut self, states: &mut [PhysicsState]) -> Result<()> {
        // Classical mechanics (positions, velocities)
        self.classical_solver.step(states, &self.constants)?;
        
        // Electromagnetic forces
        self.em_solver.step(states, &self.constants)?;
        
        // Thermodynamic processes
        self.thermo_solver.step(states, &self.constants)?;
        
        // Quantum transitions
        self.quantum_solver.step(states, &self.constants)?;
        
        // Chemical reactions
        self.chemistry_solver.step(states, &self.constants)?;
        
        Ok(())
    }

    /// Validate physics conservation laws
    pub fn validate_conservation(&self, states: &[PhysicsState]) -> Result<()> {
        validation::check_energy_conservation(states, &self.constants)?;
        validation::check_momentum_conservation(states)?;
        validation::check_entropy_increase(states)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_element_table() {
        let mut table = ElementTable::new();
        table.set_abundance(1, 1000); // Hydrogen
        assert_eq!(table.get_abundance(1), 1000);
        assert_eq!(table.get_abundance(200), 0); // Out of bounds
    }

    #[test]
    fn test_environment_habitability() {
        let earth = EnvironmentProfile::earth_baseline();
        assert!(earth.is_habitable());
        assert!(earth.survival_probability() > 0.8);
        
        let mars = EnvironmentProfile {
            liquid_water: 0.0,
            atmos_oxygen: 0.001,
            temp_celsius: -60.0,
            ..earth
        };
        assert!(!mars.is_habitable());
        assert!(mars.survival_probability() < 0.1);
    }

    #[test]
    fn test_physics_engine_creation() {
        let engine = PhysicsEngine::new(1e-6).unwrap();
        assert_relative_eq!(engine.time_step, 1e-6);
    }
}