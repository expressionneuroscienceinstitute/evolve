//! Planet system and habitability calculations
//!
//! This module handles planetary formation, environmental conditions, and habitability
//! calculations as specified in the instructions.

use crate::constants::{SurvivalThresholds, elements};
use crate::types::*;
use crate::{Result, SimError};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use nalgebra::{Point3, Vector3};
use rkyv::{Archive, Serialize as RkyvSerialize, Deserialize as RkyvDeserialize};

/// Planet representation with environmental and geological data
#[derive(Debug, Clone)]
pub struct Planet {
    pub id: PlanetId,
    pub position: Coord2D,
    pub mass: MassEnergy,
    pub radius: f64, // meters
    pub environment: EnvironmentProfile,
    pub elements: ElementTable,
    pub geological_layers: Vec<GeologicalLayer>,
    pub age_ticks: u64,
    pub class: PlanetClass,
    pub has_life: bool,
    pub agent_population: u32,
}

/// Environmental profile for planetary habitability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentProfile {
    /// Surface liquid water fraction (0-1)
    pub liquid_water: f64,
    /// Atmospheric oxygen percentage (0-1)
    pub atmos_oxygen: f64,
    /// Atmospheric pressure relative to Earth
    pub atmos_pressure: f64,
    /// Mean surface temperature (Celsius)
    pub temp_celsius: f64,
    /// Radiation dose (Sv/year)
    pub radiation: f64,
    /// Stellar energy flux (kW/m²)
    pub energy_flux: f64,
    /// Shelter availability index (0-1)
    pub shelter_index: f64,
    /// Hazard event rate (events/year)
    pub hazard_rate: f64,
}

impl EnvironmentProfile {
    /// Create Earth-like baseline environment
    pub fn earth_baseline() -> Self {
        Self {
            liquid_water: 0.71,
            atmos_oxygen: 0.21,
            atmos_pressure: 1.0,
            temp_celsius: 15.0,
            radiation: 0.002, // 2 mSv/year
            energy_flux: 1.361, // Solar constant
            shelter_index: 0.5,
            hazard_rate: 0.001,
        }
    }
    
    /// Check if environment supports life
    pub fn is_habitable(&self, thresholds: &SurvivalThresholds) -> bool {
        self.liquid_water >= thresholds.min_liquid_water &&
        self.liquid_water <= thresholds.max_liquid_water &&
        self.atmos_oxygen >= thresholds.min_atmos_oxygen &&
        self.atmos_oxygen <= thresholds.max_atmos_oxygen &&
        self.atmos_pressure >= thresholds.min_atmos_pressure &&
        self.atmos_pressure <= thresholds.max_atmos_pressure &&
        self.temp_celsius >= thresholds.min_temp_celsius &&
        self.temp_celsius <= thresholds.max_temp_celsius &&
        self.radiation <= thresholds.max_radiation_sv_per_year &&
        self.energy_flux >= thresholds.min_energy_flux_kw_per_m2
    }
    
    /// Calculate habitability score (0-1, higher is better)
    pub fn habitability_score(&self, thresholds: &SurvivalThresholds) -> f64 {
        if !self.is_habitable(thresholds) {
            return 0.0;
        }
        
        // Calculate normalized scores for each parameter
        let water_score = 1.0 - (self.liquid_water - 0.5).abs() * 2.0;
        let oxygen_score = 1.0 - (self.atmos_oxygen - 0.21).abs() / 0.21;
        let pressure_score = 1.0 - (self.atmos_pressure - 1.0).abs();
        let temp_score = 1.0 - (self.temp_celsius - 15.0).abs() / 50.0;
        let radiation_score = 1.0 - (self.radiation / thresholds.max_radiation_sv_per_year);
        let energy_score = (self.energy_flux / 2.0).min(1.0);
        
        // Weighted average
        (water_score * 0.25 + oxygen_score * 0.25 + pressure_score * 0.15 + 
         temp_score * 0.15 + radiation_score * 0.1 + energy_score * 0.1).max(0.0)
    }
}

/// Geological layer in planetary stratigraphy
#[derive(Debug, Clone)]
pub struct GeologicalLayer {
    pub id: LayerId,
    pub depth_m: f64,
    pub thickness_m: f64,
    pub material_type: MaterialType,
    pub bulk_density: f64, // kg/m³
    pub elements: ElementTable,
    pub age_ticks: u64,
}

/// Material types for geological layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
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

impl MaterialType {
    /// Get hardness (Mohs scale)
    pub fn hardness(&self) -> f64 {
        match self {
            MaterialType::Regolith => 1.0,
            MaterialType::Topsoil => 1.5,
            MaterialType::Subsoil => 2.0,
            MaterialType::SedimentaryRock => 3.0,
            MaterialType::IgneousRock => 6.0,
            MaterialType::MetamorphicRock => 7.0,
            MaterialType::OreVein => 5.0,
            MaterialType::Ice => 1.5,
            MaterialType::Magma => 0.5,
        }
    }
    
    /// Get typical density (kg/m³)
    pub fn density(&self) -> f64 {
        match self {
            MaterialType::Regolith => 1500.0,
            MaterialType::Topsoil => 1200.0,
            MaterialType::Subsoil => 1600.0,
            MaterialType::SedimentaryRock => 2500.0,
            MaterialType::IgneousRock => 2700.0,
            MaterialType::MetamorphicRock => 2800.0,
            MaterialType::OreVein => 4000.0,
            MaterialType::Ice => 917.0,
            MaterialType::Magma => 2400.0,
        }
    }
}

/// Planet classification system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub enum PlanetClass {
    /// Earth-like, balanced composition
    E,
    /// Desert world, low water
    D,
    /// Ice world, high water content
    I,
    /// Toxic atmosphere
    T,
    /// Gas dwarf
    G,
}

impl PlanetClass {
    /// Get typical environment for this planet class
    pub fn typical_environment(&self) -> EnvironmentProfile {
        match self {
            PlanetClass::E => EnvironmentProfile::earth_baseline(),
            PlanetClass::D => EnvironmentProfile {
                liquid_water: 0.05,
                atmos_oxygen: 0.02,
                atmos_pressure: 0.1,
                temp_celsius: 45.0,
                radiation: 0.01,
                energy_flux: 2.0,
                shelter_index: 0.8,
                hazard_rate: 0.005,
            },
            PlanetClass::I => EnvironmentProfile {
                liquid_water: 0.95,
                atmos_oxygen: 0.05,
                atmos_pressure: 0.5,
                temp_celsius: -30.0,
                radiation: 0.05,
                energy_flux: 0.5,
                shelter_index: 0.3,
                hazard_rate: 0.001,
            },
            PlanetClass::T => EnvironmentProfile {
                liquid_water: 0.1,
                atmos_oxygen: 0.0,
                atmos_pressure: 2.0,
                temp_celsius: 150.0,
                radiation: 0.1,
                energy_flux: 1.0,
                shelter_index: 0.1,
                hazard_rate: 0.02,
            },
            PlanetClass::G => EnvironmentProfile {
                liquid_water: 0.0,
                atmos_oxygen: 0.0,
                atmos_pressure: 10.0,
                temp_celsius: -100.0,
                radiation: 0.5,
                energy_flux: 0.1,
                shelter_index: 0.0,
                hazard_rate: 0.1,
            },
        }
    }
}

/// Planet generation and management system
pub struct PlanetSystem {
    planets: HashMap<PlanetId, Planet>,
    next_id: u64,
    survival_thresholds: SurvivalThresholds,
}

impl PlanetSystem {
    pub fn new() -> Self {
        Self {
            planets: HashMap::new(),
            next_id: 1,
            survival_thresholds: SurvivalThresholds::default(),
        }
    }
    
    /// Generate a new planet with procedural characteristics
    pub fn generate_planet(
        &mut self,
        position: Coord2D,
        stellar_metallicity: f64,
        distance_au: f64,
        rng: &mut impl Rng,
    ) -> Result<PlanetId> {
        let id = PlanetId::new(self.next_id);
        self.next_id += 1;
        
        // Determine planet class based on distance and metallicity
        let class = self.determine_planet_class(distance_au, stellar_metallicity, rng);
        
        // Generate mass based on class
        let mass = self.generate_planet_mass(&class, rng);
        
        // Calculate radius from mass-radius relation
        let radius = self.calculate_planet_radius(mass.as_kg(), &class);
        
        // Generate environment based on class and stellar distance
        let environment = self.generate_environment(&class, distance_au, rng);
        
        // Generate elemental composition
        let elements = self.generate_element_table(stellar_metallicity, &class, rng);
        
        // Generate geological layers
        let geological_layers = self.generate_geology(&class, mass.as_kg(), rng)?;
        
        let planet = Planet {
            id,
            position,
            mass,
            radius,
            environment,
            elements,
            geological_layers,
            age_ticks: 0,
            class,
            has_life: false,
            agent_population: 0,
        };
        
        self.planets.insert(id, planet);
        Ok(id)
    }
    
    /// Determine planet class based on formation conditions
    fn determine_planet_class(
        &self,
        distance_au: f64,
        metallicity: f64,
        rng: &mut impl Rng,
    ) -> PlanetClass {
        // Simplified planet formation model
        let ice_line = 2.7; // AU, where water freezes
        let random_factor = rng.gen::<f64>();
        
        if distance_au < 0.5 {
            // Too hot, likely toxic
            if random_factor < 0.8 { PlanetClass::T } else { PlanetClass::D }
        } else if distance_au < ice_line {
            // Habitable zone
            if metallicity > 0.01 && random_factor < 0.3 {
                PlanetClass::E
            } else if random_factor < 0.7 {
                PlanetClass::D
            } else {
                PlanetClass::T
            }
        } else if distance_au < 10.0 {
            // Beyond ice line
            if random_factor < 0.6 { PlanetClass::I } else { PlanetClass::G }
        } else {
            // Outer system
            PlanetClass::G
        }
    }
    
    /// Generate planet mass based on class
    fn generate_planet_mass(&self, class: &PlanetClass, rng: &mut impl Rng) -> MassEnergy {
        use crate::constants::physics::M_EARTH;
        
        let mass_range = match class {
            PlanetClass::E => (0.5, 2.0),     // Earth-like
            PlanetClass::D => (0.1, 1.5),     // Smaller, dry
            PlanetClass::I => (0.3, 3.0),     // Larger, icy
            PlanetClass::T => (0.8, 5.0),     // Venus-like
            PlanetClass::G => (3.0, 15.0),    // Gas dwarf
        };
        
        let mass_factor = rng.gen_range(mass_range.0..=mass_range.1);
        MassEnergy::new(mass_factor * M_EARTH)
    }
    
    /// Calculate planet radius from mass and class
    fn calculate_planet_radius(&self, mass_kg: f64, class: &PlanetClass) -> f64 {
        use crate::constants::physics::M_EARTH;
        
        let earth_radius = 6.371e6; // meters
        let mass_ratio = mass_kg / M_EARTH;
        
        // Mass-radius relation depends on composition
        let radius_scaling = match class {
            PlanetClass::E => mass_ratio.powf(0.27),      // Rocky
            PlanetClass::D => mass_ratio.powf(0.25),      // Dry rocky
            PlanetClass::I => mass_ratio.powf(0.3),       // Ice-rich
            PlanetClass::T => mass_ratio.powf(0.28),      // Dense atmosphere
            PlanetClass::G => mass_ratio.powf(0.5),       // Low density
        };
        
        earth_radius * radius_scaling
    }
    
    /// Generate environmental conditions
    fn generate_environment(
        &self,
        class: &PlanetClass,
        distance_au: f64,
        rng: &mut impl Rng,
    ) -> EnvironmentProfile {
        let mut env = class.typical_environment();
        
        // Modify based on stellar distance
        let flux_factor = 1.0 / (distance_au * distance_au);
        env.energy_flux *= flux_factor;
        
        // Temperature depends on stellar flux and atmosphere
        let base_temp = 278.0 * (flux_factor).powf(0.25) - 273.15; // Stefan-Boltzmann
        env.temp_celsius = base_temp + rng.gen_range(-20.0..20.0);
        
        // Add some randomness to other parameters
        env.liquid_water = (env.liquid_water * rng.gen_range(0.8..1.2)).clamp(0.0, 1.0);
        env.atmos_oxygen = (env.atmos_oxygen * rng.gen_range(0.5..1.5)).clamp(0.0, 1.0);
        env.atmos_pressure = env.atmos_pressure * rng.gen_range(0.5..2.0);
        env.radiation = env.radiation * rng.gen_range(0.1..5.0);
        
        env
    }
    
    /// Generate elemental composition
    fn generate_element_table(
        &self,
        stellar_metallicity: f64,
        class: &PlanetClass,
        rng: &mut impl Rng,
    ) -> ElementTable {
        let mut elements = ElementTable::new();
        
        // Base composition varies by planet class
        match class {
            PlanetClass::E => {
                // Earth-like composition
                elements = ElementTable::earth_baseline();
            },
            PlanetClass::D => {
                // Dry, silicon-rich
                elements.set_abundance(elements::SI, 400_000);
                elements.set_abundance(elements::FE, 80_000);
                elements.set_abundance(elements::O, 300_000);
                elements.set_abundance(elements::H, 50_000);
            },
            PlanetClass::I => {
                // Ice-rich, hydrogen and oxygen heavy
                elements.set_abundance(elements::H, 300_000);
                elements.set_abundance(elements::O, 600_000);
                elements.set_abundance(elements::C, 10_000);
                elements.set_abundance(elements::N, 5_000);
            },
            PlanetClass::T => {
                // Volcanic, sulfur-rich
                elements.set_abundance(elements::S, 50_000);
                elements.set_abundance(elements::SI, 300_000);
                elements.set_abundance(elements::FE, 100_000);
                elements.set_abundance(elements::CL, 20_000);
            },
            PlanetClass::G => {
                // Gas dwarf, light elements
                elements.set_abundance(elements::H, 800_000);
                elements.set_abundance(elements::HE, 150_000);
                elements.set_abundance(elements::C, 5_000);
            },
        }
        
        // Scale by stellar metallicity
        for z in 3..=92 {
            let current = elements.get_abundance(z);
            let scaled = (current as f64 * stellar_metallicity).round() as u32;
            elements.set_abundance(z, scaled);
        }
        
        elements
    }
    
    /// Generate geological layer structure
    fn generate_geology(
        &self,
        class: &PlanetClass,
        mass_kg: f64,
        rng: &mut impl Rng,
    ) -> Result<Vec<GeologicalLayer>> {
        let mut layers = Vec::new();
        let mut current_depth = 0.0;
        let mut layer_id = 0;
        
        // Different geological structures by planet class
        match class {
            PlanetClass::E | PlanetClass::D | PlanetClass::T => {
                // Rocky planet structure
                self.generate_rocky_geology(&mut layers, &mut current_depth, &mut layer_id, rng)?;
            },
            PlanetClass::I => {
                // Icy planet structure
                self.generate_icy_geology(&mut layers, &mut current_depth, &mut layer_id, rng)?;
            },
            PlanetClass::G => {
                // Gas planet (minimal solid layers)
                self.generate_gas_geology(&mut layers, &mut current_depth, &mut layer_id, rng)?;
            },
        }
        
        Ok(layers)
    }
    
    fn generate_rocky_geology(
        &self,
        layers: &mut Vec<GeologicalLayer>,
        current_depth: &mut f64,
        layer_id: &mut u64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Surface regolith
        self.add_layer(
            layers, current_depth, layer_id,
            MaterialType::Regolith,
            rng.gen_range(1.0..10.0),
            rng,
        )?;
        
        // Sedimentary layers
        for _ in 0..rng.gen_range(3..8) {
            self.add_layer(
                layers, current_depth, layer_id,
                MaterialType::SedimentaryRock,
                rng.gen_range(100.0..1000.0),
                rng,
            )?;
        }
        
        // Igneous basement
        for _ in 0..rng.gen_range(5..15) {
            self.add_layer(
                layers, current_depth, layer_id,
                MaterialType::IgneousRock,
                rng.gen_range(1000.0..10000.0),
                rng,
            )?;
        }
        
        // Occasional ore veins
        if rng.gen_bool(0.3) {
            self.add_layer(
                layers, current_depth, layer_id,
                MaterialType::OreVein,
                rng.gen_range(10.0..100.0),
                rng,
            )?;
        }
        
        Ok(())
    }
    
    fn generate_icy_geology(
        &self,
        layers: &mut Vec<GeologicalLayer>,
        current_depth: &mut f64,
        layer_id: &mut u64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Thick ice layers
        for _ in 0..rng.gen_range(5..10) {
            self.add_layer(
                layers, current_depth, layer_id,
                MaterialType::Ice,
                rng.gen_range(1000.0..10000.0),
                rng,
            )?;
        }
        
        // Rocky core
        for _ in 0..rng.gen_range(2..5) {
            self.add_layer(
                layers, current_depth, layer_id,
                MaterialType::IgneousRock,
                rng.gen_range(5000.0..20000.0),
                rng,
            )?;
        }
        
        Ok(())
    }
    
    fn generate_gas_geology(
        &self,
        layers: &mut Vec<GeologicalLayer>,
        current_depth: &mut f64,
        layer_id: &mut u64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Small rocky/icy core
        self.add_layer(
            layers, current_depth, layer_id,
            MaterialType::IgneousRock,
            rng.gen_range(1000.0..5000.0),
            rng,
        )?;
        
        Ok(())
    }
    
    fn add_layer(
        &self,
        layers: &mut Vec<GeologicalLayer>,
        current_depth: &mut f64,
        layer_id: &mut u64,
        material_type: MaterialType,
        thickness: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let elements = self.generate_layer_elements(&material_type, rng);
        
        layers.push(GeologicalLayer {
            id: LayerId::new(*layer_id),
            depth_m: *current_depth,
            thickness_m: thickness,
            bulk_density: material_type.density(),
            material_type,
            elements,
            age_ticks: 0,
        });
        
        *current_depth += thickness;
        *layer_id += 1;
        
        Ok(())
    }
    
    fn generate_layer_elements(&self, material_type: &MaterialType, rng: &mut impl Rng) -> ElementTable {
        let mut elements = ElementTable::new();
        
        match material_type {
            MaterialType::Regolith | MaterialType::Topsoil => {
                // Weathered surface material
                elements.set_abundance(elements::SI, 300_000);
                elements.set_abundance(elements::O, 400_000);
                elements.set_abundance(elements::AL, 80_000);
                elements.set_abundance(elements::FE, 50_000);
            },
            MaterialType::SedimentaryRock => {
                // Limestone, sandstone, etc.
                elements.set_abundance(elements::CA, 200_000);
                elements.set_abundance(elements::C, 120_000);
                elements.set_abundance(elements::O, 480_000);
                elements.set_abundance(elements::SI, 100_000);
            },
            MaterialType::IgneousRock => {
                // Granite, basalt
                elements.set_abundance(elements::SI, 280_000);
                elements.set_abundance(elements::O, 460_000);
                elements.set_abundance(elements::AL, 82_000);
                elements.set_abundance(elements::FE, 56_000);
                elements.set_abundance(elements::CA, 41_000);
            },
            MaterialType::OreVein => {
                // Concentrated metal deposits
                if rng.gen_bool(0.5) {
                    elements.set_abundance(elements::FE, 600_000);
                    elements.set_abundance(elements::O, 300_000);
                } else {
                    elements.set_abundance(elements::CU, 400_000);
                    elements.set_abundance(elements::S, 200_000);
                }
                
                // Rare chance of precious metals
                if rng.gen_bool(0.1) {
                    elements.set_abundance(elements::AU, 100);
                }
                if rng.gen_bool(0.01) {
                    elements.set_abundance(elements::U, 10);
                }
            },
            MaterialType::Ice => {
                // Water ice with dissolved gases
                elements.set_abundance(elements::H, 111_000);
                elements.set_abundance(elements::O, 889_000);
            },
            _ => {
                // Default silicate composition
                elements.set_abundance(elements::SI, 200_000);
                elements.set_abundance(elements::O, 400_000);
                elements.set_abundance(elements::FE, 100_000);
            },
        }
        
        elements
    }
    
    /// Get planet by ID
    pub fn get_planet(&self, id: PlanetId) -> Option<&Planet> {
        self.planets.get(&id)
    }
    
    /// Get mutable planet by ID
    pub fn get_planet_mut(&mut self, id: PlanetId) -> Option<&mut Planet> {
        self.planets.get_mut(&id)
    }
    
    /// List all planets
    pub fn list_planets(&self) -> Vec<&Planet> {
        self.planets.values().collect()
    }
    
    /// List planets by class
    pub fn list_planets_by_class(&self, class: PlanetClass) -> Vec<&Planet> {
        self.planets.values()
            .filter(|p| p.class == class)
            .collect()
    }
    
    /// Update planet environment over time
    pub fn update_planet_environment(&mut self, planet_id: PlanetId, tick: Tick) -> Result<()> {
        if let Some(planet) = self.planets.get_mut(&planet_id) {
            planet.age_ticks += 1;
            
            // TODO: Implement climate evolution, geological processes, etc.
            // This is a placeholder for more complex environmental dynamics
        }
        
        Ok(())
    }
}