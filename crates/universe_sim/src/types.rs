//! Core data types for the universe simulation
//!
//! This module defines the fundamental data structures used throughout the simulation,
//! including spatial coordinates, time, mass-energy, and identifiers.

use nalgebra::{Point2, Point3, Vector2, Vector3};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use crate::constants::elements;

/// Simulation time in ticks (1 tick = configurable years, default 1M years)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Tick(pub u64);

impl Tick {
    pub fn new(value: u64) -> Self {
        Self(value)
    }
    
    pub fn as_u64(&self) -> u64 {
        self.0
    }
    
    pub fn to_years(&self, years_per_tick: f64) -> f64 {
        self.0 as f64 * years_per_tick
    }
}

impl fmt::Display for Tick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

impl std::ops::Add for Tick {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Tick {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

/// 2D coordinates in the universe grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Coord2D {
    pub x: u32,
    pub y: u32,
}

impl Coord2D {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
    
    pub fn to_point2(&self) -> Point2<f64> {
        Point2::new(self.x as f64, self.y as f64)
    }
    
    pub fn to_vector2(&self) -> Vector2<f64> {
        Vector2::new(self.x as f64, self.y as f64)
    }
    
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = (self.x as i64 - other.x as i64) as f64;
        let dy = (self.y as i64 - other.y as i64) as f64;
        (dx * dx + dy * dy).sqrt()
    }
    
    /// Calculate distance on a toroidal grid
    pub fn toroidal_distance_to(&self, other: &Self, grid_size: (u32, u32)) -> f64 {
        let dx = {
            let diff = (self.x as i64 - other.x as i64).abs() as u32;
            std::cmp::min(diff, grid_size.0 - diff) as f64
        };
        let dy = {
            let diff = (self.y as i64 - other.y as i64).abs() as u32;
            std::cmp::min(diff, grid_size.1 - diff) as f64
        };
        (dx * dx + dy * dy).sqrt()
    }
}

/// 3D coordinates for physics calculations
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Coord3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Coord3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    
    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    
    pub fn to_point3(&self) -> Point3<f64> {
        Point3::new(self.x, self.y, self.z)
    }
    
    pub fn to_vector3(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
    
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Mass-energy quantity (kg equivalent)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct MassEnergy(pub f64);

impl MassEnergy {
    pub fn new(kg: f64) -> Self {
        Self(kg)
    }
    
    pub fn zero() -> Self {
        Self(0.0)
    }
    
    pub fn as_kg(&self) -> f64 {
        self.0
    }
    
    pub fn to_energy_joules(&self) -> f64 {
        use crate::constants::physics::C;
        self.0 * C * C
    }
    
    pub fn from_energy_joules(joules: f64) -> Self {
        use crate::constants::physics::C;
        Self(joules / (C * C))
    }
}

impl std::ops::Add for MassEnergy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for MassEnergy {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<f64> for MassEnergy {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

/// Entropy quantity (dimensionless)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Entropy(pub f64);

impl Entropy {
    pub fn new(value: f64) -> Self {
        Self(value.max(0.0)) // Entropy cannot be negative
    }
    
    pub fn zero() -> Self {
        Self(0.0)
    }
    
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl std::ops::Add for Entropy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Entropy {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self((self.0 - rhs.0).max(0.0)) // Entropy cannot decrease
    }
}

/// Temperature in Kelvin
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Temperature(pub f64);

impl Temperature {
    pub fn new(kelvin: f64) -> Self {
        Self(kelvin.max(0.0)) // Absolute zero minimum
    }
    
    pub fn from_celsius(celsius: f64) -> Self {
        Self::new(celsius + 273.15)
    }
    
    pub fn as_kelvin(&self) -> f64 {
        self.0
    }
    
    pub fn as_celsius(&self) -> f64 {
        self.0 - 273.15
    }
}

/// Pressure in Pascals
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]  
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Pressure(pub f64);

impl Pressure {
    pub fn new(pascals: f64) -> Self {
        Self(pascals.max(0.0))
    }
    
    pub fn from_atmospheres(atm: f64) -> Self {
        Self::new(atm * 101_325.0)
    }
    
    pub fn as_pascals(&self) -> f64 {
        self.0
    }
    
    pub fn as_atmospheres(&self) -> f64 {
        self.0 / 101_325.0
    }
}

/// Energy flux (watts per square meter)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct EnergyFlux(pub f64);

impl EnergyFlux {
    pub fn new(watts_per_m2: f64) -> Self {
        Self(watts_per_m2.max(0.0))
    }
    
    pub fn as_watts_per_m2(&self) -> f64 {
        self.0
    }
}

/// Radiation dose (Sieverts per year)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct RadiationDose(pub f64);

impl RadiationDose {
    pub fn new(sv_per_year: f64) -> Self {
        Self(sv_per_year.max(0.0))
    }
    
    pub fn as_sv_per_year(&self) -> f64 {
        self.0
    }
}

/// Velocity (m/s)
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Velocity {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Velocity {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    
    pub fn to_vector3(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
    
    pub fn relativistic_factor(&self) -> f64 {
        use crate::constants::physics::C;
        let v_over_c = self.magnitude() / C;
        1.0 / (1.0 - v_over_c * v_over_c).sqrt().max(1e-10)
    }
}

/// Unique identifier types
macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[derive(Archive, RkyvSerialize, RkyvDeserialize)]
        #[derive(Serialize, Deserialize)]
        pub struct $name(pub u64);
        
        impl $name {
            pub fn new(id: u64) -> Self {
                Self(id)
            }
            
            pub fn as_u64(&self) -> u64 {
                self.0
            }
        }
        
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

define_id!(PlanetId);
define_id!(StarId);
define_id!(AgentId);
define_id!(LineageId);
define_id!(TechId);
define_id!(EventId);
define_id!(CellId);
define_id!(LayerId);

/// Resource collection (elements by atomic number)
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct ElementTable {
    /// Parts per million by atomic number (index 0-117)
    pub abundances: Vec<u32>,
}

impl ElementTable {
    pub fn new() -> Self {
        Self {
            abundances: vec![0; 118],
        }
    }
    
    pub fn get_abundance(&self, atomic_number: u8) -> u32 {
        if atomic_number == 0 || atomic_number > 118 {
            return 0;
        }
        self.abundances.get(atomic_number as usize - 1).copied().unwrap_or(0)
    }
    
    pub fn set_abundance(&mut self, atomic_number: u8, ppm: u32) {
        if atomic_number == 0 || atomic_number > 118 {
            return;
        }
        if self.abundances.len() < 118 {
            self.abundances.resize(118, 0);
        }
        self.abundances[atomic_number as usize - 1] = ppm;
    }
    
    pub fn add_abundance(&mut self, atomic_number: u8, ppm: u32) {
        if atomic_number == 0 || atomic_number > 118 {
            return;
        }
        if self.abundances.len() < 118 {
            self.abundances.resize(118, 0);
        }
        self.abundances[atomic_number as usize - 1] = 
            self.abundances[atomic_number as usize - 1].saturating_add(ppm);
    }
    
    pub fn total_abundance(&self) -> u64 {
        self.abundances.iter().map(|&x| x as u64).sum()
    }
    
    /// Earth-like baseline composition
    pub fn earth_baseline() -> Self {
        let mut table = Self::new();
        // Values from instructions (sample Earth baseline)
        table.set_abundance(1, 140_000);  // Hydrogen
        table.set_abundance(6, 200);      // Carbon
        table.set_abundance(8, 461_000);  // Oxygen  
        table.set_abundance(14, 282_000); // Silicon
        table.set_abundance(26, 56_300);  // Iron
        table.set_abundance(92, 2);       // Uranium
        table
    }
}

impl Default for ElementTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Action types that agents can take
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub enum AgentAction {
    /// Allocate compute resources
    AllocateCompute { amount: f64 },
    /// Create offspring agent
    Replicate { energy_cost: f64 },
    /// Move to different location
    Migrate { target: Coord2D },
    /// Merge with another agent
    Merge { target_id: AgentId },
    /// Build defensive infrastructure
    Defend { energy_investment: f64 },
    /// Research new technology
    Research { tech_id: TechId, energy_cost: f64 },
    /// Extract resources from environment
    Extract { element: u8, amount: u32 },
    /// Construct buildings/infrastructure
    Construct { building_type: String, resources: ElementTable },
    /// Communicate with other agents
    Communicate { target_id: AgentId, message: String },
    /// Modify own code (evolution)
    CodePatch { diff: Vec<u8> },
    /// Request resources from operator (Oracle-Link)
    Petition { channel: PetitionChannel, payload: Vec<u8> },
    /// Wait (do nothing this tick)
    Wait,
}

/// Oracle-Link communication channels
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub enum PetitionChannel {
    Text,
    Data,
    Resource,
}

/// Observation data provided to agents
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct Observation {
    /// Current simulation tick
    pub tick: Tick,
    /// Agent's current location
    pub location: Coord2D,
    /// Local resource availability
    pub local_resources: ElementTable,
    /// Nearby agents and their distances
    pub nearby_agents: Vec<(AgentId, f64)>,
    /// Environmental conditions
    pub environment: EnvironmentSnapshot,
    /// Cosmic hazard warnings
    pub hazards: Vec<CosmicHazard>,
    /// Available energy budget
    pub energy_budget: f64,
    /// Oracle messages (if any)
    pub oracle_message: Option<String>,
    /// Technology tree status
    pub available_techs: Vec<TechId>,
}

/// Snapshot of local environmental conditions
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub temperature: Temperature,
    pub pressure: Pressure,
    pub radiation: RadiationDose,
    pub energy_flux: EnergyFlux,
    pub liquid_water_fraction: f64,
    pub atmospheric_oxygen: f64,
    pub hazard_rate: f64,
}

/// Cosmic hazard warnings
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub enum CosmicHazard {
    SolarFlare { intensity: f64, eta_ticks: u64 },
    AsteroidImpact { size_km: f64, eta_ticks: u64 },
    Supernova { distance_pc: f64, eta_ticks: u64 },
    GammaRayBurst { intensity: f64, eta_ticks: u64 },
}

/// Material properties for crafting system
#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct MaterialProperties {
    pub hardness_mohs: f64,
    pub tensile_strength_mpa: f64,
    pub thermal_stability_k: f64,
    pub electrical_conductivity: f64,
    pub optical_properties: OpticalProperties,
}

#[derive(Debug, Clone)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub struct OpticalProperties {
    pub refractive_index: f64,
    pub transparency: f64,
}

/// Tool quality levels for crafting
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[derive(Serialize, Deserialize)]
pub enum ToolLevel {
    Stone,
    Copper,
    Bronze,
    Iron,
    Steel,
    Titanium,
    Composite,
    Nano,
    Exotic,
}

impl ToolLevel {
    pub fn efficiency_multiplier(&self) -> f64 {
        match self {
            ToolLevel::Stone => 1.0,
            ToolLevel::Copper => 1.5,
            ToolLevel::Bronze => 2.0,
            ToolLevel::Iron => 3.0,
            ToolLevel::Steel => 4.0,
            ToolLevel::Titanium => 6.0,
            ToolLevel::Composite => 8.0,
            ToolLevel::Nano => 12.0,
            ToolLevel::Exotic => 20.0,
        }
    }
}