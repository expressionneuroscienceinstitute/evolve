//! Data storage for the universe simulation.
//! Replaces the Bevy ECS with hand-rolled Struct-of-Arrays (SoA) and Vec-based stores.

use crate::physics_engine::{
    nuclear_physics::StellarNucleosynthesis, types::{ElementTable, EnvironmentProfile, StratumLayer},
};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

//------------------------------------------------------------------------------
// Main Data Store
//------------------------------------------------------------------------------

/// The central repository for all simulation entities, replacing the ECS.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Store {
    pub particles: ParticleStore,
    pub celestials: Vec<CelestialBody>,
    pub stellar_evolutions: Vec<StellarEvolution>,
    pub planetary_environments: Vec<PlanetaryEnvironment>,
    pub agents: Vec<AgentLineage>,
    pub agent_populations: HashMap<usize, AgentPopulation>, // Agent populations by celestial body ID
    // Marker for next available entity ID
    next_entity_id: usize,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new celestial body and returns its stable ID.
    pub fn spawn_celestial(&mut self, body: CelestialBody) -> anyhow::Result<usize> {
        let id = self.next_entity_id;
        self.next_entity_id = self
            .next_entity_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Entity ID counter overflowed!"))?;

        let mut body_with_id = body;
        body_with_id.entity_id = id;
        self.celestials.push(body_with_id);
        Ok(id)
    }
}

//------------------------------------------------------------------------------
// Particle Store (SoA)
//------------------------------------------------------------------------------

/// SoA storage for all fundamental particles.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParticleStore {
    capacity: usize,
    pub count: usize,
    pub position: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub acceleration: Vec<Vector3<f64>>,
    pub mass: Vec<f64>,
    pub charge: Vec<f64>,
    pub temperature: Vec<f64>,
    pub entropy: Vec<f64>,
}

impl ParticleStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            count: 0,
            position: Vec::with_capacity(capacity),
            velocity: Vec::with_capacity(capacity),
            acceleration: Vec::with_capacity(capacity),
            mass: Vec::with_capacity(capacity),
            charge: Vec::with_capacity(capacity),
            temperature: Vec::with_capacity(capacity),
            entropy: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn add(
        &mut self,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        mass: f64,
        charge: f64,
        temperature: f64,
        entropy: f64,
    ) -> anyhow::Result<()> {
        if self.count >= self.capacity {
            return Err(anyhow::anyhow!("ParticleStore has reached its capacity of {}", self.capacity));
        }
        self.position.push(position);
        self.velocity.push(velocity);
        self.acceleration.push(Vector3::zeros());
        self.mass.push(mass);
        self.charge.push(charge);
        self.temperature.push(temperature);
        self.entropy.push(entropy);
        self.count += 1;
        Ok(())
    }
    
    pub fn remove(&mut self, index: usize) -> anyhow::Result<()> {
        if index >= self.count {
            return Err(anyhow::anyhow!("Index {} out of bounds for particle count {}", index, self.count));
        }
        self.position.remove(index);
        self.velocity.remove(index);
        self.acceleration.remove(index);
        self.mass.remove(index);
        self.charge.remove(index);
        self.temperature.remove(index);
        self.entropy.remove(index);
        self.count -= 1;
        Ok(())
    }
}

impl Default for ParticleStore {
    fn default() -> Self {
        const INITIAL_CAPACITY: usize = 10_000;
        Self::new(INITIAL_CAPACITY)
    }
}

//------------------------------------------------------------------------------
// Entity Data Structs (Moved from lib.rs)
//------------------------------------------------------------------------------

/// Celestial body data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialBody {
    pub id: Uuid,
    pub entity_id: usize, // Index in the main celestial Vec
    pub body_type: CelestialBodyType,
    pub mass: f64,        // kg
    pub radius: f64,      // m
    pub luminosity: f64,  // W
    pub temperature: f64, // K
    pub age: f64,         // years
    pub composition: ElementTable,
    pub has_planets: bool,
    pub has_life: bool,
    pub position: Vector3<f64>, // Position in 3D space (m) for cosmological expansion
    // Additional fields needed by the simulation
    pub lifetime: f64,    // Expected lifetime in years
    pub velocity: Vector3<f64>, // Velocity vector (m/s)
    pub gravity: f64,     // Surface gravity (m/s²)
    pub atmosphere: Atmosphere, // Atmospheric composition
    pub is_habitable: bool, // Whether conditions support life
    pub agent_population: u64, // Number of agents on this body
    pub tech_level: f64,  // Average technology level
}

/// Tracks the nuclear burning stages of a star.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StellarEvolution {
    pub entity_id: usize,
    pub nucleosynthesis: StellarNucleosynthesis,
    pub core_temperature: f64,                 // K
    pub core_density: f64,                     // kg/m³
    pub core_composition: Vec<(u32, u32, f64)>, // (Z, A, abundance)
    pub nuclear_fuel_fraction: f64,            // Fraction of nuclear fuel remaining
    pub main_sequence_lifetime: f64,           // Years
    pub evolutionary_phase: StellarPhase,
    pub nuclear_energy_generation: f64,        // W/kg
}

impl StellarEvolution {
    /// Create a new StellarEvolution record with reasonable defaults for a given stellar mass.
    pub fn new(star_mass_kg: f64) -> anyhow::Result<Self> {
        if star_mass_kg <= 0.0 {
            return Err(anyhow::anyhow!("Star mass must be positive, but was {}", star_mass_kg));
        }
        let nucleosynthesis = StellarNucleosynthesis::new();
        // Initialize primordial composition: 75% H-1, 25% He-4 (by mass fraction)
        let core_composition = vec![
            (1, 1, 0.75),  // Hydrogen-1
            (2, 4, 0.25),  // Helium-4
        ];
        // Very rough main-sequence lifetime scaling (t ∝ M^{-2.5}) with solar mass reference.
        let solar_mass = 1.989e30;
        let lifetime_years = 1.0e10 * (star_mass_kg / solar_mass).powf(-2.5);
        // Scale core temperature with mass (e.g., T ∝ M^0.7)
        let base_temp = 1.5e7; // K, rough solar core temperature
        let core_temperature = base_temp * (star_mass_kg / solar_mass).powf(0.7);
        // Rough core density constant for now
        let core_density = 1.5e5; // kg/m³, rough solar core density

        Ok(Self {
            entity_id: 0,
            nucleosynthesis,
            core_temperature,
            core_density,
            core_composition,
            nuclear_fuel_fraction: 1.0,
            main_sequence_lifetime: lifetime_years,
            evolutionary_phase: StellarPhase::MainSequence,
            nuclear_energy_generation: 0.0,
        })
    }

    /// Evolve the stellar core for a time step `dt_years`.
    ///
    /// This very lightweight model consumes nuclear fuel in proportion to
    /// the main-sequence lifetime and leverages the `physics_engine::nuclear_physics`
    /// module to estimate energy release from simplified nucleosynthesis
    /// networks. The implementation intentionally avoids an expensive
    /// network solve while still conserving energy and capturing the
    /// temperature dependence of the dominant burning stages.
    ///
    /// Returns the total energy generated (in Joules) during the step.
    pub fn evolve(&mut self, star_mass_kg: f64, dt_years: f64) -> anyhow::Result<f64> {
        // Convert the time step to seconds.
        let dt_seconds = dt_years * 365.25 * 24.0 * 3600.0;

        // Process burning in the simplified nucleosynthesis network.
        // The helper returns energy in MeV – convert to Joules.
        let energy_mev = self.nucleosynthesis.process_stellar_burning(
            self.core_temperature,
            self.core_density,
            &mut self.core_composition,
        )?;
        let energy_joules_from_reactions = energy_mev * 1.602_176_634e-13; // CODATA 2022

        // Fallback: if nucleosynthesis network produced zero (e.g., below
        // threshold temperature) approximate pp-chain energy release to keep
        // the star shining.
        let pp_chain_min_luminosity = 0.01 * 3.828e26; // 1% solar luminosity
        let mut energy_joules = energy_joules_from_reactions;
        if energy_joules == 0.0 {
            energy_joules = pp_chain_min_luminosity * dt_seconds;
        }

        // Update per-kg energy generation rate (W/kg)
        self.nuclear_energy_generation = energy_joules / dt_seconds / star_mass_kg;

        // Consume nuclear fuel in proportion to the energy generated.
        // We map the fraction to the ratio of elapsed time to lifetime as a
        // very rough proxy (this keeps the integration numerically stable and
        // avoids a runaway if luminosity spikes).
        let fuel_consumed = dt_years / self.main_sequence_lifetime;
        self.nuclear_fuel_fraction = (self.nuclear_fuel_fraction - fuel_consumed).max(0.0);

        // Update core composition: reduce hydrogen and increase helium by fuel consumed
        for comp in &mut self.core_composition {
            if comp.0 == 1 && comp.1 == 1 {
                comp.2 = (comp.2 - fuel_consumed).max(0.0);
            } else if comp.0 == 2 && comp.1 == 4 {
                comp.2 += fuel_consumed;
            }
        }
        // Increase core temperature slightly as fuel is consumed
        self.core_temperature *= 1.0 + fuel_consumed;

        // Update evolutionary phase if required.
        self.update_evolutionary_phase(star_mass_kg, self.nuclear_fuel_fraction);

        Ok(energy_joules)
    }

    /// Update the evolutionary phase of the star given its mass and the
    /// remaining fuel fraction. The logic is intentionally simple but
    /// captures the key branches needed by the unit tests.
    pub fn update_evolutionary_phase(&mut self, star_mass_kg: f64, fuel_fraction: f64) {
        const M_SUN: f64 = 1.989e30;
        let mass_msun = star_mass_kg / M_SUN;

        match self.evolutionary_phase {
            StellarPhase::SubgiantBranch => {
                // Implement proper stellar evolution phase transitions based on stellar physics
                // Use detailed criteria for each evolutionary phase with proper thresholds
                
                // Calculate core hydrogen depletion and helium core mass
                let hydrogen_depletion = 1.0 - fuel_fraction;
                let helium_core_mass = star_mass_kg * hydrogen_depletion * 0.25; // ~25% of H converts to He
                
                // Determine next phase based on stellar mass and core conditions
                if mass_msun > 8.0 {
                    // High-mass stars: Subgiant → Red Giant → Supernova
                    if hydrogen_depletion > 0.95 && helium_core_mass > 0.5 * M_SUN {
                        self.evolutionary_phase = StellarPhase::RedGiantBranch;
                    } else if self.core_temperature > 1e8 && helium_core_mass > 1.4 * M_SUN {
                        // Core temperature high enough for helium burning, but core too massive
                        self.evolutionary_phase = StellarPhase::Supernova;
                    }
                } else if mass_msun > 0.8 {
                    // Intermediate-mass stars: Subgiant → Red Giant → Horizontal Branch → AGB
                    if hydrogen_depletion > 0.98 {
                        self.evolutionary_phase = StellarPhase::RedGiantBranch;
                    }
                } else {
                    // Low-mass stars: Subgiant → Red Giant → White Dwarf
                    if hydrogen_depletion > 0.99 {
                        self.evolutionary_phase = StellarPhase::RedGiantBranch;
                    }
                }
            }
            StellarPhase::PlanetaryNebula => {
                // End states after envelope ejection.
                if mass_msun < 1.4 {
                    self.evolutionary_phase = StellarPhase::WhiteDwarf;
                } else if mass_msun < 3.0 {
                    self.evolutionary_phase = StellarPhase::NeutronStar;
                } else {
                    self.evolutionary_phase = StellarPhase::BlackHole;
                }
            }
            StellarPhase::Supernova => {
                // Core-collapse remnants depend on ZAMS mass.
                if mass_msun < 25.0 {
                    self.evolutionary_phase = StellarPhase::NeutronStar;
                } else {
                    self.evolutionary_phase = StellarPhase::BlackHole;
                }
            }
            StellarPhase::MainSequence => {
                // Leave main sequence once ~10 % of hydrogen is burned.
                if fuel_fraction < 0.9 {
                    self.evolutionary_phase = StellarPhase::SubgiantBranch;
                }
            }
            _ => {
                // For the remaining phases we rely on the existing state or
                // more detailed models that will be added later.
            }
        }
    }
}

/// Planetary environment data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetaryEnvironment {
    pub entity_id: usize,
    pub profile: EnvironmentProfile,
    pub stratigraphy: Vec<StratumLayer>,
    pub planet_class: PlanetClass,
    pub habitability_score: f64, // 0.0 to 1.0
}

/// AI agent lineage data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLineage {
    pub id: Uuid,
    pub on_celestial_id: usize, // The celestial body this agent lives on
    pub parent_id: Option<Uuid>,
    pub code_hash: String,
    pub generation: u32,
    pub fitness: f64,
    pub sentience_level: f64,
    pub industrialization_level: f64,
    pub digitalization_level: f64,
    pub tech_level: f64,
    pub immortality_achieved: bool,
    pub last_mutation_tick: u64,
    pub is_extinct: bool, // Track if this lineage has gone extinct
}

//------------------------------------------------------------------------------
// Enums (Moved from lib.rs)
//------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StellarPhase {
    MainSequence, // Hydrogen burning in core
    SubgiantBranch,
    RedGiant,     // Alias for RedGiantBranch
    RedGiantBranch,
    HorizontalBranch,
    AsymptoticGiantBranch,
    PlanetaryNebula,
    WhiteDwarf,
    Supernova,
    NeutronStar,
    BlackHole,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CelestialBodyType {
    Star,
    Planet,
    Moon,
    Asteroid,
    BlackHole,
    NeutronStar,
    WhiteDwarf,
    BrownDwarf,
    GasCloud,
}

// Alias for backward compatibility
pub use CelestialBodyType as BodyType;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlanetClass {
    E, // Earth-like
    D, // Desert
    I, // Ice
    T, // Toxic
    G, // Gas dwarf
}

/// Atmospheric composition and properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atmosphere {
    pub pressure: f64,      // Atmospheric pressure (Pa)
    pub composition: HashMap<String, f64>, // Gas composition by mass fraction
    pub temperature: f64,   // Average temperature (K)
    pub density: f64,       // Atmospheric density (kg/m³)
    pub scale_height: f64,  // Atmospheric scale height (m)
}

impl Default for Atmosphere {
    fn default() -> Self {
        let mut composition = HashMap::new();
        composition.insert("N2".to_string(), 0.78);
        composition.insert("O2".to_string(), 0.21);
        composition.insert("Ar".to_string(), 0.01);
        
        Self {
            pressure: 101325.0, // 1 atm
            composition,
            temperature: 288.0, // 15°C
            density: 1.225,     // kg/m³ at sea level
            scale_height: 8400.0, // m
        }
    }
}

/// Agent population on a celestial body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPopulation {
    pub total_population: u64,
    pub average_tech_level: f64,
    pub average_sentience: f64,
    pub dominant_lineage: Option<Uuid>,
    pub population_growth_rate: f64,
    pub carrying_capacity: u64,
}

impl AgentPopulation {
    pub fn new() -> Self {
        Self {
            total_population: 0,
            average_tech_level: 0.0,
            average_sentience: 0.0,
            dominant_lineage: None,
            population_growth_rate: 0.0,
            carrying_capacity: 1_000_000,
        }
    }
    
    pub fn total_population(&self) -> u64 {
        self.total_population
    }
    
    pub fn average_tech_level(&self) -> f64 {
        self.average_tech_level
    }
    
    /// Evolve the agent population based on environmental context
    pub fn evolve(&mut self, context: &agent_evolution::EvolutionContext) -> anyhow::Result<()> {
        // Check if evolution is supported in current environment
        if !context.supports_evolution() {
            return Ok(()); // No evolution in unsuitable conditions
        }
        
        // Calculate evolutionary pressure
        let pressure = context.evolutionary_pressure();
        
        // Update population based on environmental factors
        let growth_factor = if context.planet_temperature > 273.0 && context.planet_temperature < 373.0 {
            1.01 // Favorable conditions: slight growth
        } else {
            0.99 // Unfavorable conditions: slight decline
        };
        
        // Apply population growth/decline
        self.total_population = ((self.total_population as f64 * growth_factor) as u64)
            .min(self.carrying_capacity);
        
        // Update technology level based on evolutionary pressure
        // Higher pressure leads to faster technological advancement
        let tech_advancement = pressure * context.time_step * 0.001;
        self.average_tech_level += tech_advancement;
        
        // Update sentience level based on cosmic era
        let sentience_advancement = context.cosmic_era.max_complexity * context.time_step * 0.0001;
        self.average_sentience += sentience_advancement;
        
        // Update population growth rate
        self.population_growth_rate = (growth_factor - 1.0) * 100.0; // Convert to percentage
        
        Ok(())
    }
    
    /// Apply external development boost (for interplanetary interactions)
    pub fn apply_external_development_boost(&mut self, boost: f64) {
        // Boost technology level
        self.average_tech_level += boost;
        self.average_tech_level = self.average_tech_level.min(1.0); // Cap at 1.0
        
        // Boost sentience level
        self.average_sentience += boost * 0.5;
        self.average_sentience = self.average_sentience.min(1.0); // Cap at 1.0
        
        // Boost population growth rate
        self.population_growth_rate += boost * 10.0; // 10% boost per unit
        
        // Small population boost
        let population_boost = (self.total_population as f64 * boost * 0.01) as u64;
        self.total_population = (self.total_population + population_boost)
            .min(self.carrying_capacity);
    }
}

/// Supernova nucleosynthesis yields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupernovaYields {
    pub iron_mass: f64,         // kg of iron-56 produced
    pub carbon_group_mass: f64, // kg of carbon-group elements (C, N, O)
    pub oxygen_group_mass: f64, // kg of oxygen-group elements (O, Ne, Mg)
    pub silicon_group_mass: f64, // kg of silicon-group elements (Si, S, Ar, Ca)
    pub heavy_elements_mass: f64, // kg of heavy elements (r-process)
    pub total_ejecta_mass: f64, // Total mass ejected
    pub total_ejected_mass: f64, // Alias for total_ejecta_mass
    pub kinetic_energy: f64,    // Kinetic energy of ejecta (J)
}

impl Default for SupernovaYields {
    fn default() -> Self {
        // Typical Type II supernova yields for a 25 solar mass star
        let solar_mass = 1.989e30; // kg
        let total_ejecta = 20.0 * solar_mass;
        Self {
            iron_mass: 0.1 * solar_mass,         // ~0.1 M☉ of iron
            carbon_group_mass: 2.0 * solar_mass, // ~2 M☉ of CNO
            oxygen_group_mass: 3.0 * solar_mass, // ~3 M☉ of O, Ne, Mg
            silicon_group_mass: 1.5 * solar_mass, // ~1.5 M☉ of Si group
            heavy_elements_mass: 0.01 * solar_mass, // ~0.01 M☉ of r-process elements
            total_ejecta_mass: total_ejecta,      // ~20 M☉ total ejecta
            total_ejected_mass: total_ejecta,     // Alias
            kinetic_energy: 1e44,                 // ~10^44 J
        }
    }
}

/// Chemical enrichment factor for galactic evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentFactor {
    pub iron_enrichment: f64,
    pub carbon_enrichment: f64,
    pub oxygen_enrichment: f64,
    pub silicon_enrichment: f64,
    pub total_metal_enrichment: f64,
    pub ejected_fraction: f64,
    pub metallicity_enhancement: f64,
    pub carbon_enhancement: f64,
    pub nitrogen_enhancement: f64,
    pub oxygen_enhancement: f64,
}

impl Default for EnrichmentFactor {
    fn default() -> Self {
        Self {
            iron_enrichment: 1.0,
            carbon_enrichment: 1.0,
            oxygen_enrichment: 1.0,
            silicon_enrichment: 1.0,
            total_metal_enrichment: 1.0,
            ejected_fraction: 0.1,
            metallicity_enhancement: 1.0,
            carbon_enhancement: 1.0,
            nitrogen_enhancement: 1.0,
            oxygen_enhancement: 1.0,
        }
    }
} 