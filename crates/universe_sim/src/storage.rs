//! Data storage for the universe simulation.
//! Replaces the Bevy ECS with hand-rolled Struct-of-Arrays (SoA) and Vec-based stores.

use crate::physics_engine::{
    nuclear_physics::StellarNucleosynthesis, ElementTable, EnvironmentProfile, StratumLayer,
};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    // Marker for next available entity ID
    next_entity_id: usize,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new celestial body and returns its stable ID.
    pub fn spawn_celestial(&mut self, body: CelestialBody) -> usize {
        let id = self.next_entity_id;
        self.next_entity_id += 1;

        let mut body_with_id = body;
        body_with_id.entity_id = id;
        self.celestials.push(body_with_id);
        id
    }
}

//------------------------------------------------------------------------------
// Particle Store (SoA)
//------------------------------------------------------------------------------

/// SoA storage for all fundamental particles.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ParticleStore {
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
    pub fn new() -> Self {
        // Pre-allocate for performance, assuming a reasonably large number of particles.
        const INITIAL_CAPACITY: usize = 10_000;
        Self {
            count: 0,
            position: Vec::with_capacity(INITIAL_CAPACITY),
            velocity: Vec::with_capacity(INITIAL_CAPACITY),
            acceleration: Vec::with_capacity(INITIAL_CAPACITY),
            mass: Vec::with_capacity(INITIAL_CAPACITY),
            charge: Vec::with_capacity(INITIAL_CAPACITY),
            temperature: Vec::with_capacity(INITIAL_CAPACITY),
            entropy: Vec::with_capacity(INITIAL_CAPACITY),
        }
    }

    pub fn add(
        &mut self,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        mass: f64,
        charge: f64,
        temperature: f64,
        entropy: f64,
    ) {
        self.position.push(position);
        self.velocity.push(velocity);
        self.acceleration.push(Vector3::zeros());
        self.mass.push(mass);
        self.charge.push(charge);
        self.temperature.push(temperature);
        self.entropy.push(entropy);
        self.count += 1;
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
    pub fn new(star_mass_kg: f64) -> Self {
        let mut nucleosynthesis = StellarNucleosynthesis::new();
        // Initialize primordial composition: 75% H-1, 25% He-4 (by mass fraction)
        let core_composition = vec![
            (1, 1, 0.75),  // Hydrogen-1
            (2, 4, 0.25),  // Helium-4
        ];
        // Very rough main-sequence lifetime scaling (t ∝ M^{-2.5}) with solar mass reference.
        let solar_mass = 1.989e30;
        let lifetime_years = 1.0e10 * (star_mass_kg / solar_mass).powf(-2.5);

        Self {
            entity_id: 0,
            nucleosynthesis,
            core_temperature: 1.5e7, // K, rough solar core temperature
            core_density: 1.5e5,      // kg/m³, rough solar core density
            core_composition,
            nuclear_fuel_fraction: 1.0,
            main_sequence_lifetime: lifetime_years,
            evolutionary_phase: StellarPhase::MainSequence,
            nuclear_energy_generation: 0.0,
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
}

//------------------------------------------------------------------------------
// Enums (Moved from lib.rs)
//------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StellarPhase {
    MainSequence, // Hydrogen burning in core
    SubgiantBranch,
    RedGiantBranch,
    HorizontalBranch,
    AsymptoticGiantBranch,
    PlanetaryNebula,
    WhiteDwarf,
    Supernova,
    NeutronStar,
    BlackHole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CelestialBodyType {
    Star,
    Planet,
    Moon,
    Asteroid,
    BlackHole,
    NeutronStar,
    WhiteDwarf,
    BrownDwarf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlanetClass {
    E, // Earth-like
    D, // Desert
    I, // Ice
    T, // Toxic
    G, // Gas dwarf
} 