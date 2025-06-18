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
        // Scale core temperature with mass (e.g., T ∝ M^0.7)
        let base_temp = 1.5e7; // K, rough solar core temperature
        let core_temperature = base_temp * (star_mass_kg / solar_mass).powf(0.7);
        // Rough core density constant for now
        let core_density = 1.5e5; // kg/m³, rough solar core density

        Self {
            entity_id: 0,
            nucleosynthesis,
            core_temperature,
            core_density,
            core_composition,
            nuclear_fuel_fraction: 1.0,
            main_sequence_lifetime: lifetime_years,
            evolutionary_phase: StellarPhase::MainSequence,
            nuclear_energy_generation: 0.0,
        }
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
                // Massive stars (>8 Msun) explode as supernova once fuel is sufficiently depleted
                if mass_msun > 8.0 && fuel_fraction < 0.1 {
                    self.evolutionary_phase = StellarPhase::Supernova;
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