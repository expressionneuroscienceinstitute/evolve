//! Universe Simulation Core Library
//! 
//! Implements the complete universe simulation from Big Bang to far future
//! with autonomous AI agents evolving toward immortality.

use physics_engine::{PhysicsEngine, PhysicsState, ElementTable, EnvironmentProfile};
use bevy_ecs::prelude::*;
use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use uuid::Uuid;

pub mod world;
pub mod cosmic_era;
pub mod evolution;
pub mod persistence;
pub mod config;

pub use physics_engine;

/// Core universe simulation structure
pub struct UniverseSimulation {
    pub world: World,                          // ECS world
    pub physics_engine: PhysicsEngine,         // Physics simulation
    pub current_tick: u64,                     // Simulation time
    pub tick_span_years: f64,                  // Years per tick (default 1M)
    pub target_ups: f64,                       // Updates per second target
    pub cosmic_era: cosmic_era::CosmicEra,     // Current era
    pub config: config::SimulationConfig,      // Configuration
}

/// Celestial body component
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct CelestialBody {
    pub id: Uuid,
    pub body_type: CelestialBodyType,
    pub mass: f64,                             // kg
    pub radius: f64,                           // m
    pub luminosity: f64,                       // W
    pub temperature: f64,                      // K
    pub age: f64,                             // years
    pub composition: ElementTable,
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

/// Planetary environment component
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct PlanetaryEnvironment {
    pub profile: EnvironmentProfile,
    pub stratigraphy: Vec<physics_engine::StratumLayer>,
    pub planet_class: PlanetClass,
    pub habitability_score: f64,              // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanetClass {
    E, // Earth-like
    D, // Desert
    I, // Ice
    T, // Toxic
    G, // Gas dwarf
}

/// AI agent lineage component
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct AgentLineage {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub code_hash: String,                     // Hash of agent code
    pub generation: u32,
    pub fitness: f64,
    pub sentience_level: f64,                  // Progress toward sentience
    pub industrialization_level: f64,          // Energy output level
    pub digitalization_level: f64,             // Digital substrate usage
    pub tech_level: f64,                      // Technology advancement
    pub immortality_achieved: bool,
    pub last_mutation_tick: u64,
}

impl UniverseSimulation {
    /// Create a new universe simulation
    pub fn new(config: config::SimulationConfig) -> Result<Self> {
        let world = World::new();
        
        let physics_engine = PhysicsEngine::new(1e-6)?; // 1 microsecond timestep
        
        Ok(Self {
            world,
            physics_engine,
            current_tick: 0,
            tick_span_years: config.tick_span_years,
            target_ups: config.target_ups,
            cosmic_era: cosmic_era::CosmicEra::ParticleSoup,
            config,
        })
    }

    /// Initialize with Big Bang conditions
    pub fn init_big_bang(&mut self) -> Result<()> {
        // Set initial cosmic era
        self.cosmic_era = cosmic_era::CosmicEra::ParticleSoup;
        
        // Create initial particle soup
        self.spawn_initial_particles()?;
        
        // Initialize cosmic background
        self.init_cosmic_background()?;
        
        Ok(())
    }

    /// Spawn initial particles after Big Bang
    fn spawn_initial_particles(&mut self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Create primordial hydrogen and helium
        let num_particles = self.config.initial_particle_count;
        
        for _ in 0..num_particles {
            let position = Vector3::new(
                rng.gen_range(-1e15..1e15), // ±1000 light-years
                rng.gen_range(-1e15..1e15),
                rng.gen_range(-1e15..1e15),
            );
            
            let velocity = Vector3::new(
                rng.gen_range(-1e6..1e6), // ±1000 km/s
                rng.gen_range(-1e6..1e6),
                rng.gen_range(-1e6..1e6),
            );
            
            // 75% hydrogen, 25% helium (by number)
            let (mass, charge) = if rng.gen::<f64>() < 0.75 {
                // Hydrogen
                (1.67262e-27, 1.602176e-19) // Proton
            } else {
                // Helium
                (6.64466e-27, 2.0 * 1.602176e-19) // Alpha particle
            };
            
            let physics_state = PhysicsState {
                position,
                velocity,
                acceleration: Vector3::zeros(),
                mass,
                charge,
                temperature: 3000.0, // 3000 K initial temperature
                entropy: 1e-23,
            };
            
            self.world.spawn(physics_state);
        }
        
        Ok(())
    }

    /// Initialize cosmic background conditions
    fn init_cosmic_background(&mut self) -> Result<()> {
        // Set up cosmic microwave background
        // Set up dark matter (simplified as additional gravitational sources)
        // Initialize fundamental constants
        
        Ok(())
    }

    /// Run one simulation tick
    pub fn tick(&mut self) -> Result<()> {
        // Update cosmic era based on time
        self.update_cosmic_era();
        
        // Update physics for all entities
        self.update_physics()?;
        
        // Update agent evolution
        self.update_agent_evolution()?;
        
        // Update cosmic-scale processes
        self.update_cosmic_processes()?;
        
        // Check win/lose conditions
        self.check_victory_conditions()?;
        
        // Increment tick counter
        self.current_tick += 1;
        
        Ok(())
    }

    /// Update cosmic era based on current time
    fn update_cosmic_era(&mut self) {
        let age_gyr = self.current_tick as f64 * self.tick_span_years / 1e9;
        
        self.cosmic_era = match age_gyr {
            x if x < 0.0003 => cosmic_era::CosmicEra::ParticleSoup,
            x if x <= 1.0 => cosmic_era::CosmicEra::Starbirth,
            x if x < 5.0 => cosmic_era::CosmicEra::PlanetaryAge,
            x if x < 10.0 => cosmic_era::CosmicEra::Biogenesis,
            x if x < 13.0 => cosmic_era::CosmicEra::DigitalEvolution,
            _ => cosmic_era::CosmicEra::PostIntelligence,
        };
    }

    /// Update physics simulation
    fn update_physics(&mut self) -> Result<()> {
        // Get all physics states
        let mut physics_states: Vec<PhysicsState> = self.world
            .query::<&PhysicsState>()
            .iter(&self.world)
            .map(|state| state.clone())
            .collect();
        
        // Run physics step
        self.physics_engine.step(&mut physics_states)?;
        
        // Update entities with new physics states
        let mut query = self.world.query::<&mut PhysicsState>();
        for (i, mut state) in query.iter_mut(&mut self.world).enumerate() {
            if let Some(new_state) = physics_states.get(i) {
                *state = new_state.clone();
            }
        }
        
        Ok(())
    }

    /// Update agent evolution and behavior
    fn update_agent_evolution(&mut self) -> Result<()> {
        // This will be implemented in the agents module
        // For now, just a placeholder
        Ok(())
    }

    /// Update cosmic-scale processes (star formation, supernovae, etc.)
    fn update_cosmic_processes(&mut self) -> Result<()> {
        match self.cosmic_era {
            cosmic_era::CosmicEra::Starbirth => {
                self.process_star_formation()?;
            },
            cosmic_era::CosmicEra::PlanetaryAge => {
                self.process_planet_formation()?;
            },
            cosmic_era::CosmicEra::Biogenesis => {
                self.process_life_emergence()?;
            },
            _ => {}
        }
        
        Ok(())
    }

    /// Process star formation from gas clouds
    fn process_star_formation(&mut self) -> Result<()> {
        // Simplified star formation
        // In reality, this would involve complex hydrodynamics
        
        Ok(())
    }

    /// Process planet formation around stars
    fn process_planet_formation(&mut self) -> Result<()> {
        // Simplified planetary accretion
        
        Ok(())
    }

    /// Process emergence of life on suitable planets
    fn process_life_emergence(&mut self) -> Result<()> {
        // Check environmental conditions and spawn life
        
        Ok(())
    }

    /// Check victory and defeat conditions
    fn check_victory_conditions(&mut self) -> Result<()> {
        // Check if any lineage has achieved the goal chain:
        // Sentience → Industrialization → Digitalization → Trans-Tech → Immortality
        
        let mut query = self.world.query::<&AgentLineage>();
        for lineage in query.iter(&self.world) {
            if lineage.immortality_achieved {
                tracing::info!("Victory! Lineage {} achieved immortality!", lineage.id);
                // Continue simulation (no end state)
            }
        }
        
        Ok(())
    }

    /// Get current universe age in years
    pub fn universe_age_years(&self) -> f64 {
        self.current_tick as f64 * self.tick_span_years
    }

    /// Get current universe age in billion years
    pub fn universe_age_gyr(&self) -> f64 {
        self.universe_age_years() / 1e9
    }

    /// Get simulation statistics
    pub fn get_stats(&mut self) -> SimulationStats {
        let physics_count = self.world.query::<&PhysicsState>().iter(&self.world).count();
        let celestial_count = self.world.query::<&CelestialBody>().iter(&self.world).count();
        let planet_count = self.world.query::<&PlanetaryEnvironment>().iter(&self.world).count();
        let lineage_count = self.world.query::<&AgentLineage>().iter(&self.world).count();
        
        SimulationStats {
            current_tick: self.current_tick,
            universe_age_gyr: self.universe_age_gyr(),
            cosmic_era: self.cosmic_era.clone(),
            particle_count: physics_count,
            celestial_body_count: celestial_count,
            planet_count,
            lineage_count,
            target_ups: self.target_ups,
        }
    }
}

/// Simulation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStats {
    pub current_tick: u64,
    pub universe_age_gyr: f64,
    pub cosmic_era: cosmic_era::CosmicEra,
    pub particle_count: usize,
    pub celestial_body_count: usize,
    pub planet_count: usize,
    pub lineage_count: usize,
    pub target_ups: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universe_creation() {
        let config = config::SimulationConfig::default();
        let sim = UniverseSimulation::new(config).unwrap();
        
        assert_eq!(sim.current_tick, 0);
        assert_eq!(sim.universe_age_gyr(), 0.0);
    }

    #[test]
    fn test_big_bang_initialization() {
        let config = config::SimulationConfig::default();
        let mut sim = UniverseSimulation::new(config).unwrap();
        
        sim.init_big_bang().unwrap();
        
        let stats = sim.get_stats();
        assert!(stats.particle_count > 0);
    }

    #[test]
    fn test_cosmic_era_progression() {
        let config = config::SimulationConfig::default();
        let mut sim = UniverseSimulation::new(config).unwrap();
        
        // Test era transitions
        sim.current_tick = 0;
        sim.update_cosmic_era();
        assert!(matches!(sim.cosmic_era, cosmic_era::CosmicEra::ParticleSoup));
        
        sim.current_tick = 1000; // 1 Gyr
        sim.update_cosmic_era();
        assert!(matches!(sim.cosmic_era, cosmic_era::CosmicEra::Starbirth));
    }
}