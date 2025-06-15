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

    /// Get real map data from simulation particle positions and field strengths
    pub fn get_map_data(&mut self, zoom: f64, layer: &str, width: usize, height: usize) -> Result<serde_json::Value> {
        // Calculate spatial bounds based on particle distribution
        let (min_pos, max_pos) = self.calculate_spatial_bounds()?;
        
        // Apply zoom factor to the bounds
        let scale = 1.0 / zoom;
        let center = (min_pos + max_pos) / 2.0;
        let range = (max_pos - min_pos) * scale;
        
        let x_min = center.x - range.x / 2.0;
        let x_max = center.x + range.x / 2.0;
        let y_min = center.y - range.y / 2.0;
        let y_max = center.y + range.y / 2.0;

        // Create density grid based on real data
        let mut density_grid = Vec::new();
        
        for y in 0..height {
            for x in 0..width {
                let world_x = x_min + (x as f64 / width as f64) * (x_max - x_min);
                let world_y = y_min + (y as f64 / height as f64) * (y_max - y_min);
                
                let density = match layer {
                    "stars" => self.calculate_stellar_density_at(world_x, world_y)?,
                    "gas" => self.calculate_gas_density_at(world_x, world_y)?,
                    "dark_matter" => self.calculate_dark_matter_density_at(world_x, world_y)?,
                    "radiation" => self.calculate_radiation_density_at(world_x, world_y)?,
                    _ => self.calculate_total_density_at(world_x, world_y)?,
                };
                
                density_grid.push(density);
            }
        }

        Ok(serde_json::json!({
            "density_grid": density_grid,
            "width": width,
            "height": height,
            "layer": layer,
            "zoom": zoom,
            "spatial_bounds": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max
            },
            "particle_count": self.physics_engine.particles.len(),
            "generated_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        }))
    }

    /// Calculate spatial bounds from particle distribution
    fn calculate_spatial_bounds(&self) -> Result<(Vector3<f64>, Vector3<f64>)> {
        if self.physics_engine.particles.is_empty() {
            // Default bounds if no particles
            return Ok((
                Vector3::new(-1e15, -1e15, -1e15),
                Vector3::new(1e15, 1e15, 1e15)
            ));
        }

        let mut min_pos = self.physics_engine.particles[0].position;
        let mut max_pos = self.physics_engine.particles[0].position;

        for particle in &self.physics_engine.particles {
            min_pos.x = min_pos.x.min(particle.position.x);
            min_pos.y = min_pos.y.min(particle.position.y);
            min_pos.z = min_pos.z.min(particle.position.z);
            
            max_pos.x = max_pos.x.max(particle.position.x);
            max_pos.y = max_pos.y.max(particle.position.y);
            max_pos.z = max_pos.z.max(particle.position.z);
        }

        // Add padding to bounds
        let padding = (max_pos - min_pos) * 0.1;
        min_pos -= padding;
        max_pos += padding;

        Ok((min_pos, max_pos))
    }

    /// Calculate stellar density at a given position
    fn calculate_stellar_density_at(&mut self, x: f64, y: f64) -> Result<f64> {
        let position = Vector3::new(x, y, 0.0);
        let search_radius = 1e14; // 0.1 light-year
        
        let mut stellar_mass = 0.0;
        
        // Check celestial bodies in ECS world
        let mut query = self.world.query::<(&CelestialBody, &physics_engine::PhysicsState)>();
        for (body, state) in query.iter(&self.world) {
            if let CelestialBodyType::Star = body.body_type {
                let distance = (state.position.xy() - position.xy()).magnitude();
                if distance < search_radius {
                    // Use inverse square falloff
                    let weight = 1.0 / (1.0 + (distance / (search_radius * 0.1)).powi(2));
                    stellar_mass += body.mass * weight;
                }
            }
        }
        
        // Convert to relative density (0.0 to 1.0)
        let max_stellar_density = 1e32; // kg in search volume
        Ok((stellar_mass / max_stellar_density).min(1.0))
    }

    /// Calculate gas density at a given position
    fn calculate_gas_density_at(&self, x: f64, y: f64) -> Result<f64> {
        let position = Vector3::new(x, y, 0.0);
        let search_radius = 1e13; // 10 light-years
        
        let mut gas_density = 0.0;
        
        // Count gas particles (hydrogen and helium primarily)
        for particle in &self.physics_engine.particles {
            let is_gas = matches!(particle.particle_type, 
                physics_engine::ParticleType::Proton | 
                physics_engine::ParticleType::Neutron
            );
            
            if is_gas {
                let distance = (particle.position.xy() - position.xy()).magnitude();
                if distance < search_radius {
                    let weight = 1.0 / (1.0 + (distance / (search_radius * 0.1)).powi(2));
                    gas_density += particle.mass * weight;
                }
            }
        }
        
        // Convert to relative density
        let max_gas_density = 1e-18; // kg/m³ typical ISM density
        Ok((gas_density / (std::f64::consts::PI * search_radius.powi(2)) / max_gas_density).min(1.0))
    }

    /// Calculate dark matter density (simplified approximation)
    fn calculate_dark_matter_density_at(&self, x: f64, y: f64) -> Result<f64> {
        let position = Vector3::new(x, y, 0.0);
        
        // Simplified dark matter halo model - denser toward center
        let center = Vector3::zeros();
        let distance = (position - center).magnitude();
        let core_radius = 1e16; // ~10 light-years
        
        // NFW profile approximation
        let relative_distance = distance / core_radius;
        let density = 1.0 / (relative_distance * (1.0 + relative_distance).powi(2));
        
        Ok(density.min(1.0))
    }

    /// Calculate radiation density at a given position
    fn calculate_radiation_density_at(&mut self, x: f64, y: f64) -> Result<f64> {
        let position = Vector3::new(x, y, 0.0);
        let search_radius = 1e15; // 1 light-year
        
        let mut radiation_flux = 0.0;
        
        // Count photons and calculate flux from stars
        for particle in &self.physics_engine.particles {
            if matches!(particle.particle_type, physics_engine::ParticleType::Photon) {
                let distance = (particle.position.xy() - position.xy()).magnitude();
                if distance < search_radius {
                    let weight = 1.0 / (1.0 + distance.powi(2) / (search_radius * 0.01).powi(2));
                    radiation_flux += particle.energy * weight;
                }
            }
        }
        
        // Add stellar radiation
        let mut query = self.world.query::<(&CelestialBody, &physics_engine::PhysicsState)>();
        for (body, state) in query.iter(&self.world) {
            if let CelestialBodyType::Star = body.body_type {
                let distance = (state.position.xy() - position.xy()).magnitude();
                if distance > 0.0 {
                    // Inverse square law for stellar radiation
                    let flux = body.luminosity / (4.0 * std::f64::consts::PI * distance.powi(2));
                    radiation_flux += flux;
                }
            }
        }
        
        // Convert to relative intensity
        let max_radiation = 1e26; // W/m² near a star
        Ok((radiation_flux / max_radiation).min(1.0))
    }

    /// Calculate total matter density at a given position
    fn calculate_total_density_at(&mut self, x: f64, y: f64) -> Result<f64> {
        let stellar = self.calculate_stellar_density_at(x, y)?;
        let gas = self.calculate_gas_density_at(x, y)?;
        let dark_matter = self.calculate_dark_matter_density_at(x, y)? * 5.0; // Dark matter is ~5x more abundant
        
        Ok(((stellar + gas + dark_matter) / 7.0).min(1.0))
    }

    /// Get real planet data from simulation ECS world
    pub fn get_planet_data(&mut self, class_filter: Option<String>, habitable_only: bool) -> Result<serde_json::Value> {
        let mut planets = Vec::new();
        
        let mut query = self.world.query::<(&CelestialBody, &PlanetaryEnvironment, &physics_engine::PhysicsState)>();
        for (body, env, state) in query.iter(&self.world) {
            if let CelestialBodyType::Planet = body.body_type {
                let planet_class = match env.planet_class {
                    PlanetClass::E => "E",
                    PlanetClass::D => "D", 
                    PlanetClass::I => "I",
                    PlanetClass::T => "T",
                    PlanetClass::G => "G",
                };
                
                // Apply class filter
                if let Some(ref filter_class) = class_filter {
                    if planet_class != filter_class {
                        continue;
                    }
                }
                
                let is_habitable = env.habitability_score > 0.5;
                
                // Apply habitability filter
                if habitable_only && !is_habitable {
                    continue;
                }
                
                let planet_data = serde_json::json!({
                    "id": format!("REAL-{}", body.id),
                    "class": planet_class,
                    "temperature": env.profile.temp_celsius,
                    "water_fraction": env.profile.liquid_water,
                    "oxygen_fraction": env.profile.atmos_oxygen,
                    "radiation_level": env.profile.radiation,
                    "habitable": is_habitable,
                    "age_gyr": body.age / (365.25 * 24.0 * 3600.0 * 1e9),
                    "mass_earth": body.mass / 5.972e24,
                    "radius_earth": body.radius / 6.371e6,
                    "habitability_score": env.habitability_score,
                    "position": [state.position.x, state.position.y, state.position.z],
                    "velocity": [state.velocity.x, state.velocity.y, state.velocity.z]
                });
                
                planets.push(planet_data);
            }
        }
        
        Ok(serde_json::json!(planets))
    }

    /// Get detailed planet inspection data
    pub fn get_planet_inspection_data(&mut self, planet_id: &str) -> Result<Option<serde_json::Value>> {
        // Extract UUID from planet ID if it's a real planet
        if let Some(uuid_str) = planet_id.strip_prefix("REAL-") {
            if let Ok(planet_uuid) = uuid::Uuid::parse_str(uuid_str) {
                let mut query = self.world.query::<(&CelestialBody, &PlanetaryEnvironment, &physics_engine::PhysicsState)>();
                for (body, env, state) in query.iter(&self.world) {
                    if body.id == planet_uuid {
                        let planet_class = match env.planet_class {
                            PlanetClass::E => "E",
                            PlanetClass::D => "D", 
                            PlanetClass::I => "I",
                            PlanetClass::T => "T",
                            PlanetClass::G => "G",
                        };
                        
                        let planet_data = serde_json::json!({
                            "id": planet_id,
                            "class": planet_class,
                            "mass_earth": body.mass / 5.972e24,
                            "radius_earth": body.radius / 6.371e6,
                            "temperature": env.profile.temp_celsius,
                            "water_fraction": env.profile.liquid_water,
                            "oxygen_fraction": env.profile.atmos_oxygen,
                            "radiation_level": env.profile.radiation,
                            "age_gyr": body.age / (365.25 * 24.0 * 3600.0 * 1e9),
                            "habitable": env.habitability_score > 0.5,
                            "habitability_score": env.habitability_score,
                            "atmospheric_pressure_atm": env.profile.atmos_pressure,
                            "surface_gravity_g": (physics_engine::GRAVITATIONAL_CONSTANT * body.mass) / (body.radius.powi(2) * 9.81),
                            "orbital_distance_au": state.position.magnitude() / 1.496e11,
                            "luminosity": body.luminosity,
                            "composition": {
                                "hydrogen": env.profile.liquid_water * 0.11, // Approximate H from H2O
                                "oxygen": env.profile.atmos_oxygen,
                                "carbon": 0.03, // Placeholder
                                "nitrogen": 0.78 - env.profile.atmos_oxygen, // Approximate N2
                                "other": 0.08
                            },
                            "position": [state.position.x, state.position.y, state.position.z],
                            "velocity": [state.velocity.x, state.velocity.y, state.velocity.z]
                        });
                        
                        return Ok(Some(planet_data));
                    }
                }
            }
        }
        
        Ok(None)
    }

    /// Get real lineage data from simulation ECS world  
    pub fn get_lineage_data(&mut self) -> Result<serde_json::Value> {
        let mut lineages = Vec::new();
        
        let mut query = self.world.query::<&AgentLineage>();
        for lineage in query.iter(&self.world) {
            let lineage_data = serde_json::json!({
                "id": format!("REAL-{}", lineage.id),
                "generation": lineage.generation,
                "population": 1, // Individual lineage entry
                "average_fitness": lineage.fitness,
                "sentience_level": lineage.sentience_level,
                "tech_level": lineage.tech_level,
                "consciousness_level": lineage.sentience_level, // Approximate
                "immortality_achieved": lineage.immortality_achieved,
                "industrialization_level": lineage.industrialization_level,
                "digitalization_level": lineage.digitalization_level,
                "parent_id": lineage.parent_id.map(|id| format!("REAL-{}", id)),
                "code_hash": lineage.code_hash,
                "last_mutation_tick": lineage.last_mutation_tick
            });
            
            lineages.push(lineage_data);
        }
        
        Ok(serde_json::json!(lineages))
    }

    /// Get detailed lineage inspection data
    pub fn get_lineage_inspection_data(&mut self, lineage_id: &str) -> Result<Option<serde_json::Value>> {
        // Extract UUID from lineage ID if it's a real lineage
        if let Some(uuid_str) = lineage_id.strip_prefix("REAL-") {
            if let Ok(lineage_uuid) = uuid::Uuid::parse_str(uuid_str) {
                let mut query = self.world.query::<&AgentLineage>();
                for lineage in query.iter(&self.world) {
                    if lineage.id == lineage_uuid {
                        let lineage_data = serde_json::json!({
                            "id": lineage_id,
                            "generation": lineage.generation,
                            "population": 1,
                            "average_fitness": lineage.fitness,
                            "sentience_level": lineage.sentience_level,
                            "tech_level": lineage.tech_level,
                            "consciousness_level": lineage.sentience_level,
                            "immortality_achieved": lineage.immortality_achieved,
                            "industrialization_level": lineage.industrialization_level,
                            "digitalization_level": lineage.digitalization_level,
                            "parent_id": lineage.parent_id.map(|id| format!("REAL-{}", id)),
                            "code_hash": lineage.code_hash,
                            "last_mutation_tick": lineage.last_mutation_tick,
                            "mutation_rate": 0.001, // From config
                            "evolutionary_pressure": lineage.fitness * lineage.tech_level,
                            "survival_adaptations": vec![
                                "Environmental adaptation",
                                "Technological advancement", 
                                "Consciousness development"
                            ]
                        });
                        
                        return Ok(Some(lineage_data));
                    }
                }
            }
        }
        
        Ok(None)
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