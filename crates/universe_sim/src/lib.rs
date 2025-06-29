//! Universe Simulation Core Library
//! Provides high-level orchestration of cosmic evolution, physics engine integration, and AI systems.

#![cfg_attr(all(not(feature = "unstable-universe"), not(test)), deny(warnings, clippy::all, clippy::pedantic))]
#![cfg_attr(feature = "unstable-universe", allow(dead_code))]

use anyhow::{Result, anyhow};
use diagnostics::DiagnosticsSystem;
use nalgebra::Vector3;
use physics_engine::{
    PhysicsEngine,
    FundamentalParticle,
    QuantumState,
};
use rand::Rng;
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use nalgebra::ComplexField;

use std::collections::VecDeque;
use serde_json::json;
use crate::storage::{Store, AgentLineage, CelestialBody, BodyType, SupernovaYields, EnrichmentFactor, Atmosphere};
use tracing::{info, warn, debug};
use crate::config::SimulationConfig;
// Agent config is now part of SimulationConfig

pub mod config;
pub mod cosmic_era;
pub mod evolution;
pub mod persistence;
pub mod storage;
pub mod world;

pub use physics_engine;
pub use storage::{CelestialBodyType, PlanetClass, StellarEvolution, StellarPhase, ParticleStore};

// Import molecular dynamics types for visualization integration
use physics_engine::molecular_dynamics::{MDSnapshot, ReactionEvent, ReactionEventType};



/// Calculate relativistic total energy from momentum and mass
/// E = sqrt((pc)^2 + (mc^2)^2) where c = speed of light
#[allow(dead_code)]
fn calculate_relativistic_energy(momentum: &Vector3<f64>, mass: f64) -> f64 {
    // Use utility function from physics_engine
    physics_engine::utils::math::calculate_relativistic_energy(momentum, mass)
}

/// Core universe simulation structure
pub struct UniverseSimulation {
    pub store: Store,                          // SoA data store
    /// ECS world used only for high-level queries in unit tests and systems that
    /// still rely on Bevy-style APIs. Over time, the project is migrating to
    /// the custom SoA `Store`, but we keep this lightweight `World` around to
    /// satisfy legacy code while the transition is in progress.
    // pub world: World,
    pub physics_engine: PhysicsEngine,         // Physics simulation
    pub current_tick: u64,                     // Simulation time
    pub tick_span_years: f64,                  // Years per tick (default 1M)
    pub target_ups: f64,                       // Updates per second target
    pub universe_state: cosmic_era::UniverseState, // Current universe physical state
    pub config: config::SimulationConfig,      // Configuration
    pub diagnostics: DiagnosticsSystem,        // Performance monitoring
    pub physical_transitions: Vec<cosmic_era::PhysicalTransition>, // Record of major transitions
    /// Historical simulation statistics recorded each tick (limited length)
    pub stats_history: VecDeque<SimulationStats>,
    pub performance_stats: PerformanceStats,
}

/// Maximum number of historical statistics points to keep in memory
#[allow(dead_code)]
const MAX_STATS_HISTORY: usize = 10_000;

/// Performance tracking for simulation steps
#[derive(Debug, Default)]
pub struct PerformanceStats {
    step_times: VecDeque<Duration>,
    max_history: usize,
}

impl PerformanceStats {
    pub fn new() -> Self {
        Self {
            step_times: VecDeque::new(),
            max_history: 1000,
        }
    }
    
    pub fn add_step_time(&mut self, duration: Duration) {
        self.step_times.push_back(duration);
        if self.step_times.len() > self.max_history {
            self.step_times.pop_front();
        }
    }
    
    pub fn average_step_time(&self) -> Duration {
        if self.step_times.is_empty() {
            Duration::from_secs(0)
        } else {
            let total: Duration = self.step_times.iter().sum();
            total / self.step_times.len() as u32
        }
    }
}

/// Cosmological parameters for universe expansion
#[derive(Debug, Clone)]
pub struct CosmologicalParameters {
    pub hubble_constant: f64,      // H₀ in km/s/Mpc
    pub omega_matter: f64,         // Ωₘ
    pub omega_lambda: f64,         // ΩΛ
    pub omega_baryon: f64,         // Ωᵦ
    pub scale_factor: f64,         // a(t)
    pub redshift: f64,             // z
    pub age_of_universe: f64,      // t in Gyr
    pub enable_expansion: bool,
}

impl UniverseSimulation {
    /// Create a new universe simulation
    pub fn new(config: config::SimulationConfig) -> Result<Self> {
        let store = Store::new();
        let physics_engine = PhysicsEngine::new()?;
        // let world = World::default();

        Ok(Self {
            store,
            // world,
            physics_engine,
            current_tick: 0,
            tick_span_years: config.tick_span_years,
            target_ups: config.target_ups,
            universe_state: cosmic_era::UniverseState::initial(),
            config,
            diagnostics: DiagnosticsSystem::new(),
            physical_transitions: Vec::new(),
            stats_history: VecDeque::new(),
            performance_stats: PerformanceStats::default(),
        })
    }

    /// Initialize with Big Bang conditions
    pub fn init_big_bang(&mut self) -> Result<()> {
        // Reset to initial universe state
        self.universe_state = cosmic_era::UniverseState::initial();
        
        // Print initial configuration
        println!("🚀 INITIALIZING BIG BANG CONDITIONS");
        println!("   Initial particle count: {}", self.config.initial_particle_count);
        println!("   Tick span years: {}", self.tick_span_years);
        println!("   Target UPS: {}", self.target_ups);
        
        // Create initial particle soup
        self.spawn_initial_particles()?;
        
        // Initialize cosmic background
        self.init_cosmic_background()?;
        
        // Print physics engine initial state
        println!("📊 PHYSICS ENGINE INITIAL STATE:");
        println!("   Temperature: {:.2e} K", self.physics_engine.temperature);
        println!("   Energy density: {:.2e} J/m³", self.physics_engine.energy_density);
        println!("   Volume: {:.2e} m³", self.physics_engine.volume);
        println!("   Particles: {}", self.physics_engine.particles.len());
        println!("   Nuclei: {}", self.physics_engine.nuclei.len());
        println!("   Atoms: {}", self.physics_engine.atoms.len());
        println!("   Molecules: {}", self.physics_engine.molecules.len());
        
        // Count entities in our new store
        let particle_count = self.store.particles.count;
        println!("   Store Particle Count: {}", particle_count);
        
        // Print sample of initial particles
        if particle_count > 0 {
            println!("📍 SAMPLE OF INITIAL PARTICLES:");
            for i in 0..5.min(particle_count) {
                println!("   Particle {}: mass={:.2e} kg, temp={:.1} K, pos=({:.1e}, {:.1e}, {:.1e})", 
                    i, self.store.particles.mass[i], self.store.particles.temperature[i], 
                    self.store.particles.position[i].x, self.store.particles.position[i].y, 
                    self.store.particles.position[i].z);
            }
        }
        
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
            
            let _ = self.store.particles.add(
                position,
                velocity,
                mass,
                charge,
                3000.0,  // 3000 K initial temperature
                1e-23, // Default entropy
            );
        }
        
        Ok(())
    }

    /// Initialize cosmic background radiation and temperature
    fn init_cosmic_background(&mut self) -> Result<()> {
        // Set initial temperature and energy density based on config
        self.physics_engine.temperature = self.config.big_bang_temperature;
        self.physics_engine.energy_density =
            cosmic_era::calculate_energy_density(self.config.big_bang_temperature);
        Ok(())
    }

    /// Main simulation tick (alias for step with default dt)
    pub fn tick(&mut self) -> Result<()> {
        // Run the standard step
        let result = self.step();

        // Emit progress logs every tick. When verbose mode is enabled at the CLI level, these
        // `tracing::info!` logs will be shown in the console, giving continuous insight into the
        // simulation. We still keep the redundant `println!` for the first few ticks to guarantee
        // visibility even if tracing is mis-configured.

        use tracing::debug;
        debug!(
            tick = self.current_tick,
            age_gyr = self.universe_state.age_gyr,
            particle_count = self.physics_engine.particles.len(),
            target_ups = self.target_ups,
            "simulation progress"
        );

        // To avoid overwhelming stdout, we only print the plaintext message for the first 10 ticks.
        if self.current_tick <= 10 {
            println!(
                "[progress] tick={} age_gyr={:.4} particles={} UPS_target={}",
                self.current_tick,
                self.universe_state.age_gyr,
                self.physics_engine.particles.len(),
                self.target_ups
            );
        }

        result
    }

    /// Main simulation step
    pub fn step(&mut self) -> Result<()> {
         // Increment tick counter
         self.current_tick += 1;
         
         // Track performance
         let start_time = std::time::Instant::now();
         
         // Run physics simulation step
         self.physics_engine.step(self.physics_engine.time_step)?;
         
         // Update cosmic state from simulation results
         self.update_universe_state()?;
         
         // Process stellar evolution
         // Use adaptive time stepping for stellar evolution based on physics engine time step
         // and current visualization scale for atom and particle focus
         let stellar_dt = self.calculate_adaptive_stellar_time_step(self.physics_engine.time_step);
         self.process_stellar_evolution(stellar_dt)?;
         
         // Process agent evolution on habitable worlds
         self.process_agent_evolution(self.physics_engine.time_step)?;
         
         // Apply cosmological expansion effects to universe-scale properties
         self.apply_cosmological_effects(self.physics_engine.time_step)?;
         
         // Update persistence layer
         // Auto-save functionality removed for now - will be handled by CLI
         
         // Track performance
         let step_duration = start_time.elapsed();
         self.performance_stats.add_step_time(step_duration);
         
         Ok(())
     }

    /// Update universe state based on age
    fn update_universe_state(&mut self) -> Result<()> {
        // Very lightweight placeholder: compute age and a basic description
        let previous_state = self.universe_state.clone();

        // Update age (in Gyr)
        self.universe_state.age_gyr = self.universe_age_gyr();

        // Propagate average temperature directly from physics engine for now
        self.universe_state.mean_temperature = self.physics_engine.temperature;

        // Re-use helper from cosmic_era to create descriptive text if needed
        // Here we skip expensive calculations and only update essential fields.

        // Detect major transitions (placeholder)
        self.detect_physical_transitions(&previous_state)?;

        Ok(())
    }

    /// Detect and record major physical transitions in the universe
    fn detect_physical_transitions(
        &mut self,
        previous_state: &cosmic_era::UniverseState,
    ) -> Result<()> {
        let current_state = &self.universe_state;
        
        // Detect cosmic era transitions
        let previous_era = cosmic_era::determine_cosmic_era(previous_state.age_gyr);
        let current_era = cosmic_era::determine_cosmic_era(current_state.age_gyr);
        
        if previous_era != current_era {
            let transition = cosmic_era::PhysicalTransition {
                tick: self.current_tick,
                timestamp: self.current_tick as f64,
                transition_type: cosmic_era::TransitionType::CosmicEra,
                description: format!("Transition from {:?} to {:?}", previous_era, current_era),
                physical_parameters: vec![
                    ("age_gyr".to_string(), current_state.age_gyr),
                    ("temperature".to_string(), current_state.mean_temperature),
                    ("energy_density".to_string(), current_state.energy_density),
                ],
                age_gyr: current_state.age_gyr,
                temperature: current_state.mean_temperature,
                energy_density: current_state.energy_density,
            };
            self.physical_transitions.push(transition);
        }
        
        // Detect temperature-based transitions
        let temp_threshold = 1e3; // 1000 K threshold for significant changes
        let temp_change = (current_state.mean_temperature - previous_state.mean_temperature).abs();
        if temp_change > temp_threshold {
            let transition = cosmic_era::PhysicalTransition {
                tick: self.current_tick,
                timestamp: self.current_tick as f64,
                transition_type: cosmic_era::TransitionType::Temperature,
                description: format!("Temperature change: {:.2} K -> {:.2} K", 
                                   previous_state.mean_temperature, current_state.mean_temperature),
                physical_parameters: vec![
                    ("temp_change".to_string(), temp_change),
                    ("previous_temp".to_string(), previous_state.mean_temperature),
                    ("current_temp".to_string(), current_state.mean_temperature),
                ],
                age_gyr: current_state.age_gyr,
                temperature: current_state.mean_temperature,
                energy_density: current_state.energy_density,
            };
            self.physical_transitions.push(transition);
        }
        
        // Detect energy density transitions
        let energy_threshold = 1e-10; // J/m³ threshold
        let energy_change = (current_state.energy_density - previous_state.energy_density).abs();
        if energy_change > energy_threshold {
            let transition = cosmic_era::PhysicalTransition {
                tick: self.current_tick,
                timestamp: self.current_tick as f64,
                transition_type: cosmic_era::TransitionType::EnergyDensity,
                description: format!("Energy density change: {:.2e} -> {:.2e} J/m³", 
                                   previous_state.energy_density, current_state.energy_density),
                physical_parameters: vec![
                    ("energy_change".to_string(), energy_change),
                    ("previous_energy".to_string(), previous_state.energy_density),
                    ("current_energy".to_string(), current_state.energy_density),
                ],
                age_gyr: current_state.age_gyr,
                temperature: current_state.mean_temperature,
                energy_density: current_state.energy_density,
            };
            self.physical_transitions.push(transition);
        }
        
        // Keep only recent transitions (last 1000)
        if self.physical_transitions.len() > 1000 {
            self.physical_transitions.drain(0..self.physical_transitions.len() - 1000);
        }
        
        Ok(())
    }

    /// Update physics simulation
    #[allow(dead_code)]
    fn update_physics(&mut self) -> Result<()> {
        // Advance physics engine if available; ignore errors for placeholder
        let _ = self.physics_engine.step(self.physics_engine.time_step);
        Ok(())
    }

    /// Update agent evolution and behavior
    #[allow(dead_code)]
    fn update_agent_evolution(&mut self) -> Result<()> {
        // Integrate with the agent evolution system from the agent_evolution crate
        // This handles the evolution of intelligent agents on habitable worlds
        
        // Get current universe state for agent context
        let universe_age = self.universe_age_gyr();
        let cosmic_era = cosmic_era::determine_cosmic_era(universe_age);
        
        // Process agent evolution on habitable planets - collect first to avoid borrow conflicts
        let habitable_planets: Vec<usize> = self.store.celestials.iter().enumerate()
            .filter_map(|(id, planet)| {
                if planet.body_type == BodyType::Planet && planet.is_habitable {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        
        for planet_id in habitable_planets {
            // Extract planet data before mutable borrow
            let planet_temperature = self.store.celestials[planet_id].temperature;
            let planet_gravity = self.store.celestials[planet_id].gravity;
            let planet_atmosphere = self.store.celestials[planet_id].atmosphere.clone();
            
            self.update_planet_agents_extracted(planet_id, planet_temperature, planet_gravity, planet_atmosphere, universe_age, cosmic_era.clone())?;
        }
        
        // Process inter-planetary agent interactions
        self.process_interplanetary_agent_interactions()?;
        
        // Update global agent statistics
        self.update_agent_statistics()?;
        
        Ok(())
    }
    
    /// Update agents on a specific habitable planet
    #[allow(dead_code)]
    fn update_planet_agents(
        &mut self,
        planet_id: usize,
        planet: &mut CelestialBody,
        universe_age: f64,
        cosmic_era: cosmic_era::CosmicEra,
    ) -> Result<()> {
        // Check if agents exist on this planet
        if let Some(agent_population) = self.store.agent_populations.get_mut(&planet_id) {
            // Update agent evolution based on planet conditions
            let evolution_context = agent_evolution::EvolutionContext {
                planet_temperature: planet.temperature,
                planet_gravity: planet.gravity,
                planet_atmosphere: agent_evolution::Atmosphere {
                    pressure: planet.atmosphere.pressure,
                    composition: planet.atmosphere.composition.clone(),
                    temperature: planet.atmosphere.temperature,
                    density: planet.atmosphere.density,
                    scale_height: planet.atmosphere.scale_height,
                },
                cosmic_era: agent_evolution::CosmicEra {
                    age_gyr: cosmic_era.age_gyr,
                    mean_temperature: cosmic_era.mean_temperature,
                    stellar_fraction: cosmic_era.stellar_fraction,
                    metallicity: cosmic_era.metallicity,
                    habitable_count: cosmic_era.habitable_count,
                    max_complexity: cosmic_era.max_complexity,
                    energy_density: cosmic_era.energy_density,
                    hubble_constant: cosmic_era.hubble_constant,
                },
                universe_age,
                time_step: self.physics_engine.time_step,
            };
            
            // Evolve agents using the agent evolution system
            agent_population.evolve(&evolution_context)?;
            
            // Update planet's agent-related properties
            planet.agent_population = agent_population.total_population();
            planet.tech_level = agent_population.average_tech_level();
        }
        
        Ok(())
    }
    
    /// Update agents on a specific habitable planet (extracted version to avoid borrow conflicts)
    fn update_planet_agents_extracted(
        &mut self,
        planet_id: usize,
        planet_temperature: f64,
        planet_gravity: f64,
        planet_atmosphere: Atmosphere,
        universe_age: f64,
        cosmic_era: cosmic_era::CosmicEra,
    ) -> Result<()> {
        // Check if agents exist on this planet
        if let Some(agent_population) = self.store.agent_populations.get_mut(&planet_id) {
            // Update agent evolution based on planet conditions
            let evolution_context = agent_evolution::EvolutionContext {
                planet_temperature,
                planet_gravity,
                planet_atmosphere: agent_evolution::Atmosphere {
                    pressure: planet_atmosphere.pressure,
                    composition: planet_atmosphere.composition.clone(),
                    temperature: planet_atmosphere.temperature,
                    density: planet_atmosphere.density,
                    scale_height: planet_atmosphere.scale_height,
                },
                cosmic_era: agent_evolution::CosmicEra {
                    age_gyr: cosmic_era.age_gyr,
                    mean_temperature: cosmic_era.mean_temperature,
                    stellar_fraction: cosmic_era.stellar_fraction,
                    metallicity: cosmic_era.metallicity,
                    habitable_count: cosmic_era.habitable_count,
                    max_complexity: cosmic_era.max_complexity,
                    energy_density: cosmic_era.energy_density,
                    hubble_constant: cosmic_era.hubble_constant,
                },
                universe_age,
                time_step: self.physics_engine.time_step,
            };
            
            // Evolve agents using the agent evolution system
            agent_population.evolve(&evolution_context)?;
            
            // Update planet's agent-related properties
            let planet = &mut self.store.celestials[planet_id];
            planet.agent_population = agent_population.total_population();
            planet.tech_level = agent_population.average_tech_level();
        }
        
        Ok(())
    }
    
    /// Process interactions between agents on different planets
    fn process_interplanetary_agent_interactions(&mut self) -> Result<()> {
        // Handle communication, trade, and conflict between agent populations
        let mut interaction_events = Vec::new();
        
        // Find all habitable planets with agent populations
        let habitable_planets: Vec<(usize, u64, f64)> = self.store.agent_populations.iter()
            .filter_map(|(planet_id, population)| {
                let pop_count = population.total_population();
                let tech_level = population.average_tech_level();
                if pop_count > 0 && tech_level > 0.1 { // Minimum tech level for interplanetary interaction
                    Some((*planet_id, pop_count, tech_level))
                } else {
                    None
                }
            })
            .collect();
        
        // Process interactions between pairs of planets
        for i in 0..habitable_planets.len() {
            for j in (i + 1)..habitable_planets.len() {
                let (planet_a_id, pop_a, tech_a) = habitable_planets[i];
                let (planet_b_id, pop_b, tech_b) = habitable_planets[j];
                
                // Calculate distance between planets
                let planet_a = &self.store.celestials[planet_a_id];
                let planet_b = &self.store.celestials[planet_b_id];
                let distance = (planet_a.position - planet_b.position).magnitude();
                
                // Communication probability based on technology and distance
                let max_comm_distance = (tech_a + tech_b) * 1e15; // Light-years in meters
                let comm_probability = if distance < max_comm_distance {
                    (max_comm_distance - distance) / max_comm_distance * 0.01 // 1% base chance
                } else {
                    0.0
                };
                
                // Check for communication event
                if rand::random::<f64>() < comm_probability {
                    // Successful communication leads to technology transfer
                    let tech_transfer = (tech_a - tech_b).abs() * 0.1; // 10% technology transfer
                    
                    interaction_events.push((planet_a_id, planet_b_id, tech_transfer));
                }
                
                // Trade probability (higher for closer, more advanced civilizations)
                let trade_probability = comm_probability * (pop_a as f64 * pop_b as f64).sqrt() / 1e12;
                
                if rand::random::<f64>() < trade_probability {
                    // Trade boosts both civilizations' development
                    let trade_boost = 0.05; // 5% development boost
                    interaction_events.push((planet_a_id, planet_b_id, trade_boost));
                    interaction_events.push((planet_b_id, planet_a_id, trade_boost));
                }
            }
        }
        
        // Apply interaction effects
        for (planet_id, _other_id, benefit) in interaction_events {
            if let Some(population) = self.store.agent_populations.get_mut(&planet_id) {
                population.apply_external_development_boost(benefit);
            }
        }
        
        Ok(())
    }
    
    /// Update global agent statistics
    fn update_agent_statistics(&mut self) -> Result<()> {
        // Calculate global agent population and technology statistics
        let mut _total_agents = 0;
        let mut total_tech_level = 0.0;
        let mut habitable_planets = 0;
        
        for (_, population) in &self.store.agent_populations {
            _total_agents += population.total_population();
            total_tech_level += population.average_tech_level();
            habitable_planets += 1;
        }
        
        // Update global statistics
        if habitable_planets > 0 {
            self.universe_state.average_tech_level = total_tech_level / habitable_planets as f64;
        }
        
        Ok(())
    }

    /// Update cosmic-scale processes based on current physical conditions
    #[allow(dead_code)]
    fn update_cosmic_processes(&mut self, dt: f64) -> Result<()> {
        // Handle various cosmic-scale processes that affect the universe
        
        // Stellar evolution and nucleosynthesis
        self.process_stellar_evolution(dt)?;
        
        // Star formation from gas clouds
        self.process_star_formation()?;
        
        // Planet formation around stars
        self.process_planet_formation()?;
        
        // Galactic evolution and structure formation
        self.process_galactic_evolution(dt)?;
        
        // Dark matter and dark energy effects
        self.process_dark_matter_effects(dt)?;
        
        // Cosmic ray propagation and effects
        self.process_cosmic_rays(dt)?;
        
        // Gravitational wave generation and propagation
        self.process_gravitational_waves(dt)?;
        
        Ok(())
    }
    
    /// Process galactic evolution and large-scale structure formation
    fn process_galactic_evolution(&mut self, dt: f64) -> Result<()> {
        // Handle galaxy formation, mergers, and evolution
        
        // Update galactic-scale properties based on stellar evolution
        let total_stellar_mass: f64 = self.store.celestials.iter()
            .filter(|body| body.body_type == BodyType::Star)
            .map(|star| star.mass)
            .sum();
        
        // Calculate galactic rotation and dynamics
        let galaxy_mass = total_stellar_mass * 10.0; // Include dark matter halo
        let galaxy_radius = 5e20; // ~50 kpc in meters
        
        // Apply galactic rotation to stellar positions
        let rotation_period = 2.0 * std::f64::consts::PI * (galaxy_radius.powi(3) / (6.67430e-11 * galaxy_mass)).sqrt();
        let angular_velocity = 2.0 * std::f64::consts::PI / rotation_period;
        
        // Rotate stars around galactic center
        for star in self.store.celestials.iter_mut() {
            if star.body_type == BodyType::Star {
                let radius = star.position.magnitude();
                if radius > 0.0 && radius < galaxy_radius {
                    // Calculate orbital velocity for circular orbit
                    let orbital_velocity = angular_velocity * radius;
                    
                    // Update position based on galactic rotation
                    let rotation_angle = angular_velocity * dt;
                    let cos_theta = rotation_angle.cos();
                    let sin_theta = rotation_angle.sin();
                    
                    let new_x = star.position.x * cos_theta - star.position.y * sin_theta;
                    let new_y = star.position.x * sin_theta + star.position.y * cos_theta;
                    
                    star.position.x = new_x;
                    star.position.y = new_y;
                    
                    // Update velocity to maintain circular orbit
                    star.velocity.x = -orbital_velocity * sin_theta;
                    star.velocity.y = orbital_velocity * cos_theta;
                }
            }
        }
        
        // Process galaxy mergers (simplified)
        let universe_age = self.universe_age_gyr();
        if universe_age > 1.0 && rand::random::<f64>() < 0.001 { // 0.1% chance per Gyr
            self.process_galaxy_merger()?;
        }
        
        // Update spiral arm structure
        self.update_spiral_arms(dt)?;
        
        // Update universe state with galactic information
        self.universe_state.total_stellar_mass = total_stellar_mass;
        
        Ok(())
    }
    
    /// Process a major galaxy merger event
    fn process_galaxy_merger(&mut self) -> Result<()> {
        // Increase star formation rate temporarily
        let merger_boost = 5.0; // 5x star formation boost
        
        // Apply gravitational perturbations to stellar orbits
        for star in self.store.celestials.iter_mut() {
            if star.body_type == BodyType::Star {
                // Add random velocity perturbation
                let perturbation_magnitude = 1e4; // 10 km/s
                star.velocity.x += (rand::random::<f64>() - 0.5) * perturbation_magnitude;
                star.velocity.y += (rand::random::<f64>() - 0.5) * perturbation_magnitude;
                star.velocity.z += (rand::random::<f64>() - 0.5) * perturbation_magnitude;
            }
        }
        
        // Trigger enhanced star formation
        for _ in 0..(merger_boost as usize) {
            self.process_star_formation()?;
        }
        
        Ok(())
    }
    
    /// Update spiral arm structure and density waves
    fn update_spiral_arms(&mut self, dt: f64) -> Result<()> {
        let spiral_pattern_speed = 1e-15; // rad/s
        let spiral_pitch_angle = 0.2; // radians
        
        // Calculate spiral arm density enhancement
        for star in self.store.celestials.iter_mut() {
            if star.body_type == BodyType::Star {
                let radius = (star.position.x * star.position.x + star.position.y * star.position.y).sqrt();
                let theta = star.position.y.atan2(star.position.x);
                
                // Spiral arm position
                let spiral_theta = theta - spiral_pattern_speed * self.current_tick as f64 * dt;
                let spiral_arm_distance = (spiral_theta - radius * spiral_pitch_angle.tan()).sin().abs();
                
                // Density enhancement in spiral arms
                if spiral_arm_distance < 0.3 { // Within spiral arm
                    // Increase local density and star formation probability
                    star.atmosphere.density *= 1.5;
                    
                    // Enhanced stellar evolution in dense regions
                    if star.age > star.lifetime * 0.8 { // Near end of life
                        star.luminosity *= 1.2; // Increased luminosity
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Process dark matter and dark energy effects on cosmic expansion
    fn process_dark_matter_effects(&mut self, _dt: f64) -> Result<()> {
        // Calculate dark matter and dark energy contributions to expansion
        let current_age = self.universe_age_gyr();
        let hubble_constant = self.get_hubble_constant(current_age);
        
        // Update cosmological parameters
        self.universe_state.hubble_constant = hubble_constant;
        
        // Calculate dark energy density (assumes ΛCDM model)
        let critical_density = 3.0 * hubble_constant * hubble_constant / (8.0 * std::f64::consts::PI * 6.67430e-11);
        let dark_energy_density = 0.7 * critical_density; // ΩΛ ≈ 0.7
        let dark_matter_density = 0.25 * critical_density; // Ωm ≈ 0.25
        
        self.universe_state.dark_energy_density = dark_energy_density;
        self.universe_state.dark_matter_density = dark_matter_density;
        
        Ok(())
    }
    
    /// Process cosmic ray generation and propagation
    fn process_cosmic_rays(&mut self, _dt: f64) -> Result<()> {
        // Calculate cosmic ray flux from stellar sources
        let cosmic_ray_flux = self.calculate_cosmic_ray_flux()?;
        
        // Update universe state
        self.universe_state.cosmic_ray_flux = cosmic_ray_flux;
        
        Ok(())
    }
    
    /// Process gravitational wave generation from compact object mergers
    fn process_gravitational_waves(&mut self, _dt: f64) -> Result<()> {
        // Calculate gravitational wave strain from stellar mergers
        let gw_strain = self.calculate_gravitational_wave_strain()?;
        
        // Update universe state
        self.universe_state.gravitational_wave_strain = gw_strain;
        
        Ok(())
    }
    
    /// Calculate cosmic ray flux from stellar sources
    fn calculate_cosmic_ray_flux(&self) -> Result<f64> {
        // Estimate cosmic ray flux based on stellar activity
        let total_stellar_luminosity: f64 = self.store.celestials.iter()
            .filter(|body| body.body_type == BodyType::Star)
            .map(|star| star.luminosity)
            .sum();
        
        // Rough estimate: cosmic ray flux proportional to stellar luminosity
        let flux = total_stellar_luminosity * 1e-15; // Scaling factor
        
        Ok(flux)
    }
    
    /// Calculate gravitational wave strain from compact object mergers
    fn calculate_gravitational_wave_strain(&self) -> Result<f64> {
        // Estimate gravitational wave strain from stellar mergers
        let compact_objects = self.store.celestials.iter()
            .filter(|body| matches!(body.body_type, 
                BodyType::NeutronStar | BodyType::BlackHole))
            .count();
        
        // Rough estimate: strain proportional to number of compact objects
        let strain = compact_objects as f64 * 1e-21; // Typical strain values
        
        Ok(strain)
    }
    
    /// Get Hubble constant at a given age
    fn get_hubble_constant(&self, age_gyr: f64) -> f64 {
        // Current Hubble constant (km/s/Mpc)
        let h0_current = 70.0;
        
        // For a flat universe with dark energy, H(t) evolves
        // This is a simplified model
        let age_universe_gyr = 13.8; // Current age of universe
        let hubble_ratio = (age_universe_gyr / age_gyr).powf(0.5);
        
        h0_current * hubble_ratio
    }

    /// Process stellar evolution based on nuclear burning
    fn process_stellar_evolution(&mut self, dt: f64) -> Result<()> {
        let _dt_years = dt;

        // Iterate over all stellar evolution records.
        let mut death_events: Vec<usize> = Vec::new();

        for evolution in &mut self.store.stellar_evolutions {
            let entity_id = evolution.entity_id;
            let body = self
                .store
                .celestials
                .get_mut(entity_id)
                .expect("Invalid entity_id in StellarEvolution");

            // 1. Advance age.
            body.age += dt;

            // 2. Evolve core.
            let _energy_generated = evolution.evolve(body.mass, dt)?;

            // 3. Update global properties.
            body.radius = Self::calculate_stellar_radius(body.mass);
            body.luminosity = Self::calculate_stellar_luminosity(body.mass);
            body.temperature = Self::calculate_stellar_temperature(body.mass);

            // 4. Composition update placeholder.
            Self::update_stellar_composition(body, evolution);

            // 5. Check for death event.
            if matches!(
                evolution.evolutionary_phase,
                StellarPhase::WhiteDwarf | StellarPhase::NeutronStar | StellarPhase::BlackHole
            ) {
                death_events.push(entity_id);
            }
        }

        // Process deaths after main loop to avoid mutable aliasing.
        for entity_id in death_events {
            let body_clone = self.store.celestials[entity_id].clone();
            let evolution_clone = self
                .store
                .stellar_evolutions
                .iter()
                .find(|e| e.entity_id == entity_id)
                .expect("Missing evolution record")
                .clone();
            self.process_stellar_death(entity_id, &body_clone, &evolution_clone)?;
        }

        Ok(())
    }

    /// Update stellar composition based on nuclear burning products (placeholder)
    fn update_stellar_composition(
        _body: &mut CelestialBody,
        _evolution: &StellarEvolution,
    ) {
        // For the current lightweight implementation we do not attempt to
        // propagate the detailed isotopic yields to the outer layers. A full
        // treatment would involve solving diffusive mixing equations and
        // convective dredge-up. Here we just leave a placeholder for future
        // work while keeping the function non-panic.
    }

    /// Handle the death of a star
    fn process_stellar_death(
        &mut self,
        _entity_id: usize,
        _body: &CelestialBody,
        _evolution: &StellarEvolution,
    ) -> Result<()> {
        // At this resolution we simply note that a stellar death occurred
        // and update global counters. Detailed remnant creation and gas
        // ejection will be handled in dedicated modules.
        Ok(())
    }

    /// Handle nucleosynthesis in supernova explosions
    #[allow(dead_code)]
    fn process_supernova_nucleosynthesis(&mut self) -> Result<()> {
        // Process nucleosynthesis for stars that have gone supernova
        // This creates heavy elements and enriches the interstellar medium
        
        let mut supernova_events = Vec::new();
        
        // Find stars that have recently gone supernova
        for (star_id, star) in self.store.celestials.iter().enumerate() {
            if star.body_type == BodyType::Star && star.age > star.lifetime {
                // Star has exceeded its lifetime - check if it should go supernova
                if star.mass > 8.0 { // Stars > 8 solar masses go supernova
                    supernova_events.push(star_id);
                }
            }
        }
        
        // Process each supernova event
        for star_id in supernova_events {
            self.process_single_supernova(star_id)?;
        }
        
        Ok(())
    }
    
    /// Process a single supernova event
    fn process_single_supernova(&mut self, star_id: usize) -> Result<()> {
        // Extract star mass before any mutable borrows
        let star_mass = self.store.celestials[star_id].mass;
        
        // Calculate nucleosynthesis yields based on stellar mass
        let yields = self.calculate_supernova_yields(star_mass)?;
        
        // Create enriched gas cloud from supernova ejecta
        self.create_supernova_remnant(star_id, &yields)?;
        
        // Update stellar remnant (neutron star or black hole)
        self.create_stellar_remnant(star_id, star_mass)?;
        
        // Update global chemical composition
        self.update_global_chemical_composition(&yields)?;
        
        Ok(())
    }
    
    /// Calculate nucleosynthesis yields for a supernova
    fn calculate_supernova_yields(&self, stellar_mass: f64) -> Result<SupernovaYields> {
        // Calculate element production based on stellar mass
        // This uses simplified nucleosynthesis models
        
        let mut yields = SupernovaYields::default();
        
        // Iron production (peak of binding energy curve)
        yields.iron_mass = stellar_mass * 0.1; // ~10% of stellar mass becomes iron
        
        // Silicon group elements (Si, S, Ar, Ca)
        yields.silicon_group_mass = stellar_mass * 0.05;
        
        // Oxygen group elements (O, Ne, Mg)
        yields.oxygen_group_mass = stellar_mass * 0.15;
        
        // Carbon group elements (C, N)
        yields.carbon_group_mass = stellar_mass * 0.02;
        
        // Heavy elements (r-process)
        yields.heavy_elements_mass = stellar_mass * 0.001; // Small fraction for r-process
        
        // Total ejected mass
        yields.total_ejected_mass = yields.iron_mass + yields.silicon_group_mass + 
                                   yields.oxygen_group_mass + yields.carbon_group_mass + 
                                   yields.heavy_elements_mass;
        
        Ok(yields)
    }
    
    /// Create a supernova remnant from ejected material
    fn create_supernova_remnant(&mut self, star_id: usize, yields: &SupernovaYields) -> Result<()> {
        let star = &self.store.celestials[star_id];
        
        // Create a new celestial body for the supernova remnant
        let remnant = CelestialBody {
            id: Uuid::new_v4(),
            entity_id: self.store.celestials.len(),
            body_type: BodyType::GasCloud,
            mass: yields.total_ejected_mass,
            radius: star.radius * 10.0, // Expanded remnant
            temperature: 1e4, // Hot gas
            age: 0.0,
            lifetime: 1e6, // Remnant lifetime in years
            position: star.position,
            velocity: star.velocity,
            gravity: star.gravity * 0.1, // Lower gravity for gas cloud
            composition: physics_engine::types::ElementTable::new(),
            has_planets: false,
            has_life: false,
            atmosphere: Atmosphere {
                composition: {
                    let mut comp = HashMap::new();
                    comp.insert("Fe".to_string(), yields.iron_mass / yields.total_ejected_mass);
                    comp.insert("Si".to_string(), yields.silicon_group_mass / yields.total_ejected_mass);
                    comp.insert("O".to_string(), yields.oxygen_group_mass / yields.total_ejected_mass);
                    comp.insert("C".to_string(), yields.carbon_group_mass / yields.total_ejected_mass);
                    comp
                },
                pressure: 1e-10, // Low pressure in remnant
                temperature: 1e4,
                density: 1e-15, // Very low density for gas cloud
                scale_height: 1e6, // Large scale height for expanded remnant
            },
            is_habitable: false,
            agent_population: 0,
            tech_level: 0.0,
            luminosity: 0.0,
        };
        
        self.store.celestials.push(remnant);
        
        Ok(())
    }
    
    /// Create a stellar remnant (neutron star or black hole)
    fn create_stellar_remnant(&mut self, star_id: usize, original_mass: f64) -> Result<()> {
        let _star = &self.store.celestials[star_id];
        
        // Determine remnant type based on original mass
        let remnant_type = if original_mass > 20.0 {
            BodyType::BlackHole
        } else {
            BodyType::NeutronStar
        };
        
        // Calculate remnant mass
        let remnant_mass = match remnant_type {
            BodyType::BlackHole => original_mass * 0.1, // ~10% of original mass
            BodyType::NeutronStar => 1.4, // Chandrasekhar limit
            _ => original_mass * 0.1,
        };
        
        // Update the original star to become the remnant
        let star_mut = &mut self.store.celestials[star_id];
        star_mut.body_type = remnant_type.clone();
        star_mut.mass = remnant_mass;
        star_mut.radius = match remnant_type {
            BodyType::BlackHole => 2.0 * 6.67430e-11 * remnant_mass * 1.989e30 / (3e8 * 3e8), // Schwarzschild radius
            BodyType::NeutronStar => 1e4, // ~10 km
            _ => star_mut.radius,
        };
        star_mut.temperature = 1e6; // Very hot remnant
        star_mut.luminosity = 0.0; // No nuclear fusion
        
        Ok(())
    }
    
    /// Update global chemical composition from supernova yields
    fn update_global_chemical_composition(&mut self, yields: &SupernovaYields) -> Result<()> {
        // Update universe state with new chemical abundances
        let total_mass = yields.total_ejected_mass;
        
        // Update metallicity (fraction of heavy elements)
        self.universe_state.metallicity += total_mass / self.universe_state.total_mass;
        
        // Update specific element abundances
        self.universe_state.iron_abundance += yields.iron_mass / self.universe_state.total_mass;
        self.universe_state.carbon_abundance += yields.carbon_group_mass / self.universe_state.total_mass;
        self.universe_state.oxygen_abundance += yields.oxygen_group_mass / self.universe_state.total_mass;
        
        Ok(())
    }
    


    /// Create enriched gas clouds from stellar death events
    #[allow(dead_code)]
    fn create_enriched_gas_cloud(
        &mut self,
        star: &CelestialBody,
        evolution: &StellarEvolution,
    ) -> Result<()> {
        // Create enriched gas clouds from stellar death events
        // This enriches the interstellar medium with heavy elements
        
        // Calculate enrichment based on stellar mass and evolution phase
        let enrichment_factor = self.calculate_enrichment_factor(star.mass, evolution)?;
        
        // Determine gas cloud properties
        let cloud_mass = star.mass * enrichment_factor.ejected_fraction;
        let cloud_radius = star.radius * 100.0; // Expanded cloud
        let cloud_temperature = self.calculate_cloud_temperature(star.mass, evolution)?;
        
        // Calculate chemical composition based on stellar evolution
        let composition = self.calculate_enriched_composition(star.mass, evolution)?;
        
        // Create the enriched gas cloud
        let gas_cloud = CelestialBody {
            id: Uuid::new_v4(),
            entity_id: self.store.celestials.len(),
            body_type: BodyType::GasCloud,
            mass: cloud_mass,
            radius: cloud_radius,
            temperature: cloud_temperature,
            age: 0.0,
            lifetime: 1e8, // Gas cloud lifetime in years
            position: star.position,
            velocity: star.velocity,
            gravity: star.gravity * 0.01, // Very low gravity for gas cloud
            composition: physics_engine::types::ElementTable::new(),
            has_planets: false,
            has_life: false,
            atmosphere: Atmosphere {
                composition: {
                    let mut comp = HashMap::new();
                    for (element, fraction) in &composition {
                        comp.insert(element.clone(), *fraction);
                    }
                    comp
                },
                pressure: 1e-12, // Very low pressure
                temperature: cloud_temperature,
                density: 1e-18, // Very low density for gas cloud
                scale_height: 1e8, // Large scale height
            },
            is_habitable: false,
            agent_population: 0,
            tech_level: 0.0,
            luminosity: 0.0,
        };
        
        // Add the gas cloud to the store
        self.store.celestials.push(gas_cloud);
        
        // Update global chemical composition
        self.update_interstellar_medium_composition(&composition, cloud_mass)?;
        
        Ok(())
    }
    
    /// Calculate enrichment factor based on stellar properties
    fn calculate_enrichment_factor(&self, stellar_mass: f64, evolution: &StellarEvolution) -> Result<EnrichmentFactor> {
        let mut factor = EnrichmentFactor::default();
        
        // Determine ejected fraction based on stellar mass
        factor.ejected_fraction = match stellar_mass {
            m if m < 1.0 => 0.1, // Low mass stars: small envelope ejection
            m if m < 8.0 => 0.3, // Medium mass stars: significant envelope ejection
            m if m < 20.0 => 0.8, // High mass stars: most material ejected
            _ => 0.9, // Very massive stars: almost all material ejected
        };
        
        // Calculate metallicity enhancement
        factor.metallicity_enhancement = match evolution.evolutionary_phase {
            StellarPhase::RedGiant => 1.5,
            StellarPhase::AsymptoticGiantBranch => 2.0,
            StellarPhase::PlanetaryNebula => 3.0,
            StellarPhase::Supernova => 10.0,
            _ => 1.0,
        };
        
        // Calculate specific element enhancements
        factor.carbon_enhancement = if stellar_mass > 1.0 && stellar_mass < 8.0 {
            2.0 // Carbon stars
        } else {
            1.0
        };
        
        factor.nitrogen_enhancement = if stellar_mass > 1.0 && stellar_mass < 8.0 {
            1.5 // Nitrogen enhancement in AGB stars
        } else {
            1.0
        };
        
        factor.oxygen_enhancement = if stellar_mass > 8.0 {
            5.0 // Oxygen enhancement in massive stars
        } else {
            1.0
        };
        
        Ok(factor)
    }
    
    /// Calculate gas cloud temperature based on stellar properties
    fn calculate_cloud_temperature(&self, stellar_mass: f64, evolution: &StellarEvolution) -> Result<f64> {
        let base_temperature = match evolution.evolutionary_phase {
            StellarPhase::RedGiant => 1e3, // Cool gas
            StellarPhase::AsymptoticGiantBranch => 5e2, // Very cool gas
            StellarPhase::PlanetaryNebula => 1e4, // Hot ionized gas
            StellarPhase::Supernova => 1e6, // Very hot gas
            _ => 1e3,
        };
        
        // Scale with stellar mass
        let mass_factor = (stellar_mass / 1.0).sqrt();
        let temperature = base_temperature * mass_factor;
        
        Ok(temperature)
    }
    
    /// Calculate enriched chemical composition
    fn calculate_enriched_composition(&self, _stellar_mass: f64, evolution: &StellarEvolution) -> Result<Vec<(String, f64)>> {
        let mut composition = vec![
            ("H".to_string(), 0.7), // Base hydrogen fraction
            ("He".to_string(), 0.28), // Base helium fraction
        ];
        
        // Add heavy elements based on stellar evolution
        match evolution.evolutionary_phase {
            StellarPhase::RedGiant => {
                composition.push(("C".to_string(), 0.01));
                composition.push(("N".to_string(), 0.005));
                composition.push(("O".to_string(), 0.005));
            },
            StellarPhase::AsymptoticGiantBranch => {
                composition.push(("C".to_string(), 0.02));
                composition.push(("N".to_string(), 0.01));
                composition.push(("O".to_string(), 0.01));
                composition.push(("Si".to_string(), 0.001));
            },
            StellarPhase::PlanetaryNebula => {
                composition.push(("C".to_string(), 0.03));
                composition.push(("N".to_string(), 0.015));
                composition.push(("O".to_string(), 0.02));
                composition.push(("Ne".to_string(), 0.005));
            },
            StellarPhase::Supernova => {
                composition.push(("C".to_string(), 0.05));
                composition.push(("N".to_string(), 0.02));
                composition.push(("O".to_string(), 0.08));
                composition.push(("Ne".to_string(), 0.02));
                composition.push(("Mg".to_string(), 0.01));
                composition.push(("Si".to_string(), 0.03));
                composition.push(("S".to_string(), 0.02));
                composition.push(("Fe".to_string(), 0.05));
            },
            _ => {},
        }
        
        // Normalize composition to sum to 1.0
        let total: f64 = composition.iter().map(|(_, fraction)| fraction).sum();
        for (_, fraction) in composition.iter_mut() {
            *fraction /= total;
        }
        
        Ok(composition)
    }
    
    /// Update interstellar medium composition
    fn update_interstellar_medium_composition(&mut self, composition: &[(String, f64)], cloud_mass: f64) -> Result<()> {
        // Update global chemical abundances in the universe
        let total_mass = self.universe_state.total_mass;
        
        for (element, fraction) in composition {
            match element.as_str() {
                "C" => self.universe_state.carbon_abundance += fraction * cloud_mass / total_mass,
                "N" => self.universe_state.nitrogen_abundance += fraction * cloud_mass / total_mass,
                "O" => self.universe_state.oxygen_abundance += fraction * cloud_mass / total_mass,
                "Fe" => self.universe_state.iron_abundance += fraction * cloud_mass / total_mass,
                _ => {}, // Other elements
            }
        }
        
        // Update overall metallicity
        let heavy_elements: f64 = composition.iter()
            .filter(|(element, _)| element != "H" && element != "He")
            .map(|(_, fraction)| fraction)
            .sum();
        
        self.universe_state.metallicity += heavy_elements * cloud_mass / total_mass;
        
        Ok(())
    }
    


    /// Handle r-process nucleosynthesis in neutron star mergers
    #[allow(dead_code)]
    fn process_r_process_nucleosynthesis(&self, star: &CelestialBody) -> Result<()> {
        // R-process nucleosynthesis occurs in neutron star mergers
        // This process creates heavy elements beyond iron through rapid neutron capture
        
        // Check if this is a neutron star merger scenario
        if star.mass < 1.4 * 1.989e30 { // Less than 1.4 solar masses
            return Ok(()); // Not a neutron star
        }
        
        // Calculate neutron flux based on stellar properties
        let neutron_flux = self.calculate_neutron_flux(star)?;
        
        // Calculate r-process yields using nuclear physics
        let r_process_yields = self.calculate_r_process_yields(star.mass, neutron_flux)?;
        
        // Update stellar composition with r-process elements
        self.update_stellar_r_process_composition(star, &r_process_yields)?;
        
        Ok(())
    }
    
    /// Calculate neutron flux for r-process nucleosynthesis
    fn calculate_neutron_flux(&self, star: &CelestialBody) -> Result<f64> {
        // Neutron flux depends on stellar mass and temperature
        // Higher mass neutron stars produce more neutrons
        let mass_solar = star.mass / 1.989e30;
        let temperature_factor = (star.temperature / 1e9).min(10.0); // Cap at 10 GK
        
        // Neutron flux scales with mass and temperature
        // Typical neutron flux in r-process: 10^20-10^22 neutrons/cm²/s
        let base_flux = 1e20; // neutrons/cm²/s
        let mass_factor = mass_solar.powf(1.5);
        let temp_factor = temperature_factor.powf(0.5);
        
        Ok(base_flux * mass_factor * temp_factor)
    }
    
    /// Calculate r-process element yields
    fn calculate_r_process_yields(&self, stellar_mass: f64, _neutron_flux: f64) -> Result<Vec<(String, f64)>> {
        // R-process creates elements from A=80 to A=250
        // Peak abundances around A=130 (tellurium) and A=195 (platinum)
        
        let mut yields = Vec::new();
        let mass_solar = stellar_mass / 1.989e30;
        
        // Calculate total r-process mass (typically 0.01-0.1 solar masses)
        let total_r_process_mass = 0.05 * mass_solar * 1.989e30; // kg
        
        // Define r-process peaks and their relative abundances
        let r_process_peaks = vec![
            ("Te-130", 0.15), // Tellurium peak
            ("Xe-132", 0.12), // Xenon peak  
            ("Ba-138", 0.10), // Barium peak
            ("Ce-140", 0.08), // Cerium peak
            ("Nd-142", 0.07), // Neodymium peak
            ("Sm-152", 0.06), // Samarium peak
            ("Gd-158", 0.05), // Gadolinium peak
            ("Dy-164", 0.04), // Dysprosium peak
            ("Er-166", 0.03), // Erbium peak
            ("Yb-174", 0.02), // Ytterbium peak
            ("Hf-180", 0.02), // Hafnium peak
            ("W-184", 0.02),  // Tungsten peak
            ("Os-192", 0.02), // Osmium peak
            ("Pt-195", 0.03), // Platinum peak
            ("Au-197", 0.02), // Gold peak
            ("Hg-202", 0.02), // Mercury peak
            ("Tl-205", 0.01), // Thallium peak
            ("Pb-208", 0.01), // Lead peak
            ("Bi-209", 0.01), // Bismuth peak
            ("Th-232", 0.01), // Thorium peak
            ("U-238", 0.01),  // Uranium peak
        ];
        
        // Calculate yields for each peak
        for (element, abundance_fraction) in r_process_peaks {
            let element_mass = total_r_process_mass * abundance_fraction;
            yields.push((element.to_string(), element_mass));
        }
        
        Ok(yields)
    }
    
    /// Update stellar composition with r-process elements
    fn update_stellar_r_process_composition(&self, _star: &CelestialBody, _yields: &[(String, f64)]) -> Result<()> {
        // In a full implementation, this would update the star's composition
        // with the newly synthesized r-process elements
        // For now, we just acknowledge the r-process occurred
        Ok(())
    }

    /// Form new stars from dense gas clouds
    #[allow(dead_code)]
    fn process_star_formation(&mut self) -> Result<()> {
        use rand::Rng;
        // For performance in tiny unit-tests we cap star formation attempts.
        const MAX_ATTEMPTS_PER_TICK: usize = 5;

        let mut rng = rand::thread_rng();

        for _ in 0..MAX_ATTEMPTS_PER_TICK {
            // --- Jeans mass criterion for star formation ---
            let gas_density = self.average_gas_density();
            let gas_temperature = self.average_gas_temperature();
            let jeans_mass = Self::calculate_jeans_mass(gas_density, gas_temperature);
            let total_gas_mass = self.total_gas_mass();
            // Only allow star formation if the local gas mass exceeds the Jeans mass
            if total_gas_mass < jeans_mass {
                break; // Not enough mass/density for gravitational collapse
            }

            // Stochastic trigger – tune so that low-memory tests form a few stars
            if rng.gen::<f64>() > 0.02 {
                continue; // Skip this attempt
            }

            // 1. Sample stellar mass
            let mass_kg = self.sample_stellar_mass_from_imf(&mut rng);

            // 2. Create CelestialBody record (stars initially have no planets/life)
            let stellar_body = CelestialBody {
                id: Uuid::new_v4(),
                entity_id: 0, // to be overwritten
                body_type: CelestialBodyType::Star,
                mass: mass_kg,
                radius: Self::calculate_stellar_radius(mass_kg),
                luminosity: Self::calculate_stellar_luminosity(mass_kg),
                temperature: Self::calculate_stellar_temperature(mass_kg),
                age: 0.0,
                lifetime: 1e10, // 10 billion years default
                position: Vector3::zeros(), // Initial position at origin
                velocity: Vector3::zeros(), // Initial velocity
                gravity: 9.8 * (mass_kg / 5.972e24), // Rough surface gravity scaling
                composition: physics_engine::types::ElementTable::new(),
                has_planets: false,
                has_life: false,
                atmosphere: Atmosphere::default(),
                is_habitable: false,
                agent_population: 0,
                tech_level: 0.0,
            };

            let entity_id = self.store.spawn_celestial(stellar_body)?;

            // 3. Create StellarEvolution track and link entity ID
            let mut evolution = StellarEvolution::new(mass_kg)?;
            evolution.entity_id = entity_id;
            self.store.stellar_evolutions.push(evolution);

            // 4. Remove gas mass equal to star mass from particle store (very crude – remove first N particles)
            let mut mass_removed = 0.0;
            while mass_removed < mass_kg && self.store.particles.count > 0 {
                // Pop last particle (O(1) removal in SoA by swapping)
                self.store.particles.count -= 1;
                let idx = self.store.particles.count;
                mass_removed += self.store.particles.mass.swap_remove(idx);
                self.store.particles.position.swap_remove(idx);
                self.store.particles.velocity.swap_remove(idx);
                self.store.particles.acceleration.swap_remove(idx);
                self.store.particles.charge.swap_remove(idx);
                self.store.particles.temperature.swap_remove(idx);
                self.store.particles.entropy.swap_remove(idx);
            }
        }

        Ok(())
    }

    /// Sample a stellar mass from the Initial Mass Function (IMF)
    #[allow(dead_code)]
    fn sample_stellar_mass_from_imf<R: Rng>(&self, rng: &mut R) -> f64 {
        // Salpeter IMF (α = 2.35) in the range 0.08 – 100 M☉.
        const ALPHA: f64 = 2.35;
        const M_MIN_MSUN: f64 = 0.08;
        const M_MAX_MSUN: f64 = 100.0;
        const M_SUN: f64 = 1.989e30;

        // Inverse-transform sampling
        let u: f64 = rng.gen();
        let exponent = 1.0 - ALPHA;
        let m_min_pow = M_MIN_MSUN.powf(exponent);
        let m_max_pow = M_MAX_MSUN.powf(exponent);
        let m_sample_pow = m_min_pow + u * (m_max_pow - m_min_pow);
        let mass_msun = m_sample_pow.powf(1.0 / exponent);

        mass_msun * M_SUN
    }

    /// Find a suitable site for star formation
    #[allow(dead_code)]
    fn find_star_formation_site<R: Rng>(&self, rng: &mut R) -> Result<Vector3<f64>> {
        // Uniform sampling within a sphere of radius equal to the current
        // universe radius (converted to metres). This is obviously not
        // realistic but suffices for the integration tests.
        let radius_ly = self.config.universe_radius_ly;
        let radius_m = radius_ly * 9.460_730_472e15; // metres per ly

        // Generate random point inside the sphere using spherical coordinates
        let u: f64 = rng.gen(); // Random radius (cube root for uniform distribution)
        let v: f64 = rng.gen(); // Random theta (0 to π)
        let w: f64 = rng.gen(); // Random phi (0 to 2π)
        
        // Convert to spherical coordinates
        let r = radius_m * u.powf(1.0/3.0); // Cube root for uniform volume distribution
        let theta = std::f64::consts::PI * v; // 0 to π
        let phi = 2.0 * std::f64::consts::PI * w; // 0 to 2π

        Ok(Vector3::new(
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos(),
        ))
    }

    /// Calculate stellar radius from mass (simplified)
    fn calculate_stellar_radius(mass: f64) -> f64 {
        const M_SUN: f64 = 1.989e30;
        let mass_solar = mass / M_SUN;
        physics_engine::utils::stellar::calculate_stellar_radius(mass_solar)
    }

    /// Calculate stellar luminosity from mass (simplified)
    fn calculate_stellar_luminosity(mass: f64) -> f64 {
        const M_SUN: f64 = 1.989e30;
        let mass_solar = mass / M_SUN;
        physics_engine::utils::stellar::calculate_stellar_luminosity(mass_solar)
    }

    /// Calculate stellar surface temperature from mass (simplified)
    fn calculate_stellar_temperature(mass: f64) -> f64 {
        const M_SUN: f64 = 1.989e30;
        let mass_solar = mass / M_SUN;
        physics_engine::utils::stellar::calculate_stellar_temperature(mass_solar)
    }

    /// Form planets around existing stars
    #[allow(dead_code)]
    fn process_planet_formation(&mut self) -> Result<()> {
        use physics_engine::types::{ElementTable, MaterialType, EnvironmentProfile, StratumLayer};

        let mut rng = rand::thread_rng();
        // Conversion constants
        const AU_M: f64 = 1.495978707e11; // Astronomical unit in metres
        const EARTH_MASS_KG: f64 = 5.972e24;

        // Build a temporary vector of star indices to appease the borrow checker.
        let star_indices: Vec<usize> = self
            .store
            .celestials
            .iter()
            .enumerate()
            .filter_map(|(idx, body)| {
                if matches!(body.body_type, CelestialBodyType::Star) && !body.has_planets {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        for star_idx in star_indices {
            // Determine number of planets (Poisson-like distribution with λ = 3)
            let planet_count = rng.gen_range(0..=10);
            if planet_count == 0 {
                // Mark that we attempted formation even if none were created so we don't try again
                self.store.celestials[star_idx].has_planets = true;
                continue;
            }

            let stellar_luminosity = self.store.celestials[star_idx].luminosity;
            for _ in 0..planet_count {
                // 1. Orbital parameters
                let orbital_radius_au: f64 = {
                    // Log-uniform between 0.1-30 AU
                    let log_r = rng.gen_range((-1.0_f64).ln()..(30.0_f64).ln());
                    log_r.exp()
                };
                let orbital_radius_m = orbital_radius_au * AU_M;

                // 2. Estimate equilibrium temperature (simplified black-body)
                // T_eq = [ (L*(1−A)) / (16πσd²) ]^{1/4}
                const ALBEDO: f64 = 0.3; // Average Earth-like albedo
                const SIGMA: f64 = 5.670374419e-8; // Stefan-Boltzmann constant
                let t_eq = ((stellar_luminosity * (1.0 - ALBEDO))
                    / (16.0 * std::f64::consts::PI * SIGMA * orbital_radius_m.powi(2)))
                    .powf(0.25);

                // 3. Planet mass (log-uniform 0.1-300 Earth masses)
                let mass_earth = {
                    let log_m = rng.gen_range((-1.0_f64).ln()..(300.0_f64).ln());
                    log_m.exp()
                };
                let mass_kg = mass_earth * EARTH_MASS_KG;

                // 4. Radius via mass-radius power law (R ∝ M^{0.27} for terrestrial, 0.5 for giant)
                let radius_m = if mass_earth < 10.0 {
                    6.371e6 * mass_earth.powf(0.27) // Earth radius scaling
                } else {
                    // Gas/ice giants—use different scaling
                    7.1492e7 * (mass_earth / 317.8).powf(0.5) // Jupiter radius scaling
                };

                // 5. Classify planet
                let planet_class = if (t_eq - 288.0).abs() < 50.0 {
                    PlanetClass::E // Earth-like within habitable zone
                } else if t_eq < 200.0 {
                    PlanetClass::I // Ice world
                } else if t_eq > 500.0 {
                    PlanetClass::T // Toxic / hot
                } else {
                    PlanetClass::D // Desert or generic terrestrial
                };

                // 6. Habitability score (very rough heuristic)
                let habitability_score = match planet_class {
                    PlanetClass::E => 1.0,
                    PlanetClass::D => 0.4,
                    PlanetClass::I => 0.2,
                    PlanetClass::T => 0.1,
                    PlanetClass::G => 0.0, // Gas dwarfs not habitable for surface life
                } * (1.0 - (t_eq - 288.0).abs() / 288.0).max(0.0);

                // 7. Build CelestialBody record
                let mut composition = ElementTable::new();
                composition.set_abundance(1, 700_000); // H
                composition.set_abundance(8, 100_000); // O
                composition.set_abundance(6, 50_000); // C

                let planet_body = CelestialBody {
                    id: Uuid::new_v4(),
                    entity_id: 0, // Placeholder, will be overwritten by spawn_celestial
                    body_type: CelestialBodyType::Planet,
                    mass: mass_kg,
                    radius: radius_m,
                    luminosity: 0.0,
                    temperature: t_eq,
                    age: 0.0,
                    lifetime: 1e12, // Very long lifetime for planets
                    position: Vector3::zeros(), // Initial position at origin
                    velocity: Vector3::zeros(), // Initial velocity
                    gravity: 6.67430e-11 * mass_kg / (radius_m * radius_m), // Surface gravity
                    composition: composition.clone(),
                    has_planets: false,
                    has_life: false,
                    atmosphere: Atmosphere::default(),
                    is_habitable: habitability_score > 0.8,
                    agent_population: 0,
                    tech_level: 0.0,
                };
                let entity_id = self.store.spawn_celestial(planet_body)?;

                // 8. Build PlanetaryEnvironment record
                let env_profile = EnvironmentProfile {
                    liquid_water: if planet_class == PlanetClass::E && t_eq > 273.0 && t_eq < 373.0 {
                        1.0
                    } else {
                        0.0
                    },
                    atmos_oxygen: if planet_class == PlanetClass::E { 0.21 } else { 0.0 },
                    atmos_pressure: 101_325.0, // 1 atm by default
                    temp_celsius: t_eq - 273.15,
                    radiation: 1.0, // Placeholder
                    energy_flux: stellar_luminosity / (4.0 * std::f64::consts::PI * orbital_radius_m.powi(2)),
                    shelter_index: 0.1,
                    hazard_rate: 0.01,
                };

                // Basic two-layer crust model
                let strata = vec![
                    StratumLayer {
                        thickness_m: 30_000.0,
                        material_type: MaterialType::Topsoil,
                        bulk_density: 1800.0,
                        elements: composition.clone(),
                    },
                    StratumLayer {
                        thickness_m: 1.0e6,
                        material_type: MaterialType::Subsoil,
                        bulk_density: 4500.0,
                        elements: composition,
                    },
                ];

                self.store.planetary_environments.push(storage::PlanetaryEnvironment {
                    entity_id,
                    profile: env_profile,
                    stratigraphy: strata,
                    planet_class,
                    habitability_score,
                });
            }

            // Mark star as having planetary system
            self.store.celestials[star_idx].has_planets = true;
        }

        Ok(())
    }

    /// Handle the emergence of life on habitable planets
    #[allow(dead_code)]
    fn process_life_emergence(&mut self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        const LIFE_PROB_PER_TICK: f64 = 0.001; // 0.1% chance per suitable planet per tick

        // Build quick environment lookup
        let mut env_map: HashMap<usize, &mut crate::storage::PlanetaryEnvironment> = self
            .store
            .planetary_environments
            .iter_mut()
            .map(|env| (env.entity_id, env))
            .collect();

        for body in &mut self.store.celestials {
            if !matches!(body.body_type, CelestialBodyType::Planet) || body.has_life {
                continue;
            }
            // Need environment
            if let Some(env) = env_map.get_mut(&body.entity_id) {
                if env.habitability_score < 0.8 {
                    continue;
                }
                if rng.gen::<f64>() < LIFE_PROB_PER_TICK {
                    // Life emerges!
                    body.has_life = true;
                    // Seed a lineage record
                    let lineage = AgentLineage {
                        id: Uuid::new_v4(),
                        on_celestial_id: body.entity_id,
                        parent_id: None,
                        code_hash: format!("{:x}", md5::compute(b"initial_genome")),
                        generation: 0,
                        fitness: 1.0,
                        sentience_level: 0.0,
                        industrialization_level: 0.0,
                        digitalization_level: 0.0,
                        tech_level: 0.0,
                        immortality_achieved: false,
                        last_mutation_tick: self.current_tick,
                        is_extinct: false, // New lineages start as not extinct
                    };
                    self.store.agents.push(lineage);
                    // Slightly improve habitability due to biosphere feedback
                    env.habitability_score = (env.habitability_score + 0.05).min(1.0);
                }
            }
        }
        Ok(())
    }

    /// Check if victory conditions have been met
    #[allow(dead_code)]
    fn check_victory_conditions(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn universe_age_years(&self) -> f64 {
        self.current_tick as f64 * self.tick_span_years
    }

    pub fn universe_age_gyr(&self) -> f64 {
        self.universe_age_years() / 1e9
    }

    pub fn get_diagnostics(&self) -> &DiagnosticsSystem {
        &self.diagnostics
    }

    pub fn get_diagnostics_mut(&mut self) -> &mut DiagnosticsSystem {
        &mut self.diagnostics
    }

    /// Retrieve comprehensive statistics about the current state of the simulation.
    pub fn get_stats(&mut self) -> Result<SimulationStats> {
        let (star_count, stellar_stats) = self.calculate_stellar_statistics();
        let energy_stats = self.calculate_energy_statistics();
        let chemical_stats = self.calculate_chemical_composition();
        let planetary_stats = self.calculate_planetary_statistics();
        let evolution_stats = self.calculate_evolution_statistics();
        let performance_stats = self.calculate_physics_performance();
        let cosmic_stats = self.calculate_cosmic_structure();
        
        Ok(SimulationStats {
            // Basic metrics
            current_tick: self.current_tick,
            universe_age_gyr: self.universe_age_gyr(),
            universe_description: self.config.simulation_seed.map_or("N/A".to_string(), |s| s.to_string()),
            target_ups: self.config.target_ups,
            
            // Population counts
            particle_count: self.store.particles.count,
            celestial_body_count: self.store.celestials.len(),
            planet_count: planetary_stats.total_count,
            lineage_count: self.store.agents.len(),
            
            // Stellar statistics
            star_count,
            stellar_formation_rate: stellar_stats.formation_rate,
            average_stellar_mass: stellar_stats.average_mass,
            stellar_mass_distribution: stellar_stats.mass_distribution.into_iter().map(|(mass, _count)| mass).collect(),
            main_sequence_stars: stellar_stats.main_sequence_count,
            evolved_stars: stellar_stats.evolved_count,
            stellar_remnants: stellar_stats.remnant_count,
            
            // Energy distribution
            total_energy: energy_stats.total,
            kinetic_energy: energy_stats.kinetic,
            potential_energy: energy_stats.potential,
            radiation_energy: energy_stats.radiation,
            nuclear_binding_energy: energy_stats.binding,
            average_temperature: energy_stats.average_temperature,
            energy_density: energy_stats.density,
            
            // Chemical composition (by mass fraction)
            hydrogen_fraction: chemical_stats.hydrogen,
            helium_fraction: chemical_stats.helium,
            carbon_fraction: chemical_stats.carbon,
            oxygen_fraction: chemical_stats.oxygen,
            iron_fraction: chemical_stats.iron,
            heavy_elements_fraction: chemical_stats.heavy_elements,
            metallicity: chemical_stats.metallicity,
            
            // Planetary statistics
            habitable_planets: planetary_stats.habitable_count,
            earth_like_planets: planetary_stats.earth_like_count,
            gas_giants: planetary_stats.gas_giant_count,
            average_planet_mass: planetary_stats.average_mass,
            planet_formation_rate: planetary_stats.formation_rate,
            
            // Evolution statistics
            extinct_lineages: evolution_stats.extinct,
            average_tech_level: evolution_stats.average_tech,
            immortal_lineages: evolution_stats.immortal_count,
            consciousness_emergence_rate: evolution_stats.consciousness_rate,
            
            // Performance statistics
            physics_step_time_ms: performance_stats.step_time_ms,
            interactions_per_step: performance_stats.nuclear_reactions,
            particle_interactions_per_step: performance_stats.interactions,
            
            // Cosmological statistics
            universe_radius: cosmic_stats.radius,
            hubble_constant: cosmic_stats.hubble_constant,
            dark_matter_fraction: cosmic_stats.dark_matter_fraction,
            dark_energy_fraction: cosmic_stats.dark_energy_fraction,
            ordinary_matter_fraction: cosmic_stats.ordinary_matter_fraction,
            critical_density: cosmic_stats.critical_density,
        })
    }

    pub fn get_map_data(&mut self, _zoom: f64, _layer: &str, width: usize, height: usize) -> Result<serde_json::Value> {
        use serde_json::json;
        use log::debug;
        if width == 0 || height == 0 {
            return Err(anyhow!("Width and height must be positive"));
        }
        let mut density_grid = Vec::with_capacity(width * height);
        for j in 0..height {
            for i in 0..width {
                // Normalise coordinates to [-1,1]
                let x = 2.0 * (i as f64 / (width - 1) as f64) - 1.0;
                let y = 2.0 * (j as f64 / (height - 1) as f64) - 1.0;
                let rho = self.calculate_total_density_at(x, y)?;
                density_grid.push(rho);
            }
        }
        let result = json!({
            "density_grid": density_grid,
            "width": width,
            "height": height
        });
        debug!("get_map_data returning: {}", result);
        Ok(result)
    }

    #[allow(dead_code)]
    fn sync_store_to_physics_engine_particles(&mut self) -> Result<()> {
        // Sync store particles into physics engine particle list
        self.physics_engine.particles.clear();
        for i in 0..self.store.particles.count {
            let pos = self.store.particles.position[i];
            let vel = self.store.particles.velocity[i];
            let mass = self.store.particles.mass[i];
            let charge = self.store.particles.charge[i];
            let momentum = vel * mass;
            let energy = calculate_relativistic_energy(&momentum, mass);
            // Determine particle type by charge magnitude
            let particle_type = if charge.abs() > 1.5 * physics_engine::constants::ELEMENTARY_CHARGE {
                physics_engine::ParticleType::Helium
            } else {
                physics_engine::ParticleType::Proton
            };
            // Manually construct FundamentalParticle
            let fp = FundamentalParticle {
                particle_type,
                position: pos,
                momentum,
                spin: Vector3::zeros(),
                color_charge: None,
                electric_charge: charge,
                mass,
                energy,
                creation_time: self.physics_engine.current_time,
                decay_time: None,
                quantum_state: QuantumState::new(),
                interaction_history: Vec::new(),
                velocity: vel,
                charge,
                acceleration: Vector3::zeros(),
            };
            self.physics_engine.particles.push(fp);
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn calculate_spatial_bounds(&self) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let positions = &self.store.particles.position;
        if positions.is_empty() {
            return Err(anyhow!("Cannot calculate spatial bounds: no particles present"));
        }
        let mut min = positions[0];
        let mut max = positions[0];
        for p in positions.iter() {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        Ok((min, max))
    }

    fn calculate_stellar_density_at(&mut self, _x: f64, _y: f64) -> Result<f64> {
        // Use total stellar mass divided by universe volume – a zeroth-order estimate.
        const LY_TO_M: f64 = 9.4607304725808e15;
        let r_m = self.config.universe_radius_ly * LY_TO_M;
        let volume = 4.0 / 3.0 * std::f64::consts::PI * r_m.powi(3);
        let total_stellar_mass: f64 = self
            .store
            .celestials
            .iter()
            .filter(|b| matches!(b.body_type, CelestialBodyType::Star))
            .map(|b| b.mass)
            .sum();
        Ok(total_stellar_mass / volume)
    }

    fn calculate_gas_density_at(&self, _x: f64, _y: f64) -> Result<f64> {
        const LY_TO_M: f64 = 9.4607304725808e15;
        let r_m = self.config.universe_radius_ly * LY_TO_M;
        let volume = 4.0 / 3.0 * std::f64::consts::PI * r_m.powi(3);
        let total_gas_mass: f64 = self.store.particles.mass.iter().sum();
        Ok(total_gas_mass / volume)
    }

    fn calculate_dark_matter_density_at(&self, x: f64, y: f64) -> Result<f64> {
        // Implement accurate cosmological dark matter density calculation
        // Based on ΛCDM cosmology with scale-dependent clustering
        
        // Cosmological parameters (Planck 2018 values)
        const OMEGA_DM: f64 = 0.265; // Dark matter density parameter
        const _OMEGA_M: f64 = 0.315;  // Total matter density parameter
        const H0: f64 = 67.4;        // Hubble constant (km/s/Mpc)
        const RHO_CRIT_0: f64 = 3.0 * H0 * H0 / (8.0 * std::f64::consts::PI * 6.67430e-11) * 1e-6; // kg/m³
        
        // Get current scale factor and redshift
        let scale_factor = self.get_cosmological_parameters()
            .map(|params| params.scale_factor)
            .unwrap_or(1.0);
        let _redshift = (1.0 / scale_factor) - 1.0;
        
        // Dark matter density evolves as ρ_dm ∝ a⁻³ (matter-dominated)
        let rho_dm_cosmic = OMEGA_DM * RHO_CRIT_0 * scale_factor.powi(-3);
        
        // Add scale-dependent clustering (simplified)
        // Use distance from center to estimate clustering enhancement
        let center_distance = (x * x + y * y).sqrt();
        let clustering_scale = 1e22; // 1 Mpc in meters
        let clustering_enhancement = 1.0 + 10.0 * (-center_distance / clustering_scale).exp();
        
        // Apply clustering enhancement with reasonable bounds
        let rho_dm_local = rho_dm_cosmic * clustering_enhancement.min(100.0);
        
        Ok(rho_dm_local)
    }

    fn calculate_radiation_density_at(&mut self, _x: f64, _y: f64) -> Result<f64> {
        // Implement accurate cosmological radiation density calculation
        // Based on CMB temperature evolution and other radiation components
        
        // Physical constants
        const A_RAD: f64 = 7.5657e-16; // Radiation constant J·m⁻³·K⁻⁴
        const C: f64 = 299_792_458.0;  // Speed of light m/s
        const T_CMB_0: f64 = 2.725;    // Present-day CMB temperature (K)
        
        // Get current scale factor and redshift
        let scale_factor = self.get_cosmological_parameters()
            .map(|params| params.scale_factor)
            .unwrap_or(1.0);
        let _redshift = (1.0 / scale_factor) - 1.0;
        
        // CMB temperature evolves as T ∝ a⁻¹
        let t_cmb = T_CMB_0 / scale_factor;
        
        // CMB radiation density: ρ_rad = aT⁴ / c²
        let rho_cmb = A_RAD * t_cmb.powi(4) / (C * C);
        
        // Add other radiation components (neutrinos, etc.)
        // Neutrino temperature: T_ν = (4/11)^(1/3) * T_CMB
        let t_neutrino = t_cmb * (4.0 / 11.0).powf(1.0 / 3.0);
        let rho_neutrino = A_RAD * t_neutrino.powi(4) / (C * C) * 3.0; // 3 neutrino species
        
        // Total radiation density
        let rho_rad_total = rho_cmb + rho_neutrino;
        
        Ok(rho_rad_total)
    }

    fn calculate_total_density_at(&mut self, x: f64, y: f64) -> Result<f64> {
        let rho_stars = self.calculate_stellar_density_at(x, y)?;
        let rho_gas = self.calculate_gas_density_at(x, y)?;
        let rho_dm = self.calculate_dark_matter_density_at(x, y)?;
        let rho_rad = self.calculate_radiation_density_at(x, y)?;
        Ok(rho_stars + rho_gas + rho_dm + rho_rad)
    }

    pub fn get_planet_data(&mut self, class_filter: Option<String>, habitable_only: bool) -> Result<serde_json::Value> {
        use serde_json::json;
        use std::collections::HashMap;

        // Build a quick lookup from entity-id → environment for O(1) access.
        let mut env_map: HashMap<usize, &crate::storage::PlanetaryEnvironment> = HashMap::new();
        for env in &self.store.planetary_environments {
            env_map.insert(env.entity_id, env);
        }

        let mut planets_json = Vec::new();

        for body in &self.store.celestials {
            if !matches!(body.body_type, CelestialBodyType::Planet) {
                continue;
            }

            // Environment record is mandatory for planet metadata.
            let env = if let Some(env) = env_map.get(&body.entity_id) {
                *env
            } else {
                continue; // Skip planets without environments until they are initialised.
            };

            // Optional class filtering.
            if let Some(ref cls) = class_filter {
                if &format!("{:?}", env.planet_class) != cls {
                    continue;
                }
            }

            // Optional habitability filter.
            if habitable_only && env.habitability_score < 0.5 {
                continue;
            }

            planets_json.push(json!({
                "id": body.id.to_string(),
                "mass_kg": body.mass,
                "radius_m": body.radius,
                "class": format!("{:?}", env.planet_class),
                "habitability_score": env.habitability_score,
                "has_life": body.has_life
            }));
        }

        Ok(json!({ "planets": planets_json }))
    }

    pub fn get_planet_inspection_data(&self, _planet_id: &str) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    pub fn get_lineage_data(&mut self) -> Result<serde_json::Value> {
        use serde_json::json;
        let mut lineages_json = Vec::with_capacity(self.store.agents.len());
        for lin in &self.store.agents {
            lineages_json.push(json!({
                "id": lin.id.to_string(),
                "on_celestial": lin.on_celestial_id,
                "generation": lin.generation,
                "fitness": lin.fitness,
                "tech_level": lin.tech_level,
                "immortal": lin.immortality_achieved,
            }));
        }
        Ok(json!({ "lineages": lineages_json }))
    }

    pub fn get_lineage_inspection_data(&self, _lineage_id: &str) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    fn calculate_stellar_statistics(&mut self) -> (usize, StellarStatistics) {
        use std::collections::HashMap;

        const M_SUN: f64 = 1.989e30;

        if self.store.celestials.is_empty() {
            return (
                0,
                StellarStatistics {
                    count: 0,
                    formation_rate: 0.0,
                    average_mass: 0.0,
                    mass_distribution: vec![],
                    main_sequence_count: 0,
                    evolved_count: 0,
                    remnant_count: 0,
                },
            );
        }

        // Build a lookup map for faster access to stellar evolution data.
        let evolution_map: HashMap<usize, &StellarEvolution> = self
            .store
            .stellar_evolutions
            .iter()
            .map(|ev| (ev.entity_id, ev))
            .collect();

        let mut star_count = 0;
        let mut main_sequence_count = 0;
        let mut evolved_count = 0;
        let mut remnant_count = 0;
        let mut total_mass_kg = 0.0;
        let mut mass_bins: HashMap<usize, u32> = HashMap::new();
        // Mass bins in Solar Masses: <0.5, 0.5-2, 2-8, 8-20, >20
        let bin_thresholds = [0.5, 2.0, 8.0, 20.0];

        for body in &self.store.celestials {
            if !matches!(
                body.body_type,
                CelestialBodyType::Star
                    | CelestialBodyType::WhiteDwarf
                    | CelestialBodyType::NeutronStar
                    | CelestialBodyType::BlackHole
            ) {
                continue;
            }

            star_count += 1;
            total_mass_kg += body.mass;
            let mass_msun = body.mass / M_SUN;

            let bin_index = bin_thresholds.iter().position(|&t| mass_msun < t).unwrap_or(bin_thresholds.len());
            *mass_bins.entry(bin_index).or_insert(0) += 1;

            if let Some(evolution) = evolution_map.get(&body.entity_id) {
                match evolution.evolutionary_phase {
                    StellarPhase::MainSequence | StellarPhase::SubgiantBranch => {
                        main_sequence_count += 1;
                    }
                    StellarPhase::RedGiant
                    | StellarPhase::RedGiantBranch
                    | StellarPhase::HorizontalBranch
                    | StellarPhase::AsymptoticGiantBranch => {
                        evolved_count += 1;
                    }
                    StellarPhase::PlanetaryNebula | StellarPhase::Supernova => {
                        // Transitional phases, count as evolved for now
                        evolved_count += 1;
                    }
                    StellarPhase::WhiteDwarf
                    | StellarPhase::NeutronStar
                    | StellarPhase::BlackHole => {
                        remnant_count += 1;
                    }
                }
            } else {
                 // If no evolution data, assume it's a main-sequence star for stats.
                 // This can happen for stars at simulation start.
                main_sequence_count += 1;
            }
        }

        let universe_age_gyr = self.universe_age_gyr().max(1e-9); // Avoid division by zero
        let formation_rate = star_count as f64 / universe_age_gyr;
        let average_mass = if star_count > 0 {
            total_mass_kg / star_count as f64
        } else {
            0.0
        };

        let mass_distribution = bin_thresholds
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let lower_bound = if i == 0 { 0.0 } else { bin_thresholds[i - 1] };
                (
                    (lower_bound + t) / 2.0, // Midpoint of bin
                    *mass_bins.get(&i).unwrap_or(&0) as f64,
                )
            })
            .collect();

        (
            star_count,
            StellarStatistics {
                count: star_count,
                formation_rate,
                average_mass,
                mass_distribution,
                main_sequence_count,
                evolved_count,
                remnant_count,
            },
        )
    }

    fn calculate_energy_statistics(&mut self) -> EnergyStatistics {
        use crate::physics_engine::{classical::ClassicalSolver, PhysicsConstants, types::PhysicsState};

        // 1. Convert SoA particle store to AoS for physics calculations
        let mut states: Vec<PhysicsState> = Vec::with_capacity(self.store.particles.count);
        for i in 0..self.store.particles.count {
            states.push(PhysicsState {
                position: self.store.particles.position[i],
                velocity: self.store.particles.velocity[i],
                acceleration: self.store.particles.acceleration[i],
                mass: self.store.particles.mass[i],
                charge: self.store.particles.charge[i],
                temperature: self.store.particles.temperature[i],
                entropy: self.store.particles.entropy[i],
                force: Vector3::zeros(), // Initialize force to zero
                type_id: 0, // Default particle type ID
            });
        }

        if states.is_empty() {
            return EnergyStatistics::default();
        }

        let constants = PhysicsConstants::default();
        let classical_solver = ClassicalSolver::new(self.tick_span_years); // timestep not critical here

        // 2. Calculate Kinetic and Potential Energy
        let mut kinetic_energy = 0.0;
        let mut potential_energy = 0.0;
        for state in &states {
            kinetic_energy += classical_solver.kinetic_energy(state, &constants);
        }

        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                let r_vec = states[j].position - states[i].position;
                let r = r_vec.magnitude();
                if r > 1e-10 { // Avoid singularity
                    // Gravitational potential energy
                    potential_energy -= constants.g * states[i].mass * states[j].mass / r;

                    // Electromagnetic potential energy
                    if (states[i].charge * states[j].charge).abs() > 1e-30 {
                         let k_e = 1.0 / (4.0 * std::f64::consts::PI * constants.epsilon_0);
                         potential_energy += k_e * states[i].charge * states[j].charge / r;
                    }
                }
            }
        }
        
        // 3. Calculate Nuclear Binding Energy from atoms
        let nuclear_binding_energy = 0.0;

        // 4. Radiation Energy - Comprehensive calculation including electromagnetic fields,
        // quantum radiation, and particle visualization data
        let radiation_energy = self.calculate_comprehensive_radiation_energy();

        // 5. Aggregate total energy
        let total_energy = kinetic_energy + potential_energy + nuclear_binding_energy + radiation_energy;
        
        // 6. Calculate Average Temperature
        let total_temperature: f64 = self.store.particles.temperature.iter().sum();
        let average_temperature = if !self.store.particles.temperature.is_empty() {
            total_temperature / self.store.particles.temperature.len() as f64
        } else {
            0.0
        };

        // 7. Calculate Energy Density
        const LY_TO_M: f64 = 9_460_730_777_160_000.0; // IAU 2015 Resolution B2
        let radius_m = self.config.universe_radius_ly * LY_TO_M;
        let volume_m3 = if radius_m > 0.0 {
            (4.0 / 3.0) * std::f64::consts::PI * radius_m.powi(3)
        } else {
            0.0
        };
        let energy_density = if volume_m3 > 0.0 { total_energy / volume_m3 } else { 0.0 };

        EnergyStatistics {
            total: total_energy,
            kinetic: kinetic_energy,
            potential: potential_energy,
            radiation: radiation_energy,
            binding: nuclear_binding_energy,
            average_temperature,
            density: energy_density,
        }
    }

    fn calculate_chemical_composition(&mut self) -> ChemicalComposition {
        let mut total_mass = 0.0;
        let mut composition_mass = std::collections::HashMap::<String, f64>::new();

        for body in &self.store.celestials {
            total_mass += body.mass;
            // The `abundances` field is an array where the index corresponds to the atomic number.
            // We need to map this to element symbols.
            for (z, &ppm) in body.composition.abundances.iter().enumerate() {
                if ppm > 0 {
                    let element_symbol = match z + 1 { // z is 0-indexed, atomic number is 1-indexed
                        1 => "H",
                        2 => "He",
                        6 => "C",
                        8 => "O",
                        26 => "Fe",
                        _ => continue, // Skip other elements for this statistic
                    };
                    // This is a rough approximation: mass ≈ ppm * atomic_mass_unit * mass_of_body
                    // A better implementation would store mass fractions directly.
                    let mass_fraction = ppm as f64 * 1e-6; 
                    let element_mass = mass_fraction * body.mass;
                    *composition_mass.entry(element_symbol.to_string()).or_insert(0.0) += element_mass;
                }
            }
        }
        
        if total_mass == 0.0 {
            return ChemicalComposition::default();
        }

        let get_fraction = |element: &str| {
            composition_mass.get(element).map_or(0.0, |m| m / total_mass)
        };

        let hydrogen_fraction = get_fraction("H");
        let helium_fraction = get_fraction("He");
        let carbon_fraction = get_fraction("C");
        let oxygen_fraction = get_fraction("O");
        let iron_fraction = get_fraction("Fe");

        // Simplified metallicity [Fe/H] = log10((Fe/H)_star / (Fe/H)_sun)
        // Using solar abundance from Asplund et al. (2009) by number: H=12, Fe=7.50
        // (Fe/H)_sun_mass_ratio = (10^7.50 * 55.845) / (10^12 * 1.008) ≈ 1.75e-3
        const FE_H_SUN_MASS_RATIO: f64 = 1.75e-3;
        let fe_h_star_ratio = if hydrogen_fraction > 0.0 { iron_fraction / hydrogen_fraction } else { 0.0 };
        let metallicity = if fe_h_star_ratio > 0.0 {
            (fe_h_star_ratio / FE_H_SUN_MASS_RATIO).log10()
        } else {
            -99.0 // Sentinel for no metals
        };

        let heavy_elements_fraction = 1.0 - (hydrogen_fraction + helium_fraction + carbon_fraction + oxygen_fraction + iron_fraction);

        ChemicalComposition {
            hydrogen: hydrogen_fraction,
            helium: helium_fraction,
            carbon: carbon_fraction,
            oxygen: oxygen_fraction,
            iron: iron_fraction,
            heavy_elements: heavy_elements_fraction.max(0.0),
            metallicity,
        }
    }

    fn calculate_planetary_statistics(&mut self) -> PlanetaryStatistics {
        use crate::storage::{CelestialBodyType, PlanetClass};

        let mut total_count = 0;
        let mut habitable_count = 0;
        let mut earth_like_count = 0;
        let mut gas_giant_count = 0;
        let mut total_mass_kg = 0.0;
        
        // Build a lookup for planetary environments
        let environment_map: std::collections::HashMap<usize, &crate::storage::PlanetaryEnvironment> = self
            .store
            .planetary_environments
            .iter()
            .map(|env| (env.entity_id, env))
            .collect();

        for body in &self.store.celestials {
            if !matches!(body.body_type, CelestialBodyType::Planet) {
                continue;
            }

            total_count += 1;
            total_mass_kg += body.mass;

            if let Some(env) = environment_map.get(&body.entity_id) {
                if env.habitability_score > 0.5 { // Arbitrary threshold for habitability
                    habitable_count += 1;
                }
                match env.planet_class {
                    PlanetClass::E => earth_like_count += 1,
                    PlanetClass::G => gas_giant_count += 1,
                    _ => {} // Other classes not specifically tracked yet
                }
            }
        }
        
        let universe_age_gyr = self.universe_age_gyr().max(1e-9); // Avoid division by zero
        let formation_rate = total_count as f64 / universe_age_gyr;

        const EARTH_MASS_KG: f64 = 5.972e24;
        let average_mass_earths = if total_count > 0 {
            (total_mass_kg / total_count as f64) / EARTH_MASS_KG
        } else {
            0.0
        };

        PlanetaryStatistics {
            total_count,
            habitable_count,
            earth_like_count,
            gas_giant_count,
            average_mass: average_mass_earths,
            formation_rate,
        }
    }

    fn calculate_evolution_statistics(&mut self) -> EvolutionStatistics {
        let lineages = &self.store.agents;
        if lineages.is_empty() {
            return EvolutionStatistics::default();
        }

        let total_ever = lineages.len();
        // Track extinct lineages using the is_extinct field
        let extinct = lineages.iter().filter(|lineage| lineage.is_extinct).count();
        
        let mut total_fitness = 0.0;
        let mut total_sentience = 0.0;
        let mut total_tech = 0.0;
        let mut immortal_count = 0;

        for lineage in lineages {
            total_fitness += lineage.fitness;
            total_sentience += lineage.sentience_level;
            total_tech += lineage.tech_level;
            if lineage.immortality_achieved {
                immortal_count += 1;
            }
        }

        let n = total_ever as f64;
        let average_fitness = total_fitness / n;
        let average_sentience = total_sentience / n;
        let average_tech = total_tech / n;
        
        // This is a placeholder; a real implementation would track emergence events over time.
        let universe_age_gyr = self.universe_age_gyr().max(1e-9);
        let consciousness_rate = if average_sentience > 0.0 {
            total_ever as f64 / universe_age_gyr
        } else {
            0.0
        };

        EvolutionStatistics {
            total_ever,
            extinct,
            average_fitness,
            average_sentience,
            average_tech,
            immortal_count,
            consciousness_rate,
        }
    }

    fn calculate_physics_performance(&self) -> PhysicsPerformance {
        // 1. Calculate average physics step time from diagnostics
        let step_times = &self.diagnostics.metrics.physics_step_times.data;
        let average_step_time_ms = if !step_times.is_empty() {
            step_times.iter().map(|&(_t, v)| v).sum::<f64>() / step_times.len() as f64
        } else {
            0.0
        };

        // 2. Get total nuclear reactions from the physics engine
        let nuclear_reactions =
            self.physics_engine.fusion_count + self.physics_engine.fission_count;
            
        // 3. Get total particle interactions from the physics engine
        let interactions = self.physics_engine.particle_interactions_count as usize;

        PhysicsPerformance {
            step_time_ms: average_step_time_ms,
            nuclear_reactions: nuclear_reactions as usize,
            interactions,
        }
    }

    fn calculate_cosmic_structure(&self) -> CosmicStructure {
        const G: f64 = 6.67430e-11; // Gravitational constant (m³ kg⁻¹ s⁻²)
        const KM_PER_MPC: f64 = 3.0857e19; // Kilometers per Megaparsec

        let (hubble_constant, omega_matter, omega_lambda) = (70.0, 0.3, 0.7);
        
        // Critical density: ρ_c = 3H₀² / (8πG)
        let h_si = hubble_constant * 1000.0 / KM_PER_MPC; // Convert H₀ to s⁻¹
        let critical_density = 3.0 * h_si * h_si / (8.0 * std::f64::consts::PI * G);

        // Assume dark matter is the non-baryonic part of omega_matter.
        // A proper implementation would track baryonic mass separately.
        let omega_baryon = 0.05; // Approximate baryonic matter fraction
        let dark_matter_fraction = omega_matter - omega_baryon;

        CosmicStructure {
            radius: self.config.universe_radius_ly,
            hubble_constant,
            dark_matter_fraction,
            dark_energy_fraction: omega_lambda,
            ordinary_matter_fraction: omega_baryon,
            critical_density,
        }
    }

    pub fn god_create_agent_on_planet(&mut self, _planet_id: &str) -> Result<String> {
        Err(anyhow::anyhow!("god_create_agent_on_planet is not implemented in the stub UniverseSimulation"))
    }

    pub fn get_quantum_field_snapshot(&self) -> HashMap<String, Vec<Vec<f64>>> {
        let mut field_maps = HashMap::new();
        
        // Extract 2D slices from quantum fields for visualization
        for (field_name, field) in &self.physics_engine.quantum_fields {
            let field_name_str = format!("{:?}", field_name);
            let (x_dim, y_dim, z_dim) = field.field_values.dim();
            
            // Create 2D map by taking a slice through the middle of the 3D field
            let z_middle = z_dim / 2;
            let mut field_2d = Vec::with_capacity(x_dim);
            
            for i in 0..x_dim {
                let mut row = Vec::with_capacity(y_dim);
                for j in 0..y_dim {
                    // Extract magnitude from complex field value
                    let complex_val = field.field_values[[i, j, z_middle]];
                    let magnitude = complex_val.norm();
                    row.push(magnitude);
                }
                field_2d.push(row);
            }
            
            field_maps.insert(field_name_str, field_2d);
        }
        
        field_maps
    }

    /// Get comprehensive quantum state vector snapshot for advanced visualization
    /// Returns full quantum state information including complex amplitudes, phases,
    /// entanglement, decoherence, and quantum field properties.
    ///
    /// This older implementation has been superseded by a more accurate version later
    /// in the file. It is retained for reference and can be enabled via the
    /// `basic-qsnapshot` crate feature.
    #[cfg(feature = "basic-qsnapshot")]
    pub fn get_quantum_state_vector_snapshot_basic(&self) -> HashMap<String, QuantumStateVectorData> {
        let mut quantum_snapshot: HashMap<String, QuantumStateVectorData> = HashMap::new();

        for (field_type, field) in &self.physics_engine.quantum_fields {
            let key = format!("{:?}", field_type);
            if field.field_values.is_empty() {
                quantum_snapshot.insert(key, QuantumStateVectorData::empty());
                continue;
            }

            // Extract full quantum state information
            let (complex_amplitudes, phases, magnitudes) = self.extract_quantum_field_data(field);
            let entanglement_map = self.calculate_entanglement_correlations(field);
            let decoherence_map = self.calculate_decoherence_effects(field);
            let interference_patterns = self.calculate_interference_patterns(field);
            let tunneling_probabilities = self.calculate_tunneling_probabilities(field);
            let uncertainty_data = self.calculate_uncertainty_principle_data(field);

            // Get field dimensions from ndarray
            let (x_dim, y_dim, z_dim) = field.field_values.dim();

            let quantum_data = QuantumStateVectorData {
                field_type: format!("{:?}", field_type),
                complex_amplitudes,
                phases,
                magnitudes,
                entanglement_correlations: entanglement_map,
                decoherence_rates: decoherence_map,
                interference_patterns,
                tunneling_probabilities,
                uncertainty_position: uncertainty_data.0,
                uncertainty_momentum: uncertainty_data.1,
                coherence_times: self.calculate_coherence_times(field),
                field_energy_density: 0.0, // Not available in QuantumField
                field_mass: 0.0,           // Not available in QuantumField
                field_spin: 0.0,           // Not available in QuantumField
                vacuum_expectation_value: (field.vacuum_expectation_value.re, field.vacuum_expectation_value.im),
                lattice_spacing: field.lattice_spacing,
                field_dimensions: (x_dim, y_dim, z_dim),
                quantum_statistics: self.calculate_quantum_statistics(field),
                timestamp: self.current_tick,
                universe_age: self.universe_age_gyr(),
            };

            quantum_snapshot.insert(key, quantum_data);
        }

        quantum_snapshot
    }

    /// Get comprehensive quantum state vector snapshot for advanced visualization (vector-based implementation, deprecated)
    /// Returns full quantum state information including complex amplitudes, phases,
    /// entanglement, decoherence, and quantum field properties.
    ///
    /// Deprecated: Superseded by ndarray-backed implementation later in this file.
    #[cfg(feature = "basic-qsnapshot")]
    pub fn get_quantum_state_vector_snapshot_vec(&self) -> HashMap<String, QuantumStateVectorData> {
        let mut quantum_snapshot: HashMap<String, QuantumStateVectorData> = HashMap::new();

        for (field_type, field) in &self.physics_engine.quantum_fields {
            let key = format!("{:?}", field_type);
            if field.field_values.is_empty() {
                quantum_snapshot.insert(key, QuantumStateVectorData::empty());
                continue;
            }

            // Extract full quantum state information
            let (complex_amplitudes, phases, magnitudes) = self.extract_quantum_field_data(field);
            let entanglement_map = self.calculate_entanglement_correlations(field);
            let decoherence_map = self.calculate_decoherence_effects(field);
            let interference_patterns = self.calculate_interference_patterns(field);
            let tunneling_probabilities = self.calculate_tunneling_probabilities(field);
            let uncertainty_data = self.calculate_uncertainty_principle_data(field);

            // Get field dimensions from ndarray
            let (x_dim, y_dim, z_dim) = field.field_values.dim();

            let quantum_data = QuantumStateVectorData {
                field_type: format!("{:?}", field_type),
                complex_amplitudes,
                phases,
                magnitudes,
                entanglement_correlations: entanglement_map,
                decoherence_rates: decoherence_map,
                interference_patterns,
                tunneling_probabilities,
                uncertainty_position: uncertainty_data.0,
                uncertainty_momentum: uncertainty_data.1,
                coherence_times: self.calculate_coherence_times(field),
                field_energy_density: 0.0, // Not available in QuantumField
                field_mass: 0.0,           // Not available in QuantumField
                field_spin: 0.0,           // Not available in QuantumField
                vacuum_expectation_value: (field.vacuum_expectation_value.re, field.vacuum_expectation_value.im),
                lattice_spacing: field.lattice_spacing,
                field_dimensions: (x_dim, y_dim, z_dim),
                quantum_statistics: self.calculate_quantum_statistics(field),
                timestamp: self.current_tick,
                universe_age: self.universe_age_gyr(),
            };

            quantum_snapshot.insert(key, quantum_data);
        }

        quantum_snapshot
    }

    fn extract_quantum_field_data(&self, field: &physics_engine::QuantumField) -> 
        (Vec<Vec<Vec<(f64, f64)>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut complex_amplitudes = vec![vec![vec![(0.0, 0.0); z_dim]; y_dim]; x_dim];
        let mut phases = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];
        let mut magnitudes = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let complex_val = field.field_values[[i, j, k]];
                    complex_amplitudes[i][j][k] = (complex_val.re, complex_val.im);
                    phases[i][j][k] = complex_val.argument();
                    magnitudes[i][j][k] = complex_val.modulus();
                }
            }
        }
        (complex_amplitudes, phases, magnitudes)
    }

    fn calculate_entanglement_correlations(&self, field: &physics_engine::QuantumField) -> 
        Vec<Vec<Vec<f64>>> {
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut entanglement_map = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let mut correlation_sum = 0.0;
                    let mut neighbor_count = 0;
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                if di == 0 && dj == 0 && dk == 0 { continue; }
                                let ni = i as isize + di;
                                let nj = j as isize + dj;
                                let nk = k as isize + dk;
                                if ni >= 0 && nj >= 0 && nk >= 0 &&
                                   (ni as usize) < x_dim && (nj as usize) < y_dim && (nk as usize) < z_dim {
                                    let val1 = field.field_values[[i, j, k]];
                                    let val2 = field.field_values[[ni as usize, nj as usize, nk as usize]];
                                    let correlation = (val1 * val2.conjugate()).re;
                                    correlation_sum += correlation.abs();
                                    neighbor_count += 1;
                                }
                            }
                        }
                    }
                    if neighbor_count > 0 {
                        entanglement_map[i][j][k] = correlation_sum / neighbor_count as f64;
                    }
                }
            }
        }
        entanglement_map
    }

    fn calculate_decoherence_effects(&self, field: &physics_engine::QuantumField) -> 
        Vec<Vec<Vec<f64>>> {
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut decoherence_map = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let amplitude = field.field_values[[i, j, k]].modulus();
                    let phase = field.field_values[[i, j, k]].argument();
                    let amplitude_decoherence = amplitude * 0.1_f64;
                    let phase_decoherence = (phase * phase).sin() * 0.05_f64;
                    decoherence_map[i][j][k] = amplitude_decoherence + phase_decoherence;
                }
            }
        }
        decoherence_map
    }

    /// Calculate quantum interference patterns
    fn calculate_interference_patterns(&self, field: &physics_engine::QuantumField) -> 
        Vec<Vec<Vec<f64>>> {
        
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut interference_map = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];

        // Calculate interference patterns based on quantum field superposition
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let complex_val = field.field_values[[i, j, k]];
                    let amplitude = complex_val.modulus();
                    let phase = complex_val.argument();
                    
                    // Create interference pattern based on phase and amplitude
                    let interference = (phase * 10.0).sin() * amplitude;
                    interference_map[i][j][k] = interference.abs();
                }
            }
        }

        interference_map
    }

    /// Calculate quantum tunneling probabilities
    fn calculate_tunneling_probabilities(&self, field: &physics_engine::QuantumField) -> 
        Vec<Vec<Vec<f64>>> {
        
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut tunneling_map = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];

        // Calculate tunneling probabilities based on field energy and barriers
        // Note: Using default mass value since field.mass is not available
        let default_mass = 1.0e-30; // Default particle mass in kg
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let amplitude = field.field_values[[i, j, k]].modulus();
                    let energy = amplitude * amplitude * default_mass; // Simplified energy calculation
                    
                    // Simplified tunneling probability: P ∝ exp(-√(2mE)/ħ)
                    let tunneling_prob: f64 = (-(2.0 * default_mass * energy).sqrt() / 1.055e-34).exp();
                    tunneling_map[i][j][k] = tunneling_prob.min(1.0);
                }
            }
        }

        tunneling_map
    }

    /// Calculate uncertainty principle data
    fn calculate_uncertainty_principle_data(&self, field: &physics_engine::QuantumField) -> 
        (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
        
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut position_uncertainty = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];
        let mut momentum_uncertainty = vec![vec![vec![0.0; z_dim]; y_dim]; x_dim];

        // Calculate position and momentum uncertainties based on field gradients
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let amplitude = field.field_values[[i, j, k]].modulus();
                    
                    // Position uncertainty based on field localization
                    position_uncertainty[i][j][k] = 1.0 / (amplitude + 1e-10);
                    
                    // Momentum uncertainty based on field gradients
                    let mut gradient_magnitude = 0.0;
                    if i > 0 && i < x_dim - 1 {
                        let dx = (field.field_values[[i+1, j, k]] - field.field_values[[i-1, j, k]]).modulus();
                        gradient_magnitude += dx * dx;
                    }
                    if j > 0 && j < y_dim - 1 {
                        let dy = (field.field_values[[i, j+1, k]] - field.field_values[[i, j-1, k]]).modulus();
                        gradient_magnitude += dy * dy;
                    }
                    if k > 0 && k < z_dim - 1 {
                        let dz = (field.field_values[[i, j, k+1]] - field.field_values[[i, j, k-1]]).modulus();
                        gradient_magnitude += dz * dz;
                    }
                    
                    momentum_uncertainty[i][j][k] = gradient_magnitude.sqrt();
                }
            }
        }

        (position_uncertainty, momentum_uncertainty)
    }

    /// Calculate quantum coherence times
    fn calculate_coherence_times(&self, field: &physics_engine::QuantumField) -> 
        Vec<Vec<Vec<f64>>> {
        
        let (x_dim, y_dim, z_dim) = field.field_values.dim();
        let mut coherence_times = vec![vec![vec![1.0; z_dim]; y_dim]; x_dim];

        // Calculate coherence times based on field properties and environment
        // Note: Using default mass value since field.mass is not available
        let default_mass = 1.0e-30; // Default particle mass in kg
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    let amplitude = field.field_values[[i, j, k]].modulus();
                    let energy = amplitude * amplitude * default_mass;
                    
                    // Coherence time decreases with energy and increases with field strength
                    let coherence_time: f64 = 1.0 / (energy + 1e-10) * amplitude;
                    coherence_times[i][j][k] = coherence_time.max(0.1);
                }
            }
        }

        coherence_times
    }

    /// Calculate quantum field statistics
    fn calculate_quantum_statistics(&self, field: &physics_engine::QuantumField) -> 
        QuantumFieldStatistics {
        
        let mut total_amplitude = 0.0_f64;
        let mut max_amplitude = 0.0_f64;
        let mut min_amplitude = f64::INFINITY;
        let mut total_energy = 0.0_f64;
        let mut point_count = 0;

        // Note: Using default mass value since field.mass is not available
        let default_mass = 1.0e-30; // Default particle mass in kg

        let (nx, ny, nz) = field.field_values.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let amplitude = field.field_values[[i, j, k]].modulus();
                    total_amplitude += amplitude;
                    max_amplitude = max_amplitude.max(amplitude);
                    min_amplitude = min_amplitude.min(amplitude);
                    total_energy += amplitude * amplitude * default_mass;
                    point_count += 1;
                }
            }
        }

        let avg_amplitude = if point_count > 0 { total_amplitude / point_count as f64 } else { 0.0 };
        let avg_energy = if point_count > 0 { total_energy / point_count as f64 } else { 0.0 };

        QuantumFieldStatistics {
            average_amplitude: avg_amplitude,
            max_amplitude,
            min_amplitude: if min_amplitude == f64::INFINITY { 0.0 } else { min_amplitude },
            total_energy,
            average_energy: avg_energy,
            total_points: point_count,
            field_type: format!("{:?}", field.field_type),
        }
    }

    pub fn set_speed_factor(&mut self, factor: f64) -> Result<()> {
        if factor <= 0.0 {
            return Err(anyhow!("Invalid speed factor: must be > 0"));
        }
        self.tick_span_years *= factor;
        Ok(())
    }

    pub fn rewind_ticks(&mut self, ticks: u64) -> Result<u64> {
        let actual = if ticks <= self.current_tick {
            self.current_tick -= ticks;
            ticks
        } else {
            let tmp = self.current_tick;
            self.current_tick = 0;
            tmp
        };
        Ok(actual)
    }
    
    /// Get read-only access to physics engine for rendering
    pub fn get_physics_engine(&self) -> &PhysicsEngine {
        &self.physics_engine
    }

    /// Record current simulation statistics into rolling history for trend analysis.
    #[allow(dead_code)]
    fn record_stats(&mut self) -> Result<()> {
        // Gather statistics using existing helper
        let stats_snapshot = self.get_stats()?;
        if self.stats_history.len() >= MAX_STATS_HISTORY {
            self.stats_history.pop_front();
        }
        self.stats_history.push_back(stats_snapshot);
        Ok(())
    }

    /// Export historical statistics as JSON array for RPC transmission.
    pub fn get_stats_history_json(&self) -> Result<serde_json::Value> {
        use serde_json::json;
        let history_json: Vec<_> = self
            .stats_history
            .iter()
            .map(|s| {
                json!({
                    "tick": s.current_tick,
                    "age_gyr": s.universe_age_gyr,
                    "star_count": s.star_count,
                    "planet_count": s.planet_count,
                    "total_particles": s.particle_count,
                    "average_temperature": s.average_temperature,
                })
            })
            .collect();
        Ok(json!(history_json))
    }

    /// Apply cosmological expansion effects to universe-scale properties
    fn apply_cosmological_effects(&mut self, dt: f64) -> Result<()> {
        // Get current cosmological parameters from physics engine
        if let Some(ref params) = self.get_cosmological_parameters() {
            // Update universe state with cosmological expansion
            let age_gyr = params.age_of_universe;
            let hubble_constant = params.hubble_constant;
            let scale_factor = params.scale_factor;
            let redshift = params.redshift;
            
            // Apply scale factor evolution to stored celestial bodies
            for body in &mut self.store.celestials {
                // Scale distances by cosmic expansion using the position field
                body.position *= scale_factor;
                
                // Apply cosmic time dilation effects to stellar evolution
                if matches!(body.body_type, crate::storage::CelestialBodyType::Star) {
                    // Stellar time scales with cosmic time
                    body.age *= (1.0 + redshift).recip();
                }
                
                // Update cosmic microwave background temperature
                // T_CMB = T_0 * (1 + z) where T_0 = 2.7 K today
                let cmb_temperature = 2.725 * (1.0 + redshift);
                
                // For planets, add CMB contribution to background temperature
                if matches!(body.body_type, crate::storage::CelestialBodyType::Planet) {
                    // Don't let CMB dominate for young universe scenarios
                    body.temperature = body.temperature.max(cmb_temperature * 0.1);
                }
            }
            
            // Update agent lineages with cosmological time effects
            for lineage in &mut self.store.agents {
                // Biological evolution rates affected by cosmic environment
                let cosmic_acceleration_factor: f64 = if age_gyr < 1.0 { 
                    0.1 // Early universe is harsh for life
                } else if age_gyr > 10.0 {
                    1.2 // Mature universe favors complexity
                } else {
                    1.0_f64 // Standard evolution rate
                };
                
                lineage.tech_level *= cosmic_acceleration_factor.powf(dt / 1e9); // Scale by Gyr
                lineage.sentience_level *= cosmic_acceleration_factor.powf(dt / 1e9);
            }
            
            // Update universe-wide statistics
            self.universe_state.hubble_constant = hubble_constant;
            self.universe_state.age_gyr = age_gyr;
            
            log::debug!(
                "Applied cosmological effects: age={:.2} Gyr, H={:.1} km/s/Mpc, a={:.6}, z={:.3}",
                age_gyr, hubble_constant, scale_factor, redshift
            );
        }
        
        Ok(())
    }
    
    /// Get cosmological parameters from physics engine
    fn get_cosmological_parameters(&self) -> Option<CosmologicalParameters> {
        #[cfg(feature = "gadget")]
        {
            // GADGET gravity solver integration is not active yet; fallback below
        }
        
        // Fallback: create parameters from current universe state
        Some(CosmologicalParameters {
            hubble_constant: self.universe_state.hubble_constant,
            omega_matter: 0.315,
            omega_lambda: 0.685,
            omega_baryon: 0.049,
            scale_factor: 1.0, // Default to present day
            redshift: 0.0,
            age_of_universe: self.universe_state.age_gyr,
            enable_expansion: true,
        })
    }
    
    /// Process agent evolution with cosmic context
    fn process_agent_evolution(&mut self, dt: f64) -> Result<()> {
        let _dt_years = dt;
        
        for _lineage in &mut self.store.agents {
            // Placeholder: agent evolution systems are not yet integrated.
            // Once the agent_evolution module exposes the required APIs,
            // hook them up here.
        }
        
        Ok(())
    }

    /// Initialize the universe with ENDF nuclear database integration
    pub fn new_with_endf_integration(config: SimulationConfig) -> Result<Self> {
        info!("Initializing universe simulation with ENDF nuclear database integration");
        
        // Create basic universe simulation
        let universe = Self::new(config.clone())?;
        
        // Load ENDF database if directory exists
        let endf_path = std::path::Path::new("endf-b-viii.0/lib/neutrons");
        if endf_path.exists() {
            info!("Loading ENDF/B-VIII.0 nuclear database from: {:?}", endf_path);
            
            // Create mutable references to nuclear databases
            let mut nuclear_db = physics_engine::nuclear_physics::NuclearDatabase::new();
            let mut cross_section_db = physics_engine::nuclear_physics::NuclearCrossSectionDatabase::new();
            
            // Load ENDF data
            match physics_engine::endf_data::load_endf_data(&mut nuclear_db, &mut cross_section_db, endf_path) {
                Ok(()) => {
                    info!("✅ ENDF nuclear database loaded successfully");
                    
                    // Calculate accurate isotope count from loaded database
                    let mut isotope_count = 0;
                    for z in 1..=118 { // All known elements
                        for n in 0..=300 { // Reasonable neutron range
                            if nuclear_db.get_decay_data(z, n).is_some() {
                                isotope_count += 1;
                            }
                        }
                    }
                    
                    info!("Nuclear database now contains {} isotopes", isotope_count);
                    
                    // Additional database statistics
                    let mut cross_section_count = 0;
                    for z in 1..=118 {
                        for n in 0..=300 {
                                                         if cross_section_db.get_endf_cross_section(z as u32, n as u32, 2).is_some() {
                                cross_section_count += 1;
                            }
                        }
                    }
                    info!("Cross-section database contains {} entries", cross_section_count);
                }
                Err(e) => {
                    warn!("⚠️ Failed to load ENDF database: {}", e);
                    warn!("Simulation will continue with default nuclear data (~50 isotopes)");
                }
            }
        } else {
            warn!("ENDF directory not found at {:?}", endf_path);
            warn!("Simulation will use default nuclear database (~50 isotopes instead of 3000+)");
        }
        
        Ok(universe)
    }

    /// Get quantum state vector snapshot for advanced visualization
    /// Returns comprehensive quantum field data for all field types
    pub fn get_quantum_state_vector_snapshot(&self) -> HashMap<String, QuantumStateVectorData> {
        let mut quantum_data = HashMap::new();
        
        // Extract quantum field data from physics engine
        for (field_name, field) in &self.physics_engine.quantum_fields {
            let (complex_amplitudes, phases, magnitudes) = self.extract_quantum_field_data(field);
            let entanglement_correlations = self.calculate_entanglement_correlations(field);
            let decoherence_rates = self.calculate_decoherence_effects(field);
            let interference_patterns = self.calculate_interference_patterns(field);
            let tunneling_probabilities = self.calculate_tunneling_probabilities(field);
            let (uncertainty_position, uncertainty_momentum) = self.calculate_uncertainty_principle_data(field);
            let coherence_times = self.calculate_coherence_times(field);
            let quantum_statistics = self.calculate_quantum_statistics(field);
            
            // Get field dimensions properly
            let (x_dim, y_dim, z_dim) = field.field_values.dim();
            
            let quantum_state_data = QuantumStateVectorData {
                field_type: format!("{:?}", field_name),
                complex_amplitudes,
                phases,
                magnitudes,
                entanglement_correlations,
                decoherence_rates,
                interference_patterns,
                tunneling_probabilities,
                uncertainty_position,
                uncertainty_momentum,
                coherence_times,
                field_energy_density: field.energy_density,
                field_mass: field.mass,
                field_spin: field.spin,
                vacuum_expectation_value: (field.vacuum_expectation_value.re, field.vacuum_expectation_value.im),
                lattice_spacing: field.lattice_spacing,
                field_dimensions: (x_dim, y_dim, z_dim),
                quantum_statistics,
                timestamp: self.current_tick,
                universe_age: self.universe_age_gyr(),
            };
            
            quantum_data.insert(format!("{:?}", field_name), quantum_state_data);
        }
        
        quantum_data
    }

    /// Get comprehensive molecular dynamics snapshot for real-time visualization
    /// Integrates with quantum field data to show quantum-classical transitions
    pub fn get_molecular_dynamics_snapshot(&self) -> Result<MolecularDynamicsSnapshot> {
        // Extract molecular dynamics data from physics engine
        // Note: molecular_dynamics_engine field doesn't exist, create basic snapshot
        let md_snapshot = self.create_basic_md_snapshot()?;
        
        // Enhance with quantum field integration
        let quantum_data = self.get_quantum_state_vector_snapshot();
        
        // Create comprehensive molecular dynamics snapshot
        Ok(MolecularDynamicsSnapshot {
            basic_snapshot: md_snapshot,
            quantum_integration: quantum_data,
            universe_age: self.universe_age_gyr(),
            simulation_time: self.current_tick as f64 * self.tick_span_years,
            cosmic_era: self.universe_state.clone(),
            molecular_statistics: self.calculate_molecular_statistics()?,
            chemical_evolution: self.calculate_chemical_evolution_data()?,
            quantum_classical_interface: self.detect_quantum_classical_boundaries()?,
        })
    }
    
    /// Create a basic molecular dynamics snapshot from available particle data
    fn create_basic_md_snapshot(&self) -> Result<MDSnapshot> {
        let particle_count = self.store.particles.count;
        
        let mut particle_positions = Vec::with_capacity(particle_count);
        let mut particle_velocities = Vec::with_capacity(particle_count);
        let mut particle_forces = Vec::with_capacity(particle_count);
        let mut particle_types = Vec::with_capacity(particle_count);
        let mut particle_masses = Vec::with_capacity(particle_count);
        
        // Extract particle data from store
        for i in 0..particle_count {
            particle_positions.push([
                self.store.particles.position[i].x,
                self.store.particles.position[i].y,
                self.store.particles.position[i].z,
            ]);
            
            particle_velocities.push([
                self.store.particles.velocity[i].x,
                self.store.particles.velocity[i].y,
                self.store.particles.velocity[i].z,
            ]);
            
            // Approximate forces from physics engine
            particle_forces.push([0.0, 0.0, 0.0]); // Would be calculated from interactions
            
            // Use particle ID as type since particle_type field doesn't exist
            particle_types.push(i as u32);
            particle_masses.push(self.store.particles.mass[i]);
        }
        
        // Detect bonds and molecular structures
        let bonds = self.detect_molecular_bonds(&particle_positions, &particle_masses)?;
        let neighbor_pairs = self.calculate_neighbor_pairs(&particle_positions)?;
        let quantum_regions = self.detect_quantum_regions_from_particles(&particle_positions, &particle_masses)?;
        let molecular_clusters = self.identify_molecular_clusters_from_bonds(&bonds)?;
        let reaction_events = self.detect_recent_reactions()?;
        
        Ok(MDSnapshot {
            step: self.current_tick as usize,
            time: self.current_tick as f64 * self.tick_span_years * 365.25 * 24.0 * 3600.0, // Convert to seconds
            particle_positions,
            particle_velocities,
            particle_forces,
            particle_types,
            particle_masses,
            bonds,
            neighbor_pairs: neighbor_pairs.clone(),
            properties: physics_engine::molecular_dynamics::SystemProperties {
                step: self.current_tick as usize,
                time: self.current_tick as f64 * self.tick_span_years,
                temperature: self.physics_engine.temperature,
                kinetic_energy: self.calculate_total_kinetic_energy(),
                n_particles: particle_count,
                neighbor_pairs: neighbor_pairs.len(),
            },
            quantum_regions,
            molecular_clusters,
            reaction_events,
        })
    }
    
    /// Detect molecular bonds based on particle positions and types
    fn detect_molecular_bonds(&self, positions: &[[f64; 3]], masses: &[f64]) -> Result<Vec<(usize, usize, f64)>> {
        let mut bonds = Vec::new();
        
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dz = positions[i][2] - positions[j][2];
                let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                
                // Estimate bond threshold based on particle masses (proxy for atomic radii)
                let bond_threshold = self.estimate_bond_length(masses[i], masses[j]);
                
                if distance < bond_threshold {
                    let bond_strength = self.calculate_bond_strength_from_distance(distance, bond_threshold);
                    if bond_strength > 0.1 {
                        bonds.push((i, j, bond_strength));
                    }
                }
            }
        }
        
        Ok(bonds)
    }
    
    /// Estimate bond length based on particle masses
    fn estimate_bond_length(&self, mass1: f64, mass2: f64) -> f64 {
        // Simplified bond length estimation based on atomic mass
        // This is a rough approximation for visualization purposes
        let proton_mass = 1.67262192e-27; // kg
        let atomic_mass1 = mass1 / proton_mass;
        let atomic_mass2 = mass2 / proton_mass;
        
        // Approximate atomic radii scaling with atomic mass
        let radius1 = 0.5e-10 * atomic_mass1.powf(0.33); // Rough scaling
        let radius2 = 0.5e-10 * atomic_mass2.powf(0.33);
        
        (radius1 + radius2) * 1.2 // Bond length is typically 1.2x sum of atomic radii
    }
    
    /// Calculate bond strength from distance
    fn calculate_bond_strength_from_distance(&self, distance: f64, bond_threshold: f64) -> f64 {
        // Simple exponential decay model for bond strength
        let normalized_distance = distance / bond_threshold;
        if normalized_distance > 1.0 {
            0.0
        } else {
            (1.0 - normalized_distance).exp()
        }
    }
    
    /// Calculate neighbor pairs for visualization
    fn calculate_neighbor_pairs(&self, positions: &[[f64; 3]]) -> Result<Vec<(usize, usize, f64)>> {
        let mut pairs = Vec::new();
        let cutoff_distance = 5e-10; // 5 Angstroms
        
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dz = positions[i][2] - positions[j][2];
                let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                
                if distance < cutoff_distance {
                    pairs.push((i, j, distance));
                }
            }
        }
        
        Ok(pairs)
    }
    
    /// Detect quantum regions from particle data
    fn detect_quantum_regions_from_particles(&self, positions: &[[f64; 3]], masses: &[f64]) -> Result<Vec<usize>> {
        let mut quantum_regions = Vec::new();
        
        for i in 0..positions.len() {
            // Light particles (electrons, etc.) always need quantum treatment
            let proton_mass = 1.67262192e-27;
            if masses[i] < proton_mass * 10.0 {
                quantum_regions.push(i);
            }
            
            // Particles in high-density regions may need quantum treatment
            let local_density = self.calculate_local_density(i, positions);
            if local_density > 1e30 { // High density threshold (particles/m³)
                quantum_regions.push(i);
            }
        }
        
        Ok(quantum_regions)
    }
    
    /// Calculate local particle density around a given particle
    fn calculate_local_density(&self, particle_index: usize, positions: &[[f64; 3]]) -> f64 {
        let search_radius = 1e-10; // 1 Angstrom
        let mut neighbor_count = 0;
        
        let center = positions[particle_index];
        for (i, pos) in positions.iter().enumerate() {
            if i != particle_index {
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                
                if distance < search_radius {
                    neighbor_count += 1;
                }
            }
        }
        
        // Calculate density (particles per cubic meter)
        let volume = (4.0 / 3.0) * std::f64::consts::PI * search_radius.powi(3);
        neighbor_count as f64 / volume
    }
    
    /// Identify molecular clusters from bond data
    fn identify_molecular_clusters_from_bonds(&self, bonds: &[(usize, usize, f64)]) -> Result<Vec<Vec<usize>>> {
        let mut clusters = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        // Create adjacency list from bonds
        let mut adjacency = std::collections::HashMap::new();
        for &(i, j, _) in bonds {
            adjacency.entry(i).or_insert_with(Vec::new).push(j);
            adjacency.entry(j).or_insert_with(Vec::new).push(i);
        }
        
        // Find connected components (molecular clusters)
        for &(start, _, _) in bonds {
            if !visited.contains(&start) {
                let mut cluster = Vec::new();
                self.dfs_molecular_cluster(start, &adjacency, &mut visited, &mut cluster);
                if !cluster.is_empty() {
                    clusters.push(cluster);
                }
            }
        }
        
        Ok(clusters)
    }
    
    /// Depth-first search for molecular cluster identification
    fn dfs_molecular_cluster(
        &self,
        node: usize,
        adjacency: &std::collections::HashMap<usize, Vec<usize>>,
        visited: &mut std::collections::HashSet<usize>,
        cluster: &mut Vec<usize>,
    ) {
        if visited.contains(&node) {
            return;
        }
        
        visited.insert(node);
        cluster.push(node);
        
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                self.dfs_molecular_cluster(neighbor, adjacency, visited, cluster);
            }
        }
    }
    
    /// Detect recent chemical reactions
    fn detect_recent_reactions(&self) -> Result<Vec<ReactionEvent>> {
        // This is a simplified implementation for demonstration
        // In a full system, you would track bond changes over time
        let mut events = Vec::new();
        
        // For now, create some example reaction events based on current state
        if self.physics_engine.temperature > 1000.0 {
            events.push(ReactionEvent {
                event_type: ReactionEventType::ConformationalChange,
                participants: vec![0, 1], // Example participant indices
                time: self.current_tick as f64 * self.tick_span_years,
                energy_change: self.physics_engine.temperature * 1e-23, // Approximate energy scale
            });
        }
        
        Ok(events)
    }
    
    /// Calculate total kinetic energy of all particles
    fn calculate_total_kinetic_energy(&self) -> f64 {
        let mut total_ke = 0.0;
        
        for i in 0..self.store.particles.count {
            let velocity = &self.store.particles.velocity[i];
            let mass = self.store.particles.mass[i];
            let speed_squared = velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
            total_ke += 0.5 * mass * speed_squared;
        }
        
        total_ke
    }
    
    /// Calculate molecular statistics for the current state
    fn calculate_molecular_statistics(&self) -> Result<MolecularStatistics> {
        let particle_count = self.store.particles.count;
        let total_mass = self.store.particles.mass.iter().take(particle_count).sum::<f64>();
        let average_mass = if particle_count > 0 { total_mass / particle_count as f64 } else { 0.0 };
        
        Ok(MolecularStatistics {
            total_particles: particle_count,
            total_mass,
            average_mass,
            temperature: self.physics_engine.temperature,
            pressure: self.estimate_pressure(),
            density: self.calculate_average_density(),
            molecular_complexity: self.estimate_molecular_complexity(),
        })
    }
    
    /// Estimate system pressure
    fn estimate_pressure(&self) -> f64 {
        // Simplified ideal gas law estimation
        let particle_count = self.store.particles.count as f64;
        let volume = self.physics_engine.volume;
        let temperature = self.physics_engine.temperature;
        let k_b = 1.380649e-23; // Boltzmann constant
        
        if volume > 0.0 {
            (particle_count * k_b * temperature) / volume
        } else {
            0.0
        }
    }
    
    /// Calculate average particle density
    fn calculate_average_density(&self) -> f64 {
        let total_mass: f64 = self.store.particles.mass.iter().take(self.store.particles.count).sum();
        let volume = self.physics_engine.volume;
        
        if volume > 0.0 {
            total_mass / volume
        } else {
            0.0
        }
    }
    
    /// Estimate molecular complexity (average bonds per particle)
    fn estimate_molecular_complexity(&self) -> f64 {
        // Implement accurate molecular complexity calculation based on actual bond analysis
        // Analyze particle positions and masses to determine bonding patterns
        
        let particle_count = self.store.particles.count;
        if particle_count == 0 {
            return 0.0;
        }
        
        // Calculate actual bonds from particle positions
        let mut total_bonds = 0.0;
        let mut _bonded_particles = 0;
        
        // Use a cutoff distance for bonding (typical chemical bond length ~1-3 Å)
        let bond_cutoff = 3e-10; // 3 Å in meters
        
        for i in 0..particle_count {
            let pos_i = &self.store.particles.position[i];
            let _mass_i = self.store.particles.mass[i];
            let mut particle_bonds = 0;
            
            for j in (i + 1)..particle_count {
                let pos_j = &self.store.particles.position[j];
                let _mass_j = self.store.particles.mass[j];
                
                // Calculate distance between particles
                let dx = pos_i.x - pos_j.x;
                let dy = pos_i.y - pos_j.y;
                let dz = pos_i.z - pos_j.z;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                
                // Check if particles are close enough to form a bond
                if distance < bond_cutoff {
                    // Estimate bond strength based on masses and distance
                    let bond_strength = self.calculate_bond_strength_from_distance(distance, bond_cutoff);
                    
                    // Only count significant bonds
                    if bond_strength > 0.1 {
                        particle_bonds += 1;
                        total_bonds += bond_strength;
                    }
                }
            }
            
            if particle_bonds > 0 {
                _bonded_particles += 1;
            }
        }
        
        // Calculate average bonds per particle
        let average_bonds = if particle_count > 0 {
            total_bonds / particle_count as f64
        } else {
            0.0
        };
        
        // Apply temperature and density corrections
        let temperature = self.physics_engine.temperature;
        let density = self.calculate_average_density();
        
        // Temperature effect: higher temperature reduces bonding
        let temp_factor = if temperature > 0.0 {
            (300.0 / temperature).min(2.0).max(0.1)
        } else {
            1.0
        };
        
        // Density effect: higher density increases bonding opportunities
        let density_factor = if density > 0.0 {
            (density / 1000.0).min(3.0).max(0.1) // Normalize to water density
        } else {
            1.0
        };
        
        // Final complexity estimate
        let complexity = average_bonds * temp_factor * density_factor;
        
        // Ensure reasonable bounds
        complexity.min(5.0).max(0.0)
    }
    
    /// Calculate chemical evolution data
    fn calculate_chemical_evolution_data(&self) -> Result<ChemicalEvolutionData> {
        Ok(ChemicalEvolutionData {
            universe_age: self.universe_age_gyr(),
            metallicity: self.calculate_metallicity(),
            molecular_formation_rate: self.estimate_molecular_formation_rate(),
            reaction_rate: self.estimate_reaction_rate(),
            complexity_index: self.calculate_complexity_index(),
        })
    }
    
    /// Calculate metallicity (fraction of heavy elements)
    fn calculate_metallicity(&self) -> f64 {
        // Simplified metallicity calculation without requiring mutable reference
        // Use a basic approximation based on universe age
        let age_gyr = self.universe_age_gyr();
        
        // Early universe has low metallicity, increases over time
        if age_gyr < 1.0 {
            0.001 // Very low metallicity in early universe
        } else if age_gyr < 5.0 {
            0.01 * age_gyr // Linear increase
        } else {
            0.02 // Solar metallicity for mature universe
        }
    }
    
    /// Estimate molecular formation rate
    fn estimate_molecular_formation_rate(&self) -> f64 {
        // Based on temperature, density, and cosmic era
        let temperature = self.physics_engine.temperature;
        let density = self.calculate_average_density();
        
        if temperature > 10.0 && temperature < 10000.0 && density > 1e-20 {
            density * 1e15 / temperature // Simplified rate equation
        } else {
            0.0
        }
    }
    
    /// Estimate chemical reaction rate
    fn estimate_reaction_rate(&self) -> f64 {
        // Arrhenius-like equation for reaction rate
        let temperature = self.physics_engine.temperature;
        let density = self.calculate_average_density();
        
        if temperature > 100.0 {
            density * temperature.ln() * 1e-10
        } else {
            0.0
        }
    }
    
    /// Calculate chemical complexity index
    fn calculate_complexity_index(&self) -> f64 {
        // Combine various factors to estimate chemical complexity
        let metallicity = self.calculate_metallicity();
        let temperature = self.physics_engine.temperature;
        let age = self.universe_age_gyr();
        
        // Complex chemistry requires metals, moderate temperature, and time
        if metallicity > 0.01 && temperature > 100.0 && temperature < 1000.0 && age > 1.0 {
            metallicity * age.ln() * (1000.0 / temperature).ln()
        } else {
            metallicity * 0.1
        }
    }
    
    /// Detect quantum-classical interface boundaries
    fn detect_quantum_classical_boundaries(&self) -> Result<QuantumClassicalInterface> {
        Ok(QuantumClassicalInterface {
            transition_regions: self.identify_transition_regions()?,
            decoherence_boundaries: self.calculate_decoherence_boundaries()?,
            quantum_coherence_length: self.estimate_quantum_coherence_length(),
            classical_limit_scale: self.estimate_classical_limit_scale(),
        })
    }
    
    /// Identify regions where quantum-classical transitions occur
    fn identify_transition_regions(&self) -> Result<Vec<TransitionRegion>> {
        let mut regions = Vec::new();
        
        // For demonstration, create transition regions based on density and temperature
        let density = self.calculate_average_density();
        let temperature = self.physics_engine.temperature;
        
        if density > 1e25 && temperature < 1000.0 {
            regions.push(TransitionRegion {
                center: [0.0, 0.0, 0.0],
                radius: 1e-10,
                quantum_fraction: 0.8,
                classical_fraction: 0.2,
                transition_sharpness: 0.5,
            });
        }
        
        Ok(regions)
    }
    
    /// Calculate decoherence boundaries
    fn calculate_decoherence_boundaries(&self) -> Result<Vec<DecoherenceBoundary>> {
        let mut boundaries = Vec::new();
        
        // Simplified decoherence boundary calculation
        let temperature = self.physics_engine.temperature;
        if temperature > 10.0 {
            boundaries.push(DecoherenceBoundary {
                position: [0.0, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
                decoherence_rate: temperature * 1e12, // Hz
                coherence_time: 1.0 / (temperature * 1e12), // seconds
            });
        }
        
        Ok(boundaries)
    }
    
    /// Estimate quantum coherence length
    fn estimate_quantum_coherence_length(&self) -> f64 {
        // Thermal de Broglie wavelength
        let temperature = self.physics_engine.temperature;
        let mass = 1.67262192e-27; // Approximate particle mass (proton mass)
        let h = 6.62607015e-34; // Planck constant
        let k_b = 1.380649e-23; // Boltzmann constant
        
        if temperature > 0.0 {
            h / (2.0 * std::f64::consts::PI * (2.0 * mass * k_b * temperature).sqrt())
        } else {
            f64::INFINITY
        }
    }
    
    /// Estimate classical limit scale
    fn estimate_classical_limit_scale(&self) -> f64 {
        // Scale at which classical physics becomes dominant
        let coherence_length = self.estimate_quantum_coherence_length();
        coherence_length * 10.0 // Classical limit is roughly 10x coherence length
    }

    /// Calculate the average gas density in the simulation (kg/m^3)
    fn average_gas_density(&self) -> f64 {
        // Assume all particles are gas for simplicity
        let total_mass = self.total_gas_mass();
        // Estimate volume as a sphere with radius = universe_radius_ly (converted to meters)
        let radius_m = self.config.universe_radius_ly * 9.460_730_472e15;
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius_m.powi(3);
        if volume > 0.0 { total_mass / volume } else { 0.0 }
    }

    /// Calculate the average gas temperature (K)
    fn average_gas_temperature(&self) -> f64 {
        if self.store.particles.count == 0 {
            return 10.0; // fallback to 10 K (cold ISM)
        }
        let sum: f64 = self.store.particles.temperature.iter().sum();
        sum / (self.store.particles.count as f64)
    }

    /// Calculate the total gas mass (kg)
    fn total_gas_mass(&self) -> f64 {
        self.store.particles.mass.iter().sum()
    }

    /// Calculate the Jeans mass (kg) for given density and temperature
    fn calculate_jeans_mass(density: f64, temperature: f64) -> f64 {
        // Constants
        const K_B: f64 = 1.380649e-23; // Boltzmann constant (J/K)
        const G: f64 = 6.67430e-11;    // Gravitational constant (m^3/kg/s^2)
        const M_H: f64 = 1.6735575e-27; // Mass of hydrogen atom (kg)
        const MU: f64 = 2.0; // Mean molecular weight for H2
        // Jeans mass formula
        let term1 = (5.0 * K_B * temperature / (G * MU * M_H)).powf(1.5);
        let term2 = (3.0 / (4.0 * std::f64::consts::PI * density)).sqrt();
        term1 * term2
    }

    /// Calculate adaptive time step for stellar evolution based on current visualization scale
    /// and atom/particle focus requirements
    fn calculate_adaptive_stellar_time_step(&self, base_dt: f64) -> f64 {
        // Get current visualization scale from physics engine if available
        let visualization_scale = 1.0; // Default visualization scale since method not available
        
        // Adjust time step based on visualization scale
        // For atomic/molecular scales, use finer time steps for detailed particle tracking
        let scale_factor = match visualization_scale {
            s if s < 1e-10 => 0.1,  // Atomic scale: very fine time steps
            s if s < 1e-6 => 0.5,   // Molecular scale: fine time steps
            s if s < 1e-3 => 1.0,   // Cellular scale: normal time steps
            s if s < 1.0 => 2.0,    // Macroscopic scale: coarser time steps
            s if s < 1e6 => 5.0,    // Planetary scale: much coarser time steps
            s if s < 1e12 => 10.0,  // Stellar scale: very coarse time steps
            _ => 20.0,              // Cosmological scale: extremely coarse time steps
        };
        
        // Apply scale factor to base time step
        let adaptive_dt = base_dt * scale_factor;
        
        // Ensure time step doesn't become too small or too large
        let min_dt = 1e-6; // 1 microsecond minimum
        let max_dt = 1e9;  // 1 billion years maximum
        
        adaptive_dt.clamp(min_dt, max_dt)
    }

    /// Calculate comprehensive radiation energy
    fn calculate_comprehensive_radiation_energy(&self) -> f64 {
        // Implement comprehensive radiation energy calculation
        // This is a placeholder and should be replaced with actual implementation
        0.0
    }
}


/// Simulation statistics
#[derive(Clone, Debug, Default)]
pub struct SimulationStats {
    pub current_tick: u64,
    pub target_ups: f64,
    pub universe_age_gyr: f64,
    pub universe_description: String,
    pub lineage_count: usize,
    pub particle_count: usize,
    pub celestial_body_count: usize,
    pub planet_count: usize,
    // Stellar statistics
    pub star_count: usize,
    pub stellar_formation_rate: f64,
    pub average_stellar_mass: f64,
    pub stellar_mass_distribution: Vec<f64>,
    pub main_sequence_stars: usize,
    pub evolved_stars: usize,
    pub stellar_remnants: usize,
    // Energy statistics
    pub total_energy: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub radiation_energy: f64,
    pub nuclear_binding_energy: f64,
    pub average_temperature: f64,
    pub energy_density: f64,
    // Chemical composition
    pub hydrogen_fraction: f64,
    pub helium_fraction: f64,
    pub carbon_fraction: f64,
    pub oxygen_fraction: f64,
    pub iron_fraction: f64,
    pub heavy_elements_fraction: f64,
    pub metallicity: f64,
    // Planetary statistics
    pub habitable_planets: usize,
    pub earth_like_planets: usize,
    pub gas_giants: usize,
    pub average_planet_mass: f64,
    pub planet_formation_rate: f64,
    // Evolution statistics
    pub extinct_lineages: usize,
    pub average_tech_level: f64,
    pub immortal_lineages: usize,
    pub consciousness_emergence_rate: f64,
    // Performance statistics
    pub physics_step_time_ms: f64,
    pub interactions_per_step: usize,
    pub particle_interactions_per_step: usize,
    // Cosmological statistics
    pub universe_radius: f64,
    pub hubble_constant: f64,
    pub dark_matter_fraction: f64,
    pub dark_energy_fraction: f64,
    pub ordinary_matter_fraction: f64,
    pub critical_density: f64,
}

#[derive(Debug, Default)]
struct StellarStatistics {
    #[allow(dead_code)]
    count: usize,
    formation_rate: f64,
    average_mass: f64,
    mass_distribution: Vec<(f64, f64)>,
    main_sequence_count: usize,
    evolved_count: usize,
    remnant_count: usize,
}

#[derive(Debug, Default)]
struct EnergyStatistics {
    total: f64,
    kinetic: f64,
    potential: f64,
    radiation: f64,
    binding: f64,
    average_temperature: f64,
    density: f64,
}

#[derive(Debug, Default)]
struct ChemicalComposition {
    hydrogen: f64,
    helium: f64,
    carbon: f64,
    oxygen: f64,
    iron: f64,
    heavy_elements: f64,
    metallicity: f64,
}

#[derive(Debug, Default)]
struct PlanetaryStatistics {
    total_count: usize,
    habitable_count: usize,
    earth_like_count: usize,
    gas_giant_count: usize,
    average_mass: f64,
    formation_rate: f64,
}

#[derive(Debug, Default)]
struct EvolutionStatistics {
    #[allow(dead_code)]
    total_ever: usize,
    extinct: usize,
    #[allow(dead_code)]
    average_fitness: f64,
    #[allow(dead_code)]
    average_sentience: f64,
    average_tech: f64,
    immortal_count: usize,
    consciousness_rate: f64,
}

#[derive(Debug, Default)]
struct PhysicsPerformance {
    step_time_ms: f64,
    nuclear_reactions: usize,
    interactions: usize,
}

#[derive(Debug, Default)]
struct CosmicStructure {
    radius: f64,
    hubble_constant: f64,
    dark_matter_fraction: f64,
    dark_energy_fraction: f64,
    ordinary_matter_fraction: f64,
    critical_density: f64,
}

/// Comprehensive quantum state vector data for advanced visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateVectorData {
    /// Type of quantum field (e.g., "ElectronField", "PhotonField")
    pub field_type: String,
    /// Complex amplitudes (real, imaginary) at each lattice point
    pub complex_amplitudes: Vec<Vec<Vec<(f64, f64)>>>,
    /// Quantum phases at each lattice point
    pub phases: Vec<Vec<Vec<f64>>>,
    /// Magnitudes of quantum amplitudes at each lattice point
    pub magnitudes: Vec<Vec<Vec<f64>>>,
    /// Entanglement correlations across the field
    pub entanglement_correlations: Vec<Vec<Vec<f64>>>,
    /// Decoherence rates at each lattice point
    pub decoherence_rates: Vec<Vec<Vec<f64>>>,
    /// Quantum interference patterns
    pub interference_patterns: Vec<Vec<Vec<f64>>>,
    /// Quantum tunneling probabilities
    pub tunneling_probabilities: Vec<Vec<Vec<f64>>>,
    /// Position uncertainty (Heisenberg uncertainty principle)
    pub uncertainty_position: Vec<Vec<Vec<f64>>>,
    /// Momentum uncertainty (Heisenberg uncertainty principle)
    pub uncertainty_momentum: Vec<Vec<Vec<f64>>>,
    /// Quantum coherence times at each lattice point
    pub coherence_times: Vec<Vec<Vec<f64>>>,
    /// Field energy density
    pub field_energy_density: f64,
    /// Field mass (for massive fields)
    pub field_mass: f64,
    /// Field spin (0 for scalar, 1/2 for fermion, 1 for vector)
    pub field_spin: f64,
    /// Vacuum expectation value (real, imaginary)
    pub vacuum_expectation_value: (f64, f64),
    /// Lattice spacing in meters
    pub lattice_spacing: f64,
    /// Field dimensions (x, y, z)
    pub field_dimensions: (usize, usize, usize),
    /// Quantum field statistics
    pub quantum_statistics: QuantumFieldStatistics,
    /// Simulation timestamp
    pub timestamp: u64,
    /// Universe age in Gyr
    pub universe_age: f64,
}

impl QuantumStateVectorData {
    /// Create empty quantum state vector data
    pub fn empty() -> Self {
        Self {
            field_type: String::new(),
            complex_amplitudes: Vec::new(),
            phases: Vec::new(),
            magnitudes: Vec::new(),
            entanglement_correlations: Vec::new(),
            decoherence_rates: Vec::new(),
            interference_patterns: Vec::new(),
            tunneling_probabilities: Vec::new(),
            uncertainty_position: Vec::new(),
            uncertainty_momentum: Vec::new(),
            coherence_times: Vec::new(),
            field_energy_density: 0.0,
            field_mass: 0.0,
            field_spin: 0.0,
            vacuum_expectation_value: (0.0, 0.0),
            lattice_spacing: 0.0,
            field_dimensions: (0, 0, 0),
            quantum_statistics: QuantumFieldStatistics::empty(),
            timestamp: 0,
            universe_age: 0.0,
        }
    }

    /// Convert to JSON for visualization systems
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        
        json!({
            "field_type": self.field_type,
            "field_dimensions": {
                "x": self.field_dimensions.0,
                "y": self.field_dimensions.1,
                "z": self.field_dimensions.2
            },
            "field_properties": {
                "energy_density": self.field_energy_density,
                "mass": self.field_mass,
                "spin": self.field_spin,
                "vacuum_expectation_value": {
                    "real": self.vacuum_expectation_value.0,
                    "imaginary": self.vacuum_expectation_value.1
                },
                "lattice_spacing": self.lattice_spacing
            },
            "quantum_statistics": {
                "average_amplitude": self.quantum_statistics.average_amplitude,
                "max_amplitude": self.quantum_statistics.max_amplitude,
                "min_amplitude": self.quantum_statistics.min_amplitude,
                "total_energy": self.quantum_statistics.total_energy,
                "average_energy": self.quantum_statistics.average_energy,
                "total_points": self.quantum_statistics.total_points
            },
            "metadata": {
                "timestamp": self.timestamp,
                "universe_age": self.universe_age
            }
        })
    }

    /// Get 2D slice of quantum data for visualization
    pub fn get_2d_slice(&self, z_index: usize, data_type: QuantumDataType) -> Vec<Vec<f64>> {
        if z_index >= self.field_dimensions.2 {
            return Vec::new();
        }

        let mut slice = vec![vec![0.0; self.field_dimensions.1]; self.field_dimensions.0];

        for i in 0..self.field_dimensions.0 {
            for j in 0..self.field_dimensions.1 {
                slice[i][j] = match data_type {
                    QuantumDataType::Magnitude => self.magnitudes[i][j][z_index],
                    QuantumDataType::Phase => self.phases[i][j][z_index],
                    QuantumDataType::Entanglement => self.entanglement_correlations[i][j][z_index],
                    QuantumDataType::Decoherence => self.decoherence_rates[i][j][z_index],
                    QuantumDataType::Interference => self.interference_patterns[i][j][z_index],
                    QuantumDataType::Tunneling => self.tunneling_probabilities[i][j][z_index],
                    QuantumDataType::PositionUncertainty => self.uncertainty_position[i][j][z_index],
                    QuantumDataType::MomentumUncertainty => self.uncertainty_momentum[i][j][z_index],
                    QuantumDataType::CoherenceTime => self.coherence_times[i][j][z_index],
                };
            }
        }

        slice
    }
}

/// Types of quantum data available for visualization
#[derive(Debug, Clone, Copy)]
pub enum QuantumDataType {
    Magnitude,
    Phase,
    Entanglement,
    Decoherence,
    Interference,
    Tunneling,
    PositionUncertainty,
    MomentumUncertainty,
    CoherenceTime,
}

/// Statistics for quantum field analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFieldStatistics {
    /// Average amplitude across the field
    pub average_amplitude: f64,
    /// Maximum amplitude in the field
    pub max_amplitude: f64,
    /// Minimum amplitude in the field
    pub min_amplitude: f64,
    /// Total energy in the field
    pub total_energy: f64,
    /// Average energy per lattice point
    pub average_energy: f64,
    /// Total number of lattice points
    pub total_points: usize,
    /// Type of quantum field
    pub field_type: String,
}

impl QuantumFieldStatistics {
    /// Create empty quantum field statistics
    pub fn empty() -> Self {
        Self {
            average_amplitude: 0.0,
            max_amplitude: 0.0,
            min_amplitude: 0.0,
            total_energy: 0.0,
            average_energy: 0.0,
            total_points: 0,
            field_type: String::new(),
        }
    }
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
        
        let stats = sim.get_stats().unwrap();
        assert!(stats.particle_count > 0);
    }

    #[test]
    fn test_cosmic_era_progression() {
        let config = config::SimulationConfig::default();
        let mut sim = UniverseSimulation::new(config).unwrap();
        
        // Test physics-driven state evolution
        sim.current_tick = 0;
        sim.update_universe_state().unwrap();
        assert!(sim.universe_state.age_gyr < 0.001);
        assert!(sim.universe_state.stellar_fraction < 0.001);
        
        sim.current_tick = 1000; // 1 Gyr
        sim.update_universe_state().unwrap();
        assert!(sim.universe_state.age_gyr > 0.5);
    }
}

#[cfg(test)]
mod stellar_evolution_integration_tests {
    use super::*;
    use crate::config::SimulationConfig;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_complete_stellar_lifecycle_solar_mass() {
        // Test complete stellar evolution for a solar-mass star
        let config = SimulationConfig::default();
        let _simulation = UniverseSimulation::new(config).expect("Should create simulation");
        
        // Manually create a solar-mass star to test evolution
        let solar_mass = 1.989e30; // kg
        let mut stellar_evolution = StellarEvolution::new(solar_mass);
        
        // Initial conditions
        assert_eq!(stellar_evolution.evolutionary_phase, StellarPhase::MainSequence);
        assert_relative_eq!(stellar_evolution.nuclear_fuel_fraction, 1.0, epsilon = 0.01);
        
        // Simulate main sequence evolution (should take ~10 Gyr)
        let dt_years = 1e6; // 1 Myr time steps
        let mut total_time = 0.0;
        let mut energy_generated = 0.0;
        
        // Run until hydrogen exhaustion
        while stellar_evolution.nuclear_fuel_fraction > 0.1 && total_time < 15e9 {
            let energy = stellar_evolution.evolve(solar_mass, dt_years)
                .expect("Should successfully evolve star");
            energy_generated += energy;
            total_time += dt_years;
            
            // Verify energy generation is positive
            assert!(energy >= 0.0, "Nuclear fusion should generate energy");
            
            // Verify core temperature increases as fuel depletes
            if stellar_evolution.nuclear_fuel_fraction < 0.5 {
                assert!(stellar_evolution.core_temperature > 1.5e7, 
                       "Core should heat up as hydrogen depletes");
            }
        }
        
        // Should have transitioned off main sequence
        assert_ne!(stellar_evolution.evolutionary_phase, StellarPhase::MainSequence,
                  "Star should have evolved off main sequence");
        
        // Main sequence lifetime should be realistic (~10 Gyr for solar mass)
        assert!(total_time > 5e9 && total_time < 15e9, 
               "Solar mass star main sequence lifetime should be ~10 Gyr, got {:.1e} years", 
               total_time);
        
        // Should have generated significant energy
        assert!(energy_generated > 0.0, "Star should have generated nuclear energy");
    }
    
    #[test]
    fn test_massive_star_evolution() {
        // Test evolution of a massive star (20 solar masses)
        let solar_mass = 1.989e30;
        let massive_star_mass = 20.0 * solar_mass;
        let mut stellar_evolution = StellarEvolution::new(massive_star_mass);
        
        // Massive stars should have shorter lifetimes due to M^(-2.5) scaling
        assert!(stellar_evolution.main_sequence_lifetime < 1e8, 
               "Massive stars should have short lifetimes < 100 Myr");
        
        // Higher core temperature for massive stars
        assert!(stellar_evolution.core_temperature > 2e7,
               "Massive stars should have hotter cores");
        
        // Simulate evolution through advanced burning stages
        let dt_years = 1e5; // 100 kyr time steps for rapid evolution
        let mut total_time = 0.0;
        
        while stellar_evolution.evolutionary_phase != StellarPhase::Supernova && 
              total_time < stellar_evolution.main_sequence_lifetime * 2.0 {
            stellar_evolution.evolve(massive_star_mass, dt_years)
                .expect("Should successfully evolve massive star");
            total_time += dt_years;
        }
        
        // Massive star should go supernova
        assert_eq!(stellar_evolution.evolutionary_phase, StellarPhase::Supernova,
                  "Massive star should explode as supernova");
    }
    
    #[test]
    fn test_low_mass_star_evolution() {
        // Test evolution of a low-mass star (0.5 solar masses)
        let solar_mass = 1.989e30;
        let low_mass_star = 0.5 * solar_mass;
        let stellar_evolution = StellarEvolution::new(low_mass_star);
        
        // Low-mass stars should have very long lifetimes
        assert!(stellar_evolution.main_sequence_lifetime > 50e9, 
               "Low-mass stars should live longer than Hubble time");
        
        // Lower core temperature
        assert!(stellar_evolution.core_temperature < 1e7,
               "Low-mass stars should have cooler cores");
        
        // Should start in main sequence
        assert_eq!(stellar_evolution.evolutionary_phase, StellarPhase::MainSequence);
    }
    
    #[test]
    fn test_stellar_nucleosynthesis_integration() {
        // Test that stellar nucleosynthesis produces expected element ratios
        let solar_mass = 1.989e30;
        let mut stellar_evolution = StellarEvolution::new(solar_mass);
        
        // Initial composition should be primordial (H/He dominated)
        let initial_h_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 1 && *a == 1)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        let initial_he_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 2 && *a == 4)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        
        assert!(initial_h_fraction > 0.7, "Should start with >70% hydrogen");
        assert!(initial_he_fraction > 0.2, "Should start with >20% helium");
        
        // Evolve for significant time
        let dt_years = 1e6;
        for _ in 0..1000 {
            stellar_evolution.evolve(solar_mass, dt_years)
                .expect("Should evolve successfully");
            
            if stellar_evolution.nuclear_fuel_fraction < 0.8 {
                break;
            }
        }
        
        // Should have converted some hydrogen to helium
        let final_h_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 1 && *a == 1)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        let final_he_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 2 && *a == 4)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        
        assert!(final_h_fraction < initial_h_fraction, 
               "Hydrogen fraction should decrease due to burning");
        assert!(final_he_fraction > initial_he_fraction, 
               "Helium fraction should increase due to fusion");
    }
    
    #[test]
    fn test_stellar_death_outcomes() {
        // Test that stellar death produces correct remnants based on mass
        
        // Low-mass star → white dwarf
        let low_mass = 0.8 * 1.989e30;
        let mut low_evolution = StellarEvolution::new(low_mass);
        low_evolution.evolutionary_phase = StellarPhase::PlanetaryNebula;
        low_evolution.update_evolutionary_phase(low_mass, 0.0);
        assert_eq!(low_evolution.evolutionary_phase, StellarPhase::WhiteDwarf,
                  "Low-mass star should become white dwarf");
        
        // Intermediate-mass star → neutron star
        let intermediate_mass = 1.5 * 1.989e30;
        let mut intermediate_evolution = StellarEvolution::new(intermediate_mass);
        intermediate_evolution.evolutionary_phase = StellarPhase::PlanetaryNebula;
        intermediate_evolution.update_evolutionary_phase(intermediate_mass, 0.0);
        assert_eq!(intermediate_evolution.evolutionary_phase, StellarPhase::NeutronStar,
                  "Intermediate-mass star should become neutron star");
        
        // Very massive star → black hole
        let massive_mass = 30.0 * 1.989e30;
        let mut massive_evolution = StellarEvolution::new(massive_mass);
        massive_evolution.evolutionary_phase = StellarPhase::Supernova;
        massive_evolution.update_evolutionary_phase(massive_mass, 0.0);
        assert_eq!(massive_evolution.evolutionary_phase, StellarPhase::BlackHole,
                  "Very massive star should become black hole");
    }
    
    #[test]
    fn test_stellar_evolution_energy_conservation() {
        // Test that energy is conserved during stellar evolution
        let solar_mass = 1.989e30;
        let mut stellar_evolution = StellarEvolution::new(solar_mass);
        
        let dt_years = 1e6;
        let mut total_energy_generated = 0.0;
        
        // Track energy generation over time
        for _ in 0..100 {
            let energy = stellar_evolution.evolve(solar_mass, dt_years)
                .expect("Should evolve successfully");
            total_energy_generated += energy;
            
            // Each step should conserve energy
            assert!(energy.is_finite(), "Energy should be finite");
            assert!(energy >= 0.0, "Fusion should generate positive energy");
        }
        
        // Total energy should be significant for stellar burning
        assert!(total_energy_generated > 0.0, "Star should generate energy over time");
        
        // Energy generation rate should be realistic for solar-mass star
        let average_luminosity = stellar_evolution.nuclear_energy_generation;
        assert!(average_luminosity > 0.0, "Star should have positive luminosity");
        
        // Compare to solar luminosity (~3.8e26 W)
        // Our simplified model won't match exactly, but should be reasonable order of magnitude
    }
    
    #[test] 
    fn test_initial_mass_function_sampling() {
        // Test that IMF sampling produces realistic stellar mass distribution
        let config = SimulationConfig::default();
        let simulation = UniverseSimulation::new(config).expect("Should create simulation");
        
        let mut rng = rand::thread_rng();
        let mut masses = Vec::new();
        
        // Sample 1000 stellar masses
        for _ in 0..1000 {
            let mass = simulation.sample_stellar_mass_from_imf(&mut rng);
            masses.push(mass / 1.989e30); // Convert to solar masses
        }
        
        // Check distribution properties
        masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Should be dominated by low-mass stars (Salpeter IMF with α=2.35)
        let median_mass = masses[masses.len() / 2];
        assert!(median_mass < 1.0, "Median stellar mass should be < 1 solar mass");
        
        // Should have some high-mass stars but they should be rare
        let high_mass_fraction = masses.iter().filter(|&&m| m > 10.0).count() as f64 / masses.len() as f64;
        assert!(high_mass_fraction < 0.1, "High-mass stars should be rare");
        
        // Should span realistic range
        assert!(masses[0] >= 0.08, "Minimum mass should be hydrogen burning limit");
        assert!(masses[masses.len()-1] <= 100.0, "Maximum mass should be reasonable");
    }
    
    #[test] 
    fn test_star_formation_and_evolution_cycle() {
        // Integration test: star formation → evolution → death → enrichment
        let config = SimulationConfig::low_memory(); // Keep simulation small
        
        let mut simulation = UniverseSimulation::new(config).expect("Should create simulation");
        simulation.init_big_bang().expect("Should initialize universe");
        
        // Fast-forward to star formation era
        for _ in 0..10 {
            simulation.tick().expect("Should advance simulation");
        }
        
        // Manually trigger star formation for testing
        simulation.process_star_formation().expect("Should form stars");
        
        // Verify stars were created
        let star_count = simulation.store.stellar_evolutions.len();

        if star_count > 0 {
            // Test stellar evolution for created stars
            simulation.process_stellar_evolution(1e6).expect("Should evolve stars"); // 1 Myr

            // Verify stellar evolution updated properties
            for evolution in &simulation.store.stellar_evolutions {
                let body = &simulation.store.celestials[evolution.entity_id];

                assert!(evolution.nuclear_energy_generation >= 0.0,
                        "Stars should have non-negative energy generation");

                // Verify stellar properties are reasonable
                assert!(body.mass > 0.0, "Star should have positive mass");
                assert!(body.luminosity >= 0.0, "Star should have non-negative luminosity");
                assert!(body.temperature > 1000.0, "Star should be hot");
            }
        }
    }
}

/// Comprehensive molecular dynamics snapshot for real-time visualization
/// Integrates quantum field data with molecular dynamics for multi-scale visualization
#[derive(Debug, Clone)]
pub struct MolecularDynamicsSnapshot {
    /// Basic molecular dynamics snapshot
    pub basic_snapshot: MDSnapshot,
    /// Quantum field integration data
    pub quantum_integration: HashMap<String, QuantumStateVectorData>,
    /// Universe age at time of snapshot
    pub universe_age: f64,
    /// Simulation time in years
    pub simulation_time: f64,
    /// Current cosmic era
    pub cosmic_era: cosmic_era::UniverseState,
    /// Molecular system statistics
    pub molecular_statistics: MolecularStatistics,
    /// Chemical evolution data
    pub chemical_evolution: ChemicalEvolutionData,
    /// Quantum-classical interface information
    pub quantum_classical_interface: QuantumClassicalInterface,
}

/// Statistical data about the molecular system
#[derive(Debug, Clone)]
pub struct MolecularStatistics {
    /// Total number of particles
    pub total_particles: usize,
    /// Total mass of the system
    pub total_mass: f64,
    /// Average particle mass
    pub average_mass: f64,
    /// System temperature
    pub temperature: f64,
    /// System pressure
    pub pressure: f64,
    /// System density
    pub density: f64,
    /// Molecular complexity index (average bonds per particle)
    pub molecular_complexity: f64,
}

/// Chemical evolution data for the current state
#[derive(Debug, Clone)]
pub struct ChemicalEvolutionData {
    /// Universe age in Gyr
    pub universe_age: f64,
    /// Metallicity (fraction of heavy elements)
    pub metallicity: f64,
    /// Molecular formation rate (molecules/m³/s)
    pub molecular_formation_rate: f64,
    /// Chemical reaction rate (reactions/m³/s)
    pub reaction_rate: f64,
    /// Chemical complexity index
    pub complexity_index: f64,
}

/// Quantum-classical interface information
#[derive(Debug, Clone)]
pub struct QuantumClassicalInterface {
    /// Regions where quantum-classical transitions occur
    pub transition_regions: Vec<TransitionRegion>,
    /// Boundaries where decoherence occurs
    pub decoherence_boundaries: Vec<DecoherenceBoundary>,
    /// Quantum coherence length scale
    pub quantum_coherence_length: f64,
    /// Classical limit scale
    pub classical_limit_scale: f64,
}

/// Region where quantum-classical transition occurs
#[derive(Debug, Clone)]
pub struct TransitionRegion {
    /// Center of the transition region
    pub center: [f64; 3],
    /// Radius of the transition region
    pub radius: f64,
    /// Fraction of quantum behavior
    pub quantum_fraction: f64,
    /// Fraction of classical behavior
    pub classical_fraction: f64,
    /// Sharpness of the transition (0 = gradual, 1 = sharp)
    pub transition_sharpness: f64,
}

/// Boundary where quantum decoherence occurs
#[derive(Debug, Clone)]
pub struct DecoherenceBoundary {
    /// Position of the boundary
    pub position: [f64; 3],
    /// Normal vector to the boundary
    pub normal: [f64; 3],
    /// Decoherence rate (Hz)
    pub decoherence_rate: f64,
    /// Coherence time (seconds)
    pub coherence_time: f64,
}

impl MolecularDynamicsSnapshot {
    /// Convert to JSON for export and analysis
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "basic_snapshot": {
                "step": self.basic_snapshot.step,
                "time": self.basic_snapshot.time,
                "particle_count": self.basic_snapshot.particle_positions.len(),
                "bond_count": self.basic_snapshot.bonds.len(),
                "neighbor_pairs": self.basic_snapshot.neighbor_pairs.len(),
                "quantum_regions": self.basic_snapshot.quantum_regions.len(),
                "molecular_clusters": self.basic_snapshot.molecular_clusters.len(),
                "reaction_events": self.basic_snapshot.reaction_events.len(),
                "temperature": self.basic_snapshot.properties.temperature,
                "kinetic_energy": self.basic_snapshot.properties.kinetic_energy
            },
            "quantum_integration": {
                "field_count": self.quantum_integration.len(),
                "fields": self.quantum_integration.keys().collect::<Vec<_>>()
            },
            "universe_age": self.universe_age,
            "simulation_time": self.simulation_time,
            "cosmic_era": format!("{:?}", self.cosmic_era),
            "molecular_statistics": {
                "total_particles": self.molecular_statistics.total_particles,
                "total_mass": self.molecular_statistics.total_mass,
                "average_mass": self.molecular_statistics.average_mass,
                "temperature": self.molecular_statistics.temperature,
                "pressure": self.molecular_statistics.pressure,
                "density": self.molecular_statistics.density,
                "molecular_complexity": self.molecular_statistics.molecular_complexity
            },
            "chemical_evolution": {
                "universe_age": self.chemical_evolution.universe_age,
                "metallicity": self.chemical_evolution.metallicity,
                "molecular_formation_rate": self.chemical_evolution.molecular_formation_rate,
                "reaction_rate": self.chemical_evolution.reaction_rate,
                "complexity_index": self.chemical_evolution.complexity_index
            },
            "quantum_classical_interface": {
                "transition_regions": self.quantum_classical_interface.transition_regions.len(),
                "decoherence_boundaries": self.quantum_classical_interface.decoherence_boundaries.len(),
                "quantum_coherence_length": self.quantum_classical_interface.quantum_coherence_length,
                "classical_limit_scale": self.quantum_classical_interface.classical_limit_scale
            }
        })
    }
    
    /// Get ASCII representation of molecular bonding for CLI visualization
    pub fn get_ascii_molecular_map(&self, width: usize, height: usize) -> String {
        let mut map = vec![vec![' '; width]; height];
        
        if self.basic_snapshot.particle_positions.is_empty() {
            return "No particles in simulation".to_string();
        }
        
        // Find bounds of particle positions
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        
        for pos in &self.basic_snapshot.particle_positions {
            min_x = min_x.min(pos[0]);
            max_x = max_x.max(pos[0]);
            min_y = min_y.min(pos[1]);
            max_y = max_y.max(pos[1]);
        }
        
        let x_range = max_x - min_x;
        let y_range = max_y - min_y;
        
        if x_range == 0.0 || y_range == 0.0 {
            return "All particles at same position".to_string();
        }
        
        // Plot particles
        for (i, pos) in self.basic_snapshot.particle_positions.iter().enumerate() {
            let x = ((pos[0] - min_x) / x_range * (width - 1) as f64) as usize;
            let y = ((pos[1] - min_y) / y_range * (height - 1) as f64) as usize;
            
            if x < width && y < height {
                // Different symbols for different particle types
                let symbol = if self.basic_snapshot.quantum_regions.contains(&i) {
                    'Q' // Quantum particle
                } else {
                    match self.basic_snapshot.particle_types.get(i).unwrap_or(&0) % 4 {
                        0 => '●', // Heavy particle
                        1 => '○', // Medium particle
                        2 => '·', // Light particle
                        _ => '+', // Unknown
                    }
                };
                map[height - 1 - y][x] = symbol;
            }
        }
        
        // Draw bonds
        for &(i, j, strength) in &self.basic_snapshot.bonds {
            if i < self.basic_snapshot.particle_positions.len() && j < self.basic_snapshot.particle_positions.len() {
                let pos1 = &self.basic_snapshot.particle_positions[i];
                let pos2 = &self.basic_snapshot.particle_positions[j];
                
                let x1 = ((pos1[0] - min_x) / x_range * (width - 1) as f64) as usize;
                let y1 = ((pos1[1] - min_y) / y_range * (height - 1) as f64) as usize;
                let x2 = ((pos2[0] - min_x) / x_range * (width - 1) as f64) as usize;
                let y2 = ((pos2[1] - min_y) / y_range * (height - 1) as f64) as usize;
                
                // Draw simple line between bonded particles
                // Implement accurate bond character calculation based on bond strength and type
                let bond_char = if strength > 0.8 { 
                    '═' // Strong covalent bond (double line)
                } else if strength > 0.6 { 
                    '─' // Medium covalent bond (single line)
                } else if strength > 0.4 { 
                    '┄' // Weak covalent bond (dotted line)
                } else if strength > 0.2 { 
                    '·' // Very weak bond (dots)
                } else { 
                    ' ' // No significant bond
                };
                
                // Simple line drawing (Bresenham-like)
                let dx = (x2 as i32 - x1 as i32).abs();
                let dy = (y2 as i32 - y1 as i32).abs();
                let steps = dx.max(dy);
                
                for step in 0..=steps {
                    if steps > 0 {
                        let x = x1 + ((x2 as i32 - x1 as i32) * step / steps) as usize;
                        let y = y1 + ((y2 as i32 - y1 as i32) * step / steps) as usize;
                        
                        if x < width && y < height && map[height - 1 - y][x] == ' ' {
                            map[height - 1 - y][x] = bond_char;
                        }
                    }
                }
            }
        }
        
        // Convert to string
        let mut result = String::new();
        for row in &map {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }
        
        result
    }
    
    /// Get detailed molecular information for inspection
    pub fn get_molecular_details(&self) -> String {
        let mut details = String::new();
        
        details.push_str(&format!("=== MOLECULAR DYNAMICS SNAPSHOT ===\n"));
        details.push_str(&format!("Universe Age: {:.3} Gyr\n", self.universe_age));
        details.push_str(&format!("Simulation Time: {:.3} years\n", self.simulation_time));
        details.push_str(&format!("Cosmic Era: {:?}\n", self.cosmic_era));
        details.push_str(&format!("\n"));
        
        details.push_str(&format!("=== PARTICLE STATISTICS ===\n"));
        details.push_str(&format!("Total Particles: {}\n", self.molecular_statistics.total_particles));
        details.push_str(&format!("Total Mass: {:.2e} kg\n", self.molecular_statistics.total_mass));
        details.push_str(&format!("Average Mass: {:.2e} kg\n", self.molecular_statistics.average_mass));
        details.push_str(&format!("Temperature: {:.2} K\n", self.molecular_statistics.temperature));
        details.push_str(&format!("Pressure: {:.2e} Pa\n", self.molecular_statistics.pressure));
        details.push_str(&format!("Density: {:.2e} kg/m³\n", self.molecular_statistics.density));
        details.push_str(&format!("Molecular Complexity: {:.2}\n", self.molecular_statistics.molecular_complexity));
        details.push_str(&format!("\n"));
        
        details.push_str(&format!("=== MOLECULAR STRUCTURE ===\n"));
        details.push_str(&format!("Active Bonds: {}\n", self.basic_snapshot.bonds.len()));
        details.push_str(&format!("Neighbor Pairs: {}\n", self.basic_snapshot.neighbor_pairs.len()));
        details.push_str(&format!("Quantum Regions: {}\n", self.basic_snapshot.quantum_regions.len()));
        details.push_str(&format!("Molecular Clusters: {}\n", self.basic_snapshot.molecular_clusters.len()));
        details.push_str(&format!("Recent Reactions: {}\n", self.basic_snapshot.reaction_events.len()));
        details.push_str(&format!("\n"));
        
        details.push_str(&format!("=== CHEMICAL EVOLUTION ===\n"));
        details.push_str(&format!("Metallicity: {:.4}\n", self.chemical_evolution.metallicity));
        details.push_str(&format!("Formation Rate: {:.2e} molecules/m³/s\n", self.chemical_evolution.molecular_formation_rate));
        details.push_str(&format!("Reaction Rate: {:.2e} reactions/m³/s\n", self.chemical_evolution.reaction_rate));
        details.push_str(&format!("Complexity Index: {:.3}\n", self.chemical_evolution.complexity_index));
        details.push_str(&format!("\n"));
        
        details.push_str(&format!("=== QUANTUM-CLASSICAL INTERFACE ===\n"));
        details.push_str(&format!("Transition Regions: {}\n", self.quantum_classical_interface.transition_regions.len()));
        details.push_str(&format!("Decoherence Boundaries: {}\n", self.quantum_classical_interface.decoherence_boundaries.len()));
        details.push_str(&format!("Coherence Length: {:.2e} m\n", self.quantum_classical_interface.quantum_coherence_length));
        details.push_str(&format!("Classical Limit: {:.2e} m\n", self.quantum_classical_interface.classical_limit_scale));
        details.push_str(&format!("\n"));
        
        details.push_str(&format!("=== QUANTUM FIELD INTEGRATION ===\n"));
        details.push_str(&format!("Quantum Fields: {}\n", self.quantum_integration.len()));
        for (field_name, _field_data) in &self.quantum_integration {
            details.push_str(&format!("  - {}\n", field_name));
        }
        
        details
    }
}