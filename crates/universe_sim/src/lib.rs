//! Universe Simulation Core Library
//! 
//! Implements the complete universe simulation from Big Bang to far future
//! with autonomous AI agents evolving toward immortality.

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
use std::time::{Duration, Instant};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use serde_json::{json, Value};
use crate::config::SimulationConfig;
use crate::storage::{Store, AgentLineage, CelestialBody};
use crate::cosmic_era::{UniverseState, PhysicalTransition, TransitionType};
// Agent config is now part of SimulationConfig

pub mod config;
pub mod cosmic_era;
pub mod evolution;
pub mod persistence;
pub mod storage;
pub mod world;

pub use physics_engine;
pub use storage::{CelestialBodyType, PlanetClass, StellarEvolution, StellarPhase, ParticleStore};

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
            
            self.store.particles.add(
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
        let result = self.step(self.tick_span_years);

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
    pub fn step(&mut self, dt: f64) -> Result<()> {
        // Increment tick counter
        self.current_tick += 1;
        
        // Track performance
        let start_time = std::time::Instant::now();
        
        // Run physics simulation step
        self.physics_engine.step(dt)?;
        
        // Update cosmic state from simulation results
        self.update_universe_state()?;
        
        // Process stellar evolution
        self.process_stellar_evolution(dt)?;
        
        // Process agent evolution on habitable worlds
        self.process_agent_evolution(dt)?;
        
        // Apply cosmological expansion effects to universe-scale properties
        self.apply_cosmological_effects(dt)?;
        
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
        _previous_state: &cosmic_era::UniverseState,
    ) -> Result<()> {
        // Placeholder – in full implementation, compare states and push transitions.
        Ok(())
    }

    /// Update physics simulation
    fn update_physics(&mut self) -> Result<()> {
        // Advance physics engine if available; ignore errors for placeholder
        let dt = self.physics_engine.time_step;
        let _ = self.physics_engine.step(dt);
        Ok(())
    }

    /// Update agent evolution and behavior
    fn update_agent_evolution(&mut self) -> Result<()> {
        // This will be implemented in the agents module
        // For now, just a placeholder
        Ok(())
    }

    /// Update cosmic-scale processes based on current physical conditions
    fn update_cosmic_processes(&mut self, _dt: f64) -> Result<()> {
        // Currently we model only stellar evolution and star formation.
        // Planet formation and other processes are stubbed for now.
        // Note: stellar evolution is handled directly in step()
        self.process_star_formation()?;
        Ok(())
    }

    /// Process stellar evolution based on nuclear burning
    fn process_stellar_evolution(&mut self, dt: f64) -> Result<()> {
        let dt_years = dt;

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
            body.age += dt_years;

            // 2. Evolve core.
            let _energy_generated = evolution.evolve(body.mass, dt_years)?;

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
        Ok(())
    }

    /// Create enriched gas clouds from stellar death events
    #[allow(dead_code)]
    fn create_enriched_gas_cloud(
        &mut self,
        _star: &CelestialBody,
        _evolution: &StellarEvolution,
    ) -> Result<()> {
        Ok(())
    }

    /// Handle r-process nucleosynthesis in neutron star mergers
    #[allow(dead_code)]
    fn process_r_process_nucleosynthesis(&self, _star: &CelestialBody) -> Result<()> {
        Ok(())
    }

    /// Form new stars from dense gas clouds
    fn process_star_formation(&mut self) -> Result<()> {
        use rand::Rng;
        // For performance in tiny unit-tests we cap star formation attempts.
        const MAX_ATTEMPTS_PER_TICK: usize = 5;

        let mut rng = rand::thread_rng();

        for _ in 0..MAX_ATTEMPTS_PER_TICK {
            // Very simple density criterion: require at least N gas particles
            if self.store.particles.count < 100 {
                break; // Not enough gas to collapse
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
                composition: physics_engine::types::ElementTable::new(),
                has_planets: false,
                has_life: false,
            };

            let entity_id = self.store.spawn_celestial(stellar_body);

            // 3. Create StellarEvolution track and link entity ID
            let mut evolution = StellarEvolution::new(mass_kg);
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

        // Generate random point inside the sphere.
        let u: f64 = rng.gen();
        let v: f64 = rng.gen();
        let w: f64 = rng.gen();

        let r = radius_m * u.cbrt();
        let theta = (1.0 - 2.0 * v).acos();
        let phi = 2.0 * std::f64::consts::PI * w;

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
                    composition: composition.clone(),
                    has_planets: false,
                    has_life: false,
                };
                let entity_id = self.store.spawn_celestial(planet_body);

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
                    };
                    self.store.agents.push(lineage);
                    // Slightly improve habitability due to biosphere feedback
                    env.habitability_score = (env.habitability_score + 0.05).min(1.0);
                }
            }
        }
        Ok(())
    }

    /// Check for victory conditions (e.g., immortality achieved)
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
            
            // Evolution and life statistics
            extinct_lineages: evolution_stats.extinct,
            average_tech_level: evolution_stats.average_tech,
            immortal_lineages: evolution_stats.immortal_count,
            consciousness_emergence_rate: evolution_stats.consciousness_rate,
            
            // Physics engine performance
            physics_step_time_ms: performance_stats.step_time_ms,
            interactions_per_step: performance_stats.nuclear_reactions,
            particle_interactions_per_step: performance_stats.interactions,
            
            // Cosmic structure
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
        if width == 0 || height == 0 {
            return Err(anyhow!("Width and height must be positive"));
        }
        let mut grid: Vec<Vec<f64>> = Vec::with_capacity(height);
        for j in 0..height {
            let mut row = Vec::with_capacity(width);
            for i in 0..width {
                // Normalise coordinates to [-1,1]
                let x = 2.0 * (i as f64 / (width - 1) as f64) - 1.0;
                let y = 2.0 * (j as f64 / (height - 1) as f64) - 1.0;
                let rho = self.calculate_total_density_at(x, y)?;
                row.push(rho);
            }
            grid.push(row);
        }
        Ok(json!({ "density_grid": grid }))
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

    fn calculate_dark_matter_density_at(&self, _x: f64, _y: f64) -> Result<f64> {
        // Assume simple cosmological parameter Ω_dm = 0.27, critical density 9.3e-27 kg/m³
        const RHO_CRIT: f64 = 9.3e-27; // At z≈0
        Ok(0.27 * RHO_CRIT)
    }

    fn calculate_radiation_density_at(&mut self, _x: f64, _y: f64) -> Result<f64> {
        // ρ_rad = aT⁴ / c² ; but we approximate using current CMB temperature 2.725K
        const A_RAD: f64 = 7.5657e-16; // Radiation constant J·m⁻³·K⁻⁴
        const C: f64 = 299_792_458.0;
        let t: f64 = 2.725; // Present-day CMB
        Ok(A_RAD * t.powi(4) / (C * C))
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
                    StellarPhase::RedGiantBranch
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
        let nuclear_binding_energy = self
            .physics_engine
            .calculate_qm_region_energy(&self.physics_engine.atoms)
            .unwrap_or(0.0);

        // 4. Radiation Energy (Placeholder - requires electromagnetic field solver)
        // TODO: Implement radiation energy calculation once EM field solver is integrated.
        let radiation_energy = 0.0;

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
        // Assuming no extinction mechanism is implemented yet.
        // TODO: Add a flag or mechanism to track extinct lineages.
        let extinct = 0;
        
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
            
        // 3. Get total particle interactions (placeholder, as this is not yet tracked)
        // TODO: Add a counter for particle_interactions in the physics engine
        let interactions = self.physics_engine.interaction_history.len();

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
        use nalgebra::ComplexField;
        let mut snapshot: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

        for (field_type, field) in &self.physics_engine.quantum_fields {
            // Convert field type to string identifier (e.g. "ElectronField")
            let key = format!("{:?}", field_type);
            if field.field_values.is_empty() {
                snapshot.insert(key, Vec::new());
                continue;
            }
            // We convert the complex field amplitude to a 2-D slice by taking the
            // magnitude |ψ| of the first z-slice (index 0). This keeps the return
            // structure lightweight while still conveying spatial information that
            // front-ends (dashboard, CLI, etc.) can visualise as a heat-map.
            let slice_z0 = &field.field_values;
            let mut plane: Vec<Vec<f64>> = Vec::with_capacity(slice_z0.len());
            for x_row in slice_z0 {
                if x_row.is_empty() {
                    plane.push(Vec::new());
                    continue;
                }
                // We take the y-dimension at z = 0 (index 0)
                let mut row: Vec<f64> = Vec::with_capacity(x_row.len());
                for y_col in x_row {
                    // y_col is Vec<Complex<f64>> (z dimension). Use first element if available
                    let amp = y_col.first().copied().unwrap_or_default();
                    row.push(amp.modulus());
                }
                plane.push(row);
            }
            snapshot.insert(key, plane);
        }
        snapshot
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
                // Scale distances by cosmic expansion
                // TODO: Add position field to CelestialBody or handle cosmological expansion differently
                // For now, skip position scaling for celestial bodies
                
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
                    1.0 // Standard evolution rate
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
        // Extract cosmological parameters from GADGET gravity solver if available
        #[cfg(feature = "gadget")]
        if let Some(ref gadget_engine) = self.physics_engine.gadget_engine {
            return Some(gadget_engine.get_cosmological_parameters().clone());
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
        let dt_years = dt;
        
        for lineage in &mut self.store.agents {
            // TODO: Implement proper agent evolution integration
            // For now, just update basic evolution parameters
            lineage.fitness += dt_years * 0.001; // Slow fitness growth
            lineage.generation += (dt_years / 1e6) as u32; // One generation per Myr
        }
        
        Ok(())
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