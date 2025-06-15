//! Universe Simulation Core Library
//! 
//! Implements the complete universe simulation from Big Bang to far future
//! with autonomous AI agents evolving toward immortality.

use physics_engine::{PhysicsEngine, PhysicsState, ElementTable, EnvironmentProfile};
use physics_engine::nuclear_physics::{StellarNucleosynthesis, NeutronCaptureProcess, process_neutron_capture, Nucleus};
use bevy_ecs::prelude::*;
use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use uuid::Uuid;
use rand::Rng;
use md5;

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

/// Stellar evolution component for tracking nuclear burning stages
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct StellarEvolution {
    pub nucleosynthesis: StellarNucleosynthesis,
    pub core_temperature: f64,                 // K
    pub core_density: f64,                     // kg/m³
    pub core_composition: Vec<(u32, u32, f64)>, // (Z, A, abundance)
    pub nuclear_fuel_fraction: f64,            // Fraction of nuclear fuel remaining
    pub main_sequence_lifetime: f64,           // Years
    pub evolutionary_phase: StellarPhase,
    pub nuclear_energy_generation: f64,        // W/kg
}

/// Stellar evolutionary phases based on nuclear burning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StellarPhase {
    MainSequence,        // Hydrogen burning in core
    SubgiantBranch,      // Hydrogen exhausted, core contracting
    RedGiantBranch,      // Hydrogen shell burning
    HorizontalBranch,    // Helium burning in core (low-mass stars)
    AsymptoticGiantBranch, // Helium shell burning
    PlanetaryNebula,     // Mass loss phase
    WhiteDwarf,          // Stellar remnant
    Supernova,           // Massive star explosion
    NeutronStar,         // Compact remnant
    BlackHole,           // Ultimate compact object
}

#[derive(Component)]
pub struct HasPlanets;

#[derive(Component)]
pub struct HasLife;

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
        // Process stellar evolution and nuclear burning (always active for existing stars)
        self.process_stellar_evolution()?;
        
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
        
        // Process supernova nucleosynthesis and enrichment (always check)
        self.process_supernova_nucleosynthesis()?;
        
        Ok(())
    }
    
    /// Process stellar evolution based on nuclear burning
    /// This drives realistic stellar evolution timescales and element production
    fn process_stellar_evolution(&mut self) -> Result<()> {
        let dt_years = self.tick_span_years;
        let mut stellar_deaths = Vec::new();
        
        // Query all stars with stellar evolution component
        let mut stellar_query = self.world.query::<(Entity, &mut CelestialBody, &mut StellarEvolution)>();
        let stellar_data: Vec<(Entity, CelestialBody, StellarEvolution)> = stellar_query.iter_mut(&mut self.world)
            .filter(|(_, body, _)| matches!(body.body_type, CelestialBodyType::Star))
            .map(|(entity, body, evolution)| (entity, body.clone(), evolution.clone()))
            .collect();
        
        for (entity, mut body, mut evolution) in stellar_data {
            // Update stellar age
            body.age += dt_years;
            
            // Process nuclear burning and stellar evolution
            let energy_released = evolution.evolve(body.mass, dt_years)?;
            
            // Update stellar properties based on evolution
            body.luminosity = energy_released * body.mass; // Total luminosity
            body.temperature = self.calculate_stellar_temperature(body.mass);
            
            // Update composition with fusion products
            self.update_stellar_composition(&mut body, &evolution);
            
            // Check for stellar death
            if matches!(evolution.evolutionary_phase, 
                       StellarPhase::Supernova | 
                       StellarPhase::PlanetaryNebula |
                       StellarPhase::WhiteDwarf |
                       StellarPhase::NeutronStar |
                       StellarPhase::BlackHole) {
                stellar_deaths.push((entity, body.clone(), evolution.clone()));
            }
            
            // Update components in ECS
            if let Some(mut stellar_entity) = self.world.get_entity_mut(entity) {
                stellar_entity.insert(body);
                stellar_entity.insert(evolution);
            }
        }
        
        // Process stellar deaths
        for (entity, body, evolution) in stellar_deaths {
            self.process_stellar_death(entity, &body, &evolution)?;
        }
        
        Ok(())
    }
    
    /// Update stellar composition based on nuclear burning products
    fn update_stellar_composition(&self, body: &mut CelestialBody, evolution: &StellarEvolution) {
        // Update composition based on nuclear fusion products
        // This is simplified - in reality would track detailed isotopic abundances
        
        // Add fusion products to stellar composition
        for (z, _a, abundance) in &evolution.core_composition {
            if *abundance > 0.01 { // Only significant abundances
                // Convert to element table using set_abundance method
                let ppm = (*abundance * 1e6) as u32; // Convert to parts per million
                if *z <= 118 && *z > 0 {
                    body.composition.set_abundance(*z as usize, ppm);
                }
            }
        }
    }
    
    /// Process stellar death and remnant formation
    fn process_stellar_death(&mut self, entity: Entity, body: &CelestialBody, evolution: &StellarEvolution) -> Result<()> {
        let solar_mass = 1.989e30;
        let mass_ratio = body.mass / solar_mass;
        
        match evolution.evolutionary_phase {
            StellarPhase::WhiteDwarf => {
                // Form white dwarf remnant
                let wd_mass = body.mass * 0.6; // Typical WD mass fraction
                let wd_radius = 5e6; // ~Earth radius
                
                let white_dwarf = CelestialBody {
                    id: Uuid::new_v4(),
                    body_type: CelestialBodyType::WhiteDwarf,
                    mass: wd_mass,
                    radius: wd_radius,
                    luminosity: body.luminosity * 0.01, // Cooling WD
                    temperature: 50000.0, // Hot surface
                    age: body.age,
                    composition: body.composition.clone(),
                };
                
                // Replace star with white dwarf
                if let Some(mut entity_mut) = self.world.get_entity_mut(entity) {
                    entity_mut.insert(white_dwarf);
                    entity_mut.remove::<StellarEvolution>();
                }
                
                tracing::info!("White dwarf formed from {:.2} M☉ star", mass_ratio);
            },
            
            StellarPhase::NeutronStar => {
                // Form neutron star
                let ns_mass = 1.4 * solar_mass; // Typical NS mass
                let ns_radius = 12e3; // ~12 km radius
                
                let neutron_star = CelestialBody {
                    id: Uuid::new_v4(),
                    body_type: CelestialBodyType::NeutronStar,
                    mass: ns_mass,
                    radius: ns_radius,
                    luminosity: 1e28, // X-ray luminosity
                    temperature: 1e6, // Surface temperature
                    age: body.age,
                    composition: ElementTable::new(), // Mostly neutrons
                };
                
                if let Some(mut entity_mut) = self.world.get_entity_mut(entity) {
                    entity_mut.insert(neutron_star);
                    entity_mut.remove::<StellarEvolution>();
                }
                
                tracing::info!("Neutron star formed from {:.2} M☉ star", mass_ratio);
            },
            
            StellarPhase::BlackHole => {
                // Form black hole
                let bh_mass = body.mass; // Assume minimal mass loss
                let bh_radius = 2.95e3 * (bh_mass / solar_mass); // Schwarzschild radius
                
                let black_hole = CelestialBody {
                    id: Uuid::new_v4(),
                    body_type: CelestialBodyType::BlackHole,
                    mass: bh_mass,
                    radius: bh_radius,
                    luminosity: 0.0, // No thermal emission
                    temperature: 0.0,
                    age: body.age,
                    composition: ElementTable::new(), // Information is lost
                };
                
                if let Some(mut entity_mut) = self.world.get_entity_mut(entity) {
                    entity_mut.insert(black_hole);
                    entity_mut.remove::<StellarEvolution>();
                }
                
                tracing::info!("Black hole formed from {:.2} M☉ star", mass_ratio);
            },
            
            _ => {
                // Other phases handled elsewhere
            }
        }
        
        Ok(())
    }
    
    /// Process supernova nucleosynthesis and chemical enrichment
    /// This distributes fusion products to the interstellar medium
    fn process_supernova_nucleosynthesis(&mut self) -> Result<()> {
        let mut supernova_events = Vec::new();
        
        // Find stars in supernova phase
        let mut query = self.world.query::<(Entity, &CelestialBody, &StellarEvolution)>();
        for (entity, body, evolution) in query.iter(&self.world) {
            if matches!(evolution.evolutionary_phase, StellarPhase::Supernova) {
                supernova_events.push((entity, body.clone(), evolution.clone()));
            }
        }
        
        for (_entity, body, evolution) in supernova_events {
            // Create enriched gas cloud from supernova ejecta
            self.create_enriched_gas_cloud(&body, &evolution)?;
            
            // Process r-process nucleosynthesis in neutron-rich environment
            self.process_r_process_nucleosynthesis(&body)?;
            
            tracing::info!("Supernova explosion enriched interstellar medium with heavy elements");
        }
        
        Ok(())
    }
    
    /// Create enriched gas cloud from supernova ejecta
    fn create_enriched_gas_cloud(&mut self, star: &CelestialBody, _evolution: &StellarEvolution) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Eject fusion products into interstellar medium
        let num_ejecta_particles = 1000; // Simplified
        let ejecta_velocity = 1e7; // 10,000 km/s typical supernova velocity
        
        for _ in 0..num_ejecta_particles {
            let direction = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ).normalize();
            
            // Position around the exploding star
            let ejecta_position = Vector3::new(
                star.id.as_u128() as f64 * 1e12 % 1e15, // Use ID for deterministic position
                star.id.as_u128() as f64 * 1e11 % 1e15,
                star.id.as_u128() as f64 * 1e10 % 1e15,
            ) + direction * rng.gen_range(1e13..1e15); // 0.1-10 pc from star
            
            let ejecta_velocity_vec = direction * ejecta_velocity * rng.gen_range(0.5..1.5);
            
            // Create enriched gas particle
            let enriched_particle = PhysicsState {
                position: ejecta_position,
                velocity: ejecta_velocity_vec,
                acceleration: Vector3::zeros(),
                mass: 1.67e-24, // ~10^3 proton masses per particle
                charge: 0.0,
                temperature: 1e6, // Hot supernova ejecta
                entropy: 1e-20,
            };
            
            self.world.spawn(enriched_particle);
        }
        
        Ok(())
    }
    
    /// Process r-process nucleosynthesis in supernova environment
    fn process_r_process_nucleosynthesis(&self, _star: &CelestialBody) -> Result<()> {
        // Create seed nuclei for r-process
        let mut seed_nuclei = vec![
            Nucleus::new(26, 56), // ⁵⁶Fe seed
            Nucleus::new(28, 62), // ⁶²Ni seed
        ];
        
        // High neutron flux typical of supernova r-process site
        let neutron_flux = 1e25; // neutrons/cm²/s
        let temperature = 1e9; // 1 billion K
        let dt = 1.0; // 1 second timescale
        
        // Process r-process neutron capture
        let _energy_released = process_neutron_capture(
            &mut seed_nuclei,
            neutron_flux,
            temperature,
            NeutronCaptureProcess::RProcess,
            dt,
        )?;
        
        // The heavy r-process elements would be distributed with the ejecta
        // This completes the nucleosynthesis pathway for elements heavier than iron
        
        Ok(())
    }

    /// Process star formation from gas clouds
    /// Based on the Initial Mass Function (IMF) and stellar evolution theory
    fn process_star_formation(&mut self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Only form stars in appropriate cosmic eras
        let can_form_stars = matches!(self.cosmic_era, 
            cosmic_era::CosmicEra::Starbirth | 
            cosmic_era::CosmicEra::PlanetaryAge |
            cosmic_era::CosmicEra::Biogenesis
        );
        
        if !can_form_stars {
            return Ok(());
        }
        
        // Star formation rate depends on cosmic era and available gas
        let star_formation_rate = match self.cosmic_era {
            cosmic_era::CosmicEra::Starbirth => 0.01,        // Peak star formation
            cosmic_era::CosmicEra::PlanetaryAge => 0.005,    // Star formation continues
            cosmic_era::CosmicEra::Biogenesis => 0.001,      // Late star formation
            _ => 0.0,
        };
        
        // Count available gas particles (simplified)
        let gas_particle_count = self.world.query::<&PhysicsState>()
            .iter(&self.world)
            .filter(|state| state.temperature < 10000.0) // Cool gas
            .count();
        
        if gas_particle_count < 1000 {
            return Ok(()); // Not enough gas for star formation
        }
        
        // Determine if star formation occurs this tick
        let formation_probability = star_formation_rate * (gas_particle_count as f64 / 1e6);
        if !rng.gen_bool(formation_probability) {
            return Ok(());
        }
        
        // Sample stellar mass from Initial Mass Function (IMF)
        // Using Salpeter IMF: dN/dM ∝ M^(-2.35)
        let stellar_mass = self.sample_stellar_mass_from_imf(&mut rng);
        
        // Find suitable location for star formation (gas-rich region)
        let star_position = self.find_star_formation_site(&mut rng)?;
        
        // Create stellar physics state
        let stellar_radius = self.calculate_stellar_radius(stellar_mass);
        let stellar_luminosity = self.calculate_stellar_luminosity(stellar_mass);
        let stellar_temperature = self.calculate_stellar_temperature(stellar_mass);
        
        let stellar_body = CelestialBody {
            id: Uuid::new_v4(),
            body_type: CelestialBodyType::Star,
            mass: stellar_mass,
            radius: stellar_radius,
            luminosity: stellar_luminosity,
            temperature: stellar_temperature,
            age: 0.0,
            composition: ElementTable::new(), // Start with primordial composition
        };
        
        let stellar_state = PhysicsState {
            position: star_position,
            velocity: Vector3::new(
                rng.gen_range(-1e4..1e4), // ±10 km/s
                rng.gen_range(-1e4..1e4),
                rng.gen_range(-1e4..1e4),
            ),
            acceleration: Vector3::zeros(),
            mass: stellar_mass,
            charge: 0.0,
            temperature: stellar_temperature,
            entropy: 0.0,
        };
        
        let stellar_evolution = StellarEvolution::new(stellar_mass);
        
        // Spawn the star
        self.world.spawn((stellar_body, stellar_state, stellar_evolution));
        
        tracing::info!("Star formed with mass {:.2} M☉ at position {:?}", 
                      stellar_mass / 1.989e30, star_position);
        
        Ok(())
    }
    
    /// Sample stellar mass from Initial Mass Function (Salpeter IMF)
    /// Based on observational data: dN/dM ∝ M^(-2.35)
    fn sample_stellar_mass_from_imf<R: Rng>(&self, rng: &mut R) -> f64 {
        let solar_mass = 1.989e30; // kg
        
        // IMF parameters
        let alpha: f64 = 2.35; // Salpeter slope
        let m_min: f64 = 0.08; // Minimum mass (M☉)
        let m_max: f64 = 100.0; // Maximum mass (M☉)
        
        // Inverse transform sampling for power law distribution
        let u = rng.gen::<f64>();
        let exponent: f64 = 1.0 - alpha;
        
        let mass_solar = if exponent.abs() < 1e-6 {
            // Special case for α = 1
            m_min * (m_max / m_min).powf(u)
        } else {
            let numerator = m_min.powf(exponent) + u * (m_max.powf(exponent) - m_min.powf(exponent));
            numerator.powf(1.0 / exponent)
        };
        
        mass_solar * solar_mass
    }
    
    /// Find suitable location for star formation (gas-rich region)
    fn find_star_formation_site<R: Rng>(&self, rng: &mut R) -> Result<Vector3<f64>> {
        // Simplified: random position within simulation bounds
        // In reality, would use hydrodynamic simulation to find dense gas regions
        
        Ok(Vector3::new(
            rng.gen_range(-1e16..1e16), // ±10,000 light-years
            rng.gen_range(-1e16..1e16),
            rng.gen_range(-1e16..1e16),
        ))
    }
    
    /// Calculate stellar radius using mass-radius relation
    /// Based on stellar structure theory
    fn calculate_stellar_radius(&self, mass: f64) -> f64 {
        let solar_mass = 1.989e30;
        let solar_radius = 6.96e8; // m
        let mass_ratio = mass / solar_mass;
        
        // Main sequence mass-radius relation: R ∝ M^0.8
        solar_radius * mass_ratio.powf(0.8)
    }
    
    /// Calculate stellar luminosity using mass-luminosity relation
    /// Based on stellar structure theory
    fn calculate_stellar_luminosity(&self, mass: f64) -> f64 {
        let solar_mass = 1.989e30;
        let solar_luminosity = 3.828e26; // W
        let mass_ratio = mass / solar_mass;
        
        // Main sequence mass-luminosity relation
        if mass_ratio < 0.43 {
            // Low mass: L ∝ M^2.3
            solar_luminosity * mass_ratio.powf(2.3)
        } else if mass_ratio < 2.0 {
            // Intermediate mass: L ∝ M^4
            solar_luminosity * mass_ratio.powf(4.0)
        } else {
            // High mass: L ∝ M^3.5
            solar_luminosity * mass_ratio.powf(3.5)
        }
    }
    
    /// Calculate stellar surface temperature
    /// Based on mass-temperature relation
    fn calculate_stellar_temperature(&self, mass: f64) -> f64 {
        let solar_mass = 1.989e30;
        let solar_temperature = 5778.0; // K
        let mass_ratio = mass / solar_mass;
        
        // Approximate main sequence temperature relation: T ∝ M^0.54
        solar_temperature * mass_ratio.powf(0.54)
    }

    /// Process planet formation around stars
    fn process_planet_formation(&mut self) -> Result<()> {
        let mut rng = rand::thread_rng();
        let mut new_planets = Vec::new();
        let mut stars_to_mark = Vec::new();

        let mut query = self.world.query_filtered::<(Entity, &CelestialBody, &PhysicsState), Without<HasPlanets>>();
        let star_entities: Vec<(Entity, CelestialBody, PhysicsState)> = query.iter(&self.world)
            .filter(|(_, body, _)| matches!(body.body_type, CelestialBodyType::Star))
            .map(|(entity, body, state)| (entity, body.clone(), state.clone()))
            .collect();

        for (star_entity, star_body, star_state) in star_entities {
            // 80% chance to form planets
            if rng.gen::<f64>() > 0.8 {
                stars_to_mark.push(star_entity);
                continue;
            }

            let num_planets = rng.gen_range(1..=8);
            for i in 0..num_planets {
                let orbital_radius = 1.0e11 * (i as f64 + 1.5); // ~0.6 AU spacing
                let planet_mass = rng.gen_range(1.0e22..1.0e25); // From Moon to Super-Earth mass
                let planet_radius = rng.gen_range(1.0e6..1.0e7); // Earth-like radius range

                // Position in a circular orbit in the XY plane
                let angle = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                let planet_pos_relative = Vector3::new(
                    orbital_radius * angle.cos(),
                    orbital_radius * angle.sin(),
                    0.0,
                );
                let planet_pos = star_state.position + planet_pos_relative;

                // Orbital velocity
                let orbital_speed = (physics_engine::GRAVITATIONAL_CONSTANT * star_body.mass / orbital_radius).sqrt();
                let orbital_velocity = Vector3::new(
                    -orbital_speed * angle.sin(),
                    orbital_speed * angle.cos(),
                    0.0,
                );

                let planet_body = CelestialBody {
                    id: Uuid::new_v4(),
                    body_type: CelestialBodyType::Planet,
                    mass: planet_mass,
                    radius: planet_radius,
                    luminosity: 0.0,
                    temperature: 288.0, // Placeholder temperature
                    age: 0.0,
                    composition: ElementTable::new(),
                };

                let planet_state = PhysicsState {
                    position: planet_pos,
                    velocity: star_state.velocity + orbital_velocity,
                    acceleration: Vector3::zeros(),
                    mass: planet_mass,
                    charge: 0.0,
                    temperature: 288.0,
                    entropy: 0.0,
                };

                let planet_class = match rng.gen_range(0..5) {
                    0 => PlanetClass::E,
                    1 => PlanetClass::D,
                    2 => PlanetClass::I,
                    3 => PlanetClass::T,
                    _ => PlanetClass::G,
                };
                
                let habitability_score = if matches!(planet_class, PlanetClass::E) {
                    rng.gen_range(0.6..0.9)
                } else {
                    rng.gen_range(0.0..0.4)
                };

                let planet_env = PlanetaryEnvironment {
                    profile: EnvironmentProfile::default(), // Placeholder
                    stratigraphy: Vec::new(), // Placeholder
                    planet_class,
                    habitability_score,
                };

                new_planets.push((planet_body, planet_state, planet_env));
            }
            stars_to_mark.push(star_entity);
        }

        for (body, state, env) in new_planets {
            self.world.spawn((body, state, env));
        }

        for star_entity in stars_to_mark {
            self.world.entity_mut(star_entity).insert(HasPlanets);
        }

        Ok(())
    }

    /// Process emergence of life on suitable planets
    fn process_life_emergence(&mut self) -> Result<()> {
        let mut rng = rand::thread_rng();
        let mut planets_to_mark = Vec::new();
        let mut new_lineages = Vec::new();

        let mut query = self.world.query_filtered::<(Entity, &PlanetaryEnvironment), Without<HasLife>>();
        
        let potential_planets: Vec<(Entity, f64)> = query.iter(&self.world)
            .map(|(entity, env)| (entity, env.habitability_score))
            .collect();

        for (planet_entity, habitability_score) in potential_planets {
            // Low probability for abiogenesis on highly habitable planets
            if habitability_score > 0.7 && rng.gen::<f64>() < 0.01 { // 1% chance per tick
                let new_lineage = AgentLineage {
                    id: Uuid::new_v4(),
                    parent_id: None,
                    code_hash: format!("{:x}", md5::compute(rng.gen::<[u8; 16]>())),
                    generation: 1,
                    fitness: rng.gen(),
                    sentience_level: 0.01,
                    industrialization_level: 0.0,
                    digitalization_level: 0.0,
                    tech_level: 0.01,
                    immortality_achieved: false,
                    last_mutation_tick: self.current_tick,
                };
                new_lineages.push(new_lineage);
                planets_to_mark.push(planet_entity);
            } else if habitability_score > 0.5 {
                // If not creating life, still mark it to avoid re-checking every time
                if rng.gen::<f64>() < 0.1 { // 10% chance to stop checking
                    planets_to_mark.push(planet_entity);
                }
            }
        }

        for lineage in new_lineages {
            self.world.spawn(lineage);
        }

        for entity in planets_to_mark {
            self.world.entity_mut(entity).insert(HasLife);
        }

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
        // TODO: This is a temporary fix to synchronize the ECS world with the physics
        // engine's particle list for visualization. The core issue is the architectural
        // mismatch between the ECS-based UniverseSimulation and the vec-based PhysicsEngine.
        // A proper fix involves refactoring `update_physics` to handle this correctly.
        self.sync_ecs_to_physics_engine_particles()?;

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

    /// TEMPORARY: Synchronizes PhysicsState components from ECS to physics_engine.particles
    fn sync_ecs_to_physics_engine_particles(&mut self) -> Result<()> {
        use physics_engine::{FundamentalParticle, ParticleType, QuantumState};

        self.physics_engine.particles.clear();
        let mut query = self.world.query::<&physics_engine::PhysicsState>();

        for state in query.iter(&self.world) {
            // Determine particle type based on mass (approximation for initial particles)
            let particle_type = if (state.mass - 1.67e-27).abs() < 1e-29 {
                ParticleType::Proton
            } else if (state.mass - 6.64e-27).abs() < 1e-29 {
                ParticleType::Helium
            } else {
                ParticleType::DarkMatter // Fallback for unknown particles
            };

            // Create a default quantum state
            let quantum_state = QuantumState {
                wave_function: Vec::new(),
                entanglement_partners: Vec::new(),
                decoherence_time: 0.0,
                measurement_basis: physics_engine::MeasurementBasis::Position,
                superposition_amplitudes: std::collections::HashMap::new(),
            };

            let particle = FundamentalParticle {
                particle_type,
                position: state.position,
                momentum: state.velocity * state.mass,
                spin: nalgebra::Vector3::new(nalgebra::Complex::new(0.5, 0.0), nalgebra::Complex::new(0.0, 0.0), nalgebra::Complex::new(0.0, 0.0)),
                color_charge: None,
                electric_charge: state.charge,
                mass: state.mass,
                energy: 0.0, // TODO: Calculate from momentum and mass
                creation_time: 0.0,
                decay_time: None,
                quantum_state,
                interaction_history: Vec::new(),
            };
            self.physics_engine.particles.push(particle);
        }

        Ok(())
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
                physics_engine::ParticleType::Neutron |
                physics_engine::ParticleType::Helium
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

impl StellarEvolution {
    /// Create new stellar evolution data for a given stellar mass
    /// Based on stellar evolution theory (Kippenhahn & Weigert)
    pub fn new(stellar_mass: f64) -> Self {
        let solar_mass = 1.989e30; // kg
        let mass_ratio = stellar_mass / solar_mass;
        
        // Main sequence lifetime scales as M^(-2.5) approximately
        let main_sequence_lifetime = 10e9 * mass_ratio.powf(-2.5); // years
        
        // Core temperature and density from stellar structure equations
        // T_core ∝ M/R and ρ_core ∝ M/R³
        let core_temperature = 1.5e7 * mass_ratio.powf(0.8); // K
        let core_density = 1.5e5 * mass_ratio.powf(2.0); // kg/m³
        
        // Initial composition: primordial hydrogen/helium
        let core_composition = vec![
            (1, 1, 0.75),  // 75% Hydrogen
            (2, 4, 0.25),  // 25% Helium
            // Heavy elements negligible for Pop III stars
        ];
        
        Self {
            nucleosynthesis: StellarNucleosynthesis::new(),
            core_temperature,
            core_density,
            core_composition,
            nuclear_fuel_fraction: 1.0,
            main_sequence_lifetime,
            evolutionary_phase: StellarPhase::MainSequence,
            nuclear_energy_generation: 0.0,
        }
    }
    
    /// Update stellar evolution based on nuclear burning
    pub fn evolve(&mut self, stellar_mass: f64, dt_years: f64) -> Result<f64> {
        let dt_seconds = dt_years * 365.25 * 24.0 * 3600.0;
        
        // Process nuclear burning in stellar core
        let energy_released = self.nucleosynthesis.process_stellar_burning(
            self.core_temperature,
            self.core_density,
            &mut self.core_composition
        )?;
        
        // Convert energy release rate to luminosity (W/kg)
        self.nuclear_energy_generation = energy_released / dt_seconds;
        
        // Update nuclear fuel fraction
        let hydrogen_abundance = self.core_composition.iter()
            .find(|(z, a, _)| *z == 1 && *a == 1)
            .map(|(_, _, abundance)| *abundance)
            .unwrap_or(0.0);
        
        self.nuclear_fuel_fraction = hydrogen_abundance / 0.75; // Initial H fraction
        
        // Determine evolutionary phase transition
        self.update_evolutionary_phase(stellar_mass, dt_years);
        
        Ok(energy_released)
    }
    
    fn update_evolutionary_phase(&mut self, stellar_mass: f64, _dt_years: f64) {
        let solar_mass = 1.989e30;
        let mass_ratio = stellar_mass / solar_mass;
        
        match self.evolutionary_phase {
            StellarPhase::MainSequence => {
                // Transition when core hydrogen is exhausted
                if self.nuclear_fuel_fraction < 0.1 {
                    self.evolutionary_phase = StellarPhase::SubgiantBranch;
                    // Core contracts and heats up
                    self.core_temperature *= 1.5;
                    self.core_density *= 10.0;
                }
            },
            StellarPhase::SubgiantBranch => {
                // Quick transition to red giant branch
                self.evolutionary_phase = StellarPhase::RedGiantBranch;
            },
            StellarPhase::RedGiantBranch => {
                // For massive stars (M > 8 M☉), go to supernova
                if mass_ratio > 8.0 && self.core_temperature > 3e9 {
                    self.evolutionary_phase = StellarPhase::Supernova;
                }
                // For intermediate mass stars, helium ignition
                else if self.core_temperature > 1e8 {
                    self.evolutionary_phase = StellarPhase::HorizontalBranch;
                }
            },
            StellarPhase::HorizontalBranch => {
                // Helium burning phase
                let helium_abundance = self.core_composition.iter()
                    .find(|(z, a, _)| *z == 2 && *a == 4)
                    .map(|(_, _, abundance)| *abundance)
                    .unwrap_or(0.0);
                
                if helium_abundance < 0.05 {
                    self.evolutionary_phase = StellarPhase::AsymptoticGiantBranch;
                }
            },
            StellarPhase::AsymptoticGiantBranch => {
                // Mass loss leads to planetary nebula
                self.evolutionary_phase = StellarPhase::PlanetaryNebula;
            },
            StellarPhase::PlanetaryNebula => {
                // Final fate depends on mass
                if mass_ratio < 1.4 {
                    self.evolutionary_phase = StellarPhase::WhiteDwarf;
                } else if mass_ratio < 3.0 {
                    self.evolutionary_phase = StellarPhase::NeutronStar;
                } else {
                    self.evolutionary_phase = StellarPhase::BlackHole;
                }
            },
            StellarPhase::Supernova => {
                // Supernova explosion -> compact remnant
                if mass_ratio < 25.0 {
                    self.evolutionary_phase = StellarPhase::NeutronStar;
                } else {
                    self.evolutionary_phase = StellarPhase::BlackHole;
                }
            },
            _ => {
                // End states: no further evolution
            }
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