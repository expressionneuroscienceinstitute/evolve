//! Main universe simulation orchestrator

use crate::agent::{Agent, Lineage};
use crate::chemistry::ChemistryEngine;
use crate::config::SimulationConfig;
use crate::entropy::EntropyEngine;
use crate::events::{EventSystem, CosmicEvent};
use crate::physics::{PhysicsEngine, CelestialBody};
use crate::planet::{Planet, PlanetSystem};
use crate::tech::{TechTree, TechMilestone};
use crate::types::*;
use crate::{Result, SimError};
use std::collections::HashMap;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Main universe simulation
pub struct Universe {
    /// Current simulation tick
    current_tick: Tick,
    /// Configuration
    config: SimulationConfig,
    /// Random number generator
    rng: ChaCha8Rng,
    
    /// Physics engine
    physics: PhysicsEngine,
    /// Chemistry engine
    chemistry: ChemistryEngine,
    /// Entropy engine
    entropy: EntropyEngine,
    /// Event system
    events: EventSystem,
    /// Planet system
    planets: PlanetSystem,
    /// Technology tree
    tech_tree: TechTree,
    
    /// Celestial bodies (stars, planets, etc.)
    celestial_bodies: HashMap<StarId, CelestialBody>,
    /// Agents
    agents: HashMap<AgentId, Agent>,
    /// Lineages
    lineages: HashMap<LineageId, Lineage>,
    
    /// Simulation state
    running: bool,
    ups_target: f64,
    last_tick_time: std::time::Instant,
    
    /// Win condition tracking
    victory_achieved: bool,
    victory_lineage: Option<LineageId>,
}

impl Universe {
    /// Create a new universe simulation
    pub fn new(config: SimulationConfig) -> Result<Self> {
        let seed = config.simulation.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });
        
        let rng = ChaCha8Rng::seed_from_u64(seed);
        
        Ok(Self {
            current_tick: Tick::new(0),
            config: config.clone(),
            rng,
            
            physics: PhysicsEngine::new(&config.physics),
            chemistry: ChemistryEngine::new(&config.physics.chemistry),
            entropy: EntropyEngine::new(),
            events: EventSystem::new(),
            planets: PlanetSystem::new(),
            tech_tree: TechTree::new(),
            
            celestial_bodies: HashMap::new(),
            agents: HashMap::new(),
            lineages: HashMap::new(),
            
            running: false,
            ups_target: config.simulation.target_ups,
            last_tick_time: std::time::Instant::now(),
            
            victory_achieved: false,
            victory_lineage: None,
        })
    }
    
    /// Initialize the universe (Big Bang to early stellar formation)
    pub fn initialize(&mut self) -> Result<()> {
        log::info!("Initializing universe at tick {}", self.current_tick);
        
        // Generate initial stellar population
        self.generate_initial_stars()?;
        
        // Generate initial planets
        self.generate_initial_planets()?;
        
        // Seed initial agent population on habitable worlds
        self.seed_initial_agents()?;
        
        log::info!("Universe initialized with {} stars, {} planets, {} agents",
                  self.celestial_bodies.len(),
                  self.planets.list_planets().len(),
                  self.agents.len());
        
        Ok(())
    }
    
    /// Run one simulation tick
    pub fn tick(&mut self) -> Result<UniverseState> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Physics update
        let physics_results = self.physics.tick(
            self.current_tick,
            &mut self.celestial_bodies,
            self.config.simulation.years_per_tick,
        )?;
        
        // Step 2: Chemistry update
        self.chemistry.tick(self.current_tick)?;
        
        // Step 3: Entropy update
        self.entropy.tick(self.current_tick, physics_results.entropy_delta)?;
        
        // Step 4: Events processing
        let triggered_events = self.events.tick(self.current_tick);
        for event in triggered_events {
            self.process_cosmic_event(event)?;
        }
        
        // Step 5: Agent actions
        self.process_agent_actions()?;
        
        // Step 6: Planet evolution
        self.update_planetary_environments()?;
        
        // Step 7: Check win conditions
        self.check_victory_conditions()?;
        
        // Update tick counter
        self.current_tick = self.current_tick + Tick::new(1);
        
        // Calculate performance metrics
        let tick_duration = start_time.elapsed();
        let actual_ups = if tick_duration.as_secs_f64() > 0.0 {
            1.0 / tick_duration.as_secs_f64()
        } else {
            f64::INFINITY
        };
        
        Ok(UniverseState {
            tick: self.current_tick,
            ups: actual_ups,
            total_agents: self.agents.len() as u32,
            total_lineages: self.lineages.len() as u32,
            total_entropy: self.entropy.get_total_entropy(),
            victory_achieved: self.victory_achieved,
            victory_lineage: self.victory_lineage,
        })
    }
    
    /// Start simulation loop
    pub fn run(&mut self) -> Result<()> {
        self.running = true;
        self.last_tick_time = std::time::Instant::now();
        
        while self.running && self.current_tick.as_u64() < self.config.simulation.max_ticks {
            let state = self.tick()?;
            
            // Rate limiting to target UPS
            let target_frame_time = std::time::Duration::from_secs_f64(1.0 / self.ups_target);
            let elapsed = self.last_tick_time.elapsed();
            
            if elapsed < target_frame_time {
                std::thread::sleep(target_frame_time - elapsed);
            }
            
            self.last_tick_time = std::time::Instant::now();
            
            // Log progress periodically
            if self.current_tick.as_u64() % 1000 == 0 {
                log::info!("Tick {}: {:.1} UPS, {} agents, entropy {:.2e}",
                          self.current_tick,
                          state.ups,
                          state.total_agents,
                          state.total_entropy.as_f64());
            }
            
            // Check for victory
            if self.victory_achieved {
                log::info!("Victory achieved by lineage {:?} at tick {}!",
                          self.victory_lineage, self.current_tick);
                break;
            }
        }
        
        Ok(())
    }
    
    /// Stop simulation
    pub fn stop(&mut self) {
        self.running = false;
    }
    
    /// Generate initial stellar population
    fn generate_initial_stars(&mut self) -> Result<()> {
        let grid_cells = self.config.world.grid_size.0 * self.config.world.grid_size.1;
        let star_count = (grid_cells as f64 * self.config.world.star_formation_rate) as usize;
        
        for i in 0..star_count {
            let position = Coord3D::new(
                self.rng.gen_range(0.0..self.config.world.grid_size.0 as f64),
                self.rng.gen_range(0.0..self.config.world.grid_size.1 as f64),
                self.rng.gen_range(-1000.0..1000.0), // parsecs
            );
            
            let mass_factor = self.rng.gen_range(0.1..50.0); // Solar masses
            let mass = MassEnergy::new(mass_factor * crate::constants::physics::M_SOL);
            
            let star = CelestialBody {
                body_type: crate::physics::CelestialBodyType::Star,
                mass,
                position,
                velocity: Velocity::zero(),
                temperature: Temperature::new(5778.0 * mass_factor.powf(0.5)),
                luminosity: crate::constants::physics::L_SOL * mass_factor.powf(3.5),
                age_ticks: 0,
            };
            
            self.celestial_bodies.insert(StarId::new(i as u64 + 1), star);
        }
        
        Ok(())
    }
    
    /// Generate initial planets around stars
    fn generate_initial_planets(&mut self) -> Result<()> {
        let star_ids: Vec<StarId> = self.celestial_bodies.keys().cloned().collect();
        
        for star_id in star_ids {
            let planet_count = self.rng.gen_range(0..8); // 0-7 planets per star
            
            for p in 0..planet_count {
                let distance_au = self.rng.gen_range(0.1..50.0);
                let stellar_metallicity = self.rng.gen_range(0.001..0.04);
                
                let position = Coord2D::new(
                    self.rng.gen_range(0..self.config.world.grid_size.0),
                    self.rng.gen_range(0..self.config.world.grid_size.1),
                );
                
                self.planets.generate_planet(
                    position,
                    stellar_metallicity,
                    distance_au,
                    &mut self.rng,
                )?;
            }
        }
        
        Ok(())
    }
    
    /// Seed initial agent population on habitable worlds
    fn seed_initial_agents(&mut self) -> Result<()> {
        let habitable_planets: Vec<&Planet> = self.planets.list_planets()
            .into_iter()
            .filter(|p| p.environment.is_habitable(&crate::constants::SurvivalThresholds::default()))
            .collect();
        
        if habitable_planets.is_empty() {
            log::warn!("No habitable planets found for initial agent seeding");
            return Ok(());
        }
        
        let agents_per_planet = self.config.evolution.initial_population / habitable_planets.len() as u32;
        let mut agent_id = 1;
        let mut lineage_id = 1;
        
        for planet in habitable_planets {
            // Create a lineage for this planet
            let lineage = Lineage::new(LineageId::new(lineage_id), self.current_tick);
            self.lineages.insert(LineageId::new(lineage_id), lineage);
            
            // Spawn agents
            for _ in 0..agents_per_planet {
                let agent = Agent::new(
                    AgentId::new(agent_id),
                    LineageId::new(lineage_id),
                    planet.position,
                );
                
                self.agents.insert(AgentId::new(agent_id), agent);
                agent_id += 1;
            }
            
            lineage_id += 1;
        }
        
        Ok(())
    }
    
    /// Process agent actions for this tick
    fn process_agent_actions(&mut self) -> Result<()> {
        let agent_ids: Vec<AgentId> = self.agents.keys().cloned().collect();
        
        for agent_id in agent_ids {
            // First, create observation without holding mutable reference
            let observation = {
                let agent = self.agents.get(&agent_id)
                    .ok_or_else(|| SimError::AgentError("Agent not found".to_string()))?;
                
                // Create observation data
                Observation {
                    tick: self.current_tick,
                    location: agent.position,
                    local_resources: ElementTable::new(),
                    nearby_agents: Vec::new(),
                    environment: EnvironmentSnapshot {
                        temperature: Temperature::new(288.0),
                        pressure: Pressure::from_atmospheres(1.0),
                        radiation: RadiationDose::new(0.002),
                        energy_flux: EnergyFlux::new(1.361),
                        liquid_water_fraction: 0.7,
                        atmospheric_oxygen: 0.21,
                        hazard_rate: 0.001,
                    },
                    hazards: Vec::new(),
                    energy_budget: agent.energy,
                    oracle_message: None,
                    available_techs: self.tech_tree.get_available_techs(agent.lineage_id),
                }
            };
            
            // Now process the action with mutable reference
            if let Some(agent) = self.agents.get_mut(&agent_id) {
                // Agent decides action
                let action = agent.observe_and_act(&observation)?;
                
                // Process action immediately within the mutable borrow
                match action {
                    AgentAction::Research { tech_id, energy_cost } => {
                        if agent.energy >= energy_cost {
                            agent.energy -= energy_cost;
                            self.tech_tree.research_tech(agent.lineage_id, tech_id)?;
                        }
                    },
                    AgentAction::Wait => {
                        // Do nothing
                    },
                    _ => {
                        // TODO: Implement other actions
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create observation for an agent
    fn create_agent_observation(&self, agent_id: AgentId) -> Result<Observation> {
        let agent = self.agents.get(&agent_id)
            .ok_or_else(|| SimError::AgentError("Agent not found".to_string()))?;
        
        // TODO: Implement comprehensive observation generation
        Ok(Observation {
            tick: self.current_tick,
            location: agent.position,
            local_resources: ElementTable::new(),
            nearby_agents: Vec::new(),
            environment: EnvironmentSnapshot {
                temperature: Temperature::new(288.0),
                pressure: Pressure::from_atmospheres(1.0),
                radiation: RadiationDose::new(0.002),
                energy_flux: EnergyFlux::new(1.361),
                liquid_water_fraction: 0.7,
                atmospheric_oxygen: 0.21,
                hazard_rate: 0.001,
            },
            hazards: Vec::new(),
            energy_budget: agent.energy,
            oracle_message: None,
            available_techs: self.tech_tree.get_available_techs(agent.lineage_id),
        })
    }
    
    /// Execute an agent action
    fn execute_agent_action(&mut self, agent_id: AgentId, action: AgentAction) -> Result<()> {
        match action {
            AgentAction::Research { tech_id, energy_cost } => {
                if let Some(agent) = self.agents.get_mut(&agent_id) {
                    if agent.energy >= energy_cost {
                        agent.energy -= energy_cost;
                        self.tech_tree.research_tech(agent.lineage_id, tech_id)?;
                    }
                }
            },
            AgentAction::Wait => {
                // Do nothing
            },
            _ => {
                // TODO: Implement other actions
            }
        }
        
        Ok(())
    }
    
    /// Process a cosmic event
    fn process_cosmic_event(&mut self, event: CosmicEvent) -> Result<()> {
        match event {
            CosmicEvent::Supernova { star_id, .. } => {
                // TODO: Implement supernova effects
                log::info!("Supernova occurred at star {}", star_id);
            },
            CosmicEvent::AsteroidImpact { planet_id, .. } => {
                // TODO: Implement impact effects
                log::info!("Asteroid impact on planet {}", planet_id);
            },
            _ => {
                // TODO: Implement other event types
            }
        }
        
        Ok(())
    }
    
    /// Update planetary environments
    fn update_planetary_environments(&mut self) -> Result<()> {
        let planet_ids: Vec<PlanetId> = self.planets.list_planets()
            .iter()
            .map(|p| p.id)
            .collect();
            
        for planet_id in planet_ids {
            self.planets.update_planet_environment(planet_id, self.current_tick)?;
        }
        
        Ok(())
    }
    
    /// Check victory conditions
    fn check_victory_conditions(&mut self) -> Result<()> {
        for (lineage_id, _lineage) in &self.lineages {
            let milestone = self.tech_tree.check_milestone(*lineage_id);
            
            if milestone >= crate::tech::TechMilestone::Immortality {
                self.victory_achieved = true;
                self.victory_lineage = Some(*lineage_id);
                break;
            }
        }
        
        Ok(())
    }
    
    /// Get current universe state
    pub fn get_state(&self) -> UniverseState {
        UniverseState {
            tick: self.current_tick,
            ups: 0.0, // Calculated in tick()
            total_agents: self.agents.len() as u32,
            total_lineages: self.lineages.len() as u32,
            total_entropy: self.entropy.get_total_entropy(),
            victory_achieved: self.victory_achieved,
            victory_lineage: self.victory_lineage,
        }
    }
}

/// Current state of the universe simulation
#[derive(Debug, Clone)]
pub struct UniverseState {
    pub tick: Tick,
    pub ups: f64,
    pub total_agents: u32,
    pub total_lineages: u32,
    pub total_entropy: Entropy,
    pub victory_achieved: bool,
    pub victory_lineage: Option<LineageId>,
}