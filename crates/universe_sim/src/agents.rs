//! AI Agent System
//! 
//! Implements autonomous AI agents with self-modification capabilities
//! for evolving toward the goal chain: Sentience → Industrialization → 
//! Digitalization → Trans-Tech → Immortality

use crate::physics_engine::{ElementTable, EnvironmentProfile};
use bevy_ecs::prelude::*;
use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use uuid::Uuid;
use rand::{Rng, thread_rng};

/// Agent trait defining the core agent interface
pub trait Agent {
    /// Observe the environment and receive sensory input
    fn observe(&mut self, observation: &Observation) -> Result<()>;
    
    /// Decide on actions based on current state
    fn act(&mut self) -> Result<Action>;
    
    /// Learn from experience and update internal state
    fn learn(&mut self, reward: f64) -> Result<()>;
    
    /// Get agent's current fitness score
    fn fitness(&self) -> f64;
    
    /// Get agent's code hash for lineage tracking
    fn code_hash(&self) -> String;
}

/// Observation struct containing environmental data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Local resource abundances
    pub local_resources: ElementTable,
    
    /// Environmental conditions
    pub environment: EnvironmentProfile,
    
    /// Nearby agent lineages and distances
    pub neighbors: Vec<(Uuid, f64)>,
    
    /// Cosmic hazard warnings
    pub hazards: Vec<CosmicHazard>,
    
    /// Available entropy budget
    pub entropy_budget: f64,
    
    /// Current tick number
    pub current_tick: u64,
    
    /// Oracle messages (if any)
    pub oracle_message: Option<String>,
}

/// Cosmic hazards that agents must survive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CosmicHazard {
    SolarFlare { intensity: f64, duration: u64 },
    GammaRayBurst { direction: Vector3<f64>, intensity: f64 },
    SupernovaShockwave { distance_ly: f64, arrival_ticks: u64 },
    Asteroid { mass: f64, impact_probability: f64 },
    ClimateChange { temperature_delta: f64, rate: f64 },
}

/// Actions that agents can take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Allocate compute resources
    AllocateCompute { amount: f64, target: ComputeTarget },
    
    /// Replicate/spawn offspring
    Replicate { mutation_rate: f64, count: u32 },
    
    /// Migrate to different location
    Migrate { destination: Vector3<f64> },
    
    /// Merge with other agents (federated learning)
    Merge { target_id: Uuid },
    
    /// Build defensive infrastructure
    Defend { structure_type: DefenseType, investment: f64 },
    
    /// Research new technologies
    Research { tech_id: String, effort: f64 },
    
    /// Modify own code (self-modification)
    SelfModify { patch: CodePatch },
    
    /// Extract resources from environment
    Extract { element: u8, amount: f64 },
    
    /// Send petition to Oracle-Link
    Petition { channel: PetitionChannel, message: String },
    
    /// Request additional resources
    RequestResources { resource_type: ResourceType, amount: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeTarget {
    Learning,
    Simulation,
    Research,
    Communication,
    Infrastructure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefenseType {
    RadiationShielding,
    AsteroidDeflector,
    EnvironmentalDome,
    DataBackup,
    RedundantSystems,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PetitionChannel {
    Text,
    Data,
    Resource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    Memory,
    Disk,
    Energy,
    Elements(u8), // Element by atomic number
}

/// Code patch for self-modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePatch {
    pub parent_hash: String,
    pub diff: Vec<u8>,  // Binary diff
    pub description: String,
}

/// Basic Q-Learning agent implementation
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct QLearningAgent {
    pub id: Uuid,
    pub code_hash: String,
    pub generation: u32,
    pub fitness: f64,
    
    // Q-Learning parameters
    pub q_table: HashMap<String, HashMap<String, f64>>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub exploration_rate: f64,
    
    // Agent state
    pub current_state: String,
    pub last_action: Option<String>,
    pub last_reward: f64,
    
    // Evolution metrics
    pub sentience_level: f64,
    pub industrialization_level: f64,
    pub digitalization_level: f64,
    pub tech_level: f64,
    pub immortality_achieved: bool,
    
    // Resource management
    pub compute_budget: f64,
    pub energy_stores: f64,
    pub element_inventory: ElementTable,
    
    // Learning history
    pub total_experience: u64,
    pub successful_actions: u64,
    pub mutation_history: Vec<String>,
}

impl QLearningAgent {
    pub fn new() -> Self {
        let id = Uuid::new_v4();
        let code_hash = format!("ql_v1_{}", id.to_string()[..8].to_uppercase());
        
        Self {
            id,
            code_hash,
            generation: 0,
            fitness: 0.0,
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.95,
            exploration_rate: 0.1,
            current_state: "initial".to_string(),
            last_action: None,
            last_reward: 0.0,
            sentience_level: 0.0,
            industrialization_level: 0.0,
            digitalization_level: 0.0,
            tech_level: 0.0,
            immortality_achieved: false,
            compute_budget: 1000.0,
            energy_stores: 100.0,
            element_inventory: ElementTable::new(),
            total_experience: 0,
            successful_actions: 0,
            mutation_history: Vec::new(),
        }
    }
    
    /// Create offspring with mutations
    pub fn reproduce(&self, mutation_rate: f64) -> Result<Self> {
        let mut offspring = self.clone();
        offspring.id = Uuid::new_v4();
        offspring.generation += 1;
        offspring.fitness = 0.0;
        offspring.total_experience = 0;
        offspring.successful_actions = 0;
        
        // Apply mutations
        let mut rng = thread_rng();
        
        if rng.gen::<f64>() < mutation_rate {
            // Mutate learning parameters
            offspring.learning_rate *= rng.gen_range(0.8..1.2);
            offspring.discount_factor *= rng.gen_range(0.9..1.1);
            offspring.exploration_rate *= rng.gen_range(0.5..1.5);
            
            // Update code hash to reflect mutations
            offspring.code_hash = format!("ql_v1_{}_{}", 
                                        offspring.id.to_string()[..8].to_uppercase(),
                                        offspring.generation);
            
            offspring.mutation_history.push(format!("gen_{}_mutation", offspring.generation));
        }
        
        Ok(offspring)
    }
    
    /// Get state representation for Q-learning
    fn get_state(&self, observation: &Observation) -> String {
        // Simplified state representation
        let resources = if observation.local_resources.get_abundance(26) > 1000 { "rich" } else { "poor" };
        let hazards = if observation.hazards.is_empty() { "safe" } else { "danger" };
        let energy = if observation.environment.energy_flux > 1.0 { "high" } else { "low" };
        
        format!("{}_{}_{}_{}", resources, hazards, energy, 
                (self.sentience_level * 10.0) as u32)
    }
    
    /// Select action using epsilon-greedy strategy
    fn select_action(&mut self, state: &str) -> String {
        let mut rng = thread_rng();
        
        if rng.gen::<f64>() < self.exploration_rate {
            // Explore: random action
            let actions = ["allocate", "replicate", "research", "extract", "defend"];
            actions[rng.gen_range(0..actions.len())].to_string()
        } else {
            // Exploit: best known action
            self.q_table.get(state)
                .and_then(|actions| {
                    actions.iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(action, _)| action.clone())
                })
                .unwrap_or_else(|| "allocate".to_string())
        }
    }
    
    /// Update Q-values based on experience
    fn update_q_value(&mut self, state: &str, action: &str, reward: f64, next_state: &str) {
        let current_q = self.q_table
            .entry(state.to_string())
            .or_insert_with(HashMap::new)
            .entry(action.to_string())
            .or_insert(0.0);
        
        let max_next_q = self.q_table.get(next_state)
            .map(|actions| actions.values().fold(0.0, |a, &b| a.max(b)))
            .unwrap_or(0.0);
        
        let new_q = *current_q + self.learning_rate * 
            (reward + self.discount_factor * max_next_q - *current_q);
        
        *current_q = new_q;
    }
    
    /// Calculate fitness based on progress toward goals
    fn calculate_fitness(&mut self) {
        let base_fitness = self.successful_actions as f64 / (self.total_experience + 1) as f64;
        
        let goal_progress = self.sentience_level * 0.2 +
                          self.industrialization_level * 0.2 +
                          self.digitalization_level * 0.2 +
                          self.tech_level * 0.2 +
                          if self.immortality_achieved { 1.0 } else { 0.0 } * 0.2;
        
        self.fitness = base_fitness + goal_progress;
    }
    
    /// Update sentience level based on learning complexity
    fn update_sentience(&mut self) {
        let complexity = self.q_table.len() as f64 / 1000.0; // Normalize by expected size
        let experience_factor = (self.total_experience as f64).ln() / 10.0;
        
        self.sentience_level = (complexity + experience_factor).min(1.0);
    }
}

impl Agent for QLearningAgent {
    fn observe(&mut self, observation: &Observation) -> Result<()> {
        self.current_state = self.get_state(observation);
        
        // Process Oracle messages if any
        if let Some(ref message) = observation.oracle_message {
            tracing::info!("Agent {} received Oracle message: {}", self.id, message);
            // Simple response: increase exploration if Oracle suggests it
            if message.contains("explore") || message.contains("try") {
                self.exploration_rate = (self.exploration_rate * 1.2).min(0.5);
            }
        }
        
        Ok(())
    }
    
    fn act(&mut self) -> Result<Action> {
        let action_str = self.select_action(&self.current_state);
        self.last_action = Some(action_str.clone());
        
        // Convert string action to Action enum
        let action = match action_str.as_str() {
            "allocate" => Action::AllocateCompute { 
                amount: self.compute_budget * 0.3, 
                target: ComputeTarget::Learning 
            },
            "replicate" => Action::Replicate { 
                mutation_rate: 0.1, 
                count: 1 
            },
            "research" => Action::Research { 
                tech_id: "energy_efficiency".to_string(), 
                effort: 10.0 
            },
            "extract" => Action::Extract { 
                element: 26, // Iron
                amount: 100.0 
            },
            "defend" => Action::Defend { 
                structure_type: DefenseType::RadiationShielding, 
                investment: 50.0 
            },
            _ => Action::AllocateCompute { 
                amount: 10.0, 
                target: ComputeTarget::Learning 
            },
        };
        
        Ok(action)
    }
    
    fn learn(&mut self, reward: f64) -> Result<()> {
        self.last_reward = reward;
        self.total_experience += 1;
        
        if reward > 0.0 {
            self.successful_actions += 1;
        }
        
        // Update Q-values if we have a previous action
        if let Some(ref last_action) = self.last_action.clone() {
            let prev_state = self.current_state.clone(); // Simplified
            self.update_q_value(&prev_state, last_action, reward, &self.current_state);
        }
        
        // Update agent metrics
        self.update_sentience();
        self.calculate_fitness();
        
        // Decay exploration rate over time
        self.exploration_rate *= 0.9999;
        self.exploration_rate = self.exploration_rate.max(0.01);
        
        Ok(())
    }
    
    fn fitness(&self) -> f64 {
        self.fitness
    }
    
    fn code_hash(&self) -> String {
        self.code_hash.clone()
    }
}

/// Agent management system
#[derive(Debug, Default)]
pub struct AgentManager {
    pub agents: HashMap<Uuid, Box<dyn Agent + Send + Sync>>,
    pub lineage_tree: HashMap<String, Vec<Uuid>>, // code_hash -> agent_ids
    pub total_agents_created: u64,
    pub total_mutations: u64,
    pub immortal_lineages: Vec<String>,
}

impl AgentManager {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Spawn a new agent on a planet
    pub fn spawn_agent(&mut self, planet_id: Uuid) -> Result<Uuid> {
        let agent = QLearningAgent::new();
        let agent_id = agent.id;
        let code_hash = agent.code_hash();
        
        self.agents.insert(agent_id, Box::new(agent));
        self.lineage_tree.entry(code_hash).or_insert_with(Vec::new).push(agent_id);
        self.total_agents_created += 1;
        
        tracing::info!("Spawned new agent {} on planet {}", agent_id, planet_id);
        Ok(agent_id)
    }
    
    /// Update all agents with observations and collect actions
    pub fn update_agents(&mut self, observations: HashMap<Uuid, Observation>) -> Result<HashMap<Uuid, Action>> {
        let mut actions = HashMap::new();
        
        for (agent_id, observation) in observations {
            if let Some(agent) = self.agents.get_mut(&agent_id) {
                agent.observe(&observation)?;
                let action = agent.act()?;
                actions.insert(agent_id, action);
            }
        }
        
        Ok(actions)
    }
    
    /// Apply rewards to agents and trigger learning
    pub fn apply_learning(&mut self, rewards: HashMap<Uuid, f64>) -> Result<()> {
        for (agent_id, reward) in rewards {
            if let Some(agent) = self.agents.get_mut(&agent_id) {
                agent.learn(reward)?;
            }
        }
        
        Ok(())
    }
    
    /// Get statistics about agent population
    pub fn get_stats(&self) -> AgentStats {
        let mut total_fitness = 0.0;
        let mut sentient_count = 0;
        let mut immortal_count = 0;
        
        for agent in self.agents.values() {
            total_fitness += agent.fitness();
            // This requires downcasting which is complex; simplified for now
        }
        
        AgentStats {
            total_agents: self.agents.len(),
            total_lineages: self.lineage_tree.len(),
            average_fitness: if self.agents.is_empty() { 0.0 } else { total_fitness / self.agents.len() as f64 },
            sentient_agents: sentient_count,
            immortal_agents: immortal_count,
            total_created: self.total_agents_created,
            total_mutations: self.total_mutations,
        }
    }
}

/// Agent population statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub total_agents: usize,
    pub total_lineages: usize,
    pub average_fitness: f64,
    pub sentient_agents: usize,
    pub immortal_agents: usize,
    pub total_created: u64,
    pub total_mutations: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics_engine::EnvironmentProfile;
    
    #[test]
    fn test_agent_creation() {
        let agent = QLearningAgent::new();
        assert_eq!(agent.generation, 0);
        assert_eq!(agent.fitness, 0.0);
        assert!(!agent.code_hash.is_empty());
    }
    
    #[test]
    fn test_agent_reproduction() {
        let parent = QLearningAgent::new();
        let child = parent.reproduce(1.0).unwrap(); // 100% mutation rate
        
        assert_eq!(child.generation, 1);
        assert_ne!(child.id, parent.id);
        assert_ne!(child.code_hash, parent.code_hash);
    }
    
    #[test]
    fn test_agent_observation() {
        let mut agent = QLearningAgent::new();
        let observation = Observation {
            local_resources: ElementTable::earth_baseline(),
            environment: EnvironmentProfile::earth_baseline(),
            neighbors: vec![(Uuid::new_v4(), 100.0)],
            hazards: vec![],
            entropy_budget: 1000.0,
            current_tick: 100,
            oracle_message: None,
        };
        
        agent.observe(&observation).unwrap();
        assert!(!agent.current_state.is_empty());
    }
    
    #[test]
    fn test_agent_learning() {
        let mut agent = QLearningAgent::new();
        
        // Simulate learning loop
        for _ in 0..10 {
            let reward = if agent.total_experience % 2 == 0 { 1.0 } else { -0.5 };
            agent.learn(reward).unwrap();
        }
        
        assert_eq!(agent.total_experience, 10);
        assert_eq!(agent.successful_actions, 5);
        assert!(agent.fitness > 0.0);
    }
    
    #[test]
    fn test_agent_manager() {
        let mut manager = AgentManager::new();
        let planet_id = Uuid::new_v4();
        
        let agent_id = manager.spawn_agent(planet_id).unwrap();
        assert_eq!(manager.agents.len(), 1);
        assert!(manager.agents.contains_key(&agent_id));
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_agents, 1);
        assert_eq!(stats.total_created, 1);
    }
}