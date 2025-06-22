//! # Embodied Agent Representation
//!
//! Defines how AI agents are physically represented in the universe simulation.
//! Agents are embodied as collections of fundamental particles with distributed
//! AI components that can interact with the physics environment.
//!
//! ## Environmental Challenges
//! - **Fields**: Spatially-varying scalar/vector fields (energy, temperature, electromagnetic)
//! - **Obstacles**: Static or dynamic obstacles that block or redirect agent movement
//! - **Energy Gradients**: Regions where energy cost or gain varies, influencing agent strategies
//! - **Agent-Agent Interaction**: Competition for resources and direct interaction capabilities

use anyhow::Result;
use nalgebra::{DVector, Vector3};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::{
    ai_core::AICore,
    meta_learning::MetaLearner,
    hypernetwork::Hypernetwork,
    curiosity::{CuriositySystem, ActionType},
    self_modification::AdvancedSelfModification,
    open_ended_evolution::OpenEndedEvolution,
    genetics::Genome,
};

/// Embodied AI agent in physics simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedAgent {
    // Identity
    pub id: Uuid,
    pub name: String,
    pub generation: u64,
    
    // Physical Body
    pub body_particles: Vec<usize>, // Indices into physics engine particle list
    pub position: Vector3<f64>,     // Center of mass
    pub velocity: Vector3<f64>,     // Overall motion
    pub energy: f64,               // Total energy
    pub mass: f64,                 // Total mass
    
    // Distributed AI Components
    pub neural_core: DistributedNeuralCore,
    pub meta_learner: DistributedMetaLearner,
    pub curiosity_system: DistributedCuriositySystem,
    pub self_modification: DistributedSelfModification,
    pub open_ended_evolution: DistributedOpenEndedEvolution,
    pub genome: Genome,
    
    // Behavioral State
    pub current_task: Option<TaskEmbedding>,
    pub performance_history: Vec<f64>,
    pub discovery_count: usize,
    pub last_action: Option<ActionType>,
    pub age: f64, // Time since creation
}

/// Distributed neural core across particle network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedNeuralCore {
    pub processing_particles: Vec<usize>, // Particles that handle computation
    pub memory_particles: Vec<usize>,     // Particles that store information
    pub communication_particles: Vec<usize>, // Particles for inter-agent communication
    pub architecture: NeuralArchitecture,
    pub learning_parameters: LearningParameters,
    pub quantum_states: Vec<QuantumState>,
}

/// Neural architecture distributed across particles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralArchitecture {
    pub input_particles: Vec<usize>,
    pub hidden_particles: Vec<Vec<usize>>, // Layers of hidden particles
    pub output_particles: Vec<usize>,
    pub connection_strengths: Vec<f64>, // Quantum entanglement strengths
    pub activation_functions: Vec<ActivationType>,
}

/// Learning parameters for distributed learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningParameters {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub quantum_coherence_time: f64,
    pub entanglement_threshold: f64,
}

/// Quantum state of a particle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub particle_index: usize,
    pub wave_function: Vec<f64>, // Real amplitudes for simplicity
    pub entanglement_partners: Vec<usize>,
    pub energy_level: f64,
    pub spin_state: Vector3<f64>,
    pub coherence_time: f64,
}

/// Distributed meta-learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedMetaLearner {
    pub meta_particles: Vec<usize>, // Particles that handle meta-learning
    pub parameter_particles: Vec<usize>, // Particles that store parameters
    pub adaptation_rate: f64,
    pub meta_parameters: Vec<MetaParameter>,
}

/// Distributed curiosity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCuriositySystem {
    pub sensory_particles: Vec<usize>,    // Particles that detect environment
    pub novelty_detector_particles: Vec<usize>, // Particles that identify novelty
    pub exploration_particles: Vec<usize>, // Particles that drive exploration
    pub prediction_particles: Vec<usize>,  // Particles that make predictions
    pub curiosity_level: f64,
    pub exploration_rate: f64,
}

/// Distributed self-modification system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSelfModification {
    pub modification_particles: Vec<usize>, // Particles that can modify others
    pub analysis_particles: Vec<usize>,     // Particles that analyze performance
    pub validation_particles: Vec<usize>,   // Particles that validate changes
    pub modification_rate: f64,
    pub safety_threshold: f64,
}

/// Distributed open-ended evolution system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedOpenEndedEvolution {
    pub evolution_particles: Vec<usize>,   // Particles that drive evolution
    pub archive_particles: Vec<usize>,     // Particles that store innovations
    pub diversity_particles: Vec<usize>,   // Particles that maintain diversity
    pub emergence_particles: Vec<usize>,   // Particles that detect emergence
    pub novelty_threshold: f64,
    pub diversity_pressure: f64,
}

/// Task embedding for embodied agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEmbedding {
    pub task_type: TaskType,
    pub complexity: f64,
    pub energy_requirement: f64,
    pub particle_requirement: usize,
    pub time_horizon: f64,
    pub success_criteria: Vec<f64>,
}

/// Types of tasks for embodied agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Exploration,
    EnergyHarvesting,
    InformationGathering,
    ToolCreation,
    SelfReplication,
    Communication,
    ProblemSolving,
    Innovation,
    Survival,
    Emergent,
}

/// Types of activation functions for neural particles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    Quantum,
    Emergent,
}

/// Meta-parameter types for distributed learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaParameter {
    LearningRate,
    ExplorationRate,
    CuriosityWeight,
    ModificationRate,
    EvolutionRate,
    CoherenceTime,
    EntanglementStrength,
}

/// Environmental field types that can affect agent behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    /// Scalar energy field - affects agent energy levels
    EnergyField { strength: f64, decay_rate: f64 },
    /// Temperature field - affects agent thermal properties
    TemperatureField { temperature: f64, conductivity: f64 },
    /// Electromagnetic field - affects charged particles
    ElectromagneticField { 
        electric_strength: Vector3<f64>, 
        magnetic_strength: Vector3<f64> 
    },
    /// Gravitational field - affects all particles
    GravitationalField { strength: f64 },
    /// Resource field - provides collectible resources
    ResourceField { resource_type: String, density: f64, regeneration_rate: f64 },
}

/// Environmental obstacle that can block or redirect agent movement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    pub id: Uuid,
    pub position: Vector3<f64>,
    pub size: Vector3<f64>,
    pub obstacle_type: ObstacleType,
    pub is_dynamic: bool,
    pub velocity: Vector3<f64>,
    pub damage_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObstacleType {
    /// Solid wall that completely blocks movement
    Wall { friction: f64 },
    /// Energy barrier that damages agents passing through
    EnergyBarrier { damage_per_second: f64 },
    /// Force field that repels or attracts agents
    ForceField { strength: f64, attraction: bool },
    /// Resource deposit that agents can interact with
    ResourceDeposit { resource_type: String, capacity: f64 },
}

/// Environmental feature that affects a region of space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalFeature {
    pub id: Uuid,
    pub position: Vector3<f64>,
    pub radius: f64,
    pub field_type: FieldType,
    pub is_active: bool,
    pub lifetime: Option<f64>, // None for permanent features
    pub age: f64,
}

/// Energy gradient that affects agent movement and energy costs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyGradient {
    pub id: Uuid,
    pub start_position: Vector3<f64>,
    pub end_position: Vector3<f64>,
    pub start_energy_cost: f64,
    pub end_energy_cost: f64,
    pub gradient_type: GradientType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientType {
    /// Linear gradient between two points
    Linear,
    /// Exponential gradient (steeper changes)
    Exponential { exponent: f64 },
    /// Step gradient (discrete energy levels)
    Step { step_size: f64 },
}

/// Resource that agents can collect and compete for
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: Uuid,
    pub position: Vector3<f64>,
    pub resource_type: String,
    pub value: f64,
    pub energy_content: f64,
    pub is_collected: bool,
    pub collection_time: Option<f64>,
    pub collected_by: Option<Uuid>,
}

/// Agent interaction capabilities and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInteraction {
    pub agent_id: Uuid,
    pub interaction_range: f64,
    pub interaction_strength: f64,
    pub current_target: Option<Uuid>,
    pub interaction_history: Vec<InteractionEvent>,
    pub resources_held: Vec<Resource>,
    pub energy_reserves: f64,
    pub max_energy_capacity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub timestamp: f64,
    pub event_type: InteractionEventType,
    pub target_id: Option<Uuid>,
    pub outcome: f64,
    pub energy_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionEventType {
    CollectResource { resource_type: String, value: f64 },
    CompeteWithAgent { opponent_id: Uuid, success: bool },
    AvoidObstacle { obstacle_id: Uuid, damage_taken: f64 },
    NavigateGradient { gradient_id: Uuid, energy_cost: f64 },
    InteractWithField { field_id: Uuid, energy_gain: f64 },
}

impl EmbodiedAgent {
    /// Create a new embodied agent
    pub fn new(name: String, position: Vector3<f64>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            generation: 0,
            body_particles: Vec::new(),
            position,
            velocity: Vector3::zeros(),
            energy: 0.0,
            mass: 0.0,
            neural_core: DistributedNeuralCore::new(),
            meta_learner: DistributedMetaLearner::new(),
            curiosity_system: DistributedCuriositySystem::new(),
            self_modification: DistributedSelfModification::new(),
            open_ended_evolution: DistributedOpenEndedEvolution::new(),
            genome: Genome::new(),
            current_task: None,
            performance_history: Vec::new(),
            discovery_count: 0,
            last_action: None,
            age: 0.0,
        }
    }
    
    /// Get sensory input from physics environment
    pub fn get_sensory_input(&self, physics_engine: &dyn PhysicsEngineInterface) -> Result<DVector<f64>> {
        let mut input = Vec::new();
        
        // Get sensory data from sensory particles
        for &particle_idx in &self.curiosity_system.sensory_particles {
            if let Some(particle_data) = physics_engine.get_particle_data(particle_idx) {
                input.push(particle_data.position.x);
                input.push(particle_data.position.y);
                input.push(particle_data.position.z);
                input.push(particle_data.energy);
                input.push(particle_data.electric_charge);
                input.push(particle_data.mass);
            }
        }
        
        // Add agent's own state
        input.push(self.position.x);
        input.push(self.position.y);
        input.push(self.position.z);
        input.push(self.velocity.x);
        input.push(self.velocity.y);
        input.push(self.velocity.z);
        input.push(self.energy);
        input.push(self.mass);
        input.push(self.age);
        
        // Pad to fixed size
        while input.len() < 100 {
            input.push(0.0);
        }
        
        Ok(DVector::from_vec(input))
    }
    
    /// Execute action in physics environment
    pub fn execute_action(&mut self, action: ActionType, physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        let outcome = match action {
            ActionType::Explore => self.execute_exploration(physics_engine)?,
            ActionType::Investigate => self.execute_investigation(physics_engine)?,
            ActionType::Experiment => self.execute_experiment(physics_engine)?,
            ActionType::Observe => self.execute_observation(physics_engine)?,
            ActionType::Interact => self.execute_interaction(physics_engine)?,
            ActionType::Learn => self.execute_learning(physics_engine)?,
            ActionType::Create => self.execute_creation(physics_engine)?,
            ActionType::Discover => self.execute_discovery(physics_engine)?,
        };
        
        self.last_action = Some(action);
        self.performance_history.push(outcome);
        
        Ok(outcome)
    }
    
    /// Execute exploration action
    fn execute_exploration(&mut self, physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        // Move exploration particles to generate motion
        for &particle_idx in &self.curiosity_system.exploration_particles {
            physics_engine.apply_force_to_particle(particle_idx, Vector3::new(1e-15, 0.0, 0.0))?;
        }
        
        // Update agent position based on particle movement
        self.update_position_from_particles(physics_engine)?;
        
        Ok(0.1) // Small positive outcome for exploration
    }
    
    /// Execute investigation action
    fn execute_investigation(&mut self, physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        let mut total_information = 0.0;
        
        // Focus sensory particles on nearby environment
        for &particle_idx in &self.curiosity_system.sensory_particles {
            if let Some(particle_data) = physics_engine.get_particle_data(particle_idx) {
                let distance = (particle_data.position - self.position).magnitude();
                if distance < 1e-7 {
                    total_information += particle_data.energy * 0.1;
                }
            }
        }
        
        Ok(total_information.min(1.0))
    }
    
    /// Execute experiment action
    fn execute_experiment(&mut self, physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        let mut interactions = 0;
        
        // Try to interact with nearby particles
        for &particle_idx in &self.body_particles {
            if let Some(particle_data) = physics_engine.get_particle_data(particle_idx) {
                for other_particle_idx in 0..physics_engine.get_particle_count() {
                    if let Some(other_data) = physics_engine.get_particle_data(other_particle_idx) {
                        let distance = (other_data.position - particle_data.position).magnitude();
                        if distance < 1e-8 {
                            interactions += 1;
                        }
                    }
                }
            }
        }
        
        Ok(interactions as f64 * 0.01)
    }
    
    /// Execute observation action
    fn execute_observation(&mut self, _physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        // Passive observation - just process sensory data
        Ok(0.05)
    }
    
    /// Execute interaction action
    fn execute_interaction(&mut self, physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        let mut energy_gained = 0.0;
        
        // Try to absorb energy from nearby particles
        for &particle_idx in &self.body_particles {
            if let Some(particle_data) = physics_engine.get_particle_data(particle_idx) {
                for other_particle_idx in 0..physics_engine.get_particle_count() {
                    if let Some(other_data) = physics_engine.get_particle_data(other_particle_idx) {
                        let distance = (other_data.position - particle_data.position).magnitude();
                        if distance < 1e-7 && other_data.energy > 0.0 {
                            energy_gained += other_data.energy * 0.1;
                            physics_engine.transfer_energy(other_particle_idx, particle_idx, other_data.energy * 0.1)?;
                        }
                    }
                }
            }
        }
        
        self.energy += energy_gained;
        Ok(energy_gained)
    }
    
    /// Execute learning action
    fn execute_learning(&mut self, _physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        // Meta-learning action - update learning parameters
        self.meta_learner.adaptation_rate *= 1.01; // Slight increase
        Ok(0.2)
    }
    
    /// Execute creation action
    fn execute_creation(&mut self, physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        // Create new particles through quantum processes
        let new_particle_idx = physics_engine.create_particle(
            self.position + Vector3::new(1e-8, 0.0, 0.0),
            Vector3::zeros(),
            9.1093837015e-31, // Electron mass
            -1.602176634e-19, // Electron charge
        )?;
        
        // Add to agent's body
        self.body_particles.push(new_particle_idx);
        
        Ok(0.3) // Good outcome for creation
    }
    
    /// Execute discovery action
    fn execute_discovery(&mut self, _physics_engine: &mut dyn PhysicsEngineInterface) -> Result<f64> {
        // Discovery action - analyze current state for novelty
        self.discovery_count += 1;
        Ok(0.4)
    }
    
    /// Update agent position based on particle positions
    fn update_position_from_particles(&mut self, physics_engine: &dyn PhysicsEngineInterface) -> Result<()> {
        let mut total_mass = 0.0;
        let mut weighted_position = Vector3::zeros();
        
        for &particle_idx in &self.body_particles {
            if let Some(particle_data) = physics_engine.get_particle_data(particle_idx) {
                weighted_position += particle_data.position * particle_data.mass;
                total_mass += particle_data.mass;
            }
        }
        
        if total_mass > 0.0 {
            self.position = weighted_position / total_mass;
        }
        
        Ok(())
    }
    
    /// Update agent age
    pub fn update_age(&mut self, dt: f64) {
        self.age += dt;
    }
    
    /// Get agent statistics
    pub fn get_statistics(&self) -> AgentStatistics {
        let average_performance = if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
        };
        
        let recent_performance = if self.performance_history.len() >= 10 {
            self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0
        } else {
            average_performance
        };
        
        AgentStatistics {
            id: self.id,
            name: self.name.clone(),
            generation: self.generation,
            age: self.age,
            particle_count: self.body_particles.len(),
            energy: self.energy,
            mass: self.mass,
            position: self.position,
            velocity: self.velocity,
            discovery_count: self.discovery_count,
            average_performance,
            recent_performance,
        }
    }
    
    /// Calculate energy cost for moving through an energy gradient
    pub fn calculate_gradient_energy_cost(&self, gradient: &EnergyGradient) -> f64 {
        let distance_to_start = (self.position - gradient.start_position).norm();
        let distance_to_end = (self.position - gradient.end_position).norm();
        let total_gradient_distance = (gradient.end_position - gradient.start_position).norm();
        
        if total_gradient_distance == 0.0 {
            return gradient.start_energy_cost;
        }
        
        let progress = distance_to_start / total_gradient_distance;
        let progress = progress.max(0.0).min(1.0);
        
        match gradient.gradient_type {
            GradientType::Linear => {
                gradient.start_energy_cost + (gradient.end_energy_cost - gradient.start_energy_cost) * progress
            }
            GradientType::Exponential { exponent } => {
                let factor = progress.powf(exponent);
                gradient.start_energy_cost + (gradient.end_energy_cost - gradient.start_energy_cost) * factor
            }
            GradientType::Step { step_size } => {
                let step_count = (progress / step_size).floor() as i32;
                let step_progress = step_count as f64 * step_size;
                gradient.start_energy_cost + (gradient.end_energy_cost - gradient.start_energy_cost) * step_progress
            }
        }
    }
    
    /// Apply environmental field effects to the agent
    pub fn apply_field_effects(&mut self, field: &EnvironmentalFeature, dt: f64) -> Result<f64> {
        let distance = (self.position - field.position).norm();
        if distance > field.radius {
            return Ok(0.0);
        }
        
        let field_strength = if field.lifetime.is_some() {
            // Decay field strength over time
            let age_factor = (field.age / field.lifetime.unwrap()).min(1.0);
            1.0 - age_factor
        } else {
            1.0
        };
        
        let distance_factor = 1.0 - (distance / field.radius);
        let total_effect = field_strength * distance_factor;
        
        match &field.field_type {
            FieldType::EnergyField { strength, decay_rate } => {
                let energy_gain = strength * total_effect * dt;
                self.energy += energy_gain;
                Ok(energy_gain)
            }
            FieldType::TemperatureField { temperature, conductivity } => {
                // Temperature affects agent thermal properties
                let thermal_effect = temperature * conductivity * total_effect * dt;
                // For now, just affect energy slightly
                self.energy += thermal_effect * 0.1;
                Ok(thermal_effect)
            }
            FieldType::ElectromagneticField { electric_strength, magnetic_strength } => {
                // EM fields affect charged particles
                let em_effect = (electric_strength.norm() + magnetic_strength.norm()) * total_effect * dt;
                // Apply force to charged particles
                Ok(em_effect)
            }
            FieldType::GravitationalField { strength } => {
                // Gravitational field affects all particles
                let gravity_effect = strength * total_effect * dt;
                // Apply gravitational force
                Ok(gravity_effect)
            }
            FieldType::ResourceField { resource_type, density, regeneration_rate } => {
                // Resource fields provide collectible resources
                let resource_available = density * total_effect * dt;
                Ok(resource_available)
            }
        }
    }
    
    /// Check for obstacle collisions and handle them
    pub fn handle_obstacle_collision(&mut self, obstacle: &Obstacle, dt: f64) -> Result<f64> {
        let distance = (self.position - obstacle.position).norm();
        let collision_threshold = obstacle.size.norm() * 0.5;
        
        if distance > collision_threshold {
            return Ok(0.0);
        }
        
        match &obstacle.obstacle_type {
            ObstacleType::Wall { friction } => {
                // Wall blocks movement and applies friction
                let friction_effect = friction * dt;
                self.velocity *= (1.0 - friction_effect).max(0.0);
                Ok(-friction_effect)
            }
            ObstacleType::EnergyBarrier { damage_per_second } => {
                // Energy barrier damages the agent
                let damage = damage_per_second * dt;
                self.energy = (self.energy - damage).max(0.0);
                Ok(-damage)
            }
            ObstacleType::ForceField { strength, attraction } => {
                // Force field repels or attracts the agent
                let direction = if *attraction {
                    (obstacle.position - self.position).normalize()
                } else {
                    (self.position - obstacle.position).normalize()
                };
                let force_effect = strength * dt;
                self.velocity += direction * force_effect;
                Ok(force_effect)
            }
            ObstacleType::ResourceDeposit { resource_type, capacity } => {
                // Resource deposit can be collected
                let collection_rate = 1.0 * dt;
                let collected = collection_rate.min(*capacity);
                Ok(collected)
            }
        }
    }
    
    /// Attempt to collect a resource
    pub fn collect_resource(&mut self, resource: &mut Resource, dt: f64) -> Result<bool> {
        let distance = (self.position - resource.position).norm();
        let collection_range = 1e-8; // 10 nanometers
        
        if distance > collection_range || resource.is_collected {
            return Ok(false);
        }
        
        // Successfully collect the resource
        resource.is_collected = true;
        resource.collection_time = Some(self.age);
        resource.collected_by = Some(self.id);
        
        // Add resource value to agent
        self.energy += resource.energy_content;
        self.discovery_count += 1;
        
        Ok(true)
    }
    
    /// Interact with another agent (competition, cooperation, etc.)
    pub fn interact_with_agent(&mut self, other_agent: &mut EmbodiedAgent, interaction_type: &str) -> Result<f64> {
        let distance = (self.position - other_agent.position).norm();
        let interaction_range = 1e-7; // 100 nanometers
        
        if distance > interaction_range {
            return Ok(0.0);
        }
        
        match interaction_type {
            "compete" => {
                // Simple competition based on energy levels
                let my_strength = self.energy * self.mass;
                let other_strength = other_agent.energy * other_agent.mass;
                
                if my_strength > other_strength {
                    // Win competition
                    let energy_transfer = other_agent.energy * 0.1;
                    self.energy += energy_transfer;
                    other_agent.energy -= energy_transfer;
                    Ok(energy_transfer)
                } else {
                    // Lose competition
                    let energy_loss = self.energy * 0.1;
                    self.energy -= energy_loss;
                    other_agent.energy += energy_loss;
                    Ok(-energy_loss)
                }
            }
            "cooperate" => {
                // Cooperation benefits both agents
                let cooperation_bonus = (self.energy + other_agent.energy) * 0.05;
                self.energy += cooperation_bonus;
                other_agent.energy += cooperation_bonus;
                Ok(cooperation_bonus)
            }
            "avoid" => {
                // Avoid collision by moving away
                let avoidance_force = 1e-15;
                let direction = (self.position - other_agent.position).normalize();
                self.velocity += direction * avoidance_force;
                Ok(0.0)
            }
            _ => Ok(0.0)
        }
    }
    
    /// Navigate through an energy gradient
    pub fn navigate_gradient(&mut self, gradient: &EnergyGradient, dt: f64) -> Result<f64> {
        let energy_cost = self.calculate_gradient_energy_cost(gradient);
        
        if self.energy < energy_cost * dt {
            // Not enough energy to navigate
            return Ok(-energy_cost * dt);
        }
        
        // Pay energy cost for navigation
        self.energy -= energy_cost * dt;
        
        // Move towards lower energy cost direction
        let gradient_direction = (gradient.end_position - gradient.start_position).normalize();
        let movement = gradient_direction * 1e-8 * dt; // Small movement
        self.position += movement;
        
        Ok(-energy_cost * dt)
    }
    
    /// Update agent's environmental awareness and adapt behavior
    pub fn update_environmental_awareness(&mut self, 
        fields: &[EnvironmentalFeature], 
        obstacles: &[Obstacle], 
        gradients: &[EnergyGradient],
        resources: &[Resource],
        other_agents: &[EmbodiedAgent],
        dt: f64
    ) -> Result<Vec<InteractionEvent>> {
        let mut events = Vec::new();
        
        // Check field effects
        for field in fields {
            if field.is_active {
                let effect = self.apply_field_effects(field, dt)?;
                if effect != 0.0 {
                    events.push(InteractionEvent {
                        timestamp: self.age,
                        event_type: InteractionEventType::InteractWithField {
                            field_id: field.id,
                            energy_gain: effect,
                        },
                        target_id: Some(field.id),
                        outcome: effect,
                        energy_cost: 0.0,
                    });
                }
            }
        }
        
        // Check obstacle collisions
        for obstacle in obstacles {
            let effect = self.handle_obstacle_collision(obstacle, dt)?;
            if effect != 0.0 {
                events.push(InteractionEvent {
                    timestamp: self.age,
                    event_type: InteractionEventType::AvoidObstacle {
                        obstacle_id: obstacle.id,
                        damage_taken: effect.abs(),
                    },
                    target_id: Some(obstacle.id),
                    outcome: effect,
                    energy_cost: effect.abs(),
                });
            }
        }
        
        // Check gradient navigation
        for gradient in gradients {
            let cost = self.navigate_gradient(gradient, dt)?;
            if cost != 0.0 {
                events.push(InteractionEvent {
                    timestamp: self.age,
                    event_type: InteractionEventType::NavigateGradient {
                        gradient_id: gradient.id,
                        energy_cost: cost.abs(),
                    },
                    target_id: Some(gradient.id),
                    outcome: -cost,
                    energy_cost: cost.abs(),
                });
            }
        }
        
        // Check resource collection
        for resource in resources {
            if let Ok(collected) = self.collect_resource(&mut resource.clone(), dt) {
                if collected {
                    events.push(InteractionEvent {
                        timestamp: self.age,
                        event_type: InteractionEventType::CollectResource {
                            resource_type: resource.resource_type.clone(),
                            value: resource.value,
                        },
                        target_id: Some(resource.id),
                        outcome: resource.value,
                        energy_cost: 0.0,
                    });
                }
            }
        }
        
        Ok(events)
    }
}

impl DistributedNeuralCore {
    pub fn new() -> Self {
        Self {
            processing_particles: Vec::new(),
            memory_particles: Vec::new(),
            communication_particles: Vec::new(),
            architecture: NeuralArchitecture::new(),
            learning_parameters: LearningParameters::default(),
            quantum_states: Vec::new(),
        }
    }
}

impl NeuralArchitecture {
    pub fn new() -> Self {
        Self {
            input_particles: Vec::new(),
            hidden_particles: vec![Vec::new()], // One hidden layer
            output_particles: Vec::new(),
            connection_strengths: Vec::new(),
            activation_functions: vec![ActivationType::Linear],
        }
    }
}

impl Default for LearningParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            quantum_coherence_time: 1e-9,
            entanglement_threshold: 0.1,
        }
    }
}

impl DistributedMetaLearner {
    pub fn new() -> Self {
        Self {
            meta_particles: Vec::new(),
            parameter_particles: Vec::new(),
            adaptation_rate: 0.001,
            meta_parameters: Vec::new(),
        }
    }
}

impl DistributedCuriositySystem {
    pub fn new() -> Self {
        Self {
            sensory_particles: Vec::new(),
            novelty_detector_particles: Vec::new(),
            exploration_particles: Vec::new(),
            prediction_particles: Vec::new(),
            curiosity_level: 0.5,
            exploration_rate: 0.1,
        }
    }
}

impl DistributedSelfModification {
    pub fn new() -> Self {
        Self {
            modification_particles: Vec::new(),
            analysis_particles: Vec::new(),
            validation_particles: Vec::new(),
            modification_rate: 0.01,
            safety_threshold: 0.8,
        }
    }
}

impl DistributedOpenEndedEvolution {
    pub fn new() -> Self {
        Self {
            evolution_particles: Vec::new(),
            archive_particles: Vec::new(),
            diversity_particles: Vec::new(),
            emergence_particles: Vec::new(),
            novelty_threshold: 0.3,
            diversity_pressure: 0.5,
        }
    }
}

/// Interface for physics engine interaction
pub trait PhysicsEngineInterface {
    fn get_particle_count(&self) -> usize;
    fn get_particle_data(&self, index: usize) -> Option<ParticleData>;
    fn apply_force_to_particle(&mut self, index: usize, force: Vector3<f64>) -> Result<()>;
    fn transfer_energy(&mut self, from: usize, to: usize, amount: f64) -> Result<()>;
    fn create_particle(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64, charge: f64) -> Result<usize>;
}

/// Particle data for physics engine interface
#[derive(Debug, Clone)]
pub struct ParticleData {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub energy: f64,
    pub mass: f64,
    pub electric_charge: f64,
}

/// Agent statistics for monitoring
#[derive(Debug, Clone)]
pub struct AgentStatistics {
    pub id: Uuid,
    pub name: String,
    pub generation: u64,
    pub age: f64,
    pub particle_count: usize,
    pub energy: f64,
    pub mass: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub discovery_count: usize,
    pub average_performance: f64,
    pub recent_performance: f64,
}
