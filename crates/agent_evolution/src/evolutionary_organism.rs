//! # Evolutionary Organism System
//!
//! This module implements true evolutionary AI organisms that learn survival strategies 
//! from scratch through self-modification, adaptation, and natural selection.
//! Unlike the demo systems, these organisms exhibit real learning, decision-making, 
//! and self-improvement capabilities.

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use rand::{thread_rng, Rng};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::ai_core::{AICore, SensoryInput, ActionType, AgentSensoryData};
use crate::natural_selection::{Fitness, FitnessLandscape};
use crate::autonomous_communication::{AutonomousCommunication, CommunicationEnvironment, SocialContext};
use crate::{PlasticityInput, PlasticityOutput};

/// A true evolutionary organism with self-modifying neural architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryOrganism {
    pub id: Uuid,
    pub generation: u64,
    pub age: u64,
    pub energy: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub neural_core: AICore,
    pub survival_instincts: SurvivalInstincts,
    pub learning_system: LearningSystem,
    pub fitness_tracker: OrganismFitness,
    pub genetic_memory: GeneticMemory,
    pub metabolism: Metabolism,
    pub sensory_organs: SensoryOrgans,
    pub motor_system: MotorSystem,
    pub reproduction_system: ReproductionSystem,
    pub communication_system: AutonomousCommunication,
}

/// Survival instincts that evolve over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalInstincts {
    pub hunger_threshold: f64,
    pub danger_sensitivity: f64,
    pub exploration_drive: f64,
    pub social_tendency: f64,
    pub risk_tolerance: f64,
    pub learning_motivation: f64,
    pub cooperation_bias: f64,
    pub competition_aggression: f64,
}

/// Adaptive learning system that modifies behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSystem {
    pub learning_rate: f64,
    pub memory_consolidation_rate: f64,
    pub pattern_recognition_strength: f64,
    pub adaptation_speed: f64,
    pub experience_buffer: Vec<LearningExperience>,
    pub behavioral_patterns: HashMap<String, BehavioralPattern>,
    pub success_strategies: Vec<Strategy>,
    pub failure_avoidance: Vec<Strategy>,
}

/// Individual learning experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningExperience {
    pub timestamp: u64,
    pub context: Vec<f64>,
    pub action_taken: ActionType,
    pub immediate_outcome: f64,
    pub long_term_impact: f64,
    pub energy_change: f64,
    pub survival_relevance: f64,
}

/// Behavioral pattern learned from experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    pub trigger_conditions: Vec<f64>,
    pub action_sequence: Vec<ActionType>,
    pub success_rate: f64,
    pub confidence: f64,
    pub last_used: u64,
    pub total_uses: u64,
}

/// Survival strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub description: String,
    pub conditions: Vec<f64>,
    pub actions: Vec<ActionType>,
    pub effectiveness: f64,
    pub energy_efficiency: f64,
}

/// Comprehensive fitness tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismFitness {
    pub survival_time: u64,
    pub energy_efficiency: f64,
    pub reproduction_success: f64,
    pub learning_progress: f64,
    pub social_success: f64,
    pub innovation_score: f64,
    pub adaptation_rate: f64,
    pub overall_fitness: f64,
    pub fitness_history: Vec<f64>,
}

/// Genetic memory that passes knowledge to offspring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticMemory {
    pub ancestral_knowledge: Vec<f64>,
    pub successful_strategies: Vec<Strategy>,
    pub environmental_adaptations: HashMap<String, f64>,
    pub social_behaviors: Vec<BehavioralPattern>,
    pub innovation_history: Vec<Innovation>,
}

/// Metabolic system for energy management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metabolism {
    pub base_metabolic_rate: f64,
    pub efficiency: f64,
    pub energy_storage_capacity: f64,
    pub current_energy: f64,
    pub hunger_level: f64,
    pub fatigue_level: f64,
}

/// Advanced sensory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryOrgans {
    pub vision_range: f64,
    pub vision_acuity: f64,
    pub hearing_sensitivity: f64,
    pub chemical_detection: f64,
    pub spatial_awareness: f64,
    pub temporal_perception: f64,
    pub social_perception: f64,
}

/// Motor and movement system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorSystem {
    pub movement_speed: f64,
    pub agility: f64,
    pub coordination: f64,
    pub strength: f64,
    pub endurance: f64,
    pub precision: f64,
}

/// Reproduction and genetic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproductionSystem {
    pub sexual_maturity: bool,
    pub mating_success: f64,
    pub offspring_quality: f64,
    pub parental_investment: f64,
    pub genetic_diversity: f64,
}

/// Innovation or creative solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Innovation {
    pub description: String,
    pub discovery_time: u64,
    pub effectiveness: f64,
    pub complexity: f64,
    pub transferability: f64,
}

impl EvolutionaryOrganism {
    /// Create a new organism with random initial traits
    pub fn new() -> Self {
        let mut rng = thread_rng();
        let id = Uuid::new_v4();
        
        Self {
            id,
            generation: 0,
            age: 0,
            energy: 100.0,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            neural_core: AICore::new(),
            survival_instincts: SurvivalInstincts {
                hunger_threshold: rng.gen_range(0.2..0.8),
                danger_sensitivity: rng.gen_range(0.3..0.9),
                exploration_drive: rng.gen_range(0.1..0.7),
                social_tendency: rng.gen_range(0.0..1.0),
                risk_tolerance: rng.gen_range(0.1..0.6),
                learning_motivation: rng.gen_range(0.5..1.0),
                cooperation_bias: rng.gen_range(0.0..0.8),
                competition_aggression: rng.gen_range(0.1..0.6),
            },
            learning_system: LearningSystem {
                learning_rate: rng.gen_range(0.01..0.1),
                memory_consolidation_rate: rng.gen_range(0.01..0.05),
                pattern_recognition_strength: rng.gen_range(0.3..0.8),
                adaptation_speed: rng.gen_range(0.1..0.5),
                experience_buffer: Vec::new(),
                behavioral_patterns: HashMap::new(),
                success_strategies: Vec::new(),
                failure_avoidance: Vec::new(),
            },
            fitness_tracker: OrganismFitness {
                survival_time: 0,
                energy_efficiency: 1.0,
                reproduction_success: 0.0,
                learning_progress: 0.0,
                social_success: 0.0,
                innovation_score: 0.0,
                adaptation_rate: 0.0,
                overall_fitness: 0.0,
                fitness_history: Vec::new(),
            },
            genetic_memory: GeneticMemory {
                ancestral_knowledge: vec![0.0; 20],
                successful_strategies: Vec::new(),
                environmental_adaptations: HashMap::new(),
                social_behaviors: Vec::new(),
                innovation_history: Vec::new(),
            },
            metabolism: Metabolism {
                base_metabolic_rate: rng.gen_range(0.1..0.3),
                efficiency: rng.gen_range(0.7..1.0),
                energy_storage_capacity: rng.gen_range(80.0..120.0),
                current_energy: 100.0,
                hunger_level: 0.0,
                fatigue_level: 0.0,
            },
            sensory_organs: SensoryOrgans {
                vision_range: rng.gen_range(50.0..200.0),
                vision_acuity: rng.gen_range(0.5..1.0),
                hearing_sensitivity: rng.gen_range(0.3..0.9),
                chemical_detection: rng.gen_range(0.2..0.8),
                spatial_awareness: rng.gen_range(0.4..0.9),
                temporal_perception: rng.gen_range(0.3..0.8),
                social_perception: rng.gen_range(0.2..0.9),
            },
            motor_system: MotorSystem {
                movement_speed: rng.gen_range(1.0..3.0),
                agility: rng.gen_range(0.5..1.0),
                coordination: rng.gen_range(0.6..1.0),
                strength: rng.gen_range(0.7..1.3),
                endurance: rng.gen_range(0.8..1.2),
                precision: rng.gen_range(0.4..0.9),
            },
            reproduction_system: ReproductionSystem {
                sexual_maturity: false,
                mating_success: 0.0,
                offspring_quality: 0.0,
                parental_investment: rng.gen_range(0.3..0.8),
                genetic_diversity: rng.gen_range(0.5..1.0),
            },
            communication_system: AutonomousCommunication::new(id),
        }
    }
    
    /// Create offspring from two parent organisms
    pub fn reproduce(&self, other: &EvolutionaryOrganism) -> EvolutionaryOrganism {
        let mut offspring = EvolutionaryOrganism::new();
        
        offspring.id = Uuid::new_v4();
        offspring.generation = (self.generation + other.generation) / 2 + 1;
        
        // Inherit neural architecture through crossover
        offspring.neural_core = self.neural_core.reproduce(&other.neural_core);
        
        // Combine and mutate survival instincts
        offspring.survival_instincts = SurvivalInstincts {
            hunger_threshold: self.crossover_trait(self.survival_instincts.hunger_threshold, other.survival_instincts.hunger_threshold, 0.1),
            danger_sensitivity: self.crossover_trait(self.survival_instincts.danger_sensitivity, other.survival_instincts.danger_sensitivity, 0.1),
            exploration_drive: self.crossover_trait(self.survival_instincts.exploration_drive, other.survival_instincts.exploration_drive, 0.1),
            social_tendency: self.crossover_trait(self.survival_instincts.social_tendency, other.survival_instincts.social_tendency, 0.1),
            risk_tolerance: self.crossover_trait(self.survival_instincts.risk_tolerance, other.survival_instincts.risk_tolerance, 0.1),
            learning_motivation: self.crossover_trait(self.survival_instincts.learning_motivation, other.survival_instincts.learning_motivation, 0.05),
            cooperation_bias: self.crossover_trait(self.survival_instincts.cooperation_bias, other.survival_instincts.cooperation_bias, 0.1),
            competition_aggression: self.crossover_trait(self.survival_instincts.competition_aggression, other.survival_instincts.competition_aggression, 0.1),
        };
        
        // Inherit genetic memory
        offspring.genetic_memory.ancestral_knowledge = self.genetic_memory.ancestral_knowledge.iter()
            .zip(&other.genetic_memory.ancestral_knowledge)
            .map(|(a, b)| self.crossover_trait(*a, *b, 0.05))
            .collect();
        
        // Combine successful strategies
        offspring.genetic_memory.successful_strategies.extend(self.genetic_memory.successful_strategies.clone());
        offspring.genetic_memory.successful_strategies.extend(other.genetic_memory.successful_strategies.clone());
        
        // Inherit and mutate physical systems
        offspring.metabolism.base_metabolic_rate = self.crossover_trait(self.metabolism.base_metabolic_rate, other.metabolism.base_metabolic_rate, 0.02);
        offspring.metabolism.efficiency = self.crossover_trait(self.metabolism.efficiency, other.metabolism.efficiency, 0.05);
        
        offspring.sensory_organs.vision_range = self.crossover_trait(self.sensory_organs.vision_range, other.sensory_organs.vision_range, 10.0);
        offspring.sensory_organs.vision_acuity = self.crossover_trait(self.sensory_organs.vision_acuity, other.sensory_organs.vision_acuity, 0.1);
        
        offspring.motor_system.movement_speed = self.crossover_trait(self.motor_system.movement_speed, other.motor_system.movement_speed, 0.2);
        offspring.motor_system.agility = self.crossover_trait(self.motor_system.agility, other.motor_system.agility, 0.1);
        
        // NEW: Inherit communication system with crossover
        offspring.communication_system = self.reproduce_communication_system(&other.communication_system);
        
        offspring
    }
    
    /// Reproduce communication system with crossover and mutation
    fn reproduce_communication_system(&self, other: &AutonomousCommunication) -> AutonomousCommunication {
        let mut offspring_comm = AutonomousCommunication::new(self.id);
        
        // Crossover vocal cords
        offspring_comm.communication_organs.vocal_cords.complexity = 
            self.crossover_trait(self.communication_system.communication_organs.vocal_cords.complexity, 
                               other.communication_organs.vocal_cords.complexity, 0.05);
        
        // Crossover visual signals
        offspring_comm.communication_organs.visual_signals.pattern_complexity = 
            self.crossover_trait(self.communication_system.communication_organs.visual_signals.pattern_complexity,
                               other.communication_organs.visual_signals.pattern_complexity, 0.05);
        
        // Crossover chemical signals
        offspring_comm.communication_organs.chemical_signals.pheromone_production = 
            self.crossover_trait(self.communication_system.communication_organs.chemical_signals.pheromone_production,
                               other.communication_organs.chemical_signals.pheromone_production, 0.05);
        
        // Inherit vocabulary (combine both parents' vocabularies)
        offspring_comm.language_evolution.vocabulary.extend(self.communication_system.language_evolution.vocabulary.clone());
        offspring_comm.language_evolution.vocabulary.extend(other.language_evolution.vocabulary.clone());
        
        // Inherit grammar rules
        offspring_comm.language_evolution.grammar_rules.extend(self.communication_system.language_evolution.grammar_rules.clone());
        offspring_comm.language_evolution.grammar_rules.extend(other.language_evolution.grammar_rules.clone());
        
        offspring_comm
    }
    
    /// Crossover trait with mutation
    fn crossover_trait(&self, trait_a: f64, trait_b: f64, mutation_strength: f64) -> f64 {
        let mut rng = thread_rng();
        let base_value = if rng.gen::<f64>() < 0.5 { trait_a } else { trait_b };
        let mutation = rng.gen_range(-mutation_strength..mutation_strength);
        (base_value + mutation).max(0.0).min(1.0)
    }
    
    /// Update organism for one simulation step
    pub fn update(&mut self, environment: &Environment, nearby_organisms: &[&EvolutionaryOrganism], delta_time: f64) -> Result<ActionType> {
        self.age += 1;
        
        // Update metabolism
        self.update_metabolism(delta_time);
        
        // NEW: Evolve communication system
        self.evolve_communication(environment, nearby_organisms)?;
        
        // Process sensory input
        let sensory_input = self.perceive_environment(environment, nearby_organisms)?;
        
        // Make decision using neural network
        let action = self.neural_core.make_decision(&sensory_input, self.age)?;
        
        // Execute action and learn from outcome
        let outcome = self.execute_action(&action, environment);
        self.learn_from_experience(action.clone(), outcome, sensory_input.to_vector().as_slice().to_vec());
        
        // Update fitness
        self.update_fitness();
        
        // Check for sexual maturity
        if self.age > 1000 && !self.reproduction_system.sexual_maturity {
            self.reproduction_system.sexual_maturity = true;
        }
        
        Ok(action)
    }
    
    /// Update metabolic state
    fn update_metabolism(&mut self, delta_time: f64) {
        let energy_cost = self.metabolism.base_metabolic_rate * delta_time;
        self.metabolism.current_energy -= energy_cost;
        self.energy = self.metabolism.current_energy;
        
        // Update hunger and fatigue
        self.metabolism.hunger_level = (100.0 - self.energy) / 100.0;
        self.metabolism.fatigue_level += 0.001 * delta_time;
    }
    
    /// Perceive the environment through sensory organs
    fn perceive_environment(&self, environment: &Environment, nearby_organisms: &[&EvolutionaryOrganism]) -> Result<SensoryInput> {
        let mut nearby_agents = Vec::new();
        
        for organism in nearby_organisms.iter().take(5) {
            let distance = self.distance_to(organism);
            if distance <= self.sensory_organs.vision_range {
                nearby_agents.push(AgentSensoryData {
                    distance,
                    energy_level: organism.energy / 100.0,
                    threat_level: organism.survival_instincts.competition_aggression,
                });
            }
        }
        
        let memory_state = self.genetic_memory.ancestral_knowledge.clone();
        
        Ok(SensoryInput::from_environment(
            self.position,
            self.energy,
            environment.temperature,
            environment.resource_density,
            &nearby_agents,
            &memory_state,
        ))
    }
    
    /// Calculate distance to another organism
    fn distance_to(&self, other: &EvolutionaryOrganism) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Execute an action and return the outcome
    fn execute_action(&mut self, action: &ActionType, environment: &Environment) -> f64 {
        let mut outcome = 0.0;
        let energy_cost = action.energy_cost();
        
        self.energy -= energy_cost;
        
        match action {
            ActionType::ExtractEnergy => {
                if environment.resource_density > 0.3 {
                    let gained_energy = environment.resource_density * 10.0 * self.metabolism.efficiency;
                    self.energy += gained_energy;
                    outcome = gained_energy;
                }
            },
            ActionType::MoveForward => {
                self.position[0] += self.motor_system.movement_speed;
                outcome = self.motor_system.movement_speed * 0.1;
            },
            ActionType::Rest => {
                self.metabolism.fatigue_level *= 0.9;
                outcome = 1.0;
            },
            ActionType::Learn => {
                self.learning_system.learning_rate *= 1.01;
                outcome = 0.5;
            },
            ActionType::Observe => {
                self.sensory_organs.vision_acuity *= 1.001;
                outcome = 0.3;
            },
            _ => {
                outcome = -0.1; // Small penalty for unknown actions
            }
        }
        
        // Survival penalty for low energy
        if self.energy < 20.0 {
            outcome -= 2.0;
        }
        
        outcome
    }
    
    /// Learn from experience
    fn learn_from_experience(&mut self, action: ActionType, outcome: f64, context: Vec<f64>) {
        let experience = LearningExperience {
            timestamp: self.age,
            context: context.clone(),
            action_taken: action.clone(),
            immediate_outcome: outcome,
            long_term_impact: 0.0, // Will be calculated later
            energy_change: outcome,
            survival_relevance: if outcome > 0.0 { 1.0 } else { -1.0 },
        };
        
        self.learning_system.experience_buffer.push(experience);
        
        // Learn from outcome using neural network
        self.neural_core.learn_from_outcome(action, outcome, context, self.age);
        
        // Update behavioral patterns
        self.update_behavioral_patterns(outcome);
        
        // Consolidate learning periodically
        if self.learning_system.experience_buffer.len() > 100 {
            self.consolidate_learning();
        }
    }
    
    /// Update behavioral patterns based on recent experiences
    fn update_behavioral_patterns(&mut self, outcome: f64) {
        // This is a simplified pattern learning system
        // In reality, this would involve more sophisticated pattern recognition
        
        if outcome > 1.0 {
            // Successful behavior - reinforce
            let pattern_key = format!("success_{}", self.learning_system.behavioral_patterns.len());
            
            if !self.learning_system.behavioral_patterns.contains_key(&pattern_key) {
                let pattern = BehavioralPattern {
                    trigger_conditions: vec![self.energy / 100.0, self.metabolism.hunger_level],
                    action_sequence: vec![ActionType::ExtractEnergy],
                    success_rate: 1.0,
                    confidence: 0.5,
                    last_used: self.age,
                    total_uses: 1,
                };
                self.learning_system.behavioral_patterns.insert(pattern_key, pattern);
            }
        }
    }
    
    /// Consolidate learning experiences into long-term knowledge
    fn consolidate_learning(&mut self) {
        // Calculate long-term impacts
        for i in 0..self.learning_system.experience_buffer.len() {
            let mut long_term_impact = 0.0;
            
            // Look at subsequent experiences to calculate long-term impact
            for j in (i + 1)..self.learning_system.experience_buffer.len().min(i + 10) {
                long_term_impact += self.learning_system.experience_buffer[j].immediate_outcome * 0.9_f64.powi((j - i) as i32);
            }
            
            self.learning_system.experience_buffer[i].long_term_impact = long_term_impact;
        }
        
        // Extract successful strategies
        let successful_experiences: Vec<_> = self.learning_system.experience_buffer.iter()
            .filter(|exp| exp.long_term_impact > 1.0)
            .collect();
        
        for exp in successful_experiences {
            let strategy = Strategy {
                description: format!("Strategy learned at age {}", exp.timestamp),
                conditions: exp.context.clone(),
                actions: vec![exp.action_taken.clone()],
                effectiveness: exp.long_term_impact,
                energy_efficiency: exp.energy_change / exp.action_taken.energy_cost().max(0.1),
            };
            
            self.learning_system.success_strategies.push(strategy);
        }
        
        // Keep only recent experiences
        if self.learning_system.experience_buffer.len() > 50 {
            self.learning_system.experience_buffer.drain(0..50);
        }
        
        // Update genetic memory
        self.update_genetic_memory();
    }
    
    /// Update genetic memory with learned knowledge
    fn update_genetic_memory(&mut self) {
        // Transfer successful strategies to genetic memory
        for strategy in &self.learning_system.success_strategies {
            if strategy.effectiveness > 2.0 {
                self.genetic_memory.successful_strategies.push(strategy.clone());
            }
        }
        
        // Update ancestral knowledge based on learning progress
        for i in 0..self.genetic_memory.ancestral_knowledge.len() {
            let learning_influence = self.learning_system.learning_rate * 0.01;
            self.genetic_memory.ancestral_knowledge[i] += learning_influence;
        }
    }
    
    /// Update fitness metrics
    fn update_fitness(&mut self) {
        self.fitness_tracker.survival_time = self.age;
        self.fitness_tracker.energy_efficiency = if self.age > 0 { self.energy / self.age as f64 } else { 0.0 };
        self.fitness_tracker.learning_progress = self.learning_system.success_strategies.len() as f64;
        self.fitness_tracker.adaptation_rate = self.learning_system.adaptation_speed;
        
        // NEW: Include communication fitness
        let communication_fitness = self.communication_system.communication_fitness.overall_fitness;
        self.fitness_tracker.social_success = communication_fitness;
        
        // Calculate overall fitness
        self.fitness_tracker.overall_fitness = 
            self.fitness_tracker.survival_time as f64 * 0.25 +
            self.fitness_tracker.energy_efficiency * 8.0 +
            self.fitness_tracker.learning_progress * 4.0 +
            self.fitness_tracker.reproduction_success * 15.0 +
            communication_fitness * 10.0; // NEW: Communication contributes to fitness
        
        self.fitness_tracker.fitness_history.push(self.fitness_tracker.overall_fitness);
        
        // Keep fitness history manageable
        if self.fitness_tracker.fitness_history.len() > 1000 {
            self.fitness_tracker.fitness_history.remove(0);
        }
    }
    
    /// Get current fitness score
    pub fn get_fitness(&self) -> f64 {
        self.fitness_tracker.overall_fitness
    }
    
    /// Check if organism is ready to reproduce
    pub fn can_reproduce(&self) -> bool {
        self.reproduction_system.sexual_maturity && 
        self.energy > 50.0 && 
        self.age > 500
    }
    
    /// Check if organism is dead
    pub fn is_dead(&self) -> bool {
        self.energy <= 0.0 || self.age > 10000
    }
    
    /// NEW: Evolve communication system based on environment and social context
    fn evolve_communication(&mut self, environment: &Environment, nearby_organisms: &[&EvolutionaryOrganism]) -> Result<()> {
        // Create communication environment
        let comm_environment = CommunicationEnvironment {
            medium: "air".to_string(), // Could be determined by actual environment
            visibility: 0.8,
            noise_level: 0.3,
            danger_level: environment.danger_level,
            technology_level: 0.1, // Starts low, can evolve
            complexity: environment.complexity,
        };
        
        // Create social context
        let social_context = SocialContext {
            complexity: nearby_organisms.len() as f64 * 0.1,
            cooperation_need: self.survival_instincts.cooperation_bias,
            competition_level: self.survival_instincts.competition_aggression,
            group_size: nearby_organisms.len(),
            hierarchy_present: false, // Could evolve
        };
        
        // Evolve communication system
        self.communication_system.evolve(&comm_environment, &social_context)?;
        
        Ok(())
    }
    
    /// NEW: Get communication capabilities
    pub fn get_communication_capabilities(&self) -> crate::autonomous_communication::CommunicationCapabilities {
        self.communication_system.get_capabilities()
    }
    
    /// NEW: Get evolved language vocabulary
    pub fn get_vocabulary(&self) -> &HashMap<String, crate::autonomous_communication::Concept> {
        &self.communication_system.language_evolution.vocabulary
    }
    
    /// NEW: Get evolved signal types
    pub fn get_signal_types(&self) -> Vec<crate::autonomous_communication::SignalType> {
        self.communication_system.signal_system.signal_types.keys().cloned().collect()
    }
}

/// Environment in which organisms live and evolve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub temperature: f64,
    pub resource_density: f64,
    pub danger_level: f64,
    pub social_pressure: f64,
    pub complexity: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            temperature: 20.0,
            resource_density: 0.5,
            danger_level: 0.3,
            social_pressure: 0.4,
            complexity: 0.5,
        }
    }
}

/// Population of evolutionary organisms
#[derive(Debug)]
pub struct EvolutionaryPopulation {
    pub organisms: Vec<EvolutionaryOrganism>,
    pub environment: Environment,
    pub generation: u64,
    pub population_size: usize,
    pub mutation_rate: f64,
    pub selection_pressure: f64,
    pub fitness_landscape: FitnessLandscape,
}

impl EvolutionaryPopulation {
    /// Create a new population
    pub fn new(size: usize) -> Self {
        let mut organisms = Vec::new();
        for _ in 0..size {
            organisms.push(EvolutionaryOrganism::new());
        }
        
        Self {
            organisms,
            environment: Environment::default(),
            generation: 0,
            population_size: size,
            mutation_rate: 0.1,
            selection_pressure: 0.3,
            fitness_landscape: FitnessLandscape::new(),
        }
    }
    
    /// Update entire population for one time step
    pub fn update(&mut self, delta_time: f64) -> Result<()> {
        // Update each organism
        for i in 0..self.organisms.len() {
            if !self.organisms[i].is_dead() {
                // Create nearby organisms list without borrowing conflicts
                let nearby_organisms: Vec<EvolutionaryOrganism> = self.organisms.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, org)| org.clone())
                    .collect();
                
                let nearby_refs: Vec<&EvolutionaryOrganism> = nearby_organisms.iter().collect();
                
                self.organisms[i].update(&self.environment, &nearby_refs, delta_time)?;
                
                // Update fitness landscape
                self.fitness_landscape.update_fitness(
                    self.organisms[i].id, 
                    self.organisms[i].get_fitness()
                );
            }
        }
        
        // Remove dead organisms
        self.organisms.retain(|org| !org.is_dead());
        
        // Check for reproduction
        self.handle_reproduction()?;
        
        // Evolution step if population gets too small
        if self.organisms.len() < self.population_size / 2 {
            self.evolve_population()?;
        }
        
        Ok(())
    }
    
    /// Handle reproduction between compatible organisms
    fn handle_reproduction(&mut self) -> Result<()> {
        let mut new_offspring = Vec::new();
        let mut rng = thread_rng();
        
        // Find organisms ready to reproduce
        let ready_to_reproduce: Vec<usize> = self.organisms.iter()
            .enumerate()
            .filter(|(_, org)| org.can_reproduce())
            .map(|(i, _)| i)
            .collect();
        
        // Randomly pair organisms for reproduction
        for chunk in ready_to_reproduce.chunks(2) {
            if chunk.len() == 2 {
                let parent1 = &self.organisms[chunk[0]];
                let parent2 = &self.organisms[chunk[1]];
                
                // Check compatibility (simplified)
                let compatibility = 1.0 - (parent1.generation as f64 - parent2.generation as f64).abs() / 10.0;
                
                if rng.gen::<f64>() < compatibility * 0.3 {
                    let offspring = parent1.reproduce(parent2);
                    new_offspring.push(offspring);
                }
            }
        }
        
        // Add offspring to population
        self.organisms.extend(new_offspring);
        
        Ok(())
    }
    
    /// Evolve the population through selection and mutation
    fn evolve_population(&mut self) -> Result<()> {
        // Sort by fitness
        self.organisms.sort_by(|a, b| {
            b.get_fitness().partial_cmp(&a.get_fitness()).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Keep top performers
        let survivors = (self.organisms.len() as f64 * (1.0 - self.selection_pressure)) as usize;
        let survivors = survivors.max(2).min(self.organisms.len());
        
        self.organisms.truncate(survivors);
        
        // Generate new organisms through mutation and crossover
        let mut new_generation = self.organisms.clone();
        
        while new_generation.len() < self.population_size {
            let parent1_idx = thread_rng().gen_range(0..self.organisms.len());
            let parent2_idx = thread_rng().gen_range(0..self.organisms.len());
            
            if parent1_idx != parent2_idx {
                let offspring = self.organisms[parent1_idx].reproduce(&self.organisms[parent2_idx]);
                new_generation.push(offspring);
            }
        }
        
        self.organisms = new_generation;
        self.generation += 1;
        
        Ok(())
    }
    
    /// Get population statistics
    pub fn get_statistics(&self) -> PopulationStatistics {
        let total_organisms = self.organisms.len();
        
        if total_organisms == 0 {
            return PopulationStatistics::default();
        }
        
        let avg_fitness = self.organisms.iter().map(|o| o.get_fitness()).sum::<f64>() / total_organisms as f64;
        let max_fitness = self.organisms.iter().map(|o| o.get_fitness()).fold(0.0, f64::max);
        let avg_age = self.organisms.iter().map(|o| o.age).sum::<u64>() / total_organisms as u64;
        let avg_energy = self.organisms.iter().map(|o| o.energy).sum::<f64>() / total_organisms as f64;
        
        let learning_organisms = self.organisms.iter()
            .filter(|o| !o.learning_system.success_strategies.is_empty())
            .count();
        
        let reproductive_organisms = self.organisms.iter()
            .filter(|o| o.can_reproduce())
            .count();
        
        PopulationStatistics {
            total_organisms,
            generation: self.generation,
            avg_fitness,
            max_fitness,
            avg_age,
            avg_energy,
            learning_organisms,
            reproductive_organisms,
            avg_learning_rate: self.organisms.iter().map(|o| o.learning_system.learning_rate).sum::<f64>() / total_organisms as f64,
        }
    }
}

/// Population statistics
#[derive(Debug, Clone, Default)]
pub struct PopulationStatistics {
    pub total_organisms: usize,
    pub generation: u64,
    pub avg_fitness: f64,
    pub max_fitness: f64,
    pub avg_age: u64,
    pub avg_energy: f64,
    pub learning_organisms: usize,
    pub reproductive_organisms: usize,
    pub avg_learning_rate: f64,
} 