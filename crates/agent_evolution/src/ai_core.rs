//! # Agent Evolution: AI Core
//!
//! This module defines the core intelligence of an agent, including its neural network,
//! sensory processing, and decision-making capabilities. The AI core is designed to be
//! modular and extensible, allowing for different neural architectures and learning algorithms.

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use crate::genetics::Genome;
use rand::{Rng, thread_rng};
use serde::{Serialize, Deserialize};

/// Represents the sensory inputs available to an agent.
/// This includes environmental data, internal state, and social information.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SensoryInput {
    pub vision: Vec<f64>,           // Visual field data
    pub audio: Vec<f64>,            // Auditory information
    pub internal_state: Vec<f64>,   // Energy, health, internal metrics
    pub social: Vec<f64>,           // Nearby agents, social signals
    pub environmental: Vec<f64>,    // Temperature, pressure, resource density
    pub memory: Vec<f64>,           // Recent experiences and learned patterns
}

impl SensoryInput {
    /// Creates a comprehensive sensory input from environment data
    pub fn from_environment(
        position: [f64; 3],
        energy: f64,
        temperature: f64,
        resource_density: f64,
        nearby_agents: &[AgentSensoryData],
        memory_state: &[f64],
    ) -> Self {
        let mut vision = Vec::new();
        let mut social = Vec::new();
        
        // Process visual field (simplified as spatial awareness)
        vision.extend_from_slice(&position);
        vision.push(resource_density);
        vision.push(temperature);
        
        // Process social information
        for agent_data in nearby_agents.iter().take(5) { // Limit to 5 nearest agents
            social.push(agent_data.distance);
            social.push(agent_data.energy_level);
            social.push(agent_data.threat_level);
        }
        
        // Pad social vector to fixed size
        while social.len() < 15 { // 5 agents * 3 attributes
            social.push(0.0);
        }
        
        // Internal state
        let internal_state = vec![
            energy,
            temperature,
            resource_density,
        ];
        
        // Environmental conditions
        let environmental = vec![
            temperature,
            resource_density,
            position[0] * 1e-12, // Normalize position
            position[1] * 1e-12,
            position[2] * 1e-12,
        ];
        
        Self {
            vision,
            audio: vec![0.0; 5], // Placeholder for audio
            internal_state,
            social,
            environmental,
            memory: memory_state.to_vec(),
        }
    }
    
    /// Flattens all sensory inputs into a single vector for the neural network.
    pub fn to_vector(&self) -> DVector<f64> {
        let mut combined = Vec::new();
        combined.extend(&self.vision);
        combined.extend(&self.audio);
        combined.extend(&self.internal_state);
        combined.extend(&self.social);
        combined.extend(&self.environmental);
        combined.extend(&self.memory);
        DVector::from_vec(combined)
    }
    
    /// Get the expected input size for neural network construction
    pub fn expected_size() -> usize {
        8 + 5 + 3 + 15 + 5 + 10 // vision + audio + internal + social + environmental + memory
    }
}

/// Sensory data about nearby agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSensoryData {
    pub distance: f64,
    pub energy_level: f64,
    pub threat_level: f64,
}

/// Neural network activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
}

impl ActivationFunction {
    /// Apply activation function to a vector
    pub fn apply(&self, input: &DVector<f64>) -> DVector<f64> {
        match self {
            ActivationFunction::Sigmoid => {
                input.map(|x| 1.0 / (1.0 + (-x).exp()))
            },
            ActivationFunction::Tanh => {
                input.map(|x| x.tanh())
            },
            ActivationFunction::ReLU => {
                input.map(|x| x.max(0.0))
            },
            ActivationFunction::Softmax => {
                let max_val = input.max();
                let exp_vals = input.map(|x| (x - max_val).exp());
                let sum = exp_vals.sum();
                exp_vals.map(|x| x / sum)
            },
        }
    }
}

/// A neural network layer with weights, biases, and activation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralLayer {
    pub weights: DMatrix<f64>,
    pub biases: DVector<f64>,
    pub activation: ActivationFunction,
}

impl NeuralLayer {
    /// Create a new neural layer with random weights
    pub fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let mut rng = thread_rng();
        
        // Initialize weights with Xavier/Glorot initialization
        let variance = 2.0 / (input_size + output_size) as f64;
        let std_dev = variance.sqrt();
        
        let weights = DMatrix::from_fn(output_size, input_size, |_, _| {
            rng.gen_range(-std_dev..std_dev)
        });
        
        let biases = DVector::zeros(output_size);
        
        Self {
            weights,
            biases,
            activation,
        }
    }
    
    /// Forward pass through the layer
    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let linear_output = &self.weights * input + &self.biases;
        self.activation.apply(&linear_output)
    }
    
    /// Mutate the layer's weights and biases
    pub fn mutate(&mut self, mutation_rate: f64, mutation_strength: f64) {
        let mut rng = thread_rng();
        
        // Mutate weights
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                if rng.gen::<f64>() < mutation_rate {
                    let delta = rng.gen_range(-mutation_strength..mutation_strength);
                    self.weights[(i, j)] += delta;
                }
            }
        }
        
        // Mutate biases
        for i in 0..self.biases.len() {
            if rng.gen::<f64>() < mutation_rate {
                let delta = rng.gen_range(-mutation_strength..mutation_strength);
                self.biases[i] += delta;
            }
        }
    }
}

/// A multi-layer feedforward neural network with evolutionary capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<NeuralLayer>,
    pub input_size: usize,
    pub output_size: usize,
    pub learning_rate: f64,
    pub mutation_rate: f64,
    pub mutation_strength: f64,
}

impl NeuralNetwork {
    /// Creates a new neural network with specified architecture
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_input_size = input_size;
        
        // Create hidden layers with ReLU activation
        for &hidden_size in hidden_sizes {
            layers.push(NeuralLayer::new(
                current_input_size,
                hidden_size,
                ActivationFunction::ReLU,
            ));
            current_input_size = hidden_size;
        }
        
        // Create output layer with softmax for decision probability distribution
        layers.push(NeuralLayer::new(
            current_input_size,
            output_size,
            ActivationFunction::Softmax,
        ));
        
        Self {
            layers,
            input_size,
            output_size,
            learning_rate,
            mutation_rate: 0.1,
            mutation_strength: 0.1,
        }
    }
    
    /// Forward pass through the entire network
    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        assert_eq!(input.len(), self.input_size, 
                  "Input vector size {} does not match network input size {}", 
                  input.len(), self.input_size);
        
        let mut current_input = input.clone();
        
        for layer in &self.layers {
            current_input = layer.forward(&current_input);
        }
        
        current_input
    }
    
    /// Evolve the network through mutation
    pub fn mutate(&mut self) {
        for layer in &mut self.layers {
            layer.mutate(self.mutation_rate, self.mutation_strength);
        }
    }
    
    /// Create offspring by combining two networks (crossover)
    pub fn crossover(&self, other: &NeuralNetwork) -> NeuralNetwork {
        assert_eq!(self.layers.len(), other.layers.len(), "Cannot crossover networks with different architectures");
        
        let mut offspring = self.clone();
        let mut rng = thread_rng();
        
        for (i, layer) in offspring.layers.iter_mut().enumerate() {
            let other_layer = &other.layers[i];
            
            // Crossover weights
            for j in 0..layer.weights.nrows() {
                for k in 0..layer.weights.ncols() {
                    if rng.gen::<f64>() < 0.5 {
                        layer.weights[(j, k)] = other_layer.weights[(j, k)];
                    }
                }
            }
            
            // Crossover biases
            for j in 0..layer.biases.len() {
                if rng.gen::<f64>() < 0.5 {
                    layer.biases[j] = other_layer.biases[j];
                }
            }
        }
        
        offspring
    }
    
    /// Adjust mutation parameters based on performance
    pub fn adapt_mutation_parameters(&mut self, fitness_change: f64) {
        // Increase mutation rate if fitness is stagnating
        if fitness_change.abs() < 0.01 {
            self.mutation_rate = (self.mutation_rate * 1.1).min(0.5);
            self.mutation_strength = (self.mutation_strength * 1.05).min(0.5);
        } else if fitness_change > 0.1 {
            // Decrease mutation rate if making good progress
            self.mutation_rate = (self.mutation_rate * 0.95).max(0.01);
            self.mutation_strength = (self.mutation_strength * 0.98).max(0.01);
        }
    }
}

/// Action types that agents can perform
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActionType {
    // Movement actions
    MoveForward,
    MoveBackward,
    TurnLeft,
    TurnRight,
    MoveUp,
    MoveDown,
    
    // Resource actions
    ExtractEnergy,
    ExtractMatter,
    ConsumeResource,
    StoreResource,
    
    // Social actions
    Communicate,
    Cooperate,
    Compete,
    ShareResource,
    
    // Survival actions
    Rest,
    Hide,
    Defend,
    Flee,
    
    // Reproduction actions
    SeekMate,
    Reproduce,
    CareForOffspring,
    
    // Innovation actions
    Experiment,
    CreateTool,
    ModifyCode,
    Learn,
    
    // Wait/Observe
    Wait,
    Observe,
}

impl ActionType {
    /// Get all possible action types
    pub fn all_actions() -> Vec<ActionType> {
        vec![
            ActionType::MoveForward,
            ActionType::MoveBackward,
            ActionType::TurnLeft,
            ActionType::TurnRight,
            ActionType::MoveUp,
            ActionType::MoveDown,
            ActionType::ExtractEnergy,
            ActionType::ExtractMatter,
            ActionType::ConsumeResource,
            ActionType::StoreResource,
            ActionType::Communicate,
            ActionType::Cooperate,
            ActionType::Compete,
            ActionType::ShareResource,
            ActionType::Rest,
            ActionType::Hide,
            ActionType::Defend,
            ActionType::Flee,
            ActionType::SeekMate,
            ActionType::Reproduce,
            ActionType::CareForOffspring,
            ActionType::Experiment,
            ActionType::CreateTool,
            ActionType::ModifyCode,
            ActionType::Learn,
            ActionType::Wait,
            ActionType::Observe,
        ]
    }
    
    /// Get the energy cost of performing this action
    pub fn energy_cost(&self) -> f64 {
        match self {
            ActionType::MoveForward | ActionType::MoveBackward => 2.0,
            ActionType::TurnLeft | ActionType::TurnRight => 1.0,
            ActionType::MoveUp | ActionType::MoveDown => 3.0,
            ActionType::ExtractEnergy => 5.0,
            ActionType::ExtractMatter => 8.0,
            ActionType::ConsumeResource => 1.0,
            ActionType::StoreResource => 2.0,
            ActionType::Communicate => 3.0,
            ActionType::Cooperate => 4.0,
            ActionType::Compete => 6.0,
            ActionType::ShareResource => 2.0,
            ActionType::Rest => -1.0, // Rest recovers energy
            ActionType::Hide => 1.0,
            ActionType::Defend => 4.0,
            ActionType::Flee => 5.0,
            ActionType::SeekMate => 3.0,
            ActionType::Reproduce => 20.0,
            ActionType::CareForOffspring => 5.0,
            ActionType::Experiment => 10.0,
            ActionType::CreateTool => 15.0,
            ActionType::ModifyCode => 25.0,
            ActionType::Learn => 8.0,
            ActionType::Wait => 0.5,
            ActionType::Observe => 1.0,
        }
    }
}

/// Memory system for agents to store and recall experiences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemory {
    pub short_term: Vec<f64>,      // Recent experiences
    pub long_term: Vec<f64>,       // Consolidated memories
    pub episodic: Vec<Episode>,    // Specific event memories
    pub semantic: Vec<Concept>,    // General knowledge
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub timestamp: u64,
    pub context: Vec<f64>,
    pub action: ActionType,
    pub outcome: f64,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub name: String,
    pub associations: Vec<f64>,
    pub strength: f64,
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self {
            short_term: vec![0.0; 10],
            long_term: vec![0.0; 50],
            episodic: Vec::new(),
            semantic: Vec::new(),
        }
    }
}

impl AgentMemory {
    /// Add a new experience to memory
    pub fn add_experience(&mut self, context: Vec<f64>, action: ActionType, outcome: f64, importance: f64, timestamp: u64) {
        let episode = Episode {
            timestamp,
            context,
            action,
            outcome,
            importance,
        };
        
        self.episodic.push(episode);
        
        // Consolidate important memories
        if importance > 0.7 {
            self.consolidate_to_long_term();
        }
        
        // Limit episodic memory size
        if self.episodic.len() > 1000 {
            self.episodic.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
            self.episodic.truncate(500);
        }
    }
    
    /// Consolidate important short-term memories to long-term storage
    fn consolidate_to_long_term(&mut self) {
        // Simple consolidation: average recent important experiences
        let recent_important: Vec<&Episode> = self.episodic
            .iter()
            .filter(|e| e.importance > 0.5)
            .collect();
        
        if recent_important.len() > 5 {
            for i in 0..self.long_term.len().min(10) {
                let avg = recent_important.iter()
                    .map(|e| e.context.get(i).unwrap_or(&0.0))
                    .sum::<f64>() / recent_important.len() as f64;
                self.long_term[i] = (self.long_term[i] + avg) / 2.0;
            }
        }
    }
    
    /// Retrieve memory state for neural network input
    pub fn get_memory_state(&self) -> Vec<f64> {
        let mut memory_state = Vec::new();
        memory_state.extend(&self.short_term);
        
        // Add truncated long-term memory
        memory_state.extend(self.long_term.iter().take(10));
        
        // Pad to fixed size
        while memory_state.len() < 20 {
            memory_state.push(0.0);
        }
        
        memory_state.truncate(20);
        memory_state
    }
}

/// The central AI core of an agent with advanced decision-making capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AICore {
    pub neural_network: NeuralNetwork,
    pub memory: AgentMemory,
    pub decision_confidence: f64,
    pub exploration_rate: f64,
    pub last_action: Option<ActionType>,
    pub performance_history: Vec<f64>,
}

impl AICore {
    /// Creates a new AI core with a sophisticated neural architecture
    pub fn new() -> Self {
        let input_size = SensoryInput::expected_size();
        let hidden_sizes = vec![64, 32, 16]; // Multi-layer architecture
        let output_size = ActionType::all_actions().len();
        
        let neural_network = NeuralNetwork::new(
            input_size,
            &hidden_sizes,
            output_size,
            0.01, // learning rate
        );
        
        Self {
            neural_network,
            memory: AgentMemory::default(),
            decision_confidence: 0.5,
            exploration_rate: 0.3,
            last_action: None,
            performance_history: Vec::new(),
        }
    }
    
    /// Creates a new AI core configured by the agent's genome
    pub fn from_genome(genome: &Genome) -> Self {
        let mut ai_core = Self::new();
        
        // Use genome to influence neural network initialization
        let genome_influence = genome.genes.iter()
            .map(|gene| gene.expression_level)
            .sum::<f64>() / genome.genes.len() as f64;
        
        ai_core.neural_network.mutation_rate = 0.05 + genome_influence * 0.1;
        ai_core.exploration_rate = 0.1 + genome_influence * 0.4;
        
        ai_core
    }
    
    /// Main decision-making process
    pub fn make_decision(&mut self, sensory_input: &SensoryInput, current_tick: u64) -> Result<ActionType> {
        // Update short-term memory with current sensory input
        let input_summary = sensory_input.internal_state.iter().sum::<f64>() / sensory_input.internal_state.len() as f64;
        self.memory.short_term.rotate_right(1);
        self.memory.short_term[0] = input_summary;
        
        // Create neural network input
        let mut network_input = sensory_input.to_vector();
        let memory_state = self.memory.get_memory_state();
        
        // Ensure memory state is included in input
        if network_input.len() < SensoryInput::expected_size() {
            let mut full_input = Vec::new();
            full_input.extend(network_input.as_slice());
            full_input.extend(&memory_state);
            network_input = DVector::from_vec(full_input);
        }
        
        // Get action probabilities from neural network
        let action_probabilities = self.neural_network.forward(&network_input);
        
        // Select action using epsilon-greedy with exploration
        let selected_action = if thread_rng().gen::<f64>() < self.exploration_rate {
            // Explore: choose random action
            let actions = ActionType::all_actions();
            actions[thread_rng().gen_range(0..actions.len())]
        } else {
            // Exploit: choose best action based on network output
            let actions = ActionType::all_actions();
            let best_action_index = action_probabilities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0);
            
            actions[best_action_index.min(actions.len() - 1)]
        };
        
        // Update decision confidence
        let max_prob = action_probabilities.max();
        self.decision_confidence = max_prob;
        
        // Store the decision
        self.last_action = Some(selected_action);
        
        Ok(selected_action)
    }
    
    /// Learn from the outcome of a decision
    pub fn learn_from_outcome(&mut self, action: ActionType, outcome: f64, context: Vec<f64>, timestamp: u64) {
        // Add experience to memory
        let importance = outcome.abs(); // More extreme outcomes are more important
        self.memory.add_experience(context, action, outcome, importance, timestamp);
        
        // Update performance history
        self.performance_history.push(outcome);
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
        
        // Adapt exploration rate based on recent performance
        let recent_performance: f64 = self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;
        
        if recent_performance > 0.5 {
            // Good performance, reduce exploration
            self.exploration_rate = (self.exploration_rate * 0.98).max(0.05);
        } else if recent_performance < -0.5 {
            // Poor performance, increase exploration
            self.exploration_rate = (self.exploration_rate * 1.02).min(0.8);
        }
        
        // Adapt neural network mutation parameters
        let fitness_change = outcome;
        self.neural_network.adapt_mutation_parameters(fitness_change);
    }
    
    /// Evolve the AI through mutation
    pub fn evolve(&mut self) {
        self.neural_network.mutate();
    }
    
    /// Create offspring by combining with another AI core
    pub fn reproduce(&self, other: &AICore) -> AICore {
        let mut offspring = self.clone();
        offspring.neural_network = self.neural_network.crossover(&other.neural_network);
        
        // Inherit some traits from both parents
        offspring.exploration_rate = (self.exploration_rate + other.exploration_rate) / 2.0;
        offspring.decision_confidence = (self.decision_confidence + other.decision_confidence) / 2.0;
        
        // Reset performance history for new individual
        offspring.performance_history.clear();
        
        offspring
    }
    
    /// Get the current fitness estimate based on recent performance
    pub fn get_fitness(&self) -> f64 {
        if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
        }
    }
}