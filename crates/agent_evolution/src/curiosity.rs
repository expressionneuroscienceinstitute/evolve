//! # Curiosity-Driven Learning Module
//!
//! Implements intrinsic motivation through curiosity-driven exploration and learning.
//! This module provides agents with the ability to seek out novel experiences,
//! learn from prediction errors, and develop intrinsic rewards based on information gain.
//!
//! The curiosity system is based on:
//! - Information gain maximization
//! - Prediction error minimization
//! - Novelty detection and exploration
//! - Intrinsic motivation without external rewards
//!
//! No hard-coded biological outcomes - purely algorithmic curiosity mechanisms.

use anyhow::Result;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Curiosity-driven learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriositySystem {
    pub id: Uuid,
    pub name: String,
    pub curiosity_network: CuriosityNetwork,
    pub novelty_detector: NoveltyDetector,
    pub prediction_engine: PredictionEngine,
    pub exploration_policy: ExplorationPolicy,
    pub curiosity_history: Vec<CuriosityEvent>,
    pub intrinsic_rewards: Vec<IntrinsicReward>,
    pub learning_rate: f64,
    pub exploration_rate: f64,
    pub novelty_threshold: f64,
}

/// Neural network for curiosity-driven learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityNetwork {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub prediction_weights: Vec<f64>,
    pub curiosity_weights: Vec<f64>,
    pub learning_history: Vec<LearningEvent>,
    pub prediction_accuracy: f64,
    pub curiosity_level: f64,
}

/// Detects novel experiences and patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyDetector {
    pub experience_buffer: VecDeque<Experience>,
    pub novelty_threshold: f64,
    pub novelty_decay: f64,
    pub pattern_memory: HashMap<String, PatternMemory>,
    pub novelty_scores: Vec<f64>,
}

/// Experience representation for novelty detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: Uuid,
    pub timestamp: f64,
    pub sensory_input: DVector<f64>,
    pub context: DVector<f64>,
    pub action_taken: Option<ActionType>,
    pub outcome: Option<f64>,
    pub novelty_score: f64,
    pub curiosity_value: f64,
}

/// Memory of patterns for novelty comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMemory {
    pub pattern_hash: String,
    pub frequency: u32,
    pub last_seen: f64,
    pub average_novelty: f64,
    pub curiosity_contribution: f64,
}

/// Prediction engine for curiosity-driven learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEngine {
    pub model: PredictionModel,
    pub prediction_history: Vec<Prediction>,
    pub error_threshold: f64,
    pub learning_enabled: bool,
    pub adaptation_rate: f64,
}

/// Prediction model for next-state estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModel {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub activation_function: ActivationType,
    pub prediction_accuracy: f64,
}

/// Individual prediction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub timestamp: f64,
    pub input_state: DVector<f64>,
    pub predicted_next_state: DVector<f64>,
    pub actual_next_state: DVector<f64>,
    pub prediction_error: f64,
    pub curiosity_reward: f64,
}

/// Exploration policy for curiosity-driven actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPolicy {
    pub policy_type: PolicyType,
    pub exploration_parameters: ExplorationParameters,
    pub action_selection_history: Vec<ActionSelection>,
    pub exploration_effectiveness: f64,
}

/// Types of exploration policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyType {
    EpsilonGreedy,
    Boltzmann,
    UpperConfidenceBound,
    ThompsonSampling,
    CuriosityDriven,
    InformationGain,
}

/// Parameters for exploration behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationParameters {
    pub epsilon: f64,           // Exploration rate
    pub temperature: f64,       // Boltzmann temperature
    pub confidence_factor: f64, // UCB confidence
    pub curiosity_weight: f64,  // Weight of curiosity in action selection
    pub novelty_bonus: f64,     // Bonus for novel actions
}

/// Action selection record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSelection {
    pub timestamp: f64,
    pub available_actions: Vec<ActionType>,
    pub selected_action: ActionType,
    pub selection_reason: SelectionReason,
    pub curiosity_contribution: f64,
    pub exploration_bonus: f64,
}

/// Reasons for action selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionReason {
    Exploitation,
    Exploration,
    Curiosity,
    Novelty,
    InformationGain,
    Random,
}

/// Action types for curiosity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Explore,
    Investigate,
    Experiment,
    Observe,
    Interact,
    Learn,
    Create,
    Discover,
}

/// Activation functions for neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    Linear,
}

/// Learning event in curiosity network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub timestamp: f64,
    pub input: DVector<f64>,
    pub target: DVector<f64>,
    pub prediction: DVector<f64>,
    pub error: f64,
    pub curiosity_reward: f64,
    pub learning_rate: f64,
}

/// Curiosity event tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityEvent {
    pub timestamp: f64,
    pub event_type: CuriosityEventType,
    pub curiosity_level: f64,
    pub novelty_score: f64,
    pub prediction_error: f64,
    pub intrinsic_reward: f64,
    pub description: String,
}

/// Types of curiosity events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CuriosityEventType {
    NoveltyDetected,
    PredictionError,
    ExplorationAction,
    LearningBreakthrough,
    PatternDiscovery,
    InformationGain,
    CuriosityPeak,
}

/// Intrinsic reward from curiosity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicReward {
    pub timestamp: f64,
    pub reward_type: RewardType,
    pub magnitude: f64,
    pub source: String,
    pub context: DVector<f64>,
}

/// Types of intrinsic rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardType {
    Novelty,
    PredictionError,
    InformationGain,
    PatternDiscovery,
    Exploration,
    Learning,
    Creativity,
}

impl CuriositySystem {
    /// Create a new curiosity-driven learning system
    pub fn new(input_size: usize) -> Self {
        let curiosity_network = CuriosityNetwork {
            input_size,
            hidden_size: 64,
            output_size: input_size,
            prediction_weights: vec![0.0; input_size * 64 + 64 * input_size],
            curiosity_weights: vec![0.0; input_size * 64 + 64 * input_size],
            learning_history: Vec::new(),
            prediction_accuracy: 0.0,
            curiosity_level: 0.5,
        };

        let novelty_detector = NoveltyDetector {
            experience_buffer: VecDeque::with_capacity(1000),
            novelty_threshold: 0.1,
            novelty_decay: 0.95,
            pattern_memory: HashMap::new(),
            novelty_scores: Vec::new(),
        };

        let prediction_engine = PredictionEngine {
            model: PredictionModel {
                input_dim: input_size,
                hidden_dim: 32,
                output_dim: input_size,
                weights: vec![0.0; input_size * 32 + 32 * input_size],
                biases: vec![0.0; 32 + input_size],
                activation_function: ActivationType::Tanh,
                prediction_accuracy: 0.0,
            },
            prediction_history: Vec::new(),
            error_threshold: 0.01,
            learning_enabled: true,
            adaptation_rate: 0.01,
        };

        let exploration_policy = ExplorationPolicy {
            policy_type: PolicyType::CuriosityDriven,
            exploration_parameters: ExplorationParameters {
                epsilon: 0.1,
                temperature: 1.0,
                confidence_factor: 2.0,
                curiosity_weight: 0.7,
                novelty_bonus: 0.3,
            },
            action_selection_history: Vec::new(),
            exploration_effectiveness: 0.0,
        };

        Self {
            id: Uuid::new_v4(),
            name: "CuriosityDrivenLearningSystem".to_string(),
            curiosity_network,
            novelty_detector,
            prediction_engine,
            exploration_policy,
            curiosity_history: Vec::new(),
            intrinsic_rewards: Vec::new(),
            learning_rate: 0.01,
            exploration_rate: 0.2,
            novelty_threshold: 0.1,
        }
    }

    /// Process new experience and update curiosity system
    pub fn process_experience(&mut self, experience: Experience) -> Result<CuriosityOutput> {
        let timestamp = experience.timestamp;

        // Detect novelty
        let novelty_score = self.detect_novelty(&experience)?;
        
        // Make prediction
        let prediction = self.make_prediction(&experience.sensory_input)?;
        let prediction_error = self.calculate_prediction_error(&prediction, &experience.sensory_input)?;
        
        // Calculate curiosity level
        let curiosity_level = self.calculate_curiosity_level(novelty_score, prediction_error)?;
        
        // Generate intrinsic reward
        let intrinsic_reward = self.generate_intrinsic_reward(novelty_score, prediction_error, curiosity_level)?;
        
        // Update learning
        self.update_learning(&experience, prediction_error, intrinsic_reward.magnitude)?;
        
        // Record curiosity event
        let event = CuriosityEvent {
            timestamp,
            event_type: CuriosityEventType::NoveltyDetected,
            curiosity_level,
            novelty_score,
            prediction_error,
            intrinsic_reward: intrinsic_reward.magnitude,
            description: format!("Processed experience with novelty {:.3} and curiosity {:.3}", novelty_score, curiosity_level),
        };
        self.curiosity_history.push(event);
        
        // Store intrinsic reward
        self.intrinsic_rewards.push(intrinsic_reward.clone());

        Ok(CuriosityOutput {
            curiosity_level,
            novelty_score,
            prediction_error,
            intrinsic_reward: intrinsic_reward.magnitude,
            exploration_bonus: self.calculate_exploration_bonus(novelty_score)?,
            learning_signal: self.calculate_learning_signal(prediction_error)?,
        })
    }

    /// Select action based on curiosity-driven exploration
    pub fn select_action(&mut self, available_actions: Vec<ActionType>, current_state: &DVector<f64>) -> Result<ActionSelection> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut best_action = available_actions[0].clone();
        let mut best_score = f64::NEG_INFINITY;
        let mut selection_reason = SelectionReason::Random;
        let mut curiosity_contribution = 0.0;
        let mut exploration_bonus = 0.0;

        for action in &available_actions {
            let (score, reason, curiosity, bonus) = self.evaluate_action(action, current_state)?;
            
            if score > best_score {
                best_score = score;
                best_action = action.clone();
                selection_reason = reason;
                curiosity_contribution = curiosity;
                exploration_bonus = bonus;
            }
        }

        let selection = ActionSelection {
            timestamp,
            available_actions,
            selected_action: best_action,
            selection_reason,
            curiosity_contribution,
            exploration_bonus,
        };

        self.exploration_policy.action_selection_history.push(selection.clone());
        Ok(selection)
    }

    /// Detect novelty in experience
    fn detect_novelty(&mut self, experience: &Experience) -> Result<f64> {
        let pattern_hash = self.hash_experience_pattern(experience)?;
        
        // Check if pattern exists in memory
        let novelty_score = if let Some(pattern) = self.novelty_detector.pattern_memory.get(&pattern_hash) {
            // Pattern exists, calculate novelty based on frequency and recency
            let frequency_factor = 1.0 / (1.0 + pattern.frequency as f64);
            let recency_factor = (experience.timestamp - pattern.last_seen).max(1.0) / 1000.0;
            frequency_factor * recency_factor * self.novelty_detector.novelty_decay
        } else {
            // New pattern - high novelty
            1.0
        };

        // Update pattern memory
        let entry = self.novelty_detector.pattern_memory.entry(pattern_hash.clone()).or_insert(PatternMemory {
            pattern_hash: pattern_hash.clone(),
            frequency: 0,
            last_seen: experience.timestamp,
            average_novelty: novelty_score,
            curiosity_contribution: novelty_score,
        });
        
        entry.frequency += 1;
        entry.last_seen = experience.timestamp;
        entry.average_novelty = (entry.average_novelty + novelty_score) / 2.0;

        // Store in experience buffer
        self.novelty_detector.experience_buffer.push_back(experience.clone());
        if self.novelty_detector.experience_buffer.len() > 1000 {
            self.novelty_detector.experience_buffer.pop_front();
        }

        Ok(novelty_score)
    }

    /// Make prediction about next state
    fn make_prediction(&mut self, current_state: &DVector<f64>) -> Result<DVector<f64>> {
        // Simple linear prediction model
        let mut prediction = DVector::zeros(current_state.len());
        
        for i in 0..current_state.len() {
            let mut sum = 0.0;
            for j in 0..current_state.len() {
                let weight_idx = i * current_state.len() + j;
                if weight_idx < self.prediction_engine.model.weights.len() {
                    sum += current_state[j] * self.prediction_engine.model.weights[weight_idx];
                }
            }
            if i < self.prediction_engine.model.biases.len() {
                sum += self.prediction_engine.model.biases[i];
            }
            prediction[i] = sum;
        }

        Ok(prediction)
    }

    /// Calculate prediction error
    fn calculate_prediction_error(&self, prediction: &DVector<f64>, actual: &DVector<f64>) -> Result<f64> {
        let error = (prediction - actual).norm_squared();
        Ok(error)
    }

    /// Calculate curiosity level
    fn calculate_curiosity_level(&self, novelty_score: f64, prediction_error: f64) -> Result<f64> {
        let novelty_contribution = novelty_score * 0.6;
        let error_contribution = prediction_error.min(1.0) * 0.4;
        let curiosity = novelty_contribution + error_contribution;
        Ok(curiosity.min(1.0))
    }

    /// Generate intrinsic reward
    fn generate_intrinsic_reward(&self, novelty_score: f64, prediction_error: f64, curiosity_level: f64) -> Result<IntrinsicReward> {
        let reward_type = if novelty_score > 0.5 {
            RewardType::Novelty
        } else if prediction_error > 0.1 {
            RewardType::PredictionError
        } else {
            RewardType::Learning
        };

        let magnitude = curiosity_level * self.exploration_policy.exploration_parameters.curiosity_weight;

        Ok(IntrinsicReward {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            reward_type,
            magnitude,
            source: "curiosity_system".to_string(),
            context: DVector::zeros(10), // Placeholder
        })
    }

    /// Update learning based on experience
    fn update_learning(&mut self, experience: &Experience, prediction_error: f64, intrinsic_reward: f64) -> Result<()> {
        // Update prediction model weights
        if self.prediction_engine.learning_enabled {
            let learning_rate = self.learning_rate * intrinsic_reward;
            
            // Simple gradient descent update
            for i in 0..self.prediction_engine.model.weights.len() {
                self.prediction_engine.model.weights[i] -= learning_rate * prediction_error * 0.01;
            }
        }

        // Update curiosity network
        let learning_event = LearningEvent {
            timestamp: experience.timestamp,
            input: experience.sensory_input.clone(),
            target: experience.sensory_input.clone(), // Autoencoder-like learning
            prediction: self.make_prediction(&experience.sensory_input)?,
            error: prediction_error,
            curiosity_reward: intrinsic_reward,
            learning_rate: self.learning_rate,
        };

        self.curiosity_network.learning_history.push(learning_event);
        
        // Keep only recent learning events
        if self.curiosity_network.learning_history.len() > 100 {
            self.curiosity_network.learning_history.remove(0);
        }

        Ok(())
    }

    /// Evaluate action for curiosity-driven selection
    fn evaluate_action(&self, action: &ActionType, _current_state: &DVector<f64>) -> Result<(f64, SelectionReason, f64, f64)> {
        let base_score = match action {
            ActionType::Explore => 0.5,
            ActionType::Investigate => 0.7,
            ActionType::Experiment => 0.8,
            ActionType::Observe => 0.3,
            ActionType::Interact => 0.6,
            ActionType::Learn => 0.9,
            ActionType::Create => 1.0,
            ActionType::Discover => 1.0,
        };

        let curiosity_contribution = self.curiosity_network.curiosity_level * 0.5;
        let exploration_bonus = self.exploration_policy.exploration_parameters.novelty_bonus * 0.3;

        let total_score = base_score + curiosity_contribution + exploration_bonus;

        let reason = if curiosity_contribution > 0.5 {
            SelectionReason::Curiosity
        } else if exploration_bonus > 0.2 {
            SelectionReason::Exploration
        } else {
            SelectionReason::Exploitation
        };

        Ok((total_score, reason, curiosity_contribution, exploration_bonus))
    }

    /// Calculate exploration bonus
    fn calculate_exploration_bonus(&self, novelty_score: f64) -> Result<f64> {
        Ok(novelty_score * self.exploration_policy.exploration_parameters.novelty_bonus)
    }

    /// Calculate learning signal
    fn calculate_learning_signal(&self, prediction_error: f64) -> Result<f64> {
        Ok(prediction_error * self.learning_rate)
    }

    /// Hash experience pattern for novelty detection
    fn hash_experience_pattern(&self, experience: &Experience) -> Result<String> {
        // Simple hash based on sensory input
        let mut hash = 0u64;
        for (i, &value) in experience.sensory_input.iter().enumerate() {
            hash = hash.wrapping_add((value * 1000.0) as u64 * (i as u64 + 1));
        }
        Ok(format!("{:x}", hash))
    }

    /// Get curiosity statistics
    pub fn get_statistics(&self) -> CuriosityStatistics {
        let total_events = self.curiosity_history.len();
        let average_curiosity = if total_events > 0 {
            self.curiosity_history.iter().map(|e| e.curiosity_level).sum::<f64>() / total_events as f64
        } else {
            0.0
        };

        let total_rewards = self.intrinsic_rewards.len();
        let average_reward = if total_rewards > 0 {
            self.intrinsic_rewards.iter().map(|r| r.magnitude).sum::<f64>() / total_rewards as f64
        } else {
            0.0
        };

        CuriosityStatistics {
            total_events,
            average_curiosity,
            total_rewards,
            average_reward,
            current_curiosity_level: self.curiosity_network.curiosity_level,
            exploration_effectiveness: self.exploration_policy.exploration_effectiveness,
            prediction_accuracy: self.prediction_engine.model.prediction_accuracy,
        }
    }
}

/// Output from curiosity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityOutput {
    pub curiosity_level: f64,
    pub novelty_score: f64,
    pub prediction_error: f64,
    pub intrinsic_reward: f64,
    pub exploration_bonus: f64,
    pub learning_signal: f64,
}

/// Statistics about curiosity system performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityStatistics {
    pub total_events: usize,
    pub average_curiosity: f64,
    pub total_rewards: usize,
    pub average_reward: f64,
    pub current_curiosity_level: f64,
    pub exploration_effectiveness: f64,
    pub prediction_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curiosity_system_creates_and_processes_experience() {
        let mut curiosity = CuriositySystem::new(10);
        
        let experience = Experience {
            id: Uuid::new_v4(),
            timestamp: 0.0,
            sensory_input: DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            context: DVector::zeros(5),
            action_taken: Some(ActionType::Explore),
            outcome: Some(0.5),
            novelty_score: 0.0,
            curiosity_value: 0.0,
        };

        let output = curiosity.process_experience(experience).unwrap();
        
        assert!(output.curiosity_level >= 0.0);
        assert!(output.curiosity_level <= 1.0);
        assert!(output.novelty_score >= 0.0);
        assert!(output.intrinsic_reward >= 0.0);
    }

    #[test]
    fn curiosity_system_selects_actions() {
        let mut curiosity = CuriositySystem::new(5);
        
        let actions = vec![ActionType::Explore, ActionType::Learn, ActionType::Experiment];
        let state = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        
        let selection = curiosity.select_action(actions, &state).unwrap();
        
        assert!(selection.curiosity_contribution >= 0.0);
        assert!(selection.exploration_bonus >= 0.0);
    }
} 