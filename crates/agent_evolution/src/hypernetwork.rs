//! # Hypernetwork Module
//!
//! Implements a hypernetwork that generates neural network architectures and weights
//! dynamically. This enables meta-learning of network structures and rapid adaptation
//! to new tasks without hard-coded biological constraints.
//!
//! The hypernetwork takes a task embedding and generates:
//! - Network architecture (layer sizes, connectivity patterns)
//! - Initial weights and biases
//! - Learning parameters (learning rate, activation functions)
//!
//! This follows recent research in Neural Architecture Search (NAS) and
//! weight-generating networks, but with a focus on physics-informed generation.

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Task embedding that describes what the generated network should do
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEmbedding {
    pub task_type: TaskType,
    pub complexity: f64,           // 0.0 = simple, 1.0 = complex
    pub input_dim: usize,
    pub output_dim: usize,
    pub constraints: Vec<NetworkConstraint>,
    pub performance_target: f64,
}

/// Types of tasks the hypernetwork can generate networks for
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
    ReinforcementLearning,
    PhysicsInformed,
    QuantumComputing,
    PatternRecognition,
    Optimization,
    Control,
}

/// Constraints on the generated network architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConstraint {
    pub constraint_type: ConstraintType,
    pub value: f64,
    pub priority: f64,  // 0.0 = soft, 1.0 = hard
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    MaxLayers,
    MaxParameters,
    MaxDepth,
    MinEfficiency,
    MaxMemoryUsage,
    MinAccuracy,
}

/// Generated neural network architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedArchitecture {
    pub id: Uuid,
    pub layer_sizes: Vec<usize>,
    pub activation_functions: Vec<ActivationType>,
    pub connectivity_pattern: ConnectivityPattern,
    pub learning_parameters: LearningParameters,
    pub generated_weights: Vec<DMatrix<f64>>,
    pub generated_biases: Vec<DVector<f64>>,
    pub architecture_score: f64,
}

/// Activation functions for generated networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU,
    Sinusoidal,
    Softmax,
    Linear,
    Custom { function: String, parameters: HashMap<String, f64> },
}

/// Connectivity patterns for network layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityPattern {
    FullyConnected,
    Sparse { sparsity: f64 },
    Convolutional { kernel_size: usize, stride: usize },
    Recurrent { memory_length: usize },
    Attention { heads: usize },
    Residual,
    SkipConnections { connections: Vec<(usize, usize)> },
}

/// Learning parameters for generated networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningParameters {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub dropout_rate: f64,
    pub batch_size: usize,
    pub optimizer_type: OptimizerType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    RMSprop,
    AdaGrad,
    Custom { name: String, parameters: HashMap<String, f64> },
}

/// Main hypernetwork that generates neural architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypernetwork {
    pub id: Uuid,
    pub name: String,
    pub generator_network: GeneratorNetwork,
    pub architecture_encoder: ArchitectureEncoder,
    pub performance_predictor: PerformancePredictor,
    pub generation_history: Vec<GenerationRecord>,
    pub adaptation_rate: f64,
}

/// Neural network that generates other neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorNetwork {
    pub layers: Vec<GeneratorLayer>,
    pub task_embedding_size: usize,
    pub architecture_embedding_size: usize,
    pub weight_generation_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: ActivationType,
    pub attention_mechanism: Option<AttentionMechanism>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMechanism {
    pub num_heads: usize,
    pub attention_dim: usize,
    pub dropout_rate: f64,
}

/// Encodes task embeddings into architecture specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureEncoder {
    pub embedding_layers: Vec<usize>,
    pub architecture_space: ArchitectureSpace,
    pub encoding_method: EncodingMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSpace {
    pub max_layers: usize,
    pub max_layer_size: usize,
    pub activation_options: Vec<ActivationType>,
    pub connectivity_options: Vec<ConnectivityPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingMethod {
    Continuous,
    Discrete,
    Mixed,
    Hierarchical,
}

/// Predicts performance of generated architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictor {
    pub predictor_network: Vec<usize>,
    pub prediction_metrics: Vec<PredictionMetric>,
    pub calibration_data: Vec<CalibrationPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionMetric {
    Accuracy,
    Efficiency,
    MemoryUsage,
    TrainingTime,
    Generalization,
    Robustness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub predicted_performance: f64,
    pub actual_performance: f64,
    pub confidence: f64,
}

/// Record of a network generation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub timestamp: f64,
    pub task_embedding: TaskEmbedding,
    pub generated_architecture: GeneratedArchitecture,
    pub actual_performance: Option<f64>,
    pub generation_time: f64,
    pub adaptation_feedback: f64,
}

impl Hypernetwork {
    /// Create a new hypernetwork with default configuration
    pub fn new() -> Self {
        let generator_network = GeneratorNetwork {
            layers: vec![
                GeneratorLayer {
                    input_size: 128,
                    output_size: 256,
                    activation: ActivationType::ReLU,
                    attention_mechanism: Some(AttentionMechanism {
                        num_heads: 8,
                        attention_dim: 64,
                        dropout_rate: 0.1,
                    }),
                },
                GeneratorLayer {
                    input_size: 256,
                    output_size: 512,
                    activation: ActivationType::ReLU,
                    attention_mechanism: None,
                },
                GeneratorLayer {
                    input_size: 512,
                    output_size: 256,
                    activation: ActivationType::ReLU,
                    attention_mechanism: None,
                },
            ],
            task_embedding_size: 64,
            architecture_embedding_size: 128,
            weight_generation_size: 256,
        };

        let architecture_encoder = ArchitectureEncoder {
            embedding_layers: vec![64, 128, 256],
            architecture_space: ArchitectureSpace {
                max_layers: 10,
                max_layer_size: 1024,
                activation_options: vec![
                    ActivationType::ReLU,
                    ActivationType::Tanh,
                    ActivationType::Sigmoid,
                    ActivationType::Swish,
                ],
                connectivity_options: vec![
                    ConnectivityPattern::FullyConnected,
                    ConnectivityPattern::Sparse { sparsity: 0.8 },
                    ConnectivityPattern::Residual,
                ],
            },
            encoding_method: EncodingMethod::Mixed,
        };

        let performance_predictor = PerformancePredictor {
            predictor_network: vec![128, 64, 32, 1],
            prediction_metrics: vec![
                PredictionMetric::Accuracy,
                PredictionMetric::Efficiency,
                PredictionMetric::MemoryUsage,
            ],
            calibration_data: Vec::new(),
        };

        Self {
            id: Uuid::new_v4(),
            name: "AdaptiveHypernetwork".to_string(),
            generator_network,
            architecture_encoder,
            performance_predictor,
            generation_history: Vec::new(),
            adaptation_rate: 0.01,
        }
    }

    /// Generate a neural network architecture for a given task
    pub fn generate_architecture(&mut self, task: &TaskEmbedding) -> Result<GeneratedArchitecture> {
        let start_time = std::time::Instant::now();

        // Encode task into architecture embedding
        let architecture_embedding = self.encode_task(task)?;

        // Generate architecture specification
        let layer_sizes = self.generate_layer_sizes(&architecture_embedding, task)?;
        let activation_functions = self.generate_activations(&architecture_embedding, &layer_sizes)?;
        let connectivity_pattern = self.generate_connectivity(&architecture_embedding, task)?;
        let learning_parameters = self.generate_learning_parameters(&architecture_embedding, task)?;

        // Generate initial weights and biases
        let (generated_weights, generated_biases) = self.generate_weights_and_biases(&layer_sizes, &architecture_embedding)?;

        // Predict architecture performance
        let architecture_score = self.predict_performance(&architecture_embedding, &layer_sizes)?;

        let architecture = GeneratedArchitecture {
            id: Uuid::new_v4(),
            layer_sizes,
            activation_functions,
            connectivity_pattern,
            learning_parameters,
            generated_weights,
            generated_biases,
            architecture_score,
        };

        // Record generation
        let generation_time = start_time.elapsed().as_secs_f64();
        let record = GenerationRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            task_embedding: task.clone(),
            generated_architecture: architecture.clone(),
            actual_performance: None,
            generation_time,
            adaptation_feedback: 0.0,
        };
        self.generation_history.push(record);

        Ok(architecture)
    }

    /// Encode task embedding into architecture specification
    fn encode_task(&self, task: &TaskEmbedding) -> Result<DVector<f64>> {
        let mut embedding = DVector::zeros(self.architecture_encoder.architecture_space.max_layers * 3);

        // Encode task type
        let task_type_encoding = match task.task_type {
            TaskType::Classification => 0.0,
            TaskType::Regression => 0.2,
            TaskType::ReinforcementLearning => 0.4,
            TaskType::PhysicsInformed => 0.6,
            TaskType::QuantumComputing => 0.8,
            TaskType::PatternRecognition => 1.0,
            TaskType::Optimization => 0.3,
            TaskType::Control => 0.7,
        };

        // Encode complexity and dimensions
        embedding[0] = task_type_encoding;
        embedding[1] = task.complexity;
        embedding[2] = (task.input_dim as f64) / 1000.0;  // Normalize
        embedding[3] = (task.output_dim as f64) / 100.0;  // Normalize
        embedding[4] = task.performance_target;

        Ok(embedding)
    }

    /// Generate layer sizes based on task requirements
    fn generate_layer_sizes(&self, embedding: &DVector<f64>, task: &TaskEmbedding) -> Result<Vec<usize>> {
        let complexity = embedding[1];
        let input_dim = task.input_dim;
        let output_dim = task.output_dim;

        // Determine number of layers based on complexity
        let num_layers = if complexity < 0.3 {
            2
        } else if complexity < 0.7 {
            3 + (complexity * 3.0) as usize
        } else {
            5 + (complexity * 5.0) as usize
        }.min(self.architecture_encoder.architecture_space.max_layers);

        let mut layer_sizes = Vec::new();
        layer_sizes.push(input_dim);

        // Generate hidden layer sizes
        for i in 1..num_layers {
            let layer_size = if i == num_layers - 1 {
                output_dim
            } else {
                let base_size = (input_dim + output_dim) / 2;
                let complexity_factor = 1.0 + complexity;
                let layer_factor = 1.0 - (i as f64 / num_layers as f64) * 0.5;
                (base_size as f64 * complexity_factor * layer_factor) as usize
            }.min(self.architecture_encoder.architecture_space.max_layer_size);
            
            layer_sizes.push(layer_size);
        }

        Ok(layer_sizes)
    }

    /// Generate activation functions for each layer
    fn generate_activations(&self, embedding: &DVector<f64>, layer_sizes: &[usize]) -> Result<Vec<ActivationType>> {
        let task_type = embedding[0];
        let mut activations = Vec::new();

        for (i, _) in layer_sizes.iter().enumerate() {
            let activation = if i == layer_sizes.len() - 1 {
                // Output layer activation
                if task_type < 0.3 {
                    ActivationType::Softmax
                } else {
                    ActivationType::Linear
                }
            } else {
                // Hidden layer activation
                let activation_choice = (task_type * 4.0 + i as f64) % 4.0;
                match activation_choice as usize {
                    0 => ActivationType::ReLU,
                    1 => ActivationType::Tanh,
                    2 => ActivationType::Sigmoid,
                    3 => ActivationType::Swish,
                    _ => ActivationType::ReLU,
                }
            };
            activations.push(activation);
        }

        Ok(activations)
    }

    /// Generate connectivity pattern
    fn generate_connectivity(&self, embedding: &DVector<f64>, _task: &TaskEmbedding) -> Result<ConnectivityPattern> {
        let complexity = embedding[1];
        
        if complexity < 0.3 {
            Ok(ConnectivityPattern::FullyConnected)
        } else if complexity < 0.7 {
            Ok(ConnectivityPattern::Sparse { sparsity: 0.8 })
        } else {
            Ok(ConnectivityPattern::Residual)
        }
    }

    /// Generate learning parameters
    fn generate_learning_parameters(&self, embedding: &DVector<f64>, _task: &TaskEmbedding) -> Result<LearningParameters> {
        let complexity = embedding[1];
        
        let learning_rate = if complexity < 0.3 {
            0.01
        } else if complexity < 0.7 {
            0.001
        } else {
            0.0001
        };

        Ok(LearningParameters {
            learning_rate,
            momentum: 0.9,
            weight_decay: 0.0001,
            dropout_rate: complexity * 0.3,
            batch_size: 32,
            optimizer_type: OptimizerType::Adam,
        })
    }

    /// Generate initial weights and biases
    fn generate_weights_and_biases(&self, layer_sizes: &[usize], embedding: &DVector<f64>) -> Result<(Vec<DMatrix<f64>>, Vec<DVector<f64>>)> {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Generate weights using Xavier initialization with task-specific scaling
            let scale = (2.0 / (input_size + output_size) as f64).sqrt() * (1.0 + embedding[1] * 0.5);
            let weight_matrix = DMatrix::from_fn(output_size, input_size, |_, _| {
                (rand::random::<f64>() - 0.5) * 2.0 * scale
            });
            weights.push(weight_matrix);

            // Generate biases
            let bias_vector = DVector::from_fn(output_size, |_, _| {
                (rand::random::<f64>() - 0.5) * 0.1
            });
            biases.push(bias_vector);
        }

        Ok((weights, biases))
    }

    /// Predict performance of generated architecture
    fn predict_performance(&self, embedding: &DVector<f64>, layer_sizes: &[usize]) -> Result<f64> {
        // Simple performance prediction based on architecture characteristics
        let complexity = embedding[1];
        let total_parameters: usize = layer_sizes.windows(2)
            .map(|w| w[0] * w[1])
            .sum();

        let parameter_efficiency = 1.0 / (1.0 + total_parameters as f64 / 10000.0);
        let complexity_efficiency = 1.0 - complexity * 0.3;
        let task_alignment = 1.0 - (embedding[0] - 0.5).abs() * 0.5;

        let predicted_performance = parameter_efficiency * complexity_efficiency * task_alignment;
        Ok(predicted_performance.max(0.0).min(1.0))
    }

    /// Update hypernetwork based on actual performance feedback
    pub fn update_from_feedback(&mut self, architecture_id: Uuid, actual_performance: f64) -> Result<()> {
        if let Some(record) = self.generation_history.iter_mut().find(|r| r.generated_architecture.id == architecture_id) {
            record.actual_performance = Some(actual_performance);
            
            // Calculate adaptation feedback
            let predicted = record.generated_architecture.architecture_score;
            let error = (actual_performance - predicted).abs();
            record.adaptation_feedback = 1.0 - error;

            // Update adaptation rate based on prediction accuracy
            self.adaptation_rate = (self.adaptation_rate * 0.99 + error * 0.01).max(0.001).min(0.1);
        }

        Ok(())
    }

    /// Get generation statistics
    pub fn get_statistics(&self) -> HypernetworkStatistics {
        let total_generations = self.generation_history.len();
        let successful_generations = self.generation_history.iter()
            .filter(|r| r.actual_performance.is_some())
            .count();

        let average_performance = if successful_generations > 0 {
            self.generation_history.iter()
                .filter_map(|r| r.actual_performance)
                .sum::<f64>() / successful_generations as f64
        } else {
            0.0
        };

        let average_generation_time = if total_generations > 0 {
            self.generation_history.iter()
                .map(|r| r.generation_time)
                .sum::<f64>() / total_generations as f64
        } else {
            0.0
        };

        HypernetworkStatistics {
            total_generations,
            successful_generations,
            average_performance,
            average_generation_time,
            adaptation_rate: self.adaptation_rate,
        }
    }
}

/// Statistics about hypernetwork performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypernetworkStatistics {
    pub total_generations: usize,
    pub successful_generations: usize,
    pub average_performance: f64,
    pub average_generation_time: f64,
    pub adaptation_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypernetwork_generates_architecture() {
        let mut hypernetwork = Hypernetwork::new();
        
        let task = TaskEmbedding {
            task_type: TaskType::Classification,
            complexity: 0.5,
            input_dim: 10,
            output_dim: 3,
            constraints: vec![],
            performance_target: 0.9,
        };

        let architecture = hypernetwork.generate_architecture(&task).unwrap();
        
        assert_eq!(architecture.layer_sizes[0], 10);
        assert_eq!(architecture.layer_sizes.last().unwrap(), &3);
        assert!(!architecture.layer_sizes.is_empty());
        assert_eq!(architecture.activation_functions.len(), architecture.layer_sizes.len());
    }

    #[test]
    fn hypernetwork_adapts_from_feedback() {
        let mut hypernetwork = Hypernetwork::new();
        let initial_rate = hypernetwork.adaptation_rate;
        
        let task = TaskEmbedding {
            task_type: TaskType::Regression,
            complexity: 0.3,
            input_dim: 5,
            output_dim: 1,
            constraints: vec![],
            performance_target: 0.8,
        };

        let architecture = hypernetwork.generate_architecture(&task).unwrap();
        hypernetwork.update_from_feedback(architecture.id, 0.7).unwrap();
        
        // Adaptation rate should have changed
        assert_ne!(hypernetwork.adaptation_rate, initial_rate);
    }
} 