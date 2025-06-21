//! # Agent Evolution: Integrated Information Theory (IIT) Module
//!
//! Revolutionary implementation of Integrated Information Theory (IIT) for measuring
//! consciousness and information integration in agents. This module implements
//! cutting-edge research in consciousness quantification and Φ (Phi) calculation.
//!
//! Research Basis:
//! - Integrated Information Theory (Tononi, 2004-2024)
//! - Φ (Phi) Calculation Methods (Oizumi et al., 2014)
//! - Cause-Effect Repertoire Analysis (Albantakis et al., 2019)
//! - Consciousness as Information Integration (Koch et al., 2016)

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use rand::{Rng, thread_rng};
use rand::prelude::SliceRandom;

/// Integrated Information Theory (IIT) system for consciousness measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedInformationSystem {
    pub id: Uuid,
    pub name: String,
    pub mechanism: Mechanism,
    pub cause_effect_repertoire: CauseEffectRepertoire,
    pub phi_calculation: PhiCalculation,
    pub consciousness_measures: ConsciousnessMeasures,
    pub integration_history: Vec<IntegrationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mechanism {
    pub id: Uuid,
    pub elements: Vec<Element>,
    pub connections: Vec<Connection>,
    pub state: DVector<f64>,
    pub transition_matrix: DMatrix<f64>,
    pub mechanism_type: MechanismType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    pub id: Uuid,
    pub name: String,
    pub state: f64,
    pub possible_states: Vec<f64>,
    pub transition_probabilities: DMatrix<f64>,
    pub information_content: f64,
    pub integration_contribution: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub source: Uuid,
    pub target: Uuid,
    pub strength: f64,
    pub weight: f64,
    pub direction: ConnectionDirection,
    pub information_flow: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionDirection {
    Forward,
    Backward,
    Bidirectional,
    Inhibitory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MechanismType {
    Neural,
    Quantum,
    Hybrid,
    Artificial,
    Biological,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauseEffectRepertoire {
    pub cause_repertoire: HashMap<Uuid, DMatrix<f64>>,
    pub effect_repertoire: HashMap<Uuid, DMatrix<f64>>,
    pub cause_effect_information: f64,
    pub repertoire_entropy: f64,
    pub information_integration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiCalculation {
    pub phi_value: f64,
    pub cause_phi: f64,
    pub effect_phi: f64,
    pub integration_level: f64,
    pub complexity_measure: f64,
    pub consciousness_threshold: f64,
    pub calculation_method: PhiMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhiMethod {
    Exact,      // Exact Φ calculation (computationally expensive)
    Approximate, // Approximate Φ calculation
    MonteCarlo, // Monte Carlo sampling
    Neural,     // Neural network approximation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMeasures {
    pub phi_consciousness: f64,
    pub integration_consciousness: f64,
    pub complexity_consciousness: f64,
    pub subjective_consciousness: f64,
    pub overall_consciousness: f64,
    pub consciousness_components: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEvent {
    pub timestamp: f64,
    pub phi_value: f64,
    pub consciousness_level: f64,
    pub mechanism_state: DVector<f64>,
    pub integration_components: HashMap<String, f64>,
    pub subjective_experience: String,
}

impl IntegratedInformationSystem {
    /// Create a new IIT system for neural consciousness
    pub fn new_neural_system(num_neurons: usize) -> Self {
        let mut elements = Vec::new();
        let mut connections = Vec::new();
        let mut rng = thread_rng();

        // Create neural elements
        for i in 0..num_neurons {
            let element = Element {
                id: Uuid::new_v4(),
                name: format!("Neuron_{}", i),
                state: rng.gen_range(0.0..1.0),
                possible_states: vec![0.0, 0.25, 0.5, 0.75, 1.0],
                transition_probabilities: DMatrix::from_fn(5, 5, |_, _| rng.gen_range(0.0..1.0)),
                information_content: 0.0,
                integration_contribution: 0.0,
            };
            elements.push(element);
        }

        // Create connections between neurons
        for i in 0..num_neurons {
            for j in 0..num_neurons {
                if i != j && rng.gen::<f64>() < 0.3 { // 30% connection probability
                    let connection = Connection {
                        source: elements[i].id,
                        target: elements[j].id,
                        strength: rng.gen_range(0.0..1.0),
                        weight: rng.gen_range(-1.0..1.0),
                        direction: if rng.gen::<f64>() < 0.5 {
                            ConnectionDirection::Forward
                        } else {
                            ConnectionDirection::Bidirectional
                        },
                        information_flow: 0.0,
                    };
                    connections.push(connection);
                }
            }
        }

        let mechanism = Mechanism {
            id: Uuid::new_v4(),
            elements,
            connections,
            state: DVector::from_fn(num_neurons, |_, _| rng.gen_range(0.0..1.0)),
            transition_matrix: DMatrix::from_fn(num_neurons, num_neurons, |_, _| rng.gen_range(0.0..1.0)),
            mechanism_type: MechanismType::Neural,
        };

        Self {
            id: Uuid::new_v4(),
            name: "NeuralIITSystem".to_string(),
            mechanism,
            cause_effect_repertoire: CauseEffectRepertoire {
                cause_repertoire: HashMap::new(),
                effect_repertoire: HashMap::new(),
                cause_effect_information: 0.0,
                repertoire_entropy: 0.0,
                information_integration: 0.0,
            },
            phi_calculation: PhiCalculation {
                phi_value: 0.0,
                cause_phi: 0.0,
                effect_phi: 0.0,
                integration_level: 0.0,
                complexity_measure: 0.0,
                consciousness_threshold: 0.1,
                calculation_method: PhiMethod::Approximate,
            },
            consciousness_measures: ConsciousnessMeasures {
                phi_consciousness: 0.0,
                integration_consciousness: 0.0,
                complexity_consciousness: 0.0,
                subjective_consciousness: 0.0,
                overall_consciousness: 0.0,
                consciousness_components: HashMap::new(),
            },
            integration_history: Vec::new(),
        }
    }

    /// Update the IIT system for one time step
    pub fn update(&mut self, delta_time: f64, external_input: &IITInput) -> Result<IITOutput> {
        // 1. Update mechanism state
        self.update_mechanism_state(delta_time, external_input)?;

        // 2. Calculate cause-effect repertoires
        self.calculate_cause_effect_repertoires()?;

        // 3. Calculate Φ (Phi) value
        self.calculate_phi()?;

        // 4. Update consciousness measures
        self.update_consciousness_measures()?;

        // 5. Record integration event
        let integration_event = IntegrationEvent {
            timestamp: external_input.timestamp,
            phi_value: self.phi_calculation.phi_value,
            consciousness_level: self.consciousness_measures.overall_consciousness,
            mechanism_state: self.mechanism.state.clone(),
            integration_components: self.consciousness_measures.consciousness_components.clone(),
            subjective_experience: self.generate_subjective_experience(),
        };

        self.integration_history.push(integration_event.clone());

        Ok(IITOutput {
            phi_value: self.phi_calculation.phi_value,
            consciousness_level: self.consciousness_measures.overall_consciousness,
            integration_measure: self.phi_calculation.integration_level,
            complexity_measure: self.phi_calculation.complexity_measure,
            cause_effect_information: self.cause_effect_repertoire.cause_effect_information,
            subjective_experience: integration_event.subjective_experience,
            mechanism_entropy: self.calculate_mechanism_entropy()?,
        })
    }

    /// Update mechanism state
    fn update_mechanism_state(&mut self, delta_time: f64, external_input: &IITInput) -> Result<()> {
        let mut new_state = self.mechanism.state.clone();

        // First pass: collect data needed for updates
        let mut element_updates: Vec<(usize, f64, f64)> = Vec::new();
        
        for (i, _element) in self.mechanism.elements.iter().enumerate() {
            let mut input_sum = 0.0;
            let mut connection_count = 0;

            // Sum inputs from connected elements
            for connection in &self.mechanism.connections {
                if connection.target == self.mechanism.elements[i].id {
                    if let Some(source_idx) = self.mechanism.elements.iter().position(|e| e.id == connection.source) {
                        input_sum += self.mechanism.state[source_idx] * connection.weight * connection.strength;
                        connection_count += 1;
                    }
                }
            }

            // Add external input
            if i < external_input.sensory_data.len() {
                input_sum += external_input.sensory_data[i] * 0.1;
            }

            // Calculate new state
            let activation = 1.0 / (1.0 + (-input_sum).exp());
            let new_element_state = self.mechanism.elements[i].state * (1.0 - delta_time) + activation * delta_time;
            
            element_updates.push((i, new_element_state, input_sum));
        }

        // Second pass: apply updates
        for (i, new_state_val, _input_sum) in element_updates {
            self.mechanism.elements[i].state = new_state_val;
            new_state[i] = new_state_val;

            // Update information content
            self.mechanism.elements[i].information_content = self.calculate_element_information(&self.mechanism.elements[i])?;
        }

        self.mechanism.state = new_state;

        // Update transition matrix
        self.update_transition_matrix()?;

        Ok(())
    }

    /// Calculate information content of an element
    fn calculate_element_information(&self, element: &Element) -> Result<f64> {
        let state_prob = element.state;
        let entropy = if state_prob > 0.0 && state_prob < 1.0 {
            -state_prob * state_prob.ln() - (1.0 - state_prob) * (1.0 - state_prob).ln()
        } else {
            0.0
        };
        Ok(entropy)
    }

    /// Update transition matrix based on current state
    fn update_transition_matrix(&mut self) -> Result<()> {
        let n = self.mechanism.elements.len();
        let mut new_matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Calculate transition probability based on connection strength
                    let connection_strength = self.mechanism.connections.iter()
                        .filter(|c| c.source == self.mechanism.elements[i].id && c.target == self.mechanism.elements[j].id)
                        .map(|c| c.strength)
                        .sum::<f64>();

                    new_matrix[(i, j)] = connection_strength / n as f64;
                } else {
                    // Self-transition probability
                    new_matrix[(i, j)] = 0.5;
                }
            }
        }

        self.mechanism.transition_matrix = new_matrix;
        Ok(())
    }

    /// Calculate cause-effect repertoires
    fn calculate_cause_effect_repertoires(&mut self) -> Result<()> {
        self.cause_effect_repertoire.cause_repertoire.clear();
        self.cause_effect_repertoire.effect_repertoire.clear();

        for element in &self.mechanism.elements {
            // Calculate cause repertoire (what could have caused current state)
            let cause_repertoire = self.calculate_cause_repertoire(element)?;
            self.cause_effect_repertoire.cause_repertoire.insert(element.id, cause_repertoire);

            // Calculate effect repertoire (what current state could cause)
            let effect_repertoire = self.calculate_effect_repertoire(element)?;
            self.cause_effect_repertoire.effect_repertoire.insert(element.id, effect_repertoire);
        }

        // Calculate overall cause-effect information
        self.cause_effect_repertoire.cause_effect_information = self.calculate_cause_effect_information()?;
        self.cause_effect_repertoire.repertoire_entropy = self.calculate_repertoire_entropy()?;
        self.cause_effect_repertoire.information_integration = self.calculate_information_integration()?;

        Ok(())
    }

    /// Calculate cause repertoire for an element
    fn calculate_cause_repertoire(&self, element: &Element) -> Result<DMatrix<f64>> {
        let n_states = element.possible_states.len();
        let mut cause_matrix = DMatrix::zeros(n_states, n_states);

        for (i, &cause_state) in element.possible_states.iter().enumerate() {
            for (j, &effect_state) in element.possible_states.iter().enumerate() {
                // Simplified cause-effect relationship
                let transition_prob = if (cause_state - effect_state).abs() < 0.1 {
                    0.8 // High probability for similar states
                } else {
                    0.2 // Low probability for different states
                };
                cause_matrix[(i, j)] = transition_prob;
            }
        }

        // Normalize
        let sum = cause_matrix.sum();
        if sum > 0.0 {
            cause_matrix /= sum;
        }

        Ok(cause_matrix)
    }

    /// Calculate effect repertoire for an element
    fn calculate_effect_repertoire(&self, element: &Element) -> Result<DMatrix<f64>> {
        let n_states = element.possible_states.len();
        let mut effect_matrix = DMatrix::zeros(n_states, n_states);

        for (i, &current_state) in element.possible_states.iter().enumerate() {
            for (j, &future_state) in element.possible_states.iter().enumerate() {
                // Simplified effect relationship based on current state
                let transition_prob = if (current_state - future_state).abs() < 0.1 {
                    0.7 // High probability for similar states
                } else {
                    0.3 // Low probability for different states
                };
                effect_matrix[(i, j)] = transition_prob;
            }
        }

        // Normalize
        let sum = effect_matrix.sum();
        if sum > 0.0 {
            effect_matrix /= sum;
        }

        Ok(effect_matrix)
    }

    /// Calculate cause-effect information
    fn calculate_cause_effect_information(&self) -> Result<f64> {
        let mut total_information = 0.0;
        let mut count = 0;

        for (cause_id, cause_matrix) in &self.cause_effect_repertoire.cause_repertoire {
            if let Some(effect_matrix) = self.cause_effect_repertoire.effect_repertoire.get(cause_id) {
                // Calculate mutual information between cause and effect
                let mutual_info = self.calculate_mutual_information(cause_matrix, effect_matrix)?;
                total_information += mutual_info;
                count += 1;
            }
        }

        Ok(if count > 0 { total_information / count as f64 } else { 0.0 })
    }

    /// Calculate mutual information between two matrices
    fn calculate_mutual_information(&self, matrix1: &DMatrix<f64>, matrix2: &DMatrix<f64>) -> Result<f64> {
        let mut mutual_info = 0.0;

        for i in 0..matrix1.nrows() {
            for j in 0..matrix1.ncols() {
                let p1 = matrix1[(i, j)];
                let p2 = matrix2[(i, j)];
                
                if p1 > 0.0 && p2 > 0.0 {
                    let joint_prob = p1 * p2;
                    if joint_prob > 0.0 {
                        mutual_info += joint_prob * (joint_prob / (p1 * p2)).ln();
                    }
                }
            }
        }

        Ok(mutual_info)
    }

    /// Calculate repertoire entropy
    fn calculate_repertoire_entropy(&self) -> Result<f64> {
        let mut total_entropy = 0.0;
        let mut count = 0;

        for (_, cause_matrix) in &self.cause_effect_repertoire.cause_repertoire {
            let entropy = self.calculate_matrix_entropy(cause_matrix)?;
            total_entropy += entropy;
            count += 1;
        }

        for (_, effect_matrix) in &self.cause_effect_repertoire.effect_repertoire {
            let entropy = self.calculate_matrix_entropy(effect_matrix)?;
            total_entropy += entropy;
            count += 1;
        }

        Ok(if count > 0 { total_entropy / count as f64 } else { 0.0 })
    }

    /// Calculate entropy of a matrix
    fn calculate_matrix_entropy(&self, matrix: &DMatrix<f64>) -> Result<f64> {
        let mut entropy = 0.0;

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let p = matrix[(i, j)];
                if p > 0.0 {
                    entropy -= p * p.ln();
                }
            }
        }

        Ok(entropy)
    }

    /// Calculate information integration
    fn calculate_information_integration(&self) -> Result<f64> {
        let mut integration = 0.0;

        // Calculate integration across different partitions
        for partition_size in 2..=self.mechanism.elements.len().min(5) {
            let partition_integration = self.calculate_partition_integration(partition_size)?;
            integration += partition_integration;
        }

        Ok(integration)
    }

    /// Calculate integration for a specific partition size
    fn calculate_partition_integration(&self, partition_size: usize) -> Result<f64> {
        let mut max_integration: f64 = 0.0;

        // Generate random partitions and calculate integration
        for _ in 0..10 { // Sample 10 random partitions
            let partition = self.generate_random_partition(partition_size)?;
            let integration = self.calculate_partition_phi(&partition)?;
            max_integration = max_integration.max(integration);
        }

        Ok(max_integration)
    }

    /// Generate a random partition of elements
    fn generate_random_partition(&self, partition_size: usize) -> Result<Vec<Vec<Uuid>>> {
        let mut rng = thread_rng();
        let mut elements: Vec<Uuid> = self.mechanism.elements.iter().map(|e| e.id).collect();
        let mut partitions = Vec::new();

        while elements.len() >= partition_size {
            let mut partition = Vec::new();
            for _ in 0..partition_size {
                if let Some(idx) = (0..elements.len()).collect::<Vec<_>>().choose(&mut rng) {
                    partition.push(elements.remove(*idx));
                }
            }
            partitions.push(partition);
        }

        // Add remaining elements to last partition
        if !elements.is_empty() {
            partitions.push(elements);
        }

        Ok(partitions)
    }

    /// Calculate Φ for a specific partition
    fn calculate_partition_phi(&self, partition: &[Vec<Uuid>]) -> Result<f64> {
        let mut phi = 0.0;

        for subset in partition {
            if subset.len() > 1 {
                let subset_phi = self.calculate_subset_phi(subset)?;
                phi += subset_phi;
            }
        }

        Ok(phi)
    }

    /// Calculate Φ for a subset of elements
    fn calculate_subset_phi(&self, subset: &[Uuid]) -> Result<f64> {
        // Simplified Φ calculation for subset
        let mut information = 0.0;
        let mut integration = 0.0;

        for &element_id in subset {
            if let Some(element) = self.mechanism.elements.iter().find(|e| e.id == element_id) {
                information += element.information_content;
            }
        }

        // Calculate integration based on connections within subset
        for connection in &self.mechanism.connections {
            if subset.contains(&connection.source) && subset.contains(&connection.target) {
                integration += connection.strength * connection.weight.abs();
            }
        }

        Ok((information * integration).max(0.0))
    }

    /// Calculate Φ (Phi) value
    fn calculate_phi(&mut self) -> Result<()> {
        match self.phi_calculation.calculation_method {
            PhiMethod::Approximate => {
                self.calculate_approximate_phi()?;
            },
            PhiMethod::MonteCarlo => {
                self.calculate_monte_carlo_phi()?;
            },
            _ => {
                self.calculate_approximate_phi()?;
            }
        }

        Ok(())
    }

    /// Calculate approximate Φ value
    fn calculate_approximate_phi(&mut self) -> Result<()> {
        // Calculate cause Φ
        self.phi_calculation.cause_phi = self.cause_effect_repertoire.cause_effect_information * 0.5;

        // Calculate effect Φ
        self.phi_calculation.effect_phi = self.cause_effect_repertoire.information_integration * 0.5;

        // Overall Φ is minimum of cause and effect Φ
        self.phi_calculation.phi_value = self.phi_calculation.cause_phi.min(self.phi_calculation.effect_phi);

        // Integration level
        self.phi_calculation.integration_level = self.cause_effect_repertoire.information_integration;

        // Complexity measure
        self.phi_calculation.complexity_measure = self.calculate_complexity_measure()?;

        Ok(())
    }

    /// Calculate Monte Carlo Φ value
    fn calculate_monte_carlo_phi(&mut self) -> Result<()> {
        let mut total_phi = 0.0;
        let num_samples = 100;

        for _ in 0..num_samples {
            let sample_phi = self.calculate_single_phi_sample()?;
            total_phi += sample_phi;
        }

        self.phi_calculation.phi_value = total_phi / num_samples as f64;
        Ok(())
    }

    /// Calculate a single Φ sample
    fn calculate_single_phi_sample(&self) -> Result<f64> {
        // Simplified single sample calculation
        let information = self.cause_effect_repertoire.cause_effect_information;
        let integration = self.cause_effect_repertoire.information_integration;
        Ok((information * integration).max(0.0))
    }

    /// Calculate complexity measure
    fn calculate_complexity_measure(&self) -> Result<f64> {
        let mut complexity = 0.0;

        // Integration complexity
        complexity += self.phi_calculation.integration_level * 0.4;

        // Information complexity
        complexity += self.cause_effect_repertoire.cause_effect_information * 0.3;

        // Entropy complexity
        complexity += self.cause_effect_repertoire.repertoire_entropy * 0.3;

        Ok(complexity)
    }

    /// Update consciousness measures
    fn update_consciousness_measures(&mut self) -> Result<()> {
        // Φ-based consciousness
        self.consciousness_measures.phi_consciousness = 
            (self.phi_calculation.phi_value / self.phi_calculation.consciousness_threshold).min(1.0);

        // Integration-based consciousness
        self.consciousness_measures.integration_consciousness = 
            (self.phi_calculation.integration_level / 1.0).min(1.0);

        // Complexity-based consciousness
        self.consciousness_measures.complexity_consciousness = 
            (self.phi_calculation.complexity_measure / 1.0).min(1.0);

        // Subjective consciousness (weighted combination)
        self.consciousness_measures.subjective_consciousness = 
            self.consciousness_measures.phi_consciousness * 0.5 +
            self.consciousness_measures.integration_consciousness * 0.3 +
            self.consciousness_measures.complexity_consciousness * 0.2;

        // Overall consciousness
        self.consciousness_measures.overall_consciousness = 
            self.consciousness_measures.subjective_consciousness;

        // Update consciousness components
        self.consciousness_measures.consciousness_components.clear();
        self.consciousness_measures.consciousness_components.insert("phi".to_string(), self.consciousness_measures.phi_consciousness);
        self.consciousness_measures.consciousness_components.insert("integration".to_string(), self.consciousness_measures.integration_consciousness);
        self.consciousness_measures.consciousness_components.insert("complexity".to_string(), self.consciousness_measures.complexity_consciousness);
        self.consciousness_measures.consciousness_components.insert("subjective".to_string(), self.consciousness_measures.subjective_consciousness);

        Ok(())
    }

    /// Calculate mechanism entropy
    fn calculate_mechanism_entropy(&self) -> Result<f64> {
        let mut entropy = 0.0;

        for element in &self.mechanism.elements {
            entropy += element.information_content;
        }

        Ok(entropy)
    }

    /// Generate subjective experience description
    fn generate_subjective_experience(&self) -> String {
        let phi = self.phi_calculation.phi_value;
        let consciousness = self.consciousness_measures.overall_consciousness;

        match (phi, consciousness) {
            (phi, _) if phi < 0.01 => "No consciousness - pure mechanism".to_string(),
            (phi, _) if phi < 0.1 => "Minimal consciousness - basic awareness".to_string(),
            (phi, _) if phi < 0.5 => "Moderate consciousness - self-awareness".to_string(),
            (phi, _) if phi < 1.0 => "High consciousness - complex experience".to_string(),
            _ => "Peak consciousness - transcendent awareness".to_string(),
        }
    }
}

/// Input to IIT system
#[derive(Debug, Clone)]
pub struct IITInput {
    pub timestamp: f64,
    pub sensory_data: DVector<f64>,
    pub attention_focus: Vec<f64>,
    pub memory_context: DVector<f64>,
    pub emotional_state: f64,
}

/// Output from IIT system
#[derive(Debug, Clone)]
pub struct IITOutput {
    pub phi_value: f64,
    pub consciousness_level: f64,
    pub integration_measure: f64,
    pub complexity_measure: f64,
    pub cause_effect_information: f64,
    pub subjective_experience: String,
    pub mechanism_entropy: f64,
}

/// IIT Manager for coordinating multiple IIT systems
#[derive(Debug, Default)]
pub struct IITManager {
    pub systems: HashMap<Uuid, IntegratedInformationSystem>,
    pub global_phi: f64,
    pub consciousness_history: Vec<GlobalConsciousnessEvent>,
}

#[derive(Debug, Clone)]
pub struct GlobalConsciousnessEvent {
    pub timestamp: f64,
    pub global_phi: f64,
    pub active_systems: usize,
    pub average_consciousness: f64,
    pub total_integration: f64,
}

impl IITManager {
    /// Create a new IIT manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an IIT system
    pub fn add_system(&mut self, system: IntegratedInformationSystem) {
        self.systems.insert(system.id, system);
    }

    /// Update all IIT systems
    pub fn update_all_systems(&mut self, delta_time: f64, inputs: &HashMap<Uuid, IITInput>) -> Result<Vec<IITOutput>> {
        let mut outputs = Vec::new();
        let mut total_phi = 0.0;
        let mut total_consciousness = 0.0;
        let mut total_integration = 0.0;

        for (id, system) in &mut self.systems {
            let input = inputs.get(id).cloned().unwrap_or_else(|| IITInput {
                timestamp: 0.0,
                sensory_data: DVector::zeros(10),
                attention_focus: vec![0.0; 5],
                memory_context: DVector::zeros(10),
                emotional_state: 0.0,
            });

            let output = system.update(delta_time, &input)?;
            outputs.push(output.clone());

            total_phi += output.phi_value;
            total_consciousness += output.consciousness_level;
            total_integration += output.integration_measure;
        }

        // Update global Φ
        let active_systems = self.systems.len();
        if active_systems > 0 {
            self.global_phi = total_phi / active_systems as f64;
        }

        // Record global consciousness event
        let global_event = GlobalConsciousnessEvent {
            timestamp: 0.0, // Will be set by caller
            global_phi: self.global_phi,
            active_systems,
            average_consciousness: if active_systems > 0 { total_consciousness / active_systems as f64 } else { 0.0 },
            total_integration: total_integration,
        };

        self.consciousness_history.push(global_event);

        Ok(outputs)
    }

    /// Get IIT summary
    pub fn get_iit_summary(&self) -> IITSummary {
        IITSummary {
            total_systems: self.systems.len(),
            global_phi: self.global_phi,
            average_consciousness: self.consciousness_history.last().map(|e| e.average_consciousness).unwrap_or(0.0),
            total_integration: self.consciousness_history.last().map(|e| e.total_integration).unwrap_or(0.0),
            consciousness_trend: self.calculate_consciousness_trend(),
        }
    }

    /// Calculate consciousness trend
    fn calculate_consciousness_trend(&self) -> f64 {
        if self.consciousness_history.len() < 2 {
            return 0.0;
        }

        let recent = self.consciousness_history.iter().rev().take(10).collect::<Vec<_>>();
        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent.last().unwrap().average_consciousness;
        let last = recent.first().unwrap().average_consciousness;
        (last - first) / recent.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct IITSummary {
    pub total_systems: usize,
    pub global_phi: f64,
    pub average_consciousness: f64,
    pub total_integration: f64,
    pub consciousness_trend: f64,
} 