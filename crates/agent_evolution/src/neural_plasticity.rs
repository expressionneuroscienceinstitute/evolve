//! # Agent Evolution: Neural Plasticity Module
//!
//! Revolutionary implementation of neural plasticity, synaptic learning, and network
//! growth for advanced agent learning and adaptation. This module implements
//! cutting-edge research in synaptic plasticity, Hebbian learning, and neural
//! network evolution.
//!
//! Research Basis:
//! - Hebbian Learning Theory (Hebb, 1949)
//! - Spike-Timing Dependent Plasticity (STDP) (Bi & Poo, 1998)
//! - Synaptic Scaling and Homeostasis (Turrigiano, 2008)
//! - Neural Network Growth and Pruning (Huttenlocher, 1990)

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;
use rand::{Rng, thread_rng};

/// Neural plasticity system for synaptic learning and network adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPlasticitySystem {
    pub id: Uuid,
    pub name: String,
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub plasticity_rules: PlasticityRules,
    pub learning_state: LearningState,
    pub network_metrics: NetworkMetrics,
    pub plasticity_history: Vec<PlasticityEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: Uuid,
    pub name: String,
    pub position: [f64; 3],
    pub membrane_potential: f64,
    pub threshold: f64,
    pub resting_potential: f64,
    pub refractory_period: f64,
    pub last_spike_time: f64,
    pub spike_history: VecDeque<SpikeEvent>,
    pub synaptic_inputs: Vec<Uuid>,
    pub synaptic_outputs: Vec<Uuid>,
    pub neuron_type: NeuronType,
    pub plasticity_state: NeuronPlasticityState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub timestamp: f64,
    pub amplitude: f64,
    pub frequency: f64,
    pub plasticity_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Sensory,
    Motor,
    Interneuron,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronPlasticityState {
    pub learning_rate: f64,
    pub synaptic_strength_modulation: f64,
    pub homeostatic_scaling: f64,
    pub metaplasticity: f64,
    pub structural_plasticity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub id: Uuid,
    pub source_neuron: Uuid,
    pub target_neuron: Uuid,
    pub weight: f64,
    pub strength: f64,
    pub conductance: f64,
    pub delay: f64,
    pub plasticity_state: SynapticPlasticityState,
    pub learning_history: Vec<LearningEvent>,
    pub synapse_type: SynapseType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticPlasticityState {
    pub ltp_threshold: f64,      // Long-term potentiation threshold
    pub ltd_threshold: f64,      // Long-term depression threshold
    pub stdp_window: f64,        // STDP time window
    pub calcium_concentration: f64,
    pub nmda_activation: f64,
    pub ampa_activation: f64,
    pub metaplasticity_state: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub timestamp: f64,
    pub event_type: LearningEventType,
    pub magnitude: f64,
    pub weight_change: f64,
    pub calcium_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningEventType {
    LTP,        // Long-term potentiation
    LTD,        // Long-term depression
    STDP,       // Spike-timing dependent plasticity
    Homeostasis, // Synaptic scaling
    Structural, // Structural plasticity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
    Modulatory,
    GapJunction,
    Electrical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityRules {
    pub hebbian_learning: HebbianRule,
    pub stdp_learning: STDPRule,
    pub homeostatic_scaling: HomeostaticRule,
    pub structural_plasticity: StructuralRule,
    pub metaplasticity: MetaplasticityRule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianRule {
    pub enabled: bool,
    pub learning_rate: f64,
    pub correlation_threshold: f64,
    pub weight_decay: f64,
    pub max_weight: f64,
    pub min_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPRule {
    pub enabled: bool,
    pub time_window: f64,
    pub ltp_strength: f64,
    pub ltd_strength: f64,
    pub asymmetry_factor: f64,
    pub calcium_dependency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostaticRule {
    pub enabled: bool,
    pub target_activity: f64,
    pub scaling_rate: f64,
    pub time_constant: f64,
    pub global_scaling: bool,
    pub local_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralRule {
    pub enabled: bool,
    pub growth_rate: f64,
    pub pruning_threshold: f64,
    pub max_connections: usize,
    pub min_connections: usize,
    pub growth_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaplasticityRule {
    pub enabled: bool,
    pub meta_learning_rate: f64,
    pub threshold_modulation: f64,
    pub plasticity_modulation: f64,
    pub adaptation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningState {
    pub current_time: f64,
    pub learning_rate: f64,
    pub global_activity: f64,
    pub average_weight: f64,
    pub plasticity_events: u64,
    pub structural_changes: u64,
    pub learning_phase: LearningPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPhase {
    Initialization,
    Exploration,
    Consolidation,
    Optimization,
    Maintenance,
}

impl Default for LearningPhase {
    fn default() -> Self {
        LearningPhase::Initialization
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub total_neurons: usize,
    pub total_synapses: usize,
    pub average_connectivity: f64,
    pub network_density: f64,
    pub clustering_coefficient: f64,
    pub path_length: f64,
    pub modularity: f64,
    pub efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityEvent {
    pub timestamp: f64,
    pub event_type: PlasticityEventType,
    pub magnitude: f64,
    pub affected_neurons: Vec<Uuid>,
    pub affected_synapses: Vec<Uuid>,
    pub learning_impact: f64,
}

impl Default for PlasticityEvent {
    fn default() -> Self {
        PlasticityEvent {
            timestamp: 0.0,
            event_type: PlasticityEventType::SynapticStrengthening,
            magnitude: 0.0,
            affected_neurons: Vec::new(),
            affected_synapses: Vec::new(),
            learning_impact: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityEventType {
    SynapticStrengthening,
    SynapticWeakening,
    SynapseFormation,
    SynapseElimination,
    NeuronGrowth,
    NeuronPruning,
    HomeostaticScaling,
    MetaplasticityChange,
}

impl NeuralPlasticitySystem {
    /// Create a new neural plasticity system
    pub fn new(num_neurons: usize) -> Self {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();
        let mut rng = thread_rng();

        // Create neurons
        for i in 0..num_neurons {
            let neuron_type = if i <= num_neurons/4 {
                NeuronType::Sensory
            } else if i <= num_neurons/2 {
                NeuronType::Interneuron
            } else if i <= 3*num_neurons/4 {
                NeuronType::Excitatory
            } else {
                NeuronType::Inhibitory
            };

            let neuron = Neuron {
                id: Uuid::new_v4(),
                name: format!("Neuron_{}", i),
                position: [
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                ],
                membrane_potential: rng.gen_range(-70.0..-50.0),
                threshold: rng.gen_range(-55.0..-45.0),
                resting_potential: -65.0,
                refractory_period: rng.gen_range(1.0..5.0),
                last_spike_time: 0.0,
                spike_history: VecDeque::with_capacity(100),
                synaptic_inputs: Vec::new(),
                synaptic_outputs: Vec::new(),
                neuron_type,
                plasticity_state: NeuronPlasticityState {
                    learning_rate: rng.gen_range(0.01..0.1),
                    synaptic_strength_modulation: 1.0,
                    homeostatic_scaling: 1.0,
                    metaplasticity: 0.0,
                    structural_plasticity: 0.0,
                },
            };
            neurons.push(neuron);
        }

        // Create synapses
        for i in 0..neurons.len() {
            for j in 0..neurons.len() {
                if i != j && rng.gen::<f64>() < 0.3 { // 30% connection probability
                    let synapse_type = match (neurons[i].neuron_type.clone(), neurons[j].neuron_type.clone()) {
                        (NeuronType::Excitatory, _) => SynapseType::Excitatory,
                        (NeuronType::Inhibitory, _) => SynapseType::Inhibitory,
                        (NeuronType::Modulatory, _) => SynapseType::Modulatory,
                        _ => SynapseType::Excitatory,
                    };

                    let synapse = Synapse {
                        id: Uuid::new_v4(),
                        source_neuron: neurons[i].id,
                        target_neuron: neurons[j].id,
                        weight: rng.gen_range(0.0..1.0),
                        strength: rng.gen_range(0.1..1.0),
                        conductance: rng.gen_range(0.1..1.0),
                        delay: rng.gen_range(0.1..5.0),
                        plasticity_state: SynapticPlasticityState {
                            ltp_threshold: rng.gen_range(0.1..0.5),
                            ltd_threshold: rng.gen_range(-0.5..-0.1),
                            stdp_window: rng.gen_range(10.0..50.0),
                            calcium_concentration: 0.0,
                            nmda_activation: 0.0,
                            ampa_activation: 0.0,
                            metaplasticity_state: 0.0,
                        },
                        learning_history: Vec::new(),
                        synapse_type,
                    };

                    synapses.push(synapse.clone());
                }
            }
        }

        // Update neuron connection lists
        for synapse in &synapses {
            if let Some(source_neuron) = neurons.iter_mut().find(|n| n.id == synapse.source_neuron) {
                source_neuron.synaptic_outputs.push(synapse.id);
            }
            if let Some(target_neuron) = neurons.iter_mut().find(|n| n.id == synapse.target_neuron) {
                target_neuron.synaptic_inputs.push(synapse.id);
            }
        }

        let plasticity_rules = PlasticityRules {
            hebbian_learning: HebbianRule {
                enabled: true,
                learning_rate: 0.01,
                correlation_threshold: 0.1,
                weight_decay: 0.001,
                max_weight: 2.0,
                min_weight: 0.0,
            },
            stdp_learning: STDPRule {
                enabled: true,
                time_window: 20.0,
                ltp_strength: 0.1,
                ltd_strength: -0.05,
                asymmetry_factor: 1.0,
                calcium_dependency: true,
            },
            homeostatic_scaling: HomeostaticRule {
                enabled: true,
                target_activity: 0.1,
                scaling_rate: 0.001,
                time_constant: 1000.0,
                global_scaling: true,
                local_scaling: false,
            },
            structural_plasticity: StructuralRule {
                enabled: true,
                growth_rate: 0.001,
                pruning_threshold: 0.01,
                max_connections: 100,
                min_connections: 1,
                growth_factor: 1.0,
            },
            metaplasticity: MetaplasticityRule {
                enabled: true,
                meta_learning_rate: 0.001,
                threshold_modulation: 0.1,
                plasticity_modulation: 0.1,
                adaptation_rate: 0.01,
            },
        };

        Self {
            id: Uuid::new_v4(),
            name: "NeuralPlasticitySystem".to_string(),
            neurons: neurons.clone(),
            synapses: synapses.clone(),
            plasticity_rules,
            learning_state: LearningState {
                current_time: 0.0,
                learning_rate: 0.01,
                global_activity: 0.0,
                average_weight: 0.0,
                plasticity_events: 0,
                structural_changes: 0,
                learning_phase: LearningPhase::Initialization,
            },
            network_metrics: NetworkMetrics {
                total_neurons: neurons.len(),
                total_synapses: synapses.len(),
                average_connectivity: 0.0,
                network_density: 0.0,
                clustering_coefficient: 0.0,
                path_length: 0.0,
                modularity: 0.0,
                efficiency: 0.0,
            },
            plasticity_history: Vec::new(),
        }
    }

    /// Update the neural plasticity system for one time step
    pub fn update(&mut self, delta_time: f64, input: &PlasticityInput) -> Result<PlasticityOutput> {
        // Update current time
        self.learning_state.current_time += delta_time;

        // Process input
        self.process_input(input)?;

        // Collect neuron indices for processing
        let neuron_indices: Vec<usize> = (0..self.neurons.len()).collect();

        // Update neurons
        for &idx in &neuron_indices {
            let neuron = &mut self.neurons[idx];
            
            // Update membrane potential
            neuron.membrane_potential += delta_time * 0.1;
            
            // Check for spike generation
            if neuron.membrane_potential > neuron.threshold {
                generate_spike_internal(neuron, input.timestamp)?;
            }
        }

        // Apply plasticity rules
        self.apply_plasticity_rules(delta_time)?;

        // Update network structure
        self.update_network_structure(delta_time)?;

        // Update network metrics
        self.update_network_metrics()?;

        // Calculate learning impact
        let learning_impact = self.calculate_learning_impact()?;

        // Create plasticity event
        let plasticity_event = PlasticityEvent {
            timestamp: input.timestamp,
            event_type: PlasticityEventType::SynapticStrengthening,
            magnitude: learning_impact,
            affected_neurons: self.neurons.iter().map(|n| n.id).collect(),
            affected_synapses: self.synapses.iter().map(|s| s.id).collect(),
            learning_impact,
        };

        self.plasticity_history.push(plasticity_event.clone());

        Ok(PlasticityOutput {
            learning_rate: self.learning_state.learning_rate,
            global_activity: self.learning_state.global_activity,
            average_weight: self.learning_state.average_weight,
            plasticity_events: self.learning_state.plasticity_events,
            structural_changes: self.learning_state.structural_changes,
            network_efficiency: self.network_metrics.efficiency,
            learning_phase: self.learning_state.learning_phase.clone(),
            plasticity_event,
        })
    }

    /// Process input and generate spikes
    fn process_input(&mut self, input: &PlasticityInput) -> Result<()> {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if i < input.sensory_inputs.len() {
                // Add sensory input to membrane potential
                neuron.membrane_potential += input.sensory_inputs[i] * 0.1;

                // Check for spike generation
                if neuron.membrane_potential > neuron.threshold {
                    generate_spike_internal(neuron, input.timestamp)?;
                }
            }
        }
        Ok(())
    }

    /// Apply plasticity rules
    fn apply_plasticity_rules(&mut self, delta_time: f64) -> Result<()> {
        // Apply Hebbian learning
        if self.plasticity_rules.hebbian_learning.enabled {
            self.apply_hebbian_learning(delta_time)?;
        }

        // Apply STDP learning
        if self.plasticity_rules.stdp_learning.enabled {
            self.apply_stdp_learning(delta_time)?;
        }

        // Apply homeostatic scaling
        if self.plasticity_rules.homeostatic_scaling.enabled {
            self.apply_homeostatic_scaling(delta_time)?;
        }

        // Apply metaplasticity
        if self.plasticity_rules.metaplasticity.enabled {
            self.apply_metaplasticity(delta_time)?;
        }

        Ok(())
    }

    /// Apply Hebbian learning rule
    fn apply_hebbian_learning(&mut self, delta_time: f64) -> Result<()> {
        for synapse in &mut self.synapses {
            if let (Some(source_neuron), Some(target_neuron)) = (
                self.neurons.iter().find(|n| n.id == synapse.source_neuron),
                self.neurons.iter().find(|n| n.id == synapse.target_neuron)
            ) {
                // Calculate correlation between pre- and post-synaptic activity
                let source_activity = if source_neuron.spike_history.len() > 0 {
                    source_neuron.spike_history.back().unwrap().frequency
                } else {
                    0.0
                };

                let target_activity = if target_neuron.spike_history.len() > 0 {
                    target_neuron.spike_history.back().unwrap().frequency
                } else {
                    0.0
                };

                let correlation = source_activity * target_activity;

                if correlation > self.plasticity_rules.hebbian_learning.correlation_threshold {
                    // Long-term potentiation
                    let weight_change = self.plasticity_rules.hebbian_learning.learning_rate * correlation * delta_time;
                    synapse.weight = (synapse.weight + weight_change).min(self.plasticity_rules.hebbian_learning.max_weight);

                    // Record learning event
                    let learning_event = LearningEvent {
                        timestamp: self.learning_state.current_time,
                        event_type: LearningEventType::LTP,
                        magnitude: weight_change,
                        weight_change,
                        calcium_level: synapse.plasticity_state.calcium_concentration,
                    };
                    synapse.learning_history.push(learning_event);

                    self.learning_state.plasticity_events += 1;
                } else if correlation < -self.plasticity_rules.hebbian_learning.correlation_threshold {
                    // Long-term depression
                    let weight_change = -self.plasticity_rules.hebbian_learning.learning_rate * correlation.abs() * delta_time;
                    synapse.weight = (synapse.weight + weight_change).max(self.plasticity_rules.hebbian_learning.min_weight);

                    let learning_event = LearningEvent {
                        timestamp: self.learning_state.current_time,
                        event_type: LearningEventType::LTD,
                        magnitude: weight_change.abs(),
                        weight_change,
                        calcium_level: synapse.plasticity_state.calcium_concentration,
                    };
                    synapse.learning_history.push(learning_event);

                    self.learning_state.plasticity_events += 1;
                }
            }
        }
        Ok(())
    }

    /// Apply STDP learning rule
    fn apply_stdp_learning(&mut self, delta_time: f64) -> Result<()> {
        for synapse in &mut self.synapses {
            if let (Some(source_neuron), Some(target_neuron)) = (
                self.neurons.iter().find(|n| n.id == synapse.source_neuron),
                self.neurons.iter().find(|n| n.id == synapse.target_neuron)
            ) {
                // Calculate spike timing difference
                if let (Some(source_spike), Some(target_spike)) = (
                    source_neuron.spike_history.back(),
                    target_neuron.spike_history.back()
                ) {
                    let time_diff = source_spike.timestamp - target_spike.timestamp;

                    if time_diff.abs() < self.plasticity_rules.stdp_learning.time_window {
                        let weight_change = if time_diff > 0.0 {
                            // Pre-synaptic spike before post-synaptic spike (LTP)
                            self.plasticity_rules.stdp_learning.ltp_strength * (-time_diff / self.plasticity_rules.stdp_learning.time_window).exp()
                        } else {
                            // Post-synaptic spike before pre-synaptic spike (LTD)
                            self.plasticity_rules.stdp_learning.ltd_strength * (time_diff / self.plasticity_rules.stdp_learning.time_window).exp()
                        };

                        synapse.weight = (synapse.weight + weight_change).max(0.0).min(2.0);

                        let learning_event = LearningEvent {
                            timestamp: self.learning_state.current_time,
                            event_type: LearningEventType::STDP,
                            magnitude: weight_change.abs(),
                            weight_change,
                            calcium_level: synapse.plasticity_state.calcium_concentration,
                        };
                        synapse.learning_history.push(learning_event);

                        self.learning_state.plasticity_events += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply homeostatic scaling
    fn apply_homeostatic_scaling(&mut self, delta_time: f64) -> Result<()> {
        let target_activity = self.plasticity_rules.homeostatic_scaling.target_activity;
        let current_activity = self.learning_state.global_activity;
        let scaling_rate = self.plasticity_rules.homeostatic_scaling.scaling_rate;

        let scaling_factor = if current_activity > target_activity {
            // Scale down weights to reduce activity
            1.0 - scaling_rate * delta_time
        } else {
            // Scale up weights to increase activity
            1.0 + scaling_rate * delta_time
        };

        for synapse in &mut self.synapses {
            synapse.weight *= scaling_factor;
        }

        Ok(())
    }

    /// Apply metaplasticity
    fn apply_metaplasticity(&mut self, delta_time: f64) -> Result<()> {
        for synapse in &mut self.synapses {
            // Update metaplasticity state based on learning history
            let recent_learning = synapse.learning_history.iter()
                .rev()
                .take(10)
                .map(|e| e.magnitude)
                .sum::<f64>();

            synapse.plasticity_state.metaplasticity_state += 
                self.plasticity_rules.metaplasticity.meta_learning_rate * recent_learning * delta_time;

            // Adjust plasticity thresholds based on metaplasticity state
            synapse.plasticity_state.ltp_threshold += 
                synapse.plasticity_state.metaplasticity_state * self.plasticity_rules.metaplasticity.threshold_modulation * delta_time;
            synapse.plasticity_state.ltd_threshold -= 
                synapse.plasticity_state.metaplasticity_state * self.plasticity_rules.metaplasticity.threshold_modulation * delta_time;
        }
        Ok(())
    }

    /// Update network structure
    fn update_network_structure(&mut self, delta_time: f64) -> Result<()> {
        if self.plasticity_rules.structural_plasticity.enabled {
            // Synapse formation
            self.form_new_synapses(delta_time)?;

            // Synapse elimination
            self.eliminate_weak_synapses(delta_time)?;

            // Neuron growth/pruning
            self.update_neuron_structure(delta_time)?;
        }
        Ok(())
    }

    /// Form new synapses
    fn form_new_synapses(&mut self, delta_time: f64) -> Result<()> {
        let growth_rate = self.plasticity_rules.structural_plasticity.growth_rate;
        let max_connections = self.plasticity_rules.structural_plasticity.max_connections;

        for neuron in &self.neurons {
            if neuron.synaptic_outputs.len() < max_connections {
                // Find potential new connections
                for other_neuron in &self.neurons {
                    if neuron.id != other_neuron.id && 
                       !neuron.synaptic_outputs.iter().any(|&syn_id| {
                           self.synapses.iter().find(|s| s.id == syn_id).map(|s| s.target_neuron == other_neuron.id).unwrap_or(false)
                       }) {
                        
                        if thread_rng().gen::<f64>() < growth_rate * delta_time {
                            // Create new synapse
                            let new_synapse = Synapse {
                                id: Uuid::new_v4(),
                                source_neuron: neuron.id,
                                target_neuron: other_neuron.id,
                                weight: thread_rng().gen_range(0.0..0.5),
                                strength: thread_rng().gen_range(0.1..0.5),
                                conductance: thread_rng().gen_range(0.1..0.5),
                                delay: thread_rng().gen_range(0.1..2.0),
                                plasticity_state: SynapticPlasticityState {
                                    ltp_threshold: 0.3,
                                    ltd_threshold: -0.3,
                                    stdp_window: 20.0,
                                    calcium_concentration: 0.0,
                                    nmda_activation: 0.0,
                                    ampa_activation: 0.0,
                                    metaplasticity_state: 0.0,
                                },
                                learning_history: Vec::new(),
                                synapse_type: SynapseType::Excitatory,
                            };

                            self.synapses.push(new_synapse.clone());
                            self.learning_state.structural_changes += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Eliminate weak synapses
    fn eliminate_weak_synapses(&mut self, delta_time: f64) -> Result<()> {
        let pruning_threshold = self.plasticity_rules.structural_plasticity.pruning_threshold;
        let min_connections = self.plasticity_rules.structural_plasticity.min_connections;

        self.synapses.retain(|synapse| {
            let should_eliminate = synapse.weight < pruning_threshold && 
                                 self.neurons.iter().find(|n| n.id == synapse.source_neuron)
                                     .map(|n| n.synaptic_outputs.len() > min_connections)
                                     .unwrap_or(false);

            if should_eliminate {
                self.learning_state.structural_changes += 1;
            }

            !should_eliminate
        });
        Ok(())
    }

    /// Update neuron structure
    fn update_neuron_structure(&mut self, delta_time: f64) -> Result<()> {
        for neuron in &mut self.neurons {
            // Update structural plasticity based on activity
            let activity = if neuron.spike_history.len() > 0 {
                neuron.spike_history.back().unwrap().frequency
            } else {
                0.0
            };

            neuron.plasticity_state.structural_plasticity += 
                self.plasticity_rules.structural_plasticity.growth_factor * activity * delta_time;
        }
        Ok(())
    }

    /// Update network metrics
    fn update_network_metrics(&mut self) -> Result<()> {
        self.network_metrics.total_neurons = self.neurons.len();
        self.network_metrics.total_synapses = self.synapses.len();

        // Calculate average connectivity
        let total_connections: usize = self.neurons.iter().map(|n| n.synaptic_outputs.len()).sum();
        self.network_metrics.average_connectivity = total_connections as f64 / self.neurons.len() as f64;

        // Calculate network density
        let max_possible_connections = self.neurons.len() * (self.neurons.len() - 1);
        self.network_metrics.network_density = self.synapses.len() as f64 / max_possible_connections as f64;

        // Calculate average weight
        let total_weight: f64 = self.synapses.iter().map(|s| s.weight).sum();
        self.learning_state.average_weight = total_weight / self.synapses.len() as f64;

        // Calculate efficiency (simplified)
        self.network_metrics.efficiency = self.network_metrics.network_density * self.learning_state.average_weight;

        Ok(())
    }

    /// Calculate learning impact
    fn calculate_learning_impact(&self) -> Result<f64> {
        let mut impact = 0.0;

        // Weight changes
        for synapse in &self.synapses {
            if let Some(recent_event) = synapse.learning_history.last() {
                impact += recent_event.magnitude;
            }
        }

        // Structural changes
        impact += self.learning_state.structural_changes as f64 * 0.1;

        // Activity changes
        impact += self.learning_state.global_activity * 0.5;

        Ok(impact)
    }
}

/// Input to neural plasticity system
#[derive(Debug, Clone)]
pub struct PlasticityInput {
    pub timestamp: f64,
    pub sensory_inputs: Vec<f64>,
    pub learning_signal: f64,
    pub attention_focus: Vec<f64>,
    pub reward_signal: f64,
}

/// Output from neural plasticity system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlasticityOutput {
    pub learning_rate: f64,
    pub global_activity: f64,
    pub average_weight: f64,
    pub plasticity_events: u64,
    pub structural_changes: u64,
    pub network_efficiency: f64,
    pub learning_phase: LearningPhase,
    pub plasticity_event: PlasticityEvent,
}

/// Neural Plasticity Manager for coordinating multiple plasticity systems
#[derive(Debug, Default)]
pub struct NeuralPlasticityManager {
    pub systems: HashMap<Uuid, NeuralPlasticitySystem>,
    pub global_learning_rate: f64,
    pub plasticity_history: Vec<GlobalPlasticityEvent>,
}

#[derive(Debug, Clone)]
pub struct GlobalPlasticityEvent {
    pub timestamp: f64,
    pub global_learning_rate: f64,
    pub active_systems: usize,
    pub average_efficiency: f64,
    pub total_plasticity_events: u64,
}

impl NeuralPlasticityManager {
    /// Create a new neural plasticity manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a plasticity system
    pub fn add_system(&mut self, system: NeuralPlasticitySystem) {
        self.systems.insert(system.id, system);
    }

    /// Update all plasticity systems
    pub fn update_all_systems(&mut self, delta_time: f64, inputs: &HashMap<Uuid, PlasticityInput>) -> Result<Vec<PlasticityOutput>> {
        let mut outputs = Vec::new();
        let mut total_learning_rate = 0.0;
        let mut total_efficiency = 0.0;
        let mut total_plasticity_events = 0;

        for (id, system) in &mut self.systems {
            let input = inputs.get(id).cloned().unwrap_or_else(|| PlasticityInput {
                timestamp: 0.0,
                sensory_inputs: vec![0.0; 10],
                learning_signal: 0.0,
                attention_focus: vec![0.0; 5],
                reward_signal: 0.0,
            });

            let output = system.update(delta_time, &input)?;
            outputs.push(output.clone());

            total_learning_rate += output.learning_rate;
            total_efficiency += output.network_efficiency;
            total_plasticity_events += output.plasticity_events;
        }

        // Update global learning rate
        let active_systems = self.systems.len();
        if active_systems > 0 {
            self.global_learning_rate = total_learning_rate / active_systems as f64;
        }

        // Record global plasticity event
        let global_event = GlobalPlasticityEvent {
            timestamp: 0.0, // Will be set by caller
            global_learning_rate: self.global_learning_rate,
            active_systems,
            average_efficiency: if active_systems > 0 { total_efficiency / active_systems as f64 } else { 0.0 },
            total_plasticity_events,
        };

        self.plasticity_history.push(global_event);

        Ok(outputs)
    }

    /// Get plasticity summary
    pub fn get_plasticity_summary(&self) -> PlasticitySummary {
        PlasticitySummary {
            total_systems: self.systems.len(),
            global_learning_rate: self.global_learning_rate,
            average_efficiency: self.plasticity_history.last().map(|e| e.average_efficiency).unwrap_or(0.0),
            total_plasticity_events: self.plasticity_history.last().map(|e| e.total_plasticity_events).unwrap_or(0),
            learning_trend: self.calculate_learning_trend(),
        }
    }

    /// Calculate learning trend
    fn calculate_learning_trend(&self) -> f64 {
        if self.plasticity_history.len() < 2 {
            return 0.0;
        }

        let recent = self.plasticity_history.iter().rev().take(10).collect::<Vec<_>>();
        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent.last().unwrap().global_learning_rate;
        let last = recent.first().unwrap().global_learning_rate;
        (last - first) / recent.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct PlasticitySummary {
    pub total_systems: usize,
    pub global_learning_rate: f64,
    pub average_efficiency: f64,
    pub total_plasticity_events: u64,
    pub learning_trend: f64,
}

fn generate_spike_internal(neuron: &mut Neuron, timestamp: f64) -> Result<()> {
    // Create spike event
    let spike_event = SpikeEvent {
        timestamp,
        amplitude: 1.0,
        frequency: 1.0 / (timestamp - neuron.last_spike_time).max(0.001),
        plasticity_impact: 0.1,
    };

    // Add to spike history
    neuron.spike_history.push_back(spike_event);

    // Reset membrane potential
    neuron.membrane_potential = neuron.resting_potential;
    neuron.last_spike_time = timestamp;

    Ok(())
} 