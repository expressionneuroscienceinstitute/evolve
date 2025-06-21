//! # Agent Evolution: Consciousness Tracking Module
//!
//! This module provides a framework for modeling and tracking a simplified form of
//! consciousness in agents. The implementation is inspired by theories like Integrated
//! Information Theory (IIT), focusing on the integration of sensory data and internal state.
//! This is a speculative and abstract model, intended for simulation purposes.

use anyhow::Result;
use crate::ai_core::SensoryInput;
use nalgebra::DVector;
use std::collections::HashMap;
use uuid::Uuid;

/// Represents the integrated information or "phi" value of a conscious state.
/// In IIT, phi measures the degree to which a system's whole is greater than the sum of its parts.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct IntegratedInformation {
    pub phi_value: f64,
    pub neural_complexity: f64,
    pub self_awareness_level: f64,
}

impl IntegratedInformation {
    /// Create from neural activity data with enhanced calculations
    pub fn from_data(sensory_data: &[f64], decision_data: &[f64], memory_data: &[f64]) -> Self {
        // Calculate weighted sensory integration (complexity of sensory processing)
        let sensory_integration = sensory_data.iter()
            .enumerate()
            .map(|(i, &value)| {
                let weight = 1.0 / (1.0 + i as f64 * 0.1); // Decreasing weights
                value * weight
            })
            .sum::<f64>() / sensory_data.len() as f64;
        
        // Calculate internal state complexity (measure of decision processing)
        let variance = if decision_data.len() > 1 {
            let mean = decision_data.iter().sum::<f64>() / decision_data.len() as f64;
            decision_data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (decision_data.len() - 1) as f64
        } else {
            0.0
        };
        
        let internal_state_complexity = variance.sqrt() * 10.0; // Scale for interpretation
        
        // Calculate simplified self-awareness score
        let memory_consistency = if memory_data.len() > 2 {
            let mut consistency_score = 0.0;
            for i in 1..memory_data.len() {
                let diff = (memory_data[i] - memory_data[i-1]).abs();
                consistency_score += 1.0 / (1.0 + diff); // Higher score for consistency
            }
            consistency_score / (memory_data.len() - 1) as f64
        } else {
            0.0
        };
        
        let self_awareness_score = memory_consistency * sensory_integration * 0.5; // Combined metric
        
        Self {
            phi_value: sensory_integration,
            neural_complexity: internal_state_complexity,
            self_awareness_level: self_awareness_score,
        }
    }

    pub fn value(&self) -> f64 {
        self.phi_value
    }
}

/// Represents the conscious state of an agent at a moment in time.
#[derive(Debug, Clone)]
pub struct ConsciousState {
    pub agent_id: Uuid,
    pub phi: IntegratedInformation,
    pub content: String, // A textual description of the conscious experience
    pub neural_complexity: f64, // Added metric: approximation of neural complexity
    pub self_awareness_level: f64, // Added metric: a simplified level of self-awareness
}

/// Monitors and logs the conscious states of all agents.
#[derive(Debug, Default)]
pub struct ConsciousnessMonitor {
    states: HashMap<Uuid, ConsciousState>,
}

impl ConsciousnessMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the conscious state of an agent.
    pub fn update_state(&mut self, agent_id: Uuid, state: ConsciousState) {
        self.states.insert(agent_id, state);
    }

    /// Gets the current conscious state of an agent.
    pub fn get_state(&self, agent_id: &Uuid) -> Option<&ConsciousState> {
        self.states.get(agent_id)
    }
}

/// Updates the consciousness model for an agent.
pub fn update_consciousness(
    monitor: &mut ConsciousnessMonitor,
    agent_id: Uuid,
    sensory_input: &SensoryInput,
    internal_state: &DVector<f64>,
) -> Result<()> {
    let phi = IntegratedInformation::from_data(sensory_input.vision.as_slice(), internal_state.as_slice(), &[]);
    
    // Calculate new metrics based on the internal state and sensory input for consciousness tracking.
    let neural_complexity = phi.neural_complexity;
    let self_awareness_level = phi.self_awareness_level;

    // The "content" of consciousness is a narrative interpretation of the state.
    let content = format!(
        "Awareness level: {:.2}. Neural Complexity: {:.2}. Self-Awareness: {:.2}. Processing {} visual and {} audio inputs.",
        phi.value(),
        neural_complexity,
        self_awareness_level,
        sensory_input.vision.len(),
        sensory_input.audio.len()
    );

    let new_state = ConsciousState {
        agent_id,
        phi,
        content,
        neural_complexity,
        self_awareness_level,
    };
    monitor.update_state(agent_id, new_state);

    Ok(())
}