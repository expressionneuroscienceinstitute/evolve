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
pub struct IntegratedInformation(f64);

impl IntegratedInformation {
    /// Calculates a simplified "phi" value based on the complexity and integration of data.
    /// This is a highly abstract placeholder for a real IIT calculation.
    pub fn from_data(sensory_input: &SensoryInput, internal_state: &DVector<f64>) -> Self {
        // A mock calculation: combines the number of inputs and the variance of the internal state.
        let input_diversity = (sensory_input.vision.len() + sensory_input.audio.len()) as f64;
        let state_variance = internal_state.variance();
        let phi = (input_diversity * state_variance).log(2.0).max(0.0);
        IntegratedInformation(phi)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Represents the conscious state of an agent at a moment in time.
#[derive(Debug, Clone)]
pub struct ConsciousState {
    pub agent_id: Uuid,
    pub phi: IntegratedInformation,
    pub content: String, // A textual description of the conscious experience
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
    let phi = IntegratedInformation::from_data(sensory_input, internal_state);
    
    // The "content" of consciousness is a narrative interpretation of the state.
    let content = format!(
        "Awareness level: {:.2}. Processing {} visual and {} audio inputs.",
        phi.value(),
        sensory_input.vision.len(),
        sensory_input.audio.len()
    );

    let new_state = ConsciousState { agent_id, phi, content };
    monitor.update_state(agent_id, new_state);

    Ok(())
}