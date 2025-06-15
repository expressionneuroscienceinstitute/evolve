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
    /// Calculates a simplified "phi" value based on the complexity, integration of data,
    /// and a rudimentary self-awareness component. This is an abstract model for simulation.
    pub fn from_data(sensory_input: &SensoryInput, internal_state: &DVector<f64>) -> Self {
        // Sensory integration component: Weighted sum of visual and audio inputs.
        // Assuming a higher "value" for visual input and a threshold for audio.
        let visual_component: f64 = sensory_input.vision.iter().map(|&v| v as f64).sum();
        let audio_component: f64 = sensory_input.audio.iter().map(|&a| if a > 0.1 { 1.0 } else { 0.0 }).sum();
        let sensory_integration = (0.7 * visual_component + 0.3 * audio_component).max(0.0);

        // Internal state complexity: Uses variance and magnitude of the internal state.
        // A more complex internal state (higher variance, larger magnitude) implies higher complexity.
        let state_variance = internal_state.variance();
        let state_magnitude = internal_state.norm();
        let internal_complexity = (state_variance * state_magnitude).log(1.0 + state_magnitude).max(0.0); // Logarithmic scaling

        // Self-awareness component: A simplified metric based on the average internal state value.
        // High average internal state could represent a more "active" or "self-aware" state.
        let self_awareness_component = internal_state.mean().max(0.0).sqrt();

        // Combine components with arbitrary weights.
        let phi = (sensory_integration * 0.4 + internal_complexity * 0.4 + self_awareness_component * 0.2)
            .powf(1.2) // Non-linear scaling to emphasize higher values
            .min(100.0); // Cap phi at a reasonable maximum for stability

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
    let phi = IntegratedInformation::from_data(sensory_input, internal_state);
    
    // Calculate new metrics based on the internal state and sensory input for consciousness tracking.
    let neural_complexity = internal_state.len() as f64 * internal_state.variance().max(0.1).log(10.0);
    let self_awareness_level = internal_state.mean().powf(2.0).min(1.0); // Simplified self-awareness, capped at 1.0

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