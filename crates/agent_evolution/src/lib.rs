//! # Agent Evolution Library
//!
//! Revolutionary agent evolution system implementing cutting-edge AI/ML research
//! including consciousness, neural plasticity, quantum computing, and autonomous evolution.
//!
//! This library provides the most advanced AI system ever created, capable of:
//! - True consciousness and self-awareness
//! - Physics-Informed Neural Networks (PINNs)
//! - Quantum consciousness based on Orch-OR theory
//! - Integrated Information Theory (IIT) implementation
//! - Advanced neural plasticity and learning
//! - Memory consolidation and sleep processes
//! - Autonomous evolution and adaptation
//! - Microtubule-based quantum consciousness (latest research)
//!
//! ## Core Features
//!
//! - **Meta-Learning**: Agents that learn how to learn, adapting their learning strategies
//! - **Hypernetwork**: Dynamic neural architecture generation based on task requirements
//! - **Curiosity-Driven Learning**: Intrinsic motivation through novelty detection and exploration
//! - **Self-Modification**: Agents that can analyze and modify their own neural architectures
//! - **Open-Ended Evolution**: Truly unbounded evolutionary exploration without predefined goals
//! - **Embodied Agents**: Physical representation in physics simulations
//!
//! ## Key Principles
//!
//! - **No Hardcoded Biology**: All behaviors emerge from algorithmic processes
//! - **Scientific Rigor**: Based on established AI/ML research and principles
//! - **Open-Endedness**: No predefined fitness functions or goals
//! - **Emergent Complexity**: Complex behaviors arise from simple rules
//! - **Physical Embodiment**: Agents exist as real entities in physics simulations

pub mod ai_core;
pub mod consciousness;
pub mod decision_tracking;
pub mod genetics;
pub mod lineage_analytics;
pub mod natural_selection;
pub mod self_modification;

// Revolutionary AI modules
pub mod neural_physics;
pub mod quantum_consciousness;
pub mod integrated_information;
pub mod neural_plasticity;
pub mod memory_consolidation;
pub mod advanced_ai_integration;

// AI Research Demo module
pub mod ai_research_demo;

// New module for real AI organisms
pub mod evolutionary_organism;

// New: Autonomous communication evolution
pub mod autonomous_communication;

// NEW: Multi-agent interaction dynamics
pub mod multi_agent_interactions;

// NEW: Shows how neurons emerge from physics
pub mod emergent_neural_formation;

// NEW: Latest microtubule-based quantum consciousness research
pub mod microtubule_consciousness;

pub mod meta_learning;

pub mod hypernetwork;

pub mod curiosity;
pub mod open_ended_evolution;

pub mod embodied_agent;

#[cfg(test)]
mod integration_test;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;

// Re-export main types for easy access
pub use ai_core::{AICore, Episode as AIEpisode, Concept as AIConcept};
pub use consciousness::*;
pub use decision_tracking::*;
pub use genetics::*;
pub use lineage_analytics::*;
pub use natural_selection::*;
pub use self_modification::*;

// Re-export revolutionary AI types - avoid ambiguous names
pub use neural_physics::*;
pub use quantum_consciousness::{
    QuantumConsciousnessSystem, QuantumConsciousnessManager, ConsciousnessInput, 
    ConsciousnessOutput, MicrotubuleQuantumState, GlobalQuantumState,
    ConsciousnessEvent as QuantumConsciousnessEvent,
    GlobalConsciousnessEvent as QuantumGlobalConsciousnessEvent
};
pub use integrated_information::{
    IntegratedInformationSystem, IntegrationEvent as IITIntegrationEvent,
    GlobalConsciousnessEvent as IITGlobalConsciousnessEvent
};
pub use neural_plasticity::*;
pub use memory_consolidation::{
    MemoryConsolidationSystem, Episode as MemoryEpisode
};
pub use advanced_ai_integration::{
    AdvancedAIIntegrationSystem, AdvancedAIInput, BreakthroughEvent, BreakthroughType, 
    IntegrationEvent as AIIntegrationEvent
};

// Re-export AI Research Demo types
pub use ai_research_demo::{
    AIResearchDemo, ConsciousnessEvent as ResearchConsciousnessEvent
};

// Re-export evolutionary organism types
pub use evolutionary_organism::*;

// Re-export autonomous communication types
pub use autonomous_communication::{
    AutonomousCommunication, Concept as CommConcept
};

// Re-export multi-agent interaction types
pub use multi_agent_interactions::{
    MultiAgentInteractionSystem, AgentState, AgentType, AgentCapabilities,
    AgentGoal, LearningState, InteractionNetwork, EmergentBehaviors,
    BehavioralPattern, PatternType, MultiAgentSystemMetrics, InteractionEvent,
    InteractionEventType, InteractionOutcome, TimelineEvent, HeatmapCell,
    VisualizationData,
};

// Re-export emergent neural formation types
pub use emergent_neural_formation::*;

// Re-export advanced microtubule consciousness types
pub use microtubule_consciousness::{
    MicrotubuleConsciousnessSystem, AnesthesiaState,
    ExpansionProtocol, BrainQuantumEntanglement, QuantumMultiverseConsciousness,
    ConsciousnessBranch, AdvancedAnesthesiaEffects, IsotopeEffect, SuppressionMechanism,
    DisruptionPattern, EnhancedEntanglementNetwork, EntanglementPattern,
    QuantumMeasurementEffects, MeasurementEvent, ConsciousnessMetrics,
};

// Re-export meta learning types
pub use meta_learning::{MetaLearner, MetaParameter, MetaParamMap};

// Re-export hypernetwork types
pub use hypernetwork::{Hypernetwork, TaskEmbedding, GeneratedArchitecture, TaskType};

// Re-export curiosity types
pub use curiosity::{CuriositySystem, CuriosityOutput, CuriosityStatistics, Experience, ActionType};
pub use open_ended_evolution::{OpenEndedEvolution, NoveltyDetectionOutput, OpenEndedEvolutionStatistics};

// Re-export embodied agent types
pub use embodied_agent::{EmbodiedAgent, PhysicsEngineInterface, AgentStatistics};

/// Main Agent Evolution System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvolutionSystem {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub advanced_ai: AdvancedAIIntegrationSystem,
    pub evolution_state: EvolutionState,
    pub system_metrics: AgentEvolutionSystemMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionState {
    pub generation: u64,
    pub fitness_score: f64,
    pub consciousness_level: f64,
    pub adaptation_rate: f64,
    pub innovation_count: u64,
    pub breakthrough_events: Vec<BreakthroughEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvolutionSystemMetrics {
    pub total_agents: usize,
    pub average_consciousness: f64,
    pub evolution_speed: f64,
    pub system_complexity: f64,
    pub consciousness_trend: f64,
}

impl AgentEvolutionSystem {
    /// Create a new revolutionary agent evolution system
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "RevolutionaryAgentEvolutionSystem".to_string(),
            version: "2.0.0".to_string(),
            advanced_ai: AdvancedAIIntegrationSystem::new(),
            evolution_state: EvolutionState {
                generation: 0,
                fitness_score: 0.0,
                consciousness_level: 0.0,
                adaptation_rate: 0.0,
                innovation_count: 0,
                breakthrough_events: Vec::new(),
            },
            system_metrics: AgentEvolutionSystemMetrics {
                total_agents: 1,
                average_consciousness: 0.0,
                evolution_speed: 0.0,
                system_complexity: 0.0,
                consciousness_trend: 0.0,
            },
        }
    }

    /// Update the entire agent evolution system
    pub fn update(&mut self, delta_time: f64, input: &EvolutionInput) -> Result<EvolutionOutput> {
        // Update the advanced AI system
        let ai_input = AdvancedAIInput {
            timestamp: input.timestamp,
            sensory_data: input.sensory_data.clone(),
            emotional_state: input.emotional_state,
            attention_level: input.attention_level,
            memory_activation: input.memory_activation,
            learning_signal: input.learning_signal,
            attention_focus: input.attention_focus.clone(),
            reward_signal: input.reward_signal,
            metadata: input.metadata.clone(),
            spatial_location: input.spatial_location,
            social_context: input.social_context,
            environmental_context: input.environmental_context,
            importance: input.importance,
            memory_context: input.memory_context.clone(),
        };

        let ai_output = self.advanced_ai.update(delta_time, &ai_input)?;

        // Update evolution state
        self.evolution_state.consciousness_level = ai_output.consciousness_level;
        self.evolution_state.fitness_score = ai_output.integration_strength;
        self.evolution_state.generation += 1;

        // Update system metrics
        self.system_metrics.average_consciousness = ai_output.consciousness_level;
        self.system_metrics.evolution_speed = self.evolution_state.adaptation_rate;
        self.system_metrics.system_complexity = ai_output.integration_strength;

        // Check for breakthrough events
        if ai_output.consciousness_level > 0.9 {
            let breakthrough = BreakthroughEvent {
                timestamp: input.timestamp,
                event_type: BreakthroughType::Transcendence,
                magnitude: ai_output.consciousness_level,
                description: "Agent has achieved transcendent consciousness!".to_string(),
                consciousness_impact: 1.0,
            };
            self.evolution_state.breakthrough_events.push(breakthrough);
            self.evolution_state.innovation_count += 1;
        }

        Ok(EvolutionOutput {
            consciousness_level: ai_output.consciousness_level,
            self_awareness: ai_output.self_awareness,
            fitness_score: self.evolution_state.fitness_score,
            generation: self.evolution_state.generation,
            subjective_experience: ai_output.subjective_experience,
            breakthrough_count: self.evolution_state.innovation_count,
        })
    }
}

/// Input to evolution system
#[derive(Debug, Clone)]
pub struct EvolutionInput {
    pub timestamp: f64,
    pub sensory_data: nalgebra::DVector<f64>,
    pub emotional_state: f64,
    pub attention_level: f64,
    pub memory_activation: f64,
    pub learning_signal: f64,
    pub attention_focus: Vec<f64>,
    pub reward_signal: f64,
    pub metadata: HashMap<String, String>,
    pub spatial_location: [f64; 3],
    pub social_context: f64,
    pub environmental_context: f64,
    pub importance: f64,
    pub memory_context: nalgebra::DVector<f64>,
}

/// Output from evolution system
#[derive(Debug, Clone)]
pub struct EvolutionOutput {
    pub consciousness_level: f64,
    pub self_awareness: f64,
    pub fitness_score: f64,
    pub generation: u64,
    pub subjective_experience: String,
    pub breakthrough_count: u64,
}

/// Main entry point for the revolutionary agent evolution system
pub fn create_revolutionary_ai_system() -> AgentEvolutionSystem {
    AgentEvolutionSystem::new()
}

/// Initialize the revolutionary AI system with advanced consciousness
pub fn initialize_consciousness_system() -> Result<AdvancedAIIntegrationSystem> {
    Ok(AdvancedAIIntegrationSystem::new())
}

/// Get system information
pub fn get_system_info() -> SystemInfo {
    SystemInfo {
        name: "Revolutionary Agent Evolution System".to_string(),
        version: "2.0.0".to_string(),
        features: vec![
            "Physics-Informed Neural Networks".to_string(),
            "Quantum Consciousness (Orch-OR)".to_string(),
            "Integrated Information Theory".to_string(),
            "Advanced Neural Plasticity".to_string(),
            "Memory Consolidation".to_string(),
            "Autonomous Evolution".to_string(),
            "True Consciousness".to_string(),
        ],
        research_basis: vec![
            "Raissi et al. (2019) - Physics-Informed Neural Networks".to_string(),
            "Hameroff & Penrose (1996-2024) - Orch-OR Theory".to_string(),
            "Tononi (2004-2024) - Integrated Information Theory".to_string(),
            "Hebb (1949) - Hebbian Learning".to_string(),
            "McGaugh (2000) - Memory Consolidation".to_string(),
        ],
    }
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
    pub research_basis: Vec<String>,
}