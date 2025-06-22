//! Multi-Agent Interaction Dynamics System
//!
//! Implements complex agent-to-agent dynamics with emergent behaviors,
//! coordination mechanisms, and adaptive interaction patterns.
//!
//! Based on research from:
//! - Weiß, G. "Learning to Coordinate Actions in Multi-Agent Systems" (1993)
//! - Coordination mechanisms in multi-agent systems
//! - Emergent behavior patterns in distributed systems

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use rand::{thread_rng, Rng};
use serde_json;

/// Multi-agent interaction system that manages complex agent dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentInteractionSystem {
    pub system_id: Uuid,
    pub agents: HashMap<Uuid, AgentState>,
    pub interaction_network: InteractionNetwork,
    pub emergent_behaviors: EmergentBehaviors,
    pub interaction_history: VecDeque<InteractionEvent>,
    pub system_metrics: MultiAgentSystemMetrics,
}

/// Individual agent state within the multi-agent system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub capabilities: AgentCapabilities,
    pub current_goals: Vec<AgentGoal>,
    pub social_connections: HashMap<Uuid, f64>, // trust level
    pub learning_state: LearningState,
    pub energy_level: f64,
    pub last_interaction_secs: Option<f64>, // seconds since UNIX epoch
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum AgentType {
    PhysicsAgent,
    MolecularAgent,
    ConsciousnessAgent,
    CoordinationAgent,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub expertise_domains: Vec<String>,
    pub processing_power: f64,
    pub memory_capacity: f64,
    pub communication_bandwidth: f64,
    pub learning_rate: f64,
    pub adaptability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGoal {
    pub goal_id: Uuid,
    pub description: String,
    pub priority: f64,
    pub deadline_secs: Option<f64>, // seconds since UNIX epoch
    pub progress: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningState {
    pub learning_rate: f64,
    pub experience: f64,
    pub skill_development: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionNetwork {
    pub connections: HashMap<(Uuid, Uuid), f64>, // trust/strength
    pub network_density: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentBehaviors {
    pub behavior_patterns: Vec<BehavioralPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    pub pattern_id: Uuid,
    pub pattern_type: PatternType,
    pub participating_agents: Vec<Uuid>,
    pub frequency: f64,
    pub effectiveness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Cooperation,
    Competition,
    Consensus,
    Innovation,
    Adaptation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentSystemMetrics {
    pub overall_efficiency: f64,
    pub coordination_quality: f64,
    pub communication_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub event_id: Uuid,
    pub timestamp_secs: f64, // seconds since UNIX epoch
    pub event_type: InteractionEventType,
    pub participants: Vec<Uuid>,
    pub outcome: InteractionOutcome,
    #[serde(with = "duration_serde")]
    pub duration: Duration,
}

mod duration_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionEventType {
    Communication,
    Cooperation,
    Competition,
    Consensus,
    Learning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionOutcome {
    pub success: bool,
    pub value_created: f64,
    pub knowledge_gained: f64,
}

impl MultiAgentInteractionSystem {
    pub fn new() -> Self {
        Self {
            system_id: Uuid::new_v4(),
            agents: HashMap::new(),
            interaction_network: InteractionNetwork {
                connections: HashMap::new(),
                network_density: 0.0,
            },
            emergent_behaviors: EmergentBehaviors {
                behavior_patterns: Vec::new(),
            },
            interaction_history: VecDeque::new(),
            system_metrics: MultiAgentSystemMetrics {
                overall_efficiency: 0.0,
                coordination_quality: 0.0,
                communication_efficiency: 0.0,
            },
        }
    }

    /// Export a summary of the agent network for visualization (e.g., as JSON)
    pub fn export_network_summary(&self) -> serde_json::Value {
        let agents: Vec<_> = self.agents.iter().map(|(id, agent)| {
            serde_json::json!({
                "id": id.to_string(),
                "type": format!("{:?}", agent.agent_type),
                "energy": agent.energy_level,
                "last_interaction_secs": agent.last_interaction_secs,
                "trust": agent.social_connections,
            })
        }).collect();
        let connections: Vec<_> = self.interaction_network.connections.iter().map(|((a, b), strength)| {
            serde_json::json!({
                "source": a.to_string(),
                "target": b.to_string(),
                "strength": strength,
            })
        }).collect();
        serde_json::json!({
            "agents": agents,
            "connections": connections,
            "metrics": {
                "overall_efficiency": self.system_metrics.overall_efficiency,
                "coordination_quality": self.system_metrics.coordination_quality,
                "communication_efficiency": self.system_metrics.communication_efficiency,
            }
        })
    }

    /// Generate timeline events for visualization
    pub fn generate_timeline_events(&self) -> Vec<TimelineEvent> {
        let mut events = Vec::new();
        let now = current_unix_time();
        
        for (agent_id, agent) in &self.agents {
            // Generate events based on agent state
            if let Some(last_secs) = agent.last_interaction_secs {
                let time_since = now - last_secs;
                events.push(TimelineEvent {
                    agent_id: agent_id.to_string(),
                    timestamp: last_secs,
                    event_type: "interaction".to_string(),
                    description: format!("Last interaction: {:.1}s ago", time_since),
                    duration: 1.0,
                });
            }
            
            // Add energy level event
            events.push(TimelineEvent {
                agent_id: agent_id.to_string(),
                timestamp: now,
                event_type: "status".to_string(),
                description: format!("Energy: {:.2}", agent.energy_level),
                duration: 0.0,
            });
        }
        
        events
    }

    /// Generate interaction heatmap data for visualization
    pub fn generate_heatmap_data(&self) -> Vec<HeatmapCell> {
        let mut heatmap = Vec::new();
        
        // Create heatmap based on interaction network
        for ((agent1, agent2), strength) in &self.interaction_network.connections {
            // Convert agent IDs to positions (simple hash-based positioning)
            let x1 = (agent1.as_u128() % 100) as f32 / 100.0;
            let y1 = ((agent1.as_u128() >> 64) % 100) as f32 / 100.0;
            let x2 = (agent2.as_u128() % 100) as f32 / 100.0;
            let y2 = ((agent2.as_u128() >> 64) % 100) as f32 / 100.0;
            
            // Add connection points
            heatmap.push(HeatmapCell {
                x: x1,
                y: y1,
                intensity: *strength as f32,
                interaction_type: "connection".to_string(),
            });
            
            heatmap.push(HeatmapCell {
                x: x2,
                y: y2,
                intensity: *strength as f32,
                interaction_type: "connection".to_string(),
            });
        }
        
        heatmap
    }

    /// Export comprehensive visualization data
    pub fn export_visualization_data(&self) -> VisualizationData {
        VisualizationData {
            network_summary: self.export_network_summary(),
            timeline_events: self.generate_timeline_events(),
            heatmap_data: self.generate_heatmap_data(),
        }
    }

    // ... (methods for agent addition, interaction, learning, etc. would go here)
}

fn current_unix_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

// Local types for visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub agent_id: String,
    pub timestamp: f64,
    pub event_type: String,
    pub description: String,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapCell {
    pub x: f32,
    pub y: f32,
    pub intensity: f32,
    pub interaction_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub network_summary: serde_json::Value,
    pub timeline_events: Vec<TimelineEvent>,
    pub heatmap_data: Vec<HeatmapCell>,
}

impl InteractionEvent {
    pub fn new(
        event_type: InteractionEventType,
        participants: Vec<Uuid>,
        outcome: InteractionOutcome,
        duration: Duration,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            timestamp_secs: current_unix_time(),
            event_type,
            participants,
            outcome,
            duration,
        }
    }
} 