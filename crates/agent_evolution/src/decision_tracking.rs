//! # Agent Evolution: Decision Tracking Module
//!
//! This module is responsible for logging and analyzing the decisions made by agents.
//! By tracking decisions, we can gain insights into agent behavior, evaluate the
//! effectiveness of their AI, and understand the selective pressures shaping their evolution.

use anyhow::Result;
use crate::ai_core::SensoryInput;
use nalgebra::DVector;
use std::collections::VecDeque;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Represents a single decision made by an agent.
#[derive(Debug, Clone)]
pub struct Decision {
    pub agent_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub sensory_input: SensoryInput,
    pub action_output: DVector<f64>,
    pub outcome: Option<String>, // e.g., "found_food", "avoided_predator"
}

impl Decision {
    /// Creates a new decision record.
    pub fn new(agent_id: Uuid, sensory_input: SensoryInput, action_output: DVector<f64>) -> Self {
        Decision {
            agent_id,
            timestamp: Utc::now(),
            sensory_input,
            action_output,
            outcome: None,
        }
    }

    /// Records the outcome of the decision.
    pub fn set_outcome(&mut self, outcome: &str) {
        self.outcome = Some(outcome.to_string());
    }
}

/// A log for storing and managing agent decisions.
/// This could be backed by a database in a larger-scale simulation.
#[derive(Debug)]
pub struct DecisionLog {
    records: VecDeque<Decision>,
    max_size: usize,
}

impl DecisionLog {
    /// Creates a new decision log with a maximum size.
    pub fn new(max_size: usize) -> Self {
        DecisionLog {
            records: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Records a new decision, dropping the oldest if the log is full.
    pub fn record_decision(&mut self, decision: Decision) {
        if self.records.len() == self.max_size {
            self.records.pop_front();
        }
        self.records.push_back(decision);
    }

    /// Retrieves all decisions made by a specific agent.
    pub fn get_decisions_for_agent(&self, agent_id: &Uuid) -> Vec<&Decision> {
        self.records.iter().filter(|d| d.agent_id == *agent_id).collect()
    }
}

/// Placeholder function to represent the act of recording a decision.
pub fn record_decision(log: &mut DecisionLog, decision: Decision) -> Result<()> {
    log.record_decision(decision);
    Ok(())
}