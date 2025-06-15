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

/// Performs analysis on agent decisions to identify patterns and evaluate effectiveness.
#[derive(Debug, Default)]
pub struct DecisionAnalyzer {
    // In a real system, this might contain more complex models, e.g., for reinforcement learning
    // For now, it will focus on statistical analysis and simple pattern recognition.
}

impl DecisionAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyzes a single decision and updates internal learning models (placeholder).
    /// Returns feedback based on the outcome.
    pub fn analyze_decision(&mut self, decision: &Decision) -> String {
        match &decision.outcome {
            Some(outcome_str) if outcome_str.contains("success") => {
                format!("Decision (ID: {}) resulted in success. Learning positive reinforcement.", decision.agent_id)
            },
            Some(outcome_str) if outcome_str.contains("failure") => {
                format!("Decision (ID: {}) resulted in failure. Learning negative reinforcement.", decision.agent_id)
            },
            Some(outcome_str) => {
                format!("Decision (ID: {}) outcome: {}. Analyzing further.", decision.agent_id, outcome_str)
            },
            None => {
                format!("Decision (ID: {}) recorded, outcome pending.", decision.agent_id)
            },
        }
    }

    /// Provides a summary of decision patterns for a given agent (placeholder).
    pub fn summarize_agent_decisions(&self, agent_id: &Uuid, decisions: &[&Decision]) -> String {
        let total_decisions = decisions.len();
        let successful_outcomes = decisions.iter()
            .filter(|d| d.outcome.as_ref().map_or(false, |o| o.contains("success")))
            .count();

        if total_decisions == 0 {
            return format!("No decisions recorded for agent {}.", agent_id);
        }

        let success_rate = (successful_outcomes as f64 / total_decisions as f64) * 100.0;
        format!(
            "Agent {} made {} decisions. Success rate: {:.2}%.",
            agent_id,
            total_decisions,
            success_rate
        )
    }
}

/// A log for storing and managing agent decisions.
/// This could be backed by a database in a larger-scale simulation.
#[derive(Debug)]
pub struct DecisionLog {
    records: VecDeque<Decision>,
    max_size: usize,
    analyzer: DecisionAnalyzer, // Added decision analyzer
}

impl DecisionLog {
    /// Creates a new decision log with a maximum size.
    pub fn new(max_size: usize) -> Self {
        DecisionLog {
            records: VecDeque::with_capacity(max_size),
            max_size,
            analyzer: DecisionAnalyzer::new(), // Initialize the analyzer
        }
    }

    /// Records a new decision, dropping the oldest if the log is full.
    /// After recording, it triggers the decision analysis.
    pub fn record_decision(&mut self, decision: Decision) {
        // Analyze the decision immediately after recording
        let feedback = self.analyzer.analyze_decision(&decision);
        println!("Decision Analysis Feedback: {}", feedback); // For demonstration

        if self.records.len() == self.max_size {
            self.records.pop_front();
        }
        self.records.push_back(decision);
    }

    /// Retrieves all decisions made by a specific agent.
    pub fn get_decisions_for_agent(&self, agent_id: &Uuid) -> Vec<&Decision> {
        self.records.iter().filter(|d| d.agent_id == *agent_id).collect()
    }

    /// Provides a summary of all decisions in the log.
    pub fn summarize_all_decisions(&self) -> String {
        let mut agent_summaries = String::new();
        let mut distinct_agents: Vec<Uuid> = self.records.iter().map(|d| d.agent_id).collect();
        distinct_agents.sort();
        distinct_agents.dedup();

        for agent_id in distinct_agents {
            let agent_decisions = self.get_decisions_for_agent(&agent_id);
            agent_summaries.push_str(&self.analyzer.summarize_agent_decisions(&agent_id, &agent_decisions));
            agent_summaries.push_str("\n");
        }
        agent_summaries
    }
}

/// Placeholder function to represent the act of recording a decision.
pub fn record_decision(log: &mut DecisionLog, decision: Decision) -> Result<()> {
    log.record_decision(decision);
    Ok(())
}