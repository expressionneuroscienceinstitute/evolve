//! # Agent Evolution: Lineage Analytics Module
//!
//! This module provides tools for tracking the ancestry and evolutionary history of agents.
//! By maintaining lineage records, we can analyze how populations change over time,
//! trace the origins of successful adaptations, and visualize the tree of life.

use anyhow::Result;
use crate::genetics::Genome;
use std::collections::HashMap;
use uuid::Uuid;

/// Represents the ancestral line of an agent.
#[derive(Debug, Clone)]
pub struct Lineage {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub generation: u64,
    pub genome_snapshot: Genome,
}

impl Lineage {
    /// Creates a new founding lineage for a progenitor agent.
    pub fn new(genome: Genome) -> Self {
        Lineage {
            id: Uuid::new_v4(),
            parent_id: None,
            generation: 0,
            genome_snapshot: genome,
        }
    }

    /// Creates a new lineage that descends from a parent.
    pub fn new_descendant(parent: &Lineage, genome: Genome) -> Self {
        Lineage {
            id: Uuid::new_v4(),
            parent_id: Some(parent.id),
            generation: parent.generation + 1,
            genome_snapshot: genome,
        }
    }
}

/// Manages all lineages in the simulation.
#[derive(Debug, Default)]
pub struct LineageTracker {
    lineages: HashMap<Uuid, Lineage>,
}

impl LineageTracker {
    /// Creates a new, empty lineage tracker.
    pub fn new() -> Self {
        LineageTracker {
            lineages: HashMap::new(),
        }
    }

    /// Adds a new lineage to the tracker.
    pub fn add_lineage(&mut self, lineage: Lineage) {
        self.lineages.insert(lineage.id, lineage);
    }

    /// Retrieves a lineage by its ID.
    pub fn get_lineage(&self, id: &Uuid) -> Option<&Lineage> {
        self.lineages.get(id)
    }

    /// Traces the ancestry of a given lineage back to its origin.
    pub fn trace_ancestry(&self, start_id: &Uuid) -> Vec<&Lineage> {
        let mut ancestry = Vec::new();
        let mut current_id = Some(start_id);

        while let Some(id) = current_id {
            if let Some(lineage) = self.get_lineage(id) {
                ancestry.push(lineage);
                current_id = lineage.parent_id.as_ref();
            } else {
                break;
            }
        }
        ancestry.reverse(); // Return in chronological order
        ancestry
    }
}

/// Placeholder function for lineage-related updates.
/// In a full simulation, this would be called periodically to process new data.
pub fn update_lineages(tracker: &mut LineageTracker) -> Result<()> {
    // In a real application, this function would handle tasks such as:
    // - Pruning old or extinct lineages to save memory.
    // - Calculating statistics on lineage diversity and adaptation.
    // - Persisting lineage data to a database.
    log::info!("Updating lineage analytics. Total lineages: {}", tracker.lineages.len());
    Ok(())
}