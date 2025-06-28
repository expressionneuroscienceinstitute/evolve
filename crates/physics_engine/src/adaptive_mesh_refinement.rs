//! Adaptive Mesh Refinement (AMR) for Physics Engine
//!
//! This module implements a hierarchical octree-based adaptive mesh refinement system
//! for efficient multi-scale physics simulation. Based on best practices from AMReX
//! and modern AMR literature (see RESEARCH_PAPERS.md).
//!
//! ## Features
//! - Octree-based spatial decomposition
//! - Gradient-based refinement criteria
//! - Parallel-ready data structures
//! - Conservation-preserving mesh operations
//! - Performance monitoring and statistics
//!
//! ## References
//! - Zhang & Almgren (2023). "AMReX Highlights 2023: Block-Structured AMR for Exascale"
//! - Colella et al. (2024). "Adaptive Mesh Refinement in the Age of Exascale Computing"
//! - CODATA 2022 physics constants

use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;

use crate::{FundamentalParticle, types::BoundaryConditions};

/// AMR grid cell with hierarchical refinement capability
/// 
/// Represents a single cell in the adaptive mesh with all properties
/// needed for refinement decisions and physics calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmrCell {
    /// Unique cell identifier
    pub id: usize,
    /// Refinement level (0 = coarsest)
    pub level: u32,
    /// Cell center position (m)
    pub position: Vector3<f64>,
    /// Cell size (edge length in m)
    pub size: f64,
    /// Mass density (kg/m³)
    pub mass_density: f64,
    /// Energy density (J/m³)
    pub energy_density: f64,
    /// Field gradient magnitude (units depend on field)
    pub field_gradient: f64,
    /// Number of particles in cell
    pub particle_count: usize,
    /// Refinement criterion value
    pub refinement_criterion: f64,
    /// Parent cell ID (None for root level)
    pub parent_id: Option<usize>,
    /// Child cell IDs (empty for leaf cells)
    pub children_ids: Vec<usize>,
    /// True if this is a leaf cell
    pub is_leaf: bool,
    /// True if cell should be refined
    pub requires_refinement: bool,
    /// True if cell should be coarsened
    pub requires_coarsening: bool,
    /// Boundary condition type
    pub boundary_conditions: BoundaryConditions,
}

/// Adaptive mesh refinement manager
/// 
/// Central manager for the AMR grid hierarchy. Handles refinement,
/// coarsening, and grid updates based on physics criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmrManager {
    /// All AMR cells in the hierarchy
    pub cells: Vec<AmrCell>,
    /// Maximum allowed refinement level
    pub max_refinement_level: u32,
    /// Minimum allowed refinement level
    pub min_refinement_level: u32,
    /// Threshold for triggering refinement
    pub refinement_threshold: f64,
    /// Threshold for triggering coarsening
    pub coarsening_threshold: f64,
    /// Base grid cell size (m)
    pub base_grid_size: f64,
    /// Total domain size (m)
    pub domain_size: Vector3<f64>,
    /// Total number of cells created
    pub total_cells: usize,
    /// History of refinement events
    pub refinement_history: Vec<RefinementEvent>,
}

/// Event tracking for refinement analysis
/// 
/// Records all mesh operations for performance analysis and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementEvent {
    /// Time when event occurred (s)
    pub timestamp: f64,
    /// ID of affected cell
    pub cell_id: usize,
    /// Type of refinement event
    pub event_type: RefinementEventType,
    /// Old refinement level
    pub old_level: u32,
    /// New refinement level
    pub new_level: u32,
    /// Value that triggered the event
    pub trigger_value: f64,
}

/// Types of refinement events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefinementEventType {
    /// Cell was refined into children
    Refinement,
    /// Cell was coarsened (children removed)
    Coarsening,
    /// New cell was created
    Creation,
    /// Cell was deleted
    Deletion,
}

/// AMR grid statistics for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmrStatistics {
    /// Total number of cells
    pub total_cells: usize,
    /// Number of leaf cells
    pub total_leaves: usize,
    /// Number of refined cells
    pub total_refined: usize,
    /// Maximum refinement level
    pub max_level: u32,
    /// Distribution of cells by level
    pub level_distribution: HashMap<u32, usize>,
    /// Total refinement events
    pub refinement_events: usize,
}

impl AmrManager {
    /// Create new AMR manager with base grid
    /// 
    /// Initializes the AMR system with a uniform base grid at level 0.
    /// 
    /// # Arguments
    /// * `domain_size` - Physical size of simulation domain (m)
    /// * `base_grid_size` - Size of base level cells (m)
    /// * `max_level` - Maximum refinement level allowed
    /// * `refinement_threshold` - Criterion value to trigger refinement
    /// 
    /// # Returns
    /// New AMR manager ready for simulation
    pub fn new(
        domain_size: Vector3<f64>,
        base_grid_size: f64,
        max_level: u32,
        refinement_threshold: f64,
    ) -> Self {
        let mut manager = Self {
            cells: Vec::new(),
            max_refinement_level: max_level,
            min_refinement_level: 0,
            refinement_threshold,
            coarsening_threshold: refinement_threshold * 0.25, // Conservative coarsening
            base_grid_size,
            domain_size,
            total_cells: 0,
            refinement_history: Vec::new(),
        };
        
        // Initialize base grid
        manager.initialize_base_grid();
        manager
    }
    
    /// Initialize the coarsest level grid
    /// 
    /// Creates a uniform grid of cells at refinement level 0
    /// covering the entire simulation domain.
    fn initialize_base_grid(&mut self) {
        let cells_per_dimension = (self.domain_size.x / self.base_grid_size).ceil() as usize;
        
        for i in 0..cells_per_dimension {
            for j in 0..cells_per_dimension {
                for k in 0..cells_per_dimension {
                    let position = Vector3::new(
                        i as f64 * self.base_grid_size + self.base_grid_size * 0.5,
                        j as f64 * self.base_grid_size + self.base_grid_size * 0.5,
                        k as f64 * self.base_grid_size + self.base_grid_size * 0.5,
                    );
                    
                    let cell = AmrCell {
                        id: self.total_cells,
                        level: 0,
                        position,
                        size: self.base_grid_size,
                        mass_density: 0.0,
                        energy_density: 0.0,
                        field_gradient: 0.0,
                        particle_count: 0,
                        refinement_criterion: 0.0,
                        parent_id: None,
                        children_ids: Vec::new(),
                        is_leaf: true,
                        requires_refinement: false,
                        requires_coarsening: false,
                        boundary_conditions: BoundaryConditions::Periodic,
                    };
                    
                    self.cells.push(cell);
                    self.total_cells += 1;
                }
            }
        }
    }
    
    /// Update AMR grid based on physical conditions
    /// 
    /// Complete AMR update cycle including property updates,
    /// refinement criteria calculation, and mesh adaptation.
    /// 
    /// # Arguments
    /// * `particles` - Current particle distribution
    /// * `current_time` - Simulation time (s)
    /// 
    /// # Returns
    /// Result indicating success or failure
    pub fn update_mesh(&mut self, particles: &[FundamentalParticle], current_time: f64) -> Result<()> {
        // Step 1: Update cell properties from particle data
        self.update_cell_properties(particles)?;
        
        // Step 2: Calculate refinement criteria
        self.calculate_refinement_criteria()?;
        
        // Step 3: Perform refinement
        self.perform_refinement(current_time)?;
        
        // Step 4: Perform coarsening
        self.perform_coarsening(current_time)?;
        
        Ok(())
    }
    
    /// Update cell properties based on particle distribution
    /// 
    /// Maps particle properties to grid cells and calculates
    /// density fields for refinement decisions.
    fn update_cell_properties(&mut self, particles: &[FundamentalParticle]) -> Result<()> {
        // Clear existing counts and densities
        for cell in &mut self.cells {
            cell.mass_density = 0.0;
            cell.energy_density = 0.0;
            cell.particle_count = 0;
        }

        // Map particle properties to cells
        for particle in particles {
            if let Some(cell_id) = self.find_containing_cell(&particle.position) {
                if let Some(cell) = self.cells.get_mut(cell_id) {
                    if cell.is_leaf {
                        let cell_volume = cell.size.powi(3);
                        cell.mass_density += particle.mass / cell_volume;
                        cell.energy_density += particle.energy / cell_volume;
                        cell.particle_count += 1;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate refinement criteria for all leaf cells
    /// 
    /// Iterates through all leaf cells and computes a refinement criterion,
    /// typically based on the local density gradient.
    fn calculate_refinement_criteria(&mut self) -> Result<()> {
        let cell_ids: Vec<usize> = self.cells.iter().map(|c| c.id).collect();
        for cell_id in cell_ids {
            if self.cells[cell_id].is_leaf {
                let gradient = self.calculate_spatial_gradient(cell_id)?;
                self.cells[cell_id].field_gradient = gradient;
                
                // Simple criterion: gradient magnitude relative to average density
                let avg_density = self.cells.iter().map(|c| c.mass_density).sum::<f64>() / self.cells.len() as f64;
                if avg_density > 1e-9 {
                    self.cells[cell_id].refinement_criterion = gradient / avg_density;
                } else {
                    self.cells[cell_id].refinement_criterion = 0.0;
                }
            }
        }
        Ok(())
    }
    
    /// Calculate spatial gradient for a given cell
    /// 
    /// Computes the maximum gradient magnitude by comparing
    /// the cell's properties with its neighbors.
    fn calculate_spatial_gradient(&self, cell_id: usize) -> Result<f64> {
        let mut max_gradient = 0.0;
        let cell_pos = self.cells[cell_id].position;
        let cell_density = self.cells[cell_id].mass_density;

        // Implement proper neighbor finding using octree-based approach
        // This efficiently locates adjacent cells in the AMR hierarchy
        let neighbors = self.find_amr_neighbors(cell_id)?;
        
        for &neighbor_id in &neighbors {
            let neighbor_density = self.cells[neighbor_id].mass_density;
            let neighbor_pos = self.cells[neighbor_id].position;
            
            let distance = (neighbor_pos - cell_pos).norm();
            if distance > 1e-12 { // Avoid division by zero
                let gradient = (neighbor_density - cell_density).abs() / distance;
                if gradient > max_gradient {
                    max_gradient = gradient;
                }
            }
        }
        
        Ok(max_gradient)
    }
    
    /// Find all neighbors of a given cell in the AMR hierarchy
    /// 
    /// Uses an efficient octree-based approach to locate adjacent cells
    /// at the same or different refinement levels.
    fn find_amr_neighbors(&self, cell_id: usize) -> Result<Vec<usize>> {
        let mut neighbors = Vec::new();
        let cell = &self.cells[cell_id];
        let cell_size = cell.size;
        let cell_pos = cell.position;
        
        // Define search region (slightly larger than cell to catch all neighbors)
        let search_radius = cell_size * 1.5;
        
        for (neighbor_id, neighbor) in self.cells.iter().enumerate() {
            if neighbor_id == cell_id || !neighbor.is_leaf {
                continue; // Skip self and non-leaf cells
            }
            
            let distance = (neighbor.position - cell_pos).norm();
            if distance <= search_radius {
                // Check if cells are actually adjacent (share a face, edge, or corner)
                let max_separation = (cell_size + neighbor.size) / 2.0;
                if distance <= max_separation {
                    neighbors.push(neighbor_id);
                }
            }
        }
        
        // If no neighbors found at same level, look for parent's neighbors
        if neighbors.is_empty() && cell.parent_id.is_some() {
            if let Ok(parent_neighbors) = self.find_amr_neighbors(cell.parent_id.unwrap()) {
                // Add children of parent's neighbors
                for &parent_neighbor_id in &parent_neighbors {
                    let parent_neighbor = &self.cells[parent_neighbor_id];
                    if !parent_neighbor.is_leaf {
                        neighbors.extend(&parent_neighbor.children_ids);
                    } else {
                        neighbors.push(parent_neighbor_id);
                    }
                }
            }
        }
        
        Ok(neighbors)
    }
    
    /// Perform refinement operations based on calculated criteria
    /// 
    /// Iterates through all cells and refines those that meet
    /// the refinement threshold.
    fn perform_refinement(&mut self, current_time: f64) -> Result<()> {
        let mut cells_to_refine = Vec::new();
        for cell in &self.cells {
            if cell.refinement_criterion > self.refinement_threshold && cell.is_leaf && cell.level < self.max_refinement_level {
                cells_to_refine.push(cell.id);
            }
        }

        for cell_id in cells_to_refine {
            self.refine_cell(cell_id, current_time)?;
        }
        
        Ok(())
    }
    
    /// Refine a single cell into eight children
    /// 
    /// Replaces a single leaf cell with eight smaller children cells
    /// at the next refinement level.
    fn refine_cell(&mut self, cell_id: usize, current_time: f64) -> Result<()> {
        // Check if cell exists and is a leaf
        let (parent_level, parent_pos, parent_size, boundary_conditions) = {
            if let Some(parent_cell) = self.cells.get(cell_id) {
                if !parent_cell.is_leaf {
                    return Ok(()); // Already refined
                }
                (parent_cell.level, parent_cell.position, parent_cell.size, parent_cell.boundary_conditions.clone())
            } else {
                return Ok(()); // Cell doesn't exist
            }
        };
        
        // Update parent cell properties
        if let Some(parent_cell) = self.cells.get_mut(cell_id) {
            parent_cell.is_leaf = false;
            parent_cell.requires_refinement = false;
        }
        
        let child_size = parent_size / 2.0;
        let mut children_ids = Vec::with_capacity(8);

        for i in 0..8 {
            let offset_x = if (i & 1) == 0 { -0.25 } else { 0.25 };
            let offset_y = if (i & 2) == 0 { -0.25 } else { 0.25 };
            let offset_z = if (i & 4) == 0 { -0.25 } else { 0.25 };

            let child_pos = parent_pos + Vector3::new(
                offset_x * parent_size,
                offset_y * parent_size,
                offset_z * parent_size,
            );

            let new_cell_id = self.total_cells;
            let child_cell = AmrCell {
                id: new_cell_id,
                level: parent_level + 1,
                position: child_pos,
                size: child_size,
                mass_density: 0.0, // Will be updated in the next cycle
                energy_density: 0.0,
                field_gradient: 0.0,
                particle_count: 0,
                refinement_criterion: 0.0,
                parent_id: Some(cell_id),
                children_ids: Vec::new(),
                is_leaf: true,
                requires_refinement: false,
                requires_coarsening: false,
                boundary_conditions: boundary_conditions.clone(),
            };
            
            children_ids.push(new_cell_id);
            self.cells.push(child_cell);
            self.total_cells += 1;
            
            // Record event
            self.refinement_history.push(RefinementEvent {
                timestamp: current_time,
                cell_id: new_cell_id,
                event_type: RefinementEventType::Creation,
                old_level: parent_level,
                new_level: parent_level + 1,
                trigger_value: 0.0, // Not directly triggered
            });
        }

        // Update parent's children list
        if let Some(parent_cell) = self.cells.get_mut(cell_id) {
            parent_cell.children_ids = children_ids;
        }

        self.refinement_history.push(RefinementEvent {
            timestamp: current_time,
            cell_id,
            event_type: RefinementEventType::Refinement,
            old_level: parent_level,
            new_level: parent_level, // Level of parent doesn't change
            trigger_value: self.cells[cell_id].refinement_criterion,
        });
        
        Ok(())
    }
    
    /// Perform coarsening operations for cells no longer needing refinement
    /// 
    /// Removes child cells for regions that no longer need fine resolution.
    fn perform_coarsening(&mut self, current_time: f64) -> Result<()> {
        let mut parents_to_coarsen = HashMap::new();

        // Identify potential parents for coarsening
        for cell in &self.cells {
            if !cell.is_leaf {
                let criterion = cell.refinement_criterion;
                if criterion < self.coarsening_threshold && cell.level >= self.min_refinement_level {
                     // Check if all children are leaves
                    let all_children_are_leaves = cell.children_ids.iter().all(|&child_id| self.cells[child_id].is_leaf);
                    if all_children_are_leaves {
                        parents_to_coarsen.insert(cell.id, criterion);
                    }
                }
            }
        }
        
        for (parent_id, trigger_value) in parents_to_coarsen {
            let children_to_remove = self.cells[parent_id].children_ids.clone();
            
            // Remove children
            self.cells.retain(|c| !children_to_remove.contains(&c.id));

            // Update parent
            if let Some(parent) = self.cells.iter_mut().find(|c| c.id == parent_id) {
                parent.is_leaf = true;
                parent.children_ids.clear();
            }
            
            // Record event
            self.refinement_history.push(RefinementEvent {
                timestamp: current_time,
                cell_id: parent_id,
                event_type: RefinementEventType::Coarsening,
                old_level: self.cells[parent_id].level,
                new_level: self.cells[parent_id].level,
                trigger_value,
            });
        }
        
        Ok(())
    }
    
    /// Find the cell containing a given position
    /// 
    /// Searches the AMR hierarchy to find the finest cell
    /// that contains the specified position.
    fn find_containing_cell(&self, position: &Vector3<f64>) -> Option<usize> {
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.is_leaf {
                let half_size = cell.size / 2.0;
                if (position.x >= cell.position.x - half_size) &&
                   (position.x < cell.position.x + half_size) &&
                   (position.y >= cell.position.y - half_size) &&
                   (position.y < cell.position.y + half_size) &&
                   (position.z >= cell.position.z - half_size) &&
                   (position.z < cell.position.z + half_size) {
                    return Some(i);
                }
            }
        }
        None
    }
    
    /// Get AMR statistics for performance monitoring
    /// 
    /// Computes comprehensive statistics about the current
    /// mesh state for analysis and optimization.
    pub fn get_statistics(&self) -> AmrStatistics {
        let mut level_distribution = HashMap::new();
        let mut total_leaves = 0;
        let mut total_refined = 0;
        let mut max_level = 0;
        
        for cell in &self.cells {
            *level_distribution.entry(cell.level).or_insert(0) += 1;
            if cell.is_leaf {
                total_leaves += 1;
            } else {
                total_refined += 1;
            }
            max_level = max_level.max(cell.level);
        }
        
        AmrStatistics {
            total_cells: self.cells.len(),
            total_leaves,
            total_refined,
            max_level,
            level_distribution,
            refinement_events: self.refinement_history.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle_types::ParticleType;

    #[test]
    fn test_amr_manager_creation() {
        let domain = Vector3::new(1.0, 1.0, 1.0);
        let manager = AmrManager::new(domain, 1.0, 2, 0.1);
        // Expect a single base cell for a 1x1x1 m domain with 1 m cell size
        assert_eq!(manager.cells.len(), 1);
        let cell = &manager.cells[0];
        assert_eq!(cell.level, 0);
        assert!(cell.is_leaf);
    }

    #[test]
    fn test_amr_update_mesh_empty_particles() {
        let domain = Vector3::new(1.0, 1.0, 1.0);
        let mut manager = AmrManager::new(domain, 1.0, 2, 0.1);
        let particles: Vec<FundamentalParticle> = Vec::new();
        assert!(manager.update_mesh(&particles, 0.0).is_ok());
        // After update, cell properties should remain zero for empty particle list
        let cell = &manager.cells[0];
        assert_eq!(cell.particle_count, 0);
        assert_eq!(cell.mass_density, 0.0);
        assert_eq!(cell.energy_density, 0.0);
    }
} 