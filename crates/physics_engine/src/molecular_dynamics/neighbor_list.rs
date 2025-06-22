//! Neighbor List Algorithms for Molecular Dynamics
//!
//! Implements efficient neighbor finding algorithms to reduce computational complexity
//! from O(N²) to O(N) for force calculations in molecular dynamics simulations.
//! - Cell-linked lists (primary method)
//! - Verlet neighbor lists
//! - Adaptive neighbor list updates
//!
//! References:
//! - https://cpvr.rantai.dev/docs/part-iv/chapter-17/
//! - Project RESEARCH_PAPERS.md

use anyhow::Result;
use nalgebra::Vector3;
use crate::PhysicsState;

/// Cell-linked list for efficient neighbor finding
/// 
/// Divides space into cubic cells and maintains lists of particles in each cell.
/// Reduces neighbor search complexity from O(N²) to O(N) for uniform particle distributions.
pub struct CellList {
    /// Size of each cell (should be >= cutoff distance)
    pub cell_size: f64,
    /// Number of cells in each dimension
    pub grid_size: [usize; 3],
    /// Bounds of the simulation domain
    pub domain_bounds: [[f64; 2]; 3],
    /// List of particles in each cell
    pub cells: Vec<Vec<usize>>,
    /// Mapping from particle index to cell index
    pub particle_to_cell: Vec<usize>,
}

impl CellList {
    /// Create a new cell list with given parameters
    pub fn new(cell_size: f64, domain_bounds: [[f64; 2]; 3]) -> Self {
        let grid_size = [
            ((domain_bounds[0][1] - domain_bounds[0][0]) / cell_size).ceil() as usize,
            ((domain_bounds[1][1] - domain_bounds[1][0]) / cell_size).ceil() as usize,
            ((domain_bounds[2][1] - domain_bounds[2][0]) / cell_size).ceil() as usize,
        ];
        
        let total_cells = grid_size[0] * grid_size[1] * grid_size[2];
        
        Self {
            cell_size,
            grid_size,
            domain_bounds,
            cells: vec![Vec::new(); total_cells],
            particle_to_cell: Vec::new(),
        }
    }

    /// Build the cell list from particle positions
    pub fn build(&mut self, particles: &[PhysicsState]) -> Result<()> {
        // Clear existing data
        for cell in &mut self.cells {
            cell.clear();
        }
        self.particle_to_cell.resize(particles.len(), 0);

        // Assign particles to cells
        for (particle_idx, particle) in particles.iter().enumerate() {
            let cell_idx = self.position_to_cell_index(&particle.position)?;
            if cell_idx < self.cells.len() {
                self.cells[cell_idx].push(particle_idx);
                self.particle_to_cell[particle_idx] = cell_idx;
            }
        }

        Ok(())
    }

    /// Find all neighbors of a particle within cutoff distance
    pub fn find_neighbors(&self, particle_idx: usize, cutoff: f64, particles: &[PhysicsState]) -> Result<Vec<usize>> {
        if particle_idx >= particles.len() {
            return Ok(Vec::new());
        }

        let particle_pos = &particles[particle_idx].position;
        let cutoff_squared = cutoff * cutoff;
        let mut neighbors = Vec::new();

        // Get the cell containing the particle
        let center_cell = self.position_to_cell_index(particle_pos)?;
        
        // Check the center cell and all adjacent cells
        for &neighbor_cell in &self.get_neighboring_cells(center_cell)? {
            if neighbor_cell < self.cells.len() {
                for &neighbor_particle_idx in &self.cells[neighbor_cell] {
                    if neighbor_particle_idx != particle_idx {
                        let neighbor_pos = &particles[neighbor_particle_idx].position;
                        let distance_squared = (particle_pos - neighbor_pos).norm_squared();
                        
                        if distance_squared <= cutoff_squared {
                            neighbors.push(neighbor_particle_idx);
                        }
                    }
                }
            }
        }

        Ok(neighbors)
    }

    /// Convert 3D position to cell index
    fn position_to_cell_index(&self, position: &Vector3<f64>) -> Result<usize> {
        let mut cell_coords = [0; 3];
        
        for dim in 0..3 {
            let relative_pos = position[dim] - self.domain_bounds[dim][0];
            let cell_coord = (relative_pos / self.cell_size).floor() as i32;
            
            if cell_coord < 0 || cell_coord >= self.grid_size[dim] as i32 {
                return Err(anyhow::anyhow!("Particle position outside domain bounds"));
            }
            
            cell_coords[dim] = cell_coord as usize;
        }

        Ok(cell_coords[0] + cell_coords[1] * self.grid_size[0] + cell_coords[2] * self.grid_size[0] * self.grid_size[1])
    }

    /// Get indices of cells neighboring a given cell
    fn get_neighboring_cells(&self, center_cell: usize) -> Result<Vec<usize>> {
        let mut neighbors = Vec::new();
        
        // Convert cell index back to 3D coordinates
        let z = center_cell / (self.grid_size[0] * self.grid_size[1]);
        let remainder = center_cell % (self.grid_size[0] * self.grid_size[1]);
        let y = remainder / self.grid_size[0];
        let x = remainder % self.grid_size[0];

        // Check all 27 neighboring cells (including the center cell)
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    
                    if nx >= 0 && nx < self.grid_size[0] as i32 &&
                       ny >= 0 && ny < self.grid_size[1] as i32 &&
                       nz >= 0 && nz < self.grid_size[2] as i32 {
                        let neighbor_idx = nx as usize + 
                                         ny as usize * self.grid_size[0] + 
                                         nz as usize * self.grid_size[0] * self.grid_size[1];
                        neighbors.push(neighbor_idx);
                    }
                }
            }
        }

        Ok(neighbors)
    }
}

/// Verlet neighbor list with automatic updates
/// 
/// Maintains a list of neighbors for each particle and automatically
/// rebuilds the list when particles move significantly.
#[derive(Debug)]
pub struct VerletNeighborList {
    /// Cutoff distance for neighbor inclusion
    pub cutoff: f64,
    /// Skin distance for list updates (should be > 0)
    pub skin_distance: f64,
    /// Neighbor lists for each particle
    pub neighbor_lists: Vec<Vec<usize>>,
    /// Last positions when the list was built
    pub last_positions: Vec<Vector3<f64>>,
    /// Whether the list needs to be rebuilt
    pub needs_rebuild: bool,
}

impl VerletNeighborList {
    /// Create a new Verlet neighbor list
    pub fn new(cutoff: f64, skin_distance: f64) -> Self {
        Self {
            cutoff,
            skin_distance,
            neighbor_lists: Vec::new(),
            last_positions: Vec::new(),
            needs_rebuild: true,
        }
    }

    /// Build the neighbor list using a cell list
    pub fn build(&mut self, particles: &[PhysicsState], cell_list: &mut CellList) -> Result<()> {
        cell_list.build(particles)?;
        
        self.neighbor_lists.clear();
        self.neighbor_lists.resize(particles.len(), Vec::new());
        self.last_positions.clear();
        self.last_positions.extend(particles.iter().map(|p| p.position));
        
        // Build neighbor lists for each particle
        for particle_idx in 0..particles.len() {
            let neighbors = cell_list.find_neighbors(particle_idx, self.cutoff + self.skin_distance, particles)?;
            self.neighbor_lists[particle_idx] = neighbors;
        }
        
        self.needs_rebuild = false;
        Ok(())
    }

    /// Check if the neighbor list needs to be rebuilt
    pub fn check_rebuild_needed(&mut self, particles: &[PhysicsState]) -> Result<bool> {
        if self.needs_rebuild {
            return Ok(true);
        }

        let max_displacement_squared = (self.cutoff + self.skin_distance) * (self.cutoff + self.skin_distance) / 4.0;
        
        for (particle_idx, particle) in particles.iter().enumerate() {
            if particle_idx < self.last_positions.len() {
                let displacement_squared = (particle.position - self.last_positions[particle_idx]).norm_squared();
                if displacement_squared > max_displacement_squared {
                    self.needs_rebuild = true;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get neighbors for a specific particle
    pub fn get_neighbors(&self, particle_idx: usize) -> &[usize] {
        if particle_idx < self.neighbor_lists.len() {
            &self.neighbor_lists[particle_idx]
        } else {
            &[]
        }
    }

    /// Mark the list for rebuild
    pub fn mark_for_rebuild(&mut self) {
        self.needs_rebuild = true;
    }
}

/// Adaptive neighbor list that optimizes update frequency
/// 
/// Dynamically adjusts the skin distance based on particle velocities
/// to minimize the number of list rebuilds while maintaining accuracy.
pub struct AdaptiveNeighborList {
    /// Base cutoff distance
    pub base_cutoff: f64,
    /// Minimum skin distance
    pub min_skin_distance: f64,
    /// Maximum skin distance
    pub max_skin_distance: f64,
    /// Current skin distance
    pub current_skin_distance: f64,
    /// Target number of rebuilds per simulation step
    pub target_rebuild_frequency: f64,
    /// Current rebuild frequency
    pub current_rebuild_frequency: f64,
    /// Number of steps since last rebuild
    pub steps_since_rebuild: usize,
    /// Total number of steps
    pub total_steps: usize,
}

impl AdaptiveNeighborList {
    /// Create a new adaptive neighbor list
    pub fn new(base_cutoff: f64, min_skin_distance: f64, max_skin_distance: f64, target_frequency: f64) -> Self {
        Self {
            base_cutoff,
            min_skin_distance,
            max_skin_distance,
            current_skin_distance: min_skin_distance,
            target_rebuild_frequency: target_frequency,
            current_rebuild_frequency: 0.0,
            steps_since_rebuild: 0,
            total_steps: 0,
        }
    }

    /// Update the skin distance based on current rebuild frequency
    pub fn adapt_skin_distance(&mut self) {
        if self.total_steps > 0 {
            self.current_rebuild_frequency = self.steps_since_rebuild as f64 / self.total_steps as f64;
            
            // Adjust skin distance based on rebuild frequency
            if self.current_rebuild_frequency > self.target_rebuild_frequency {
                // Too many rebuilds, increase skin distance
                self.current_skin_distance = (self.current_skin_distance * 1.1).min(self.max_skin_distance);
            } else if self.current_rebuild_frequency < self.target_rebuild_frequency * 0.5 {
                // Too few rebuilds, decrease skin distance
                self.current_skin_distance = (self.current_skin_distance * 0.9).max(self.min_skin_distance);
            }
        }
    }

    /// Get the current effective cutoff distance
    pub fn get_effective_cutoff(&self) -> f64 {
        self.base_cutoff + self.current_skin_distance
    }

    /// Record a rebuild event
    pub fn record_rebuild(&mut self) {
        self.steps_since_rebuild = 0;
        self.total_steps += 1;
    }

    /// Record a step without rebuild
    pub fn record_step(&mut self) {
        self.steps_since_rebuild += 1;
        self.total_steps += 1;
    }
} 