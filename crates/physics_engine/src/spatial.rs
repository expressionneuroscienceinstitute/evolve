//! Spatial partitioning for optimizing particle interactions
//! 
//! This module provides spatial data structures to accelerate particle interaction
//! calculations from O(N²) to O(N) complexity using spatial hash grids and octrees.

use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};
use crate::FundamentalParticle;
use serde::{Serialize, Deserialize};

/// Spatial hash grid for fast neighbor finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialHashGrid {
    /// Grid cell size (in meters)
    cell_size: f64,
    /// Hash map from cell coordinates to particle indices
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    /// Maximum interaction range (determines cell size)
    max_interaction_range: f64,
}

impl SpatialHashGrid {
    /// Create a new spatial hash grid
    pub fn new(max_interaction_range: f64) -> Self {
        // Cell size should be at least as large as max interaction range
        // to ensure particles only need to check neighboring cells
        let cell_size = max_interaction_range * 1.1; // 10% buffer
        
        Self {
            cell_size,
            grid: HashMap::new(),
            max_interaction_range,
        }
    }
    
    /// Clear and rebuild the grid with current particle positions
    pub fn update(&mut self, particles: &[FundamentalParticle]) {
        self.grid.clear();
        
        for (i, particle) in particles.iter().enumerate() {
            let cell_coords = self.position_to_cell(&particle.position);
            self.grid.entry(cell_coords).or_insert_with(Vec::new).push(i);
        }
    }
    
    /// Convert position to grid cell coordinates
    fn position_to_cell(&self, position: &Vector3<f64>) -> (i32, i32, i32) {
        // Clamp coordinates to prevent overflow when converting to i32
        const MAX_CELL_COORD: f64 = (i32::MAX / 2) as f64; // Use half of i32::MAX for safety
        const MIN_CELL_COORD: f64 = (i32::MIN / 2) as f64;
        
        let x = (position.x / self.cell_size).floor().clamp(MIN_CELL_COORD, MAX_CELL_COORD) as i32;
        let y = (position.y / self.cell_size).floor().clamp(MIN_CELL_COORD, MAX_CELL_COORD) as i32;
        let z = (position.z / self.cell_size).floor().clamp(MIN_CELL_COORD, MAX_CELL_COORD) as i32;
        (x, y, z)
    }
    
    /// Find all potential interaction partners for a particle
    pub fn find_neighbors(&self, particle_idx: usize, particles: &[FundamentalParticle]) -> Vec<usize> {
        if particle_idx >= particles.len() {
            return Vec::new();
        }
        
        let particle = &particles[particle_idx];
        let center_cell = self.position_to_cell(&particle.position);
        let mut neighbors = Vec::new();
        
        // Check the center cell and all 26 neighboring cells (3x3x3 - 1)
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    // Use checked_add to prevent overflow
                    let cell = match (
                        center_cell.0.checked_add(dx),
                        center_cell.1.checked_add(dy),
                        center_cell.2.checked_add(dz),
                    ) {
                        (Some(x), Some(y), Some(z)) => (x, y, z),
                        _ => continue, // Skip this cell if overflow would occur
                    };
                    
                    if let Some(cell_particles) = self.grid.get(&cell) {
                        for &other_idx in cell_particles {
                            if other_idx != particle_idx {
                                let distance = (particles[other_idx].position - particle.position).norm();
                                if distance <= self.max_interaction_range {
                                    neighbors.push(other_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        neighbors
    }
    
    /// Find all particle pairs within interaction range (optimized)
    pub fn find_interaction_pairs(&self, particles: &[FundamentalParticle]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let mut processed = HashSet::new();
        
        for &particle_idx in self.grid.values().flatten() {
            if processed.contains(&particle_idx) {
                continue;
            }
            
            let neighbors = self.find_neighbors(particle_idx, particles);
            for neighbor_idx in neighbors {
                // Avoid duplicate pairs
                if particle_idx < neighbor_idx {
                    pairs.push((particle_idx, neighbor_idx));
                }
            }
            
            processed.insert(particle_idx);
        }
        
        pairs
    }
    
    /// Get grid statistics for diagnostics
    pub fn get_statistics(&self) -> SpatialGridStats {
        let total_cells = self.grid.len();
        let mut total_particles = 0;
        let mut max_particles_per_cell = 0;
        let mut non_empty_cells = 0;
        
        for cell_particles in self.grid.values() {
            let count = cell_particles.len();
            total_particles += count;
            max_particles_per_cell = max_particles_per_cell.max(count);
            if count > 0 {
                non_empty_cells += 1;
            }
        }
        
        let avg_particles_per_cell = if non_empty_cells > 0 {
            total_particles as f64 / non_empty_cells as f64
        } else {
            0.0
        };
        
        SpatialGridStats {
            cell_size: self.cell_size,
            total_cells,
            non_empty_cells,
            total_particles,
            max_particles_per_cell,
            avg_particles_per_cell,
        }
    }

    /// Get the maximum interaction range (in meters) configured for this grid.
    #[inline]
    pub fn max_interaction_range(&self) -> f64 {
        self.max_interaction_range
    }
}

/// Statistics about the spatial grid
#[derive(Debug, Clone)]
pub struct SpatialGridStats {
    pub cell_size: f64,
    pub total_cells: usize,
    pub non_empty_cells: usize,
    pub total_particles: usize,
    pub max_particles_per_cell: usize,
    pub avg_particles_per_cell: f64,
}

/// Hierarchical spatial partitioning using octree
#[derive(Debug, Clone)]
pub struct SpatialOctree {
    root: OctreeNode,
    max_depth: usize,
    max_particles_per_node: usize,
    bounds: (Vector3<f64>, Vector3<f64>), // (min, max)
}

#[derive(Debug, Clone)]
struct OctreeNode {
    bounds: (Vector3<f64>, Vector3<f64>),
    particles: Vec<usize>,
    children: Option<Box<[OctreeNode; 8]>>,
    depth: usize,
}

impl SpatialOctree {
    /// Create a new octree
    pub fn new(bounds: (Vector3<f64>, Vector3<f64>), max_depth: usize, max_particles_per_node: usize) -> Self {
        let root = OctreeNode {
            bounds,
            particles: Vec::new(),
            children: None,
            depth: 0,
        };
        
        Self {
            root,
            max_depth,
            max_particles_per_node,
            bounds,
        }
    }
    
    /// Rebuild the octree with current particle positions
    pub fn rebuild(&mut self, particles: &[FundamentalParticle]) {
        // Reset root node
        self.root = OctreeNode {
            bounds: self.bounds,
            particles: Vec::new(),
            children: None,
            depth: 0,
        };
        
        // Insert all particles
        for (i, particle) in particles.iter().enumerate() {
            self.insert_particle(i, &particle.position, particles);
        }
    }
    
    /// Insert a particle into the octree
    fn insert_particle(&mut self, particle_idx: usize, position: &Vector3<f64>, particles: &[FundamentalParticle]) {
        let position = *position; // Clone position to avoid borrow conflicts
        self.insert_recursive_with_position(particle_idx, position, particles);
    }
    
    /// Helper method to avoid borrowing conflicts
    fn insert_recursive_with_position(&mut self, particle_idx: usize, position: Vector3<f64>, particles: &[FundamentalParticle]) {
        // Implementation that doesn't create borrow conflicts
        if !point_in_bounds(&position, &self.root.bounds) {
            return;
        }
        // This is where the recursive insertion should start from the root
        insert_recursive_internal(&mut self.root, particle_idx, position, particles, self.max_depth, self.max_particles_per_node);
    }

    /// Find neighbors within a radius using the octree
    pub fn find_neighbors_in_radius(&self, query_point: &Vector3<f64>, radius: f64, particles: &[FundamentalParticle]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        self.find_neighbors_recursive(&self.root, query_point, radius, particles, &mut neighbors);
        neighbors
    }

    /// Recursive neighbor search
    fn find_neighbors_recursive(
        &self,
        node: &OctreeNode,
        query_point: &Vector3<f64>,
        radius: f64,
        particles: &[FundamentalParticle],
        result: &mut Vec<usize>,
    ) {
        // Check if query sphere intersects with node
        if !self.sphere_intersects_bounds(query_point, radius, &node.bounds) {
            return;
        }

        if let Some(ref children) = node.children {
            // Recursively search children
            for child in children.iter() {
                self.find_neighbors_recursive(child, query_point, radius, particles, result);
            }
        } else {
            // Leaf node - check all particles
            for &particle_idx in &node.particles {
                if particle_idx < particles.len() {
                    let distance = (particles[particle_idx].position - query_point).norm();
                    if distance <= radius {
                        result.push(particle_idx);
                    }
                }
            }
        }
    }

    /// Check if a sphere intersects with a bounding box
    fn sphere_intersects_bounds(&self, center: &Vector3<f64>, radius: f64, bounds: &(Vector3<f64>, Vector3<f64>)) -> bool {
        let (min, max) = bounds;

        // Find the closest point on the bounding box to the sphere center
        let closest_x = center.x.max(min.x).min(max.x);
        let closest_y = center.y.max(min.y).min(max.y);
        let closest_z = center.z.max(min.z).min(max.z);

        let closest_point = Vector3::new(closest_x, closest_y, closest_z);
        let distance = (closest_point - center).norm();

        distance <= radius
    }
}

/// Recursive particle insertion
fn insert_recursive_internal(
    node: &mut OctreeNode,
    particle_idx: usize,
    position: Vector3<f64>,
    particles: &[FundamentalParticle],
    max_depth: usize,
    max_particles_per_node: usize,
) {
    // Check if position is within node bounds
    if !point_in_bounds(&position, &node.bounds) {
        return;
    }

    // If this is a leaf node and hasn't reached capacity, add particle
    if node.children.is_none() {
        node.particles.push(particle_idx);

        // Subdivide if we've exceeded capacity and haven't reached max depth
        if node.particles.len() > max_particles_per_node && node.depth < max_depth {
            subdivide_node(node, particles);
        }
    } else {
        // Insert into appropriate child
        if let Some(ref mut children) = node.children {
            let child_idx = determine_child_index(&position, &node.bounds);
            insert_recursive_internal(&mut children[child_idx], particle_idx, position, particles, max_depth, max_particles_per_node);
        }
    }
}

impl OctreeNode {
    fn new(bounds: (Vector3<f64>, Vector3<f64>), depth: usize) -> Self {
        Self {
            bounds,
            particles: Vec::new(),
            children: None,
            depth,
        }
    }
}

/// Check if a point is within bounds
fn point_in_bounds(point: &Vector3<f64>, bounds: &(Vector3<f64>, Vector3<f64>)) -> bool {
    let (min, max) = bounds;
    point.x >= min.x && point.x <= max.x &&
    point.y >= min.y && point.y <= max.y &&
    point.z >= min.z && point.z <= max.z
}

/// Determine which child index a point belongs to
fn determine_child_index(point: &Vector3<f64>, bounds: &(Vector3<f64>, Vector3<f64>)) -> usize {
    let (min, max) = bounds;
    let center = (min + max) * 0.5;
    
    let mut index = 0;
    if point.x >= center.x { index += 1; }
    if point.y >= center.y { index += 2; }
    if point.z >= center.z { index += 4; }
    
    index
}

/// Subdivide a node into 8 children
fn subdivide_node(node: &mut OctreeNode, particles: &[FundamentalParticle]) {
    let (min, max) = node.bounds;
    let center = (min + max) * 0.5;
    
    let mut new_children: Box<[OctreeNode; 8]> = Box::new([
        // Bottom layer (z = min)
        OctreeNode::new((min, center), node.depth + 1),
        OctreeNode::new((Vector3::new(center.x, min.y, min.z), Vector3::new(max.x, center.y, center.z)), node.depth + 1),
        OctreeNode::new((Vector3::new(min.x, center.y, min.z), Vector3::new(center.x, max.y, center.z)), node.depth + 1),
        OctreeNode::new((Vector3::new(center.x, center.y, min.z), Vector3::new(max.x, max.y, center.z)), node.depth + 1),
        // Top layer (z = max)
        OctreeNode::new((Vector3::new(min.x, min.y, center.z), Vector3::new(center.x, center.y, max.z)), node.depth + 1),
        OctreeNode::new((Vector3::new(center.x, min.y, center.z), Vector3::new(max.x, center.y, max.z)), node.depth + 1),
        OctreeNode::new((Vector3::new(min.x, center.y, center.z), Vector3::new(center.x, max.y, max.z)), node.depth + 1),
        OctreeNode::new((center, max), node.depth + 1),
    ]);
    
    // Redistribute particles to children
    let particles_to_redistribute = std::mem::take(&mut node.particles);
    for particle_idx in particles_to_redistribute {
        if particle_idx < particles.len() {
            let particle_position = particles[particle_idx].position;
            let child_idx = determine_child_index(&particle_position, &node.bounds);
            // Directly push to the new_children's particle list
            new_children[child_idx].particles.push(particle_idx);
        }
    }
    node.children = Some(new_children);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParticleType, FundamentalParticle};
    
    fn create_test_particle(position: Vector3<f64>) -> FundamentalParticle {
        FundamentalParticle {
            particle_type: ParticleType::Electron,
            position,
            momentum: Vector3::zeros(),
            spin: Vector3::new(Complex::new(0.5, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            color_charge: None,
            electric_charge: -1.602e-19,
            mass: 9.109e-31,
            energy: 1e-13,
            creation_time: 0.0,
            decay_time: None,
            quantum_state: QuantumState::new(),
            interaction_history: Vec::new(),
            velocity: Vector3::zeros(),
            charge: -1.602e-19,
        }
    }
    
    #[test]
    fn test_spatial_hash_grid() {
        let mut grid = SpatialHashGrid::new(1e-14); // 10 fm interaction range
        
        let particles = vec![
            create_test_particle(Vector3::new(0.0, 0.0, 0.0)),
            create_test_particle(Vector3::new(5e-15, 0.0, 0.0)),  // 5 fm away
            create_test_particle(Vector3::new(2e-14, 0.0, 0.0)),  // 20 fm away
        ];
        
        grid.update(&particles);
        
        // First particle should have second as neighbor, but not third
        let neighbors = grid.find_neighbors(0, &particles);
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
        
        // Get statistics
        let stats = grid.get_statistics();
        assert_eq!(stats.total_particles, 3);
    }
    
    #[test]
    fn test_octree() {
        let bounds = (
            Vector3::new(-1e-13, -1e-13, -1e-13),
            Vector3::new(1e-13, 1e-13, 1e-13),
        );
        let mut octree = SpatialOctree::new(bounds, 5, 4);
        
        let particles = vec![
            create_test_particle(Vector3::new(0.0, 0.0, 0.0)),
            create_test_particle(Vector3::new(5e-14, 5e-14, 5e-14)),
            create_test_particle(Vector3::new(-3e-14, -3e-14, -3e-14)),
        ];
        
        octree.rebuild(&particles);
        
        // Find neighbors within 1e-13 m radius
        let neighbors = octree.find_neighbors_in_radius(&Vector3::zeros(), 1e-13, &particles);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }
} 