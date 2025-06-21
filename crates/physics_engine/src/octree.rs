//! Octree for spatial partitioning of particles with Barnes-Hut tree algorithm.
//! 
//! This module implements the Barnes-Hut tree algorithm for efficient N-body gravitational
//! force calculations, reducing complexity from O(N²) to O(N log N).
//! 
//! The algorithm works by:
//! 1. Building a hierarchical octree from particle positions
//! 2. Computing mass centers and multipole moments for each node
//! 3. Using the opening criterion θ = s/d to decide when to approximate distant groups
//! 4. Computing forces by traversing the tree with the opening criterion
//! 
//! Reference: Barnes & Hut (1986) "A hierarchical O(N log N) force-calculation algorithm"

use nalgebra::Vector3;
use std::collections::VecDeque;
use anyhow::Result;

const MAX_PARTICLES_PER_NODE: usize = 8;
const MAX_DEPTH: u32 = 16;

#[derive(Debug, Clone)]
pub struct Octree {
    root: Box<OctreeNode>,
    /// Opening criterion θ for Barnes-Hut algorithm (typically 0.5-1.0)
    theta: f64,
    /// Gravitational constant G
    g_constant: f64,
}

#[derive(Debug, Clone)]
struct OctreeNode {
    boundary: AABB,
    particles: Vec<(usize, Vector3<f64>)>, // (index, position)
    children: Option<[Box<OctreeNode>; 8]>,
    is_leaf: bool,
    
    // Barnes-Hut specific fields
    mass_center: Vector3<f64>,    // Center of mass
    total_mass: f64,              // Total mass in this node
    multipole_moment: f64,        // Quadrupole moment for higher-order corrections
    node_size: f64,               // Size of this node's bounding box
}

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    center: Vector3<f64>,
    half_dimension: Vector3<f64>,
}

impl AABB {
    pub fn new(center: Vector3<f64>, half_dimension: Vector3<f64>) -> Self {
        Self { center, half_dimension }
    }

    pub fn contains(&self, point: &Vector3<f64>) -> bool {
        (self.center.x - self.half_dimension.x <= point.x && point.x <= self.center.x + self.half_dimension.x) &&
        (self.center.y - self.half_dimension.y <= point.y && point.y <= self.center.y + self.half_dimension.y) &&
        (self.center.z - self.half_dimension.z <= point.z && point.z <= self.center.z + self.half_dimension.z)
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        (self.center.x - self.half_dimension.x < other.center.x + other.half_dimension.x && self.center.x + self.half_dimension.x > other.center.x - other.half_dimension.x) &&
        (self.center.y - self.half_dimension.y < other.center.y + other.half_dimension.y && self.center.y + self.half_dimension.y > other.center.y - other.half_dimension.y) &&
        (self.center.z - self.half_dimension.z < other.center.z + other.half_dimension.z && self.center.z + self.half_dimension.z > other.center.z - other.half_dimension.z)
    }
    
    /// Get the size (diameter) of the bounding box
    pub fn size(&self) -> f64 {
        2.0 * self.half_dimension.max()
    }
}

impl OctreeNode {
    fn new(boundary: AABB) -> Self {
        Self {
            boundary,
            particles: Vec::with_capacity(MAX_PARTICLES_PER_NODE),
            children: None,
            is_leaf: true,
            mass_center: Vector3::zeros(),
            total_mass: 0.0,
            multipole_moment: 0.0,
            node_size: boundary.size(),
        }
    }

    fn subdivide(&mut self, depth: u32) {
        let hd = self.boundary.half_dimension / 2.0;
        let center = self.boundary.center;

        let children_boundaries = [
            AABB::new(Vector3::new(center.x - hd.x, center.y - hd.y, center.z - hd.z), hd), // 0
            AABB::new(Vector3::new(center.x + hd.x, center.y - hd.y, center.z - hd.z), hd), // 1
            AABB::new(Vector3::new(center.x - hd.x, center.y + hd.y, center.z - hd.z), hd), // 2
            AABB::new(Vector3::new(center.x + hd.x, center.y + hd.y, center.z - hd.z), hd), // 3
            AABB::new(Vector3::new(center.x - hd.x, center.y - hd.y, center.z + hd.z), hd), // 4
            AABB::new(Vector3::new(center.x + hd.x, center.y - hd.y, center.z + hd.z), hd), // 5
            AABB::new(Vector3::new(center.x - hd.x, center.y + hd.y, center.z + hd.z), hd), // 6
            AABB::new(Vector3::new(center.x + hd.x, center.y + hd.y, center.z + hd.z), hd), // 7
        ];

        self.children = Some([
            Box::new(OctreeNode::new(children_boundaries[0])),
            Box::new(OctreeNode::new(children_boundaries[1])),
            Box::new(OctreeNode::new(children_boundaries[2])),
            Box::new(OctreeNode::new(children_boundaries[3])),
            Box::new(OctreeNode::new(children_boundaries[4])),
            Box::new(OctreeNode::new(children_boundaries[5])),
            Box::new(OctreeNode::new(children_boundaries[6])),
            Box::new(OctreeNode::new(children_boundaries[7])),
        ]);
        self.is_leaf = false;

        // Move particles to children
        let particles = std::mem::take(&mut self.particles);
        for (idx, pos) in particles {
            let quadrant = self.get_quadrant(&pos);
            if let Some(children) = &mut self.children {
                children[quadrant].insert(idx, &pos, depth + 1);
            }
        }
    }

    fn get_quadrant(&self, point: &Vector3<f64>) -> usize {
        let mut index = 0;
        if point.x > self.boundary.center.x { index |= 1; }
        if point.y > self.boundary.center.y { index |= 2; }
        if point.z > self.boundary.center.z { index |= 4; }
        index
    }

    fn insert(&mut self, particle_index: usize, position: &Vector3<f64>, depth: u32) -> bool {
        if !self.boundary.contains(position) {
            return false;
        }

        if self.is_leaf {
            if self.particles.len() < MAX_PARTICLES_PER_NODE || depth >= MAX_DEPTH {
                self.particles.push((particle_index, *position));
                return true;
            } else {
                self.subdivide(depth);
            }
        }
        
        let quadrant = self.get_quadrant(position);
        if let Some(children) = &mut self.children {
            return children[quadrant].insert(particle_index, position, depth + 1);
        }

        unreachable!("Should have children after subdivide");
    }
    
    /// Compute mass center and multipole moments for Barnes-Hut algorithm
    fn compute_mass_properties(&mut self, masses: &[f64]) {
        if self.is_leaf {
            // Leaf node: compute from actual particles
            let mut total_mass = 0.0;
            let mut weighted_position = Vector3::zeros();
            
            for (particle_idx, _) in &self.particles {
                let mass = masses[*particle_idx];
                total_mass += mass;
                weighted_position += masses[*particle_idx] * self.particles.iter()
                    .find(|(idx, _)| idx == particle_idx)
                    .map(|(_, pos)| *pos)
                    .unwrap_or(Vector3::zeros());
            }
            
            if total_mass > 0.0 {
                self.mass_center = weighted_position / total_mass;
            } else {
                self.mass_center = self.boundary.center;
            }
            self.total_mass = total_mass;
            
            // Simple quadrupole moment approximation
            self.multipole_moment = 0.0;
            for (particle_idx, _) in &self.particles {
                let mass = masses[*particle_idx];
                let pos = self.particles.iter()
                    .find(|(idx, _)| idx == particle_idx)
                    .map(|(_, pos)| *pos)
                    .unwrap_or(Vector3::zeros());
                let r = (pos - self.mass_center).norm();
                self.multipole_moment += mass * r * r;
            }
        } else {
            // Internal node: compute from children
            let mut total_mass = 0.0;
            let mut weighted_position = Vector3::zeros();
            let mut total_multipole = 0.0;
            
            if let Some(children) = &mut self.children {
                for child in children.iter_mut() {
                    child.compute_mass_properties(masses);
                    total_mass += child.total_mass;
                    weighted_position += child.total_mass * child.mass_center;
                    total_multipole += child.multipole_moment;
                }
            }
            
            if total_mass > 0.0 {
                self.mass_center = weighted_position / total_mass;
            } else {
                self.mass_center = self.boundary.center;
            }
            self.total_mass = total_mass;
            
            // Add contribution from child mass centers
            if let Some(children) = &self.children {
                for child in children.iter() {
                    let r = (child.mass_center - self.mass_center).norm();
                    total_multipole += child.total_mass * r * r;
                }
            }
            self.multipole_moment = total_multipole;
        }
    }
    
    /// Compute gravitational force on a particle using Barnes-Hut algorithm
    fn compute_force(&self, 
                    particle_pos: &Vector3<f64>, 
                    particle_mass: f64,
                    g_constant: f64,
                    theta: f64,
                    softening_length: f64) -> Vector3<f64> {
        let mut force = Vector3::zeros();
        let distance = (self.mass_center - *particle_pos).norm();
        
        // Skip self-interaction
        if distance < 1e-15 {
            return force;
        }
        
        // Barnes-Hut opening criterion: θ = s/d
        // If θ < s/d, we can approximate this node as a point mass
        let opening_criterion = self.node_size / distance;
        
        if opening_criterion < theta || self.is_leaf {
            // Use this node as a point mass
            let softened_distance = (distance * distance + softening_length * softening_length).sqrt();
            let force_magnitude = g_constant * particle_mass * self.total_mass / (softened_distance * softened_distance);
            let direction = (self.mass_center - *particle_pos).normalize();
            force += direction * force_magnitude;
        } else {
            // Recursively compute force from children
            if let Some(children) = &self.children {
                for child in children.iter() {
                    force += child.compute_force(particle_pos, particle_mass, g_constant, theta, softening_length);
                }
            }
        }
        
        force
    }
}

impl Octree {
    pub fn new(boundary: AABB) -> Self {
        Self {
            root: Box::new(OctreeNode::new(boundary)),
            theta: 0.5, // Standard Barnes-Hut opening criterion
            g_constant: 6.67430e-11, // Gravitational constant
        }
    }
    
    /// Create a new Barnes-Hut tree with custom parameters
    pub fn new_barnes_hut(boundary: AABB, theta: f64, g_constant: f64) -> Self {
        Self {
            root: Box::new(OctreeNode::new(boundary)),
            theta,
            g_constant,
        }
    }

    pub fn insert(&mut self, particle_index: usize, position: &Vector3<f64>) {
        // A real implementation would handle resizing the octree if a particle is outside.
        // For now, we assume all particles are within the initial boundary.
        self.root.insert(particle_index, position, 0);
    }
    
    /// Build the Barnes-Hut tree and compute mass properties
    pub fn build_tree(&mut self, positions: &[Vector3<f64>], masses: &[f64]) -> Result<()> {
        // Clear existing tree
        self.clear();
        
        // Insert all particles
        for (i, pos) in positions.iter().enumerate() {
            self.insert(i, pos);
        }
        
        // Compute mass properties for Barnes-Hut algorithm
        self.root.compute_mass_properties(masses);
        
        Ok(())
    }
    
    /// Compute gravitational forces using Barnes-Hut algorithm
    pub fn compute_gravitational_forces(&self, 
                                      positions: &[Vector3<f64>], 
                                      masses: &[f64],
                                      softening_length: f64) -> Vec<Vector3<f64>> {
        let mut forces = Vec::with_capacity(positions.len());
        
        for (i, pos) in positions.iter().enumerate() {
            let force = self.root.compute_force(pos, masses[i], self.g_constant, self.theta, softening_length);
            forces.push(force);
        }
        
        forces
    }
    
    /// Compute gravitational forces in parallel using Barnes-Hut algorithm
    pub fn compute_gravitational_forces_parallel(&self, 
                                               positions: &[Vector3<f64>], 
                                               masses: &[f64],
                                               softening_length: f64) -> Vec<Vector3<f64>> {
        use rayon::prelude::*;
        
        (0..positions.len())
            .into_par_iter()
            .map(|i| {
                self.root.compute_force(&positions[i], masses[i], self.g_constant, self.theta, softening_length)
            })
            .collect()
    }

    pub fn query_range(&self, range: &AABB) -> Vec<usize> {
        let mut found_indices = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(&*self.root);

        while let Some(node) = queue.pop_front() {
            if !node.boundary.intersects(range) {
                continue;
            }

            for (particle_index, particle_pos) in &node.particles {
                if range.contains(particle_pos) {
                    found_indices.push(*particle_index);
                }
            }

            if !node.is_leaf {
                if let Some(children) = &node.children {
                    for child in children.iter() {
                        queue.push_back(child);
                    }
                }
            }
        }
        found_indices
    }
    
    pub fn clear(&mut self) {
        self.root.particles.clear();
        self.root.children = None;
        self.root.is_leaf = true;
        self.root.mass_center = Vector3::zeros();
        self.root.total_mass = 0.0;
        self.root.multipole_moment = 0.0;
    }
    
    /// Get the opening criterion θ
    pub fn theta(&self) -> f64 {
        self.theta
    }
    
    /// Set the opening criterion θ
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
    }
    
    /// Get the gravitational constant
    pub fn g_constant(&self) -> f64 {
        self.g_constant
    }
    
    /// Set the gravitational constant
    pub fn set_g_constant(&mut self, g_constant: f64) {
        self.g_constant = g_constant;
    }
    
    /// Get tree statistics for performance analysis
    pub fn get_stats(&self) -> TreeStats {
        let mut stats = TreeStats {
            total_nodes: 0,
            leaf_nodes: 0,
            max_depth: 0,
            total_particles: 0,
        };
        
        self.compute_stats_recursive(&*self.root, 0, &mut stats);
        stats
    }
    
    fn compute_stats_recursive(&self, node: &OctreeNode, depth: u32, stats: &mut TreeStats) {
        stats.total_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        
        if node.is_leaf {
            stats.leaf_nodes += 1;
            stats.total_particles += node.particles.len();
        } else {
            if let Some(children) = &node.children {
                for child in children.iter() {
                    self.compute_stats_recursive(child, depth + 1, stats);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: u32,
    pub total_particles: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_barnes_hut_tree_creation() {
        let boundary = AABB::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0)
        );
        let tree = Octree::new_barnes_hut(boundary, 0.5, 6.67430e-11);
        assert_eq!(tree.theta(), 0.5);
        assert_eq!(tree.g_constant(), 6.67430e-11);
    }
    
    #[test]
    fn test_barnes_hut_force_calculation() {
        let boundary = AABB::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 10.0, 10.0)
        );
        let mut tree = Octree::new_barnes_hut(boundary, 0.5, 6.67430e-11);
        
        // Create a simple 2-particle system
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let masses = vec![1.0, 1.0];
        
        // Build the tree
        tree.build_tree(&positions, &masses).unwrap();
        
        // Compute forces
        let forces = tree.compute_gravitational_forces(&positions, &masses, 0.01);
        
        // Check that forces are computed (should be non-zero for 2-body system)
        assert_eq!(forces.len(), 2);
        assert!(forces[0].norm() > 0.0);
        assert!(forces[1].norm() > 0.0);
        
        // Check that forces are equal and opposite (Newton's 3rd law)
        let force_diff = (forces[0] + forces[1]).norm();
        assert!(force_diff < 1e-10, "Forces should be equal and opposite, got: {:?}", force_diff);
    }
    
    #[test]
    fn test_tree_stats() {
        let boundary = AABB::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0)
        );
        let mut tree = Octree::new(boundary);
        
        // Insert some particles
        for i in 0..10 {
            let pos = Vector3::new(i as f64 * 0.1, 0.0, 0.0);
            tree.insert(i, &pos);
        }
        
        let stats = tree.get_stats();
        assert!(stats.total_nodes > 0);
        assert!(stats.total_particles == 10);
    }
} 