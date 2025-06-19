//! Octree for spatial partitioning of particles.
use nalgebra::Vector3;
use std::collections::VecDeque;

const MAX_PARTICLES_PER_NODE: usize = 8;
const MAX_DEPTH: u32 = 16;

#[derive(Debug, Clone)]
pub struct Octree {
    root: Box<OctreeNode>,
}

#[derive(Debug, Clone)]
struct OctreeNode {
    boundary: AABB,
    particles: Vec<(usize, Vector3<f64>)>, // (index, position)
    children: Option<[Box<OctreeNode>; 8]>,
    is_leaf: bool,
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
}

impl OctreeNode {
    fn new(boundary: AABB) -> Self {
        Self {
            boundary,
            particles: Vec::with_capacity(MAX_PARTICLES_PER_NODE),
            children: None,
            is_leaf: true,
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
}

impl Octree {
    pub fn new(boundary: AABB) -> Self {
        Self {
            root: Box::new(OctreeNode::new(boundary)),
        }
    }

    pub fn insert(&mut self, particle_index: usize, position: &Vector3<f64>) {
        // A real implementation would handle resizing the octree if a particle is outside.
        // For now, we assume all particles are within the initial boundary.
        self.root.insert(particle_index, position, 0);
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
    }
} 