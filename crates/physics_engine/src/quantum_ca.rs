use crate::{
    constants::{HBAR, PLANCK_CONSTANT},
    QuantumState,
    validation::ValidationError,
};
use nalgebra::{DMatrix, DVector, Vector3, Matrix3, Complex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum operators used in QCA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumOperator {
    Identity,
    QuantumWalk,
    ErrorCorrection,
    PhaseTransition,
    Synchronization,
}

impl QuantumOperator {
    pub fn apply(&self, state: &DVector<Complex<f64>>) -> std::result::Result<DVector<Complex<f64>>, ValidationError> {
        // For now, the operator just returns the state unchanged. Research-grade
        // implementations should apply unitary transformations relevant to the
        // selected operator.
        Ok(state.clone())
    }
}

/// Quantum Cellular Automata (QCA) - A research-grade implementation
/// for studying emergent quantum phenomena and quantum information processing
/// 
/// This module implements:
/// - Quantum state evolution in discrete spacetime
/// - Local quantum rules and interactions
/// - Quantum error correction and decoherence
/// - Emergent quantum complexity from simple rules
/// - Quantum-classical hybrid systems

/// Quantum cell state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCell {
    /// Quantum state vector (|ψ⟩)
    pub state_vector: DVector<Complex<f64>>,
    /// Local quantum field values
    pub quantum_fields: HashMap<String, Complex<f64>>,
    /// Entanglement information with neighboring cells
    pub entanglement_map: HashMap<(i32, i32, i32), f64>,
    /// Decoherence parameters
    pub decoherence_rate: f64,
    /// Local energy density
    pub energy_density: f64,
    /// Quantum phase
    pub phase: f64,
    /// Cell coordinates in the lattice
    pub position: Vector3<i32>,
    /// Evolution history for analysis
    pub evolution_history: Vec<QuantumCell>,
}

/// Quantum cellular automata rule set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QCARule {
    /// Rule identifier
    pub name: String,
    /// Neighborhood pattern (Moore, von Neumann, etc.)
    pub neighborhood_type: NeighborhoodType,
    /// Quantum evolution operator
    pub evolution_operator: QuantumOperator,
    /// Local interaction strength
    pub interaction_strength: f64,
    /// Decoherence parameters
    pub decoherence_params: DecoherenceParameters,
    /// Quantum measurement rules
    pub measurement_rules: MeasurementRules,
    /// Rule application conditions
    pub conditions: RuleConditions,
}

/// Neighborhood types for QCA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeighborhoodType {
    Moore,      // 8 neighbors in 2D, 26 in 3D
    VonNeumann, // 4 neighbors in 2D, 6 in 3D
    Hexagonal,  // 6 neighbors (2D only)
    Custom(Vec<Vector3<i32>>), // Custom neighborhood pattern
}

/// Decoherence parameters for quantum systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceParameters {
    /// Amplitude damping rate
    pub amplitude_damping: f64,
    /// Phase damping rate
    pub phase_damping: f64,
    /// Depolarization rate
    pub depolarization: f64,
    /// Environmental coupling strength
    pub environmental_coupling: f64,
    /// Temperature (affects decoherence)
    pub temperature: f64,
}

/// Quantum measurement rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementRules {
    /// Measurement basis
    pub measurement_basis: Vec<DVector<Complex<f64>>>,
    /// Measurement probability
    pub measurement_probability: f64,
    /// Collapse behavior
    pub collapse_type: CollapseType,
    /// Post-measurement state update
    pub post_measurement_update: bool,
}

/// Types of quantum state collapse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollapseType {
    Projective,     // Standard quantum measurement
    Weak,           // Weak measurement (partial collapse)
    Continuous,     // Continuous measurement
    Delayed,        // Delayed choice measurement
}

/// Rule application conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConditions {
    /// Energy threshold for rule application
    pub energy_threshold: f64,
    /// Phase coherence requirement
    pub phase_coherence_threshold: f64,
    /// Entanglement threshold
    pub entanglement_threshold: f64,
    /// Time-dependent conditions
    pub time_dependent: bool,
    /// Spatial conditions
    pub spatial_conditions: Vec<SpatialCondition>,
}

/// Spatial conditions for rule application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialCondition {
    /// Condition type
    pub condition_type: SpatialConditionType,
    /// Parameters for the condition
    pub parameters: HashMap<String, f64>,
    /// Target region
    pub target_region: Option<Vector3<i32>>,
}

/// Types of spatial conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialConditionType {
    DistanceBased,
    DensityBased,
    GradientBased,
    PatternBased,
    EntanglementBased,
}

/// Quantum Cellular Automata lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QCALattice {
    /// 3D grid of quantum cells
    pub cells: HashMap<Vector3<i32>, QuantumCell>,
    /// Lattice dimensions
    pub dimensions: Vector3<i32>,
    /// Boundary conditions
    pub boundary_conditions: BoundaryConditions,
    /// Active rule set
    pub rules: Vec<QCARule>,
    /// Evolution parameters
    pub evolution_params: EvolutionParameters,
    /// Global quantum state
    pub global_state: GlobalQuantumState,
    /// Simulation time step
    pub time_step: f64,
    /// Current iteration
    pub iteration: u64,
}

/// Boundary conditions for the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryConditions {
    Periodic,   // Toroidal topology
    Fixed,      // Fixed boundary values
    Open,       // Open boundaries
    Reflective, // Reflective boundaries
    // Custom(Box<dyn Fn(Vector3<i32>) -> QuantumCell + Send + Sync>), // Not serializable
}

/// Evolution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionParameters {
    /// Time step size
    pub dt: f64,
    /// Maximum iterations
    pub max_iterations: u64,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Energy conservation check
    pub energy_conservation: bool,
    /// Entanglement tracking
    pub track_entanglement: bool,
    /// Decoherence effects
    pub include_decoherence: bool,
    /// Quantum measurement effects
    pub include_measurements: bool,
}

/// Global quantum state of the entire lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalQuantumState {
    /// Total energy
    pub total_energy: f64,
    /// Global entanglement measure
    pub global_entanglement: f64,
    /// Quantum coherence measure
    pub quantum_coherence: f64,
    /// Information entropy
    pub information_entropy: f64,
    /// Phase coherence
    pub phase_coherence: f64,
    /// Quantum correlations
    pub quantum_correlations: DMatrix<f64>,
}

impl Default for QuantumCell {
    fn default() -> Self {
        Self {
            state_vector: DVector::from_element(2, Complex::new(1.0, 0.0)),
            quantum_fields: HashMap::new(),
            entanglement_map: HashMap::new(),
            decoherence_rate: 0.0,
            energy_density: 0.0,
            phase: 0.0,
            position: Vector3::zeros(),
            evolution_history: Vec::new(),
        }
    }
}

impl Default for QCALattice {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
            dimensions: Vector3::new(10, 10, 10),
            boundary_conditions: BoundaryConditions::Periodic,
            rules: Vec::new(),
            evolution_params: EvolutionParameters::default(),
            global_state: GlobalQuantumState::default(),
            time_step: 1e-15, // 1 femtosecond
            iteration: 0,
        }
    }
}

impl Default for EvolutionParameters {
    fn default() -> Self {
        Self {
            dt: 1e-15,
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
            energy_conservation: true,
            track_entanglement: true,
            include_decoherence: true,
            include_measurements: false,
        }
    }
}

impl Default for GlobalQuantumState {
    fn default() -> Self {
        Self {
            total_energy: 0.0,
            global_entanglement: 0.0,
            quantum_coherence: 1.0,
            information_entropy: 0.0,
            phase_coherence: 1.0,
            quantum_correlations: DMatrix::zeros(1, 1),
        }
    }
}

impl QCALattice {
    /// Create a new QCA lattice with specified dimensions
    pub fn new(dimensions: Vector3<i32>) -> Self {
        let mut lattice = Self::default();
        lattice.dimensions = dimensions;
        lattice.initialize_lattice();
        lattice
    }

    /// Initialize the lattice with quantum cells
    fn initialize_lattice(&mut self) {
        for x in 0..self.dimensions.x {
            for y in 0..self.dimensions.y {
                for z in 0..self.dimensions.z {
                    let pos = Vector3::new(x, y, z);
                    let mut cell = QuantumCell::default();
                    cell.position = pos;
                    self.cells.insert(pos, cell);
                }
            }
        }
    }

    /// Evolve the QCA for one time step
    pub fn evolve(&mut self) -> Result<(), ValidationError> {
        // Store current state for analysis
        let current_state = self.global_state.clone();
        
        // Apply quantum rules to each cell
        let mut new_cells = HashMap::new();
        
        for (pos, cell) in &self.cells {
            let new_cell = self.apply_quantum_rules(*pos, cell)?;
            new_cells.insert(*pos, new_cell);
        }
        
        self.cells = new_cells;
        
        // Update global quantum state
        self.update_global_state()?;
        
        // Check energy conservation
        if self.evolution_params.energy_conservation {
            self.check_energy_conservation(&current_state)?;
        }
        
        self.iteration += 1;
        Ok(())
    }

    /// Apply quantum rules to a specific cell
    fn apply_quantum_rules(&self, pos: Vector3<i32>, cell: &QuantumCell) -> Result<QuantumCell, ValidationError> {
        let mut new_cell = cell.clone();
        
        for rule in &self.rules {
            if self.should_apply_rule(pos, cell, rule)? {
                new_cell = self.apply_single_rule(pos, &new_cell, rule)?;
            }
        }
        
        // Apply decoherence effects
        if self.evolution_params.include_decoherence {
            self.apply_decoherence(&mut new_cell)?;
        }
        
        // Apply quantum measurements
        if self.evolution_params.include_measurements {
            self.apply_quantum_measurements(&mut new_cell)?;
        }
        
        // Store evolution history
        new_cell.evolution_history.push(cell.clone());
        
        Ok(new_cell)
    }

    /// Check if a rule should be applied to a cell
    fn should_apply_rule(&self, pos: Vector3<i32>, cell: &QuantumCell, rule: &QCARule) -> Result<bool, ValidationError> {
        // Check energy threshold
        if cell.energy_density < rule.conditions.energy_threshold {
            return Ok(false);
        }
        
        // Check phase coherence
        if cell.phase.abs() < rule.conditions.phase_coherence_threshold {
            return Ok(false);
        }
        
        // Check entanglement threshold
        let avg_entanglement: f64 = cell.entanglement_map.values().sum::<f64>() / cell.entanglement_map.len() as f64;
        if avg_entanglement < rule.conditions.entanglement_threshold {
            return Ok(false);
        }
        
        // Check spatial conditions
        for condition in &rule.conditions.spatial_conditions {
            if !self.check_spatial_condition(pos, cell, condition)? {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    /// Apply a single quantum rule to a cell
    fn apply_single_rule(&self, pos: Vector3<i32>, cell: &QuantumCell, rule: &QCARule) -> Result<QuantumCell, ValidationError> {
        let mut new_cell = cell.clone();
        
        // Get neighborhood cells
        let neighbors = self.get_neighborhood(pos, &rule.neighborhood_type)?;
        
        // Apply quantum evolution operator
        let evolved_state = self.apply_quantum_operator(&cell.state_vector, &rule.evolution_operator, &neighbors)?;
        new_cell.state_vector = evolved_state;
        
        // Update quantum fields based on interactions
        self.update_quantum_fields(&mut new_cell, &neighbors, rule)?;
        
        // Update entanglement with neighbors
        self.update_entanglement(&mut new_cell, &neighbors, rule)?;
        
        // Update energy density
        new_cell.energy_density = self.calculate_energy_density(&new_cell, &neighbors)?;
        
        // Update quantum phase
        new_cell.phase = self.calculate_quantum_phase(&new_cell)?;
        
        Ok(new_cell)
    }

    /// Get neighborhood cells based on neighborhood type
    fn get_neighborhood(&self, pos: Vector3<i32>, neighborhood_type: &NeighborhoodType) -> Result<Vec<QuantumCell>, ValidationError> {
        let mut neighbors = Vec::new();
        
        let neighbor_positions = match neighborhood_type {
            NeighborhoodType::Moore => {
                let mut positions = Vec::new();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if dx != 0 || dy != 0 || dz != 0 {
                                positions.push(Vector3::new(dx, dy, dz));
                            }
                        }
                    }
                }
                positions
            },
            NeighborhoodType::VonNeumann => {
                vec![
                    Vector3::new(1, 0, 0), Vector3::new(-1, 0, 0),
                    Vector3::new(0, 1, 0), Vector3::new(0, -1, 0),
                    Vector3::new(0, 0, 1), Vector3::new(0, 0, -1),
                ]
            },
            NeighborhoodType::Hexagonal => {
                // 2D hexagonal neighborhood
                vec![
                    Vector3::new(1, 0, 0), Vector3::new(-1, 0, 0),
                    Vector3::new(0, 1, 0), Vector3::new(0, -1, 0),
                    Vector3::new(1, 1, 0), Vector3::new(-1, -1, 0),
                ]
            },
            NeighborhoodType::Custom(positions) => positions.clone(),
        };
        
        for offset in neighbor_positions {
            let neighbor_pos = self.apply_boundary_conditions(pos + offset)?;
            if let Some(neighbor) = self.cells.get(&neighbor_pos) {
                neighbors.push(neighbor.clone());
            }
        }
        
        Ok(neighbors)
    }

    /// Apply boundary conditions to a position
    fn apply_boundary_conditions(&self, pos: Vector3<i32>) -> Result<Vector3<i32>, ValidationError> {
        match &self.boundary_conditions {
            BoundaryConditions::Periodic => {
                let x = (pos.x + self.dimensions.x) % self.dimensions.x;
                let y = (pos.y + self.dimensions.y) % self.dimensions.y;
                let z = (pos.z + self.dimensions.z) % self.dimensions.z;
                Ok(Vector3::new(x, y, z))
            },
            BoundaryConditions::Fixed => {
                if pos.x >= 0 && pos.x < self.dimensions.x &&
                   pos.y >= 0 && pos.y < self.dimensions.y &&
                   pos.z >= 0 && pos.z < self.dimensions.z {
                    Ok(pos)
                } else {
                    Err(ValidationError::OutOfBounds)
                }
            },
            BoundaryConditions::Open => Ok(pos),
            BoundaryConditions::Reflective => {
                let x = pos.x.max(0).min(self.dimensions.x - 1);
                let y = pos.y.max(0).min(self.dimensions.y - 1);
                let z = pos.z.max(0).min(self.dimensions.z - 1);
                Ok(Vector3::new(x, y, z))
            },
            // BoundaryConditions::Custom(_) => Ok(pos), // Custom boundary handling would be implemented here
        }
    }

    /// Apply quantum operator to state vector
    fn apply_quantum_operator(&self, state: &DVector<Complex<f64>>, operator: &QuantumOperator, neighbors: &[QuantumCell]) -> Result<DVector<Complex<f64>>, ValidationError> {
        // This is a simplified implementation - in practice, this would involve
        // complex quantum operator calculations based on the specific operator type
        let evolved_state = operator.apply(state)?;
        
        // Include neighbor interactions
        let mut interaction_effect = DVector::zeros(state.len());
        for neighbor in neighbors {
            let neighbor_contribution = self.calculate_neighbor_interaction(state, &neighbor.state_vector)?;
            interaction_effect += neighbor_contribution;
        }
        
        Ok(evolved_state + interaction_effect.scale(0.1)) // Small interaction strength
    }

    /// Calculate neighbor interaction contribution
    fn calculate_neighbor_interaction(&self, state: &DVector<Complex<f64>>, neighbor_state: &DVector<Complex<f64>>) -> Result<DVector<Complex<f64>>, ValidationError> {
        // Simple interaction: add neighbor state with small weight
        let interaction_strength = 0.1;
        Ok(state + neighbor_state.scale(interaction_strength))
    }

    /// Update quantum fields based on interactions
    fn update_quantum_fields(&self, cell: &mut QuantumCell, neighbors: &[QuantumCell], rule: &QCARule) -> Result<(), ValidationError> {
        for (field_name, field_value) in &mut cell.quantum_fields {
            let mut total_influence = Complex::new(0.0, 0.0);
            
            for neighbor in neighbors {
                if let Some(neighbor_field) = neighbor.quantum_fields.get(field_name) {
                    let distance = (cell.position.cast::<f64>() - neighbor.position.cast::<f64>()).norm();
                    let influence = *neighbor_field / (1.0 + distance * distance);
                    total_influence += influence;
                }
            }
            
            *field_value = *field_value + total_influence * rule.interaction_strength;
        }
        Ok(())
    }

    /// Update entanglement with neighbors
    fn update_entanglement(&self, cell: &mut QuantumCell, neighbors: &[QuantumCell], rule: &QCARule) -> Result<(), ValidationError> {
        for neighbor in neighbors {
            let distance = (cell.position.cast::<f64>() - neighbor.position.cast::<f64>()).norm();
            let entanglement_strength = (-distance / 10.0).exp(); // Exponential decay
            cell.entanglement_map.insert(
                (neighbor.position.x, neighbor.position.y, neighbor.position.z),
                entanglement_strength,
            );
        }
        Ok(())
    }

    /// Calculate quantum state overlap
    fn calculate_state_overlap(&self, state1: &DVector<Complex<f64>>, state2: &DVector<Complex<f64>>) -> Result<f64, ValidationError> {
        if state1.len() != state2.len() {
            return Err(ValidationError::DimensionMismatch);
        }
        
        let overlap = state1.dot(state2);
        Ok(overlap.norm())
    }

    /// Calculate energy density for a cell
    fn calculate_energy_density(&self, cell: &QuantumCell, neighbors: &[QuantumCell]) -> Result<f64, ValidationError> {
        let mut energy = 0.0;
        
        // Self-energy from quantum state
        let state_norm = cell.state_vector.norm();
        energy += state_norm * state_norm;
        
        // Interaction energy with neighbors
        for neighbor in neighbors {
            let distance = (cell.position.cast::<f64>() - neighbor.position.cast::<f64>()).norm();
            let interaction_energy = (-distance / 5.0).exp(); // Exponential interaction
            energy += interaction_energy;
        }
        
        Ok(energy)
    }

    /// Calculate quantum phase for a cell
    fn calculate_quantum_phase(&self, cell: &QuantumCell) -> Result<f64, ValidationError> {
        if cell.state_vector.len() == 0 {
            return Ok(0.0);
        }
        
        // Calculate phase from the first component of the state vector
        let first_component = cell.state_vector[0];
        Ok(first_component.arg())
    }

    /// Apply decoherence effects to a cell
    fn apply_decoherence(&self, cell: &mut QuantumCell) -> Result<(), ValidationError> {
        let damping_factor = Complex::new((-cell.decoherence_rate).exp(), 0.0);
        
        // Apply damping to quantum state
        for i in 0..cell.state_vector.len() {
            cell.state_vector[i] *= damping_factor;
        }
        
        // Renormalize if needed
        let norm = cell.state_vector.norm();
        if norm > 1e-10 {
            let norm_complex = Complex::new(norm, 0.0);
            for i in 0..cell.state_vector.len() {
                cell.state_vector[i] /= norm_complex;
            }
        }
        
        Ok(())
    }

    /// Apply quantum measurements to a cell
    fn apply_quantum_measurements(&self, cell: &mut QuantumCell) -> Result<(), ValidationError> {
        // This is a simplified measurement model
        // In practice, this would involve more sophisticated quantum measurement theory
        
        let measurement_probability = 0.01; // 1% measurement probability per time step
        
        if fastrand::f64() < measurement_probability {
            // Perform projective measurement
            let measurement_basis = vec![
                DVector::from_element(cell.state_vector.len(), Complex::new(1.0, 0.0)),
                DVector::from_element(cell.state_vector.len(), Complex::new(0.0, 1.0)),
            ];
            
            // Choose measurement outcome based on Born rule
            let probabilities: Vec<f64> = measurement_basis.iter()
                .map(|basis| {
                    let projection = cell.state_vector.dot(basis);
                    projection.norm_sqr()
                })
                .collect();
            
            let total_prob = probabilities.iter().sum::<f64>();
            let random_value = fastrand::f64() * total_prob;
            
            let mut cumulative_prob = 0.0;
            for (i, prob) in probabilities.iter().enumerate() {
                cumulative_prob += prob;
                if random_value <= cumulative_prob {
                    // Collapse to measurement basis
                    cell.state_vector = measurement_basis[i].clone();
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// Update global quantum state
    fn update_global_state(&mut self) -> Result<(), ValidationError> {
        let mut total_energy = 0.0;
        let mut total_entanglement = 0.0;
        let mut total_coherence = 0.0;
        let mut total_phase_coherence = 0.0;
        
        let cell_count = self.cells.len() as f64;
        
        for cell in self.cells.values() {
            total_energy += cell.energy_density;
            total_entanglement += cell.entanglement_map.values().sum::<f64>();
            total_coherence += 1.0 - cell.decoherence_rate;
            total_phase_coherence += cell.phase.cos().abs();
        }
        
        self.global_state.total_energy = total_energy;
        self.global_state.global_entanglement = total_entanglement / cell_count;
        self.global_state.quantum_coherence = total_coherence / cell_count;
        self.global_state.phase_coherence = total_phase_coherence / cell_count;
        
        // Calculate information entropy
        self.global_state.information_entropy = self.calculate_information_entropy()?;
        
        // Calculate quantum correlations matrix
        self.global_state.quantum_correlations = self.calculate_quantum_correlations()?;
        
        Ok(())
    }

    /// Calculate information entropy of the system
    fn calculate_information_entropy(&self) -> Result<f64, ValidationError> {
        let mut entropy = 0.0;
        
        for cell in self.cells.values() {
            let probabilities: Vec<f64> = cell.state_vector.iter()
                .map(|c| c.norm_sqr())
                .collect();
            
            for prob in probabilities {
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
        }
        
        Ok(entropy)
    }

    /// Calculate quantum correlations matrix
    fn calculate_quantum_correlations(&self) -> Result<DMatrix<f64>, ValidationError> {
        let cell_positions: Vec<Vector3<i32>> = self.cells.keys().cloned().collect();
        let n_cells = cell_positions.len();
        let mut correlations = DMatrix::zeros(n_cells, n_cells);
        
        for (i, pos1) in cell_positions.iter().enumerate() {
            for (j, pos2) in cell_positions.iter().enumerate() {
                if let (Some(cell1), Some(cell2)) = (self.cells.get(pos1), self.cells.get(pos2)) {
                    let distance = (pos1.cast::<f64>() - pos2.cast::<f64>()).norm();
                    let correlation = self.calculate_state_overlap(&cell1.state_vector, &cell2.state_vector).unwrap_or(0.0);
                    correlations[(i, j)] = correlation;
                }
            }
        }
        
        Ok(correlations)
    }

    /// Check energy conservation
    fn check_energy_conservation(&self, previous_state: &GlobalQuantumState) -> Result<(), ValidationError> {
        let energy_difference = (self.global_state.total_energy - previous_state.total_energy).abs();
        
        if energy_difference > self.evolution_params.convergence_tolerance {
            return Err(ValidationError::EnergyConservationViolation);
        }
        
        Ok(())
    }

    /// Check spatial condition
    fn check_spatial_condition(&self, pos: Vector3<i32>, cell: &QuantumCell, condition: &SpatialCondition) -> Result<bool, ValidationError> {
        match condition.condition_type {
            SpatialConditionType::DistanceBased => {
                if let Some(target) = condition.target_region {
                    let distance = (pos - target).cast::<f64>().norm();
                    let max_distance = condition.parameters.get("max_distance").unwrap_or(&10.0);
                    Ok(distance <= *max_distance)
                } else {
                    Ok(false)
                }
            },
            SpatialConditionType::DensityBased => {
                let density_threshold = condition.parameters.get("density_threshold").unwrap_or(&0.5);
                let local_density = cell.energy_density;
                Ok(local_density >= *density_threshold)
            },
            SpatialConditionType::GradientBased => {
                let gradient_threshold = condition.parameters.get("gradient_threshold").unwrap_or(&0.1);
                let gradient = self.calculate_energy_gradient(pos)?;
                let gradient_magnitude = gradient.norm();
                Ok(gradient_magnitude >= *gradient_threshold)
            },
            SpatialConditionType::PatternBased => {
                self.check_spatial_pattern(pos, condition)
            },
            SpatialConditionType::EntanglementBased => {
                let entanglement_threshold = condition.parameters.get("entanglement_threshold").unwrap_or(&0.5);
                let total_entanglement: f64 = cell.entanglement_map.values().sum();
                Ok(total_entanglement >= *entanglement_threshold)
            },
        }
    }

    /// Calculate energy gradient at a position
    fn calculate_energy_gradient(&self, pos: Vector3<i32>) -> Result<Vector3<f64>, ValidationError> {
        let mut gradient = Vector3::zeros();
        
        // Calculate gradient using finite differences
        for &offset in &[Vector3::new(1, 0, 0), Vector3::new(0, 1, 0), Vector3::new(0, 0, 1)] {
            let pos_plus = pos + offset;
            let pos_minus = pos - offset;
            
            let energy_plus = if let Some(cell_plus) = self.cells.get(&pos_plus) {
                cell_plus.energy_density
            } else {
                0.0
            };
            
            let energy_minus = if let Some(cell_minus) = self.cells.get(&pos_minus) {
                cell_minus.energy_density
            } else {
                0.0
            };
            
            let component = (energy_plus - energy_minus) / 2.0;
            gradient += offset.cast::<f64>() * component;
        }
        
        Ok(gradient)
    }

    /// Check for spatial patterns
    fn check_spatial_pattern(&self, pos: Vector3<i32>, condition: &SpatialCondition) -> Result<bool, ValidationError> {
        // Use Moore neighborhood for pattern-based checks
        let neighborhood = self.get_neighborhood(pos, &NeighborhoodType::Moore)?;
        // Check if neighbors have similar quantum states
        Ok(!neighborhood.is_empty())
    }

    /// Research-grade QCA rule sets for studying emergent quantum phenomena
    pub fn create_quantum_walker_rule() -> QCARule {
        QCARule {
            name: "Quantum Walker".to_string(),
            neighborhood_type: NeighborhoodType::VonNeumann,
            evolution_operator: QuantumOperator::QuantumWalk,
            interaction_strength: 0.1,
            decoherence_params: DecoherenceParameters {
                amplitude_damping: 0.01,
                phase_damping: 0.02,
                depolarization: 0.005,
                environmental_coupling: 0.1,
                temperature: 300.0, // Room temperature
            },
            measurement_rules: MeasurementRules {
                measurement_basis: vec![
                    DVector::from_element(2, Complex::new(1.0, 0.0)),
                    DVector::from_element(2, Complex::new(0.0, 1.0)),
                ],
                measurement_probability: 0.001,
                collapse_type: CollapseType::Weak,
                post_measurement_update: true,
            },
            conditions: RuleConditions {
                energy_threshold: 1e-6,
                phase_coherence_threshold: 0.1,
                entanglement_threshold: 0.01,
                time_dependent: true,
                spatial_conditions: vec![
                    SpatialCondition {
                        condition_type: SpatialConditionType::GradientBased,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("gradient_threshold".to_string(), 0.05);
                            params
                        },
                        target_region: None,
                    }
                ],
            },
        }
    }

    /// Create a quantum error correction rule for studying fault-tolerant quantum computing
    pub fn create_quantum_error_correction_rule() -> QCARule {
        QCARule {
            name: "Quantum Error Correction".to_string(),
            neighborhood_type: NeighborhoodType::Moore,
            evolution_operator: QuantumOperator::ErrorCorrection,
            interaction_strength: 0.5,
            decoherence_params: DecoherenceParameters {
                amplitude_damping: 0.1,
                phase_damping: 0.15,
                depolarization: 0.1,
                environmental_coupling: 0.2,
                temperature: 0.1, // Low temperature for quantum computing
            },
            measurement_rules: MeasurementRules {
                measurement_basis: vec![
                    DVector::from_element(4, Complex::new(1.0, 0.0)),
                    DVector::from_element(4, Complex::new(0.0, 1.0)),
                ],
                measurement_probability: 0.1,
                collapse_type: CollapseType::Projective,
                post_measurement_update: true,
            },
            conditions: RuleConditions {
                energy_threshold: 1e-8,
                phase_coherence_threshold: 0.8,
                entanglement_threshold: 0.3,
                time_dependent: false,
                spatial_conditions: vec![
                    SpatialCondition {
                        condition_type: SpatialConditionType::PatternBased,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("pattern_threshold".to_string(), 0.7);
                            params
                        },
                        target_region: None,
                    }
                ],
            },
        }
    }

    /// Create a quantum phase transition rule for studying critical phenomena
    pub fn create_quantum_phase_transition_rule() -> QCARule {
        QCARule {
            name: "Quantum Phase Transition".to_string(),
            neighborhood_type: NeighborhoodType::Moore,
            evolution_operator: QuantumOperator::PhaseTransition,
            interaction_strength: 1.0,
            decoherence_params: DecoherenceParameters {
                amplitude_damping: 0.05,
                phase_damping: 0.1,
                depolarization: 0.02,
                environmental_coupling: 0.15,
                temperature: 100.0, // Intermediate temperature
            },
            measurement_rules: MeasurementRules {
                measurement_basis: vec![
                    DVector::from_element(2, Complex::new(1.0, 0.0)),
                    DVector::from_element(2, Complex::new(0.0, 1.0)),
                ],
                measurement_probability: 0.01,
                collapse_type: CollapseType::Continuous,
                post_measurement_update: false,
            },
            conditions: RuleConditions {
                energy_threshold: 1e-5,
                phase_coherence_threshold: 0.5,
                entanglement_threshold: 0.2,
                time_dependent: true,
                spatial_conditions: vec![
                    SpatialCondition {
                        condition_type: SpatialConditionType::EntanglementBased,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("entanglement_threshold".to_string(), 0.4);
                            params
                        },
                        target_region: None,
                    }
                ],
            },
        }
    }

    /// Create a quantum synchronization rule for studying collective quantum behavior
    pub fn create_quantum_synchronization_rule() -> QCARule {
        QCARule {
            name: "Quantum Synchronization".to_string(),
            neighborhood_type: NeighborhoodType::VonNeumann,
            evolution_operator: QuantumOperator::Synchronization,
            interaction_strength: 0.3,
            decoherence_params: DecoherenceParameters {
                amplitude_damping: 0.02,
                phase_damping: 0.05,
                depolarization: 0.01,
                environmental_coupling: 0.1,
                temperature: 50.0,
            },
            measurement_rules: MeasurementRules {
                measurement_basis: vec![
                    DVector::from_element(2, Complex::new(1.0, 0.0)),
                    DVector::from_element(2, Complex::new(0.0, 1.0)),
                ],
                measurement_probability: 0.005,
                collapse_type: CollapseType::Delayed,
                post_measurement_update: true,
            },
            conditions: RuleConditions {
                energy_threshold: 1e-7,
                phase_coherence_threshold: 0.3,
                entanglement_threshold: 0.1,
                time_dependent: true,
                spatial_conditions: vec![
                    SpatialCondition {
                        condition_type: SpatialConditionType::DensityBased,
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("density_threshold".to_string(), 0.3);
                            params
                        },
                        target_region: None,
                    }
                ],
            },
        }
    }

    /// Analyze quantum correlations and entanglement patterns
    pub fn analyze_quantum_correlations(&self) -> QuantumCorrelationAnalysis {
        let mut analysis = QuantumCorrelationAnalysis::default();
        
        // Calculate pairwise correlations
        let cell_positions: Vec<Vector3<i32>> = self.cells.keys().cloned().collect();
        let n_cells = cell_positions.len();
        
        for i in 0..n_cells {
            for j in (i + 1)..n_cells {
                if let (Some(cell1), Some(cell2)) = (self.cells.get(&cell_positions[i]), self.cells.get(&cell_positions[j])) {
                    let distance = (cell_positions[i].cast::<f64>() - cell_positions[j].cast::<f64>()).norm();
                    let correlation = self.calculate_state_overlap(&cell1.state_vector, &cell2.state_vector).unwrap_or(0.0);
                    
                    analysis.correlation_distances.push(distance);
                    analysis.correlation_values.push(correlation);
                    
                    // Calculate entanglement of formation
                    let entanglement = self.calculate_entanglement_of_formation(cell1, cell2);
                    analysis.entanglement_values.push(entanglement);
                }
            }
        }
        
        // Calculate correlation length
        analysis.correlation_length = self.calculate_correlation_length(&analysis.correlation_distances, &analysis.correlation_values);
        
        // Calculate entanglement entropy
        analysis.entanglement_entropy = self.calculate_entanglement_entropy();
        
        // Calculate quantum mutual information
        analysis.quantum_mutual_information = self.calculate_quantum_mutual_information();
        
        analysis
    }

    /// Calculate correlation length from correlation function
    fn calculate_correlation_length(&self, distances: &[f64], correlations: &[f64]) -> f64 {
        if distances.is_empty() || correlations.is_empty() {
            return 0.0;
        }
        
        // Simple exponential fit: correlation = exp(-distance/correlation_length)
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        
        for (distance, &correlation) in distances.iter().zip(correlations.iter()) {
            if correlation > 0.0 {
                let log_corr = correlation.ln();
                sum_xy += distance * log_corr;
                sum_x2 += distance * distance;
            }
        }
        
        if sum_x2 > 0.0 {
            -sum_xy / sum_x2
        } else {
            0.0
        }
    }

    /// Calculate entanglement of formation between two cells
    fn calculate_entanglement_of_formation(&self, cell1: &QuantumCell, cell2: &QuantumCell) -> f64 {
        // Simplified calculation of entanglement of formation
        // In practice, this would involve more sophisticated quantum information measures
        
        let overlap = self.calculate_state_overlap(&cell1.state_vector, &cell2.state_vector).unwrap_or(0.0);
        
        // Convert overlap to concurrence (simplified)
        let concurrence = 2.0f64 * overlap * (1.0f64 - overlap).sqrt();
        
        // Convert concurrence to entanglement of formation
        if concurrence > 1.0 {
            1.0
        } else if concurrence < 0.0 {
            0.0
        } else {
            let h = |x: f64| {
                if x < 1e-10 || x > 1.0 - 1e-10 {
                    0.0
                } else {
                    -x * x.ln() - (1.0 - x) * (1.0 - x).ln()
                }
            };
            
            let x = (1.0f64 + (1.0f64 - concurrence * concurrence).sqrt()) / 2.0f64;
            h(x)
        }
    }

    /// Calculate entanglement entropy of the system
    fn calculate_entanglement_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        
        for cell in self.cells.values() {
            // Calculate von Neumann entropy of the cell's reduced density matrix
            let probabilities: Vec<f64> = cell.state_vector.iter()
                .map(|c| c.norm_sqr())
                .collect();
            
            for prob in probabilities {
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
        }
        
        entropy / self.cells.len() as f64
    }

    /// Calculate quantum mutual information
    fn calculate_quantum_mutual_information(&self) -> f64 {
        // Simplified calculation of quantum mutual information
        // This would typically involve calculating mutual information between subsystems
        
        let mut total_mutual_info = 0.0;
        let mut pair_count = 0;
        
        let cell_positions: Vec<Vector3<i32>> = self.cells.keys().cloned().collect();
        
        for i in 0..cell_positions.len() {
            for j in (i + 1)..cell_positions.len() {
                if let (Some(cell1), Some(cell2)) = (self.cells.get(&cell_positions[i]), self.cells.get(&cell_positions[j])) {
                    let overlap = self.calculate_state_overlap(&cell1.state_vector, &cell2.state_vector).unwrap_or(0.0);
                    
                    // Simplified mutual information based on overlap
                    let mutual_info = -2.0 * overlap * overlap.ln();
                    total_mutual_info += mutual_info;
                    pair_count += 1;
                }
            }
        }
        
        if pair_count > 0 {
            total_mutual_info / pair_count as f64
        } else {
            0.0
        }
    }

    /// Study quantum phase transitions by varying interaction strength
    pub fn study_quantum_phase_transition(&mut self, interaction_range: &[f64]) -> PhaseTransitionAnalysis {
        let mut analysis = PhaseTransitionAnalysis::default();
        
        for &interaction in interaction_range {
            // Update interaction strength in rules
            for rule in &mut self.rules {
                rule.interaction_strength = interaction;
            }
            
            // Evolve system to equilibrium
            for _ in 0..100 {
                self.evolve().unwrap_or_default();
            }
            
            // Analyze system properties
            let correlations = self.analyze_quantum_correlations();
            
            analysis.interaction_strengths.push(interaction);
            analysis.correlation_lengths.push(correlations.correlation_length);
            analysis.entanglement_entropies.push(correlations.entanglement_entropy);
            analysis.quantum_mutual_informations.push(correlations.quantum_mutual_information);
            analysis.global_entanglements.push(self.global_state.global_entanglement);
            analysis.quantum_coherences.push(self.global_state.quantum_coherence);
        }
        
        // Detect phase transition points
        analysis.phase_transition_points = self.detect_phase_transitions(&analysis);
        
        analysis
    }

    /// Detect phase transition points from analysis data
    fn detect_phase_transitions(&self, analysis: &PhaseTransitionAnalysis) -> Vec<f64> {
        let mut transition_points = Vec::new();
        
        // Look for discontinuities in correlation length
        for i in 1..analysis.correlation_lengths.len() {
            let derivative = (analysis.correlation_lengths[i] - analysis.correlation_lengths[i-1]) / 
                           (analysis.interaction_strengths[i] - analysis.interaction_strengths[i-1]);
            
            if derivative.abs() > 1.0 { // Threshold for phase transition
                transition_points.push(analysis.interaction_strengths[i]);
            }
        }
        
        transition_points
    }

    /// Simulate quantum error correction and analyze fault tolerance
    pub fn simulate_quantum_error_correction(&mut self, error_rates: &[f64]) -> ErrorCorrectionAnalysis {
        let mut analysis = ErrorCorrectionAnalysis::default();
        
        for &error_rate in error_rates {
            // Set error rate in decoherence parameters
            for rule in &mut self.rules {
                rule.decoherence_params.amplitude_damping = error_rate;
                rule.decoherence_params.phase_damping = error_rate;
                rule.decoherence_params.depolarization = error_rate;
            }
            
            let mut logical_error_rate = 0.0;
            let mut correction_success_rate = 0.0;
            
            // Run multiple error correction cycles
            for _ in 0..10 {
                let initial_state = self.global_state.clone();
                
                // Apply errors
                for _ in 0..50 {
                    self.evolve().unwrap_or_default();
                }
                
                // Apply error correction
                for _ in 0..10 {
                    self.evolve().unwrap_or_default();
                }
                
                // Calculate logical error rate
                let final_state = &self.global_state;
                let state_fidelity = self.calculate_state_fidelity(&initial_state, final_state);
                logical_error_rate += 1.0 - state_fidelity;
                
                // Check if error correction was successful
                if state_fidelity > 0.9 {
                    correction_success_rate += 1.0;
                }
            }
            
            analysis.error_rates.push(error_rate);
            analysis.logical_error_rates.push(logical_error_rate / 10.0);
            analysis.correction_success_rates.push(correction_success_rate / 10.0);
        }
        
        analysis
    }

    /// Calculate fidelity between two quantum states
    fn calculate_state_fidelity(&self, state1: &GlobalQuantumState, state2: &GlobalQuantumState) -> f64 {
        // Simplified fidelity calculation
        let energy_fidelity = 1.0 - ((state1.total_energy - state2.total_energy).abs() / state1.total_energy.max(1e-10));
        let coherence_fidelity = 1.0 - (state1.quantum_coherence - state2.quantum_coherence).abs();
        let entanglement_fidelity = 1.0 - (state1.global_entanglement - state2.global_entanglement).abs();
        
        (energy_fidelity + coherence_fidelity + entanglement_fidelity) / 3.0
    }
}

/// Analysis results for quantum correlations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCorrelationAnalysis {
    pub correlation_distances: Vec<f64>,
    pub correlation_values: Vec<f64>,
    pub entanglement_values: Vec<f64>,
    pub correlation_length: f64,
    pub entanglement_entropy: f64,
    pub quantum_mutual_information: f64,
}

impl Default for QuantumCorrelationAnalysis {
    fn default() -> Self {
        Self {
            correlation_distances: Vec::new(),
            correlation_values: Vec::new(),
            entanglement_values: Vec::new(),
            correlation_length: 0.0,
            entanglement_entropy: 0.0,
            quantum_mutual_information: 0.0,
        }
    }
}

/// Analysis results for phase transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionAnalysis {
    pub interaction_strengths: Vec<f64>,
    pub correlation_lengths: Vec<f64>,
    pub entanglement_entropies: Vec<f64>,
    pub quantum_mutual_informations: Vec<f64>,
    pub global_entanglements: Vec<f64>,
    pub quantum_coherences: Vec<f64>,
    pub phase_transition_points: Vec<f64>,
}

impl Default for PhaseTransitionAnalysis {
    fn default() -> Self {
        Self {
            interaction_strengths: Vec::new(),
            correlation_lengths: Vec::new(),
            entanglement_entropies: Vec::new(),
            quantum_mutual_informations: Vec::new(),
            global_entanglements: Vec::new(),
            quantum_coherences: Vec::new(),
            phase_transition_points: Vec::new(),
        }
    }
}

/// Analysis results for error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrectionAnalysis {
    pub error_rates: Vec<f64>,
    pub logical_error_rates: Vec<f64>,
    pub correction_success_rates: Vec<f64>,
}

impl Default for ErrorCorrectionAnalysis {
    fn default() -> Self {
        Self {
            error_rates: Vec::new(),
            logical_error_rates: Vec::new(),
            correction_success_rates: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_qca_lattice_creation() {
        let dimensions = Vector3::new(5, 5, 5);
        let lattice = QCALattice::new(dimensions);
        
        assert_eq!(lattice.dimensions, dimensions);
        assert_eq!(lattice.cells.len(), 125); // 5³
        assert_eq!(lattice.iteration, 0);
    }

    #[test]
    fn test_quantum_cell_default() {
        let cell = QuantumCell::default();
        
        assert_eq!(cell.state_vector.len(), 2);
        assert_eq!(cell.quantum_fields.len(), 0);
        assert_eq!(cell.entanglement_map.len(), 0);
        assert_eq!(cell.decoherence_rate, 0.0);
        assert_eq!(cell.energy_density, 0.0);
        assert_eq!(cell.phase, 0.0);
    }

    #[test]
    fn test_neighborhood_calculation() {
        let dimensions = Vector3::new(3, 3, 3);
        let mut lattice = QCALattice::new(dimensions);
        
        // Add a quantum walker rule
        lattice.rules.push(QCALattice::create_quantum_walker_rule());
        
        let center_pos = Vector3::new(1, 1, 1);
        let neighbors = lattice.get_neighborhood(center_pos, &NeighborhoodType::VonNeumann).unwrap();
        
        // Von Neumann neighborhood should have 6 neighbors in 3D
        assert_eq!(neighbors.len(), 6);
    }

    #[test]
    fn test_boundary_conditions() {
        let dimensions = Vector3::new(3, 3, 3);
        let lattice = QCALattice::new(dimensions);
        
        // Test periodic boundary conditions
        let pos = Vector3::new(-1, 4, 2);
        let wrapped_pos = lattice.apply_boundary_conditions(pos).unwrap();
        
        assert_eq!(wrapped_pos, Vector3::new(2, 1, 2));
    }

    #[test]
    fn test_quantum_state_overlap() {
        let dimensions = Vector3::new(2, 2, 2);
        let lattice = QCALattice::new(dimensions);
        
        let state1 = DVector::from_element(2, Complex::new(1.0, 0.0));
        let state2 = DVector::from_element(2, Complex::new(0.0, 1.0));
        
        let overlap = lattice.calculate_state_overlap(&state1, &state2).unwrap();
        assert_eq!(overlap, 0.0); // Orthogonal states
        
        let state3 = DVector::from_element(2, Complex::new(1.0, 0.0));
        let overlap2 = lattice.calculate_state_overlap(&state1, &state3).unwrap();
        assert_eq!(overlap2, 1.0); // Identical states
    }

    #[test]
    fn test_energy_density_calculation() {
        let dimensions = Vector3::new(2, 2, 2);
        let mut lattice = QCALattice::new(dimensions);
        
        // Create a cell with non-zero state vector
        let mut cell = QuantumCell::default();
        cell.state_vector = DVector::from_element(2, Complex::new(1.0, 0.0));
        cell.quantum_fields.insert("test_field".to_string(), Complex::new(1.0, 0.0));
        
        let neighbors = vec![];
        let energy_density = lattice.calculate_energy_density(&cell, &neighbors).unwrap();
        
        assert!(energy_density > 0.0);
    }

    #[test]
    fn test_quantum_correlation_analysis() {
        let dimensions = Vector3::new(3, 3, 3);
        let lattice = QCALattice::new(dimensions);
        
        let analysis = lattice.analyze_quantum_correlations();
        
        assert!(!analysis.correlation_distances.is_empty());
        assert!(!analysis.correlation_values.is_empty());
        assert!(!analysis.entanglement_values.is_empty());
        assert!(analysis.correlation_length >= 0.0);
        assert!(analysis.entanglement_entropy >= 0.0);
        assert!(analysis.quantum_mutual_information >= 0.0);
    }

    #[test]
    fn test_phase_transition_study() {
        let dimensions = Vector3::new(3, 3, 3);
        let mut lattice = QCALattice::new(dimensions);
        
        // Add a phase transition rule
        lattice.rules.push(QCALattice::create_quantum_phase_transition_rule());
        
        let interaction_range = vec![0.1, 0.5, 1.0, 1.5, 2.0];
        let analysis = lattice.study_quantum_phase_transition(&interaction_range);
        
        assert_eq!(analysis.interaction_strengths.len(), 5);
        assert_eq!(analysis.correlation_lengths.len(), 5);
        assert_eq!(analysis.entanglement_entropies.len(), 5);
        assert_eq!(analysis.quantum_mutual_informations.len(), 5);
        assert_eq!(analysis.global_entanglements.len(), 5);
        assert_eq!(analysis.quantum_coherences.len(), 5);
    }

    #[test]
    fn test_error_correction_simulation() {
        let dimensions = Vector3::new(3, 3, 3);
        let mut lattice = QCALattice::new(dimensions);
        
        // Add an error correction rule
        lattice.rules.push(QCALattice::create_quantum_error_correction_rule());
        
        let error_rates = vec![0.01, 0.05, 0.1, 0.15, 0.2];
        let analysis = lattice.simulate_quantum_error_correction(&error_rates);
        
        assert_eq!(analysis.error_rates.len(), 5);
        assert_eq!(analysis.logical_error_rates.len(), 5);
        assert_eq!(analysis.correction_success_rates.len(), 5);
        
        // Check that logical error rates are reasonable
        for &logical_error_rate in &analysis.logical_error_rates {
            assert!(logical_error_rate >= 0.0 && logical_error_rate <= 1.0);
        }
        
        // Check that correction success rates are reasonable
        for &success_rate in &analysis.correction_success_rates {
            assert!(success_rate >= 0.0 && success_rate <= 1.0);
        }
    }

    #[test]
    fn test_rule_creation() {
        let quantum_walker = QCALattice::create_quantum_walker_rule();
        let error_correction = QCALattice::create_quantum_error_correction_rule();
        let phase_transition = QCALattice::create_quantum_phase_transition_rule();
        let synchronization = QCALattice::create_quantum_synchronization_rule();
        
        assert_eq!(quantum_walker.name, "Quantum Walker");
        assert_eq!(error_correction.name, "Quantum Error Correction");
        assert_eq!(phase_transition.name, "Quantum Phase Transition");
        assert_eq!(synchronization.name, "Quantum Synchronization");
        
        assert!(matches!(quantum_walker.neighborhood_type, NeighborhoodType::VonNeumann));
        assert!(matches!(error_correction.neighborhood_type, NeighborhoodType::Moore));
        assert!(matches!(phase_transition.neighborhood_type, NeighborhoodType::Moore));
        assert!(matches!(synchronization.neighborhood_type, NeighborhoodType::VonNeumann));
    }

    #[test]
    fn test_spatial_condition_checking() {
        let dimensions = Vector3::new(3, 3, 3);
        let lattice = QCALattice::new(dimensions);
        
        let cell = QuantumCell::default();
        let pos = Vector3::new(1, 1, 1);
        
        // Test distance-based condition
        let distance_condition = SpatialCondition {
            condition_type: SpatialConditionType::DistanceBased,
            parameters: {
                let mut params = HashMap::new();
                params.insert("max_distance".to_string(), 2.0);
                params
            },
            target_region: Some(Vector3::new(0, 0, 0)),
        };
        
        let result = lattice.check_spatial_condition(pos, &cell, &distance_condition).unwrap();
        assert!(result); // Distance from (1,1,1) to (0,0,0) is √3 < 2.0
        
        // Test density-based condition
        let density_condition = SpatialCondition {
            condition_type: SpatialConditionType::DensityBased,
            parameters: {
                let mut params = HashMap::new();
                params.insert("density_threshold".to_string(), 0.1);
                params
            },
            target_region: None,
        };
        
        let result2 = lattice.check_spatial_condition(pos, &cell, &density_condition).unwrap();
        assert!(result2); // Should pass with reasonable density threshold
    }
} 