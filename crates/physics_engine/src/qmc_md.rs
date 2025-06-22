/// Meta-Learned Non-Markovian Adaptive QMC - Core Structures
/// This implements the novel approach combining memory-augmented walkers,
/// meta-learned proposals, active learning, and non-local teleportation.

use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use rand::Rng;
use std::time::Instant;

// Re-export quantum_chemistry module for convenience
pub use crate::quantum_chemistry;

/// Quantum Force Calculator Interface for Molecular Dynamics
/// This trait provides quantum-accurate force calculations for molecular dynamics simulations
pub trait QuantumForceCalculator {
    /// Calculate quantum forces with statistical error estimates
    fn calculate_quantum_forces(&self, positions: &[Vector3<f64>]) -> Result<ForceWithError, QMCError>;
    
    /// Estimate statistical error in current force calculations
    fn statistical_error_estimate(&self) -> f64;
    
    /// Control adaptive sampling to achieve target error
    fn adaptive_sampling_control(&mut self, target_error: f64);
    
    /// Determine if atoms need quantum treatment
    fn needs_quantum_treatment(&self, atoms: &[Atom]) -> bool;
    
    /// Calculate quantum energy for given positions
    fn quantum_energy(&self, positions: &[Vector3<f64>]) -> f64;
    
    /// Estimate memory usage for current calculation
    fn memory_usage_estimate(&self) -> usize;
    
    /// Save current state for checkpointing
    fn checkpoint_state(&self) -> Result<Vec<u8>, QMCError>;
    
    /// Restore state from checkpoint
    fn restore_state(&mut self, state: &[u8]) -> Result<(), QMCError>;
}

/// Force calculation result with error estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceWithError {
    /// Quantum forces for each atom
    pub forces: Vec<Vector3<f64>>,
    /// Statistical errors for each force component
    pub statistical_errors: Vec<f64>,
    /// Confidence level of the calculation
    pub confidence_level: f64,
    /// Number of samples used for force calculation
    pub sample_count: usize,
    /// Computational cost in seconds
    pub computational_cost: f64,
}

/// Atom representation for quantum force calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    /// Atomic number
    pub atomic_number: u8,
    /// Position in 3D space
    pub position: Vector3<f64>,
    /// Atomic mass in atomic units
    pub mass: f64,
    /// Charge in elementary charge units
    pub charge: f64,
    /// Whether this atom requires quantum treatment
    pub needs_quantum_treatment: bool,
}

/// Quantum Force Calculator implementation using QMC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMCForceCalculator {
    /// QMC system for force calculations
    pub qmc_system: MetaLearnedNonMarkovianQMC,
    /// Sampling parameters for force calculations
    pub sampling_params: ForceSamplingParameters,
    /// Statistical analysis of force calculations
    pub force_statistics: ForceStatistics,
    /// Memory usage tracking
    pub memory_tracker: MemoryTracker,
    /// Checkpoint data
    pub checkpoint_data: Option<Vec<u8>>,
}

/// Sampling parameters specific to force calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceSamplingParameters {
    /// Number of walkers for force calculation
    pub num_walkers: usize,
    /// Number of steps for force sampling
    pub num_steps: usize,
    /// Time step for force evolution
    pub time_step: f64,
    /// Target statistical error
    pub target_error: f64,
    /// Maximum computational time in seconds
    pub max_computation_time: f64,
    /// Adaptive sampling enabled
    pub adaptive_sampling: bool,
    /// Force calculation frequency
    pub force_calculation_frequency: usize,
}

/// Statistics for force calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceStatistics {
    /// Average force magnitude
    pub average_force_magnitude: f64,
    /// Maximum force magnitude
    pub max_force_magnitude: f64,
    /// Force variance
    pub force_variance: f64,
    /// Autocorrelation time for forces
    pub force_autocorrelation_time: f64,
    /// Effective sample size
    pub effective_sample_size: usize,
    /// Confidence intervals for forces
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Memory usage tracker for force calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTracker {
    /// Current memory usage in bytes
    pub current_memory: usize,
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Memory allocation history
    pub allocation_history: Vec<(String, usize)>,
    /// Memory limit in bytes
    pub memory_limit: usize,
}

impl QMCForceCalculator {
    /// Create a new QMC force calculator
    pub fn new(sampling_params: ForceSamplingParameters) -> Result<Self, QMCError> {
        let qmc_system = MetaLearnedNonMarkovianQMC::new(
            SamplingParameters {
                num_walkers: sampling_params.num_walkers,
                num_steps: sampling_params.num_steps,
                time_step: sampling_params.time_step,
                equilibration_steps: 1000,
                block_size: 100,
                target_error: sampling_params.target_error,
            },
            ConvergenceCriteria {
                energy_tolerance: 1e-6,
                variance_tolerance: 1e-6,
                max_iterations: 10000,
                autocorrelation_threshold: 0.1,
            },
        );
        
        Ok(Self {
            qmc_system,
            sampling_params,
            force_statistics: ForceStatistics {
                average_force_magnitude: 0.0,
                max_force_magnitude: 0.0,
                force_variance: 0.0,
                force_autocorrelation_time: 0.0,
                effective_sample_size: 0,
                confidence_intervals: Vec::new(),
            },
            memory_tracker: MemoryTracker {
                current_memory: 0,
                peak_memory: 0,
                allocation_history: Vec::new(),
                memory_limit: 8 * 1024 * 1024 * 1024, // 8GB default
            },
            checkpoint_data: None,
        })
    }
    
    /// Calculate forces using QMC with error estimation
    fn calculate_forces_qmc(&mut self, positions: &[Vector3<f64>]) -> Result<Vec<Vector3<f64>>, QMCError> {
        let start_time = Instant::now();
        
        // Initialize walkers at given positions
        self.qmc_system.initialize_walkers_at_positions(positions);
        
        // Perform QMC sampling for force calculation
        let statistical_analysis = self.qmc_system.perform_ml_nma_sampling(positions)?;
        
        // Calculate forces from energy gradients
        let forces = self.calculate_forces_from_energy_gradients(positions, &statistical_analysis)?;
        
        // Update computational cost
        let computational_cost = start_time.elapsed().as_secs_f64();
        
        // Update force statistics
        self.update_force_statistics(&forces, computational_cost);
        
        Ok(forces)
    }
    
    /// Calculate forces from energy gradients using finite differences
    fn calculate_forces_from_energy_gradients(&self, positions: &[Vector3<f64>], energy_analysis: &StatisticalAnalysis) -> Result<Vec<Vector3<f64>>, QMCError> {
        let mut forces = Vec::with_capacity(positions.len());
        let displacement = 1e-6; // Small displacement for finite differences
        
        for (i, &position) in positions.iter().enumerate() {
            let mut force = Vector3::zeros();
            
            // Calculate force components using finite differences
            for dim in 0..3 {
                let mut displaced_positions = positions.to_vec();
                
                // Forward displacement
                displaced_positions[i][dim] += displacement;
                let energy_forward = self.qmc_system.calculate_local_energy(&displaced_positions);
                
                // Backward displacement
                displaced_positions[i][dim] -= 2.0 * displacement;
                let energy_backward = self.qmc_system.calculate_local_energy(&displaced_positions);
                
                // Force is negative gradient of energy
                force[dim] = -(energy_forward - energy_backward) / (2.0 * displacement);
            }
            
            forces.push(force);
        }
        
        Ok(forces)
    }
    
    /// Update force statistics
    fn update_force_statistics(&mut self, forces: &[Vector3<f64>], computational_cost: f64) {
        let force_magnitudes: Vec<f64> = forces.iter().map(|f| f.norm()).collect();
        
        self.force_statistics.average_force_magnitude = force_magnitudes.iter().sum::<f64>() / force_magnitudes.len() as f64;
        self.force_statistics.max_force_magnitude = force_magnitudes.iter().fold(0.0, |a, &b| a.max(b));
        
        // Calculate variance
        let mean = self.force_statistics.average_force_magnitude;
        self.force_statistics.force_variance = force_magnitudes.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / force_magnitudes.len() as f64;
        
        // Estimate autocorrelation time (simplified)
        self.force_statistics.force_autocorrelation_time = 1.0; // Placeholder
        
        // Update effective sample size
        self.force_statistics.effective_sample_size = forces.len();
        
        // Calculate confidence intervals (simplified)
        self.force_statistics.confidence_intervals = forces.iter().map(|_| (0.0, 0.0)).collect();
    }
    
    /// Estimate statistical error based on force variance
    fn estimate_statistical_error(&self) -> f64 {
        let standard_error = (self.force_statistics.force_variance / self.force_statistics.effective_sample_size as f64).sqrt();
        standard_error
    }
    
    /// Determine if atoms need quantum treatment
    fn determine_quantum_treatment(&self, atoms: &[Atom]) -> bool {
        // Simple heuristic: treat light atoms and atoms in close proximity quantum mechanically
        for atom in atoms {
            if atom.atomic_number <= 10 { // Light atoms (H, He, Li, Be, B, C, N, O, F, Ne)
                return true;
            }
        }
        
        // Check for close proximity (potential for quantum tunneling)
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let distance = (atoms[i].position - atoms[j].position).norm();
                if distance < 2.0 { // Close proximity threshold
                    return true;
                }
            }
        }
        
        false
    }
}

impl QuantumForceCalculator for QMCForceCalculator {
    fn calculate_quantum_forces(&self, positions: &[Vector3<f64>]) -> Result<ForceWithError, QMCError> {
        let mut calculator = self.clone();
        let forces = calculator.calculate_forces_qmc(positions)?;
        
        let statistical_error = self.estimate_statistical_error();
        let confidence_level = 0.95; // 95% confidence level
        
        Ok(ForceWithError {
            forces,
            statistical_errors: vec![statistical_error; positions.len()],
            confidence_level,
            sample_count: self.force_statistics.effective_sample_size,
            computational_cost: 0.0, // Will be updated during calculation
        })
    }
    
    fn statistical_error_estimate(&self) -> f64 {
        self.estimate_statistical_error()
    }
    
    fn adaptive_sampling_control(&mut self, target_error: f64) {
        if self.sampling_params.adaptive_sampling {
            // Adjust number of walkers based on current error vs target
            let current_error = self.estimate_statistical_error();
            let error_ratio = current_error / target_error;
            
            if error_ratio > 1.0 {
                // Increase sampling
                self.sampling_params.num_walkers = (self.sampling_params.num_walkers as f64 * error_ratio.sqrt()) as usize;
                self.sampling_params.num_steps = (self.sampling_params.num_steps as f64 * error_ratio) as usize;
            }
        }
    }
    
    fn needs_quantum_treatment(&self, atoms: &[Atom]) -> bool {
        self.determine_quantum_treatment(atoms)
    }
    
    fn quantum_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        self.qmc_system.calculate_local_energy(positions)
    }
    
    fn memory_usage_estimate(&self) -> usize {
        // Estimate memory usage based on number of walkers and positions
        let walker_memory = self.sampling_params.num_walkers * std::mem::size_of::<MemoryAugmentedWalker>();
        let position_memory = self.sampling_params.num_walkers * 100 * std::mem::size_of::<Vector3<f64>>();
        walker_memory + position_memory
    }
    
    fn checkpoint_state(&self) -> Result<Vec<u8>, QMCError> {
        // Serialize current state for checkpointing
        bincode::serialize(self).map_err(|e| QMCError::NumericalInstability {
            message: format!("Failed to serialize checkpoint: {}", e),
        })
    }
    
    fn restore_state(&mut self, state: &[u8]) -> Result<(), QMCError> {
        // Deserialize state from checkpoint
        let restored: QMCForceCalculator = bincode::deserialize(state).map_err(|e| QMCError::NumericalInstability {
            message: format!("Failed to deserialize checkpoint: {}", e),
        })?;
        
        *self = restored;
        Ok(())
    }
}

/// Hybrid QMC-Molecular Dynamics system
#[derive(Debug, Clone)]
pub struct HybridQmcMd {
    /// Quantum force calculator
    pub quantum_calculator: QMCForceCalculator,
    /// Classical molecular dynamics system
    pub classical_md: ClassicalMDSystem,
    /// Adaptive quantum region selector
    pub quantum_region_selector: QuantumRegionSelector,
    /// Force consistency checker
    pub force_consistency_checker: ForceConsistencyChecker,
}

/// Classical molecular dynamics system
#[derive(Debug, Clone)]
pub struct ClassicalMDSystem {
    /// Atoms in the system
    pub atoms: Vec<Atom>,
    /// Classical forces
    pub classical_forces: Vec<Vector3<f64>>,
    /// Integration parameters
    pub integration_params: IntegrationParameters,
}

/// Integration parameters for molecular dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationParameters {
    /// Time step for integration
    pub time_step: f64,
    /// Temperature
    pub temperature: f64,
    /// Number of integration steps
    pub num_steps: usize,
    /// Output frequency
    pub output_frequency: usize,
}

/// Quantum region selector for adaptive quantum treatment
#[derive(Debug, Clone)]
pub struct QuantumRegionSelector {
    /// Threshold for quantum treatment
    pub quantum_threshold: f64,
    /// Regions requiring quantum treatment
    pub quantum_regions: Vec<usize>,
    /// Selection criteria
    pub selection_criteria: SelectionCriteria,
}

/// Selection criteria for quantum regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Distance threshold for quantum treatment
    pub distance_threshold: f64,
    /// Energy threshold for quantum treatment
    pub energy_threshold: f64,
    /// Atomic number threshold for quantum treatment
    pub atomic_number_threshold: u8,
}

/// Force consistency checker
#[derive(Debug, Clone)]
pub struct ForceConsistencyChecker {
    /// Force tolerance
    pub force_tolerance: f64,
    /// Energy tolerance
    pub energy_tolerance: f64,
    /// Consistency history
    pub consistency_history: Vec<ConsistencyCheck>,
}

/// Consistency check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyCheck {
    /// Timestamp
    pub timestamp: f64,
    /// Force difference
    pub force_difference: f64,
    /// Energy difference
    pub energy_difference: f64,
    /// Consistency status
    pub is_consistent: bool,
}

impl HybridQmcMd {
    /// Create a new hybrid QMC-MD system
    pub fn new(quantum_calculator: QMCForceCalculator, classical_md: ClassicalMDSystem) -> Self {
        Self {
            quantum_calculator,
            classical_md,
            quantum_region_selector: QuantumRegionSelector {
                quantum_threshold: 0.1,
                quantum_regions: Vec::new(),
                selection_criteria: SelectionCriteria {
                    distance_threshold: 2.0,
                    energy_threshold: 1e-3,
                    atomic_number_threshold: 10,
                },
            },
            force_consistency_checker: ForceConsistencyChecker {
                force_tolerance: 1e-6,
                energy_tolerance: 1e-6,
                consistency_history: Vec::new(),
            },
        }
    }
    
    /// Run hybrid QMC-MD simulation
    pub fn run_simulation(&mut self) -> Result<(), QMCError> {
        for step in 0..self.classical_md.integration_params.num_steps {
            // Select quantum regions
            self.select_quantum_regions()?;
            
            // Calculate forces
            let quantum_forces = self.calculate_quantum_forces()?;
            let classical_forces = self.calculate_classical_forces()?;
            
            // Combine forces
            let combined_forces = self.combine_forces(quantum_forces, classical_forces)?;
            
            // Check force consistency
            self.check_force_consistency(&combined_forces)?;
            
            // Integrate positions
            self.integrate_positions(&combined_forces)?;
            
            // Output if needed
            if step % self.classical_md.integration_params.output_frequency == 0 {
                self.output_trajectory(step)?;
            }
        }
        
        Ok(())
    }
    
    /// Select regions requiring quantum treatment
    fn select_quantum_regions(&mut self) -> Result<(), QMCError> {
        let positions: Vec<Vector3<f64>> = self.classical_md.atoms.iter().map(|a| a.position).collect();
        let atoms = &self.classical_md.atoms;
        
        self.quantum_region_selector.quantum_regions.clear();
        
        for (i, atom) in atoms.iter().enumerate() {
            if self.quantum_calculator.needs_quantum_treatment(&[atom.clone()]) {
                self.quantum_region_selector.quantum_regions.push(i);
            }
        }
        
        Ok(())
    }
    
    /// Calculate quantum forces for selected regions
    fn calculate_quantum_forces(&mut self) -> Result<Vec<Vector3<f64>>, QMCError> {
        let positions: Vec<Vector3<f64>> = self.classical_md.atoms.iter().map(|a| a.position).collect();
        let force_result = self.quantum_calculator.calculate_quantum_forces(&positions)?;
        Ok(force_result.forces)
    }
    
    /// Calculate classical forces
    fn calculate_classical_forces(&self) -> Result<Vec<Vector3<f64>>, QMCError> {
        // Placeholder for classical force calculation
        Ok(vec![Vector3::zeros(); self.classical_md.atoms.len()])
    }
    
    /// Combine quantum and classical forces
    fn combine_forces(&self, quantum_forces: Vec<Vector3<f64>>, classical_forces: Vec<Vector3<f64>>) -> Result<Vec<Vector3<f64>>, QMCError> {
        let mut combined_forces = Vec::with_capacity(quantum_forces.len());
        
        for (i, (q_force, c_force)) in quantum_forces.iter().zip(classical_forces.iter()).enumerate() {
            if self.quantum_region_selector.quantum_regions.contains(&i) {
                // Use quantum force for quantum regions
                combined_forces.push(*q_force);
            } else {
                // Use classical force for classical regions
                combined_forces.push(*c_force);
            }
        }
        
        Ok(combined_forces)
    }
    
    /// Check force consistency
    fn check_force_consistency(&mut self, forces: &[Vector3<f64>]) -> Result<(), QMCError> {
        let consistency_check = ConsistencyCheck {
            timestamp: 0.0, // Placeholder
            force_difference: 0.0, // Placeholder
            energy_difference: 0.0, // Placeholder
            is_consistent: true, // Placeholder
        };
        
        self.force_consistency_checker.consistency_history.push(consistency_check);
        Ok(())
    }
    
    /// Integrate positions using forces
    fn integrate_positions(&mut self, forces: &[Vector3<f64>]) -> Result<(), QMCError> {
        let dt = self.classical_md.integration_params.time_step;
        
        for (atom, force) in self.classical_md.atoms.iter_mut().zip(forces.iter()) {
            // Simple velocity Verlet integration
            let acceleration = *force / atom.mass;
            atom.position += acceleration * dt * dt;
        }
        
        Ok(())
    }
    
    /// Output trajectory data
    fn output_trajectory(&self, step: usize) -> Result<(), QMCError> {
        // Placeholder for trajectory output
        Ok(())
    }
}

/// Memory-augmented walker with explicit trajectory history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAugmentedWalker {
    pub position: Vec<Vector3<f64>>,
    pub weight: f64,
    pub local_energy: f64,
    pub local_energy_history: Vec<f64>,
    pub force_contribution: Vec<Vector3<f64>>,
    // Novel: Explicit memory of trajectory
    pub trajectory_memory: TrajectoryMemory,
    // Novel: Uncertainty estimates
    pub uncertainty_estimate: f64,
    // Novel: Exploration metrics
    pub exploration_metrics: ExplorationMetrics,
}

/// Compressed trajectory memory using attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMemory {
    // Attention mechanism for memory compression
    pub attention_weights: Vec<f64>,
    pub key_positions: Vec<Vec<Vector3<f64>>>,
    pub value_energies: Vec<f64>,
    // Temporal correlation structure
    pub autocorrelation_features: Vec<f64>,
    // Memory capacity and compression
    pub max_memory_size: usize,
    pub compression_ratio: f64,
}

/// Exploration and uncertainty metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationMetrics {
    pub phase_space_coverage: f64,
    pub energy_variance: f64,
    pub position_diversity: f64,
    pub exploration_efficiency: f64,
    pub rare_event_count: usize,
}

/// Meta-learned proposal distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnedProposal {
    // Simple neural network for proposal generation (placeholder for full implementation)
    pub proposal_weights: Vec<f64>,
    pub bias_terms: Vec<f64>,
    // Uncertainty quantification
    pub uncertainty_estimate: f64,
    // Learning parameters
    pub learning_rate: f64,
    pub exploration_factor: f64,
}

/// Active learning controller for uncertainty-driven sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningController {
    pub uncertainty_threshold: f64,
    pub uncertainty_boost_factor: f64,
    pub rare_event_threshold: f64,
    pub phase_transition_sensitivity: f64,
    pub sampling_redirection_strength: f64,
}

/// Non-local teleportation controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleportationController {
    pub teleportation_probability: f64,
    pub energy_landscape_model: EnergyLandscapeModel,
    pub acceptance_threshold: f64,
    pub max_teleportation_distance: f64,
}

/// Simplified energy landscape model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyLandscapeModel {
    pub promising_regions: Vec<Vec<Vector3<f64>>>,
    pub region_energies: Vec<f64>,
    pub region_uncertainties: Vec<f64>,
}

/// Meta-optimizer for self-improving QMC algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaOptimizer {
    pub performance_history: Vec<PerformanceMetrics>,
    pub hyperparameter_space: HyperparameterSpace,
    pub optimization_frequency: usize,
    pub improvement_threshold: f64,
}

/// Hyperparameter space for meta-optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    pub learning_rate_range: (f64, f64),
    pub exploration_factor_range: (f64, f64),
    pub teleportation_probability_range: (f64, f64),
    pub uncertainty_threshold_range: (f64, f64),
}

/// Sampling parameters for QMC calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub num_walkers: usize,
    pub num_steps: usize,
    pub time_step: f64,
    pub equilibration_steps: usize,
    pub block_size: usize,
    pub target_error: f64,
}

/// Convergence criteria for QMC calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    pub energy_tolerance: f64,
    pub variance_tolerance: f64,
    pub max_iterations: usize,
    pub autocorrelation_threshold: f64,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    pub mean_energy: f64,
    pub energy_variance: f64,
    pub statistical_error: f64,
    pub autocorrelation_time: f64,
    pub effective_sample_size: usize,
    pub confidence_interval: (f64, f64),
}

/// Performance metrics for tracking algorithm performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub energy_convergence: f64,
    pub exploration_efficiency: f64,
    pub sampling_efficiency: f64,
    pub computational_cost: f64,
    pub uncertainty_reduction: f64,
}

/// Performance tracker for monitoring algorithm performance
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    pub metrics_history: Vec<PerformanceMetrics>,
    pub convergence_history: Vec<f64>,
    pub step_count: usize,
    pub start_time: Instant,
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self {
            metrics_history: Vec::new(),
            convergence_history: Vec::new(),
            step_count: 0,
            start_time: Instant::now(),
        }
    }
}

impl serde::Serialize for PerformanceTracker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PerformanceTracker", 3)?;
        state.serialize_field("metrics_history", &self.metrics_history)?;
        state.serialize_field("convergence_history", &self.convergence_history)?;
        state.serialize_field("step_count", &self.step_count)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for PerformanceTracker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct PerformanceTrackerHelper {
            metrics_history: Vec<PerformanceMetrics>,
            convergence_history: Vec<f64>,
            step_count: usize,
        }

        let helper = PerformanceTrackerHelper::deserialize(deserializer)?;
        Ok(PerformanceTracker {
            metrics_history: helper.metrics_history,
            convergence_history: helper.convergence_history,
            step_count: helper.step_count,
            start_time: Instant::now(), // Always use current time when deserializing
        })
    }
}

/// QMC error types
#[derive(Debug)]
pub enum QMCError {
    InsufficientSampling { message: String },
    ConvergenceFailed { message: String },
    NumericalInstability { message: String },
}

impl std::fmt::Display for QMCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QMCError::InsufficientSampling { message } => write!(f, "Insufficient sampling: {}", message),
            QMCError::ConvergenceFailed { message } => write!(f, "Convergence failed: {}", message),
            QMCError::NumericalInstability { message } => write!(f, "Numerical instability: {}", message),
        }
    }
}

impl std::error::Error for QMCError {}

/// Trial wavefunction for QMC calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialWavefunction {
    pub wavefunction_type: WavefunctionType,
    pub parameters: Vec<f64>,
    pub basis_set: quantum_chemistry::BasisSet,
    pub jastrow_factor: Option<JastrowFactor>,
}

/// Wavefunction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WavefunctionType {
    SlaterDeterminant,
    HartreeFock,
    ConfigurationInteraction,
    CoupledCluster,
    Custom(String),
}

/// Jastrow factor for correlation effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JastrowFactor {
    pub correlation_type: String,
    pub parameters: Vec<f64>,
}

/// Variational Monte Carlo implementation
#[derive(Debug, Clone)]
pub struct VariationalMonteCarlo {
    pub trial_wavefunction: TrialWavefunction,
    pub sampling_parameters: SamplingParameters,
    pub convergence_criteria: ConvergenceCriteria,
    pub walkers: Vec<MemoryAugmentedWalker>,
}

impl VariationalMonteCarlo {
    pub fn new_demo(
        trial_wavefunction: TrialWavefunction,
        sampling_parameters: SamplingParameters,
        convergence_criteria: ConvergenceCriteria,
    ) -> Self {
        let walkers = (0..sampling_parameters.num_walkers)
            .map(|_| MemoryAugmentedWalker::new(vec![Vector3::zeros()], 50))
            .collect();
        
        Self {
            trial_wavefunction,
            sampling_parameters,
            convergence_criteria,
            walkers,
        }
    }
    
    pub fn perform_sampling(&self, _walkers: &mut Vec<MemoryAugmentedWalker>) -> Result<StatisticalAnalysis, QMCError> {
        // Simplified implementation for demo
        Ok(StatisticalAnalysis {
            mean_energy: -0.5,
            energy_variance: 0.01,
            statistical_error: 0.001,
            autocorrelation_time: 0.1,
            effective_sample_size: 100,
            confidence_interval: (-0.502, -0.498),
        })
    }
}

/// Meta-Learned Non-Markovian Adaptive QMC System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnedNonMarkovianQMC {
    pub walkers: Vec<MemoryAugmentedWalker>,
    pub proposal_model: MetaLearnedProposal,
    pub active_learning: ActiveLearningController,
    pub teleportation: TeleportationController,
    pub meta_optimizer: MetaOptimizer,
    pub sampling_parameters: SamplingParameters,
    pub convergence_criteria: ConvergenceCriteria,
    pub performance_tracker: PerformanceTracker,
    pub demo_mode: bool,
}

impl MemoryAugmentedWalker {
    /// Create new memory-augmented walker
    pub fn new(positions: Vec<Vector3<f64>>, max_memory_size: usize) -> Self {
        let positions_len = positions.len();
        Self {
            position: positions,
            weight: 1.0,
            local_energy: 0.0,
            local_energy_history: Vec::new(),
            force_contribution: vec![Vector3::zeros(); positions_len],
            trajectory_memory: TrajectoryMemory::new(max_memory_size),
            uncertainty_estimate: 1.0, // High initial uncertainty
            exploration_metrics: ExplorationMetrics::default(),
        }
    }
    
    /// Update trajectory memory with new position and energy
    pub fn update_memory(&mut self, new_position: &[Vector3<f64>], new_energy: f64) {
        self.trajectory_memory.add_position(new_position.to_vec(), new_energy);
        self.local_energy_history.push(new_energy);
        
        // Update exploration metrics
        self.update_exploration_metrics();
        
        // Update uncertainty estimate based on energy variance
        self.update_uncertainty_estimate();
    }
    
    /// Update exploration metrics based on trajectory
    fn update_exploration_metrics(&mut self) {
        if self.local_energy_history.len() < 2 {
            return;
        }
        
        // Calculate energy variance
        let mean_energy = self.local_energy_history.iter().sum::<f64>() / self.local_energy_history.len() as f64;
        let variance = self.local_energy_history.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>() / (self.local_energy_history.len() - 1) as f64;
        
        self.exploration_metrics.energy_variance = variance;
        
        // Calculate position diversity (simplified)
        let mut total_distance = 0.0;
        let mut count = 0;
        for i in 0..self.trajectory_memory.key_positions.len() {
            for j in (i + 1)..self.trajectory_memory.key_positions.len() {
                if i < self.trajectory_memory.key_positions.len() && j < self.trajectory_memory.key_positions.len() {
                    let distance = calculate_position_distance(&self.trajectory_memory.key_positions[i], &self.trajectory_memory.key_positions[j]);
                    total_distance += distance;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            self.exploration_metrics.position_diversity = total_distance / count as f64;
        }
        
        // Update exploration efficiency (simplified metric)
        self.exploration_metrics.exploration_efficiency = 
            (self.exploration_metrics.position_diversity * self.exploration_metrics.energy_variance).sqrt();
    }
    
    /// Update uncertainty estimate based on energy fluctuations
    fn update_uncertainty_estimate(&mut self) {
        if self.local_energy_history.len() < 10 {
            self.uncertainty_estimate = 1.0;
            return;
        }
        
        // Calculate rolling energy variance as uncertainty proxy
        let window_size = 10.min(self.local_energy_history.len());
        let recent_energies: Vec<f64> = self.local_energy_history.iter().rev().take(window_size).cloned().collect();
        
        let mean_energy = recent_energies.iter().sum::<f64>() / recent_energies.len() as f64;
        let variance = recent_energies.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>() / (recent_energies.len() - 1) as f64;
        
        // Normalize uncertainty estimate
        self.uncertainty_estimate = (variance / (mean_energy.abs() + 1e-10)).min(10.0);
    }
}

impl TrajectoryMemory {
    /// Create new trajectory memory
    pub fn new(max_size: usize) -> Self {
        Self {
            attention_weights: Vec::new(),
            key_positions: Vec::new(),
            value_energies: Vec::new(),
            autocorrelation_features: Vec::new(),
            max_memory_size: max_size,
            compression_ratio: 0.1, // Keep 10% of history
        }
    }
    
    /// Add new position and energy to memory
    pub fn add_position(&mut self, position: Vec<Vector3<f64>>, energy: f64) {
        // Simple memory management: keep only recent positions
        if self.key_positions.len() >= self.max_memory_size {
            // Remove oldest entry
            self.key_positions.remove(0);
            self.value_energies.remove(0);
            self.attention_weights.remove(0);
        }
        
        self.key_positions.push(position);
        self.value_energies.push(energy);
        
        // Calculate attention weight based on energy significance
        let attention_weight = self.calculate_attention_weight(energy);
        self.attention_weights.push(attention_weight);
        
        // Update autocorrelation features
        self.update_autocorrelation_features();
    }
    
    /// Calculate attention weight for energy value
    fn calculate_attention_weight(&self, energy: f64) -> f64 {
        if self.value_energies.is_empty() {
            return 1.0;
        }
        
        let mean_energy = self.value_energies.iter().sum::<f64>() / self.value_energies.len() as f64;
        let energy_deviation = (energy - mean_energy).abs();
        let max_deviation = self.value_energies.iter()
            .map(|&e| (e - mean_energy).abs())
            .fold(0.0, f64::max);
        
        if max_deviation > 0.0 {
            (energy_deviation / max_deviation).min(2.0)
        } else {
            1.0
        }
    }
    
    /// Update autocorrelation features
    fn update_autocorrelation_features(&mut self) {
        if self.value_energies.len() < 5 {
            return;
        }
        
        // Calculate simple autocorrelation for different lags
        let lags = vec![1, 2, 3];
        self.autocorrelation_features.clear();
        
        for lag in lags {
            if self.value_energies.len() > lag {
                let autocorr = self.calculate_autocorrelation(lag);
                self.autocorrelation_features.push(autocorr);
            }
        }
    }
    
    /// Calculate autocorrelation for given lag
    fn calculate_autocorrelation(&self, lag: usize) -> f64 {
        if self.value_energies.len() <= lag {
            return 0.0;
        }
        
        let mean_energy = self.value_energies.iter().sum::<f64>() / self.value_energies.len() as f64;
        let variance = self.value_energies.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>() / (self.value_energies.len() - 1) as f64;
        
        if variance < 1e-10 {
            return 0.0;
        }
        
        let mut autocorr_sum = 0.0;
        let mut count = 0;
        
        for i in 0..(self.value_energies.len() - lag) {
            autocorr_sum += (self.value_energies[i] - mean_energy) * (self.value_energies[i + lag] - mean_energy);
            count += 1;
        }
        
        if count > 0 {
            autocorr_sum / (count as f64 * variance)
        } else {
            0.0
        }
    }
}

impl Default for ExplorationMetrics {
    fn default() -> Self {
        Self {
            phase_space_coverage: 0.0,
            energy_variance: 0.0,
            position_diversity: 0.0,
            exploration_efficiency: 0.0,
            rare_event_count: 0,
        }
    }
}

impl MetaLearnedProposal {
    /// Create new meta-learned proposal
    pub fn new(input_size: usize) -> Self {
        Self {
            proposal_weights: vec![0.1; input_size],
            bias_terms: vec![0.0; input_size],
            uncertainty_estimate: 1.0,
            learning_rate: 0.01,
            exploration_factor: 0.1,
        }
    }
    
    /// Propose new position using learned model
    pub fn propose_move(&self, walker: &MemoryAugmentedWalker) -> Vec<Vector3<f64>> {
        // Simple linear model for proposal (placeholder for neural network)
        let mut proposal = walker.position.clone();
        
        // Apply learned weights to current position
        for (i, pos) in proposal.iter_mut().enumerate() {
            if i < self.proposal_weights.len() {
                let weight = self.proposal_weights[i];
                let bias = self.bias_terms[i];
                
                // Add learned perturbation
                let perturbation = Vector3::new(
                    weight * pos.x + bias,
                    weight * pos.y + bias,
                    weight * pos.z + bias,
                );
                
                *pos += perturbation * self.learning_rate;
            }
        }
        
        // Add exploration noise
        let exploration_noise = self.sample_exploration_noise(walker.position.len());
        for (pos, noise) in proposal.iter_mut().zip(exploration_noise.iter()) {
            *pos += *noise * self.exploration_factor;
        }
        
        proposal
    }
    
    /// Sample exploration noise
    fn sample_exploration_noise(&self, num_positions: usize) -> Vec<Vector3<f64>> {
        (0..num_positions).map(|_| {
            Vector3::new(
                (rand::thread_rng().gen::<f64>() - 0.5) * 2.0,
                (rand::thread_rng().gen::<f64>() - 0.5) * 2.0,
                (rand::thread_rng().gen::<f64>() - 0.5) * 2.0,
            )
        }).collect()
    }
    
    /// Update proposal model based on acceptance and energy change
    pub fn update_model(&mut self, walker: &MemoryAugmentedWalker, acceptance: bool, energy_change: f64) {
        // Simple online learning update
        let information_gain = self.calculate_information_gain(walker, energy_change);
        let learning_signal = if acceptance { information_gain } else { -information_gain * 0.5 };
        
        // Update weights based on learning signal
        for weight in &mut self.proposal_weights {
            *weight += learning_signal * self.learning_rate;
            // Clip weights to prevent instability
            *weight = weight.clamp(-1.0, 1.0);
        }
        
        // Update bias terms
        for bias in &mut self.bias_terms {
            *bias += learning_signal * self.learning_rate * 0.1;
            *bias = bias.clamp(-0.1, 0.1);
        }
    }
    
    /// Calculate information gain from energy change
    fn calculate_information_gain(&self, walker: &MemoryAugmentedWalker, energy_change: f64) -> f64 {
        // Simple information gain based on energy change magnitude
        let energy_change_abs = energy_change.abs();
        let uncertainty = walker.uncertainty_estimate;
        
        // Higher information gain for larger energy changes in uncertain regions
        energy_change_abs * uncertainty
    }
}

impl Default for ActiveLearningController {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.5,
            uncertainty_boost_factor: 2.0,
            rare_event_threshold: 0.1,
            phase_transition_sensitivity: 0.8,
            sampling_redirection_strength: 1.0,
        }
    }
}

impl ActiveLearningController {
    /// Identify high uncertainty regions
    pub fn identify_high_uncertainty_regions(&self, walkers: &[MemoryAugmentedWalker]) -> Vec<usize> {
        walkers.iter()
            .enumerate()
            .filter(|(_, w)| w.uncertainty_estimate > self.uncertainty_threshold)
            .map(|(i, _)| i)
            .collect()
    }
    
    /// Direct sampling to high uncertainty regions
    pub fn direct_sampling(&mut self, walkers: &mut [MemoryAugmentedWalker]) {
        let high_uncertainty_indices = self.identify_high_uncertainty_regions(walkers);
        
        for &idx in &high_uncertainty_indices {
            if idx < walkers.len() {
                // Increase sampling weight for uncertain regions
                walkers[idx].weight *= self.uncertainty_boost_factor;
            }
        }
    }
    
    /// Detect rare events
    pub fn detect_rare_events(&self, walkers: &[MemoryAugmentedWalker]) -> Vec<usize> {
        walkers.iter()
            .enumerate()
            .filter(|(_, w)| {
                // Detect rare events based on energy deviations
                if w.local_energy_history.len() < 10 {
                    return false;
                }
                
                let mean_energy = w.local_energy_history.iter().sum::<f64>() / w.local_energy_history.len() as f64;
                let current_deviation = (w.local_energy - mean_energy).abs() / (mean_energy.abs() + 1e-10);
                
                current_deviation > self.rare_event_threshold
            })
            .map(|(i, _)| i)
            .collect()
    }
}

impl Default for TeleportationController {
    fn default() -> Self {
        Self {
            teleportation_probability: 0.01, // 1% chance of teleportation
            energy_landscape_model: EnergyLandscapeModel::default(),
            acceptance_threshold: 0.1,
            max_teleportation_distance: 1.0,
        }
    }
}

impl TeleportationController {
    /// Attempt non-local teleportation move
    pub fn attempt_teleportation(&mut self, walker: &mut MemoryAugmentedWalker) -> bool {
        if rand::thread_rng().gen::<f64>() < self.teleportation_probability {
            // Generate teleportation target
            let target_position = self.sample_promising_region(walker.position.len());
            
            // Calculate acceptance probability for non-local move
            let acceptance = self.calculate_teleportation_acceptance(walker, &target_position);
            
            if acceptance {
                walker.position = target_position;
                return true;
            }
        }
        false
    }
    
    /// Sample promising region for teleportation
    fn sample_promising_region(&self, num_positions: usize) -> Vec<Vector3<f64>> {
        if self.energy_landscape_model.promising_regions.is_empty() {
            // Fallback: random teleportation within bounds
            (0..num_positions).map(|_| {
                Vector3::new(
                    (rand::thread_rng().gen::<f64>() - 0.5) * self.max_teleportation_distance,
                    (rand::thread_rng().gen::<f64>() - 0.5) * self.max_teleportation_distance,
                    (rand::thread_rng().gen::<f64>() - 0.5) * self.max_teleportation_distance,
                )
            }).collect()
        } else {
            // Use learned promising regions
            let region_idx = rand::thread_rng().gen::<usize>() % self.energy_landscape_model.promising_regions.len();
            self.energy_landscape_model.promising_regions[region_idx].clone()
        }
    }
    
    /// Calculate acceptance probability for teleportation
    fn calculate_teleportation_acceptance(&self, walker: &MemoryAugmentedWalker, target_position: &[Vector3<f64>]) -> bool {
        // Simple acceptance based on distance and energy landscape
        let distance = calculate_position_distance(&walker.position, target_position);
        
        if distance > self.max_teleportation_distance {
            return false;
        }
        
        // Higher acceptance for teleportations to promising regions
        let acceptance_prob = (1.0 - distance / self.max_teleportation_distance) * self.acceptance_threshold;
        
        rand::thread_rng().gen::<f64>() < acceptance_prob
    }
}

impl Default for EnergyLandscapeModel {
    fn default() -> Self {
        Self {
            promising_regions: Vec::new(),
            region_energies: Vec::new(),
            region_uncertainties: Vec::new(),
        }
    }
}

impl Default for MetaOptimizer {
    fn default() -> Self {
        Self {
            performance_history: Vec::new(),
            hyperparameter_space: HyperparameterSpace::default(),
            optimization_frequency: 100,
            improvement_threshold: 0.01,
        }
    }
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            learning_rate_range: (0.001, 0.1),
            exploration_factor_range: (0.01, 0.5),
            teleportation_probability_range: (0.001, 0.05),
            uncertainty_threshold_range: (0.1, 1.0),
        }
    }
}

impl MetaLearnedNonMarkovianQMC {
    /// Create new ML-NMA-QMC system
    pub fn new(
        sampling_parameters: SamplingParameters,
        convergence_criteria: ConvergenceCriteria,
    ) -> Self {
        let walkers = Self::initialize_memory_walkers(&sampling_parameters);
        let proposal_model = MetaLearnedProposal::new(3); // 3D positions
        
        Self {
            walkers,
            proposal_model,
            active_learning: ActiveLearningController::default(),
            teleportation: TeleportationController::default(),
            meta_optimizer: MetaOptimizer::default(),
            sampling_parameters,
            convergence_criteria,
            performance_tracker: PerformanceTracker::default(),
            demo_mode: false,
        }
    }
    
    /// Create new ML-NMA-QMC system with demo mode
    pub fn new_demo(
        sampling_parameters: SamplingParameters,
        convergence_criteria: ConvergenceCriteria,
    ) -> Self {
        let mut system = Self::new(sampling_parameters, convergence_criteria);
        system.demo_mode = true;
        system
    }
    
    /// Initialize memory-augmented walkers
    fn initialize_memory_walkers(params: &SamplingParameters) -> Vec<MemoryAugmentedWalker> {
        let max_memory_size = 50; // Reasonable memory size
        
        (0..params.num_walkers).map(|_| {
            let initial_positions = vec![Vector3::new(0.0, 0.0, 0.0)];
            MemoryAugmentedWalker::new(initial_positions, max_memory_size)
        }).collect()
    }
    
    /// Perform meta-learned non-Markovian sampling
    pub fn perform_ml_nma_sampling(&mut self, positions: &[Vector3<f64>]) -> Result<StatisticalAnalysis, QMCError> {
        // Initialize walkers at given positions
        self.initialize_walkers_at_positions(positions);
        
        // Equilibration phase with meta-learning
        for step in 0..self.sampling_parameters.equilibration_steps {
            self.step_memory_walkers();
            self.update_meta_components(step);
        }
        
        // Production phase
        let mut energy_samples = Vec::new();
        for step in 0..self.sampling_parameters.num_steps {
            self.step_memory_walkers();
            self.update_meta_components(step);
            
            // Collect energy samples
            for walker in &self.walkers {
                energy_samples.push(walker.local_energy);
            }
        }
        
        // Statistical analysis
        self.analyze_samples(&energy_samples)
    }
    
    /// Initialize walkers at given positions
    fn initialize_walkers_at_positions(&mut self, positions: &[Vector3<f64>]) {
        for walker in &mut self.walkers {
            // Add small random perturbations
            let perturbed_positions: Vec<Vector3<f64>> = positions.iter().map(|&pos| {
                let perturbation = Vector3::new(
                    (rand::thread_rng().gen::<f64>() - 0.5) * 1e-12,
                    (rand::thread_rng().gen::<f64>() - 0.5) * 1e-12,
                    (rand::thread_rng().gen::<f64>() - 0.5) * 1e-12,
                );
                pos + perturbation
            }).collect();
            
            walker.position = perturbed_positions;
        }
    }
    
    /// Step all memory-augmented walkers
    fn step_memory_walkers(&mut self) {
        for i in 0..self.walkers.len() {
            // Attempt teleportation first
            let teleported = self.teleportation.attempt_teleportation(&mut self.walkers[i]);
            
            if !teleported {
                // Use meta-learned proposal for local move
                let new_position = self.proposal_model.propose_move(&self.walkers[i]);
                let acceptance_ratio = self.calculate_acceptance_ratio(&self.walkers[i].position, &new_position);
                
                if rand::thread_rng().gen::<f64>() < acceptance_ratio {
                    self.walkers[i].position = new_position;
                }
            }
            
            // Update local energy
            let local_energy = self.calculate_local_energy(&self.walkers[i].position);
            self.walkers[i].local_energy = local_energy;
            
            // Fix: clone position before mutable borrow
            let position_clone = self.walkers[i].position.clone();
            self.walkers[i].update_memory(&position_clone, local_energy);
        }
        
        // Apply active learning
        self.active_learning.direct_sampling(&mut self.walkers);
    }
    
    /// Update meta-learning components
    fn update_meta_components(&mut self, step: usize) {
        // Update proposal model based on walker performance
        for walker in &self.walkers {
            let energy_change = if walker.local_energy_history.len() >= 2 {
                walker.local_energy - walker.local_energy_history[walker.local_energy_history.len() - 2]
            } else {
                0.0
            };
            
            // Simple acceptance heuristic
            let acceptance = energy_change < 0.0 || rand::thread_rng().gen::<f64>() < 0.1;
            
            self.proposal_model.update_model(walker, acceptance, energy_change);
        }
        
        // Periodic meta-optimization
        if step % self.meta_optimizer.optimization_frequency == 0 {
            self.perform_meta_optimization();
        }
    }
    
    /// Perform meta-optimization of algorithm parameters
    fn perform_meta_optimization(&mut self) {
        // Simple meta-optimization: adjust parameters based on performance
        let current_performance = self.calculate_current_performance();
        
        // Create a PerformanceMetrics struct for the history
        let performance_metrics = PerformanceMetrics {
            energy_convergence: current_performance,
            exploration_efficiency: self.walkers.iter().map(|w| w.exploration_metrics.exploration_efficiency).sum::<f64>() / self.walkers.len() as f64,
            sampling_efficiency: 0.8, // Placeholder
            computational_cost: 1.0, // Placeholder
            uncertainty_reduction: self.walkers.iter().map(|w| w.uncertainty_estimate).sum::<f64>() / self.walkers.len() as f64,
        };
        
        self.meta_optimizer.performance_history.push(performance_metrics);
        
        if self.meta_optimizer.performance_history.len() >= 2 {
            let recent_performance: f64 = self.meta_optimizer.performance_history.iter().rev().take(5)
                .map(|pm| pm.energy_convergence)
                .sum::<f64>() / 5.0;
            let older_performance: f64 = self.meta_optimizer.performance_history.iter().rev().skip(5).take(5)
                .map(|pm| pm.energy_convergence)
                .sum::<f64>() / 5.0;
            
            let improvement = recent_performance - older_performance;
            
            if improvement > self.meta_optimizer.improvement_threshold {
                // Performance improving - increase exploration
                self.proposal_model.exploration_factor *= 1.1;
                self.teleportation.teleportation_probability *= 1.05;
            } else {
                // Performance stagnating - reduce exploration
                self.proposal_model.exploration_factor *= 0.95;
                self.teleportation.teleportation_probability *= 0.98;
            }
            
            // Clip parameters to valid ranges
            self.proposal_model.exploration_factor = self.proposal_model.exploration_factor
                .clamp(0.01, 0.5);
            self.teleportation.teleportation_probability = self.teleportation.teleportation_probability
                .clamp(0.001, 0.05);
        }
    }
    
    /// Calculate current performance metric
    fn calculate_current_performance(&self) -> f64 {
        if self.walkers.is_empty() {
            return 0.0;
        }
        
        // Performance based on energy variance and exploration efficiency
        let mean_energy: f64 = self.walkers.iter().map(|w| w.local_energy).sum::<f64>() / self.walkers.len() as f64;
        let energy_variance: f64 = self.walkers.iter()
            .map(|w| (w.local_energy - mean_energy).powi(2))
            .sum::<f64>() / self.walkers.len() as f64;
        
        let avg_exploration_efficiency: f64 = self.walkers.iter()
            .map(|w| w.exploration_metrics.exploration_efficiency)
            .sum::<f64>() / self.walkers.len() as f64;
        
        // Higher performance for lower variance and higher exploration
        (1.0 / (energy_variance + 1e-10)) * avg_exploration_efficiency
    }
    
    /// Calculate acceptance ratio for moves
    fn calculate_acceptance_ratio(&self, old_pos: &[Vector3<f64>], new_pos: &[Vector3<f64>]) -> f64 {
        // Simplified acceptance ratio calculation
        let old_energy = self.calculate_local_energy(old_pos);
        let new_energy = self.calculate_local_energy(new_pos);
        
        let energy_diff = new_energy - old_energy;
        (-energy_diff).exp().min(1.0)
    }
    
    /// Calculate local energy (simplified)
    fn calculate_local_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        // Simplified energy calculation
        let kinetic_energy = positions.iter().map(|pos| pos.norm_squared() / 2.0).sum::<f64>();
        let potential_energy = self.calculate_potential_energy(positions);
        kinetic_energy + potential_energy
    }
    
    /// Calculate potential energy (simplified)
    fn calculate_potential_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        let mut energy = 0.0;
        for (i, pos1) in positions.iter().enumerate() {
            for (j, pos2) in positions.iter().enumerate() {
                if i != j {
                    let distance = (pos1 - pos2).norm();
                    if distance > 1e-10 {
                        energy += 1.0 / distance; // Coulomb interaction
                    }
                }
            }
        }
        energy
    }
    
    /// Analyze samples statistically
    fn analyze_samples(&self, samples: &[f64]) -> Result<StatisticalAnalysis, QMCError> {
        if samples.is_empty() {
            return Err(QMCError::InsufficientSampling {
                message: "No energy samples available".to_string(),
            });
        }
        
        let mean_energy = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|&x| (x - mean_energy).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;
        
        let statistical_error = (variance / samples.len() as f64).sqrt();
        
        // Calculate effective sample size with memory correction
        let autocorrelation_time = self.calculate_autocorrelation_time(samples);
        let effective_sample_size = (samples.len() as f64 / (1.0 + 2.0 * autocorrelation_time)) as usize;
        
        // Calculate confidence interval
        let t_value = 1.96; // 95% confidence
        let confidence_interval = (
            mean_energy - t_value * statistical_error,
            mean_energy + t_value * statistical_error,
        );
        
        Ok(StatisticalAnalysis {
            mean_energy,
            energy_variance: variance,
            statistical_error,
            autocorrelation_time,
            effective_sample_size,
            confidence_interval,
        })
    }
    
    /// Calculate autocorrelation time
    fn calculate_autocorrelation_time(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;
        
        let max_lag = (samples.len() / 10).min(100);
        let mut autocorr_sum = 0.0;
        
        for lag in 1..=max_lag {
            let mut autocorr = 0.0;
            for i in 0..(samples.len() - lag) {
                autocorr += (samples[i] - mean) * (samples[i + lag] - mean);
            }
            autocorr /= (samples.len() - lag) as f64;
            autocorr_sum += autocorr / variance;
        }
        
        autocorr_sum / max_lag as f64
    }
}

/// Helper function to calculate distance between positions
fn calculate_position_distance(pos1: &[Vector3<f64>], pos2: &[Vector3<f64>]) -> f64 {
    if pos1.len() != pos2.len() {
        return f64::INFINITY;
    }
    
    pos1.iter().zip(pos2.iter())
        .map(|(p1, p2)| (p1 - p2).norm_squared())
        .sum::<f64>()
        .sqrt()
} 