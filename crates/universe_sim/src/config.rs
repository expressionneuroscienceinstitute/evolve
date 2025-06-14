//! Configuration system for the universe simulation
//!
//! This module handles loading and validation of simulation configuration from YAML files,
//! environment variables, and command-line arguments.

use crate::constants::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Environment variable error: {0}")]
    EnvVar(String),
}

/// Main simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Core simulation parameters
    pub simulation: SimulationParams,
    /// Physics engine configuration
    pub physics: PhysicsConfig,
    /// World generation settings
    pub world: WorldConfig,
    /// Agent evolution parameters
    pub evolution: EvolutionConfig,
    /// Host resource limits
    pub resources: ResourceLimits,
    /// Network and clustering
    pub network: NetworkConfig,
    /// God-mode configuration
    pub god_mode: GodModeConfig,
    /// Oracle-Link settings
    pub oracle: OracleConfig,
    /// Persistence and checkpointing
    pub persistence: PersistenceConfig,
    /// Monitoring and metrics
    pub monitoring: MonitoringConfig,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            simulation: SimulationParams::default(),
            physics: PhysicsConfig::default(),
            world: WorldConfig::default(),
            evolution: EvolutionConfig::default(),
            resources: ResourceLimits::default(),
            network: NetworkConfig::default(),
            god_mode: GodModeConfig::default(),
            oracle: OracleConfig::default(),
            persistence: PersistenceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

/// Core simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    /// Time scale: years per simulation tick
    pub years_per_tick: f64,
    /// Target updates per second (UPS)
    pub target_ups: f64,
    /// Maximum simulation time in ticks
    pub max_ticks: u64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Enable low-memory mode
    pub low_memory_mode: bool,
    /// Validation strictness
    pub validation_level: ValidationLevel,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            years_per_tick: sim::DEFAULT_TICK_YEARS,
            target_ups: 1000.0,
            max_ticks: (sim::MAX_SIMULATION_TIME / sim::DEFAULT_TICK_YEARS) as u64,
            seed: None,
            low_memory_mode: false,
            validation_level: ValidationLevel::Standard,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    None,
    Basic,
    Standard,
    Strict,
    Paranoid,
}

/// Physics engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Enable relativistic corrections
    pub relativistic: bool,
    /// Gravitational simulation accuracy
    pub gravity_accuracy: GravityAccuracy,
    /// Thermodynamics solver settings
    pub thermodynamics: ThermodynamicsConfig,
    /// Chemistry engine settings
    pub chemistry: ChemistryConfig,
    /// Nuclear physics settings
    pub nuclear: NuclearConfig,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            relativistic: true,
            gravity_accuracy: GravityAccuracy::Standard,
            thermodynamics: ThermodynamicsConfig::default(),
            chemistry: ChemistryConfig::default(),
            nuclear: NuclearConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GravityAccuracy {
    Fast,      // Newtonian only
    Standard,  // Relativistic corrections for v > 0.1c
    Precise,   // Full GR for strong fields
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsConfig {
    /// Equation of state model
    pub equation_of_state: EosModel,
    /// Temperature solver tolerance
    pub temperature_tolerance: f64,
    /// Pressure solver tolerance
    pub pressure_tolerance: f64,
}

impl Default for ThermodynamicsConfig {
    fn default() -> Self {
        Self {
            equation_of_state: EosModel::PengRobinson,
            temperature_tolerance: 1e-6,
            pressure_tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EosModel {
    IdealGas,
    VanDerWaals,
    PengRobinson,
    RedlichKwong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemistryConfig {
    /// Enable quantum tunneling corrections
    pub quantum_tunneling: bool,
    /// Reaction rate database
    pub reaction_database: String,
    /// Kinetics solver method
    pub kinetics_solver: KineticsSolver,
    /// Enable catalysis modeling
    pub catalysis: bool,
}

impl Default for ChemistryConfig {
    fn default() -> Self {
        Self {
            quantum_tunneling: true,
            reaction_database: "GRI-Mech-3.0".to_string(),
            kinetics_solver: KineticsSolver::Cvode,
            catalysis: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KineticsSolver {
    Euler,
    RungeKutta4,
    Cvode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearConfig {
    /// Nuclear decay database
    pub decay_database: String,
    /// Enable fission modeling
    pub fission: bool,
    /// Enable fusion modeling
    pub fusion: bool,
    /// Neutron cross-section database
    pub cross_sections: String,
}

impl Default for NuclearConfig {
    fn default() -> Self {
        Self {
            decay_database: "ENSDF-2024".to_string(),
            fission: true,
            fusion: true,
            cross_sections: "ENDF-VIII.0".to_string(),
        }
    }
}

/// World generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Universe grid dimensions
    pub grid_size: (u32, u32),
    /// Initial matter density
    pub initial_density: f64,
    /// Star formation rate
    pub star_formation_rate: f64,
    /// Planet formation parameters
    pub planet_formation: PlanetFormationConfig,
    /// Geological evolution settings
    pub geology: GeologyConfig,
    /// Climate modeling
    pub climate: ClimateConfig,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            grid_size: (sim::DEFAULT_GRID_SIZE.0 as u32, sim::DEFAULT_GRID_SIZE.1 as u32),
            initial_density: 1e-29, // kg/m³ - roughly cosmic background
            star_formation_rate: 1e-3, // per million years per cell
            planet_formation: PlanetFormationConfig::default(),
            geology: GeologyConfig::default(),
            climate: ClimateConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetFormationConfig {
    /// Protoplanetary disk mass range (solar masses)
    pub disk_mass_range: (f64, f64),
    /// Planet formation efficiency
    pub formation_efficiency: f64,
    /// Water delivery probability
    pub water_delivery_prob: f64,
    /// Heavy element enrichment factor
    pub heavy_element_factor: f64,
}

impl Default for PlanetFormationConfig {
    fn default() -> Self {
        Self {
            disk_mass_range: (0.01, 0.1),
            formation_efficiency: 0.1,
            water_delivery_prob: 0.3,
            heavy_element_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeologyConfig {
    /// Enable plate tectonics
    pub plate_tectonics: bool,
    /// Volcanic activity factor
    pub volcanic_activity: f64,
    /// Erosion rate factor
    pub erosion_rate: f64,
    /// Maximum geological layers per cell
    pub max_layers: usize,
}

impl Default for GeologyConfig {
    fn default() -> Self {
        Self {
            plate_tectonics: true,
            volcanic_activity: 1.0,
            erosion_rate: 1.0,
            max_layers: sim::MAX_STRATA_LAYERS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateConfig {
    /// Enable greenhouse effect modeling
    pub greenhouse_effect: bool,
    /// Enable ice-albedo feedback
    pub ice_albedo_feedback: bool,
    /// Ocean mixing timescale (years)
    pub ocean_mixing_timescale: f64,
    /// Atmospheric mixing timescale (years)
    pub atmosphere_mixing_timescale: f64,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        Self {
            greenhouse_effect: true,
            ice_albedo_feedback: true,
            ocean_mixing_timescale: 1000.0,
            atmosphere_mixing_timescale: 1.0,
        }
    }
}

/// Agent evolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Initial agent population
    pub initial_population: u32,
    /// Mutation rate per generation
    pub mutation_rate: f64,
    /// Selection pressure strength
    pub selection_pressure: f64,
    /// Code mutation settings
    pub code_mutation: CodeMutationConfig,
    /// Fitness function parameters
    pub fitness: FitnessConfig,
    /// Speciation parameters
    pub speciation: SpeciationConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            initial_population: 100,
            mutation_rate: evolution::DEFAULT_MUTATION_RATE,
            selection_pressure: evolution::SELECTION_PRESSURE,
            code_mutation: CodeMutationConfig::default(),
            fitness: FitnessConfig::default(),
            speciation: SpeciationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMutationConfig {
    /// Maximum patch size per tick (bytes)
    pub max_patch_size: usize,
    /// Compilation timeout (milliseconds)
    pub compile_timeout_ms: u64,
    /// Maximum binary size (bytes)
    pub max_binary_size: usize,
    /// Enable unsafe code
    pub allow_unsafe: bool,
    /// Hot-swap rate limit (per N ticks)
    pub hotswap_rate_limit: u64,
}

impl Default for CodeMutationConfig {
    fn default() -> Self {
        Self {
            max_patch_size: 65536, // 64 KB
            compile_timeout_ms: 200,
            max_binary_size: 5 * 1024 * 1024, // 5 MB
            allow_unsafe: false,
            hotswap_rate_limit: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessConfig {
    /// Resource efficiency weight
    pub resource_efficiency_weight: f64,
    /// Resilience weight
    pub resilience_weight: f64,
    /// Cooperation weight
    pub cooperation_weight: f64,
    /// Entropy cost weight
    pub entropy_cost_weight: f64,
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            resource_efficiency_weight: 0.4,
            resilience_weight: 0.3,
            cooperation_weight: 0.2,
            entropy_cost_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciationConfig {
    /// Code similarity threshold for speciation
    pub similarity_threshold: f64,
    /// Minimum lineage age for speciation
    pub min_lineage_age: u64,
    /// Maximum species count
    pub max_species: u32,
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: evolution::SPECIATION_THRESHOLD,
            min_lineage_age: 1000,
            max_species: 1000,
        }
    }
}

/// Oracle-Link configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleConfig {
    /// Enable Oracle-Link communication
    pub enabled: bool,
    /// Maximum message size (bytes)
    pub max_message_size: usize,
    /// Rate limit (messages per minute per lineage)
    pub rate_limit: u32,
    /// Translation model settings
    pub translation: TranslationConfig,
    /// Response policies
    pub policies: OraclePolicyConfig,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_message_size: 4096,
            rate_limit: 10,
            translation: TranslationConfig::default(),
            policies: OraclePolicyConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Translation model path
    pub model_path: Option<PathBuf>,
    /// Confidence threshold for translation
    pub confidence_threshold: f64,
    /// Enable vocabulary learning
    pub vocabulary_learning: bool,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            confidence_threshold: 0.7,
            vocabulary_learning: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePolicyConfig {
    /// Prohibit religious influence
    pub no_religious_influence: bool,
    /// Maximum messages per day
    pub max_messages_per_day: u32,
    /// Require multi-sig for resource grants
    pub require_multisig: bool,
    /// Auto-pause on policy violation
    pub auto_pause_violations: bool,
}

impl Default for OraclePolicyConfig {
    fn default() -> Self {
        Self {
            no_religious_influence: true,
            max_messages_per_day: 100,
            require_multisig: true,
            auto_pause_violations: true,
        }
    }
}

/// Persistence and checkpointing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,
    /// Checkpoint interval (ticks)
    pub checkpoint_interval: u64,
    /// Maximum checkpoint file size (MB)
    pub max_checkpoint_size_mb: usize,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Archive settings
    pub archive: ArchiveConfig,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            checkpoint_interval: sim::DEFAULT_CHECKPOINT_INTERVAL,
            max_checkpoint_size_mb: sim::MAX_CHECKPOINT_SIZE_MB,
            compression_level: 6,
            archive: ArchiveConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveConfig {
    /// Enable automatic archiving
    pub enabled: bool,
    /// Archive interval (days)
    pub interval_days: u32,
    /// Retention period (days)
    pub retention_days: u32,
    /// Compression format
    pub compression: CompressionFormat,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_days: 7,
            retention_days: 90,
            compression: CompressionFormat::Xz,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionFormat {
    None,
    Gzip,
    Xz,
    Zstd,
}

/// Monitoring and metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub prometheus: bool,
    /// Metrics scrape interval (seconds)
    pub scrape_interval: u64,
    /// Enable Grafana dashboard
    pub grafana: bool,
    /// Log level
    pub log_level: LogLevel,
    /// Enable performance profiling
    pub profiling: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            prometheus: true,
            scrape_interval: 15,
            grafana: false,
            log_level: LogLevel::Info,
            profiling: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl SimulationConfig {
    /// Load configuration from YAML file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to YAML file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration with environment variable overrides
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = Self::default();
        
        // Override with environment variables
        if let Ok(val) = std::env::var("UNIVERSE_YEARS_PER_TICK") {
            config.simulation.years_per_tick = val.parse()
                .map_err(|e| ConfigError::EnvVar(format!("Invalid UNIVERSE_YEARS_PER_TICK: {}", e)))?;
        }
        
        if let Ok(val) = std::env::var("UNIVERSE_TARGET_UPS") {
            config.simulation.target_ups = val.parse()
                .map_err(|e| ConfigError::EnvVar(format!("Invalid UNIVERSE_TARGET_UPS: {}", e)))?;
        }
        
        if let Ok(val) = std::env::var("UNIVERSE_SEED") {
            config.simulation.seed = Some(val.parse()
                .map_err(|e| ConfigError::EnvVar(format!("Invalid UNIVERSE_SEED: {}", e)))?);
        }
        
        if let Ok(val) = std::env::var("UNIVERSE_LOW_MEMORY") {
            config.simulation.low_memory_mode = val.parse()
                .map_err(|e| ConfigError::EnvVar(format!("Invalid UNIVERSE_LOW_MEMORY: {}", e)))?;
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate simulation parameters
        if self.simulation.years_per_tick <= 0.0 {
            return Err(ConfigError::Validation("years_per_tick must be positive".to_string()));
        }
        
        if self.simulation.target_ups <= 0.0 {
            return Err(ConfigError::Validation("target_ups must be positive".to_string()));
        }
        
        if self.simulation.max_ticks == 0 {
            return Err(ConfigError::Validation("max_ticks must be positive".to_string()));
        }
        
        // Validate world parameters
        if self.world.grid_size.0 == 0 || self.world.grid_size.1 == 0 {
            return Err(ConfigError::Validation("grid_size dimensions must be positive".to_string()));
        }
        
        if self.world.initial_density < 0.0 {
            return Err(ConfigError::Validation("initial_density cannot be negative".to_string()));
        }
        
        // Validate evolution parameters
        if self.evolution.mutation_rate < 0.0 || self.evolution.mutation_rate > 1.0 {
            return Err(ConfigError::Validation("mutation_rate must be between 0 and 1".to_string()));
        }
        
        if self.evolution.selection_pressure < 0.0 {
            return Err(ConfigError::Validation("selection_pressure cannot be negative".to_string()));
        }
        
        // Validate resource limits
        if self.resources.cpu_percent <= 0.0 || self.resources.cpu_percent > 100.0 {
            return Err(ConfigError::Validation("cpu_percent must be between 0 and 100".to_string()));
        }
        
        if self.resources.memory_gb <= 0.0 {
            return Err(ConfigError::Validation("memory_gb must be positive".to_string()));
        }
        
        if self.resources.disk_gb <= 0.0 {
            return Err(ConfigError::Validation("disk_gb must be positive".to_string()));
        }
        
        // Validate god-mode parameters
        if self.god_mode.max_body_mass_fraction < 0.0 || self.god_mode.max_body_mass_fraction > 1.0 {
            return Err(ConfigError::Validation("max_body_mass_fraction must be between 0 and 1".to_string()));
        }
        
        if self.god_mode.max_time_warp_factor <= 0.0 {
            return Err(ConfigError::Validation("max_time_warp_factor must be positive".to_string()));
        }
        
        Ok(())
    }
    
    /// Get configuration summary for logging
    pub fn summary(&self) -> String {
        format!(
            "Universe Config: {}x{} grid, {:.0}yr/tick, {:.0} UPS target, {} agents",
            self.world.grid_size.0,
            self.world.grid_size.1,
            self.simulation.years_per_tick,
            self.simulation.target_ups,
            self.evolution.initial_population
        )
    }
}