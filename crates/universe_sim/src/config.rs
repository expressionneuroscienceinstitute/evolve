//! Simulation Configuration
//! 
//! Manages all configurable parameters for the universe simulation

use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::path::Path;

/// Main simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Time simulation parameters
    pub tick_span_years: f64,          // Years per tick (default 1M)
    pub target_ups: f64,               // Updates per second target
    pub max_ticks: Option<u64>,        // Maximum ticks (None = infinite)
    
    /// Initial conditions
    pub initial_particle_count: usize, // Starting particles
    pub universe_radius_ly: f64,       // Initial universe radius (light-years)
    pub big_bang_temperature: f64,     // Initial temperature (K)
    
    /// Physics parameters
    pub physics_timestep: f64,         // Physics integration timestep (s)
    pub enable_relativity: bool,       // Enable relativistic corrections
    pub enable_quantum: bool,          // Enable quantum effects
    pub enable_chemistry: bool,        // Enable chemical reactions
    
    /// Agent evolution parameters
    pub agent_mutation_rate: f64,      // Probability per tick
    pub max_agents_per_planet: usize,  // Population limit
    pub agent_lifespan_ticks: u64,     // Maximum individual lifespan
    pub enable_self_modification: bool, // Allow agents to modify their code
    
    /// Performance and resource limits
    pub memory_limit_gb: f64,          // Memory usage limit
    pub cpu_limit_percent: f64,        // CPU usage limit (0-100)
    pub auto_save_interval: u64,       // Ticks between autosaves
    pub auto_save_path: String,        // Path for autosaved checkpoints
    pub checkpoint_retention: u32,     // Number of checkpoints to keep
    
    /// God-mode parameters
    pub god_mode_enabled: bool,        // Enable divine intervention
    pub max_miracles_per_epoch: u32,   // Miracle limit per cosmic era
    pub divine_entropy_cost: f64,      // Entropy cost for miracles
    
    /// Networking and distribution
    pub enable_clustering: bool,       // Enable distributed execution
    pub cluster_node_limit: u32,       // Maximum cluster nodes
    pub enable_oracle_link: bool,      // Enable agent-operator communication
    
    /// Validation and safety
    pub strict_physics_validation: bool, // Enforce conservation laws strictly
    pub enable_safety_guards: bool,    // Enable resource protection
    pub simulation_seed: Option<u64>,  // Random seed for reproducibility
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            // Time parameters
            tick_span_years: 1e6,          // 1 million years per tick
            target_ups: 1000.0,            // 1000 updates per second
            max_ticks: None,                // Infinite simulation
            
            // Initial conditions
            initial_particle_count: 1000,   // Modest particle count
            universe_radius_ly: 1000.0,     // 1000 light-year radius
            big_bang_temperature: 3000.0,   // 3000 K initial temperature
            
            // Physics parameters
            physics_timestep: 1e-6,         // 1 microsecond
            enable_relativity: true,
            enable_quantum: true,
            enable_chemistry: true,
            
            // Agent evolution
            agent_mutation_rate: 0.001,     // 0.1% per tick
            max_agents_per_planet: 10000,
            agent_lifespan_ticks: 1000,     // 1 billion years
            enable_self_modification: true,
            
            // Performance limits
            memory_limit_gb: 4.0,           // 4 GB default
            cpu_limit_percent: 70.0,        // 70% CPU usage
            auto_save_interval: 10000,      // Every 10k ticks
            auto_save_path: "./checkpoints/".to_string(), // Default save path
            checkpoint_retention: 10,       // Keep 10 checkpoints
            
            // God-mode
            god_mode_enabled: false,        // Disabled by default
            max_miracles_per_epoch: 5,
            divine_entropy_cost: 1e20,     // High entropy cost
            
            // Networking
            enable_clustering: false,       // Single-node by default
            cluster_node_limit: 16,
            enable_oracle_link: true,
            
            // Validation
            strict_physics_validation: true,
            enable_safety_guards: true,
            simulation_seed: None,          // Random seed
        }
    }
}

impl SimulationConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: SimulationConfig = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.tick_span_years <= 0.0 {
            return Err(anyhow::anyhow!("tick_span_years must be positive"));
        }
        
        if self.target_ups <= 0.0 {
            return Err(anyhow::anyhow!("target_ups must be positive"));
        }
        
        if self.initial_particle_count == 0 {
            return Err(anyhow::anyhow!("initial_particle_count must be non-zero"));
        }
        
        if self.physics_timestep <= 0.0 {
            return Err(anyhow::anyhow!("physics_timestep must be positive"));
        }
        
        if self.memory_limit_gb <= 0.0 {
            return Err(anyhow::anyhow!("memory_limit_gb must be positive"));
        }
        
        if self.cpu_limit_percent < 1.0 || self.cpu_limit_percent > 100.0 {
            return Err(anyhow::anyhow!("cpu_limit_percent must be between 1 and 100"));
        }
        
        Ok(())
    }

    /// Create a low-memory configuration
    pub fn low_memory() -> Self {
        Self {
            initial_particle_count: 100,   // Fewer particles
            memory_limit_gb: 0.5,          // 512 MB limit
            target_ups: 100.0,             // Lower UPS target
            auto_save_interval: 1000,      // More frequent saves
            auto_save_path: "./checkpoints_low_mem/".to_string(),
            checkpoint_retention: 3,       // Fewer checkpoints
            strict_physics_validation: false, // Less validation overhead
            ..Default::default()
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            initial_particle_count: 10000, // More particles
            memory_limit_gb: 16.0,         // 16 GB limit
            target_ups: 5000.0,            // Higher UPS target
            cpu_limit_percent: 90.0,       // Use more CPU
            enable_clustering: true,       // Enable distribution
            physics_timestep: 1e-7,        // Finer timestep
            ..Default::default()
        }
    }

    /// Create a benchmark configuration
    pub fn benchmark() -> Self {
        Self {
            max_ticks: Some(10000),        // Limited duration
            initial_particle_count: 1000,
            target_ups: f64::INFINITY,     // Max speed
            auto_save_interval: u64::MAX,  // No autosaves
            auto_save_path: "".to_string(), // No autosave path for benchmark
            strict_physics_validation: false,
            enable_safety_guards: false,
            simulation_seed: Some(42),     // Reproducible
            ..Default::default()
        }
    }

    /// Get estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> f64 {
        // Rough estimation based on particle count and features
        let particle_memory = self.initial_particle_count as f64 * 1024.0; // ~1KB per particle
        let physics_memory = if self.enable_quantum { 2.0 } else { 1.0 } * 1024.0 * 1024.0;
        let agent_memory = self.max_agents_per_planet as f64 * 10.0 * 1024.0; // ~10KB per agent
        
        particle_memory + physics_memory + agent_memory
    }

    /// Get simulation duration in real time (approximate)
    pub fn estimated_duration_hours(&self) -> Option<f64> {
        self.max_ticks.map(|max_ticks| max_ticks as f64 / (self.target_ups * 3600.0))
    }

    /// Check if configuration is suitable for current system
    pub fn check_system_compatibility(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        // Check memory limit
        let estimated_mb = self.estimated_memory_usage() / (1024.0 * 1024.0);
        let limit_mb = self.memory_limit_gb * 1024.0;
        
        if estimated_mb > limit_mb {
            warnings.push(format!(
                "Estimated memory usage ({:.1} MB) exceeds limit ({:.1} MB)",
                estimated_mb, limit_mb
            ));
        }
        
        // Check CPU usage
        if self.cpu_limit_percent > 80.0 && !self.enable_safety_guards {
            warnings.push("High CPU usage without safety guards may impact system responsiveness".to_string());
        }
        
        // Check particle count
        if self.initial_particle_count > 50000 {
            warnings.push("Large particle count may impact performance".to_string());
        }
        
        Ok(warnings)
    }
}

/// Configuration preset
#[derive(Debug, Clone, Copy)]
pub enum ConfigPreset {
    Default,
    LowMemory,
    HighPerformance,
    Benchmark,
}

impl ConfigPreset {
    pub fn to_config(self) -> SimulationConfig {
        match self {
            ConfigPreset::Default => SimulationConfig::default(),
            ConfigPreset::LowMemory => SimulationConfig::low_memory(),
            ConfigPreset::HighPerformance => SimulationConfig::high_performance(),
            ConfigPreset::Benchmark => SimulationConfig::benchmark(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use super::*;

    #[test]
    fn test_default_config() {
        let config = SimulationConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.tick_span_years, 1e6);
        assert_eq!(config.initial_particle_count, 1000);
    }

    #[test]
    fn test_config_validation() {
        let mut config = SimulationConfig::default();
        
        // Test invalid tick span
        config.tick_span_years = -1.0;
        assert!(config.validate().is_err());
        
        // Test invalid UPS
        config.tick_span_years = 1e6;
        config.target_ups = 0.0;
        assert!(config.validate().is_err());
        
        // Test invalid CPU limit
        config.target_ups = 1000.0;
        config.cpu_limit_percent = 150.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_io() {
        let config = SimulationConfig::default();
        
        // Use a temporary file for testing
        let temp_content = toml::to_string_pretty(&config).unwrap();
        let temp_config: SimulationConfig = toml::from_str(&temp_content).unwrap();
        
        assert_eq!(config.tick_span_years, temp_config.tick_span_years);
        assert_eq!(config.initial_particle_count, temp_config.initial_particle_count);
    }

    #[test]
    fn test_config_presets() {
        let low_mem = ConfigPreset::LowMemory.to_config();
        assert_eq!(low_mem.memory_limit_gb, 0.5);
        assert_eq!(low_mem.initial_particle_count, 100);
        
        let high_perf = ConfigPreset::HighPerformance.to_config();
        assert_eq!(high_perf.memory_limit_gb, 16.0);
        assert_eq!(high_perf.initial_particle_count, 10000);
        
        let benchmark = ConfigPreset::Benchmark.to_config();
        assert_eq!(benchmark.max_ticks, Some(10000));
        assert_eq!(benchmark.simulation_seed, Some(42));
    }

    #[test]
    fn test_memory_estimation() {
        let config = SimulationConfig::default();
        let memory = config.estimated_memory_usage();
        assert!(memory > 0.0);
        
        // Memory should scale with particle count
        let mut high_particle_config = config.clone();
        high_particle_config.initial_particle_count *= 10;
        let high_memory = high_particle_config.estimated_memory_usage();
        assert!(high_memory > memory);
    }

    #[test]
    fn test_system_compatibility() {
        let config = SimulationConfig::default();
        let warnings = config.check_system_compatibility().unwrap();
        // Should not have warnings for default config
        assert!(warnings.is_empty() || warnings.len() <= 1);
        
        // Create problematic config
        let mut problem_config = config;
        problem_config.initial_particle_count = 100000;
        problem_config.memory_limit_gb = 0.1; // Too small
        
        let warnings = problem_config.check_system_compatibility().unwrap();
        assert!(!warnings.is_empty());
    }
}