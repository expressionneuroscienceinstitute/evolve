//! Fundamental constants and simulation parameters
//!
//! This module contains the immutable physics constants that govern the simulation,
//! as well as configurable parameters that control the behavior of various subsystems.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Fundamental physics constants (immutable)
pub mod physics {
    /// Speed of light in vacuum (m/s)
    pub const C: f64 = 299_792_458.0;
    
    /// Gravitational constant (m³/kg/s²)
    pub const G: f64 = 6.674_30e-11;
    
    /// Planck constant (J⋅s)
    pub const H: f64 = 6.626_070_15e-34;
    
    /// Boltzmann constant (J/K)
    pub const K_B: f64 = 1.380_649e-23;
    
    /// Elementary charge (C)
    pub const E: f64 = 1.602_176_634e-19;
    
    /// Electron mass (kg)
    pub const M_E: f64 = 9.109_383_70e-31;
    
    /// Proton mass (kg)
    pub const M_P: f64 = 1.672_621_90e-27;
    
    /// Neutron mass (kg)
    pub const M_N: f64 = 1.674_927_50e-27;
    
    /// Fine structure constant (dimensionless)
    pub const ALPHA: f64 = 7.297_352_566e-3;
    
    /// Avogadro constant (mol⁻¹)
    pub const N_A: f64 = 6.022_140_76e23;
    
    /// Solar mass (kg)
    pub const M_SOL: f64 = 1.988_47e30;
    
    /// Solar luminosity (W)
    pub const L_SOL: f64 = 3.828e26;
    
    /// Earth mass (kg)
    pub const M_EARTH: f64 = 5.972_168e24;
    
    /// Astronomical unit (m)
    pub const AU: f64 = 1.495_978_707e11;
    
    /// Parsec (m)
    pub const PC: f64 = 3.085_677_581e16;
}

/// Simulation configuration constants
pub mod sim {
    /// Default time scale: 1 tick = 1 million years
    pub const DEFAULT_TICK_YEARS: f64 = 1e6;
    
    /// Maximum simulation time (years) - heat death
    pub const MAX_SIMULATION_TIME: f64 = 1e100;
    
    /// Default universe grid size
    pub const DEFAULT_GRID_SIZE: (usize, usize) = (4096, 4096);
    
    /// Maximum geological layers per cell
    pub const MAX_STRATA_LAYERS: usize = 64;
    
    /// Default checkpoint interval (ticks)
    pub const DEFAULT_CHECKPOINT_INTERVAL: u64 = 10_000;
    
    /// Maximum checkpoint file size (MB)
    pub const MAX_CHECKPOINT_SIZE_MB: usize = 50;
    
    /// Agent CPU cycle budget per tick
    pub const AGENT_CPU_BUDGET_CYCLES: u64 = 1_000_000;
    
    /// Minimum fusion threshold (solar masses)
    pub const FUSION_THRESHOLD_M_SOL: f64 = 0.08;
    
    /// Sentience threshold for self-awareness metric
    pub const SENTIENCE_THRESHOLD: f64 = 0.7;
    
    /// Industrialization energy threshold (watts)
    pub const INDUSTRIALIZATION_THRESHOLD: f64 = 1e15;
    
    /// Digitalization threshold (fraction of cognitive processes)
    pub const DIGITALIZATION_THRESHOLD: f64 = 0.5;
    
    /// Immortality requirement (ticks without entropy death)
    pub const IMMORTALITY_TICKS: u64 = 1_000_000;
}

/// Survival thresholds for planetary habitability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalThresholds {
    pub min_liquid_water: f64,
    pub max_liquid_water: f64,
    pub min_atmos_oxygen: f64,
    pub max_atmos_oxygen: f64,
    pub min_atmos_pressure: f64,
    pub max_atmos_pressure: f64,
    pub min_temp_celsius: f64,
    pub max_temp_celsius: f64,
    pub max_radiation_sv_per_year: f64,
    pub min_energy_flux_kw_per_m2: f64,
}

impl Default for SurvivalThresholds {
    fn default() -> Self {
        Self {
            min_liquid_water: 0.2,
            max_liquid_water: 0.8,
            min_atmos_oxygen: 0.05,
            max_atmos_oxygen: 0.4,
            min_atmos_pressure: 0.3,
            max_atmos_pressure: 3.0,
            min_temp_celsius: -20.0,
            max_temp_celsius: 80.0,
            max_radiation_sv_per_year: 5.0,
            min_energy_flux_kw_per_m2: 0.1,
        }
    }
}

/// Element atomic numbers for quick reference
pub mod elements {
    pub const H: u8 = 1;   // Hydrogen
    pub const HE: u8 = 2;  // Helium
    pub const LI: u8 = 3;  // Lithium
    pub const BE: u8 = 4;  // Beryllium
    pub const B: u8 = 5;   // Boron
    pub const C: u8 = 6;   // Carbon
    pub const N: u8 = 7;   // Nitrogen
    pub const O: u8 = 8;   // Oxygen
    pub const F: u8 = 9;   // Fluorine
    pub const NE: u8 = 10; // Neon
    pub const NA: u8 = 11; // Sodium
    pub const MG: u8 = 12; // Magnesium
    pub const AL: u8 = 13; // Aluminum
    pub const SI: u8 = 14; // Silicon
    pub const P: u8 = 15;  // Phosphorus
    pub const S: u8 = 16;  // Sulfur
    pub const CL: u8 = 17; // Chlorine
    pub const AR: u8 = 18; // Argon
    pub const K: u8 = 19;  // Potassium
    pub const CA: u8 = 20; // Calcium
    pub const FE: u8 = 26; // Iron
    pub const CU: u8 = 29; // Copper
    pub const AU: u8 = 79; // Gold
    pub const U: u8 = 92;  // Uranium
}

/// Cosmic event probabilities and parameters
pub mod cosmic_events {
    /// Supernova probability per massive star per million years
    pub const SUPERNOVA_PROB_PER_MYEAR: f64 = 1e-6;
    
    /// Gamma ray burst probability per galaxy per million years
    pub const GRB_PROB_PER_MYEAR: f64 = 1e-9;
    
    /// Asteroid impact probability (km-scale) per planet per million years
    pub const ASTEROID_IMPACT_PROB_PER_MYEAR: f64 = 1e-3;
    
    /// Solar flare intensity multiplier range
    pub const SOLAR_FLARE_INTENSITY_RANGE: (f64, f64) = (0.1, 10.0);
}

/// Agent evolution parameters
pub mod evolution {
    /// Mutation rate per generation
    pub const DEFAULT_MUTATION_RATE: f64 = 0.01;
    
    /// Selection pressure strength
    pub const SELECTION_PRESSURE: f64 = 0.1;
    
    /// Speciation threshold (code similarity)
    pub const SPECIATION_THRESHOLD: f64 = 0.8;
    
    /// Maximum agent parameter count for emergent intelligence
    pub const INTELLIGENCE_PARAM_THRESHOLD: u64 = 1_000_000;
    
    /// Cooperation payoff matrix values
    pub const COOPERATION_PAYOFF: [[f64; 2]; 2] = [[3.0, 0.0], [5.0, 1.0]];
}

/// Host resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_percent: f32,
    pub memory_gb: f32,
    pub disk_gb: f32,
    pub max_file_descriptors: u32,
    pub max_processes: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_percent: 70.0,
            memory_gb: 4.0,
            disk_gb: 10.0,
            max_file_descriptors: 2048,
            max_processes: 0, // Fork disabled by default
        }
    }
}

/// Network configuration for distributed simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub quic_port: u16,
    pub rpc_port: u16,
    pub dashboard_port: u16,
    pub max_bandwidth_mbps: u32,
    pub cluster_mode: ClusterMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMode {
    Standalone,
    MultiCoreSmp,
    LanCluster,
    CloudGrid,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            quic_port: 7000,
            rpc_port: 50051,
            dashboard_port: 8080,
            max_bandwidth_mbps: 200,
            cluster_mode: ClusterMode::Standalone,
        }
    }
}

/// God-mode capabilities and restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodModeConfig {
    pub enabled: bool,
    pub max_body_mass_fraction: f64,
    pub max_miracles_per_tick_k: u32,
    pub max_time_warp_factor: f64,
    pub require_double_confirm: bool,
    pub audit_webhook_url: Option<String>,
}

impl Default for GodModeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_body_mass_fraction: 1e-3,
            max_miracles_per_tick_k: 1,
            max_time_warp_factor: 1000.0,
            require_double_confirm: true,
            audit_webhook_url: None,
        }
    }
}