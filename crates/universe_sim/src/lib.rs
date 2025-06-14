//! # Universe Simulation Engine
//!
//! A comprehensive physics-based simulation engine for modeling the evolution
//! of matter, stars, planets, life, and intelligent civilization from the Big Bang
//! to the far future.
//!
//! ## Core Features
//!
//! - Peer-reviewed physics fidelity with conservation laws
//! - Autonomous AI agent evolution without human intervention
//! - Resource-constrained planetary survival mechanics
//! - Emergent technology trees and civilization development
//! - Headless operation with optional visualization
//! - Distributed cluster execution support

pub mod config;
pub mod world;
pub mod physics;
pub mod entropy;
pub mod agents;
pub mod resources;
pub mod tick;
pub mod save;
pub mod metrics;
pub mod universe;

// Re-export core types
pub use config::{Config, SimulationConfig, PhysicsConfig};
pub use world::{World, WorldCell, CosmicEra, Planet, PlanetClass};
pub use physics::{PhysicsEngine, ConservationLaws};
pub use agents::{Agent, AgentTrait, LineageId, Fitness};
pub use resources::{ElementTable, EnvironmentProfile, ResourceExtraction};
pub use tick::{TickScheduler, TickResult};
pub use save::{SaveState, CheckpointManager};
pub use universe::{Universe, UniverseState};

/// The main simulation result type
pub type SimResult<T> = anyhow::Result<T>;

/// Default tick span in years (1 million years per tick)
pub const DEFAULT_TICK_SPAN: u64 = 1_000_000;

/// Maximum world grid size 
pub const MAX_WORLD_SIZE: usize = 4096;

/// Maximum geological layers per cell
pub const MAX_STRATA_LAYERS: usize = 64;

/// Universal constants (peer-reviewed values from CODATA-2023)
pub mod constants {
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
    
    /// Gravitational constant (m³/kg·s²)
    pub const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11;
    
    /// Planck constant (J·s)
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;
    
    /// Solar mass (kg)
    pub const SOLAR_MASS: f64 = 1.98847e30;
    
    /// Fusion threshold (fraction of solar mass)
    pub const FUSION_THRESHOLD: f64 = 0.08;
    
    /// Boltzmann constant (J/K)
    pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_sanity() {
        assert!(constants::SPEED_OF_LIGHT > 0.0);
        assert!(constants::GRAVITATIONAL_CONSTANT > 0.0);
        assert!(constants::FUSION_THRESHOLD < 1.0);
    }
}