//! # Universe Simulation Engine
//!
//! Core simulation engine for "Evolve: The Game of Life" - a comprehensive
//! simulation of cosmic evolution from the Big Bang to intelligent civilization.
//!
//! ## Architecture
//!
//! The simulation is built around several key components:
//! - **Physics Engine**: Implements fundamental laws (conservation, entropy, gravity, etc.)
//! - **Chemistry Engine**: Handles element interactions and compound formation
//! - **Agent System**: Autonomous AI entities that can evolve their own code
//! - **World Representation**: 2D toroidal grid with stratified geological columns
//! - **Persistence**: Zero-copy serialization for checkpointing and save/load

pub mod agent;
pub mod chemistry;
pub mod config;
pub mod constants;
pub mod entropy;
pub mod events;
pub mod fusion;
pub mod physics;
pub mod planet;
pub mod tech;
pub mod types;
pub mod universe;
pub mod utils;

// Re-export core types for convenience
pub use agent::{Agent, Lineage};
pub use config::SimulationConfig;
pub use constants::*;
pub use entropy::EntropyEngine;
pub use events::CosmicEvent;
pub use physics::PhysicsEngine;
pub use planet::{Planet, EnvironmentProfile};
pub use tech::TechTree;
pub use types::*;
pub use universe::Universe;
pub use utils::*;

use thiserror::Error;

/// Main error type for the simulation
#[derive(Error, Debug)]
pub enum SimError {
    #[error("Physics violation: {0}")]
    PhysicsViolation(String),
    
    #[error("Conservation law violated: {0}")]
    ConservationViolation(String),
    
    #[error("Agent error: {0}")]
    AgentError(String),
    
    #[error("Persistence error: {0}")]
    PersistenceError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, SimError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get git hash if available
pub fn git_hash() -> &'static str {
    option_env!("VERGEN_GIT_SHA").unwrap_or("unknown")
}

/// Initialize the simulation library
pub fn init() -> Result<()> {
    log::info!("Universe Simulation Engine v{} ({})", VERSION, git_hash());
    Ok(())
}