//! Persistence & Checkpointing
//! 
//! Handles saving and loading simulation state using `bincode`.

use crate::{
    config::SimulationConfig,
    cosmic_era::{PhysicalTransition, UniverseState},
    physics_engine::PhysicsEngine,
    storage::Store,
    UniverseSimulation,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Get the default simulation checkpoint path
pub fn get_simulation_path() -> Result<PathBuf> {
    let mut path = std::env::current_dir()?;
    path.push("checkpoints");
    path.push("universe_simulation.bin");
    Ok(path)
}

/// Load a simulation from the default checkpoint path
pub fn load_simulation(path: &Path) -> Result<UniverseSimulation> {
    load_checkpoint(path)
}

/// A serializable representation of the entire simulation state.
#[derive(Serialize, Deserialize)]
struct SerializableUniverse {
    // Note: PhysicsEngine is not serialized due to complexity
    // It will be recreated from scratch on load
    current_tick: u64,
    tick_span_years: f64,
    target_ups: f64,
    universe_state: UniverseState,
    physical_transitions: Vec<PhysicalTransition>,
    config: SimulationConfig,
    store: Store,
}

/// Saves the complete state of the simulation to a checkpoint file.
pub fn save_checkpoint(sim: &mut UniverseSimulation, path: &Path) -> Result<()> {
    // Create directory structure if it doesn't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let serializable_universe = SerializableUniverse {
        current_tick: sim.current_tick,
        tick_span_years: sim.tick_span_years,
        target_ups: sim.target_ups,
        universe_state: sim.universe_state.clone(),
        physical_transitions: sim.physical_transitions.clone(),
        config: sim.config.clone(),
        store: sim.store.clone(),
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    
    // Use bincode for efficient binary serialization of the entire simulation state
    // The Store's SoA layout serializes efficiently due to Vec<T> being contiguous
    bincode::serialize_into(writer, &serializable_universe)
        .map_err(|e| anyhow::anyhow!("Failed to serialize simulation state: {}", e))?;

    // Log successful checkpoint creation
    log::info!("Simulation checkpoint saved to: {}", path.display());

    Ok(())
}

/// Loads a simulation state from a checkpoint file.
pub fn load_checkpoint(path: &Path) -> Result<UniverseSimulation> {
    // Verify checkpoint file exists and is readable
    if !path.exists() {
        return Err(anyhow::anyhow!("Checkpoint file does not exist: {}", path.display()));
    }
    
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    // Deserialize the entire simulation state from binary format
    // The SoA Store structure deserializes efficiently back to contiguous memory
    let serializable_universe: SerializableUniverse = bincode::deserialize_from(reader)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize checkpoint: {} - This may indicate a corrupted file or version mismatch", e))?;

    // Reconstruct the simulation with deserialized state
    // Note: PhysicsEngine is recreated from scratch as it's not serialized due to complexity
    let mut sim = UniverseSimulation {
        store: serializable_universe.store,
        physics_engine: PhysicsEngine::new()
            .map_err(|e| anyhow::anyhow!("Failed to recreate physics engine during load: {}", e))?,
        current_tick: serializable_universe.current_tick,
        tick_span_years: serializable_universe.tick_span_years,
        target_ups: serializable_universe.target_ups,
        universe_state: serializable_universe.universe_state,
        physical_transitions: serializable_universe.physical_transitions,
        config: serializable_universe.config,
        diagnostics: crate::DiagnosticsSystem::new(),
        stats_history: std::collections::VecDeque::new(),
        performance_stats: crate::PerformanceStats::default(),
    };

    // Validate loaded state for basic consistency
    validate_loaded_simulation(&mut sim)?;
    
    // Log successful checkpoint loading
    log::info!("Simulation checkpoint loaded from: {} (tick: {}, particles: {})", 
        path.display(), sim.current_tick, sim.store.particles.len());

    Ok(sim)
}

/// Validates the consistency of a loaded simulation state
fn validate_loaded_simulation(sim: &mut UniverseSimulation) -> Result<()> {
    // Validate particle store consistency
    let particles = &sim.store.particles;
    if particles.position.len() != particles.velocity.len() 
        || particles.position.len() != particles.mass.len()
        || particles.position.len() != particles.charge.len() {
        return Err(anyhow::anyhow!(
            "Particle store arrays have inconsistent lengths: pos={}, vel={}, mass={}, charge={}",
            particles.position.len(), particles.velocity.len(), 
            particles.mass.len(), particles.charge.len()
        ));
    }

    // Check for reasonable physics values
    for (i, &mass) in particles.mass.iter().enumerate() {
        if mass <= 0.0 || mass.is_infinite() || mass.is_nan() {
            return Err(anyhow::anyhow!("Invalid particle mass at index {}: {}", i, mass));
        }
    }

    for (i, &temp) in particles.temperature.iter().enumerate() {
        if temp < 0.0 || temp.is_infinite() || temp.is_nan() {
            return Err(anyhow::anyhow!("Invalid particle temperature at index {}: {}", i, temp));
        }
    }

    // Validate celestial bodies
    for (i, body) in sim.store.celestials.iter().enumerate() {
        if body.mass <= 0.0 || body.radius <= 0.0 {
            return Err(anyhow::anyhow!(
                "Invalid celestial body {} at index {}: mass={}, radius={}", 
                body.id, i, body.mass, body.radius
            ));
        }
    }

    // Validate simulation time parameters
    if sim.tick_span_years <= 0.0 || sim.target_ups <= 0.0 {
        return Err(anyhow::anyhow!(
            "Invalid simulation time parameters: tick_span_years={}, target_ups={}", 
            sim.tick_span_years, sim.target_ups
        ));
    }

    Ok(())
}