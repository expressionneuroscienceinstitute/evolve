//! Persistence & Checkpointing
//! 
//! Handles saving and loading simulation state using `bincode`.

use crate::{
    config::SimulationConfig,
    cosmic_era::{PhysicalTransition, UniverseState},
    physics_engine::{PhysicsEngine, PhysicsState},
    storage::{AgentLineage, CelestialBody, PlanetaryEnvironment, Store},
    UniverseSimulation,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use bevy_ecs::prelude::World;

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
    // TODO: Re-implement serialization for the new SoA-based Store.
    // This will likely involve serializing each Vec in the Store.
    
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
    bincode::serialize_into(writer, &serializable_universe)?;

    Ok(())
}

/// Loads a simulation state from a checkpoint file.
pub fn load_checkpoint(path: &Path) -> Result<UniverseSimulation> {
    // TODO: Re-implement deserialization for the new SoA-based Store.
    
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let serializable_universe: SerializableUniverse = bincode::deserialize_from(reader)?;

    let sim = UniverseSimulation {
        store: serializable_universe.store,
        world: World::default(),
        physics_engine: PhysicsEngine::new()?, // Recreate physics engine
        current_tick: serializable_universe.current_tick,
        tick_span_years: serializable_universe.tick_span_years,
        target_ups: serializable_universe.target_ups,
        universe_state: serializable_universe.universe_state,
        physical_transitions: serializable_universe.physical_transitions,
        config: serializable_universe.config,
        diagnostics: crate::DiagnosticsSystem::new(),
    };

    Ok(sim)
}