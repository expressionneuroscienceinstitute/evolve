//! Persistence & Checkpointing
//! 
//! Handles saving and loading simulation state using `bincode`.

use crate::{
    UniverseSimulation,
    CelestialBody,
    PlanetaryEnvironment,
    AgentLineage,
    config::SimulationConfig,
    cosmic_era::CosmicEra,
};
use crate::physics_engine::{PhysicsEngine, PhysicsState};
use bevy_ecs::prelude::*;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::path::Path;
use std::fs::File;
use std::io::{BufWriter, BufReader};

/// A serializable representation of the entire simulation state.
#[derive(Serialize, Deserialize)]
struct SerializableUniverse {
    // Note: PhysicsEngine is not serialized due to complexity
    // It will be recreated from scratch on load
    current_tick: u64,
    tick_span_years: f64,
    target_ups: f64,
    cosmic_era: CosmicEra,
    config: SimulationConfig,
    entities: Vec<SerializableEntity>,
}

/// A serializable representation of a single entity and its components.
#[derive(Serialize, Deserialize)]
struct SerializableEntity {
    physics_state: Option<PhysicsState>,
    celestial_body: Option<CelestialBody>,
    planetary_environment: Option<PlanetaryEnvironment>,
    agent_lineage: Option<AgentLineage>,
}

/// Saves the complete state of the simulation to a checkpoint file.
pub fn save_checkpoint(sim: &mut UniverseSimulation, path: &Path) -> Result<()> {
    let mut serializable_entities = Vec::new();
    let mut query = sim.world.query::<(
        Entity,
        Option<&PhysicsState>,
        Option<&CelestialBody>,
        Option<&PlanetaryEnvironment>,
        Option<&AgentLineage>,
    )>();

    for (
        _entity,
        physics_state,
        celestial_body,
        planetary_environment,
        agent_lineage
    ) in query.iter_mut(&mut sim.world) {
        serializable_entities.push(SerializableEntity {
            physics_state: physics_state.cloned(),
            celestial_body: celestial_body.cloned(),
            planetary_environment: planetary_environment.cloned(),
            agent_lineage: agent_lineage.cloned(),
        });
    }

    let serializable_universe = SerializableUniverse {
        current_tick: sim.current_tick,
        tick_span_years: sim.tick_span_years,
        target_ups: sim.target_ups,
        cosmic_era: sim.cosmic_era.clone(),
        config: sim.config.clone(),
        entities: serializable_entities,
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &serializable_universe)?;

    Ok(())
}

/// Loads a simulation state from a checkpoint file.
pub fn load_checkpoint(path: &Path) -> Result<UniverseSimulation> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let serializable_universe: SerializableUniverse = bincode::deserialize_from(reader)?;

    let mut sim = UniverseSimulation {
        world: World::new(),
        physics_engine: PhysicsEngine::new()?, // Recreate physics engine
        current_tick: serializable_universe.current_tick,
        tick_span_years: serializable_universe.tick_span_years,
        target_ups: serializable_universe.target_ups,
        cosmic_era: serializable_universe.cosmic_era,
        config: serializable_universe.config,
        diagnostics: crate::DiagnosticsSystem::new(),
    };

    for s_entity in serializable_universe.entities {
        let mut entity_builder = sim.world.spawn_empty();
        if let Some(c) = s_entity.physics_state {
            entity_builder.insert(c);
        }
        if let Some(c) = s_entity.celestial_body {
            entity_builder.insert(c);
        }
        if let Some(c) = s_entity.planetary_environment {
            entity_builder.insert(c);
        }
        if let Some(c) = s_entity.agent_lineage {
            entity_builder.insert(c);
        }
    }

    Ok(sim)
}