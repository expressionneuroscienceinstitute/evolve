# Project TODO List

This file tracks the stubs and TODO items found in the codebase.

## Stubs

- [x] `viz_web/Cargo.toml`: `src/bin/stub.rs` exists. Investigate and implement.
- [x] `crates/agent_evolution/src/genetics.rs`: Implement genetics.
- [x] `crates/agent_evolution/src/self_modification.rs`: Implement self-modification.
- [x] `crates/agent_evolution/src/lineage_analytics.rs`: Implement lineage analytics.
- [x] `crates/agent_evolution/src/ai_core.rs`: Implement AI core.
- [x] `crates/agent_evolution/src/decision_tracking.rs`: Implement decision tracking.
- [x] `crates/agent_evolution/src/natural_selection.rs`: Implement natural selection.
- [x] `crates/agent_evolution/src/consciousness.rs`: Implement consciousness tracking.
- [x] `crates/physics_engine/src/emergent_properties.rs`: Implement emergent properties.
- [x] `crates/physics_engine/src/molecular_dynamics.rs`: Implement molecular dynamics.
- [x] `crates/physics_engine/src/quantum_fields.rs`: Implement quantum field lattice.
- [x] `crates/physics_engine/src/nuclear_physics.rs`: Implement nuclear physics.
- [x] `crates/physics_engine/src/phase_transitions.rs`: Implement phase transitions.
- [x] `crates/physics_engine/src/atomic_physics.rs`: Implement atomic physics.
- [x] `crates/physics_engine/tests/qcd_validation.rs`: Remove demo stub tolerance.

## TODOs

### `cli/src/main.rs`

- [x] L294: Connect to running simulation via RPC
- [x] L354: Implement checkpoint loading
- [ ] L427: Send stop signal to running simulation
- [ ] L436: Generate ASCII map from simulation data
- [ ] L465: Query running simulation for planets
- [ ] L499: Get planet data from simulation
- [ ] L505: Get lineage data from simulation
- [ ] L511: Get universe statistics
- [ ] L517: Get physics engine diagnostics
- [ ] L530: Export simulation state to file
- [ ] L544: Send speed change command to simulation
- [ ] L553: Implement rewind functionality
- [ ] L567: Implement body creation
- [ ] L571: Implement body deletion
- [ ] L575: Implement constant modification
- [ ] L579: Implement lineage spawning
- [ ] L589: Implement miracles
- [ ] L593: Implement time warp
- [ ] L597: Implement expression evaluation
- [ ] L611: Show pending resource requests
- [ ] L619: Implement resource granting
- [ ] L624: Show current resource usage
- [ ] L631: Implement resource reload
- [ ] L644: Show pending messages from agents
- [ ] L652: Implement reply functionality
- [ ] L657: Show communication statistics

### `crates/universe_sim/src/world.rs`

- [ ] L448: Calculate orbital velocity

### `crates/physics_engine/src/particles.rs`

- [ ] L13: Populate with PDG constants and Standard Model parameters

### `crates/physics_engine/src/interactions.rs`

- [ ] L232: Implement proper spin 