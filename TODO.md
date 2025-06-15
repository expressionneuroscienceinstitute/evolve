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
- [x] L427: Send stop signal to running simulation
- [ ] L436: Generate ASCII map from simulation data
- [ ] L465: Query running simulation for planets
- [ ] L499: Get planet data from simulation
- [ ] L505: Get lineage data from simulation
- [ ] L511: Get universe statistics
- [ ] L517: Get physics engine diagnostics
- [ ] L530: Export simulation state to file
- [x] L544: Send speed change command to simulation
- [x] L553: Implement rewind functionality
- [x] L567: Implement body creation
- [x] L571: Implement body deletion
- [x] L575: Implement constant modification
- [x] L579: Implement lineage spawning
- [x] L589: Implement miracles
- [x] L593: Implement time warp
- [x] L597: Implement expression evaluation
- [x] L611: Show pending resource requests
- [x] L619: Implement resource granting
- [x] L624: Show current resource usage
- [x] L631: Implement resource reload
- [x] L644: Show pending messages from agents
- [x] L652: Implement reply functionality
- [x] L657: Show communication statistics
- [x] L972: Calculate save file age for status response
- [x] L1045: Implement graceful shutdown

### `crates/universe_sim/src/world.rs`

- [x] L448: Calculate orbital velocity

### `crates/physics_engine/src/particles.rs`

- [x] L13: Populate with PDG constants and Standard Model parameters (already implemented)

### `crates/physics_engine/src/interactions.rs`

- [x] L232: Implement proper spin

## Additional TODOs Found

### `cli/src/main.rs` - Placeholder Implementations

- [ ] L479: Implement actual ASCII map generation from simulation data (currently uses dummy data)
- [ ] L506: Implement actual planet listing from simulation (currently uses hardcoded sample data)
- [ ] L537: Implement actual entity inspection functionality
- [ ] L542: Implement actual snapshot export functionality

### `crates/networking/src/lib.rs`

- [ ] L4: Implement distributed simulation networking (currently placeholder)

### `crates/diagnostics/src/lib.rs` 

- [ ] L4: Implement simulation diagnostics and monitoring (currently placeholder)

### `crates/physics_engine/src/lib.rs` - Placeholder Physics Methods

- [ ] L955: Implement nuclear fission processing
- [ ] L956: Implement nuclear shell updates
- [ ] L957: Implement fusion reaction checking
- [ ] L958: Implement fusion reaction calculations
- [ ] L959: Implement fusion reaction execution
- [ ] L961: Implement atomic physics updates
- [ ] L963: Implement molecular dynamics updates
- [ ] L965: Implement phase transition processing
- [ ] L967: Implement emergent properties updates
- [ ] L969: Implement running coupling updates
- [ ] L971: Implement symmetry breaking checks
- [ ] L973: Implement spacetime curvature updates

### `crates/universe_sim/src/lib.rs` - Simulation Core Placeholders

- [ ] L239: Implement agent evolution updates
- [ ] L250: Implement star formation processing
- [ ] L256: Implement planet formation processing  
- [ ] L262: Implement life emergence processing

### `crates/agent_evolution/src/lib.rs` - Evolution System Placeholders

- [ ] L660: Implement consciousness level tracking
- [ ] L665: Implement decision analysis
- [ ] L670: Implement lineage tracking updates

### `viz_web/src/main.rs` - Visualization Placeholders

- [ ] L415: Implement render loop functionality
- [ ] L440: Implement complete visualization rendering

### `crates/universe_sim/src/evolution.rs`

- [ ] L16: Implement evolution logic (currently just increments generation)

### `crates/physics_engine/src/quantum.rs`

- [ ] L80: Implement Pauli exclusion principle enforcement
- [ ] L133: Implement quantum state generation for electrons 