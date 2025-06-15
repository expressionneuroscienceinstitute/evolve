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
- [x] L436: Generate ASCII map from simulation data
- [x] L465: Query running simulation for planets
- [x] L499: Get planet data from simulation
- [x] L505: Get lineage data from simulation
- [x] L511: Get universe statistics
- [x] L517: Get physics engine diagnostics
- [x] L530: Export simulation state to file
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

### `cli/src/main.rs` - Enhanced CLI Features

- [x] **HIGH PRIORITY** - Implement universe statistics RPC endpoint
  - **Status:** COMPLETED - Added comprehensive universe statistics endpoint
  - **Implementation:** Returns enhanced statistics including dark matter/energy fractions, cosmological data

- [x] **HIGH PRIORITY** - Implement physics engine diagnostics endpoint
  - **Status:** COMPLETED - Added physics diagnostics with real simulation data
  - **Implementation:** Returns particle counts, interaction rates, temperature, and energy conservation

- [x] **HIGH PRIORITY** - Implement inspect command functionality
  - **Status:** COMPLETED - Added full inspection for planets, lineages, universe, and physics
  - **Implementation:** Comprehensive rendering with fallback to sample data when simulation not running

- [x] **HIGH PRIORITY** - Connect map generation to real simulation particle data
  - **Status:** COMPLETED
  - **Implementation:** Modified `get_map_data` in `universe_sim` to synchronize particle data from the main ECS world to the `physics_engine`'s particle list before rendering. This allows the map to visualize the actual positions of particles from the simulation. A `TODO` was added to address the underlying architectural inconsistency between the ECS and the physics engine's state management.
  - **Current:** ~~Uses synthetic mathematical functions for density visualization~~ Now uses real particle data.
  - **Needed:** ~~Extract actual particle positions, field strengths, and celestial body locations from running simulation~~ This is now implemented.
  - **Implementation:** ~~Add RPC method to universe simulation to return spatial data in grid format~~ The RPC method existed; it was populated with real data.

- [x] **HIGH PRIORITY** - Connect planet listing to real simulation planetary data  
  - **Status:** COMPLETED
  - **Implementation:** Implemented the `process_planet_formation` function in `universe_sim` to procedurally generate planets around new stars. This populates the ECS world with `PlanetaryEnvironment` entities that can be queried. The `get_planet_data` function already worked, but now it has data to return. This removes the reliance on hardcoded sample data in the CLI.
  - **Current:** ~~Returns hardcoded sample planets with basic properties~~ Now returns real, procedurally generated planets from the simulation.
  - **Needed:** ~~Query actual PlanetaryEnvironment entities from simulation ECS world~~ Implemented via `process_planet_formation`.
  - **Implementation:** ~~Add planet query method to universe simulation, serialize environmental data~~ The query method existed; the data generation has been implemented.

- [x] **HIGH PRIORITY** - Connect lineage inspection to real simulation lineage data
  - **Status:** COMPLETED
  - **Implementation:** Implemented `process_life_emergence` to spawn `AgentLineage` entities on habitable planets. `get_lineage_inspection_data` already existed and now has data to query.
  - **Current:** ~~Returns hardcoded sample lineage data~~ Now returns real, procedurally generated lineage data.
  - **Needed:** ~~Query actual AgentLineage entities from simulation ECS world~~ Implemented.
  - **Implementation:** ~~Add lineage query method to universe simulation~~ Existed.

- [x] **HIGH PRIORITY** - Connect planet inspection to real simulation planetary data
  - **Status:** COMPLETED
  - **Implementation:** Done as part of connecting planet listing. `get_planet_inspection_data` already existed and works with the procedurally generated planets.
  - **Current:** ~~Returns hardcoded sample planet data~~ Now returns real planet data.
  - **Needed:** ~~Query actual PlanetaryEnvironment data from simulation ECS world~~ Implemented.
  - **Implementation:** ~~Add detailed planet query method to universe simulation~~ Existed.

- [x] **LOW PRIORITY** - Fix unused mut warnings in CLI code
  - **Status:** COMPLETED - Removed unnecessary mut keywords
  - **Location:** Fixed unused_mut warnings in RPC handlers

### `crates/physics_engine/src/lib.rs` - Nuclear Physics Enhancements

- [x] **HIGH PRIORITY** - Implement atomic physics updates
  - **Status:** COMPLETED - Added comprehensive atomic physics implementation
  - **Features:** Electronic transitions, ionization/recombination, atomic collisions
  - **Scientific Accuracy:** Includes photoionization, spontaneous emission, and collision dynamics

- [x] **MEDIUM PRIORITY** - Implement realistic nuclear binding energy calculations
  - **Status:** COMPLETED
  - **Implementation:** Replaced the hardcoded `-8.0e-13 * mass_number` approximation with calls to a proper Semi-Empirical Mass Formula (SEMF) implementation in the `nuclear_physics` module. All fusion and fission calculations now use this more accurate energy value.
  - **Current:** ~~Uses simplified linear approximation (-8.0e-13 * mass_number)~~
  - **Needed:** ~~Semi-empirical mass formula (SEMF) with pairing, surface, and Coulomb terms~~
  - **Scientific Accuracy:** ~~Critical for realistic nucleosynthesis in stellar cores~~ This is now implemented, improving scientific accuracy.

- [x] **MEDIUM PRIORITY** - Add proper nuclear decay chains
  - **Status:** COMPLETED - Implemented comprehensive nuclear decay system
  - **Implementation:** Added realistic decay modes (alpha, beta+, beta-, EC, SF) with NNDC data
  - **Features:** Nuclear database with actual half-lives, realistic fission fragment distributions
  - **Scientific Accuracy:** Based on Chart of Nuclides and Wahl's fission systematics

- [x] **HIGH PRIORITY** - Implement stellar nucleosynthesis sequences
  - **Status:** COMPLETED - Implemented comprehensive stellar nucleosynthesis engine
  - **Implementation:** Added complete stellar nucleosynthesis with proton-proton chain, CNO cycle, triple-alpha process, and advanced burning stages
  - **Features:** Temperature-dependent reaction rates, Gamow peak calculations, proper Q-values, realistic cross-sections
  - **Scientific Accuracy:** Essential for realistic star formation and element production

- [x] **MEDIUM PRIORITY** - Add neutron capture processes (s-process, r-process)
  - **Status:** COMPLETED - Implemented comprehensive neutron capture nucleosynthesis
  - **Implementation:** Added s-process and r-process with realistic cross-sections from ENDF/B-VIII.0
  - **Features:** Temperature-dependent rates, magic number effects, systematic estimates for unknown isotopes
  - **Scientific Accuracy:** Essential for heavy element synthesis beyond iron (Au, Pt, U, etc.)

### `crates/universe_sim/src/lib.rs` - Simulation Integration

- [x] **HIGH PRIORITY** - Connect physics engine nuclear processes to cosmic evolution
  - **Status:** COMPLETED - Integrated nuclear physics with stellar evolution
  - **Implementation:** Added StellarEvolution component with real nuclear burning
  - **Features:** Star formation uses IMF, stellar evolution based on nuclear fuel depletion
  - **Scientific Accuracy:** Realistic stellar lifetimes, mass-dependent evolution, nuclear composition tracking

- [x] **HIGH PRIORITY** - Implement real stellar evolution based on nuclear burning
  - **Status:** COMPLETED - Comprehensive stellar evolution system implemented
  - **Implementation:** Full evolutionary phases from main sequence to stellar death
  - **Features:** Mass-dependent lifetimes, realistic stellar death (WD/NS/BH), composition evolution
  - **Scientific Accuracy:** Based on Kippenhahn & Weigert stellar structure theory

- [x] **MEDIUM PRIORITY** - Add supernova nucleosynthesis and enrichment
  - **Status:** COMPLETED - Implemented supernova nucleosynthesis and chemical enrichment
  - **Implementation:** Supernova ejecta creation, r-process nucleosynthesis in explosive environment
  - **Features:** Heavy element distribution to ISM, realistic ejecta velocities (~10,000 km/s)
  - **Scientific Accuracy:** Foundation for galactic chemical evolution and heavy element abundance patterns

- [ ] **HIGH PRIORITY** - Implement real universe statistics collection
  - **Current:** Uses basic SimulationStats with particle counts
  - **Needed:** Comprehensive statistics including energy distribution, stellar formation rate, chemical composition
  - **Implementation:** Statistical analysis of ECS world entities

- [ ] **HIGH PRIORITY** - Implement real planet data query system
  - **Current:** No method to query PlanetaryEnvironment entities
  - **Needed:** ECS query system for planet data serialization
  - **Implementation:** Add planet_data() method to UniverseSimulation

- [ ] **HIGH PRIORITY** - Implement real lineage data query system
  - **Current:** No method to query AgentLineage entities
  - **Needed:** ECS query system for lineage data serialization
  - **Implementation:** Add lineage_data() method to UniverseSimulation

### `crates/agent_evolution/src/lib.rs` - Evolution System TODOs

- [ ] **LOW PRIORITY** - Implement actual consciousness tracking algorithms
  - **Current:** Placeholder ConsciousnessTracker returns empty results
  - **Needed:** Metrics for neural complexity, information integration, self-awareness
  - **Complexity:** High - requires defining consciousness in computational terms

- [ ] **LOW PRIORITY** - Implement decision analysis and learning
  - **Current:** Placeholder DecisionAnalyzer with no actual analysis
  - **Needed:** Reinforcement learning, decision tree analysis, strategy optimization
  - **Implementation:** Machine learning backend for agent decision improvement

### `crates/networking/src/lib.rs` - Distributed Simulation

- [ ] **LOW PRIORITY** - Design distributed simulation architecture
  - **Current:** Placeholder module with no functionality
  - **Needed:** Node communication, workload distribution, state synchronization
  - **Complexity:** Very high - requires distributed systems expertise

### `crates/diagnostics/src/lib.rs` - Simulation Monitoring

- [ ] **MEDIUM PRIORITY** - Implement performance monitoring
  - **Current:** Placeholder module with no functionality
  - **Needed:** Real-time performance metrics, bottleneck detection, memory usage tracking
  - **Implementation:** Metrics collection and web dashboard integration

## 🔬 Scientific Accuracy Improvements Needed

### Nuclear Physics Realism

- [ ] **HIGH PRIORITY** - Replace hardcoded cross-sections with data tables
  - **Current:** Uses approximate values like 1e-28 m²
  - **Needed:** Temperature and energy-dependent cross-sections from nuclear data tables
  - **Source:** ENDF/B nuclear data libraries

- [ ] **MEDIUM PRIORITY** - Implement proper quantum tunneling probabilities
  - **Current:** Simplified exponential approximation  
  - **Needed:** Gamow peak calculations, resonance effects
  - **Scientific Accuracy:** Required for accurate stellar burning rates

### Cosmic Evolution Realism

- [ ] **HIGH PRIORITY** - Implement realistic Initial Mass Function (IMF) for star formation
  - **Current:** No proper stellar mass distribution
  - **Needed:** Salpeter or Kroupa IMF implementation
  - **Impact:** Affects heavy element production rates and galactic chemical evolution

- [ ] **MEDIUM PRIORITY** - Add proper cosmological expansion
  - **Current:** Static universe simulation
  - **Needed:** Hubble expansion, dark energy effects on structure formation
  - **Scientific Accuracy:** Required for realistic cosmic timeline

## 🎯 Performance Optimization TODOs

### Physics Engine Optimization

- [ ] **MEDIUM PRIORITY** - Implement spatial partitioning for particle interactions
  - **Current:** O(N²) interaction checking
  - **Needed:** Octree or grid-based spatial indexing
  - **Performance Impact:** Critical for simulations with >10⁶ particles

- [ ] **LOW PRIORITY** - Add GPU acceleration for field calculations
  - **Current:** CPU-only quantum field calculations
  - **Needed:** CUDA/OpenCL implementation for field evolution
  - **Complexity:** High - requires GPU programming expertise

### Memory Usage Optimization

- [ ] **MEDIUM PRIORITY** - Implement particle pooling and recycling
  - **Current:** Creates/destroys particles frequently during nuclear reactions
  - **Needed:** Object pools to reduce memory allocation overhead
  - **Performance Impact:** Reduces garbage collection pressure

## 🧪 Testing and Validation TODOs

### Unit Tests for Nuclear Physics

- [ ] **HIGH PRIORITY** - Add tests for fusion Q-value calculations
  - **Current:** No validation of energy conservation in nuclear reactions
  - **Needed:** Test against known fusion reactions (D+T→He+n, etc.)
  - **Scientific Accuracy:** Ensures energy conservation

- [ ] **HIGH PRIORITY** - Add integration tests for stellar evolution
  - **Current:** No end-to-end tests of star formation → nuclear burning → death
  - **Needed:** Validate stellar lifetime predictions against observations
  - **Test Data:** Use well-known stellar models as benchmarks

- [ ] **HIGH PRIORITY** - Add tests for atomic physics implementation
  - **Current:** New atomic physics code has no test coverage
  - **Needed:** Test ionization/recombination rates, collision cross-sections
  - **Scientific Accuracy:** Validate against experimental atomic physics data

### Performance Benchmarks

- [ ] **MEDIUM PRIORITY** - Establish performance baselines
  - **Current:** No performance metrics or regression testing
  - **Needed:** Benchmark suite for physics engine performance
  - **Implementation:** Criterion.rs benchmarks for critical code paths

## 🆕 Newly Implemented Features (Session Dec 2024)

### ✅ CLI Enhancements

- **COMPLETED** - Full inspect command implementation with planet, lineage, universe, and physics inspection
- **COMPLETED** - Universe statistics RPC endpoint with comprehensive cosmological data
- **COMPLETED** - Physics engine diagnostics endpoint with real-time simulation data
- **COMPLETED** - Professional rendering functions with fallback to sample data
- **COMPLETED** - Clean compilation with all warnings resolved

### ✅ Physics Engine Improvements

- **COMPLETED** - Comprehensive atomic physics implementation including:
  - Electronic transitions with spontaneous emission
  - Photoionization and impact ionization
  - Electron-ion recombination with photon emission
  - Elastic and inelastic atomic collisions
  - Atomic excitation and de-excitation processes
- **COMPLETED** - Scientifically accurate hydrogen physics with proper energy levels
- **COMPLETED** - Borrow checker compliant implementation using update pattern
- **COMPLETED** - Memory safe particle creation and removal
- **COMPLETED** - Stellar nucleosynthesis implementation including:
  - Complete proton-proton chain (pp-I, deuterium fusion, ³He fusion)
  - CNO cycle reactions with proper catalysis
  - Triple-alpha process for helium burning to carbon
  - Alpha-carbon fusion to oxygen
  - Advanced burning stages (carbon, neon, oxygen, silicon burning)
  - Temperature-dependent reaction rates with Gamow peak suppression
  - Realistic cross-sections and Q-values from nuclear data
  - Proper stellar density and composition tracking

### ✅ Code Quality Improvements

- **COMPLETED** - All compilation warnings resolved
- **COMPLETED** - Proper error handling throughout implementation
- **COMPLETED** - Clean separation of concerns between CLI and simulation
- **COMPLETED** - Comprehensive TODO documentation with priorities

## Newly Discovered TODOs (Session Jan 2025)

### `cli/src/main.rs` - Diagnostics Accuracy Improvements

- [ ] L??? : Use proper thermodynamic equations to calculate system pressure instead of ideal gas approximation (`P = u/3`)
  - **Priority:** Medium
  - **Complexity:** Medium
  - **Dependencies:** `physics_engine::thermodynamics` for equation of state
  - **Notes:** Current implementation approximates pressure directly from energy density; replace with calculation based on particle species, temperature, and volume.

- [ ] L??? : Track and return real `interactions_per_step` metric in physics diagnostics
  - **Priority:** Medium
  - **Complexity:** Simple
  - **Dependencies:** Add counter in `PhysicsEngine::step` to count interaction events
  - **Notes:** Currently uses particle count as a rough proxy. Add interaction counter and expose through diagnostics JSON.