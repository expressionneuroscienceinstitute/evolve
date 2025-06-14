# Evolve: The Game of Life

A comprehensive universe simulation modeling cosmic evolution from Big Bang to intelligent civilization.

## Project Status: ✅ BUILDING SUCCESSFULLY

All compilation errors have been resolved! The project now builds cleanly across all components.

## Quick Start

```bash
# Build the project
cargo build --release

# Generate example configuration
./target/release/universectl gen-config config.yaml

# Run simulation
./target/release/universectl run --config config.yaml

# View available commands
./target/release/universectl --help
```

## Architecture

### Core Components
- **Physics Engine**: Implements 5 fundamental laws (mass-energy conservation, entropy, gravity, fusion, nucleosynthesis)
- **Agent System**: Autonomous AI entities with evolutionary code modification
- **Planet System**: Geological stratification and environmental modeling
- **Technology Tree**: Stone Age → Warp Drive progression system
- **CLI Interface**: Command-line tools for simulation control

### Win Condition
Agent lineages must achieve: Sentience → Industrialization → Digitalization → Trans-Tech Frontier → Immortality

## Build Fixes Applied

The following major compilation issues were resolved:
- ✅ Environment variable handling
- ✅ Missing dependencies and imports  
- ✅ Private type exports
- ✅ Lifetime annotations
- ✅ Type ambiguity issues
- ✅ Serialization compatibility
- ✅ Borrowing conflicts
- ✅ Trait implementations

See `BUILD_FIXES.md` for detailed information.

## Components

- `crates/universe_sim/` - Core simulation library
- `cli/` - Command-line interface (`universectl`)
- `viz_web/` - Web visualization (placeholder)
- `trainer/` - Distributed training (placeholder)
- `universewd/` - Watchdog daemon (placeholder)

## Features

- 🌌 **Physics Simulation**: Relativistic gravity, stellar evolution, nucleosynthesis
- 🤖 **AI Agents**: Self-modifying code with survival mechanics
- 🌍 **Planet Generation**: Procedural worlds with geological layers
- ⚗️ **Chemistry Engine**: Element interactions and compound formation
- 🔬 **Technology Research**: Progressive tech tree advancement
- 📊 **Benchmarking**: Performance testing for physics and AI systems
- ⚙️ **Configuration**: YAML-based settings with CLI overrides
- 🎯 **God Mode**: Divine intervention capabilities
- 📡 **Oracle Link**: Agent-operator communication channel

## Development

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Build all targets
cargo build --all-targets

# Generate documentation
cargo doc --open
```

## Configuration

The simulation supports extensive configuration through YAML files:

```yaml
simulation:
  years_per_tick: 1000000.0
  target_ups: 1000.0
  max_ticks: 18446744073709551615

physics:
  relativistic: true
  gravity_accuracy: Standard

world:
  grid_size: [4096, 4096]
  star_formation_rate: 0.001

evolution:
  initial_population: 100
  mutation_rate: 0.01
```

## License

MIT OR Apache-2.0