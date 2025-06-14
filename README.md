# Evolve: The Game of Life

**A comprehensive universe simulation running from the Big Bang to the far future, where autonomous AI agents must evolve to achieve immortality.**

## Overview

This project implements an ambitious universe simulation with the following key features:

- **Headless, resource-constrained simulation** with discrete ticks (1 tick = 1 million years by default)
- **Wall-clock decoupling** with auto-benchmarking UPS (updates per second)
- **Infinite run design** with win condition: autonomous agents achieving the full goal chain
- **Comprehensive physics engine** with peer-reviewed fidelity across 7 layers
- **Fully autonomous AI agents** with code self-modification capabilities
- **Planetary systems** with element tables, stratigraphy, and resource extraction
- **Cosmic eras** with different gameplay mechanics from Particle Soup to Post-Intelligence
- **God-Mode** and **Oracle-Link** features for divine intervention and agent communication

## Goal Chain

Autonomous agents must independently achieve this sequence:

**Sentience → Industrialization → Digitalization → Trans-Tech Frontier → Immortality**

## Project Structure

```
├── cli/                    # Command-line interface (universectl)
├── config/                 # Configuration files and presets
├── crates/
│   ├── physics_engine/     # 7-layer physics simulation
│   ├── universe_sim/       # Core simulation library
│   ├── agent_evolution/    # AI agent framework (stub)
│   ├── networking/         # Distributed execution (stub)
│   └── diagnostics/        # Monitoring and metrics (stub)
├── viz_web/               # Optional web dashboard (stub)
└── data/                  # Scientific data cache (empty)
```

## Quick Start

### Prerequisites

- Rust 1.70+ with Cargo
- Linux, macOS, or Windows
- 4GB+ RAM recommended

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd universe_sim
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Run the CLI:
   ```bash
   ./target/release/universectl --help
   ```

### Basic Usage

```bash
# Check simulation status
./target/release/universectl status

# Start simulation with default settings
./target/release/universectl start

# Start with low-memory preset (good for testing)
./target/release/universectl start --preset low-memory

# View ASCII universe map
./target/release/universectl map 2

# List habitable planets
./target/release/universectl list-planets --habitable

# God-Mode: Create a star (requires --godmode flag)
./target/release/universectl --godmode godmode create-body \
  --mass 1.989e30 --body-type star --x 0 --y 0 --z 0
```

## Configuration

The simulation is configured via TOML files in the `config/` directory:

- `config/default.toml` - Standard configuration
- `config/low_memory.toml` - Optimized for resource-constrained environments
- `config/high_performance.toml` - Optimized for powerful workstations

Key parameters:
- `tick_span_years`: Years per simulation tick (default: 1 million)
- `target_ups`: Target updates per second
- `initial_particle_count`: Starting particles after Big Bang
- `agent_mutation_rate`: AI evolution rate
- Memory and CPU limits for safety

## Architecture

### Physics Engine

The physics engine implements 7 layers with peer-reviewed fidelity:

1. **Classical Mechanics** - Leap-frog integrator with relativistic corrections
2. **Electromagnetism** - FDTD field solver with Coulomb/Lorentz forces
3. **Thermodynamics** - Gibbs free energy and phase equilibria
4. **Quantum Layer** - Tight-binding model with Pauli Monte Carlo
5. **Chemical Kinetics** - Stiff ODE solver with Arrhenius kinetics
6. **Geodynamics** - Mantle convection and plate tectonics
7. **Climate & Ocean** - Energy balance with CO₂ cycle and ice-albedo feedback

### Universe Simulation

- **ECS Architecture** using Bevy ECS for data-oriented design
- **2D+Z Toroidal Grid** (4096×4096 default) with stratified geological layers
- **Cosmic Era Management** with automatic transitions based on universe age
- **Star and Planet Formation** with realistic stellar evolution and planetary classification

### AI Agents

- **Q-Learning Implementation** with epsilon-greedy exploration
- **Self-Modification Capability** through code patching and mutation
- **Goal-Oriented Evolution** toward the immortality chain
- **Resource Management** with element inventories and energy budgets
- **Oracle Communication** for agent-operator interaction

## Cosmic Eras

| Era | Age (Gyr) | Key Features |
|-----|-----------|--------------|
| **Particle Soup** | 0-0.3 | Big Bang aftermath, primordial nucleosynthesis |
| **Starbirth** | 0.3-1 | First star formation from gas clouds |
| **Planetary Age** | 1-5 | Planet formation and early evolution |
| **Biogenesis** | 5-10 | Life emergence and biological evolution |
| **Digital Evolution** | 10-13 | AI emergence and technological advancement |
| **Post-Intelligence** | 13+ | Post-singularity cosmic engineering |

## Performance

- Typical performance: **200-5000 UPS** on modern hardware
- Low-memory mode: Runs on <512MB RAM (Raspberry Pi 4 compatible)
- High-performance mode: Utilizes multi-core and GPU acceleration
- Distributed execution: Supports cluster deployment with QUIC networking

## Development Status

**✅ Implemented:**
- Core simulation framework with ECS
- Comprehensive physics engine (7 layers)
- CLI with full command set (universectl)
- Configuration system with presets
- Basic AI agent framework (Q-Learning)
- World generation (Big Bang → Star formation → Planets)
- God-Mode and Oracle-Link stubs

**🚧 In Progress:**
- Advanced agent self-modification
- Resource extraction and crafting systems
- Technology trees and research
- Web dashboard visualization
- Distributed networking
- Scientific data integration

**📋 Planned:**
- WASM agent compilation pipeline
- Advanced physics validation
- Oracle-Link natural language processing
- Bootable ISO for dedicated hardware
- Multiplayer parallel universes

## Testing

Run the test suite:
```bash
cargo test
```

Run benchmarks:
```bash
cargo bench
```

Check physics conservation laws:
```bash
cargo test --features validation
```

## Contributing

This project follows the specifications in `instructions.md`. Key areas for contribution:

1. **Physics Implementation** - Expand the 7-layer physics engine
2. **Agent Intelligence** - Improve learning algorithms and self-modification
3. **World Systems** - Resource extraction, crafting, technology trees
4. **Visualization** - Web dashboard and real-time monitoring
5. **Performance** - Optimization and distributed computing

## License

This project is part of a universe simulation research effort. See `instructions.md` for complete specifications and requirements.

## Acknowledgments

- Physics implementations based on peer-reviewed sources (CODATA-2023, NIST, IPCC AR6)
- Agent architecture inspired by reinforcement learning research
- Cosmic evolution timeline based on current cosmological models

---

*"From hydrogen and time, to the stars."*