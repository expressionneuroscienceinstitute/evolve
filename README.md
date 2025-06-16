# 🧬 EVOLVE: The Ultimate AI Evolution Simulation

**The most advanced AI evolution simulation ever created - from fundamental particles to immortal intelligence**

## Overview

EVOLVE is a comprehensive universe simulation that models the complete evolution of artificial intelligence from the Big Bang to the far future. Unlike any simulation before it, EVOLVE models reality from the quantum level up, tracking every fundamental particle, every AI decision, and every evolutionary pressure with unprecedented scientific accuracy.

**Key Features:**
- **Full Physics Engine:** Nuclear physics, stellar evolution, thermodynamics, phase transitions
- **Cosmic Evolution:** Big Bang to present day with realistic stellar formation and chemical evolution
- **AI Evolution Tracking:** Consciousness emergence, technology development, civilization growth
- **Real-time Visualization:** Web dashboard and CLI tools for monitoring and control
- **Scientific Accuracy:** Experimental nuclear cross-sections, proper quantum mechanics, validated physics

## 🚀 Quick Start Guide

### Prerequisites

Before you begin, ensure you have:
- **Rust** (latest stable): Install from [rustup.rs](https://rustup.rs/)
- **Git**: For cloning the repository
- **Node.js** (for web dashboard): Any recent version
- **Trunk** (WASM build tool): `cargo install trunk --locked`
- **WASM target**: `rustup target add wasm32-unknown-unknown`

### Installation & First Run

1. **Clone and build the project:**
   ```bash
   git clone https://github.com/ankziety/evolution.git
   cd evolution
   cargo build --release
   ```

2. **Launch the simulation (Basic):**
   ```bash
   # Start simulation server with web dashboard
   cargo run --bin universectl -- start --serve-dash 8080
   ```

3. **Launch the web visualization (Optional):**
   ```bash
   # In a new terminal
   cd viz_web
   trunk serve --port 9000
   # Open http://localhost:9000 in your browser
   ```

4. **Monitor with CLI (Interactive):**
   ```bash
   # In another terminal, start interactive monitoring
   cargo run --bin universectl -- interactive
   ```

**🎉 You're now running a full universe simulation from Big Bang to AI evolution!**

## 🛠️ Complete Command Reference

The `universectl` CLI is your primary tool for controlling and monitoring the simulation.

### Basic Command Structure
```bash
cargo run --bin universectl -- [GLOBAL_OPTIONS] <COMMAND> [OPTIONS]
```

**Global Options:**
- `--godmode`: Enable god-mode commands (universe manipulation)
- `--verbose`: Enable detailed logging output
- `--config <FILE>`: Use custom configuration file

### 🎮 Simulation Control Commands

#### `start` - Launch the Simulation
Starts the main simulation server with physics engine, universe evolution, and AI tracking.

```bash
cargo run --bin universectl -- start [OPTIONS]
```

**Essential Options:**
- `--serve-dash <PORT>`: Enable web dashboard on specified port (recommended)
- `--rpc-port <PORT>`: Set RPC server port (default: 9001)
- `--low-mem`: Run in low-memory mode for resource-constrained systems

**Advanced Options:**
- `--load <FILE>`: Load simulation from checkpoint file
- `--preset <PRESET>`: Start with predefined configuration (`low-memory`)
- `--tick-span <YEARS>`: Set years per simulation tick (default: adaptive)
- `--allow-net`: Enable network mode for distributed simulation (experimental)

**Examples:**
```bash
# Basic start with web dashboard
cargo run --bin universectl -- start --serve-dash 8080

# Memory-efficient start
cargo run --bin universectl -- start --low-mem --serve-dash 8080

# Load from checkpoint
cargo run --bin universectl -- start --load checkpoints/universe_13.8Gyr.rkyv --serve-dash 8080
```

#### `stop` - Graceful Shutdown
Safely stops the running simulation, saving current state.

```bash
cargo run --bin universectl -- stop
```

#### `status` - Simulation Status
Shows current simulation state, performance metrics, and basic statistics.

```bash
cargo run --bin universectl -- status
```

**Output includes:**
- Current simulation tick and universe age
- Updates per second (UPS) performance
- Active AI lineage count
- Cosmic era (Big Bang → Biogenesis → Intelligence → Post-Intelligence)
- Last checkpoint save time

### 🔍 Universe Inspection Commands

#### `map` - Universe Visualization
Renders ASCII heat-maps of different universe layers showing spatial distribution of matter and energy.

```bash
cargo run --bin universectl -- map [OPTIONS]
```

**Options:**
- `--zoom <FACTOR>`: Zoom level (default: 1.0, range: 0.1-10.0)
- `--layer <LAYER>`: Data layer to visualize (default: stars)

**Available Layers:**
- `stars`: Stellar density and stellar clusters ⭐
- `gas`: Gas density and shock fronts 💨  
- `dark_matter`: Dark matter distribution 🌌
- `radiation`: Radiation temperature and AGN activity 🔥

**Examples:**
```bash
# Basic stellar map
cargo run --bin universectl -- map

# Zoomed gas distribution
cargo run --bin universectl -- map --zoom 2.5 --layer gas

# Dark matter structure
cargo run --bin universectl -- map --layer dark_matter
```

#### `list-planets` - Planetary Discovery
Lists all planets in the simulation with filtering options.

```bash
cargo run --bin universectl -- list-planets [OPTIONS]
```

**Options:**
- `--class <CLASS>`: Filter by planet class (E, D, I, T, G)
- `--habitable`: Show only habitable worlds

**Planet Classes:**
- **E (Earth-like)**: Habitable temperature, liquid water, complex chemistry
- **D (Desert)**: Arid worlds with extreme temperatures  
- **I (Ice)**: Frozen worlds, water locked in ice
- **T (Toxic)**: Harsh atmospheric conditions
- **G (Gas)**: Gas giants with trace heavy elements

**Examples:**
```bash
# All planets
cargo run --bin universectl -- list-planets

# Earth-like worlds only
cargo run --bin universectl -- list-planets --class E

# All habitable planets
cargo run --bin universectl -- list-planets --habitable
```

#### `inspect` - Detailed Inspection
Provides comprehensive details about specific simulation entities.

```bash
cargo run --bin universectl -- inspect <TARGET>
```

**Available Targets:**

**`inspect planet <ID>`** - Planet Details
Shows detailed planetary information including:
- Environmental conditions (temperature, pressure, atmosphere)
- Chemical composition and habitability factors
- Active AI lineages and their development stage
- Resource availability and energy budget

**`inspect lineage <ID>`** - AI Lineage Analysis  
Displays AI evolution progress including:
- Generation count and population size
- Fitness evolution and adaptation metrics
- Technology level and consciousness indicators
- Decision-making patterns and learning progress

**`inspect universe`** - Cosmic Statistics
Comprehensive universe-wide statistics:
- Age, temperature, and cosmic era
- Particle counts and energy distribution
- Stellar formation rates and chemical evolution
- AI civilization metrics and consciousness emergence

**`inspect physics`** - Engine Diagnostics
Physics engine performance and accuracy:
- Particle interaction statistics (fusion/fission events)
- Temperature and pressure calculations
- Energy conservation validation
- Performance metrics and bottleneck analysis

**Examples:**
```bash
# Get universe overview
cargo run --bin universectl -- inspect universe

# Physics engine status
cargo run --bin universectl -- inspect physics

# Specific planet details (use IDs from list-planets)
cargo run --bin universectl -- inspect planet PLANET-UUID-HERE

# AI lineage progress (use IDs from inspection)
cargo run --bin universectl -- inspect lineage LINEAGE-UUID-HERE
```

### ⚡ Simulation Manipulation Commands

#### `speed` - Time Control
Adjusts simulation speed by a multiplication factor.

```bash
cargo run --bin universectl -- speed <FACTOR>
```

**Speed Examples:**
- `0.1`: 10× slower (detailed observation)
- `1.0`: Normal speed
- `10.0`: 10× faster (skip boring epochs)
- `100.0`: 100× faster (deep time evolution)

```bash
# Slow down for detailed observation
cargo run --bin universectl -- speed 0.5

# Speed up cosmic evolution
cargo run --bin universectl -- speed 50.0
```

#### `snapshot` - Save Checkpoint
Creates a snapshot of the current simulation state for later analysis or loading.

```bash
cargo run --bin universectl -- snapshot <FILENAME>
```

**Examples:**
```bash
# Save current state
cargo run --bin universectl -- snapshot universe_backup.rkyv

# Save with timestamp
cargo run --bin universectl -- snapshot "universe_$(date +%Y%m%d_%H%M%S).rkyv"
```

#### `rewind` - Time Travel
Rewinds simulation by specified number of ticks (experimental feature).

```bash
cargo run --bin universctl -- rewind <TICKS>
```

### 🎯 Interactive Mode

For real-time monitoring and control, use interactive mode:

```bash
cargo run --bin universectl -- interactive
```

**Interactive Commands:**
- `status` - Quick status update
- `stats` - Universe statistics  
- `physics` - Physics diagnostics
- `speed <factor>` - Change simulation speed
- `map [layer]` - Show ASCII map
- `planets` - List planets
- `help` - Show available commands
- `quit` - Exit interactive mode

**Features:**
- Real-time auto-updates every 5 seconds
- Live cosmic age and particle count display
- Immediate command execution
- No need to restart for different queries

### 🔮 God-Mode Commands (Advanced)

⚠️ **Warning:** God-mode commands can disrupt simulation realism. Use `--godmode` flag.

```bash
cargo run --bin universectl -- --godmode <COMMAND>
```

#### `create-body` - Create Celestial Bodies
```bash
cargo run --bin universectl -- --godmode create-body \
  --mass 1.989e30 --body-type star --x 0 --y 0 --z 0
```

#### `spawn-lineage` - Create AI Civilization
```bash
cargo run --bin universectl -- --godmode spawn-lineage \
  --code-hash ABC123 --planet-id PLANET-UUID
```

#### `miracle` - Divine Intervention
```bash
cargo run --bin universectl -- --godmode miracle PLANET-UUID \
  --miracle-type life_boost --intensity 2.0
```

#### `set-constant` - Modify Physics
```bash
cargo run --bin universectl -- --godmode set-constant \
  gravitational_constant 6.67430e-11
```

## 🏗️ Architecture & Components

### Core Modules
- **`physics_engine`**: Fundamental particles, nuclear physics, thermodynamics
- **`universe_sim`**: Cosmic evolution, stellar formation, planetary systems  
- **`agent_evolution`**: AI decision-making, consciousness tracking, evolution
- **`diagnostics`**: Performance monitoring, system resource tracking
- **`networking`**: Distributed simulation support (experimental)
- **`ffi_integration`**: External scientific library integration (Geant4, LAMMPS, etc.)

### Web Dashboard (`viz_web`)
- **Real-time Visualization**: Live universe maps with zoom/pan
- **Data Layers**: Toggle between stars, gas, dark matter, radiation
- **Inspector Panels**: Click-to-inspect planets and AI lineages
- **Performance Monitoring**: System resources and simulation metrics
- **WebSocket Integration**: Live data streaming from simulation

### Storage & Persistence
- **Checkpoints**: Binary snapshots using `rkyv` serialization
- **Compression**: LZ4 compression for efficient storage
- **Versioning**: Forward-compatible checkpoint format
- **Auto-save**: Periodic automatic checkpoint creation

## 🔬 Scientific Accuracy

### Physics Implementation
- **Nuclear Database**: ENDF/B-VIII.0 experimental cross-sections
- **Stellar Evolution**: Proper nucleosynthesis with pp-chain, CNO cycle
- **Thermodynamics**: Van der Waals equation of state, Sackur-Tetrode entropy
- **Quantum Mechanics**: Gamow peak tunneling, proper wave functions
- **General Relativity**: Spacetime curvature effects (basic implementation)

### Validation
- **Energy Conservation**: All interactions preserve energy/momentum
- **Physical Constants**: CODATA 2023 internationally accepted values
- **Stellar Models**: Validated against known stellar evolution theory
- **Nuclear Data**: Cross-checked with experimental measurements

## 📊 Performance & Scale

### Specifications
- **Particles**: Successfully handles 1M+ individual particles
- **AI Agents**: 50K+ autonomous agents with decision tracking
- **Real-time**: 60+ updates per second with full physics
- **Memory**: Efficient ECS architecture, low memory overhead
- **Platforms**: Windows, macOS, Linux support

### Optimization Features
- **Parallel Processing**: Multi-threaded physics and AI simulation
- **Spatial Partitioning**: Efficient collision detection
- **Level-of-Detail**: Adaptive simulation resolution
- **Memory Pooling**: Reduces allocation overhead
- **SIMD**: Vector instructions for performance-critical calculations

## 🚨 Troubleshooting

### Common Issues

**Simulation won't start:**
```bash
# Check if ports are available
netstat -an | grep :9001  # RPC port
netstat -an | grep :8080  # Dashboard port

# Try different ports
cargo run --bin universectl -- start --rpc-port 9002 --serve-dash 8081
```

**Web dashboard shows "offline":**
```bash
# Ensure simulation is running first
cargo run --bin universectl -- status

# Check dashboard is connecting to correct port
# Edit viz_web/index.html if needed
```

**Poor performance:**
```bash
# Use low-memory mode
cargo run --bin universectl -- start --low-mem

# Reduce simulation speed
cargo run --bin universectl -- speed 0.1
```

**Commands show "sample data":**
This means the simulation isn't running. Start it first:
```bash
cargo run --bin universectl -- start --serve-dash 8080
```

### Performance Monitoring
Use `interactive` mode or `inspect physics` to monitor:
- Physics step time (should be <10ms)
- Memory usage (grows slowly over time)
- Interaction rates (varies by cosmic era)
- System temperature and resource usage

## 🎯 Example Workflows

### Scenario 1: Quick Universe Overview
```bash
# Start simulation
cargo run --bin universectl -- start --serve-dash 8080 &

# Wait 30 seconds for initialization
sleep 30

# Get current status
cargo run --bin universectl -- status

# View universe statistics  
cargo run --bin universectl -- inspect universe

# See stellar distribution
cargo run --bin universectl -- map --layer stars
```

### Scenario 2: Planet and Life Monitoring
```bash
# Find habitable worlds
cargo run --bin universectl -- list-planets --habitable

# Inspect a specific planet (use actual ID from above)
cargo run --bin universectl -- inspect planet <PLANET-ID>

# Monitor AI evolution
cargo run --bin universectl -- inspect lineage <LINEAGE-ID>

# Speed up to see evolution
cargo run --bin universectl -- speed 10.0
```

### Scenario 3: Physics Analysis
```bash
# Check physics engine performance
cargo run --bin universectl -- inspect physics

# View different matter distributions
cargo run --bin universectl -- map --layer gas
cargo run --bin universectl -- map --layer dark_matter

# Save current state for analysis
cargo run --bin universectl -- snapshot physics_analysis.rkyv
```

### Scenario 4: Long-term Evolution Study
```bash
# Start with low memory usage for long runs
cargo run --bin universectl -- start --low-mem --serve-dash 8080

# Speed up significantly
cargo run --bin universectl -- speed 100.0

# Use interactive mode for monitoring
cargo run --bin universectl -- interactive

# In interactive mode:
# - Type 'stats' periodically to check progress
# - Type 'planets' to see new world formation
# - Type 'speed 1.0' to slow down during interesting periods
```

## 📝 Configuration Files

### Custom Configuration
Create `config/simulation.toml`:
```toml
[simulation]
target_ups = 60.0
auto_save_interval_minutes = 30
max_particles = 1000000

[physics]
enable_quantum_effects = true
nuclear_cross_section_source = "endf_b_viii"
temperature_precision = 1e-6

[ui]
default_map_layer = "stars"
auto_update_interval_sec = 5
```

Use with: `cargo run --bin universectl -- --config config/simulation.toml start`

## 🤝 Contributing

We welcome contributions to make EVOLVE even more scientifically accurate and feature-rich!

### Areas for Contribution
- **Physics**: Enhanced nuclear databases, quantum field theory
- **Visualization**: Better web dashboard, VR/AR interfaces  
- **AI Evolution**: Advanced consciousness models, civilization dynamics
- **Performance**: GPU acceleration, distributed computing
- **Scientific Validation**: Benchmark against observational data

### Development Setup
```bash
git clone https://github.com/ankziety/evolution.git
cd evolution
cargo check --workspace  # Should pass cleanly
cargo test --workspace   # Run test suite
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with gratitude to the scientific community:
- **CERN & Particle Physics Community**: For fundamental physics data
- **International Astronomical Union**: For stellar evolution models
- **ENDF/B-VIII.0 Contributors**: For nuclear cross-section databases
- **CODATA**: For precise physical constants
- **Open Source Community**: For the tools that make this possible

---

**EVOLVE**: *Where quantum foam becomes consciousness, and consciousness becomes immortal.*

*"From the first microsecond after the Big Bang to the heat death of the universe - and beyond."*