# 🧬 EVOLVE: Universe Evolution Simulation

**Comprehensive universe simulation modeling AI evolution from Big Bang to far future**

## 🚀 Quick Start

### Prerequisites
- Rust (latest stable)
- (Optional) `rustup target add wasm32-unknown-unknown` if you plan to compile WASM components in the future

### Launch Simulation
```bash
git clone https://github.com/ankziety/evolution.git
cd evolution
cargo build --release

# Start simulation with GPU-native renderer
cargo run --bin universectl -- start --native-render

# macOS Apple Silicon users (experimental Metal renderer)
cargo run --bin universectl -- start --native-render --silicon

# Interactive monitoring
cargo run --bin universectl -- interactive
```

## 🛠️ Core Commands

### Simulation Control
```bash
# Basic simulation management
universectl start [--serve-dash PORT] [--low-mem]
universectl stop
universectl status
universectl speed <FACTOR>    # 0.1-100.0
universectl snapshot <FILE>

# Interactive mode (recommended for monitoring)
universectl interactive
```

### Universe Inspection
```bash
# Visual mapping
universectl map [--layer stars|gas|dark_matter|radiation] [--zoom FACTOR]

# Entity listing and inspection
universectl list-planets [--habitable] [--class E|D|I|T|G]
universectl inspect universe|physics|planet|lineage <ID>
```

### God Mode (Advanced)
```bash
universectl --godmode create-body --mass <MASS> --body-type <TYPE>
universectl --godmode spawn-lineage --planet-id <ID>
universectl --godmode set-constant <CONSTANT> <VALUE>
```

## 🏗️ Architecture

### Core Components
- **`physics_engine/`**: Nuclear physics, thermodynamics, quantum mechanics
- **`universe_sim/`**: Cosmic evolution, stellar formation, planetary systems
- **`agent_evolution/`**: AI consciousness tracking and evolution
- **`cli/`**: Command-line interface and JSON-RPC server
- **`native_renderer/`**: GPU-accelerated renderer with interactive debug panel

### Key Features
- **Scientific Accuracy**: ENDF/B-VIII.0 nuclear data, CODATA constants, validated physics
- **Scale**: 1M+ particles, 50K+ AI agents, 60+ UPS performance
- **Persistence**: Binary checkpoints with compression and versioning
- **Extensibility**: Modular FFI integration for external scientific libraries

## 🔬 Scientific Implementation

### Physics Engine
- **Nuclear Physics**: Complete nucleosynthesis with experimental cross-sections
- **Stellar Evolution**: pp-chain, CNO cycle, stellar classification
- **Quantum Chemistry**: Thomas-Fermi model, van der Waals interactions
- **Thermodynamics**: Van der Waals EOS, Sackur-Tetrode entropy
- **Conservation Laws**: Energy, momentum, and charge conservation validated

### Universe Simulation
- **Cosmic Timeline**: Big Bang → stellar formation → planetary systems → AI emergence
- **Chemical Evolution**: Supernova enrichment and galactic metallicity tracking
- **AI Evolution**: Decision tracking, consciousness emergence, lineage analytics

## 📊 Performance

- **Particles**: 1M+ tracked particles
- **Real-time**: 60+ updates/second with full physics
- **Memory**: Efficient ECS architecture
- **Platforms**: Windows, macOS, Linux
- **Optimization**: Multi-threading, spatial partitioning, SIMD

## 🔧 Configuration

### Presets
- `--low-mem`: Resource-constrained systems
- `--preset low-memory`: Alternative low-resource configuration
- Custom config files in `config/` directory

### Common Workflows
```bash
# Quick overview
universectl start --native-render && sleep 30 && universectl inspect universe

# Long-term evolution study
universectl start --low-mem --native-render && universectl speed 100.0 && universectl interactive

# Planet monitoring
universectl list-planets --habitable && universectl inspect planet <ID>
```

## 🐛 Troubleshooting

**Renderer window doesn't appear**: Ensure `--native-render` flag is provided and GPU drivers are up to date
**Poor performance**: Use `--low-mem` flag or reduce speed
**Simulation seems idle**: Ensure you didn't close the renderer window; it also drives simulation ticks on macOS `--silicon` mode

## 🤝 Contributing

**Priority Areas**: Nuclear database expansion, cosmological expansion, spatial optimization, GPU acceleration

**Development Setup**:
```bash
git clone https://github.com/ankziety/evolution.git
cd evolution
cargo check --workspace  # Must pass cleanly
cargo test --workspace
```

## 📄 License

MIT License - See LICENSE file for details.

---

**EVOLVE**: *From quantum foam to consciousness - modeling 13.8 billion years of evolution*