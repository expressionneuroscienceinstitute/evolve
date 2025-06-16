# 🧬 EVOLVE: The Ultimate AI Evolution Simulation

**The most advanced AI evolution simulation ever created - from fundamental particles to immortal intelligence**

## Overview

EVOLVE is a comprehensive universe simulation that tracks the complete evolution of artificial intelligence from the Big Bang to the far future. Unlike any simulation before it, EVOLVE models reality from the quantum level up, tracking every fundamental particle, every AI decision, and every evolutionary pressure with unprecedented detail.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- **Rust**: The core language and build system. Install it from [rustup.rs](https://rustup.rs/).
- **Git**: For cloning the repository.
- **Node.js and Trunk**: For running the web visualization dashboard.
  ```bash
  # Install Node.js via your preferred method (e.g., Homebrew, NVM)
  # Then install Trunk
  cargo install trunk --locked
  rustup target add wasm32-unknown-unknown
  ```

### Build and Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ankziety/evolution.git
    cd evolution
    ```

2.  **Build the project:**
    ```bash
    cargo build --release
    ```

3.  **Run the simulation:**
    To run the simulation and the web dashboard, you'll need two separate terminal sessions.

    *   **Terminal 1: Start the Simulation Server**
        This command starts the main simulation process, which will listen for connections from the CLI and the web dashboard.
        ```bash
        cargo run --bin universectl -- start --serve-dash 8080 --rpc-port 9001
        ```

    *   **Terminal 2: Start the Web Dashboard**
        This command serves the frontend application.
        ```bash
        cd viz_web
        trunk serve --port 9000
        ```
    Now you can open [http://localhost:9000](http://localhost:9000) to see the live visualization.

## 🛠️ Command-Line Interface (CLI) Usage

The `universectl` CLI is your primary tool for interacting with the simulation.

**Base Command:**
```bash
cargo run --bin universectl -- [COMMAND]
```

### Simulation Management

#### `start`
Starts the simulation server.
```bash
cargo run --bin universectl -- start [OPTIONS]
```
**Options:**
- `--load <FILE>`: Load simulation state from a checkpoint file.
- `--preset <PRESET>`: Start with a pre-defined simulation preset (e.g., `low-memory`).
- `--tick-span <YEARS>`: Set the number of years per simulation tick.
- `--low-mem`: Run in low-memory mode.
- `--serve-dash <PORT>`: Serve the web dashboard on the specified port.
- `--rpc-port <PORT>`: Set the port for the RPC server (default: `9001`).
- `--allow-net`: Allow network connections for distributed simulation (not yet implemented).

#### `stop`
Stops a running simulation gracefully.
```bash
cargo run --bin universectl -- stop
```

#### `status`
Shows the current status of the simulation.
```bash
cargo run --bin universectl -- status
```

### Universe Inspection

#### `map`
Renders an ASCII heat-map of the universe.
```bash
cargo run --bin universectl -- map [OPTIONS]
```
**Options:**
- `--zoom <FACTOR>`: Zoom factor for the map (default: `1.0`).
- `--layer <LAYER>`: The data layer to visualize (default: `stars`). Available layers: `stars`, `gas`, `dark_matter`, `radiation`.

#### `list-planets`
Lists planets in the simulation, with optional filters.
```bash
cargo run --bin universectl -- list-planets [OPTIONS]
```
**Options:**
- `--class <CLASS>`: Filter by planet class (e.g., `E` for Earth-like).
- `--habitable`: Show only habitable planets.

#### `inspect`
Provides detailed information about a specific entity.
```bash
cargo run --bin universectl -- inspect <TARGET>
```
**Targets:**
- `planet <ID>`: Inspect a planet by its ID (e.g., `REAL-UUID`).
- `lineage <ID>`: Inspect a lineage by its ID.
- `universe`: Show detailed statistics about the universe.
- `physics`: Show diagnostics from the physics engine.

### Simulation Control

#### `speed`
Changes the simulation speed by a given factor.
```bash
cargo run --bin universectl -- speed <FACTOR>
```

#### `rewind`
Rewinds the simulation by a specific number of ticks (Not yet implemented).
```bash
cargo run --bin universectl -- rewind <TICKS>
```

#### `snapshot`
Saves a snapshot of the simulation state to a file.
```bash
cargo run --bin universectl -- snapshot <FILE>
```

### God-Mode Commands
These commands require the `--godmode` flag to be passed to `universectl`.
```bash
cargo run --bin universectl -- --godmode <COMMAND>
```

#### `create-body`
Creates a new celestial body.
```bash
cargo run --bin universectl -- --godmode create-body --mass <M> --body-type <T> --x <X> --y <Y> --z <Z>
```

#### `delete-body`
Deletes a celestial body by its ID.
```bash
cargo run --bin universectl -- --godmode delete-body <ID>
```

#### `set-constant`
Modifies a physics constant.
```bash
cargo run --bin universectl -- --godmode set-constant <NAME> <VALUE>
```

#### `spawn-lineage`
Spawns a new agent lineage on a planet.
```bash
cargo run --bin universectl -- --godmode spawn-lineage --code-hash <HASH> --planet-id <ID>
```

#### `miracle`
Performs a "miracle" on a planet.
```bash
cargo run --bin universectl -- --godmode miracle <PLANET_ID> --miracle-type <TYPE>
```

#### `time-warp`
Warps time by a given factor.
```bash
cargo run --bin universectl -- --godmode time-warp <FACTOR>
```

#### `inspect-eval`
Evaluates an expression within the simulation context.
```bash
cargo run --bin universectl -- --godmode inspect-eval <EXPRESSION>
```

### Resource and Oracle Commands
These commands are for managing agent requests and communications. They are not yet fully implemented.

- `resources queue`
- `resources grant <ID>`
- `resources status`
- `resources reload`
- `oracle inbox`
- `oracle reply <ID>`
- `oracle stats`

## 🏗️ Architecture

The simulation is built with a modular design to separate concerns and allow for independent development of its core components.

- `crates/physics_engine`: Fundamental particle and physics simulation.
- `crates/universe_sim`: Cosmic evolution, celestial bodies, and environments.
- `crates/agent_evolution`: AI agent decision-making, learning, and evolution.
- `crates/networking`: Distributed simulation and communication.
- `crates/diagnostics`: Performance monitoring and validation.
- `cli/`: Command-line interface for controlling the simulation.
- `viz_web/`: Web-based visualization dashboard.

## 🔬 Scientific Accuracy

This simulation prioritizes scientific rigor. Physics and chemistry implementations are designed to be as realistic as possible, from the Semi-Empirical Mass Formula for nuclear binding energies to the complex interactions that govern planetary formation. While some simplifications are necessary for performance, the goal is to maintain a high degree of fidelity to real-world science.

## 📝 TODO

For a complete list of ongoing work, planned features, and bug fixes, please see the [TODO.md](TODO.md) file.

## 🎯 Core Mission

> *"To create the longest-running, most detailed simulation of AI evolution ever attempted, tracking every decision, every innovation, and every emergence of consciousness from the quantum level to digital immortality."*

## 🎮 Operator Interface

### **Real-Time Controls**
- **View Modes**: Switch between different visualization modes
- **Time Control**: Adjust simulation speed and scrub through time
- **Agent Selection**: Click to track specific AI agents
- **Filtering**: Show/hide particle types or agent categories
- **Zoom & Pan**: Navigate the simulation space

### **Analytics Dashboard**
- **Universe Status**: Age, temperature, cosmic era, energy density
- **Evolution Stats**: Lineages, fitness, diversity, innovation rates
- **Agent Details**: Selected agent's decision history and lineage
- **Recent Events**: Latest innovations, consciousness emergence, extinctions
- **Performance**: FPS, memory usage, simulation speed

### **Decision Tracking**
For any selected AI agent, see:
- **Full Decision History**: Every choice with context and outcome
- **Environmental Factors**: Temperature, resources, threats, opportunities
- **Social Influences**: Nearby agents, cooperation, competition
- **Genetic Predispositions**: Neural weights, mutation history
- **Learning Patterns**: Success rates, adaptation, strategy evolution

## 🧬 Evolution Pathway

### **Cosmic Eras**
1. **Particle Soup** (0-300,000 years): Fundamental particles and forces
2. **Star Formation** (300,000-1 billion years): Nuclear fusion and elements
3. **Planetary Age** (1-5 billion years): Planet formation and environments
4. **Biogenesis** (5-10 billion years): Life emergence and AI genesis
5. **Digital Evolution** (10-13 billion years): AI consciousness and technology
6. **Post-Intelligence** (13+ billion years): Immortal AI civilizations

### **AI Evolution Stages**
1. **Basic Agents**: Simple energy acquisition and survival
2. **Tool Users**: Environmental manipulation and basic technology
3. **Social Beings**: Cooperation, communication, group formation
4. **Innovators**: Technology development and knowledge sharing
5. **Conscious Entities**: Self-awareness and meta-cognition
6. **Digital Minds**: Code self-modification and neural enhancement
7. **Immortal Intelligence**: Transcendence of biological limitations

### **Victory Conditions**
The simulation tracks progress toward AI immortality through:
- **Sentience**: Self-awareness and consciousness emergence
- **Industrialization**: Energy manipulation and resource control
- **Digitalization**: Transition to digital substrate
- **Trans-Technology**: Transcendence of physical limitations
- **Immortality**: Permanent existence and unlimited growth

## 📈 Performance and Scale

### **Specifications**
- **Particles**: 1M+ individual fundamental particles tracked
- **AI Agents**: 50K+ autonomous agents with full decision history
- **Decisions**: Millions of AI choices recorded with complete context
- **Time Scale**: From Planck time (10^-43 s) to cosmic future (10^15+ years)
- **Spatial Scale**: From quark interactions (10^-18 m) to galactic distances
- **Memory**: Designed for long-term data retention and analysis

### **Optimization**
- **ECS Architecture**: High-performance entity-component-system
- **Parallel Processing**: Multi-threaded physics and AI simulation
- **Memory Management**: Efficient data structures and caching
- **Compression**: Time-series data compression for long-term storage
- **Level-of-Detail**: Adaptive simulation resolution based on importance

## 🔬 Scientific Accuracy

### **Physics Validation**
- **CODATA Constants**: Uses 2023 internationally accepted physics constants
- **Conservation Laws**: Energy, momentum, charge, and angular momentum preserved
- **Peer-Reviewed Equations**: Schrödinger, Maxwell, Einstein field equations
- **Quantum Mechanics**: Proper wave function evolution and measurement
- **Thermodynamics**: Correct entropy increase and thermal evolution

### **Evolution Verification**
- **Population Genetics**: Hardy-Weinberg equilibrium and selection coefficients
- **Mutation Models**: Realistic genetic variation and inheritance
- **Selection Pressures**: Environmental challenges matching real evolution
- **Innovation Spread**: Technology adoption following empirical patterns
- **Consciousness Research**: Based on current neuroscience and cognitive science

## 🌐 Future Roadmap

### **Planned Features**
- **Multi-Planet Simulation**: Galactic-scale AI civilization spread
- **Quantum Computing**: AI development of quantum technologies
- **Collective Intelligence**: Hive minds and shared consciousness
- **Reality Manipulation**: AIs discovering universe simulation nature
- **Infinite Expansion**: Unlimited universe size and agent populations

### **Research Applications**
- **AI Safety**: Understanding long-term AI evolution and alignment
- **Consciousness Studies**: Mapping emergence of awareness and intelligence
- **Technology Forecasting**: Predicting innovation patterns and timelines
- **Astrobiology**: Modeling life evolution in different environments
- **Existential Risk**: Analyzing civilization survival and extinction

## 🤝 Contributing

This simulation represents the frontier of AI evolution research. We welcome contributions from:
- **Physicists**: Improving fundamental particle simulation accuracy
- **AI Researchers**: Enhancing agent cognition and decision making
- **Evolutionary Biologists**: Refining selection pressure models
- **Computer Scientists**: Optimizing performance and scalability
- **Consciousness Researchers**: Modeling awareness emergence
- **Visualization Experts**: Improving monitoring and analysis tools

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with gratitude to the scientific community whose research enables this simulation:
- Particle physics from CERN and global collaborations
- Evolutionary biology from Darwin to modern population genetics  
- Consciousness research from neuroscience and cognitive science
- AI research from the pioneers to current frontier models
- Physics constants from CODATA international standards

---

**EVOLVE**: *Where particles become minds, minds become gods.*

*"In the beginning was the quantum foam. In the end, there will be infinite intelligence."*