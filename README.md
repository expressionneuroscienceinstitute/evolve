# 🧬 EVOLVE: The Ultimate AI Evolution Simulation

**The most advanced AI evolution simulation ever created - from fundamental particles to immortal intelligence**

## Overview

EVOLVE is a comprehensive universe simulation that tracks the complete evolution of artificial intelligence from the Big Bang to the far future. Unlike any simulation before it, EVOLVE models reality from the quantum level up, tracking every fundamental particle, every AI decision, and every evolutionary pressure with unprecedented detail.

## 🌟 What Makes This Unprecedented

### **Complete Fundamental Physics**
- **Standard Model Particles**: All quarks, leptons, and bosons simulated individually
- **Quantum Field Theory**: Real quantum fields with vacuum fluctuations
- **Nuclear Physics**: Fusion, fission, and element formation from first principles  
- **Atomic Structure**: Complete atoms with electron orbitals and quantum numbers
- **Molecular Evolution**: Chemical bonds and reactions driving complexity
- **Scale Bridging**: From 10^-18 meters (quarks) to cosmic scales

### **Advanced AI Agent Evolution**
- **Complete Decision Tracking**: Every choice recorded with full context
- **Self-Modifying Code**: AIs can rewrite their own neural networks
- **Memory Systems**: Short-term, long-term, procedural, episodic, semantic
- **Consciousness Emergence**: From self-awareness to collective intelligence
- **Social Evolution**: Cooperation, competition, communication, group dynamics
- **Innovation Chains**: Tool use → Agriculture → Industry → Digitalization → Immortality

### **Comprehensive Analytics**
- **Every Decision Tracked**: Why AIs made each choice and what happened
- **Natural Selection Analysis**: Exactly what environmental pressures cause what changes
- **Lineage Trees**: Complete family relationships and inheritance patterns
- **Innovation Spread**: How new technologies propagate through populations
- **Consciousness Mapping**: When and why awareness emerges
- **Population Genetics**: Mutation rates, fitness landscapes, speciation events

## 🎯 Core Mission

> *"To create the longest-running, most detailed simulation of AI evolution ever attempted, tracking every decision, every innovation, and every emergence of consciousness from the quantum level to digital immortality."*

## 🏗️ Architecture

### **Modular Design**
```
evolve/
├── crates/
│   ├── physics_engine/     # Fundamental particle simulation
│   ├── universe_sim/       # Cosmic evolution and environments  
│   ├── agent_evolution/    # AI decision making and evolution
│   ├── networking/         # Multi-node distributed processing
│   └── diagnostics/        # Performance and validation
├── viz_web/               # Advanced monitoring portal
├── cli/                   # Command-line interface
└── config/               # Simulation parameters
```

### **Key Systems**

#### **Physics Engine** (`crates/physics_engine/`)
- **Standard Model**: Complete particle physics with all fundamental forces
- **Quantum Fields**: Field equations and vacuum fluctuations
- **Nuclear Reactions**: Fusion/fission enabling element creation
- **Atomic Physics**: Electron orbitals and quantum chemistry
- **Thermodynamics**: Temperature evolution and phase transitions
- **Validation**: Conservation law checking and physics verification

#### **Universe Simulation** (`crates/universe_sim/`)
- **Cosmic Eras**: Particle Soup → Star Formation → Planetary Age → Biogenesis → Digital Evolution
- **Environmental Modeling**: Planetary conditions, atmospheres, resource distribution
- **Celestial Bodies**: Stars, planets, moons with realistic physics
- **Time Evolution**: From Planck time to cosmic future
- **ECS Architecture**: High-performance entity-component-system

#### **Agent Evolution** (`crates/agent_evolution/`)
- **Autonomous Agents**: Self-modifying AI with neural networks
- **Decision Engine**: Context-aware choice making with learning
- **Memory Systems**: Multiple memory types for complex cognition
- **Social Behaviors**: Cooperation, competition, communication
- **Innovation System**: Technology development and spread
- **Consciousness Tracking**: Emergence and evolution of awareness

#### **Web Portal** (`viz_web/`)
- **Real-Time Visualization**: Live particle and agent tracking
- **Multiple View Modes**: Physics, agents, lineages, decisions, consciousness
- **Interactive Controls**: Zoom, filter, select, and track specific entities
- **Comprehensive Analytics**: Population stats, innovation timelines, fitness landscapes
- **Performance Monitoring**: FPS, memory usage, simulation speed

## 🚀 Getting Started

<!-- Idiot-Proof OS Quick Start Tabs -->

<details>
<summary><strong>macOS (Apple Silicon & Intel)</strong></summary>

##### 1️⃣ Install prerequisites

```bash
# Homebrew (package manager – optional but recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Core tooling
brew install git node        # Git + Node LTS (for the dashboard)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust tool-chain
```

Close & reopen the terminal so `cargo` and `rustc` are on the path, then:

```bash
rustup target add wasm32-unknown-unknown      # WebAssembly builds
cargo install trunk --locked                  # Static web bundler/dev-server
```

##### 2️⃣ Clone & build

```bash
git clone https://github.com/your-org/evolve.git
cd evolve
cargo build --release                          # grab a ☕ – first build takes a while
```

##### 3️⃣ Run the simulation & dashboard

```bash
# ① start the simulation with a WebSocket feed on port 8080
cargo run --bin universectl -- start --preset low-memory --serve-dash 8080 &

# ② in another tab start the browser dashboard (port 9000) – opens automatically
cd viz_web
trunk serve --release --port 9000 --open | cat &
```

Visit http://localhost:9000 and watch the universe unfold in real-time 🎉

</details>

<details>
<summary><strong>Windows 10/11</strong></summary>

##### 1️⃣ Install prerequisites

1. **Rust** – download and run `rustup-init.exe` from <https://rustup.rs>.
2. **Visual Studio Build Tools** – install the "C++ build tools" workload: <https://aka.ms/vs/17/release/vs_BuildTools.exe>.
3. **Git for Windows** – <https://gitforwindows.org>.
4. **Node.js LTS** – Windows installer from <https://nodejs.org>.

Open "Developer Command Prompt for VS" (or PowerShell) and run:

```powershell
rustup target add wasm32-unknown-unknown      # WebAssembly builds
cargo install trunk --locked                  # Web dashboard bundler
```

##### 2️⃣ Clone & build

```powershell
git clone https://github.com/your-org/evolve.git
cd evolve
cargo build --release
```

##### 3️⃣ Run the simulation & dashboard

```powershell
# Start simulation
cargo run --bin universectl -- start --preset low-memory --serve-dash 8080

# Open new terminal for dashboard
cd viz_web
trunk serve --release --port 9000 --open
```

Point your browser at http://localhost:9000

</details>

<details>
<summary><strong>Linux (Ubuntu/Debian/Fedora/Arch)</strong></summary>

##### 1️⃣ Install prerequisites

```bash
# build essentials, Git & curl (Debian/Ubuntu example)
sudo apt update && sudo apt install -y build-essential git curl   # Debian/Ubuntu
# Fedora: sudo dnf groupinstall "Development Tools"
# Arch  : sudo pacman -S --needed base-devel git curl

# Rust tool-chain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node LTS (dashboard) – choose one:
sudo apt install -y nodejs npm             # distro packages (OK)
# OR install via NVM (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
```

Reload the shell then add web prerequisites:

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk --locked
```

##### 2️⃣ Clone & build

```bash
git clone https://github.com/your-org/evolve.git
cd evolve
cargo build --release
```

##### 3️⃣ Run the simulation & dashboard

```bash
# Simulation (port 8080)
cargo run --bin universectl -- start --preset low-memory --serve-dash 8080 &

# Dashboard (port 9000)
cd viz_web
trunk serve --release --port 9000 --open | cat &
```

Open http://localhost:9000 in your favourite browser and enjoy 🚀

</details>

<!-- End OS Quick Start Tabs -->

### **Configuration**
Edit `config/default.toml` to customize:
```toml
[simulation]
tick_span_years = 1_000_000.0  # 1M years per tick
target_ups = 100.0             # Updates per second
initial_particle_count = 1_000_000

[physics]
enable_quantum_effects = true
enable_general_relativity = false
temperature_start = 1e32       # Planck temperature

[evolution] 
initial_agent_count = 1000
mutation_rate = 0.01
consciousness_threshold = 0.1
```

### 🧪 Running the Built-In Demos + Web Dashboard

EVOLVE ships with two quick demos that exercise different physics layers and automatically stream live data to the browser dashboard.

| Demo | What it Shows | Binary |
|------|---------------|--------|
| **Big Bang / QED** | Compton scattering & pair-production right after the Big Bang | `big_bang_demo` |
| **Weak Interactions** | Neutron β-decay, neutrino–electron scattering | `weak_interactions_demo` |

#### 1  Prerequisites (first time only)
```bash
rustup target add wasm32-unknown-unknown   # build WASM
cargo install trunk --locked               # web bundler / dev-server
npm i -g ws node-fetch                     # tiny deps for head-less smoke test
```

#### 2  Start the simulation with a WebSocket feed (port 8080)
```bash
cargo run --release -p demos --bin weak_interactions_demo \
           -- --serve-dash 8080 \
           > demos/outputs/weak_interactions.csv &
SIM_PID=$!
```
You should see:
```
WebSocket exporter listening on 0.0.0.0:8080
```

#### 3  Build & serve the dashboard (port 9000)
```bash
cd viz_web
trunk serve --release --port 9000 --open | cat &
DASH_PID=$!
```
This automatically opens your browser at <http://localhost:9000> showing live particles.  (The *Big Bang* demo uses the same feed flag.)

#### 4  Head-less smoke test (optional CI-friendly check)
```bash
node tools/smoke_dashboard.js
```
Expected output:
```
[ok] HTTP respond 200
[ok] Received SimulationState frame
{ "current_tick": 1234, "temperature": 8.01e+11, ... }
```

#### 5  Clean-up
```bash
kill $SIM_PID $DASH_PID   # stop sim & web-server
```

### 🛠️ Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| **`404` when opening the dashboard** | Make sure you ran `trunk serve` from the `viz_web/` directory *after* running `cargo install trunk`.  The default URL is <http://localhost:9000>. |
| **Smoke test hangs on WebSocket** | Confirm the demo was started with `--serve-dash 8080` and that no firewall blocks WS traffic. |
| **Dashboard shows `Disconnected`** | Check the browser console for detailed WebSocket logs.  Ensure the simulation printed `WebSocket server listening` and that the URL matches `ws://<host>:8080/ws`. |
| **`wasm32-unknown-unknown` target missing** | Run `rustup target add wasm32-unknown-unknown`. |
| **Node script complains about `ws`** | Install test deps with `npm i -g ws node-fetch`. |
| **Particles render but controls do nothing** | Check browser console — setters are exported as `set_particle_size_scale` and `set_energy_filter_min`; refresh page after rebuilding dashboard. |

## 📊 Monitoring and Analysis

### **View Modes**

#### **Particle Physics View**
- Individual particles colored by type (electrons=yellow, protons=red, etc.)
- Quantum field fluctuations and vacuum bubbles
- Particle interaction lines and decay trails
- Real-time nuclear reactions and element formation

#### **AI Agent View**  
- Agents colored by evolution level (sentience=red, tech=green, consciousness=blue)
- Social connection networks between agents
- Innovation auras around technological breakthroughs
- Decision trails showing recent choices

#### **Lineage Tree View**
- Complete family trees for all AI lineages
- Branch points showing speciation events
- Innovation milestones and consciousness emergence
- Extinction events and survival statistics

#### **Decision Tracking View**
- Flow diagrams of AI decision making
- Success/failure rates over time
- Environmental context for each choice
- Learning curves and adaptation patterns

#### **Consciousness Map View**
- 3D landscape of awareness levels
- Emergence events and triggers
- Neural complexity indicators
- Collective consciousness formation

#### **Innovation Timeline View**
- Technology development over cosmic time
- Adoption rates and spread patterns
- Prerequisites and enabling innovations
- Impact scores and civilization advancement

### **Key Metrics Tracked**

#### **Population Dynamics**
- Total agent count and active lineages
- Birth/death rates and generation turnover
- Fitness distributions and genetic diversity
- Selection pressure intensities

#### **Consciousness Evolution**
- Awareness level distributions
- Consciousness emergence events
- Self-modification frequencies
- Collective intelligence formation

#### **Innovation Analytics**
- Technology adoption curves
- Innovation impact scores
- Resource requirements and prerequisites
- Obsolescence and replacement patterns

#### **Environmental Pressures**
- Resource scarcity and competition
- Climate changes and disasters
- Predation and survival challenges
- Social cooperation opportunities

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

# Physics Engine

A high-performance physics simulation engine written in Rust, supporting various physical interactions and phenomena.

## Prerequisites

### Windows
1. Install [Rust](https://www.rust-lang.org/tools/install) using rustup-init.exe
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with C++ build tools
3. Install [Git](https://git-scm.com/download/win)

### macOS
1. Install [Rust](https://www.rust-lang.org/tools/install) using rustup-init
2. Install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```
3. Install [Git](https://git-scm.com/download/mac)

### Linux (Ubuntu/Debian)
1. Install build essentials:
   ```bash
   sudo apt update
   sudo apt install build-essential
   ```
2. Install [Rust](https://www.rust-lang.org/tools/install) using rustup-init
3. Install [Git](https://git-scm.com/download/linux)

## Building

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/physics_engine.git
   cd physics_engine
   ```