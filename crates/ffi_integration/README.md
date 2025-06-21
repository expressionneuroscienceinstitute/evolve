# Geant4 FFI Integration

This directory contains the Foreign Function Interface (FFI) integration for Geant4, providing high-fidelity particle physics simulation capabilities to the EVOLVE universe simulation.

## Overview

The FFI integration supports two modes:

1. **Stub Mode (Default)**: Uses lightweight C stubs that provide the API without actual Geant4 functionality
2. **Real Geant4 Mode**: Uses a dynamically linked C++ wrapper that provides full Geant4 Monte Carlo simulation

## Quick Start

### Option 1: Using Stubs (No Geant4 Required)

```bash
# Build with stubs (default)
cd /path/to/evolution
cargo build --features geant4

# The system will use stub implementations
```

### Option 2: Using Real Geant4

```bash
# 1. Install Geant4 (see Installation section below)
# 2. Build the wrapper library
cd crates/ffi_integration
./build_wrapper.sh

# 3. Build the Rust project
cd ../..
cargo build --features geant4

# The system will automatically detect and use the real library
```

## Installation

### Installing Geant4

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install geant4-dev geant4-data libgeant4-dev
```

#### macOS (via Homebrew)
```bash
brew install geant4
```

#### From Source
```bash
# Download Geant4 source
wget https://geant4-data.web.cern.ch/releases/geant4-v11.1.1.tar.gz
tar -xzf geant4-v11.1.1.tar.gz
cd geant4-v11.1.1

# Build and install
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/geant4 \
      -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_BUILD_MULTITHREADED=ON \
      ..
make -j$(nproc)
sudo make install

# Set environment
export GEANT4_DIR=/usr/local/geant4
export PATH=$GEANT4_DIR/bin:$PATH
```

## Building the Wrapper Library

### Automatic Build

The `build_wrapper.sh` script automates the entire process:

```bash
cd crates/ffi_integration

# Check if Geant4 is available
./build_wrapper.sh --check

# Build the wrapper library
./build_wrapper.sh

# Build with debug symbols
./build_wrapper.sh --debug

# Clean build artifacts
./build_wrapper.sh --clean
```

### Manual Build

Using the provided Makefile:

```bash
cd crates/ffi_integration

# Show build configuration
make config

# Build the library
make all

# Test the library
make test

# Install system-wide (optional)
sudo make install

# Clean build artifacts
make clean
```

## File Structure

```
crates/ffi_integration/
├── src/
│   ├── geant4_wrapper.h      # C header interface
│   ├── geant4_wrapper.cpp    # C++ implementation (real Geant4)
│   ├── geant4_stubs.c        # C stub implementation (fallback)
│   └── geant4.rs             # Rust FFI wrapper
├── build/                    # Build artifacts (created during build)
│   ├── obj/                  # Object files
│   └── lib/                  # Built libraries
├── Makefile                  # Build configuration
├── build_wrapper.sh          # Automated build script
├── build.rs                  # Cargo build script
└── README.md                 # This file
```

## API Overview

The FFI wrapper provides a C interface to Geant4 functionality:

### Core Functions

```c
// Availability and initialization
int g4_is_available(void);
int g4_global_initialize(void);
void g4_global_cleanup(void);

// Run management
G4RunManager* g4_create_run_manager(void);
void g4_delete_run_manager(G4RunManager* manager);
void g4_run_beam_on(G4RunManager* manager, int n_events);

// Particle simulation
G4ParticleGun* g4_create_particle_gun(void);
void g4_set_particle_gun_particle(G4ParticleGun* gun, int pdg_code);
void g4_set_particle_gun_energy(G4ParticleGun* gun, double energy_mev);

// Physics queries
double g4_get_cross_section(const char* particle, const char* material, 
                           const char* process, double energy_mev);
double g4_get_stopping_power(const char* particle, const char* material, 
                            double energy_mev);
```

### Rust Interface

```rust
use ffi_integration::geant4::{Geant4Engine, is_available};

// Check if real Geant4 is available
if is_available() {
    println!("Real Geant4 is available!");
    
    // Create and configure engine
    let mut engine = Geant4Engine::new("FTFP_BERT")?;
    
    // Simulate particle transport
    let interactions = engine.transport_particle(&particle, "G4_WATER", 1.0)?;
    
    // Calculate cross-sections
    let xsec = engine.calculate_cross_section(&particle_type, "G4_WATER", 
                                             "photoelectric", 1.0)?;
} else {
    println!("Using Geant4 stubs");
}
```

## Environment Variables

- `GEANT4_DIR`: Path to Geant4 installation (if `geant4-config` not in PATH)
- `CXX`: C++ compiler to use (default: `g++`)
- `LD_LIBRARY_PATH`: Should include the path to built wrapper library

## Troubleshooting

### Build Issues

**Problem**: `geant4-config not found`
```bash
# Solution: Set GEANT4_DIR
export GEANT4_DIR=/path/to/geant4/installation
./build_wrapper.sh
```

**Problem**: Missing Geant4 headers
```bash
# Solution: Install development packages
sudo apt-get install geant4-dev libgeant4-dev  # Ubuntu/Debian
brew install geant4                            # macOS
```

**Problem**: Linking errors
```bash
# Solution: Check library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GEANT4_DIR/lib:$LD_LIBRARY_PATH
```

### Runtime Issues

**Problem**: `g4_is_available()` returns 0 even with real library
- Check that the wrapper library was built successfully
- Verify Geant4 data files are installed
- Check environment variables

**Problem**: Segmentation faults
- Ensure Geant4 is properly initialized before use
- Check that detector construction and physics list are set
- Verify particle definitions are valid

### Verification

Test that everything is working:

```bash
cd crates/ffi_integration

# Check Geant4 installation
./build_wrapper.sh --check

# Build and test wrapper
./build_wrapper.sh
make test

# Build Rust project
cd ../..
cargo build --features geant4

# Run a demo
cargo run --bin universectl -- --help
```

## Performance Notes

- **Stub Mode**: Very fast, no physics calculations
- **Real Geant4**: Slower but scientifically accurate
- **Memory Usage**: Real Geant4 requires ~100MB+ for physics tables
- **Multithreading**: Supported if Geant4 was built with MT support

## Scientific Accuracy

The real Geant4 implementation provides:

- ✅ Experimental cross-sections from evaluated databases
- ✅ Monte Carlo transport with detailed physics processes
- ✅ Accurate material properties and interactions
- ✅ Validated against experimental data
- ✅ Support for complex detector geometries

The stub implementation provides:
- ❌ No real physics calculations
- ❌ Constant/zero return values
- ✅ API compatibility for development
- ✅ Fast compilation and testing

## Contributing

When modifying the FFI interface:

1. Update the C header (`geant4_wrapper.h`)
2. Implement in both C++ wrapper (`geant4_wrapper.cpp`) and C stubs (`geant4_stubs.c`)
3. Update the Rust interface (`geant4.rs`)
4. Test both stub and real implementations
5. Update this README

## License

This FFI wrapper follows the same license as the main EVOLVE project. Note that Geant4 itself is distributed under the [Geant4 Software License](https://geant4.web.cern.ch/license). 