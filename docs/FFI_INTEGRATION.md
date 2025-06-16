# High-Fidelity Scientific Library Integration Guide

This guide explains how to integrate proven C/C++ scientific libraries with EVOLVE for maximum accuracy and performance.

## Overview

EVOLVE uses Foreign Function Interface (FFI) to integrate with battle-tested scientific libraries:

- **Geant4**: Particle physics Monte Carlo simulation
- **LAMMPS**: Molecular dynamics simulation
- **GADGET**: Cosmological N-body simulation
- **ENDF Libraries**: Nuclear data evaluation

## Prerequisites

### System Requirements

- **Compiler**: GCC 9+ or Clang 10+
- **CMake**: 3.16+
- **Python**: 3.8+ (for build scripts)
- **Memory**: 16GB+ recommended for large simulations
- **Disk**: 20GB+ for all scientific libraries

### Environment Variables

```bash
# Core library paths
export GEANT4_DIR=/usr/local/geant4
export LAMMPS_DIR=/usr/local/lammps
export GADGET_SRC=/usr/local/src/gadget
export ENDF_LIB_DIR=/usr/local/endf

# Compiler and build settings
export CC=gcc
export CXX=g++
export CMAKE_BUILD_TYPE=Release
export RUST_BACKTRACE=1
```

## Library Installation

### 1. Geant4 Installation

```bash
# Download Geant4
cd /tmp
wget https://geant4-data.web.cern.ch/releases/geant4-v11.2.0.tar.gz
tar -xzf geant4-v11.2.0.tar.gz
cd geant4-v11.2.0

# Build and install
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/geant4 \
      -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_BUILD_MULTITHREADED=ON \
      -DGEANT4_USE_GDML=ON \
      -DGEANT4_USE_QT=OFF \
      ..
make -j$(nproc)
sudo make install

# Set environment
export GEANT4_DIR=/usr/local/geant4
source $GEANT4_DIR/bin/geant4.sh
```

### 2. LAMMPS Installation

```bash
# Clone LAMMPS
cd /tmp
git clone -b stable https://github.com/lammps/lammps.git
cd lammps

# Build shared library
mkdir build && cd build
cmake -C ../cmake/presets/basic.cmake \
      -DCMAKE_INSTALL_PREFIX=/usr/local/lammps \
      -DBUILD_SHARED_LIBS=ON \
      -DLAMMPS_EXCEPTIONS=ON \
      -DPKG_MOLECULE=ON \
      -DPKG_MANYBODY=ON \
      -DPKG_KSPACE=ON \
      ../cmake
make -j$(nproc)
sudo make install

export LAMMPS_DIR=/usr/local/lammps
export LD_LIBRARY_PATH=$LAMMPS_DIR/lib:$LD_LIBRARY_PATH
```

### 3. GADGET Installation

```bash
# Download GADGET (requires registration)
# https://www.h-its.org/2018/02/22/gadget-code/
cd /usr/local/src
sudo wget [GADGET_URL] -O gadget.tar.gz
sudo tar -xzf gadget.tar.gz
sudo mv gadget-4 gadget
cd gadget

# Create FFI-compatible Makefile
sudo cp Makefile.template Makefile
sudo sed -i 's/^SYSTYPE=.*/SYSTYPE="Generic-gcc"/' Makefile
sudo sed -i 's/^#DOUBLEPRECISION/DOUBLEPRECISION/' Makefile
sudo sed -i 's/^#GADGET_FFI/GADGET_FFI/' Makefile

export GADGET_SRC=/usr/local/src/gadget
```

### 4. ENDF Data Libraries

```bash
# Download ENDF/B-VIII.0 data
cd /tmp
wget https://www.nndc.bnl.gov/endf/b8.0/download/ENDF-B-VIII.0_neutrons.tar.gz
tar -xzf ENDF-B-VIII.0_neutrons.tar.gz

# Install ENDF parser library
git clone https://github.com/nuclearkatie/endf-parser.git
cd endf-parser
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/endf ..
make -j$(nproc)
sudo make install

export ENDF_LIB_DIR=/usr/local/endf
```

## Building EVOLVE with FFI

### 1. Enable FFI Features

```bash
# Build with all scientific libraries
cargo build --release --features "geant4,lammps,gadget,endf"

# Build with specific libraries only
cargo build --release --features "geant4,lammps"
```

### 2. Cargo.toml Configuration

```toml
[features]
default = []
geant4 = []
lammps = []
gadget = []
endf = []
all_ffi = ["geant4", "lammps", "gadget", "endf"]

[dependencies]
libc = "0.2"
libloading = "0.8"

[build-dependencies]
cc = "1.0"
pkg-config = "0.3"
bindgen = "0.69"
```

### 3. Runtime Configuration

```bash
# Check library availability
cargo run --bin universectl -- check-ffi

# Run with maximum fidelity
cargo run --bin universectl -- start --ffi-mode --max-fidelity
```

## Usage Examples

### Geant4 Particle Physics

```rust
use crate::ffi::geant4::Geant4Engine;

// Initialize Geant4 with physics list
let mut geant4 = Geant4Engine::new("QBBC")?;

// Transport electron through silicon
let electron = FundamentalParticle::new_electron(1e6); // 1 MeV
let interactions = geant4.transport_particle(&electron, "Silicon", 1.0)?;

// Get cross-section data
let cross_section = geant4.calculate_cross_section(
    &ParticleType::Electron,
    "Silicon", 
    "ComptonScattering",
    1.0 // 1 MeV
)?;
```

### LAMMPS Molecular Dynamics

```rust
use crate::ffi::lammps::LammpsEngine;

// Initialize LAMMPS
let mut lammps = LammpsEngine::new()?;

// Set up water simulation
lammps.create_box([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0])?;
lammps.create_atoms_random(1000, "H2O")?;
lammps.set_force_field("TIP3P")?;

// Run molecular dynamics
lammps.run_dynamics(1000, 0.001)?; // 1000 steps, 1 fs timestep
```

### GADGET N-body Gravity

```rust
use crate::ffi::gadget::GadgetEngine;

// Initialize cosmological simulation
let mut gadget = GadgetEngine::new()?;

// Set cosmological parameters (Planck 2018)
gadget.set_cosmology(0.674, 0.315, 0.685)?; // H0, Omega_m, Omega_lambda

// Add dark matter particles
gadget.add_dark_matter_particles(1_000_000)?;

// Evolve universe
gadget.evolve_to_redshift(0.0, 1000)?; // z=0, 1000 timesteps
```

## Performance Optimization

### Compilation Flags

```bash
# Maximum optimization
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Link-time optimization
export CARGO_PROFILE_RELEASE_LTO=true
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
```

### Memory Management

```rust
// Configure FFI memory limits
let ffi_config = FfiConfig {
    max_particles: 10_000_000,
    thread_count: num_cpus::get(),
    memory_limit_mb: 32768, // 32 GB
    precision: FfiPrecision::Double,
};
```

### Parallel Processing

```bash
# Set OpenMP threads for scientific libraries
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
```

## Troubleshooting

### Common Issues

1. **Library Not Found**
   ```bash
   error: could not find native library 'G4run'
   ```
   Solution: Check `GEANT4_DIR` and `LD_LIBRARY_PATH`

2. **Binding Generation Failed**
   ```bash
   error: Unable to generate Geant4 bindings
   ```
   Solution: Install `clang-dev` and check header paths

3. **Runtime Crashes**
   ```bash
   Segmentation fault in g4_transport_particle
   ```
   Solution: Verify library versions and initialization order

### Debug Mode

```bash
# Enable FFI debugging
export RUST_LOG=debug
export EVOLVE_FFI_DEBUG=1

# Run with detailed logging
cargo run --bin universectl -- start --ffi-mode --verbose
```

### Performance Profiling

```bash
# Profile FFI calls
perf record -g cargo run --bin universectl -- start --ffi-mode
perf report

# Memory profiling
valgrind --tool=memcheck --leak-check=full ./target/release/universectl start --ffi-mode
```

## Integration Testing

### Unit Tests

```bash
# Test FFI integration
cargo test --features "all_ffi" ffi::

# Test specific libraries
cargo test --features "geant4" ffi::geant4::
```

### Validation Tests

```bash
# Compare with reference results
cargo run --bin validate-ffi -- --library geant4 --test compton-scattering
cargo run --bin validate-ffi -- --library lammps --test water-dynamics
```

## Scientific Accuracy Validation

### Geant4 Validation

- **Compton Scattering**: Klein-Nishina formula
- **Photoelectric Effect**: Experimental cross-sections
- **Pair Production**: Bethe-Heitler theory
- **Energy Loss**: Bethe-Bloch formula

### LAMMPS Validation

- **Thermodynamics**: Maxwell-Boltzmann distribution
- **Radial Distribution Functions**: Experimental data
- **Diffusion Coefficients**: Literature values
- **Phase Transitions**: Known critical points

### GADGET Validation

- **Cosmological Evolution**: Millennium Simulation
- **Halo Mass Functions**: Sheth-Tormen theory
- **Power Spectrum**: Linear theory at high-z
- **Gravitational Lensing**: Weak lensing surveys

## Future Enhancements

### Planned Integrations

1. **OpenMM**: GPU-accelerated molecular dynamics
2. **HOOMD-blue**: Particle simulation framework
3. **ChaNGa**: Parallel N-body solver
4. **FLASH**: Astrophysical simulation code

### GPU Acceleration

```bash
# Enable CUDA support
cargo build --features "geant4,lammps,cuda"
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Distributed Computing

```bash
# MPI support for large-scale simulations
mpirun -np 64 cargo run --bin universectl -- start --ffi-mode --distributed
```

## References

- [Geant4 User Guide](https://geant4-userdoc.web.cern.ch/)
- [LAMMPS Manual](https://docs.lammps.org/)
- [GADGET Documentation](https://www.h-its.org/2018/02/22/gadget-code/)
- [ENDF/B-VIII.0 Release](https://www.nndc.bnl.gov/endf/b8.0/)
- [Rust FFI Guide](https://doc.rust-lang.org/nomicon/ffi.html) 