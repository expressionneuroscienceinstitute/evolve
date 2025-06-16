#!/bin/bash
# Setup script for high-fidelity scientific libraries FFI integration
# This script installs Geant4, LAMMPS, GADGET, and ENDF libraries for maximum accuracy

set -e  # Exit on error

# Configuration
INSTALL_PREFIX="/usr/local"
BUILD_DIR="/tmp/evolve_ffi_build"
THREADS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root for system installation
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root - installing system-wide"
        SUDO=""
    else
        log_info "Running as user - will use sudo for system installation"
        SUDO="sudo"
    fi
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        $SUDO apt-get update
        $SUDO apt-get install -y \
            build-essential \
            cmake \
            git \
            wget \
            curl \
            python3 \
            python3-dev \
            python3-pip \
            libopenmpi-dev \
            openmpi-bin \
            libfftw3-dev \
            libeigen3-dev \
            libhdf5-dev \
            libboost-all-dev \
            libblas-dev \
            liblapack-dev \
            gfortran \
            clang-dev \
            llvm-dev \
            pkg-config
    elif command -v yum &> /dev/null; then
        $SUDO yum groupinstall -y "Development Tools"
        $SUDO yum install -y \
            cmake \
            git \
            wget \
            python3-devel \
            openmpi-devel \
            fftw3-devel \
            eigen3-devel \
            hdf5-devel \
            boost-devel \
            blas-devel \
            lapack-devel \
            gcc-gfortran \
            clang-devel \
            llvm-devel \
            pkgconfig
    elif command -v brew &> /dev/null; then
        brew install \
            cmake \
            git \
            wget \
            python3 \
            open-mpi \
            fftw \
            eigen \
            hdf5 \
            boost \
            openblas \
            lapack \
            gfortran \
            llvm \
            pkg-config
    else
        log_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
    
    log_success "System dependencies installed"
}

# Setup build environment
setup_build_env() {
    log_info "Setting up build environment..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Set environment variables
    export CC=gcc
    export CXX=g++
    export FC=gfortran
    export CMAKE_BUILD_TYPE=Release
    export OMP_NUM_THREADS=$THREADS
    
    log_success "Build environment ready"
}

# Install Geant4
install_geant4() {
    log_info "Installing Geant4..."
    
    cd "$BUILD_DIR"
    
    # Download Geant4
    if [ ! -f "geant4-v11.2.0.tar.gz" ]; then
        log_info "Downloading Geant4..."
        wget https://geant4-data.web.cern.ch/releases/geant4-v11.2.0.tar.gz
    fi
    
    # Extract and build
    tar -xzf geant4-v11.2.0.tar.gz
    cd geant4-v11.2.0
    mkdir -p build && cd build
    
    log_info "Configuring Geant4..."
    cmake \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX/geant4" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGEANT4_INSTALL_DATA=ON \
        -DGEANT4_BUILD_MULTITHREADED=ON \
        -DGEANT4_USE_GDML=ON \
        -DGEANT4_USE_QT=OFF \
        -DGEANT4_USE_OPENGL_X11=OFF \
        -DGEANT4_BUILD_CXXSTD=17 \
        -DGEANT4_USE_SYSTEM_EXPAT=ON \
        ..
    
    log_info "Building Geant4 (this may take a while)..."
    make -j$THREADS
    
    log_info "Installing Geant4..."
    $SUDO make install
    
    # Set environment variables
    echo "export GEANT4_DIR=$INSTALL_PREFIX/geant4" | $SUDO tee -a /etc/environment
    export GEANT4_DIR="$INSTALL_PREFIX/geant4"
    
    log_success "Geant4 installed successfully"
}

# Install LAMMPS
install_lammps() {
    log_info "Installing LAMMPS..."
    
    cd "$BUILD_DIR"
    
    # Clone LAMMPS
    if [ ! -d "lammps" ]; then
        log_info "Cloning LAMMPS..."
        git clone -b stable --depth 1 https://github.com/lammps/lammps.git
    fi
    
    cd lammps
    mkdir -p build && cd build
    
    log_info "Configuring LAMMPS..."
    cmake \
        -C ../cmake/presets/basic.cmake \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX/lammps" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DLAMMPS_EXCEPTIONS=ON \
        -DPKG_MOLECULE=ON \
        -DPKG_MANYBODY=ON \
        -DPKG_KSPACE=ON \
        -DPKG_RIGID=ON \
        -DPKG_MISC=ON \
        -DPKG_EXTRA-MOLECULE=ON \
        -DPKG_PYTHON=ON \
        -DWITH_GZIP=ON \
        -DWITH_FFMPEG=OFF \
        ../cmake
    
    log_info "Building LAMMPS..."
    make -j$THREADS
    
    log_info "Installing LAMMPS..."
    $SUDO make install
    
    # Set environment variables
    echo "export LAMMPS_DIR=$INSTALL_PREFIX/lammps" | $SUDO tee -a /etc/environment
    export LAMMPS_DIR="$INSTALL_PREFIX/lammps"
    
    log_success "LAMMPS installed successfully"
}

# Install GADGET (Note: Requires manual download due to licensing)
install_gadget() {
    log_info "Setting up GADGET..."
    
    GADGET_URL="https://www.h-its.org/2018/02/22/gadget-code/"
    
    log_warning "GADGET requires manual download due to licensing requirements"
    log_info "Please:"
    log_info "1. Visit: $GADGET_URL"
    log_info "2. Register and download GADGET-4"
    log_info "3. Place the tarball in $BUILD_DIR/gadget.tar.gz"
    log_info "4. Run this script again"
    
    if [ -f "$BUILD_DIR/gadget.tar.gz" ]; then
        log_info "Found GADGET tarball, proceeding with installation..."
        
        cd "$BUILD_DIR"
        tar -xzf gadget.tar.gz
        
        # Find extracted directory
        GADGET_DIR=$(find . -maxdepth 1 -type d -name "*gadget*" | head -1)
        if [ -z "$GADGET_DIR" ]; then
            log_error "Could not find GADGET directory after extraction"
            return 1
        fi
        
        cd "$GADGET_DIR"
        
        # Create FFI-compatible Makefile
        if [ -f "Makefile.template" ]; then
            cp Makefile.template Makefile
            
            # Configure for FFI usage
            sed -i 's/^SYSTYPE=.*/SYSTYPE="Generic-gcc"/' Makefile
            sed -i 's/^#DOUBLEPRECISION/DOUBLEPRECISION/' Makefile
            echo "GADGET_FFI" >> Makefile
            
            # Build library version
            make -j$THREADS
            
            # Install manually
            $SUDO mkdir -p "$INSTALL_PREFIX/gadget"
            $SUDO cp -r . "$INSTALL_PREFIX/gadget/"
            
            # Set environment variables
            echo "export GADGET_SRC=$INSTALL_PREFIX/gadget" | $SUDO tee -a /etc/environment
            export GADGET_SRC="$INSTALL_PREFIX/gadget"
            
            log_success "GADGET installed successfully"
        else
            log_error "Could not find Makefile.template in GADGET directory"
            return 1
        fi
    else
        log_warning "GADGET tarball not found - skipping installation"
        return 1
    fi
}

# Install ENDF data libraries
install_endf() {
    log_info "Installing ENDF data libraries..."
    
    cd "$BUILD_DIR"
    
    # Download ENDF/B-VIII.0 data
    if [ ! -f "ENDF-B-VIII.0_neutrons.tar.gz" ]; then
        log_info "Downloading ENDF/B-VIII.0 neutron data..."
        wget https://www.nndc.bnl.gov/endf/b8.0/download/ENDF-B-VIII.0_neutrons.tar.gz
    fi
    
    # Install data
    $SUDO mkdir -p "$INSTALL_PREFIX/endf/data"
    cd "$INSTALL_PREFIX/endf/data"
    $SUDO tar -xzf "$BUILD_DIR/ENDF-B-VIII.0_neutrons.tar.gz"
    
    # Clone and build ENDF parser
    cd "$BUILD_DIR"
    if [ ! -d "endf-parser" ]; then
        log_info "Cloning ENDF parser..."
        git clone https://github.com/nuclearkatie/endf-parser.git
    fi
    
    cd endf-parser
    mkdir -p build && cd build
    
    log_info "Building ENDF parser..."
    cmake \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX/endf" \
        -DCMAKE_BUILD_TYPE=Release \
        ..
    
    make -j$THREADS
    $SUDO make install
    
    # Set environment variables
    echo "export ENDF_LIB_DIR=$INSTALL_PREFIX/endf" | $SUDO tee -a /etc/environment
    export ENDF_LIB_DIR="$INSTALL_PREFIX/endf"
    
    log_success "ENDF libraries installed successfully"
}

# Create wrapper libraries for FFI
create_ffi_wrappers() {
    log_info "Creating FFI wrapper libraries..."
    
    # This would create C wrapper libraries that expose
    # simple C interfaces for the C++ libraries
    # For now, we'll just create placeholder directories
    
    $SUDO mkdir -p "$INSTALL_PREFIX/evolve_ffi"
    
    # Future: Compile actual wrapper libraries here
    log_warning "FFI wrapper creation not yet implemented"
    log_info "Will use direct library linking for now"
}

# Update library paths
update_library_paths() {
    log_info "Updating library paths..."
    
    # Update LD_LIBRARY_PATH
    LIBRARY_PATHS="$INSTALL_PREFIX/geant4/lib64:$INSTALL_PREFIX/geant4/lib:$INSTALL_PREFIX/lammps/lib:$INSTALL_PREFIX/endf/lib"
    
    echo "export LD_LIBRARY_PATH=$LIBRARY_PATHS:\$LD_LIBRARY_PATH" | $SUDO tee -a /etc/environment
    
    # Update ldconfig
    if command -v ldconfig &> /dev/null; then
        echo "$INSTALL_PREFIX/geant4/lib64" | $SUDO tee /etc/ld.so.conf.d/geant4.conf
        echo "$INSTALL_PREFIX/geant4/lib" | $SUDO tee -a /etc/ld.so.conf.d/geant4.conf
        echo "$INSTALL_PREFIX/lammps/lib" | $SUDO tee /etc/ld.so.conf.d/lammps.conf
        echo "$INSTALL_PREFIX/endf/lib" | $SUDO tee /etc/ld.so.conf.d/endf.conf
        $SUDO ldconfig
    fi
    
    log_success "Library paths updated"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Geant4
    if [ -d "$INSTALL_PREFIX/geant4" ]; then
        log_success "Geant4: Found"
    else
        log_error "Geant4: Missing"
    fi
    
    # Check LAMMPS
    if [ -d "$INSTALL_PREFIX/lammps" ]; then
        log_success "LAMMPS: Found"
    else
        log_error "LAMMPS: Missing"
    fi
    
    # Check GADGET
    if [ -d "$INSTALL_PREFIX/gadget" ]; then
        log_success "GADGET: Found"
    else
        log_warning "GADGET: Missing (requires manual download)"
    fi
    
    # Check ENDF
    if [ -d "$INSTALL_PREFIX/endf" ]; then
        log_success "ENDF: Found"
    else
        log_error "ENDF: Missing"
    fi
}

# Generate build instructions for EVOLVE
generate_build_instructions() {
    log_info "Generating EVOLVE build instructions..."
    
    cat > "$BUILD_DIR/evolve_build_instructions.txt" << EOF
# EVOLVE High-Fidelity Build Instructions

To build EVOLVE with maximum scientific accuracy using the installed libraries:

## Environment Setup
source /etc/environment
export GEANT4_DIR=$INSTALL_PREFIX/geant4
export LAMMPS_DIR=$INSTALL_PREFIX/lammps
export GADGET_SRC=$INSTALL_PREFIX/gadget
export ENDF_LIB_DIR=$INSTALL_PREFIX/endf

## Build Commands

# Build with all high-fidelity libraries
cargo build --release --features "geant4,lammps,gadget,endf"

# Build with subset (if some libraries unavailable)
cargo build --release --features "geant4,lammps"

## Runtime Configuration

# Check library availability
cargo run --bin universectl -- check-ffi

# Run with maximum fidelity
cargo run --bin universectl -- start --ffi-mode --max-fidelity

## Performance Optimization
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CARGO_PROFILE_RELEASE_LTO=true
export OMP_NUM_THREADS=$(nproc)

## Troubleshooting

If you encounter library loading issues:
1. Check LD_LIBRARY_PATH: echo \$LD_LIBRARY_PATH
2. Verify library installation: ldd target/release/universectl
3. Run ldconfig: sudo ldconfig
4. Check environment: env | grep -E "(GEANT4|LAMMPS|GADGET|ENDF)"
EOF

    log_success "Build instructions saved to $BUILD_DIR/evolve_build_instructions.txt"
}

# Cleanup
cleanup() {
    log_info "Cleaning up build directory..."
    rm -rf "$BUILD_DIR"
    log_success "Cleanup complete"
}

# Main execution
main() {
    log_info "Starting EVOLVE FFI library installation..."
    log_info "This will install Geant4, LAMMPS, GADGET, and ENDF libraries"
    log_info "Installation prefix: $INSTALL_PREFIX"
    log_info "Build directory: $BUILD_DIR"
    log_info "Using $THREADS threads for compilation"
    
    check_root
    install_dependencies
    setup_build_env
    
    # Install libraries
    install_geant4
    install_lammps
    install_gadget || log_warning "GADGET installation skipped"
    install_endf
    
    # Finalize installation
    create_ffi_wrappers
    update_library_paths
    verify_installation
    generate_build_instructions
    
    log_success "FFI library installation complete!"
    log_info "Please source your environment: source /etc/environment"
    log_info "Then follow instructions in: $BUILD_DIR/evolve_build_instructions.txt"
    
    # Optional cleanup
    read -p "Remove build directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --prefix DIR     Installation prefix (default: /usr/local)"
            echo "  --threads N      Number of build threads (default: $(nproc))"
            echo "  --no-cleanup     Don't remove build directory"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main 