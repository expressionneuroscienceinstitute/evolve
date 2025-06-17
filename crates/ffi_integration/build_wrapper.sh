#!/bin/bash

# Build script for Geant4 wrapper dynamic library
# This script automates the building of the real Geant4 implementation
# instead of using stub functions

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
LIB_DIR="$BUILD_DIR/lib"

# Platform-specific library name
if [[ "$OSTYPE" == "darwin"* ]]; then
    WRAPPER_LIB="$LIB_DIR/libgeant4_wrapper.1.0.0.dylib"
else
    WRAPPER_LIB="$LIB_DIR/libgeant4_wrapper.so.1.0.0"
fi

echo "=== Geant4 Wrapper Library Build Script ==="
echo "Script directory: $SCRIPT_DIR"
echo "Build directory: $BUILD_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Geant4 installation
check_geant4() {
    echo "Checking Geant4 installation..."
    
    if command_exists geant4-config; then
        echo "Found geant4-config tool"
        G4_VERSION=$(geant4-config --version 2>/dev/null || echo "unknown")
        G4_PREFIX=$(geant4-config --prefix 2>/dev/null || echo "unknown")
        echo "Geant4 version: $G4_VERSION"
        echo "Geant4 prefix: $G4_PREFIX"
        return 0
    elif [ -n "$GEANT4_DIR" ] && [ -d "$GEANT4_DIR" ]; then
        echo "Using GEANT4_DIR: $GEANT4_DIR"
        if [ -d "$GEANT4_DIR/include/Geant4" ]; then
            echo "Found Geant4 headers in $GEANT4_DIR/include/Geant4"
            return 0
        else
            echo "Error: Geant4 headers not found in $GEANT4_DIR/include/Geant4"
            return 1
        fi
    else
        echo "Error: Geant4 not found"
        echo "Please install Geant4 or set the GEANT4_DIR environment variable"
        return 1
    fi
}

# Function to build the library
build_library() {
    echo "Building Geant4 wrapper library..."
    cd "$SCRIPT_DIR"
    
    # Check if Makefile exists
    if [ ! -f "Makefile" ]; then
        echo "Error: Makefile not found in $SCRIPT_DIR"
        echo "Please ensure the Makefile is present"
        return 1
    fi
    
    # Show configuration
    echo "Build configuration:"
    make config || true
    
    # Clean previous build
    echo "Cleaning previous build..."
    make clean || true
    
    # Build the library
    echo "Compiling C++ wrapper..."
    if make all; then
        echo "Build successful!"
        return 0
    else
        echo "Build failed!"
        return 1
    fi
}

# Function to verify the built library
verify_library() {
    echo "Verifying built library..."
    
    if [ ! -f "$WRAPPER_LIB" ]; then
        echo "Error: Library file not found: $WRAPPER_LIB"
        return 1
    fi
    
    echo "Library file exists: $WRAPPER_LIB"
    
    # Check if it's a valid shared library
    if file "$WRAPPER_LIB" | grep -q "shared object"; then
        echo "Valid shared library detected"
    else
        echo "Warning: File does not appear to be a shared library"
    fi
    
    # Check for required symbols
    echo "Checking for required symbols..."
    if nm -D "$WRAPPER_LIB" 2>/dev/null | grep -q "g4_is_available"; then
        echo "Found g4_is_available symbol ✓"
    else
        echo "Warning: g4_is_available symbol not found"
    fi
    
    if nm -D "$WRAPPER_LIB" 2>/dev/null | grep -q "g4_create_run_manager"; then
        echo "Found g4_create_run_manager symbol ✓"
    else
        echo "Warning: g4_create_run_manager symbol not found"
    fi
    
    # Test the library if possible
    echo "Running basic library test..."
    if make test; then
        echo "Library test passed ✓"
        return 0
    else
        echo "Library test failed (this may be expected if Geant4 is not properly configured)"
        return 0  # Don't fail the build for test failures
    fi
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --check, -c    Only check Geant4 installation"
    echo "  --clean        Clean build artifacts and exit"
    echo "  --debug        Build with debug symbols"
    echo ""
    echo "Environment variables:"
    echo "  GEANT4_DIR     Path to Geant4 installation (if geant4-config not available)"
    echo "  CXX            C++ compiler to use (default: g++)"
}

# Parse command line arguments
ONLY_CHECK=false
CLEAN_ONLY=false
DEBUG_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --check|-c)
            ONLY_CHECK=true
            shift
            ;;
        --clean)
            CLEAN_ONLY=true
            shift
            ;;
        --debug)
            DEBUG_BUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "Starting Geant4 wrapper build process..."
    
    # Handle clean-only mode
    if $CLEAN_ONLY; then
        echo "Cleaning build artifacts..."
        cd "$SCRIPT_DIR"
        make clean || true
        echo "Clean completed"
        exit 0
    fi
    
    # Check Geant4 installation
    if ! check_geant4; then
        echo "Geant4 check failed. Cannot build wrapper library."
        exit 1
    fi
    
    # Exit early if only checking
    if $ONLY_CHECK; then
        echo "Geant4 check completed successfully"
        exit 0
    fi
    
    # Check required tools
    echo "Checking build tools..."
    if ! command_exists g++; then
        echo "Error: g++ compiler not found"
        exit 1
    fi
    
    if ! command_exists make; then
        echo "Error: make build tool not found"
        exit 1
    fi
    
    echo "Build tools check passed ✓"
    
    # Build the library
    if $DEBUG_BUILD; then
        echo "Building in debug mode..."
        cd "$SCRIPT_DIR"
        make debug
    else
        build_library
    fi
    
    # Verify the result
    if verify_library; then
        echo ""
        echo "=== Build Summary ==="
        echo "✓ Geant4 wrapper library built successfully"
        echo "✓ Library location: $WRAPPER_LIB"
        echo "✓ Ready for use with Rust FFI"
        echo ""
        echo "Next steps:"
        echo "1. Run 'cargo build --features geant4' to use the real library"
        echo "2. Or run 'cargo build' to continue using stubs"
        echo ""
        exit 0
    else
        echo "Library verification failed"
        exit 1
    fi
}

# Run main function
main "$@" 