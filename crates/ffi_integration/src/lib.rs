//! # FFI Integration Library
//!
//! High-fidelity scientific simulation using proven C/C++ libraries.
//! This crate provides safe Rust wrappers around:
//! - Geant4: Particle physics Monte Carlo simulation
//! - LAMMPS: Molecular dynamics simulation  
//! - GADGET: Cosmological N-body simulation
//! - ENDF: Nuclear data library

pub mod geant4;
pub mod lammps;
pub mod gadget;
pub mod endf;
pub mod mod_file;

#[cfg(feature = "demo")]
pub mod demo;

pub use mod_file::*;

use anyhow::Result;

/// Initialize all available FFI libraries
pub fn initialize_all_libraries() -> Result<()> {
    mod_file::initialize_ffi_libraries()
}

/// Cleanup all FFI libraries
pub fn cleanup_all_libraries() -> Result<()> {
    mod_file::cleanup_ffi_libraries()
}

/// Check status of all scientific libraries
pub fn check_library_status() -> mod_file::LibraryStatus {
    mod_file::check_library_availability()
}

/// Print status report of available libraries
pub fn print_status_report() {
    let status = check_library_status();
    println!("{}", status.status_report());
    
    if status.all_available() {
        log::info!("🚀 High-Fidelity Mode: ALL scientific libraries available!");
    } else {
        log::warn!("⚠️  Mixed Fidelity Mode: Some libraries missing, using fallbacks");
    }
}

/// Configuration for FFI operations
pub struct FfiConfiguration {
    pub max_particles: usize,
    pub memory_limit_gb: f64,
    pub thread_count: usize,
    pub enable_gpu: bool,
    pub precision: FfiPrecision,
}

impl Default for FfiConfiguration {
    fn default() -> Self {
        Self {
            max_particles: 1_000_000,
            memory_limit_gb: 16.0,
            thread_count: num_cpus::get(),
            enable_gpu: false,
            precision: FfiPrecision::Double,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FfiPrecision {
    Single,
    Double,
    Extended,
}

// Re-export commonly used types
pub use geant4::{Geant4Engine};
pub use lammps::{LammpsEngine, ForceFieldType, MolecularState, ThermodynamicState};
pub use gadget::{GadgetEngine, GadgetParticle, CosmologicalParameters, Halo};
pub use endf::{EndfEngine, ReactionType, ThermalCrossSections, ResonanceParameter}; 