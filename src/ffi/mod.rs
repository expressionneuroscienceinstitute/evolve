//! # Foreign Function Interface (FFI) Module
//!
//! This module provides safe Rust wrappers around high-fidelity C/C++ scientific libraries.
//! Following the PDF recommendation to use proven open source implementations for maximum accuracy.

pub mod geant4;
pub mod lammps; 
pub mod gadget;
pub mod endf;

use anyhow::Result;

/// Initialize all FFI libraries
pub fn initialize_ffi_libraries() -> Result<()> {
    log::info!("Initializing FFI scientific libraries...");
    
    // Initialize libraries in order of dependency
    geant4::initialize()?;
    lammps::initialize()?;
    gadget::initialize()?;
    endf::initialize()?;
    
    log::info!("All FFI libraries initialized successfully");
    Ok(())
}

/// Cleanup all FFI libraries
pub fn cleanup_ffi_libraries() -> Result<()> {
    log::info!("Cleaning up FFI scientific libraries...");
    
    // Cleanup in reverse order
    endf::cleanup()?;
    gadget::cleanup()?;
    lammps::cleanup()?;
    geant4::cleanup()?;
    
    log::info!("All FFI libraries cleaned up successfully");
    Ok(())
}

/// Check which libraries are available at runtime
pub fn check_library_availability() -> LibraryStatus {
    LibraryStatus {
        geant4_available: geant4::is_available(),
        lammps_available: lammps::is_available(),
        gadget_available: gadget::is_available(),
        endf_available: endf::is_available(),
    }
}

/// Status of available scientific libraries
#[derive(Debug, Clone)]
pub struct LibraryStatus {
    pub geant4_available: bool,
    pub lammps_available: bool,
    pub gadget_available: bool,
    pub endf_available: bool,
}

impl LibraryStatus {
    /// Check if all critical libraries are available
    pub fn all_available(&self) -> bool {
        self.geant4_available && self.lammps_available && self.gadget_available && self.endf_available
    }
    
    /// Get human-readable status report
    pub fn status_report(&self) -> String {
        format!(
            "Scientific Library Status:\n\
             - Geant4 (Particle Physics): {}\n\
             - LAMMPS (Molecular Dynamics): {}\n\
             - GADGET (N-body Gravity): {}\n\
             - ENDF (Nuclear Data): {}",
            if self.geant4_available { "Available" } else { "Missing" },
            if self.lammps_available { "Available" } else { "Missing" },
            if self.gadget_available { "Available" } else { "Missing" },
            if self.endf_available { "Available" } else { "Missing" }
        )
    }
}

/// Global FFI configuration
pub struct FfiConfig {
    pub max_particles: usize,
    pub thread_count: usize,
    pub memory_limit_mb: usize,
    pub precision: FfiPrecision,
}

#[derive(Debug, Clone, Copy)]
pub enum FfiPrecision {
    Single,
    Double,
    Extended,
}

impl Default for FfiConfig {
    fn default() -> Self {
        Self {
            max_particles: 1_000_000,
            thread_count: num_cpus::get(),
            memory_limit_mb: 8192,
            precision: FfiPrecision::Double,
        }
    }
} 