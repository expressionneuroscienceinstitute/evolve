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
#[cfg(feature = "gadget")]
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
#[cfg(feature = "gadget")]
pub use gadget::{GadgetEngine, GadgetParticle, GadgetParticleType, CosmologicalParameters, Halo};
pub use endf::{EndfEngine, ReactionType, ThermalCrossSections, ResonanceParameter};

// Compile-time fallback: Provide dummy C symbols so that linking succeeds on
// systems where the heavy native libraries are not present. These stubs are
// lightweight and have zero runtime cost when the real libraries are linked
// dynamically because the dynamic linker will prefer the real symbols.
mod stub_syms;

#[cfg(not(feature = "gadget"))]
mod gadget_stub {
    //! Minimal no-op stand-ins for the GADGET API so that downstream crates can
    //! compile even when the heavy `gadget` feature is disabled.  These stubs
    //! purposefully implement the *public* interface required by the physics
    //! engine but perform no work.  Every method either returns an error or a
    //! sensible default so that calling sites can still execute in low-fidelity
    //! mode.

    use anyhow::Result;
    use nalgebra::Vector3;

    /// Enumeration of particle categories used by the real GADGET codebase.
    /// The numerical discriminants intentionally match the real wrapper so that
    /// serialized data remains forward-compatible.
    #[derive(Debug, Clone, Copy)]
    pub enum GadgetParticleType {
        DarkMatter = 1,
        Stars = 2,
        Gas = 3,
        BlackHole = 4,
        Boundary = 5,
    }

    /// Cosmological integration parameters mirrored from the real interface.
    #[derive(Debug, Clone)]
    pub struct CosmologicalParameters {
        pub hubble_constant: f64,
        pub omega_matter: f64,
        pub omega_lambda: f64,
        pub omega_baryon: f64,
        pub scale_factor: f64,
        pub redshift: f64,
        pub age_of_universe: f64,
        pub enable_expansion: bool,
    }

    /// Simplified halo structure – enough for compile-time compatibility.
    #[derive(Debug, Clone)]
    pub struct Halo {
        pub id: usize,
        pub n_particles: usize,
        pub total_mass: f64,
        pub center_of_mass: Vector3<f64>,
        pub virial_radius: f64,
        pub velocity_dispersion: f64,
        pub particle_ids: Vec<usize>,
    }

    /// Particle state as expected by the upstream physics engine.
    #[derive(Debug, Clone)]
    pub struct GadgetParticle {
        pub id: usize,
        pub particle_type: GadgetParticleType,
        pub position: Vector3<f64>,
        pub velocity: Vector3<f64>,
        pub mass: f64,
        pub acceleration: Vector3<f64>,
        pub gravitational_potential: f64,
        pub softening_length: f64,
        pub time_step: f64,
        pub active: bool,
        pub density: f64,
    }

    /// Stubbed engine that fulfils the public contract while performing no work.
    #[derive(Debug, Default)]
    pub struct GadgetEngine;

    impl GadgetEngine {
        /// Construct a new stub – always succeeds.
        pub fn new() -> Result<Self> { Ok(Self) }

        /// Library availability query – returns `false` for the stub.
        pub fn is_available() -> bool { false }

        /// The following methods simply return `Ok(())` so that higher-level code
        /// can continue its control flow without feature-gated branches.
        pub fn clear_particles(&mut self) -> Result<()> { Ok(()) }
        pub fn add_particle(&mut self, _particle: GadgetParticle) -> Result<()> { Ok(()) }
        pub fn calculate_forces(&mut self) -> Result<()> { Ok(()) }
        pub fn integrate_step(&mut self, _dt: f64) -> Result<()> { Ok(()) }

        /// Returns an empty particle vector – no simulation performed.
        pub fn get_particle_data(&self) -> Result<Vec<GadgetParticle>> { Ok(Vec::new()) }
    }
}

#[cfg(not(feature = "gadget"))]
pub use gadget_stub::{
    GadgetEngine,
    GadgetParticle,
    GadgetParticleType,
    CosmologicalParameters,
    Halo,
}; 