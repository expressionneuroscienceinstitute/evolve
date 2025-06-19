//! Minimal demo stub
//!
//! The original interactive Geant4 FFI demonstration relied on an outdated
//! particle data model.  In order to keep the public `demo` feature available
//! without breaking the main build we provide a *compile-time* placeholder that
//! prints a short notice when executed.

use anyhow::Result;

/// Entry point for the demo (does not run heavy physics).
#[allow(dead_code)]
pub fn run_geant4_demo() -> Result<()> {
    println!(
        "Geant4 FFI demo has been temporarily disabled while the particle data \n\
         model is being refactored.  Please refer to the updated README for \n\
         instructions on running the full demonstration with a compatible \n\
         binary release."
    );
    Ok(())
} 