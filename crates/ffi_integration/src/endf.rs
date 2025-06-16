//! # ENDF FFI Integration
//!
//! High-fidelity nuclear data using the ENDF/B-VIII.0 evaluated nuclear data library.
//! This provides access to experimental nuclear cross-sections and reaction data.

use anyhow::{Result, anyhow};
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::collections::HashMap;

// Include generated bindings
#[cfg(feature = "endf")]
include!(concat!(env!("OUT_DIR"), "/endf_bindings.rs"));

/// Safe Rust wrapper around ENDF functionality
pub struct EndfEngine {
    data_loaded: bool,
    available_isotopes: Vec<u32>,
    temperature_points: Vec<f64>,
}

unsafe impl Send for EndfEngine {}
unsafe impl Sync for EndfEngine {}

impl EndfEngine {
    /// Create new ENDF engine
    pub fn new() -> Result<Self> {
        if !is_available() {
            return Err(anyhow!("ENDF library not available"));
        }
        
        Ok(Self {
            data_loaded: false,
            available_isotopes: Vec::new(),
            temperature_points: Vec::new(),
        })
    }
    
    /// Load ENDF nuclear data library
    pub fn load_nuclear_data(&mut self, library_path: &str) -> Result<()> {
        let path_c = CString::new(library_path)?;
        
        unsafe {
            let result = endf_load_library(path_c.as_ptr());
            if result != 0 {
                return Err(anyhow!("Failed to load ENDF library: {}", result));
            }
            
            // Get available isotopes
            let n_isotopes = endf_get_n_isotopes();
            let mut isotopes = vec![0u32; n_isotopes as usize];
            endf_get_isotope_list(isotopes.as_mut_ptr());
            self.available_isotopes = isotopes;
            
            // Set up temperature grid
            self.temperature_points = vec![
                300.0, 600.0, 900.0, 1200.0, 1500.0, 2100.0, 2400.0
            ]; // Standard ENDF temperature points in Kelvin
            
            for &temp in &self.temperature_points {
                endf_set_temperature(temp);
            }
        }
        
        self.data_loaded = true;
        Ok(())
    }
    
    /// Get neutron absorption cross-section for isotope at given energy
    pub fn get_absorption_cross_section(
        &self,
        isotope: u32,
        energy_ev: f64,
        temperature_k: f64,
    ) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            // Set temperature
            endf_set_temperature(temperature_k);
            
            // Get cross-section
            let xs = endf_get_cross_section(
                isotope as c_int,
                ENDF_MT_ABSORPTION,
                energy_ev,
            );
            
            if xs < 0.0 {
                return Err(anyhow!("Invalid cross-section data for isotope {}", isotope));
            }
            
            Ok(xs) // Cross-section in barns
        }
    }
    
    /// Get fission cross-section for fissile isotope
    pub fn get_fission_cross_section(
        &self,
        isotope: u32,
        energy_ev: f64,
        temperature_k: f64,
    ) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            endf_set_temperature(temperature_k);
            
            let xs = endf_get_cross_section(
                isotope as c_int,
                ENDF_MT_FISSION,
                energy_ev,
            );
            
            Ok(xs.max(0.0)) // Return 0 for non-fissile isotopes
        }
    }
    
    /// Get elastic scattering cross-section
    pub fn get_elastic_cross_section(
        &self,
        isotope: u32,
        energy_ev: f64,
        temperature_k: f64,
    ) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            endf_set_temperature(temperature_k);
            
            let xs = endf_get_cross_section(
                isotope as c_int,
                ENDF_MT_ELASTIC,
                energy_ev,
            );
            
            Ok(xs.max(0.0))
        }
    }
    
    /// Get Q-value for nuclear reaction
    pub fn get_reaction_q_value(
        &self,
        isotope: u32,
        reaction_type: ReactionType,
    ) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        let mt = match reaction_type {
            ReactionType::Absorption => ENDF_MT_ABSORPTION,
            ReactionType::Fission => ENDF_MT_FISSION,
            ReactionType::Elastic => ENDF_MT_ELASTIC,
            ReactionType::Inelastic => ENDF_MT_INELASTIC,
            ReactionType::AlphaCapturs => ENDF_MT_ALPHA_CAPTURE,
            ReactionType::ProtonCapture => ENDF_MT_PROTON_CAPTURE,
        };
        
        unsafe {
            let q_value = endf_get_q_value(isotope as c_int, mt);
            Ok(q_value) // Q-value in MeV
        }
    }
    
    /// Get decay constant for radioactive isotope
    pub fn get_decay_constant(&self, isotope: u32) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let half_life = endf_get_half_life(isotope as c_int);
            if half_life <= 0.0 {
                return Ok(0.0); // Stable isotope
            }
            
            // Convert half-life to decay constant: λ = ln(2) / t_half
            Ok(std::f64::consts::LN_2 / half_life)
        }
    }
    
    /// Get fission yield data for fissile isotope
    pub fn get_fission_yields(
        &self,
        parent_isotope: u32,
        neutron_energy_ev: f64,
    ) -> Result<HashMap<u32, f64>> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let n_products = endf_get_n_fission_products(parent_isotope as c_int);
            if n_products <= 0 {
                return Ok(HashMap::new()); // No fission products
            }
            
            let mut products = vec![0u32; n_products as usize];
            let mut yields = vec![0.0f64; n_products as usize];
            
            endf_get_fission_yields(
                parent_isotope as c_int,
                neutron_energy_ev,
                products.as_mut_ptr(),
                yields.as_mut_ptr(),
            );
            
            let mut yield_map = HashMap::new();
            for (i, &product) in products.iter().enumerate() {
                yield_map.insert(product, yields[i]);
            }
            
            Ok(yield_map)
        }
    }
    
    /// Calculate resonance parameters for isotope
    pub fn get_resonance_parameters(
        &self,
        isotope: u32,
        energy_range: (f64, f64),
    ) -> Result<Vec<ResonanceParameter>> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let n_resonances = endf_get_n_resonances(
                isotope as c_int,
                energy_range.0,
                energy_range.1,
            );
            
            if n_resonances <= 0 {
                return Ok(Vec::new());
            }
            
            let mut resonances = Vec::new();
            
            for i in 0..n_resonances {
                let mut energy = 0.0;
                let mut gamma_n = 0.0;
                let mut gamma_gamma = 0.0;
                let mut gamma_f = 0.0;
                
                endf_get_resonance_data(
                    isotope as c_int,
                    i,
                    &mut energy,
                    &mut gamma_n,
                    &mut gamma_gamma,
                    &mut gamma_f,
                );
                
                resonances.push(ResonanceParameter {
                    energy_ev: energy,
                    neutron_width: gamma_n,
                    gamma_width: gamma_gamma,
                    fission_width: gamma_f,
                    total_width: gamma_n + gamma_gamma + gamma_f,
                });
            }
            
            Ok(resonances)
        }
    }
    
    /// Calculate thermal cross-sections at given temperature
    pub fn get_thermal_cross_sections(
        &self,
        isotope: u32,
        temperature_k: f64,
    ) -> Result<ThermalCrossSections> {
        let thermal_energy = BOLTZMANN_EV * temperature_k; // kT in eV
        
        Ok(ThermalCrossSections {
            absorption: self.get_absorption_cross_section(isotope, thermal_energy, temperature_k)?,
            fission: self.get_fission_cross_section(isotope, thermal_energy, temperature_k)?,
            elastic: self.get_elastic_cross_section(isotope, thermal_energy, temperature_k)?,
            temperature: temperature_k,
        })
    }
    
    /// Get list of available isotopes
    pub fn get_available_isotopes(&self) -> &[u32] {
        &self.available_isotopes
    }
    
    /// Check if isotope is fissile
    pub fn is_fissile(&self, isotope: u32) -> bool {
        if !self.data_loaded {
            return false;
        }
        
        // Check if fission cross-section exists for thermal neutrons
        self.get_fission_cross_section(isotope, 0.025, 300.0)
            .map(|xs| xs > 0.0)
            .unwrap_or(false)
    }
}

/// Nuclear reaction types
#[derive(Debug, Clone, Copy)]
pub enum ReactionType {
    Absorption,
    Fission,
    Elastic,
    Inelastic,
    AlphaCapturs,
    ProtonCapture,
}

/// Resonance parameter data
#[derive(Debug, Clone)]
pub struct ResonanceParameter {
    pub energy_ev: f64,
    pub neutron_width: f64,
    pub gamma_width: f64,
    pub fission_width: f64,
    pub total_width: f64,
}

/// Thermal cross-sections at specific temperature
#[derive(Debug, Clone)]
pub struct ThermalCrossSections {
    pub absorption: f64,
    pub fission: f64,
    pub elastic: f64,
    pub temperature: f64,
}

// ENDF MT (Material-Temperature) numbers for reaction types
const ENDF_MT_ELASTIC: c_int = 2;
const ENDF_MT_INELASTIC: c_int = 4;
const ENDF_MT_ABSORPTION: c_int = 27;
const ENDF_MT_FISSION: c_int = 18;
const ENDF_MT_ALPHA_CAPTURE: c_int = 107;
const ENDF_MT_PROTON_CAPTURE: c_int = 103;

// Physical constants
const BOLTZMANN_EV: f64 = 8.617333e-5; // Boltzmann constant in eV/K

/// Check if ENDF library is available
pub fn is_available() -> bool {
    #[cfg(feature = "endf")]
    {
        unsafe { endf_version() > 0 }
    }
    #[cfg(not(feature = "endf"))]
    {
        false
    }
}

/// Initialize ENDF library
pub fn initialize() -> Result<()> {
    if is_available() {
        unsafe {
            endf_init();
        }
        log::info!("ENDF library initialized successfully");
        Ok(())
    } else {
        log::warn!("ENDF library not available - using fallback implementation");
        Ok(())
    }
}

/// Cleanup ENDF library
pub fn cleanup() -> Result<()> {
    if is_available() {
        unsafe {
            endf_cleanup();
        }
    }
    log::info!("ENDF library cleaned up");
    Ok(())
}

// Stub implementations when ENDF is not available
#[cfg(not(feature = "endf"))]
mod stubs {
    use super::*;
    
    pub unsafe extern "C" fn endf_version() -> c_int { 0 }
    pub unsafe extern "C" fn endf_init() {}
    pub unsafe extern "C" fn endf_cleanup() {}
    pub unsafe extern "C" fn endf_load_library(_: *const c_char) -> c_int { -1 }
    pub unsafe extern "C" fn endf_get_n_isotopes() -> c_int { 0 }
    pub unsafe extern "C" fn endf_get_isotope_list(_: *mut u32) {}
    pub unsafe extern "C" fn endf_set_temperature(_: c_double) {}
    pub unsafe extern "C" fn endf_get_cross_section(_: c_int, _: c_int, _: c_double) -> c_double { 0.0 }
    pub unsafe extern "C" fn endf_get_q_value(_: c_int, _: c_int) -> c_double { 0.0 }
    pub unsafe extern "C" fn endf_get_half_life(_: c_int) -> c_double { 0.0 }
    pub unsafe extern "C" fn endf_get_n_fission_products(_: c_int) -> c_int { 0 }
    pub unsafe extern "C" fn endf_get_fission_yields(
        _: c_int, _: c_double, _: *mut u32, _: *mut c_double
    ) {}
    pub unsafe extern "C" fn endf_get_n_resonances(_: c_int, _: c_double, _: c_double) -> c_int { 0 }
    pub unsafe extern "C" fn endf_get_resonance_data(
        _: c_int, _: c_int, _: *mut c_double, _: *mut c_double, _: *mut c_double, _: *mut c_double
    ) {}
}

#[cfg(not(feature = "endf"))]
use stubs::*; 