//! # ENDF FFI Integration
//!
//! High-fidelity nuclear data using the ENDF/B-VIII.0 evaluated nuclear data library.
//! This provides access to experimental nuclear cross-sections and reaction data.

use anyhow::{Result, anyhow};
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::collections::HashMap;

// Include generated bindings when feature is enabled
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
    pub fn load_data_library(&mut self, library_path: &str) -> Result<()> {
        if !is_available() {
            return Err(anyhow!("ENDF library not available"));
        }
        
        let path_c = CString::new(library_path)?;
        
        unsafe {
            if endf_load_library(path_c.as_ptr()) != 0 {
                return Err(anyhow!("Failed to load ENDF library from: {}", library_path));
            }
            
            // Get available isotopes
            let n_isotopes = endf_get_n_isotopes();
            if n_isotopes > 0 {
                self.available_isotopes.resize(n_isotopes as usize, 0);
                endf_get_isotope_list(self.available_isotopes.as_mut_ptr());
            }
        }
        
        self.data_loaded = true;
        log::info!("Loaded ENDF data for {} isotopes", self.available_isotopes.len());
        Ok(())
    }
    
    /// Set temperature for thermal cross-sections
    pub fn set_temperature(&mut self, temperature_k: f64) -> Result<()> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            endf_set_temperature(temperature_k);
        }
        
        Ok(())
    }
    
    /// Get reaction cross-section
    pub fn get_cross_section(
        &self,
        target_za: u32,
        reaction_mt: u32,
        energy_ev: f64,
    ) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let cross_section = endf_get_cross_section(
                target_za as c_int,
                reaction_mt as c_int,
                energy_ev,
            );
            Ok(cross_section)
        }
    }
    
    /// Get Q-value for nuclear reaction
    pub fn get_q_value(&self, target_za: u32, reaction_mt: u32) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let q_value = endf_get_q_value(target_za as c_int, reaction_mt as c_int);
            Ok(q_value)
        }
    }
    
    /// Get half-life for radioactive isotope
    pub fn get_half_life(&self, isotope_za: u32) -> Result<f64> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let half_life = endf_get_half_life(isotope_za as c_int);
            Ok(half_life)
        }
    }
    
    /// Get thermal cross-sections at current temperature
    pub fn get_thermal_cross_sections(&self, isotope_za: u32) -> Result<ThermalCrossSections> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        // Common thermal energies (0.0253 eV for thermal neutrons)
        let thermal_energy = 0.0253;
        
        Ok(ThermalCrossSections {
            absorption: self.get_cross_section(isotope_za, ENDF_MT_NEUTRON_ABSORPTION as u32, thermal_energy)?,
            scattering: self.get_cross_section(isotope_za, ENDF_MT_ELASTIC_SCATTERING as u32, thermal_energy)?,
            fission: self.get_cross_section(isotope_za, ENDF_MT_FISSION as u32, thermal_energy).unwrap_or(0.0),
            capture: self.get_cross_section(isotope_za, ENDF_MT_NEUTRON_CAPTURE as u32, thermal_energy)?,
            total: self.get_cross_section(isotope_za, ENDF_MT_TOTAL as u32, thermal_energy)?,
        })
    }
    
    /// Get fission yields for fissile isotope
    pub fn get_fission_yields(
        &self,
        fissile_za: u32,
        neutron_energy_ev: f64,
    ) -> Result<HashMap<u32, f64>> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let n_products = endf_get_n_fission_products(fissile_za as c_int);
            
            if n_products > 0 {
                let mut product_za = vec![0u32; n_products as usize];
                let mut yields = vec![0.0f64; n_products as usize];
                
                endf_get_fission_yields(
                    fissile_za as c_int,
                    neutron_energy_ev,
                    product_za.as_mut_ptr(),
                    yields.as_mut_ptr(),
                );
                
                let mut yield_map = HashMap::new();
                for (za, yield_val) in product_za.into_iter().zip(yields.into_iter()) {
                    yield_map.insert(za, yield_val);
                }
                
                Ok(yield_map)
            } else {
                Ok(HashMap::new())
            }
        }
    }
    
    /// Get resonance parameters for resolved resonance region
    pub fn get_resonance_parameters(
        &self,
        isotope_za: u32,
        energy_min_ev: f64,
        energy_max_ev: f64,
    ) -> Result<Vec<ResonanceParameter>> {
        if !self.data_loaded {
            return Err(anyhow!("ENDF data not loaded"));
        }
        
        unsafe {
            let n_resonances = endf_get_n_resonances(
                isotope_za as c_int,
                energy_min_ev,
                energy_max_ev,
            );
            
            let mut resonances = Vec::new();
            
            for i in 0..n_resonances {
                let mut energy = 0.0;
                let mut gamma_n = 0.0;
                let mut gamma_gamma = 0.0;
                let mut gamma_f = 0.0;
                
                endf_get_resonance_data(
                    isotope_za as c_int,
                    i,
                    &mut energy,
                    &mut gamma_n,
                    &mut gamma_gamma,
                    &mut gamma_f,
                );
                
                resonances.push(ResonanceParameter {
                    energy_ev: energy,
                    neutron_width_ev: gamma_n,
                    gamma_width_ev: gamma_gamma,
                    fission_width_ev: gamma_f,
                    total_width_ev: gamma_n + gamma_gamma + gamma_f,
                });
            }
            
            Ok(resonances)
        }
    }
    
    /// Get available isotopes in library
    pub fn get_available_isotopes(&self) -> &[u32] {
        &self.available_isotopes
    }
    
    /// Check if isotope data is available
    pub fn has_isotope_data(&self, isotope_za: u32) -> bool {
        self.available_isotopes.contains(&isotope_za)
    }
}

/// Nuclear reaction types (MT numbers from ENDF)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReactionType {
    Total = 1,
    Elastic = 2,
    Nonelastic = 3,
    Inelastic = 4,
    Fission = 18,
    NeutronCapture = 102,
    ProtonCapture = 103,
    AlphaCapture = 107,
    // Add more as needed
}

/// Thermal neutron cross-sections
#[derive(Debug, Clone)]
pub struct ThermalCrossSections {
    pub absorption: f64,  // barns
    pub scattering: f64,  // barns
    pub fission: f64,     // barns
    pub capture: f64,     // barns
    pub total: f64,       // barns
}

/// Resonance parameter data
#[derive(Debug, Clone)]
pub struct ResonanceParameter {
    pub energy_ev: f64,
    pub neutron_width_ev: f64,
    pub gamma_width_ev: f64,
    pub fission_width_ev: f64,
    pub total_width_ev: f64,
}

// ENDF MT reaction numbers (commonly used)
const ENDF_MT_TOTAL: c_int = 1;
const ENDF_MT_ELASTIC_SCATTERING: c_int = 2;
const ENDF_MT_NONELASTIC: c_int = 3;
const ENDF_MT_FISSION: c_int = 18;
const ENDF_MT_NEUTRON_ABSORPTION: c_int = 27;
const ENDF_MT_NEUTRON_CAPTURE: c_int = 102;
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

// Conditional compilation for FFI functions and stubs
#[cfg(feature = "endf")]
extern "C" {
    fn endf_version() -> c_int;
    fn endf_init();
    fn endf_cleanup();
    fn endf_load_library(path: *const c_char) -> c_int;
    fn endf_get_n_isotopes() -> c_int;
    fn endf_get_isotope_list(isotopes: *mut u32);
    fn endf_set_temperature(temperature_k: c_double);
    fn endf_get_cross_section(target_za: c_int, reaction_mt: c_int, energy_ev: c_double) -> c_double;
    fn endf_get_q_value(target_za: c_int, reaction_mt: c_int) -> c_double;
    fn endf_get_half_life(isotope_za: c_int) -> c_double;
    fn endf_get_n_fission_products(fissile_za: c_int) -> c_int;
    fn endf_get_fission_yields(
        fissile_za: c_int, neutron_energy_ev: c_double, 
        product_za: *mut u32, yields: *mut c_double
    );
    fn endf_get_n_resonances(isotope_za: c_int, energy_min_ev: c_double, energy_max_ev: c_double) -> c_int;
    fn endf_get_resonance_data(
        isotope_za: c_int, resonance_index: c_int,
        energy_ev: *mut c_double, gamma_n: *mut c_double, 
        gamma_gamma: *mut c_double, gamma_f: *mut c_double
    );
}

// Stub implementations when ENDF is not available
#[cfg(not(feature = "endf"))]
unsafe fn endf_version() -> c_int { 0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_init() {}
#[cfg(not(feature = "endf"))]
unsafe fn endf_cleanup() {}
#[cfg(not(feature = "endf"))]
unsafe fn endf_load_library(_: *const c_char) -> c_int { -1 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_n_isotopes() -> c_int { 0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_isotope_list(_: *mut u32) {}
#[cfg(not(feature = "endf"))]
unsafe fn endf_set_temperature(_: c_double) {}
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_cross_section(_: c_int, _: c_int, _: c_double) -> c_double { 0.0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_q_value(_: c_int, _: c_int) -> c_double { 0.0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_half_life(_: c_int) -> c_double { 0.0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_n_fission_products(_: c_int) -> c_int { 0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_fission_yields(
    _: c_int, _: c_double, _: *mut u32, _: *mut c_double
) {}
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_n_resonances(_: c_int, _: c_double, _: c_double) -> c_int { 0 }
#[cfg(not(feature = "endf"))]
unsafe fn endf_get_resonance_data(
    _: c_int, _: c_int, _: *mut c_double, _: *mut c_double, _: *mut c_double, _: *mut c_double
) {} 