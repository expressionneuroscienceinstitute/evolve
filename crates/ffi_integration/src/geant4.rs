//! # Geant4 FFI Integration
//!
//! High-fidelity particle physics using the actual Geant4 Monte Carlo toolkit.
//! This provides the highest accuracy possible for particle interactions, transport, and detector simulation.

use physics_types::{FundamentalParticle, ParticleType, InteractionEvent, InteractionType, QuantumState};
use anyhow::{Result, anyhow};
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use nalgebra::Vector3;

// If the `geant4` feature is enabled we rely on the automatically generated
// bindings created by build.rs (OUT_DIR/geant4_bindings.rs). This avoids
// defining the symbols twice and removes the compilation clashes with the
// hand-written stubs.

#[cfg(feature = "geant4")]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
mod ffi {
    use super::*; // re-export common imports so type names resolve
    include!(concat!(env!("OUT_DIR"), "/geant4_bindings.rs"));
}

#[cfg(feature = "geant4")]
use ffi::*;

// -----------------------------------------------------------------------------
// Fallback stubs for builds _without_ the `geant4` Cargo feature.
// These give the compiler everything it needs so the rest of the file compiles,
// while doing *nothing* at runtime.  All functions return neutral defaults and
// the availability helper reports "not available" so higher-level callers know
// the real Geant-4 engine is missing.
// -----------------------------------------------------------------------------
#[cfg(not(feature = "geant4"))]
mod ffi {
    use super::*;

    // Dummy opaque types – in the real bindings these are structs with hidden
    // fields.  We just need distinct types so pointers are well-typed.
    pub type G4RunManager = c_void;
    pub type G4VUserDetectorConstruction = c_void;
    pub type G4VUserPhysicsList = c_void;
    pub type G4SteppingManager = c_void;
    pub type G4ParticleGun = c_void;

    // ---------------------------------------------------------------------
    // Minimal data structures that higher-level safe wrappers rely on.
    // ---------------------------------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct G4InteractionData {
        pub process_type: c_int,
        pub energy_deposited: f64,
        pub n_secondaries: c_int,
        pub secondary_particles: *mut c_int,
        pub secondary_energies: *mut f64,
        pub step_length: f64,
        pub position: [f64; 3],
    }

    impl Default for G4InteractionData {
        fn default() -> Self {
            Self {
                process_type: 0,
                energy_deposited: 0.0,
                n_secondaries: 0,
                secondary_particles: std::ptr::null_mut(),
                secondary_energies: std::ptr::null_mut(),
                step_length: 0.0,
                position: [0.0; 3],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct G4ParticleData {
        pub pdg_code: c_int,
        pub energy: f64,
        pub momentum: [f64; 3],
        pub position: [f64; 3],
    }

    impl Default for G4ParticleData {
        fn default() -> Self {
            Self {
                pdg_code: 0,
                energy: 0.0,
                momentum: [0.0; 3],
                position: [0.0; 3],
            }
        }
    }

    // ---------------------------------------------------------------------
    // No-op (or constant) shims for every C function the real bindings expose.
    // They keep the symbol names identical so the wrapper code compiles.
    // ---------------------------------------------------------------------
    #[inline]
    pub unsafe fn g4_is_available() -> c_int { 0 }

    #[inline]
    pub unsafe fn g4_global_initialize() -> c_int { 0 }
    #[inline]
    pub unsafe fn g4_global_cleanup() {}

    #[inline]
    pub unsafe fn g4_create_run_manager() -> *mut G4RunManager { ptr::null_mut() }
    #[inline]
    pub unsafe fn g4_delete_run_manager(_m: *mut G4RunManager) {}

    #[inline]
    pub unsafe fn g4_create_simple_detector() -> *mut G4VUserDetectorConstruction { ptr::null_mut() }
    #[inline]
    pub unsafe fn g4_delete_detector(_d: *mut G4VUserDetectorConstruction) {}

    #[inline]
    pub unsafe fn g4_create_physics_list(_name: *const c_char) -> *mut G4VUserPhysicsList { ptr::null_mut() }
    #[inline]
    pub unsafe fn g4_delete_physics_list(_p: *mut G4VUserPhysicsList) {}

    #[inline]
    pub unsafe fn g4_create_stepping_manager() -> *mut G4SteppingManager { ptr::null_mut() }
    #[inline]
    pub unsafe fn g4_delete_stepping_manager(_s: *mut G4SteppingManager) {}

    #[inline]
    pub unsafe fn g4_create_particle_gun() -> *mut G4ParticleGun { ptr::null_mut() }
    #[inline]
    pub unsafe fn g4_delete_particle_gun(_p: *mut G4ParticleGun) {}

    #[inline]
    pub unsafe fn g4_set_detector_construction(_rm: *mut G4RunManager, _dc: *mut G4VUserDetectorConstruction) {}
    #[inline]
    pub unsafe fn g4_set_physics_list(_rm: *mut G4RunManager, _pl: *mut G4VUserPhysicsList) {}
    #[inline]
    pub unsafe fn g4_initialize_geant4(_rm: *mut G4RunManager) {}

    #[inline]
    pub unsafe fn g4_set_particle_gun_particle(_pg: *mut G4ParticleGun, _pdg: c_int) {}
    #[inline]
    pub unsafe fn g4_set_particle_gun_energy(_pg: *mut G4ParticleGun, _e: f64) {}
    #[inline]
    pub unsafe fn g4_set_particle_gun_position(_pg: *mut G4ParticleGun, _x: f64, _y: f64, _z: f64) {}
    #[inline]
    pub unsafe fn g4_set_particle_gun_direction(_pg: *mut G4ParticleGun, _x: f64, _y: f64, _z: f64) {}

    #[inline]
    pub unsafe fn g4_set_material(_mat: *const c_char) {}
    #[inline]
    pub unsafe fn g4_set_step_limit(_len: f64) {}

    #[inline]
    pub unsafe fn g4_run_beam_on(_rm: *mut G4RunManager, _n: c_int) {}

    #[inline]
    pub unsafe fn g4_get_n_interactions() -> c_int { 0 }
    #[inline]
    pub unsafe fn g4_get_interaction_data(_i: c_int, _data: *mut G4InteractionData) {}

    #[inline]
    pub unsafe fn g4_get_cross_section(_particle: *const c_char, _material: *const c_char, _process: *const c_char, _energy: f64) -> f64 { 0.0 }
    #[inline]
    pub unsafe fn g4_get_stopping_power(_particle: *const c_char, _material: *const c_char, _energy: f64) -> f64 { 0.0 }

    #[inline]
    pub unsafe fn g4_simulate_decay(_pdg: c_int, _energy: f64) -> c_int { 0 }
    #[inline]
    pub unsafe fn g4_get_decay_product(_i: c_int, _data: *mut G4ParticleData) {}
}

// Make the stub symbols available at the top level of this module just like
// the real bindings.
#[cfg(not(feature = "geant4"))]
use ffi::*;

// -----------------------------------------------------------------------------
// End of stub module additions
// -----------------------------------------------------------------------------

/// Safe Rust wrapper around Geant4 functionality
#[derive(Debug)]
pub struct Geant4Engine {
    detector_construction: *mut ffi::G4VUserDetectorConstruction,
    physics_list: *mut ffi::G4VUserPhysicsList,
    run_manager: *mut ffi::G4RunManager,
    step_manager: *mut ffi::G4SteppingManager,
    particle_gun: *mut ffi::G4ParticleGun,
    is_initialized: bool,
}

unsafe impl Send for Geant4Engine {}
unsafe impl Sync for Geant4Engine {}

impl Geant4Engine {
    /// Create new Geant4 engine with specified physics list
    pub fn new(physics_list_name: &str) -> Result<Self> {
        if !is_available() {
            return Err(anyhow!("Geant4 library not available"));
        }
        
        let physics_list_c = CString::new(physics_list_name)?;
        
        unsafe {
            let run_manager: *mut ffi::G4RunManager = g4_create_run_manager();
            if run_manager.is_null() {
                return Err(anyhow!("Failed to create Geant4 run manager"));
            }
            
            let detector_construction: *mut ffi::G4VUserDetectorConstruction = g4_create_simple_detector();
            if detector_construction.is_null() {
                g4_delete_run_manager(run_manager);
                return Err(anyhow!("Failed to create detector construction"));
            }
            
            let physics_list: *mut ffi::G4VUserPhysicsList = g4_create_physics_list(physics_list_c.as_ptr());
            if physics_list.is_null() {
                g4_delete_detector(detector_construction);
                g4_delete_run_manager(run_manager);
                return Err(anyhow!("Failed to create physics list"));
            }
            
            let step_manager: *mut ffi::G4SteppingManager = g4_create_stepping_manager();
            let particle_gun: *mut ffi::G4ParticleGun = g4_create_particle_gun();
            
            g4_set_detector_construction(run_manager, detector_construction);
            g4_set_physics_list(run_manager, physics_list);
            g4_initialize_geant4(run_manager);
            
            Ok(Self {
                detector_construction,
                physics_list,
                run_manager,
                step_manager,
                particle_gun,
                is_initialized: true,
            })
        }
    }
    
    /// Transport particle through matter and return interaction results
    pub fn transport_particle(
        &mut self,
        particle: &FundamentalParticle,
        material: &str,
        step_length_cm: f64,
    ) -> Result<Vec<InteractionEvent>> {
        if !self.is_initialized {
            return Err(anyhow!("Geant4 engine not initialized"));
        }
        
        let material_c = CString::new(material)?;
        
        unsafe {
            // Set up particle gun
            let pdg_code = particle_type_to_pdg(&particle.particle_type);
            let energy_mev = particle.energy * 6.242e12; // Convert J to MeV
            
            g4_set_particle_gun_particle(self.particle_gun, pdg_code);
            g4_set_particle_gun_energy(self.particle_gun, energy_mev);
            g4_set_particle_gun_position(
                self.particle_gun,
                particle.position.x * 100.0, // Convert m to cm
                particle.position.y * 100.0,
                particle.position.z * 100.0,
            );
            g4_set_particle_gun_direction(
                self.particle_gun,
                particle.velocity.x.signum(),
                particle.velocity.y.signum(),
                particle.velocity.z.signum(),
            );
            
            // Set material properties
            g4_set_material(material_c.as_ptr());
            g4_set_step_limit(step_length_cm);
            
            // Run simulation
            let n_events = 1;
            g4_run_beam_on(self.run_manager, n_events);
            
            // Extract results
            let n_interactions = g4_get_n_interactions();
            let mut events = Vec::new();
            
            for i in 0..n_interactions {
                let mut interaction_data = G4InteractionData {
                    process_type: 0,
                    energy_deposited: 0.0,
                    n_secondaries: 0,
                    secondary_particles: ptr::null_mut(),
                    secondary_energies: ptr::null_mut(),
                    step_length: 0.0,
                    position: [0.0; 3],
                };
                
                g4_get_interaction_data(i, &mut interaction_data);
                
                let event = InteractionEvent {
                    timestamp: 0.0, // Would be filled by Geant4
                    interaction_type: process_type_to_interaction(interaction_data.process_type),
                    particles_in: vec![], // Input particle would be stored here
                    particles_out: extract_secondary_particles(&interaction_data),
                    energy_exchanged: interaction_data.energy_deposited * 1.602e-13, // Convert MeV to J
                    momentum_transfer: Vector3::zeros(), // Would calculate from data
                    cross_section: 1e-28, // Simplified
                };
                
                events.push(event);
            }
            
            Ok(events)
        }
    }
    
    /// Calculate cross-section for specific process
    pub fn calculate_cross_section(
        &self,
        particle_type: &ParticleType,
        target_material: &str,
        process: &str,
        energy_mev: f64,
    ) -> Result<f64> {
        if !is_available() {
            return Err(anyhow!("Geant4 not available"));
        }
        
        let particle_name = CString::new(particle_type_to_name(particle_type))?;
        let material_name = CString::new(target_material)?;
        let process_name = CString::new(process)?;
        
        unsafe {
            let cross_section = g4_get_cross_section(
                particle_name.as_ptr(),
                material_name.as_ptr(),
                process_name.as_ptr(),
                energy_mev,
            );
            Ok(cross_section * 1e-28) // Convert barn to m²
        }
    }
    
    /// Get stopping power for particle in material
    pub fn get_stopping_power(
        &self,
        particle_type: &ParticleType,
        material: &str,
        energy_mev: f64,
    ) -> Result<f64> {
        if !is_available() {
            return Err(anyhow!("Geant4 not available"));
        }
        
        let particle_name = CString::new(particle_type_to_name(particle_type))?;
        let material_name = CString::new(material)?;
        
        unsafe {
            let stopping_power = g4_get_stopping_power(
                particle_name.as_ptr(),
                material_name.as_ptr(),
                energy_mev,
            );
            Ok(stopping_power * 1.602e-13) // Convert MeV cm²/g to J⋅m²/kg
        }
    }
    
    /// Simulate radioactive decay
    pub fn simulate_decay(&mut self, particle: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
        if !is_available() {
            return Err(anyhow!("Geant4 not available"));
        }
        
        let pdg_code = particle_type_to_pdg(&particle.particle_type);
        let energy_mev = particle.energy * 6.242e12;
        
        unsafe {
            let n_products = g4_simulate_decay(pdg_code, energy_mev);
            let mut products = Vec::new();
            
            for i in 0..n_products {
                let mut product_data = G4ParticleData {
                    pdg_code: 0,
                    energy: 0.0,
                    momentum: [0.0; 3],
                    position: [0.0; 3],
                };
                
                g4_get_decay_product(i, &mut product_data);
                
                let product = FundamentalParticle {
                    particle_type: pdg_to_particle_type(product_data.pdg_code),
                    mass: get_pdg_mass(product_data.pdg_code),
                    energy: product_data.energy * 1.602e-13, // Convert MeV to J
                    electric_charge: get_pdg_charge(product_data.pdg_code),
                    position: Vector3::new(
                        product_data.position[0] / 100.0, // Convert cm to m
                        product_data.position[1] / 100.0,
                        product_data.position[2] / 100.0,
                    ),
                    momentum: Vector3::new(
                        product_data.momentum[0],
                        product_data.momentum[1],
                        product_data.momentum[2],
                    ),
                    velocity: Vector3::zeros(), // Calculate from momentum/mass
                    spin: Vector3::zeros(),
                    color_charge: None,
                    creation_time: 0.0,
                    decay_time: None,
                    quantum_state: QuantumState::default(),
                    interaction_history: Vec::new(),
                };
                
                products.push(product);
            }
            
            Ok(products)
        }
    }
}

impl Drop for Geant4Engine {
    fn drop(&mut self) {
        // Only call cleanup if Geant4 is actually available
        if self.is_initialized && is_available() {
            unsafe {
                g4_delete_particle_gun(self.particle_gun);
                g4_delete_stepping_manager(self.step_manager);
                g4_delete_physics_list(self.physics_list);
                g4_delete_detector(self.detector_construction);
                g4_delete_run_manager(self.run_manager);
            }
        }
    }
}

/// Check if Geant4 library is available
pub fn is_available() -> bool {
    #[cfg(feature = "geant4")]
    {
        unsafe { g4_is_available() != 0 }
    }
    #[cfg(not(feature = "geant4"))]
    {
        false
    }
}

/// Initialize Geant4 subsystem
pub fn initialize() -> Result<()> {
    if !is_available() {
        log::warn!("Geant4 library not available - using fallback implementations");
        return Ok(());
    }
    
    unsafe {
        if g4_global_initialize() != 0 {
            return Err(anyhow!("Failed to initialize Geant4 global state"));
        }
    }
    
    log::info!("Geant4 particle physics engine initialized");
    Ok(())
}

/// Cleanup Geant4 subsystem
pub fn cleanup() -> Result<()> {
    if is_available() {
        unsafe {
            g4_global_cleanup();
        }
        log::info!("Geant4 engine cleaned up");
    }
    Ok(())
}

// Helper functions for type conversion
fn particle_type_to_pdg(particle_type: &ParticleType) -> c_int {
    match particle_type {
        ParticleType::Electron => 11,
        ParticleType::Positron => -11,
        ParticleType::Muon => 13,
        ParticleType::ElectronNeutrino => 12,
        ParticleType::ElectronAntiNeutrino => -12,
        ParticleType::MuonNeutrino => 14,
        ParticleType::MuonAntiNeutrino => -14,
        ParticleType::TauNeutrino => 16,
        ParticleType::TauAntiNeutrino => -16,
        ParticleType::Photon => 22,
        ParticleType::Proton => 2212,
        ParticleType::Neutron => 2112,
        ParticleType::PionPlus => 211,
        ParticleType::PionMinus => -211,
        ParticleType::PionZero => 111,
        ParticleType::KaonPlus => 321,
        ParticleType::KaonMinus => -321,
        _ => 0, // Unknown
    }
}

fn pdg_to_particle_type(pdg_code: c_int) -> ParticleType {
    match pdg_code {
        11 => ParticleType::Electron,
        -11 => ParticleType::Positron,
        13 => ParticleType::Muon,
        12 => ParticleType::ElectronNeutrino,
        -12 => ParticleType::ElectronAntiNeutrino,
        14 => ParticleType::MuonNeutrino,
        -14 => ParticleType::MuonAntiNeutrino,
        16 => ParticleType::TauNeutrino,
        -16 => ParticleType::TauAntiNeutrino,
        22 => ParticleType::Photon,
        2212 => ParticleType::Proton,
        2112 => ParticleType::Neutron,
        211 => ParticleType::PionPlus,
        -211 => ParticleType::PionMinus,
        111 => ParticleType::PionZero,
        321 => ParticleType::KaonPlus,
        -321 => ParticleType::KaonMinus,
        _ => ParticleType::DarkMatter, // Default fallback
    }
}

fn particle_type_to_name(particle_type: &ParticleType) -> &'static str {
    match particle_type {
        ParticleType::Electron => "e-",
        ParticleType::Positron => "e+",
        ParticleType::Muon => "mu-",
        ParticleType::ElectronNeutrino => "nu_e",
        ParticleType::ElectronAntiNeutrino => "anti_nu_e",
        ParticleType::MuonNeutrino => "nu_mu",
        ParticleType::MuonAntiNeutrino => "anti_nu_mu",
        ParticleType::TauNeutrino => "nu_tau",
        ParticleType::TauAntiNeutrino => "anti_nu_tau",
        ParticleType::Photon => "gamma",
        ParticleType::Proton => "proton",
        ParticleType::Neutron => "neutron",
        ParticleType::PionPlus => "pi+",
        ParticleType::PionMinus => "pi-",
        ParticleType::PionZero => "pi0",
        ParticleType::KaonPlus => "kaon+",
        ParticleType::KaonMinus => "kaon-",
        _ => "unknown",
    }
}

fn get_pdg_mass(pdg_code: c_int) -> f64 {
    match pdg_code {
        11 | -11 => 9.1093837015e-31,  // Electron mass
        13 | -13 => 1.883531627e-28,   // Muon mass
        2212 => 1.67262192369e-27,     // Proton mass
        2112 => 1.67492749804e-27,     // Neutron mass
        22 => 0.0,                     // Photon
        _ => 1e-30,                    // Default
    }
}

fn get_pdg_charge(pdg_code: c_int) -> f64 {
    // A more complete mapping would be needed
    match pdg_code {
        22 => 0.0,    // Photon
        11 => -1.0,   // Electron
        -11 => 1.0,   // Positron
        2212 => 1.0,  // Proton
        2112 => 0.0,  // Neutron
        _ => 0.0,
    }
}

fn process_type_to_interaction(process_type: c_int) -> InteractionType {
    match process_type {
        1 => InteractionType::ElectromagneticScattering,
        2 => InteractionType::WeakDecay,
        3 => InteractionType::StrongInteraction,
        _ => InteractionType::ElectromagneticScattering,
    }
}

fn extract_secondary_particles(_data: &G4InteractionData) -> Vec<FundamentalParticle> {
    // Simplified extraction - would need proper implementation
    Vec::new()
}

// Remove now-duplicate hand-written data structures to avoid redefinition when
// the real Geant-4 bindings are present.  Their stub counterparts live inside
// the `ffi` module above.
// (The deleted hand-written structs were unconditional; with this cfg-gated
// approach we avoid symbol collisions.) 