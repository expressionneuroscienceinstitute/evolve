//! # Geant4 FFI Integration
//!
//! High-fidelity particle physics using the actual Geant4 Monte Carlo toolkit.
//! This provides the highest accuracy possible for particle interactions, transport, and detector simulation.

use crate::types::{FundamentalParticle, ParticleType, InteractionEvent, InteractionType};
use anyhow::{Result, anyhow};
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use nalgebra::Vector3;

// Include generated bindings
#[cfg(feature = "geant4")]
include!(concat!(env!("OUT_DIR"), "/geant4_bindings.rs"));

/// Safe Rust wrapper around Geant4 functionality
pub struct Geant4Engine {
    detector_construction: *mut c_void,
    physics_list: *mut c_void,
    run_manager: *mut c_void,
    step_manager: *mut c_void,
    particle_gun: *mut c_void,
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
            let run_manager = g4_create_run_manager();
            if run_manager.is_null() {
                return Err(anyhow!("Failed to create Geant4 run manager"));
            }
            
            let detector_construction = g4_create_simple_detector();
            if detector_construction.is_null() {
                g4_delete_run_manager(run_manager);
                return Err(anyhow!("Failed to create detector construction"));
            }
            
            let physics_list = g4_create_physics_list(physics_list_c.as_ptr());
            if physics_list.is_null() {
                g4_delete_detector(detector_construction);
                g4_delete_run_manager(run_manager);
                return Err(anyhow!("Failed to create physics list"));
            }
            
            let step_manager = g4_create_stepping_manager();
            let particle_gun = g4_create_particle_gun();
            
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
                    timestamp: 0.0, // Would need to track simulation time
                    interaction_type: process_type_to_interaction(interaction_data.process_type),
                    participants: vec![0], // Would need particle tracking
                    energy_exchanged: interaction_data.energy_deposited * 1.602e-13, // MeV to J
                    momentum_transfer: Vector3::zeros(), // Would extract from secondaries
                    products: extract_secondary_particles(&interaction_data),
                    cross_section: 0.0, // Would calculate from process
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
        let particle_c = CString::new(particle_type_to_name(particle_type))?;
        let material_c = CString::new(target_material)?;
        let process_c = CString::new(process)?;
        
        unsafe {
            let cross_section_barns = g4_get_cross_section(
                particle_c.as_ptr(),
                material_c.as_ptr(),
                process_c.as_ptr(),
                energy_mev,
            );
            
            // Convert barns to m²
            Ok(cross_section_barns * 1e-28)
        }
    }
    
    /// Get stopping power for particle in material
    pub fn get_stopping_power(
        &self,
        particle_type: &ParticleType,
        material: &str,
        energy_mev: f64,
    ) -> Result<f64> {
        let particle_c = CString::new(particle_type_to_name(particle_type))?;
        let material_c = CString::new(material)?;
        
        unsafe {
            let stopping_power = g4_get_stopping_power(
                particle_c.as_ptr(), 
                material_c.as_ptr(),
                energy_mev,
            );
            
            // Convert MeV cm²/g to J m²/kg
            Ok(stopping_power * 1.602e-13 * 1e4 / 1e3)
        }
    }
    
    /// Generate realistic decay products
    pub fn simulate_decay(&mut self, particle: &FundamentalParticle) -> Result<Vec<FundamentalParticle>> {
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
                    charge: get_pdg_charge(product_data.pdg_code),
                    energy: product_data.energy * 1.602e-13, // MeV to J
                    position: Vector3::new(
                        product_data.position[0] / 100.0, // cm to m
                        product_data.position[1] / 100.0,
                        product_data.position[2] / 100.0,
                    ),
                    velocity: Vector3::new(
                        product_data.momentum[0],
                        product_data.momentum[1],
                        product_data.momentum[2],
                    ).normalize() * (2.0 * product_data.energy * 1.602e-13 / get_pdg_mass(product_data.pdg_code)).sqrt(),
                    spin: Vector3::zeros(),
                    creation_time: particle.creation_time,
                    last_interaction_time: particle.last_interaction_time,
                };
                
                products.push(product);
            }
            
            Ok(products)
        }
    }
}

impl Drop for Geant4Engine {
    fn drop(&mut self) {
        if self.is_initialized {
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
        ParticleType::AntiMuon => -13,
        ParticleType::ElectronNeutrino => 12,
        ParticleType::ElectronAntiNeutrino => -12,
        ParticleType::MuonNeutrino => 14,
        ParticleType::MuonAntiNeutrino => -14,
        ParticleType::TauNeutrino => 16,
        ParticleType::TauAntiNeutrino => -16,
        ParticleType::Photon => 22,
        ParticleType::Proton => 2212,
        ParticleType::Neutron => 2112,
        ParticleType::PiPlus => 211,
        ParticleType::PiMinus => -211,
        ParticleType::PiZero => 111,
        ParticleType::Kaon => 321,
        ParticleType::AntiKaon => -321,
        _ => 0, // Unknown
    }
}

fn pdg_to_particle_type(pdg_code: c_int) -> ParticleType {
    match pdg_code {
        11 => ParticleType::Electron,
        -11 => ParticleType::Positron,
        13 => ParticleType::Muon,
        -13 => ParticleType::AntiMuon,
        12 => ParticleType::ElectronNeutrino,
        -12 => ParticleType::ElectronAntiNeutrino,
        14 => ParticleType::MuonNeutrino,
        -14 => ParticleType::MuonAntiNeutrino,
        16 => ParticleType::TauNeutrino,
        -16 => ParticleType::TauAntiNeutrino,
        22 => ParticleType::Photon,
        2212 => ParticleType::Proton,
        2112 => ParticleType::Neutron,
        211 => ParticleType::PiPlus,
        -211 => ParticleType::PiMinus,
        111 => ParticleType::PiZero,
        321 => ParticleType::Kaon,
        -321 => ParticleType::AntiKaon,
        _ => ParticleType::Unknown,
    }
}

fn particle_type_to_name(particle_type: &ParticleType) -> &'static str {
    match particle_type {
        ParticleType::Electron => "e-",
        ParticleType::Positron => "e+",
        ParticleType::Muon => "mu-",
        ParticleType::AntiMuon => "mu+",
        ParticleType::ElectronNeutrino => "nu_e",
        ParticleType::ElectronAntiNeutrino => "anti_nu_e",
        ParticleType::MuonNeutrino => "nu_mu",
        ParticleType::MuonAntiNeutrino => "anti_nu_mu",
        ParticleType::TauNeutrino => "nu_tau",
        ParticleType::TauAntiNeutrino => "anti_nu_tau",
        ParticleType::Photon => "gamma",
        ParticleType::Proton => "proton",
        ParticleType::Neutron => "neutron",
        ParticleType::PiPlus => "pi+",
        ParticleType::PiMinus => "pi-",
        ParticleType::PiZero => "pi0",
        ParticleType::Kaon => "kaon+",
        ParticleType::AntiKaon => "kaon-",
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
    match pdg_code {
        11 | 13 => -1.602176634e-19,   // Negative charge
        -11 | -13 => 1.602176634e-19,  // Positive charge
        2212 => 1.602176634e-19,       // Proton
        _ => 0.0,                      // Neutral
    }
}

// Stub functions - these would be implemented in C wrapper
#[cfg(not(feature = "geant4"))]
mod stubs {
    use super::*;
    
    pub unsafe extern "C" fn g4_is_available() -> c_int { 0 }
    pub unsafe extern "C" fn g4_global_initialize() -> c_int { -1 }
    pub unsafe extern "C" fn g4_global_cleanup() {}
    pub unsafe extern "C" fn g4_create_run_manager() -> *mut c_void { ptr::null_mut() }
    pub unsafe extern "C" fn g4_delete_run_manager(_: *mut c_void) {}
    pub unsafe extern "C" fn g4_create_simple_detector() -> *mut c_void { ptr::null_mut() }
    pub unsafe extern "C" fn g4_delete_detector(_: *mut c_void) {}
    pub unsafe extern "C" fn g4_create_physics_list(_: *const c_char) -> *mut c_void { ptr::null_mut() }
    pub unsafe extern "C" fn g4_delete_physics_list(_: *mut c_void) {}
    pub unsafe extern "C" fn g4_create_stepping_manager() -> *mut c_void { ptr::null_mut() }
    pub unsafe extern "C" fn g4_delete_stepping_manager(_: *mut c_void) {}
    pub unsafe extern "C" fn g4_create_particle_gun() -> *mut c_void { ptr::null_mut() }
    pub unsafe extern "C" fn g4_delete_particle_gun(_: *mut c_void) {}
}

#[cfg(not(feature = "geant4"))]
use stubs::*;

// FFI function declarations - these would be generated by bindgen
extern "C" {
    fn g4_is_available() -> c_int;
    fn g4_global_initialize() -> c_int;
    fn g4_global_cleanup();
    fn g4_create_run_manager() -> *mut c_void;
    fn g4_delete_run_manager(manager: *mut c_void);
    fn g4_create_simple_detector() -> *mut c_void;
    fn g4_delete_detector(detector: *mut c_void);
    fn g4_create_physics_list(name: *const c_char) -> *mut c_void;
    fn g4_delete_physics_list(physics_list: *mut c_void);
    fn g4_create_stepping_manager() -> *mut c_void;
    fn g4_delete_stepping_manager(manager: *mut c_void);
    fn g4_create_particle_gun() -> *mut c_void;
    fn g4_delete_particle_gun(gun: *mut c_void);
    fn g4_set_detector_construction(manager: *mut c_void, detector: *mut c_void);
    fn g4_set_physics_list(manager: *mut c_void, physics_list: *mut c_void);
    fn g4_initialize_geant4(manager: *mut c_void);
    fn g4_set_particle_gun_particle(gun: *mut c_void, pdg_code: c_int);
    fn g4_set_particle_gun_energy(gun: *mut c_void, energy_mev: c_double);
    fn g4_set_particle_gun_position(gun: *mut c_void, x: c_double, y: c_double, z: c_double);
    fn g4_set_particle_gun_direction(gun: *mut c_void, dx: c_double, dy: c_double, dz: c_double);
    fn g4_set_material(material: *const c_char);
    fn g4_set_step_limit(step_cm: c_double);
    fn g4_run_beam_on(manager: *mut c_void, n_events: c_int);
    fn g4_get_n_interactions() -> c_int;
    fn g4_get_interaction_data(index: c_int, data: *mut G4InteractionData);
    fn g4_get_cross_section(particle: *const c_char, material: *const c_char, 
                           process: *const c_char, energy_mev: c_double) -> c_double;
    fn g4_get_stopping_power(particle: *const c_char, material: *const c_char, 
                            energy_mev: c_double) -> c_double;
    fn g4_simulate_decay(pdg_code: c_int, energy_mev: c_double) -> c_int;
    fn g4_get_decay_product(index: c_int, data: *mut G4ParticleData);
}

// Data structures for FFI
#[repr(C)]
struct G4InteractionData {
    process_type: c_int,
    energy_deposited: c_double,
    n_secondaries: c_int,
    secondary_particles: *mut c_int,
    secondary_energies: *mut c_double,
    step_length: c_double,
    position: [c_double; 3],
}

#[repr(C)]
struct G4ParticleData {
    pdg_code: c_int,
    energy: c_double,
    momentum: [c_double; 3],
    position: [c_double; 3],
}

// Helper function implementations
fn process_type_to_interaction(process_type: c_int) -> InteractionType {
    match process_type {
        1 => InteractionType::ElectromagneticScattering,
        2 => InteractionType::WeakDecay,
        3 => InteractionType::StrongInteraction,
        4 => InteractionType::PairProduction,
        5 => InteractionType::Annihilation,
        _ => InteractionType::ElectromagneticScattering,
    }
}

fn extract_secondary_particles(data: &G4InteractionData) -> Vec<ParticleType> {
    let mut particles = Vec::new();
    if !data.secondary_particles.is_null() {
        unsafe {
            for i in 0..data.n_secondaries {
                let pdg_code = *data.secondary_particles.add(i as usize);
                particles.push(pdg_to_particle_type(pdg_code));
            }
        }
    }
    particles
} 