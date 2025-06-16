//! # GADGET FFI Integration
//!
//! High-fidelity cosmological N-body simulation using the GADGET simulation code.
//! This provides access to the same algorithms used in the Millennium Simulation.

use anyhow::{Result, anyhow};
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use nalgebra::Vector3;

// Include generated bindings
#[cfg(feature = "gadget")]
include!(concat!(env!("OUT_DIR"), "/gadget_bindings.rs"));

/// Safe Rust wrapper around GADGET functionality
pub struct GadgetEngine {
    n_particles: c_int,
    box_size: f64,
    time_current: f64,
    time_step: f64,
    is_initialized: bool,
    particles: Vec<GadgetParticle>,
}

unsafe impl Send for GadgetEngine {}
unsafe impl Sync for GadgetEngine {}

impl GadgetEngine {
    /// Create new GADGET engine
    pub fn new() -> Result<Self> {
        if !is_available() {
            return Err(anyhow!("GADGET library not available"));
        }
        
        Ok(Self {
            n_particles: 0,
            box_size: 0.0,
            time_current: 0.0,
            time_step: 0.0,
            is_initialized: false,
            particles: Vec::new(),
        })
    }
    
    /// Initialize cosmological simulation
    pub fn initialize_cosmology(
        &mut self,
        particles: Vec<CosmologicalParticle>,
        box_size: f64,
        cosmological_params: CosmologicalParameters,
    ) -> Result<()> {
        self.n_particles = particles.len() as c_int;
        self.box_size = box_size;
        
        // Convert to GADGET particle format
        self.particles = particles.into_iter().map(|p| GadgetParticle {
            position: p.position,
            velocity: p.velocity,
            mass: p.mass,
            particle_type: p.particle_type as c_int,
            softening: p.softening.unwrap_or(box_size / 1000.0), // Default softening
            potential: 0.0,
            acceleration: Vector3::zeros(),
        }).collect();
        
        unsafe {
            // Initialize GADGET with cosmological parameters
            gadget_set_cosmology(
                cosmological_params.omega_matter,
                cosmological_params.omega_lambda,
                cosmological_params.hubble_param,
            );
            
            // Set box size
            gadget_set_box_size(box_size);
            
            // Initialize particle data
            gadget_init_particles(self.n_particles);
            
            for (i, particle) in self.particles.iter().enumerate() {
                gadget_set_particle_data(
                    i as c_int,
                    particle.position.x,
                    particle.position.y, 
                    particle.position.z,
                    particle.velocity.x,
                    particle.velocity.y,
                    particle.velocity.z,
                    particle.mass,
                    particle.particle_type,
                );
            }
            
            // Initialize tree structure
            gadget_init_tree();
        }
        
        self.is_initialized = true;
        Ok(())
    }
    
    /// Evolve N-body system using GADGET algorithms
    pub fn evolve_system(
        &mut self,
        target_time: f64,
        max_timestep: f64,
    ) -> Result<Vec<GadgetParticle>> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET system not initialized"));
        }
        
        unsafe {
            // Set evolution parameters
            gadget_set_time_parameters(self.time_current, target_time, max_timestep);
            
            // Main evolution loop (simplified)
            while self.time_current < target_time {
                // Calculate gravitational forces using tree algorithm
                gadget_calculate_forces();
                
                // Update particle positions and velocities
                let dt = gadget_get_timestep();
                gadget_update_particles(dt);
                
                self.time_current += dt;
                self.time_step = dt;
                
                // Periodic tree reconstruction
                if self.time_current.fract() < dt {
                    gadget_rebuild_tree();
                }
            }
            
            // Extract updated particle data
            for (i, particle) in self.particles.iter_mut().enumerate() {
                let mut pos = [0.0; 3];
                let mut vel = [0.0; 3];
                let mut acc = [0.0; 3];
                let mut pot = 0.0;
                
                gadget_get_particle_data(
                    i as c_int,
                    pos.as_mut_ptr(),
                    vel.as_mut_ptr(),
                    acc.as_mut_ptr(),
                    &mut pot,
                );
                
                particle.position = Vector3::new(pos[0], pos[1], pos[2]);
                particle.velocity = Vector3::new(vel[0], vel[1], vel[2]);
                particle.acceleration = Vector3::new(acc[0], acc[1], acc[2]);
                particle.potential = pot;
            }
        }
        
        Ok(self.particles.clone())
    }
    
    /// Calculate gravitational potential energy
    pub fn calculate_potential_energy(&self) -> Result<f64> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET system not initialized"));
        }
        
        unsafe {
            Ok(gadget_calculate_potential_energy())
        }
    }
    
    /// Calculate kinetic energy
    pub fn calculate_kinetic_energy(&self) -> Result<f64> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET system not initialized"));
        }
        
        let mut kinetic_energy = 0.0;
        for particle in &self.particles {
            kinetic_energy += 0.5 * particle.mass * particle.velocity.magnitude_squared();
        }
        
        Ok(kinetic_energy)
    }
    
    /// Get halo catalog using friends-of-friends algorithm
    pub fn find_halos(&self, linking_length: f64) -> Result<Vec<Halo>> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET system not initialized"));
        }
        
        unsafe {
            gadget_run_fof(linking_length);
            let n_halos = gadget_get_n_halos();
            let mut halos = Vec::new();
            
            for i in 0..n_halos {
                let mut halo_data = HaloData {
                    n_particles: 0,
                    total_mass: 0.0,
                    center_of_mass: [0.0; 3],
                    virial_radius: 0.0,
                    velocity_dispersion: 0.0,
                };
                
                gadget_get_halo_data(i, &mut halo_data);
                
                halos.push(Halo {
                    id: i as u64,
                    n_particles: halo_data.n_particles as usize,
                    total_mass: halo_data.total_mass,
                    center_of_mass: Vector3::new(
                        halo_data.center_of_mass[0],
                        halo_data.center_of_mass[1], 
                        halo_data.center_of_mass[2]
                    ),
                    virial_radius: halo_data.virial_radius,
                    velocity_dispersion: halo_data.velocity_dispersion,
                });
            }
            
            Ok(halos)
        }
    }
    
    /// Get current simulation statistics
    pub fn get_simulation_stats(&self) -> SimulationStats {
        SimulationStats {
            current_time: self.time_current,
            timestep: self.time_step,
            n_particles: self.n_particles as usize,
            box_size: self.box_size,
            potential_energy: self.calculate_potential_energy().unwrap_or(0.0),
            kinetic_energy: self.calculate_kinetic_energy().unwrap_or(0.0),
        }
    }
}

/// GADGET particle representation
#[derive(Debug, Clone)]
pub struct GadgetParticle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub acceleration: Vector3<f64>,
    pub mass: f64,
    pub particle_type: c_int,
    pub softening: f64,
    pub potential: f64,
}

/// Input cosmological particle
#[derive(Debug, Clone)]
pub struct CosmologicalParticle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub mass: f64,
    pub particle_type: ParticleType,
    pub softening: Option<f64>,
}

/// Particle types in cosmological simulation
#[derive(Debug, Clone, Copy)]
pub enum ParticleType {
    DarkMatter = 1,
    Gas = 0,
    Star = 4,
    BlackHole = 5,
}

/// Cosmological parameters
#[derive(Debug, Clone)]
pub struct CosmologicalParameters {
    pub omega_matter: f64,    // Matter density parameter
    pub omega_lambda: f64,    // Dark energy density parameter
    pub hubble_param: f64,    // Hubble parameter (h)
    pub sigma8: f64,          // Power spectrum normalization
    pub spectral_index: f64,  // Spectral index (n_s)
}

impl Default for CosmologicalParameters {
    fn default() -> Self {
        // Planck 2018 values
        Self {
            omega_matter: 0.315,
            omega_lambda: 0.685,
            hubble_param: 0.674,
            sigma8: 0.811,
            spectral_index: 0.965,
        }
    }
}

/// Dark matter halo
#[derive(Debug, Clone)]
pub struct Halo {
    pub id: u64,
    pub n_particles: usize,
    pub total_mass: f64,
    pub center_of_mass: Vector3<f64>,
    pub virial_radius: f64,
    pub velocity_dispersion: f64,
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct SimulationStats {
    pub current_time: f64,
    pub timestep: f64,
    pub n_particles: usize,
    pub box_size: f64,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
}

/// C structure for halo data
#[repr(C)]
struct HaloData {
    n_particles: c_int,
    total_mass: c_double,
    center_of_mass: [c_double; 3],
    virial_radius: c_double,
    velocity_dispersion: c_double,
}

/// Check if GADGET library is available
pub fn is_available() -> bool {
    #[cfg(feature = "gadget")]
    {
        unsafe { gadget_version() > 0 }
    }
    #[cfg(not(feature = "gadget"))]
    {
        false
    }
}

/// Initialize GADGET library
pub fn initialize() -> Result<()> {
    if is_available() {
        unsafe {
            gadget_init();
        }
        log::info!("GADGET library initialized successfully");
        Ok(())
    } else {
        log::warn!("GADGET library not available - using fallback implementation");
        Ok(())
    }
}

/// Cleanup GADGET library
pub fn cleanup() -> Result<()> {
    if is_available() {
        unsafe {
            gadget_cleanup();
        }
    }
    log::info!("GADGET library cleaned up");
    Ok(())
}

// Stub implementations when GADGET is not available
#[cfg(not(feature = "gadget"))]
mod stubs {
    use super::*;
    
    pub unsafe extern "C" fn gadget_version() -> c_int { 0 }
    pub unsafe extern "C" fn gadget_init() {}
    pub unsafe extern "C" fn gadget_cleanup() {}
    pub unsafe extern "C" fn gadget_set_cosmology(_: c_double, _: c_double, _: c_double) {}
    pub unsafe extern "C" fn gadget_set_box_size(_: c_double) {}
    pub unsafe extern "C" fn gadget_init_particles(_: c_int) {}
    pub unsafe extern "C" fn gadget_set_particle_data(
        _: c_int, _: c_double, _: c_double, _: c_double,
        _: c_double, _: c_double, _: c_double, _: c_double, _: c_int
    ) {}
    pub unsafe extern "C" fn gadget_init_tree() {}
    pub unsafe extern "C" fn gadget_set_time_parameters(_: c_double, _: c_double, _: c_double) {}
    pub unsafe extern "C" fn gadget_calculate_forces() {}
    pub unsafe extern "C" fn gadget_get_timestep() -> c_double { 0.0 }
    pub unsafe extern "C" fn gadget_update_particles(_: c_double) {}
    pub unsafe extern "C" fn gadget_rebuild_tree() {}
    pub unsafe extern "C" fn gadget_get_particle_data(
        _: c_int, _: *mut c_double, _: *mut c_double, _: *mut c_double, _: *mut c_double
    ) {}
    pub unsafe extern "C" fn gadget_calculate_potential_energy() -> c_double { 0.0 }
    pub unsafe extern "C" fn gadget_run_fof(_: c_double) {}
    pub unsafe extern "C" fn gadget_get_n_halos() -> c_int { 0 }
    pub unsafe extern "C" fn gadget_get_halo_data(_: c_int, _: *mut HaloData) {}
}

#[cfg(not(feature = "gadget"))]
use stubs::*; 