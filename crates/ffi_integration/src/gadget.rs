//! # GADGET FFI Integration
//!
//! High-fidelity cosmological N-body simulation using the GADGET simulation code.
//! This provides access to the same algorithms used in the Millennium Simulation.

use anyhow::{Result, anyhow};
use std::os::raw::{c_double, c_int};
use nalgebra::Vector3;

// Include generated bindings when feature is enabled
#[cfg(feature = "gadget")]
include!(concat!(env!("OUT_DIR"), "/gadget_bindings.rs"));

/// Safe Rust wrapper around GADGET functionality
#[derive(Debug)]
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
    pub fn initialize_cosmology(&mut self, omega_m: f64, omega_l: f64, hubble_h: f64) -> Result<()> {
        if !is_available() {
            return Err(anyhow!("GADGET not available"));
        }
        
        unsafe {
            gadget_init();
            gadget_set_cosmology(omega_m, omega_l, hubble_h);
        }
        
        self.is_initialized = true;
        Ok(())
    }
    
    /// Set simulation box size
    pub fn set_box_size(&mut self, box_size: f64) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
        }
        
        unsafe {
            gadget_set_box_size(box_size);
        }
        
        self.box_size = box_size;
        Ok(())
    }
    
    /// Add particles to simulation
    pub fn add_particles(&mut self, particles: &[GadgetParticle]) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
        }
        
        unsafe {
            gadget_init_particles(particles.len() as c_int);
            
            for (i, particle) in particles.iter().enumerate() {
                gadget_set_particle_data(
                    i as c_int,
                    particle.position.x,
                    particle.position.y,
                    particle.position.z,
                    particle.velocity.x,
                    particle.velocity.y,
                    particle.velocity.z,
                    particle.mass,
                    particle.particle_type as c_int,
                );
            }
        }
        
        self.particles = particles.to_vec();
        self.n_particles = particles.len() as c_int;
        Ok(())
    }
    
    /// Set time integration parameters
    pub fn set_time_parameters(&mut self, time_begin: f64, time_end: f64, time_step: f64) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
        }
        
        unsafe {
            gadget_set_time_parameters(time_begin, time_end, time_step);
        }
        
        self.time_current = time_begin;
        self.time_step = time_step;
        Ok(())
    }
    
    /// Perform one integration step
    pub fn step(&mut self) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
        }
        
        unsafe {
            // Build tree and calculate forces
            gadget_init_tree();
            gadget_calculate_forces();
            
            // Get adaptive timestep
            let dt = gadget_get_timestep();
            
            // Update particle positions and velocities
            gadget_update_particles(dt);
            
            // Rebuild tree for next step
            gadget_rebuild_tree();
        }
        
        self.time_current += self.time_step;
        Ok(())
    }
    
    /// Extract particle data after simulation
    pub fn get_particles(&self) -> Result<Vec<GadgetParticle>> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
        }
        
        let mut particles = Vec::new();
        
        unsafe {
            for i in 0..self.n_particles {
                let mut pos = [0.0; 3];
                let mut vel = [0.0; 3];
                let mut mass = 0.0;
                
                gadget_get_particle_data(i, pos.as_mut_ptr(), vel.as_mut_ptr(), &mut mass);
                
                particles.push(GadgetParticle {
                    id: i as usize,
                    particle_type: GadgetParticleType::DarkMatter, // Would need to track
                    position: Vector3::new(pos[0], pos[1], pos[2]),
                    velocity: Vector3::new(vel[0], vel[1], vel[2]),
                    mass,
                    acceleration: Vector3::zeros(),
                    gravitational_potential: 0.0,
                    softening_length: 0.01, // Default value
                    time_step: self.time_step,
                    active: true,
                });
            }
        }
        
        Ok(particles)
    }
    
    /// Calculate total potential energy
    pub fn get_potential_energy(&self) -> Result<f64> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
        }
        
        unsafe {
            Ok(gadget_calculate_potential_energy())
        }
    }
    
    /// Run Friends-of-Friends halo finder
    pub fn find_halos(&mut self, linking_length: f64) -> Result<Vec<Halo>> {
        if !self.is_initialized {
            return Err(anyhow!("GADGET not initialized"));
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
                    id: i as usize,
                    n_particles: halo_data.n_particles as usize,
                    total_mass: halo_data.total_mass,
                    center_of_mass: Vector3::new(
                        halo_data.center_of_mass[0],
                        halo_data.center_of_mass[1],
                        halo_data.center_of_mass[2],
                    ),
                    virial_radius: halo_data.virial_radius,
                    velocity_dispersion: halo_data.velocity_dispersion,
                    particle_ids: Vec::new(), // Would need separate call to get
                });
            }
            
            Ok(halos)
        }
    }
}

impl Drop for GadgetEngine {
    fn drop(&mut self) {
        if self.is_initialized && is_available() {
            unsafe {
                gadget_cleanup();
            }
        }
    }
}

/// Cosmological particle for N-body simulation
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
}

/// Types of particles in cosmological simulation
#[derive(Debug, Clone, Copy)]
pub enum GadgetParticleType {
    DarkMatter = 1,
    Stars = 2,
    Gas = 3,
    BlackHole = 4,
    Boundary = 5,
}

/// Cosmological parameters for simulation
#[derive(Debug, Clone)]
pub struct CosmologicalParameters {
    pub hubble_constant: f64,      // H₀ in km/s/Mpc
    pub omega_matter: f64,         // Ωₘ
    pub omega_lambda: f64,         // ΩΛ
    pub omega_baryon: f64,         // Ωᵦ
    pub scale_factor: f64,         // a(t)
    pub redshift: f64,             // z
    pub age_of_universe: f64,      // t in Gyr
    pub enable_expansion: bool,
}

/// Dark matter halo structure
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

// Internal FFI data structure
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

// Conditional compilation for FFI functions and stubs
#[cfg(feature = "gadget")]
extern "C" {
    fn gadget_version() -> c_int;
    fn gadget_init();
    fn gadget_cleanup();
    fn gadget_set_cosmology(omega_m: c_double, omega_l: c_double, hubble_h: c_double);
    fn gadget_set_box_size(box_size: c_double);
    fn gadget_init_particles(n_particles: c_int);
    fn gadget_set_particle_data(
        id: c_int, x: c_double, y: c_double, z: c_double,
        vx: c_double, vy: c_double, vz: c_double, mass: c_double, ptype: c_int
    );
    fn gadget_init_tree();
    fn gadget_set_time_parameters(time_begin: c_double, time_end: c_double, time_step: c_double);
    fn gadget_calculate_forces();
    fn gadget_get_timestep() -> c_double;
    fn gadget_update_particles(dt: c_double);
    fn gadget_rebuild_tree();
    fn gadget_get_particle_data(
        id: c_int, pos: *mut c_double, vel: *mut c_double, mass: *mut c_double
    );
    fn gadget_calculate_potential_energy() -> c_double;
    fn gadget_run_fof(linking_length: c_double);
    fn gadget_get_n_halos() -> c_int;
    fn gadget_get_halo_data(halo_id: c_int, data: *mut HaloData);
}

// Stub implementations when GADGET is not available
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_version() -> c_int { 0 }
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_init() {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_cleanup() {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_set_cosmology(_: c_double, _: c_double, _: c_double) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_set_box_size(_: c_double) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_init_particles(_: c_int) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_set_particle_data(
    _: c_int, _: c_double, _: c_double, _: c_double,
    _: c_double, _: c_double, _: c_double, _: c_double, _: c_int
) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_init_tree() {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_set_time_parameters(_: c_double, _: c_double, _: c_double) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_calculate_forces() {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_get_timestep() -> c_double { 0.0 }
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_update_particles(_: c_double) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_rebuild_tree() {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_get_particle_data(
    _: c_int, _: *mut c_double, _: *mut c_double, _: *mut c_double
) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_calculate_potential_energy() -> c_double { 0.0 }
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_run_fof(_: c_double) {}
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_get_n_halos() -> c_int { 0 }
#[cfg(not(feature = "gadget"))]
unsafe fn gadget_get_halo_data(_: c_int, _: *mut HaloData) {} 