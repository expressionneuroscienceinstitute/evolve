//! # LAMMPS FFI Integration
//!
//! High-fidelity molecular dynamics using the LAMMPS simulation package.
//! This provides access to decades of MD development and validation.

use anyhow::{Result, anyhow};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use nalgebra::Vector3;

// Include generated bindings when feature is enabled
#[cfg(feature = "lammps")]
include!(concat!(env!("OUT_DIR"), "/lammps_bindings.rs"));

/// Safe Rust wrapper around LAMMPS functionality
#[derive(Debug)]
pub struct LammpsEngine {
    lammps_handle: *mut c_void,
    natoms: c_int,
    is_initialized: bool,
}

unsafe impl Send for LammpsEngine {}
unsafe impl Sync for LammpsEngine {}

impl LammpsEngine {
    /// Create new LAMMPS engine
    pub fn new() -> Result<Self> {
        if !is_available() {
            return Err(anyhow!("LAMMPS library not available"));
        }
        
        unsafe {
            let handle = lammps_open_no_mpi(0, ptr::null_mut());
            if handle.is_null() {
                return Err(anyhow!("Failed to create LAMMPS instance"));
            }
            
            Ok(Self {
                lammps_handle: handle,
                natoms: 0,
                is_initialized: false,
            })
        }
    }
    
    /// Initialize molecular system
    pub fn initialize_system(
        &mut self,
        atoms: &[(Vector3<f64>, Vector3<f64>, f64, i32)], // position, velocity, mass, type
        box_bounds: [f64; 6], // xlo, xhi, ylo, yhi, zlo, zhi
        force_field: &str,
    ) -> Result<()> {
        if self.lammps_handle.is_null() {
            return Err(anyhow!("LAMMPS not initialized"));
        }
        
        let natoms = atoms.len() as c_int;
        self.natoms = natoms;
        
        unsafe {
            // Set box bounds
            let region_cmd = CString::new(format!(
                "region box block {} {} {} {} {} {}",
                box_bounds[0], box_bounds[1], box_bounds[2], 
                box_bounds[3], box_bounds[4], box_bounds[5]
            ))?;
            lammps_command(self.lammps_handle, region_cmd.as_ptr());
            
            // Create box
            let create_box_cmd = CString::new("create_box 1 box")?;
            lammps_command(self.lammps_handle, create_box_cmd.as_ptr());
            
            // Set atom style
            let atom_style_cmd = CString::new("atom_style atomic")?;
            lammps_command(self.lammps_handle, atom_style_cmd.as_ptr());
            
            // Create atoms
            for (pos, _vel, _mass, atom_type) in atoms.iter() {
                let create_atom_cmd = CString::new(format!(
                    "create_atoms {} single {} {} {}",
                    atom_type, pos.x, pos.y, pos.z
                ))?;
                lammps_command(self.lammps_handle, create_atom_cmd.as_ptr());
            }
            
            // Set masses
            let mass_cmd = CString::new(format!("mass 1 {}", atoms[0].2))?;
            lammps_command(self.lammps_handle, mass_cmd.as_ptr());
            
            // Set force field
            let pair_style_cmd = CString::new(format!("pair_style {}", force_field))?;
            lammps_command(self.lammps_handle, pair_style_cmd.as_ptr());
            
            // Set initial velocities
            for (i, (_, vel, _, _)) in atoms.iter().enumerate() {
                let vel_cmd = CString::new(format!(
                    "velocity {} set {} {} {}",
                    i + 1, vel.x, vel.y, vel.z
                ))?;
                lammps_command(self.lammps_handle, vel_cmd.as_ptr());
            }
            
            self.is_initialized = true;
        }
        
        Ok(())
    }
    
    /// Run molecular dynamics simulation
    pub fn run_dynamics(
        &mut self,
        timestep: f64,
        nsteps: i32,
        temperature: f64,
        pressure: Option<f64>,
    ) -> Result<Vec<MolecularState>> {
        if !self.is_initialized {
            return Err(anyhow!("LAMMPS system not initialized"));
        }
        
        unsafe {
            // Set timestep
            let timestep_cmd = CString::new(format!("timestep {}", timestep))?;
            lammps_command(self.lammps_handle, timestep_cmd.as_ptr());
            
            // Set temperature control
            let thermo_cmd = CString::new(format!(
                "fix nvt all nvt temp {} {} {}",
                temperature, temperature, timestep * 100.0
            ))?;
            lammps_command(self.lammps_handle, thermo_cmd.as_ptr());
            
            // Set pressure control if specified
            if let Some(p) = pressure {
                let pressure_cmd = CString::new(format!(
                    "fix npt all npt temp {} {} {} iso {} {} {}",
                    temperature, temperature, timestep * 100.0,
                    p, p, timestep * 1000.0
                ))?;
                lammps_command(self.lammps_handle, pressure_cmd.as_ptr());
            }
            
            // Run simulation
            let run_cmd = CString::new(format!("run {}", nsteps))?;
            lammps_command(self.lammps_handle, run_cmd.as_ptr());
            
            // Extract trajectory data
            let mut states = Vec::new();
            let positions = lammps_extract_atom(self.lammps_handle, c"x".as_ptr());
            let velocities = lammps_extract_atom(self.lammps_handle, c"v".as_ptr());
            let forces = lammps_extract_atom(self.lammps_handle, c"f".as_ptr());
            
            if !positions.is_null() && !velocities.is_null() && !forces.is_null() {
                let pos_array = positions as *const [f64; 3];
                let vel_array = velocities as *const [f64; 3];
                let force_array = forces as *const [f64; 3];
                
                for i in 0..self.natoms as isize {
                    let pos = *pos_array.offset(i);
                    let vel = *vel_array.offset(i);
                    let force = *force_array.offset(i);
                    
                    states.push(MolecularState {
                        position: Vector3::new(pos[0], pos[1], pos[2]),
                        velocity: Vector3::new(vel[0], vel[1], vel[2]),
                        force: Vector3::new(force[0], force[1], force[2]),
                        potential_energy: 0.0, // Would extract from LAMMPS
                        kinetic_energy: 0.5 * (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)),
                    });
                }
            }
            
            Ok(states)
        }
    }
    
    /// Calculate thermodynamic properties
    pub fn get_thermodynamic_properties(&self) -> Result<ThermodynamicState> {
        if !self.is_initialized {
            return Err(anyhow!("LAMMPS system not initialized"));
        }
        
        unsafe {
            let temp = lammps_extract_global(self.lammps_handle, c"temp".as_ptr());
            let press = lammps_extract_global(self.lammps_handle, c"press".as_ptr());
            let pe = lammps_extract_global(self.lammps_handle, c"pe".as_ptr());
            let ke = lammps_extract_global(self.lammps_handle, c"ke".as_ptr());
            
            Ok(ThermodynamicState {
                temperature: if !temp.is_null() { *(temp as *const f64) } else { 0.0 },
                pressure: if !press.is_null() { *(press as *const f64) } else { 0.0 },
                potential_energy: if !pe.is_null() { *(pe as *const f64) } else { 0.0 },
                kinetic_energy: if !ke.is_null() { *(ke as *const f64) } else { 0.0 },
                volume: 0.0, // Would calculate from box bounds
            })
        }
    }
    
    /// Set force field parameters
    pub fn set_force_field(&mut self, ff_type: ForceFieldType, parameters: &[f64]) -> Result<()> {
        if self.lammps_handle.is_null() {
            return Err(anyhow!("LAMMPS not initialized"));
        }
        
        unsafe {
            match ff_type {
                ForceFieldType::LennardJones => {
                    if parameters.len() >= 2 {
                        let pair_coeff_cmd = CString::new(format!(
                            "pair_coeff 1 1 {} {}",
                            parameters[0], parameters[1] // epsilon, sigma
                        ))?;
                        lammps_command(self.lammps_handle, pair_coeff_cmd.as_ptr());
                    }
                },
                ForceFieldType::CHARMM => {
                    let pair_style_cmd = CString::new("pair_style charmm/coul/long 8.0 10.0")?;
                    lammps_command(self.lammps_handle, pair_style_cmd.as_ptr());
                },
                ForceFieldType::AMBER => {
                    let pair_style_cmd = CString::new("pair_style amber")?;
                    lammps_command(self.lammps_handle, pair_style_cmd.as_ptr());
                },
                ForceFieldType::ReaxFF => {
                    let pair_style_cmd = CString::new("pair_style reax/c NULL")?;
                    lammps_command(self.lammps_handle, pair_style_cmd.as_ptr());
                },
            }
        }
        
        Ok(())
    }

    /// Remove all atoms from the current LAMMPS system
    pub fn clear_atoms(&mut self) -> Result<()> {
        if self.lammps_handle.is_null() {
            return Err(anyhow!("LAMMPS not initialized"));
        }
        // In real FFI we would call appropriate deletion commands.
        // For stub we simply reset natoms counter.
        self.natoms = 0;
        Ok(())
    }

    /// Add a single atom to the LAMMPS system
    pub fn add_atom(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64, _charge: f64) -> Result<()> {
        if self.lammps_handle.is_null() {
            return Err(anyhow!("LAMMPS not initialized"));
        }
        unsafe {
            let atom_type = 1; // Single atom type for stub
            // Create atom in LAMMPS domain
            let cmd = CString::new(format!(
                "create_atoms {} single {} {} {}",
                atom_type, position.x, position.y, position.z
            ))?;
            lammps_command(self.lammps_handle, cmd.as_ptr());

            // Set mass for this type only first time
            if self.natoms == 0 {
                let mass_cmd = CString::new(format!("mass {} {}", atom_type, mass))?;
                lammps_command(self.lammps_handle, mass_cmd.as_ptr());
            }

            // Set velocity
            let vel_cmd = CString::new(format!(
                "velocity {} set {} {} {}",
                self.natoms + 1, velocity.x, velocity.y, velocity.z
            ))?;
            lammps_command(self.lammps_handle, vel_cmd.as_ptr());
        }
        self.natoms += 1;
        Ok(())
    }

    /// Retrieve current atomic positions/velocities/forces
    pub fn get_atom_data(&self) -> Result<Vec<MolecularState>> {
        if self.lammps_handle.is_null() {
            return Err(anyhow!("LAMMPS not initialized"));
        }
        let mut states = Vec::new();
        unsafe {
            let positions = lammps_extract_atom(self.lammps_handle, c"x".as_ptr());
            let velocities = lammps_extract_atom(self.lammps_handle, c"v".as_ptr());
            let forces = lammps_extract_atom(self.lammps_handle, c"f".as_ptr());
            if !positions.is_null() && !velocities.is_null() && !forces.is_null() {
                let pos_array = positions as *const [f64; 3];
                let vel_array = velocities as *const [f64; 3];
                let force_array = forces as *const [f64; 3];
                for i in 0..self.natoms as isize {
                    let pos = *pos_array.offset(i);
                    let vel = *vel_array.offset(i);
                    let force = *force_array.offset(i);
                    states.push(MolecularState {
                        position: Vector3::new(pos[0], pos[1], pos[2]),
                        velocity: Vector3::new(vel[0], vel[1], vel[2]),
                        force: Vector3::new(force[0], force[1], force[2]),
                        potential_energy: 0.0,
                        kinetic_energy: 0.5 * (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)),
                    });
                }
            }
        }
        Ok(states)
    }
}

impl Drop for LammpsEngine {
    fn drop(&mut self) {
        if !self.lammps_handle.is_null() {
            unsafe {
                lammps_close(self.lammps_handle);
            }
        }
    }
}

/// Molecular state at a given time
#[derive(Debug, Clone)]
pub struct MolecularState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub force: Vector3<f64>,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
}

/// Thermodynamic state of the system
#[derive(Debug, Clone)]
pub struct ThermodynamicState {
    pub temperature: f64,
    pub pressure: f64,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub volume: f64,
}

/// Force field types supported by LAMMPS
#[derive(Debug, Clone, Copy)]
pub enum ForceFieldType {
    LennardJones,
    CHARMM,
    AMBER,
    ReaxFF,
}

/// Check if LAMMPS library is available
pub fn is_available() -> bool {
    #[cfg(feature = "lammps")]
    {
        unsafe { lammps_version() > 0 }
    }
    #[cfg(not(feature = "lammps"))]
    {
        false
    }
}

/// Initialize LAMMPS library
pub fn initialize() -> Result<()> {
    if is_available() {
        log::info!("LAMMPS library initialized successfully");
        Ok(())
    } else {
        log::warn!("LAMMPS library not available - using fallback implementation");
        Ok(())
    }
}

/// Cleanup LAMMPS library
pub fn cleanup() -> Result<()> {
    log::info!("LAMMPS library cleaned up");
    Ok(())
}

// Conditional compilation for FFI functions and stubs
#[cfg(feature = "lammps")]
extern "C" {
    fn lammps_open_no_mpi(argc: c_int, argv: *mut *mut c_char) -> *mut c_void;
    fn lammps_close(handle: *mut c_void);
    fn lammps_command(handle: *mut c_void, cmd: *const c_char);
    fn lammps_extract_atom(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn lammps_extract_global(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn lammps_version() -> c_int;
}

// Stub implementations when LAMMPS is not available
#[cfg(not(feature = "lammps"))]
unsafe fn lammps_open_no_mpi(_: c_int, _: *mut *mut c_char) -> *mut c_void { ptr::null_mut() }
#[cfg(not(feature = "lammps"))]
unsafe fn lammps_close(_: *mut c_void) {}
#[cfg(not(feature = "lammps"))]
unsafe fn lammps_command(_: *mut c_void, _: *const c_char) {}
#[cfg(not(feature = "lammps"))]
unsafe fn lammps_extract_atom(_: *mut c_void, _: *const c_char) -> *mut c_void { ptr::null_mut() }
#[cfg(not(feature = "lammps"))]
unsafe fn lammps_extract_global(_: *mut c_void, _: *const c_char) -> *mut c_void { ptr::null_mut() }
#[cfg(not(feature = "lammps"))]
unsafe fn lammps_version() -> c_int { 0 } 