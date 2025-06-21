//! GADGET-style cosmological gravity implementation
//! 
//! Provides cosmological parameters and gravity calculations for large-scale structure formation

use nalgebra::Vector3;
use serde::{Serialize, Deserialize};

/// Cosmological parameters for ΛCDM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmologicalParameters {
    /// Hubble parameter at z=0 (km/s/Mpc)
    pub h0: f64,
    /// Matter density parameter
    pub omega_m: f64,
    /// Dark energy density parameter
    pub omega_lambda: f64,
    /// Baryon density parameter
    pub omega_b: f64,
    /// Radiation density parameter
    pub omega_r: f64,
    /// Curvature density parameter
    pub omega_k: f64,
    /// Spectral index of primordial power spectrum
    pub n_s: f64,
    /// Amplitude of primordial power spectrum
    pub sigma_8: f64,
    /// Current age of universe (seconds)
    pub age_universe: f64,
}

impl Default for CosmologicalParameters {
    fn default() -> Self {
        // Planck 2018 best-fit values
        Self {
            h0: 67.66, // km/s/Mpc
            omega_m: 0.3111,
            omega_lambda: 0.6889,
            omega_b: 0.0490,
            omega_r: 9.0e-5,
            omega_k: 0.0,
            n_s: 0.9665,
            sigma_8: 0.8102,
            age_universe: 13.787e9 * 365.25 * 24.0 * 3600.0, // seconds
        }
    }
}

impl CosmologicalParameters {
    /// Calculate scale factor from cosmic time
    pub fn scale_factor_from_time(&self, time_seconds: f64) -> f64 {
        // Simplified calculation - in practice this would use numerical integration
        // of the Friedmann equation
        let h0_s = self.h0 * 1000.0 / (3.086e22); // Convert to s^-1
        let t_hubble = 1.0 / h0_s;
        
        // Approximate scale factor for matter-dominated era
        (time_seconds / t_hubble).powf(2.0/3.0)
    }
    
    /// Calculate cosmic time from scale factor
    pub fn time_from_scale_factor(&self, scale_factor: f64) -> f64 {
        let h0_s = self.h0 * 1000.0 / (3.086e22);
        let t_hubble = 1.0 / h0_s;
        
        // Approximate time for matter-dominated era
        t_hubble * scale_factor.powf(3.0/2.0)
    }
    
    /// Calculate Hubble parameter at given scale factor
    pub fn hubble_parameter(&self, scale_factor: f64) -> f64 {
        let h0_s = self.h0 * 1000.0 / (3.086e22);
        let a = scale_factor;
        
        // H(a) = H0 * sqrt(Ωm/a³ + Ωr/a⁴ + ΩΛ + Ωk/a²)
        let h_squared = self.omega_m / (a * a * a) + 
                       self.omega_r / (a * a * a * a) + 
                       self.omega_lambda + 
                       self.omega_k / (a * a);
        
        h0_s * h_squared.sqrt()
    }
}

/// GADGET-style gravity solver for cosmological simulations
pub struct GadgetGravitySolver {
    pub cosmological_params: CosmologicalParameters,
    pub softening_length: f64,
    pub force_accuracy: f64,
}

impl GadgetGravitySolver {
    pub fn new(params: CosmologicalParameters) -> Self {
        Self {
            cosmological_params: params,
            softening_length: 1.0e-6, // 1 μm default
            force_accuracy: 1.0e-4,   // 0.01% accuracy
        }
    }
    
    /// Calculate gravitational force between two particles
    pub fn calculate_gravitational_force(
        &self,
        pos1: Vector3<f64>,
        pos2: Vector3<f64>,
        mass1: f64,
        mass2: f64,
    ) -> Vector3<f64> {
        use crate::constants::G;
        
        let r_vec = pos2 - pos1;
        let r_squared = r_vec.dot(&r_vec);
        let r = r_squared.sqrt();
        
        // Apply softening
        let r_soft = (r_squared + self.softening_length * self.softening_length).sqrt();
        
        // Newtonian force with softening
        let force_magnitude = G * mass1 * mass2 / (r_soft * r_soft);
        r_vec.normalize() * force_magnitude
    }
} 