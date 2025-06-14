//! Nuclear fusion engine for stellar nuclear fusion processes

use crate::types::*;
use crate::{Result, SimError};

/// Fusion engine for stellar nuclear processes
pub struct FusionEngine {
    fusion_threshold_solar_masses: f64,
}

impl FusionEngine {
    pub fn new() -> Self {
        Self {
            fusion_threshold_solar_masses: crate::constants::sim::FUSION_THRESHOLD_M_SOL,
        }
    }
    
    /// Check if a star can sustain fusion
    pub fn can_fuse(&self, mass_solar: f64) -> bool {
        mass_solar >= self.fusion_threshold_solar_masses
    }
    
    /// Calculate fusion rate for a star
    pub fn calculate_fusion_rate(&self, mass_solar: f64, temperature_k: f64) -> f64 {
        if !self.can_fuse(mass_solar) {
            return 0.0;
        }
        
        // Simplified pp-chain fusion rate
        let base_rate = 1e-20; // kg/s for 1 solar mass at 1.5e7 K
        let mass_factor = mass_solar.powf(4.0);
        let temp_factor = (temperature_k / 1.5e7).powf(4.0);
        
        base_rate * mass_factor * temp_factor
    }
}