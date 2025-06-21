//! # Physics Engine: Phase Transitions Module
//!
//! This module provides a framework for modeling phase transitions, the process by
//! which a substance changes from one state of matter (solid, liquid, gas, plasma)
//! to another.

use anyhow::Result;
use crate::emergent_properties::{Temperature, Pressure, Density};
use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

// Global map tracking the last phase printed for each substance so we only log
// when the phase actually changes. `Lazy`+`Mutex` is sufficient because the
// overhead is negligible compared to a full simulation tick and it avoids
// bringing in a heavier dependency.
static LAST_PHASE: Lazy<Mutex<HashMap<String, Phase>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Represents the possible phases of matter for a substance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Solid,
    Liquid,
    Gas,
    Plasma,
}

/// A simplified model for determining the phase of a substance based on its
/// temperature, pressure, and density. This uses a generic phase diagram structure
/// with more realistic curve approximations.
pub struct PhaseTransitionModel {
    pub substance_name: String,
    pub triple_point_temp: f64,  // Triple point temperature (K)
    pub triple_point_pres: f64,  // Triple point pressure (Pa)
    pub critical_point_temp: f64, // Critical point temperature (K)
    pub critical_point_pres: f64, // Critical point pressure (Pa)
    // Approximated coefficients for Clausius-Clapeyron like curves (for water-like substance)
    pub sublimation_const: f64, // Constant for solid-gas boundary (ln P = A - B/T)
    pub sublimation_slope: f64,
    pub vaporization_const: f64, // Constant for liquid-gas boundary
    pub vaporization_slope: f64,
    pub melting_slope: f64, // Slope for solid-liquid boundary (P = P_ref + m(T - T_ref))
    pub melting_ref_point: (f64, f64), // (T_ref, P_ref) for melting curve
}

impl PhaseTransitionModel {
    /// Creates a new phase transition model for a specific substance.
    pub fn new(substance: &str) -> Result<Self> {
        match substance.to_lowercase().as_str() {
            "water" => Ok(PhaseTransitionModel {
                substance_name: "Water".to_string(),
                triple_point_temp: 273.16, // K (0.01 C)
                triple_point_pres: 611.65, // Pa (6.1165 mbar)
                critical_point_temp: 647.096, // K (373.946 C)
                critical_point_pres: 2.2064e7, // Pa (220.64 bar)
                sublimation_const: 23.32,
                sublimation_slope: 6111.9,
                vaporization_const: 23.2,
                vaporization_slope: 3816.4,
                melting_slope: -1.35e7, // Pa/K (negative slope for water)
                melting_ref_point: (273.15, 101325.0),
            }),
            "hydrogen" => Ok(PhaseTransitionModel {
                substance_name: "Hydrogen".to_string(),
                triple_point_temp: 13.8033,    // K
                triple_point_pres: 7.04e3,     // Pa
                critical_point_temp: 33.18,    // K
                critical_point_pres: 1.30e6,   // Pa
                // NOTE: Coefficients derived from Clausius-Clapeyron equation and empirical data.
                // Melting slope from dP/dT = ΔH_fus / (T * ΔV_fus).
                sublimation_const: 17.755,      // ln(Pa)
                sublimation_slope: 122.8,       // K
                vaporization_const: 16.734,     // ln(Pa)
                vaporization_slope: 108.7,      // K
                melting_slope: 2.53e6, // Pa/K, positive slope for hydrogen
                melting_ref_point: (13.8033, 7.04e3),
            }),
            // Future substances will be added here
            _ => Err(anyhow::anyhow!("Substance '{}' not supported.", substance)),
        }
    }

    /// Determines the phase of a substance based on temperature, pressure, and a simplified density check.
    pub fn determine_phase(&self, temperature: &Temperature, pressure: &Pressure, density: &Density) -> Phase {
        let temp = temperature.as_kelvin();
        let pres = pressure.as_pascals();
        let dens = density.as_kg_per_m3();

        // Check for supercritical fluid/plasma region first
        if temp > self.critical_point_temp && pres > self.critical_point_pres {
            return Phase::Plasma; // Simplification: assume supercritical is plasma-like in this context
        }

        // Calculate phase boundary pressures at current temperature based on Clausius-Clapeyron approximation
        let sublimation_pressure = (self.sublimation_const - self.sublimation_slope / temp).exp();
        let vaporization_pressure = (self.vaporization_const - self.vaporization_slope / temp).exp();

        // Calculate melting temperature at current pressure
        let melting_temperature = self.melting_ref_point.0 + (pres - self.melting_ref_point.1) / self.melting_slope;

        // Determine phase based on comparisons to boundary curves and triple point
        if temp < self.triple_point_temp && pres < sublimation_pressure {
            Phase::Gas // Below triple point, below sublimation curve
        } else if temp < self.triple_point_temp && pres >= sublimation_pressure {
            Phase::Solid // Below triple point, above sublimation curve
        } else if temp >= self.critical_point_temp && pres < self.critical_point_pres {
            Phase::Gas // Above critical temperature but below critical pressure (gas or supercritical gas)
        } else if temp < melting_temperature {
            Phase::Solid // Below melting curve
        } else if temp >= melting_temperature && pres > vaporization_pressure {
            Phase::Liquid // Above melting curve, above vaporization curve
        } else if temp >= melting_temperature && pres <= vaporization_pressure {
            Phase::Gas // Above melting curve, below vaporization curve
        } else {
            // Fallback for edge cases or intermediate states not perfectly covered
            // Use density as a tie-breaker or coarse indicator if other conditions are ambiguous.
            // This part might need further refinement for specific substance behavior.
            if dens > 900.0 { // Arbitrary density threshold for liquid (e.g., water)
                Phase::Liquid
            } else if dens < 10.0 { // Arbitrary density threshold for gas
                Phase::Gas
            } else {
                Phase::Plasma // Default or unknown high energy state
            }
        }
    }

    /// Notifies about a phase transition event.
    fn log_phase_transition(&self, phase: &Phase, temp: &Temperature, pres: &Pressure, density: &Density) {
        // Don't log transitions with zero or invalid values as they indicate uninitialized state
        let temp_k = temp.as_kelvin();
        let pres_pa = pres.as_pascals();
        let dens_kg_m3 = density.as_kg_per_m3();
        
        if temp_k <= 0.0 || pres_pa <= 0.0 || dens_kg_m3 <= 0.0 {
            return; // Skip logging for uninitialized or invalid states
        }
        
        // Check if the phase actually changed compared with last recorded for this substance
        let mut map = LAST_PHASE.lock().unwrap();
        let entry = map.entry(self.substance_name.clone()).or_insert(*phase);

        if *entry != *phase {
            // Update stored phase
            *entry = *phase;

            // Use proper logging instead of direct stdout to avoid interfering with CLI
            log::info!(
                "Phase transition: {} to {:?} at {:.2} K, {:.2} Pa, {:.2} kg/m³",
                self.substance_name, phase, temp_k, pres_pa, dens_kg_m3
            );
        }
    }
}

/// Evaluates and reports the phase of a system.
pub fn evaluate_phase_transitions(substance: &str, temp: Temperature, pres: Pressure, density: Density) -> Result<Phase> {
    let model = PhaseTransitionModel::new(substance)?;
    let phase = model.determine_phase(&temp, &pres, &density);
    model.log_phase_transition(&phase, &temp, &pres, &density);
    Ok(phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emergent_properties::{Temperature, Pressure, Density};

    #[test]
    fn test_water_phase_transitions() {
        let temp = Temperature::from_kelvin(300.0); // Room temperature
        let pres = Pressure::from_pascals(101325.0); // 1 atm
        let dens = Density::from_kg_per_m3(1000.0);
        let phase = evaluate_phase_transitions("water", temp, pres, dens).unwrap();
        assert_eq!(phase, Phase::Liquid);
    }

    #[test]
    fn test_hydrogen_phase_model_creation() {
        let model = PhaseTransitionModel::new("hydrogen");
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.substance_name, "Hydrogen");
        assert_eq!(model.triple_point_temp, 13.8033);
    }

    #[test]
    fn test_hydrogen_phase_determination() {
        // Test conditions around hydrogen's triple point (13.8 K, 7.04 kPa)
        
        // Solid phase: below triple point temperature and above sublimation pressure (which is very low)
        let temp_solid = Temperature::from_kelvin(10.0);
        let pres_solid = Pressure::from_pascals(8.0e3); 
        let dens_solid = Density::from_kg_per_m3(80.0);
        let phase_solid = evaluate_phase_transitions("hydrogen", temp_solid, pres_solid, dens_solid).unwrap();
        // With physically derived slopes, this test should now be accurate.
        assert_eq!(phase_solid, Phase::Solid);

        // Gas phase: above triple point temp, below vaporization pressure
        let temp_gas = Temperature::from_kelvin(20.0); 
        let pres_gas = Pressure::from_pascals(1.0e3); // Well below triple point pressure
        let dens_gas = Density::from_kg_per_m3(0.1);
        let phase_gas = evaluate_phase_transitions("hydrogen", temp_gas, pres_gas, dens_gas).unwrap();
        assert_eq!(phase_gas, Phase::Gas);

        // Liquid phase: between triple and critical points, at a pressure above vaporization curve
        let temp_liquid = Temperature::from_kelvin(25.0);
        let pres_liquid = Pressure::from_pascals(3.0e5); // Above vaporization pressure for 25K
        let dens_liquid = Density::from_kg_per_m3(70.0);
        let phase_liquid = evaluate_phase_transitions("hydrogen", temp_liquid, pres_liquid, dens_liquid).unwrap();
        // This test now uses a pressure that is correctly in the liquid phase for this temperature.
        assert_eq!(phase_liquid, Phase::Liquid);
    }
}