//! # Physics Engine: Phase Transitions Module
//!
//! This module provides a framework for modeling phase transitions, the process by
//! which a substance changes from one state of matter (solid, liquid, gas, plasma)
//! to another.

use anyhow::Result;
use crate::emergent_properties::{Temperature, Pressure};

/// Represents the possible phases of matter for a substance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Solid,
    Liquid,
    Gas,
    Plasma,
}

/// A simplified model for determining the phase of a substance based on its
/// temperature and pressure. This uses a generic phase diagram structure.
pub struct PhaseTransitionModel {
    // These points define the boundaries on the phase diagram.
    // (Temperature in Kelvin, Pressure in Pascals)
    pub triple_point: (f64, f64),
    pub critical_point: (f64, f64),
}

impl PhaseTransitionModel {
    /// Creates a new phase transition model for a substance.
    /// This would typically be initialized with known data for a specific material.
    pub fn new(triple_point: (f64, f64), critical_point: (f64, f64)) -> Self {
        PhaseTransitionModel { triple_point, critical_point }
    }

    /// Determines the phase of a substance based on temperature and pressure.
    pub fn determine_phase(&self, temperature: &Temperature, pressure: &Pressure) -> Phase {
        let temp = temperature.as_kelvin();
        let pres = pressure.as_pascals();
        
        let (triple_temp, triple_pres) = self.triple_point;
        let (crit_temp, crit_pres) = self.critical_point;

        if temp > crit_temp && pres > crit_pres {
            // Above the critical point, we have a supercritical fluid, often grouped with plasma.
            Phase::Plasma
        } else if pres > crit_pres {
            Phase::Liquid
        } else if temp > crit_temp {
            Phase::Gas
        } else {
            // Simplified logic for below the critical point.
            if temp < triple_temp && pres < triple_pres {
                Phase::Gas // Or could be solid depending on the sublimation curve.
            } else if temp < triple_temp {
                Phase::Solid
            } else if pres < triple_pres {
                 Phase::Gas
            } else {
                // In the region between the triple and critical points.
                // A more accurate model would use the vaporization curve.
                // For simplicity, we'll make a rough distinction.
                // This is not physically accurate, just a placeholder.
                let vaporization_pressure_approx = triple_pres + (crit_pres - triple_pres) * (temp - triple_temp) / (crit_temp - triple_temp);
                if pres > vaporization_pressure_approx {
                    Phase::Liquid
                } else {
                    Phase::Gas
                }
            }
        }
    }

    /// Notifies about a phase transition event.
    fn log_phase_transition(&self, phase: &Phase, temp: &Temperature, pres: &Pressure) {
        println!(
            "Phase transition to {:?}. Conditions: {:.2} K, {:.2} Pa.",
            phase,
            temp.as_kelvin(),
            pres.as_pascals()
        );
    }
}

/// Evaluates and reports the phase of a system.
pub fn evaluate_phase_transitions(model: &PhaseTransitionModel, temp: Temperature, pres: Pressure) -> Result<Phase> {
    let phase = model.determine_phase(&temp, &pres);
    model.log_phase_transition(&phase, &temp, &pres);
    Ok(phase)
}