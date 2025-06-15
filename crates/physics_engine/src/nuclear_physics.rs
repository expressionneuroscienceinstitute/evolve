//! # Physics Engine: Nuclear Physics Utilities
//!
//! This module provides utilities for simulating nuclear physics phenomena, such as
//! nuclear binding energies, stability, and radioactive decay.

use anyhow::Result;
use rand::prelude::*;

/// Represents an atomic nucleus, composed of protons and neutrons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nucleus {
    pub protons: u32,  // Z
    pub neutrons: u32, // N
}

impl Nucleus {
    /// Creates a new nucleus.
    pub fn new(protons: u32, neutrons: u32) -> Self {
        Nucleus { protons, neutrons }
    }

    /// Returns the mass number (A), the total number of nucleons.
    pub fn mass_number(&self) -> u32 {
        self.protons + self.neutrons
    }

    /// Calculates the binding energy of the nucleus using the semi-empirical mass formula (SEMF).
    /// This provides an approximation of the energy required to disassemble the nucleus.
    /// Returns the binding energy in MeV (Mega-electronvolts).
    pub fn binding_energy(&self) -> f64 {
        let z = self.protons as f64;
        let n = self.neutrons as f64;
        let a = self.mass_number() as f64;

        if a == 0.0 { return 0.0; }

        // Coefficients for the SEMF (in MeV)
        const A_V: f64 = 15.75;  // Volume term
        const A_S: f64 = 17.8;   // Surface term
        const A_C: f64 = 0.711;  // Coulomb term
        const A_A: f64 = 23.7;   // Asymmetry term
        const A_P: f64 = 11.18;  // Pairing term

        let volume_term = A_V * a;
        let surface_term = A_S * a.powf(2.0 / 3.0);
        let coulomb_term = A_C * z * (z - 1.0) / a.powf(1.0 / 3.0);
        let asymmetry_term = A_A * (n - z).powi(2) / a;
        
        // Pairing term depends on the parity of Z and N
        let pairing_term = if self.protons % 2 == 0 && self.neutrons % 2 == 0 {
            A_P / a.powf(0.5)
        } else if self.protons % 2 != 0 && self.neutrons % 2 != 0 {
            -A_P / a.powf(0.5)
        } else {
            0.0
        };

        volume_term - surface_term - coulomb_term - asymmetry_term + pairing_term
    }

    /// Simulates one of the possible radioactive decay modes for an unstable nucleus.
    /// This is a simplified model and does not cover all decay types.
    /// Returns the decay product(s) if decay occurs.
    pub fn radioactive_decay<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<Vec<Nucleus>> {
        // A very simple probabilistic model for decay.
        // A real model would depend on the half-life of the specific nuclide.
        let decay_probability = 1.0 - (-self.binding_energy() / self.mass_number() as f64).exp();

        if rng.gen_bool(decay_probability) {
            // Simplified decay: assume alpha decay is a possibility for heavy nuclei.
            if self.protons > 82 {
                return Some(vec![
                    Nucleus::new(self.protons - 2, self.neutrons - 2), // Daughter nucleus
                    Nucleus::new(2, 2), // Alpha particle (Helium nucleus)
                ]);
            }
            // Could add beta decay, gamma decay, etc. here.
        }
        None
    }
}

/// Updates the state of a collection of nuclei, simulating radioactive decay.
pub fn update_nuclear_state(nuclei: &mut Vec<Nucleus>) -> Result<()> {
    let mut rng = thread_rng();
    let mut new_nuclei = Vec::new();
    
    nuclei.retain_mut(|nucleus| {
        if let Some(products) = nucleus.radioactive_decay(&mut rng) {
            new_nuclei.extend(products);
            false // Remove the decayed nucleus
        } else {
            true // Keep the stable nucleus
        }
    });

    nuclei.append(&mut new_nuclei);
    Ok(())
}