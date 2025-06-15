//! # Physics Engine: Atomic Physics Helpers
//!
//! This module provides structures and functions for modeling atoms, including their
//! electronic structure, ionization, and interaction with photons.

use anyhow::Result;
use crate::nuclear_physics::Nucleus;

/// Represents an electron shell in an atom.
/// This is a simplified model based on principal quantum numbers.
#[derive(Debug, Clone)]
pub struct ElectronShell {
    pub quantum_number: u32, // n
    pub electrons: u32,
}

impl ElectronShell {
    /// The maximum number of electrons a shell can hold (2n^2).
    pub fn capacity(&self) -> u32 {
        2 * self.quantum_number.pow(2)
    }

    /// Checks if the shell is full.
    pub fn is_full(&self) -> bool {
        self.electrons >= self.capacity()
    }
}

/// Represents an atom, consisting of a nucleus and its electron shells.
#[derive(Debug, Clone)]
pub struct Atom {
    pub nucleus: Nucleus,
    pub shells: Vec<ElectronShell>,
}

impl Atom {
    /// Creates a new, neutral atom with a default electron configuration.
    pub fn new(nucleus: Nucleus) -> Self {
        let mut atom = Atom { nucleus: nucleus.clone(), shells: Vec::new() };
        let mut remaining_electrons = nucleus.protons;
        let mut n = 1;

        while remaining_electrons > 0 {
            let mut shell = ElectronShell { quantum_number: n, electrons: 0 };
            let capacity = shell.capacity();
            let electrons_to_add = remaining_electrons.min(capacity);
            shell.electrons = electrons_to_add;
            atom.shells.push(shell);
            remaining_electrons -= electrons_to_add;
            n += 1;
        }
        atom
    }

    /// Calculates the charge of the atom (ion charge).
    pub fn charge(&self) -> i32 {
        let total_electrons: u32 = self.shells.iter().map(|s| s.electrons).sum();
        self.nucleus.protons as i32 - total_electrons as i32
    }

    /// Simulates the ionization of the atom by removing an electron.
    /// This requires energy and typically happens to the outermost electron.
    /// Returns the energy required for ionization (a positive value).
    pub fn ionize(&mut self) -> Result<f64> {
        if let Some(outer_shell) = self.shells.last_mut() {
            if outer_shell.electrons > 0 {
                outer_shell.electrons -= 1;
                // Simplified ionization energy calculation (placeholder).
                // A real calculation would be much more complex (e.g., using Hartree-Fock).
                let energy = 13.6 * (self.nucleus.protons as f64).powi(2) / (outer_shell.quantum_number as f64).powi(2);
                return Ok(energy);
            }
        }
        Err(anyhow::anyhow!("Atom has no electrons to ionize."))
    }

    /// Simulates spectral emission when an electron transitions to a lower energy level.
    /// Returns the energy of the emitted photon.
    pub fn spectral_emission(&mut self, from_shell_n: u32, to_shell_n: u32) -> Result<f64> {
        if from_shell_n <= to_shell_n {
            return Err(anyhow::anyhow!("Electron must transition to a lower energy shell."));
        }

        let from_shell = self.shells.get_mut((from_shell_n - 1) as usize);
        let to_shell = self.shells.get_mut((to_shell_n - 1) as usize);

        if let (Some(from), Some(to)) = (from_shell, to_shell) {
            if from.electrons > 0 && !to.is_full() {
                from.electrons -= 1;
                to.electrons += 1;

                // Rydberg formula for energy of emitted photon (for hydrogen-like atoms).
                const RYDBERG_CONSTANT: f64 = 13.6; // eV
                let z = self.nucleus.protons as f64;
                let n1 = to_shell_n as f64;
                let n2 = from_shell_n as f64;
                let energy = RYDBERG_CONSTANT * z.powi(2) * (1.0 / n1.powi(2) - 1.0 / n2.powi(2));
                return Ok(energy);
            }
        }
        Err(anyhow::anyhow!("Invalid shell transition for spectral emission."))
    }
}

/// A placeholder function to compute various atomic properties.
pub fn compute_atomic_properties(atom: &Atom) -> Result<()> {
    log::info!(
        "Atom properties: Z={}, A={}, Charge={}",
        atom.nucleus.protons,
        atom.nucleus.mass_number(),
        atom.charge()
    );
    Ok(())
}