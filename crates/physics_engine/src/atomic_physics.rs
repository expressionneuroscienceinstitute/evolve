//! # Physics Engine: Atomic Physics Helpers
//!
//! This module provides structures and functions for modeling atoms, including their
//! electronic structure, ionization, and interaction with photons.

use anyhow::{anyhow, bail, Result};

use crate::constants::RYDBERG_CONSTANT;
use crate::Electron;
use crate::nuclear_physics::Nucleus;

/// Represents an electron shell in an atom.
/// This is a simplified model based on principal quantum numbers.
#[derive(Debug, Clone)]
pub struct ElectronShell {
    /// The principal quantum number (n=1, 2, 3...).
    pub quantum_number: u32,
    /// The electrons currently in this shell.
    pub electrons: Vec<Electron>,
    /// The energy level of this shell in electron-volts (eV).
    pub energy_level: f64,
}

impl ElectronShell {
    /// Creates a new electron shell.
    pub fn new(quantum_number: u32, atomic_number: u32) -> Self {
        // Bohr model energy levels for hydrogen-like atoms: E = -13.6 * Z^2 / n^2
        let energy_level = -RYDBERG_CONSTANT * (atomic_number as f64).powi(2)
            / (quantum_number as f64).powi(2);
        ElectronShell {
            quantum_number,
            electrons: Vec::new(),
            energy_level,
        }
    }

    /// The maximum number of electrons a shell can hold (2n^2).
    pub fn capacity(&self) -> usize {
        2 * (self.quantum_number as usize).pow(2)
    }

    /// Checks if the shell is at its maximum capacity.
    pub fn is_full(&self) -> bool {
        self.electrons.len() >= self.capacity()
    }

    /// Adds an electron to the shell if there is capacity.
    pub fn add_electron(&mut self, electron: Electron) -> Result<()> {
        if self.electrons.len() < self.capacity() {
            self.electrons.push(electron);
            Ok(())
        } else {
            Err(anyhow!("Electron shell is full."))
        }
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
            if n as usize > atom.shells.len() {
                atom.shells.push(ElectronShell::new(n, nucleus.protons));
            }
            let shell = &mut atom.shells[(n - 1) as usize];
            let electrons_to_add = remaining_electrons.min(shell.capacity() as u32 - shell.electrons.len() as u32);
            
            for _ in 0..electrons_to_add {
                shell.add_electron(Electron::default()).unwrap();
            }

            remaining_electrons -= electrons_to_add;
            n += 1;
        }
        atom
    }

    /// Calculates the charge of the atom (ion charge).
    pub fn charge(&self) -> i32 {
        let total_electrons: u32 = self.shells.iter().map(|s| s.electrons.len() as u32).sum();
        self.nucleus.protons as i32 - total_electrons as i32
    }

    /// Simulates the ionization of the atom by removing an electron.
    /// This requires energy and typically happens to the outermost electron.
    /// Returns the energy required for ionization (a positive value).
    pub fn ionize(&mut self) -> Result<f64> {
        if let Some(outer_shell) = self.shells.last_mut() {
            if !outer_shell.electrons.is_empty() {
                let energy = outer_shell.energy_level;
                outer_shell.electrons.pop();
                return Ok(energy);
            }
        }
        Err(anyhow!("Atom has no electrons to ionize."))
    }

    /// Simulates spectral emission when an electron transitions to a lower energy level.
    /// Returns the energy of the emitted photon.
    pub fn spectral_emission(&mut self, from_shell_n: u32, to_shell_n: u32) -> Result<f64> {
        if from_shell_n <= to_shell_n {
            return Err(anyhow!("Electron must transition to a lower energy shell."));
        }

        let from_idx = (from_shell_n - 1) as usize;
        let to_idx = (to_shell_n - 1) as usize;

        if from_idx >= self.shells.len() || to_idx >= self.shells.len() {
            bail!("Shell index out of bounds");
        }
        
        // Use split_at_mut to safely get two mutable references
        let (shells1, shells2) = self.shells.split_at_mut(from_idx.max(to_idx));
        let (from_shell, to_shell) = if from_idx < to_idx {
            (&mut shells1[from_idx], &mut shells2[0])
        } else {
            (&mut shells2[0], &mut shells1[to_idx])
        };

        if !from_shell.electrons.is_empty() && !to_shell.is_full() {
            let energy = to_shell.energy_level - from_shell.energy_level;
            let electron = from_shell.electrons.pop().ok_or_else(|| anyhow!("Electron not found in from_shell"))?;
            to_shell.electrons.push(electron);
            return Ok(energy);
        }
        
        Err(anyhow!("Invalid shell transition for spectral emission."))
    }

    /// This simulates the photoelectric effect or electron capture.
    pub fn transition_electron(
        &mut self,
        electron_index: usize,
        from_shell_n: u32,
        to_shell_n: u32,
    ) -> Result<()> {
        if from_shell_n == to_shell_n {
            return Ok(()); // No transition
        }

        if from_shell_n > self.shells.len() as u32 || to_shell_n > self.shells.len() as u32 {
            bail!("Shell index out of bounds");
        }

        let (from_idx, to_idx) = (
            (from_shell_n - 1) as usize,
            (to_shell_n - 1) as usize,
        );

        if from_idx >= self.shells.len() || to_idx >= self.shells.len() {
            bail!("Shell index out of bounds");
        }
        
        // Use split_at_mut to safely get two mutable references
        let (shells1, shells2) = self.shells.split_at_mut(from_idx.max(to_idx));
        let (from_shell, to_shell) = if from_idx < to_idx {
            (&mut shells1[from_idx], &mut shells2[0])
        } else {
            (&mut shells2[0], &mut shells1[to_idx])
        };

        let _electron = from_shell
            .electrons
            .get(electron_index)
            .ok_or_else(|| anyhow!("Electron index out of bounds in from_shell"))?;

        let energy_diff = to_shell.energy_level - from_shell.energy_level;

        if to_shell.electrons.len() >= to_shell.capacity() {
            bail!("Target shell is full");
        }

        // Emit or absorb a photon corresponding to the transition.
        // Negative energy_diff → emission, positive → absorption.
        if energy_diff < 0.0 {
            // Emission: create a photon with |energy_diff| eV
            log::debug!("Photon emitted: {:.4} eV", -energy_diff);
        } else if energy_diff > 0.0 {
            // Absorption: photon of energy_diff eV is absorbed
            log::debug!("Photon absorbed: {:.4} eV", energy_diff);
        }

        let electron = from_shell.electrons.remove(electron_index);
        to_shell.electrons.push(electron);

        println!(
            "Electron transitioned from n={} to n={}, energy change: {}",
            from_shell_n, to_shell_n, energy_diff
        );

        Ok(())
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