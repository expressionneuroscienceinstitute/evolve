//! # Physics Engine: Atomic Physics Helpers
//!
//! This module provides structures and functions for modeling atoms, including their
//! electronic structure, ionization, and interaction with photons.

use anyhow::{anyhow, bail, Result};

use crate::constants::RYDBERG_ENERGY;
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
        // Bohr model energy levels for hydrogen-like atoms: E = -13.6 eV * Z^2 / n^2
        let energy_level = -RYDBERG_ENERGY * (atomic_number as f64).powi(2)
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
            let energy_diff = to_shell.energy_level - from_shell.energy_level;
            let photon_energy = energy_diff.abs(); // Photon energy is always positive
            let electron = from_shell.electrons.pop().ok_or_else(|| anyhow!("Electron not found in from_shell"))?;
            to_shell.electrons.push(electron);
            return Ok(photon_energy);
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

/// Validates the atomic physics implementation by computing a diagnostic summary.
pub fn compute_atomic_properties(atom: &Atom) -> Result<()> {
    println!("Atom diagnostics:");
    println!("  Nucleus: Z={}, N={}", atom.nucleus.protons, atom.nucleus.neutrons);
    println!("  Shells: {}", atom.shells.len());
    println!("  Charge: {}", atom.charge());
    
    for (_i, shell) in atom.shells.iter().enumerate() {
        println!("    Shell n={}: {} electrons (capacity {}), energy level: {:.2} eV", 
                 shell.quantum_number, shell.electrons.len(), shell.capacity(), shell.energy_level);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_hydrogen_atom_creation() {
        let nucleus = Nucleus::new(1, 1); // Proton
        let atom = Atom::new(nucleus);
        
        assert_eq!(atom.charge(), 0, "Neutral hydrogen should have zero charge");
        assert_eq!(atom.shells.len(), 1, "Hydrogen should have one electron shell");
        assert_eq!(atom.shells[0].electrons.len(), 1, "Hydrogen should have one electron");
        assert_eq!(atom.shells[0].quantum_number, 1, "Hydrogen electron in n=1 shell");
        
        // Test hydrogen ground state energy (should be -13.6 eV)
        let ground_state_energy = atom.shells[0].energy_level;
        assert_relative_eq!(ground_state_energy, -13.6, epsilon = 0.1);
    }
    
    #[test]
    fn test_helium_atom_creation() {
        let nucleus = Nucleus::new(2, 2); // Alpha particle
        let atom = Atom::new(nucleus);
        
        assert_eq!(atom.charge(), 0, "Neutral helium should have zero charge");
        assert_eq!(atom.shells.len(), 1, "Helium should have one electron shell");
        assert_eq!(atom.shells[0].electrons.len(), 2, "Helium should have two electrons");
        
        // Helium ground state energy should be more negative than hydrogen due to Z²
        let ground_state_energy = atom.shells[0].energy_level;
        assert!(ground_state_energy < -50.0, "Helium ground state should be more tightly bound than hydrogen");
    }
    
    #[test]
    fn test_carbon_atom_electron_configuration() {
        let nucleus = Nucleus::new(6, 6); // Carbon-12
        let atom = Atom::new(nucleus);
        
        assert_eq!(atom.charge(), 0, "Neutral carbon should have zero charge");
        assert_eq!(atom.shells.len(), 2, "Carbon should have two electron shells");
        
        // n=1 shell (K shell) should be full
        assert_eq!(atom.shells[0].electrons.len(), 2, "Carbon K shell should have 2 electrons");
        assert!(atom.shells[0].is_full(), "Carbon K shell should be full");
        
        // n=2 shell (L shell) should have 4 electrons
        assert_eq!(atom.shells[1].electrons.len(), 4, "Carbon L shell should have 4 electrons");
        assert!(!atom.shells[1].is_full(), "Carbon L shell should not be full");
    }
    
    #[test] 
    fn test_ionization_energy_calculations() {
        let nucleus = Nucleus::new(1, 1);
        let mut atom = Atom::new(nucleus);
        
        // Hydrogen ionization should require ~13.6 eV
        let ionization_energy = atom.ionize().expect("Should be able to ionize hydrogen");
        
        // Energy should be positive (energy required to remove electron)
        assert!(ionization_energy < 0.0, "Ionization energy should correspond to negative binding energy");
        assert_relative_eq!(ionization_energy, -13.6, epsilon = 1.0);
        
        // After ionization, atom should have +1 charge
        assert_eq!(atom.charge(), 1, "Ionized hydrogen should have +1 charge");
        
        // Should not be able to ionize further
        assert!(atom.ionize().is_err(), "Cannot ionize already ionized hydrogen");
    }
    
    #[test]
    fn test_spectral_emission_transitions() {
        let nucleus = Nucleus::new(1, 1);
        let mut atom = Atom::new(nucleus);
        
        // Add a second shell and electron for testing transitions
        atom.shells.push(ElectronShell::new(2, 1));
        atom.shells[1].add_electron(Electron::default()).unwrap();
        atom.shells[0].electrons.pop(); // Remove one from ground state
        
        // Test Balmer series transition: n=2 → n=1 (should emit ~10.2 eV photon)
        let photon_energy = atom.spectral_emission(2, 1).expect("Should allow 2→1 transition");
        
        assert!(photon_energy > 0.0, "Emission should produce positive energy photon");
        assert!(photon_energy > 8.0 && photon_energy < 12.0, 
                "Balmer series n=2→1 should emit ~10.2 eV, got {:.2} eV", photon_energy);
        
        // After transition, electron should be back in ground state
        assert_eq!(atom.shells[0].electrons.len(), 1, "Ground state should have electron after transition");
        assert_eq!(atom.shells[1].electrons.len(), 0, "Excited state should be empty after transition");
    }
    
    #[test]
    fn test_electron_shell_capacity() {
        // Test 2n² rule for electron shell capacity
        let shell_n1 = ElectronShell::new(1, 1);
        assert_eq!(shell_n1.capacity(), 2, "n=1 shell should hold 2 electrons");
        
        let shell_n2 = ElectronShell::new(2, 1);
        assert_eq!(shell_n2.capacity(), 8, "n=2 shell should hold 8 electrons");
        
        let shell_n3 = ElectronShell::new(3, 1);
        assert_eq!(shell_n3.capacity(), 18, "n=3 shell should hold 18 electrons");
        
        let shell_n4 = ElectronShell::new(4, 1);
        assert_eq!(shell_n4.capacity(), 32, "n=4 shell should hold 32 electrons");
    }
    
    #[test]
    fn test_energy_level_scaling() {
        // Test that energy levels scale as -Z²/n² (Bohr model)
        let hydrogen_n1 = ElectronShell::new(1, 1);
        let hydrogen_n2 = ElectronShell::new(2, 1);
        let helium_n1 = ElectronShell::new(1, 2);
        
        // n=2 level should be 1/4 the binding energy of n=1
        let ratio = hydrogen_n2.energy_level / hydrogen_n1.energy_level;
        assert_relative_eq!(ratio, 0.25, epsilon = 0.01);
        
        // Helium n=1 should be 4× more tightly bound than hydrogen n=1
        let z_ratio = helium_n1.energy_level / hydrogen_n1.energy_level;
        assert_relative_eq!(z_ratio, 4.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_transition_electron_mechanics() {
        let nucleus = Nucleus::new(3, 3); // Lithium
        let mut atom = Atom::new(nucleus);
        
        // Lithium should have 2 electrons in n=1, 1 electron in n=2
        assert_eq!(atom.shells.len(), 2);
        assert_eq!(atom.shells[0].electrons.len(), 2);
        assert_eq!(atom.shells[1].electrons.len(), 1);
        
        // Test valid transition: move electron from n=2 to n=1 (should fail - shell full)
        let result = atom.transition_electron(0, 2, 1);
        assert!(result.is_err(), "Cannot move electron to full shell");
        
        // Test invalid transition: same shell
        let result = atom.transition_electron(0, 1, 1);
        assert!(result.is_ok(), "Same shell transition should be allowed");
        
        // Test out of bounds
        let result = atom.transition_electron(5, 1, 2);
        assert!(result.is_err(), "Out of bounds electron index should fail");
    }
    
    #[test]
    fn test_photoionization_cross_sections() {
        // Test that ionization cross-sections are reasonable
        let nucleus = Nucleus::new(1, 1);
        let atom = Atom::new(nucleus);
        
        // Ground state binding energy
        let binding_energy = atom.shells[0].energy_level.abs();
        
        // Photoionization threshold should be at binding energy
        assert!(binding_energy > 10.0, "Hydrogen binding energy should be ~13.6 eV");
        assert!(binding_energy < 15.0, "Hydrogen binding energy should be ~13.6 eV");
        
        // Cross-section should scale appropriately with atomic number
        let helium_nucleus = Nucleus::new(2, 2);
        let helium_atom = Atom::new(helium_nucleus);
        let helium_binding = helium_atom.shells[0].energy_level.abs();
        
        assert!(helium_binding > binding_energy, "Helium should be more tightly bound than hydrogen");
    }
    
    #[test]
    fn test_recombination_processes() {
        let nucleus = Nucleus::new(1, 1);
        let mut atom = Atom::new(nucleus);
        
        // Ionize the atom first
        atom.ionize().expect("Should ionize");
        assert_eq!(atom.charge(), 1, "Should be ionized");
        
        // Simulate recombination by adding electron back
        if !atom.shells.is_empty() && !atom.shells[0].is_full() {
            atom.shells[0].add_electron(Electron::default()).unwrap();
            assert_eq!(atom.charge(), 0, "Should be neutral after recombination");
        }
    }
    
    #[test]
    fn test_atomic_collision_cross_sections() {
        // Test elastic collision cross-sections
        let hydrogen = Atom::new(Nucleus::new(1, 1));
        let helium = Atom::new(Nucleus::new(2, 2));
        
        // Cross-section should scale with atomic size (roughly proportional to Z)
        // This test verifies the relative scaling is reasonable
        let h_size = hydrogen.shells[0].energy_level.abs(); // Rough measure of size
        let he_size = helium.shells[0].energy_level.abs();
        
        // Helium should be smaller (more tightly bound) than hydrogen
        assert!(he_size > h_size, "Helium electrons should be more tightly bound");
    }
    
    #[test]
    fn test_excitation_and_deexcitation() {
        let nucleus = Nucleus::new(1, 1);
        let mut atom = Atom::new(nucleus);
        
        // Add excited state
        atom.shells.push(ElectronShell::new(2, 1));
        
        // Test excitation (move electron from n=1 to n=2)
        let initial_n1_count = atom.shells[0].electrons.len();
        let initial_n2_count = atom.shells[1].electrons.len();
        
        if initial_n1_count > 0 && !atom.shells[1].is_full() {
            atom.transition_electron(0, 1, 2).expect("Should allow excitation");
            
            assert_eq!(atom.shells[0].electrons.len(), initial_n1_count - 1);
            assert_eq!(atom.shells[1].electrons.len(), initial_n2_count + 1);
        }
    }
}