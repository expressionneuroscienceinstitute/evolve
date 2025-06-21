//! # Physics Engine: Atomic Physics Helpers
//!
//! This module provides structures and functions for modeling atoms, including their
//! electronic structure, ionization, and interaction with photons.

//! Routines for atomic-level phenomena.

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

    /// Estimates the atomic radius using the Bohr model radius of the outermost shell.
    /// r_n = n^2 * a_0 / Z
    /// where a_0 is the Bohr radius (~5.29e-11 m).
    pub fn atomic_radius(&self) -> f64 {
        const BOHR_RADIUS: f64 = 5.29177e-11; // meters
        if let Some(outer_shell) = self.shells.last() {
            if !outer_shell.electrons.is_empty() {
                let n = outer_shell.quantum_number as f64;
                let z = self.nucleus.protons as f64;
                return n.powi(2) * BOHR_RADIUS / z;
            }
        }
        // For a bare nucleus, return a default small radius.
        BOHR_RADIUS / (self.nucleus.protons as f64)
    }
}

/// Calculates the photoionization cross-section for a hydrogen-like atom.
///
/// This uses a simplified model, valid for photon energies above the ionization threshold.
/// The cross-section is given by σ(E) ≈ σ_L * (I/E)^3, where σ_L is the cross-section
/// at the Lyman limit (ionization threshold).
///
/// # Arguments
/// * `atomic_number` - The atomic number (Z) of the nucleus.
/// * `photon_energy_ev` - The energy of the incoming photon in electron-volts (eV).
///
/// # Returns
/// The photoionization cross-section in square meters (m^2), or 0 if the photon energy
/// is below the ionization threshold.
pub fn photoionization_cross_section(atomic_number: u32, photon_energy_ev: f64) -> f64 {
    // Ionization energy for a hydrogen-like atom: I = 13.6 eV * Z^2
    let ionization_energy = RYDBERG_ENERGY * (atomic_number as f64).powi(2);

    if photon_energy_ev < ionization_energy {
        return 0.0; // Photon energy is not sufficient to ionize the atom.
    }

    // Cross-section at the ionization threshold for hydrogen (Z=1).
    // From NIST data, this is approximately 6.3e-18 cm^2.
    const SIGMA_L_H: f64 = 6.3e-18; // cm^2

    // The cross-section for a hydrogen-like ion scales as Z^-2.
    let sigma_l = SIGMA_L_H / (atomic_number as f64).powi(2);

    // Calculate the cross-section using the approximate formula σ ~ (I/E)^3
    let cross_section_cm2 = sigma_l * (ionization_energy / photon_energy_ev).powi(3);

    // Convert from cm^2 to m^2 (1 m^2 = 10^4 cm^2)
    cross_section_cm2 / 1.0e4
}

/// Calculates the radiative recombination rate for hydrogen.
///
/// This rate represents the number of recombination events per unit volume per second.
/// The formula used is Rate = α(T) * n_e * n_p, where α(T) is the recombination
/// coefficient, which is temperature-dependent.
///
/// # Arguments
/// * `electron_density` - Number of electrons per cubic meter (m^-3).
/// * `ion_density` - Number of ions (protons) per cubic meter (m^-3).
/// * `temperature` - Gas temperature in Kelvin.
///
/// # Returns
/// The recombination rate in events per cubic meter per second.
pub fn radiative_recombination_rate(electron_density: f64, ion_density: f64, temperature: f64) -> f64 {
    // Recombination coefficient α(T) for hydrogen.
    // This is an approximation for Case B recombination, valid for T ~ 10^4 K.
    // α_B(T) ≈ 2.59e-13 * (T / 10^4 K)^-0.7 cm^3 s^-1
    let alpha_cm3_s = 2.59e-13 * (temperature / 1.0e4).powf(-0.7);
    
    // Convert α from cm^3 s^-1 to m^3 s^-1 (1 m^3 = 10^6 cm^3)
    let alpha_m3_s = alpha_cm3_s / 1.0e6;

    // Rate = α * n_e * n_p
    alpha_m3_s * electron_density * ion_density
}

/// Validates the atomic physics implementation by computing a diagnostic summary.
pub fn compute_atomic_properties(atom: &Atom) -> Result<()> {
    println!("Atom diagnostics:");
    println!("  Nucleus: Z={}, N={}", atom.nucleus.protons, atom.nucleus.neutrons);
    println!("  Shells: {}", atom.shells.len());
    println!("  Charge: {}", atom.charge());
    
    for shell in atom.shells.iter() {
        println!("    Shell n={}: {} electrons (capacity {}), energy level: {:.2} eV", 
                 shell.quantum_number, shell.electrons.len(), shell.capacity(), shell.energy_level);
    }
    
    Ok(())
}

/// Calculates the geometric cross-section for an elastic collision between two atoms.
/// This is a simple approximation based on the sum of their atomic radii.
pub fn elastic_collision_cross_section(atom1: &Atom, atom2: &Atom) -> f64 {
    let r1 = atom1.atomic_radius();
    let r2 = atom2.atomic_radius();
    std::f64::consts::PI * (r1 + r2).powi(2)
}

/// Calculates the binding energy of an electron in a hydrogen-like atom.
pub fn hydrogen_like_binding_energy(z: u32, n: u32) -> f64 {
    // Using the Bohr model: E_n = - Z^2 * R_H / n^2
    // where R_H ≈ 13.6 eV is the Rydberg constant for hydrogen.
    const RYDBERG_EV: f64 = 13.605693122994; // more precise value

    if n == 0 {
        return 0.0; // avoid division by zero; caller should ensure n>=1
    }

    let z_f = z as f64;
    let n_f = n as f64;
    -RYDBERG_EV * z_f.powi(2) / n_f.powi(2)
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
        // Test photoionization cross-section for hydrogen.
        // The threshold energy is exactly the Rydberg energy: 13.605693122994 eV
        let h_ionization_energy = RYDBERG_ENERGY; // Use exact value

        // At threshold, the cross-section should be ~6.3e-22 m^2.
        let cs_at_threshold = photoionization_cross_section(1, h_ionization_energy);
        assert_relative_eq!(cs_at_threshold, 6.3e-22, epsilon = 1e-23);

        // For a photon with twice the ionization energy, the cross-section should be 1/8.
        let cs_at_2x = photoionization_cross_section(1, h_ionization_energy * 2.0);
        assert_relative_eq!(cs_at_2x, 6.3e-22 / 8.0, epsilon = 1e-24);
        
        // Below threshold, cross-section should be zero.
        let cs_below_threshold = photoionization_cross_section(1, h_ionization_energy - 1.0);
        assert_eq!(cs_below_threshold, 0.0);

        // Test for Helium (Z=2). Ionization energy is 4x hydrogen's.
        // Threshold cross-section should be sigma_L / Z^2
        let he_ionization_energy = RYDBERG_ENERGY * 4.0;
        let cs_he_threshold = photoionization_cross_section(2, he_ionization_energy);
        assert_relative_eq!(cs_he_threshold, 6.3e-22 / 4.0, epsilon = 1e-24);
    }
    
    #[test]
    fn test_recombination_processes() {
        // Test radiative recombination rate for typical astrophysical conditions.
        let electron_density_m3 = 1.0e6; // 1 electron per cm^3
        let ion_density_m3 = 1.0e6;      // 1 ion per cm^3
        let temperature_k = 1.0e4;       // 10,000 K

        let rate = radiative_recombination_rate(electron_density_m3, ion_density_m3, temperature_k);

        // At 10^4 K, α ≈ 2.59e-13 cm^3/s = 2.59e-19 m^3/s
        // Rate = 2.59e-19 * 1e6 * 1e6 = 2.59e-7 events / m^3 / s
        let expected_alpha = 2.59e-19;
        let expected_rate = expected_alpha * electron_density_m3 * ion_density_m3;

        assert_relative_eq!(rate, expected_rate, epsilon = 1e-9);

        // Test temperature dependence (rate should increase as T decreases)
        let rate_at_lower_temp = radiative_recombination_rate(electron_density_m3, ion_density_m3, temperature_k / 2.0);
        assert!(rate_at_lower_temp > rate, "Recombination rate should increase at lower temperatures");
    }
    
    #[test]
    fn test_atomic_collision_cross_sections() {
        // Test elastic collision cross-sections using a geometric model.
        let hydrogen = Atom::new(Nucleus::new(1, 0)); // Z=1
        let helium = Atom::new(Nucleus::new(2, 2));   // Z=2

        // Radii: r_H ~ 1^2 * a_0 / 1 = a_0
        //        r_He ~ 1^2 * a_0 / 2 = 0.5 * a_0
        let h_radius = hydrogen.atomic_radius();
        let he_radius = helium.atomic_radius();

        assert_relative_eq!(h_radius, 5.29177e-11, epsilon = 1e-15);
        assert_relative_eq!(he_radius, 5.29177e-11 / 2.0, epsilon = 1e-15);

        // Cross section for H-H collision
        let cs_hh = elastic_collision_cross_section(&hydrogen, &hydrogen);
        let expected_cs_hh = std::f64::consts::PI * (h_radius * 2.0).powi(2);
        assert_relative_eq!(cs_hh, expected_cs_hh, epsilon = 1e-25);

        // Cross section for H-He collision
        let cs_h_he = elastic_collision_cross_section(&hydrogen, &helium);
        let expected_cs_h_he = std::f64::consts::PI * (h_radius + he_radius).powi(2);
        assert_relative_eq!(cs_h_he, expected_cs_h_he, epsilon = 1e-25);

        // A larger atom (e.g. Lithium) should have a larger cross-section.
        let lithium = Atom::new(Nucleus::new(3, 4)); // Z=3, outer shell n=2
        let li_radius = lithium.atomic_radius();
        assert!(li_radius > h_radius, "Lithium should be larger than hydrogen");
        
        let cs_h_li = elastic_collision_cross_section(&hydrogen, &lithium);
        assert!(cs_h_li > cs_h_he, "H-Li cross section should be larger than H-He");
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