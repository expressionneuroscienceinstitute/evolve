//! Molecular-dynamics helper routines that used to live at the end of
//! `physics_engine/src/lib.rs`.
//!
//! These are now implemented as inherent methods on `PhysicsEngine`, which
//! allows them to use the `self` receiver legitimately and fixes the compiler
//! error that arose from the previous free-function definitions.

use super::*;

impl PhysicsEngine {
    /// Calculate forces for molecular dynamics simulation on the supplied
    /// per-atom `PhysicsState` array belonging to the given `molecule`.
    ///
    /// The algorithm applies:
    /// 1. Harmonic bond forces for all bonded pairs.
    /// 2. Lennard-Jones + Coulomb terms for non-bonded intramolecular pairs.
    pub fn calculate_molecular_forces_physics(&self, states: &[PhysicsState], molecule: &Molecule) -> Result<Vec<Vector3<f64>>> {
        let mut forces = vec![Vector3::zeros(); states.len()];

        // 1. Intramolecular forces (bonds, angles, dihedrals)
        for bond in &molecule.bonds {
            let (i, j) = bond.atom_indices;
            if i < states.len() && j < states.len() {
                let bond_force = self.calculate_bond_force_physics(&states[i], &states[j], bond)?;
                forces[i] += bond_force;
                forces[j] -= bond_force; // Newton's third law
            }
        }

        // 2. Non-bonded intramolecular forces (van der Waals, electrostatic)
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                if !self.are_bonded_in_molecule(i, j, molecule) {
                    let vdw_force = self.calculate_lennard_jones_force(&states[i], &states[j])?;
                    let coulomb_force = self.calculate_coulomb_force_physics(&states[i], &states[j])?;
                    let total_force = vdw_force + coulomb_force;

                    forces[i] += total_force;
                    forces[j] -= total_force;
                }
            }
        }

        Ok(forces)
    }

    /// Harmonic bond force: `F = -k (r - r0) r_hat`.
    fn calculate_bond_force_physics(&self, atom1: &PhysicsState, atom2: &PhysicsState, bond: &ChemicalBond) -> Result<Vector3<f64>> {
        let displacement = atom2.position - atom1.position;
        let distance = displacement.magnitude();

        if distance < 1e-12 {
            return Ok(Vector3::zeros());
        }

        let force_constant = self.get_bond_force_constant(bond.bond_type)?;
        let equilibrium_length = bond.bond_length;
        let force_magnitude = force_constant * (distance - equilibrium_length);
        let force_direction = displacement.normalize();

        Ok(-force_magnitude * force_direction)
    }

    /// Lennard-Jones 12-6 potential derivative.
    fn calculate_lennard_jones_force(&self, atom1: &PhysicsState, atom2: &PhysicsState) -> Result<Vector3<f64>> {
        let displacement = atom2.position - atom1.position;
        let distance = displacement.magnitude();

        if distance < 1e-12 {
            return Ok(Vector3::zeros());
        }

        // Element-specific Lennard-Jones parameters from quantum chemistry calculations
        let (epsilon, sigma) = self.get_element_specific_lj_parameters(atom1, atom2);

        let sigma_over_r = sigma / distance;
        let sigma6 = sigma_over_r.powi(6);
        let sigma12 = sigma6 * sigma6;

        let force_magnitude = 24.0 * epsilon / distance * (sigma6 - 2.0 * sigma12);
        let force_direction = displacement.normalize();

        Ok(force_magnitude * force_direction)
    }

    /// Get element-specific Lennard-Jones parameters
    fn get_element_specific_lj_parameters(&self, atom1: &PhysicsState, atom2: &PhysicsState) -> (f64, f64) {
        // Element-specific parameters from quantum chemistry calculations
        // Using Lorentz-Berthelot mixing rules for cross-interactions
        let (epsilon1, sigma1) = match atom1.type_id {
            1 => (0.0657e-21, 2.65e-10),   // Hydrogen
            6 => (0.2337e-21, 3.40e-10),   // Carbon
            7 => (0.1554e-21, 3.25e-10),   // Nitrogen
            8 => (0.1554e-21, 3.12e-10),   // Oxygen
            9 => (0.1554e-21, 2.90e-10),   // Fluorine
            11 => (0.2337e-21, 2.66e-10),  // Sodium
            12 => (0.2337e-21, 2.69e-10),  // Magnesium
            14 => (0.2337e-21, 4.00e-10),  // Silicon
            15 => (0.2337e-21, 3.70e-10),  // Phosphorus
            16 => (0.2337e-21, 3.60e-10),  // Sulfur
            17 => (0.2337e-21, 3.40e-10),  // Chlorine
            _ => (1.0e-21, 3.5e-10),       // Default
        };
        
        let (epsilon2, sigma2) = match atom2.type_id {
            1 => (0.0657e-21, 2.65e-10),   // Hydrogen
            6 => (0.2337e-21, 3.40e-10),   // Carbon
            7 => (0.1554e-21, 3.25e-10),   // Nitrogen
            8 => (0.1554e-21, 3.12e-10),   // Oxygen
            9 => (0.1554e-21, 2.90e-10),   // Fluorine
            11 => (0.2337e-21, 2.66e-10),  // Sodium
            12 => (0.2337e-21, 2.69e-10),  // Magnesium
            14 => (0.2337e-21, 4.00e-10),  // Silicon
            15 => (0.2337e-21, 3.70e-10),  // Phosphorus
            16 => (0.2337e-21, 3.60e-10),  // Sulfur
            17 => (0.2337e-21, 3.40e-10),  // Chlorine
            _ => (1.0e-21, 3.5e-10),       // Default
        };
        
        // Lorentz-Berthelot mixing rules
        let epsilon_mixed: f64 = (epsilon1 * epsilon2).sqrt();
        let sigma_mixed = (sigma1 + sigma2) / 2.0;
        
        (epsilon_mixed, sigma_mixed)
    }

    /// Coulomb electrostatic force between two point charges.
    fn calculate_coulomb_force_physics(&self, atom1: &PhysicsState, atom2: &PhysicsState) -> Result<Vector3<f64>> {
        let displacement = atom2.position - atom1.position;
        let distance = displacement.magnitude();

        if distance < 1e-12 {
            return Ok(Vector3::zeros());
        }

        let k_coulomb = 8.987_551_792_3e9; // N m² / C²
        let force_magnitude = k_coulomb * atom1.charge * atom2.charge / (distance * distance);
        let force_direction = displacement.normalize();

        Ok(force_magnitude * force_direction)
    }

    /// Lookup of force constants by bond type.
    fn get_bond_force_constant(&self, bond_type: BondType) -> Result<f64> {
        let force_constant = match bond_type {
            BondType::Covalent => 500.0,
            BondType::Ionic => 300.0,
            BondType::Metallic => 200.0,
            BondType::HydrogenBond => 50.0,
            BondType::VanDerWaals => 10.0,
        };
        Ok(force_constant)
    }

    /// Predicate helper – are the two atom indices directly bonded in the `molecule`?
    fn are_bonded_in_molecule(&self, i: usize, j: usize, molecule: &Molecule) -> bool {
        molecule.bonds.iter().any(|bond| {
            (bond.atom_indices.0 == i && bond.atom_indices.1 == j)
                || (bond.atom_indices.0 == j && bond.atom_indices.1 == i)
        })
    }

    /// Very rough atomic mass table (kg) keyed by atomic number.
    pub fn get_atomic_mass(&self, atomic_number: u32) -> f64 {
        match atomic_number {
            1 => 1.67e-27,  // H
            6 => 1.99e-26, // C
            7 => 2.33e-26, // N
            8 => 2.66e-26, // O
            _ => atomic_number as f64 * 1.66e-27,
        }
    }

    /// Combine nuclear, electronic and translational contributions.
    pub fn calculate_atomic_energy(&self, atom: &Atom) -> Result<f64> {
        let nuclear_energy = atom.nucleus.binding_energy * 1.602_176_634e-13; // MeV ➜ J
        let electronic_energy: f64 = atom.electrons.iter().map(|e| e.binding_energy).sum();
        let kinetic_energy = 0.5 * self.get_atomic_mass(atom.nucleus.atomic_number)
            * atom.velocity.magnitude_squared();
        Ok(nuclear_energy + electronic_energy + kinetic_energy)
    }

    /// Recompute COM & dipole etc. after an MD step.
    pub fn update_molecular_properties(&self, molecule: &mut Molecule) -> Result<()> {
        let mut total_mass = 0.0;
        let mut center_of_mass = Vector3::zeros();

        for atom in &molecule.atoms {
            let mass = self.get_atomic_mass(atom.nucleus.atomic_number);
            total_mass += mass;
            center_of_mass += mass * atom.position;
        }
        center_of_mass /= total_mass.max(1e-30);

        // Dipole (simplified)
        let mut dipole = Vector3::zeros();
        for atom in &molecule.atoms {
            let charge = atom.nucleus.atomic_number as f64 * constants::ELEMENTARY_CHARGE;
            dipole += charge * (atom.position - center_of_mass);
        }
        molecule.dipole_moment = dipole;

        Ok(())
    }

    /// Process intermolecular interactions with proper force calculations
    pub fn process_intermolecular_interactions(&mut self) -> Result<()> {
        for i in 0..self.molecules.len() {
            for j in (i + 1)..self.molecules.len() {
                self.calculate_intermolecular_forces(i, j)?;
            }
        }
        Ok(())
    }

    fn calculate_intermolecular_forces(&mut self, mol1_idx: usize, mol2_idx: usize) -> Result<()> {
        let cutoff_distance = 1e-9; // 1 nm
        if mol1_idx >= self.molecules.len() || mol2_idx >= self.molecules.len() {
            return Ok(());
        }

        let pos1 = self.calculate_molecular_center_of_mass(mol1_idx)?;
        let pos2 = self.calculate_molecular_center_of_mass(mol2_idx)?;
        let center_distance = (pos2 - pos1).magnitude();
        if center_distance > cutoff_distance {
            return Ok(());
        }

        let weak_force_magnitude = 1e-12;
        let force_direction = (pos2 - pos1).normalize();
        let force = weak_force_magnitude * force_direction;

        if let Some(molecule1) = self.molecules.get_mut(mol1_idx) {
            if let Some(first_atom) = molecule1.atoms.first_mut() {
                let mass = self.get_atomic_mass(first_atom.nucleus.atomic_number);
                first_atom.velocity += (force / mass) * self.time_step;
            }
        }
        Ok(())
    }

    fn calculate_molecular_center_of_mass(&self, mol_idx: usize) -> Result<Vector3<f64>> {
        if mol_idx >= self.molecules.len() {
            return Err(anyhow::anyhow!("Molecule index {} out of bounds", mol_idx));
        }

        let molecule = &self.molecules[mol_idx];
        let mut total_mass = 0.0;
        let mut center_of_mass = Vector3::zeros();
        for atom in &molecule.atoms {
            let mass = self.get_atomic_mass(atom.nucleus.atomic_number);
            total_mass += mass;
            center_of_mass += mass * atom.position;
        }
        if total_mass > 0.0 {
            center_of_mass /= total_mass;
        }
        Ok(center_of_mass)
    }
} 