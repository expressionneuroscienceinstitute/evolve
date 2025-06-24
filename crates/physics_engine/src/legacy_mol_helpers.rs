//! Legacy molecular dynamics helper functions
//! 
//! These functions were originally written as free functions with `&self` receivers,
//! which caused compilation errors. They have been moved to `molecular_helpers.rs`
//! as proper `impl PhysicsEngine` methods.
//!
//! This file exists only for reference and is not compiled.

#[cfg(FALSE)] // Disable compilation of this legacy code
pub mod legacy_molecular_helpers {
    use super::*;

    /// Calculate forces for molecular dynamics simulation
    fn calculate_molecular_forces(&self, states: &[PhysicsState], molecule: &Molecule) -> Result<Vec<Vector3<f64>>> {
        let mut forces = vec![Vector3::zeros(); states.len()];
        
        // 1. Intramolecular forces (bonds, angles, dihedrals)
        for bond in &molecule.bonds {
            let (i, j) = bond.atom_indices;
            if i < states.len() && j < states.len() {
                let bond_force = self.calculate_bond_force(&states[i], &states[j], bond)?;
                forces[i] += bond_force;
                forces[j] -= bond_force; // Newton's third law
            }
        }
        
        // 2. Non-bonded intramolecular forces (van der Waals, electrostatic)
        for i in 0..states.len() {
            for j in (i+1)..states.len() {
                if !self.are_bonded_in_molecule(i, j, molecule) {
                    let vdw_force = self.calculate_lennard_jones_force(&states[i], &states[j])?;
                    let coulomb_force = self.calculate_coulomb_force(&states[i], &states[j])?;
                    let total_force = vdw_force + coulomb_force;
                    
                    forces[i] += total_force;
                    forces[j] -= total_force;
                }
            }
        }
        
        Ok(forces)
    }

    /// Calculate harmonic bond force
    fn calculate_bond_force(&self, atom1: &PhysicsState, atom2: &PhysicsState, bond: &ChemicalBond) -> Result<Vector3<f64>> {
        let displacement = atom2.position - atom1.position;
        let distance = displacement.magnitude();
        
        if distance < 1e-12 {
            return Ok(Vector3::zeros());
        }
        
        // Harmonic bond potential: V = 0.5 * k * (r - r₀)²
        // Force: F = -k * (r - r₀) * r̂
        let force_constant = self.get_bond_force_constant(bond.bond_type)?;
        let equilibrium_length = bond.bond_length;
        let force_magnitude = force_constant * (distance - equilibrium_length);
        let force_direction = displacement.normalize();
        
        Ok(-force_magnitude * force_direction)
    }

    /// Calculate Lennard-Jones force between two atoms
    fn calculate_lennard_jones_force(&self, atom1: &PhysicsState, atom2: &PhysicsState) -> Result<Vector3<f64>> {
        let displacement = atom2.position - atom1.position;
        let distance = displacement.magnitude();
        
        if distance < 1e-12 {
            return Ok(Vector3::zeros());
        }
        
        // Lennard-Jones parameters (simplified - should be element-specific)
        let epsilon = 1e-21; // J (roughly 0.6 kJ/mol)
        let sigma = 3.5e-10;  // m (typical atomic diameter)
        
        // LJ potential: V = 4ε[(σ/r)¹² - (σ/r)⁶]
        // Force: F = 24ε/r * [(σ/r)⁶ - 2(σ/r)¹²] * r̂
        let sigma_over_r = sigma / distance;
        let sigma6 = sigma_over_r.powi(6);
        let sigma12 = sigma6 * sigma6;
        
        let force_magnitude = 24.0 * epsilon / distance * (sigma6 - 2.0 * sigma12);
        let force_direction = displacement.normalize();
        
        Ok(force_magnitude * force_direction)
    }

    /// Calculate Coulomb electrostatic force
    fn calculate_coulomb_force(&self, atom1: &PhysicsState, atom2: &PhysicsState) -> Result<Vector3<f64>> {
        let displacement = atom2.position - atom1.position;
        let distance = displacement.magnitude();
        
        if distance < 1e-12 {
            return Ok(Vector3::zeros());
        }
        
        // Coulomb force: F = k * q₁ * q₂ / r² * r̂
        let k_coulomb = 8.9875517923e9; // N⋅m²/C² (Coulomb's constant)
        let force_magnitude = k_coulomb * atom1.charge * atom2.charge / (distance * distance);
        let force_direction = displacement.normalize();
        
        Ok(force_magnitude * force_direction)
    }

    /// Get bond force constant for different bond types
    fn get_bond_force_constant(&self, bond_type: BondType) -> Result<f64> {
        // Force constants in N/m (typical values)
        let force_constant = match bond_type {
            BondType::Covalent => 500.0,      // Strong covalent bonds
            BondType::Ionic => 300.0,         // Ionic bonds
            BondType::Metallic => 200.0,      // Metallic bonds
            BondType::HydrogenBond => 50.0,   // Hydrogen bonds
            BondType::VanDerWaals => 10.0,    // van der Waals interactions
        };
        Ok(force_constant)
    }

    /// Check if two atoms are bonded within a molecule
    fn are_bonded_in_molecule(&self, i: usize, j: usize, molecule: &Molecule) -> bool {
        molecule.bonds.iter().any(|bond| {
            (bond.atom_indices.0 == i && bond.atom_indices.1 == j) ||
            (bond.atom_indices.0 == j && bond.atom_indices.1 == i)
        })
    }

    /// Get atomic mass for a given atomic number
    fn get_atomic_mass(&self, atomic_number: u32) -> f64 {
        // Simplified atomic masses in kg (approximate values)
        match atomic_number {
            1 => 1.67e-27,    // Hydrogen
            6 => 1.99e-26,    // Carbon
            7 => 2.33e-26,    // Nitrogen
            8 => 2.66e-26,    // Oxygen
            _ => atomic_number as f64 * 1.66e-27, // Rough approximation
        }
    }

    /// Calculate atomic energy based on current state
    fn calculate_atomic_energy(&self, atom: &Atom) -> Result<f64> {
        // Calculate total atomic energy including electronic and nuclear contributions
        let nuclear_energy = atom.nucleus.binding_energy * 1.602176634e-13; // Convert MeV to J
        let electronic_energy = atom.electrons.iter().map(|e| e.binding_energy).sum::<f64>();
        let kinetic_energy = 0.5 * self.get_atomic_mass(atom.nucleus.atomic_number) * 
                           atom.velocity.magnitude_squared();
        
        Ok(nuclear_energy + electronic_energy + kinetic_energy)
    }

    /// Update molecular properties after MD step
    fn update_molecular_properties(&self, molecule: &mut Molecule) -> Result<()> {
        // Update center of mass
        let mut total_mass = 0.0;
        let mut center_of_mass = Vector3::zeros();
        
        for atom in &molecule.atoms {
            let mass = self.get_atomic_mass(atom.nucleus.atomic_number);
            total_mass += mass;
            center_of_mass += mass * atom.position;
        }
        center_of_mass /= total_mass;
        
        // Update dipole moment (simplified calculation)
        let mut dipole = Vector3::zeros();
        for atom in &molecule.atoms {
            let charge = atom.nucleus.atomic_number as f64 * constants::ELEMENTARY_CHARGE;
            dipole += charge * (atom.position - center_of_mass);
        }
        molecule.dipole_moment = dipole;
        
        Ok(())
    }

    /// Process intermolecular interactions between different molecules
    fn process_intermolecular_interactions(&mut self) -> Result<()> {
        // Simplified intermolecular interactions (van der Waals, hydrogen bonding)
        for i in 0..self.molecules.len() {
            for j in (i+1)..self.molecules.len() {
                self.calculate_intermolecular_forces(i, j)?;
            }
        }
        Ok(())
    }

    /// Calculate forces between different molecules
    fn calculate_intermolecular_forces(&mut self, mol1_idx: usize, mol2_idx: usize) -> Result<()> {
        // Simplified intermolecular interaction calculation
        // In practice, this would involve detailed force field calculations
        
        let cutoff_distance = 1e-9; // 1 nm cutoff for efficiency
        
        // Get molecules (need to work around borrowing rules)
        if mol1_idx >= self.molecules.len() || mol2_idx >= self.molecules.len() {
            return Ok(());
        }
        
        // Calculate center-to-center distance first
        let pos1 = self.calculate_molecular_center_of_mass(mol1_idx)?;
        let pos2 = self.calculate_molecular_center_of_mass(mol2_idx)?;
        let center_distance = (pos2 - pos1).magnitude();
        
        // Skip if molecules are too far apart
        if center_distance > cutoff_distance {
            return Ok(());
        }
        
        // Apply weak intermolecular forces (simplified)
        let weak_force_magnitude = 1e-12; // Very small force in N
        let force_direction = (pos2 - pos1).normalize();
        let force = weak_force_magnitude * force_direction;
        
        // Apply forces to molecular centers (simplified application)
        if let Some(molecule1) = self.molecules.get_mut(mol1_idx) {
            if let Some(first_atom) = molecule1.atoms.first_mut() {
                let mass = self.get_atomic_mass(first_atom.nucleus.atomic_number);
                let acceleration = force / mass;
                first_atom.velocity += acceleration * self.time_step;
            }
        }
        
        Ok(())
    }

    /// Calculate molecular center of mass
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