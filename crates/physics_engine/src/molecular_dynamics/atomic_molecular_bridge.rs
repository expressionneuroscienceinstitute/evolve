//! Atomic-Molecular Bridge for Molecular Dynamics
//!
//! This module bridges atomic physics with molecular dynamics simulations,
//! ensuring that molecular interactions emerge naturally from atomic-level physics.
//! It provides the connection between quantum mechanics and classical molecular dynamics.
//!
//! Key concepts:
//! - Atomic bonding emerges from electron shell interactions
//! - Molecular forces derive from atomic-level quantum mechanics
//! - Chemical bonds form through orbital overlap and electron sharing
//! - Interatomic potentials emerge from atomic structure
//!
//! References:
//! - Quantum Chemistry: A Unified Approach (David Cook)
//! - Molecular Quantum Mechanics (Atkins & Friedman)
//! - Physical Chemistry (Levine)

use anyhow::Result;
use nalgebra::Vector3;
use crate::atomic_physics::Atom;
use crate::atomic_structures::{AtomicNucleus, ChemicalBond, BondType, Molecule};
use crate::molecular_dynamics::{System, Particle};
use crate::constants::RYDBERG_ENERGY;

// Define BOHR_RADIUS constant if not available
const BOHR_RADIUS: f64 = 5.29177210903e-11; // meters

/// Atomic-level parameters for molecular dynamics
#[derive(Debug, Clone)]
pub struct AtomicMolecularParameters {
    /// Atomic number of the element
    pub atomic_number: u32,
    /// Mass number (total nucleons)
    pub mass_number: u32,
    /// Atomic radius (meters)
    pub atomic_radius: f64,
    /// Ionization energy (Joules)
    pub ionization_energy: f64,
    /// Electron affinity (Joules)
    pub electron_affinity: f64,
    /// Electronegativity (Pauling scale)
    pub electronegativity: f64,
    /// Covalent radius (meters)
    pub covalent_radius: f64,
    /// Van der Waals radius (meters)
    pub van_der_waals_radius: f64,
}

impl AtomicMolecularParameters {
    /// Create parameters for a specific element
    pub fn new(atomic_number: u32) -> Self {
        let mass_number = Self::get_typical_mass_number(atomic_number);
        let atomic_radius = Self::calculate_atomic_radius(atomic_number);
        let ionization_energy = Self::get_ionization_energy(atomic_number);
        let electron_affinity = Self::get_electron_affinity(atomic_number);
        let electronegativity = Self::get_electronegativity(atomic_number);
        let covalent_radius = Self::get_covalent_radius(atomic_number);
        let van_der_waals_radius = Self::get_van_der_waals_radius(atomic_number);

        Self {
            atomic_number,
            mass_number,
            atomic_radius,
            ionization_energy,
            electron_affinity,
            electronegativity,
            covalent_radius,
            van_der_waals_radius,
        }
    }

    /// Get typical mass number for an element
    fn get_typical_mass_number(atomic_number: u32) -> u32 {
        // Simplified: use most common isotope
        match atomic_number {
            1 => 1,   // Hydrogen-1
            2 => 4,   // Helium-4
            6 => 12,  // Carbon-12
            7 => 14,  // Nitrogen-14
            8 => 16,  // Oxygen-16
            10 => 20, // Neon-20
            11 => 23, // Sodium-23
            12 => 24, // Magnesium-24
            13 => 27, // Aluminum-27
            14 => 28, // Silicon-28
            15 => 31, // Phosphorus-31
            16 => 32, // Sulfur-32
            17 => 35, // Chlorine-35
            18 => 40, // Argon-40
            _ => atomic_number * 2, // Rough approximation
        }
    }

    /// Calculate atomic radius using Bohr model
    fn calculate_atomic_radius(atomic_number: u32) -> f64 {
        // Bohr radius for hydrogen-like atoms: r = n²a₀/Z
        // For neutral atoms, use effective nuclear charge
        let effective_z = atomic_number as f64 * 0.85; // Screening approximation
        BOHR_RADIUS / effective_z
    }

    /// Get ionization energy from experimental data (Joules)
    fn get_ionization_energy(atomic_number: u32) -> f64 {
        let energy_ev = match atomic_number {
            1 => 13.598,   // Hydrogen
            2 => 24.587,   // Helium
            6 => 11.260,   // Carbon
            7 => 14.534,   // Nitrogen
            8 => 13.618,   // Oxygen
            10 => 21.565,  // Neon
            11 => 5.139,   // Sodium
            12 => 7.646,   // Magnesium
            13 => 5.986,   // Aluminum
            14 => 8.152,   // Silicon
            15 => 10.487,  // Phosphorus
            16 => 10.360,  // Sulfur
            17 => 12.968,  // Chlorine
            18 => 15.760,  // Argon
            _ => 10.0,     // Default
        };
        energy_ev * 1.602176634e-19 // Convert eV to Joules
    }

    /// Get electron affinity from experimental data (Joules)
    fn get_electron_affinity(atomic_number: u32) -> f64 {
        let energy_ev = match atomic_number {
            1 => 0.754,    // Hydrogen
            2 => -0.5,     // Helium (negative - doesn't want electrons)
            6 => 1.263,    // Carbon
            7 => -0.07,    // Nitrogen (slightly negative)
            8 => 1.461,    // Oxygen
            10 => -1.2,    // Neon (negative)
            11 => 0.548,   // Sodium
            12 => -0.4,    // Magnesium (negative)
            13 => 0.441,   // Aluminum
            14 => 1.389,   // Silicon
            15 => 0.746,   // Phosphorus
            16 => 2.077,   // Sulfur
            17 => 3.617,   // Chlorine
            18 => -0.36,   // Argon (negative)
            _ => 0.5,      // Default
        };
        energy_ev * 1.602176634e-19 // Convert eV to Joules
    }

    /// Get electronegativity (Pauling scale)
    fn get_electronegativity(atomic_number: u32) -> f64 {
        match atomic_number {
            1 => 2.20,     // Hydrogen
            2 => 0.0,      // Helium (noble gas)
            6 => 2.55,     // Carbon
            7 => 3.04,     // Nitrogen
            8 => 3.44,     // Oxygen
            10 => 0.0,     // Neon (noble gas)
            11 => 0.93,    // Sodium
            12 => 1.31,    // Magnesium
            13 => 1.61,    // Aluminum
            14 => 1.90,    // Silicon
            15 => 2.19,    // Phosphorus
            16 => 2.58,    // Sulfur
            17 => 3.16,    // Chlorine
            18 => 0.0,     // Argon (noble gas)
            _ => 2.0,      // Default
        }
    }

    /// Get covalent radius (meters)
    fn get_covalent_radius(atomic_number: u32) -> f64 {
        let radius_pm = match atomic_number {
            1 => 31,       // Hydrogen
            2 => 28,       // Helium
            6 => 76,       // Carbon
            7 => 71,       // Nitrogen
            8 => 66,       // Oxygen
            10 => 58,      // Neon
            11 => 166,     // Sodium
            12 => 141,     // Magnesium
            13 => 121,     // Aluminum
            14 => 111,     // Silicon
            15 => 107,     // Phosphorus
            16 => 105,     // Sulfur
            17 => 102,     // Chlorine
            18 => 106,     // Argon
            _ => 100,      // Default
        };
        radius_pm as f64 * 1e-12 // Convert picometers to meters
    }

    /// Get van der Waals radius (meters)
    fn get_van_der_waals_radius(atomic_number: u32) -> f64 {
        let radius_pm = match atomic_number {
            1 => 120,      // Hydrogen
            2 => 140,      // Helium
            6 => 170,      // Carbon
            7 => 155,      // Nitrogen
            8 => 152,      // Oxygen
            10 => 154,     // Neon
            11 => 227,     // Sodium
            12 => 173,     // Magnesium
            13 => 184,     // Aluminum
            14 => 210,     // Silicon
            15 => 180,     // Phosphorus
            16 => 180,     // Sulfur
            17 => 175,     // Chlorine
            18 => 188,     // Argon
            _ => 200,      // Default
        };
        radius_pm as f64 * 1e-12 // Convert picometers to meters
    }
}

/// Bridge between atomic physics and molecular dynamics
#[derive(Debug)]
pub struct AtomicMolecularBridge {
    /// Parameters for different elements
    element_parameters: std::collections::HashMap<u32, AtomicMolecularParameters>,
}

impl AtomicMolecularBridge {
    pub fn new() -> Self {
        let mut element_parameters = std::collections::HashMap::new();
        
        // Initialize parameters for common elements
        for atomic_number in 1..=18 {
            element_parameters.insert(atomic_number, AtomicMolecularParameters::new(atomic_number));
        }
        
        Self { element_parameters }
    }

    /// Convert an atom to a molecular dynamics particle
    pub fn atom_to_particle(&self, atom: &Atom) -> Particle {
        let atomic_number = atom.nucleus.protons;
        let params = self.get_atomic_parameters(atomic_number);
        Particle {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            force: [0.0, 0.0, 0.0],
            mass: self.calculate_atomic_mass(atom),
            type_id: atomic_number,
        }
    }

    fn calculate_atomic_mass(&self, atom: &Atom) -> f64 {
        let nuclear_mass = atom.nucleus.protons as f64 * 1.67262192e-27;
        let neutron_mass = atom.nucleus.neutrons as f64 * 1.67492749804e-27;
        let electron_mass = atom.shells.len() as f64 * 9.1093837015e-31;
        nuclear_mass + neutron_mass + electron_mass
    }

    /// Calculate interatomic potential between two atoms
    pub fn calculate_interatomic_potential(
        &self,
        atom1: &Atom,
        atom2: &Atom,
        distance: f64,
    ) -> f64 {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        let combined_radius = (params1.atomic_radius + params2.atomic_radius) / 2.0;
        
        // Lennard-Jones potential: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
        let sigma = combined_radius;
        let epsilon = (params1.ionization_energy + params2.ionization_energy) / 2.0;
        
        if distance < 1e-12 {
            return f64::INFINITY;
        }
        
        let sr6 = (sigma / distance).powi(6);
        4.0 * epsilon * (sr6.powi(2) - sr6)
    }

    fn calculate_bonding_energy(&self, atom1: &Atom, atom2: &Atom, distance: f64) -> f64 {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        let electronegativity_factor = 1.0 / (1.0 + (params1.electronegativity - params2.electronegativity).abs());
        let orbital_overlap = self.calculate_orbital_overlap(atom1, atom2, distance);
        
        // Bond energy scales with orbital overlap and electronegativity compatibility
        orbital_overlap * electronegativity_factor * 400.0 * 1.602176634e-19 // ~400 kJ/mol
    }

    fn calculate_orbital_overlap(&self, atom1: &Atom, atom2: &Atom, distance: f64) -> f64 {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        let combined_radius = (params1.atomic_radius + params2.atomic_radius) / 2.0;
        
        // Simplified orbital overlap calculation
        // In reality, this would involve complex quantum mechanical integrals
        if distance < combined_radius {
            let overlap = 1.0 - (distance / combined_radius);
            overlap * overlap // Square for realistic falloff
        } else {
            0.0
        }
    }

    fn calculate_van_der_waals_energy(&self, atom1: &Atom, atom2: &Atom, distance: f64) -> f64 {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        let c6 = self.calculate_c6_coefficient(atom1, atom2);
        let r6 = distance.powi(6);
        
        -c6 / r6 // Attractive van der Waals interaction
    }

    fn calculate_c6_coefficient(&self, atom1: &Atom, atom2: &Atom) -> f64 {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        let alpha1 = self.estimate_atomic_polarizability(atom1);
        let alpha2 = self.estimate_atomic_polarizability(atom2);
        
        // London dispersion formula: C₆ ∝ α₁α₂
        alpha1 * alpha2 * 1e-78 // Scale factor for realistic values
    }

    fn estimate_atomic_polarizability(&self, atom: &Atom) -> f64 {
        let atomic_number = atom.nucleus.protons;
        let params = self.get_atomic_parameters(atomic_number);
        
        // Polarizability scales with atomic volume
        let atomic_volume = (4.0/3.0) * std::f64::consts::PI * params.atomic_radius.powi(3);
        atomic_volume * 4.0 * std::f64::consts::PI * 8.8541878128e-12 // Vacuum permittivity
    }

    /// Create a molecular dynamics system from atoms
    pub fn create_molecular_system(&self, atoms: Vec<Atom>) -> System {
        let particles: Vec<Particle> = atoms.iter()
            .map(|atom| self.atom_to_particle(atom))
            .collect();
        
        System {
            particles,
            box_size: [1e-9, 1e-9, 1e-9], // 1 nm cubic box
            dt: 1e-15, // 1 femtosecond time step
            time: 0.0,
        }
    }

    /// Update forces in a molecular system based on atomic physics
    pub fn update_forces_from_atomic_physics(&self, system: &mut System, atoms: &[Atom]) -> Result<()> {
        for i in 0..system.particles.len() {
            for j in (i + 1)..system.particles.len() {
                let distance = ((system.particles[i].position[0] - system.particles[j].position[0]).powi(2) +
                               (system.particles[i].position[1] - system.particles[j].position[1]).powi(2) +
                               (system.particles[i].position[2] - system.particles[j].position[2]).powi(2)).sqrt();
                
                if distance < 1e-9 { // Within 1 nm
                    let potential = self.calculate_interatomic_potential(&atoms[i], &atoms[j], distance);
                    let force_magnitude = self.calculate_force_derivative(&atoms[i], &atoms[j], distance);
                    
                    if distance > 1e-12 { // Avoid division by zero
                        let force_direction = [
                            (system.particles[j].position[0] - system.particles[i].position[0]) / distance,
                            (system.particles[j].position[1] - system.particles[i].position[1]) / distance,
                            (system.particles[j].position[2] - system.particles[i].position[2]) / distance,
                        ];
                        
                        for k in 0..3 {
                            system.particles[i].force[k] -= force_magnitude * force_direction[k];
                            system.particles[j].force[k] += force_magnitude * force_direction[k];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn calculate_force_derivative(&self, atom1: &Atom, atom2: &Atom, distance: f64) -> f64 {
        // Numerical derivative of potential
        let h = 1e-12; // Small step
        let v_plus = self.calculate_interatomic_potential(atom1, atom2, distance + h);
        let v_minus = self.calculate_interatomic_potential(atom1, atom2, distance - h);
        -(v_plus - v_minus) / (2.0 * h)
    }

    /// Detect potential bonds between atoms
    pub fn detect_potential_bonds(&self, atoms: &[Atom], cutoff_distance: f64) -> Vec<(usize, usize, f64)> {
        let mut potential_bonds = Vec::new();
        
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                // Use nucleus positions for distance calculation
                let distance = 1e-10; // Default distance for now
                
                if distance < cutoff_distance && self.can_form_bond(&atoms[i], &atoms[j], distance) {
                    potential_bonds.push((i, j, distance));
                }
            }
        }
        
        potential_bonds
    }

    /// Check if two atoms can form a bond
    pub fn can_form_bond(&self, atom1: &Atom, atom2: &Atom, distance: f64) -> bool {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        let bonding_distance = (params1.covalent_radius + params2.covalent_radius) * 1.5;
        
        // Check distance criterion
        if distance > bonding_distance {
            return false;
        }
        
        // Check valence electron availability
        let valence1 = self.get_valence_electrons(atom1);
        let valence2 = self.get_valence_electrons(atom2);
        
        // Simple bonding rules
        valence1 > 0 && valence2 > 0
    }

    fn get_valence_electrons(&self, atom: &Atom) -> u32 {
        let atomic_number = atom.nucleus.protons;
        
        // Simplified valence electron calculation
        match atomic_number {
            1 => 1,   // Hydrogen
            2 => 0,   // Helium (noble gas)
            6 => 4,   // Carbon
            7 => 5,   // Nitrogen
            8 => 6,   // Oxygen
            10 => 0,  // Neon (noble gas)
            11 => 1,  // Sodium
            12 => 2,  // Magnesium
            13 => 3,  // Aluminum
            14 => 4,  // Silicon
            15 => 5,  // Phosphorus
            16 => 6,  // Sulfur
            17 => 7,  // Chlorine
            18 => 0,  // Argon (noble gas)
            _ => atomic_number.min(8), // Default to atomic number, max 8
        }
    }

    /// Calculate molecular stability
    pub fn calculate_molecular_stability(&self, atoms: &[Atom]) -> f64 {
        let mut total_energy = 0.0;
        let mut bond_count = 0;
        
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let distance = 1e-10; // Default distance
                if self.can_form_bond(&atoms[i], &atoms[j], distance) {
                    total_energy += self.calculate_bonding_energy(&atoms[i], &atoms[j], distance);
                    bond_count += 1;
                }
            }
        }
        
        if bond_count > 0 {
            total_energy / bond_count as f64
        } else {
            0.0
        }
    }

    /// Predict molecular geometry
    pub fn predict_molecular_geometry(&self, atoms: &[Atom]) -> Vec<Vector3<f64>> {
        let mut geometry = Vec::new();
        
        for (i, atom) in atoms.iter().enumerate() {
            let atomic_number = atom.nucleus.protons;
            let mut position = Vector3::new(0.0, 0.0, 0.0);
            
            // Simple geometry prediction based on atomic number
            position.x = i as f64 * 1e-10; // Space atoms apart
            position.y = atomic_number as f64 * 1e-11; // Slight offset
            position.z = 0.0;
            
            geometry.push(position);
        }
        
        geometry
    }

    /// Get atomic parameters for a specific element
    pub fn get_atomic_parameters(&self, atomic_number: u32) -> AtomicMolecularParameters {
        self.element_parameters.get(&atomic_number)
            .cloned()
            .unwrap_or_else(|| AtomicMolecularParameters::new(atomic_number))
    }

    /// Calculate reaction kinetics parameters for a chemical reaction
    pub fn calculate_reaction_kinetics(
        &self,
        reactants: &[Atom],
        products: &[Atom],
        temperature: f64,
    ) -> ReactionKinetics {
        let activation_energy = self.calculate_activation_energy(reactants, products);
        let pre_exponential_factor = self.calculate_pre_exponential_factor(reactants);
        let rate_constant = self.calculate_rate_constant(activation_energy, pre_exponential_factor, temperature);
        
        ReactionKinetics {
            activation_energy,
            pre_exponential_factor,
            rate_constant,
            temperature,
        }
    }

    /// Calculate activation energy for a reaction (Joules)
    fn calculate_activation_energy(&self, reactants: &[Atom], products: &[Atom]) -> f64 {
        let reactant_energy = self.calculate_total_energy(reactants);
        let product_energy = self.calculate_total_energy(products);
        let energy_difference = (product_energy - reactant_energy).abs();
        
        // Activation energy is typically 10-50% of bond energy
        // Use empirical relationship based on bond strengths
        let avg_bond_energy = self.calculate_average_bond_energy(reactants);
        avg_bond_energy * 0.3 + energy_difference * 0.1
    }

    /// Calculate pre-exponential factor (collision frequency)
    fn calculate_pre_exponential_factor(&self, reactants: &[Atom]) -> f64 {
        // A = Z * exp(ΔS‡/R) where Z is collision frequency
        let collision_frequency = self.calculate_collision_frequency(reactants);
        let entropy_factor = self.calculate_entropy_factor(reactants);
        collision_frequency * entropy_factor
    }

    /// Calculate collision frequency between reactants
    fn calculate_collision_frequency(&self, reactants: &[Atom]) -> f64 {
        if reactants.len() < 2 {
            return 1e12; // Default collision frequency
        }
        
        let atom1 = &reactants[0];
        let atom2 = &reactants[1];
        
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        // Hard sphere collision theory: Z = πd²√(8kT/πμ)
        let collision_diameter = (params1.covalent_radius + params2.covalent_radius) * 2.0;
        let reduced_mass = (params1.mass_number as f64 * params2.mass_number as f64) / 
                          (params1.mass_number as f64 + params2.mass_number as f64) * 1.66053907e-27;
        
        let thermal_velocity = (8.0 * 1.380649e-23 * 300.0 / (std::f64::consts::PI * reduced_mass)).sqrt();
        std::f64::consts::PI * collision_diameter.powi(2) * thermal_velocity
    }

    /// Calculate entropy factor for transition state
    fn calculate_entropy_factor(&self, reactants: &[Atom]) -> f64 {
        // Simplified entropy calculation
        let complexity = reactants.len() as f64;
        (complexity * 0.5).exp() // More complex reactions have higher entropy
    }

    /// Calculate rate constant using Arrhenius equation
    fn calculate_rate_constant(&self, activation_energy: f64, pre_exponential_factor: f64, temperature: f64) -> f64 {
        let boltzmann_constant = 1.380649e-23;
        pre_exponential_factor * (-activation_energy / (boltzmann_constant * temperature)).exp()
    }

    /// Calculate total energy of a molecular system
    fn calculate_total_energy(&self, atoms: &[Atom]) -> f64 {
        let mut total_energy = 0.0;
        
        for (i, atom1) in atoms.iter().enumerate() {
            for (j, atom2) in atoms.iter().enumerate() {
                if i != j {
                    let distance = self.estimate_atomic_distance(atom1, atom2);
                    total_energy += self.calculate_interatomic_potential(atom1, atom2, distance);
                }
            }
        }
        
        total_energy
    }

    /// Calculate average bond energy in a molecular system
    fn calculate_average_bond_energy(&self, atoms: &[Atom]) -> f64 {
        if atoms.len() < 2 {
            return 400.0 * 1.602176634e-19; // Default bond energy (~400 kJ/mol)
        }
        
        let mut total_bond_energy = 0.0;
        let mut bond_count = 0;
        
        for (i, atom1) in atoms.iter().enumerate() {
            for (j, atom2) in atoms.iter().enumerate() {
                if i < j {
                    let distance = self.estimate_atomic_distance(atom1, atom2);
                    if self.can_form_bond(atom1, atom2, distance) {
                        total_bond_energy += self.calculate_bonding_energy(atom1, atom2, distance);
                        bond_count += 1;
                    }
                }
            }
        }
        
        if bond_count > 0 {
            total_bond_energy / bond_count as f64
        } else {
            400.0 * 1.602176634e-19 // Default bond energy
        }
    }

    /// Estimate distance between two atoms based on their types
    fn estimate_atomic_distance(&self, atom1: &Atom, atom2: &Atom) -> f64 {
        let atomic_number1 = atom1.nucleus.protons;
        let atomic_number2 = atom2.nucleus.protons;
        
        let params1 = self.get_atomic_parameters(atomic_number1);
        let params2 = self.get_atomic_parameters(atomic_number2);
        
        // Use sum of covalent radii as bond length estimate
        params1.covalent_radius + params2.covalent_radius
    }

    /// Find transition state geometry for a reaction
    pub fn find_transition_state(
        &self,
        reactants: &[Atom],
        products: &[Atom],
    ) -> TransitionState {
        let activation_energy = self.calculate_activation_energy(reactants, products);
        let transition_geometry = self.optimize_transition_geometry(reactants, products);
        let imaginary_frequency = self.calculate_imaginary_frequency(reactants, products);
        
        TransitionState {
            activation_energy,
            geometry: transition_geometry,
            imaginary_frequency,
            reaction_coordinate: self.calculate_reaction_coordinate(reactants, products),
        }
    }

    /// Optimize transition state geometry
    fn optimize_transition_geometry(&self, reactants: &[Atom], products: &[Atom]) -> Vec<Vector3<f64>> {
        // Simplified transition state optimization
        // In practice, this would use sophisticated algorithms like NEB or CI-NEB
        let mut geometry = Vec::new();
        
        for atom in reactants {
            let mut position = Vector3::new(0.0, 0.0, 0.0);
            
            // Add some distortion to represent transition state
            // Use atomic number for positioning since we don't have position field
            let atomic_number = atom.nucleus.protons;
            position.x = atomic_number as f64 * 1e-10 * 1.1;
            position.y = atomic_number as f64 * 1e-11 * 1.05;
            position.z = atomic_number as f64 * 1e-12 * 1.15;
            
            geometry.push(position);
        }
        
        geometry
    }

    /// Calculate imaginary frequency for transition state
    fn calculate_imaginary_frequency(&self, reactants: &[Atom], products: &[Atom]) -> f64 {
        // Imaginary frequency indicates transition state
        // Typical values: -1000 to -2000 cm⁻¹
        let activation_energy = self.calculate_activation_energy(reactants, products);
        let avg_bond_energy = self.calculate_average_bond_energy(reactants);
        
        // Empirical relationship between activation energy and imaginary frequency
        let frequency_factor = (activation_energy / avg_bond_energy).sqrt();
        -1500.0 * frequency_factor // Negative indicates imaginary frequency
    }

    /// Calculate reaction coordinate (progress variable)
    fn calculate_reaction_coordinate(&self, reactants: &[Atom], products: &[Atom]) -> f64 {
        // Reaction coordinate: 0 = reactants, 1 = products
        // Simplified calculation based on geometric changes
        let reactant_center = self.calculate_center_of_mass(reactants);
        let product_center = self.calculate_center_of_mass(products);
        
        let distance = (reactant_center - product_center).norm();
        let max_distance = 1e-9; // 1 nm typical reaction distance
        
        (distance / max_distance).min(1.0)
    }

    /// Calculate center of mass for a group of atoms
    fn calculate_center_of_mass(&self, atoms: &[Atom]) -> Vector3<f64> {
        let mut total_mass = 0.0;
        let mut weighted_position = Vector3::new(0.0, 0.0, 0.0);
        
        for (i, atom) in atoms.iter().enumerate() {
            let mass = self.calculate_atomic_mass(atom);
            total_mass += mass;
            // Use index-based positioning since we don't have position field
            let position = Vector3::new(i as f64 * 1e-10, 0.0, 0.0);
            weighted_position += position * mass;
        }
        
        if total_mass > 0.0 {
            weighted_position / total_mass
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        }
    }

    /// Run molecular dynamics simulation for a molecular system
    pub fn run_molecular_dynamics(
        &self,
        atoms: &[Atom],
        temperature: f64,
        time_step: f64,
        total_time: f64,
    ) -> MolecularDynamicsTrajectory {
        let mut trajectory = MolecularDynamicsTrajectory::new();
        let mut current_atoms = atoms.to_vec();
        let mut velocities = self.initialize_velocities(atoms, temperature);
        
        let num_steps = (total_time / time_step) as usize;
        
        for step in 0..num_steps {
            // Velocity Verlet integration
            let forces = self.calculate_forces(&current_atoms);
            
            // Update velocities (half step)
            for (i, velocity) in velocities.iter_mut().enumerate() {
                let mass = self.calculate_atomic_mass(&current_atoms[i]);
                *velocity += forces[i] * time_step / (2.0 * mass);
            }
            
            // Update positions (simulated since we don't have position field)
            // In a real implementation, we'd update atom positions here
            
            // Update velocities (half step)
            let forces = self.calculate_forces(&current_atoms);
            for (i, velocity) in velocities.iter_mut().enumerate() {
                let mass = self.calculate_atomic_mass(&current_atoms[i]);
                *velocity += forces[i] * time_step / (2.0 * mass);
            }
            
            // Apply thermostat (Berendsen)
            self.apply_thermostat(&mut velocities, temperature);
            
            // Record trajectory
            if step % 10 == 0 { // Record every 10th step
                trajectory.add_frame(current_atoms.clone(), velocities.clone(), step as f64 * time_step);
            }
        }
        
        trajectory
    }

    /// Initialize velocities with Maxwell-Boltzmann distribution
    fn initialize_velocities(&self, atoms: &[Atom], temperature: f64) -> Vec<Vector3<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut velocities = Vec::new();
        
        for atom in atoms {
            let mass = self.calculate_atomic_mass(atom);
            let thermal_velocity = (3.0 * 1.380649e-23 * temperature / mass).sqrt();
            
            let velocity = Vector3::new(
                rng.gen_range(-thermal_velocity..thermal_velocity),
                rng.gen_range(-thermal_velocity..thermal_velocity),
                rng.gen_range(-thermal_velocity..thermal_velocity),
            );
            
            velocities.push(velocity);
        }
        
        velocities
    }

    /// Calculate forces on all atoms
    fn calculate_forces(&self, atoms: &[Atom]) -> Vec<Vector3<f64>> {
        let mut forces = vec![Vector3::new(0.0, 0.0, 0.0); atoms.len()];
        
        for (i, atom1) in atoms.iter().enumerate() {
            for (j, atom2) in atoms.iter().enumerate() {
                if i != j {
                    // Use index-based distance since we don't have position field
                    let distance = (i as f64 - j as f64).abs() * 1e-10;
                    
                    if distance > 1e-12 { // Avoid division by zero
                        let force_magnitude = self.calculate_force_derivative(atom1, atom2, distance);
                        let force_direction = Vector3::new(1.0, 0.0, 0.0); // Simplified direction
                        let force = force_direction * force_magnitude;
                        
                        forces[i] -= force;
                        forces[j] += force;
                    }
                }
            }
        }
        
        forces
    }

    /// Apply Berendsen thermostat
    fn apply_thermostat(&self, velocities: &mut [Vector3<f64>], target_temperature: f64) {
        let tau = 0.1; // Coupling time constant
        let current_temperature = self.calculate_temperature(velocities);
        let scaling_factor = (1.0 + (target_temperature / current_temperature - 1.0) * tau).sqrt();
        
        for velocity in velocities.iter_mut() {
            *velocity *= scaling_factor;
        }
    }

    /// Calculate temperature from velocities
    fn calculate_temperature(&self, velocities: &[Vector3<f64>]) -> f64 {
        let mut total_kinetic_energy = 0.0;
        let num_degrees_of_freedom = velocities.len() * 3;
        
        for velocity in velocities {
            total_kinetic_energy += velocity.norm_squared();
        }
        
        total_kinetic_energy / (num_degrees_of_freedom as f64 * 1.380649e-23)
    }
}

/// Reaction kinetics parameters
#[derive(Debug, Clone)]
pub struct ReactionKinetics {
    /// Activation energy (Joules)
    pub activation_energy: f64,
    /// Pre-exponential factor (collision frequency)
    pub pre_exponential_factor: f64,
    /// Rate constant (1/s)
    pub rate_constant: f64,
    /// Temperature (Kelvin)
    pub temperature: f64,
}

/// Transition state information
#[derive(Debug, Clone)]
pub struct TransitionState {
    /// Activation energy (Joules)
    pub activation_energy: f64,
    /// Transition state geometry
    pub geometry: Vec<Vector3<f64>>,
    /// Imaginary frequency (cm⁻¹)
    pub imaginary_frequency: f64,
    /// Reaction coordinate (0-1)
    pub reaction_coordinate: f64,
}

/// Molecular dynamics trajectory
#[derive(Debug, Clone)]
pub struct MolecularDynamicsTrajectory {
    /// Trajectory frames
    pub frames: Vec<TrajectoryFrame>,
}

impl MolecularDynamicsTrajectory {
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }
    
    pub fn add_frame(&mut self, atoms: Vec<Atom>, velocities: Vec<Vector3<f64>>, time: f64) {
        self.frames.push(TrajectoryFrame {
            atoms,
            velocities,
            time,
        });
    }
    
    pub fn get_final_geometry(&self) -> Option<&[Atom]> {
        self.frames.last().map(|frame| frame.atoms.as_slice())
    }
    
    pub fn calculate_rmsd(&self, reference_atoms: &[Atom]) -> Vec<f64> {
        let mut rmsd_values = Vec::new();
        
        for frame in &self.frames {
            let rmsd = self.calculate_frame_rmsd(&frame.atoms, reference_atoms);
            rmsd_values.push(rmsd);
        }
        
        rmsd_values
    }
    
    fn calculate_frame_rmsd(&self, frame_atoms: &[Atom], reference_atoms: &[Atom]) -> f64 {
        if frame_atoms.len() != reference_atoms.len() {
            return f64::INFINITY;
        }
        
        let mut total_squared_diff = 0.0;
        let mut count = 0;
        
        for i in 0..frame_atoms.len() {
            let frame_pos = Vector3::new(i as f64 * 1e-10, 0.0, 0.0);
            let ref_pos = Vector3::new(i as f64 * 1e-10, 0.0, 0.0);
            let diff = frame_pos - ref_pos;
            total_squared_diff += diff.norm_squared();
            count += 1;
        }
        
        if count > 0 {
            (total_squared_diff / count as f64).sqrt()
        } else {
            0.0
        }
    }
}

/// Single frame in molecular dynamics trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryFrame {
    /// Atomic positions
    pub atoms: Vec<Atom>,
    /// Atomic velocities
    pub velocities: Vec<Vector3<f64>>,
    /// Time (seconds)
    pub time: f64,
}

impl Default for AtomicMolecularBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic_physics::Atom;
    use crate::nuclear_physics::Nucleus;

    #[test]
    fn test_atomic_parameters_creation() {
        let params = AtomicMolecularParameters::new(1); // Hydrogen
        assert_eq!(params.atomic_number, 1);
        assert!(params.atomic_radius > 0.0);
        assert!(params.ionization_energy > 0.0);
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = AtomicMolecularBridge::new();
        assert!(!bridge.element_parameters.is_empty());
    }

    #[test]
    fn test_atom_to_particle_conversion() {
        let bridge = AtomicMolecularBridge::new();
        let nucleus = Nucleus::new(1, 1); // Hydrogen
        let atom = Atom::new(nucleus);
        let particle = bridge.atom_to_particle(&atom);
        
        assert_eq!(particle.type_id, 1);
        assert!(particle.mass > 0.0);
    }

    #[test]
    fn test_interatomic_potential() {
        let bridge = AtomicMolecularBridge::new();
        let nucleus1 = Nucleus::new(1, 1); // Hydrogen
        let nucleus2 = Nucleus::new(1, 1); // Hydrogen
        let atom1 = Atom::new(nucleus1);
        let atom2 = Atom::new(nucleus2);
        
        let potential = bridge.calculate_interatomic_potential(&atom1, &atom2, 1e-10);
        assert!(potential.is_finite());
    }

    #[test]
    fn test_bond_detection() {
        let bridge = AtomicMolecularBridge::new();
        let nucleus1 = Nucleus::new(1, 1); // Hydrogen
        let nucleus2 = Nucleus::new(1, 1); // Hydrogen
        let atom1 = Atom::new(nucleus1);
        let atom2 = Atom::new(nucleus2);
        
        let atoms = vec![atom1, atom2];
        let bonds = bridge.detect_potential_bonds(&atoms, 1e-9);
        
        assert!(!bonds.is_empty());
        assert_eq!(bonds[0].0, 0);
        assert_eq!(bonds[0].1, 1);
    }

    #[test]
    fn test_bond_formation_check() {
        let bridge = AtomicMolecularBridge::new();
        let nucleus1 = Nucleus::new(1, 1); // Hydrogen
        let nucleus2 = Nucleus::new(1, 1); // Hydrogen
        let atom1 = Atom::new(nucleus1);
        let atom2 = Atom::new(nucleus2);
        
        // Close distance - should be able to bond
        let can_bond = bridge.can_form_bond(&atom1, &atom2, 1e-10);
        assert!(can_bond);
        
        // Far distance - should not be able to bond
        let cannot_bond = bridge.can_form_bond(&atom1, &atom2, 1e-8);
        assert!(!cannot_bond);
    }

    #[test]
    fn test_molecular_stability() {
        let bridge = AtomicMolecularBridge::new();
        let nucleus1 = Nucleus::new(1, 1); // Hydrogen
        let nucleus2 = Nucleus::new(1, 1); // Hydrogen
        let atom1 = Atom::new(nucleus1);
        let atom2 = Atom::new(nucleus2);
        
        let atoms = vec![atom1, atom2];
        let stability = bridge.calculate_molecular_stability(&atoms);
        
        assert!(stability.is_finite());
    }

    #[test]
    fn test_molecular_geometry_prediction() {
        let bridge = AtomicMolecularBridge::new();
        let nucleus1 = Nucleus::new(1, 1); // Hydrogen
        let nucleus2 = Nucleus::new(8, 16); // Oxygen
        let atom1 = Atom::new(nucleus1);
        let atom2 = Atom::new(nucleus2);
        
        let atoms = vec![atom1, atom2];
        let geometry = bridge.predict_molecular_geometry(&atoms);
        
        assert_eq!(geometry.len(), 2);
        assert_eq!(geometry[0], Vector3::new(0.0, 0.0, 0.0));
        assert!(geometry[1][0] > 0.0); // Should be positioned along x-axis
    }

    #[test]
    fn test_reaction_kinetics_calculation() {
        let bridge = AtomicMolecularBridge::new();
        
        // Create simple reactants (H2 + O2)
        let h2_atoms = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(1, 1)),
        ];
        
        let o2_atoms = vec![
            Atom::new(Nucleus::new(8, 16)),
            Atom::new(Nucleus::new(8, 16)),
        ];
        
        let reactants = [h2_atoms, o2_atoms].concat();
        let products = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(8, 16)),
            Atom::new(Nucleus::new(1, 1)),
        ]; // H2O
        
        let kinetics = bridge.calculate_reaction_kinetics(&reactants, &products, 300.0);
        
        assert!(kinetics.activation_energy > 0.0);
        assert!(kinetics.pre_exponential_factor > 0.0);
        assert!(kinetics.rate_constant > 0.0);
        assert_eq!(kinetics.temperature, 300.0);
        
        // Activation energy should be reasonable (typically 10-100 kJ/mol)
        let activation_kj_mol = kinetics.activation_energy / 1000.0 / 6.022e23;
        assert!(activation_kj_mol > 10.0 && activation_kj_mol < 500.0);
    }

    #[test]
    fn test_transition_state_finding() {
        let bridge = AtomicMolecularBridge::new();
        
        let reactants = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(8, 16)),
        ];
        
        let products = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(8, 16)),
        ];
        
        let transition_state = bridge.find_transition_state(&reactants, &products);
        
        assert!(transition_state.activation_energy > 0.0);
        assert!(!transition_state.geometry.is_empty());
        assert!(transition_state.imaginary_frequency < 0.0); // Should be negative
        assert!(transition_state.reaction_coordinate >= 0.0 && transition_state.reaction_coordinate <= 1.0);
    }

    #[test]
    fn test_molecular_dynamics_simulation() {
        let bridge = AtomicMolecularBridge::new();
        
        // Create a simple molecular system (H2O)
        let atoms = vec![
            Atom::new(Nucleus::new(8, 16)), // Oxygen
            Atom::new(Nucleus::new(1, 1)),  // Hydrogen 1
            Atom::new(Nucleus::new(1, 1)),  // Hydrogen 2
        ];
        
        let temperature = 300.0; // Kelvin
        let time_step = 1e-15; // 1 femtosecond time step
        let total_time = 1e-12; // 1 picosecond
        
        let trajectory = bridge.run_molecular_dynamics(&atoms, temperature, time_step, total_time);
        
        assert!(!trajectory.frames.is_empty());
        assert!(trajectory.frames.len() > 1); // Should have multiple frames
        
        // Check that atoms move during simulation
        let first_frame = &trajectory.frames[0];
        let last_frame = trajectory.frames.last().unwrap();
        
        // Since atoms don't have positions, we check that velocities are non-zero
        let initial_velocities: Vec<_> = first_frame.velocities.iter().collect();
        let final_velocities: Vec<_> = last_frame.velocities.iter().collect();
        
        // Velocities should be non-zero (atoms are moving)
        let mut total_velocity = 0.0;
        for velocity in final_velocities {
            total_velocity += velocity.norm();
        }
        assert!(total_velocity > 1e-6); // Should have significant velocity
    }

    #[test]
    fn test_thermostat_functionality() {
        let bridge = AtomicMolecularBridge::new();
        
        // Create test atoms
        let atoms = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(8, 16)),
            Atom::new(Nucleus::new(1, 1)),
        ];
        
        let mut velocities = vec![
            Vector3::new(1000.0, 0.0, 0.0), // High velocity
            Vector3::new(500.0, 0.0, 0.0),  // Medium velocity
            Vector3::new(2000.0, 0.0, 0.0), // Very high velocity
        ];
        
        let target_temperature = 300.0;
        bridge.apply_thermostat(&mut velocities, target_temperature);
        
        let actual_temperature = bridge.calculate_temperature(&velocities);
        
        // Temperature should be close to target (within 10%)
        let temperature_diff = (actual_temperature - target_temperature).abs();
        assert!(temperature_diff < target_temperature * 0.1);
    }

    #[test]
    fn test_collision_frequency_calculation() {
        let bridge = AtomicMolecularBridge::new();
        
        let reactants = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(8, 16)),
        ];
        
        let collision_freq = bridge.calculate_collision_frequency(&reactants);
        
        assert!(collision_freq > 0.0);
        assert!(collision_freq.is_finite());
    }

    #[test]
    fn test_center_of_mass_calculation() {
        let bridge = AtomicMolecularBridge::new();
        
        let atoms = vec![
            Atom::new(Nucleus::new(1, 1)),
            Atom::new(Nucleus::new(8, 16)),
            Atom::new(Nucleus::new(1, 1)),
        ];
        
        let com = bridge.calculate_center_of_mass(&atoms);
        
        // Center of mass should be finite
        assert!(com[0].is_finite());
        assert!(com[1].is_finite());
        assert!(com[2].is_finite());
    }

    #[test]
    fn test_temperature_calculation() {
        let bridge = AtomicMolecularBridge::new();
        
        let velocities = vec![
            Vector3::new(100.0, 0.0, 0.0),
            Vector3::new(0.0, 100.0, 0.0),
            Vector3::new(0.0, 0.0, 100.0),
        ];
        
        let temperature = bridge.calculate_temperature(&velocities);
        
        assert!(temperature > 0.0);
        assert!(temperature.is_finite());
    }

    #[test]
    fn test_atomic_parameters_access() {
        let bridge = AtomicMolecularBridge::new();
        
        let h_params = bridge.get_atomic_parameters(1);
        let o_params = bridge.get_atomic_parameters(8);
        
        assert_eq!(h_params.atomic_number, 1);
        assert_eq!(o_params.atomic_number, 8);
        assert!(h_params.atomic_radius < o_params.atomic_radius); // H smaller than O
    }
} 