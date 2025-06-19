//! Atomic and nuclear structure definitions for EVOLVE universe simulation
//! 
//! This module provides comprehensive data structures for atoms, nuclei, electrons,
//! and molecules, supporting realistic quantum chemistry and nuclear physics calculations.
//! 
//! Based on authoritative references:
//! - NIST Atomic Spectra Database
//! - CRC Handbook of Chemistry and Physics
//! - Quantum chemistry textbooks (Szabo & Ostlund, Levine)
//! - Nuclear data from NNDC (National Nuclear Data Center)

use nalgebra::{Vector3, Matrix3, Complex};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::particle_types::{ParticleType, ColorCharge, QuarkType, NucleonType, QuantumNumbers, 
                          GluonField, NuclearShellState, ElectronicState};

/// Quark structure for hadron composition
/// Based on QCD (Quantum Chromodynamics) and experimental data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quark {
    pub quark_type: QuarkType,
    pub color: ColorCharge,
    pub position: Vector3<f64>,          // Position within hadron (fm)
    pub momentum: Vector3<f64>,          // Momentum (GeV/c)
    pub spin: Vector3<Complex<f64>>,     // Spin state
    pub confinement_potential: f64,      // QCD confinement energy (GeV)
}

impl Quark {
    /// Create a new quark with specified type and color
    pub fn new(quark_type: QuarkType, color: ColorCharge) -> Self {
        Self {
            quark_type,
            color,
            position: Vector3::zeros(),
            momentum: Vector3::zeros(),
            spin: Vector3::zeros(),
            confinement_potential: Self::get_standard_confinement_energy(quark_type),
        }
    }
    
    /// Get standard confinement energy for quark types (GeV)
    /// Based on experimental hadron spectroscopy
    fn get_standard_confinement_energy(quark_type: QuarkType) -> f64 {
        match quark_type {
            QuarkType::Up | QuarkType::Down => 0.3,     // Light quarks
            QuarkType::Strange => 0.5,                   // Strange quark
            QuarkType::Charm => 1.3,                     // Charm quark
            QuarkType::Bottom => 4.2,                    // Bottom quark
            QuarkType::Top => 173.0,                     // Top quark (very heavy)
        }
    }
    
    /// Get electric charge in units of elementary charge
    pub fn electric_charge(&self) -> f64 {
        match self.quark_type {
            QuarkType::Up | QuarkType::Charm | QuarkType::Top => 2.0/3.0,
            QuarkType::Down | QuarkType::Strange | QuarkType::Bottom => -1.0/3.0,
        }
    }
}

/// Nucleon (proton or neutron) structure
/// Contains three quarks bound by the strong force
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nucleon {
    pub nucleon_type: NucleonType,
    pub quarks: [Quark; 3],              // Three-quark structure
    pub gluon_field: GluonField,         // Gluon field binding quarks
    pub position_in_nucleus: Vector3<f64>, // Position within atomic nucleus (fm)
    pub momentum: Vector3<f64>,          // Nucleon momentum (MeV/c)
    pub spin: Vector3<f64>,              // Nuclear spin
    pub isospin: f64,                    // Isospin quantum number
}

impl Nucleon {
    /// Create a new nucleon with proper quark composition
    pub fn new(nucleon_type: NucleonType) -> Self {
        let quarks = match nucleon_type {
            NucleonType::Proton => [
                Quark::new(QuarkType::Up, ColorCharge::Red),
                Quark::new(QuarkType::Up, ColorCharge::Green),
                Quark::new(QuarkType::Down, ColorCharge::Blue),
            ],
            NucleonType::Neutron => [
                Quark::new(QuarkType::Up, ColorCharge::Red),
                Quark::new(QuarkType::Down, ColorCharge::Green),
                Quark::new(QuarkType::Down, ColorCharge::Blue),
            ],
        };
        
        Self {
            nucleon_type,
            quarks,
            gluon_field: Vec::new(),
            position_in_nucleus: Vector3::zeros(),
            momentum: Vector3::zeros(),
            spin: Vector3::new(0.5, 0.0, 0.0), // Spin-1/2 particle
            isospin: match nucleon_type {
                NucleonType::Proton => 0.5,
                NucleonType::Neutron => -0.5,
            },
        }
    }
    
    /// Calculate total electric charge from constituent quarks
    pub fn electric_charge(&self) -> f64 {
        self.quarks.iter().map(|q| q.electric_charge()).sum()
    }
    
    /// Get rest mass in kg (CODATA 2022 values)
    pub fn rest_mass(&self) -> f64 {
        match self.nucleon_type {
            NucleonType::Proton => 1.672621898e-27,   // kg
            NucleonType::Neutron => 1.674927471e-27,  // kg
        }
    }
}

/// Atomic nucleus structure with nucleons and nuclear physics properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicNucleus {
    pub mass_number: u32,                      // A (total nucleons)
    pub atomic_number: u32,                    // Z (protons)
    pub protons: Vec<Nucleon>,                 // Proton collection
    pub neutrons: Vec<Nucleon>,                // Neutron collection
    pub binding_energy: f64,                   // Nuclear binding energy (MeV)
    pub nuclear_spin: Vector3<f64>,            // Total nuclear spin
    pub magnetic_moment: Vector3<f64>,         // Nuclear magnetic moment (μN)
    pub electric_quadrupole_moment: f64,       // Quadrupole moment (b)
    pub nuclear_radius: f64,                   // RMS charge radius (fm)
    pub shell_model_state: NuclearShellState,  // Shell model configuration
    pub position: Vector3<f64>,                // Nucleus position (m)
    pub momentum: Vector3<f64>,                // Nuclear momentum (kg⋅m/s)
    pub excitation_energy: f64,                // Nuclear excitation energy (MeV)
}

impl AtomicNucleus {
    /// Create a new atomic nucleus with specified Z and A
    pub fn new(atomic_number: u32, mass_number: u32) -> Self {
        let neutron_number = mass_number - atomic_number;
        
        // Create nucleons
        let protons: Vec<Nucleon> = (0..atomic_number)
            .map(|_| Nucleon::new(NucleonType::Proton))
            .collect();
        let neutrons: Vec<Nucleon> = (0..neutron_number)
            .map(|_| Nucleon::new(NucleonType::Neutron))
            .collect();
        
        let binding_energy = Self::calculate_binding_energy(atomic_number, mass_number);
        let nuclear_radius = Self::calculate_nuclear_radius(mass_number);
        
        Self {
            mass_number,
            atomic_number,
            protons,
            neutrons,
            binding_energy,
            nuclear_spin: Vector3::zeros(),
            magnetic_moment: Vector3::zeros(),
            electric_quadrupole_moment: 0.0,
            nuclear_radius,
            shell_model_state: HashMap::new(),
            position: Vector3::zeros(),
            momentum: Vector3::zeros(),
            excitation_energy: 0.0,
        }
    }
    
    /// Calculate nuclear binding energy using semi-empirical mass formula
    /// Based on Weizsäcker formula with modern parameters
    fn calculate_binding_energy(z: u32, a: u32) -> f64 {
        let z = z as f64;
        let a = a as f64;
        let n = a - z;
        
        // Semi-empirical mass formula coefficients (MeV)
        let a_v = 15.75;    // Volume term
        let a_s = 17.8;     // Surface term
        let a_c = 0.711;    // Coulomb term
        let a_a = 23.7;     // Asymmetry term
        let a_p = 11.18;    // Pairing term
        
        let volume_term = a_v * a;
        let surface_term = -a_s * a.powf(2.0/3.0);
        let coulomb_term = -a_c * z * z / a.powf(1.0/3.0);
        let asymmetry_term = -a_a * (n - z).powi(2) / a;
        
        // Pairing term depends on Z and N parity
        let pairing_term = if z as u32 % 2 == 0 && n as u32 % 2 == 0 {
            a_p / a.sqrt()  // Even-even
        } else if z as u32 % 2 == 1 && n as u32 % 2 == 1 {
            -a_p / a.sqrt() // Odd-odd
        } else {
            0.0             // Even-odd
        };
        
        volume_term + surface_term + coulomb_term + asymmetry_term + pairing_term
    }
    
    /// Calculate nuclear radius using empirical formula
    /// R = r₀ × A^(1/3) where r₀ ≈ 1.2 fm
    fn calculate_nuclear_radius(mass_number: u32) -> f64 {
        let r0 = 1.2e-15; // meters (1.2 fm)
        r0 * (mass_number as f64).powf(1.0/3.0)
    }
    
    /// Get nuclear mass in kg
    pub fn nuclear_mass(&self) -> f64 {
        // Sum of nucleon masses minus binding energy
        let proton_mass = 1.672621898e-27; // kg
        let neutron_mass = 1.674927471e-27; // kg
        let nucleon_mass = self.atomic_number as f64 * proton_mass + 
                          (self.mass_number - self.atomic_number) as f64 * neutron_mass;
        
        // Convert binding energy from MeV to kg
        let mev_to_kg = 1.782661907e-30; // kg/MeV (E = mc²)
        let binding_energy_kg = self.binding_energy * mev_to_kg;
        
        nucleon_mass - binding_energy_kg
    }
    
    /// Check if nucleus is stable
    pub fn is_stable(&self) -> bool {
        // Simple stability criteria based on known stable isotopes
        let n = self.mass_number - self.atomic_number;
        let z = self.atomic_number;
        
        // Magic numbers (closed shells)
        let magic_numbers = [2, 8, 20, 28, 50, 82, 126];
        let is_magic_z = magic_numbers.contains(&z);
        let is_magic_n = magic_numbers.contains(&n);
        
        // Enhanced stability for magic numbers
        if is_magic_z || is_magic_n {
            return true;
        }
        
        // General stability criteria
        z <= 83 && self.binding_energy > 0.0 && (n as f64 / z as f64) < 1.5
    }
}

/// Electron structure with quantum mechanical properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Electron {
    pub position_probability: Vec<Vec<Vec<f64>>>, // 3D probability density |ψ|²
    pub momentum_distribution: Vec<Vector3<f64>>, // Momentum space distribution
    pub spin: Vector3<Complex<f64>>,              // Electron spin state
    pub orbital_angular_momentum: Vector3<f64>,   // Orbital angular momentum
    pub quantum_numbers: QuantumNumbers,          // n, l, mₗ, mₛ
    pub binding_energy: f64,                      // Ionization energy (J)
}

impl Electron {
    /// Create a new electron in a specified orbital
    pub fn new(quantum_numbers: QuantumNumbers) -> Self {
        let binding_energy = Self::calculate_binding_energy(&quantum_numbers);
        
        Self {
            position_probability: Vec::new(),
            momentum_distribution: Vec::new(),
            spin: Vector3::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.5, 0.0)),
            orbital_angular_momentum: Vector3::zeros(),
            quantum_numbers,
            binding_energy,
        }
    }
    
    /// Calculate approximate binding energy using hydrogen-like model
    /// E_n = -13.6 eV × Z_eff² / n²
    fn calculate_binding_energy(quantum_numbers: &QuantumNumbers) -> f64 {
        let rydberg_energy = 13.6; // eV
        let z_eff = 1.0; // Effective nuclear charge (simplified)
        let n = quantum_numbers.n as f64;
        
        let binding_energy_ev = -rydberg_energy * z_eff * z_eff / (n * n);
        binding_energy_ev * 1.602176634e-19 // Convert eV to J
    }
    
    /// Get electron rest mass (CODATA 2022)
    pub fn rest_mass() -> f64 {
        9.1093837015e-31 // kg
    }
}

/// Atomic orbital types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrbitalType {
    S, P, D, F,
}

/// Atomic orbital structure with wave function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicOrbital {
    pub orbital_type: OrbitalType,
    pub wave_function: Vec<Vec<Vec<Complex<f64>>>>, // 3D wave function ψ(r)
    pub energy: f64,                                // Orbital energy (J)
    pub occupation_number: f64,                     // Electron occupation (0-2)
    pub quantum_numbers: QuantumNumbers,            // Orbital quantum numbers
}

impl AtomicOrbital {
    /// Create a new atomic orbital
    pub fn new(orbital_type: OrbitalType, quantum_numbers: QuantumNumbers) -> Self {
        Self {
            orbital_type,
            wave_function: Vec::new(),
            energy: 0.0,
            occupation_number: 0.0,
            quantum_numbers,
        }
    }
    
    /// Check if orbital can accommodate more electrons
    pub fn can_add_electron(&self) -> bool {
        self.occupation_number < 2.0
    }
}

/// Complete atom structure with nucleus and electrons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub nucleus: AtomicNucleus,              // Atomic nucleus
    pub electrons: Vec<Electron>,            // Electron collection
    pub electron_orbitals: Vec<AtomicOrbital>, // Orbital structure
    pub total_energy: f64,                   // Total atomic energy (J)
    pub ionization_energy: f64,              // First ionization energy (J)
    pub electron_affinity: f64,              // Electron affinity (J)
    pub atomic_radius: f64,                  // Atomic radius (m)
    pub position: Vector3<f64>,              // Atom position (m)
    pub velocity: Vector3<f64>,              // Atom velocity (m/s)
    pub electronic_state: ElectronicState,   // Electronic configuration
}

impl Atom {
    /// Create a new neutral atom with specified atomic number
    pub fn new(atomic_number: u32) -> Self {
        let mass_number = Self::get_typical_mass_number(atomic_number);
        let nucleus = AtomicNucleus::new(atomic_number, mass_number);
        
        // Create electrons to balance nuclear charge
        let electrons: Vec<Electron> = (0..atomic_number)
            .map(|i| {
                let quantum_numbers = Self::get_ground_state_quantum_numbers(i);
                Electron::new(quantum_numbers)
            })
            .collect();
        
        let atomic_radius = Self::calculate_atomic_radius(atomic_number);
        let ionization_energy = Self::get_ionization_energy(atomic_number);
        
        Self {
            nucleus,
            electrons,
            electron_orbitals: Vec::new(),
            total_energy: 0.0,
            ionization_energy,
            electron_affinity: 0.0,
            atomic_radius,
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            electronic_state: HashMap::new(),
        }
    }
    
    /// Get typical mass number for an element (most abundant isotope)
    fn get_typical_mass_number(atomic_number: u32) -> u32 {
        match atomic_number {
            1 => 1,    // Hydrogen
            2 => 4,    // Helium
            3 => 7,    // Lithium
            6 => 12,   // Carbon
            7 => 14,   // Nitrogen
            8 => 16,   // Oxygen
            26 => 56,  // Iron
            _ => atomic_number * 2, // Rough approximation
        }
    }
    
    /// Get ground state quantum numbers for nth electron
    fn get_ground_state_quantum_numbers(electron_index: u32) -> QuantumNumbers {
        // Simplified electron configuration (Aufbau principle)
        match electron_index {
            0 | 1 => QuantumNumbers { n: 1, l: 0, m_l: 0, m_s: 0.5 },
            2..=9 => QuantumNumbers { n: 2, l: 0, m_l: 0, m_s: 0.5 },
            _ => QuantumNumbers { n: 3, l: 0, m_l: 0, m_s: 0.5 }, // Simplified
        }
    }
    
    /// Calculate atomic radius using empirical data
    fn calculate_atomic_radius(atomic_number: u32) -> f64 {
        // Rough approximation based on periodic trends (pm to m)
        let radius_pm = match atomic_number {
            1 => 37.0,   // Hydrogen
            2 => 32.0,   // Helium
            3 => 134.0,  // Lithium
            6 => 67.0,   // Carbon
            8 => 48.0,   // Oxygen
            26 => 156.0, // Iron
            _ => 100.0,  // Default
        };
        radius_pm * 1e-12 // Convert pm to m
    }
    
    /// Get first ionization energy (experimental values in J)
    fn get_ionization_energy(atomic_number: u32) -> f64 {
        let ionization_ev = match atomic_number {
            1 => 13.598,  // Hydrogen
            2 => 24.587,  // Helium
            3 => 5.392,   // Lithium
            6 => 11.260,  // Carbon
            8 => 13.618,  // Oxygen
            26 => 7.902,  // Iron
            _ => 10.0,    // Default
        };
        ionization_ev * 1.602176634e-19 // Convert eV to J
    }
    
    /// Get particle type corresponding to this atom
    pub fn get_particle_type(&self) -> ParticleType {
        match self.nucleus.atomic_number {
            1 => ParticleType::HydrogenAtom,
            2 => ParticleType::HeliumAtom,
            6 => ParticleType::CarbonAtom,
            8 => ParticleType::OxygenAtom,
            26 => ParticleType::IronAtom,
            _ => ParticleType::HydrogenAtom, // Default fallback
        }
    }
    
    /// Calculate total atomic mass
    pub fn total_mass(&self) -> f64 {
        let electron_mass = Electron::rest_mass();
        self.nucleus.nuclear_mass() + self.electrons.len() as f64 * electron_mass
    }
}

/// Chemical bond types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BondType {
    Ionic, 
    Covalent, 
    Metallic, 
    HydrogenBond, 
    VanDerWaals,
}

/// Chemical bond structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalBond {
    pub atom_indices: (usize, usize),    // Indices of bonded atoms
    pub bond_type: BondType,             // Type of chemical bond
    pub bond_length: f64,                // Bond length (m)
    pub bond_energy: f64,                // Bond dissociation energy (J)
    pub bond_order: f64,                 // Bond order (1, 2, 3, etc.)
    pub electron_density: f64,           // Electron density at bond critical point
    pub overlap_integral: f64,           // Orbital overlap integral
}

impl ChemicalBond {
    /// Create a new chemical bond between two atoms
    pub fn new(atom1_idx: usize, atom2_idx: usize, bond_type: BondType) -> Self {
        let (bond_length, bond_energy) = Self::get_typical_bond_parameters(bond_type);
        
        Self {
            atom_indices: (atom1_idx, atom2_idx),
            bond_type,
            bond_length,
            bond_energy,
            bond_order: 1.0,
            electron_density: 0.0,
            overlap_integral: 0.0,
        }
    }
    
    /// Get typical bond parameters for different bond types
    fn get_typical_bond_parameters(bond_type: BondType) -> (f64, f64) {
        match bond_type {
            BondType::Covalent => (1.5e-10, 6.0e-19),      // ~1.5 Å, ~4 eV
            BondType::Ionic => (2.0e-10, 8.0e-19),         // ~2.0 Å, ~5 eV
            BondType::Metallic => (2.5e-10, 4.0e-19),      // ~2.5 Å, ~2.5 eV
            BondType::HydrogenBond => (2.8e-10, 0.8e-19),  // ~2.8 Å, ~0.5 eV
            BondType::VanDerWaals => (4.0e-10, 0.16e-19),  // ~4.0 Å, ~0.1 eV
        }
    }
}

/// Type aliases for molecular structures
pub type MolecularOrbital = AtomicOrbital;
pub type VibrationalMode = Vector3<f64>;
pub type PotentialEnergySurface = Vec<Vec<Vec<f64>>>;
pub type ReactionCoordinate = Vector3<f64>;

/// Molecule structure with atoms and bonds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub atoms: Vec<Atom>,                           // Constituent atoms
    pub bonds: Vec<ChemicalBond>,                   // Chemical bonds
    pub molecular_orbitals: Vec<MolecularOrbital>,  // Molecular orbitals
    pub vibrational_modes: Vec<VibrationalMode>,    // Vibrational normal modes
    pub rotational_constants: Vector3<f64>,         // Rotational constants (Hz)
    pub dipole_moment: Vector3<f64>,                // Electric dipole moment (C⋅m)
    pub polarizability: Matrix3<f64>,               // Polarizability tensor (C⋅m²/N)
    pub potential_energy_surface: PotentialEnergySurface, // PES for reactions
    pub reaction_coordinates: Vec<ReactionCoordinate>,    // Reaction pathways
}

impl Molecule {
    /// Create a new molecule from a collection of atoms
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            bonds: Vec::new(),
            molecular_orbitals: Vec::new(),
            vibrational_modes: Vec::new(),
            rotational_constants: Vector3::zeros(),
            dipole_moment: Vector3::zeros(),
            polarizability: Matrix3::zeros(),
            potential_energy_surface: Vec::new(),
            reaction_coordinates: Vec::new(),
        }
    }
    
    /// Calculate molecular mass
    pub fn molecular_mass(&self) -> f64 {
        self.atoms.iter().map(|atom| atom.total_mass()).sum()
    }
    
    /// Get center of mass
    pub fn center_of_mass(&self) -> Vector3<f64> {
        let total_mass = self.molecular_mass();
        if total_mass > 0.0 {
            let weighted_position: Vector3<f64> = self.atoms.iter()
                .map(|atom| atom.position * atom.total_mass())
                .sum();
            weighted_position / total_mass
        } else {
            Vector3::zeros()
        }
    }
    
    /// Add a chemical bond between two atoms
    pub fn add_bond(&mut self, atom1_idx: usize, atom2_idx: usize, bond_type: BondType) {
        if atom1_idx < self.atoms.len() && atom2_idx < self.atoms.len() {
            let bond = ChemicalBond::new(atom1_idx, atom2_idx, bond_type);
            self.bonds.push(bond);
        }
    }
} 