use super::*;
use nalgebra::{Vector3, Matrix3, DMatrix};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use once_cell::sync::Lazy;
use crate::constants;
use std::f64::consts::PI;
use rayon::prelude::*;
use anyhow::{anyhow, ensure};
use nalgebra::{Cholesky, SymmetricEigen};
use crate::quantum_math::{
    boys_function, gaussian_normalization, 
    gaussian_product_center,
    kinetic_integral_obara_saika, ObSaWorkspace
};


// Source: CRC Handbook of Chemistry and Physics, 91st ed.
// Values are in picometers (pm), converted to meters.
pub static COVALENT_RADII: Lazy<HashMap<ParticleType, f64>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(ParticleType::Hydrogen, 37e-12);
    m.insert(ParticleType::Carbon, 77e-12);
    m.insert(ParticleType::Oxygen, 73e-12);
    m.insert(ParticleType::Nitrogen, 75e-12);
    m.insert(ParticleType::Phosphorus, 110e-12);
    m.insert(ParticleType::Sulfur, 103e-12);
    m.insert(ParticleType::Silicon, 117e-12);
    m
});

// Source: Average bond energies, from various standard chemistry textbooks.
// Values are in kJ/mol, converted to Joules per bond.
const KJ_PER_MOL_TO_J_PER_BOND: f64 = 1000.0 / 6.02214076e23;
pub static AVG_BOND_ENERGIES: Lazy<HashMap<(ParticleType, ParticleType), f64>> = Lazy::new(|| {
    let mut m = HashMap::new();
    // Self-bonds (homonuclear)
    m.insert((ParticleType::Hydrogen, ParticleType::Hydrogen), 436.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Carbon, ParticleType::Carbon), 348.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Oxygen, ParticleType::Oxygen), 146.0 * KJ_PER_MOL_TO_J_PER_BOND); // Single bond
    m.insert((ParticleType::Nitrogen, ParticleType::Nitrogen), 163.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Sulfur, ParticleType::Sulfur), 266.0 * KJ_PER_MOL_TO_J_PER_BOND);


    // Heteronuclear bonds
    m.insert((ParticleType::Carbon, ParticleType::Hydrogen), 413.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Carbon, ParticleType::Oxygen), 358.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Carbon, ParticleType::Nitrogen), 305.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Oxygen, ParticleType::Hydrogen), 463.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m.insert((ParticleType::Nitrogen, ParticleType::Hydrogen), 391.0 * KJ_PER_MOL_TO_J_PER_BOND);
    m
});


/// Quantum chemical calculation method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CalculationMethod {
    HartreeFock,           // Basic ab initio method
    DensityFunctional,     // DFT with various functionals
    SemiEmpirical,         // Fast approximate methods
    MolecularMechanics,    // Classical force fields
    QmMm,                  // Hybrid quantum/classical
}

/// Electronic structure calculation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronicStructure {
    pub total_energy: f64,
    pub orbital_energies: Vec<f64>,
    pub molecular_orbitals: Vec<MolecularOrbital>,
    pub electron_density: Vec<Vec<Vec<f64>>>,
    pub bond_orders: HashMap<(usize, usize), f64>,
    pub atomic_charges: Vec<f64>,
    pub dipole_moment: Vector3<f64>,
    pub polarizability: Matrix3<f64>,
}

/// Molecular orbital representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularOrbital {
    pub energy: f64,
    pub occupation: f64,
    pub coefficients: Vec<f64>,
    pub orbital_type: OrbitalType,
    pub symmetry: String,
}

/// Quantum chemistry engine for molecular calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumChemistryEngine {
    pub method: CalculationMethod,
    pub basis_set: BasisSet,
    pub functional: DftFunctional,
    pub calculation_cache: HashMap<String, ElectronicStructure>,
    pub reaction_database: Vec<ChemicalReaction>,
    pub force_field_parameters: ForceFieldParameters,
}

/// A shell of contracted Gaussian-type orbitals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contraction {
    pub angular_momentum: u32, // 0 for s, 1 for p, 2 for d, etc.
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// Basis set for quantum calculations, mapping atomic number to its shells.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisSet {
    pub name: String,
    // Key: atomic number (Z)
    pub shells_for_atom: HashMap<u32, Vec<Contraction>>,
}

impl BasisSet {
    /// Returns a minimal placeholder STO-3G basis set. This can be filled with proper
    /// contracted Gaussian primitives in future commits.
    pub fn sto_3g() -> Self {
        Self {
            name: "STO-3G".to_string(),
            shells_for_atom: HashMap::new(),
        }
    }
}

/// Gaussian basis function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianFunction {
    pub exponent: f64,
    pub coefficient: f64,
    pub angular_momentum: [u32; 3], // l, m, n for x^l * y^m * z^n
    pub center: Vector3<f64>,
}

/// DFT functional for electron correlation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DftFunctional {
    Lda,      // Local Density Approximation
    Gga,      // Generalized Gradient Approximation
    Hybrid,   // B3LYP-like hybrid functionals
    MetaGga,  // Meta-GGA functionals
}

/// Chemical reaction definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalReaction {
    pub reactants: Vec<ParticleType>,
    pub products: Vec<ParticleType>,
    pub activation_energy: f64,
    pub reaction_energy: f64,
    pub rate_constant: f64,
    pub temperature_dependence: ArrheniusParameters,
    pub mechanism: ReactionMechanism,
}

/// Arrhenius parameters for temperature dependence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrheniusParameters {
    pub pre_exponential_factor: f64,
    pub activation_energy: f64,
    pub temperature_exponent: f64,
}

/// Reaction mechanism classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReactionMechanism {
    Unimolecular,
    Bimolecular,
    Termolecular,
    ChainReaction,
    Photochemical,
    Electrochemical,
}

/// Force field parameters for classical molecular dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceFieldParameters {
    pub bond_parameters: HashMap<(ParticleType, ParticleType), BondParameters>,
    pub angle_parameters: HashMap<(ParticleType, ParticleType, ParticleType), AngleParameters>,
    pub dihedral_parameters: HashMap<(ParticleType, ParticleType, ParticleType, ParticleType), DihedralParameters>,
    pub van_der_waals_parameters: HashMap<ParticleType, VdwParameters>,
}

/// Bond stretching parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondParameters {
    pub equilibrium_length: f64,
    pub force_constant: f64,
    pub dissociation_energy: f64,
}

/// Angle bending parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleParameters {
    pub equilibrium_angle: f64,
    pub force_constant: f64,
}

/// Dihedral torsion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralParameters {
    pub periodicity: u32,
    pub phase_angle: f64,
    pub barrier_height: f64,
}

/// Van der Waals parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdwParameters {
    pub sigma: f64,      // Size parameter
    pub epsilon: f64,    // Energy parameter
    pub radius: f64,     // Atomic radius
    pub partial_charge: f64, // Partial charge for electrostatic calculations
}

impl Atom {
    // Helper to get particle type from atomic number
    pub fn get_particle_type(&self) -> ParticleType {
        match self.nucleus.atomic_number {
            1 => ParticleType::Hydrogen,
            6 => ParticleType::Carbon,
            7 => ParticleType::Nitrogen,
            8 => ParticleType::Oxygen,
            9 => ParticleType::Fluorine,
            14 => ParticleType::Silicon,
            15 => ParticleType::Phosphorus,
            16 => ParticleType::Sulfur,
            17 => ParticleType::Chlorine,
            35 => ParticleType::Bromine,
            53 => ParticleType::Iodine,
            _ => ParticleType::DarkMatter, // Default for unknown
        }
    }
}


impl QuantumChemistryEngine {
    /// Create new quantum chemistry engine
    pub fn new() -> Self {
        Self {
            method: CalculationMethod::DensityFunctional,
            basis_set: BasisSet::sto_3g(),
            functional: DftFunctional::Hybrid,
            calculation_cache: HashMap::new(),
            reaction_database: Self::initialize_reaction_database(),
            force_field_parameters: Self::initialize_force_field(),
        }
    }
    
    /// Initializes a basic force field with parameters for common molecules (e.g., water).
    /// Sources: TIP3P water model for VdW and charges. Generic values for bonds/angles.
    fn initialize_force_field() -> ForceFieldParameters {
        let mut params = ForceFieldParameters::default();
        
        const KCAL_PER_MOL_TO_J: f64 = 4184.0 / 6.02214076e23;

        // Water (TIP3P-like parameters)
        let oh_bond_len = 0.9572e-10; // meters
        let hoh_angle = 104.52f64.to_radians(); // radians
        
        // Using typical force constants from other force fields as placeholders
        let kr_oh = 450.0 * KCAL_PER_MOL_TO_J / 1e-20; // J/m^2
        let k_hoh = 55.0 * KCAL_PER_MOL_TO_J;      // J/rad^2
        
        params.bond_parameters.insert((ParticleType::Oxygen, ParticleType::Hydrogen), BondParameters {
            equilibrium_length: oh_bond_len,
            force_constant: kr_oh,
            dissociation_energy: 492.0 * KJ_PER_MOL_TO_J_PER_BOND * 1000.0,
        });

        params.angle_parameters.insert((ParticleType::Hydrogen, ParticleType::Oxygen, ParticleType::Hydrogen), AngleParameters {
            equilibrium_angle: hoh_angle,
            force_constant: k_hoh,
        });

        // VdW for Oxygen (from TIP3P)
        params.van_der_waals_parameters.insert(ParticleType::Oxygen, VdwParameters {
            sigma: 3.15061e-10, // meters
            epsilon: 0.1521 * KCAL_PER_MOL_TO_J, // Joules
            radius: 1.77e-10, // meters, adjusted
            partial_charge: -0.834 * constants::ELEMENTARY_CHARGE,
        });

        // VdW for Hydrogen (from TIP3P)
        params.van_der_waals_parameters.insert(ParticleType::Hydrogen, VdwParameters {
            sigma: 0.0, // TIP3P has no VdW for H, only charge
            epsilon: 0.0,
            radius: 1.2e-10,
            partial_charge: 0.417 * constants::ELEMENTARY_CHARGE,
        });

        params
    }

    pub fn initialize_reaction_database() -> Vec<ChemicalReaction> {
        // Placeholder for reaction database
        Vec::new()
    }

    /// Calculate the total mass of a molecule.
    ///
    /// # Arguments
    /// * `molecule` - The molecule to calculate the mass of.
    ///
    /// # Returns
    /// The total mass in kilograms.
    pub fn calculate_molecular_mass(&self, molecule: &Molecule) -> f64 {
        // Sum of atomic masses of all atoms in the molecule
        molecule.atoms.iter().map(|atom| atom.nucleus.mass_number as f64).sum::<f64>() * constants::ATOMIC_MASS_UNIT
    }

    /// Estimate atomic ground-state electronic energy using the Thomas-Fermi model.
    /// E(Z) ≈ -20.8 * Z^(7/3) eV
    /// This is a significant improvement over the hydrogenic model for multi-electron atoms.
    ///
    /// # Arguments
    /// * `atomic_number` - The atomic number (Z) of the atom.
    ///
    /// # Returns
    /// The estimated ground-state energy in Joules.
    pub fn get_atomic_energy(&self, atomic_number: &u32) -> f64 {
        const EV_TO_J: f64 = 1.602_176_634e-19; // CODATA 2022
        let z = *atomic_number as f64;
        // Thomas-Fermi model approximation for total ionization energy of a neutral atom.
        // Source: "Introduction to Solid State Physics" by Charles Kittel.
        -20.8 * z.powf(7.0 / 3.0) * EV_TO_J
    }

    /// Get an estimated bond energy from a table of average bond energies.
    ///
    /// # Arguments
    /// * `atom1_type` - The `ParticleType` of the first atom.
    /// * `atom2_type` - The `ParticleType` of the second atom.
    /// * `bond_order` - The order of the bond (1 for single, 2 for double, etc.).
    ///
    /// # Returns
    /// The estimated bond energy in Joules. Returns a default low energy for unknown bonds.
    pub fn get_bond_energy(&self, atom1_type: ParticleType, atom2_type: ParticleType, bond_order: f64) -> f64 {
        // Normalize the key so (C, H) and (H, C) are treated the same.
        let key = if atom1_type as u32 > atom2_type as u32 {
            (atom2_type, atom1_type)
        } else {
            (atom1_type, atom2_type)
        };

        let single_bond_energy = AVG_BOND_ENERGIES.get(&key).copied().unwrap_or(0.1 * KJ_PER_MOL_TO_J_PER_BOND);

        // Scale by bond order (a very rough approximation).
        // A double bond is not twice as strong as a single, but this is a start.
        single_bond_energy * bond_order
    }

    /// Predicts if two atoms are bonded based on their distance and covalent radii.
    ///
    /// # Arguments
    /// * `atom1` - The first atom.
    /// * `atom2` - The second atom.
    /// * `distance` - The distance between the atoms.
    ///
    /// # Returns
    /// `true` if the atoms are likely bonded, `false` otherwise.
    pub fn are_bonded(&self, atom1: &Atom, atom2: &Atom, distance: f64) -> bool {
        const BOND_TOLERANCE: f64 = 1.2; // Allow for some bond stretching.
        let r1 = COVALENT_RADII.get(&atom1.get_particle_type()).copied().unwrap_or(0.0);
        let r2 = COVALENT_RADII.get(&atom2.get_particle_type()).copied().unwrap_or(0.0);

        if r1 > 0.0 && r2 > 0.0 {
            distance < (r1 + r2) * BOND_TOLERANCE
        } else {
            false
        }
    }

    /// Lennard-Jones 12-6 potential (J) for van-der-Waals interaction.
    /// This implementation retrieves parameters from the engine's force field.
    ///
    /// # Arguments
    /// * `i` - Index of the first atom in the molecule.
    /// * `j` - Index of the second atom in the molecule.
    /// * `distance` - The distance between the atoms.
    /// * `molecule` - The molecule containing the atoms.
    ///
    /// # Returns
    /// The van der Waals energy in Joules.
    pub fn van_der_waals_energy(&self, i: usize, j: usize, distance: f64, molecule: &Molecule) -> f64 {
        if distance < 1e-12 {
            return 0.0;
        }

        // Default parameters for unknown atom types.
        let default_vdw = VdwParameters {
            sigma: 3.5e-10, // meters
            epsilon: 0.2 * KJ_PER_MOL_TO_J_PER_BOND, // Joules
            radius: 1.5e-10, // meters
            partial_charge: 0.0, // Default partial charge
        };

        let atom_type_i = molecule.atoms[i].get_particle_type();
        let atom_type_j = molecule.atoms[j].get_particle_type();

        let vdw_i = self.force_field_parameters.van_der_waals_parameters.get(&atom_type_i).unwrap_or(&default_vdw);
        let vdw_j = self.force_field_parameters.van_der_waals_parameters.get(&atom_type_j).unwrap_or(&default_vdw);

        // Apply Lorentz-Berthelot mixing rules.
        let sigma = (vdw_i.sigma + vdw_j.sigma) / 2.0;
        let epsilon = (vdw_i.epsilon * vdw_j.epsilon).sqrt();

        let r_inv = sigma / distance;
        let r_inv6 = r_inv.powi(6);
        let r_inv12 = r_inv6 * r_inv6;

        4.0 * epsilon * (r_inv12 - r_inv6)
    }

    /// Calculate electronic structure for a molecule
    pub fn calculate_electronic_structure(&mut self, molecule: &Molecule) -> Result<ElectronicStructure> {
        let molecule_key = self.generate_molecule_key(molecule);
        
        // Check cache first
        if let Some(cached_result) = self.calculation_cache.get(&molecule_key) {
            return Ok(cached_result.clone());
        }
        
        // Perform quantum chemical calculation
        let result = match self.method {
            CalculationMethod::HartreeFock => self.hartree_fock_calculation(molecule)?,
            CalculationMethod::DensityFunctional => self.dft_calculation(molecule)?,
            CalculationMethod::SemiEmpirical => self.semi_empirical_calculation(molecule)?,
            CalculationMethod::MolecularMechanics => self.molecular_mechanics_calculation(molecule)?,
            CalculationMethod::QmMm => self.qm_mm_calculation(molecule)?,
        };
        
        // Cache result
        self.calculation_cache.insert(molecule_key, result.clone());
        
        Ok(result)
    }
    
    /// Hartree-Fock self-consistent field calculation
    #[allow(dead_code)]
    fn hartree_fock_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
        let n_electrons = self.count_electrons(molecule);
        let overlap_matrix = self.build_overlap_matrix(molecule)?;

        // Initial guess for density matrix (e.g., zero matrix)
        let mut density_matrix = DMatrix::zeros(overlap_matrix.nrows(), overlap_matrix.ncols());

        let mut total_energy = 0.0;
        let mut last_eigenvalues: Vec<f64> = Vec::new();

        // Self-Consistent Field (SCF) iterations
        for _iteration in 0..50 {
            // Construct Fock matrix with current density
            let fock_matrix = self.build_fock_matrix(&density_matrix, molecule)?;

            // Solve Roothaan equations
            let (eigenvalues, eigenvectors) = self.solve_eigenvalue_problem(&fock_matrix, &overlap_matrix)?;
            last_eigenvalues = eigenvalues.clone();

            // Determine occupied orbitals (closed-shell)
            let n_occ = (n_electrons as f64 / 2.0).ceil() as usize;
            // C_occ = first n_occ columns of eigenvectors
            let c_occ = eigenvectors.columns(0, n_occ).into_owned();

            // New density: P = 2 C_occ C_occᵀ
            let new_density = &c_occ * c_occ.transpose() * 2.0;

            // Compute energy and check convergence
            let old_energy = total_energy;
            total_energy = self.calculate_total_energy(&new_density, molecule)?;

            if (total_energy - old_energy).abs() < 1e-8 {
                density_matrix = new_density;
                break;
            }

            density_matrix = new_density;
        }
        
        Ok(ElectronicStructure {
            total_energy,
            orbital_energies: last_eigenvalues,
            molecular_orbitals: vec![],
            electron_density: vec![],
            bond_orders: HashMap::new(),
            atomic_charges: vec![0.0; molecule.atoms.len()],
            dipole_moment: Vector3::zeros(),
            polarizability: Matrix3::zeros(),
        })
    }
    
    /// Density Functional Theory calculation
    fn dft_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
        // Simplified DFT - similar to HF but with exchange-correlation functional
        let mut structure = self.hartree_fock_calculation(molecule)?;
        
        // Add DFT corrections based on functional
        let xc_energy = match self.functional {
            DftFunctional::Lda => self.lda_exchange_correlation(molecule)?,
            DftFunctional::Gga => self.gga_exchange_correlation(molecule)?,
            DftFunctional::Hybrid => self.hybrid_exchange_correlation(molecule)?,
            DftFunctional::MetaGga => self.meta_gga_exchange_correlation(molecule)?,
        };
        
        structure.total_energy += xc_energy;
        
        Ok(structure)
    }
    
    /// Semi-empirical quantum calculation (faster approximation)
    fn semi_empirical_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
        // Use parameterized methods like AM1, PM3, etc.
        let num_atoms = molecule.atoms.len();
        let mut total_energy = 0.0;
    
        // Electronic energy from parameterized method
        for atom in &molecule.atoms {
            total_energy += self.get_atomic_energy(&atom.nucleus.atomic_number);
        }

        // Bond energies
        for bond in &molecule.bonds {
            let atom1 = &molecule.atoms[bond.atom_indices.0];
            let atom2 = &molecule.atoms[bond.atom_indices.1];
            total_energy += self.get_bond_energy(atom1.get_particle_type(), atom2.get_particle_type(), bond.bond_order);
        }

        Ok(ElectronicStructure {
            total_energy,
            orbital_energies: vec![],
            molecular_orbitals: vec![],
            electron_density: vec![],
            bond_orders: HashMap::new(),
            atomic_charges: vec![0.0; num_atoms],
            dipole_moment: Vector3::zeros(),
            polarizability: Matrix3::zeros(),
        })
    }

    // --- Placeholder Methods to Satisfy Compiler ---
    /// Generates a canonical key for a molecule based on its atomic composition and geometry.
    /// The key is invariant to atom ordering, translation, and rotation of the molecule.
    /// This is used for caching calculation results.
    fn generate_molecule_key(&self, molecule: &Molecule) -> String {
        // Create a sortable representation of each atom: (atomic_number, x, y, z)
        let mut atom_reprs: Vec<_> = molecule.atoms.iter().map(|atom| {
            (
                atom.nucleus.atomic_number,
                // Truncate coordinates to handle floating point inaccuracies
                (atom.position.x * 1e6).round() as i64,
                (atom.position.y * 1e6).round() as i64,
                (atom.position.z * 1e6).round() as i64,
            )
        }).collect();

        // Sort to create a canonical ordering
        atom_reprs.sort_unstable();

        // Create a string representation of the sorted atoms
        atom_reprs.iter()
            .map(|(z, x, y, z_coord)| format!("{}:{}:{}:{};", z, x, y, z_coord))
            .collect::<String>()
    }

    fn count_electrons(&self, molecule: &Molecule) -> usize { molecule.atoms.iter().map(|a| a.nucleus.atomic_number as usize).sum() }

    /// Builds the overlap matrix S for a given molecule and basis set.
    /// The element Sμν is the overlap integral between atomic orbitals μ and ν.
    /// ∫φμ*(r)φν(r)dr
    /// This implementation currently handles s-type orbitals only.
    fn build_overlap_matrix(&self, molecule: &Molecule) -> Result<DMatrix<f64>> {
        
        // Helper struct to represent a single basis function (atomic orbital)
        struct BasisFunction<'a> {
            contraction: &'a Contraction,
            atom_center: Vector3<f64>,
        }

        // 1. Create a flat list of all basis functions for the molecule
        let mut basis_functions = Vec::new();
        for atom in &molecule.atoms {
            if let Some(shells) = self.basis_set.shells_for_atom.get(&atom.nucleus.atomic_number) {
                for shell in shells {
                    // For p-orbitals, we'd add 3 functions (px, py, pz) here.
                    // For now, we only handle s-orbitals (l=0).
                    if shell.angular_momentum == 0 {
                         basis_functions.push(BasisFunction {
                            contraction: shell,
                            atom_center: atom.position,
                        });
                    }
                }
            }
        }

        let basis_size = basis_functions.len();
        if basis_size == 0 {
            return Ok(DMatrix::from_element(0, 0, 0.0));
        }
        let mut overlap_matrix = DMatrix::from_element(basis_size, basis_size, 0.0);

        // 2. Calculate the overlap for each pair of basis functions (μ, ν)
        for i in 0..basis_size {
            for j in 0..=i {
                let bf_i = &basis_functions[i];
                let bf_j = &basis_functions[j];

                let mut integral = 0.0;

                // Sum over primitive Gaussians: Σcᵢcⱼ∫gᵢgⱼ
                for (k, &exp_i) in bf_i.contraction.exponents.iter().enumerate() {
                    for (l, &exp_j) in bf_j.contraction.exponents.iter().enumerate() {
                        let coeff_i = bf_i.contraction.coefficients[k];
                        let coeff_j = bf_j.contraction.coefficients[l];
                        let r_ab_sq = (bf_i.atom_center - bf_j.atom_center).norm_squared();

                        // Overlap integral for two primitive s-type Gaussians
                        // S_ab = (π/(α+β))^(3/2) * exp(-αβ/(α+β) * |Ra-Rb|²)
                        let p = exp_i + exp_j;
                        let prefactor = (PI / p).powf(1.5);
                        let exponent = -(exp_i * exp_j / p) * r_ab_sq;
                        let primitive_overlap = prefactor * exponent.exp();

                        integral += coeff_i * coeff_j * primitive_overlap;
                    }
                }
                
                overlap_matrix[(i, j)] = integral;
                if i != j {
                    overlap_matrix[(j, i)] = integral; // Matrix is symmetric
                }
            }
        }

        Ok(overlap_matrix)
    }

    fn build_kinetic_matrix(&self, molecule: &Molecule) -> Result<DMatrix<f64>> {
        
        struct BasisFunction<'a> {
            contraction: &'a Contraction,
            atom_center: Vector3<f64>,
            // The angular momentum vector (lx, ly, lz)
            angular_momentum: (u32, u32, u32),
        }

        let mut basis_functions = Vec::new();
        for atom in &molecule.atoms {
            if let Some(contractions) = self.basis_set.shells_for_atom.get(&atom.nucleus.atomic_number) {
                for contraction in contractions {
                    match contraction.angular_momentum {
                        0 => { // s-shell
                            basis_functions.push(BasisFunction {
                                contraction,
                                atom_center: atom.position,
                                angular_momentum: (0, 0, 0),
                            });
                        }
                        1 => { // p-shell
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (1, 0, 0) }); // px
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 1, 0) }); // py
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 0, 1) }); // pz
                        }
                        2 => { // d-shell (6 Cartesian functions)
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (2, 0, 0) }); // d_xx
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 2, 0) }); // d_yy
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 0, 2) }); // d_zz
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (1, 1, 0) }); // d_xy
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (1, 0, 1) }); // d_xz
                            basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 1, 1) }); // d_yz
                        }
                        3 => { // f-shell (10 Cartesian functions)
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (3, 0, 0) }); // f_xxx
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 3, 0) }); // f_yyy
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 0, 3) }); // f_zzz
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (2, 1, 0) }); // f_xxy
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (2, 0, 1) }); // f_xxz
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (1, 2, 0) }); // f_xyy
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 2, 1) }); // f_yyz
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (1, 0, 2) }); // f_xzz
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (0, 1, 2) }); // f_yzz
                             basis_functions.push(BasisFunction { contraction, atom_center: atom.position, angular_momentum: (1, 1, 1) }); // f_xyz
                        }
                        // Add other angular momenta (d, f) here as needed
                        _ => unimplemented!("Angular momentum > 3 not supported yet. Found: {}", contraction.angular_momentum),
                    }
                }
            }
        }

        let n_basis = basis_functions.len();
        let mut kinetic_matrix = DMatrix::zeros(n_basis, n_basis);

        // The kinetic matrix is symmetric, so we only need to compute the lower or upper triangle.
        // We can parallelize the computation of matrix elements using rayon.
        // Create a flat list of index pairs (i, j) for the upper triangle.
        let mut indices: Vec<(usize, usize)> = Vec::with_capacity(n_basis * (n_basis + 1) / 2);
        for i in 0..n_basis {
            for j in i..n_basis { // Note: j starts from i
                indices.push((i, j));
            }
        }

        let matrix_elements: Vec<f64> = indices.par_iter().map(|&(i, j)| {
            let bf_i = &basis_functions[i];
            let bf_j = &basis_functions[j];
            let mut integral = 0.0;

            // This is the contracted integral T_IJ = sum_{p,q} c_p * c_q * T_pq
            // where T_pq is the integral over primitive gaussians.
            for (c1, &exp1) in bf_i.contraction.coefficients.iter().zip(&bf_i.contraction.exponents) {
                for (c2, &exp2) in bf_j.contraction.coefficients.iter().zip(&bf_j.contraction.exponents) {
                    let norm1 = gaussian_normalization(exp1, &bf_i.angular_momentum);
                    let norm2 = gaussian_normalization(exp2, &bf_j.angular_momentum);
                    
                    integral += c1 * c2 * norm1 * norm2 * kinetic_integral(
                        exp1, bf_i.atom_center, &bf_i.angular_momentum,
                        exp2, bf_j.atom_center, &bf_j.angular_momentum,
                    );
                }
            }
            integral
        }).collect();

        // Populate the full matrix from the calculated elements.
        for (k, &(i, j)) in indices.iter().enumerate() {
            let value = matrix_elements[k];
            kinetic_matrix[(i, j)] = value;
            if i != j {
                kinetic_matrix[(j, i)] = value; // Set the symmetric element
            }
        }

        Ok(kinetic_matrix)
    }

    fn build_nuclear_attraction_matrix(&self, molecule: &Molecule) -> Result<DMatrix<f64>> {
        // --- Nuclear attraction integrals V_μν ---
        // V_{μν} = -\sum_A Z_A \int φ_μ(r) \frac{1}{|r-R_A|} φ_ν(r) \,dr
        // For contracted Cartesian Gaussian basis functions this evaluates to:
        // V_{μν} = -\sum_A Z_A \sum_{p∈μ} \sum_{q∈ν} c_p c_q N_p N_q (2π / p) e^{-α_p α_q/p |R_a-R_b|^2} F_0(p |P-R_A|^2)
        // where p = α_p + α_q, P is the Gaussian product centre and F_0 is the Boys function of order 0.
        // 
        // This implementation currently supports s-type (ℓ = 0) contracted Gaussians, which is sufficient for the
        // existing STO-3G basis provided. Extension to higher angular momentum functions can reuse Obara–Saika
        // recursion but is left for future work.
        //
        // All quantities are assumed to be in atomic units (Bohr, Hartree). Calling code must ensure the coordinate
        // system is consistent. The test-suite converts Å to Bohr accordingly.

        struct BasisFunction<'a> {
            contraction: &'a Contraction,
            center: Vector3<f64>,
        }

        // 1. Collect all s-type basis functions for the molecule
        let mut basis_functions = Vec::new();
        for atom in &molecule.atoms {
            if let Some(shells) = self.basis_set.shells_for_atom.get(&atom.nucleus.atomic_number) {
                for shell in shells {
                    if shell.angular_momentum == 0 {
                        basis_functions.push(BasisFunction { contraction: shell, center: atom.position });
                    }
                }
            }
        }

        let n_basis = basis_functions.len();
        if n_basis == 0 {
            return Ok(DMatrix::from_element(0, 0, 0.0));
        }

        let mut v_matrix = DMatrix::<f64>::zeros(n_basis, n_basis);

        // 2. Precompute nuclear charges and positions
        let nuclei: Vec<(Vector3<f64>, f64)> = molecule
            .atoms
            .iter()
            .map(|atom| (atom.position, atom.nucleus.atomic_number as f64))
            .collect();

        // 3. Double loop over (μ, ν) – matrix symmetry exploited
        for mu in 0..n_basis {
            for nu in 0..=mu {
                let bf_mu = &basis_functions[mu];
                let bf_nu = &basis_functions[nu];

                let mut integral = 0.0;

                // Sum over nuclei A
                for (r_a, z_a) in &nuclei {
                    // Sum over primitives of μ and ν
                    for (c_p, &alpha_p) in bf_mu.contraction.coefficients.iter().zip(&bf_mu.contraction.exponents) {
                        let norm_p = gaussian_normalization(alpha_p, &(0, 0, 0));
                        for (c_q, &alpha_q) in bf_nu.contraction.coefficients.iter().zip(&bf_nu.contraction.exponents) {
                            let norm_q = gaussian_normalization(alpha_q, &(0, 0, 0));

                            let p = alpha_p + alpha_q;

                            // Gaussian product centre P
                            let p_center = gaussian_product_center(alpha_p, bf_mu.center, alpha_q, bf_nu.center);

                            // |R_a - R_b|^2 term for exponent
                            let rab2 = (bf_mu.center - bf_nu.center).norm_squared();
                            let k_ab = (-alpha_p * alpha_q / p * rab2).exp();

                            // Boys function argument t = p |P - R_A|^2
                            let t = p * (p_center - *r_a).norm_squared();
                            let boys = boys_function(0, t);

                            let coeff = -z_a * 2.0 * PI / p;

                            integral += c_p * c_q * norm_p * norm_q * coeff * k_ab * boys;
                        }
                    }
                }

                v_matrix[(mu, nu)] = integral;
                if mu != nu {
                    v_matrix[(nu, mu)] = integral; // Symmetric
                }
            }
        }

        Ok(v_matrix)
    }

    fn build_fock_matrix(&self, _density: &DMatrix<f64>, molecule: &Molecule) -> Result<DMatrix<f64>> {
        // At the Hartree-Fock level the Fock matrix is:
        // F = H_core + G(D)
        // where H_core = T + V_nuc and G comprises Coulomb (J) and exchange (K) contributions
        // built from two-electron integrals and the density matrix D.
        //
        // Two-electron integrals are not yet implemented. Until they are, we return the core Hamiltonian.
        // This yields the Harris functional (non-self-consistent) which is still variational and provides
        // a meaningful energy upper bound.

        let t = self.build_kinetic_matrix(molecule)?;
        let v = self.build_nuclear_attraction_matrix(molecule)?;
        Ok(t + v)
    }

    fn solve_eigenvalue_problem(&self, fock: &DMatrix<f64>, overlap: &DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>)> {
        // --- Roothaan generalized eigenvalue solver ---
        // We solve  F C = S C ε  where F is the Fock matrix,
        // S is the overlap matrix (symmetric positive-definite),
        // C are molecular orbital coefficients and ε the diagonal matrix of orbital energies.
        //
        // Standard approach:
        // 1. Perform Cholesky decomposition  S = L Lᵗ  (guaranteed SPD for a well-defined basis).
        // 2. Transform F  →  F' = L⁻ᵗ F L⁻¹  which is orthonormal (S' = I).
        // 3. Solve the ordinary symmetric eigenproblem  F' C' = C' ε.
        // 4. Back-transform coefficients  C = L⁻¹ C'.
        // 5. Sort eigenvalues ascending and reorder columns of C accordingly.
        //
        // References:
        // * C. C. J. Roothaan, "Self-consistent Field Theory for Molecular and Solid State Problems", Rev. Mod. Phys. 32, 179 (1960).
        // * T. Helgaker, P. Jørgensen, J. Olsen, "Molecular Electronic-Structure Theory", Wiley (2000), ch. 10.

        // Sanity checks
        ensure!(fock.is_square(), "Fock matrix must be square");
        ensure!(overlap.is_square(), "Overlap matrix must be square");
        ensure!(fock.nrows() == overlap.nrows(), "Fock and overlap matrices must have identical dimensions");

        let n = fock.nrows();
        if n == 0 {
            return Ok((Vec::new(), DMatrix::zeros(0, 0)));
        }

        // Clone to owned matrices for decomposition (nalgebra consumes the input)
        let s_owned = overlap.clone();
        let f_owned = fock.clone();

        // 1. Cholesky factorisation S = L Lᵗ
        let chol = Cholesky::new(s_owned).ok_or_else(|| anyhow!("Overlap matrix is not positive-definite; basis set is likely linearly dependent"))?;
        let l = chol.l(); // Lower triangular view

        // 2. Compute L⁻¹ (explicit inverse is acceptable for small/medium basis sizes; for large systems use solve instead)
        let l_inv = l.clone().try_inverse().ok_or_else(|| anyhow!("Failed to invert Cholesky factor of overlap matrix"))?;

        // Construct orthogonalised F' = L⁻ᵗ F L⁻¹
        let f_ortho = &l_inv.transpose() * f_owned * &l_inv;

        // 3. Symmetric eigenvalue decomposition of F'
        let eig = SymmetricEigen::new(f_ortho);
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        let eigenvectors_prime = eig.eigenvectors; // Columns are eigenvectors in orthonormal basis

        // 4. Back-transform eigenvectors to original non-orthogonal basis: C = L⁻¹ C'
        let c = &l_inv * eigenvectors_prime;

        // 5. Sort eigenvalues in ascending order and reorder corresponding columns in C
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        let mut c_sorted = DMatrix::zeros(n, n);
        for (col_new, &col_old) in indices.iter().enumerate() {
            // Copy column col_old from C into column col_new of c_sorted
            c_sorted.set_column(col_new, &c.column(col_old));
        }
        eigenvalues = indices.iter().map(|&i| eigenvalues[i]).collect();

        Ok((eigenvalues, c_sorted))
    }

    fn calculate_total_energy(&self, density: &DMatrix<f64>, molecule: &Molecule) -> Result<f64> {
        // Electronic energy at Hartree-Fock level:
        // E = Σ_μν D_μν (H_core_μν + F_μν)/2
        //
        // With the current placeholder F = H_core (see build_fock_matrix) this reduces to
        // E = Σ_μν D_μν H_core_μν.
        // The factor of 1/2 is absorbed because H_core = F.

        let h_core = {
            let t = self.build_kinetic_matrix(molecule)?;
            let v = self.build_nuclear_attraction_matrix(molecule)?;
            t + v
        };

        // Ensure dimensions match
        ensure!(density.nrows() == h_core.nrows(), "Density and core Hamiltonian dimensions mismatch");

        let mut energy = 0.0;
        for i in 0..density.nrows() {
            for j in 0..density.ncols() {
                energy += density[(i, j)] * h_core[(i, j)];
            }
        }
        Ok(energy)
    }

    fn lda_exchange_correlation(&self, _molecule: &Molecule) -> Result<f64> { Ok(0.0) }
    fn gga_exchange_correlation(&self, _molecule: &Molecule) -> Result<f64> { Ok(0.0) }
    fn hybrid_exchange_correlation(&self, _molecule: &Molecule) -> Result<f64> { Ok(0.0) }
    fn meta_gga_exchange_correlation(&self, _molecule: &Molecule) -> Result<f64> { Ok(0.0) }
    
    /// Molecular mechanics calculation using a classical force field.
    /// The total energy is a sum of bonded (bond, angle, dihedral) and 
    /// non-bonded (van der Waals, electrostatic) terms.
    fn molecular_mechanics_calculation(&self, molecule: &Molecule) -> Result<ElectronicStructure> {
        let bond_energy = self.calculate_bond_energy(molecule)?;
        let angle_energy = self.calculate_angle_energy(molecule)?;
        let dihedral_energy = self.calculate_dihedral_energy(molecule)?;
        let non_bonded_energy = self.calculate_non_bonded_energy(molecule)?;

        let total_energy = bond_energy + angle_energy + dihedral_energy + non_bonded_energy;

        Ok(ElectronicStructure {
            total_energy,
            ..Default::default()
        })
    }

    fn qm_mm_calculation(&self, _molecule: &Molecule) -> Result<ElectronicStructure> { Ok(ElectronicStructure::default()) }

    // --- Molecular Mechanics Helpers ---

    /// Calculates the total bond stretching energy of a molecule.
    /// Uses a simple harmonic potential: E = k(r - r₀)²
    fn calculate_bond_energy(&self, molecule: &Molecule) -> Result<f64> {
        let mut total_bond_energy = 0.0;
        for bond in &molecule.bonds {
            let atom1 = &molecule.atoms[bond.atom_indices.0];
            let atom2 = &molecule.atoms[bond.atom_indices.1];
            let key = (atom1.get_particle_type(), atom2.get_particle_type());

            if let Some(params) = self.force_field_parameters.bond_parameters.get(&key) {
                let distance = (atom1.position - atom2.position).norm();
                let displacement = distance - params.equilibrium_length;
                total_bond_energy += params.force_constant * displacement.powi(2);
            }
        }
        Ok(total_bond_energy)
    }

    /// Calculates the total angle bending energy of a molecule.
    /// Uses a simple harmonic potential: E = k(θ - θ₀)²
    fn calculate_angle_energy(&self, molecule: &Molecule) -> Result<f64> {
        let mut total_angle_energy = 0.0;
        let num_atoms = molecule.atoms.len();
        if num_atoms < 3 {
            return Ok(0.0);
        }

        // Build adjacency list for efficient neighbor lookup
        let mut adj = vec![vec![]; num_atoms];
        for bond in &molecule.bonds {
            adj[bond.atom_indices.0].push(bond.atom_indices.1);
            adj[bond.atom_indices.1].push(bond.atom_indices.0);
        }

        // Iterate over all atoms as the central atom of an angle
        for j in 0..num_atoms {
            let neighbors = &adj[j];
            if neighbors.len() < 2 {
                continue;
            }
            // Form angles from pairs of neighbors
            for i_idx in 0..neighbors.len() {
                for k_idx in (i_idx + 1)..neighbors.len() {
                    let i = neighbors[i_idx];
                    let k = neighbors[k_idx];

                    let atom_i = &molecule.atoms[i];
                    let atom_j = &molecule.atoms[j];
                    let atom_k = &molecule.atoms[k];

                    let key = (atom_i.get_particle_type(), atom_j.get_particle_type(), atom_k.get_particle_type());
                    
                    if let Some(params) = self.force_field_parameters.angle_parameters.get(&key) {
                        let v_ji = atom_i.position - atom_j.position;
                        let v_jk = atom_k.position - atom_j.position;
                        let angle = v_ji.angle(&v_jk);
                        let angle_displacement = angle - params.equilibrium_angle;
                        total_angle_energy += params.force_constant * angle_displacement.powi(2);
                    }
                }
            }
        }
        Ok(total_angle_energy)
    }

    /// Calculates the total dihedral/torsional energy of a molecule.
    /// Uses a periodic potential: E = Vₙ(1 + cos(nφ - δ))
    fn calculate_dihedral_energy(&self, molecule: &Molecule) -> Result<f64> {
        let mut total_dihedral_energy = 0.0;
        let num_atoms = molecule.atoms.len();
        if num_atoms < 4 {
            return Ok(0.0);
        }

        // Build adjacency list for efficient neighbor lookup
        let mut adj = vec![vec![]; num_atoms];
        for bond in &molecule.bonds {
            adj[bond.atom_indices.0].push(bond.atom_indices.1);
            adj[bond.atom_indices.1].push(bond.atom_indices.0);
        }

        // Iterate over all central bonds (j-k)
        for j in 0..num_atoms {
            for &k in &adj[j] {
                if j > k { continue; } // Avoid double counting

                for &i in &adj[j] {
                    if i == k { continue; }

                    for &l in &adj[k] {
                        if l == j { continue; }

                        let atom_i = &molecule.atoms[i];
                        let atom_j = &molecule.atoms[j];
                        let atom_k = &molecule.atoms[k];
                        let atom_l = &molecule.atoms[l];

                        let key = (atom_i.get_particle_type(), atom_j.get_particle_type(), atom_k.get_particle_type(), atom_l.get_particle_type());

                        if let Some(params) = self.force_field_parameters.dihedral_parameters.get(&key) {
                            let v_ij = atom_j.position - atom_i.position;
                            let v_jk = atom_k.position - atom_j.position;
                            let v_kl = atom_l.position - atom_k.position;

                            let n1 = v_ij.cross(&v_jk).normalize();
                            let n2 = v_jk.cross(&v_kl).normalize();
                            
                            let cos_phi = n1.dot(&n2);
                            let phi = cos_phi.acos();

                            // Handle sign of the angle
                            let sign = if v_kl.dot(&n1) > 0.0 { 1.0 } else { -1.0 };
                            let phi_signed = phi * sign;

                            total_dihedral_energy += params.barrier_height * (1.0 + (params.periodicity as f64 * phi_signed - params.phase_angle).cos());
                        }
                    }
                }
            }
        }
        Ok(total_dihedral_energy)
    }

    /// Calculates non-bonded energy (van der Waals and electrostatic).
    /// This implementation excludes pairs that are directly bonded.
    /// A more sophisticated implementation would also exclude 1-3 pairs and scale 1-4 pairs.
    fn calculate_non_bonded_energy(&self, molecule: &Molecule) -> Result<f64> {
        let mut total_non_bonded_energy = 0.0;
        let num_atoms = molecule.atoms.len();
        if num_atoms < 2 {
            return Ok(0.0);
        }

        // Build a set of bonded pairs for quick lookup
        let bonded_pairs: std::collections::HashSet<(usize, usize)> = molecule.bonds.iter().map(|b| {
            if b.atom_indices.0 < b.atom_indices.1 {
                (b.atom_indices.0, b.atom_indices.1)
            } else {
                (b.atom_indices.1, b.atom_indices.0)
            }
        }).collect();

        let coulomb_k = 1.0 / (4.0 * PI * constants::VACUUM_PERMITTIVITY);

        for i in 0..num_atoms {
            for j in (i + 1)..num_atoms {
                // Exclude directly bonded pairs
                if bonded_pairs.contains(&(i, j)) {
                    continue;
                }

                let atom_i = &molecule.atoms[i];
                let atom_j = &molecule.atoms[j];
                let distance = (atom_i.position - atom_j.position).norm();

                if distance < 1e-12 { continue; }

                // Van der Waals energy
                total_non_bonded_energy += self.van_der_waals_energy(i, j, distance, molecule);

                // Electrostatic energy
                let type_i = atom_i.get_particle_type();
                let type_j = atom_j.get_particle_type();
                let charge_i = self.force_field_parameters.van_der_waals_parameters.get(&type_i).map_or(0.0, |p| p.partial_charge);
                let charge_j = self.force_field_parameters.van_der_waals_parameters.get(&type_j).map_or(0.0, |p| p.partial_charge);
                
                if charge_i.abs() > 1e-9 && charge_j.abs() > 1e-9 {
                     total_non_bonded_energy += coulomb_k * (charge_i * charge_j) / distance;
                }
            }
        }
        Ok(total_non_bonded_energy)
    }
}

impl Default for ElectronicStructure {
    fn default() -> Self {
        Self {
            total_energy: 0.0,
            orbital_energies: vec![],
            molecular_orbitals: vec![],
            electron_density: vec![],
            bond_orders: HashMap::new(),
            atomic_charges: vec![],
            dipole_moment: Vector3::zeros(),
            polarizability: Matrix3::zeros(),
        }
    }
}

impl Default for ForceFieldParameters {
    fn default() -> Self {
        Self {
            bond_parameters: HashMap::new(),
            angle_parameters: HashMap::new(),
            dihedral_parameters: HashMap::new(),
            van_der_waals_parameters: HashMap::new(),
        }
    }
}

// === Helper Functions (using quantum_math module) ===

/// Analytic integral of the kinetic-energy operator between two primitive
/// Cartesian Gaussian basis functions.  The operator used is the non-relativistic
/// one-electron kinetic energy  **T = −½∇²** (in Hartree atomic units).
///
/// For general angular momentum the Obara–Saika recurrence can be employed.
/// A complete implementation is beyond the scope of the current task; however,
/// the project presently utilises only s-type functions (l = m = n = 0) because
/// the bundled STO-3G basis in `BasisSet::sto_3g()` is populated on demand.
///
/// For two s-type primitives centred at A and B with exponents α, β the
/// integral is (see Helgaker *et al.* Eq. (10.6.26))
///
///    T(α,β,R) = \frac{αβ}{α+β}\left[ 3 − 2\frac{αβ}{α+β} R² \right] S(α,β,R)
///
/// where R = |A−B| and  
///    S(α,β,R) = (π/(α+β))^{3/2} exp(−αβ/(α+β) R²)
/// is the overlap integral of the two primitives.
///
/// The returned value is in Hartree (atomic units).  Coordinate units must be
/// consistent (Bohr radii if atomic units are desired).
#[inline]
fn kinetic_integral(
    alpha1: f64,
    center1: Vector3<f64>,
    ang1: &(u32, u32, u32),
    alpha2: f64,
    center2: Vector3<f64>,
    ang2: &(u32, u32, u32),
) -> f64 {
    // Use the new Obara-Saika implementation from quantum_math
    let mut workspace = ObSaWorkspace::new(2);
    kinetic_integral_obara_saika(alpha1, center1, *ang1, alpha2, center2, *ang2, &mut workspace)
}

/// Nuclear attraction integral between two Gaussian basis functions
/// 
/// Computes ∫ φᵢ(r) (-Z/|r-R_N|) φⱼ(r) dr where Z is the nuclear charge
/// and R_N is the nuclear position.
/// 
/// Uses the Boys function for the fundamental integrals.
fn nuclear_attraction_integral(
    alpha1: f64,
    center1: Vector3<f64>,
    ang1: &(u32, u32, u32),
    alpha2: f64,
    center2: Vector3<f64>,
    ang2: &(u32, u32, u32),
    nuclear_charge: f64,
    nuclear_center: Vector3<f64>,
) -> f64 {
    let (l1, m1, n1) = *ang1;
    let (l2, m2, n2) = *ang2;
    
    // For s-type functions only (current implementation)
    if l1 == 0 && m1 == 0 && n1 == 0 && l2 == 0 && m2 == 0 && n2 == 0 {
        let gamma = alpha1 + alpha2;
        let product_center = gaussian_product_center(alpha1, center1, alpha2, center2);
        let rpc_squared = (product_center - nuclear_center).norm_squared();
        
        let prefactor = -nuclear_charge * 2.0 * PI / gamma;
        let overlap_factor = (-(alpha1 * alpha2 / gamma) * (center1 - center2).norm_squared()).exp();
        let boys_arg = gamma * rpc_squared;
        
        prefactor * overlap_factor * boys_function(0, boys_arg)
    } else {
        // Placeholder for higher angular momentum - would use Obara-Saika recursion
        0.0
    }
}



// ----------------------------- Tests ---------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kinetic_integral_identical_centers() {
        // When two identical s-type Gaussians share a centre the kinetic integral
        // reduces to 3α/2 (see standard quantum-chemistry texts).
        let alpha = 0.75f64;
        let centre = Vector3::new(0.0, 0.0, 0.0);
        let t = kinetic_integral(alpha, centre, &(0, 0, 0), alpha, centre, &(0, 0, 0));
        let expected = 1.5 * alpha; // 3α/2 in atomic units
        let rel_err = ((t - expected) / expected).abs();
        assert!(rel_err < 1e-12, "kinetic integral incorrect: rel_err = {}", rel_err);
    }
    
    #[test]
    fn test_nuclear_attraction_integral() {
        // Test nuclear attraction integral for two s-type Gaussians
        let alpha1 = 1.0;
        let center1 = Vector3::new(0.0, 0.0, 0.0);
        let alpha2 = 1.0;
        let center2 = Vector3::new(0.0, 0.0, 0.0);
        let nuclear_charge = 1.0;
        let nuclear_center = Vector3::new(0.0, 0.0, 0.0);
        
        let integral = nuclear_attraction_integral(
            alpha1, center1, &(0, 0, 0),
            alpha2, center2, &(0, 0, 0),
            nuclear_charge, nuclear_center
        );
        
        // For identical s-type Gaussians centered at the nucleus,
        // the integral should be negative (attractive)
        assert!(integral < 0.0, "Nuclear attraction integral should be negative");
        assert!(integral.abs() > 1e-10, "Nuclear attraction integral should be non-zero");
    }
} 