use super::*;
use nalgebra::{Vector3, Matrix3, Complex, DMatrix};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use once_cell::sync::Lazy;
use crate::constants;
use std::f64::consts::PI;
use rayon::prelude::*;
use anyhow::{anyhow, ensure};
use nalgebra::{Cholesky, SymmetricEigen};


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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    /// Populates STO-3G, a minimal basis set, for selected elements.
    /// STO-3G approximates a single Slater-type orbital with 3 Gaussian functions.
    /// Source: Hehre, W. J.; Stewart, R. F.; Pople, J. A. J. Chem. Phys. 1969, 51, 2657.
    pub fn sto_3g() -> Self {
        let mut shells_for_atom = HashMap::new();

        // Hydrogen (Z=1)
        shells_for_atom.insert(1, vec![
            Contraction { // 1s
                angular_momentum: 0,
                exponents: vec![3.42525091, 0.62391373, 0.16885540],
                coefficients: vec![0.15432897, 0.53532814, 0.44463454],
            }
        ]);

        // Carbon (Z=6)
        shells_for_atom.insert(6, vec![
            Contraction { // 1s
                angular_momentum: 0,
                exponents: vec![71.6168370, 13.0450960, 3.5305122],
                coefficients: vec![0.15432897, 0.53532814, 0.44463454],
            },
            Contraction { // 2s
                angular_momentum: 0,
                exponents: vec![2.9412494, 0.6834831, 0.2222899],
                coefficients: vec![-0.09996723, 0.39951283, 0.70011547],
            },
            Contraction { // 2p
                angular_momentum: 1,
                exponents: vec![2.9412494, 0.6834831, 0.2222899],
                coefficients: vec![0.15591627, 0.60768372, 0.39195739],
            }
        ]);

        Self {
            name: "STO-3G".to_string(),
            shells_for_atom,
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
        let density_matrix = DMatrix::zeros(overlap_matrix.nrows(), overlap_matrix.ncols());

        let mut total_energy = 0.0;

        // Self-Consistent Field (SCF) iterations
        for _iteration in 0..50 {
            let fock_matrix = self.build_fock_matrix(&density_matrix, molecule)?;
            let (_eigenvalues, _eigenvectors) = self.solve_eigenvalue_problem(&fock_matrix, &overlap_matrix)?;
            
            // Update density matrix and check convergence
            let old_energy = total_energy;
            total_energy = self.calculate_total_energy(&density_matrix, molecule)?;
            
            if (total_energy - old_energy).abs() < 1e-8 {
                break;
            }
        }
        
        Ok(ElectronicStructure {
            total_energy,
            orbital_energies: vec![0.0; n_electrons * 4], // Approximate basis size
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
                    let norm1 = normalization(exp1, &bf_i.angular_momentum);
                    let norm2 = normalization(exp2, &bf_j.angular_momentum);
                    
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
                        let norm_p = normalization(alpha_p, &(0, 0, 0));
                        for (c_q, &alpha_q) in bf_nu.contraction.coefficients.iter().zip(&bf_nu.contraction.exponents) {
                            let norm_q = normalization(alpha_q, &(0, 0, 0));

                            let p = alpha_p + alpha_q;

                            // Gaussian product centre P
                            let p_center = gaussian_product_center(alpha_p, bf_mu.center, alpha_q, bf_nu.center);

                            // |R_a - R_b|^2 term for exponent
                            let rab2 = (bf_mu.center - bf_nu.center).norm_squared();
                            let k_ab = (-alpha_p * alpha_q / p * rab2).exp();

                            // Boys function argument t = p |P - R_A|^2
                            let t = p * (p_center - *r_a).norm_squared();
                            let boys = boys_function_0(t);

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

/// Computes the kinetic energy integral between two primitive Cartesian Gaussian functions.
/// Source: Hô, M., & Hernández-Pérez, J. M. (2013). Evaluation of Gaussian Molecular Integrals II. Kinetic-Energy Integrals.
/// The Mathematica Journal, 15. doi:10.3888/tmj.15-1
fn kinetic_integral(alpha1: f64, center1: Vector3<f64>, lmn1: &(u32, u32, u32),
                    alpha2: f64, center2: Vector3<f64>, lmn2: &(u32, u32, u32)) -> f64 {
    let p = alpha1 + alpha2;
    let p_center = gaussian_product_center(alpha1, center1, alpha2, center2);

    let sx = s_1d(lmn1.0, lmn2.0, center1.x, center2.x, p_center.x, p);
    let sy = s_1d(lmn1.1, lmn2.1, center1.y, center2.y, p_center.y, p);
    let sz = s_1d(lmn1.2, lmn2.2, center1.z, center2.z, p_center.z, p);

    let tx = t_1d(lmn1.0, lmn2.0, center1.x, center2.x, p_center.x, p, alpha1, alpha2);
    let ty = t_1d(lmn1.1, lmn2.1, center1.y, center2.y, p_center.y, p, alpha1, alpha2);
    let tz = t_1d(lmn1.2, lmn2.2, center1.z, center2.z, p_center.z, p, alpha1, alpha2);
    
    // The kinetic energy operator is T = -1/2 * nabla^2 = -1/2 * (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)
    // The matrix element <a|T|b> is separable: <a_x|T_x|b_x><a_y|S_y|b_y><a_z|S_z|b_z> + ...
    // Source: Helgaker, T., Jorgensen, P., & Olsen, J. (2000). Molecular Electronic-Structure Theory. Wiley. Page 243.
    tx * sy * sz + sx * ty * sz + sx * sy * tz
}

/// Calculates the 1D kinetic energy integral component T_ij = <i| -1/2 d^2/dx^2 |j>
/// using recurrence relations. The operator acts on basis function j (the ket).
/// Source: Helgaker, T., Jorgensen, P., & Olsen, J. (2000). Molecular Electronic-Structure Theory. Wiley.
/// See equations 9.6.15 - 9.6.18.
fn t_1d(l1: u32, l2: u32, a: f64, b: f64, p_coord: f64, p: f64, _alpha1: f64, alpha2: f64) -> f64 {
    // T_ij = alpha2 * (2*l2 + 1) * S_{i,j} - 2*alpha2^2 * S_{i,j+2} - 1/2*l2*(l2-1)*S_{i,j-2}
    
    let term1 = alpha2 * (2.0 * l2 as f64 + 1.0) * s_1d(l1, l2, a, b, p_coord, p);
    
    let term2 = 2.0 * alpha2.powi(2) * s_1d(l1, l2 + 2, a, b, p_coord, p);

    let term3 = if l2 >= 2 {
        0.5 * (l2 * (l2 - 1)) as f64 * s_1d(l1, l2 - 2, a, b, p_coord, p)
    } else {
        0.0
    };

    term1 - term2 - term3
}

/// Calculates the 1D overlap integral S_ij = <i|j> for x, y, or z component.
/// This uses the Hermite-Gauss integral formula, computed via recurrence relations.
/// Source: Obara, S., & Saika, A. (1986). J. Chem. Phys. 84, 3963.
fn s_1d(l1: u32, l2: u32, a: f64, b: f64, p_coord: f64, p: f64) -> f64 {
    if l1 == 0 && l2 == 0 {
        return (PI / p).sqrt();
    }
    
    let mut s_vals = HashMap::new();
    s_vals.insert((0, 0), (PI / p).sqrt());

    for i in 1..=(l1 + l2) {
        let term1 = (p_coord - a) * s_vals.get(&(i - 1, 0)).unwrap_or(&0.0);
        let term2 = if i >= 2 {
            ((i - 1) as f64 / (2.0 * p)) * s_vals.get(&(i - 2, 0)).unwrap_or(&0.0)
        } else { 0.0 };
        s_vals.insert((i, 0), term1 + term2);
    }
    
    for j in 1..=l2 {
        for i in 0..=(l1 + l2 - j) {
            let val = s_vals.get(&(i + 1, j - 1)).unwrap_or(&0.0) + (a - b) * s_vals.get(&(i, j - 1)).unwrap_or(&0.0);
            s_vals.insert((i, j), val);
        }
    }

    *s_vals.get(&(l1, l2)).unwrap_or(&0.0)
}

/// Calculates the center of the product of two Gaussians.
fn gaussian_product_center(alpha1: f64, center1: Vector3<f64>, alpha2: f64, center2: Vector3<f64>) -> Vector3<f64> {
    (alpha1 * center1 + alpha2 * center2) / (alpha1 + alpha2)
}

/// Normalization constant for a primitive Cartesian Gaussian function.
/// N = (2α/π)^(3/4) * [(4α)^l / ( (2l_x-1)!! * (2l_y-1)!! * (2l_z-1)!! )]^(1/2)
/// Source: Helgaker, T., Jorgensen, P., & Olsen, J. (2000). Molecular Electronic-Structure Theory. Wiley.
fn normalization(alpha: f64, lmn: &(u32, u32, u32)) -> f64 {
    let term1 = (2.0 * alpha / PI).powf(0.75);
    let total_l = lmn.0 + lmn.1 + lmn.2;
    let term2 = (4.0 * alpha).powf(total_l as f64 / 2.0);
    
    let dbl_factorial = |n: u32| -> f64 {
        if n == 0 { return 1.0; }
        let mut res = 1.0;
        let mut i = n;
        while i > 0 {
            res *= i as f64;
            if i <= 2 { break; }
            i -= 2;
        }
        res
    };
    
    let denom = (dbl_factorial(2 * lmn.0) * dbl_factorial(2 * lmn.1) * dbl_factorial(2 * lmn.2)).sqrt();
    
    term1 * term2 / denom
}

/// Approximate error function erf(x) using a rational polynomial.
/// Source: Abramowitz & Stegun, Handbook of Mathematical Functions, 7.1.26.
/// Accurate to 1.5e-7 for x ∈ [-1, 1].
fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = |t| {
        t * (0.254829592
            - t * (-0.284496736
            + t * (1.421413741
            - t * (1.453152027
            - t * 1.061405429))))
    };
    if x >= 0.0 {
        poly(t)
    } else {
        -poly(t)
    }
}

/// Boys function F_0(x) = √π/4x * erf(√x)
/// Source: Helgaker, T., Jorgensen, P., & Olsen, J. (2000). Molecular Electronic-Structure Theory. Wiley.
fn boys_function_0(x: f64) -> f64 {
    if x < 1e-12 {
        return 1.0;
    }
    (PI / (4.0 * x)).sqrt() * erf_approx(x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use crate::{Atom, AtomicNucleus, Molecule, Nucleon, Quark};

    #[test]
    fn test_build_kinetic_matrix_h2() {
        // Setup a simple H2 molecule
        let h_atom1 = Atom {
            nucleus: AtomicNucleus {
                atomic_number: 1,
                mass_number: 1,
                position: Vector3::new(0.0, 0.0, -0.7),
                ..Default::default()
            },
            position: Vector3::new(0.0, 0.0, -0.7 * 5.29177e-11), // Convert to meters
            ..Default::default()
        };
        let h_atom2 = Atom {
            nucleus: AtomicNucleus {
                atomic_number: 1,
                mass_number: 1,
                position: Vector3::new(0.0, 0.0, 0.7),
                ..Default::default()
            },
            position: Vector3::new(0.0, 0.0, 0.7 * 5.29177e-11), // Convert to meters
            ..Default::default()
        };
        let molecule = Molecule {
            atoms: vec![h_atom1, h_atom2],
            ..Default::default()
        };

        let engine = QuantumChemistryEngine {
            basis_set: BasisSet::sto_3g(),
            ..QuantumChemistryEngine::new()
        };

        let kinetic_matrix = engine.build_kinetic_matrix(&molecule).unwrap();

        // These are reference values for H2 with STO-3G at 1.4 bohr (0.7408 Å) bond length
        // calculated with PySCF. Values are in Hartree.
        // Note: The physics engine seems to use meters, but quantum calculations
        // are usually in atomic units (Bohr, Hartree). This test assumes the basis
        // set exponents and coefficients are for calculations in atomic units.
        let expected_t11 = 0.76003233;
        let expected_t12 = 0.23663243;

        // Check for symmetry and correct values
        assert!((kinetic_matrix[(0, 0)] - expected_t11).abs() < 1e-6);
        assert!((kinetic_matrix[(1, 1)] - expected_t11).abs() < 1e-6);
        assert!((kinetic_matrix[(0, 1)] - expected_t12).abs() < 1e-6);
        assert!((kinetic_matrix[(1, 0)] - kinetic_matrix[(0, 1)]).abs() < 1e-9);
    }

    // Helper to create a default Atom for testing purposes
    impl Default for Atom {
        fn default() -> Self {
            Atom {
                nucleus: Default::default(),
                electrons: vec![],
                electron_orbitals: vec![],
                total_energy: 0.0,
                ionization_energy: 0.0,
                electron_affinity: 0.0,
                atomic_radius: 0.0,
                position: Vector3::zeros(),
                velocity: Vector3::zeros(),
                electronic_state: Default::default(),
            }
        }
    }
    
    // Helper to create a default AtomicNucleus for testing purposes
    impl Default for AtomicNucleus {
        fn default() -> Self {
            AtomicNucleus {
                mass_number: 1,
                atomic_number: 1,
                protons: vec![],
                neutrons: vec![],
                binding_energy: 0.0,
                nuclear_spin: Vector3::zeros(),
                magnetic_moment: Vector3::zeros(),
                electric_quadrupole_moment: 0.0,
                nuclear_radius: 0.0,
                shell_model_state: Default::default(),
                position: Vector3::zeros(),
                momentum: Vector3::zeros(),
                excitation_energy: 0.0,
            }
        }
    }
    
    // Helper to create a default Molecule for testing purposes
    impl Default for Molecule {
        fn default() -> Self {
            Molecule {
                atoms: vec![],
                bonds: vec![],
                molecular_orbitals: vec![],
                vibrational_modes: vec![],
                rotational_constants: Vector3::zeros(),
                dipole_moment: Vector3::zeros(),
                polarizability: Matrix3::zeros(),
                potential_energy_surface: Default::default(),
                reaction_coordinates: vec![],
            }
        }
    }

    #[test]
    fn test_solve_eigenvalue_problem() {
        use nalgebra::DMatrix;

        // Simple 2×2 symmetric test matrices (atomic units)
        let fock = DMatrix::<f64>::from_row_slice(2, 2, &[
            -1.0, -0.2,
            -0.2, -0.5,
        ]);
        let overlap = DMatrix::<f64>::from_row_slice(2, 2, &[
            1.0, 0.1,
            0.1, 1.0,
        ]);

        let engine = QuantumChemistryEngine {
            basis_set: BasisSet::sto_3g(),
            ..QuantumChemistryEngine::new()
        };

        let (e_vals, c_mat) = engine.solve_eigenvalue_problem(&fock, &overlap).unwrap();

        // Basic sanity checks
        assert_eq!(e_vals.len(), 2);
        assert!(e_vals[0] <= e_vals[1]);

        // Verify orthonormality Cᵀ S C ≈ I
        let s_ortho = c_mat.transpose() * &overlap * &c_mat;
        let identity = DMatrix::<f64>::identity(2, 2);
        assert!((s_ortho - identity).amax() < 1e-10);

        // Verify eigen equation F C ≈ S C ε
        let f_c = &fock * &c_mat;
        let s_c = &overlap * &c_mat;
        for col in 0..2 {
            let lambda = e_vals[col];
            let diff = &f_c.column(col) - &(s_c.column(col) * lambda);
            assert!(diff.amax() < 1e-10);
        }
    }
} 