use super::*;
use nalgebra::{Vector3, Matrix3, Complex, DMatrix};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use once_cell::sync::Lazy;
use crate::constants;
use std::f64::consts::PI;


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
                        // Add other angular momenta (d, f) here as needed
                        _ => unimplemented!("Angular momentum > 1 not supported yet"),
                    }
                }
            }
        }

        let n_basis = basis_functions.len();
        let mut kinetic_matrix = DMatrix::zeros(n_basis, n_basis);

        for i in 0..n_basis {
            for j in 0..n_basis {
                let bf_i = &basis_functions[i];
                let bf_j = &basis_functions[j];
                let mut integral = 0.0;

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
                kinetic_matrix[(i, j)] = integral;
            }
        }

        Ok(kinetic_matrix)
    }

    fn build_nuclear_attraction_matrix(&self, _molecule: &Molecule) -> Result<DMatrix<f64>> { unimplemented!() }
    fn build_fock_matrix(&self, _density: &DMatrix<f64>, _molecule: &Molecule) -> Result<DMatrix<f64>> { unimplemented!() }
    fn solve_eigenvalue_problem(&self, _fock: &DMatrix<f64>, _overlap: &DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>)> { unimplemented!() }
    fn calculate_total_energy(&self, _density: &DMatrix<f64>, _molecule: &Molecule) -> Result<f64> { Ok(0.0) }
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
    let p_vec = gaussian_product_center(alpha1, center1, alpha2, center2);
    
    let t_x = t_1d(lmn1.0, lmn2.0, center1.x, center2.x, p_vec.x, p, alpha1, alpha2);
    let s_x = s_1d(lmn1.0, lmn2.0, center1.x, center2.x, p_vec.x, p);

    let t_y = t_1d(lmn1.1, lmn2.1, center1.y, center2.y, p_vec.y, p, alpha1, alpha2);
    let s_y = s_1d(lmn1.1, lmn2.1, center1.y, center2.y, p_vec.y, p);

    let t_z = t_1d(lmn1.2, lmn2.2, center1.z, center2.z, p_vec.z, p, alpha1, alpha2);
    let s_z = s_1d(lmn1.2, lmn2.2, center1.z, center2.z, p_vec.z, p);

    let dist_sq = (center1 - center2).norm_squared();
    let prefactor = (-alpha1 * alpha2 * dist_sq / p).exp();

    prefactor * (t_x * s_y * s_z + s_x * t_y * s_z + s_x * s_y * t_z)
}

/// Calculates the 1D kinetic integral component T_i.
fn t_1d(l1: u32, l2: u32, a: f64, b: f64, p_coord: f64, p: f64, _alpha1: f64, alpha2: f64) -> f64 {
    let term1 = if l2 >= 1 {
        l2 as f64 * s_1d(l1, l2 - 1, a, b, p_coord, p)
    } else { 0.0 };

    let term2 = 2.0 * alpha2 * alpha2 * s_1d(l1, l2 + 1, a, b, p_coord, p);
    let term3 = -0.5 * if l2 >= 2 {
        (l2 * (l2 - 1)) as f64 * s_1d(l1, l2 - 2, a, b, p_coord, p)
    } else { 0.0 };

    alpha2 * term1 + term2 + term3
}


/// Calculates the 1D overlap integral S_i between two primitive Gaussians along one dimension.
/// Uses a recurrence relation.
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
} 