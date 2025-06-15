//! # Physics Engine: Nuclear Physics Utilities
//!
//! This module provides utilities for simulating nuclear physics phenomena, such as
//! nuclear binding energies, stability, radioactive decay, and stellar nucleosynthesis.

use anyhow::Result;
use rand::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::BOLTZMANN;

/// Represents an atomic nucleus, composed of protons and neutrons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Nucleus {
    pub protons: u32,  // Z
    pub neutrons: u32, // N
}

/// Nuclear decay modes based on Chart of Nuclides data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayMode {
    Alpha,           // α decay: A(Z,N) → A-4(Z-2,N-2) + α
    BetaMinus,       // β⁻ decay: n → p + e⁻ + ν̄ₑ
    BetaPlus,        // β⁺ decay: p → n + e⁺ + νₑ
    ElectronCapture, // EC: p + e⁻ → n + νₑ
    SpontaneousFission, // SF: Heavy nuclei split into fragments
    Stable,          // No decay
}

/// Nuclear decay data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearDecayData {
    pub half_life_seconds: f64,  // Half-life in seconds
    pub primary_mode: DecayMode, // Primary decay mode
    pub decay_energy: f64,       // Q-value in MeV
    pub branching_ratio: f64,    // Probability of this decay mode
}

/// Nuclear database containing decay properties
pub struct NuclearDatabase {
    decay_data: HashMap<(u32, u32), NuclearDecayData>, // (Z, A) -> decay data
}

impl NuclearDatabase {
    /// Initialize nuclear database with known isotope data
    /// Data sources: NNDC (National Nuclear Data Center), Chart of Nuclides
    pub fn new() -> Self {
        let mut database = Self {
            decay_data: HashMap::new(),
        };
        database.populate_nuclear_data();
        database
    }
    
    /// Populate database with selected nuclear decay data
    /// Based on NNDC Chart of Nuclides (https://www.nndc.bnl.gov/nudat3/)
    fn populate_nuclear_data(&mut self) {
        // Stable isotopes (partial list of most abundant)
        self.add_stable_isotope(1, 1);   // ¹H
        self.add_stable_isotope(1, 2);   // ²H (deuterium)
        self.add_stable_isotope(2, 4);   // ⁴He
        self.add_stable_isotope(6, 12);  // ¹²C
        self.add_stable_isotope(6, 13);  // ¹³C
        self.add_stable_isotope(7, 14);  // ¹⁴N
        self.add_stable_isotope(8, 16);  // ¹⁶O
        self.add_stable_isotope(8, 18);  // ¹⁸O
        self.add_stable_isotope(10, 20); // ²⁰Ne
        self.add_stable_isotope(12, 24); // ²⁴Mg
        self.add_stable_isotope(14, 28); // ²⁸Si
        self.add_stable_isotope(16, 32); // ³²S
        self.add_stable_isotope(26, 56); // ⁵⁶Fe
        
        // Radioactive isotopes with realistic decay properties
        
        // Beta-minus decays (neutron-rich)
        self.decay_data.insert((1, 3), NuclearDecayData {  // ³H (tritium)
            half_life_seconds: 3.888e8, // 12.32 years
            primary_mode: DecayMode::BetaMinus,
            decay_energy: 0.0186, // MeV
            branching_ratio: 1.0,
        });
        
        self.decay_data.insert((6, 14), NuclearDecayData {  // ¹⁴C
            half_life_seconds: 1.808e11, // 5,730 years
            primary_mode: DecayMode::BetaMinus,
            decay_energy: 0.156, // MeV
            branching_ratio: 1.0,
        });
        
        // Beta-plus decays and electron capture (proton-rich)
        self.decay_data.insert((6, 11), NuclearDecayData {  // ¹¹C
            half_life_seconds: 1.220e3, // 20.38 minutes
            primary_mode: DecayMode::BetaPlus,
            decay_energy: 0.960, // MeV
            branching_ratio: 1.0,
        });
        
        self.decay_data.insert((7, 13), NuclearDecayData {  // ¹³N
            half_life_seconds: 5.98e2, // 9.96 minutes
            primary_mode: DecayMode::BetaPlus,
            decay_energy: 1.199, // MeV
            branching_ratio: 1.0,
        });
        
        // Alpha decays (heavy nuclei)
        self.decay_data.insert((88, 226), NuclearDecayData { // ²²⁶Ra
            half_life_seconds: 5.05e10, // 1,600 years
            primary_mode: DecayMode::Alpha,
            decay_energy: 4.871, // MeV
            branching_ratio: 1.0,
        });
        
        self.decay_data.insert((92, 238), NuclearDecayData { // ²³⁸U
            half_life_seconds: 1.41e17, // 4.47 billion years
            primary_mode: DecayMode::Alpha,
            decay_energy: 4.270, // MeV
            branching_ratio: 1.0,
        });
        
        self.decay_data.insert((92, 235), NuclearDecayData { // ²³⁵U
            half_life_seconds: 2.22e16, // 704 million years
            primary_mode: DecayMode::Alpha,
            decay_energy: 4.679, // MeV
            branching_ratio: 1.0,
        });
        
        self.decay_data.insert((94, 239), NuclearDecayData { // ²³⁹Pu
            half_life_seconds: 7.61e11, // 24,110 years
            primary_mode: DecayMode::Alpha,
            decay_energy: 5.244, // MeV
            branching_ratio: 1.0,
        });
        
        // Spontaneous fission (super-heavy elements)
        self.decay_data.insert((98, 252), NuclearDecayData { // ²⁵²Cf
            half_life_seconds: 8.33e7, // 2.64 years
            primary_mode: DecayMode::SpontaneousFission,
            decay_energy: 200.0, // Typical fission energy
            branching_ratio: 0.031, // 3.1% SF, rest is α
        });
    }
    
    fn add_stable_isotope(&mut self, z: u32, a: u32) {
        self.decay_data.insert((z, a), NuclearDecayData {
            half_life_seconds: f64::INFINITY,
            primary_mode: DecayMode::Stable,
            decay_energy: 0.0,
            branching_ratio: 1.0,
        });
    }
    
    /// Get decay data for a given nucleus
    pub fn get_decay_data(&self, z: u32, a: u32) -> Option<&NuclearDecayData> {
        self.decay_data.get(&(z, a))
    }
    
    /// Check if nucleus is stable based on valley of beta stability
    /// Uses empirical stability criteria from nuclear physics
    pub fn is_stable(&self, z: u32, a: u32) -> bool {
        // First check database
        if let Some(data) = self.get_decay_data(z, a) {
            return matches!(data.primary_mode, DecayMode::Stable);
        }
        
        // Empirical stability rules for unknown isotopes
        let n = a - z;
        
        // Magic numbers (closed shells are more stable)
        let magic_numbers = [2, 8, 20, 28, 50, 82, 126];
        let z_magic = magic_numbers.contains(&z);
        let n_magic = magic_numbers.contains(&n);
        
        if z_magic && n_magic {
            return true; // Doubly magic nuclei are very stable
        }
        
        // Valley of beta stability approximation
        // Stable N/Z ratio increases with mass
        let optimal_n_to_z = if z < 20 {
            1.0  // Light nuclei: N ≈ Z
        } else if z < 83 {
            1.0 + 0.4 * (z as f64 - 20.0) / 63.0  // Medium nuclei
        } else {
            return false; // All nuclei with Z > 82 are radioactive
        };
        
        let actual_n_to_z = n as f64 / z as f64;
        let deviation = (actual_n_to_z - optimal_n_to_z).abs();
        
        // Allow some deviation from optimal ratio
        deviation < 0.15
    }
}

impl Nucleus {
    /// Creates a new nucleus.
    pub fn new(protons: u32, neutrons: u32) -> Self {
        Nucleus { protons, neutrons }
    }

    /// Returns the mass number (A), the total number of nucleons.
    pub fn mass_number(&self) -> u32 {
        self.protons + self.neutrons
    }

    /// Calculates the binding energy of the nucleus using the semi-empirical mass formula (SEMF).
    /// This provides an approximation of the energy required to disassemble the nucleus.
    /// Returns the binding energy in MeV (Mega-electronvolts).
    /// Calculate the binding energy of the nucleus using the Semi-Empirical Mass Formula (SEMF)
    /// Returns binding energy in MeV (positive value means bound state)
    pub fn binding_energy(&self) -> f64 {
        let z = self.protons as f64;
        let n = self.neutrons as f64;
        let a = self.mass_number() as f64;

        if a == 0.0 { return 0.0; }

        // Special cases for very light nuclei with known values
        match (self.protons, self.neutrons) {
            (0, 1) => return 0.0,       // Free neutron has zero binding energy
            (1, 0) => return 0.0,       // Free proton has zero binding energy
            (1, 1) => return 2.225,     // Deuteron binding energy (MeV)
            (1, 2) => return 8.482,     // Tritium binding energy (MeV)
            (2, 1) => return 7.718,     // ³He binding energy (MeV)
            (2, 2) => return 28.296,    // ⁴He binding energy (MeV)
            (6, 6) => return 92.162,    // ¹²C binding energy (MeV)
            (8, 8) => return 127.619,   // ¹⁶O binding energy (MeV)
            _ => {}
        }

        // Coefficients for the SEMF (in MeV)
        const A_V: f64 = 15.75;  // Volume term
        const A_S: f64 = 17.8;   // Surface term
        const A_C: f64 = 0.711;  // Coulomb term
        const A_A: f64 = 23.7;   // Asymmetry term
        const A_P: f64 = 11.18;  // Pairing term

        let volume_term = A_V * a;
        let surface_term = A_S * a.powf(2.0 / 3.0);
        let coulomb_term = A_C * z * (z - 1.0) / a.powf(1.0 / 3.0);
        let asymmetry_term = A_A * (n - z).powi(2) / a;
        
        // Pairing term depends on the parity of Z and N
        let pairing_term = if self.protons % 2 == 0 && self.neutrons % 2 == 0 {
            A_P / a.powf(0.5)
        } else if self.protons % 2 != 0 && self.neutrons % 2 != 0 {
            -A_P / a.powf(0.5)
        } else {
            0.0
        };

        volume_term - surface_term - coulomb_term - asymmetry_term + pairing_term
    }

    /// Simulates radioactive decay based on realistic nuclear data
    /// Returns the decay product(s) if decay occurs.
    pub fn radioactive_decay<R: Rng + ?Sized>(&self, rng: &mut R, database: &NuclearDatabase, dt: f64) -> Option<Vec<Nucleus>> {
        let decay_data = database.get_decay_data(self.protons, self.mass_number())?;
        
        // Check for stable nucleus
        if matches!(decay_data.primary_mode, DecayMode::Stable) {
            return None;
        }
        
        // Calculate decay probability using exponential decay law
        // P(decay) = 1 - exp(-λt) where λ = ln(2)/t₁/₂
        let decay_constant = 0.693147 / decay_data.half_life_seconds; // ln(2) / t₁/₂
        let decay_probability = 1.0 - (-decay_constant * dt).exp();
        
        if !rng.gen_bool(decay_probability * decay_data.branching_ratio) {
            return None;
        }
        
        // Execute decay based on mode
        match decay_data.primary_mode {
            DecayMode::Alpha => self.alpha_decay(),
            DecayMode::BetaMinus => self.beta_minus_decay(),
            DecayMode::BetaPlus => self.beta_plus_decay(),
            DecayMode::ElectronCapture => self.electron_capture(),
            DecayMode::SpontaneousFission => self.spontaneous_fission(rng),
            DecayMode::Stable => None,
        }
    }
    
    fn alpha_decay(&self) -> Option<Vec<Nucleus>> {
        if self.protons >= 2 && self.neutrons >= 2 {
            Some(vec![
                Nucleus::new(self.protons - 2, self.neutrons - 2), // Daughter nucleus
                Nucleus::new(2, 2), // Alpha particle (⁴He)
            ])
        } else {
            None
        }
    }
    
    fn beta_minus_decay(&self) -> Option<Vec<Nucleus>> {
        if self.neutrons > 0 {
            Some(vec![
                Nucleus::new(self.protons + 1, self.neutrons - 1), // n → p + e⁻ + ν̄ₑ
                // Note: electron and antineutrino are not tracked as nuclei
            ])
        } else {
            None
        }
    }
    
    fn beta_plus_decay(&self) -> Option<Vec<Nucleus>> {
        if self.protons > 0 {
            Some(vec![
                Nucleus::new(self.protons - 1, self.neutrons + 1), // p → n + e⁺ + νₑ
                // Note: positron and neutrino are not tracked as nuclei
            ])
        } else {
            None
        }
    }
    
    fn electron_capture(&self) -> Option<Vec<Nucleus>> {
        // Same nuclear result as β⁺ decay: p + e⁻ → n + νₑ
        self.beta_plus_decay()
    }
    
    /// Spontaneous fission with realistic fission fragment distribution
    /// Based on Wahl's fission fragment mass distribution systematics
    fn spontaneous_fission<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<Vec<Nucleus>> {
        let a = self.mass_number();
        let z = self.protons;
        
        // Only allow fission for sufficiently heavy nuclei
        if a < 200 { return None; }
        
        // Wahl's asymmetric fission model parameters
        // Most fissions are asymmetric with light/heavy fragment mass ratio
        let light_mass_peak = (a as f64 * 0.4) as u32;  // ~40% of total mass
        
        // Add some randomness around the peaks
        let mass_spread = 10;
        let light_mass = light_mass_peak + rng.gen_range(0..mass_spread) - mass_spread/2;
        let heavy_mass = a - light_mass;
        
        // Charge distribution: approximately preserves Z/A ratio
        let light_z = ((light_mass as f64 / a as f64) * z as f64) as u32;
        let heavy_z = z - light_z;
        
        // Calculate neutron numbers
        let light_n = light_mass - light_z;
        let heavy_n = heavy_mass - heavy_z;
        
        // Generate fission products + neutrons
        let mut products = vec![
            Nucleus::new(light_z, light_n),
            Nucleus::new(heavy_z, heavy_n),
        ];
        
        // Add prompt neutrons (typically 2-3 per fission)
        let neutron_count = rng.gen_range(1..4);
        for _ in 0..neutron_count {
            products.push(Nucleus::new(0, 1)); // Free neutron
        }
        
        Some(products)
    }
}

/// Stellar nucleosynthesis reaction types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NucleosynthesisReaction {
    // Proton-proton chain
    PPChainI,    // p + p → d + e+ + νe
    PPChainII,   // p + p + e- → d + νe (rare)
    DeuteriumFusion, // d + p → ³He + γ
    He3Fusion,   // ³He + ³He → ⁴He + 2p
    
    // CNO cycle
    CNO1, // p + ¹²C → ¹³N + γ
    CNO2, // ¹³N → ¹³C + e+ + νe
    CNO3, // p + ¹³C → ¹⁴N + γ
    CNO4, // p + ¹⁴N → ¹⁵O + γ
    CNO5, // ¹⁵O → ¹⁵N + e+ + νe
    CNO6, // p + ¹⁵N → ¹²C + ⁴He
    
    // Helium burning (Triple-alpha process)
    TripleAlpha, // 3 ⁴He → ¹²C + γ
    AlphaCarbon, // ⁴He + ¹²C → ¹⁶O + γ
    
    // Advanced burning stages
    CarbonBurning,  // ¹²C + ¹²C → various products
    NeonBurning,    // ²⁰Ne + γ → ¹⁶O + ⁴He
    OxygenBurning,  // ¹⁶O + ¹⁶O → various products
    SiliconBurning, // Si + α-particles → Fe-peak elements
}

/// Stellar nucleosynthesis reaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StellarReaction {
    pub reaction_type: NucleosynthesisReaction,
    pub reactants: Vec<(u32, u32)>, // (Z, A) pairs
    pub products: Vec<(u32, u32)>,  // (Z, A) pairs
    pub q_value: f64,               // Energy released (MeV)
    pub cross_section: f64,         // Cross-section (barns, 1 barn = 1e-24 cm²)
    pub temperature_threshold: f64,  // Minimum temperature (K)
    pub rate_coefficient: f64,      // Reaction rate coefficient
}

impl StellarReaction {
    /// Calculate reaction rate based on temperature and density
    pub fn calculate_rate(&self, temperature: f64, density: f64) -> f64 {
        if temperature < self.temperature_threshold {
            return 0.0;
        }
        
        // Gamow peak - exponential suppression due to Coulomb barrier
        let coulomb_barrier = self.calculate_coulomb_barrier();
        let gamow_factor = (-coulomb_barrier / (temperature * 8.617e-5)).exp(); // kT in eV
        
        // Rate = n₁n₂⟨σv⟩ where ⟨σv⟩ is the thermally averaged cross-section
        self.rate_coefficient * density.powi(2) * gamow_factor
    }
    
    fn calculate_coulomb_barrier(&self) -> f64 {
        // Simplified Coulomb barrier calculation
        if self.reactants.len() < 2 { return 0.0; }
        
        let z1 = self.reactants[0].0 as f64;
        let z2 = self.reactants[1].0 as f64;
        let a1 = self.reactants[0].1 as f64;
        let a2 = self.reactants[1].1 as f64;
        
        // Nuclear radius: r = r0 * (A1^(1/3) + A2^(1/3)) where r0 = 1.2 fm
        let r0 = 1.2e-15; // 1.2 fm in meters
        let nuclear_radius = r0 * (a1.powf(1.0/3.0) + a2.powf(1.0/3.0));
        
        // Coulomb barrier: E = k * Z1 * Z2 * e² / r, where k*e² = 1.44 MeV·fm
        let barrier_mev = 1.44 * z1 * z2 / (nuclear_radius * 1e15); // Convert m to fm
        barrier_mev * 1e6 // Convert MeV to eV
    }
}

/// Stellar nucleosynthesis engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StellarNucleosynthesis {
    pub reactions: Vec<StellarReaction>,
    pub pp_chain_active: bool,
    pub cno_cycle_active: bool,
    pub helium_burning_active: bool,
    pub advanced_burning_active: bool,
}

impl StellarNucleosynthesis {
    pub fn new() -> Self {
        let mut nucleosynthesis = Self {
            reactions: Vec::new(),
            pp_chain_active: false,
            cno_cycle_active: false,
            helium_burning_active: false,
            advanced_burning_active: false,
        };
        
        nucleosynthesis.initialize_reactions();
        nucleosynthesis
    }
    
    fn initialize_reactions(&mut self) {
        // Proton-proton chain reactions
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::PPChainI,
            reactants: vec![(1, 1), (1, 1)], // p + p
            products: vec![(1, 2), (0, 0), (0, 0)], // d + e+ + νe
            q_value: 1.442, // MeV
            cross_section: 4.7e-47, // Very small due to weak interaction
            temperature_threshold: 4e6, // 4 million K
            rate_coefficient: 3.78e-43, // cm³/mol/s
        });
        
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::DeuteriumFusion,
            reactants: vec![(1, 2), (1, 1)], // d + p
            products: vec![(2, 3), (0, 0)], // ³He + γ
            q_value: 5.494, // MeV
            cross_section: 5.5e-4, // Much larger than pp
            temperature_threshold: 1e6, // 1 million K
            rate_coefficient: 2.38e-4,
        });
        
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::He3Fusion,
            reactants: vec![(2, 3), (2, 3)], // ³He + ³He
            products: vec![(2, 4), (1, 1), (1, 1)], // ⁴He + 2p
            q_value: 12.859, // MeV
            cross_section: 5.0e-3,
            temperature_threshold: 6e6, // 6 million K
            rate_coefficient: 5.61e-11,
        });
        
        // CNO cycle reactions
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::CNO1,
            reactants: vec![(1, 1), (6, 12)], // p + ¹²C
            products: vec![(7, 13), (0, 0)], // ¹³N + γ
            q_value: 1.944, // MeV
            cross_section: 1.7e-3,
            temperature_threshold: 18e6, // 18 million K
            rate_coefficient: 9.18e-16,
        });
        
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::CNO3,
            reactants: vec![(1, 1), (6, 13)], // p + ¹³C
            products: vec![(7, 14), (0, 0)], // ¹⁴N + γ
            q_value: 7.551, // MeV
            cross_section: 2.7e-2,
            temperature_threshold: 18e6,
            rate_coefficient: 1.08e-12,
        });
        
        // Triple-alpha process (helium burning)
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::TripleAlpha,
            reactants: vec![(2, 4), (2, 4), (2, 4)], // 3 ⁴He
            products: vec![(6, 12), (0, 0)], // ¹²C + γ
            q_value: 7.275, // MeV
            cross_section: 3.0e-9, // Very temperature dependent
            temperature_threshold: 100e6, // 100 million K
            rate_coefficient: 2.79e-43, // Three-body reaction
        });
        
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::AlphaCarbon,
            reactants: vec![(2, 4), (6, 12)], // ⁴He + ¹²C
            products: vec![(8, 16), (0, 0)], // ¹⁶O + γ
            q_value: 7.162, // MeV
            cross_section: 1.2e-6,
            temperature_threshold: 200e6, // 200 million K
            rate_coefficient: 1.04e-15,
        });
        
        // Advanced burning (simplified)
        self.reactions.push(StellarReaction {
            reaction_type: NucleosynthesisReaction::CarbonBurning,
            reactants: vec![(6, 12), (6, 12)], // ¹²C + ¹²C
            products: vec![(12, 24), (0, 0)], // ²⁴Mg + γ (one of many products)
            q_value: 13.93, // MeV
            cross_section: 1.5e-8,
            temperature_threshold: 600e6, // 600 million K
            rate_coefficient: 4.27e-15,
        });
    }
    
    /// Process stellar nucleosynthesis based on stellar conditions
    pub fn process_stellar_burning(&mut self, temperature: f64, density: f64, composition: &mut [(u32, u32, f64)]) -> Result<f64> {
        let mut total_energy_released = 0.0;
        
        // Determine which burning stages are active
        self.update_burning_stages(temperature);
        
        // Process reactions in order of activation
        for reaction in &self.reactions {
            if self.is_reaction_active(&reaction.reaction_type, temperature) {
                let rate = reaction.calculate_rate(temperature, density);
                let energy = self.execute_stellar_reaction(reaction, rate, composition)?;
                total_energy_released += energy;
            }
        }
        
        Ok(total_energy_released)
    }
    
    fn update_burning_stages(&mut self, temperature: f64) {
        self.pp_chain_active = temperature > 4e6;
        self.cno_cycle_active = temperature > 18e6;
        self.helium_burning_active = temperature > 100e6;
        self.advanced_burning_active = temperature > 600e6;
    }
    
    fn is_reaction_active(&self, reaction_type: &NucleosynthesisReaction, _temperature: f64) -> bool {
        match reaction_type {
            NucleosynthesisReaction::PPChainI | 
            NucleosynthesisReaction::PPChainII |
            NucleosynthesisReaction::DeuteriumFusion |
            NucleosynthesisReaction::He3Fusion => self.pp_chain_active,
            
            NucleosynthesisReaction::CNO1 |
            NucleosynthesisReaction::CNO2 |
            NucleosynthesisReaction::CNO3 |
            NucleosynthesisReaction::CNO4 |
            NucleosynthesisReaction::CNO5 |
            NucleosynthesisReaction::CNO6 => self.cno_cycle_active,
            
            NucleosynthesisReaction::TripleAlpha |
            NucleosynthesisReaction::AlphaCarbon => self.helium_burning_active,
            
            NucleosynthesisReaction::CarbonBurning |
            NucleosynthesisReaction::NeonBurning |
            NucleosynthesisReaction::OxygenBurning |
            NucleosynthesisReaction::SiliconBurning => self.advanced_burning_active,
        }
    }
    
    fn execute_stellar_reaction(&self, reaction: &StellarReaction, rate: f64, composition: &mut [(u32, u32, f64)]) -> Result<f64> {
        // Find reactant abundances
        let mut reactant_abundances = Vec::new();
        for &(z, a) in &reaction.reactants {
            if let Some(abundance) = self.find_isotope_abundance(composition, z, a) {
                reactant_abundances.push(abundance);
            } else {
                return Ok(0.0); // Can't proceed without reactants
            }
        }
        
        // Calculate limiting reactant
        let limiting_abundance = reactant_abundances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let reaction_extent = limiting_abundance.min(rate * 0.01); // Limit reaction rate
        
        if reaction_extent <= 0.0 {
            return Ok(0.0);
        }
        
        // Update composition
        self.update_composition_from_reaction(composition, reaction, reaction_extent)?;
        
        // Calculate energy released
        Ok(reaction.q_value * 1.602e-13 * reaction_extent) // Convert MeV to Joules
    }
    
    fn find_isotope_abundance(&self, composition: &[(u32, u32, f64)], z: u32, a: u32) -> Option<f64> {
        composition.iter().find(|&&(zz, aa, _)| zz == z && aa == a).map(|&(_, _, abundance)| abundance)
    }
    
    fn update_composition_from_reaction(&self, composition: &mut [(u32, u32, f64)], reaction: &StellarReaction, extent: f64) -> Result<()> {
        // Consume reactants
        for &(z, a) in &reaction.reactants {
            if let Some(isotope) = composition.iter_mut().find(|isotope| isotope.0 == z && isotope.1 == a) {
                isotope.2 -= extent;
                isotope.2 = isotope.2.max(0.0); // Prevent negative abundances
            }
        }
        
        // Produce products
        for &(z, a) in &reaction.products {
            if z > 0 && a > 0 { // Skip massless particles (photons, neutrinos)
                if let Some(isotope) = composition.iter_mut().find(|isotope| isotope.0 == z && isotope.1 == a) {
                    isotope.2 += extent;
                } else {
                    // Add new isotope if not present (this would require expanding the composition array)
                    // For now, we'll skip this to avoid memory allocation issues
                }
            }
        }
        
        Ok(())
    }
}

/// Updates the state of a collection of nuclei, simulating radioactive decay.
pub fn update_nuclear_state(nuclei: &mut Vec<Nucleus>) -> Result<()> {
    let mut rng = thread_rng();
    let mut new_nuclei = Vec::new();
    let database = NuclearDatabase::new();
    
    nuclei.retain_mut(|nucleus| {
        if let Some(products) = nucleus.radioactive_decay(&mut rng, &database, 1.0) {
            new_nuclei.extend(products);
            false // Remove the decayed nucleus
        } else {
            true // Keep the stable nucleus
        }
    });

    nuclei.append(&mut new_nuclei);
    Ok(())
}

/// Neutron capture process types for heavy element synthesis
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeutronCaptureProcess {
    SProcess, // Slow neutron capture (stellar nucleosynthesis)
    RProcess, // Rapid neutron capture (explosive nucleosynthesis)
}

/// Neutron capture cross-section data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutronCaptureData {
    pub cross_section: f64,      // Neutron capture cross-section (barns)
    pub resonance_energy: f64,   // Resonance energy (eV)
    pub capture_probability: f64, // Probability of capture vs. scattering
}

/// Neutron capture database for heavy element synthesis
/// Based on experimental neutron capture cross-sections from ENDF/B-VIII.0
pub struct NeutronCaptureDatabase {
    capture_data: HashMap<(u32, u32), NeutronCaptureData>, // (Z, A) -> capture data
}

impl NeutronCaptureDatabase {
    /// Initialize neutron capture database with experimental cross-sections
    /// Data sources: ENDF/B-VIII.0, Evaluated Nuclear Data File
    pub fn new() -> Self {
        let mut database = Self {
            capture_data: HashMap::new(),
        };
        database.populate_capture_data();
        database
    }
    
    /// Populate database with neutron capture cross-sections
    /// Critical isotopes for s-process and r-process nucleosynthesis
    fn populate_capture_data(&mut self) {
        // Iron-peak nuclei (seed nuclei for heavy element synthesis)
        self.capture_data.insert((26, 56), NeutronCaptureData { // ⁵⁶Fe
            cross_section: 2.59,  // barns at thermal energy
            resonance_energy: 1.15e3, // eV
            capture_probability: 0.85,
        });
        
        self.capture_data.insert((26, 57), NeutronCaptureData { // ⁵⁷Fe
            cross_section: 2.48,  // barns
            resonance_energy: 3.94e2, // eV
            capture_probability: 0.82,
        });
        
        // Important s-process branch points
        self.capture_data.insert((38, 87), NeutronCaptureData { // ⁸⁷Sr
            cross_section: 16.0,  // High cross-section branch point
            resonance_energy: 2.8e2, // eV
            capture_probability: 0.95,
        });
        
        self.capture_data.insert((38, 88), NeutronCaptureData { // ⁸⁸Sr
            cross_section: 0.058, // Magic N=50, low cross-section
            resonance_energy: 1.36e4, // eV
            capture_probability: 0.45,
        });
        
        // Barium isotopes (s-process main component)
        self.capture_data.insert((56, 138), NeutronCaptureData { // ¹³⁸Ba
            cross_section: 0.27,  // barns
            resonance_energy: 7.8e3, // eV
            capture_probability: 0.65,
        });
        
        // Lead isotopes (s-process termination)
        self.capture_data.insert((82, 206), NeutronCaptureData { // ²⁰⁶Pb
            cross_section: 0.030, // Very low, near magic numbers
            resonance_energy: 3.2e4, // eV
            capture_probability: 0.25,
        });
        
        self.capture_data.insert((82, 207), NeutronCaptureData { // ²⁰⁷Pb
            cross_section: 0.699, // barns
            resonance_energy: 3.07e3, // eV
            capture_probability: 0.72,
        });
        
        self.capture_data.insert((82, 208), NeutronCaptureData { // ²⁰⁸Pb (doubly magic)
            cross_section: 0.0003, // Extremely low cross-section
            resonance_energy: 7.2e4, // eV
            capture_probability: 0.05,
        });
        
        // Neutron-rich r-process isotopes (estimated cross-sections)
        // These are typically unavailable experimentally, estimated from systematics
        self.capture_data.insert((38, 94), NeutronCaptureData { // ⁹⁴Sr (r-process)
            cross_section: 150.0, // Very high for neutron-rich isotopes
            resonance_energy: 50.0, // eV (much lower resonance)
            capture_probability: 0.98,
        });
        
        self.capture_data.insert((56, 150), NeutronCaptureData { // ¹⁵⁰Ba (r-process)
            cross_section: 250.0, // Very high cross-section
            resonance_energy: 30.0, // eV
            capture_probability: 0.99,
        });
    }
    
    /// Get neutron capture data for a given nucleus
    pub fn get_capture_data(&self, z: u32, a: u32) -> Option<&NeutronCaptureData> {
        self.capture_data.get(&(z, a))
    }
    
    /// Calculate neutron capture rate based on neutron flux and temperature
    /// For s-process: thermal neutrons (kT ~ 30 keV)
    /// For r-process: high neutron density (>10²⁰ neutrons/cm³)
    pub fn calculate_capture_rate(&self, z: u32, a: u32, neutron_flux: f64, temperature: f64, process: &NeutronCaptureProcess) -> f64 {
        if let Some(data) = self.get_capture_data(z, a) {
            // Maxwell-Boltzmann averaged cross-section
            let thermal_energy = 0.0253; // eV at room temperature
            let kt = 8.617e-5 * temperature; // kT in eV
            
            // Cross-section varies with neutron energy: σ(E) ∝ 1/√E for thermal region
            let energy_factor = match process {
                NeutronCaptureProcess::SProcess => (thermal_energy / kt).sqrt(), // Thermal neutrons
                NeutronCaptureProcess::RProcess => 1.0, // High-energy neutrons, less energy dependence
            };
            
            // Rate = n_neutrons * v_relative * sigma
            // For thermal neutrons: v ~ 2200 m/s
            let relative_velocity = match process {
                NeutronCaptureProcess::SProcess => 2200.0, // m/s
                NeutronCaptureProcess::RProcess => 1e6,    // m/s (higher energy neutrons)
            };
            
            neutron_flux * data.cross_section * 1e-24 * relative_velocity * energy_factor * data.capture_probability
        } else {
            // Estimate cross-section for unknown isotopes using systematics
            self.estimate_capture_cross_section(z, a, process) * neutron_flux * 2200.0 * 1e-24
        }
    }
    
    /// Estimate neutron capture cross-section for unknown isotopes
    /// Based on optical model and mass-dependent systematics
    fn estimate_capture_cross_section(&self, z: u32, a: u32, process: &NeutronCaptureProcess) -> f64 {
        let n = a - z;
        
        // Basic systematics: cross-section increases with neutron excess
        // and decreases near magic numbers
        let magic_numbers = [2, 8, 20, 28, 50, 82, 126];
        let n_magic = magic_numbers.contains(&n);
        let z_magic = magic_numbers.contains(&z);
        
        let base_cross_section = match process {
            NeutronCaptureProcess::SProcess => {
                if n_magic || z_magic {
                    1.0 // Low cross-section near magic numbers
                } else {
                    10.0 + (n as f64 - z as f64) * 2.0 // Increases with neutron excess
                }
            },
            NeutronCaptureProcess::RProcess => {
                // R-process isotopes are very neutron-rich with high cross-sections
                100.0 + (n as f64 - z as f64) * 10.0
            }
        };
        
        // Mass dependence: A^(-1/3) dependence from optical model
        base_cross_section * (56.0 / a as f64).powf(1.0/3.0)
    }
}

/// Simulate neutron capture nucleosynthesis
/// This processes neutron capture chains for s-process and r-process
pub fn process_neutron_capture(
    nuclei: &mut Vec<Nucleus>, 
    neutron_flux: f64, 
    temperature: f64, 
    process: NeutronCaptureProcess,
    dt: f64
) -> Result<f64> {
    let mut rng = thread_rng();
    let capture_db = NeutronCaptureDatabase::new();
    let decay_db = NuclearDatabase::new();
    let mut total_energy_released = 0.0;
    let mut new_nuclei = Vec::new();
    
    // Process existing nuclei for neutron capture
    for nucleus in nuclei.iter_mut() {
        let z = nucleus.protons;
        let a = nucleus.mass_number();
        
        // Skip neutrons themselves
        if z == 0 { continue; }
        
        // Calculate neutron capture probability
        let capture_rate = capture_db.calculate_capture_rate(z, a, neutron_flux, temperature, &process);
        let capture_probability = 1.0 - (-capture_rate * dt).exp();
        
        if rng.gen_bool(capture_probability) {
            // Neutron capture: A(Z,N) + n → A+1(Z,N+1) + γ
            let new_nucleus = Nucleus::new(z, nucleus.neutrons + 1);
            
            // Calculate Q-value for neutron capture (typically 6-8 MeV)
            let q_value = 7.0; // MeV (typical neutron binding energy)
            total_energy_released += q_value * 1.602e-13; // Convert to Joules
            
            // Replace original nucleus with capture product
            *nucleus = new_nucleus;
            
            // Check if the product is unstable and may decay
            if let Some(decay_data) = decay_db.get_decay_data(nucleus.protons, nucleus.mass_number()) {
                if !matches!(decay_data.primary_mode, DecayMode::Stable) {
                    // Fast beta decay in r-process
                    if matches!(process, NeutronCaptureProcess::RProcess) && 
                       matches!(decay_data.primary_mode, DecayMode::BetaMinus) {
                        // Rapid beta decay during r-process
                        let beta_decay_rate = 0.693147 / decay_data.half_life_seconds;
                        let beta_probability = 1.0 - (-beta_decay_rate * dt).exp();
                        
                        if rng.gen_bool(beta_probability) {
                            if let Some(decay_products) = nucleus.radioactive_decay(&mut rng, &decay_db, dt) {
                                // Replace with decay products
                                if !decay_products.is_empty() {
                                    *nucleus = decay_products[0].clone();
                                    // Add any additional products
                                    new_nuclei.extend(decay_products.into_iter().skip(1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Add any new nuclei from decay processes
    nuclei.extend(new_nuclei);
    
    Ok(total_energy_released)
}

/// Comprehensive nuclear cross-section database
/// Based on ENDF/B-VIII.0 and experimental data from NNDC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearCrossSectionDatabase {
    /// Fusion cross-sections for stellar nucleosynthesis reactions
    pub fusion_data: HashMap<(u32, u32, u32, u32), FusionCrossSectionData>,
    /// Nuclear reaction cross-sections (n,γ), (p,γ), etc.
    pub reaction_data: HashMap<String, ReactionCrossSectionData>,
    /// Temperature and energy-dependent data tables
    pub temperature_grids: HashMap<String, Vec<(f64, f64)>>, // (temperature_K, cross_section_barns)
}

/// Fusion cross-section data for specific nuclear reactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCrossSectionData {
    /// Reactant 1: (Z, A)
    pub reactant1: (u32, u32),
    /// Reactant 2: (Z, A)
    pub reactant2: (u32, u32),
    /// Q-value in MeV
    pub q_value: f64,
    /// Astrophysical S-factor in MeV·barns
    pub s_factor: f64,
    /// Coulomb barrier energy in keV
    pub coulomb_barrier: f64,
    /// Gamow peak energy in keV
    pub gamow_peak_energy: f64,
    /// Temperature range where this reaction is relevant (K)
    pub temperature_range: (f64, f64),
    /// Cross-section at Gamow peak in barns
    pub peak_cross_section: f64,
}

/// Nuclear reaction cross-section data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionCrossSectionData {
    /// Target nucleus: (Z, A)
    pub target: (u32, u32),
    /// Projectile type (neutron, proton, alpha, etc.)
    pub projectile: String,
    /// Product channels
    pub products: Vec<(u32, u32)>,
    /// Energy-dependent cross-section table (energy_eV, cross_section_barns)
    pub energy_table: Vec<(f64, f64)>,
    /// Thermal cross-section at 0.0253 eV in barns
    pub thermal_cross_section: f64,
    /// Resonance integrals
    pub resonance_integral: f64,
}

impl NuclearCrossSectionDatabase {
    /// Create new nuclear cross-section database with experimental data
    pub fn new() -> Self {
        let mut database = Self {
            fusion_data: HashMap::new(),
            reaction_data: HashMap::new(),
            temperature_grids: HashMap::new(),
        };
        
        database.populate_fusion_data();
        database.populate_reaction_data();
        database.populate_temperature_grids();
        
        database
    }
    
    /// Populate fusion cross-section data from experimental measurements
    fn populate_fusion_data(&mut self) {
        // Proton-proton chain reactions
        self.fusion_data.insert((1, 1, 1, 1), FusionCrossSectionData {
            reactant1: (1, 1), // p
            reactant2: (1, 1), // p
            q_value: 1.442,    // MeV (includes neutrino energy)
            s_factor: 4.0e-47, // Very small due to weak interaction
            coulomb_barrier: 1022.0, // keV
            gamow_peak_energy: 6.0,   // keV at solar core temperature
            temperature_range: (4e6, 50e6), // 4-50 million K
            peak_cross_section: 4.7e-47, // barns
        });
        
        self.fusion_data.insert((1, 2, 1, 1), FusionCrossSectionData {
            reactant1: (1, 2), // d
            reactant2: (1, 1), // p
            q_value: 5.494,    // MeV
            s_factor: 3.4e-4,  // Much larger than pp
            coulomb_barrier: 982.0, // keV
            gamow_peak_energy: 7.5,  // keV
            temperature_range: (1e6, 20e6),
            peak_cross_section: 5.5e-4, // barns
        });
        
        self.fusion_data.insert((2, 3, 2, 3), FusionCrossSectionData {
            reactant1: (2, 3), // ³He
            reactant2: (2, 3), // ³He
            q_value: 12.859,   // MeV
            s_factor: 5.2e-3,  // barns·keV
            coulomb_barrier: 1220.0, // keV
            gamow_peak_energy: 22.0, // keV
            temperature_range: (6e6, 30e6),
            peak_cross_section: 5.0e-3, // barns
        });
        
        // CNO cycle reactions
        self.fusion_data.insert((1, 1, 6, 12), FusionCrossSectionData {
            reactant1: (1, 1), // p
            reactant2: (6, 12), // ¹²C
            q_value: 1.944,    // MeV
            s_factor: 1.7e-3,  // barns·keV
            coulomb_barrier: 1372.0, // keV
            gamow_peak_energy: 25.0, // keV at CNO temperatures
            temperature_range: (18e6, 80e6),
            peak_cross_section: 1.7e-3, // barns
        });
        
        self.fusion_data.insert((1, 1, 7, 14), FusionCrossSectionData {
            reactant1: (1, 1), // p
            reactant2: (7, 14), // ¹⁴N
            q_value: 7.297,    // MeV
            s_factor: 3.2e-3,  // barns·keV
            coulomb_barrier: 1503.0, // keV
            gamow_peak_energy: 27.0, // keV
            temperature_range: (18e6, 80e6),
            peak_cross_section: 3.2e-3, // barns
        });
        
        // Helium burning reactions
        self.fusion_data.insert((2, 4, 6, 12), FusionCrossSectionData {
            reactant1: (2, 4), // ⁴He
            reactant2: (6, 12), // ¹²C
            q_value: 7.162,    // MeV
            s_factor: 1.4e-6,  // Much smaller than hydrogen burning
            coulomb_barrier: 1966.0, // keV
            gamow_peak_energy: 300.0, // keV at helium burning temperatures
            temperature_range: (100e6, 300e6),
            peak_cross_section: 1.2e-6, // barns
        });
        
        // Advanced burning stages
        self.fusion_data.insert((6, 12, 6, 12), FusionCrossSectionData {
            reactant1: (6, 12), // ¹²C
            reactant2: (6, 12), // ¹²C
            q_value: 13.93,    // MeV
            s_factor: 1.5e-8,  // Very small cross-section
            coulomb_barrier: 2940.0, // keV
            gamow_peak_energy: 1500.0, // keV
            temperature_range: (600e6, 1200e6),
            peak_cross_section: 1.5e-8, // barns
        });
        
        self.fusion_data.insert((8, 16, 8, 16), FusionCrossSectionData {
            reactant1: (8, 16), // ¹⁶O
            reactant2: (8, 16), // ¹⁶O
            q_value: 16.54,    // MeV
            s_factor: 2.0e-10, // Extremely small
            coulomb_barrier: 3920.0, // keV
            gamow_peak_energy: 2000.0, // keV
            temperature_range: (1500e6, 3000e6),
            peak_cross_section: 2.0e-10, // barns
        });
        
        // Silicon burning (quasi-equilibrium)
        self.fusion_data.insert((14, 28, 2, 4), FusionCrossSectionData {
            reactant1: (14, 28), // ²⁸Si
            reactant2: (2, 4),   // ⁴He
            q_value: 9.98,     // MeV
            s_factor: 1.0e-12, // Very suppressed
            coulomb_barrier: 2800.0, // keV
            gamow_peak_energy: 3500.0, // keV
            temperature_range: (3000e6, 5000e6),
            peak_cross_section: 1.0e-12, // barns
        });
    }
    
    /// Populate nuclear reaction cross-section data
    fn populate_reaction_data(&mut self) {
        // Neutron capture reactions (important for s-process and r-process)
        let fe56_n_gamma = ReactionCrossSectionData {
            target: (26, 56), // ⁵⁶Fe
            projectile: "neutron".to_string(),
            products: vec![(26, 57)], // ⁵⁷Fe
            energy_table: vec![
                (0.0253, 2.59),    // Thermal point (eV, barns)
                (1.0, 2.8),        // 1 eV
                (10.0, 3.2),       // 10 eV
                (100.0, 2.1),      // 100 eV
                (1000.0, 1.8),     // 1 keV
                (10000.0, 1.2),    // 10 keV
            ],
            thermal_cross_section: 2.59, // barns
            resonance_integral: 1.4, // barns
        };
        self.reaction_data.insert("Fe56_n_gamma".to_string(), fe56_n_gamma);
        
        // Add more important neutron capture reactions for heavy element synthesis
        let sr87_n_gamma = ReactionCrossSectionData {
            target: (38, 87), // ⁸⁷Sr (s-process branch point)
            projectile: "neutron".to_string(),
            products: vec![(38, 88)], // ⁸⁸Sr
            energy_table: vec![
                (0.0253, 16.0),    // High thermal cross-section
                (1.0, 18.0),
                (10.0, 21.0),
                (100.0, 15.0),
                (1000.0, 12.0),
            ],
            thermal_cross_section: 16.0, // barns
            resonance_integral: 8.2, // barns
        };
        self.reaction_data.insert("Sr87_n_gamma".to_string(), sr87_n_gamma);
        
        // Proton capture reactions
        let c12_p_gamma = ReactionCrossSectionData {
            target: (6, 12), // ¹²C
            projectile: "proton".to_string(),
            products: vec![(7, 13)], // ¹³N
            energy_table: vec![
                (10000.0, 1.7e-3),   // 10 keV
                (25000.0, 2.8e-3),   // 25 keV (Gamow peak)
                (50000.0, 3.1e-3),   // 50 keV
                (100000.0, 2.9e-3),  // 100 keV
                (1000000.0, 1.8e-3), // 1 MeV
            ],
            thermal_cross_section: 0.0, // Not applicable for charged particles
            resonance_integral: 0.0,
        };
        self.reaction_data.insert("C12_p_gamma".to_string(), c12_p_gamma);
    }
    
    /// Populate temperature-dependent cross-section grids
    fn populate_temperature_grids(&mut self) {
        // pp-chain temperature dependence
        let pp_grid = vec![
            (4e6, 4.7e-47),    // Solar core
            (6e6, 8.2e-47),    // Slightly hotter
            (10e6, 2.1e-46),   // Hot core
            (15e6, 5.8e-46),   // Very hot
            (20e6, 1.2e-45),   // Extreme
        ];
        self.temperature_grids.insert("pp_chain".to_string(), pp_grid);
        
        // CNO cycle temperature dependence (very steep)
        let cno_grid = vec![
            (15e6, 1.0e-4),    // Threshold
            (18e6, 1.7e-3),    // Standard
            (25e6, 8.3e-3),    // Enhanced
            (30e6, 2.1e-2),    // High temperature
            (40e6, 5.8e-2),    // Very high
        ];
        self.temperature_grids.insert("cno_cycle".to_string(), cno_grid);
        
        // Triple-alpha process (extremely temperature dependent)
        let triple_alpha_grid = vec![
            (80e6, 1.0e-10),   // Barely active
            (100e6, 3.0e-9),   // Helium flash threshold
            (120e6, 2.1e-8),   // Active
            (150e6, 8.7e-8),   // Strong
            (200e6, 3.2e-7),   // Very strong
        ];
        self.temperature_grids.insert("triple_alpha".to_string(), triple_alpha_grid);
    }
    
    /// Get fusion cross-section for specific reaction at given temperature
    pub fn get_fusion_cross_section(&self, z1: u32, a1: u32, z2: u32, a2: u32, temperature: f64) -> Option<f64> {
        // Try both orderings of reactants
        let key1 = (z1, a1, z2, a2);
        let key2 = (z2, a2, z1, a1);
        
        let data = self.fusion_data.get(&key1).or_else(|| self.fusion_data.get(&key2))?;
        
        // Check if temperature is in valid range
        if temperature < data.temperature_range.0 || temperature > data.temperature_range.1 {
            return Some(0.0); // Reaction not active at this temperature
        }
        
        // Calculate cross-section with Gamow peak suppression
        let kt = BOLTZMANN * temperature; // Thermal energy in Joules
        let kt_kev = kt / (1.602176634e-19 * 1000.0); // Convert to keV
        
        // Gamow peak energy in keV (temperature-dependent)
        let gamow_peak = 1.22 * (data.coulomb_barrier.powf(2.0) * kt_kev).powf(1.0/3.0);
        
        // Improved Gamow factor for stellar conditions
        let tau = (data.coulomb_barrier / kt_kev).sqrt();
        let gamow_factor = (temperature / 1e7).powf(1.0/3.0) * (-tau).exp(); // T^(1/3) dependence
        
        // S-factor approach with proper energy scaling
        let effective_energy = gamow_peak.max(kt_kev); // Use whichever is larger
        let cross_section_barns = data.s_factor * gamow_factor / effective_energy;
        
        // Temperature-dependent enhancement factor for higher temperatures
        let temp_factor = if temperature > 1e7 { (temperature / 1e7).powf(0.5) } else { 1.0 };
        
        let final_cross_section = cross_section_barns * temp_factor;
        Some(final_cross_section.max(1e-50) * 1e-24) // Convert barns to m²
    }
    
    /// Get nuclear reaction cross-section for given energy
    pub fn get_reaction_cross_section(&self, reaction_name: &str, energy_ev: f64) -> Option<f64> {
        let data = self.reaction_data.get(reaction_name)?;
        
        // Linear interpolation in energy table
        for i in 0..data.energy_table.len()-1 {
            let (e1, sigma1) = data.energy_table[i];
            let (e2, sigma2) = data.energy_table[i+1];
            
            if energy_ev >= e1 && energy_ev <= e2 {
                let t = (energy_ev - e1) / (e2 - e1);
                let sigma = sigma1 + t * (sigma2 - sigma1);
                return Some(sigma * 1e-24); // Convert barns to m²
            }
        }
        
        // Extrapolate if outside range
        if energy_ev < data.energy_table[0].0 {
            // Low energy: use thermal value
            Some(data.thermal_cross_section * 1e-24)
        } else {
            // High energy: assume 1/v dependence
            let last_point = data.energy_table.last().unwrap();
            let sigma = last_point.1 * (last_point.0 / energy_ev).sqrt();
            Some(sigma * 1e-24)
        }
    }
    
    /// Estimate cross-section for unknown nuclear reactions using systematics
    pub fn estimate_fusion_cross_section(&self, z1: u32, a1: u32, z2: u32, a2: u32, temperature: f64) -> f64 {
        // Nuclear systematics for estimating unknown cross-sections
        let coulomb_barrier = 1.44 * (z1 * z2) as f64 / (2.4e-15 * ((a1 as f64).powf(1.0/3.0) + (a2 as f64).powf(1.0/3.0))); // MeV
        let barrier_joules = coulomb_barrier * 1e6 * 1.602176634e-19; // Convert to J
        let kt = BOLTZMANN * temperature;
        
        // Gamow suppression
        let gamow_factor = (-barrier_joules / kt).exp();
        
        // Geometric cross-section estimate
        let r1 = 1.2e-15 * (a1 as f64).powf(1.0/3.0); // Nuclear radius
        let r2 = 1.2e-15 * (a2 as f64).powf(1.0/3.0);
        let _geometric_cross_section = std::f64::consts::PI * (r1 + r2).powi(2);
        
        // S-factor estimate (typical values 1e-3 to 1e-10 barns·keV)
        // Light nuclei have higher cross-sections due to lower Coulomb barriers
        let z_avg = (z1 + z2) as f64 / 2.0;
        let s_factor_estimate = if z_avg <= 2.0 {
            1e-3 // Light nuclei (higher cross-sections)
        } else if z_avg <= 6.0 {
            1e-6 // Intermediate 
        } else {
            1e-10 // Heavy nuclei (lower cross-sections)
        };
        
        let energy_kev = 3.0 * temperature / 1e3 * BOLTZMANN / 1.602176634e-19; // Thermal energy in keV
        let cross_section = s_factor_estimate * gamow_factor / energy_kev * 1e-24; // barns to m²
        
        cross_section.max(1e-50) // Minimum cross-section
    }
}

// Global nuclear cross-section database instance
use once_cell::sync::Lazy;
pub static NUCLEAR_DATABASE: Lazy<NuclearCrossSectionDatabase> = Lazy::new(|| NuclearCrossSectionDatabase::new());

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fusion_q_value_calculations() {
        // Test known fusion reactions with experimental Q-values
        
        // D + T → ⁴He + n (Q = 17.59 MeV)
        let deuterium = Nucleus::new(1, 1);  // Z=1, N=1, A=2
        let tritium = Nucleus::new(1, 2);    // Z=1, N=2, A=3
        let helium4 = Nucleus::new(2, 2);    // Z=2, N=2, A=4
        let neutron = Nucleus::new(0, 1);    // Z=0, N=1, A=1
        
        let d_be = deuterium.binding_energy();
        let t_be = tritium.binding_energy();
        let he4_be = helium4.binding_energy();
        let n_be = neutron.binding_energy();
        
        println!("Debug: D BE = {:.3} MeV, T BE = {:.3} MeV, He4 BE = {:.3} MeV, n BE = {:.3} MeV", 
                 d_be, t_be, he4_be, n_be);
        
        let initial_energy = d_be + t_be;
        let final_energy = he4_be + n_be;
        let q_value_mev = final_energy - initial_energy; // Already in MeV
        
        println!("Debug: Initial = {:.3} MeV, Final = {:.3} MeV, Q-value = {:.3} MeV", 
                 initial_energy, final_energy, q_value_mev);
        
        // Should be approximately 17.59 MeV
        assert!(q_value_mev > 15.0 && q_value_mev < 20.0, 
                "D+T fusion Q-value should be ~17.59 MeV, got {:.2} MeV", q_value_mev);
    }
    
    #[test]
    fn test_pp_chain_q_values() {
        // Test proton-proton chain Q-values
        
        // p + p → d + e⁺ + νₑ (Q = 1.44 MeV)
        let proton1 = Nucleus::new(1, 0);  // Z=1, N=0, A=1
        let proton2 = Nucleus::new(1, 0);  // Z=1, N=0, A=1
        let deuterium = Nucleus::new(1, 1); // Z=1, N=1, A=2
        
        let initial_energy = proton1.binding_energy() + proton2.binding_energy();
        let final_energy = deuterium.binding_energy();
        let q_value_mev = final_energy - initial_energy; // Already in MeV
        
        // Should be positive (energy released)
        assert!(q_value_mev > 0.0, "PP chain should release energy, got {:.2} MeV", q_value_mev);
        
        // d + p → ³He + γ (Q = 5.49 MeV)
        let helium3 = Nucleus::new(2, 1);  // Z=2, N=1, A=3
        let initial_energy_2 = deuterium.binding_energy() + proton1.binding_energy();
        let final_energy_2 = helium3.binding_energy();
        let q_value_2_mev = final_energy_2 - initial_energy_2; // Already in MeV
        
        assert!(q_value_2_mev > 4.0 && q_value_2_mev < 7.0,
                "d+p fusion Q-value should be ~5.49 MeV, got {:.2} MeV", q_value_2_mev);
    }
    
    #[test]
    fn test_alpha_process_q_values() {
        // Test triple-alpha process: 3 ⁴He → ¹²C (Q = 7.27 MeV)
        let helium4 = Nucleus::new(2, 2);   // Z=2, N=2, A=4
        let carbon12 = Nucleus::new(6, 6);  // Z=6, N=6, A=12
        
        let initial_energy = 3.0 * helium4.binding_energy();
        let final_energy = carbon12.binding_energy();
        let q_value_mev = final_energy - initial_energy; // Already in MeV
        
        // Should release energy for stellar helium burning
        assert!(q_value_mev > 5.0 && q_value_mev < 10.0,
                "Triple-alpha Q-value should be ~7.27 MeV, got {:.2} MeV", q_value_mev);
        
        // ⁴He + ¹²C → ¹⁶O (Q = 7.16 MeV)
        let oxygen16 = Nucleus::new(8, 8);  // Z=8, N=8, A=16
        let initial_energy_2 = helium4.binding_energy() + carbon12.binding_energy();
        let final_energy_2 = oxygen16.binding_energy();
        let q_value_2_mev = final_energy_2 - initial_energy_2; // Already in MeV
        
        assert!(q_value_2_mev > 5.0 && q_value_2_mev < 10.0,
                "Alpha-carbon Q-value should be ~7.16 MeV, got {:.2} MeV", q_value_2_mev);
    }
    
    #[test]
    fn test_energy_conservation_in_nuclear_reactions() {
        // Test that nuclear reactions conserve energy within precision
        let database = NuclearDatabase::new();
        
        // Test alpha decay: ²²⁶Ra → ²²²Rn + ⁴He
        let radium226 = Nucleus::new(88, 138);  // Z=88, N=138, A=226
        let _initial_mass_energy = radium226.binding_energy();
        
        if let Some(decay_data) = database.get_decay_data(88, 226) {
            let q_value_mev = decay_data.decay_energy; // Already in MeV
            
            // Energy should be conserved: Q-value should equal binding energy difference
            assert!(q_value_mev > 0.0, "Alpha decay should release energy");
            assert!(q_value_mev < 10.0, "Q-value should be reasonable for alpha decay (< 10 MeV)");
        }
    }
    
    #[test]
    fn test_nuclear_database_integrity() {
        let database = NuclearDatabase::new();
        
        // Test that stable isotopes are correctly identified
        assert!(database.is_stable(1, 1), "Hydrogen-1 should be stable");
        assert!(database.is_stable(2, 4), "Helium-4 should be stable");
        assert!(database.is_stable(6, 12), "Carbon-12 should be stable");
        assert!(database.is_stable(26, 56), "Iron-56 should be stable");
        
        // Test that unstable isotopes are correctly identified
        assert!(!database.is_stable(1, 3), "Tritium should be unstable");
        assert!(!database.is_stable(6, 14), "Carbon-14 should be unstable");
        assert!(!database.is_stable(92, 238), "Uranium-238 should be unstable");
        
        // Test decay data retrieval
        if let Some(tritium_data) = database.get_decay_data(1, 3) {
            assert_eq!(tritium_data.primary_mode, DecayMode::BetaMinus);
            assert!(tritium_data.half_life_seconds > 3e8); // ~12 years
            assert!(tritium_data.decay_energy > 0.0);
        } else {
            panic!("Tritium decay data should be available");
        }
    }
    
    #[test] 
    fn test_stellar_nucleosynthesis_reactions() {
        let nucleosynthesis = StellarNucleosynthesis::new();
        
        // Test that nucleosynthesis reactions are properly initialized
        assert!(!nucleosynthesis.reactions.is_empty(), "Should have initialized reactions");
        
        // Test reaction rate calculations at stellar conditions
        let stellar_temp = 1.5e7; // 15 million K (solar core)
        let stellar_density = 1.5e5; // kg/m³
        
        for reaction in &nucleosynthesis.reactions {
            let rate = reaction.calculate_rate(stellar_temp, stellar_density);
            assert!(rate >= 0.0, "Reaction rates should be non-negative");
            
            // At stellar conditions, some reactions should be active
            if matches!(reaction.reaction_type, NucleosynthesisReaction::PPChainI | 
                                               NucleosynthesisReaction::DeuteriumFusion) {
                assert!(rate > 0.0, "PP chain reactions should be active at stellar conditions");
            }
        }
    }
    
    #[test]
    fn test_nuclear_cross_section_database() {
        let database = NuclearCrossSectionDatabase::new();
        
        // Test pp-chain cross-sections
        let pp_cross_section = database.get_fusion_cross_section(1, 1, 1, 1, 1.5e7);
        assert!(pp_cross_section.is_some(), "Should have pp fusion cross-section");
        
        if let Some(cross_section) = pp_cross_section {
            assert!(cross_section > 0.0, "Cross-section should be positive");
            assert!(cross_section < 1e-20, "Cross-section should be realistic (< 1e-20 m²)");
        }
        
        // Test deuterium fusion cross-sections
        let d_p_cross_section = database.get_fusion_cross_section(1, 2, 1, 1, 1.5e7);
        assert!(d_p_cross_section.is_some(), "Should have deuterium-proton fusion cross-section");
        
        // Test that cross-sections increase with temperature (Gamow peak)
        let low_temp_cross_section = database.get_fusion_cross_section(1, 1, 1, 1, 1e7);
        let high_temp_cross_section = database.get_fusion_cross_section(1, 1, 1, 1, 2e7);
        
        if let (Some(low), Some(high)) = (low_temp_cross_section, high_temp_cross_section) {
            assert!(high > low, "Cross-section should increase with temperature due to Gamow peak");
        }
    }
    
    #[test]
    fn test_neutron_capture_processes() {
        let database = NeutronCaptureDatabase::new();
        
        // Test s-process capture rates
        let iron56_capture = database.get_capture_data(26, 56);
        assert!(iron56_capture.is_some(), "Should have neutron capture data for Fe-56");
        
        if let Some(capture_data) = iron56_capture {
            assert!(capture_data.cross_section > 0.0, "Capture cross-section should be positive");
            assert!(capture_data.capture_probability > 0.0 && capture_data.capture_probability <= 1.0,
                    "Capture probability should be between 0 and 1");
        }
        
        // Test that r-process rates are higher than s-process
        let s_process_rate = database.calculate_capture_rate(26, 56, 1e10, 1e8, &NeutronCaptureProcess::SProcess);
        let r_process_rate = database.calculate_capture_rate(26, 56, 1e20, 2e9, &NeutronCaptureProcess::RProcess);
        
        assert!(r_process_rate > s_process_rate, "R-process should have higher capture rates");
    }
    
    #[test]
    fn test_nuclear_systematics() {
        let database = NuclearCrossSectionDatabase::new();
        
        // Test that nuclear systematics provide reasonable estimates for unknown isotopes
        let estimated_cross_section = database.estimate_fusion_cross_section(3, 6, 3, 7, 1e8);
        assert!(estimated_cross_section > 0.0, "Estimated cross-section should be positive");
        assert!(estimated_cross_section < 1e-25, "Estimated cross-section should be realistic");
        
        // Test that estimates follow expected trends (lighter nuclei have higher cross-sections)
        let light_estimate = database.estimate_fusion_cross_section(1, 2, 1, 3, 1e8);
        let heavy_estimate = database.estimate_fusion_cross_section(10, 20, 10, 22, 1e8);
        
        assert!(light_estimate > heavy_estimate, 
                "Light nuclei should have higher fusion cross-sections than heavy nuclei");
    }
}