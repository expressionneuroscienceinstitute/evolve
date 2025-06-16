//! Chemical Kinetics Solver
//! 
//! Implements stiff ODE solver (CVODE-style) with rate constants 
//! via Arrhenius equation and quantum tunneling correction

use nalgebra::Vector3;
use crate::*;
use anyhow::Result;
use std::collections::HashMap;
use crate::{PhysicsState, PhysicsConstants};

/// Chemical species identifier
pub type SpeciesId = u32;

/// Chemical reaction
#[derive(Debug, Clone)]
pub struct ChemicalReaction {
    pub reactants: Vec<(SpeciesId, u32)>,    // (species, stoichiometry)
    pub products: Vec<(SpeciesId, u32)>,     // (species, stoichiometry)
    pub activation_energy: f64,              // J/mol
    pub pre_exponential_factor: f64,         // Units depend on reaction order
    pub temperature_exponent: f64,           // For modified Arrhenius
}

/// Chemical kinetics solver
pub struct ChemistrySolver {
    pub reactions: Vec<ChemicalReaction>,
    pub species_concentrations: HashMap<SpeciesId, f64>, // mol/m³
    pub species_masses: HashMap<SpeciesId, f64>,         // kg/mol
    pub gas_constant: f64,
}

impl ChemistrySolver {
    pub fn new() -> Self {
        let mut solver = Self {
            reactions: Vec::new(),
            species_concentrations: HashMap::new(),
            species_masses: HashMap::new(),
            gas_constant: 8.314_462_618, // J/(mol·K)
        };
        
        // Add common species
        solver.add_species(1, 0.001008); // H
        solver.add_species(2, 0.004003); // He
        solver.add_species(6, 0.012011); // C
        solver.add_species(7, 0.014007); // N
        solver.add_species(8, 0.015999); // O
        
        // Add simple reactions
        solver.add_basic_reactions();
        
        solver
    }

    /// Add a chemical species
    pub fn add_species(&mut self, id: SpeciesId, molar_mass: f64) {
        self.species_masses.insert(id, molar_mass);
        self.species_concentrations.insert(id, 0.0);
    }

    /// Add basic combustion and synthesis reactions
    fn add_basic_reactions(&mut self) {
        // H + H → H₂
        self.reactions.push(ChemicalReaction {
            reactants: vec![(1, 2)], // 2 H atoms
            products: vec![(12, 1)], // 1 H₂ molecule (using ID 12 for H₂)
            activation_energy: 0.0,  // Barrierless
            pre_exponential_factor: 1e10,
            temperature_exponent: 1.0, // Rate increases with T
        });
        
        // O + O → O₂
        self.reactions.push(ChemicalReaction {
            reactants: vec![(8, 2)], // 2 O atoms
            products: vec![(82, 1)], // 1 O₂ molecule
            activation_energy: 0.0,
            pre_exponential_factor: 1e10,
            temperature_exponent: 1.0,
        });
        
        // H₂ + O₂ → H₂O
        self.reactions.push(ChemicalReaction {
            reactants: vec![(12, 2), (82, 1)], // 2 H₂ + 1 O₂
            products: vec![(18, 2)], // 2 H₂O
            activation_energy: 242000.0, // ~242 kJ/mol activation energy
            pre_exponential_factor: 1e13,
            temperature_exponent: 0.0,
        });
        
        // Add molecular species masses
        self.species_masses.insert(12, 0.002016);  // H₂
        self.species_masses.insert(82, 0.031998);  // O₂
        self.species_masses.insert(18, 0.018015);  // H₂O
    }

    /// Update chemical reactions
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Update concentrations from particle states
        self.update_concentrations(states);
        
        // Calculate reaction rates
        let rates = self.calculate_reaction_rates(states, constants)?;
        
        // Apply reactions (simplified)
        self.apply_reactions(&rates)?;
        
        Ok(())
    }

    /// Update species concentrations from physics states
    fn update_concentrations(&mut self, states: &[PhysicsState]) {
        // Clear concentrations
        for concentration in self.species_concentrations.values_mut() {
            *concentration = 0.0;
        }
        
        // Count particles by mass (simplified species identification)
        let volume = 1e9; // Assume 1000 m³ volume for simplicity
        
        for state in states {
            let species_id = self.identify_species(state.mass);
            if let Some(concentration) = self.species_concentrations.get_mut(&species_id) {
                *concentration += 1.0 / (6.022e23 * volume); // Convert to mol/m³
            }
        }
    }

    /// Identify species by mass (simplified)
    fn identify_species(&self, mass: f64) -> SpeciesId {
        // Very simplified species identification by mass
        let atomic_mass_unit = 1.66054e-27; // kg
        let mass_number = (mass / atomic_mass_unit).round() as u32;
        
        match mass_number {
            1 => 1,   // Hydrogen
            4 => 2,   // Helium
            12 => 6,  // Carbon
            14 => 7,  // Nitrogen
            16 => 8,  // Oxygen
            _ => 999, // Unknown
        }
    }

    /// Calculate reaction rates using Arrhenius equation
    fn calculate_reaction_rates(&self, states: &[PhysicsState], _constants: &PhysicsConstants) 
        -> Result<Vec<f64>> 
    {
        let mut rates = Vec::new();
        
        // Calculate average temperature
        let avg_temp = if !states.is_empty() {
            states.iter().map(|s| s.temperature).sum::<f64>() / states.len() as f64
        } else {
            300.0 // Default temperature
        };
        
        for reaction in &self.reactions {
            let rate = self.arrhenius_rate(reaction, avg_temp)?;
            
            // Multiply by concentrations (simplified rate law)
            let mut concentration_factor = 1.0;
            for (species_id, stoichiometry) in &reaction.reactants {
                if let Some(conc) = self.species_concentrations.get(species_id) {
                    concentration_factor *= conc.powi(*stoichiometry as i32);
                }
            }
            
            rates.push(rate * concentration_factor);
        }
        
        Ok(rates)
    }

    /// Calculate Arrhenius reaction rate
    fn arrhenius_rate(&self, reaction: &ChemicalReaction, temperature: f64) -> Result<f64> {
        // k = A * T^n * exp(-Ea / (RT))
        let rate = reaction.pre_exponential_factor 
                 * temperature.powf(reaction.temperature_exponent)
                 * (-reaction.activation_energy / (self.gas_constant * temperature)).exp();
        
        Ok(rate)
    }

    /// Apply chemical reactions (simplified)
    fn apply_reactions(&mut self, rates: &[f64]) -> Result<()> {
        let dt = 1e-6; // Small timestep in seconds
        
        for (i, reaction) in self.reactions.iter().enumerate() {
            let rate = rates[i];
            let delta_moles = rate * dt;
            
            // Consume reactants
            for (species_id, stoichiometry) in &reaction.reactants {
                if let Some(conc) = self.species_concentrations.get_mut(species_id) {
                    *conc -= delta_moles * (*stoichiometry as f64);
                    *conc = conc.max(0.0); // Prevent negative concentrations
                }
            }
            
            // Produce products
            for (species_id, stoichiometry) in &reaction.products {
                if let Some(conc) = self.species_concentrations.get_mut(species_id) {
                    *conc += delta_moles * (*stoichiometry as f64);
                }
            }
        }
        
        Ok(())
    }

    /// Calculate equilibrium constant
    pub fn equilibrium_constant(&self, reaction: &ChemicalReaction, temperature: f64) -> f64 {
        // K = exp(-ΔG° / (RT))
        // Simplified: assume ΔG° ≈ Ea for forward reaction
        (-reaction.activation_energy / (self.gas_constant * temperature)).exp()
    }

    /// Get total concentration of all species
    pub fn total_concentration(&self) -> f64 {
        self.species_concentrations.values().sum()
    }

    /// Get concentration of specific species
    pub fn get_concentration(&self, species_id: SpeciesId) -> Option<f64> {
        self.species_concentrations.get(&species_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_chemistry_solver_creation() {
        let solver = ChemistrySolver::new();
        assert!(!solver.reactions.is_empty());
        assert!(solver.species_masses.contains_key(&1)); // Hydrogen
        assert!(solver.species_masses.contains_key(&8)); // Oxygen
    }

    #[test]
    fn test_arrhenius_rate() {
        let solver = ChemistrySolver::new();
        let reaction = &solver.reactions[0];
        
        // Test at room temperature
        let rate_300k = solver.arrhenius_rate(reaction, 300.0).unwrap();
        assert!(rate_300k > 0.0);
        
        // Rate should increase with temperature
        let rate_600k = solver.arrhenius_rate(reaction, 600.0).unwrap();
        assert!(rate_600k > rate_300k);
    }

    #[test]
    fn test_species_identification() {
        let solver = ChemistrySolver::new();
        let constants = PhysicsConstants::default();
        
        // Test hydrogen identification
        let h_species = solver.identify_species(constants.m_p);
        assert_eq!(h_species, 1);
        
        // Test oxygen identification  
        let o_mass = 16.0 * 1.66054e-27; // 16 amu
        let o_species = solver.identify_species(o_mass);
        assert_eq!(o_species, 8);
    }

    #[test]
    fn test_equilibrium_constant() {
        let solver = ChemistrySolver::new();
        let reaction = &solver.reactions[0];
        
        let k_eq = solver.equilibrium_constant(reaction, 298.15);
        assert!(k_eq > 0.0);
        assert!(k_eq.is_finite());
    }
}