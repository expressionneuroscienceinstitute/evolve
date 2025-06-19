//! Interaction events and decay processes for particle physics simulations
//! 
//! This module defines interaction types, decay channels, and reaction mechanisms
//! used throughout the EVOLVE universe simulation. Based on experimental particle
//! physics data and nuclear reaction databases.
//! 
//! References:
//! - PDG (Particle Data Group) decay data
//! - NIST nuclear reaction databases  
//! - ENDF/B-VIII.0 cross-section libraries
//! - Experimental high-energy physics results

use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use crate::particle_types::{ParticleType, InteractionType};

/// Comprehensive interaction event record
/// Stores all relevant information about particle interactions and decays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub timestamp: f64,                    // Time of interaction (s)
    pub interaction_type: InteractionType, // Type of interaction
    pub participants: Vec<usize>,          // Particle indices involved
    pub energy_exchanged: f64,             // Energy transfer (J)
    pub momentum_transfer: Vector3<f64>,   // Momentum transfer vector (kg⋅m/s)
    pub products: Vec<ParticleType>,       // Particles produced
    pub cross_section: f64,                // Interaction cross-section (m²)
}

impl InteractionEvent {
    /// Create a new interaction event
    pub fn new(
        timestamp: f64,
        interaction_type: InteractionType,
        participants: Vec<usize>,
        energy_exchanged: f64,
        momentum_transfer: Vector3<f64>,
        products: Vec<ParticleType>,
        cross_section: f64,
    ) -> Self {
        Self {
            timestamp,
            interaction_type,
            participants,
            energy_exchanged,
            momentum_transfer,
            products,
            cross_section,
        }
    }
    
    /// Check if this is a particle decay event (single participant)
    pub fn is_decay(&self) -> bool {
        self.participants.len() == 1
    }
    
    /// Check if this is a scattering event (two participants)
    pub fn is_scattering(&self) -> bool {
        self.participants.len() == 2
    }
    
    /// Get the total energy scale of the interaction
    pub fn energy_scale(&self) -> f64 {
        self.energy_exchanged.abs()
    }
}

/// Decay channel for unstable particles
/// Based on experimental branching ratios and decay constants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayChannel {
    pub products: Vec<ParticleType>,  // Decay products
    pub branching_ratio: f64,         // Probability of this decay mode
    pub decay_constant: f64,          // Decay rate (s⁻¹)
}

impl DecayChannel {
    /// Create a new decay channel
    pub fn new(products: Vec<ParticleType>, branching_ratio: f64, decay_constant: f64) -> Self {
        Self {
            products,
            branching_ratio,
            decay_constant,
        }
    }
    
    /// Get the half-life of this decay mode
    pub fn half_life(&self) -> f64 {
        if self.decay_constant > 0.0 {
            (2.0_f64).ln() / self.decay_constant
        } else {
            f64::INFINITY
        }
    }
    
    /// Calculate probability of decay within time interval dt
    pub fn decay_probability(&self, dt: f64) -> f64 {
        1.0 - (-self.decay_constant * dt).exp()
    }
}

/// Nuclear fusion reaction definition
/// Includes Q-value calculations and cross-section data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionReaction {
    pub reactant_indices: Vec<usize>,    // Indices of reacting nuclei
    pub product_mass_number: u32,        // Mass number of fusion product
    pub product_atomic_number: u32,      // Atomic number of fusion product
    pub q_value: f64,                    // Energy released (J)
    pub cross_section: f64,              // Reaction cross-section (m²)
    pub requires_catalysis: bool,        // Whether catalysis is needed
}

impl Default for FusionReaction {
    fn default() -> Self {
        Self {
            reactant_indices: Vec::new(),
            product_mass_number: 0,
            product_atomic_number: 0,
            q_value: 0.0,
            cross_section: 0.0,
            requires_catalysis: false,
        }
    }
}

impl FusionReaction {
    /// Create a new fusion reaction
    pub fn new(
        reactant_indices: Vec<usize>,
        product_mass_number: u32,
        product_atomic_number: u32,
        q_value: f64,
        cross_section: f64,
    ) -> Self {
        Self {
            reactant_indices,
            product_mass_number,
            product_atomic_number,
            q_value,
            cross_section,
            requires_catalysis: false,
        }
    }
    
    /// Check if this is an exothermic reaction
    pub fn is_exothermic(&self) -> bool {
        self.q_value > 0.0
    }
    
    /// Get the Coulomb barrier height for this reaction (simplified)
    pub fn coulomb_barrier(&self) -> f64 {
        // Simplified Coulomb barrier calculation
        // E_barrier ≈ k * Z1 * Z2 / (A1^(1/3) + A2^(1/3))
        let z1 = self.product_atomic_number as f64; // Simplified
        let z2 = 1.0; // Assume second reactant is hydrogen
        let a1 = self.product_mass_number as f64;
        let a2: f64 = 1.0;
        
        let ke = 1.44; // MeV⋅fm (Coulomb constant)
        let r0 = 1.2;  // fm (nuclear radius parameter)
        
        ke * z1 * z2 / (r0 * (a1.powf(1.0/3.0) + a2.powf(1.0/3.0))) * 1.602e-13 // Convert MeV to J
    }
}

/// Comprehensive decay data for particles
/// Based on PDG (Particle Data Group) experimental compilations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayData {
    pub half_life_seconds: f64,                        // Particle half-life (s)
    pub decay_modes: Vec<DecayMode>,                   // All possible decay modes
    pub q_value_mev: f64,                             // Q-value for decay (MeV)
    pub daughter_products: Vec<(ParticleType, f64)>,   // (particle, branching_ratio)
}

/// Individual decay mode with specific products and probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayMode {
    pub products: Vec<ParticleType>,  // Decay products
    pub branching_ratio: f64,         // Fraction of decays via this mode
    pub q_value: f64,                // Energy released in this mode (MeV)
}

impl DecayMode {
    /// Create a new decay mode
    pub fn new(products: Vec<ParticleType>, branching_ratio: f64, q_value: f64) -> Self {
        Self {
            products,
            branching_ratio,
            q_value,
        }
    }
}

impl DecayData {
    /// Create comprehensive decay data for common particles
    pub fn for_particle(particle_type: ParticleType) -> Option<Self> {
        match particle_type {
            ParticleType::Neutron => Some(Self {
                half_life_seconds: 879.4, // Free neutron lifetime
                decay_modes: vec![
                    DecayMode::new(
                        vec![ParticleType::Proton, ParticleType::Electron, ParticleType::ElectronAntiNeutrino],
                        1.0, // 100% beta-minus decay
                        0.782, // Q-value in MeV
                    ),
                ],
                q_value_mev: 0.782,
                daughter_products: vec![
                    (ParticleType::Proton, 1.0),
                    (ParticleType::Electron, 1.0),
                    (ParticleType::ElectronAntiNeutrino, 1.0),
                ],
            }),
            
            ParticleType::Muon => Some(Self {
                half_life_seconds: 2.197e-6, // Muon lifetime
                decay_modes: vec![
                    DecayMode::new(
                        vec![ParticleType::Electron, ParticleType::ElectronAntiNeutrino, ParticleType::MuonNeutrino],
                        1.0, // 100% muon decay
                        105.7, // Q-value in MeV
                    ),
                ],
                q_value_mev: 105.7,
                daughter_products: vec![
                    (ParticleType::Electron, 1.0),
                    (ParticleType::ElectronAntiNeutrino, 1.0),
                    (ParticleType::MuonNeutrino, 1.0),
                ],
            }),
            
            ParticleType::PionPlus => Some(Self {
                half_life_seconds: 2.603e-8, // Charged pion lifetime
                decay_modes: vec![
                    DecayMode::new(
                        vec![ParticleType::Muon, ParticleType::MuonNeutrino],
                        0.9998, // 99.98% to muon + neutrino
                        33.9, // Q-value in MeV
                    ),
                    DecayMode::new(
                        vec![ParticleType::Electron, ParticleType::ElectronNeutrino],
                        0.0002, // 0.02% to electron + neutrino
                        33.9, // Q-value in MeV
                    ),
                ],
                q_value_mev: 33.9,
                daughter_products: vec![
                    (ParticleType::Muon, 0.9998),
                    (ParticleType::MuonNeutrino, 0.9998),
                    (ParticleType::Electron, 0.0002),
                    (ParticleType::ElectronNeutrino, 0.0002),
                ],
            }),
            
            ParticleType::PionZero => Some(Self {
                half_life_seconds: 8.52e-17, // Neutral pion lifetime
                decay_modes: vec![
                    DecayMode::new(
                        vec![ParticleType::Photon, ParticleType::Photon],
                        0.9882, // 98.82% to two photons
                        135.0, // Q-value in MeV
                    ),
                ],
                q_value_mev: 135.0,
                daughter_products: vec![
                    (ParticleType::Photon, 1.9764), // Two photons
                ],
            }),
            
            // Stable particles return None
            ParticleType::Electron | ParticleType::Proton | 
            ParticleType::ElectronNeutrino | ParticleType::MuonNeutrino | 
            ParticleType::TauNeutrino => None,
            
            // Default for unknown particles
            _ => None,
        }
    }
    
    /// Get the mean lifetime (τ = t₁/₂ / ln(2))
    pub fn mean_lifetime(&self) -> f64 {
        self.half_life_seconds / (2.0_f64).ln()
    }
    
    /// Calculate decay constant (λ = 1/τ)
    pub fn decay_constant(&self) -> f64 {
        1.0 / self.mean_lifetime()
    }
    
    /// Select a random decay mode based on branching ratios
    pub fn select_decay_mode(&self, random_value: f64) -> Option<&DecayMode> {
        let mut cumulative_probability = 0.0;
        for mode in &self.decay_modes {
            cumulative_probability += mode.branching_ratio;
            if random_value <= cumulative_probability {
                return Some(mode);
            }
        }
        None
    }
}

/// Material properties for particle transport simulations
/// Used with Geant4 integration and stopping power calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub name: String,                           // Material name
    pub density_g_cm3: f64,                    // Density (g/cm³)
    pub atomic_composition: Vec<(u32, f64)>,   // (Z, fraction by weight)
    pub mean_excitation_energy_ev: f64,        // Mean excitation energy (eV)
    pub radiation_length_cm: f64,              // Radiation length (cm)
    pub nuclear_interaction_length_cm: f64,    // Nuclear interaction length (cm)
}

impl MaterialProperties {
    /// Create material properties for common materials
    pub fn standard_material(name: &str) -> Option<Self> {
        match name {
            "Vacuum" => Some(Self {
                name: "Vacuum".to_string(),
                density_g_cm3: 0.0,
                atomic_composition: vec![],
                mean_excitation_energy_ev: 0.0,
                radiation_length_cm: f64::INFINITY,
                nuclear_interaction_length_cm: f64::INFINITY,
            }),
            
            "Air" => Some(Self {
                name: "Air".to_string(),
                density_g_cm3: 0.001225, // At STP
                atomic_composition: vec![
                    (7, 0.755),  // Nitrogen
                    (8, 0.232),  // Oxygen
                    (18, 0.013), // Argon
                ],
                mean_excitation_energy_ev: 85.7,
                radiation_length_cm: 36.66,
                nuclear_interaction_length_cm: 70.78,
            }),
            
            "Water" => Some(Self {
                name: "Water".to_string(),
                density_g_cm3: 1.0,
                atomic_composition: vec![
                    (1, 0.111),  // Hydrogen
                    (8, 0.889),  // Oxygen
                ],
                mean_excitation_energy_ev: 78.0,
                radiation_length_cm: 36.08,
                nuclear_interaction_length_cm: 83.46,
            }),
            
            "Lead" => Some(Self {
                name: "Lead".to_string(),
                density_g_cm3: 11.34,
                atomic_composition: vec![
                    (82, 1.0),   // Lead
                ],
                mean_excitation_energy_ev: 823.0,
                radiation_length_cm: 0.561,
                nuclear_interaction_length_cm: 18.26,
            }),
            
            _ => None,
        }
    }
    
    /// Calculate effective atomic number for compound material
    pub fn effective_atomic_number(&self) -> f64 {
        let mut sum_z_weight = 0.0;
        let mut sum_weight = 0.0;
        
        for (z, fraction) in &self.atomic_composition {
            sum_z_weight += *z as f64 * fraction;
            sum_weight += fraction;
        }
        
        if sum_weight > 0.0 {
            sum_z_weight / sum_weight
        } else {
            0.0
        }
    }
}

/// Stopping power table for particle energy loss calculations
/// Based on NIST stopping power and range tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingPowerTable {
    pub energies_mev: Vec<f64>,                 // Kinetic energies (MeV)
    pub stopping_powers_mev_cm2_g: Vec<f64>,    // dE/dx values (MeV⋅cm²/g)
    pub range_mev_cm2_g: Vec<f64>,             // Range values (MeV⋅cm²/g)
    pub material: String,                       // Material name
}

impl StoppingPowerTable {
    /// Create a stopping power table for electrons in a material
    pub fn for_electrons(material: &MaterialProperties) -> Self {
        // Simplified Bethe-Bloch formula for demonstration
        // In practice, would use tabulated NIST data
        let energies: Vec<f64> = (0..100).map(|i| 0.01 * (i + 1) as f64).collect(); // 0.01 to 1 MeV
        let stopping_powers: Vec<f64> = energies.iter()
            .map(|&energy| Self::bethe_bloch_stopping_power(energy, &material))
            .collect();
        let ranges: Vec<f64> = energies.iter()
            .map(|&energy| Self::csda_range(energy, &material))
            .collect();
        
        Self {
            energies_mev: energies,
            stopping_powers_mev_cm2_g: stopping_powers,
            range_mev_cm2_g: ranges,
            material: material.name.clone(),
        }
    }
    
    /// Simplified Bethe-Bloch stopping power calculation
    fn bethe_bloch_stopping_power(energy_mev: f64, material: &MaterialProperties) -> f64 {
        // Very simplified - in reality use full Bethe-Bloch formula
        let z_eff = material.effective_atomic_number();
        let density = material.density_g_cm3;
        
        // Approximate stopping power (MeV⋅cm²/g)
        0.307 * z_eff / energy_mev * density.ln()
    }
    
    /// Simplified CSDA range calculation
    fn csda_range(energy_mev: f64, _material: &MaterialProperties) -> f64 {
        // Very simplified range calculation
        // R ≈ 0.412 * T^(1.265-0.0954*ln(T)) for T in MeV, R in g/cm²
        0.412 * energy_mev.powf(1.265 - 0.0954 * energy_mev.ln())
    }
    
    /// Interpolate stopping power at given energy
    pub fn stopping_power_at_energy(&self, energy_mev: f64) -> f64 {
        if let Some(index) = self.energies_mev.iter().position(|&e| e >= energy_mev) {
            if index == 0 {
                self.stopping_powers_mev_cm2_g[0]
            } else {
                // Linear interpolation
                let e1 = self.energies_mev[index - 1];
                let e2 = self.energies_mev[index];
                let sp1 = self.stopping_powers_mev_cm2_g[index - 1];
                let sp2 = self.stopping_powers_mev_cm2_g[index];
                
                sp1 + (sp2 - sp1) * (energy_mev - e1) / (e2 - e1)
            }
        } else {
            // Extrapolate or use last value
            self.stopping_powers_mev_cm2_g.last().copied().unwrap_or(0.0)
        }
    }
} 