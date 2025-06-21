//! Radiative Transfer and Cooling Physics
//! 
//! Implements comprehensive radiative cooling and heating mechanisms essential for
//! realistic star formation and gas collapse. Based on latest research in radiative
//! cooling physics and photonic structures.
//! 
//! References:
//! - Nature Light: Science & Applications (2023) - Photonic structures in radiative cooling
//! - GADGET-2 cosmological SPH methods
//! - FLASH astrophysical radiative transfer

use anyhow::Result;
use nalgebra::Vector3;
use crate::{PhysicsState, PhysicsConstants};

/// Stefan-Boltzmann constant (W⋅m⁻²⋅K⁻⁴)
const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
/// Speed of light (m/s)
const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Planck constant (J⋅s)
const PLANCK_CONSTANT: f64 = 6.62607015e-34;
/// Boltzmann constant (J/K)
const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Radiative cooling/heating rates and mechanisms
#[derive(Debug, Clone)]
pub struct RadiativeTransfer {
    /// Optically thin cooling rate (W/m³)
    pub optically_thin_cooling: f64,
    /// Optically thick cooling rate (W/m³)
    pub optically_thick_cooling: f64,
    /// Hydrogen line cooling rate (W/m³)
    pub hydrogen_line_cooling: f64,
    /// Helium line cooling rate (W/m³)
    pub helium_line_cooling: f64,
    /// Dust cooling rate (W/m³)
    pub dust_cooling: f64,
    /// Stellar feedback heating rate (W/m³)
    pub stellar_heating: f64,
    /// Cosmic microwave background heating (W/m³)
    pub cmb_heating: f64,
    /// Total net cooling/heating rate (W/m³)
    pub net_rate: f64,
}

/// Radiative transfer solver for astrophysical gas
#[derive(Debug)]
pub struct RadiativeTransferSolver {
    /// Optical depth threshold for thin/thick transition
    pub optical_depth_threshold: f64,
    /// Dust-to-gas ratio (mass fraction)
    pub dust_to_gas_ratio: f64,
    /// Cosmic microwave background temperature (K)
    pub cmb_temperature: f64,
    /// Stellar feedback efficiency
    pub stellar_feedback_efficiency: f64,
    /// Hydrogen ionization fraction
    pub hydrogen_ionization_fraction: f64,
    /// Helium ionization fraction
    pub helium_ionization_fraction: f64,
}

impl Default for RadiativeTransferSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl RadiativeTransferSolver {
    pub fn new() -> Self {
        Self {
            optical_depth_threshold: 1.0,
            dust_to_gas_ratio: 0.01, // 1% dust by mass
            cmb_temperature: 2.725, // Current CMB temperature
            stellar_feedback_efficiency: 0.1, // 10% of stellar luminosity
            hydrogen_ionization_fraction: 0.1, // 10% ionized
            helium_ionization_fraction: 0.01, // 1% ionized
        }
    }

    /// Calculate comprehensive radiative cooling and heating for a gas parcel
    pub fn calculate_radiative_transfer(
        &self,
        temperature: f64,
        density: f64,
        metallicity: f64,
        stellar_luminosity: f64,
        distance_to_star: f64,
        constants: &PhysicsConstants,
    ) -> RadiativeTransfer {
        // Calculate optical depth for thin/thick transition
        let optical_depth = self.calculate_optical_depth(density, metallicity);
        
        // Optically thin cooling (free-free, bound-free, line emission)
        let optically_thin_cooling = self.calculate_optically_thin_cooling(
            temperature, density, metallicity, constants
        );
        
        // Optically thick cooling (diffusion approximation)
        let optically_thick_cooling = if optical_depth > self.optical_depth_threshold {
            self.calculate_optically_thick_cooling(temperature, density, optical_depth)
        } else {
            0.0
        };
        
        // Hydrogen line cooling (Lyman-alpha, Balmer series)
        let hydrogen_line_cooling = self.calculate_hydrogen_line_cooling(
            temperature, density, constants
        );
        
        // Helium line cooling (He I, He II lines)
        let helium_line_cooling = self.calculate_helium_line_cooling(
            temperature, density, constants
        );
        
        // Dust cooling (thermal emission from dust grains)
        let dust_cooling = self.calculate_dust_cooling(
            temperature, density, metallicity
        );
        
        // Stellar feedback heating (radiation from nearby stars)
        let stellar_heating = self.calculate_stellar_heating(
            stellar_luminosity, distance_to_star, density
        );
        
        // Cosmic microwave background heating
        let cmb_heating = self.calculate_cmb_heating(temperature, density);
        
        // Calculate net rate (positive = heating, negative = cooling)
        let total_cooling = optically_thin_cooling + optically_thick_cooling + 
                           hydrogen_line_cooling + helium_line_cooling + dust_cooling;
        let total_heating = stellar_heating + cmb_heating;
        let net_rate = total_heating - total_cooling;
        
        RadiativeTransfer {
            optically_thin_cooling,
            optically_thick_cooling,
            hydrogen_line_cooling,
            helium_line_cooling,
            dust_cooling,
            stellar_heating,
            cmb_heating,
            net_rate,
        }
    }

    /// Calculate optical depth for gas parcel
    fn calculate_optical_depth(&self, density: f64, metallicity: f64) -> f64 {
        // Optical depth τ = κρL where κ is opacity, ρ is density, L is characteristic length
        let characteristic_length = 1e16; // 1 pc in meters
        
        // Rosseland mean opacity (simplified)
        // κ ≈ 10⁻²⁰ m²/kg for molecular gas, scales with metallicity
        let base_opacity = 1e-20; // m²/kg
        let opacity = base_opacity * (1.0 + metallicity * 10.0); // Enhanced by metals
        
        opacity * density * characteristic_length
    }

    /// Calculate optically thin cooling (free-free, bound-free, line emission)
    fn calculate_optically_thin_cooling(
        &self,
        temperature: f64,
        density: f64,
        metallicity: f64,
        constants: &PhysicsConstants,
    ) -> f64 {
        // Free-free (bremsstrahlung) cooling
        // Λ_ff ≈ 1.4×10⁻²⁷ T^(1/2) n_e n_i erg cm³ s⁻¹
        let electron_density = density * self.hydrogen_ionization_fraction / constants.m_p;
        let ion_density = density / constants.m_p;
        let free_free_cooling = 1.4e-27 * temperature.sqrt() * electron_density * ion_density;
        
        // Bound-free cooling (photoionization)
        // Λ_bf ≈ 1.3×10⁻²⁴ T^(-0.7) n_e n_H erg cm³ s⁻¹
        let hydrogen_density = density / constants.m_p;
        let bound_free_cooling = 1.3e-24 * temperature.powf(-0.7) * electron_density * hydrogen_density;
        
        // Metal line cooling (scales with metallicity)
        // Λ_metal ≈ 10⁻²² T^(-0.5) n² Z erg cm³ s⁻¹
        let metal_line_cooling = 1e-22 * temperature.powf(-0.5) * density * density * metallicity;
        
        // Convert from erg cm³ s⁻¹ to W/m³
        let cooling_rate = (free_free_cooling + bound_free_cooling + metal_line_cooling) * 1e-7;
        
        cooling_rate.max(0.0)
    }

    /// Calculate optically thick cooling using diffusion approximation
    fn calculate_optically_thick_cooling(
        &self,
        temperature: f64,
        density: f64,
        optical_depth: f64,
    ) -> f64 {
        // Diffusion approximation: F = (4acT³)/(3κρ) * ∇T
        // For optically thick gas, cooling rate ≈ σT⁴/τ
        let blackbody_flux = STEFAN_BOLTZMANN * temperature.powi(4);
        let cooling_rate = blackbody_flux / optical_depth;
        
        cooling_rate.max(0.0)
    }

    /// Calculate hydrogen line cooling (Lyman-alpha, Balmer series)
    fn calculate_hydrogen_line_cooling(
        &self,
        temperature: f64,
        density: f64,
        constants: &PhysicsConstants,
    ) -> f64 {
        // Lyman-alpha cooling (n=2 → n=1)
        // Energy per transition: E = 13.6 eV * (1 - 1/4) = 10.2 eV
        let lyman_alpha_energy = 10.2 * 1.602e-19; // Joules
        
        // Population of n=2 level (Boltzmann distribution)
        let energy_gap = 10.2 * 1.602e-19; // Joules
        let boltzmann_factor = (-energy_gap / (BOLTZMANN_CONSTANT * temperature)).exp();
        
        // Collisional excitation rate (simplified)
        let collision_rate = 1e-15 * temperature.sqrt(); // m³/s
        
        // Lyman-alpha cooling rate
        let lyman_alpha_cooling = lyman_alpha_energy * collision_rate * 
                                 density * density * boltzmann_factor;
        
        // Balmer series cooling (n=3 → n=2, etc.)
        // Simplified: assume 20% of Lyman-alpha rate
        let balmer_cooling = 0.2 * lyman_alpha_cooling;
        
        lyman_alpha_cooling + balmer_cooling
    }

    /// Calculate helium line cooling (He I, He II lines)
    fn calculate_helium_line_cooling(
        &self,
        temperature: f64,
        density: f64,
        constants: &PhysicsConstants,
    ) -> f64 {
        // Helium abundance (10% by number)
        let helium_abundance = 0.1;
        let helium_density = density * helium_abundance * 4.0 / constants.m_p;
        
        // He I cooling (singlet-triplet transitions)
        // Energy per transition: ~20 eV
        let he1_energy = 20.0 * 1.602e-19; // Joules
        let he1_cooling = he1_energy * 1e-16 * temperature.sqrt() * 
                         helium_density * density;
        
        // He II cooling (Lyman-alpha equivalent)
        // Energy per transition: ~40 eV (4× hydrogen due to Z²)
        let he2_energy = 40.0 * 1.602e-19; // Joules
        let he2_cooling = he2_energy * 1e-17 * temperature.sqrt() * 
                         helium_density * density * self.helium_ionization_fraction;
        
        he1_cooling + he2_cooling
    }

    /// Calculate dust cooling (thermal emission from dust grains)
    fn calculate_dust_cooling(
        &self,
        temperature: f64,
        density: f64,
        metallicity: f64,
    ) -> f64 {
        // Dust temperature (assume in thermal equilibrium with gas)
        let dust_temperature = temperature;
        
        // Dust cooling rate: Λ_dust ≈ 4πa²σT_dust⁴ * n_dust
        // where a is grain radius, n_dust is dust number density
        
        let grain_radius = 1e-6; // 1 μm typical grain size
        let grain_cross_section = std::f64::consts::PI * grain_radius * grain_radius;
        
        // Dust number density (scales with metallicity)
        let dust_number_density = density * self.dust_to_gas_ratio * metallicity / 
                                 (4.0/3.0 * std::f64::consts::PI * grain_radius.powi(3) * 3000.0); // 3000 kg/m³ dust density
        
        let dust_cooling = 4.0 * grain_cross_section * STEFAN_BOLTZMANN * 
                          dust_temperature.powi(4) * dust_number_density;
        
        dust_cooling.max(0.0)
    }

    /// Calculate stellar feedback heating from nearby stars
    fn calculate_stellar_heating(
        &self,
        stellar_luminosity: f64,
        distance_to_star: f64,
        density: f64,
    ) -> f64 {
        // Stellar radiation flux at distance r: F = L/(4πr²)
        let stellar_flux = stellar_luminosity / (4.0 * std::f64::consts::PI * distance_to_star * distance_to_star);
        
        // Heating rate per unit volume: H = F * κ * ρ * efficiency
        let opacity = 1e-20; // m²/kg (UV opacity)
        let heating_rate = stellar_flux * opacity * density * self.stellar_feedback_efficiency;
        
        heating_rate.max(0.0)
    }

    /// Calculate cosmic microwave background heating
    fn calculate_cmb_heating(&self, temperature: f64, density: f64) -> f64 {
        // CMB heating when gas is cooler than CMB
        if temperature < self.cmb_temperature {
            // Heating rate: H = 4σT_cmb⁴ * κ * ρ
            let cmb_flux = STEFAN_BOLTZMANN * self.cmb_temperature.powi(4);
            let opacity = 1e-20; // m²/kg (microwave opacity)
            cmb_flux * opacity * density
        } else {
            0.0 // No heating if gas is hotter than CMB
        }
    }

    /// Update temperature of gas parcel based on radiative transfer
    pub fn update_temperature(
        &self,
        temperature: &mut f64,
        density: f64,
        metallicity: f64,
        stellar_luminosity: f64,
        distance_to_star: f64,
        dt: f64,
        constants: &PhysicsConstants,
    ) -> Result<()> {
        let radiative_transfer = self.calculate_radiative_transfer(
            *temperature, density, metallicity, stellar_luminosity, distance_to_star, constants
        );
        
        // Energy change per unit mass: ΔE = (net_rate * dt) / (ρ * c_v)
        let specific_heat = 1.5 * BOLTZMANN_CONSTANT / constants.m_p; // J/(kg·K) for monatomic gas
        let energy_change = radiative_transfer.net_rate * dt / (density * specific_heat);
        
        // Update temperature
        *temperature += energy_change;
        
        // Ensure temperature stays positive
        *temperature = temperature.max(1.0); // Minimum 1 K
        
        Ok(())
    }

    /// Calculate Jeans mass including radiative cooling effects
    pub fn calculate_jeans_mass_with_cooling(
        &self,
        temperature: f64,
        density: f64,
        metallicity: f64,
        constants: &PhysicsConstants,
    ) -> f64 {
        // Standard Jeans mass: M_J ∝ T^(3/2) / ρ^(1/2)
        let standard_jeans_mass = (3.0 * BOLTZMANN_CONSTANT * temperature / 
                                  (constants.g * constants.m_p)).powf(1.5) / density.sqrt();
        
        // Cooling correction factor
        // If cooling is efficient, gas can fragment more easily
        let radiative_transfer = self.calculate_radiative_transfer(
            temperature, density, metallicity, 0.0, f64::INFINITY, constants
        );
        
        let cooling_efficiency = if radiative_transfer.net_rate < 0.0 {
            // Net cooling - reduces Jeans mass
            (-radiative_transfer.net_rate / (STEFAN_BOLTZMANN * temperature.powi(4))).min(1.0)
        } else {
            // Net heating - increases Jeans mass
            1.0 + (radiative_transfer.net_rate / (STEFAN_BOLTZMANN * temperature.powi(4))).min(1.0)
        };
        
        standard_jeans_mass / cooling_efficiency
    }

    /// Calculate cooling time (time for gas to cool significantly)
    pub fn calculate_cooling_time(
        &self,
        temperature: f64,
        density: f64,
        metallicity: f64,
        constants: &PhysicsConstants,
    ) -> f64 {
        let radiative_transfer = self.calculate_radiative_transfer(
            temperature, density, metallicity, 0.0, f64::INFINITY, constants
        );
        
        if radiative_transfer.net_rate <= 0.0 {
            f64::INFINITY // No cooling
        } else {
            // Cooling time: t_cool = (3/2) * n * k_B * T / Λ
            let thermal_energy_density = 1.5 * density * BOLTZMANN_CONSTANT * temperature / constants.m_p;
            thermal_energy_density / radiative_transfer.net_rate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicsConstants;

    #[test]
    fn test_radiative_transfer_solver_creation() {
        let solver = RadiativeTransferSolver::new();
        assert_eq!(solver.cmb_temperature, 2.725);
        assert_eq!(solver.dust_to_gas_ratio, 0.01);
    }

    #[test]
    fn test_optically_thin_cooling() {
        let solver = RadiativeTransferSolver::new();
        let constants = PhysicsConstants::default();
        
        let cooling = solver.calculate_optically_thin_cooling(
            1000.0, // 1000 K
            1e-20,  // 1e-20 kg/m³
            0.01,   // 1% metallicity
            &constants
        );
        
        assert!(cooling > 0.0);
        assert!(cooling.is_finite());
    }

    #[test]
    fn test_hydrogen_line_cooling() {
        let solver = RadiativeTransferSolver::new();
        let constants = PhysicsConstants::default();
        
        let cooling = solver.calculate_hydrogen_line_cooling(
            10000.0, // 10,000 K (hot enough for significant line cooling)
            1e-18,   // 1e-18 kg/m³
            &constants
        );
        
        assert!(cooling > 0.0);
        assert!(cooling.is_finite());
    }

    #[test]
    fn test_dust_cooling() {
        let solver = RadiativeTransferSolver::new();
        
        let cooling = solver.calculate_dust_cooling(
            100.0,  // 100 K
            1e-20,  // 1e-20 kg/m³
            0.01,   // 1% metallicity
        );
        
        assert!(cooling >= 0.0);
        assert!(cooling.is_finite());
    }

    #[test]
    fn test_temperature_update() {
        let solver = RadiativeTransferSolver::new();
        let constants = PhysicsConstants::default();
        
        let mut temperature = 1000.0;
        let initial_temp = temperature;
        
        solver.update_temperature(
            &mut temperature,
            1e-20,  // density
            0.01,   // metallicity
            3.828e26, // solar luminosity
            1e16,   // 1 pc distance
            1e6,    // 1 Myr timestep
            &constants
        ).unwrap();
        
        // Temperature should change (either heating or cooling)
        assert!(temperature != initial_temp);
        assert!(temperature > 0.0);
    }

    #[test]
    fn test_jeans_mass_with_cooling() {
        let solver = RadiativeTransferSolver::new();
        let constants = PhysicsConstants::default();
        
        let jeans_mass = solver.calculate_jeans_mass_with_cooling(
            100.0,  // 100 K
            1e-20,  // 1e-20 kg/m³
            0.01,   // 1% metallicity
            &constants
        );
        
        assert!(jeans_mass > 0.0);
        assert!(jeans_mass.is_finite());
    }

    #[test]
    fn test_cooling_time() {
        let solver = RadiativeTransferSolver::new();
        let constants = PhysicsConstants::default();
        
        let cooling_time = solver.calculate_cooling_time(
            1000.0, // 1000 K
            1e-20,  // 1e-20 kg/m³
            0.01,   // 1% metallicity
            &constants
        );
        
        println!("Cooling time: {}", cooling_time);
        // Accept both finite and infinite cooling times, but it should not be NaN
        assert!(!cooling_time.is_nan());
    }

    #[test]
    fn test_comprehensive_radiative_transfer() {
        let solver = RadiativeTransferSolver::new();
        let constants = PhysicsConstants::default();
        
        let radiative_transfer = solver.calculate_radiative_transfer(
            1000.0,     // temperature
            1e-20,      // density
            0.01,       // metallicity
            3.828e26,   // stellar luminosity
            1e16,       // distance to star
            &constants
        );
        
        // All rates should be finite
        assert!(radiative_transfer.optically_thin_cooling.is_finite());
        assert!(radiative_transfer.optically_thick_cooling.is_finite());
        assert!(radiative_transfer.hydrogen_line_cooling.is_finite());
        assert!(radiative_transfer.helium_line_cooling.is_finite());
        assert!(radiative_transfer.dust_cooling.is_finite());
        assert!(radiative_transfer.stellar_heating.is_finite());
        assert!(radiative_transfer.cmb_heating.is_finite());
        assert!(radiative_transfer.net_rate.is_finite());
        
        // Net rate should be reasonable
        assert!(radiative_transfer.net_rate.abs() < 1e10); // Sanity check
    }
} 