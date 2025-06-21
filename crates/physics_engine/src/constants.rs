//! Physical Constants (CODATA-2023)
//! 
//! All fundamental constants used in the physics engine with peer-reviewed values

use serde::{Serialize, Deserialize};

/// Collection of fundamental physical constants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConstants {
    // Universal constants
    pub c: f64,              // Speed of light (m/s)
    pub h: f64,              // Planck constant (J⋅s)
    pub hbar: f64,           // Reduced Planck constant (J⋅s)
    pub g: f64,              // Gravitational constant (m³⋅kg⁻¹⋅s⁻²)
    pub k_b: f64,            // Boltzmann constant (J/K)
    pub n_a: f64,            // Avogadro constant (mol⁻¹)
    pub r: f64,              // Gas constant (J⋅mol⁻¹⋅K⁻¹)
    
    // Electromagnetic constants
    pub epsilon_0: f64,      // Vacuum permittivity (F/m)
    pub mu_0: f64,           // Vacuum permeability (H/m)
    pub e: f64,              // Elementary charge (C)
    pub alpha: f64,          // Fine structure constant
    
    // Atomic/nuclear constants
    pub m_e: f64,            // Electron mass (kg)
    pub m_p: f64,            // Proton mass (kg)
    pub m_n: f64,            // Neutron mass (kg)
    pub u: f64,              // Atomic mass unit (kg)
    
    // Astronomical constants
    pub m_sun: f64,          // Solar mass (kg)
    pub r_sun: f64,          // Solar radius (m)
    pub l_sun: f64,          // Solar luminosity (W)
    pub au: f64,             // Astronomical unit (m)
    pub pc: f64,             // Parsec (m)
    
    // Earth reference values
    pub m_earth: f64,        // Earth mass (kg)
    pub r_earth: f64,        // Earth radius (m)
    pub g_earth: f64,        // Earth surface gravity (m/s²)
    
    // Fusion thresholds
    pub fusion_threshold: f64, // Minimum stellar mass for fusion (solar masses)
    pub brown_dwarf_limit: f64, // Brown dwarf mass limit (solar masses)
    
    // Nucleosynthesis
    pub supernova_threshold: f64, // Minimum mass for core collapse (solar masses)
    pub neutron_star_max: f64,    // Chandrasekhar limit approximation (solar masses)
}

impl Default for PhysicsConstants {
    fn default() -> Self {
        Self {
            // Universal constants (CODATA-2023)
            c: 299_792_458.0,
            h: 6.626_070_15e-34,
            hbar: 1.054_571_817e-34,
            g: 6.674_30e-11,
            k_b: 1.380_649e-23,
            n_a: 6.022_140_76e23,
            r: 8.314_462_618,
            
            // Electromagnetic constants
            epsilon_0: 8.854_187_812_8e-12,
            mu_0: 1.256_637_062_12e-6,  // Vacuum permeability (μ₀) is fixed by definition since 2019 SI redefinition
            e: 1.602_176_634e-19,
            alpha: 7.297_352_569_3e-3,
            
            // Particle masses
            m_e: 9.109_383_701_5e-31,
            m_p: 1.672_621_923_69e-27,
            m_n: 1.674_927_498_04e-27,
            u: 1.660_539_066_60e-27,  // Atomic mass unit (CODATA 2022 value)
            
            // Astronomical constants
            m_sun: 1.988_47e30,
            r_sun: 6.957e8,
            l_sun: 3.828e26,
            au: 1.495_978_707e11,
            pc: 3.085_677_581e16,
            
            // Earth reference
            m_earth: 5.972_168e24,
            r_earth: 6.371e6,
            g_earth: 9.806_65,
            
            // Stellar evolution thresholds
            fusion_threshold: 0.08,      // 0.08 solar masses
            brown_dwarf_limit: 0.075,    // 0.075 solar masses
            supernova_threshold: 8.0,    // 8 solar masses
            neutron_star_max: 2.17,      // Tolman-Oppenheimer-Volkoff limit
        }
    }
}

impl PhysicsConstants {
    /// Create constants with custom values (for testing or alternate physics)
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Calculate gravitational force between two masses
    pub fn gravitational_force(&self, m1: f64, m2: f64, r: f64) -> f64 {
        self.g * m1 * m2 / (r * r)
    }
    
    /// Calculate Coulomb force between two charges
    pub fn coulomb_force(&self, q1: f64, q2: f64, r: f64) -> f64 {
        q1 * q2 / (4.0 * std::f64::consts::PI * self.epsilon_0 * r * r)
    }
    
    /// Check if mass is above fusion threshold
    pub fn can_fuse(&self, mass_solar: f64) -> bool {
        mass_solar >= self.fusion_threshold
    }
    
    /// Check if mass will undergo core collapse supernova
    pub fn will_supernova(&self, mass_solar: f64) -> bool {
        mass_solar >= self.supernova_threshold
    }
    
    /// Calculate Schwarzschild radius
    pub fn schwarzschild_radius(&self, mass: f64) -> f64 {
        2.0 * self.g * mass / (self.c * self.c)
    }
    
    /// Calculate escape velocity
    pub fn escape_velocity(&self, mass: f64, radius: f64) -> f64 {
        (2.0 * self.g * mass / radius).sqrt()
    }
    
    /// Calculate thermal velocity for given temperature and mass
    pub fn thermal_velocity(&self, temperature: f64, mass: f64) -> f64 {
        (3.0 * self.k_b * temperature / mass).sqrt()
    }
    
    /// Calculate blackbody temperature from luminosity and radius
    pub fn blackbody_temperature(&self, luminosity: f64, radius: f64) -> f64 {
        let sigma = 2.0 * std::f64::consts::PI.powi(5) * self.k_b.powi(4) 
                    / (15.0 * self.h.powi(3) * self.c.powi(2)); // Stefan-Boltzmann constant
        (luminosity / (4.0 * std::f64::consts::PI * radius * radius * sigma)).powf(0.25)
    }
    
    /// Calculate relativistic factor (gamma)
    pub fn lorentz_factor(&self, velocity: f64) -> f64 {
        let beta = velocity / self.c;
        1.0 / (1.0 - beta * beta).sqrt()
    }
    
    /// Check if velocity requires relativistic correction
    pub fn is_relativistic(&self, velocity: f64) -> bool {
        velocity >= 0.1 * self.c
    }
}

// Stand-alone constant aliases for ergonomics in other modules
// (These are kept in sync with the CODATA values above.)
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;          // m/s
pub const C: f64 = SPEED_OF_LIGHT;                      // Alias for speed of light
pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;      // J⋅s
pub const REDUCED_PLANCK_CONSTANT: f64 = 1.054_571_817e-34; // J⋅s
pub const GRAVITATIONAL_CONSTANT: f64 = 6.674_30e-11;   // m³⋅kg⁻¹⋅s⁻²
pub const G: f64 = GRAVITATIONAL_CONSTANT;              // Alias for gravitational constant
pub const BOLTZMANN: f64 = 1.380_649e-23;               // J/K
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;   // C
pub const FINE_STRUCTURE_CONSTANT: f64 = 7.297_352_569_3e-3;
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12; // F/m

pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;      // kg
pub const MUON_MASS: f64 = 1.883_531_627e-28;           // kg (CODATA 2018)
pub const TAU_MASS: f64 = 3.167_54e-27;                 // kg (CODATA 2018)
pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;      // kg
pub const NEUTRON_MASS: f64 = 1.674_927_498_04e-27;     // kg

/// Speed of light squared for energy calculations
pub const C_SQUARED: f64 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

/// Rydberg constant in m⁻¹ (CODATA-2023)
pub const RYDBERG_CONSTANT: f64 = 1.097_373_156_816_0e7;

/// Rydberg energy in eV (13.605693122994 eV)
pub const RYDBERG_ENERGY: f64 = 13.605693122994;

/// Hartree energy (2 Rydbergs) in Joules (4.3597447222071e-18 J)
/// This is the atomic unit of energy used in quantum chemistry
pub const HARTREE_TO_JOULE: f64 = 4.3597447222071e-18;

/// The unified atomic mass unit (u), in kilograms.
/// Also known as the Dalton (Da).
/// Source: CODATA 2018
pub const ATOMIC_MASS_UNIT: f64 = 1.66053906660e-27; // kg

pub const LIGHT_YEAR_TO_METERS: f64 = 9.461e15;     // meters

// Legacy alias – some modules still reference `E_CHARGE`. Keep for compatibility.
pub const E_CHARGE: f64 = ELEMENTARY_CHARGE;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_fundamental_constants() {
        let constants = PhysicsConstants::default();
        
        // Test speed of light
        assert_relative_eq!(constants.c, 299_792_458.0, epsilon = 1e-10);
        
        // Test that c² = 1/(ε₀μ₀)
        let c_calculated = 1.0 / (constants.epsilon_0 * constants.mu_0).sqrt();
        assert_relative_eq!(constants.c, c_calculated, epsilon = 1e-5);
        
        // Test fine structure constant α = e²/(4πε₀ℏc)
        let alpha_calculated = constants.e * constants.e 
            / (4.0 * std::f64::consts::PI * constants.epsilon_0 * constants.hbar * constants.c);
        assert_relative_eq!(constants.alpha, alpha_calculated, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gravitational_force() {
        let constants = PhysicsConstants::default();
        
        // Test Earth-Moon system
        let earth_moon_force = constants.gravitational_force(
            constants.m_earth, 
            7.342e22, // Moon mass
            3.844e8   // Earth-Moon distance
        );
        
        // Should be approximately 1.98e20 N
        assert!((earth_moon_force - 1.98e20).abs() / 1.98e20 < 0.1);
    }
    
    #[test]
    fn test_stellar_thresholds() {
        let constants = PhysicsConstants::default();
        
        assert!(constants.can_fuse(0.1));  // Above threshold
        assert!(!constants.can_fuse(0.05)); // Below threshold
        
        assert!(constants.will_supernova(10.0)); // Massive star
        assert!(!constants.will_supernova(1.0)); // Sun-like star
    }
    
    #[test]
    fn test_relativistic_calculations() {
        let constants = PhysicsConstants::default();
        
        // Non-relativistic case
        let v_slow = 1000.0; // 1 km/s
        assert!(!constants.is_relativistic(v_slow));
        assert_relative_eq!(constants.lorentz_factor(v_slow), 1.0, epsilon = 1e-10);
        
        // Relativistic case
        let v_fast = 0.5 * constants.c;
        assert!(constants.is_relativistic(v_fast));
        let gamma = constants.lorentz_factor(v_fast);
        assert_relative_eq!(gamma, 1.0 / (0.75_f64).sqrt(), epsilon = 1e-10);
    }
}