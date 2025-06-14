//! Particle interactions and cross-sections (QED, QCD, Weak)

use nalgebra::Vector3;
use rand::Rng;
use std::f64::consts::PI;

use crate::{FundamentalParticle, ParticleType};
use crate::constants::*;

/// Represents a two-body interaction
pub struct Interaction {
    pub particle_indices: (usize, usize),
    pub interaction_type: InteractionType,
    pub cross_section: f64,  // in m²
    pub probability: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum InteractionType {
    ComptonScattering,
    PairProduction,
    Annihilation,
    ElasticScattering,
    Decay,
}

/// Klein-Nishina cross-section for Compton scattering
/// PDG 2024 Eq. 34.27
/// 
/// # Arguments
/// * `photon_energy` - Initial photon energy in Joules
/// * `electron_mass_energy` - Electron rest mass energy (m_e c²) in Joules
/// 
/// # Returns
/// Total cross-section in m²
pub fn klein_nishina_cross_section(photon_energy: f64, electron_mass_energy: f64) -> f64 {
    let epsilon = photon_energy / electron_mass_energy;
    
    // Classical electron radius
    let r_e = ELEMENTARY_CHARGE.powi(2) / (4.0 * PI * 8.854187817e-12 * electron_mass_energy);
    let r_e_squared = r_e * r_e;
    
    // Klein-Nishina formula
    let term1 = (1.0 + epsilon) / epsilon.powi(3);
    let term2 = (2.0 * epsilon * (1.0 + epsilon)) / (1.0 + 2.0 * epsilon) - (1.0 + 2.0 * epsilon).ln();
    let term3 = (1.0 + 2.0 * epsilon).ln() / (2.0 * epsilon);
    let term4 = (1.0 + 3.0 * epsilon) / (1.0 + 2.0 * epsilon).powi(2);
    
    2.0 * PI * r_e_squared * (term1 * term2 + term3 - term4)
}

/// Bethe-Heitler cross-section for pair production in nuclear field
/// PDG 2024 Section 34.5
/// 
/// # Arguments
/// * `photon_energy` - Photon energy in Joules
/// * `z` - Atomic number of nucleus
/// 
/// # Returns
/// Cross-section in m²
pub fn bethe_heitler_cross_section(photon_energy: f64, z: u32) -> f64 {
    let electron_mass_energy = ELECTRON_MASS * SPEED_OF_LIGHT.powi(2);
    
    // Threshold: need at least 2 electron masses
    if photon_energy < 2.0 * electron_mass_energy {
        return 0.0;
    }
    
    // Classical electron radius
    let r_e = ELEMENTARY_CHARGE.powi(2) / (4.0 * PI * 8.854187817e-12 * electron_mass_energy);
    
    // For complete screening (high energy), PDG Eq. 34.31
    let z_squared = (z as f64).powi(2);
    let alpha = FINE_STRUCTURE_CONSTANT;
    
    // Coulomb correction factor
    let f_c = alpha.powi(2) * z_squared;
    
    // High-energy asymptotic form (complete screening)
    // This is simplified; full form includes screening functions
    let ln_term = if z == 1 {
        5.81  // Hydrogen
    } else {
        (183.0 / (z as f64).powf(1.0/3.0)).ln()
    };
    
    // Cross-section in the high-energy limit
    4.0 * alpha * r_e.powi(2) * z_squared * (ln_term - f_c)
}

/// Perform Compton scattering between photon and electron
/// Updates momenta and energies according to kinematics
pub fn scatter_compton(
    photon: &mut FundamentalParticle,
    electron: &mut FundamentalParticle,
    rng: &mut impl Rng,
) {
    let electron_mass_energy = ELECTRON_MASS * SPEED_OF_LIGHT.powi(2);
    let initial_photon_energy = photon.energy;
    
    // Dimensionless photon energy
    let epsilon = initial_photon_energy / electron_mass_energy;
    
    // Sample scattering angle using rejection method
    // The differential cross-section is given by Klein-Nishina formula
    let cos_theta = sample_compton_angle(epsilon, rng);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    
    // Azimuthal angle is uniform
    let phi = rng.gen::<f64>() * 2.0 * PI;
    
    // Final photon energy from Compton formula
    let final_photon_energy = initial_photon_energy / (1.0 + epsilon * (1.0 - cos_theta));
    
    // Energy transferred to electron
    let energy_transfer = initial_photon_energy - final_photon_energy;
    
    // Update photon
    photon.energy = final_photon_energy;
    
    // Calculate new photon direction
    let initial_direction = photon.momentum.normalize();
    let new_direction = rotate_vector(initial_direction, cos_theta, sin_theta, phi);
    photon.momentum = new_direction * (final_photon_energy / SPEED_OF_LIGHT);
    
    // Update electron (initially at rest in lab frame)
    let electron_momentum = initial_direction * (initial_photon_energy / SPEED_OF_LIGHT) 
        - photon.momentum;
    electron.momentum += electron_momentum;
    electron.energy = (electron_mass_energy.powi(2) 
        + (electron_momentum.norm() * SPEED_OF_LIGHT).powi(2)).sqrt();
}

/// Sample Compton scattering angle using rejection method
fn sample_compton_angle(epsilon: f64, rng: &mut impl Rng) -> f64 {
    // Use rejection sampling with the Klein-Nishina differential cross-section
    loop {
        // Sample cos(theta) uniformly
        let cos_theta = 2.0 * rng.gen::<f64>() - 1.0;
        
        // Compton formula for energy ratio
        let x = 1.0 / (1.0 + epsilon * (1.0 - cos_theta));
        
        // Klein-Nishina angular distribution (normalized)
        let kn_value = x * (x + 1.0/x - 1.0 + cos_theta.powi(2));
        
        // Maximum value for rejection sampling
        let kn_max = 2.0 + 8.0 / (2.0 + epsilon);
        
        if rng.gen::<f64>() * kn_max < kn_value {
            return cos_theta;
        }
    }
}

/// Rotate a vector by given angles
fn rotate_vector(v: Vector3<f64>, cos_theta: f64, sin_theta: f64, phi: f64) -> Vector3<f64> {
    // Create rotation axis perpendicular to v
    let axis = if v.z.abs() < 0.9 {
        Vector3::new(0.0, 0.0, 1.0).cross(&v).normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0).cross(&v).normalize()
    };
    
    // Rodrigues' rotation formula
    let k = axis;
    let v_rot = v * cos_theta + k.cross(&v) * sin_theta + k * k.dot(&v) * (1.0 - cos_theta);
    
    // Additional rotation around original vector by phi
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    let v_perp = v.normalize().cross(&v_rot);
    
    v_rot * cos_phi + v_perp * sin_phi
}

/// Create electron-positron pair from photon
/// Returns (electron, positron) if successful, None if below threshold
pub fn pair_produce(
    photon: &FundamentalParticle,
    rng: &mut impl Rng,
) -> Option<(FundamentalParticle, FundamentalParticle)> {
    let electron_mass_energy = ELECTRON_MASS * SPEED_OF_LIGHT.powi(2);
    
    // Check threshold
    if photon.energy < 2.0 * electron_mass_energy {
        return None;
    }
    
    // Available kinetic energy
    let kinetic_energy = photon.energy - 2.0 * electron_mass_energy;
    
    // Energy sharing between electron and positron
    // For simplicity, use symmetric sharing with small asymmetry
    let asymmetry = 0.1 * (2.0 * rng.gen::<f64>() - 1.0);
    let electron_ke = kinetic_energy * (0.5 + asymmetry);
    let positron_ke = kinetic_energy * (0.5 - asymmetry);
    
    // Total energies
    let electron_energy = electron_mass_energy + electron_ke;
    let positron_energy = electron_mass_energy + positron_ke;
    
    // Momenta (back-to-back in CM frame, then boost to lab)
    let electron_momentum_mag = (electron_energy.powi(2) - electron_mass_energy.powi(2)).sqrt() / SPEED_OF_LIGHT;
    let positron_momentum_mag = (positron_energy.powi(2) - electron_mass_energy.powi(2)).sqrt() / SPEED_OF_LIGHT;
    
    // Opening angle from momentum conservation
    let cos_opening = (photon.energy.powi(2) - electron_energy.powi(2) - positron_energy.powi(2)) 
        / (2.0 * electron_momentum_mag * positron_momentum_mag * SPEED_OF_LIGHT.powi(2));
    
    // Generate random direction for electron
    let theta = rng.gen::<f64>() * PI;
    let phi = rng.gen::<f64>() * 2.0 * PI;
    
    let electron_dir = Vector3::new(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    );
    
    // Positron direction from momentum conservation
    let photon_dir = photon.momentum.normalize();
    let positron_dir = (photon_dir * (photon.energy / SPEED_OF_LIGHT) 
        - electron_dir * electron_momentum_mag).normalize();
    
    // Create particles
    let mut electron = FundamentalParticle {
        particle_type: ParticleType::Electron,
        position: photon.position,
        momentum: electron_dir * electron_momentum_mag,
        energy: electron_energy,
        mass: ELECTRON_MASS,
        electric_charge: -ELEMENTARY_CHARGE,
        spin: Vector3::zeros(), // TODO: proper spin
        color_charge: None,
        creation_time: photon.creation_time, // Will be updated by engine
        decay_time: None,
        quantum_state: crate::QuantumState::new(),
        interaction_history: vec![],
    };
    
    let mut positron = electron.clone();
    positron.particle_type = ParticleType::Positron;
    positron.momentum = positron_dir * positron_momentum_mag;
    positron.energy = positron_energy;
    positron.electric_charge = ELEMENTARY_CHARGE;
    
    Some((electron, positron))
}

/// Check if two particles can interact via Compton scattering
pub fn can_compton_scatter(p1: &FundamentalParticle, p2: &FundamentalParticle) -> bool {
    (p1.particle_type == ParticleType::Photon && p2.particle_type == ParticleType::Electron) ||
    (p2.particle_type == ParticleType::Photon && p1.particle_type == ParticleType::Electron)
}

/// Calculate interaction probability for given cross-section and conditions
pub fn interaction_probability(
    cross_section: f64,
    number_density: f64,
    relative_velocity: f64,
    time_step: f64,
) -> f64 {
    // P = 1 - exp(-σ n v Δt)
    let rate = cross_section * number_density * relative_velocity;
    1.0 - (-rate * time_step).exp()
}

// ========================= Weak Interactions =============================

/// Dirac gamma-matrices in chiral representation (only needed for documentation / further work).
pub mod gamma {
    use nalgebra::Matrix4;
    
    /// γ⁰ matrix in chiral representation
    pub fn gamma0() -> Matrix4<f64> { 
        Matrix4::new( 0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0 ) 
    }
    
    /// γⁱ matrices (i = 1,2,3) in chiral representation
    pub fn gamma_vec(i: usize) -> Matrix4<f64> {
        match i {
            1 => Matrix4::new( 0.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0,-1.0, 0.0, 0.0,
                              -1.0, 0.0, 0.0, 0.0 ),
            2 => Matrix4::new( 0.0, 0.0, 0.0,-1.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               1.0, 0.0, 0.0, 0.0 ),
            3 => Matrix4::new( 0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0,-1.0,
                              -1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0 ),
            _ => Matrix4::zeros()
        }
    }
    
    /// γ⁵ = iγ⁰γ¹γ²γ³ matrix
    pub fn gamma5() -> Matrix4<f64> {
        Matrix4::new( 1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0,-1.0, 0.0,
                      0.0, 0.0, 0.0,-1.0 )
    }
}

/// V-A current structure for weak interactions
#[derive(Debug, Clone)]
pub struct WeakCurrent {
    pub vector_part: f64,      // Vector coupling strength
    pub axial_part: f64,       // Axial coupling strength
    pub ckm_element: f64,      // CKM matrix element for quarks
}

impl WeakCurrent {
    /// Charged current for quark transitions (u → d)
    pub fn quark_charged_current() -> Self {
        Self {
            vector_part: 1.0,
            axial_part: G_A,
            ckm_element: V_UD,
        }
    }
    
    /// Leptonic charged current (e → νₑ)
    pub fn lepton_charged_current() -> Self {
        Self {
            vector_part: 1.0,
            axial_part: 1.0,
            ckm_element: 1.0,
        }
    }
    
    /// Neutral current for neutrino-electron scattering
    pub fn neutral_current(flavour: u8, is_antineutrino: bool) -> Self {
        // Vector and axial couplings for neutral current
        let g_v = if flavour == 0 { // electron neutrino
            0.5 + 2.0 * SIN2_THETA_W
        } else { // muon/tau neutrino
            -0.5 + 2.0 * SIN2_THETA_W
        };
        let g_a = 0.5;
        
        let (v_eff, a_eff) = if is_antineutrino {
            (-g_v, -g_a)
        } else {
            (g_v, g_a)
        };
        
        Self {
            vector_part: v_eff,
            axial_part: a_eff,
            ckm_element: 1.0,
        }
    }
}

pub const G_F_GEV2: f64 = 1.1663787e-5;
pub const SIN2_THETA_W: f64 = 0.23122;
pub const V_UD: f64 = 0.97420;
pub const G_A: f64 = 1.2723;

// Beta-decay phase-space factor for neutron (unitless). PDG value ~1.6887
pub const PHASE_SPACE_N: f64 = 1.6887;

/// Beta-decay total width Γ_n (s⁻¹) using Fermi's golden rule
pub fn neutron_beta_width() -> f64 {
    // Q-value for neutron decay (MeV)
    let q_mev = 0.782; // n - p mass difference
    let m_e_mev = 0.511;
    
    // Convert to GeV
    let q_gev = q_mev / 1000.0;
    let m_e = m_e_mev / 1000.0;
    
    // Fermi's golden rule: Γ = (G_F² |V_ud|² / 2π³) (1 + 3g_A²) ∫ phase_space
    let coupling_factor = G_F_GEV2.powi(2) * V_UD.powi(2) * (1.0 + 3.0 * G_A.powi(2));
    let prefactor = coupling_factor / (2.0 * std::f64::consts::PI.powi(3));
    
    // Phase space integral: ∫ p_e E_e (E_max - E_e)² F(Z,E_e) dE_e
    let phase_space = calculate_phase_space_integral(q_gev, m_e);
    
    // Convert GeV⁵ to s⁻¹ using ħc
    let hbar_c_gev_m = 1.973269804e-16; // ħc in GeV·m
    let conversion = 1.0 / (hbar_c_gev_m * 1e-15); // GeV to s⁻¹
    
    prefactor * phase_space * conversion
}

/// Calculate phase space integral for beta decay
fn calculate_phase_space_integral(q_gev: f64, m_e_gev: f64) -> f64 {
    let e_max = q_gev + m_e_gev;
    let mut integral = 0.0;
    let n_points = 1000;
    let de = (e_max - m_e_gev) / n_points as f64;
    
    // Numerical integration using trapezoidal rule
    for i in 0..n_points {
        let e_e = m_e_gev + (i as f64 + 0.5) * de;
        let p_e = (e_e * e_e - m_e_gev * m_e_gev).sqrt();
        
        // Fermi function F(Z=1, E_e) ≈ 1 for low Z
        let fermi_function = 1.0;
        
        let integrand = p_e * e_e * (e_max - e_e).powi(2) * fermi_function;
        integral += integrand * de;
    }
    
    integral
}

/// Sample three-body beta decay kinematics: n → p + e⁻ + ν̄ₑ
pub fn sample_beta_decay_kinematics(
    neutron_mass_gev: f64,
    proton_mass_gev: f64,
    electron_mass_gev: f64,
    rng: &mut impl Rng,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    // Q-value
    let q_value = neutron_mass_gev - proton_mass_gev;
    
    // Sample electron energy from beta spectrum
    let e_electron = sample_electron_energy(q_value - electron_mass_gev, rng);
    let p_electron_mag = (e_electron * e_electron - electron_mass_gev * electron_mass_gev).sqrt();
    
    // Remaining energy for neutrino
    let e_neutrino = q_value + electron_mass_gev - e_electron;
    let p_neutrino_mag = e_neutrino; // massless neutrino
    
    // Sample isotropic directions
    let theta_e = rng.gen::<f64>() * std::f64::consts::PI;
    let phi_e = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
    
    let theta_nu = rng.gen::<f64>() * std::f64::consts::PI;
    let phi_nu = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
    
    // Electron momentum
    let p_electron = Vector3::new(
        p_electron_mag * theta_e.sin() * phi_e.cos(),
        p_electron_mag * theta_e.sin() * phi_e.sin(),
        p_electron_mag * theta_e.cos(),
    );
    
    // Neutrino momentum
    let p_neutrino = Vector3::new(
        p_neutrino_mag * theta_nu.sin() * phi_nu.cos(),
        p_neutrino_mag * theta_nu.sin() * phi_nu.sin(),
        p_neutrino_mag * theta_nu.cos(),
    );
    
    // Proton momentum from conservation
    let p_proton = -(p_electron + p_neutrino);
    
    (p_proton, p_electron, p_neutrino)
}

pub fn neutron_lifetime() -> f64 { 1.0 / neutron_beta_width() }

/// Sample electron energy (in GeV) from allowed β spectrum
pub fn sample_electron_energy(q_value_gev: f64, rng: &mut impl Rng) -> f64 {
    // Rejection sampling in [m_e, E_max]
    let m_e = 0.00051099895;
    let e_max = q_value_gev + m_e;
    loop {
        let e_e = rng.gen_range(m_e..e_max);
        let p = (e_e * e_e - m_e * m_e).sqrt();
        let weight = p * e_e * (e_max - e_e).powi(2);
        let max_w = ( (e_max*e_max - m_e*m_e).sqrt() * e_max ).powi(1) ;
        if rng.gen::<f64>() * max_w < weight { return e_e; }
    }
}

/// Complete ν-e elastic scattering cross-section with V-A structure (m²)
pub fn neutrino_e_scattering_complete(flavour: u8, e_nu_gev: f64, is_antineutrino: bool) -> f64 {
    let m_e = 0.000511; // electron mass in GeV
    
    // Get weak current couplings
    let current = WeakCurrent::neutral_current(flavour, is_antineutrino);
    let g_v = current.vector_part;
    let g_a = current.axial_part;
    
    // Kinematic variables
    let s = 2.0 * m_e * e_nu_gev; // Mandelstam s
    let y_max = 1.0; // Maximum inelasticity for elastic scattering
    
    // Differential cross-section integrated over recoil energy
    // dσ/dy = (G_F² m_e E_ν)/(2π) [(g_V + g_A)² + (g_V - g_A)²(1-y)² - (g_V² - g_A²)(m_e/E_ν)y]
    
    let term1 = (g_v + g_a).powi(2);
    let term2 = (g_v - g_a).powi(2) * (1.0_f64 - y_max).powi(2);
    let term3 = (g_v.powi(2) - g_a.powi(2)) * (m_e / e_nu_gev) * y_max;
    
    let differential = (G_F_GEV2.powi(2) * m_e * e_nu_gev) / (2.0 * std::f64::consts::PI) * 
                      (term1 + term2 - term3);
    
    // Integrate over y from 0 to y_max (for elastic scattering, this is simplified)
    let integrated = differential * y_max;
    
    // Convert GeV⁻² to m²
    integrated * 3.8938e-32
}

/// Sample neutrino-electron scattering kinematics
pub fn sample_nu_e_scattering(
    nu_energy_gev: f64,
    nu_momentum: Vector3<f64>,
    electron_momentum: Vector3<f64>,
    rng: &mut impl Rng,
) -> (Vector3<f64>, Vector3<f64>) {
    let m_e = 0.000511; // GeV
    
    // Sample scattering angle in CM frame
    let cos_theta_cm = 2.0 * rng.gen::<f64>() - 1.0;
    let sin_theta_cm = (1.0 - cos_theta_cm * cos_theta_cm).sqrt();
    let phi_cm = 2.0 * std::f64::consts::PI * rng.gen::<f64>();
    
    // For elastic scattering, energy transfer is determined by angle
    let nu_energy_initial = nu_energy_gev;
    let electron_energy_initial = (electron_momentum.norm_squared() + m_e * m_e).sqrt();
    
    // Elastic scattering formula
    let energy_transfer = (2.0 * m_e * nu_energy_initial.powi(2)) / 
                         (2.0 * m_e * nu_energy_initial + m_e.powi(2)) * 
                         (1.0 - cos_theta_cm);
    
    let nu_energy_final = nu_energy_initial - energy_transfer;
    let electron_energy_final = electron_energy_initial + energy_transfer;
    
    // Final momenta magnitudes
    let nu_momentum_mag = nu_energy_final; // massless
    let electron_momentum_mag = (electron_energy_final.powi(2) - m_e.powi(2)).sqrt();
    
    // Scattered neutrino direction
    let nu_dir_initial = nu_momentum.normalize();
    let scattered_nu_dir = rotate_vector_by_angle(nu_dir_initial, cos_theta_cm, sin_theta_cm, phi_cm);
    
    // Scattered electron momentum from conservation
    let nu_momentum_final = scattered_nu_dir * nu_momentum_mag;
    let electron_momentum_final = nu_momentum + electron_momentum - nu_momentum_final;
    
    (nu_momentum_final, electron_momentum_final)
}

/// Rotate vector by given angles (helper function)
fn rotate_vector_by_angle(v: Vector3<f64>, cos_theta: f64, sin_theta: f64, phi: f64) -> Vector3<f64> {
    // Create perpendicular vectors
    let w = if v.z.abs() < 0.9 {
        Vector3::new(0.0, 0.0, 1.0).cross(&v).normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0).cross(&v).normalize()
    };
    let u = v.cross(&w).normalize();
    
    // Rotate in the plane perpendicular to v
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    
    v * cos_theta + (u * cos_phi + w * sin_phi) * sin_theta
}