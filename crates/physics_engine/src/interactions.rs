//! Particle interactions and cross-sections (QED, QCD, Weak)

use nalgebra::Vector3;
use rand::Rng;
use std::f64::consts::PI;
use nalgebra::Complex;

use crate::{FundamentalParticle, ParticleType};
use crate::constants::*;

/// Represents a two-body interaction
pub struct Interaction {
    pub particle_indices: (usize, usize),
    pub interaction_type: InteractionType,
    pub cross_section: f64,  // in m²
    pub probability: f64,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            particle_indices: (0, 0),
            interaction_type: InteractionType::ElasticScattering,
            cross_section: 0.0,
            probability: 0.0,
        }
    }
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
pub fn klein_nishina_cross_section_joules(photon_energy: f64, electron_mass_energy: f64) -> f64 {
    let r_e: f64 = 2.8179403227e-15; // classical electron radius (m)
    let _sigma_thomson = 8.0 * std::f64::consts::PI / 3.0 * r_e.powi(2);
    
    let epsilon = photon_energy / electron_mass_energy;
    
    // Klein-Nishina formula
    let term1 = (1.0 + epsilon) / epsilon.powi(3);
    let term2 = (2.0 * epsilon * (1.0 + epsilon)) / (1.0 + 2.0 * epsilon) - (1.0 + 2.0 * epsilon).ln();
    let term3 = (1.0 + 2.0 * epsilon).ln() / (2.0 * epsilon);
    let term4 = (1.0 + 3.0 * epsilon) / (1.0 + 2.0 * epsilon).powi(2);
    
    2.0 * std::f64::consts::PI * r_e.powi(2) * (term1 * term2 + term3 - term4)
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
    let _energy_transfer = initial_photon_energy - final_photon_energy;
    
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
    let _cos_opening = (photon.energy.powi(2) - electron_energy.powi(2) - positron_energy.powi(2)) 
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
    let electron = FundamentalParticle {
        particle_type: ParticleType::Electron,
        position: photon.position,
        momentum: electron_dir * electron_momentum_mag,
        energy: electron_energy,
        mass: ELECTRON_MASS,
        electric_charge: -ELEMENTARY_CHARGE,
        // Spin vector as complex components (helicity approximation, magnitude ħ/2).
        spin: Vector3::new(
            Complex::new(electron_dir.x * 0.5, 0.0),
            Complex::new(electron_dir.y * 0.5, 0.0),
            Complex::new(electron_dir.z * 0.5, 0.0),
        ),
        color_charge: None,
        creation_time: photon.creation_time,
        decay_time: None, // Stable
        quantum_state: Default::default(),
        interaction_history: vec![],
        velocity: electron_dir * (electron_ke * 2.0 / ELECTRON_MASS).sqrt(),
    };
    
    let positron = FundamentalParticle {
        particle_type: ParticleType::Positron,
        position: photon.position,
        momentum: positron_dir * positron_momentum_mag,
        energy: positron_energy,
        mass: ELECTRON_MASS,
        electric_charge: ELEMENTARY_CHARGE,
        spin: Vector3::new(
            Complex::new(positron_dir.x * 0.5, 0.0),
            Complex::new(positron_dir.y * 0.5, 0.0),
            Complex::new(positron_dir.z * 0.5, 0.0),
        ),
        color_charge: None,
        creation_time: photon.creation_time,
        decay_time: None, // Stable
        quantum_state: Default::default(),
        interaction_history: vec![],
        velocity: positron_dir * (positron_ke * 2.0 / ELECTRON_MASS).sqrt(),
    };
    
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
    // Neutron at rest in lab frame
    let _neutron_momentum: Vector3<f64> = Vector3::zeros();
    let q_value = neutron_mass_gev - proton_mass_gev - electron_mass_gev;

    // Sample electron energy from the allowed spectrum
    let electron_energy = sample_electron_energy(q_value, rng);
    let electron_momentum_mag =
        (electron_energy.powi(2) - electron_mass_gev.powi(2)).sqrt();

    // The rest of the energy goes to the antineutrino
    let neutrino_energy = q_value - electron_energy;

    // Electron and neutrino are emitted isotropically in the rest frame
    let electron_dir = Vector3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    )
    .normalize();

    // Recoil proton momentum balances electron and neutrino
    let neutrino_dir = -electron_dir;

    // A more accurate model would account for angular correlations
    // (the a_eν coefficient), but this is a good start.
    let electron_momentum = electron_dir * electron_momentum_mag;
    let neutrino_momentum = neutrino_dir * neutrino_energy; // E ≈ pc for neutrino
    let proton_momentum = -(electron_momentum + neutrino_momentum);

    (proton_momentum, electron_momentum, neutrino_momentum)
}

/// Calculate the lifetime of a free neutron
pub fn neutron_lifetime() -> f64 { 1.0 / neutron_beta_width() }

/// Sample electron energy from beta decay spectrum
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
    let _s = 2.0 * m_e * e_nu_gev; // Mandelstam s
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
    _nu_momentum: Vector3<f64>,
    electron_momentum: Vector3<f64>,
    rng: &mut impl Rng,
) -> (Vector3<f64>, Vector3<f64>) {
    let s_hat = 2.0 * electron_momentum.norm() * nu_energy_gev;
    let _cos_theta = sample_nu_e_scattering_angle(nu_energy_gev, s_hat, &electron_momentum, rng);
    
    // Final state kinematics
    let final_electron_momentum = Vector3::new(1.0, 0.0, 0.0);
    let final_nu_momentum = Vector3::new(-1.0, 0.0, 0.0);

    (final_electron_momentum, final_nu_momentum)
}

/// Sample the scattering angle for neutrino-electron scattering
fn sample_nu_e_scattering_angle(nu_energy_gev: f64, s_hat: f64, electron_momentum: &Vector3<f64>, rng: &mut impl Rng) -> f64 {
    // Sample cos(theta) uniformly
    let cos_theta = 2.0 * rng.gen::<f64>() - 1.0;
    
    // Calculate scattering angle
    let theta = (s_hat * (1.0 - cos_theta * cos_theta)).sqrt();
    
    // Calculate scattering angle
    let cos_theta_final = (s_hat - theta) / (2.0 * electron_momentum.norm() * nu_energy_gev);
    
    cos_theta_final
}

/// Bethe–Heitler pair production total cross section (m²)
/// energy_gev – photon energy, z – nuclear charge
pub fn bethe_heitler_pair_production(energy_gev: f64, z: u32) -> f64 {
    if energy_gev < 0.001022 { return 0.0; }
    let alpha = 1.0/137.035999;
    let r_e = 2.8179403227e-15;
    let z_f = z as f64;
    let factor = (7.0/9.0)*(alpha*r_e*r_e)*z_f.powi(2);
    let l = (2.0*energy_gev/0.000511).ln();
    factor * (l - 1.0)
}

/// Klein–Nishina total cross section for Compton scattering (m²)
/// energy_gev – photon energy in GeV
pub fn klein_nishina_cross_section(energy_gev: f64) -> f64 {
    let r_e: f64 = 2.8179403227e-15; // m
    let _sigma_thomson = 8.0 * std::f64::consts::PI / 3.0 * r_e.powi(2);

    let epsilon = energy_gev / 0.000511; // epsilon = E / (m_e c^2)  (m_e c^2 = 0.511 MeV)

    // Avoid division by zero for very low energy
    if epsilon == 0.0 {
        return _sigma_thomson;
    }

    let term1 = (1.0 + epsilon) / epsilon.powi(2) * ((2.0 * (1.0 + epsilon)) / (1.0 + 2.0 * epsilon) - (1.0 / epsilon) * epsilon.ln());
    let term2 = (1.0 / (2.0 * epsilon)) * ((1.0 + 3.0 * epsilon) / (1.0 + 2.0 * epsilon).powi(2));

    _sigma_thomson * (term1 + term2)
}

pub struct InteractionResult {
    pub particles: Vec<ParticleType>,
    pub p_momentum: Vector3<f64>,
    pub e_momentum: Vector3<f64>,
    _nu_momentum: Vector3<f64>,
}