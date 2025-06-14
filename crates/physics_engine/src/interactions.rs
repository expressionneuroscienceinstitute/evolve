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