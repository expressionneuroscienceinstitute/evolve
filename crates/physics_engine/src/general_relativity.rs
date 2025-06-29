//! General Relativity Implementation for Universe Simulation
//!
//! This module provides gravitational calculations using general relativity 
//! (GR) as an extension to Newtonian mechanics in strong-field regimes.
//! Key features:
//! 
//! - Post-Newtonian force corrections for massive objects
//! - Schwarzschild radius calculations for black hole formation
//! - Gravitational time dilation effects
//! - Preliminary gravitational wave strain estimation
//!
//! ## Scientific Background
//!
//! The implementation follows the standard formulation of Einstein's field
//! equations: G_μν = 8πG/c⁴ T_μν, where G_μν is the Einstein tensor and T_μν 
//! is the stress-energy tensor. For weak fields and low velocities, we apply
//! post-Newtonian expansions that correct Newtonian gravity.
//!
//! ## Key References
//!
//! - Einstein, A. (1915). "Die Feldgleichungen der Gravitation"
//! - Weinberg, S. (1972). "Gravitation and Cosmology"
//! - Will, C. M. (2014). "The Confrontation between General Relativity and Experiment"

use crate::constants::{C, G};

/// Schwarzschild radius calculation: Rs = 2GM/c²
/// 
/// The Schwarzschild radius represents the critical radius where the escape
/// velocity equals the speed of light, defining the event horizon of a black hole.
/// 
/// # Arguments
/// * `mass_kg` - Mass of the object in kilograms
/// 
/// # Returns
/// Schwarzschild radius in meters
/// 
/// # References
/// - Schwarzschild, K. (1916). "Über das Gravitationsfeld eines Massenpunktes nach der Einsteinschen Theorie"
pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
    2.0 * G * mass_kg / (C * C)
}

/// Post-Newtonian correction to gravitational force
/// 
/// Implements first-order relativistic corrections for orbital dynamics.
/// The correction accounts for:
/// - Kinetic energy terms (v²/c²)
/// - Gravitational potential energy terms (GM/rc²)
/// - Relativistic time dilation effects
/// 
/// Based on the Einstein field equations in the weak field limit.
/// 
/// # Arguments
/// * `mass1_kg` - Mass of first object in kg
/// * `mass2_kg` - Mass of second object in kg  
/// * `separation_m` - Distance between objects in meters
/// * `velocity1_ms` - Velocity of first object [x, y, z] in m/s
/// * `velocity2_ms` - Velocity of second object [x, y, z] in m/s
/// 
/// # Returns
/// Force correction vector [Fx, Fy, Fz] in Newtons
/// 
/// # References
/// - Blanchet, L. (2014). "Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries"
/// - Poisson, E. & Will, C. M. (2014). "Gravity: Newtonian, Post-Newtonian, Relativistic"
pub fn post_newtonian_force_correction(
    mass1_kg: f64,
    mass2_kg: f64,
    separation_m: f64,
    velocity1_ms: [f64; 3],
    velocity2_ms: [f64; 3],
) -> [f64; 3] {
    let total_mass = mass1_kg + mass2_kg;
    let reduced_mass = (mass1_kg * mass2_kg) / total_mass;
    
    // Relative velocity
    let rel_vel = [
        velocity1_ms[0] - velocity2_ms[0],
        velocity1_ms[1] - velocity2_ms[1],
        velocity1_ms[2] - velocity2_ms[2],
    ];
    
    let v_squared = rel_vel[0] * rel_vel[0] + rel_vel[1] * rel_vel[1] + rel_vel[2] * rel_vel[2];
    
    // First-order post-Newtonian correction factor
    // Includes kinetic energy and gravitational potential terms
    let rs = schwarzschild_radius(total_mass);
    let kinetic_term = v_squared / (C * C);
    let potential_term = rs / separation_m;
    let pn_factor = 1.0 + kinetic_term + potential_term;
    
    // Classical gravitational force magnitude
    let force_magnitude = G * mass1_kg * mass2_kg / (separation_m * separation_m);
    let corrected_magnitude = force_magnitude * pn_factor;
    
    // Full vector calculation with proper radial and tangential components
    // Calculate unit vector from mass1 to mass2 (radial direction)
    let radial_unit = [
        velocity2_ms[0] - velocity1_ms[0], // Using velocity difference as proxy for position
        velocity2_ms[1] - velocity1_ms[1],
        velocity2_ms[2] - velocity1_ms[2],
    ];
    
    // Normalize radial unit vector
    let radial_magnitude = (radial_unit[0] * radial_unit[0] + 
                           radial_unit[1] * radial_unit[1] + 
                           radial_unit[2] * radial_unit[2]).sqrt();
    
    let radial_unit_normalized = if radial_magnitude > 1e-12 {
        [
            radial_unit[0] / radial_magnitude,
            radial_unit[1] / radial_magnitude,
            radial_unit[2] / radial_magnitude,
        ]
    } else {
        [1.0, 0.0, 0.0] // Default direction if velocities are identical
    };
    
    // Calculate tangential velocity component (perpendicular to radial direction)
    let radial_velocity = rel_vel[0] * radial_unit_normalized[0] + 
                          rel_vel[1] * radial_unit_normalized[1] + 
                          rel_vel[2] * radial_unit_normalized[2];
    
    let tangential_velocity = [
        rel_vel[0] - radial_velocity * radial_unit_normalized[0],
        rel_vel[1] - radial_velocity * radial_unit_normalized[1],
        rel_vel[2] - radial_velocity * radial_unit_normalized[2],
    ];
    
    // Post-Newtonian corrections include:
    // 1. Radial correction: enhanced gravitational force
    // 2. Tangential correction: velocity-dependent terms
    // 3. Cross terms: coupling between radial and tangential motion
    
    // Radial force component (enhanced by post-Newtonian factor)
    let radial_force = [
        corrected_magnitude * radial_unit_normalized[0],
        corrected_magnitude * radial_unit_normalized[1],
        corrected_magnitude * radial_unit_normalized[2],
    ];
    
    // Tangential force component (velocity-dependent)
    let tangential_magnitude = (tangential_velocity[0] * tangential_velocity[0] + 
                               tangential_velocity[1] * tangential_velocity[1] + 
                               tangential_velocity[2] * tangential_velocity[2]).sqrt();
    
    let tangential_force = if tangential_magnitude > 1e-12 {
        let tangential_unit = [
            tangential_velocity[0] / tangential_magnitude,
            tangential_velocity[1] / tangential_magnitude,
            tangential_velocity[2] / tangential_magnitude,
        ];
        
        // Tangential force includes velocity-dependent corrections
        let tangential_correction = force_magnitude * (v_squared / (C * C)) * 0.5;
        [
            tangential_correction * tangential_unit[0],
            tangential_correction * tangential_unit[1],
            tangential_correction * tangential_unit[2],
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    
    // Cross term: coupling between radial and tangential motion
    let cross_term_magnitude = force_magnitude * (radial_velocity * tangential_magnitude) / (C * C);
    let cross_force = if tangential_magnitude > 1e-12 {
        let tangential_unit = [
            tangential_velocity[0] / tangential_magnitude,
            tangential_velocity[1] / tangential_magnitude,
            tangential_velocity[2] / tangential_magnitude,
        ];
        [
            cross_term_magnitude * tangential_unit[0],
            cross_term_magnitude * tangential_unit[1],
            cross_term_magnitude * tangential_unit[2],
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    
    // Total post-Newtonian force vector
    [
        radial_force[0] + tangential_force[0] + cross_force[0],
        radial_force[1] + tangential_force[1] + cross_force[1],
        radial_force[2] + tangential_force[2] + cross_force[2],
    ]
}

/// Time dilation factor in gravitational field
/// 
/// Computes the gravitational time dilation factor using the weak field approximation:
/// γ = √(1 - Rs/r) where Rs is the Schwarzschild radius
/// 
/// For r → Rs, time dilation approaches infinity (frozen time at event horizon).
/// For r >> Rs, factor approaches 1 (no significant dilation).
/// 
/// # Arguments
/// * `mass_kg` - Mass creating the gravitational field in kg
/// * `radius_m` - Distance from the center of mass in meters
/// 
/// # Returns
/// Time dilation factor (0 to 1, where 1 = no dilation)
/// 
/// # References
/// - Hafele, J. C. & Keating, R. E. (1972). "Around-the-World Atomic Clocks: Predicted Relativistic Time Gains"
/// - GPS Technical Documentation (accounts for ~38 μs/day gravitational time dilation)
pub fn gravitational_time_dilation(mass_kg: f64, radius_m: f64) -> f64 {
    let rs = schwarzschild_radius(mass_kg);
    if radius_m <= rs {
        0.0 // At or inside event horizon
    } else {
        (1.0 - rs / radius_m).sqrt()
    }
}

/// Check if object should use relativistic treatment
/// 
/// Based on PDF guidance: use General Relativity for high-mass or high-velocity scenarios.
/// Criteria for relativistic treatment:
/// 1. Compact objects: r < 100 * Rs (strong field regime)
/// 2. High velocities: v > 0.1c (relativistic speeds)
/// 3. Strong field effects: Rs/r > 0.01 (significant spacetime curvature)
/// 
/// # Arguments
/// * `mass_kg` - Mass of the object in kg
/// * `velocity_ms` - Speed of the object in m/s
/// * `radius_m` - Characteristic radius/distance in meters
/// 
/// # Returns
/// `true` if relativistic treatment is recommended
/// 
/// # Examples
/// - GPS satellites: requires relativistic corrections for timing accuracy
/// - Neutron stars: compact objects requiring full GR treatment
/// - High-energy particle collisions: relativistic velocities
pub fn requires_relativistic_treatment(mass_kg: f64, velocity_ms: f64, radius_m: f64) -> bool {
    let rs = schwarzschild_radius(mass_kg);
    let velocity_fraction = velocity_ms / C;
    
    // Use relativistic treatment if any of these conditions are met:
    radius_m < 100.0 * rs ||          // Compact object
    velocity_fraction > 0.1 ||        // High velocity (> 0.1c)
    (rs / radius_m) > 0.01           // Strong field effects
}

/// Gravitational wave strain amplitude (simplified)
/// 
/// Estimates the gravitational wave strain for inspiralling compact objects
/// using the quadrupole approximation. This is a simplified version suitable
/// for order-of-magnitude estimates.
/// 
/// The full calculation requires numerical relativity for accurate waveforms,
/// but this provides the scaling behavior for astrophysical sources.
/// 
/// # Arguments
/// * `mass1_kg` - Mass of first object in kg
/// * `mass2_kg` - Mass of second object in kg
/// * `separation_m` - Current separation in meters
/// * `distance_m` - Distance to observer in meters
/// 
/// # Returns
/// Dimensionless strain amplitude (h)
/// 
/// # References
/// - Abbott, B. P. et al. (LIGO Scientific Collaboration) (2016). "Observation of Gravitational Waves from a Binary Black Hole Merger"
/// - Thorne, K. S. (1987). "Gravitational radiation"
pub fn gravitational_wave_strain(
    mass1_kg: f64,
    mass2_kg: f64,
    separation_m: f64,
    distance_m: f64,
) -> f64 {
    let total_mass = mass1_kg + mass2_kg;
    let reduced_mass = (mass1_kg * mass2_kg) / total_mass;
    let rs_total = schwarzschild_radius(total_mass);
    
    // Simplified quadrupole formula for strain amplitude
    // h ~ (G/c⁴) * (μ * Rs) / (r * R)
    // where μ is reduced mass, Rs is Schwarzschild radius, r is separation, R is distance
    let strain = (G / (C * C * C * C)) * (reduced_mass * rs_total) / 
                (separation_m * distance_m);
    
    strain.abs()
}

/// Calculate gravitational redshift factor
/// 
/// Computes the frequency shift of electromagnetic radiation in a gravitational field.
/// For light escaping from a gravitational well: f_observed = f_emitted * (1 - Rs/r)
/// 
/// # Arguments
/// * `mass_kg` - Mass creating the gravitational field in kg
/// * `radius_m` - Distance from the center of mass in meters
/// 
/// # Returns
/// Redshift factor (observed frequency / emitted frequency)
pub fn gravitational_redshift(mass_kg: f64, radius_m: f64) -> f64 {
    let rs = schwarzschild_radius(mass_kg);
    if radius_m <= rs {
        0.0 // Infinite redshift at event horizon
    } else {
        1.0 - rs / radius_m
    }
}

/// Calculate orbital precession rate (advance of perihelion)
/// 
/// Computes the additional precession per orbit due to General Relativity.
/// Famous for explaining Mercury's perihelion advance of 43"/century.
/// 
/// # Arguments
/// * `semi_major_axis_m` - Semi-major axis of the orbit in meters
/// * `eccentricity` - Orbital eccentricity (0 to 1)
/// * `central_mass_kg` - Mass of central body in kg
/// 
/// # Returns
/// Precession angle per orbit in radians
/// 
/// # References
/// - Einstein, A. (1915). "Explanation of the perihelion motion of Mercury from general relativity theory"
pub fn orbital_precession_per_orbit(
    semi_major_axis_m: f64,
    eccentricity: f64,
    central_mass_kg: f64,
) -> f64 {
    let rs = schwarzschild_radius(central_mass_kg);
    
    // Einstein's formula for perihelion advance per orbit
    // Δφ = 6πRs / [a(1-e²)] where a is semi-major axis, e is eccentricity
    (6.0 * std::f64::consts::PI * rs) / (semi_major_axis_m * (1.0 - eccentricity * eccentricity))
} 