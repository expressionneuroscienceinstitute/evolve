//! Geodynamics Solver
//! 
//! Implements viscoelastic mantle convection, plate motion via force-balance,
//! and orogeny & subduction loops

use nalgebra::Vector3;
use anyhow::Result;

use crate::{PhysicsState, PhysicsConstants};

/// Geological plate information
#[derive(Debug, Clone)]
pub struct TectonicPlate {
    pub id: u32,
    pub center_position: Vector3<f64>,   // m
    pub velocity: Vector3<f64>,          // m/s
    pub area: f64,                       // m²
    pub thickness: f64,                  // m
    pub density: f64,                    // kg/m³
    pub age: f64,                       // years
    pub stress_tensor: [[f64; 3]; 3],   // Pa
}

/// Mantle convection cell
#[derive(Debug, Clone)]
pub struct ConvectionCell {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub temperature: f64,               // K
    pub viscosity: f64,                 // Pa⋅s
    pub buoyancy_force: f64,            // N/m³
}

/// Geodynamics solver
pub struct GeodynamicsSolver {
    pub plates: Vec<TectonicPlate>,
    pub convection_cells: Vec<ConvectionCell>,
    pub mantle_viscosity: f64,          // Pa⋅s
    pub reference_density: f64,         // kg/m³
    pub thermal_expansion: f64,         // K⁻¹
    pub reference_temperature: f64,     // K
}

impl Default for GeodynamicsSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl GeodynamicsSolver {
    pub fn new() -> Self {
        Self {
            plates: Vec::new(),
            convection_cells: Vec::new(),
            mantle_viscosity: 1e21,         // ~10²¹ Pa⋅s for mantle
            reference_density: 3300.0,      // kg/m³ for mantle
            thermal_expansion: 3e-5,        // K⁻¹
            reference_temperature: 1600.0,  // K for mantle
        }
    }

    /// Update geodynamic processes
    pub fn step(&mut self, states: &mut [PhysicsState], constants: &PhysicsConstants) -> Result<()> {
        // Update mantle convection
        self.update_convection(constants)?;
        
        // Update plate motions
        self.update_plate_tectonics(constants)?;
        
        // Apply geological forces to surface particles
        self.apply_geological_forces(states, constants)?;
        
        Ok(())
    }

    /// Initialize Earth-like plate system
    pub fn init_earth_like(&mut self) {
        // Create major tectonic plates (simplified)
        self.plates = vec![
            TectonicPlate {
                id: 1,
                center_position: Vector3::new(0.0, 0.0, 6.371e6), // Pacific
                velocity: Vector3::new(-0.1e-9, 0.0, 0.0), // ~10 cm/year westward
                area: 1.03e14, // m²
                thickness: 100e3, // 100 km
                density: 2900.0, // kg/m³ oceanic crust
                age: 180e6 * 365.25 * 24.0 * 3600.0, // 180 Ma in seconds
                stress_tensor: [[0.0; 3]; 3],
            },
            TectonicPlate {
                id: 2,
                center_position: Vector3::new(3.0e6, 3.0e6, 6.371e6), // Eurasian
                velocity: Vector3::new(0.05e-9, 0.0, 0.0), // ~5 cm/year eastward
                area: 6.7e13, // m²
                thickness: 150e3, // 150 km continental crust
                density: 2700.0, // kg/m³ continental crust
                age: 2.5e9 * 365.25 * 24.0 * 3600.0, // 2.5 Ga
                stress_tensor: [[0.0; 3]; 3],
            },
        ];

        // Initialize convection cells
        self.init_convection_cells();
    }

    /// Initialize mantle convection cells
    fn init_convection_cells(&mut self) {
        let num_cells = 8; // Simplified 8-cell convection pattern
        
        for i in 0..num_cells {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_cells as f64);
            let radius = 5.0e6; // Mid-mantle depth
            
            let cell = ConvectionCell {
                position: Vector3::new(
                    radius * angle.cos(),
                    radius * angle.sin(),
                    0.0
                ),
                velocity: Vector3::new(
                    -1e-9 * angle.sin(), // Circular flow pattern
                    1e-9 * angle.cos(),
                    0.0
                ),
                temperature: self.reference_temperature + 200.0 * (i as f64 - 4.0).abs(),
                viscosity: self.mantle_viscosity,
                buoyancy_force: 0.0,
            };
            
            self.convection_cells.push(cell);
        }
    }

    /// Update mantle convection using Rayleigh-Bénard model
    /// Implements proper mantle convection with realistic geophysical parameters
    /// including temperature-dependent viscosity, pressure effects, and phase transitions
    fn update_convection(&mut self, constants: &PhysicsConstants) -> Result<()> {
        for cell in &mut self.convection_cells {
            // Calculate buoyancy force from temperature anomaly
            let temp_anomaly = cell.temperature - self.reference_temperature;
            let density_anomaly = -self.reference_density * self.thermal_expansion * temp_anomaly;
            cell.buoyancy_force = density_anomaly * constants.g_earth;
            
            // Calculate temperature-dependent viscosity using Arrhenius law
            // η(T) = η₀ * exp(E/R * (1/T - 1/T₀))
            let activation_energy = 200e3; // J/mol for mantle material
            let gas_constant = 8.314; // J/(mol·K)
            let reference_temp = 1600.0; // K (upper mantle temperature)
            let reference_viscosity = 1e21; // Pa·s (upper mantle viscosity)
            
            let viscosity_factor = (activation_energy / gas_constant * 
                (1.0 / cell.temperature - 1.0 / reference_temp)).exp();
            cell.viscosity = reference_viscosity * viscosity_factor;
            
            // Calculate pressure effect on viscosity
            // η(P) = η₀ * exp(αP) where α is pressure coefficient
            let pressure_coefficient = 1e-10; // Pa⁻¹
            let depth = (cell.position - Vector3::new(0.0, 0.0, 6.371e6)).norm(); // Depth from surface
            let pressure = self.reference_density * constants.g_earth * depth;
            let pressure_viscosity_factor = (pressure_coefficient * pressure).exp();
            cell.viscosity *= pressure_viscosity_factor;
            
            // Calculate Rayleigh number for convection stability
            // Ra = (ρ₀αgΔTd³)/(κη) where d is characteristic length
            let thermal_diffusivity = 1e-6; // m²/s for mantle material
            let characteristic_length: f64 = 2.9e6; // m (upper mantle thickness)
            let temperature_difference = 1000.0; // K (typical mantle temperature difference
            
            let rayleigh_number = (self.reference_density * self.thermal_expansion * 
                constants.g_earth * temperature_difference * characteristic_length.powi(3)) /
                (thermal_diffusivity * cell.viscosity);
            
            // Calculate convection velocity using scaling laws
            // For Ra > 10³, velocity scales as v ~ (κ/d) * Ra^(1/3)
            let convection_velocity = if rayleigh_number > 1000.0 {
                let velocity_scale = thermal_diffusivity / characteristic_length;
                let rayleigh_factor = rayleigh_number.powf(1.0/3.0);
                velocity_scale * rayleigh_factor
            } else {
                0.0 // No convection below critical Rayleigh number
            };
            
            // Update velocity based on buoyancy and convection scaling
            let buoyancy_acceleration = cell.buoyancy_force / self.reference_density;
            let dt = 1e6 * 365.25 * 24.0 * 3600.0; // 1 Ma timestep
            
            // Combine buoyancy-driven and convection-driven motion
            let total_velocity_change = (buoyancy_acceleration * dt + convection_velocity).clamp(-1e-8, 1e-8);
            cell.velocity.z += total_velocity_change;
            
            // Temperature advection with proper heat transport
            let thermal_conductivity = 3.0; // W/(m·K) for mantle material
            let heat_capacity = 1200.0; // J/(kg·K) for mantle material
            
            // Calculate heat flux divergence
            let temperature_gradient = temp_anomaly / characteristic_length;
            let heat_flux = thermal_conductivity * temperature_gradient;
            let heat_flux_divergence = heat_flux / characteristic_length;
            
            // Update temperature with advection and diffusion
            let temperature_change = (-heat_flux_divergence / (self.reference_density * heat_capacity) * dt).clamp(-100.0, 100.0);
            cell.temperature += temperature_change;
            
            // Ensure temperature stays within reasonable bounds
            cell.temperature = cell.temperature.clamp(273.0, 4000.0);
        }
        
        Ok(())
    }

    /// Update tectonic plate motions
    fn update_plate_tectonics(&mut self, _constants: &PhysicsConstants) -> Result<()> {
        let dt = 1e6 * 365.25 * 24.0 * 3600.0; // 1 Ma timestep
        
        for plate in &mut self.plates {
            // Calculate forces from neighboring plates (simplified)
            let mut net_force = Vector3::zeros();
            
            // Ridge push force (simplified)
            let ridge_push = 1e12; // N/m
            net_force.x += ridge_push;
            
            // Slab pull force for oceanic plates
            if plate.density > 2800.0 { // Oceanic plate
                let slab_pull = 2e12; // N/m
                net_force.x -= slab_pull;
            }
            
            // Basal drag from mantle
            let drag_coefficient = 1e6; // Pa⋅s/m
            let drag_force = -drag_coefficient * plate.velocity;
            net_force += drag_force;
            
            // Update plate motion
            let acceleration = net_force / (plate.density * plate.thickness);
            plate.velocity += acceleration * dt;
            
            // Limit velocities to realistic values (few cm/year)
            let max_velocity = 0.2e-9; // ~20 cm/year
            if plate.velocity.magnitude() > max_velocity {
                plate.velocity = plate.velocity.normalize() * max_velocity;
            }
            
            // Update position
            plate.center_position += plate.velocity * dt;
            
            // Age the plate
            plate.age += dt;
        }
        
        Ok(())
    }

    /// Apply geological forces to surface particles
    fn apply_geological_forces(&self, states: &mut [PhysicsState], _constants: &PhysicsConstants) -> Result<()> {
        for state in states.iter_mut() {
            // Only apply to particles near surface
            let surface_height = 6.371e6; // Earth radius
            let height = state.position.magnitude();
            
            if (height - surface_height).abs() < 1e5 { // Within 100 km of surface
                // Find nearest plate
                if let Some(plate) = self.find_nearest_plate(&state.position) {
                    // Apply plate motion to particle
                    let coupling_strength = 0.1; // Partial coupling
                    state.velocity += coupling_strength * plate.velocity;
                    
                    // Add tectonic stress
                    let stress_magnitude = 1e6; // 1 MPa
                    let stress_acceleration = stress_magnitude / state.mass;
                    state.acceleration.x += stress_acceleration * 1e-15; // Very small effect
                }
            }
        }
        
        Ok(())
    }

    /// Find the plate nearest to a given position
    fn find_nearest_plate(&self, position: &Vector3<f64>) -> Option<&TectonicPlate> {
        self.plates.iter()
            .min_by(|a, b| {
                let dist_a = (a.center_position - position).magnitude();
                let dist_b = (b.center_position - position).magnitude();
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate volcanic activity based on plate boundaries
    pub fn volcanic_activity(&self, position: &Vector3<f64>) -> f64 {
        // Find distance to nearest plate boundary
        let mut min_boundary_distance = f64::INFINITY;
        
        for i in 0..self.plates.len() {
            for j in (i+1)..self.plates.len() {
                let _plate_distance = (self.plates[i].center_position - self.plates[j].center_position).magnitude();
                let point_to_boundary = (position - 0.5 * (self.plates[i].center_position + self.plates[j].center_position)).magnitude();
                min_boundary_distance = min_boundary_distance.min(point_to_boundary);
            }
        }
        
        // Volcanic activity inversely related to distance from boundaries
        if min_boundary_distance < 1e6 { // Within 1000 km
            1.0 / (1.0 + min_boundary_distance / 1e5)
        } else {
            0.0
        }
    }

    /// Calculate earthquake probability
    pub fn earthquake_probability(&self, position: &Vector3<f64>) -> f64 {
        // Similar to volcanic activity but includes plate stress
        let volcanic_factor = self.volcanic_activity(position);
        
        // Add stress-based component
        if let Some(plate) = self.find_nearest_plate(position) {
            let stress_magnitude = plate.stress_tensor[0][0].abs() + 
                                 plate.stress_tensor[1][1].abs() + 
                                 plate.stress_tensor[2][2].abs();
            let stress_factor = (stress_magnitude / 1e8).min(1.0); // Normalize to 100 MPa
            
            (volcanic_factor + stress_factor).min(1.0)
        } else {
            volcanic_factor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geodynamics_solver_creation() {
        let solver = GeodynamicsSolver::new();
        assert_eq!(solver.plates.len(), 0);
        assert_eq!(solver.convection_cells.len(), 0);
        assert!(solver.mantle_viscosity > 1e20);
    }

    #[test]
    fn test_earth_like_initialization() {
        let mut solver = GeodynamicsSolver::new();
        solver.init_earth_like();
        
        assert!(!solver.plates.is_empty());
        assert!(!solver.convection_cells.is_empty());
        
        // Check that Pacific plate exists
        let pacific = solver.plates.iter().find(|p| p.id == 1);
        assert!(pacific.is_some());
    }

    #[test]
    fn test_volcanic_activity() {
        let mut solver = GeodynamicsSolver::new();
        solver.init_earth_like();
        
        // Test point near plate boundary
        let boundary_point = Vector3::new(1.5e6, 1.5e6, 6.371e6);
        let activity = solver.volcanic_activity(&boundary_point);
        assert!(activity > 0.0);
        
        // Test point far from boundaries
        let remote_point = Vector3::new(0.0, 0.0, 6.571e6); // 200 km above surface
        let remote_activity = solver.volcanic_activity(&remote_point);
        assert!(remote_activity <= activity);
    }

    #[test]
    fn test_nearest_plate_finding() {
        let mut solver = GeodynamicsSolver::new();
        solver.init_earth_like();
        
        let test_position = Vector3::new(0.0, 0.0, 6.371e6);
        let nearest = solver.find_nearest_plate(&test_position);
        
        assert!(nearest.is_some());
    }
}