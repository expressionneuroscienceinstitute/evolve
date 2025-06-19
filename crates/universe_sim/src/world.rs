//! World Representation System
//! 
//! Manages the 2D+Z toroidal grid universe with stratified geological layers,
//! celestial bodies, and planetary systems with resource extraction.

use crate::physics_engine::{ElementTable, EnvironmentProfile, StratumLayer, MaterialType};
use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use uuid::Uuid;
use rand::{Rng, thread_rng};

/// Main world structure containing the universe grid
#[derive(Debug)]
pub struct World {
    pub grid: WorldGrid,
    pub celestial_bodies: HashMap<Uuid, CelestialBodyData>,
    pub star_systems: HashMap<Uuid, StarSystem>,
    pub time_step: f64,
    pub universe_age: f64,
}

/// 2D toroidal grid with Z-layers
#[derive(Debug, Clone)]
pub struct WorldGrid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<WorldCell>>,
    pub scale_per_cell: f64,  // Meters per cell
}

/// Individual cell in the world grid
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldCell {
    pub position: Vector3<f64>,
    pub cell_type: CellType,
    pub temperature: f64,
    pub pressure: f64,
    pub density: f64,
    pub strata: Vec<StratumLayer>,
    pub occupants: Vec<Uuid>,  // Agent IDs present in this cell
    pub celestial_body_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellType {
    Vacuum,
    Gas,
    Plasma,
    Planetary,
    Stellar,
    Nebula,
}

/// Celestial body data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialBodyData {
    pub id: Uuid,
    pub body_type: CelestialBodyType,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub mass: f64,
    pub radius: f64,
    pub temperature: f64,
    pub luminosity: f64,
    pub age: f64,
    pub composition: ElementTable,
    pub planets: Vec<Uuid>,  // For stars
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CelestialBodyType {
    Star { stellar_class: StellarClass },
    Planet { planet_class: PlanetClass },
    Moon,
    Asteroid,
    BlackHole,
    NeutronStar,
    WhiteDwarf,
    BrownDwarf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StellarClass {
    O, B, A, F, G, K, M,  // Main sequence
    WD,  // White dwarf
    NS,  // Neutron star
    BH,  // Black hole
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanetClass {
    E,  // Earth-like
    D,  // Desert
    I,  // Ice
    T,  // Toxic
    G,  // Gas dwarf
}

/// Star system containing multiple bodies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarSystem {
    pub id: Uuid,
    pub primary_star: Uuid,
    pub planets: Vec<Uuid>,
    pub asteroid_belts: Vec<AsteroidBelt>,
    pub system_age: f64,
    pub metallicity: f64,  // Z_star
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsteroidBelt {
    pub orbital_radius: f64,
    pub total_mass: f64,
    pub composition: ElementTable,
}

/// Planetary environment with full habitability model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetaryData {
    pub id: Uuid,
    pub environment: EnvironmentProfile,
    pub stratigraphy: Vec<StratumLayer>,
    pub class: PlanetClass,
    pub orbital_radius: f64,
    pub orbital_period: f64,
    pub rotation_period: f64,
    pub axial_tilt: f64,
    pub magnetic_field: f64,
    pub atmosphere: AtmosphereComposition,
    pub surface_features: SurfaceFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphereComposition {
    pub total_pressure: f64,  // Pascals
    pub gas_fractions: HashMap<String, f64>,  // Gas name -> fraction
    pub greenhouse_effect: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceFeatures {
    pub continents: Vec<Continent>,
    pub oceans: Vec<Ocean>,
    pub ice_caps: Vec<IceCap>,
    pub volcanic_activity: f64,
    pub tectonic_activity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Continent {
    pub area: f64,
    pub elevation: f64,
    pub climate_zones: Vec<ClimateZone>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ocean {
    pub area: f64,
    pub depth: f64,
    pub salinity: f64,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceCap {
    pub area: f64,
    pub thickness: f64,
    pub albedo: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClimateZone {
    Tropical,
    Temperate,
    Arctic,
    Desert,
    Rainforest,
    Grassland,
    Tundra,
}

impl World {
    /// Create a new universe world
    pub fn new(width: usize, height: usize, scale_per_cell: f64) -> Self {
        let grid = WorldGrid::new(width, height, scale_per_cell);
        
        Self {
            grid,
            celestial_bodies: HashMap::new(),
            star_systems: HashMap::new(),
            time_step: 1e-6,
            universe_age: 0.0,
        }
    }
    
    /// Initialize with Big Bang conditions
    pub fn init_big_bang(&mut self) -> Result<()> {
        // Fill universe with primordial gas
        self.fill_primordial_gas()?;
        
        // Set initial temperature and density gradients
        self.set_initial_conditions()?;
        
        Ok(())
    }
    
    /// Fill universe with primordial hydrogen and helium
    fn fill_primordial_gas(&mut self) -> Result<()> {
        let mut rng = thread_rng();
        
        for row in &mut self.grid.cells {
            for cell in row {
                cell.cell_type = CellType::Gas;
                cell.temperature = rng.gen_range(2000.0..4000.0);  // 2-4K initial temp
                cell.density = rng.gen_range(1e-30..1e-28);  // Very low density
                
                // Add primordial gas composition to strata
                let mut primordial_composition = ElementTable::new();
                primordial_composition.set_abundance(1, 750_000);  // 75% H
                primordial_composition.set_abundance(2, 250_000);  // 25% He
                
                let gas_layer = StratumLayer {
                    thickness_m: 1e15,  // Very thick gas layer
                    material_type: MaterialType::Gas,
                    bulk_density: cell.density,
                    elements: primordial_composition,
                };
                
                cell.strata.push(gas_layer);
            }
        }
        
        Ok(())
    }
    
    /// Set initial temperature and density gradients
    fn set_initial_conditions(&mut self) -> Result<()> {
        let center_x = self.grid.width / 2;
        let center_y = self.grid.height / 2;
        
        for (y, row) in self.grid.cells.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                // Distance from center
                let dx = (x as f64 - center_x as f64) / center_x as f64;
                let dy = (y as f64 - center_y as f64) / center_y as f64;
                let r = (dx * dx + dy * dy).sqrt();
                
                // Temperature decreases with distance from center
                cell.temperature *= (1.0 - 0.5 * r).max(0.1);
                
                // Density varies randomly but decreases with distance
                cell.density *= (1.0 - 0.3 * r).max(0.1);
            }
        }
        
        Ok(())
    }
    
    /// Process star formation from gas clouds
    pub fn process_star_formation(&mut self) -> Result<Vec<Uuid>> {
        let mut new_stars = Vec::new();
        let mut rng = thread_rng();
        
        // Collect candidate positions first to avoid borrowing issues
        let mut candidates = Vec::new();
        
        // Look for dense gas regions suitable for star formation
        for (y, row) in self.grid.cells.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                if matches!(cell.cell_type, CellType::Gas) && 
                   cell.density > 1e-24 &&   // Density threshold
                   cell.temperature < 100.0  // Cold gas
                {
                    // Probabilistic star formation
                    if rng.gen::<f64>() < 0.001 {  // 0.1% chance per tick
                        candidates.push((x, y));
                    }
                }
            }
        }
        
        // Now create stars at the candidate positions
        for (x, y) in candidates {
            let star_id = self.create_star(x, y)?;
            new_stars.push(star_id);
        }
        
        Ok(new_stars)
    }
    
    /// Create a new star at the specified grid position
    fn create_star(&mut self, x: usize, y: usize) -> Result<Uuid> {
        let mut rng = thread_rng();
        let star_id = Uuid::new_v4();
        
        let star_mass = rng.gen_range(0.08..50.0); // 0.08 to 50 solar masses
        
        // Calculate realistic stellar properties based on mass
        let stellar_class = self.classify_star_by_mass(star_mass);
        let stellar_radius = self.calculate_stellar_radius(star_mass);
        let stellar_temp = self.calculate_stellar_temperature(star_mass);
        let stellar_luminosity = self.calculate_stellar_luminosity(star_mass);

        let new_star = CelestialBodyData {
            id: star_id,
            body_type: CelestialBodyType::Star { stellar_class },
            position: Vector3::new(x as f64, y as f64, 0.0),
            velocity: Vector3::zeros(),
            mass: star_mass * 1.98847e30, // Convert to kg
            radius: stellar_radius,
            temperature: stellar_temp,
            luminosity: stellar_luminosity,
            age: 0.0,
            composition: self.generate_stellar_composition(star_mass),
            planets: Vec::new(),
        };
        
        self.celestial_bodies.insert(star_id, new_star);
        
        Ok(star_id)
    }
    
    /// Classify star by mass using Harvard stellar classification
    fn classify_star_by_mass(&self, mass_solar: f64) -> StellarClass {
        match mass_solar {
            m if m >= 15.0 => StellarClass::O,  // Blue supergiants (15-90 M☉)
            m if m >= 2.1  => StellarClass::B,  // Blue main sequence (2.1-16 M☉)
            m if m >= 1.4  => StellarClass::A,  // White main sequence (1.4-2.1 M☉)
            m if m >= 1.04 => StellarClass::F,  // Yellow-white main sequence (1.04-1.4 M☉)
            m if m >= 0.8  => StellarClass::G,  // Yellow main sequence (0.8-1.04 M☉) - like Sun
            m if m >= 0.45 => StellarClass::K,  // Orange main sequence (0.45-0.8 M☉)
            m if m >= 0.08 => StellarClass::M,  // Red dwarfs (0.08-0.45 M☉)
            _ => StellarClass::BH,  // Below hydrogen burning limit -> failed star
        }
    }

    /// Calculate stellar radius using mass-radius relationships
    fn calculate_stellar_radius(&self, mass_solar: f64) -> f64 {
        let solar_radius = 6.957e8; // meters
        
        // Mass-radius relation depends on stellar mass range
        let radius_ratio = if mass_solar > 1.0 {
            // High mass stars: R ~ M^0.8 (radiation pressure dominates)
            mass_solar.powf(0.8)
        } else {
            // Low mass stars: R ~ M^0.9 (more sensitive to mass)
            mass_solar.powf(0.9)
        };
        
        solar_radius * radius_ratio
    }

    /// Calculate stellar temperature using mass-temperature relationships
    fn calculate_stellar_temperature(&self, mass_solar: f64) -> f64 {
        // Main sequence temperature-mass relation: T ~ M^0.5 for nuclear burning
        let _solar_temp = 5778.0; // Kelvin
        
        match mass_solar {
            m if m >= 15.0 => 30000.0 + (m - 15.0) * 2000.0, // O-type: 30,000-50,000 K
            m if m >= 2.1  => 10000.0 + (m - 2.1) * 1540.0,  // B-type: 10,000-30,000 K
            m if m >= 1.4  => 7500.0 + (m - 1.4) * 3571.0,   // A-type: 7,500-10,000 K
            m if m >= 1.04 => 6000.0 + (m - 1.04) * 4167.0,  // F-type: 6,000-7,500 K
            m if m >= 0.8  => 5200.0 + (m - 0.8) * 3333.0,   // G-type: 5,200-6,000 K
            m if m >= 0.45 => 3700.0 + (m - 0.45) * 4286.0,  // K-type: 3,700-5,200 K
            m if m >= 0.08 => 2400.0 + (m - 0.08) * 3514.0,  // M-type: 2,400-3,700 K
            _ => 2000.0, // Brown dwarfs
        }
    }

    /// Calculate stellar luminosity using mass-luminosity relationships
    fn calculate_stellar_luminosity(&self, mass_solar: f64) -> f64 {
        let solar_luminosity = 3.828e26; // Watts
        
        // Mass-luminosity relation: L ~ M^α where α depends on mass
        let luminosity_ratio = if mass_solar > 0.43 {
            // Main sequence stars: L ~ M^4 (CNO cycle dominates for high mass)
            if mass_solar > 1.5 {
                mass_solar.powf(3.5) // Slightly less steep for massive stars
            } else {
                mass_solar.powf(4.0) // Steep dependence for intermediate mass
            }
        } else {
            // Low mass stars: L ~ M^2.3 (pp-chain, convective cores)
            mass_solar.powf(2.3)
        };
        
        solar_luminosity * luminosity_ratio
    }

    /// Generate composition for a new star
    fn generate_stellar_composition(&self, mass: f64) -> ElementTable {
        // Higher mass stars form from slightly more metal-rich gas due to galactic evolution
        let metallicity_factor = (mass / 10.0).clamp(0.5, 2.0); // 0.5x to 2x solar metallicity
        
        let mut composition = ElementTable::new();
        composition.set_abundance(1, 730_000);  // 73% H (constant)
        composition.set_abundance(2, 250_000);  // 25% He (constant)
        
        // Metals scale with mass (galactic chemical evolution effect)
        let base_metals = 20_000; // 2% metals total
        let metal_abundance = (base_metals as f64 * metallicity_factor) as u32;
        
        // Distribute metals among common elements
        composition.set_abundance(8, metal_abundance / 2);  // Oxygen (most abundant metal)
        composition.set_abundance(6, metal_abundance / 4);  // Carbon
        composition.set_abundance(10, metal_abundance / 8); // Neon
        composition.set_abundance(26, metal_abundance / 16); // Iron
        composition.set_abundance(14, metal_abundance / 16); // Silicon
        
        composition
    }
    
    /// Process planet formation around existing stars
    pub fn process_planet_formation(&mut self) -> Result<Vec<Uuid>> {
        let mut new_planets = Vec::new();
        let mut rng = thread_rng();
        
        // Check each star system for planet formation opportunities
        let star_ids: Vec<Uuid> = self.celestial_bodies.keys()
            .filter(|&&id| {
                matches!(self.celestial_bodies[&id].body_type, CelestialBodyType::Star { .. })
            })
            .cloned()
            .collect();
        
        for star_id in star_ids {
            let star = &self.celestial_bodies[&star_id];
            
            // Only form planets around young stars (< 100 Myr)
            if star.age < 1e8 * 365.25 * 24.0 * 3600.0 && rng.gen::<f64>() < 0.01 {
                let planet_id = self.create_planet(star_id)?;
                new_planets.push(planet_id);
            }
        }
        
        Ok(new_planets)
    }
    
    /// Create a planet around a star
    fn create_planet(&mut self, star_id: Uuid) -> Result<Uuid> {
        let mut rng = thread_rng();
        let planet_id = Uuid::new_v4();
        
        let star = self.celestial_bodies.get(&star_id).ok_or_else(|| anyhow::anyhow!("Star not found"))?;
        
        // Random orbital radius (0.1 to 50 AU)
        let orbital_radius = rng.gen_range(0.1..50.0) * 1.496e11;  // AU in meters
        
        // Planet mass based on orbital radius and star mass
        let planet_mass = if orbital_radius < 2.0 * 1.496e11 {
            // Rocky planets closer to star
            rng.gen_range(0.1..10.0) * 5.972e24  // Earth masses
        } else {
            // Gas giants farther out
            rng.gen_range(10.0..1000.0) * 5.972e24
        };
        
        let planet_radius = (planet_mass / 5.972e24_f64).powf(0.33) * 6.371e6;  // Earth radii scaling
        
        // Determine planet class based on orbital radius and mass
        let planet_class = self.classify_planet(orbital_radius, planet_mass, star.luminosity)?;
        
        // Calculate orbital position relative to star
        let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let planet_position = star.position + Vector3::new(
            orbital_radius * angle.cos(),
            orbital_radius * angle.sin(),
            rng.gen_range(-0.1..0.1) * orbital_radius  // Small Z component
        );
        
        // --- Orbital velocity calculation ---
        // v = sqrt(G * M_star / r)
        // Direction is perpendicular to the radius vector in the XY plane.
        const G: f64 = 6.67430e-11; // m^3 kg^-1 s^-2
        let orbital_speed = (G * star.mass / orbital_radius).sqrt();
        let velocity_direction = Vector3::new(-angle.sin(), angle.cos(), 0.0);
        let planet_velocity = velocity_direction * orbital_speed;

        // Simplified temperature calculation based on stellar luminosity and orbital radius
        let au = 1.496e11; // Astronomical Unit in meters
        let _distance_au = orbital_radius / au;
        let temperature = self.calculate_planet_temperature(orbital_radius, star.luminosity);
        
        let new_planet = CelestialBodyData {
            id: planet_id,
            body_type: CelestialBodyType::Planet { planet_class: planet_class.clone() },
            position: planet_position,
            velocity: planet_velocity,
            mass: planet_mass,
            radius: planet_radius,
            temperature,
            luminosity: 0.0,  // Planets don't emit light
            age: 0.0,
            composition: self.generate_planetary_composition(&planet_class, orbital_radius),
            planets: Vec::new(),  // Planets don't have planets
        };
        
        // Add planet to star system
        if let Some(system) = self.star_systems.values_mut()
            .find(|s| s.primary_star == star_id) {
            system.planets.push(planet_id);
        }
        
        // Add planet to star's planet list
        if let Some(star) = self.celestial_bodies.get_mut(&star_id) {
            star.planets.push(planet_id);
        }
        
        self.celestial_bodies.insert(planet_id, new_planet);
        
        tracing::info!("Created planet {} around star {} at {:.2} AU", 
                      planet_id, star_id, orbital_radius / 1.496e11);
        
        Ok(planet_id)
    }
    
    /// Classify planet based on orbital characteristics
    fn classify_planet(&self, orbital_radius: f64, mass: f64, stellar_luminosity: f64) -> Result<PlanetClass> {
        let au = 1.496e11;
        let earth_mass = 5.972e24;
        
        // Calculate equilibrium temperature
        let temp = self.calculate_planet_temperature(orbital_radius, stellar_luminosity);
        
        let class = match (orbital_radius / au, mass / earth_mass, temp) {
            // Earth-like: moderate distance, rocky, temperate
            (r, m, t) if r > 0.5 && r < 2.0 && m < 10.0 && t > 250.0 && t < 350.0 => PlanetClass::E,
            
            // Desert: close to star or low water
            (r, m, t) if r < 1.5 && m < 10.0 && t > 350.0 => PlanetClass::D,
            
            // Ice: far from star
            (r, m, t) if r > 2.0 && m < 10.0 && t < 250.0 => PlanetClass::I,
            
            // Gas dwarf: massive planets
            (_, m, _) if m > 10.0 => PlanetClass::G,
            
            // Default to toxic for edge cases
            _ => PlanetClass::T,
        };
        
        Ok(class)
    }
    
    /// Calculate planet equilibrium temperature
    fn calculate_planet_temperature(&self, orbital_radius: f64, stellar_luminosity: f64) -> f64 {
        let stefan_boltzmann = 5.67e-8;
        let albedo = 0.3;  // Simplified average albedo
        
        // Solar flux at planet's orbit
        let flux = stellar_luminosity / (4.0 * std::f64::consts::PI * orbital_radius.powi(2));
        
        // Equilibrium temperature
        ((1.0 - albedo) * flux / (4.0 * stefan_boltzmann)).powf(0.25)
    }
    
    /// Generate planetary composition based on class and formation distance
    fn generate_planetary_composition(&self, class: &PlanetClass, orbital_radius: f64) -> ElementTable {
        let mut composition = ElementTable::new();
        let au = 1.496e11;
        let _distance_au = orbital_radius / au;
        
        match class {
            PlanetClass::E => {
                // Earth-like composition
                composition.set_abundance(8, 461_000);   // O
                composition.set_abundance(14, 282_000);  // Si
                composition.set_abundance(13, 82_300);   // Al
                composition.set_abundance(26, 56_300);   // Fe
                composition.set_abundance(20, 41_500);   // Ca
                composition.set_abundance(11, 23_600);   // Na
                composition.set_abundance(19, 20_900);   // K
                composition.set_abundance(12, 20_300);   // Mg
            },
            
            PlanetClass::D => {
                // Desert world - depleted in volatiles
                composition.set_abundance(14, 350_000);  // Si (high)
                composition.set_abundance(26, 80_000);   // Fe
                composition.set_abundance(8, 300_000);   // O (reduced)
                composition.set_abundance(13, 60_000);   // Al
                composition.set_abundance(12, 40_000);   // Mg
            },
            
            PlanetClass::I => {
                // Ice world - high volatiles
                composition.set_abundance(1, 200_000);   // H (water ice)
                composition.set_abundance(8, 600_000);   // O (water ice)
                composition.set_abundance(6, 50_000);    // C (organics)
                composition.set_abundance(7, 30_000);    // N (ammonia)
                composition.set_abundance(14, 100_000);  // Si (rocky core)
            },
            
            PlanetClass::T => {
                // Toxic world - unusual composition
                composition.set_abundance(16, 100_000);  // S (sulfur compounds)
                composition.set_abundance(17, 50_000);   // Cl (chlorine)
                composition.set_abundance(8, 400_000);   // O
                composition.set_abundance(14, 200_000);  // Si
                composition.set_abundance(26, 30_000);   // Fe
            },
            
            PlanetClass::G => {
                // Gas dwarf - dominated by light elements
                composition.set_abundance(1, 800_000);   // H
                composition.set_abundance(2, 150_000);   // He
                composition.set_abundance(6, 20_000);    // C (methane)
                composition.set_abundance(7, 15_000);    // N
                composition.set_abundance(8, 10_000);    // O
            },
        }
        
        composition
    }
    
    /// Get world statistics
    pub fn get_stats(&self) -> WorldStats {
        let mut star_count = 0;
        let mut planet_count = 0;
        let system_count = self.star_systems.len();
        
        for body in self.celestial_bodies.values() {
            match body.body_type {
                CelestialBodyType::Star { .. } => star_count += 1,
                CelestialBodyType::Planet { .. } => planet_count += 1,
                _ => {},
            }
        }
        
        WorldStats {
            grid_size: (self.grid.width, self.grid.height),
            star_count,
            planet_count,
            system_count,
            universe_age: self.universe_age,
            total_mass: self.calculate_total_mass(),
        }
    }
    
    /// Calculate total mass in the universe
    fn calculate_total_mass(&self) -> f64 {
        self.celestial_bodies.values()
            .map(|body| body.mass)
            .sum()
    }
}

impl WorldGrid {
    pub fn new(width: usize, height: usize, scale_per_cell: f64) -> Self {
        let mut cells = Vec::with_capacity(height);
        
        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let position = Vector3::new(
                    x as f64 * scale_per_cell,
                    y as f64 * scale_per_cell,
                    0.0
                );
                
                let cell = WorldCell {
                    position,
                    cell_type: CellType::Vacuum,
                    temperature: 2.7,  // CMB temperature
                    pressure: 0.0,
                    density: 0.0,
                    strata: Vec::new(),
                    occupants: Vec::new(),
                    celestial_body_id: None,
                };
                
                row.push(cell);
            }
            cells.push(row);
        }
        
        Self {
            width,
            height,
            cells,
            scale_per_cell,
        }
    }
    
    /// Get cell with toroidal wrapping
    pub fn get_cell(&self, x: isize, y: isize) -> &WorldCell {
        let wrapped_x = ((x % self.width as isize) + self.width as isize) % self.width as isize;
        let wrapped_y = ((y % self.height as isize) + self.height as isize) % self.height as isize;
        
        &self.cells[wrapped_y as usize][wrapped_x as usize]
    }
    
    /// Get mutable cell with toroidal wrapping
    pub fn get_cell_mut(&mut self, x: isize, y: isize) -> &mut WorldCell {
        let wrapped_x = ((x % self.width as isize) + self.width as isize) % self.width as isize;
        let wrapped_y = ((y % self.height as isize) + self.height as isize) % self.height as isize;
        
        &mut self.cells[wrapped_y as usize][wrapped_x as usize]
    }
}

/// World statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldStats {
    pub grid_size: (usize, usize),
    pub star_count: usize,
    pub planet_count: usize,
    pub system_count: usize,
    pub universe_age: f64,
    pub total_mass: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_world_creation() {
        let world = World::new(100, 100, 1e15);
        assert_eq!(world.grid.width, 100);
        assert_eq!(world.grid.height, 100);
        assert_eq!(world.celestial_bodies.len(), 0);
    }
    
    #[test]
    fn test_grid_wrapping() {
        let grid = WorldGrid::new(10, 10, 1e15);
        
        // Test wrapping
        let cell1 = grid.get_cell(-1, -1);
        let cell2 = grid.get_cell(9, 9);
        assert_eq!(cell1.position.x, cell2.position.x);
        assert_eq!(cell1.position.y, cell2.position.y);
    }
    
    #[test]
    fn test_big_bang_initialization() {
        let mut world = World::new(10, 10, 1e15);
        world.init_big_bang().unwrap();
        
        // Check that cells have been initialized with gas
        let cell = &world.grid.cells[5][5];
        assert!(matches!(cell.cell_type, CellType::Gas));
        assert!(cell.temperature > 0.0);
        assert!(!cell.strata.is_empty());
    }
    
    #[test]
    fn test_star_formation() {
        let mut world = World::new(100, 100, 1e15);
        world.init_big_bang().unwrap();
        
        // Manually set up conditions for star formation
        world.grid.cells[50][50].density = 1e-20;  // High density
        world.grid.cells[50][50].temperature = 50.0;  // Cold
        
        let _stars = world.process_star_formation().unwrap();
        // Star formation is probabilistic, so we can't guarantee results
        // but the function should run without error
    }
    
    #[test]
    fn test_planet_classification() {
        let world = World::new(10, 10, 1e15);
        
        // Test Earth-like classification
        let class = world.classify_planet(1.5e11, 5.972e24, 3.828e26).unwrap();
        assert!(matches!(class, PlanetClass::E));
        
        // Test gas giant classification
        let class = world.classify_planet(5.0e11, 100.0 * 5.972e24, 3.828e26).unwrap();
        assert!(matches!(class, PlanetClass::G));
    }
}