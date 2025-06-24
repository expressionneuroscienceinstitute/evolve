use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main simulation state containing all universe data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub tick: u64,
    pub ups: f64, // Updates per second
    pub current_time: f64, // Time since Big Bang in years
    pub box_size: f64, // Simulation box size in megaparsecs
    pub particles: Vec<Particle>,
    pub dark_matter: Vec<DarkMatterParticle>,
    pub galaxies: Vec<Galaxy>,
    pub stars: Vec<Star>,
    pub planets: Vec<Planet>,
    pub lineages: Vec<AgentLineage>,
    pub cosmological_params: CosmologicalParameters,
    pub entropy: f64,
    pub temperature: f64,
    pub density_field: DensityField,
}

impl SimulationState {
    /// Create a mock state for development
    pub fn mock() -> Self {
        let mut galaxies = Vec::new();
        let mut stars = Vec::new();
        let mut planets = Vec::new();
        
        // Create some galaxy clusters with realistic distribution
        for cluster_id in 0..5 {
            let cluster_center = Vector3::new(
                (cluster_id as f64 * 20.0) - 40.0,
                ((cluster_id as f64 * 15.0) - 30.0).sin() * 20.0,
                ((cluster_id as f64 * 10.0) - 20.0).cos() * 15.0,
            );
            
            // Create galaxies in each cluster
            for i in 0..20 {
                let offset = Vector3::new(
                    (i as f64 * 0.7).sin() * 5.0,
                    (i as f64 * 0.5).cos() * 5.0,
                    (i as f64 * 0.3).sin() * 3.0,
                );
                
                let galaxy = Galaxy {
                    id: format!("GAL-{}-{}", cluster_id, i),
                    position: cluster_center + offset,
                    mass: 1e12, // Solar masses
                    stellar_mass: 1e11,
                    dark_matter_mass: 9e11,
                    metallicity: 0.02,
                    star_formation_rate: 10.0, // Solar masses per year
                    galaxy_type: GalaxyType::Spiral,
                    redshift: 0.1 + (cluster_id as f64 * 0.05),
                };
                galaxies.push(galaxy);
                
                // Add some stars to each galaxy
                for j in 0..5 {
                    stars.push(Star {
                        id: format!("STAR-{}-{}-{}", cluster_id, i, j),
                        position: cluster_center + offset + Vector3::new(
                            (j as f64 * 0.1).sin(),
                            (j as f64 * 0.1).cos(),
                            0.0
                        ),
                        mass: 1.0 + (j as f64 * 0.5),
                        temperature: 5778.0 * (1.0 + (j as f64 * 0.2)),
                        luminosity: 1.0 * (1.0 + j as f64).powf(3.5),
                        spectral_class: SpectralClass::G,
                        age: 4.6e9, // years
                        metallicity: 0.02,
                    });
                    
                    // Add a planet to some stars
                    if j % 2 == 0 {
                        planets.push(Planet {
                            id: format!("PLANET-{}-{}-{}", cluster_id, i, j),
                            star_id: format!("STAR-{}-{}-{}", cluster_id, i, j),
                            position: cluster_center + offset + Vector3::new(
                                (j as f64 * 0.1).sin() + 0.01,
                                (j as f64 * 0.1).cos() + 0.01,
                                0.01
                            ),
                            mass: 1.0, // Earth masses
                            radius: 1.0, // Earth radii
                            temperature: 288.0, // Kelvin
                            atmosphere_pressure: 1.0, // atm
                            water_fraction: 0.7,
                            planet_class: PlanetClass::EarthLike,
                            habitability_score: 0.8,
                            has_life: j == 0,
                            population: if j == 0 { 1000000 } else { 0 },
                        });
                    }
                }
            }
        }
        
        // Create density field
        let density_field = DensityField::new(128, 100.0);
        
        Self {
            tick: 1000000,
            ups: 60.0,
            current_time: 13.8e9, // 13.8 billion years
            box_size: 100.0, // 100 Mpc
            particles: vec![],
            dark_matter: vec![],
            galaxies,
            stars,
            planets,
            lineages: vec![],
            cosmological_params: CosmologicalParameters::default(),
            entropy: 1.0e23,
            temperature: 2.725, // CMB temperature
            density_field,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub mass: f64,
    pub particle_type: ParticleType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParticleType {
    Baryon,
    DarkMatter,
    Photon,
    Neutrino,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkMatterParticle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub mass: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Galaxy {
    pub id: String,
    pub position: Vector3<f64>,
    pub mass: f64,
    pub stellar_mass: f64,
    pub dark_matter_mass: f64,
    pub metallicity: f64,
    pub star_formation_rate: f64,
    pub galaxy_type: GalaxyType,
    pub redshift: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GalaxyType {
    Spiral,
    Elliptical,
    Irregular,
    Dwarf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Star {
    pub id: String,
    pub position: Vector3<f64>,
    pub mass: f64, // Solar masses
    pub temperature: f64, // Kelvin
    pub luminosity: f64, // Solar luminosities
    pub spectral_class: SpectralClass,
    pub age: f64, // years
    pub metallicity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpectralClass {
    O, B, A, F, G, K, M,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Planet {
    pub id: String,
    pub star_id: String,
    pub position: Vector3<f64>,
    pub mass: f64, // Earth masses
    pub radius: f64, // Earth radii
    pub temperature: f64, // Kelvin
    pub atmosphere_pressure: f64, // atm
    pub water_fraction: f64,
    pub planet_class: PlanetClass,
    pub habitability_score: f64,
    pub has_life: bool,
    pub population: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PlanetClass {
    EarthLike,
    Desert,
    Ice,
    Toxic,
    GasDwarf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLineage {
    pub id: String,
    pub planet_id: String,
    pub generation: u64,
    pub population: u64,
    pub technology_level: f64,
    pub has_space_travel: bool,
    pub colonized_systems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmologicalParameters {
    pub h0: f64, // Hubble constant
    pub omega_m: f64, // Matter density
    pub omega_lambda: f64, // Dark energy density
    pub omega_b: f64, // Baryon density
    pub sigma8: f64, // Power spectrum normalization
}

impl Default for CosmologicalParameters {
    fn default() -> Self {
        Self {
            h0: 67.4,
            omega_m: 0.315,
            omega_lambda: 0.685,
            omega_b: 0.049,
            sigma8: 0.811,
        }
    }
}

/// 3D density field for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityField {
    pub resolution: usize,
    pub box_size: f64,
    pub data: Vec<Vec<Vec<f64>>>, // 3D grid of density values
}

impl DensityField {
    pub fn new(resolution: usize, box_size: f64) -> Self {
        let mut data = vec![vec![vec![0.0; resolution]; resolution]; resolution];
        
        // Generate realistic cosmic web structure
        for i in 0..resolution {
            for j in 0..resolution {
                for k in 0..resolution {
                    let x = (i as f64 / resolution as f64 - 0.5) * 2.0;
                    let y = (j as f64 / resolution as f64 - 0.5) * 2.0;
                    let z = (k as f64 / resolution as f64 - 0.5) * 2.0;
                    
                    // Create filamentary structure
                    let filament1 = (-(x * x + y * y) * 5.0).exp();
                    let filament2 = (-(y * y + z * z) * 5.0).exp();
                    let filament3 = (-(x * x + z * z) * 5.0).exp();
                    
                    // Add some cluster peaks
                    let cluster1 = (-((x - 0.3).powi(2) + y.powi(2) + z.powi(2)) * 20.0).exp();
                    let cluster2 = (-((x + 0.3).powi(2) + (y - 0.3).powi(2) + z.powi(2)) * 20.0).exp();
                    
                    // Combine to create cosmic web
                    data[i][j][k] = 1.0 + 5.0 * (filament1 + filament2 + filament3) 
                                  + 20.0 * (cluster1 + cluster2);
                }
            }
        }
        
        Self {
            resolution,
            box_size,
            data,
        }
    }
    
    pub fn sample_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let i = ((x / self.box_size + 0.5) * self.resolution as f64) as usize;
        let j = ((y / self.box_size + 0.5) * self.resolution as f64) as usize;
        let k = ((z / self.box_size + 0.5) * self.resolution as f64) as usize;
        
        if i < self.resolution && j < self.resolution && k < self.resolution {
            self.data[i][j][k]
        } else {
            0.0
        }
    }
}