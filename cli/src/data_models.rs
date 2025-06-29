use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Main simulation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStatus {
    pub tick: u64,
    pub ups: f64,  // Updates per second
    pub lineage_count: u32,
    pub planet_count: u32,
    pub star_count: u32,
    pub mean_entropy: f64,
    pub max_entropy: f64,
    pub save_file_age: String,
    pub uptime: String,
    pub memory_usage: MemoryUsage,
    pub performance_metrics: PerformanceMetrics,
    pub cosmic_era: CosmicEra,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub used_mb: f64,
    pub available_mb: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage: f64,
    pub gpu_usage: Option<f64>,
    pub network_io: NetworkIO,
    pub disk_io: DiskIO,
    pub avg_tick_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIO {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIO {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub operations_read: u64,
    pub operations_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CosmicEra {
    ParticleSoup,      // 0-0.3 Gy
    Starbirth,         // 0.3-1 Gy
    PlanetaryAge,      // 1-5 Gy
    Biogenesis,        // 5-10 Gy
    DigitalEvolution,  // 10-13 Gy+
    PostIntelligence,  // Post-singularity
}

/// Planet classification and data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Planet {
    pub id: Uuid,
    pub name: String,
    pub class: PlanetClass,
    pub environment: EnvironmentProfile,
    pub element_table: ElementTable,
    pub active_lineages: Vec<LineageInfo>,
    pub energy_budget: EnergyBudget,
    pub coordinates: Coordinates,
    pub orbital_data: OrbitalData,
    pub geological_layers: Vec<GeologicalLayer>,
    pub habitability_score: f64,
    pub population_capacity: u64,
    pub current_population: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanetClass {
    E, // Earth-like
    D, // Desert
    I, // Ice
    T, // Toxic
    G, // Gas dwarf
}

impl std::fmt::Display for PlanetClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanetClass::E => write!(f, "Earth-like"),
            PlanetClass::D => write!(f, "Desert"),
            PlanetClass::I => write!(f, "Ice"),
            PlanetClass::T => write!(f, "Toxic"),
            PlanetClass::G => write!(f, "Gas"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentProfile {
    pub liquid_water: f64,      // Surface liquid H₂O fraction
    pub atmos_oxygen: f64,      // O₂ percentage of atmosphere
    pub atmos_pressure: f64,    // Relative to Earth sea-level
    pub temp_celsius: f64,      // Mean surface °C
    pub radiation: f64,         // Cosmic & solar ionizing flux (Sv/year)
    pub energy_flux: f64,       // Stellar insolation (kW/m²)
    pub shelter_index: f64,     // Availability of caves/buildable structures
    pub hazard_rate: f64,       // Meteor, quake, storm frequency (events/year)
    pub magnetic_field: f64,    // Magnetic field strength (relative to Earth)
    pub atmospheric_composition: HashMap<String, f64>, // Element percentages
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementTable {
    /// Array of 118 elements, index = proton number, value = ppm
    #[serde(with = "serde_arrays")]
    pub abundances: [u32; 118],
    pub total_mass: f64,        // Total planetary mass in kg
    pub accessible_fraction: f64, // Fraction accessible to surface mining
}

mod serde_arrays {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(array: &[u32; 118], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vec: Vec<u32> = array.to_vec();
        vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u32; 118], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<u32> = Vec::deserialize(deserializer)?;
        if vec.len() != 118 {
            return Err(serde::de::Error::custom(format!(
                "Expected exactly 118 elements, found {}",
                vec.len()
            )));
        }
        let mut array = [0u32; 118];
        array.copy_from_slice(&vec);
        Ok(array)
    }
}

impl ElementTable {
    pub fn get_element_ppm(&self, element: &str) -> Option<u32> {
        let atomic_number = match element {
            "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5, "C" => 6,
            "N" => 7, "O" => 8, "F" => 9, "Ne" => 10, "Na" => 11, "Mg" => 12,
            "Al" => 13, "Si" => 14, "P" => 15, "S" => 16, "Cl" => 17, "Ar" => 18,
            "K" => 19, "Ca" => 20, "Fe" => 26, "Cu" => 29, "Ag" => 47, "Au" => 79,
            "U" => 92,
            _ => return None,
        };
        if atomic_number > 0 && atomic_number <= 118 {
            Some(self.abundances[atomic_number - 1])
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBudget {
    pub total_available: f64,   // Total energy available (J)
    pub consumption_rate: f64,  // Current consumption rate (J/tick)
    pub generation_rate: f64,   // Current generation rate (J/tick)
    pub storage_capacity: f64,  // Maximum storage (J)
    pub efficiency: f64,        // Energy conversion efficiency (0-1)
    pub sources: Vec<EnergySource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySource {
    pub source_type: String,
    pub power_output: f64,      // Watts
    pub efficiency: f64,
    pub maintenance_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coordinates {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub sector: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalData {
    pub semi_major_axis: f64,   // AU
    pub eccentricity: f64,
    pub inclination: f64,       // degrees
    pub orbital_period: f64,    // Earth days
    pub rotation_period: f64,   // Earth days
    pub parent_star: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeologicalLayer {
    pub depth: f64,             // meters
    pub thickness: f64,         // meters
    pub material_type: MaterialType,
    pub bulk_density: f64,      // kg/m³
    pub element_composition: HashMap<String, f64>, // Element percentages
    pub extractable: bool,
    pub hardness: f64,          // Mohs scale
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    Regolith,
    Topsoil,
    Subsoil,
    SedimentaryRock,
    IgneousRock,
    MetamorphicRock,
    OreVein,
    Ice,
    Magma,
}

/// Lineage (AI agent) information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    pub id: Uuid,
    pub name: String,
    pub code_hash: String,
    pub generation: u64,
    pub parent_id: Option<Uuid>,
    pub children: Vec<Uuid>,
    pub fitness_history: Vec<FitnessRecord>,
    pub current_fitness: f64,
    pub parameter_count: u64,
    pub planet_residence: Uuid,
    pub resource_usage: ResourceUsage,
    pub capabilities: Vec<Capability>,
    pub achievements: Vec<Achievement>,
    pub communication_history: Vec<CommunicationEvent>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageInfo {
    pub id: Uuid,
    pub name: String,
    pub fitness: f64,
    pub population: u64,
    pub activity_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessRecord {
    pub tick: u64,
    pub fitness: f64,
    pub entropy_cost: f64,
    pub resource_efficiency: f64,
    pub cooperation_score: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_cycles: u64,
    pub memory_bytes: u64,
    pub energy_joules: f64,
    pub bandwidth_bytes: u64,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub level: f64,
    pub unlocked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub name: String,
    pub description: String,
    pub achieved_at: DateTime<Utc>,
    pub tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationEvent {
    pub event_type: CommunicationType,
    pub target: Option<Uuid>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationType {
    Petition,
    Response,
    Broadcast,
    DirectMessage,
    ResourceRequest,
}

/// Star system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarSystem {
    pub id: Uuid,
    pub name: String,
    pub coordinates: Coordinates,
    pub primary_star: Star,
    pub companion_stars: Vec<Star>,
    pub planets: Vec<Uuid>,
    pub asteroid_belts: Vec<AsteroidBelt>,
    pub age: f64,               // Billion years
    pub metallicity: f64,       // Z parameter
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Star {
    pub id: Uuid,
    pub name: String,
    pub stellar_class: String,  // O, B, A, F, G, K, M
    pub mass: f64,              // Solar masses
    pub radius: f64,            // Solar radii
    pub temperature: f64,       // Kelvin
    pub luminosity: f64,        // Solar luminosities
    pub age: f64,               // Billion years
    pub metallicity: f64,
    pub habitable_zone: HabitableZone,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HabitableZone {
    pub inner_radius: f64,      // AU
    pub outer_radius: f64,      // AU
    pub optimum_radius: f64,    // AU
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsteroidBelt {
    pub inner_radius: f64,      // AU
    pub outer_radius: f64,      // AU
    pub total_mass: f64,        // Earth masses
    pub element_composition: HashMap<String, f64>,
}

/// Resource management structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub id: Uuid,
    pub lineage_id: Uuid,
    pub resource_type: ResourceType,
    pub amount: f64,
    pub justification: String,
    pub requested_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub status: RequestStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    GPU,
    Memory,
    Disk,
    Network,
    Energy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestStatus {
    Pending,
    Approved,
    Denied,
    Expired,
}

/// Oracle communication structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Petition {
    pub id: Uuid,
    pub lineage_id: Uuid,
    pub planet_id: Uuid,
    pub tick: u64,
    pub channel: PetitionChannel,
    pub payload: String,
    pub timestamp: DateTime<Utc>,
    pub read: bool,
    pub responded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PetitionChannel {
    Text,
    Data,
    Resource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetitionResponse {
    pub petition_id: Uuid,
    pub action: ResponseAction,
    pub payload: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseAction {
    Ack,
    Nack,
    Grant,
    Message,
}

/// Map visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapData {
    pub width: u16,
    pub height: u16,
    pub zoom_level: u8,
    pub map_type: MapType,
    pub cells: Vec<Vec<MapCell>>,
    pub legend: Vec<LegendEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MapType {
    Stars,
    Entropy,
    Temperature,
    Density,
    Lineages,
    Resources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapCell {
    pub value: f64,
    pub symbol: char,
    pub color: Option<String>,
    pub tooltip: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendEntry {
    pub symbol: char,
    pub description: String,
    pub value_range: Option<(f64, f64)>,
}

/// Snapshot data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseSnapshot {
    pub metadata: SnapshotMetadata,
    pub simulation_state: SimulationStatus,
    pub star_systems: Vec<StarSystem>,
    pub planets: Vec<Planet>,
    pub lineages: Vec<Lineage>,
    pub resource_requests: Vec<ResourceRequest>,
    pub petitions: Vec<Petition>,
    pub performance_history: Vec<PerformanceSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub created_at: DateTime<Utc>,
    pub version: String,
    pub tick: u64,
    pub checksum: String,
    pub compression: Option<String>,
    pub full_state: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub tick: u64,
    pub ups: f64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_io: NetworkIO,
    pub disk_io: DiskIO,
}