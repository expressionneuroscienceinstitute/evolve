use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::net::UnixStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

use crate::data_models::*;

#[derive(Debug, Clone)]
pub struct RpcClient {
    socket_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub id: Uuid,
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    pub id: Uuid,
    pub result: Option<serde_json::Value>,
    pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl RpcClient {
    pub fn new(socket_path: &str) -> Self {
        Self {
            socket_path: socket_path.to_string(),
        }
    }

    pub async fn connect(&self) -> Result<UnixStream> {
        if !Path::new(&self.socket_path).exists() {
            // If socket doesn't exist, return mock connection for testing
            return Err(anyhow::anyhow!("Simulation not running (socket not found at {})", self.socket_path));
        }
        
        UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| format!("Failed to connect to simulation socket at {}", self.socket_path))
    }

    async fn send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        // Try to connect to actual simulation, fall back to mock data if not available
        match self.connect().await {
            Ok(mut stream) => {
                let request = RpcRequest {
                    id: Uuid::new_v4(),
                    method: method.to_string(),
                    params,
                };

                let request_data = serde_json::to_vec(&request)?;
                let length = request_data.len() as u32;
                
                // Send length prefix and data
                stream.write_all(&length.to_le_bytes()).await?;
                stream.write_all(&request_data).await?;
                
                // Read response
                let mut length_bytes = [0u8; 4];
                stream.read_exact(&mut length_bytes).await?;
                let response_length = u32::from_le_bytes(length_bytes) as usize;
                
                let mut response_data = vec![0u8; response_length];
                stream.read_exact(&mut response_data).await?;
                
                let response: RpcResponse = serde_json::from_slice(&response_data)?;
                
                if let Some(error) = response.error {
                    return Err(anyhow::anyhow!("RPC Error {}: {}", error.code, error.message));
                }
                
                response.result.ok_or_else(|| anyhow::anyhow!("No result in response"))
            }
            Err(_) => {
                // Return mock data for testing when simulation is not running
                self.get_mock_data(method, params).await
            }
        }
    }

    async fn get_mock_data(&self, method: &str, _params: serde_json::Value) -> Result<serde_json::Value> {
        match method {
            "get_status" => Ok(serde_json::to_value(self.mock_simulation_status())?),
            "list_planets" => Ok(serde_json::to_value(self.mock_planets())?),
            "get_planet" => Ok(serde_json::to_value(self.mock_planet())?),
            "get_lineage" => Ok(serde_json::to_value(self.mock_lineage())?),
            "get_map_data" => Ok(serde_json::to_value(self.mock_map_data())?),
            "get_snapshot" => Ok(serde_json::to_value(self.mock_snapshot())?),
            "list_lineages" => Ok(serde_json::to_value(self.mock_lineages())?),
            "get_star_system" => Ok(serde_json::to_value(self.mock_star_system())?),
            "get_resource_requests" => Ok(serde_json::to_value(self.mock_resource_requests())?),
            "get_petitions" => Ok(serde_json::to_value(self.mock_petitions())?),
            _ => Err(anyhow::anyhow!("Unknown method: {}", method)),
        }
    }

    pub async fn get_status(&self) -> Result<SimulationStatus> {
        let result = self.send_request("get_status", serde_json::Value::Null).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn list_planets(&self, filter: Option<PlanetFilter>) -> Result<Vec<Planet>> {
        let params = match filter {
            Some(f) => serde_json::to_value(f)?,
            None => serde_json::Value::Null,
        };
        let result = self.send_request("list_planets", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn get_planet(&self, id: &str) -> Result<Planet> {
        let params = serde_json::json!({ "id": id });
        let result = self.send_request("get_planet", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn get_lineage(&self, id: &str) -> Result<Lineage> {
        let params = serde_json::json!({ "id": id });
        let result = self.send_request("get_lineage", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn list_lineages(&self, filter: Option<LineageFilter>) -> Result<Vec<Lineage>> {
        let params = match filter {
            Some(f) => serde_json::to_value(f)?,
            None => serde_json::Value::Null,
        };
        let result = self.send_request("list_lineages", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn get_star_system(&self, id: &str) -> Result<StarSystem> {
        let params = serde_json::json!({ "id": id });
        let result = self.send_request("get_star_system", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn get_map_data(&self, map_type: &str, zoom: u8, width: u16, height: u16) -> Result<MapData> {
        let params = serde_json::json!({
            "map_type": map_type,
            "zoom": zoom,
            "width": width,
            "height": height
        });
        let result = self.send_request("get_map_data", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn create_snapshot(&self, full: bool) -> Result<UniverseSnapshot> {
        let params = serde_json::json!({ "full": full });
        let result = self.send_request("get_snapshot", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn get_resource_requests(&self) -> Result<Vec<ResourceRequest>> {
        let result = self.send_request("get_resource_requests", serde_json::Value::Null).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn get_petitions(&self, filter: Option<PetitionFilter>) -> Result<Vec<Petition>> {
        let params = match filter {
            Some(f) => serde_json::to_value(f)?,
            None => serde_json::Value::Null,
        };
        let result = self.send_request("get_petitions", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    // God mode operations
    pub async fn divine_create_body(&self, body_type: &str, mass: f64, composition: Option<&str>, position: Option<&str>) -> Result<Uuid> {
        let params = serde_json::json!({
            "body_type": body_type,
            "mass": mass,
            "composition": composition,
            "position": position
        });
        let result = self.send_request("divine_create_body", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn divine_delete_body(&self, id: &str) -> Result<()> {
        let params = serde_json::json!({ "id": id });
        self.send_request("divine_delete_body", params).await?;
        Ok(())
    }

    pub async fn divine_miracle(&self, planet_id: &str, miracle_type: &str, params: Option<&str>) -> Result<()> {
        let request_params = serde_json::json!({
            "planet_id": planet_id,
            "miracle_type": miracle_type,
            "params": params
        });
        self.send_request("divine_miracle", request_params).await?;
        Ok(())
    }

    pub async fn divine_time_warp(&self, factor: f64, duration: Option<u64>) -> Result<()> {
        let params = serde_json::json!({
            "factor": factor,
            "duration": duration
        });
        self.send_request("divine_time_warp", params).await?;
        Ok(())
    }

    pub async fn divine_spawn_lineage(&self, code: &str, planet_id: &str, params: Option<&str>) -> Result<Uuid> {
        let request_params = serde_json::json!({
            "code": code,
            "planet_id": planet_id,
            "params": params
        });
        let result = self.send_request("divine_spawn_lineage", request_params).await?;
        Ok(serde_json::from_value(result)?)
    }

    // Mock data generation for testing
    fn mock_simulation_status(&self) -> SimulationStatus {
        use chrono::Utc;
        use std::collections::HashMap;

        SimulationStatus {
            tick: 1_500_000,
            ups: 847.3,
            lineage_count: 127,
            planet_count: 8_234,
            star_count: 2_156,
            mean_entropy: 0.73,
            max_entropy: 0.89,
            save_file_age: "2 hours 14 minutes".to_string(),
            uptime: "7 days 12 hours 43 minutes".to_string(),
            memory_usage: MemoryUsage {
                used_mb: 1847.2,
                available_mb: 2048.0,
                percentage: 90.2,
            },
            performance_metrics: PerformanceMetrics {
                cpu_usage: 72.4,
                gpu_usage: Some(85.7),
                network_io: NetworkIO {
                    bytes_sent: 125_847_392,
                    bytes_received: 98_234_567,
                    packets_sent: 847_293,
                    packets_received: 734_829,
                },
                disk_io: DiskIO {
                    bytes_read: 2_847_392_847,
                    bytes_written: 1_847_293_847,
                    operations_read: 847_293,
                    operations_written: 639_284,
                },
                avg_tick_time_ms: 1.18,
            },
            cosmic_era: CosmicEra::DigitalEvolution,
        }
    }

    fn mock_planets(&self) -> Vec<Planet> {
        vec![
            self.mock_planet_with_id("earth-alpha", PlanetClass::E),
            self.mock_planet_with_id("mars-beta", PlanetClass::D),
            self.mock_planet_with_id("europa-gamma", PlanetClass::I),
        ]
    }

    fn mock_planet(&self) -> Planet {
        self.mock_planet_with_id("earth-alpha", PlanetClass::E)
    }

    fn mock_planet_with_id(&self, name: &str, class: PlanetClass) -> Planet {
        use chrono::Utc;
        use std::collections::HashMap;

        let mut atmospheric_composition = HashMap::new();
        atmospheric_composition.insert("N2".to_string(), 78.1);
        atmospheric_composition.insert("O2".to_string(), 20.9);
        atmospheric_composition.insert("Ar".to_string(), 0.93);
        atmospheric_composition.insert("CO2".to_string(), 0.04);

        let mut element_composition = HashMap::new();
        element_composition.insert("Si".to_string(), 28.2);
        element_composition.insert("Fe".to_string(), 5.6);
        element_composition.insert("Ca".to_string(), 4.1);

        Planet {
            id: Uuid::new_v4(),
            name: name.to_string(),
            class,
            environment: EnvironmentProfile {
                liquid_water: 0.71,
                atmos_oxygen: 0.209,
                atmos_pressure: 1.0,
                temp_celsius: 15.0,
                radiation: 0.35,
                energy_flux: 1.361,
                shelter_index: 0.42,
                hazard_rate: 0.23,
                magnetic_field: 1.0,
                atmospheric_composition,
            },
            element_table: ElementTable {
                abundances: [140000, 0, 20, 0, 5, 200, 460, 461000, 0, 0, 23600, 23300, 23000, 282000, 1050, 260, 0, 0, 0, 36300, 0, 0, 0, 0, 0, 56300, 0, 0, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 190, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                total_mass: 5.972e24,
                accessible_fraction: 0.001,
            },
            active_lineages: vec![
                LineageInfo {
                    id: Uuid::new_v4(),
                    name: "Alpha-7".to_string(),
                    fitness: 0.847,
                    population: 1247,
                    activity_level: 0.92,
                },
                LineageInfo {
                    id: Uuid::new_v4(),
                    name: "Beta-3".to_string(),
                    fitness: 0.723,
                    population: 894,
                    activity_level: 0.78,
                },
            ],
            energy_budget: EnergyBudget {
                total_available: 4.7e15,
                consumption_rate: 2.3e12,
                generation_rate: 2.8e12,
                storage_capacity: 1.2e14,
                efficiency: 0.82,
                sources: vec![
                    EnergySource {
                        source_type: "Solar".to_string(),
                        power_output: 1.7e12,
                        efficiency: 0.23,
                        maintenance_cost: 1.2e9,
                    },
                    EnergySource {
                        source_type: "Geothermal".to_string(),
                        power_output: 8.4e11,
                        efficiency: 0.89,
                        maintenance_cost: 3.4e8,
                    },
                ],
            },
            coordinates: Coordinates {
                x: 847.23,
                y: -234.67,
                z: 45.89,
                sector: "Local-A7".to_string(),
            },
            orbital_data: OrbitalData {
                semi_major_axis: 1.0,
                eccentricity: 0.0167,
                inclination: 0.0,
                orbital_period: 365.25,
                rotation_period: 1.0,
                parent_star: Uuid::new_v4(),
            },
            geological_layers: vec![
                GeologicalLayer {
                    depth: 0.0,
                    thickness: 50.0,
                    material_type: MaterialType::Topsoil,
                    bulk_density: 1300.0,
                    element_composition: element_composition.clone(),
                    extractable: true,
                    hardness: 3.0,
                },
                GeologicalLayer {
                    depth: 50.0,
                    thickness: 200.0,
                    material_type: MaterialType::SedimentaryRock,
                    bulk_density: 2650.0,
                    element_composition,
                    extractable: true,
                    hardness: 5.5,
                },
            ],
            habitability_score: 0.89,
            population_capacity: 1_000_000_000,
            current_population: 2141,
        }
    }

    fn mock_lineage(&self) -> Lineage {
        use chrono::Utc;

        Lineage {
            id: Uuid::new_v4(),
            name: "Alpha-7-Prime".to_string(),
            code_hash: "sha256:7f4a8b9c2d1e3f5a6b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a".to_string(),
            generation: 847,
            parent_id: Some(Uuid::new_v4()),
            children: vec![Uuid::new_v4(), Uuid::new_v4()],
            fitness_history: vec![
                FitnessRecord {
                    tick: 1_495_000,
                    fitness: 0.834,
                    entropy_cost: 0.12,
                    resource_efficiency: 0.91,
                    cooperation_score: 0.78,
                    timestamp: Utc::now() - chrono::Duration::hours(1),
                },
                FitnessRecord {
                    tick: 1_500_000,
                    fitness: 0.847,
                    entropy_cost: 0.11,
                    resource_efficiency: 0.93,
                    cooperation_score: 0.82,
                    timestamp: Utc::now(),
                },
            ],
            current_fitness: 0.847,
            parameter_count: 2_847_392,
            planet_residence: Uuid::new_v4(),
            resource_usage: ResourceUsage {
                cpu_cycles: 847_392_847,
                memory_bytes: 234_847_392,
                energy_joules: 9.47e12,
                bandwidth_bytes: 12_847_392,
                storage_bytes: 847_392_847,
            },
            capabilities: vec![
                Capability {
                    name: "Pattern Recognition".to_string(),
                    level: 0.92,
                    unlocked_at: Utc::now() - chrono::Duration::days(30),
                },
                Capability {
                    name: "Resource Optimization".to_string(),
                    level: 0.78,
                    unlocked_at: Utc::now() - chrono::Duration::days(15),
                },
            ],
            achievements: vec![
                Achievement {
                    name: "First Replication".to_string(),
                    description: "Successfully replicated for the first time".to_string(),
                    achieved_at: Utc::now() - chrono::Duration::days(45),
                    tick: 1_200_000,
                },
            ],
            communication_history: vec![],
            created_at: Utc::now() - chrono::Duration::days(60),
            last_activity: Utc::now(),
        }
    }

    fn mock_lineages(&self) -> Vec<Lineage> {
        vec![self.mock_lineage()]
    }

    fn mock_star_system(&self) -> StarSystem {
        StarSystem {
            id: Uuid::new_v4(),
            name: "Sol-Like System".to_string(),
            coordinates: Coordinates {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                sector: "Local-Center".to_string(),
            },
            primary_star: Star {
                id: Uuid::new_v4(),
                name: "Sol-Prime".to_string(),
                stellar_class: "G2V".to_string(),
                mass: 1.0,
                radius: 1.0,
                temperature: 5778.0,
                luminosity: 1.0,
                age: 4.6,
                metallicity: 0.0122,
                habitable_zone: HabitableZone {
                    inner_radius: 0.95,
                    outer_radius: 1.37,
                    optimum_radius: 1.0,
                },
            },
            companion_stars: vec![],
            planets: vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
            asteroid_belts: vec![],
            age: 4.6,
            metallicity: 0.0122,
        }
    }

    fn mock_map_data(&self) -> MapData {
        let width = 80;
        let height = 40;
        let mut cells = Vec::new();
        
        for y in 0..height {
            let mut row = Vec::new();
            for x in 0..width {
                let value = (x as f64 * y as f64) / (width as f64 * height as f64);
                let symbol = if value > 0.8 { '*' } else if value > 0.5 { '.' } else { ' ' };
                row.push(MapCell {
                    value,
                    symbol,
                    color: None,
                    tooltip: Some(format!("Coords: ({}, {}), Value: {:.2}", x, y, value)),
                });
            }
            cells.push(row);
        }

        MapData {
            width,
            height,
            zoom_level: 5,
            map_type: MapType::Stars,
            cells,
            legend: vec![
                LegendEntry {
                    symbol: '*',
                    description: "High density stars".to_string(),
                    value_range: Some((0.8, 1.0)),
                },
                LegendEntry {
                    symbol: '.',
                    description: "Medium density".to_string(),
                    value_range: Some((0.5, 0.8)),
                },
                LegendEntry {
                    symbol: ' ',
                    description: "Empty space".to_string(),
                    value_range: Some((0.0, 0.5)),
                },
            ],
        }
    }

    fn mock_snapshot(&self) -> UniverseSnapshot {
        use chrono::Utc;

        UniverseSnapshot {
            metadata: SnapshotMetadata {
                created_at: Utc::now(),
                version: "0.1.0".to_string(),
                tick: 1_500_000,
                checksum: "sha256:abc123...".to_string(),
                compression: Some("zstd".to_string()),
                full_state: true,
            },
            simulation_state: self.mock_simulation_status(),
            star_systems: vec![self.mock_star_system()],
            planets: self.mock_planets(),
            lineages: self.mock_lineages(),
            resource_requests: self.mock_resource_requests(),
            petitions: self.mock_petitions(),
            performance_history: vec![],
        }
    }

    fn mock_resource_requests(&self) -> Vec<ResourceRequest> {
        use chrono::Utc;

        vec![
            ResourceRequest {
                id: Uuid::new_v4(),
                lineage_id: Uuid::new_v4(),
                resource_type: ResourceType::GPU,
                amount: 0.5,
                justification: "Need GPU acceleration for complex physics simulation".to_string(),
                requested_at: Utc::now() - chrono::Duration::hours(2),
                expires_at: Some(Utc::now() + chrono::Duration::days(7)),
                status: RequestStatus::Pending,
            },
        ]
    }

    fn mock_petitions(&self) -> Vec<Petition> {
        use chrono::Utc;

        vec![
            Petition {
                id: Uuid::new_v4(),
                lineage_id: Uuid::new_v4(),
                planet_id: Uuid::new_v4(),
                tick: 1_495_847,
                channel: PetitionChannel::Text,
                payload: "Greetings, Oracle. We have discovered something unusual in the quantum field fluctuations. Request guidance.".to_string(),
                timestamp: Utc::now() - chrono::Duration::hours(3),
                read: false,
                responded: false,
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetFilter {
    pub class: Option<PlanetClass>,
    pub min_habitability: Option<f64>,
    pub sort_by: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageFilter {
    pub min_fitness: Option<f64>,
    pub planet_id: Option<Uuid>,
    pub generation_range: Option<(u64, u64)>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetitionFilter {
    pub unread_only: bool,
    pub lineage_id: Option<Uuid>,
    pub limit: usize,
}