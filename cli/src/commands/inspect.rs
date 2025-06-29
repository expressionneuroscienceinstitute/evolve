use anyhow::Result;
use colored::*;
use tabled::{Table, Tabled};
use std::collections::HashMap;

use crate::rpc::RpcClient;
use crate::data_models::*;
use crate::formatters;
use crate::{InspectTarget};

pub struct InspectCommand {
    rpc_client: RpcClient,
}

impl InspectCommand {
    pub fn new(socket_path: &str) -> Result<Self> {
        Ok(Self {
            rpc_client: RpcClient::new(socket_path),
        })
    }

    pub async fn execute(&self, target: InspectTarget) -> Result<()> {
        match target {
            InspectTarget::Planet { id, environment, lineages, energy, resources } => {
                self.inspect_planet(&id, environment, lineages, energy, resources).await
            }
            InspectTarget::Lineage { id, fitness, code, genealogy, time_range } => {
                self.inspect_lineage(&id, fitness, code, genealogy, time_range.as_deref()).await
            }
            InspectTarget::System { id, orbits, stellar } => {
                self.inspect_star_system(&id, orbits, stellar).await
            }
            InspectTarget::Performance { metric, window } => {
                self.inspect_performance(metric.as_deref(), &window).await
            }
        }
    }

    async fn inspect_planet(&self, id: &str, show_environment: bool, show_lineages: bool, show_energy: bool, show_resources: bool) -> Result<()> {
        let planet = self.rpc_client.get_planet(id).await?;
        
        println!("{}", format!("PLANET INSPECTION: {}", planet.name).cyan().bold());
        println!("{}", "=".repeat(60).cyan());
        println!();

        // Basic information
        self.display_planet_basics(&planet);

        // Environment profile (always shown by default, unless specifically disabled)
        if show_environment || (!show_lineages && !show_energy && !show_resources) {
            self.display_environment_profile(&planet.environment);
        }

        // Active lineages
        if show_lineages || (!show_environment && !show_energy && !show_resources) {
            self.display_planet_lineages(&planet.active_lineages).await?;
        }

        // Energy budget
        if show_energy || (!show_environment && !show_lineages && !show_resources) {
            self.display_energy_budget(&planet.energy_budget);
        }

        // Resource composition
        if show_resources || (!show_environment && !show_lineages && !show_energy) {
            self.display_resource_composition(&planet.element_table, &planet.geological_layers);
        }

        // Orbital data
        self.display_orbital_data(&planet.orbital_data);

        Ok(())
    }

    async fn inspect_lineage(&self, id: &str, show_fitness: bool, show_code: bool, show_genealogy: bool, time_range: Option<&str>) -> Result<()> {
        let lineage = self.rpc_client.get_lineage(id).await?;
        
        println!("{}", format!("LINEAGE INSPECTION: {}", lineage.name).cyan().bold());
        println!("{}", "=".repeat(60).cyan());
        println!();

        // Basic lineage information
        self.display_lineage_basics(&lineage);

        // Fitness history
        if show_fitness || (!show_code && !show_genealogy) {
            self.display_fitness_history(&lineage.fitness_history, time_range);
        }

        // Code evolution and parameters
        if show_code || (!show_fitness && !show_genealogy) {
            self.display_code_evolution(&lineage);
        }

        // Genealogy and relationships
        if show_genealogy || (!show_fitness && !show_code) {
            self.display_genealogy(&lineage).await?;
        }

        // Capabilities and achievements
        self.display_capabilities_and_achievements(&lineage);

        // Resource usage
        self.display_lineage_resource_usage(&lineage.resource_usage);

        Ok(())
    }

    async fn inspect_star_system(&self, id: &str, show_orbits: bool, show_stellar: bool) -> Result<()> {
        let system = self.rpc_client.get_star_system(id).await?;
        
        println!("{}", format!("STAR SYSTEM INSPECTION: {}", system.name).cyan().bold());
        println!("{}", "=".repeat(60).cyan());
        println!();

        // Basic system information
        self.display_system_basics(&system);

        // Stellar properties
        if show_stellar || !show_orbits {
            self.display_stellar_properties(&system.primary_star, &system.companion_stars);
        }

        // Orbital mechanics
        if show_orbits || !show_stellar {
            self.display_orbital_mechanics(&system).await?;
        }

        Ok(())
    }

    async fn inspect_performance(&self, metric_type: Option<&str>, window: &str) -> Result<()> {
        let status = self.rpc_client.get_status().await?;
        
        println!("{}", format!("PERFORMANCE INSPECTION ({})", window).cyan().bold());
        println!("{}", "=".repeat(60).cyan());
        println!();

        match metric_type {
            Some("cpu") => self.display_cpu_metrics(&status.performance_metrics),
            Some("memory") => self.display_memory_metrics(&status.memory_usage),
            Some("network") => self.display_network_metrics(&status.performance_metrics.network_io),
            Some("simulation") => self.display_simulation_metrics(&status),
            _ => {
                // Show all metrics
                self.display_cpu_metrics(&status.performance_metrics);
                self.display_memory_metrics(&status.memory_usage);
                self.display_network_metrics(&status.performance_metrics.network_io);
                self.display_simulation_metrics(&status);
            }
        }

        Ok(())
    }

    fn display_planet_basics(&self, planet: &Planet) {
        println!("{}", "BASIC INFORMATION".yellow().bold());
        
        let basic_info = vec![
            InfoRow::new("ID", formatters::format_uuid_short(&planet.id)),
            InfoRow::new("Name", planet.name.clone()),
            InfoRow::new("Class", format!("{} ({})", planet.class, planet.class)),
            InfoRow::new("Habitability Score", formatters::format_fitness(planet.habitability_score)),
            InfoRow::new("Coordinates", formatters::format_coordinates(
                planet.coordinates.x, planet.coordinates.y, planet.coordinates.z
            )),
            InfoRow::new("Sector", planet.coordinates.sector.clone()),
            InfoRow::new("Total Mass", formatters::format_mass(planet.element_table.total_mass)),
            InfoRow::new("Population", if planet.current_population > 0 {
                format!("{} / {}", 
                    formatters::format_number(planet.current_population), 
                    formatters::format_number(planet.population_capacity)
                )
            } else {
                "Uninhabited".dimmed().to_string()
            }),
        ];

        let table = Table::new(basic_info);
        println!("{}", table);
        println!();
    }

    fn display_environment_profile(&self, env: &EnvironmentProfile) {
        println!("{}", "ENVIRONMENT PROFILE".green().bold());
        
        let env_info = vec![
            InfoRow::new("Temperature", formatters::format_temperature(env.temp_celsius)),
            InfoRow::new("Atmospheric Pressure", formatters::format_pressure(env.atmos_pressure)),
            InfoRow::new("Liquid Water Coverage", format!("{:.1}%", env.liquid_water * 100.0)),
            InfoRow::new("Oxygen Concentration", format!("{:.2}%", env.atmos_oxygen * 100.0)),
            InfoRow::new("Radiation Level", formatters::format_radiation(env.radiation)),
            InfoRow::new("Energy Flux", format!("{:.3} kW/m²", env.energy_flux)),
            InfoRow::new("Shelter Index", format!("{:.2}", env.shelter_index)),
            InfoRow::new("Hazard Rate", format!("{:.2} events/year", env.hazard_rate)),
            InfoRow::new("Magnetic Field", format!("{:.2}× Earth", env.magnetic_field)),
        ];

        let table = Table::new(env_info);
        println!("{}", table);
        
        // Atmospheric composition
        if !env.atmospheric_composition.is_empty() {
            println!("\n{}", "Atmospheric Composition:".blue());
            let mut composition: Vec<_> = env.atmospheric_composition.iter().collect();
            composition.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            for (element, percentage) in composition {
                println!("  {}: {:.2}%", element, percentage);
            }
        }
        println!();
    }

    async fn display_planet_lineages(&self, lineages: &[LineageInfo]) -> Result<()> {
        println!("{}", "ACTIVE LINEAGES".blue().bold());
        
        if lineages.is_empty() {
            println!("{}", "No active lineages on this planet.".dimmed());
            println!();
            return Ok(());
        }

        let lineage_rows: Vec<LineageRow> = lineages.iter().map(|l| LineageRow::from_lineage_info(l)).collect();
        let table = Table::new(lineage_rows);
        println!("{}", table);
        println!();

        Ok(())
    }

    fn display_energy_budget(&self, energy: &EnergyBudget) {
        println!("{}", "ENERGY BUDGET".magenta().bold());
        
        let energy_info = vec![
            InfoRow::new("Total Available", formatters::format_energy(energy.total_available)),
            InfoRow::new("Generation Rate", format!("{}/tick", formatters::format_energy(energy.generation_rate))),
            InfoRow::new("Consumption Rate", format!("{}/tick", formatters::format_energy(energy.consumption_rate))),
            InfoRow::new("Net Rate", format!("{}/tick", formatters::format_energy(energy.generation_rate - energy.consumption_rate))),
            InfoRow::new("Storage Capacity", formatters::format_energy(energy.storage_capacity)),
            InfoRow::new("Efficiency", format!("{:.1}%", energy.efficiency * 100.0)),
            InfoRow::new("Utilization", formatters::format_progress_bar(
                energy.total_available, energy.storage_capacity, 20
            )),
        ];

        let table = Table::new(energy_info);
        println!("{}", table);
        
        // Energy sources
        if !energy.sources.is_empty() {
            println!("\n{}", "Energy Sources:".yellow());
            for source in &energy.sources {
                println!("  {}: {} ({:.1}% efficiency, {} maintenance)",
                    source.source_type,
                    formatters::format_energy(source.power_output),
                    source.efficiency * 100.0,
                    formatters::format_energy(source.maintenance_cost)
                );
            }
        }
        println!();
    }

    fn display_resource_composition(&self, element_table: &ElementTable, geological_layers: &[GeologicalLayer]) {
        println!("{}", "RESOURCE COMPOSITION".red().bold());
        
        // Key elements
        let key_elements = vec![
            ("Hydrogen", "H"), ("Carbon", "C"), ("Oxygen", "O"), ("Silicon", "Si"),
            ("Iron", "Fe"), ("Copper", "Cu"), ("Silver", "Ag"), ("Gold", "Au"), ("Uranium", "U"),
        ];

        println!("{}", "Key Elements:".green());
        for (name, symbol) in key_elements {
            if let Some(ppm) = element_table.get_element_ppm(symbol) {
                if ppm > 0 {
                    println!("  {}: {}", name, formatters::format_ppm(ppm));
                }
            }
        }
        println!();

        // Geological layers
        if !geological_layers.is_empty() {
            println!("{}", "Geological Layers:".blue());
            for (i, layer) in geological_layers.iter().enumerate() {
                println!("  Layer {}: {:.0}m {} ({:?}, {:.1} kg/m³, Mohs {:.1})",
                    i + 1,
                    layer.thickness,
                    if layer.extractable { "extractable".green() } else { "non-extractable".red() },
                    layer.material_type,
                    layer.bulk_density,
                    layer.hardness
                );
            }
        }
        println!();
    }

    fn display_orbital_data(&self, orbital: &OrbitalData) {
        println!("{}", "ORBITAL MECHANICS".cyan().bold());
        
        let orbital_info = vec![
            InfoRow::new("Semi-major Axis", format!("{:.3} AU", orbital.semi_major_axis)),
            InfoRow::new("Eccentricity", format!("{:.4}", orbital.eccentricity)),
            InfoRow::new("Inclination", format!("{:.2}°", orbital.inclination)),
            InfoRow::new("Orbital Period", format!("{:.1} Earth days", orbital.orbital_period)),
            InfoRow::new("Rotation Period", format!("{:.2} Earth days", orbital.rotation_period)),
            InfoRow::new("Parent Star", formatters::format_uuid_short(&orbital.parent_star)),
        ];

        let table = Table::new(orbital_info);
        println!("{}", table);
        println!();
    }

    fn display_lineage_basics(&self, lineage: &Lineage) {
        println!("{}", "BASIC INFORMATION".yellow().bold());
        
        let basic_info = vec![
            InfoRow::new("ID", formatters::format_uuid_short(&lineage.id)),
            InfoRow::new("Name", lineage.name.clone()),
            InfoRow::new("Generation", formatters::format_generation(lineage.generation)),
            InfoRow::new("Code Hash", format!("{}...", &lineage.code_hash[0..16])),
            InfoRow::new("Current Fitness", formatters::format_fitness(lineage.current_fitness)),
            InfoRow::new("Parameter Count", formatters::format_number(lineage.parameter_count)),
            InfoRow::new("Planet Residence", formatters::format_uuid_short(&lineage.planet_residence)),
            InfoRow::new("Created", lineage.created_at.format("%Y-%m-%d %H:%M UTC").to_string()),
            InfoRow::new("Last Activity", lineage.last_activity.format("%Y-%m-%d %H:%M UTC").to_string()),
        ];

        let table = Table::new(basic_info);
        println!("{}", table);
        println!();
    }

    fn display_fitness_history(&self, history: &[FitnessRecord], time_range: Option<&str>) {
        println!("{}", "FITNESS HISTORY".green().bold());
        
        if history.is_empty() {
            println!("{}", "No fitness records available.".dimmed());
            println!();
            return;
        }

        // Apply time range filter if specified
        let filtered_history: Vec<&FitnessRecord> = if let Some(_range) = time_range {
            // For now, just show recent records
            history.iter().rev().take(10).collect()
        } else {
            history.iter().rev().take(10).collect()
        };

        for record in filtered_history {
            println!("  Tick {}: {} (entropy: {:.3}, efficiency: {:.3}, cooperation: {:.3})",
                formatters::format_number(record.tick),
                formatters::format_fitness(record.fitness),
                record.entropy_cost,
                record.resource_efficiency,
                record.cooperation_score
            );
        }
        
        if let (Some(first), Some(last)) = (history.first(), history.last()) {
            let improvement = last.fitness - first.fitness;
            let improvement_color = if improvement > 0.0 { "green" } else if improvement < 0.0 { "red" } else { "yellow" };
            println!("\n  {} Fitness Change: {}{:.3}",
                "📈".green(),
                if improvement > 0.0 { "+" } else { "" },
                formatters::format_table_cell(&format!("{:.3}", improvement), Some(improvement_color))
            );
        }
        println!();
    }

    fn display_code_evolution(&self, lineage: &Lineage) {
        println!("{}", "CODE EVOLUTION".blue().bold());
        
        let code_info = vec![
            InfoRow::new("Code Hash", lineage.code_hash.clone()),
            InfoRow::new("Generation", lineage.generation.to_string()),
            InfoRow::new("Parameter Count", formatters::format_number(lineage.parameter_count)),
            InfoRow::new("Parent", match lineage.parent_id {
                Some(id) => formatters::format_uuid_short(&id),
                None => "Genesis lineage".dimmed().to_string(),
            }),
            InfoRow::new("Children", format!("{} direct descendants", lineage.children.len())),
        ];

        let table = Table::new(code_info);
        println!("{}", table);
        println!();
    }

    async fn display_genealogy(&self, lineage: &Lineage) -> Result<()> {
        println!("{}", "GENEALOGY".magenta().bold());
        
        // Parent information
        if let Some(parent_id) = lineage.parent_id {
            println!("  {} Parent: {}", "👆".blue(), formatters::format_uuid_short(&parent_id));
        } else {
            println!("  {} Genesis lineage (no parent)", "🌟".yellow());
        }
        
        // Children information
        if !lineage.children.is_empty() {
            println!("  {} Children ({}):", "👇".green(), lineage.children.len());
            for child_id in &lineage.children {
                println!("    • {}", formatters::format_uuid_short(child_id));
            }
        } else {
            println!("  {} No children yet", "🚫".dimmed());
        }
        
        println!();
        Ok(())
    }

    fn display_capabilities_and_achievements(&self, lineage: &Lineage) {
        println!("{}", "CAPABILITIES & ACHIEVEMENTS".cyan().bold());
        
        // Capabilities
        if !lineage.capabilities.is_empty() {
            println!("{}", "Capabilities:".green());
            for capability in &lineage.capabilities {
                println!("  {}: {} (unlocked {})",
                    capability.name,
                    formatters::format_progress_bar(capability.level, 1.0, 15),
                    capability.unlocked_at.format("%Y-%m-%d")
                );
            }
        }
        
        // Achievements
        if !lineage.achievements.is_empty() {
            println!("\n{}", "Achievements:".yellow());
            for achievement in &lineage.achievements {
                println!("  🏆 {} - {} (Tick {})",
                    achievement.name,
                    achievement.description,
                    formatters::format_number(achievement.tick)
                );
            }
        }
        
        if lineage.capabilities.is_empty() && lineage.achievements.is_empty() {
            println!("{}", "No capabilities or achievements yet.".dimmed());
        }
        println!();
    }

    fn display_lineage_resource_usage(&self, usage: &ResourceUsage) {
        println!("{}", "RESOURCE USAGE".red().bold());
        
        let usage_info = vec![
            InfoRow::new("CPU Cycles", formatters::format_number(usage.cpu_cycles)),
            InfoRow::new("Memory", formatters::format_bytes(usage.memory_bytes)),
            InfoRow::new("Energy", formatters::format_energy(usage.energy_joules)),
            InfoRow::new("Bandwidth", formatters::format_bytes(usage.bandwidth_bytes)),
            InfoRow::new("Storage", formatters::format_bytes(usage.storage_bytes)),
        ];

        let table = Table::new(usage_info);
        println!("{}", table);
        println!();
    }

    fn display_system_basics(&self, system: &StarSystem) {
        println!("{}", "BASIC INFORMATION".yellow().bold());
        
        let basic_info = vec![
            InfoRow::new("ID", formatters::format_uuid_short(&system.id)),
            InfoRow::new("Name", system.name.clone()),
            InfoRow::new("Coordinates", formatters::format_coordinates(
                system.coordinates.x, system.coordinates.y, system.coordinates.z
            )),
            InfoRow::new("Age", formatters::format_cosmic_time(system.age * 1e9)),
            InfoRow::new("Metallicity", format!("{:.4}", system.metallicity)),
            InfoRow::new("Planets", format!("{}", system.planets.len())),
            InfoRow::new("Companion Stars", format!("{}", system.companion_stars.len())),
            InfoRow::new("Asteroid Belts", format!("{}", system.asteroid_belts.len())),
        ];

        let table = Table::new(basic_info);
        println!("{}", table);
        println!();
    }

    fn display_stellar_properties(&self, primary: &Star, companions: &[Star]) {
        println!("{}", "STELLAR PROPERTIES".blue().bold());
        
        // Primary star
        println!("{}", "Primary Star:".green());
        let stellar_info = vec![
            InfoRow::new("Name", primary.name.clone()),
            InfoRow::new("Stellar Class", primary.stellar_class.clone()),
            InfoRow::new("Mass", format!("{:.2} M☉", primary.mass)),
            InfoRow::new("Radius", format!("{:.2} R☉", primary.radius)),
            InfoRow::new("Temperature", format!("{:.0} K", primary.temperature)),
            InfoRow::new("Luminosity", format!("{:.2} L☉", primary.luminosity)),
            InfoRow::new("Age", formatters::format_cosmic_time(primary.age * 1e9)),
            InfoRow::new("Metallicity", format!("{:.4}", primary.metallicity)),
        ];

        let table = Table::new(stellar_info);
        println!("{}", table);
        
        // Habitable zone
        println!("\n{}", "Habitable Zone:".cyan());
        println!("  Inner: {:.2} AU", primary.habitable_zone.inner_radius);
        println!("  Optimum: {:.2} AU", primary.habitable_zone.optimum_radius);
        println!("  Outer: {:.2} AU", primary.habitable_zone.outer_radius);
        
        // Companion stars
        if !companions.is_empty() {
            println!("\n{}", "Companion Stars:".truecolor(255, 165, 0));
            for (i, companion) in companions.iter().enumerate() {
                println!("  {}: {} ({}, {:.2} M☉)",
                    i + 1,
                    companion.name,
                    companion.stellar_class,
                    companion.mass
                );
            }
        }
        println!();
    }

    async fn display_orbital_mechanics(&self, system: &StarSystem) -> Result<()> {
        println!("{}", "ORBITAL MECHANICS".magenta().bold());
        
        if system.planets.is_empty() {
            println!("{}", "No planets in this system.".dimmed());
            println!();
            return Ok(());
        }

        println!("{}", "Planetary Orbits:".green());
        for (i, planet_id) in system.planets.iter().enumerate() {
            // In a real implementation, we'd fetch each planet's orbital data
            println!("  Planet {}: {} (detailed orbital data available via planet inspection)",
                i + 1,
                formatters::format_uuid_short(planet_id)
            );
        }
        
        // Asteroid belts
        if !system.asteroid_belts.is_empty() {
            println!("\n{}", "Asteroid Belts:".red());
            for (i, belt) in system.asteroid_belts.iter().enumerate() {
                println!("  Belt {}: {:.2} - {:.2} AU ({:.3} M🜨)",
                    i + 1,
                    belt.inner_radius,
                    belt.outer_radius,
                    belt.total_mass
                );
            }
        }
        println!();

        Ok(())
    }

    fn display_cpu_metrics(&self, metrics: &PerformanceMetrics) {
        println!("{}", "CPU METRICS".green().bold());
        
        let cpu_info = vec![
            InfoRow::new("CPU Usage", formatters::format_cpu_usage(metrics.cpu_usage)),
            InfoRow::new("Average Tick Time", format!("{:.2} ms", metrics.avg_tick_time_ms)),
        ];

        let table = Table::new(cpu_info);
        println!("{}", table);
        println!();
    }

    fn display_memory_metrics(&self, memory: &MemoryUsage) {
        println!("{}", "MEMORY METRICS".blue().bold());
        
        let memory_info = vec![
            InfoRow::new("Used Memory", format!("{:.1} MB", memory.used_mb)),
            InfoRow::new("Available Memory", format!("{:.1} MB", memory.available_mb)),
            InfoRow::new("Usage Percentage", formatters::format_percentage(memory.percentage, 50.0, 80.0)),
            InfoRow::new("Memory Pressure", formatters::format_progress_bar(memory.used_mb, memory.used_mb + memory.available_mb, 20)),
        ];

        let table = Table::new(memory_info);
        println!("{}", table);
        println!();
    }

    fn display_network_metrics(&self, network: &NetworkIO) {
        println!("{}", "NETWORK METRICS".cyan().bold());
        
        let network_info = vec![
            InfoRow::new("Bytes Sent", formatters::format_bytes(network.bytes_sent)),
            InfoRow::new("Bytes Received", formatters::format_bytes(network.bytes_received)),
            InfoRow::new("Packets Sent", formatters::format_number(network.packets_sent)),
            InfoRow::new("Packets Received", formatters::format_number(network.packets_received)),
        ];

        let table = Table::new(network_info);
        println!("{}", table);
        println!();
    }

    fn display_simulation_metrics(&self, status: &SimulationStatus) {
        println!("{}", "SIMULATION METRICS".magenta().bold());
        
        let sim_info = vec![
            InfoRow::new("Current Tick", formatters::format_number(status.tick)),
            InfoRow::new("Updates Per Second", format!("{:.1} UPS", status.ups)),
            InfoRow::new("Cosmic Era", format!("{:?}", status.cosmic_era)),
            InfoRow::new("Mean Entropy", format!("{:.3}", status.mean_entropy)),
            InfoRow::new("Active Lineages", formatters::format_number(status.lineage_count as u64)),
            InfoRow::new("Total Planets", formatters::format_number(status.planet_count as u64)),
        ];

        let table = Table::new(sim_info);
        println!("{}", table);
        println!();
    }
}

#[derive(Tabled)]
struct InfoRow {
    #[tabled(rename = "Property")]
    property: String,
    #[tabled(rename = "Value")]
    value: String,
}

impl InfoRow {
    fn new(property: &str, value: String) -> Self {
        Self {
            property: property.to_string(),
            value,
        }
    }
}

#[derive(Tabled)]
struct LineageRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Fitness")]
    fitness: String,
    #[tabled(rename = "Population")]
    population: String,
    #[tabled(rename = "Activity")]
    activity: String,
}

impl LineageRow {
    fn from_lineage_info(lineage: &LineageInfo) -> Self {
        Self {
            id: formatters::format_uuid_short(&lineage.id),
            name: lineage.name.clone(),
            fitness: formatters::format_fitness(lineage.fitness),
            population: formatters::format_number(lineage.population),
            activity: formatters::format_progress_bar(lineage.activity_level, 1.0, 10),
        }
    }
}