use anyhow::Result;
use colored::*;
use tabled::{Table, Tabled};

use crate::rpc::{RpcClient, PlanetFilter};
use crate::data_models::{Planet, PlanetClass};
use crate::formatters;

pub struct ListPlanetsCommand {
    rpc_client: RpcClient,
}

impl ListPlanetsCommand {
    pub fn new(socket_path: &str) -> Result<Self> {
        Ok(Self {
            rpc_client: RpcClient::new(socket_path),
        })
    }

    pub async fn execute(
        &self,
        class_filter: Option<&str>,
        min_habitability: Option<f64>,
        format: &str,
        sort_field: &str,
        limit: Option<usize>,
    ) -> Result<()> {
        // Parse class filter
        let planet_class = if let Some(class_str) = class_filter {
            Some(match class_str.to_uppercase().as_str() {
                "E" => PlanetClass::E,
                "D" => PlanetClass::D,
                "I" => PlanetClass::I,
                "T" => PlanetClass::T,
                "G" => PlanetClass::G,
                _ => {
                    anyhow::bail!("Invalid planet class '{}'. Valid options: E, D, I, T, G", class_str);
                }
            })
        } else {
            None
        };

        // Create filter
        let filter = if planet_class.is_some() || min_habitability.is_some() || limit.is_some() {
            Some(PlanetFilter {
                class: planet_class,
                min_habitability,
                sort_by: Some(sort_field.to_string()),
                limit,
            })
        } else {
            None
        };

        // Fetch planets
        let mut planets = self.rpc_client.list_planets(filter).await?;

        // Apply sorting
        self.sort_planets(&mut planets, sort_field);

        // Apply limit if not already applied by filter
        if let Some(limit_count) = limit {
            planets.truncate(limit_count);
        }

        // Display results
        match format {
            "json" => {
                println!("{}", serde_json::to_string_pretty(&planets)?);
            }
            "yaml" => {
                println!("{}", serde_yaml::to_string(&planets)?);
            }
            "csv" => {
                self.display_csv(&planets);
            }
            _ => {
                self.display_table(&planets);
            }
        }

        Ok(())
    }

    fn sort_planets(&self, planets: &mut [Planet], sort_field: &str) {
        match sort_field {
            "name" => planets.sort_by(|a, b| a.name.cmp(&b.name)),
            "class" => planets.sort_by(|a, b| format!("{:?}", a.class).cmp(&format!("{:?}", b.class))),
            "habitability" => planets.sort_by(|a, b| b.habitability_score.partial_cmp(&a.habitability_score).unwrap_or(std::cmp::Ordering::Equal)),
            "population" => planets.sort_by(|a, b| b.current_population.cmp(&a.current_population)),
            "temperature" => planets.sort_by(|a, b| a.environment.temp_celsius.partial_cmp(&b.environment.temp_celsius).unwrap_or(std::cmp::Ordering::Equal)),
            "mass" => planets.sort_by(|a, b| b.element_table.total_mass.partial_cmp(&a.element_table.total_mass).unwrap_or(std::cmp::Ordering::Equal)),
            "energy" => planets.sort_by(|a, b| b.energy_budget.total_available.partial_cmp(&a.energy_budget.total_available).unwrap_or(std::cmp::Ordering::Equal)),
            _ => {
                // Default to habitability if unknown sort field
                planets.sort_by(|a, b| b.habitability_score.partial_cmp(&a.habitability_score).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
    }

    fn display_table(&self, planets: &[Planet]) {
        if planets.is_empty() {
            println!("{}", "No planets found matching the criteria.".yellow());
            return;
        }

        println!("{}", format!("PLANETS ({} found)", planets.len()).cyan().bold());
        println!("{}", "=".repeat(80).cyan());
        println!();

        let planet_rows: Vec<PlanetRow> = planets.iter().map(|p| PlanetRow::from_planet(p)).collect();
        let table = Table::new(planet_rows);
        println!("{}", table);
        
        println!();
        self.display_summary_stats(planets);
    }

    fn display_csv(&self, planets: &[Planet]) {
        // CSV header
        println!("Name,Class,Habitability,Population,Temperature,Pressure,Water,Oxygen,Energy,Coordinates");
        
        // CSV rows
        for planet in planets {
            println!(
                "{},{},{:.3},{},{:.1},{:.2},{:.2},{:.3},{},\"{}\"",
                planet.name,
                planet.class,
                planet.habitability_score,
                planet.current_population,
                planet.environment.temp_celsius,
                planet.environment.atmos_pressure,
                planet.environment.liquid_water,
                planet.environment.atmos_oxygen,
                formatters::format_energy(planet.energy_budget.total_available),
                formatters::format_coordinates(
                    planet.coordinates.x,
                    planet.coordinates.y,
                    planet.coordinates.z
                )
            );
        }
    }

    fn display_summary_stats(&self, planets: &[Planet]) {
        println!("{}", "SUMMARY STATISTICS".yellow().bold());
        
        let class_counts = self.calculate_class_distribution(planets);
        let habitability_stats = self.calculate_habitability_stats(planets);
        let population_stats = self.calculate_population_stats(planets);

        // Class distribution
        println!("{}", "Class Distribution:".green());
        for (class, count) in &class_counts {
            let percentage = (*count as f64 / planets.len() as f64) * 100.0;
            println!("  {}: {} ({:.1}%)", class, count, percentage);
        }
        println!();

        // Habitability statistics
        println!("{}", "Habitability Statistics:".blue());
        println!("  Average: {:.3}", habitability_stats.average);
        println!("  Median: {:.3}", habitability_stats.median);
        println!("  Highly Habitable (>0.8): {}", habitability_stats.highly_habitable);
        println!("  Moderately Habitable (0.5-0.8): {}", habitability_stats.moderately_habitable);
        println!("  Barely Habitable (<0.5): {}", habitability_stats.barely_habitable);
        println!();

        // Population statistics
        println!("{}", "Population Statistics:".magenta());
        println!("  Total Population: {}", formatters::format_number(population_stats.total));
        println!("  Average per Planet: {}", formatters::format_number(population_stats.average));
        println!("  Most Populated: {}", formatters::format_number(population_stats.max));
        println!("  Inhabited Planets: {} ({:.1}%)", 
            population_stats.inhabited_count,
            (population_stats.inhabited_count as f64 / planets.len() as f64) * 100.0
        );
    }

    fn calculate_class_distribution(&self, planets: &[Planet]) -> Vec<(String, usize)> {
        use std::collections::HashMap;
        
        let mut counts = HashMap::new();
        for planet in planets {
            *counts.entry(format!("{}", planet.class)).or_insert(0) += 1;
        }
        
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    fn calculate_habitability_stats(&self, planets: &[Planet]) -> HabitabilityStats {
        let mut scores: Vec<f64> = planets.iter().map(|p| p.habitability_score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let average = scores.iter().sum::<f64>() / scores.len() as f64;
        let median = if scores.len() % 2 == 0 {
            (scores[scores.len() / 2 - 1] + scores[scores.len() / 2]) / 2.0
        } else {
            scores[scores.len() / 2]
        };
        
        let highly_habitable = scores.iter().filter(|&&s| s > 0.8).count();
        let moderately_habitable = scores.iter().filter(|&&s| s >= 0.5 && s <= 0.8).count();
        let barely_habitable = scores.iter().filter(|&&s| s < 0.5).count();
        
        HabitabilityStats {
            average,
            median,
            highly_habitable,
            moderately_habitable,
            barely_habitable,
        }
    }

    fn calculate_population_stats(&self, planets: &[Planet]) -> PopulationStats {
        let populations: Vec<u64> = planets.iter().map(|p| p.current_population).collect();
        
        let total = populations.iter().sum();
        let average = if !populations.is_empty() { total / populations.len() as u64 } else { 0 };
        let max = populations.iter().max().copied().unwrap_or(0);
        let inhabited_count = populations.iter().filter(|&&p| p > 0).count();
        
        PopulationStats {
            total,
            average,
            max,
            inhabited_count,
        }
    }
}

#[derive(Tabled)]
struct PlanetRow {
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Class")]
    class: String,
    #[tabled(rename = "Habitability")]
    habitability: String,
    #[tabled(rename = "Population")]
    population: String,
    #[tabled(rename = "Temperature")]
    temperature: String,
    #[tabled(rename = "Atmosphere")]
    atmosphere: String,
    #[tabled(rename = "Lineages")]
    lineages: String,
    #[tabled(rename = "Energy")]
    energy: String,
}

impl PlanetRow {
    fn from_planet(planet: &Planet) -> Self {
        Self {
            name: planet.name.clone(),
            class: format!("{}", planet.class),
            habitability: formatters::format_fitness(planet.habitability_score),
            population: if planet.current_population > 0 {
                formatters::format_number(planet.current_population)
            } else {
                "Uninhabited".dimmed().to_string()
            },
            temperature: formatters::format_temperature(planet.environment.temp_celsius),
            atmosphere: format!(
                "{} | {}",
                formatters::format_pressure(planet.environment.atmos_pressure),
                format!("{:.1}% O₂", planet.environment.atmos_oxygen * 100.0)
            ),
            lineages: if planet.active_lineages.is_empty() {
                "None".dimmed().to_string()
            } else {
                format!("{}", planet.active_lineages.len())
            },
            energy: formatters::format_energy(planet.energy_budget.total_available),
        }
    }
}

struct HabitabilityStats {
    average: f64,
    median: f64,
    highly_habitable: usize,
    moderately_habitable: usize,
    barely_habitable: usize,
}

struct PopulationStats {
    total: u64,
    average: u64,
    max: u64,
    inhabited_count: usize,
}