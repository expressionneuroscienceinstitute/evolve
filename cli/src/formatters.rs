use anyhow::Result;
use comfy_table::{Table, Cell, Attribute, Color, ContentArrangement};
use serde::Serialize;

/// Format data as a pretty table
pub fn format_table<T: TableFormattable>(items: &[T], headers: &[&str]) -> String {
    let mut table = Table::new();
    
    // Configure table style
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.load_preset(comfy_table::presets::UTF8_FULL);
    
    // Add headers
    let mut header_cells = vec![];
    for header in headers {
        header_cells.push(
            Cell::new(header)
                .add_attribute(Attribute::Bold)
                .fg(Color::Cyan)
        );
    }
    table.set_header(header_cells);
    
    // Add rows
    for item in items {
        table.add_row(item.to_row());
    }
    
    table.to_string()
}

/// Trait for objects that can be formatted as table rows
pub trait TableFormattable {
    fn to_row(&self) -> Vec<String>;
}

/// Format data as JSON
pub fn format_json<T: Serialize>(data: &T, pretty: bool) -> Result<String> {
    if pretty {
        Ok(serde_json::to_string_pretty(data)?)
    } else {
        Ok(serde_json::to_string(data)?)
    }
}

/// Format data as YAML
pub fn format_yaml<T: Serialize>(data: &T) -> Result<String> {
    Ok(serde_yaml::to_string(data)?)
}

/// Format data as TOML
pub fn format_toml<T: Serialize>(data: &T) -> Result<String> {
    Ok(toml::to_string_pretty(data)?)
}

/// Format large numbers with appropriate units
pub fn format_number(n: f64) -> String {
    if n.abs() >= 1e15 {
        format!("{:.2}P", n / 1e15)
    } else if n.abs() >= 1e12 {
        format!("{:.2}T", n / 1e12)
    } else if n.abs() >= 1e9 {
        format!("{:.2}G", n / 1e9)
    } else if n.abs() >= 1e6 {
        format!("{:.2}M", n / 1e6)
    } else if n.abs() >= 1e3 {
        format!("{:.2}k", n / 1e3)
    } else {
        format!("{:.2}", n)
    }
}

/// Format time in appropriate units
pub fn format_time(seconds: f64) -> String {
    const YEAR_SECONDS: f64 = 365.25 * 24.0 * 3600.0;
    
    if seconds >= 1e9 * YEAR_SECONDS {
        format!("{:.2} Gyr", seconds / (1e9 * YEAR_SECONDS))
    } else if seconds >= 1e6 * YEAR_SECONDS {
        format!("{:.2} Myr", seconds / (1e6 * YEAR_SECONDS))
    } else if seconds >= 1e3 * YEAR_SECONDS {
        format!("{:.2} kyr", seconds / (1e3 * YEAR_SECONDS))
    } else if seconds >= YEAR_SECONDS {
        format!("{:.2} yr", seconds / YEAR_SECONDS)
    } else if seconds >= 86400.0 {
        format!("{:.2} days", seconds / 86400.0)
    } else if seconds >= 3600.0 {
        format!("{:.2} hours", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.2} min", seconds / 60.0)
    } else {
        format!("{:.2} s", seconds)
    }
}

/// Format distance in appropriate units
pub fn format_distance(meters: f64) -> String {
    const PARSEC: f64 = 3.0857e16; // meters
    const LIGHT_YEAR: f64 = 9.461e15; // meters
    const AU: f64 = 1.496e11; // meters
    
    if meters >= 1e9 * PARSEC {
        format!("{:.2} Gpc", meters / (1e9 * PARSEC))
    } else if meters >= 1e6 * PARSEC {
        format!("{:.2} Mpc", meters / (1e6 * PARSEC))
    } else if meters >= 1e3 * PARSEC {
        format!("{:.2} kpc", meters / (1e3 * PARSEC))
    } else if meters >= PARSEC {
        format!("{:.2} pc", meters / PARSEC)
    } else if meters >= LIGHT_YEAR {
        format!("{:.2} ly", meters / LIGHT_YEAR)
    } else if meters >= AU {
        format!("{:.2} AU", meters / AU)
    } else if meters >= 1e9 {
        format!("{:.2} Gm", meters / 1e9)
    } else if meters >= 1e6 {
        format!("{:.2} Mm", meters / 1e6)
    } else if meters >= 1e3 {
        format!("{:.2} km", meters / 1e3)
    } else {
        format!("{:.2} m", meters)
    }
}

/// Format mass in appropriate units
pub fn format_mass(kg: f64) -> String {
    const SOLAR_MASS: f64 = 1.989e30; // kg
    const EARTH_MASS: f64 = 5.972e24; // kg
    
    if kg >= SOLAR_MASS {
        format!("{:.2} M☉", kg / SOLAR_MASS)
    } else if kg >= EARTH_MASS {
        format!("{:.2} M⊕", kg / EARTH_MASS)
    } else if kg >= 1e12 {
        format!("{:.2} Tg", kg / 1e9)
    } else if kg >= 1e9 {
        format!("{:.2} Gg", kg / 1e6)
    } else if kg >= 1e6 {
        format!("{:.2} Mg", kg / 1e3)
    } else if kg >= 1e3 {
        format!("{:.2} t", kg / 1e3)
    } else {
        format!("{:.2} kg", kg)
    }
}

/// Progress bar for terminal output
pub struct ProgressBar {
    width: usize,
    prefix: String,
}

impl ProgressBar {
    pub fn new(width: usize, prefix: String) -> Self {
        Self { width, prefix }
    }
    
    pub fn render(&self, progress: f64) -> String {
        let filled = (progress * self.width as f64) as usize;
        let empty = self.width - filled;
        
        format!("{} [{}{}] {:.1}%",
            self.prefix,
            "█".repeat(filled),
            "░".repeat(empty),
            progress * 100.0
        )
    }
}