use byte_unit::{Byte, ByteUnit};
use humantime::format_duration;
use std::time::Duration;

/// Format bytes with appropriate units (B, KB, MB, GB, TB)
pub fn format_bytes(bytes: u64) -> String {
    let byte = Byte::from_bytes(bytes as u128);
    byte.get_appropriate_unit(true).to_string()
}

/// Format a duration in human-readable format
pub fn format_duration_human(seconds: u64) -> String {
    let duration = Duration::from_secs(seconds);
    format_duration(duration).to_string()
}

/// Format a large number with thousand separators
pub fn format_number(num: u64) -> String {
    let mut result = num.to_string();
    let mut reversed = result.chars().rev().collect::<Vec<_>>();
    
    // Insert commas every 3 digits from the right
    let mut formatted = Vec::new();
    for (i, ch) in reversed.iter().enumerate() {
        if i > 0 && i % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(*ch);
    }
    
    formatted.reverse();
    formatted.into_iter().collect()
}

/// Format a percentage with color coding
pub fn format_percentage(value: f64, good_threshold: f64, warning_threshold: f64) -> String {
    use colored::*;
    
    let formatted = format!("{:.1}%", value);
    
    if value <= good_threshold {
        formatted.green().to_string()
    } else if value <= warning_threshold {
        formatted.yellow().to_string()
    } else {
        formatted.red().to_string()
    }
}

/// Format a fitness score with appropriate coloring
pub fn format_fitness(fitness: f64) -> String {
    use colored::*;
    
    let formatted = format!("{:.3}", fitness);
    
    if fitness >= 0.8 {
        formatted.green().bold().to_string()
    } else if fitness >= 0.6 {
        formatted.yellow().to_string()
    } else if fitness >= 0.4 {
        formatted.truecolor(255, 165, 0).to_string()
    } else {
        formatted.red().to_string()
    }
}

/// Format coordinates in a readable format
pub fn format_coordinates(x: f64, y: f64, z: f64) -> String {
    format!("({:.2}, {:.2}, {:.2})", x, y, z)
}

/// Format energy values with appropriate scientific notation
pub fn format_energy(joules: f64) -> String {
    if joules >= 1e15 {
        format!("{:.2} PJ", joules / 1e15)
    } else if joules >= 1e12 {
        format!("{:.2} TJ", joules / 1e12)
    } else if joules >= 1e9 {
        format!("{:.2} GJ", joules / 1e9)
    } else if joules >= 1e6 {
        format!("{:.2} MJ", joules / 1e6)
    } else if joules >= 1e3 {
        format!("{:.2} kJ", joules / 1e3)
    } else {
        format!("{:.2} J", joules)
    }
}

/// Format mass values with appropriate units
pub fn format_mass(kg: f64) -> String {
    const EARTH_MASS: f64 = 5.972e24;
    const SOLAR_MASS: f64 = 1.989e30;
    
    if kg >= SOLAR_MASS {
        format!("{:.3} M☉", kg / SOLAR_MASS)
    } else if kg >= EARTH_MASS {
        format!("{:.3} M🜨", kg / EARTH_MASS)
    } else if kg >= 1e9 {
        format!("{:.2} Gg", kg / 1e9)
    } else if kg >= 1e6 {
        format!("{:.2} Mg", kg / 1e6)
    } else if kg >= 1e3 {
        format!("{:.2} kg", kg / 1e3)
    } else {
        format!("{:.2} kg", kg)
    }
}

/// Format time in cosmic scales (years, million years, billion years)
pub fn format_cosmic_time(years: f64) -> String {
    if years >= 1e9 {
        format!("{:.2} Gy", years / 1e9)
    } else if years >= 1e6 {
        format!("{:.2} My", years / 1e6)
    } else if years >= 1e3 {
        format!("{:.2} ky", years / 1e3)
    } else {
        format!("{:.0} years", years)
    }
}

/// Format temperature with appropriate units and color coding
pub fn format_temperature(celsius: f64) -> String {
    use colored::*;
    
    let kelvin = celsius + 273.15;
    let formatted = if kelvin > 1000.0 {
        format!("{:.0} K", kelvin)
    } else {
        format!("{:.1}°C", celsius)
    };
    
    // Color coding based on habitability
    if celsius >= -20.0 && celsius <= 35.0 {
        formatted.green().to_string()
    } else if celsius >= -50.0 && celsius <= 80.0 {
        formatted.yellow().to_string()
    } else {
        formatted.red().to_string()
    }
}

/// Format pressure relative to Earth atmospheric pressure
pub fn format_pressure(relative_pressure: f64) -> String {
    use colored::*;
    
    let formatted = format!("{:.2} atm", relative_pressure);
    
    // Color coding based on habitability
    if relative_pressure >= 0.8 && relative_pressure <= 1.2 {
        formatted.green().to_string()
    } else if relative_pressure >= 0.3 && relative_pressure <= 3.0 {
        formatted.yellow().to_string()
    } else {
        formatted.red().to_string()
    }
}

/// Format radiation levels with safety indicators
pub fn format_radiation(sv_per_year: f64) -> String {
    use colored::*;
    
    let formatted = format!("{:.2} Sv/yr", sv_per_year);
    
    if sv_per_year < 0.1 {
        formatted.green().to_string()
    } else if sv_per_year < 1.0 {
        formatted.yellow().to_string()
    } else if sv_per_year < 5.0 {
        formatted.truecolor(255, 165, 0).to_string()
    } else {
        formatted.red().bold().to_string()
    }
}

/// Format parts per million with appropriate scaling
pub fn format_ppm(ppm: u32) -> String {
    if ppm >= 1_000_000 {
        format!("{:.1}%", ppm as f64 / 10_000.0)
    } else if ppm >= 1_000 {
        format!("{:.1}‰", ppm as f64 / 1_000.0)
    } else {
        format!("{} ppm", ppm)
    }
}

/// Format CPU/GPU usage with color coding
pub fn format_cpu_usage(percentage: f64) -> String {
    use colored::*;
    
    let formatted = format!("{:.1}%", percentage);
    
    if percentage < 70.0 {
        formatted.green().to_string()
    } else if percentage < 85.0 {
        formatted.yellow().to_string()
    } else {
        formatted.red().to_string()
    }
}

/// Format a progress bar for various metrics
pub fn format_progress_bar(value: f64, max_value: f64, width: usize) -> String {
    use colored::*;
    
    let percentage = (value / max_value).min(1.0).max(0.0);
    let filled_width = (percentage * width as f64) as usize;
    let empty_width = width - filled_width;
    
    let filled_str = "█".repeat(filled_width);
    let empty_str = "░".repeat(empty_width);
    
    let bar = if percentage < 0.5 {
        format!("{}{}", filled_str.green(), empty_str.dimmed())
    } else if percentage < 0.8 {
        format!("{}{}", filled_str.yellow(), empty_str.dimmed())
    } else {
        format!("{}{}", filled_str.red(), empty_str.dimmed())
    };
    
    format!("{} {:.1}%", bar, percentage * 100.0)
}

/// Format a UUID in a shortened, readable format
pub fn format_uuid_short(uuid: &uuid::Uuid) -> String {
    let uuid_str = uuid.to_string();
    format!("{}...{}", &uuid_str[0..8], &uuid_str[uuid_str.len()-4..])
}

/// Format tick number with cosmic time equivalent
pub fn format_tick_with_time(tick: u64, tick_duration_years: f64) -> String {
    let cosmic_years = tick as f64 * tick_duration_years;
    format!("Tick {} ({})", format_number(tick), format_cosmic_time(cosmic_years))
}

/// Format generation number with genealogy indicators
pub fn format_generation(generation: u64) -> String {
    use colored::*;
    
    let formatted = format!("Gen {}", generation);
    
    if generation < 100 {
        formatted.cyan().to_string()
    } else if generation < 1000 {
        formatted.yellow().to_string()
    } else {
        formatted.magenta().bold().to_string()
    }
}

/// Format a table cell with optional color
pub fn format_table_cell(value: &str, color: Option<&str>) -> String {
    use colored::*;
    
    match color {
        Some("red") => value.red().to_string(),
        Some("green") => value.green().to_string(),
        Some("yellow") => value.yellow().to_string(),
        Some("blue") => value.blue().to_string(),
        Some("cyan") => value.cyan().to_string(),
        Some("magenta") => value.magenta().to_string(),
        Some("orange") => value.truecolor(255, 165, 0).to_string(),
        _ => value.to_string(),
    }
}