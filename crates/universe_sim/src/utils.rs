//! Utility functions for the simulation

use crate::types::*;
use crate::Result;
use std::path::Path;

/// Serialize data using rkyv
pub fn serialize_to_bytes<T>(data: &T) -> Result<Vec<u8>>
where
    T: rkyv::Serialize<rkyv::ser::serializers::AllocSerializer<256>>,
{
    use rkyv::ser::Serializer;
    
    let mut serializer = rkyv::ser::serializers::AllocSerializer::<256>::default();
    serializer.serialize_value(data)
        .map_err(|e| crate::SimError::SerializationError(e.to_string()))?;
    
    Ok(serializer.into_serializer().into_inner().to_vec())
}

/// Deserialize data using rkyv
pub fn deserialize_from_bytes<T>(bytes: &[u8]) -> Result<&T::Archived>
where
    T: rkyv::Archive,
    T::Archived: for<'a> rkyv::CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>,
{
    let archived = rkyv::check_archived_root::<T>(bytes)
        .map_err(|e| crate::SimError::SerializationError(e.to_string()))?;
    
    Ok(archived)
}

/// Calculate distance between two 2D coordinates on a toroidal grid
pub fn toroidal_distance(a: Coord2D, b: Coord2D, grid_size: (u32, u32)) -> f64 {
    a.toroidal_distance_to(&b, grid_size)
}

/// Linear interpolation
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Clamp value between min and max
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Calculate sigmoid function for smooth transitions
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Generate a hash for code similarity comparison
pub fn calculate_code_hash(code: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    code.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Calculate Hamming distance between two strings
pub fn hamming_distance(a: &str, b: &str) -> usize {
    if a.len() != b.len() {
        return usize::MAX; // Invalid comparison
    }
    
    a.chars()
        .zip(b.chars())
        .map(|(c1, c2)| if c1 == c2 { 0 } else { 1 })
        .sum()
}

/// Convert years to simulation ticks
pub fn years_to_ticks(years: f64, years_per_tick: f64) -> u64 {
    (years / years_per_tick).round() as u64
}

/// Convert ticks to years
pub fn ticks_to_years(ticks: u64, years_per_tick: f64) -> f64 {
    ticks as f64 * years_per_tick
}

/// Format large numbers with scientific notation
pub fn format_scientific(value: f64) -> String {
    if value.abs() >= 1e6 || (value.abs() < 1e-3 && value != 0.0) {
        format!("{:.2e}", value)
    } else {
        format!("{:.2}", value)
    }
}

/// Format time duration for human reading
pub fn format_time_duration(years: f64) -> String {
    if years < 1e3 {
        format!("{:.0} years", years)
    } else if years < 1e6 {
        format!("{:.1} thousand years", years / 1e3)
    } else if years < 1e9 {
        format!("{:.1} million years", years / 1e6)
    } else if years < 1e12 {
        format!("{:.1} billion years", years / 1e9)
    } else {
        format!("{:.1} trillion years", years / 1e12)
    }
}

/// Weighted random selection
pub fn weighted_random_select<'a, T>(
    items: &'a [(T, f64)],
    rng: &mut impl rand::Rng,
) -> Option<&'a T> {
    let total_weight: f64 = items.iter().map(|(_, w)| w).sum();
    
    if total_weight <= 0.0 {
        return None;
    }
    
    let mut random_weight = rng.gen::<f64>() * total_weight;
    
    for (item, weight) in items {
        random_weight -= weight;
        if random_weight <= 0.0 {
            return Some(item);
        }
    }
    
    // Fallback to last item
    items.last().map(|(item, _)| item)
}

/// Safe file operations
pub fn safe_file_write<P: AsRef<Path>>(path: P, data: &[u8]) -> Result<()> {
    let path = path.as_ref();
    
    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Write to temporary file first
    let temp_path = path.with_extension("tmp");
    std::fs::write(&temp_path, data)?;
    
    // Atomic rename
    std::fs::rename(temp_path, path)?;
    
    Ok(())
}

/// Safe file reading with size limits
pub fn safe_file_read<P: AsRef<Path>>(path: P, max_size: usize) -> Result<Vec<u8>> {
    let path = path.as_ref();
    
    // Check file size first
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > max_size as u64 {
        return Err(crate::SimError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("File too large: {} bytes", metadata.len()),
        )));
    }
    
    std::fs::read(path).map_err(Into::into)
}

/// Log-normal distribution sampling
pub fn sample_log_normal(mean: f64, std_dev: f64, rng: &mut impl rand::Rng) -> f64 {
    use rand_distr::{LogNormal, Distribution};
    
    let log_normal = LogNormal::new(mean, std_dev).unwrap();
    log_normal.sample(rng)
}

/// Calculate planet escape velocity
pub fn escape_velocity(mass_kg: f64, radius_m: f64) -> f64 {
    use crate::constants::physics::G;
    (2.0 * G * mass_kg / radius_m).sqrt()
}

/// Calculate stellar lifetime from mass
pub fn stellar_lifetime_years(mass_solar: f64) -> f64 {
    // Main sequence lifetime ∝ M^-2.5
    let solar_lifetime = 10e9; // 10 billion years
    solar_lifetime * mass_solar.powf(-2.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }
    
    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }
    
    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance("hello", "hello"), 0);
        assert_eq!(hamming_distance("hello", "world"), 4);
        assert_eq!(hamming_distance("abc", "def"), 3);
    }
    
    #[test]
    fn test_years_to_ticks() {
        assert_eq!(years_to_ticks(1_000_000.0, 1_000_000.0), 1);
        assert_eq!(years_to_ticks(2_000_000.0, 1_000_000.0), 2);
    }
}