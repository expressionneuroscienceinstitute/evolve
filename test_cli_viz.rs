// Standalone test of the ASCII visualization
use std::collections::HashMap;

// Simple 2D density field for testing
struct DensityField {
    resolution: usize,
    box_size: f64,
    data: Vec<Vec<f64>>,
}

impl DensityField {
    fn new(resolution: usize, box_size: f64) -> Self {
        let mut data = vec![vec![0.0; resolution]; resolution];
        
        // Generate realistic cosmic web structure
        for i in 0..resolution {
            for j in 0..resolution {
                let x = (i as f64 / resolution as f64 - 0.5) * 2.0;
                let y = (j as f64 / resolution as f64 - 0.5) * 2.0;
                
                // Create filamentary structure
                let filament1 = (-(x * x + y * y) * 5.0).exp();
                let filament2 = (-(x * x + (y - 0.3) * (y - 0.3)) * 5.0).exp();
                let filament3 = (-((x - 0.3) * (x - 0.3) + y * y) * 5.0).exp();
                
                // Add some cluster peaks
                let cluster1 = (-((x - 0.3).powi(2) + y.powi(2)) * 20.0).exp();
                let cluster2 = (-((x + 0.3).powi(2) + (y - 0.3).powi(2)) * 20.0).exp();
                let cluster3 = (-(x.powi(2) + (y + 0.4).powi(2)) * 25.0).exp();
                
                // Combine to create cosmic web
                data[i][j] = 1.0 + 5.0 * (filament1 + filament2 + filament3) 
                              + 20.0 * (cluster1 + cluster2 + cluster3);
            }
        }
        
        Self {
            resolution,
            box_size,
            data,
        }
    }
    
    fn sample_at(&self, x: f64, y: f64) -> f64 {
        let i = ((x / self.box_size + 0.5) * self.resolution as f64) as usize;
        let j = ((y / self.box_size + 0.5) * self.resolution as f64) as usize;
        
        if i < self.resolution && j < self.resolution {
            self.data[i][j]
        } else {
            0.0
        }
    }
}

fn render_density_map(width: usize, height: usize, zoom: f64) -> String {
    let density_field = DensityField::new(128, 100.0);
    
    // ASCII characters for different density levels
    let density_chars = [' ', '·', '∙', ':', '•', '○', '◉', '◎', '◈', '■', '█'];
    
    let view_size = 100.0 / zoom;
    let min_x = -view_size / 2.0;
    let max_x = view_size / 2.0;
    let min_y = -view_size / 2.0;
    let max_y = view_size / 2.0;
    
    // First pass: Calculate min/max for normalization
    let mut min_density = f64::MAX;
    let mut max_density = f64::MIN;
    let mut density_samples = vec![vec![0.0; width]; height];
    
    for y in 0..height {
        for x in 0..width {
            let world_x = min_x + (x as f64 / width as f64) * (max_x - min_x);
            let world_y = min_y + (y as f64 / height as f64) * (max_y - min_y);
            
            let density = density_field.sample_at(world_x, world_y);
            density_samples[y][x] = density;
            
            if density > 0.0 {
                min_density = min_density.min(density);
                max_density = max_density.max(density);
            }
        }
    }
    
    // Use logarithmic scaling
    let log_min = if min_density > 0.0 { min_density.ln() } else { 0.0 };
    let log_max = max_density.ln();
    let log_range = log_max - log_min;
    
    let mut output = String::new();
    
    // Header
    output.push_str(&format!("╔{}╗\n", "═".repeat(width + 20)));
    output.push_str(&format!("║ Cosmic Web Structure - Density Map (Zoom: {}x) {}║\n", 
        zoom as u8, " ".repeat(width - 35)));
    output.push_str(&format!("╠{}╣\n", "═".repeat(width + 20)));
    
    // Render the map
    for y in 0..height {
        output.push_str("║ ");
        for x in 0..width {
            let density = density_samples[y][x];
            
            if density > 0.0 {
                let log_density = density.ln();
                let normalized = ((log_density - log_min) / log_range).clamp(0.0, 1.0);
                let enhanced = normalized.powf(0.5); // Square root for better contrast
                let char_idx = (enhanced * (density_chars.len() - 1) as f64) as usize;
                output.push(density_chars[char_idx.min(density_chars.len() - 1)]);
            } else {
                output.push(' ');
            }
        }
        output.push_str(" ║\n");
    }
    
    // Footer with legend
    output.push_str(&format!("╠{}╣\n", "═".repeat(width + 20)));
    output.push_str(&format!("║ Legend: [' '=void] ['·'=low density] ['•'=medium] ['◉'=high] ['█'=cluster] {}║\n",
        " ".repeat(width - 58)));
    output.push_str(&format!("║ Scale: {:.1} Mpc/char {}║\n", 
        view_size / width as f64, " ".repeat(width - 10)));
    output.push_str(&format!("╚{}╝\n", "═".repeat(width + 20)));
    
    output
}

fn main() {
    println!("Universe Simulation - ASCII Visualization Demo\n");
    
    // Show cosmic web at different zoom levels
    println!("=== Large Scale Structure (100 Mpc view) ===");
    println!("{}", render_density_map(80, 30, 1.0));
    
    println!("\n=== Galaxy Cluster Region (33 Mpc view) ===");
    println!("{}", render_density_map(80, 30, 3.0));
    
    println!("\n=== Detailed View (10 Mpc view) ===");
    println!("{}", render_density_map(80, 30, 10.0));
}