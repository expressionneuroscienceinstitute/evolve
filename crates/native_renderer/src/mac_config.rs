// Mac-specific configuration and optimization settings
// Provides hardware detection and performance tuning for Mac GPUs

use std::env;
use std::process::Command;

/// Mac GPU types for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MacGpuType {
    Intel,
    AppleSilicon,
    Discrete,
    Unknown,
}

/// Performance modes for Mac optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MacPerformanceMode {
    Quality,     // Maximum quality, lower performance
    Balanced,    // Balanced quality and performance
    Performance, // Maximum performance, lower quality
}

/// Mac-specific configuration for renderer optimization
#[derive(Debug, Clone)]
pub struct MacConfig {
    pub gpu_type: MacGpuType,
    pub retina_scale: f32,
    pub performance_mode: MacPerformanceMode,
    pub max_particles: usize,
    pub target_fps: f32,
    pub enable_touch_bar: bool,
    pub enable_metal_optimizations: bool,
    pub buffer_pool_size_mb: usize,
    pub shader_optimization_level: u32,
}

impl Default for MacConfig {
    fn default() -> Self {
        Self {
            gpu_type: MacGpuType::Unknown,
            retina_scale: 1.0,
            performance_mode: MacPerformanceMode::Balanced,
            max_particles: 400_000, // Conservative default
            target_fps: 60.0,
            enable_touch_bar: false,
            enable_metal_optimizations: true,
            buffer_pool_size_mb: 128,
            shader_optimization_level: 1,
        }
    }
}

impl MacConfig {
    /// Detect Mac hardware and create optimized configuration
    pub fn detect_and_optimize() -> Self {
        let mut config = Self::default();
        
        // Detect GPU type
        config.gpu_type = Self::detect_gpu_type();
        
        // Detect Retina display
        config.retina_scale = Self::detect_retina_scale();
        
        // Set performance mode based on hardware
        config.performance_mode = Self::optimize_performance_mode(&config);
        
        // Adjust particle limits based on GPU
        config.max_particles = Self::calculate_max_particles(&config);
        
        // Enable Touch Bar if available
        config.enable_touch_bar = Self::detect_touch_bar();
        
        // Adjust buffer pool size based on available memory
        config.buffer_pool_size_mb = Self::calculate_buffer_pool_size(&config);
        
        config
    }
    
    /// Detect Mac GPU type using system information
    fn detect_gpu_type() -> MacGpuType {
        // Try to get GPU info from system_profiler
        if let Ok(output) = Command::new("system_profiler")
            .args(&["SPDisplaysDataType", "-json"])
            .output() 
        {
            if let Ok(json_str) = String::from_utf8(output.stdout) {
                // Simple parsing for GPU detection
                if json_str.contains("Apple M") || json_str.contains("Apple Silicon") {
                    return MacGpuType::AppleSilicon;
                } else if json_str.contains("Intel") {
                    return MacGpuType::Intel;
                } else if json_str.contains("AMD") || json_str.contains("NVIDIA") {
                    return MacGpuType::Discrete;
                }
            }
        }
        
        // Fallback: check environment variables
        if let Ok(arch) = env::var("ARCHFLAGS") {
            if arch.contains("arm64") {
                return MacGpuType::AppleSilicon;
            }
        }
        
        MacGpuType::Unknown
    }
    
    /// Detect Retina display scale factor
    fn detect_retina_scale() -> f32 {
        // Check for Retina display using system information
        if let Ok(output) = Command::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output() 
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("Retina") {
                return 2.0; // Standard Retina scale
            }
        }
        
        // Fallback: check environment variables
        if let Ok(scale) = env::var("QT_SCALE_FACTOR") {
            if let Ok(scale_f32) = scale.parse::<f32>() {
                return scale_f32;
            }
        }
        
        1.0 // Default to standard resolution
    }
    
    /// Optimize performance mode based on hardware
    fn optimize_performance_mode(config: &MacConfig) -> MacPerformanceMode {
        match config.gpu_type {
            MacGpuType::AppleSilicon => {
                // Apple Silicon can handle high quality
                MacPerformanceMode::Quality
            }
            MacGpuType::Intel => {
                // Intel GPUs need more conservative settings
                MacPerformanceMode::Performance
            }
            MacGpuType::Discrete => {
                // Discrete GPUs can handle balanced mode
                MacPerformanceMode::Balanced
            }
            MacGpuType::Unknown => {
                // Default to balanced for unknown hardware
                MacPerformanceMode::Balanced
            }
        }
    }
    
    /// Calculate maximum particles based on hardware
    fn calculate_max_particles(config: &MacConfig) -> usize {
        let base_particles = match config.gpu_type {
            MacGpuType::AppleSilicon => 800_000,
            MacGpuType::Intel => 200_000,
            MacGpuType::Discrete => 600_000,
            MacGpuType::Unknown => 400_000,
        };
        
        // Adjust based on performance mode
        let multiplier = match config.performance_mode {
            MacPerformanceMode::Quality => 0.8,     // Fewer particles for quality
            MacPerformanceMode::Balanced => 1.0,    // Standard amount
            MacPerformanceMode::Performance => 1.5, // More particles for performance
        };
        
        (base_particles as f32 * multiplier) as usize
    }
    
    /// Detect if Touch Bar is available
    fn detect_touch_bar() -> bool {
        // Check if Touch Bar is available (simplified detection)
        if let Ok(output) = Command::new("system_profiler")
            .args(&["SPUSBDataType"])
            .output() 
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return output_str.contains("Touch Bar") || output_str.contains("TouchBar");
        }
        
        false
    }
    
    /// Calculate optimal buffer pool size based on available memory
    fn calculate_buffer_pool_size(config: &MacConfig) -> usize {
        let base_size = match config.gpu_type {
            MacGpuType::AppleSilicon => 256, // Apple Silicon has unified memory
            MacGpuType::Intel => 64,         // Intel GPUs have limited VRAM
            MacGpuType::Discrete => 512,     // Discrete GPUs have more VRAM
            MacGpuType::Unknown => 128,      // Default
        };
        
        // Adjust based on performance mode
        let multiplier = match config.performance_mode {
            MacPerformanceMode::Quality => 0.5,     // Smaller pool for quality
            MacPerformanceMode::Balanced => 1.0,    // Standard size
            MacPerformanceMode::Performance => 2.0, // Larger pool for performance
        };
        
        (base_size as f32 * multiplier) as usize
    }
    
    /// Get GPU type as float for shader uniforms
    pub fn gpu_type_as_float(&self) -> f32 {
        match self.gpu_type {
            MacGpuType::Intel => 0.0,
            MacGpuType::AppleSilicon => 1.0,
            MacGpuType::Discrete => 2.0,
            MacGpuType::Unknown => 0.0,
        }
    }
    
    /// Get performance mode as float for shader uniforms
    pub fn performance_mode_as_float(&self) -> f32 {
        match self.performance_mode {
            MacPerformanceMode::Quality => 0.0,
            MacPerformanceMode::Balanced => 1.0,
            MacPerformanceMode::Performance => 2.0,
        }
    }
    
    /// Get recommended shader optimization level
    pub fn get_shader_optimization_level(&self) -> u32 {
        match self.gpu_type {
            MacGpuType::AppleSilicon => 3, // High optimization for Apple Silicon
            MacGpuType::Intel => 1,        // Conservative optimization for Intel
            MacGpuType::Discrete => 2,     // Medium optimization for discrete
            MacGpuType::Unknown => 1,      // Conservative default
        }
    }
    
    /// Get recommended Metal backend features
    pub fn get_metal_features(&self) -> Vec<&'static str> {
        let mut features = vec!["metal"];
        
        match self.gpu_type {
            MacGpuType::AppleSilicon => {
                features.extend_from_slice(&["metal_apple_silicon", "metal_unified_memory"]);
            }
            MacGpuType::Intel => {
                features.extend_from_slice(&["metal_intel", "metal_conservative"]);
            }
            MacGpuType::Discrete => {
                features.extend_from_slice(&["metal_discrete", "metal_high_performance"]);
            }
            MacGpuType::Unknown => {
                features.push("metal_conservative");
            }
        }
        
        if self.retina_scale > 1.0 {
            features.push("metal_retina");
        }
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mac_config_default() {
        let config = MacConfig::default();
        assert_eq!(config.gpu_type, MacGpuType::Unknown);
        assert_eq!(config.retina_scale, 1.0);
        assert_eq!(config.performance_mode, MacPerformanceMode::Balanced);
    }
    
    #[test]
    fn test_gpu_type_conversion() {
        let mut config = MacConfig::default();
        
        config.gpu_type = MacGpuType::AppleSilicon;
        assert_eq!(config.gpu_type_as_float(), 1.0);
        
        config.gpu_type = MacGpuType::Intel;
        assert_eq!(config.gpu_type_as_float(), 0.0);
        
        config.gpu_type = MacGpuType::Discrete;
        assert_eq!(config.gpu_type_as_float(), 2.0);
    }
    
    #[test]
    fn test_performance_mode_conversion() {
        let mut config = MacConfig::default();
        
        config.performance_mode = MacPerformanceMode::Quality;
        assert_eq!(config.performance_mode_as_float(), 0.0);
        
        config.performance_mode = MacPerformanceMode::Balanced;
        assert_eq!(config.performance_mode_as_float(), 1.0);
        
        config.performance_mode = MacPerformanceMode::Performance;
        assert_eq!(config.performance_mode_as_float(), 2.0);
    }
} 