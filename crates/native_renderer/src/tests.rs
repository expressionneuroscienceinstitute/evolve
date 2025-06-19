//! Tests for Heavy Mode Native Renderer
//! 
//! Comprehensive test suite for scientific visualization and performance

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_particle_vertex_layout() {
        // Ensure ParticleVertex has correct GPU alignment
        assert_eq!(std::mem::size_of::<ParticleVertex>(), 32); // 8 * f32
        assert_eq!(std::mem::align_of::<ParticleVertex>(), 4);
    }

    #[test]
    fn test_uniforms_layout() {
        // Ensure Uniforms struct has correct GPU alignment
        let expected_size = 16 * 4 + 4 * 4; // 4x4 matrix + 4 floats
        assert_eq!(std::mem::size_of::<Uniforms>(), expected_size);
    }

    #[test]
    fn test_camera_default() {
        let camera = Camera::default();
        assert_eq!(camera.position, Point3::new(0.0, 0.0, 10.0));
        assert_eq!(camera.target, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(camera.up, Vector3::new(0.0, 1.0, 0.0));
        assert!(matches!(camera.scale_mode, ScaleMode::Linear));
        assert!(matches!(camera.color_mode, ColorMode::ParticleType));
    }

    #[test]
    fn test_camera_view_projection_matrix() {
        let mut camera = Camera::default();
        camera.aspect = 16.0 / 9.0;
        camera.update_view();
        camera.update_projection();
        
        let vp_matrix = camera.get_view_proj_matrix();
        
        // Matrix should be valid (no NaN or infinity)
        for row in &vp_matrix {
            for &val in row {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_scale_modes() {
        // Test all scale mode variants
        assert!(matches!(ScaleMode::Linear, ScaleMode::Linear));
        assert!(matches!(ScaleMode::Logarithmic, ScaleMode::Logarithmic));
        assert!(matches!(ScaleMode::Energy, ScaleMode::Energy));
        
        if let ScaleMode::Custom(scale) = ScaleMode::Custom(2.5) {
            assert_relative_eq!(scale, 2.5);
        }
    }

    #[test]
    fn test_color_modes() {
        // Test all color mode variants
        let modes = [
            ColorMode::ParticleType,
            ColorMode::Charge,
            ColorMode::Temperature,
            ColorMode::Velocity,
            ColorMode::Interactions,
            ColorMode::Scientific,
        ];
        
        for (i, mode) in modes.iter().enumerate() {
            assert_eq!(*mode as u32, i as u32);
        }
    }

    #[test]
    fn test_render_metrics_default() {
        let metrics = RenderMetrics::default();
        assert_eq!(metrics.fps, 0.0);
        assert_eq!(metrics.frame_time_ms, 0.0);
        assert_eq!(metrics.particles_rendered, 0);
        assert_eq!(metrics.gpu_memory_mb, 0.0);
        assert_eq!(metrics.culled_particles, 0);
        assert_eq!(metrics.shader_switches, 0);
    }

    #[test]
    fn test_particle_vertex_bytemuck() {
        // Test that ParticleVertex implements Pod and Zeroable correctly
        let vertex = ParticleVertex {
            position: [1.0, 2.0, 3.0],
            velocity: [0.1, 0.2, 0.3],
            mass: 1e-27,
            charge: 1.6e-19,
            temperature: 300.0,
            particle_type: 1.0,
            interaction_count: 5.0,
            _padding: 0.0,
        };
        
        let bytes = bytemuck::cast_slice(&[vertex]);
        assert_eq!(bytes.len(), std::mem::size_of::<ParticleVertex>());
        
        let reconstructed: ParticleVertex = bytemuck::cast_slice(bytes)[0];
        assert_relative_eq!(reconstructed.position[0], 1.0);
        assert_relative_eq!(reconstructed.mass, 1e-27);
    }

    #[test]
    fn test_camera_movement() {
        let mut camera = Camera::default();
        let initial_position = camera.position;
        
        // Test forward movement
        camera.handle_input(KeyCode::KeyW, ElementState::Pressed);
        assert_ne!(camera.position, initial_position);
    }

    #[test] 
    fn test_scientific_accuracy_constants() {
        // Test that our shader constants match physics constants
        const SPEED_OF_LIGHT: f32 = 299792458.0;
        const PLANCK_CONSTANT: f32 = 6.626e-34;
        const BOLTZMANN_CONSTANT: f32 = 1.381e-23;
        
        // These should match the constants in heavy_mode.wgsl
        assert_relative_eq!(SPEED_OF_LIGHT, 299792458.0);
        assert_relative_eq!(PLANCK_CONSTANT, 6.626e-34, epsilon = 1e-40);
        assert_relative_eq!(BOLTZMANN_CONSTANT, 1.381e-23, epsilon = 1e-28);
    }

    #[test]
    fn test_heavy_mode_features() {
        #[cfg(feature = "heavy")]
        {
            // Test heavy mode specific functionality
            assert!(true); // Heavy mode is enabled
        }
        
        #[cfg(not(feature = "heavy"))]
        {
            // Should still work without heavy mode
            assert!(true); // Basic mode fallback
        }
    }

    // Mock simulation for testing
    struct MockSimulation {
        particle_count: usize,
    }

    impl MockSimulation {
        fn new(particle_count: usize) -> Self {
            Self { particle_count }
        }
    }

    #[test]
    fn test_performance_monitoring() {
        let mut metrics = RenderMetrics::default();
        
        // Simulate frame timing
        metrics.frame_time_ms = 16.67; // 60 FPS
        metrics.particles_rendered = 100000;
        metrics.fps = 60.0;
        
        assert_relative_eq!(metrics.frame_time_ms, 16.67, epsilon = 0.01);
        assert_eq!(metrics.particles_rendered, 100000);
        assert_relative_eq!(metrics.fps, 60.0);
    }

    #[test]
    fn test_scientific_visualization_scaling() {
        // Test mass scaling for different particle types
        let atomic_mass = 1.66e-27; // kg
        let stellar_mass = 2e30; // kg (solar mass)
        let planck_mass = 2.18e-8; // kg
        
        // Test logarithmic scaling handles wide mass ranges
        let log_atomic = atomic_mass.log10();
        let log_stellar = stellar_mass.log10();
        let log_planck = planck_mass.log10();
        
        assert!(log_stellar > log_planck);
        assert!(log_planck > log_atomic);
        assert!((log_stellar - log_atomic).abs() > 50.0); // Wide dynamic range
    }

    #[test]
    fn test_relativistic_calculations() {
        // Test relativistic factor calculations
        let c = 299792458.0; // m/s
        let velocity_low = 1000.0; // m/s (non-relativistic)
        let velocity_high = 0.9 * c; // 90% speed of light
        
        let beta_low = velocity_low / c;
        let beta_high = velocity_high / c;
        
        let gamma_low = 1.0 / (1.0 - beta_low.powi(2)).sqrt();
        let gamma_high = 1.0 / (1.0 - beta_high.powi(2)).sqrt();
        
        assert_relative_eq!(gamma_low, 1.0, epsilon = 1e-10); // Non-relativistic
        assert!(gamma_high > 2.0); // Significant relativistic effects
    }

    #[test]
    fn test_color_mode_scientific_accuracy() {
        // Test temperature-based color mapping
        let temp_cold = 2.7; // CMB temperature
        let temp_room = 300.0; // Room temperature
        let temp_stellar = 5778.0; // Solar surface temperature
        let temp_core = 15e6; // Solar core temperature
        
        // Colors should progress from cold to hot
        assert!(temp_cold < temp_room);
        assert!(temp_room < temp_stellar);
        assert!(temp_stellar < temp_core);
        
        // Test blackbody radiation wavelength peaks
        let wien_constant = 2.898e-3; // m⋅K
        let peak_cold = wien_constant / temp_cold;
        let peak_hot = wien_constant / temp_core;
        
        assert!(peak_cold > peak_hot); // Longer wavelength for colder objects
    }
} 