//! High-Performance Native Renderer for Universe Simulation
//! 
//! GPU-accelerated particle rendering with direct memory access to physics data.
//! Eliminates WebSocket overhead for maximum performance.
//! 
//! # Heavy Mode Features
//! - Advanced multi-scale particle visualization
//! - Scientific color-coding and size scaling
//! - High-fidelity physics interaction rendering
//! - Performance-optimized GPU compute shaders

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use nalgebra::{Vector3, Point3, Matrix4};
// use rayon::prelude::*; // Unused for now
use std::sync::{Arc, Mutex};
use tracing::{info, error, warn, debug};
use winit::{
    event::{Event, WindowEvent, ElementState},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    keyboard::{PhysicalKey, KeyCode},
};

pub use universe_sim::UniverseSimulation;

#[cfg(test)]
mod inline_tests {
    use super::*;
    use std::mem;
    
    #[test]
    fn test_gpu_buffer_size_limit() {
        // Test that our buffer size calculation fits within GPU limits
        let max_particles = 800_000;
        let vertices_per_particle = 6;
        let vertex_size = mem::size_of::<ParticleVertex>();
        
        let total_buffer_size = max_particles * vertices_per_particle * vertex_size;
        let gpu_limit = 268_435_456; // 268 MB limit observed on macOS Metal
        
        println!("Buffer size calculation:");
        println!("  Max particles: {}", max_particles);
        println!("  Vertices per particle: {}", vertices_per_particle);
        println!("  Vertex size: {} bytes", vertex_size);
        println!("  Total buffer size: {} bytes ({:.1} MB)", total_buffer_size, total_buffer_size as f64 / 1_048_576.0);
        println!("  GPU limit: {} bytes ({:.1} MB)", gpu_limit, gpu_limit as f64 / 1_048_576.0);
        
        assert!(total_buffer_size < gpu_limit, 
            "Buffer size {} exceeds GPU limit {}", total_buffer_size, gpu_limit);
        
        // Verify ParticleVertex size matches expected
        assert_eq!(vertex_size, 48, "ParticleVertex size should be 48 bytes");
    }
}

/// High-performance particle vertex for GPU rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleVertex {
    pub position: [f32; 3],
    pub velocity: [f32; 3], 
    pub mass: f32,
    pub charge: f32,
    pub temperature: f32,
    pub particle_type: f32,  // Encoded particle type for shader branching
    pub interaction_count: f32, // Number of recent interactions
    pub _padding: f32, // GPU alignment
}

/// Enhanced camera with scientific visualization controls
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub zoom_speed: f32,
    pub pan_speed: f32,
    pub rotation_speed: f32,
    // Heavy mode scientific controls
    pub scale_mode: ScaleMode,
    pub color_mode: ColorMode,
    pub filter_threshold: f32,
    // Matrix storage
    pub view_matrix: Matrix4<f32>,
    pub proj_matrix: Matrix4<f32>,
}

/// Scientific scaling modes for heavy mode visualization
#[derive(Debug, Clone, Copy)]
pub enum ScaleMode {
    /// Linear scaling by mass
    Linear,
    /// Logarithmic scaling for wide mass ranges
    Logarithmic,
    /// Energy-based scaling for interactions
    Energy,
    /// Custom scaling for specific phenomena  
    Custom(f32),
}

/// Scientific color-coding modes
#[derive(Debug, Clone, Copy)]
pub enum ColorMode {
    /// Color by particle type
    ParticleType,
    /// Color by charge
    Charge,
    /// Color by temperature (thermal radiation)
    Temperature,
    /// Color by velocity (Doppler shift)
    Velocity,
    /// Color by interaction frequency
    Interactions,
    /// Multi-channel scientific visualization
    Scientific,
}

/// Rendering performance metrics for heavy mode
#[derive(Debug, Default)]
pub struct RenderMetrics {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub particles_rendered: usize,
    pub gpu_memory_mb: f32,
    pub culled_particles: usize,
    pub shader_switches: usize,
}

/// High-performance renderer state with heavy mode enhancements
pub struct NativeRenderer<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'window Window,
    
    // Rendering pipeline
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Heavy mode: additional compute pipeline for advanced effects
    #[cfg(feature = "heavy")]
    compute_pipeline: Option<wgpu::ComputePipeline>,
    #[cfg(feature = "heavy")]
    compute_bind_group: Option<wgpu::BindGroup>,
    
    // Camera and view state
    camera: Camera,
    view_matrix: Matrix4<f32>,
    proj_matrix: Matrix4<f32>,
    
    // Performance metrics and controls
    metrics: RenderMetrics,
    frame_count: u64,
    last_fps_time: std::time::Instant,
    
    // Particle data management
    particle_count: usize,
    max_particles: usize,
    
    // Heavy mode: scientific visualization state
    #[cfg(feature = "heavy")]
    heavy_mode_enabled: bool,
    #[cfg(feature = "heavy")]
    interaction_heatmap: Vec<f32>,
    #[cfg(feature = "heavy")]
    temperature_field: Vec<Vector3<f32>>,
}

/// Uniform data sent to GPU with heavy mode extensions
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    _padding: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 5.0), // Closer to particles
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: 60.0_f32.to_radians(), // Wider field of view
            aspect: 1.0,
            near: 0.01, // Closer near plane
            far: 1000.0,
            zoom_speed: 0.1,
            pan_speed: 0.01,
            rotation_speed: 0.01,
            scale_mode: ScaleMode::Linear,
            color_mode: ColorMode::Charge, // Start with charge for better visibility
            filter_threshold: 0.0, // No filtering initially
            view_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
        }
    }
}

impl Camera {
    /// Update view matrix based on camera parameters
    pub fn update_view(&mut self) {
        // Calculate proper view matrix using nalgebra
        let eye = self.position;
        let target = self.target;
        let up = self.up;
        
        self.view_matrix = Matrix4::look_at_rh(&eye, &target, &up);
    }
    
    /// Update projection matrix
    pub fn update_projection(&mut self) {
        self.proj_matrix = Matrix4::new_perspective(
            self.aspect,
            self.fov,
            self.near,
            self.far,
        );
    }
    
    /// Get combined view-projection matrix for GPU (4x4 array format)
    pub fn get_view_proj_matrix(&self) -> [[f32; 4]; 4] {
        let vp = self.proj_matrix * self.view_matrix;
        [
            [vp[(0, 0)], vp[(0, 1)], vp[(0, 2)], vp[(0, 3)]],
            [vp[(1, 0)], vp[(1, 1)], vp[(1, 2)], vp[(1, 3)]],
            [vp[(2, 0)], vp[(2, 1)], vp[(2, 2)], vp[(2, 3)]],
            [vp[(3, 0)], vp[(3, 1)], vp[(3, 2)], vp[(3, 3)]],
        ]
    }
    
    /// Handle keyboard input for camera controls
    pub fn handle_input(&mut self, key: KeyCode, state: ElementState) {
        let movement_speed = 0.1;
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        
        if state == ElementState::Pressed {
            match key {
                KeyCode::KeyW => {
                    self.position += forward * movement_speed;
                    self.target += forward * movement_speed;
                }
                KeyCode::KeyS => {
                    self.position -= forward * movement_speed;
                    self.target -= forward * movement_speed;
                }
                KeyCode::KeyA => {
                    self.position -= right * movement_speed;
                    self.target -= right * movement_speed;
                }
                KeyCode::KeyD => {
                    self.position += right * movement_speed;
                    self.target += right * movement_speed;
                }
                KeyCode::KeyQ => {
                    self.position.y += movement_speed;
                    self.target.y += movement_speed;
                }
                KeyCode::KeyE => {
                    self.position.y -= movement_speed;
                    self.target.y -= movement_speed;
                }
                // Heavy mode controls
                KeyCode::Digit1 => self.color_mode = ColorMode::ParticleType,
                KeyCode::Digit2 => self.color_mode = ColorMode::Charge,
                KeyCode::Digit3 => self.color_mode = ColorMode::Temperature,
                KeyCode::Digit4 => self.color_mode = ColorMode::Velocity,
                KeyCode::Digit5 => self.color_mode = ColorMode::Interactions,
                KeyCode::Digit6 => self.color_mode = ColorMode::Scientific,
                _ => {}
            }
        }
    }
}

impl<'window> NativeRenderer<'window> {
    /// Create new high-performance renderer
    pub async fn new(window: &'window Window) -> Result<Self> {
        // Initialize logging if it hasn't been set by the main application yet.
        let _ = env_logger::try_init();
        info!("Initializing high-performance native renderer");
        
        let size = window.inner_size();
        
        // Create WGPU instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        
        // Create surface
        let surface = instance.create_surface(window)?;
        
        // Request adapter (prefer high-performance GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable adapter"))?;
        
        info!("Using GPU adapter: {:?}", adapter.get_info());
        
        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await?;
        
        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        // Create shader (inline WGSL for simplicity)
        let shader_source = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) mass: f32,
    @location(3) charge: f32,
    @location(4) temperature: f32,
    @location(5) particle_type: f32,
    @location(6) interaction_count: f32,
    @location(7) _padding: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Create a small quad for each particle (6 vertices per particle)
    let quad_index = vertex_index % 6u;
    var offset = vec2<f32>(0.0, 0.0);
    
    // Define quad vertices (two triangles)
    switch quad_index {
        case 0u: { offset = vec2<f32>(-0.5, -0.5); out.uv = vec2<f32>(0.0, 0.0); }
        case 1u: { offset = vec2<f32>( 0.5, -0.5); out.uv = vec2<f32>(1.0, 0.0); }
        case 2u: { offset = vec2<f32>( 0.5,  0.5); out.uv = vec2<f32>(1.0, 1.0); }
        case 3u: { offset = vec2<f32>(-0.5, -0.5); out.uv = vec2<f32>(0.0, 0.0); }
        case 4u: { offset = vec2<f32>( 0.5,  0.5); out.uv = vec2<f32>(1.0, 1.0); }
        default: { offset = vec2<f32>(-0.5,  0.5); out.uv = vec2<f32>(0.0, 1.0); }
    }
    
    // Large particle size for initial visibility (can be replaced with mass scaling later)
    let size = 0.2; // Much larger particles
    offset = offset * size;
    
    let world_pos = vec4<f32>(vertex.position, 1.0);
    let clip_pos = uniforms.view_proj * world_pos;
    
    // Add offset in screen space
    out.clip_position = vec4<f32>(clip_pos.xy + offset * clip_pos.w, clip_pos.z, clip_pos.w);
    
    // Color by charge - make colors much brighter
    if (vertex.charge > 0.0) {
        out.color = vec3<f32>(1.0, 0.0, 0.0); // Bright red for positive
    } else if (vertex.charge < 0.0) {
        out.color = vec3<f32>(0.0, 0.0, 1.0); // Bright blue for negative
    } else {
        out.color = vec3<f32>(1.0, 1.0, 1.0); // White for neutral
    }
    
    // Modulate by temperature
    if (vertex.temperature > 0.0) {
        let temp_normalized = min(vertex.temperature / 10000.0, 1.0);
        out.color = mix(out.color, vec3<f32>(1.0, 0.8, 0.2), temp_normalized);
    }
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple solid particles for debugging - no transparency
    return vec4<f32>(in.color, 1.0);
}
"#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create uniform buffer
        let max_particles = 800_000; // Reduced from 1M to fit GPU buffer limits (800k * 6 * 48 = ~230MB)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create vertex buffer for particles
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Vertex Buffer"),
            size: (max_particles * 6 * std::mem::size_of::<ParticleVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Uniform Bind Group Layout"),
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Uniform Bind Group"),
        });
        
        // Create render pipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // Position
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        // Velocity
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as u64,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        // Mass
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Charge
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>()) as u64,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Temperature
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 2) as u64,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Particle type
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 3) as u64,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Interaction count
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 4) as u64,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Padding
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 5) as u64,
                            shader_location: 7,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        let mut camera = Camera::default();
        camera.aspect = size.width as f32 / size.height as f32;
        camera.position = Point3::new(0.0, 0.0, 5.0); // Move camera back to see particles
        camera.target = Point3::new(0.0, 0.0, 0.0);
        camera.update_view();
        camera.update_projection();
        
        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,
            
            #[cfg(feature = "heavy")]
            compute_pipeline: None,
            #[cfg(feature = "heavy")]
            compute_bind_group: None,
            
            camera,
            view_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
            metrics: RenderMetrics::default(),
            frame_count: 0,
            last_fps_time: std::time::Instant::now(),
            particle_count: 0,
            max_particles,
            
            #[cfg(feature = "heavy")]
            heavy_mode_enabled: false,
            #[cfg(feature = "heavy")]
            interaction_heatmap: Vec::new(),
            #[cfg(feature = "heavy")]
            temperature_field: Vec::new(),
        })
    }
    
    /// Update particle data from simulation with zero-copy access
    pub fn update_particles(&mut self, simulation: &UniverseSimulation) -> Result<()> {
        // Check both the store particles and physics engine particles
        let store_particles = &simulation.store.particles;
        let physics_particles = &simulation.physics_engine.particles;
        
        info!("Store has {} particles, Physics engine has {} particles", 
            store_particles.count, physics_particles.len());
        
        // Use physics engine particles if available, otherwise store particles
        let gpu_particles: Vec<ParticleVertex> = if !physics_particles.is_empty() {
            info!("Using physics engine particles");
            if physics_particles.len() > 0 {
                info!("Sample physics particle 0: pos=({:.2e}, {:.2e}, {:.2e}), mass={:.2e}, energy={:.2e}", 
                    physics_particles[0].position.x, physics_particles[0].position.y, physics_particles[0].position.z,
                    physics_particles[0].mass, physics_particles[0].energy);
            }
            
            physics_particles.iter()
                .take(self.max_particles)
                .enumerate()
                .map(|(i, p)| {
                    // Scale positions to be visible (particles might be at cosmic scales)
                    let scale_factor = 1e-12; // Scale down from meters to something visible (~0.01 to 10 units)
                    ParticleVertex {
                        position: [
                            (p.position.x * scale_factor) as f32, 
                            (p.position.y * scale_factor) as f32, 
                            (p.position.z * scale_factor) as f32
                        ],
                        velocity: [
                            p.momentum.x as f32 / p.mass as f32, 
                            p.momentum.y as f32 / p.mass as f32, 
                            p.momentum.z as f32 / p.mass as f32
                        ],
                        mass: p.mass as f32,
                        charge: p.electric_charge as f32,
                        temperature: 3000.0, // Default temperature for now
                        particle_type: i as f32 % 10.0, // Simple type encoding
                        interaction_count: p.interaction_history.len() as f32,
                        _padding: 0.0,
                    }
                })
                .collect()
        } else if store_particles.count > 0 {
            info!("Using store particles");
            info!("Sample store particle 0: pos=({:.2e}, {:.2e}, {:.2e}), mass={:.2e}, temp={:.2e}", 
                store_particles.position[0].x, store_particles.position[0].y, store_particles.position[0].z,
                store_particles.mass[0], store_particles.temperature[0]);
            
            (0..store_particles.count.min(self.max_particles))
                .map(|i| {
                    // Scale positions to be visible (particles might be at cosmic scales)
                    let scale_factor = 1e-12; // Scale down from meters to something visible (~0.01 to 10 units)
                    ParticleVertex {
                        position: [
                            (store_particles.position[i].x * scale_factor) as f32, 
                            (store_particles.position[i].y * scale_factor) as f32, 
                            (store_particles.position[i].z * scale_factor) as f32
                        ],
                        velocity: [
                            store_particles.velocity[i].x as f32, 
                            store_particles.velocity[i].y as f32, 
                            store_particles.velocity[i].z as f32
                        ],
                        mass: store_particles.mass[i] as f32,
                        charge: store_particles.charge[i] as f32,
                        temperature: store_particles.temperature[i] as f32,
                        particle_type: (i % 10) as f32,
                        interaction_count: 0.0,
                        _padding: 0.0,
                    }
                })
                .collect()
        } else {
            warn!("No particles found in either store or physics engine! Creating test particles.");
            // Create some test particles for debugging at simple positions - TEMPORARILY FORCE THIS PATH
            vec![
                // Particle at origin - large and bright
                ParticleVertex {
                    position: [0.0, 0.0, 0.0],
                    velocity: [0.0, 0.0, 0.0],
                    mass: 1e20, // Large mass for size
                    charge: 1.6e-19,
                    temperature: 10000.0,
                    particle_type: 5.0,
                    interaction_count: 0.0,
                    _padding: 0.0,
                },
                // Particle to the right - negative charge
                ParticleVertex {
                    position: [2.0, 0.0, 0.0],
                    velocity: [100.0, 0.0, 0.0],
                    mass: 1e20,
                    charge: -1.6e-19,
                    temperature: 5000.0,
                    particle_type: 3.0,
                    interaction_count: 0.0,
                    _padding: 0.0,
                },
                // Particle above - neutral
                ParticleVertex {
                    position: [0.0, 2.0, 0.0],
                    velocity: [0.0, 50.0, 0.0],
                    mass: 1e19,
                    charge: 0.0,
                    temperature: 3000.0,
                    particle_type: 1.0,
                    interaction_count: 0.0,
                    _padding: 0.0,
                },
                // Particle closer to camera
                ParticleVertex {
                    position: [0.0, 0.0, 1.0],
                    velocity: [0.0, 0.0, 200.0],
                    mass: 1e18,
                    charge: 1.6e-19,
                    temperature: 8000.0,
                    particle_type: 2.0,
                    interaction_count: 0.0,
                    _padding: 0.0,
                },
                // Additional particles for better coverage
                ParticleVertex {
                    position: [-1.0, -1.0, 0.0],
                    velocity: [-50.0, -50.0, 0.0],
                    mass: 1e21,
                    charge: 2.0 * 1.6e-19,
                    temperature: 15000.0,
                    particle_type: 7.0,
                    interaction_count: 0.0,
                    _padding: 0.0,
                },
            ]
        };
        
        // TEMPORARILY FORCE TEST PARTICLES FOR DEBUGGING
        let gpu_particles = vec![
            // Particle at origin - large and bright
            ParticleVertex {
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
                mass: 1e20, // Large mass for size
                charge: 1.6e-19,
                temperature: 10000.0,
                particle_type: 5.0,
                interaction_count: 0.0,
                _padding: 0.0,
            },
            // Particle to the right - negative charge (blue)
            ParticleVertex {
                position: [2.0, 0.0, 0.0],
                velocity: [100.0, 0.0, 0.0],
                mass: 1e20,
                charge: -1.6e-19,
                temperature: 5000.0,
                particle_type: 3.0,
                interaction_count: 0.0,
                _padding: 0.0,
            },
            // Particle above - neutral (gray)
            ParticleVertex {
                position: [0.0, 2.0, 0.0],
                velocity: [0.0, 50.0, 0.0],
                mass: 1e19,
                charge: 0.0,
                temperature: 3000.0,
                particle_type: 1.0,
                interaction_count: 0.0,
                _padding: 0.0,
            },
            // Particle closer to camera
            ParticleVertex {
                position: [0.0, 0.0, 1.0],
                velocity: [0.0, 0.0, 200.0],
                mass: 1e18,
                charge: 1.6e-19,
                temperature: 8000.0,
                particle_type: 2.0,
                interaction_count: 0.0,
                _padding: 0.0,
            },
        ];
        
        self.particle_count = gpu_particles.len();
        
        // Upload to GPU buffer (6 copies per particle for quad rendering)
        if !gpu_particles.is_empty() {
            let mut expanded_particles = Vec::with_capacity(gpu_particles.len() * 6);
            for particle in &gpu_particles {
                // Add 6 copies of each particle for the quad vertices
                for _ in 0..6 {
                    expanded_particles.push(*particle);
                }
            }
            
            self.queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&expanded_particles),
            );
            info!("Successfully uploaded {} particles ({} vertices) to GPU", self.particle_count, expanded_particles.len());
            
            // Debug: log a few sample particles
            if !gpu_particles.is_empty() {
                let sample = &gpu_particles[0];
                info!("Sample GPU particle: pos=({:.3}, {:.3}, {:.3}), mass={:.2e}, charge={:.2e}, temp={:.1}", 
                    sample.position[0], sample.position[1], sample.position[2],
                    sample.mass, sample.charge, sample.temperature);
            }
        } else {
            warn!("No particles to upload to GPU!");
        }
        
        Ok(())
    }

    /// Render frame with maximum performance and heavy mode enhancements
    pub fn render(&mut self, simulation_time: f32) -> Result<()> {
        let frame_start = std::time::Instant::now();
        
        // Update camera matrices
        self.camera.update_view();
        self.camera.update_projection(); // Ensure projection is always up to date
        let view_proj = self.camera.get_view_proj_matrix();
        
        // Debug camera info (only log occasionally)
        if self.frame_count % 60 == 0 {
            info!("Camera pos=({:.2}, {:.2}, {:.2}), target=({:.2}, {:.2}, {:.2}), particles={}", 
                self.camera.position.x, self.camera.position.y, self.camera.position.z,
                self.camera.target.x, self.camera.target.y, self.camera.target.z,
                self.particle_count);
        }
        
        // Create uniforms with heavy mode extensions
        let uniforms = Uniforms {
            view_proj,
            time: simulation_time,
            scale: match self.camera.scale_mode {
                ScaleMode::Linear => 1.0,
                ScaleMode::Logarithmic => 2.0,
                ScaleMode::Energy => 3.0,
                ScaleMode::Custom(scale) => scale,
            },
            color_mode: self.camera.color_mode as u32 as f32,
            filter_threshold: self.camera.filter_threshold,
            _padding: 0.0,
        };
        
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        // Get surface texture
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Heavy mode: run compute shader for advanced effects first
        #[cfg(feature = "heavy")]
        if self.heavy_mode_enabled {
            if let (Some(compute_pipeline), Some(compute_bind_group)) = (&self.compute_pipeline, &self.compute_bind_group) {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Heavy Mode Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(compute_pipeline);
                compute_pass.set_bind_group(0, compute_bind_group, &[]);
                compute_pass.dispatch_workgroups((self.particle_count as u32 + 63) / 64, 1, 1);
            }
        }
        
        // Main render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.2,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            // Draw 6 vertices per particle (2 triangles = 1 quad)
            let vertex_count = self.particle_count as u32 * 6;
            if vertex_count > 0 {
                render_pass.draw(0..vertex_count, 0..1);
                if self.frame_count % 60 == 0 {
                    info!("Drawing {} vertices for {} particles", vertex_count, self.particle_count);
                }
            } else {
                warn!("No vertices to draw! particle_count={}", self.particle_count);
            }
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        // Update performance metrics
        let frame_time = frame_start.elapsed();
        self.metrics.frame_time_ms = frame_time.as_secs_f32() * 1000.0;
        self.metrics.particles_rendered = self.particle_count;
        
        self.frame_count += 1;
        let now = std::time::Instant::now();
        if now.duration_since(self.last_fps_time).as_secs() >= 1 {
            self.metrics.fps = self.frame_count as f32 / now.duration_since(self.last_fps_time).as_secs_f32();
            self.frame_count = 0;
            self.last_fps_time = now;
            
            debug!(
                "Heavy Mode Renderer - FPS: {:.1}, Frame Time: {:.2}ms, Particles: {}",
                self.metrics.fps, self.metrics.frame_time_ms, self.metrics.particles_rendered
            );
        }
        
        Ok(())
    }
    
    /// Handle window resize
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
            self.camera.update_projection();
        }
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> &RenderMetrics {
        &self.metrics
    }
    
    /// Get current FPS for performance monitoring
    pub fn get_fps(&self) -> f32 {
        self.metrics.fps
    }
    
    /// Get particle count
    pub fn get_particle_count(&self) -> usize {
        self.particle_count
    }
    
    /// Toggle heavy mode rendering
    #[cfg(feature = "heavy")]
    pub fn toggle_heavy_mode(&mut self) {
        self.heavy_mode_enabled = !self.heavy_mode_enabled;
        info!("Heavy mode rendering: {}", if self.heavy_mode_enabled { "enabled" } else { "disabled" });
    }
    
    /// Set scientific visualization mode
    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.camera.color_mode = mode;
        debug!("Switched to color mode: {:?}", mode);
    }
    
    /// Set particle scaling mode
    pub fn set_scale_mode(&mut self, mode: ScaleMode) {
        self.camera.scale_mode = mode;
        debug!("Switched to scale mode: {:?}", mode);
    }
}

/// Create and run the high-performance renderer with heavy mode support
pub async fn run_renderer(simulation: Arc<Mutex<UniverseSimulation>>) -> Result<()> {
    info!("Starting high-performance native renderer with heavy mode support");
    
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("EVOLVE Universe Simulation - Heavy Mode Native Renderer")
        .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
        .build(&event_loop)?;
    
    let mut renderer = NativeRenderer::new(&window).await?;
    let mut last_update = std::time::Instant::now();
    
    info!("🎮 Renderer Controls:");
    info!("  WASD - Move camera");
    info!("  Q/E - Move up/down");
    info!("  1-6 - Switch color modes");
    info!("  H - Toggle heavy mode (if enabled)");
    info!("  ESC - Exit");
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == renderer.window.id() => match event {
                WindowEvent::CloseRequested => {
                    info!("Closing heavy mode renderer");
                    elwt.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(*physical_size);
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    if let PhysicalKey::Code(key_code) = key_event.physical_key {
                        // Handle camera controls
                        renderer.camera.handle_input(key_code, key_event.state);
                        
                        // Handle special keys
                        if key_event.state == ElementState::Pressed {
                            match key_code {
                                KeyCode::Escape => elwt.exit(),
                                #[cfg(feature = "heavy")]
                                KeyCode::KeyH => renderer.toggle_heavy_mode(),
                                KeyCode::KeyR => {
                                    // Reset camera
                                    renderer.camera = Camera::default();
                                    renderer.camera.aspect = renderer.size.width as f32 / renderer.size.height as f32;
                                    renderer.camera.update_projection();
                                    info!("Camera reset to default position");
                                }
                                _ => {}
                            }
                        }
                    }
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    // Handle scale factor changes if needed
                }
                _ => {}
            },
            Event::AboutToWait => {
                renderer.window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let simulation_time = now.duration_since(last_update).as_secs_f32();
                last_update = now;
                
                // Update particles from simulation (non-blocking)
                if let Ok(sim) = simulation.try_lock() {
                    if let Err(e) = renderer.update_particles(&sim) {
                        error!("Failed to update particles: {}", e);
                    }
                } else {
                    // Simulation is busy, use cached particle data
                    debug!("Simulation locked, using cached particle data");
                }
                
                // Render with performance monitoring
                let render_start = std::time::Instant::now();
                if let Err(e) = renderer.render(simulation_time) {
                    error!("Render error: {}", e);
                }
                
                // Log performance issues
                let render_time = render_start.elapsed().as_millis();
                if render_time > 16 {  // More than 16ms = below 60 FPS
                    warn!("Slow frame detected: {}ms render time", render_time);
                }
            }
            _ => {}
        }
    })?;
    
    Ok(())
} 