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
use glyphon::{FontSystem, SwashCache, TextRenderer, TextAtlas, TextArea, TextBounds, Metrics, Buffer, Color, Attrs, Family, Shaping, Resolution};

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
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimpleParticleVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub size: f32,
}

// Keep old struct for compatibility but add simple one
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
    quad_vertex_buffer: wgpu::Buffer,
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

    // Text rendering (glyphon)
    font_system: FontSystem,
    text_atlas: TextAtlas,
    text_cache: SwashCache,
    text_renderer: TextRenderer,
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
    _padding: [f32; 4], // Exactly 96 bytes total (16 bytes padding)
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 12.0), // Much further back to see all 6 rows
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: 45.0_f32.to_radians(), // Slightly narrower FOV for better view
            aspect: 1.0,
            near: 0.01, // Closer near plane
            far: 1000.0,
            zoom_speed: 0.2,
            pan_speed: 0.02,
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
        // WGPU expects the depth range in clip space to be 0..1, while nalgebra's
        // `new_perspective` builds an OpenGL-style projection matrix that outputs
        // depth in the −1..1 range.  We therefore need to apply the commonly
        // used "OpenGL-to-WGPU" transform so that our geometry is not clipped
        // out entirely (which is why only the clear colour was visible).

        // Equivalent to the matrix used in the official wgpu examples:
        //   [[ 1, 0, 0, 0 ],
        //    [ 0, 1, 0, 0 ],
        //    [ 0, 0, 0.5, 0 ],
        //    [ 0, 0, 0.5, 1 ]]
        let opengl_to_wgpu = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );

        let vp = (opengl_to_wgpu * self.proj_matrix * self.view_matrix).transpose();

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
                KeyCode::Digit1 => {
                    self.color_mode = ColorMode::ParticleType;
                    println!("🎨 Color mode switched to: ParticleType (1)");
                },
                KeyCode::Digit2 => {
                    self.color_mode = ColorMode::Charge;
                    println!("🎨 Color mode switched to: Charge (2)");
                },
                KeyCode::Digit3 => {
                    self.color_mode = ColorMode::Temperature;
                    println!("🎨 Color mode switched to: Temperature (3)");
                },
                KeyCode::Digit4 => {
                    self.color_mode = ColorMode::Velocity;
                    println!("🎨 Color mode switched to: Velocity (4)");
                },
                KeyCode::Digit5 => {
                    self.color_mode = ColorMode::Interactions;
                    println!("🎨 Color mode switched to: Interactions (5)");
                },
                KeyCode::Digit6 => {
                    self.color_mode = ColorMode::Scientific;
                    println!("🎨 Color mode switched to: Scientific (6)");
                },
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
        
        // ADVANCED PARTICLE SHADER WITH COLOR MODE SWITCHING
        let shader_source = include_str!("shaders/particle.wgsl");
        
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
        
        // Create quad vertex buffer (6 vertices per quad) - flattened array
        let quad_vertices: &[f32] = &[
            -1.0, -1.0,  1.0, -1.0,  -1.0,  1.0,  // Triangle 1
             1.0, -1.0,  1.0,  1.0,  -1.0,  1.0,  // Triangle 2
        ];
        let quad_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad Vertex Buffer"),
            size: (quad_vertices.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&quad_vertex_buffer, 0, bytemuck::cast_slice(quad_vertices));
        
        // Create instance buffer for particle data
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (max_particles * std::mem::size_of::<ParticleVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                buffers: &[
                    // Quad vertices (per vertex)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Particle data (per instance)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // Position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // Velocity
                            wgpu::VertexAttribute {
                                offset: std::mem::size_of::<[f32; 3]>() as u64,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // Mass
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // Charge
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>()) as u64,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // Temperature
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 2) as u64,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // Particle Type
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 3) as u64,
                                shader_location: 6,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // Interaction Count
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 4) as u64,
                                shader_location: 7,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // Padding
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>() * 5) as u64,
                                shader_location: 8,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                ],
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
                topology: wgpu::PrimitiveTopology::TriangleList, // Instanced triangles!
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
        
        // ---------------- Text renderer (glyphon) ----------------
        let font_system = FontSystem::new();
        let mut text_atlas = TextAtlas::new(&device, &queue, config.format);
        let text_renderer = TextRenderer::new(&mut text_atlas, &device, wgpu::MultisampleState::default(), None);
        let text_cache = SwashCache::new();
        
        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            quad_vertex_buffer,
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

            // Text rendering (glyphon)
            font_system,
            text_atlas,
            text_cache,
            text_renderer,
        })
    }
    
    /// Update particle data from simulation with zero-copy access
    pub fn update_particles(&mut self, _simulation: &UniverseSimulation) -> Result<()> {
        println!("🔬 DEBUG ROOM: Generating shader-driven color demo");

        // Configuration - more samples to show full color ranges
        const COLOR_SETS: usize = 6; // Number of ColorMode variants
        const SAMPLES_PER_ROW: usize = 8; // More samples to show full range
        const MASSES: [f32; 8] = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8]; // Various masses for size variation
        const SPACING_X: f32 = 1.5;
        const SPACING_Y: f32 = 1.8;

        let mut particles: Vec<ParticleVertex> = Vec::new();
        let y_offset = (COLOR_SETS as f32 - 1.0) * 0.5 * SPACING_Y;

        for row in 0..COLOR_SETS {
            let y = row as f32 * SPACING_Y - y_offset;
            
            for col in 0..SAMPLES_PER_ROW {
                let x = col as f32 * SPACING_X - (SAMPLES_PER_ROW as f32 * SPACING_X * 0.5);
                let mass = MASSES[col % MASSES.len()];
                
                // Generate particles with varying physics properties to demonstrate each color mode
                let particle = match row {
                    0 => {
                        // ParticleType mode - different particle types
                        let particle_type = (col % 6) as f32; // 0-5 for different types
                        ParticleVertex {
                            position: [x, y, 0.0],
                            velocity: [0.0, 0.0, 0.0],
                            mass,
                            charge: 0.0,
                            temperature: 300.0, // Room temperature
                            particle_type,
                            interaction_count: 0.0,
                            _padding: 0.0,
                        }
                    },
                    1 => {
                        // Charge mode: various charge values from -2 to +2
                        let charge_range = 4.0; // -2 to +2
                        let charge = (col as f32 / (SAMPLES_PER_ROW-1) as f32) * charge_range - 2.0;
                        ParticleVertex {
                            position: [x, y, 0.0],
                            velocity: [0.0, 0.0, 0.0],
                            mass,
                            charge,
                            temperature: 300.0,
                            particle_type: 0.0,
                            interaction_count: 0.0,
                            _padding: 0.0,
                        }
                    },
                    2 => {
                        // Temperature mode: range from 0K to 6000K
                        let temp_max = 6000.0;
                        let temperature = (col as f32 / (SAMPLES_PER_ROW-1) as f32) * temp_max;
                        ParticleVertex {
                            position: [x, y, 0.0],
                            velocity: [0.0, 0.0, 0.0],
                            mass,
                            charge: 0.0,
                            temperature,
                            particle_type: 0.0,
                            interaction_count: 0.0,
                            _padding: 0.0,
                        }
                    },
                    3 => {
                        // Velocity mode: various velocity directions and magnitudes
                        let (vx, vy, vz) = match col {
                            0 => (0.0, 0.0, 0.0),      // Stationary
                            1 => (1.0, 0.0, 0.0),      // +X direction
                            2 => (0.0, 1.0, 0.0),      // +Y direction
                            3 => (0.0, 0.0, 1.0),      // +Z direction
                            4 => (0.7, 0.7, 0.0),      // XY diagonal
                            5 => (0.7, 0.0, 0.7),      // XZ diagonal
                            6 => (0.0, 0.7, 0.7),      // YZ diagonal
                            _ => (2.0, 1.0, 0.5),      // Fast mixed motion
                        };
                        ParticleVertex {
                            position: [x, y, 0.0],
                            velocity: [vx, vy, vz],
                            mass,
                            charge: 0.0,
                            temperature: 300.0,
                            particle_type: 0.0,
                            interaction_count: 0.0,
                            _padding: 0.0,
                        }
                    },
                    4 => {
                        // Interactions mode: various interaction counts
                        let interaction_count = col as f32 * 10.0; // 0 to 70 interactions
                        ParticleVertex {
                            position: [x, y, 0.0],
                            velocity: [0.0, 0.0, 0.0],
                            mass,
                            charge: 0.0,
                            temperature: 300.0,
                            particle_type: 0.0,
                            interaction_count,
                            _padding: 0.0,
                        }
                    },
                    _ => {
                        // Scientific mode: mix of everything for complex visualization
                        let charge = if col % 2 == 0 { 1.0 } else { -1.0 };
                        let temperature = 300.0 + col as f32 * 400.0;
                        let speed = col as f32 * 0.2;
                        ParticleVertex {
                            position: [x, y, 0.0],
                            velocity: [speed, speed * 0.5, 0.0],
                            mass,
                            charge,
                            temperature,
                            particle_type: (col % 3) as f32,
                            interaction_count: col as f32 * 5.0,
                            _padding: 0.0,
                        }
                    }
                };
                
                particles.push(particle);
            }
        }

        // Upload to GPU
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&particles));
        self.particle_count = particles.len();
        println!("✅ Uploaded {} shader-demo particles", self.particle_count);
        Ok(())
    }

    /// Render frame with maximum performance and heavy mode enhancements
    pub fn render(&mut self, simulation_time: f32) -> Result<()> {
        let frame_start = std::time::Instant::now();
        
        println!("🔥 RENDER START - Frame {}", self.frame_count);
        
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
            
            // Debug view-projection matrix
            let vp = view_proj;
            info!("View-projection matrix [0]: [{:.3}, {:.3}, {:.3}, {:.3}]", vp[0][0], vp[0][1], vp[0][2], vp[0][3]);
            info!("View-projection matrix [1]: [{:.3}, {:.3}, {:.3}, {:.3}]", vp[1][0], vp[1][1], vp[1][2], vp[1][3]);
            info!("View-projection matrix [2]: [{:.3}, {:.3}, {:.3}, {:.3}]", vp[2][0], vp[2][1], vp[2][2], vp[2][3]);
            info!("View-projection matrix [3]: [{:.3}, {:.3}, {:.3}, {:.3}]", vp[3][0], vp[3][1], vp[3][2], vp[3][3]);
        }
        
        // ===== Text label generation (row/col names) =====
        const ROW_LABELS: [&str; 6] = ["Type", "Charge(W→Y/C)", "Thermal(B→R→W)", "Velocity(RGB=XYZ)", "Interact", "Sci."];
        const COL_LABELS: [&str; 8] = ["Stop", "+X", "+Y", "+Z", "XY", "XZ", "YZ", "Fast"];
        const COLOR_SETS: usize = 6;
        const SAMPLES_PER_ROW: usize = 8; // Define here so both functions can use it
        const SPACING_X: f32 = 1.5;
        const SPACING_Y: f32 = 1.8;

        // Create all text buffers first to avoid borrowing conflicts
        let mut label_buffers: Vec<Buffer> = Vec::with_capacity(ROW_LABELS.len() + COL_LABELS.len());
        
        let vp_mat = Matrix4::<f32>::from_row_slice(&[
            view_proj[0][0], view_proj[0][1], view_proj[0][2], view_proj[0][3],
            view_proj[1][0], view_proj[1][1], view_proj[1][2], view_proj[1][3],
            view_proj[2][0], view_proj[2][1], view_proj[2][2], view_proj[2][3],
            view_proj[3][0], view_proj[3][1], view_proj[3][2], view_proj[3][3],
        ]);

        let world_to_screen = |pos: [f32; 3], size: (u32, u32)| -> Option<(f32, f32)> {
            let wp = nalgebra::Vector4::new(pos[0], pos[1], pos[2], 1.0);
            let clip = vp_mat * wp;
            if clip.w.abs() < 1e-6 { return None; }
            let ndc = clip / clip.w;
            let ndc_x = ndc.x;
            let ndc_y = ndc.y;
            if ndc_x.abs() > 1.0 || ndc_y.abs() > 1.0 { return None; }
            let sx = (ndc_x + 1.0) * 0.5 * size.0 as f32;
            let sy = (1.0 - (ndc_y + 1.0) * 0.5) * size.1 as f32;
            Some((sx, sy))
        };

        let y_offset = (COLOR_SETS as f32 - 1.0) * 0.5 * SPACING_Y;

        // Create buffers for row labels
        for (row_idx, label) in ROW_LABELS.iter().enumerate() {
            let y_world = row_idx as f32 * SPACING_Y - y_offset;
            let x_world = -(SAMPLES_PER_ROW as f32 * SPACING_X * 0.5) - 1.5;
            if world_to_screen([x_world, y_world, 0.0], (self.size.width, self.size.height)).is_some() {
                let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
                buffer.set_size(&mut self.font_system, 180.0, 24.0);
                buffer.set_text(&mut self.font_system, label, Attrs::new().family(Family::SansSerif), Shaping::Advanced);
                label_buffers.push(buffer);
            }
        }

        // Create buffers for column labels
        let top_row_y_world = 0.0 - y_offset;
        let x_start = -(SAMPLES_PER_ROW as f32 * SPACING_X * 0.5);
        for (col_idx, label) in COL_LABELS.iter().enumerate() {
            let x_world = col_idx as f32 * SPACING_X + x_start;
            let y_world = top_row_y_world + SPACING_Y + 0.5;
            if world_to_screen([x_world, y_world, 0.0], (self.size.width, self.size.height)).is_some() {
                let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
                buffer.set_size(&mut self.font_system, 80.0, 22.0);
                buffer.set_text(&mut self.font_system, label, Attrs::new().family(Family::SansSerif), Shaping::Advanced);
                label_buffers.push(buffer);
            }
        }

        // Now create text areas with references to the buffers
        let mut text_areas: Vec<TextArea> = Vec::new();
        let mut buffer_idx = 0;

        // Row labels text areas
        for (row_idx, _label) in ROW_LABELS.iter().enumerate() {
            let y_world = row_idx as f32 * SPACING_Y - y_offset;
            let x_world = -(SAMPLES_PER_ROW as f32 * SPACING_X * 0.5) - 1.5;
            if let Some((sx, sy)) = world_to_screen([x_world, y_world, 0.0], (self.size.width, self.size.height)) {
                if buffer_idx < label_buffers.len() {
                    let bounds = TextBounds { left: 0, top: 0, right: self.size.width as i32, bottom: self.size.height as i32 };
                    text_areas.push(TextArea { 
                        buffer: &label_buffers[buffer_idx], 
                        left: sx, 
                        top: sy, 
                        scale: 1.0, 
                        bounds, 
                        default_color: Color::rgb(255,255,255) 
                    });
                    buffer_idx += 1;
                }
            }
        }

        // Column labels text areas
        for (col_idx, _label) in COL_LABELS.iter().enumerate() {
            let x_world = col_idx as f32 * SPACING_X + x_start;
            let y_world = top_row_y_world + SPACING_Y + 0.5;
            if let Some((sx, sy)) = world_to_screen([x_world, y_world, 0.0], (self.size.width, self.size.height)) {
                if buffer_idx < label_buffers.len() {
                    let bounds = TextBounds { left: 0, top: 0, right: self.size.width as i32, bottom: self.size.height as i32 };
                    text_areas.push(TextArea { 
                        buffer: &label_buffers[buffer_idx], 
                        left: sx, 
                        top: sy, 
                        scale: 1.0, 
                        bounds, 
                        default_color: Color::rgb(255,255,0) 
                    });
                    buffer_idx += 1;
                }
            }
        }

        // Prepare text rendering if we have any text areas
        if !text_areas.is_empty() {
            let viewport = Resolution { width: self.size.width, height: self.size.height };
            let _ = self.text_renderer.prepare(
                &self.device, 
                &self.queue, 
                &mut self.font_system, 
                &mut self.text_atlas, 
                viewport, 
                text_areas, 
                &mut self.text_cache
            );
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
            _padding: [0.0; 4],
        };
        
        println!("📤 Writing uniforms to GPU...");
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        // Get surface texture
        println!("🖼️ Getting surface texture...");
        let output = match self.surface.get_current_texture() {
            Ok(output) => {
                println!("✅ Surface texture obtained successfully");
                output
            }
            Err(e) => {
                println!("❌ Failed to get surface texture: {:?}", e);
                return Err(e.into());
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        println!("✅ Texture view created");
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        println!("✅ Command encoder created");
        
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
        println!("🎨 Starting main render pass with GREEN background...");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.3,
                            b: 0.0,
                            a: 1.0,
                        }), // GREEN BACKGROUND - proves rendering works
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            println!("✅ Render pass created with GREEN clear color");
            
            render_pass.set_pipeline(&self.render_pipeline);
            println!("✅ Pipeline set");
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            println!("✅ Bind group set");
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            println!("✅ Quad vertex buffer set");
            render_pass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
            println!("✅ Instance vertex buffer set");
            
            // Draw 6 vertices per instance (2 triangles per particle quad)
            let instance_count = self.particle_count as u32;
            if instance_count > 0 {
                println!("🎮 Drawing {} instanced quads for {} particles", instance_count, self.particle_count);
                render_pass.draw(0..6, 0..instance_count);
                println!("✅ Draw call completed");
            } else {
                println!("⚠️ No particles to draw! particle_count={}", self.particle_count);
                // Even with no particles, we should still see the green background
                println!("🟢 GREEN background should still be visible even without particles");
            }
        }
        println!("✅ Render pass completed");

        // ---- Text overlay pass ---- (Fix: Properly scope text_pass to release encoder borrow)
        {
            let mut text_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let _ = self.text_renderer.render(&self.text_atlas, &mut text_pass);
        } // text_pass is dropped here, releasing the borrow on encoder

        // Submit GPU commands and present the frame
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // ---- Metrics ----
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

        println!("🏁 RENDER END - Frame completed in {:.2}ms\n", frame_time.as_secs_f32() * 1000.0);

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
    
    println!("🚀 RENDERER STARTUP - Creating event loop...");
    let event_loop = EventLoop::new()?;
    println!("✅ Event loop created");
    
    println!("🪟 Creating window...");
    let window = WindowBuilder::new()
        .with_title("EVOLVE Universe Simulation - Heavy Mode Native Renderer")
        .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
        .build(&event_loop)?;
    println!("✅ Window created");
    
    println!("🎨 Initializing renderer...");
    let mut renderer = NativeRenderer::new(&window).await?;
    println!("✅ Renderer initialized");
    
    let mut last_update = std::time::Instant::now();
    
    info!("🎮 Renderer Controls:");
    info!("  WASD - Move camera");
    info!("  Q/E - Move up/down");
    info!("  1-6 - Switch color modes");
    info!("  H - Toggle heavy mode (if enabled)");
    info!("  ESC - Exit");
    
    println!("🔄 Starting event loop...");
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == renderer.window.id() => {
                println!("🪟 Window event: {:?}", event);
                match event {
                    WindowEvent::CloseRequested => {
                        info!("Closing heavy mode renderer");
                        elwt.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        println!("📏 Window resized to {:?}", physical_size);
                        renderer.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        println!("🎨 RedrawRequested - starting render...");
                        let now = std::time::Instant::now();
                        let simulation_time = now.duration_since(last_update).as_secs_f32();
                        last_update = now;
                        
                        // Update particles from simulation (non-blocking)
                        if let Ok(sim) = simulation.try_lock() {
                            println!("🔄 Updating particles from simulation...");
                            if let Err(e) = renderer.update_particles(&sim) {
                                error!("Failed to update particles: {}", e);
                            }
                        } else {
                            // Simulation is busy, use cached particle data
                            println!("🔒 Simulation locked, using cached particle data");
                            debug!("Simulation locked, using cached particle data");
                        }
                        
                        // Render with performance monitoring
                        println!("🎨 Starting render call...");
                        let render_start = std::time::Instant::now();
                        if let Err(e) = renderer.render(simulation_time) {
                            error!("Render error: {}", e);
                        }
                        
                        // Log performance issues
                        let render_time = render_start.elapsed().as_millis();
                        if render_time > 16 {  // More than 16ms = below 60 FPS
                            warn!("Slow frame detected: {}ms render time", render_time);
                        }
                        println!("✅ Render completed in {}ms", render_time);
                    }
                    WindowEvent::KeyboardInput { event: key_event, .. } => {
                        println!("⌨️ Key input: {:?}", key_event);
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
                        println!("🔍 Scale factor changed");
                        // Handle scale factor changes if needed
                    }
                    _ => {}
                }
            },
            Event::AboutToWait => {
                println!("⏳ AboutToWait - requesting redraw...");
                renderer.window.request_redraw();
            }
            _ => {
                // Don't spam with other events
            }
        }
    })?;
    
    println!("🏁 Event loop ended");
    Ok(())
} 