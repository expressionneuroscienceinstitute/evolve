//! High-Performance Native Renderer for Universe Simulation
//! 
//! GPU-accelerated particle rendering with direct memory access to physics data.
//! Eliminates WebSocket overhead for maximum performance.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use nalgebra::{Vector3, Point3};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tracing::{info, warn, error};
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

pub use universe_sim::UniverseSimulation;

/// High-performance particle vertex for GPU rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleVertex {
    pub position: [f32; 3],
    pub velocity: [f32; 3], 
    pub mass: f32,
    pub charge: f32,
    pub temperature: f32,
    pub _padding: [f32; 3], // GPU alignment
}

/// Camera state for 3D navigation
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
}

/// High-performance renderer state
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
    
    // Camera and view state
    camera: Camera,
    view_matrix: [[f32; 4]; 4],
    proj_matrix: [[f32; 4]; 4],
    
    // Performance metrics
    frame_count: u64,
    last_fps_time: std::time::Instant,
    fps: f32,
    
    // Particle data
    particle_count: usize,
    max_particles: usize,
}

/// Uniform data sent to GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    time: f32,
    scale: f32,
    _padding: [f32; 2],
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: 45.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: 1000.0,
            zoom_speed: 0.1,
            pan_speed: 0.01,
        }
    }
}

impl Camera {
    /// Update view matrix based on camera parameters
    pub fn update_view(&mut self) {
        // Basic lookAt matrix calculation
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = right.cross(&forward);
        
        self.up = up;
    }
    
    /// Get view-projection matrix for GPU
    pub fn get_view_proj_matrix(&self) -> [[f32; 4]; 4] {
        // Calculate view matrix
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = right.cross(&forward);
        
        let _view = [
            [right.x, up.x, -forward.x, 0.0],
            [right.y, up.y, -forward.y, 0.0],
            [right.z, up.z, -forward.z, 0.0],
            [-right.dot(&self.position.coords), -up.dot(&self.position.coords), forward.dot(&self.position.coords), 1.0],
        ];
        
        // Calculate projection matrix
        let f = 1.0 / (self.fov / 2.0).tan();
        let proj = [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far + self.near) / (self.near - self.far), -1.0],
            [0.0, 0.0, (2.0 * self.far * self.near) / (self.near - self.far), 0.0],
        ];
        
        // Multiply view * proj (simplified for demonstration)
        // In a real implementation, use proper matrix multiplication
        proj // Return projection for now
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
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) mass: f32,
    @location(3) charge: f32,
    @location(4) temperature: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(vertex.position, 1.0);
    
    // Color based on temperature and charge
    let temp_factor = clamp(vertex.temperature / 10000.0, 0.0, 1.0);
    let charge_factor = clamp(abs(vertex.charge) / 1e-18, 0.0, 1.0);
    out.color = vec3<f32>(temp_factor, 0.5, charge_factor);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create uniform buffer
        let max_particles = 1_000_000; // 1M particles max
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create vertex buffer for particles
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Vertex Buffer"),
            size: (max_particles * std::mem::size_of::<ParticleVertex>()) as u64,
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
                topology: wgpu::PrimitiveTopology::PointList,
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
            camera,
            view_matrix: [[0.0; 4]; 4],
            proj_matrix: [[0.0; 4]; 4],
            frame_count: 0,
            last_fps_time: std::time::Instant::now(),
            fps: 0.0,
            particle_count: 0,
            max_particles,
        })
    }
    
    /// Update particle data from simulation with zero-copy access
    pub fn update_particles(&mut self, simulation: &UniverseSimulation) -> Result<()> {
        // Access the new SoA particle store
        let particles = &simulation.store.particles;
        
        // Convert particles to GPU format in parallel
        let gpu_particles: Vec<ParticleVertex> = (0..particles.count.min(self.max_particles))
            .into_par_iter()
            .map(|i| ParticleVertex {
                position: [
                    particles.position[i].x as f32, 
                    particles.position[i].y as f32, 
                    particles.position[i].z as f32
                ],
                velocity: [
                    particles.velocity[i].x as f32, 
                    particles.velocity[i].y as f32, 
                    particles.velocity[i].z as f32
                ],
                mass: particles.mass[i] as f32,
                charge: particles.charge[i] as f32,
                temperature: particles.temperature[i] as f32,
                _padding: [0.0; 3],
            })
            .collect();
        
        self.particle_count = gpu_particles.len();
        
        // Upload to GPU buffer
        self.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&gpu_particles),
        );
        
        info!("Updated {} particles on GPU", self.particle_count);
        Ok(())
    }
    
    /// Render frame with maximum performance
    pub fn render(&mut self, simulation_time: f32) -> Result<()> {
        // Update camera
        self.camera.update_view();
        let view_proj = self.camera.get_view_proj_matrix();
        
        // Update uniforms
        let uniforms = Uniforms {
            view_proj,
            time: simulation_time,
            scale: 1.0,
            _padding: [0.0; 2],
        };
        
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        // Render
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
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
            render_pass.draw(0..self.particle_count as u32, 0..1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        // Update FPS
        self.frame_count += 1;
        let now = std::time::Instant::now();
        if now.duration_since(self.last_fps_time).as_secs() >= 1 {
            self.fps = self.frame_count as f32 / now.duration_since(self.last_fps_time).as_secs_f32();
            self.frame_count = 0;
            self.last_fps_time = now;
            info!("Renderer FPS: {:.1}, Particles: {}", self.fps, self.particle_count);
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
        }
    }
    
    /// Get current FPS for performance monitoring
    pub fn get_fps(&self) -> f32 {
        self.fps
    }
    
    /// Get particle count
    pub fn get_particle_count(&self) -> usize {
        self.particle_count
    }
}

/// Create and run the high-performance renderer
pub async fn run_renderer(simulation: Arc<Mutex<UniverseSimulation>>) -> Result<()> {
    info!("Starting high-performance native renderer");
    
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Universe Simulation - Native Renderer")
        .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
        .build(&event_loop)?;
    
    let mut renderer = NativeRenderer::new(&window).await?;
    let mut last_update = std::time::Instant::now();
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == renderer.window.id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(*physical_size);
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
                let dt = now.duration_since(last_update).as_secs_f32();
                last_update = now;
                
                // Update particles from simulation (lock briefly)
                if let Ok(sim) = simulation.try_lock() {
                    if let Err(e) = renderer.update_particles(&sim) {
                        error!("Failed to update particles: {}", e);
                    }
                }
                
                // Render
                if let Err(e) = renderer.render(dt) {
                    error!("Render error: {}", e);
                }
            }
            _ => {}
        }
    })?;
    
    Ok(())
} 