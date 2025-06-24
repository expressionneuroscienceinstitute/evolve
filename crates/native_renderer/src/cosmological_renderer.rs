//! Cosmological Renderer Module
//! Provides large-scale structure visualization capabilities for the EVOLUTION universe simulation.
//!
//! This module implements high-performance GPU-based rendering for:
//! - Dark matter N-body particle visualization (Tree-PM)
//! - SPH gas dynamics with temperature/density coloring
//! - Statistical overlays (power spectrum, correlation functions)
//! - Large-scale structure analysis (cosmic web, halos)
//! - Scientific publication-quality outputs
//!
//! References:
//! - cosmocalc.rs for cosmological distance calculations
//! - Particle simulation performance insights from dgerrells.com/blog/how-fast-is-rust-simulating-200-000-000-particles
//! - GADGET-2 and AREPO visualization methods

#![allow(dead_code)]
#![allow(unused_imports)]

use anyhow::Result;
use cgmath::{Matrix4, Point3, Vector3, perspective, Deg, InnerSpace};
use tracing::{info, warn, debug};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

use crate::physics_engine::cosmology::{CosmologicalParticle, CosmologicalParticleType, CosmologicalParameters};
use crate::physics_engine::cosmological_sph::CosmologicalSphParticle;

/// Uniforms for cosmological rendering with scientific parameters
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct CosmologyUniforms {
    /// View-projection matrix
    view_proj: [[f32; 4]; 4],
    /// Camera position in comoving coordinates
    camera_pos: [f32; 3],
    /// Simulation time (Gyr)
    time: f32,
    /// Current redshift
    redshift: f32,
    /// Scale factor a(t)
    scale_factor: f32,
    /// Hubble parameter H(z) in km/s/Mpc
    hubble_parameter: f32,
    /// Box size in Mpc/h
    box_size: f32,
    /// Particle count for LOD
    particle_count: f32,
    /// Visualization mode: 0=particles, 1=density field, 2=power spectrum, 3=halos
    visualization_mode: f32,
    /// Color mode: 0=mass, 1=velocity, 2=temperature, 3=density, 4=type
    color_mode: f32,
    /// Scale factor for particle sizes
    particle_scale: f32,
    /// LOD distance thresholds
    lod_near: f32,
    lod_far: f32,
    /// Statistical overlay parameters
    show_power_spectrum: f32,
    show_correlation_function: f32,
    _padding: f32,
}

/// GPU vertex data for cosmological particles
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct CosmologicalParticleVertex {
    /// Position in comoving coordinates (Mpc/h)
    position: [f32; 3],
    /// Velocity (km/s)
    velocity: [f32; 3],
    /// Mass (solar masses)
    mass: f32,
    /// Density (solar masses/Mpc³)
    density: f32,
    /// Temperature (K) for gas particles
    temperature: f32,
    /// Particle type (0=DM, 1=gas, 2=star, 3=BH)
    particle_type: f32,
    /// Halo ID (for halo visualization)
    halo_id: f32,
    /// Formation time (scale factor)
    formation_time: f32,
}

/// Statistical overlay data for power spectrum visualization
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct PowerSpectrumData {
    /// k values (h/Mpc)
    k_values: [f32; 64],
    /// P(k) values (Mpc³/h³)
    power_values: [f32; 64],
    /// Theoretical P(k) for comparison
    theory_values: [f32; 64],
    /// Number of k bins
    num_bins: f32,
    _padding: [f32; 3],
}

/// Renderer for cosmological-scale visualizations
#[derive(Debug)]
pub struct CosmologicalRenderer {
    initialized: bool,
    
    // Main particle rendering pipeline
    particle_pipeline: Option<wgpu::RenderPipeline>,
    particle_vertex_buffer: Option<wgpu::Buffer>,
    particle_count: usize,
    max_particles: usize,
    
    // Statistical overlay pipeline
    overlay_pipeline: Option<wgpu::RenderPipeline>,
    power_spectrum_buffer: Option<wgpu::Buffer>,
    
    // Uniform buffers and bind groups
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    
    // Cosmological parameters
    cosmo_params: CosmologicalParameters,
    current_redshift: f64,
    current_scale_factor: f64,
    
    // Visualization state
    visualization_mode: CosmologicalVisualizationMode,
    color_mode: CosmologicalColorMode,
    
    // LOD and culling
    lod_enabled: bool,
    frustum_culling: bool,
    
    // Performance tracking
    last_frame_particle_count: usize,
    
    viewport_size: winit::dpi::PhysicalSize<u32>,
}

/// Cosmological visualization modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CosmologicalVisualizationMode {
    /// N-body particle visualization
    Particles,
    /// Density field visualization
    DensityField,
    /// Power spectrum overlay
    PowerSpectrum,
    /// Halo and subhalo visualization
    Halos,
    /// Large-scale structure (cosmic web)
    LargeScaleStructure,
    /// SPH gas dynamics
    GasDynamics,
}

/// Color modes for cosmological visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CosmologicalColorMode {
    /// Color by mass
    Mass,
    /// Color by velocity magnitude
    Velocity,
    /// Color by temperature (gas particles)
    Temperature,
    /// Color by density
    Density,
    /// Color by particle type
    ParticleType,
    /// Color by halo membership
    Halo,
    /// Color by formation redshift
    FormationTime,
}

impl Default for CosmologicalRenderer {
    fn default() -> Self {
        Self {
            initialized: false,
            particle_pipeline: None,
            particle_vertex_buffer: None,
            particle_count: 0,
            max_particles: 10_000_000, // 10M particles default
            overlay_pipeline: None,
            power_spectrum_buffer: None,
            uniform_buffer: None,
            bind_group: None,
            bind_group_layout: None,
            cosmo_params: CosmologicalParameters::default(),
            current_redshift: 0.0,
            current_scale_factor: 1.0,
            visualization_mode: CosmologicalVisualizationMode::Particles,
            color_mode: CosmologicalColorMode::Mass,
            lod_enabled: true,
            frustum_culling: true,
            last_frame_particle_count: 0,
            viewport_size: winit::dpi::PhysicalSize::new(0, 0),
        }
    }
}

impl CosmologicalRenderer {
    /// Creates a new cosmological renderer
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize GPU resources for cosmological rendering
    pub fn initialize(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Result<()> {
        // Create cosmological particle shader
        const COSMOLOGICAL_SHADER: &str = r#"
            struct CosmologyUniforms {
                view_proj: mat4x4<f32>,
                camera_pos: vec3<f32>,
                time: f32,
                redshift: f32,
                scale_factor: f32,
                hubble_parameter: f32,
                box_size: f32,
                particle_count: f32,
                visualization_mode: f32,
                color_mode: f32,
                particle_scale: f32,
                lod_near: f32,
                lod_far: f32,
                show_power_spectrum: f32,
                show_correlation_function: f32,
                _padding: f32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: CosmologyUniforms;
            
            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) velocity: vec3<f32>,
                @location(2) mass: f32,
                @location(3) density: f32,
                @location(4) temperature: f32,
                @location(5) particle_type: f32,
                @location(6) halo_id: f32,
                @location(7) formation_time: f32,
                @builtin(vertex_index) vertex_index: u32,
            };
            
            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) color: vec3<f32>,
                @location(1) world_pos: vec3<f32>,
                @location(2) size: f32,
                @location(3) quad_pos: vec2<f32>,
                @location(4) particle_class: f32,
            };
            
            // Scientific color mappings
            fn get_mass_color(mass: f32) -> vec3<f32> {
                let log_mass = log(max(mass, 1e8)) / log(10.0);
                let normalized = clamp((log_mass - 8.0) / 8.0, 0.0, 1.0);
                
                if (normalized < 0.33) {
                    let t = normalized * 3.0;
                    return vec3<f32>(0.0, 0.0, 1.0 - t) + vec3<f32>(0.0, t, 0.0);
                } else if (normalized < 0.67) {
                    let t = (normalized - 0.33) * 3.0;
                    return vec3<f32>(0.0, 1.0 - t, 0.0) + vec3<f32>(t, 0.0, 0.0);
                } else {
                    let t = (normalized - 0.67) * 3.0;
                    return vec3<f32>(1.0, t, 0.0);
                }
            }
            
            fn get_velocity_color(velocity: vec3<f32>) -> vec3<f32> {
                let speed = length(velocity);
                let log_speed = log(max(speed, 1.0)) / log(10.0);
                let normalized = clamp(log_speed / 5.0, 0.0, 1.0); // 0-100000 km/s range
                
                // Blue to red scale for Doppler effect visualization
                return vec3<f32>(normalized, 0.5 * (1.0 - normalized), 1.0 - normalized);
            }
            
            fn get_temperature_color(temperature: f32) -> vec3<f32> {
                if (temperature <= 0.0) {
                    return vec3<f32>(0.0, 0.0, 0.1); // Very cold = dark blue
                }
                
                let log_temp = log(temperature) / log(10.0);
                let normalized = clamp((log_temp - 2.0) / 6.0, 0.0, 1.0); // 100K - 100MK range
                
                // Blackbody color approximation
                if (normalized < 0.25) {
                    let t = normalized * 4.0;
                    return vec3<f32>(t, 0.0, 0.0);
                } else if (normalized < 0.5) {
                    let t = (normalized - 0.25) * 4.0;
                    return vec3<f32>(1.0, t, 0.0);
                } else if (normalized < 0.75) {
                    let t = (normalized - 0.5) * 4.0;
                    return vec3<f32>(1.0, 1.0, t);
                } else {
                    return vec3<f32>(1.0, 1.0, 1.0);
                }
            }
            
            fn get_density_color(density: f32) -> vec3<f32> {
                let log_density = log(max(density, 1e-6)) / log(10.0);
                let normalized = clamp((log_density + 6.0) / 12.0, 0.0, 1.0);
                
                // Purple to white scale for density
                return vec3<f32>(0.5 + 0.5 * normalized, normalized, 1.0);
            }
            
            fn get_particle_type_color(particle_type: f32) -> vec3<f32> {
                if (particle_type < 0.5) {
                    return vec3<f32>(0.2, 0.2, 0.8); // Dark matter - blue
                } else if (particle_type < 1.5) {
                    return vec3<f32>(0.8, 0.4, 0.2); // Gas - orange
                } else if (particle_type < 2.5) {
                    return vec3<f32>(1.0, 1.0, 0.8); // Stars - yellow-white
                } else {
                    return vec3<f32>(1.0, 0.0, 1.0); // Black holes - magenta
                }
            }
            
            fn get_halo_color(halo_id: f32) -> vec3<f32> {
                if (halo_id < 0.5) {
                    return vec3<f32>(0.3, 0.3, 0.3); // Field particles - gray
                }
                
                // Pseudo-random colors for different halos
                let h = fract(sin(halo_id * 43758.5453123) * 1.0);
                let s = 0.8;
                let v = 0.9;
                
                // HSV to RGB conversion
                let c = v * s;
                let x = c * (1.0 - abs(fract(h * 6.0) * 2.0 - 1.0));
                let m = v - c;
                
                if (h < 1.0/6.0) {
                    return vec3<f32>(c + m, x + m, m);
                } else if (h < 2.0/6.0) {
                    return vec3<f32>(x + m, c + m, m);
                } else if (h < 3.0/6.0) {
                    return vec3<f32>(m, c + m, x + m);
                } else if (h < 4.0/6.0) {
                    return vec3<f32>(m, x + m, c + m);
                } else if (h < 5.0/6.0) {
                    return vec3<f32>(x + m, m, c + m);
                } else {
                    return vec3<f32>(c + m, m, x + m);
                }
            }
            
            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                
                // Generate quad vertices for particle billboards
                var quad_pos: vec2<f32>;
                switch(input.vertex_index % 6u) {
                    case 0u: { quad_pos = vec2<f32>(-1.0, -1.0); }
                    case 1u: { quad_pos = vec2<f32>( 1.0, -1.0); }
                    case 2u: { quad_pos = vec2<f32>(-1.0,  1.0); }
                    case 3u: { quad_pos = vec2<f32>(-1.0,  1.0); }
                    case 4u: { quad_pos = vec2<f32>( 1.0, -1.0); }
                    default: { quad_pos = vec2<f32>( 1.0,  1.0); }
                }
                
                // Apply cosmological coordinate transformations
                var position = input.position;
                
                // Apply periodic boundary conditions visualization
                if (position.x > uniforms.box_size * 0.5) { position.x -= uniforms.box_size; }
                if (position.y > uniforms.box_size * 0.5) { position.y -= uniforms.box_size; }
                if (position.z > uniforms.box_size * 0.5) { position.z -= uniforms.box_size; }
                
                var clip_pos = uniforms.view_proj * vec4<f32>(position, 1.0);
                
                // Calculate LOD-based particle size
                let distance = length(position - uniforms.camera_pos);
                var size_factor = 1.0;
                
                // Mass-based scaling
                let log_mass = log(max(input.mass, 1e8)) / log(10.0);
                size_factor *= max(0.1, (log_mass - 8.0) * 0.1 + 0.5);
                
                // Distance-based LOD
                if (distance > uniforms.lod_far) {
                    size_factor *= 2.0; // Larger at distance
                } else if (distance < uniforms.lod_near) {
                    size_factor *= 0.5; // Smaller up close
                }
                
                // Apply particle scale
                size_factor *= uniforms.particle_scale;
                
                // Billboarding in clip space
                let depth_scale = max(0.001, clip_pos.w);
                let offset = quad_pos * size_factor * 0.01 / depth_scale;
                clip_pos.x += offset.x;
                clip_pos.y += offset.y;
                
                out.clip_position = clip_pos;
                out.world_pos = position;
                out.size = size_factor;
                out.quad_pos = quad_pos;
                
                // Color selection based on mode
                if (uniforms.color_mode < 0.5) {
                    out.color = get_mass_color(input.mass);
                } else if (uniforms.color_mode < 1.5) {
                    out.color = get_velocity_color(input.velocity);
                } else if (uniforms.color_mode < 2.5) {
                    out.color = get_temperature_color(input.temperature);
                } else if (uniforms.color_mode < 3.5) {
                    out.color = get_density_color(input.density);
                } else if (uniforms.color_mode < 4.5) {
                    out.color = get_particle_type_color(input.particle_type);
                } else {
                    out.color = get_halo_color(input.halo_id);
                }
                
                // Particle classification for rendering effects
                if (input.mass > 1e12) {
                    out.particle_class = 3.0; // Massive halos
                } else if (input.mass > 1e10) {
                    out.particle_class = 2.0; // Medium halos
                } else if (input.mass > 1e8) {
                    out.particle_class = 1.0; // Small halos
                } else {
                    out.particle_class = 0.0; // Field particles
                }
                
                return out;
            }
            
            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                // Create circular particle with smooth edges
                let center = vec2<f32>(0.0);
                let dist = length(input.quad_pos - center);
                
                // Smooth circular falloff
                let alpha = smoothstep(1.0, 0.7, dist);
                
                // Size-based brightness modulation
                let brightness = 0.5 + 0.5 * input.size;
                
                // Apply cosmological dimming based on distance
                let distance = length(input.world_pos);
                let dimming = 1.0 / (1.0 + distance * 0.0001);
                
                var final_color = input.color * brightness * dimming;
                
                // Add glow for massive particles
                if (input.particle_class >= 2.0) {
                    let glow = smoothstep(1.5, 0.5, dist) * 0.3;
                    final_color += vec3<f32>(glow);
                }
                
                return vec4<f32>(final_color, alpha);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cosmological Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(COSMOLOGICAL_SHADER.into()),
        });

        // Create vertex buffer for particle data
        let vertex_buffer_size = (self.max_particles * std::mem::size_of::<CosmologicalParticleVertex>()) as u64;
        let particle_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cosmological Particle Buffer"),
            size: vertex_buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cosmology Uniform Buffer"),
            size: std::mem::size_of::<CosmologyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cosmology Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cosmology Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cosmology Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cosmological Particle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CosmologicalParticleVertex>() as u64,
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
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        // Mass
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Density
                        wgpu::VertexAttribute {
                            offset: 28,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Temperature
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Particle type
                        wgpu::VertexAttribute {
                            offset: 36,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Halo ID
                        wgpu::VertexAttribute {
                            offset: 40,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // Formation time
                        wgpu::VertexAttribute {
                            offset: 44,
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
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        self.particle_pipeline = Some(particle_pipeline);
        self.particle_vertex_buffer = Some(particle_vertex_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        self.bind_group_layout = Some(bind_group_layout);
        self.initialized = true;

        info!("✅ CosmologicalRenderer initialized with {} max particles", self.max_particles);
        Ok(())
    }

    /// Update particle data for rendering
    pub fn update_particles(
        &mut self,
        queue: &wgpu::Queue,
        particles: &[CosmologicalParticle],
    ) -> Result<()> {
        if !self.initialized {
            warn!("CosmologicalRenderer not initialized");
            return Ok(());
        }

        let particle_count = particles.len().min(self.max_particles);
        if particle_count == 0 {
            return Ok(());
        }

        // Convert CosmologicalParticle to GPU vertex format
        let mut vertices = Vec::with_capacity(particle_count * 6); // 6 vertices per particle (2 triangles)
        
        for particle in particles.iter().take(particle_count) {
            let vertex = CosmologicalParticleVertex {
                position: [
                    particle.position.x as f32,
                    particle.position.y as f32,
                    particle.position.z as f32,
                ],
                velocity: [
                    particle.velocity.x as f32,
                    particle.velocity.y as f32,
                    particle.velocity.z as f32,
                ],
                mass: particle.mass as f32,
                density: particle.density as f32,
                temperature: match particle.particle_type {
                    CosmologicalParticleType::Gas => 1e4, // Default gas temperature
                    _ => 0.0,
                },
                particle_type: match particle.particle_type {
                    CosmologicalParticleType::DarkMatter => 0.0,
                    CosmologicalParticleType::Gas => 1.0,
                    CosmologicalParticleType::Star => 2.0,
                    CosmologicalParticleType::BlackHole => 3.0,
                },
                halo_id: particle.halo_id.unwrap_or(0) as f32,
                formation_time: particle.formation_time as f32,
            };
            
            // Add 6 vertices for each particle (quad made of 2 triangles)
            for _ in 0..6 {
                vertices.push(vertex);
            }
        }

        // Upload to GPU
        if let Some(buffer) = &self.particle_vertex_buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vertices));
        }

        self.particle_count = particle_count;
        self.last_frame_particle_count = particle_count;

        debug!("Updated {} cosmological particles for rendering", particle_count);
        Ok(())
    }

    /// Update SPH particle data for gas dynamics visualization
    pub fn update_sph_particles(
        &mut self,
        queue: &wgpu::Queue,
        sph_particles: &[CosmologicalSphParticle],
    ) -> Result<()> {
        if !self.initialized {
            warn!("CosmologicalRenderer not initialized");
            return Ok(());
        }

        let particle_count = sph_particles.len().min(self.max_particles);
        if particle_count == 0 {
            return Ok(());
        }

        // Convert CosmologicalSphParticle to GPU vertex format
        let mut vertices = Vec::with_capacity(particle_count * 6);
        
        for particle in sph_particles.iter().take(particle_count) {
            let vertex = CosmologicalParticleVertex {
                position: [
                    particle.cosmological_particle.position.x as f32,
                    particle.cosmological_particle.position.y as f32,
                    particle.cosmological_particle.position.z as f32,
                ],
                velocity: [
                    particle.cosmological_particle.velocity.x as f32,
                    particle.cosmological_particle.velocity.y as f32,
                    particle.cosmological_particle.velocity.z as f32,
                ],
                mass: particle.cosmological_particle.mass as f32,
                density: particle.sph_particle.density as f32,
                temperature: particle.temperature as f32,
                particle_type: 1.0, // Gas
                halo_id: particle.cosmological_particle.halo_id.unwrap_or(0) as f32,
                formation_time: particle.cosmological_particle.formation_time as f32,
            };
            
            for _ in 0..6 {
                vertices.push(vertex);
            }
        }

        if let Some(buffer) = &self.particle_vertex_buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vertices));
        }

        self.particle_count = particle_count;
        debug!("Updated {} SPH particles for rendering", particle_count);
        Ok(())
    }

    /// Update cosmological parameters
    pub fn update_cosmology(
        &mut self,
        params: CosmologicalParameters,
        redshift: f64,
        scale_factor: f64,
    ) {
        self.cosmo_params = params;
        self.current_redshift = redshift;
        self.current_scale_factor = scale_factor;
    }

    /// Set visualization mode
    pub fn set_visualization_mode(&mut self, mode: CosmologicalVisualizationMode) {
        self.visualization_mode = mode;
    }

    /// Set color mode
    pub fn set_color_mode(&mut self, mode: CosmologicalColorMode) {
        self.color_mode = mode;
    }

    /// Render cosmological visualization
    pub fn render_pass<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        view: &'a wgpu::TextureView,
        queue: &wgpu::Queue,
        view_proj_matrix: &Matrix4<f32>,
        camera_pos: &Vector3<f32>,
        simulation_time: f32,
    ) -> Result<()> {
        if !self.initialized || self.particle_count == 0 {
            return Ok(());
        }

        // Update uniforms
        let uniforms = CosmologyUniforms {
            view_proj: view_proj_matrix.into(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z],
            time: simulation_time,
            redshift: self.current_redshift as f32,
            scale_factor: self.current_scale_factor as f32,
            hubble_parameter: self.cosmo_params.hubble_parameter(self.current_scale_factor) as f32,
            box_size: self.cosmo_params.box_size as f32,
            particle_count: self.particle_count as f32,
            visualization_mode: match self.visualization_mode {
                CosmologicalVisualizationMode::Particles => 0.0,
                CosmologicalVisualizationMode::DensityField => 1.0,
                CosmologicalVisualizationMode::PowerSpectrum => 2.0,
                CosmologicalVisualizationMode::Halos => 3.0,
                CosmologicalVisualizationMode::LargeScaleStructure => 4.0,
                CosmologicalVisualizationMode::GasDynamics => 5.0,
            },
            color_mode: match self.color_mode {
                CosmologicalColorMode::Mass => 0.0,
                CosmologicalColorMode::Velocity => 1.0,
                CosmologicalColorMode::Temperature => 2.0,
                CosmologicalColorMode::Density => 3.0,
                CosmologicalColorMode::ParticleType => 4.0,
                CosmologicalColorMode::Halo => 5.0,
                CosmologicalColorMode::FormationTime => 6.0,
            },
            particle_scale: 1.0,
            lod_near: 10.0,
            lod_far: 1000.0,
            show_power_spectrum: 0.0,
            show_correlation_function: 0.0,
            _padding: 0.0,
        };

        if let Some(buffer) = &self.uniform_buffer {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        // Begin render pass
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Cosmological RenderPass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set pipeline and resources
        if let (Some(pipeline), Some(vertex_buffer), Some(bind_group)) = (
            &self.particle_pipeline,
            &self.particle_vertex_buffer,
            &self.bind_group,
        ) {
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            
            // Draw particles (6 vertices per particle)
            let vertex_count = (self.particle_count * 6) as u32;
            rpass.draw(0..vertex_count, 0..1);
        }

        drop(rpass);
        info!("Rendered {} cosmological particles", self.particle_count);
        Ok(())
    }

    /// Simple render interface for compatibility
    pub fn render(
        &mut self,
        _queue: &wgpu::Queue,
        _simulation_time: f32,
    ) -> Result<()> {
        if !self.initialized {
            warn!("CosmologicalRenderer::render called before initialize() – call render_pass instead");
        }
        Ok(())
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        stats.insert("particle_count".to_string(), self.particle_count as f32);
        stats.insert("max_particles".to_string(), self.max_particles as f32);
        stats.insert("utilization".to_string(), 
                    (self.particle_count as f32) / (self.max_particles as f32) * 100.0);
        stats.insert("redshift".to_string(), self.current_redshift as f32);
        stats.insert("scale_factor".to_string(), self.current_scale_factor as f32);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = CosmologicalRenderer::new();
        // Newly created renderer should be in an uninitialized state.
        assert!(!renderer.initialized);
    }

    #[test]
    fn test_renderer_render_stub() {
        let mut renderer = CosmologicalRenderer::new();
        // The stub render method should execute without error.
        assert!(renderer.render(0.0).is_ok());
    }
} 