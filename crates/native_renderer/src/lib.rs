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

#![cfg_attr(
    all(not(feature = "unstable-renderer"), not(test)),
    // deny(warnings, clippy::all, clippy::pedantic)
    allow(warnings)
)]

#![cfg_attr(feature = "unstable-renderer", allow(dead_code))]
#![cfg_attr(feature = "unstable-renderer", allow(unused_imports))]

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, Point3, Vector3};
use cgmath::prelude::*; // InnerSpace, SquareMatrix, etc.
use cgmath::{Rad, perspective};
use glyphon::{FontSystem, SwashCache, TextAtlas, TextRenderer, TextArea, TextBounds, Metrics, Buffer as GlyphonBuffer, Color as GlyphonColor, Attrs, Family, Shaping, Resolution};
use nalgebra as na; // Used only for certain math utilities in debug functions
use tracing::{info, error, warn, debug};
// Add import for StagingBelt
use wgpu::util::StagingBelt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferDescriptor,
    BufferUsages, Color, CommandEncoder, Device, Features, FragmentState, Limits,
    MultisampleState, PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, ShaderStages, Surface, SurfaceConfiguration,
    TextureFormat, TextureUsages, VertexState, VertexBufferLayout,
};
use winit::{
    event::{ElementState},
    keyboard::{KeyCode, ModifiersState},
    window::Window,
};

// pub use universe_sim::UniverseSimulation;

// === BUFFER POOL IMPLEMENTATION ===

/// Tracks a buffer's lifecycle and metadata for safe destruction
#[derive(Debug)]
pub struct TrackedBuffer {
    pub id: wgpu::Id<wgpu::Buffer>,
    pub size: u64,
    pub label: String,
    pub last_submission: Option<wgpu::SubmissionIndex>,
    pub created_at: std::time::Instant,
}

impl TrackedBuffer {
    pub fn new(id: wgpu::Id<wgpu::Buffer>, size: u64, label: String) -> Self {
        Self {
            id,
            size,
            label,
            last_submission: None,
            created_at: std::time::Instant::now(),
        }
    }

    /// Check if this buffer is safe to destroy (GPU is done with it)
    pub fn is_safe_to_destroy(&self, device: &wgpu::Device) -> bool {
        match self.last_submission {
            None => true, // Never submitted, safe to destroy
            Some(_) => {
                // Check if the submission has completed
                device.poll(wgpu::Maintain::Poll);
                // Simple heuristic: consider safe after 1 s
                self.created_at.elapsed().as_secs() > 1
            }
        }
    }
}

/// Statistics for buffer pool monitoring
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub active_buffers: usize,
    pub retired_buffers: usize,
    pub active_memory_mb: f32,
    pub total_allocated_mb: f32,
    pub total_freed_mb: f32,
    pub peak_memory_mb: f32,
    pub allocation_count: u64,
}

/// Pool for managing GPU buffer lifetimes safely
#[derive(Debug)]
pub struct BufferPool {
    active_buffers: std::collections::HashMap<wgpu::Id<wgpu::Buffer>, TrackedBuffer>,
    retired_buffers: Vec<TrackedBuffer>,
    total_allocated_bytes: u64,
    total_freed_bytes: u64,
    peak_memory_bytes: u64,
    allocation_count: u64,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            active_buffers: std::collections::HashMap::new(),
            retired_buffers: Vec::new(),
            total_allocated_bytes: 0,
            total_freed_bytes: 0,
            peak_memory_bytes: 0,
            allocation_count: 0,
        }
    }

    /// Create a new buffer and track it
    pub fn create_buffer(&mut self, device: &wgpu::Device, desc: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        let buffer = device.create_buffer(desc);
        let size = desc.size;
        let label = desc.label.unwrap_or("Unnamed Buffer").to_string();
        
        let tracked = TrackedBuffer::new(buffer.global_id(), size, label);
        self.active_buffers.insert(buffer.global_id(), tracked);
        
        self.total_allocated_bytes += size;
        self.allocation_count += 1;
        
        let current_memory = self.get_active_memory_bytes();
        if current_memory > self.peak_memory_bytes {
            self.peak_memory_bytes = current_memory;
        }
        
        buffer
    }

    /// Mark a buffer as retired (no longer needed, but may still be in use by GPU)
    pub fn retire_buffer(&mut self, buffer: &wgpu::Buffer, last_submission: Option<wgpu::SubmissionIndex>) {
        if let Some(mut tracked) = self.active_buffers.remove(&buffer.global_id()) {
            tracked.last_submission = last_submission;
            self.retired_buffers.push(tracked);
        }
    }

    /// Clean up retired buffers that are safe to destroy
    pub fn cleanup_retired(&mut self, device: &wgpu::Device) -> usize {
        let initial_count = self.retired_buffers.len();
        
        self.retired_buffers.retain(|tracked| {
            if tracked.is_safe_to_destroy(device) {
                self.total_freed_bytes += tracked.size;
                false // Remove from retired list
            } else {
                true // Keep in retired list
            }
        });
        
        initial_count - self.retired_buffers.len()
    }

    /// Force cleanup of all retired buffers (use with caution)
    pub fn force_cleanup(&mut self) {
        for tracked in &self.retired_buffers {
            self.total_freed_bytes += tracked.size;
        }
        self.retired_buffers.clear();
    }

    /// Get current statistics
    pub fn get_stats(&self) -> BufferPoolStats {
        let active_memory_bytes = self.get_active_memory_bytes();
        let retired_memory_bytes: u64 = self.retired_buffers.iter().map(|t| t.size).sum();
        
        BufferPoolStats {
            active_buffers: self.active_buffers.len(),
            retired_buffers: self.retired_buffers.len(),
            active_memory_mb: (active_memory_bytes + retired_memory_bytes) as f32 / 1024.0 / 1024.0,
            total_allocated_mb: self.total_allocated_bytes as f32 / 1024.0 / 1024.0,
            total_freed_mb: self.total_freed_bytes as f32 / 1024.0 / 1024.0,
            peak_memory_mb: self.peak_memory_bytes as f32 / 1024.0 / 1024.0,
            allocation_count: self.allocation_count,
        }
    }

    fn get_active_memory_bytes(&self) -> u64 {
        self.active_buffers.values().map(|t| t.size).sum()
    }
}

// Include buffer pool tests
#[cfg(test)]
mod buffer_pool_test;

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
#[derive(Debug)]
pub struct RenderMetrics {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub particles_rendered: usize,
    pub gpu_memory_mb: f32,
    pub culled_particles: usize,
    pub shader_switches: usize,
}

/// Interactive debug panel for comprehensive physics debugging
#[derive(Debug)]
pub struct DebugPanel {
    pub visible: bool,
    // pub simulation_stats: Option<universe_sim::SimulationStats>,
    pub selected_particle_id: Option<usize>,
    pub show_physics_details: bool,
    pub show_performance_metrics: bool,
    pub show_cosmological_data: bool,
    pub show_chemistry_details: bool,
    pub auto_follow_particle: bool,
    pub debug_mode: DebugMode,
}

/// Different debug visualization modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DebugMode {
    Overview,          // General simulation stats
    Physics,           // Detailed physics interactions
    Performance,       // Render and simulation performance
    Particles,         // Individual particle inspection
    Chemistry,         // Chemical composition and reactions
    Cosmology,         // Universal evolution and structure
}

impl Default for DebugMode {
    fn default() -> Self {
        DebugMode::Overview
    }
}

impl Default for DebugPanel {
    fn default() -> Self {
        Self {
            visible: false,
            // simulation_stats: None,
            selected_particle_id: None,
            show_physics_details: false,
            show_performance_metrics: false,
            show_cosmological_data: false,
            show_chemistry_details: false,
            auto_follow_particle: false,
            debug_mode: DebugMode::default(),
        }
    }
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            left_pressed: false,
            right_pressed: false,
            middle_pressed: false,
            last_position: (0.0, 0.0),
            current_position: (0.0, 0.0),
            is_orbiting: false,
            is_panning: false,
            is_alt_zooming: false,
            is_flythrough: false,
            modifiers: ModifiersState::default(),
        }
    }
}

impl Default for RenderMetrics {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_ms: 0.0,
            particles_rendered: 0,
            gpu_memory_mb: 0.0,
            culled_particles: 0,
            shader_switches: 0,
        }
    }
}

/// Mouse state for Unity 6.0+ style navigation (official Unity controls)
#[derive(Debug)]
pub struct MouseState {
    pub left_pressed: bool,
    pub right_pressed: bool,
    pub middle_pressed: bool,
    pub last_position: (f64, f64),
    pub current_position: (f64, f64),
    pub is_orbiting: bool,           // Alt + Left mouse = orbit (Unity official)
    pub is_panning: bool,            // Middle mouse = pan (Unity official)
    pub is_alt_zooming: bool,        // Alt + Right mouse = zoom (Unity official)
    pub is_flythrough: bool,         // Right mouse hold = flythrough mode (Unity official)
    pub modifiers: winit::keyboard::ModifiersState,
}

impl MouseState {
    pub fn update_position(&mut self, x: f64, y: f64) {
        self.last_position = self.current_position;
        self.current_position = (x, y);
    }
    
    pub fn get_delta(&self) -> (f64, f64) {
        (
            self.current_position.0 - self.last_position.0,
            self.current_position.1 - self.last_position.1,
        )
    }
}

/// High-performance renderer state with heavy mode enhancements
pub struct NativeRenderer<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    view_matrix: Matrix4<f32>,
    #[allow(dead_code)]
    proj_matrix: Matrix4<f32>,
    
    // Performance metrics and controls
    metrics: RenderMetrics,
    frame_count: u64,
    last_fps_time: std::time::Instant,
    
    // Particle data management
    particle_count: usize,
    #[allow(dead_code)]
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
    
    // Debug panel and interaction
    pub debug_panel: DebugPanel,
    pub mouse_state: MouseState,
    
    // Enhanced camera controls
    pub orbit_sensitivity: f32,
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,

    // Keep text buffers alive until GPU is done with them to avoid destroyed-buffer validation errors
    text_buffers: Vec<GlyphonBuffer>,
    // NEW: Track retired text buffers and their submission indices for safe destruction
    retired_text_buffers: Vec<(GlyphonBuffer, wgpu::SubmissionIndex)>,
    // NEW: Staging belt for safe GPU uploads
    #[allow(dead_code)]
    staging_belt: StagingBelt,
    // NEW: Buffer pool for safe GPU memory management
    buffer_pool: BufferPool,
    // Track submission indices for proper cleanup
    last_submission_index: Option<wgpu::SubmissionIndex>,
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
        
        self.view_matrix = Matrix4::look_at_rh(eye, target, up);
    }
    
    /// Update projection matrix
    pub fn update_projection(&mut self) {
        self.proj_matrix = perspective(Rad(self.fov), self.aspect, self.near, self.far);
    }
    
    /// Get combined view-projection matrix for GPU (4x4 array format)
    pub fn get_view_proj_matrix(&self) -> [[f32; 4]; 4] {
        let view_proj_mat = self.proj_matrix * self.view_matrix;
        // Safe transmute into 4x4 array
        let vp: [[f32; 4]; 4] = view_proj_mat.into();
        vp
    }
    
    /// Orbital rotation around target (Unity-style: right mouse button)
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        let radius = (self.position - self.target).magnitude();
        
        // Convert position to spherical coordinates relative to target
        let offset = self.position - self.target;
        let mut theta = offset.z.atan2(offset.x); // Azimuth angle
        let mut phi = (offset.y / radius).acos(); // Polar angle
        
        // Apply rotation with Unity-style sensitivity
        theta -= delta_x * sensitivity * 0.005; // Horizontal rotation (inverted for Unity feel)
        phi = (phi + delta_y * sensitivity * 0.005).clamp(0.01, std::f32::consts::PI - 0.01);
        
        // Convert back to Cartesian coordinates
        let new_offset = Vector3::new(
            radius * phi.sin() * theta.cos(),
            radius * phi.cos(),
            radius * phi.sin() * theta.sin(),
        );
        
        self.position = self.target + new_offset;
        self.update_view_matrix();
    }

    /// Pan camera (Unity-style: left mouse button)
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let camera_up = right.cross(forward).normalize();
        
        // Calculate pan distance based on distance to target
        let distance = (self.position - self.target).magnitude();
        let pan_scale = distance * sensitivity * 0.001;
        
        // Unity-style panning: move both camera and target together
        let pan_offset = right * (-delta_x * pan_scale) + camera_up * (delta_y * pan_scale);
        
        self.position += pan_offset;
        self.target += pan_offset;
        self.update_view_matrix();
    }
    
    /// Zoom camera in/out (Unity scroll wheel)
    pub fn zoom(&mut self, delta: f32) {
        let direction = (self.position - self.target).normalize();
        let distance = (self.position - self.target).magnitude();
        
        // Scale zoom based on distance (closer = slower zoom)
        let zoom_amount = delta * (distance * 0.1).max(0.1);
        
        let new_position = self.position - direction * zoom_amount;
        let new_distance = (new_position - self.target).magnitude();
        
        // Prevent zooming too close or too far
        if new_distance > 0.1 && new_distance < 1000.0 {
            self.position = new_position;
            self.update_view_matrix();
        }
    }
    
    /// Update view matrix from current camera state
    pub fn update_view_matrix(&mut self) {
        self.view_matrix = Matrix4::look_at_rh(self.position, self.target, self.up);
    }
    
    /// Focus camera on a specific point
    pub fn focus_on(&mut self, point: Point3<f32>, distance: Option<f32>) {
        let target_distance = distance.unwrap_or(10.0);
        let direction = (self.position - self.target).normalize();
        
        self.target = point;
        self.position = point + direction * target_distance;
    }
    
    /// Reset camera to default view
    pub fn reset_view(&mut self) {
        *self = Camera::default();
        self.update_view_matrix();
        self.update_projection();
        println!("🔄 Camera reset to default view");
    }
    
    /// Handle keyboard input for camera controls
    pub fn handle_input(&mut self, key: KeyCode, state: ElementState) {
        let movement_speed = 0.1;
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        
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
    
    /// Flythrough mouse look (Unity 6.0+ official: right mouse hold)
    pub fn flythrough_look(&mut self, delta_x: f32, delta_y: f32) {
        // Compute yaw (around global up) and pitch (around camera right) rotations using cgmath quaternions.
        let sensitivity = 0.002;
        let yaw = Rad(-delta_x * sensitivity);
        let pitch = Rad(-delta_y * sensitivity);

        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();

        let yaw_rot = cgmath::Quaternion::from_axis_angle(self.up.normalize(), yaw);
        let pitch_rot = cgmath::Quaternion::from_axis_angle(right, pitch);
        let combined = yaw_rot * pitch_rot;

        let new_forward = combined.rotate_vector(forward);
        self.target = self.position + new_forward * (self.target - self.position).magnitude();
        self.update_view_matrix();
    }
    
    /// Flythrough movement (Unity 6.0+ official: WASD while right mouse held)
    pub fn flythrough_move(&mut self, direction: Vector3<f32>, speed: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let camera_up = right.cross(forward).normalize();
        
        // Apply movement in camera-relative directions
        let movement = right * direction.x + camera_up * direction.y + forward * direction.z;
        let scaled_movement = movement * speed;
        
        self.position += scaled_movement;
        self.target += scaled_movement;
        self.update_view_matrix();
    }

    /// Focus camera on origin (Unity F key equivalent) 
    pub fn focus_on_origin(&mut self) {
        self.focus_on(Point3::new(0.0, 0.0, 0.0), Some(50.0));
        println!("🎯 Camera focused on origin");
    }

    /// Get zoom level for LOD system - returns scale factor for detail level
    pub fn get_zoom_level(&self) -> f32 {
        let distance = (self.position - self.target).magnitude();
        
        // Scale levels:
        // 0.01-0.1: Particle level (atoms, molecules)
        // 0.1-10: Cell/organism level  
        // 10-1000: Object level (rocks, trees)
        // 1000-100k: Planetary level (continents, weather)
        // 100k+: Stellar/galactic level (stars, planets as points)
        
        if distance < 0.1 {
            0.0 // Microscope mode - individual particles
        } else if distance < 10.0 {
            1.0 // Biological scale
        } else if distance < 1000.0 {
            2.0 // Object scale
        } else if distance < 100000.0 {
            3.0 // Planetary scale
        } else {
            4.0 // Stellar scale
        }
    }

    /// Get particle size multiplier based on zoom level for LOD
    pub fn get_particle_size_multiplier(&self) -> f32 {
        let distance = (self.position - self.target).magnitude();
        
        // Size scaling for visibility at different zoom levels
        if distance < 0.01 {
            1000.0  // Very close - make particles huge
        } else if distance < 0.1 {
            100.0   // Microscope level
        } else if distance < 1.0 {
            10.0    // Close inspection
        } else if distance < 10.0 {
            5.0     // Normal view
        } else if distance < 100.0 {
            2.0     // Medium zoom out
        } else {
            1.0     // Far zoom - default size
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
        
        // Create instance buffer for particle data (simple layout: position, color, size)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (max_particles * std::mem::size_of::<SimpleParticleVertex>()) as u64,
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
                        array_stride: std::mem::size_of::<SimpleParticleVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // Position (vec3)
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // Color (vec3)
                            wgpu::VertexAttribute {
                                offset: std::mem::size_of::<[f32; 3]>() as u64,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // Size (f32)
                            wgpu::VertexAttribute {
                                offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                                shader_location: 3,
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
            #[allow(dead_code)]
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
            #[allow(dead_code)]
            view_matrix: Matrix4::identity(),
            #[allow(dead_code)]
            proj_matrix: Matrix4::identity(),
            metrics: RenderMetrics::default(),
            frame_count: 0,
            last_fps_time: std::time::Instant::now(),
            particle_count: 0,
            #[allow(dead_code)]
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
            
            // Debug panel and interaction
            debug_panel: DebugPanel::default(),
            mouse_state: MouseState::default(),
            
            // Enhanced camera controls
            orbit_sensitivity: 0.01,
            pan_sensitivity: 0.01,
            zoom_sensitivity: 0.01,

            // Persistent text buffers (cleared each frame after rendering)
            text_buffers: Vec::new(),
            // Initialize retired text buffers and submission indices
            retired_text_buffers: Vec::new(),
            // Initialize staging belt with 1 MiB default chunk size
            #[allow(dead_code)]
            staging_belt: StagingBelt::new(1024 * 1024),
            // Initialize buffer pool for safe memory management
            buffer_pool: BufferPool::new(),
            // Initialize submission tracking
            last_submission_index: None,
        })
    }
    
    /// Update particle data from simulation with zero-copy access
    // TEMPORARILY COMMENTED OUT FOR TESTING
    /*
    pub fn update_particles(&mut self, simulation: &mut UniverseSimulation) -> Result<()> {
        println!("🔬 PHYSICS DATA: Syncing real simulation particles");

        // Update debug panel with latest stats
        if let Ok(stats) = simulation.get_stats() {
            // self.debug_panel.simulation_stats = Some(stats);
        }

        let mut particles: Vec<SimpleParticleVertex> = Vec::new();
        
        // Get particles from physics engine
        let physics_particles = simulation.physics_engine.get_particles();
        let store_particle_count = simulation.store.particles.count;
        
        println!("📊 Physics Engine: {} particles, Store: {} particles", 
                 physics_particles.len(), store_particle_count);

        // Convert physics engine particles to renderer format
        for (i, physics_particle) in physics_particles.iter().enumerate() {
            if i >= self.max_particles {
                warn!("Particle limit exceeded: {} > {}", physics_particles.len(), self.max_particles);
                break;
            }

            // Map particle type to numeric value for shader
            let particle_type_value = match physics_particle.particle_type {
                physics_engine::ParticleType::Electron => 0.0,
                physics_engine::ParticleType::Proton => 1.0,
                physics_engine::ParticleType::Neutron => 2.0,
                physics_engine::ParticleType::Photon => 3.0,
                physics_engine::ParticleType::Hydrogen => 4.0,
                physics_engine::ParticleType::Helium => 5.0,
                _ => 6.0, // Other particles
            };

            // Calculate kinetic energy for temperature mapping
            let velocity_magnitude = physics_particle.velocity.magnitude();
            let kinetic_energy = 0.5 * physics_particle.mass * velocity_magnitude * velocity_magnitude;
            
            // Convert to temperature using kinetic theory (E = 3/2 kT for monatomic)
            let boltzmann_constant = 1.380649e-23; // J/K
            let _temperature = if physics_particle.mass > 0.0 {
                (2.0 * kinetic_energy) / (3.0 * boltzmann_constant)
            } else {
                300.0 // Default for massless particles
            };

            // Derive a simple color from particle type value (basic rainbow mapping)
            let hue = particle_type_value / 6.0;
            let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);

            // Apply LOD-based particle scaling for better visibility at different zoom levels
            let base_size = 0.003;
            let size_multiplier = self.camera.get_particle_size_multiplier();
            let final_size = base_size * size_multiplier;

            // Scale physics particles to be visible (1e-12 makes them invisible)
            let scale_factor = 1e-9; // Still small but visible in microscope mode
            let render_particle = SimpleParticleVertex {
                position: [
                    physics_particle.position.x as f32 * scale_factor,
                    physics_particle.position.y as f32 * scale_factor,
                    physics_particle.position.z as f32 * scale_factor,
                ],
                color: [r, g, b],
                size: final_size,
            };

            particles.push(render_particle);
        }

        // Also include particles from the universe store if they exist
        for i in 0..store_particle_count.min(self.max_particles - particles.len()) {
            if particles.len() >= self.max_particles {
                break;
            }

            let (r, g, b) = (0.8, 0.8, 0.8);
            let base_size = 0.003;
            let size_multiplier = self.camera.get_particle_size_multiplier();
            let final_size = base_size * size_multiplier;
            
            // Scale store particles to be visible (1e-12 makes them invisible)
            let scale_factor = 1e-9; // Still small but visible in microscope mode
            let store_particle = SimpleParticleVertex {
                position: [
                    simulation.store.particles.position[i].x as f32 * scale_factor,
                    simulation.store.particles.position[i].y as f32 * scale_factor,
                    simulation.store.particles.position[i].z as f32 * scale_factor,
                ],
                color: [r, g, b],
                size: final_size,
            };

            particles.push(store_particle);
        }

        // If no physics particles, create a few demo particles for testing (dev room)
        if particles.is_empty() {
            println!("🧪 DEV ROOM: No physics particles found, creating visible demo particles");
            
            let base_size = 0.05; // Much larger for visibility
            let size_multiplier = self.camera.get_particle_size_multiplier();
            let final_size = base_size * size_multiplier;
            
            // Create a visible grid of demo particles
            for i in 0..5 {
                for j in 0..5 {
                    let x = (i as f32 - 2.0) * 0.2; // 0.2 units apart
                    let y = (j as f32 - 2.0) * 0.2;
                    let z = 0.0;
                    
                    // Rainbow colors for easy identification
                    let hue = (i + j) as f32 / 10.0;
                    let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
                    
                    let demo_particle = SimpleParticleVertex {
                        position: [x, y, z],
                        color: [r, g, b],
                        size: final_size,
                    };
                    particles.push(demo_particle);
                }
            }
            println!("🎨 Created {} colorful demo particles in grid formation", particles.len());
        }

        // Upload to GPU using StagingBelt with Buffer Pool tracking for safe lifetime management
        if !particles.is_empty() {
            let data_bytes = bytemuck::cast_slice(&particles);
            let size = wgpu::BufferSize::new(data_bytes.len() as u64).unwrap();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Particle Upload Encoder"),
            });
            {
                let mut view = self.staging_belt.write_buffer(&mut encoder, &self.vertex_buffer, 0, size, &self.device);
                view.copy_from_slice(data_bytes);
            }
            self.staging_belt.finish();
            
            // Submit and track the submission index for safe buffer cleanup
            let submission_index = self.queue.submit(Some(encoder.finish()));
            self.last_submission_index = Some(submission_index);
            
            // Recall staging belt - it handles its own buffer lifetimes
            self.staging_belt.recall();
            
            // Clean up any retired buffers that are now safe to destroy
            let cleaned = self.buffer_pool.cleanup_retired(&self.device);
            if cleaned > 0 {
                debug!("BufferPool: Cleaned {} retired buffers during particle upload", cleaned);
            }
            
            self.particle_count = particles.len();
            println!("✅ Uploaded {} real physics particles", self.particle_count);
        } else {
            println!("❌ No particles to render");
        }

        Ok(())
    }
    */

    /// Render frame with maximum performance and heavy mode enhancements
    pub fn render(&mut self, simulation_time: f32) -> Result<()> {
        let frame_start = std::time::Instant::now();
        
        println!("🔥 RENDER START - Frame {}", self.frame_count);
        
        // Wait for GPU to finish with previous text buffers before clearing them
        if !self.text_buffers.is_empty() {
            let submission_index = self.last_submission_index.clone().unwrap_or_else(|| self.queue.submit(std::iter::empty()));
            for buf in self.text_buffers.drain(..) {
                self.retired_text_buffers.push((buf, submission_index.clone()));
            }
        }
        self.cleanup_retired_text_buffers();
        
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

        // ---- helper: project world-space to screen-space ----
        let vp_mat = na::Matrix4::<f32>::from_row_slice(&[
            view_proj[0][0], view_proj[0][1], view_proj[0][2], view_proj[0][3],
            view_proj[1][0], view_proj[1][1], view_proj[1][2], view_proj[1][3],
            view_proj[2][0], view_proj[2][1], view_proj[2][2], view_proj[2][3],
            view_proj[3][0], view_proj[3][1], view_proj[3][2], view_proj[3][3],
        ]);

        let world_to_screen = |pos: [f32; 3], size: (u32, u32)| -> Option<(f32, f32)> {
            let wp = na::Vector4::new(pos[0], pos[1], pos[2], 1.0);
            let clip = vp_mat * wp;
            if clip.w.abs() < 1e-6 {
                return None;
            }
            let ndc = clip / clip.w;
            if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 {
                return None;
            }
            Some((
                (ndc.x + 1.0) * 0.5 * size.0 as f32,
                (1.0 - (ndc.y + 1.0) * 0.5) * size.1 as f32,
            ))
        };

        // First pass: create and store all text buffers without creating TextAreas
        // Clear old text buffers to prevent memory buildup
        self.text_buffers.clear();

        // Collect buffer creation info first
        let mut buffer_info = Vec::new();

        // Row labels buffer info
        for (row_idx, _label) in ROW_LABELS.iter().enumerate() {
            let y_world = row_idx as f32 * SPACING_Y - (COLOR_SETS as f32 - 1.0) * 0.5 * SPACING_Y;
            let x_world = -(SAMPLES_PER_ROW as f32 * SPACING_X * 0.5) - 1.5;
            if let Some((sx, sy)) = world_to_screen([x_world, y_world, 0.0], (self.size.width, self.size.height)) {
                buffer_info.push((ROW_LABELS[row_idx], sx, sy, GlyphonColor::rgb(255, 255, 255), 16.0, 180.0, 24.0));
            }
        }

        // Column labels buffer info
        let top_row_y_world = 0.0 - (COLOR_SETS as f32 - 1.0) * 0.5 * SPACING_Y;
        let x_start = -(SAMPLES_PER_ROW as f32 * SPACING_X * 0.5);
        for (col_idx, label) in COL_LABELS.iter().enumerate() {
            let x_world = col_idx as f32 * SPACING_X + x_start;
            let y_world = top_row_y_world + SPACING_Y + 0.5;
            if let Some((sx, sy)) = world_to_screen([x_world, y_world, 0.0], (self.size.width, self.size.height)) {
                buffer_info.push((label, sx, sy, GlyphonColor::rgb(255, 255, 0), 14.0, 80.0, 22.0));
            }
        }

        // Create all text buffers
        for (text, _sx, _sy, _color, font_size, width, height) in &buffer_info {
            let mut buffer = GlyphonBuffer::new(&mut self.font_system, Metrics::new(*font_size, font_size + 4.0));
            buffer.set_size(&mut self.font_system, *width, *height);
            buffer.set_text(&mut self.font_system, text, Attrs::new().family(Family::SansSerif), Shaping::Advanced);
            self.text_buffers.push(buffer);
        }

        // Now create text areas with references to the buffers
        let mut text_areas: Vec<TextArea> = Vec::new();
        for (buffer_idx, (_text, sx, sy, color, _font_size, _width, _height)) in buffer_info.iter().enumerate() {
            if buffer_idx < self.text_buffers.len() {
                let bounds = TextBounds { left: 0, top: 0, right: self.size.width as i32, bottom: self.size.height as i32 };
                text_areas.push(TextArea { 
                    buffer: &self.text_buffers[buffer_idx], 
                    left: *sx, 
                    top: *sy, 
                    scale: 1.0, 
                    bounds, 
                    default_color: *color 
                });
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
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }), // BLACK SPACE-LIKE BACKGROUND for better particle visibility
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

        // ---- Debug Panel Overlay ----
        if let Err(e) = self.render_debug_panel(&mut encoder, &view) {
            warn!("Debug panel rendering failed: {}", e);
        }

        // Submit GPU commands and present the frame
        let submission_index = self.queue.submit(std::iter::once(encoder.finish()));
        self.last_submission_index = Some(submission_index);
        output.present();

        // Clean up buffer pool periodically to prevent memory leaks
        if self.frame_count % 60 == 0 { // Clean up every 60 frames (~1 second at 60fps)
            let cleaned = self.buffer_pool.cleanup_retired(&self.device);
            if cleaned > 0 {
                debug!("BufferPool: Periodic cleanup removed {} retired buffers", cleaned);
            }
        }

        // === NEW: GPU lifecycle safeguard for debug resources ===
        // When the debug panel is active we allocate and drop many text buffers
        // every frame.  If any of those GPU buffers are freed while the GPU is
        // still processing the previous frame wgpu will raise a validation
        // error ("Buffer is destroyed").  To guarantee safety we synchronise
        // with the GPU once per frame while the debug UI is visible.  In normal
        // rendering mode we keep the previous non-blocking behaviour for
        // performance.
        if self.debug_panel.visible {
            // Non-blocking poll every frame keeps the driver alive without stalling
            // the CPU.  We now rely on an explicit poll-and-drop inside
            // `render_debug_text` (added above) to guarantee safe destruction of
            // old text buffers, so we can always use the lightweight mode here.
            self.device.poll(wgpu::Maintain::Poll);
        } else {
            // Lightweight polling – just allow the driver to make progress.
            self.device.poll(wgpu::Maintain::Poll);
        }
        // === END safeguard ===

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
    
    /// Get buffer pool statistics for debugging
    pub fn get_buffer_pool_stats(&self) -> BufferPoolStats {
        self.buffer_pool.get_stats()
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
    
    /// Toggle debug panel visibility
    pub fn toggle_debug_panel(&mut self) {
        self.debug_panel.visible = !self.debug_panel.visible;
        println!("🎛️ Debug panel: {}", if self.debug_panel.visible { "ON" } else { "OFF" });
        
        if !self.debug_panel.visible {
            self.device.poll(wgpu::Maintain::Wait);
            let submission_index = self.last_submission_index.clone().unwrap_or_else(|| self.queue.submit(std::iter::empty()));
            for buf in self.text_buffers.drain(..) {
                self.retired_text_buffers.push((buf, submission_index.clone()));
            }
            self.cleanup_retired_text_buffers();
            println!("🧹 Safely cleared debug panel text buffers after GPU sync");
        }
    }
    
    /// Cycle through debug panel modes
    pub fn cycle_debug_mode(&mut self) {
        self.debug_panel.debug_mode = match self.debug_panel.debug_mode {
            DebugMode::Overview => DebugMode::Physics,
            DebugMode::Physics => DebugMode::Performance,
            DebugMode::Performance => DebugMode::Particles,
            DebugMode::Particles => DebugMode::Chemistry,
            DebugMode::Chemistry => DebugMode::Cosmology,
            DebugMode::Cosmology => DebugMode::Overview,
        };
        println!("🔄 Debug mode: {:?}", self.debug_panel.debug_mode);
    }
    
    /// Handle mouse button press/release (Unity 6.0+ official controls)
    pub fn handle_mouse_button(&mut self, button: winit::event::MouseButton, state: ElementState) {
        match button {
            winit::event::MouseButton::Left => {
                self.mouse_state.left_pressed = state == ElementState::Pressed;
                if state == ElementState::Pressed && self.mouse_state.modifiers.alt_key() {
                    self.mouse_state.is_orbiting = true;  // Unity: Alt + Left = Orbit
                    println!("🖱️ Unity Navigation: ALT + LEFT - ORBIT MODE activated");
                } else {
                    self.mouse_state.is_orbiting = false;
                }
            },
            winit::event::MouseButton::Right => {
                self.mouse_state.right_pressed = state == ElementState::Pressed;
                if state == ElementState::Pressed {
                    if self.mouse_state.modifiers.alt_key() {
                        self.mouse_state.is_alt_zooming = true;  // Unity: Alt + Right = Zoom
                        println!("🖱️ Unity Navigation: ALT + RIGHT - ZOOM MODE activated");
                    } else {
                        self.mouse_state.is_flythrough = true;  // Unity: Right = Flythrough
                        println!("🖱️ Unity Navigation: RIGHT HOLD - FLYTHROUGH MODE activated (use WASD)");
                    }
                } else {
                    self.mouse_state.is_alt_zooming = false;
                    self.mouse_state.is_flythrough = false;
                }
            },
            winit::event::MouseButton::Middle => {
                self.mouse_state.middle_pressed = state == ElementState::Pressed;
                if state == ElementState::Pressed {
                    self.mouse_state.is_panning = true;  // Unity: Middle = Pan
                    println!("🖱️ Unity Navigation: MIDDLE - PAN MODE activated");
                } else {
                    self.mouse_state.is_panning = false;
                }
            },
            _ => {}
        }
    }
    
    /// Handle mouse movement with Unity 6.0+ official navigation
    pub fn handle_mouse_move(&mut self, x: f64, y: f64) {
        self.mouse_state.update_position(x, y);
        let (delta_x, delta_y) = self.mouse_state.get_delta();
        
        // Apply Unity 6.0+ official navigation based on modifier keys and mouse buttons
        if self.mouse_state.is_orbiting && self.mouse_state.modifiers.alt_key() {
            // Alt + Left mouse: Orbit around target (Unity official)
            self.camera.orbit(
                delta_x as f32, 
                delta_y as f32, 
                self.orbit_sensitivity
            );
            if delta_x.abs() > 1.0 || delta_y.abs() > 1.0 {
                println!("🌀 Unity ORBIT (Alt+Left): delta({:.1}, {:.1})", delta_x, delta_y);
            }
        } else if self.mouse_state.is_panning {
            // Middle mouse: Pan camera (Unity official)
            self.camera.pan(
                delta_x as f32, 
                delta_y as f32, 
                self.pan_sensitivity
            );
            if delta_x.abs() > 1.0 || delta_y.abs() > 1.0 {
                println!("📐 Unity PAN (Middle): delta({:.1}, {:.1})", delta_x, delta_y);
            }
        } else if self.mouse_state.is_alt_zooming && self.mouse_state.modifiers.alt_key() {
            // Alt + Right mouse: Zoom camera (Unity official alternative to scroll wheel)
            let zoom_delta = delta_y as f32 * 0.01;
            self.camera.zoom(zoom_delta);
            if delta_y.abs() > 1.0 {
                println!("🔍 Unity ZOOM (Alt+Right): delta_y({:.1})", delta_y);
            }
        } else if self.mouse_state.is_flythrough {
            // Right mouse: Flythrough mode mouse look (Unity official)
            self.camera.flythrough_look(
                delta_x as f32 * 0.002, // Mouse sensitivity for look
                delta_y as f32 * 0.002
            );
            if delta_x.abs() > 1.0 || delta_y.abs() > 1.0 {
                println!("✈️ Unity FLYTHROUGH LOOK: delta({:.1}, {:.1})", delta_x, delta_y);
            }
        }
    }
    
    /// Handle mouse wheel for zooming
    pub fn handle_mouse_wheel(&mut self, delta: f32) {
        self.camera.zoom(delta);
        println!("🔍 Unity ZOOM (scroll): delta({:.1})", delta);
    }
    
    /// Handle keyboard input for debug panel and camera controls
    pub fn handle_debug_input(&mut self, key: KeyCode, state: ElementState) {
        if state == ElementState::Pressed {
            match key {
                KeyCode::F1 => self.toggle_debug_panel(),
                KeyCode::F2 => {
                    self.debug_panel.debug_mode = match self.debug_panel.debug_mode {
                        DebugMode::Overview => DebugMode::Physics,
                        DebugMode::Physics => DebugMode::Performance,
                        DebugMode::Performance => DebugMode::Particles,
                        DebugMode::Particles => DebugMode::Chemistry,
                        DebugMode::Chemistry => DebugMode::Cosmology,
                        DebugMode::Cosmology => DebugMode::Overview,
                    };
                    println!("🔄 Debug mode: {:?}", self.debug_panel.debug_mode);
                },
                KeyCode::F3 => {
                    self.debug_panel.show_physics_details = !self.debug_panel.show_physics_details;
                },
                KeyCode::F4 => {
                    self.debug_panel.show_performance_metrics = !self.debug_panel.show_performance_metrics;
                },
                KeyCode::F5 => {
                    self.camera_microscope_reset();
                    println!("🔬 Microscope view reset - focusing on particle");
                },
                KeyCode::Space => {
                    // Focus on center of particle system
                    self.camera.focus_on(Point3::new(0.0, 0.0, 0.0), Some(20.0));
                    println!("🎯 Camera focused on origin");
                },
                _ => {}
            }
        }
    }
    
    /// Handle modifier keys
    pub fn handle_modifiers(&mut self, modifiers: winit::keyboard::ModifiersState) {
        self.mouse_state.modifiers = modifiers;
        println!("🔧 Modifiers changed: {:?}", modifiers);
    }

    /// Render comprehensive debug panel overlay
    pub fn render_debug_panel(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> Result<()> {
        if !self.debug_panel.visible {
            return Ok(());
        }

        // Temporarily disable simulation stats since we commented out the field
        // let Some(ref stats) = self.debug_panel.simulation_stats else {
            // If no stats available, show a placeholder message
            let stats = self.get_buffer_pool_stats();
            let formatted_text = format!(
                "🔬 EVOLUTION Universe Simulation - Debug Panel [F1=Toggle]\n\
                📊 Buffer Pool Testing Mode...\n\
                \n\
                🎮 Controls:\n\
                • F1: Toggle debug panel\n\
                • F2: Cycle debug modes\n\
                • F5: Reset camera view\n\
                \n\
                🔧 Buffer Pool Stats:\n\
                • Active buffers: {}\n\
                • Retired buffers: {}\n\
                • Active memory: {:.2} MB\n\
                • Total allocated: {:.2} MB\n\
                • Peak memory: {:.2} MB",
                stats.active_buffers,
                stats.retired_buffers,
                stats.active_memory_mb,
                stats.total_allocated_mb,
                stats.peak_memory_mb
            );
            return self.render_debug_text(encoder, view, &formatted_text);
        // };

        // Create debug text content based on current mode
        // TEMPORARILY COMMENTED OUT - stats not available
        /*
        let debug_text = match self.debug_panel.debug_mode {
            DebugMode::Overview => {
                format!(
                    "🔬 EVOLUTION Universe Simulation - Debug Panel [F1=Toggle F2=Mode F5=Reset]\n\
                     📊 Overview Mode (F2 to cycle modes)\n\
                     \n\
                     🌌 Universe: Age {:.2} Gyr, Tick {}, Description: {}\n\
                     ⚛️  Particles: {} total, {} rendered\n\
                     ⭐ Stars: {} total, {} main sequence, {} evolved, {} remnants\n\
                     🪐 Planets: {} total, {} habitable, {} Earth-like\n\
                     👽 Lineages: {} active, {} extinct, {} immortal\n\
                     \n\
                     🎮 Controls:\n\
                     • Left Mouse: Orbit camera around target\n\
                     • Middle Mouse: Pan camera\n\
                     • Scroll Wheel: Zoom in/out (LOD scaling)\n\
                     • Space: Focus on origin\n\
                     • F5: Microscope view (particle close-up)\n\
                     • F3: Toggle physics details\n\
                     • F4: Toggle performance metrics\n\
                     \n\
                     📈 Performance: {:.1} ms/step, {:.1} fps",
                    stats.universe_age_gyr,
                    stats.current_tick,
                    stats.universe_description,
                    stats.particle_count,
                    self.particle_count,
                    stats.star_count,
                    stats.main_sequence_stars,
                    stats.evolved_stars,
                    stats.stellar_remnants,
                    stats.planet_count,
                    stats.habitable_planets,
                    stats.earth_like_planets,
                    stats.lineage_count,
                    stats.extinct_lineages,
                    stats.immortal_lineages,
                    stats.physics_step_time_ms,
                    self.metrics.fps
                )
            },
            DebugMode::Physics => {
                format!(
                    "⚛️  Physics Engine Debug [F2 to change mode]\n\
                     \n\
                     🔬 Particle Interactions:\n\
                     • Total interactions/step: {}\n\
                     • Particle interactions/step: {}\n\
                     • Physics step time: {:.3} ms\n\
                     \n\
                     🌡️  Thermodynamics:\n\
                     • Average temperature: {:.1} K\n\
                     • Total energy: {:.2e} J\n\
                     • Kinetic energy: {:.2e} J\n\
                     • Potential energy: {:.2e} J\n\
                     • Radiation energy: {:.2e} J\n\
                     • Nuclear binding energy: {:.2e} J\n\
                     • Energy density: {:.2e} J/m³\n\
                     \n\
                     🧪 Chemical Composition:\n\
                     • Hydrogen: {:.1}%\n\
                     • Helium: {:.1}%\n\
                     • Carbon: {:.1}%\n\
                     • Oxygen: {:.1}%\n\
                     • Iron: {:.1}%\n\
                     • Heavy elements: {:.1}%\n\
                     • Metallicity: {:.3}",
                    stats.interactions_per_step,
                    stats.particle_interactions_per_step,
                    stats.physics_step_time_ms,
                    stats.average_temperature,
                    stats.total_energy,
                    stats.kinetic_energy,
                    stats.potential_energy,
                    stats.radiation_energy,
                    stats.nuclear_binding_energy,
                    stats.energy_density,
                    stats.hydrogen_fraction * 100.0,
                    stats.helium_fraction * 100.0,
                    stats.carbon_fraction * 100.0,
                    stats.oxygen_fraction * 100.0,
                    stats.iron_fraction * 100.0,
                    stats.heavy_elements_fraction * 100.0,
                    stats.metallicity
                )
            },
            DebugMode::Performance => {
                format!(
                    "⚡ Performance Metrics [F2 to change mode]\n\
                     \n\
                     🖥️  Rendering:\n\
                     • FPS: {:.1}\n\
                     • Frame time: {:.2} ms\n\
                     • Particles rendered: {}\n\
                     • GPU memory: {:.1} MB\n\
                     • Culled particles: {}\n\
                     • Shader switches: {}\n\
                     \n\
                     🔬 Physics Simulation:\n\
                     • Physics step time: {:.3} ms\n\
                     • Interactions per step: {}\n\
                     • Target UPS: {:.1}\n\
                     \n\
                     📊 System:\n\
                     • Current tick: {}\n\
                     • Tick span: {:.1} years\n\
                     • Camera position: ({:.1}, {:.1}, {:.1})\n\
                     • Camera target: ({:.1}, {:.1}, {:.1})\n\
                     • Zoom level: {:.1} (distance: {:.4})\n\
                     • Current color mode: {:?}",
                    self.metrics.fps,
                    self.metrics.frame_time_ms,
                    self.metrics.particles_rendered,
                    self.metrics.gpu_memory_mb,
                    self.metrics.culled_particles,
                    self.metrics.shader_switches,
                    stats.physics_step_time_ms,
                    stats.interactions_per_step,
                    stats.target_ups,
                    stats.current_tick,
                    1e6, // Default tick span years
                    self.camera.position.x,
                    self.camera.position.y,
                    self.camera.position.z,
                    self.camera.target.x,
                    self.camera.target.y,
                    self.camera.target.z,
                    self.camera.get_zoom_level(),
                    (self.camera.position - self.camera.target).magnitude(),
                    self.camera.color_mode
                )
            },
            DebugMode::Particles => {
                format!(
                    "🎯 Particle Inspector [F2 to change mode]\n\
                     \n\
                     📊 Particle Count Summary:\n\
                     • Total particles: {}\n\
                     • Rendered particles: {}\n\
                     • Max particles: {}\n\
                     \n\
                     🔍 Visualization Settings:\n\
                     • Color mode: {:?}\n\
                     • Scale mode: {:?}\n\
                     • Filter threshold: {:.3}\n\
                     \n\
                     🎨 Color Modes Available:\n\
                     • ParticleType: Color by particle type\n\
                     • Charge: Color by electric charge\n\
                     • Temperature: Thermal radiation colors\n\
                     • Velocity: Doppler shift visualization\n\
                     • Interactions: Activity-based coloring\n\
                     • Scientific: Multi-channel visualization\n\
                     \n\
                     ⚡ Particle Selection:\n\
                     {}",
                    stats.particle_count,
                    self.particle_count,
                    self.max_particles,
                    self.camera.color_mode,
                    self.camera.scale_mode,
                    self.camera.filter_threshold,
                    if let Some(id) = self.debug_panel.selected_particle_id {
                        format!("• Selected particle ID: {}", id)
                    } else {
                        "• No particle selected (click to select)".to_string()
                    }
                )
            },
            DebugMode::Chemistry => {
                format!(
                    "🧪 Chemistry & Nucleosynthesis [F2 to change mode]\n\
                     \n\
                     🌟 Stellar Evolution:\n\
                     • Star formation rate: {:.2e} stars/year\n\
                     • Average stellar mass: {:.1} M☉\n\
                     • Main sequence stars: {}\n\
                     • Evolved stars: {}\n\
                     • Stellar remnants: {}\n\
                     \n\
                     🧬 Chemical Evolution:\n\
                     • Hydrogen fraction: {:.3}\n\
                     • Helium fraction: {:.3}\n\
                     • Carbon fraction: {:.5}\n\
                     • Oxygen fraction: {:.5}\n\
                     • Iron fraction: {:.5}\n\
                     • Heavy elements: {:.5}\n\
                     • Metallicity [Fe/H]: {:.3}\n\
                     \n\
                     🌌 Planet Formation:\n\
                     • Total planets: {}\n\
                     • Planet formation rate: {:.2e} planets/year\n\
                     • Average planet mass: {:.2e} kg\n\
                     • Gas giants: {}",
                    stats.stellar_formation_rate,
                    stats.average_stellar_mass,
                    stats.main_sequence_stars,
                    stats.evolved_stars,
                    stats.stellar_remnants,
                    stats.hydrogen_fraction,
                    stats.helium_fraction,
                    stats.carbon_fraction,
                    stats.oxygen_fraction,
                    stats.iron_fraction,
                    stats.heavy_elements_fraction,
                    stats.metallicity,
                    stats.planet_count,
                    stats.planet_formation_rate,
                    stats.average_planet_mass,
                    stats.gas_giants
                )
            },
            DebugMode::Cosmology => {
                format!(
                    "🌌 Cosmology & Structure [F2 to change mode]\n\
                     \n\
                     📐 Universal Structure:\n\
                     • Universe radius: {:.2e} m\n\
                     • Hubble constant: {:.1} km/s/Mpc\n\
                     • Critical density: {:.2e} kg/m³\n\
                     \n\
                     🌑 Dark Sector:\n\
                     • Dark matter fraction: {:.1}%\n\
                     • Dark energy fraction: {:.1}%\n\
                     • Ordinary matter fraction: {:.1}%\n\
                     \n\
                     ⏰ Cosmic Evolution:\n\
                     • Universe age: {:.2} Gyr\n\
                     • Current tick: {}\n\
                     • Simulation speed: {:.1} UPS target\n\
                     \n\
                     🧬 Life & Intelligence:\n\
                     • Extinct lineages: {}\n\
                     • Average tech level: {:.2}\n\
                     • Immortal lineages: {}\n\
                     • Consciousness emergence rate: {:.2e}/year",
                    stats.universe_radius,
                    stats.hubble_constant,
                    stats.critical_density,
                    stats.dark_matter_fraction * 100.0,
                    stats.dark_energy_fraction * 100.0,
                    stats.ordinary_matter_fraction * 100.0,
                    stats.universe_age_gyr,
                    stats.current_tick,
                    stats.target_ups,
                    stats.extinct_lineages,
                    stats.average_tech_level,
                    stats.immortal_lineages,
                    stats.consciousness_emergence_rate
                )
            },
        };

        self.render_debug_text(encoder, view, &debug_text)
        */
    }

    /// Render debug text with proper buffer lifecycle management
    fn render_debug_text(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, text: &str) -> Result<()> {
        // Never clear text buffers during debug panel rendering to prevent 
        // buffer destruction validation errors. Only clear when panel is closed.
        if self.text_buffers.len() > 20 {
            if !self.debug_panel.visible {
                self.device.poll(wgpu::Maintain::Wait);
                let submission_index = self.last_submission_index.clone().unwrap_or_else(|| self.queue.submit(std::iter::empty()));
                for buf in self.text_buffers.drain(..) {
                    self.retired_text_buffers.push((buf, submission_index.clone()));
                }
                self.cleanup_retired_text_buffers();
                println!("🧹 Cleared old text buffers safely");
            }
        }

        // CRITICAL: Store buffer BEFORE creating TextArea to ensure proper lifecycle
        let mut debug_buffer = GlyphonBuffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
        debug_buffer.set_size(&mut self.font_system, self.size.width as f32 * 0.4, self.size.height as f32 * 0.8);
        debug_buffer.set_text(
            &mut self.font_system,
            text,
            Attrs::new().family(Family::Monospace),
            Shaping::Advanced
        );
        debug_buffer.shape_until_scroll(&mut self.font_system);

        // Store buffer FIRST to ensure it stays alive during GPU operations
        self.text_buffers.push(debug_buffer);
        
        // Get reference to the buffer we just stored
        let buffer_ref = self.text_buffers.last().unwrap();

        // Create text area with reference to stored buffer
        let text_area = TextArea {
            buffer: buffer_ref,
            left: 20.0,
            top: 20.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 20,
                top: 20,
                right: (self.size.width / 2) as i32,
                bottom: (self.size.height - 20) as i32,
            },
            default_color: GlyphonColor::rgb(255, 255, 255), // Bright white text for better visibility
        };

        // Prepare text rendering
        if let Err(e) = self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            Resolution {
                width: self.size.width,
                height: self.size.height,
            },
            [text_area],
            &mut self.text_cache,
        ) {
            warn!("Text preparation failed: {}", e);
            return Ok(()); // Don't crash, just skip text rendering
        }

        // Render the text
        {
            let mut text_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug Text Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if let Err(e) = self.text_renderer.render(&self.text_atlas, &mut text_pass) {
                warn!("Text rendering failed: {}", e);
                // Don't return error, just log and continue
            }
        }

        Ok(())
    }

    /// Handle flythrough movement (Unity WASD controls when right mouse is held)
    pub fn handle_flythrough_movement(&mut self, key_code: winit::keyboard::KeyCode, state: ElementState) {
        if !self.mouse_state.is_flythrough {
            return; // Only process WASD when in flythrough mode
        }
        
        let is_pressed = state == ElementState::Pressed;
        let speed = 0.5; // Flythrough movement speed
        
        if is_pressed {
            let mut movement = Vector3::new(0.0, 0.0, 0.0);
            
            match key_code {
                winit::keyboard::KeyCode::KeyW => movement.z = speed,     // Forward
                winit::keyboard::KeyCode::KeyS => movement.z = -speed,    // Backward  
                winit::keyboard::KeyCode::KeyA => movement.x = -speed,    // Left
                winit::keyboard::KeyCode::KeyD => movement.x = speed,     // Right
                winit::keyboard::KeyCode::KeyQ => movement.y = -speed,    // Down (Unity style)
                winit::keyboard::KeyCode::KeyE => movement.y = speed,     // Up (Unity style)
                _ => return,
            }
            
            self.camera.flythrough_move(movement, 1.0);
            println!("✈️ Unity Flythrough: {:?} pressed, movement: {:?}", key_code, movement);
        }
    }

    /// Reset camera to microscope view - focus on a particle up close
    pub fn camera_microscope_reset(&mut self) {
        // Position camera very close to origin to see particles like through a microscope
        self.camera.position = Point3::new(0.01, 0.01, 0.05); // Very close
        self.camera.target = Point3::new(0.0, 0.0, 0.0);      // Looking at origin
        self.camera.up = Vector3::new(0.0, 1.0, 0.0);         // Standard up vector
        self.camera.fov = 45.0;                                // Good viewing angle
        
        // Update camera matrices
        self.camera.update_view();
        self.camera.update_projection();
        
        println!("🔬 MICROSCOPE MODE: Camera positioned at distance {:.4} for particle inspection", 
                (self.camera.position - self.camera.target).magnitude());
        println!("🔍 Use scroll wheel to zoom in/out, drag to pan around particles");
    }

    /// Clean up retired text buffers by polling the device and then dropping all retired buffers
    fn cleanup_retired_text_buffers(&mut self) {
        if self.retired_text_buffers.is_empty() {
            return;
        }
        self.device.poll(wgpu::Maintain::Wait);
        self.retired_text_buffers.clear();
    }
}

#[allow(dead_code)]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// === PUBLIC RENDERER ENTRY POINT ===

/// Run the native renderer event loop for the given shared simulation.
/// This sets up a winit event loop, opens a window, instantiates the `NativeRenderer`,
/// and continuously ticks the physics simulation while rendering each frame.
///
/// The function blocks until the window is closed. All interaction is handled on the
/// main thread in accordance with winit requirements.
pub async fn run_renderer(sim: std::sync::Arc<std::sync::Mutex<universe_sim::UniverseSimulation>>) -> anyhow::Result<()> {
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::window::WindowBuilder;

    // Create the winit event loop – the new API returns a Result.
    let event_loop: EventLoop<()> = EventLoop::new()?;

    // Build the window that will host our renderer.
    let window = WindowBuilder::new()
        .with_title("EVOLUTION – Native Renderer")
        .build(&event_loop)?;

    // Initialise the renderer (async) – we block on the future here since the
    // outer function is already async.
    let mut renderer = pollster::block_on(NativeRenderer::new(&window))?;

    // Clone the simulation Arc so we can move it into the event loop closure.
    let sim_arc = sim.clone();
    // Track the time between frames for a stable simulation delta.
    let mut last_frame_inst = std::time::Instant::now();
    // Capture window ID and initial size before moving window into closure
    let window_id = window.id();
    let mut window_size = window.inner_size();

    // Main event loop – ownership of window and renderer is moved into the
    // closure so their lifetimes outlive the loop itself.
    event_loop.run(move |event, elwt| {
        use winit::event::{Event, WindowEvent, MouseScrollDelta};
        use tracing::error;
        
        match event {
            // === Window-specific events ===
            Event::WindowEvent { window_id: win_id, event } if win_id == window_id => {
                match event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    WindowEvent::Resized(size) => {
                        window_size = size;
                        renderer.resize(size);
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        renderer.resize(window_size);
                    }
                    WindowEvent::RedrawRequested => {
                        // --- Simulation update ---
                        let now = std::time::Instant::now();
                        let _dt = (now - last_frame_inst).as_secs_f64();
                        last_frame_inst = now;

                        // Tick the simulation inside a critical section.
                        if let Ok(mut sim) = sim_arc.lock() {
                            if let Err(e) = sim.tick() {
                                error!("Simulation tick failed: {}", e);
                            }
                            // Use universe age as a time parameter in seconds for the renderer.
                            let sim_time = sim.universe_age_years() as f32;
                            if let Err(e) = renderer.render(sim_time) {
                                error!("Render error: {}", e);
                            }
                        }
                    }
                    WindowEvent::KeyboardInput { event: key_event, .. } => {
                        // Convert physical key to `KeyCode` where possible and forward to renderer.
                        if let winit::keyboard::PhysicalKey::Code(code) = key_event.physical_key {
                            renderer.handle_debug_input(code, key_event.state);
                            renderer.handle_flythrough_movement(code, key_event.state);
                        }
                    }
                    WindowEvent::ModifiersChanged(mods) => {
                        renderer.handle_modifiers(mods.state());
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        renderer.handle_mouse_button(button, state);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        renderer.handle_mouse_move(position.x, position.y);
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        // Normalise the wheel delta to a single float for simplicity.
                        let scroll_amount = match delta {
                            MouseScrollDelta::LineDelta(_x, y) => y * 32.0,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32,
                        };
                        renderer.handle_mouse_wheel(scroll_amount);
                    }
                    _ => {}
                }
            }

            // === Event loop is about to sleep – ensure we redraw continuously ===
            Event::AboutToWait => {
                elwt.set_control_flow(ControlFlow::Poll);
            }

            _ => {}
        }
    })?;

    Ok(())
} 