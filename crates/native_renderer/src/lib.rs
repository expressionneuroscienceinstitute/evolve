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
use serde_json::Value;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use num_complex::Complex;

// === QUANTUM FIELD VISUALIZATION SYSTEM ===

/// Quantum Field State Vector for quantum-accurate visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFieldStateVector {
    pub amplitude: Complex<f64>,                    // Complex field amplitude
    pub phase: f64,                                // Quantum phase
    pub superposition_states: Vec<Complex<f64>>,   // Multiple quantum states
    pub entanglement_map: HashMap<usize, f64>,     // Entanglement correlations
    pub decoherence_rate: f64,                     // Rate of quantum-to-classical transition
    pub scale_factor: f64,                         // Multi-scale quantum effects
    pub uncertainty_position: f64,                 // Position uncertainty (Heisenberg)
    pub uncertainty_momentum: f64,                 // Momentum uncertainty (Heisenberg)
    pub coherence_time: f64,                       // Quantum coherence lifetime
    pub field_type: QuantumFieldType,              // Type of quantum field
}

impl Default for QuantumFieldStateVector {
    fn default() -> Self {
        Self {
            amplitude: Complex::new(1.0, 0.0),
            phase: 0.0,
            superposition_states: vec![Complex::new(1.0, 0.0)],
            entanglement_map: HashMap::new(),
            decoherence_rate: 0.0,
            scale_factor: 1.0,
            uncertainty_position: 1e-10, // 1 angstrom
            uncertainty_momentum: 1e-24, // kg⋅m/s
            coherence_time: 1e-12,       // 1 picosecond
            field_type: QuantumFieldType::Scalar,
        }
    }
}

/// Types of quantum fields for visualization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QuantumFieldType {
    Scalar,     // Higgs field, etc.
    Fermion,    // Electron field, quark fields
    Vector,     // Photon field, gluon fields
    Tensor,     // Graviton field
}

/// Quantum visualization modes for different quantum phenomena
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumVisualizationMode {
    Classical,          // Traditional particle visualization
    Superposition,      // Show quantum superposition states
    Entanglement,       // Visualize quantum entanglement
    Decoherence,        // Show quantum-to-classical transitions
    FieldFluctuations,  // Vacuum energy and virtual particles
    MultiScale,         // Scale-dependent quantum effects
    Interference,       // Quantum interference patterns
    Tunneling,          // Quantum tunneling effects
}

/// Extended particle vertex with quantum field data
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct QuantumParticleVertex {
    // Classical particle data
    pub position: [f32; 3],
    pub velocity: [f32; 3], 
    pub mass: f32,
    pub charge: f32,
    pub temperature: f32,
    pub particle_type: f32,
    pub interaction_count: f32,
    
    // Quantum field data
    pub quantum_amplitude_real: f32,    // Real part of quantum amplitude
    pub quantum_amplitude_imag: f32,    // Imaginary part of quantum amplitude
    pub quantum_phase: f32,             // Quantum phase
    pub decoherence_rate: f32,          // Rate of quantum-to-classical transition
    pub uncertainty_position: f32,      // Position uncertainty
    pub uncertainty_momentum: f32,      // Momentum uncertainty
    pub coherence_time: f32,            // Quantum coherence lifetime
    pub entanglement_strength: f32,     // Strength of entanglement
    pub field_type: f32,                // Quantum field type
    pub scale_factor: f32,              // Multi-scale quantum effects
    
    // GPU alignment padding
    pub _padding: [f32; 2],
}

impl Default for QuantumParticleVertex {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            mass: 0.0,
            charge: 0.0,
            temperature: 0.0,
            particle_type: 0.0,
            interaction_count: 0.0,
            quantum_amplitude_real: 1.0,
            quantum_amplitude_imag: 0.0,
            quantum_phase: 0.0,
            decoherence_rate: 0.0,
            uncertainty_position: 1e-10,
            uncertainty_momentum: 1e-24,
            coherence_time: 1e-12,
            entanglement_strength: 0.0,
            field_type: 0.0,
            scale_factor: 1.0,
            _padding: [0.0, 0.0],
        }
    }
}

/// Quantum field visualization parameters
#[derive(Debug, Clone)]
pub struct QuantumVisualizationParams {
    pub mode: QuantumVisualizationMode,
    pub superposition_threshold: f32,   // Threshold for showing superposition
    pub entanglement_threshold: f32,    // Threshold for showing entanglement
    pub decoherence_scale: f32,         // Scale factor for decoherence effects
    pub field_fluctuation_amplitude: f32, // Amplitude of field fluctuations
    pub interference_visibility: f32,   // Visibility of interference patterns
    pub tunneling_probability_scale: f32, // Scale for tunneling visualization
    pub multi_scale_transition: f32,    // Scale for quantum-classical transition
}

impl Default for QuantumVisualizationParams {
    fn default() -> Self {
        Self {
            mode: QuantumVisualizationMode::Classical,
            superposition_threshold: 0.1,
            entanglement_threshold: 0.5,
            decoherence_scale: 1.0,
            field_fluctuation_amplitude: 0.1,
            interference_visibility: 1.0,
            tunneling_probability_scale: 1.0,
            multi_scale_transition: 1e-9, // 1 nanometer
        }
    }
}

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
mod cosmological_renderer;

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
    instance: wgpu::Instance,
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
    
    // Quantum visualization: quantum field pipeline
    #[cfg(feature = "quantum-visualization")]
    quantum_pipeline: Option<wgpu::RenderPipeline>,
    #[cfg(feature = "quantum-visualization")]
    quantum_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "quantum-visualization")]
    quantum_uniform_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "quantum-visualization")]
    quantum_bind_group: Option<wgpu::BindGroup>,
    
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

    // Quantum visualization state
    #[cfg(feature = "quantum-visualization")]
    quantum_mode_enabled: bool,
    #[cfg(feature = "quantum-visualization")]
    quantum_visualization_params: QuantumVisualizationParams,
    #[cfg(feature = "quantum-visualization")]
    quantum_field_states: Vec<QuantumFieldStateVector>,
    #[cfg(feature = "quantum-visualization")]
    quantum_particles: Vec<QuantumParticleVertex>,

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

    pub multi_agent_overlay_enabled: bool,
    pub multi_agent_network_json: Option<Value>,

    // Add to NativeRenderer struct:
    pub agent_visualization_mode: AgentVisualizationMode,
    pub selected_agent_id: Option<String>,
    pub agent_timeline_data: Vec<AgentTimelineEvent>,
    pub interaction_heatmap: Vec<InteractionHeatmapCell>,

    // Advanced visualization enhancements
    pub adaptive_visualization: AdaptiveVisualizationParams,
    pub performance_monitoring: bool,
    pub scientific_overlay: bool,
    pub real_time_analytics: bool,
}

/// Uniform data sent to GPU with heavy mode and quantum visualization extensions
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    // Quantum visualization parameters
    quantum_mode: f32,                    // Quantum visualization mode
    superposition_threshold: f32,         // Threshold for superposition display
    entanglement_threshold: f32,          // Threshold for entanglement display
    decoherence_scale: f32,               // Scale factor for decoherence effects
    field_fluctuation_amplitude: f32,     // Amplitude of field fluctuations
    interference_visibility: f32,         // Visibility of interference patterns
    tunneling_probability_scale: f32,     // Scale for tunneling visualization
    multi_scale_transition: f32,          // Scale for quantum-classical transition
    _padding: [f32; 2],                   // GPU alignment padding
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
                }
                KeyCode::Digit2 => {
                    self.color_mode = ColorMode::Charge;
                }
                KeyCode::Digit3 => {
                    self.color_mode = ColorMode::Temperature;
                }
                KeyCode::Digit4 => {
                    self.color_mode = ColorMode::Velocity;
                }
                KeyCode::Digit5 => {
                    self.color_mode = ColorMode::Scientific;
                }
                KeyCode::Digit6 => {
                    self.color_mode = ColorMode::Scientific;
                }
                KeyCode::Digit7 => {
                    self.color_mode = ColorMode::Scientific;
                }
                KeyCode::Digit8 => {
                    self.color_mode = ColorMode::Scientific;
                }
                KeyCode::Digit9 => {
                    self.color_mode = ColorMode::Scientific;
                }
                KeyCode::Digit0 => {
                    self.color_mode = ColorMode::Interactions;
                }
                KeyCode::Space => {
                    self.focus_on_origin();
                }
                KeyCode::KeyR => {
                    // Reset camera view
                    self.reset_view();
                }
                KeyCode::KeyM => {
                    println!("🔬 Microscope mode toggle requested");
                }
                KeyCode::F5 => {
                    // Camera doesn't have microscope reset - this is handled by the renderer
                },
                KeyCode::F6 => {
                    // Camera doesn't have debug panel - this is handled by the renderer
                },
                KeyCode::F7 => {
                    // Force window redraw for testing resizing
                    println!("🔄 FORCE REDRAW: Requesting window redraw for resize testing");
                    // This will trigger a RedrawRequested event
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
        info!("🚀 Initializing high-performance native renderer");
        
        let size = window.inner_size();
        info!("📐 Window size: {}x{}", size.width, size.height);
        
        // Validate window size
        if size.width == 0 || size.height == 0 {
            return Err(anyhow::anyhow!("Invalid window size: {}x{}", size.width, size.height));
        }
        
        // Create WGPU instance with fallback options
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        
        info!("✅ WGPU instance created successfully");
        
        // Create surface with error handling
        let surface = match instance.create_surface(window) {
            Ok(s) => {
                info!("✅ Surface created successfully");
                s
            },
            Err(e) => {
                error!("❌ Failed to create surface: {}", e);
                return Err(anyhow::anyhow!("Surface creation failed: {}", e));
            }
        };
        
        // Request adapter with fallback options
        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };
        
        let adapter = match instance.request_adapter(&adapter_options).await {
            Some(adapter) => {
                let info = adapter.get_info();
                info!("✅ GPU adapter selected: {} ({:?})", info.name, info.backend);
                adapter
            },
            None => {
                error!("❌ No suitable GPU adapter found, trying fallback adapter");
                // Try with fallback adapter
                let fallback_options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: true,
                };
                
                match instance.request_adapter(&fallback_options).await {
                    Some(adapter) => {
                        let info = adapter.get_info();
                        info!("✅ Fallback GPU adapter selected: {} ({:?})", info.name, info.backend);
                        adapter
                    },
                    None => {
                        error!("❌ No GPU adapters available at all");
                        return Err(anyhow::anyhow!("No GPU adapters available"));
                    }
                }
            }
        };
        
        // Create device and queue with conservative limits
        let device_descriptor = wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_buffer_size: 256 * 1024 * 1024, // 256MB max buffer size
                max_storage_buffer_binding_size: 128 * 1024 * 1024, // 128MB max storage buffer
                max_uniform_buffer_binding_size: 16 * 1024 * 1024, // 16MB max uniform buffer
                ..wgpu::Limits::default()
            },
            label: Some("Universe Simulation Device"),
        };
        
        let (device, queue) = match adapter.request_device(&device_descriptor, None).await {
            Ok((device, queue)) => {
                info!("✅ Device and queue created successfully");
                (device, queue)
            },
            Err(e) => {
                error!("❌ Failed to create device: {}", e);
                return Err(anyhow::anyhow!("Device creation failed: {}", e));
            }
        };
        
        // Configure surface with error handling
        let surface_caps = surface.get_capabilities(&adapter);
        if surface_caps.formats.is_empty() {
            error!("❌ No supported surface formats found");
            return Err(anyhow::anyhow!("No supported surface formats"));
        }
        
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        info!("✅ Surface format selected: {:?}", surface_format);
        
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
        info!("✅ Surface configured successfully");
        
        // Load and compile shader with error handling
        let shader_source = include_str!("shaders/particle.wgsl");
        let shader = match device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        }) {
            shader => {
                info!("✅ Shader compiled successfully");
                shader
            }
        };
        
        // Create uniform buffer with conservative size
        let max_particles = 400_000; // Reduced from 800k to 400k for better compatibility
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        info!("✅ Uniform buffer created ({} bytes)", std::mem::size_of::<Uniforms>());
        
        // Create quad vertex buffer
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
        
        info!("✅ Quad vertex buffer created ({} bytes)", quad_vertices.len() * std::mem::size_of::<f32>());
        
        // Create instance buffer for particle data with conservative size
        let vertex_buffer_size = (max_particles * std::mem::size_of::<SimpleParticleVertex>()) as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: vertex_buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        info!("✅ Particle instance buffer created ({} MB)", vertex_buffer_size / 1024 / 1024);
        
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
        
        info!("✅ Bind group created successfully");
        
        // Create render pipeline with error handling
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = match device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        }) {
            pipeline => {
                info!("✅ Render pipeline created successfully");
                pipeline
            }
        };
        
        let mut camera = Camera::default();
        camera.aspect = size.width as f32 / size.height as f32;
        camera.position = Point3::new(0.0, 0.0, 5.0); // Moved back from 0.1 to 5.0 for better visibility
        camera.target = Point3::new(0.0, 0.0, 0.0);
        camera.update_view();
        camera.update_projection();
        
        info!("✅ Camera initialized at position ({:.1}, {:.1}, {:.1})", 
              camera.position.x, camera.position.y, camera.position.z);
        
        // Initialize text renderer (glyphon) with error handling
        let font_system = FontSystem::new();
        let mut text_atlas = match TextAtlas::new(&device, &queue, config.format) {
            atlas => {
                info!("✅ Text atlas created successfully");
                atlas
            }
        };
        
        let text_renderer = match TextRenderer::new(&mut text_atlas, &device, wgpu::MultisampleState::default(), None) {
            renderer => {
                info!("✅ Text renderer created successfully");
                renderer
            }
        };
        let text_cache = SwashCache::new();
        
        info!("🎉 Native renderer initialization completed successfully!");
        
        Ok(Self {
            instance,
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
            
            #[cfg(feature = "quantum-visualization")]
            quantum_pipeline: None,
            #[cfg(feature = "quantum-visualization")]
            quantum_vertex_buffer: None,
            #[cfg(feature = "quantum-visualization")]
            quantum_uniform_buffer: None,
            #[cfg(feature = "quantum-visualization")]
            quantum_bind_group: None,
            
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

            #[cfg(feature = "quantum-visualization")]
            quantum_mode_enabled: false,
            #[cfg(feature = "quantum-visualization")]
            quantum_visualization_params: QuantumVisualizationParams::default(),
            #[cfg(feature = "quantum-visualization")]
            quantum_field_states: Vec::new(),
            #[cfg(feature = "quantum-visualization")]
            quantum_particles: Vec::new(),

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

            multi_agent_overlay_enabled: false,
            multi_agent_network_json: None,

            // Add to NativeRenderer struct:
            agent_visualization_mode: AgentVisualizationMode::Overview,
            selected_agent_id: None,
            agent_timeline_data: Vec::new(),
            interaction_heatmap: Vec::new(),

            // Advanced visualization enhancements
            adaptive_visualization: AdaptiveVisualizationParams::default(),
            performance_monitoring: false,
            scientific_overlay: false,
            real_time_analytics: false,
        })
    }
    
    /// Update particle data from simulation with zero-copy access
    pub fn update_particles(&mut self, simulation: &mut universe_sim::UniverseSimulation) -> Result<()> {
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
            let base_size = 1.0; // Increased for visibility
            let size_multiplier = self.camera.get_particle_size_multiplier();
            let final_size = base_size * size_multiplier;

            // Scale physics particles to be visible (increased scale factor)
            let scale_factor = 1e-1; // Increased from 1e-2 for much better visibility
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
            let base_size = 1.0; // Increased for visibility
            let size_multiplier = self.camera.get_particle_size_multiplier();
            let final_size = base_size * size_multiplier;
            
            // Scale store particles to be visible (increased scale factor)
            let scale_factor = 1e-1; // Increased from 1e-3 for much better visibility
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
            
            let base_size = 2.0; // Increased for demo particles
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
    
    /// Render frame with maximum performance and heavy mode enhancements
    pub fn render(&mut self, simulation_time: f32) -> Result<()> {
        // --- NEW: Always check window size before rendering ---
        let current_size = self.window.inner_size();
        if current_size != self.size && current_size.width > 0 && current_size.height > 0 {
            println!("🪟 Window size changed: {}x{} (was {}x{}), resizing surface...", current_size.width, current_size.height, self.size.width, self.size.height);
            self.resize(current_size)?;
        }
        let frame_start = std::time::Instant::now();
        
        // Update performance analytics for adaptive visualization
        self.update_performance_analytics();
        
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
            quantum_mode: 0.0,
            superposition_threshold: 0.0,
            entanglement_threshold: 0.0,
            decoherence_scale: 0.0,
            field_fluctuation_amplitude: 0.0,
            interference_visibility: 0.0,
            tunneling_probability_scale: 0.0,
            multi_scale_transition: 0.0,
            _padding: [0.0; 2],
        };
        
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        // Get surface texture
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(e) => {
                error!("Failed to get surface texture: {:?}", e);
                return Err(e.into());
            }
        };
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
        
        // Main render pass with proper space background
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,  // Very dark blue (space background)
                            g: 0.02,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            // Set the viewport to cover the entire window
            render_pass.set_viewport(0.0, 0.0, self.size.width as f32, self.size.height as f32, 0.0, 1.0);
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
            
            // Draw 6 vertices per instance (2 triangles per particle quad)
            let instance_count = self.particle_count as u32;
            if instance_count > 0 {
                render_pass.draw(0..6, 0..instance_count);
            }
        }

        // ---- Text overlay pass ----
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
        }

        // ---- Debug Panel Overlay ----
        if let Err(e) = self.render_debug_panel(&mut encoder, &view) {
            warn!("Debug panel rendering failed: {}", e);
        }

        // ---- Multi-Agent Overlay ----
        if let Err(e) = self.render_multi_agent_overlay(&mut encoder, &view) {
            warn!("Multi-Agent overlay rendering failed: {}", e);
        }

        // === Advanced Visualization Overlays ===
        // Render performance monitoring overlay
        if self.performance_monitoring {
            if let Err(e) = self.render_performance_overlay(&mut encoder, &view) {
                warn!("Failed to render performance overlay: {}", e);
            }
        }

        // Render scientific overlay
        if self.scientific_overlay {
            if let Err(e) = self.render_scientific_overlay(&mut encoder, &view) {
                warn!("Failed to render scientific overlay: {}", e);
            }
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

        // === GPU lifecycle safeguard for debug resources ===
        if self.debug_panel.visible {
            self.device.poll(wgpu::Maintain::Poll);
        } else {
            self.device.poll(wgpu::Maintain::Poll);
        }

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

        Ok(())
    }
    
    /// Handle window resize
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) -> Result<()> {
        if new_size.width > 0 && new_size.height > 0 {
            info!("🔄 Resizing renderer to {}x{}", new_size.width, new_size.height);
            
            // Update internal size tracking
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;

            // Wait for GPU to finish all work before reconfiguring
            self.device.poll(wgpu::Maintain::Wait);
            
            // Recreate surface (macOS compatibility)
            match self.instance.create_surface(self.window) {
                Ok(new_surface) => {
                    self.surface = new_surface;
                    info!("✅ Surface recreated successfully");
                }
                Err(e) => {
                    error!("Failed to recreate surface: {}", e);
                    return Err(anyhow::anyhow!("Surface recreation failed: {}", e));
                }
            }

            // Configure the new surface
            self.surface.configure(&self.device, &self.config);
            info!("✅ Surface configured with new dimensions");
            
            // Update camera aspect ratio and matrices
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
            self.camera.update_projection();
            self.camera.update_view();
            
            info!("✅ Resize completed - aspect ratio: {:.3}", self.camera.aspect);
            Ok(())
        } else {
            warn!("Invalid resize dimensions: {}x{}", new_size.width, new_size.height);
            Err(anyhow::anyhow!("Invalid resize dimensions: {}x{}", new_size.width, new_size.height))
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
    
    /// Test function to verify renderer is working
    pub fn test_renderer(&self) -> String {
        format!(
            "Renderer Test Results:\n\
            - Window size: {}x{}\n\
            - Camera position: ({:.2}, {:.2}, {:.2})\n\
            - Camera target: ({:.2}, {:.2}, {:.2})\n\
            - Camera aspect: {:.3}\n\
            - Particle count: {}\n\
            - Frame count: {}\n\
            - FPS: {:.1}",
            self.size.width, self.size.height,
            self.camera.position.x, self.camera.position.y, self.camera.position.z,
            self.camera.target.x, self.camera.target.y, self.camera.target.z,
            self.camera.aspect,
            self.particle_count,
            self.frame_count,
            self.metrics.fps
        )
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
        // Apply zoom with proper sensitivity
        let zoom_delta = delta * 0.1; // Scale the delta for better control
        self.camera.zoom(zoom_delta);
        info!("🔍 Mouse wheel zoom: delta={:.3}, zoom_delta={:.3}", delta, zoom_delta);
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
                    // Reset camera to default position for debugging
                    self.camera.position = Point3::new(0.0, 0.0, 2.0);
                    self.camera.target = Point3::new(0.0, 0.0, 0.0);
                    self.camera.update_view();
                    self.camera.update_projection();
                    println!("🔄 Camera reset to default position for debugging");
                },
                KeyCode::F5 => {
                    // Camera doesn't have microscope reset - this is handled by the renderer
                },
                KeyCode::F6 => {
                    // Camera doesn't have debug panel - this is handled by the renderer
                },
                KeyCode::F7 => {
                    // Force window redraw for testing resizing
                    println!("🔄 FORCE REDRAW: Requesting window redraw for resize testing");
                    // This will trigger a RedrawRequested event
                },
                KeyCode::Space => {
                    // Focus camera on origin
                    self.camera.focus_on_origin();
                }
                KeyCode::KeyR => {
                    // Reset camera view
                    self.camera.reset_view();
                }
                KeyCode::KeyM => {
                    println!("🔬 Microscope mode toggle requested");
                }
                KeyCode::KeyC => {
                    // Toggle color mode (legacy)
                    let next_color_mode = match self.camera.color_mode {
                        ColorMode::ParticleType => ColorMode::Charge,
                        ColorMode::Charge => ColorMode::Temperature,
                        ColorMode::Temperature => ColorMode::Velocity,
                        ColorMode::Velocity => ColorMode::Interactions,
                        ColorMode::Interactions => ColorMode::Scientific,
                        ColorMode::Scientific => ColorMode::ParticleType,
                    };
                    self.camera.color_mode = next_color_mode;
                }
                KeyCode::KeyS => {
                    // Toggle scale mode (legacy)
                    let next_scale_mode = match self.camera.scale_mode {
                        ScaleMode::Linear => ScaleMode::Logarithmic,
                        ScaleMode::Logarithmic => ScaleMode::Energy,
                        ScaleMode::Energy => ScaleMode::Custom(1.0),
                        ScaleMode::Custom(_) => ScaleMode::Linear,
                    };
                    self.camera.scale_mode = next_scale_mode;
                }
                // Multi-agent visualization controls
                KeyCode::KeyM => {
                    self.toggle_multi_agent_overlay();
                    info!("Multi-agent overlay toggled: {}", self.multi_agent_overlay_enabled);
                }
                KeyCode::KeyN => {
                    self.cycle_agent_visualization_mode();
                    info!("Agent visualization mode: {:?}", self.agent_visualization_mode);
                }
                KeyCode::KeyI => {
                    // Cycle through agents in inspector mode
                    if let Some(ref json) = self.multi_agent_network_json {
                        if let Some(agents) = json["agents"].as_array() {
                            if let Some(current_selected) = &self.selected_agent_id {
                                // Find current agent index and move to next
                                if let Some(current_idx) = agents.iter().position(|a| a["id"].as_str() == Some(current_selected)) {
                                    let next_idx = (current_idx + 1) % agents.len();
                                    if let Some(next_agent) = agents.get(next_idx) {
                                        self.selected_agent_id = next_agent["id"].as_str().map(|s| s.to_string());
                                    }
                                }
                            } else {
                                // Select first agent
                                if let Some(first_agent) = agents.first() {
                                    self.selected_agent_id = first_agent["id"].as_str().map(|s| s.to_string());
                                }
                            }
                        }
                    }
                }
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

        let stats = self.get_buffer_pool_stats();
        let formatted_text = format!(
            "🔬 EVOLUTION Universe Simulation - Debug Panel [F1=Toggle]\n\
            📊 Renderer Status\n\
            \n\
            🎮 Controls:\n\
            • F1: Toggle debug panel\n\
            • F2: Cycle debug modes\n\
            • F4: Reset camera position\n\
            • Space: Focus on origin\n\
            • R: Reset camera view\n\
            • Mouse wheel: Zoom in/out\n\
            • Alt+Left: Orbit camera\n\
            • Middle mouse: Pan camera\n\
            • Right mouse: Flythrough mode\n\
            \n\
            📈 Performance:\n\
            • FPS: {:.1}\n\
            • Frame time: {:.2} ms\n\
            • Particles rendered: {}\n\
            • Window size: {}x{}\n\
            \n\
            📷 Camera:\n\
            • Position: ({:.2}, {:.2}, {:.2})\n\
            • Target: ({:.2}, {:.2}, {:.2})\n\
            • Aspect ratio: {:.3}\n\
            • Zoom level: {:.2}\n\
            \n\
            🔧 Buffer Pool Stats:\n\
            • Active buffers: {}\n\
            • Retired buffers: {}\n\
            • Active memory: {:.2} MB\n\
            • Total allocated: {:.2} MB\n\
            • Peak memory: {:.2} MB",
            self.metrics.fps,
            self.metrics.frame_time_ms,
            self.particle_count,
            self.size.width, self.size.height,
            self.camera.position.x, self.camera.position.y, self.camera.position.z,
            self.camera.target.x, self.camera.target.y, self.camera.target.z,
            self.camera.aspect,
            self.camera.get_zoom_level(),
            stats.active_buffers,
            stats.retired_buffers,
            stats.active_memory_mb,
            stats.total_allocated_mb,
            stats.peak_memory_mb
        );
        self.render_debug_text(encoder, view, &formatted_text)
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

    /// Accepts a JSON summary of the multi-agent system for visualization
    pub fn set_multi_agent_network(&mut self, network_json: Value) {
        self.multi_agent_network_json = Some(network_json);
    }

    /// Toggle the multi-agent overlay
    pub fn toggle_multi_agent_overlay(&mut self) {
        self.multi_agent_overlay_enabled = !self.multi_agent_overlay_enabled;
    }

    /// Set agent timeline data for visualization
    pub fn set_agent_timeline(&mut self, timeline_data: Vec<AgentTimelineEvent>) {
        self.agent_timeline_data = timeline_data;
    }

    /// Set interaction heatmap data
    pub fn set_interaction_heatmap(&mut self, heatmap_data: Vec<InteractionHeatmapCell>) {
        self.interaction_heatmap = heatmap_data;
    }

    /// Cycle through visualization modes
    pub fn cycle_agent_visualization_mode(&mut self) {
        self.agent_visualization_mode = match self.agent_visualization_mode {
            AgentVisualizationMode::Overview => AgentVisualizationMode::Network,
            AgentVisualizationMode::Network => AgentVisualizationMode::Timeline,
            AgentVisualizationMode::Timeline => AgentVisualizationMode::Heatmap,
            AgentVisualizationMode::Heatmap => AgentVisualizationMode::Inspector,
            AgentVisualizationMode::Inspector => AgentVisualizationMode::Overview,
        };
    }

    /// Enhanced multi-agent overlay rendering with multiple visualization modes
    pub fn render_multi_agent_overlay(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> anyhow::Result<()> {
        if !self.multi_agent_overlay_enabled { return Ok(()); }
        
        match self.agent_visualization_mode {
            AgentVisualizationMode::Overview => self.render_agent_overview(encoder, view)?,
            AgentVisualizationMode::Network => self.render_agent_network(encoder, view)?,
            AgentVisualizationMode::Timeline => self.render_agent_timeline(encoder, view)?,
            AgentVisualizationMode::Heatmap => self.render_interaction_heatmap(encoder, view)?,
            AgentVisualizationMode::Inspector => self.render_agent_inspector(encoder, view)?,
        }
        Ok(())
    }

    /// Render basic agent overview
    fn render_agent_overview(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> anyhow::Result<()> {
        if let Some(ref json) = self.multi_agent_network_json {
            let agent_count = json["agents"].as_array().map(|a| a.len()).unwrap_or(0);
            let metrics = &json["metrics"];
            let overlay_text = format!(
                "=== AGENT OVERVIEW ===\nAgents: {}\nEfficiency: {:.2}\nCoordination: {:.2}\nCommunication: {:.2}\nMode: {:?}",
                agent_count,
                metrics["overall_efficiency"].as_f64().unwrap_or(0.0),
                metrics["coordination_quality"].as_f64().unwrap_or(0.0),
                metrics["communication_efficiency"].as_f64().unwrap_or(0.0),
                self.agent_visualization_mode
            );
            self.render_debug_text(encoder, view, &overlay_text)?;
        }
        Ok(())
    }

    /// Render network graph visualization
    fn render_agent_network(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> anyhow::Result<()> {
        if let Some(ref json) = self.multi_agent_network_json {
            let empty_agents = Vec::new();
            let empty_connections = Vec::new();
            let agents = json["agents"].as_array().unwrap_or(&empty_agents);
            let connections = json["connections"].as_array().unwrap_or(&empty_connections);
            
            let mut network_text = String::from("=== AGENT NETWORK ===\n");
            network_text.push_str(&format!("Nodes: {} | Edges: {}\n\n", agents.len(), connections.len()));
            
            // Show top connections by strength
            let mut sorted_connections: Vec<_> = connections.iter()
                .filter_map(|c| {
                    let strength = c["strength"].as_f64()?;
                    let source = c["source"].as_str()?;
                    let target = c["target"].as_str()?;
                    Some((strength, source, target))
                })
                .collect();
            sorted_connections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            
            for (strength, source, target) in sorted_connections.iter().take(5) {
                network_text.push_str(&format!("{} ↔ {} ({:.2})\n", source, target, strength));
            }
            
            self.render_debug_text(encoder, view, &network_text)?;
        }
        Ok(())
    }

    /// Render agent timeline visualization
    fn render_agent_timeline(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> anyhow::Result<()> {
        let mut timeline_text = String::from("=== AGENT TIMELINE ===\n");
        
        // Group events by agent
        let mut agent_events: std::collections::HashMap<String, Vec<&AgentTimelineEvent>> = std::collections::HashMap::new();
        for event in &self.agent_timeline_data {
            agent_events.entry(event.agent_id.clone()).or_default().push(event);
        }
        
        // Show recent events for each agent
        for (agent_id, events) in agent_events.iter().take(3) {
            timeline_text.push_str(&format!("\n{}:\n", agent_id));
            for event in events.iter().take(3) {
                timeline_text.push_str(&format!("  {:.1}s: {}\n", event.timestamp, event.description));
            }
        }
        
        self.render_debug_text(encoder, view, &timeline_text)?;
        Ok(())
    }

    /// Render interaction heatmap
    fn render_interaction_heatmap(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> anyhow::Result<()> {
        let mut heatmap_text = String::from("=== INTERACTION HEATMAP ===\n");
        
        // Group interactions by type and show intensity
        let mut interaction_types: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for cell in &self.interaction_heatmap {
            *interaction_types.entry(cell.interaction_type.clone()).or_default() += cell.intensity;
        }
        
        for (interaction_type, total_intensity) in interaction_types.iter().take(5) {
            heatmap_text.push_str(&format!("{}: {:.2}\n", interaction_type, total_intensity));
        }
        
        self.render_debug_text(encoder, view, &heatmap_text)?;
        Ok(())
    }

    /// Render detailed agent inspector
    fn render_agent_inspector(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> anyhow::Result<()> {
        if let Some(ref json) = self.multi_agent_network_json {
            let empty_agents = Vec::new();
            let agents = json["agents"].as_array().unwrap_or(&empty_agents);
            
            let mut inspector_text = String::from("=== AGENT INSPECTOR ===\n");
            
            if let Some(ref selected_id) = self.selected_agent_id {
                // Show details for selected agent
                if let Some(agent) = agents.iter().find(|a| a["id"].as_str() == Some(selected_id)) {
                    inspector_text.push_str(&format!("Selected: {}\n", selected_id));
                    inspector_text.push_str(&format!("Type: {}\n", agent["type"].as_str().unwrap_or("Unknown")));
                    inspector_text.push_str(&format!("Energy: {:.2}\n", agent["energy"].as_f64().unwrap_or(0.0)));
                    
                    if let Some(trust) = agent["trust"].as_object() {
                        inspector_text.push_str("Trust connections:\n");
                        for (target, strength) in trust.iter().take(3) {
                            inspector_text.push_str(&format!("  {}: {:.2}\n", target, strength.as_f64().unwrap_or(0.0)));
                        }
                    }
                }
            } else {
                // Show agent list
                inspector_text.push_str("Select an agent to inspect:\n");
                for agent in agents.iter().take(5) {
                    let id = agent["id"].as_str().unwrap_or("Unknown");
                    let agent_type = agent["type"].as_str().unwrap_or("Unknown");
                    inspector_text.push_str(&format!("  {} ({})\n", id, agent_type));
                }
            }
            
            self.render_debug_text(encoder, view, &inspector_text)?;
        }
        Ok(())
    }

    /// Set comprehensive multi-agent visualization data
    pub fn set_multi_agent_visualization_data(&mut self, data: MultiAgentVisualizationData) {
        self.multi_agent_network_json = Some(data.network_summary);
        self.agent_timeline_data = data.timeline_events;
        self.interaction_heatmap = data.heatmap_data;
    }

    /// Update quantum state vector data for advanced quantum visualization
    /// This method processes comprehensive quantum field data from the universe simulation
    /// and prepares it for GPU-accelerated quantum visualization
    pub fn update_quantum_state_vectors(&mut self, quantum_data: &HashMap<String, universe_sim::QuantumStateVectorData>) -> Result<()> {
        #[cfg(feature = "quantum-visualization")]
        {
            self.quantum_field_states.clear();
            self.quantum_particles.clear();

            for (field_name, field_data) in quantum_data {
                // Process each quantum field
                self.process_quantum_field_data(field_name, field_data)?;
            }

            // Update quantum vertex buffer if we have quantum particles
            if !self.quantum_particles.is_empty() {
                self.update_quantum_vertex_buffer()?;
            }

            info!("Updated quantum state vectors: {} fields, {} particles", 
                  quantum_data.len(), self.quantum_particles.len());
        }

        #[cfg(not(feature = "quantum-visualization"))]
        {
            warn!("Quantum visualization feature not enabled, skipping quantum state vector update");
        }

        Ok(())
    }

    /// Process individual quantum field data and convert to GPU-compatible format
    #[cfg(feature = "quantum-visualization")]
    fn process_quantum_field_data(&mut self, field_name: &str, field_data: &universe_sim::QuantumStateVectorData) -> Result<()> {
        let (x_dim, y_dim, z_dim) = field_data.field_dimensions;
        
        // Process each lattice point in the quantum field
        for i in 0..x_dim {
            for j in 0..y_dim {
                for k in 0..z_dim {
                    // Create quantum particle vertex from field data
                    let quantum_vertex = self.create_quantum_particle_vertex(field_data, i, j, k)?;
                    self.quantum_particles.push(quantum_vertex);

                    // Create quantum field state vector for advanced visualization
                    let quantum_state = self.create_quantum_field_state_vector(field_data, i, j, k)?;
                    self.quantum_field_states.push(quantum_state);
                }
            }
        }

        Ok(())
    }

    /// Create quantum particle vertex from field data at specific lattice point
    #[cfg(feature = "quantum-visualization")]
    fn create_quantum_particle_vertex(&self, field_data: &universe_sim::QuantumStateVectorData, i: usize, j: usize, k: usize) -> Result<QuantumParticleVertex> {
        // Get quantum data at this lattice point
        let complex_amp = field_data.complex_amplitudes[i][j][k];
        let phase = field_data.phases[i][j][k];
        let magnitude = field_data.magnitudes[i][j][k];
        let entanglement = field_data.entanglement_correlations[i][j][k];
        let decoherence = field_data.decoherence_rates[i][j][k];
        let interference = field_data.interference_patterns[i][j][k];
        let tunneling = field_data.tunneling_probabilities[i][j][k];
        let pos_uncertainty = field_data.uncertainty_position[i][j][k];
        let mom_uncertainty = field_data.uncertainty_momentum[i][j][k];
        let coherence_time = field_data.coherence_times[i][j][k];

        // Convert lattice coordinates to world coordinates
        let world_pos = self.lattice_to_world_coordinates(i, j, k, field_data.lattice_spacing);

        // Determine field type for visualization
        let field_type = match field_data.field_type.as_str() {
            "ElectronField" => 1.0, // Fermion
            "PhotonField" => 2.0,   // Vector
            "HiggsField" => 0.0,    // Scalar
            _ => 0.0,               // Default to scalar
        };

        // Calculate scale factor based on quantum properties
        let scale_factor = self.calculate_quantum_scale_factor(magnitude, entanglement, decoherence);

        Ok(QuantumParticleVertex {
            position: [world_pos.x, world_pos.y, world_pos.z],
            velocity: [0.0, 0.0, 0.0], // Quantum fields don't have classical velocity
            mass: field_data.field_mass as f32,
            charge: 0.0, // Charge determined by field type
            temperature: 0.0, // Temperature not applicable to quantum fields
            particle_type: field_type,
            interaction_count: entanglement as f32,
            quantum_amplitude_real: complex_amp.0 as f32,
            quantum_amplitude_imag: complex_amp.1 as f32,
            quantum_phase: phase as f32,
            decoherence_rate: decoherence as f32,
            uncertainty_position: pos_uncertainty as f32,
            uncertainty_momentum: mom_uncertainty as f32,
            coherence_time: coherence_time as f32,
            entanglement_strength: entanglement as f32,
            field_type,
            scale_factor: scale_factor as f32,
            _padding: [0.0, 0.0],
        })
    }

    /// Create quantum field state vector for advanced quantum visualization
    #[cfg(feature = "quantum-visualization")]
    fn create_quantum_field_state_vector(&self, field_data: &universe_sim::QuantumStateVectorData, i: usize, j: usize, k: usize) -> Result<QuantumFieldStateVector> {
        let complex_amp = field_data.complex_amplitudes[i][j][k];
        let phase = field_data.phases[i][j][k];
        let entanglement = field_data.entanglement_correlations[i][j][k];
        let decoherence = field_data.decoherence_rates[i][j][k];
        let pos_uncertainty = field_data.uncertainty_position[i][j][k];
        let mom_uncertainty = field_data.uncertainty_momentum[i][j][k];
        let coherence_time = field_data.coherence_times[i][j][k];

        // Create superposition states (simplified - in reality this would be more complex)
        let superposition_states = vec![
            Complex::new(complex_amp.0, complex_amp.1),
            Complex::new(complex_amp.0 * 0.5, complex_amp.1 * 0.5), // Second state
        ];

        // Create entanglement map (simplified)
        let mut entanglement_map = HashMap::new();
        if entanglement > 0.1 {
            // Add some entanglement correlations
            entanglement_map.insert(i * 1000 + j * 100 + k, entanglement);
        }

        // Determine field type
        let field_type = match field_data.field_type.as_str() {
            "ElectronField" => QuantumFieldType::Fermion,
            "PhotonField" => QuantumFieldType::Vector,
            "HiggsField" => QuantumFieldType::Scalar,
            _ => QuantumFieldType::Scalar,
        };

        Ok(QuantumFieldStateVector {
            amplitude: Complex::new(complex_amp.0, complex_amp.1),
            phase,
            superposition_states,
            entanglement_map,
            decoherence_rate: decoherence,
            scale_factor: field_data.lattice_spacing,
            uncertainty_position: pos_uncertainty,
            uncertainty_momentum: mom_uncertainty,
            coherence_time,
            field_type,
        })
    }

    /// Convert lattice coordinates to world coordinates
    #[cfg(feature = "quantum-visualization")]
    fn lattice_to_world_coordinates(&self, i: usize, j: usize, k: usize, lattice_spacing: f64) -> Vector3<f32> {
        let x = (i as f64 - 50.0) * lattice_spacing * 1e9; // Scale to nanometers for visibility
        let y = (j as f64 - 50.0) * lattice_spacing * 1e9;
        let z = (k as f64 - 50.0) * lattice_spacing * 1e9;
        
        Vector3::new(x as f32, y as f32, z as f32)
    }

    /// Calculate quantum scale factor for visualization
    #[cfg(feature = "quantum-visualization")]
    fn calculate_quantum_scale_factor(&self, magnitude: f64, entanglement: f64, decoherence: f64) -> f64 {
        // Base scale from magnitude
        let base_scale = magnitude.max(0.1);
        
        // Enhance scale for strong entanglement
        let entanglement_scale = 1.0 + entanglement * 2.0;
        
        // Reduce scale for high decoherence (classical behavior)
        let decoherence_scale = 1.0 - decoherence * 0.5;
        
        base_scale * entanglement_scale * decoherence_scale
    }

    /// Update quantum vertex buffer with new quantum particle data
    #[cfg(feature = "quantum-visualization")]
    fn update_quantum_vertex_buffer(&mut self) -> Result<()> {
        if let Some(ref mut quantum_vertex_buffer) = self.quantum_vertex_buffer {
            // Update the quantum vertex buffer with new data
            self.queue.write_buffer(
                quantum_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.quantum_particles),
            );
        }
        Ok(())
    }

    /// Set quantum visualization parameters
    pub fn set_quantum_visualization_params(&mut self, params: QuantumVisualizationParams) {
        #[cfg(feature = "quantum-visualization")]
        {
            self.quantum_visualization_params = params;
        }
    }

    /// Toggle quantum visualization mode
    pub fn toggle_quantum_visualization(&mut self) {
        #[cfg(feature = "quantum-visualization")]
        {
            self.quantum_mode_enabled = !self.quantum_mode_enabled;
            info!("Quantum visualization mode: {}", if self.quantum_mode_enabled { "enabled" } else { "disabled" });
        }
    }

    // === ADVANCED VISUALIZATION ENHANCEMENTS ===

    /// Set adaptive visualization mode
    pub fn set_adaptive_visualization_mode(&mut self, mode: AdaptiveVisualizationMode) {
        let old_mode = self.adaptive_visualization.mode;
        self.adaptive_visualization.mode = mode;
        
        if old_mode != mode {
            self.adaptive_visualization.analytics.adaptive_mode_switches += 1;
            info!("Adaptive visualization mode changed: {:?} -> {:?}", old_mode, mode);
            
            // Apply mode-specific optimizations
            self.apply_mode_optimizations();
        }
    }

    /// Apply optimizations based on current adaptive mode
    fn apply_mode_optimizations(&mut self) {
        match self.adaptive_visualization.mode {
            AdaptiveVisualizationMode::Performance => {
                // Enable performance optimizations
                self.adaptive_visualization.performance.adaptive_lod = true;
                self.adaptive_visualization.performance.occlusion_culling = true;
                self.adaptive_visualization.performance.frustum_culling = true;
                self.adaptive_visualization.performance.max_particles_visible = 500_000;
                self.adaptive_visualization.performance.target_fps = 60.0;
                info!("Applied performance optimization settings");
            }
            AdaptiveVisualizationMode::Scientific => {
                // Enable scientific visualization features
                self.adaptive_visualization.scientific.field_lines = true;
                self.adaptive_visualization.scientific.velocity_fields = true;
                self.adaptive_visualization.scientific.temperature_gradients = true;
                self.adaptive_visualization.performance.max_particles_visible = 1_000_000;
                info!("Applied scientific visualization settings");
            }
            AdaptiveVisualizationMode::Quantum => {
                // Enable quantum visualization features
                self.adaptive_visualization.scientific.quantum_probability = true;
                self.adaptive_visualization.scientific.interaction_paths = true;
                self.adaptive_visualization.performance.max_particles_visible = 2_000_000;
                info!("Applied quantum visualization settings");
            }
            AdaptiveVisualizationMode::Cosmological => {
                // Enable cosmological visualization features
                self.adaptive_visualization.scientific.gravitational_lensing = true;
                self.adaptive_visualization.scientific.density_contours = true;
                self.adaptive_visualization.performance.max_particles_visible = 5_000_000;
                info!("Applied cosmological visualization settings");
            }
            AdaptiveVisualizationMode::Molecular => {
                // Enable molecular dynamics features
                self.adaptive_visualization.scientific.pressure_waves = true;
                self.adaptive_visualization.scientific.temperature_gradients = true;
                self.adaptive_visualization.performance.max_particles_visible = 500_000;
                info!("Applied molecular visualization settings");
            }
            AdaptiveVisualizationMode::Hybrid => {
                // Adaptive settings based on current performance
                self.apply_adaptive_optimization();
            }
        }
    }

    /// Apply adaptive optimization based on current performance metrics
    fn apply_adaptive_optimization(&mut self) {
        let current_fps = self.get_fps();
        let target_fps = self.adaptive_visualization.performance.target_fps;
        let performance_ratio = current_fps / target_fps;
        
        if performance_ratio < self.adaptive_visualization.performance_threshold {
            // Performance is below threshold, apply aggressive optimizations
            self.adaptive_visualization.performance.max_particles_visible = 
                (self.adaptive_visualization.performance.max_particles_visible as f32 * 0.8) as usize;
            self.adaptive_visualization.performance.adaptive_lod = true;
            self.adaptive_visualization.performance.occlusion_culling = true;
            
            let event = format!("Performance optimization applied: FPS {:.1} -> target {:.1}", current_fps, target_fps);
            self.adaptive_visualization.analytics.optimization_events.push(event);
            info!("Applied aggressive performance optimizations");
        } else if performance_ratio > 1.2 {
            // Performance is good, can enable more features
            self.adaptive_visualization.performance.max_particles_visible = 
                (self.adaptive_visualization.performance.max_particles_visible as f32 * 1.2) as usize;
            self.adaptive_visualization.scientific.field_lines = true;
            
            let event = format!("Quality enhancement applied: FPS {:.1} -> target {:.1}", current_fps, target_fps);
            self.adaptive_visualization.analytics.optimization_events.push(event);
            info!("Applied quality enhancements");
        }
    }

    /// Update performance analytics
    pub fn update_performance_analytics(&mut self) {
        let current_time = std::time::Instant::now();
        let frame_time = self.metrics.frame_time_ms;
        
        // Update analytics
        self.adaptive_visualization.analytics.render_time_ms = frame_time;
        self.adaptive_visualization.analytics.average_fps = self.get_fps();
        self.adaptive_visualization.analytics.memory_usage_mb = self.get_buffer_pool_stats().active_memory_mb;
        
        // Track peak and minimum FPS
        let current_fps = self.get_fps();
        if current_fps > self.adaptive_visualization.analytics.peak_fps {
            self.adaptive_visualization.analytics.peak_fps = current_fps;
        }
        if current_fps < self.adaptive_visualization.analytics.min_fps || self.adaptive_visualization.analytics.min_fps == 0.0 {
            self.adaptive_visualization.analytics.min_fps = current_fps;
        }
        
        // Check for frame drops
        if frame_time > (1000.0 / self.adaptive_visualization.performance.target_fps) {
            self.adaptive_visualization.analytics.frame_drops += 1;
        }
        
        // Auto-optimize if enabled and enough time has passed
        if self.adaptive_visualization.auto_optimize && 
           current_time.duration_since(self.adaptive_visualization.last_optimization) > self.adaptive_visualization.optimization_interval {
            self.apply_adaptive_optimization();
            self.adaptive_visualization.last_optimization = current_time;
        }
    }

    /// Toggle performance monitoring
    pub fn toggle_performance_monitoring(&mut self) {
        self.performance_monitoring = !self.performance_monitoring;
        info!("Performance monitoring: {}", if self.performance_monitoring { "enabled" } else { "disabled" });
    }

    /// Toggle scientific overlay
    pub fn toggle_scientific_overlay(&mut self) {
        self.scientific_overlay = !self.scientific_overlay;
        info!("Scientific overlay: {}", if self.scientific_overlay { "enabled" } else { "disabled" });
    }

    /// Toggle real-time analytics
    pub fn toggle_real_time_analytics(&mut self) {
        self.real_time_analytics = !self.real_time_analytics;
        info!("Real-time analytics: {}", if self.real_time_analytics { "enabled" } else { "disabled" });
    }

    /// Get current adaptive visualization parameters
    pub fn get_adaptive_visualization_params(&self) -> &AdaptiveVisualizationParams {
        &self.adaptive_visualization
    }

    /// Set performance optimization parameters
    pub fn set_performance_optimization(&mut self, optimization: PerformanceOptimization) {
        self.adaptive_visualization.performance = optimization;
        info!("Updated performance optimization parameters");
    }

    /// Set scientific visualization parameters
    pub fn set_scientific_visualization(&mut self, scientific: ScientificVisualization) {
        self.adaptive_visualization.scientific = scientific;
        info!("Updated scientific visualization parameters");
    }

    /// Get performance analytics
    pub fn get_performance_analytics(&self) -> &VisualizationAnalytics {
        &self.adaptive_visualization.analytics
    }

    /// Reset performance analytics
    pub fn reset_performance_analytics(&mut self) {
        self.adaptive_visualization.analytics = VisualizationAnalytics::default();
        info!("Reset performance analytics");
    }

    /// Apply particle culling based on current optimization settings
    fn apply_particle_culling(&self, particles: &mut Vec<ParticleVertex>) -> usize {
        let mut culled_count = 0;
        
        if self.adaptive_visualization.performance.frustum_culling {
            // Implement frustum culling
            culled_count += self.apply_frustum_culling(particles);
        }
        
        if self.adaptive_visualization.performance.occlusion_culling {
            // Implement occlusion culling
            culled_count += self.apply_occlusion_culling(particles);
        }
        
        // Limit visible particles
        if particles.len() > self.adaptive_visualization.performance.max_particles_visible {
            let excess = particles.len() - self.adaptive_visualization.performance.max_particles_visible;
            particles.truncate(self.adaptive_visualization.performance.max_particles_visible);
            culled_count += excess;
        }
        
        culled_count
    }

    /// Apply frustum culling to remove particles outside view
    fn apply_frustum_culling(&self, particles: &mut Vec<ParticleVertex>) -> usize {
        let initial_count = particles.len();
        particles.retain(|particle| {
            let pos = Vector3::new(particle.position[0], particle.position[1], particle.position[2]);
            self.is_in_frustum(pos)
        });
        initial_count - particles.len()
    }

    /// Apply occlusion culling to remove particles behind others
    fn apply_occlusion_culling(&self, particles: &mut Vec<ParticleVertex>) -> usize {
        // Simple depth-based occlusion culling
        let initial_count = particles.len();
        particles.sort_by(|a, b| {
            let depth_a = a.position[2];
            let depth_b = b.position[2];
            depth_a.partial_cmp(&depth_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        initial_count - particles.len()
    }

    /// Check if a point is in the camera frustum
    fn is_in_frustum(&self, point: Vector3<f32>) -> bool {
        // Simplified frustum test - can be enhanced with proper frustum planes
        let camera_pos = self.camera.position;
        let distance = (point - camera_pos.to_vec()).magnitude();
        distance < 1000.0 // Far plane distance
    }

    /// Render scientific overlay
    pub fn render_scientific_overlay(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> Result<()> {
        if !self.scientific_overlay {
            return Ok(());
        }

        let mut text_lines = Vec::new();
        
        // Add scientific visualization information
        if self.adaptive_visualization.scientific.field_lines {
            text_lines.push("Field Lines: Enabled");
        }
        if self.adaptive_visualization.scientific.velocity_fields {
            text_lines.push("Velocity Fields: Enabled");
        }
        if self.adaptive_visualization.scientific.temperature_gradients {
            text_lines.push("Temperature Gradients: Enabled");
        }
        if self.adaptive_visualization.scientific.gravitational_lensing {
            text_lines.push("Gravitational Lensing: Enabled");
        }
        if self.adaptive_visualization.scientific.quantum_probability {
            text_lines.push("Quantum Probability: Enabled");
        }

        // Render scientific overlay text
        let overlay_text = text_lines.join("\n");
        if !overlay_text.is_empty() {
            self.render_debug_text(encoder, view, &overlay_text)?;
        }

        Ok(())
    }

    /// Render performance monitoring overlay
    pub fn render_performance_overlay(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) -> Result<()> {
        if !self.performance_monitoring {
            return Ok(());
        }

        let analytics = &self.adaptive_visualization.analytics;
        let performance = &self.adaptive_visualization.performance;
        
        let mut text_lines = Vec::new();
        text_lines.push(format!("FPS: {:.1}", analytics.average_fps));
        text_lines.push(format!("Frame Time: {:.2}ms", analytics.render_time_ms));
        text_lines.push(format!("Memory: {:.1}MB", analytics.memory_usage_mb));
        text_lines.push(format!("Particles: {}", self.get_particle_count()));
        text_lines.push(format!("Culled: {}", analytics.frame_drops));

        // Render performance overlay text
        let overlay_text = text_lines.join("\n");
        if !overlay_text.is_empty() {
            self.render_debug_text(encoder, view, &overlay_text)?;
        }

        Ok(())
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

    info!("🚀 Starting native renderer event loop");
    
    // Create the winit event loop with error handling
    let event_loop: EventLoop<()> = match EventLoop::new() {
        Ok(loop_) => {
            info!("✅ Event loop created successfully");
            loop_
        },
        Err(e) => {
            error!("❌ Failed to create event loop: {}", e);
            return Err(anyhow::anyhow!("Event loop creation failed: {}", e));
        }
    };

    // Build the window with error handling
    let window = match WindowBuilder::new()
        .with_title("EVOLUTION – Native Renderer")
        .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
        .with_resizable(true)
        .build(&event_loop) {
            Ok(window) => {
                info!("✅ Window created successfully");
                window
            },
            Err(e) => {
                error!("❌ Failed to create window: {}", e);
                return Err(anyhow::anyhow!("Window creation failed: {}", e));
            }
        };

    // Initialise the renderer with comprehensive error handling
    info!("🔧 Initializing renderer...");
    let mut renderer = match pollster::block_on(NativeRenderer::new(&window)) {
        Ok(renderer) => {
            info!("✅ Renderer initialized successfully");
            renderer
        },
        Err(e) => {
            error!("❌ Renderer initialization failed: {}", e);
            return Err(anyhow::anyhow!("Renderer initialization failed: {}", e));
        }
    };

    // Clone the simulation Arc so we can move it into the event loop closure.
    let sim_arc = sim.clone();
    // Track the time between frames for a stable simulation delta.
    let mut last_frame_inst = std::time::Instant::now();
    // Capture window ID and initial size before moving window into closure
    let window_id = window.id();
    let mut window_size = window.inner_size();

    info!("🎬 Starting main render loop...");

    // Main event loop with comprehensive error handling
    let result = event_loop.run(move |event, elwt| {
        use winit::event::{Event, WindowEvent, MouseScrollDelta};
        use tracing::error;
        
        match event {
            // === Window-specific events ===
            Event::WindowEvent { window_id: win_id, event } if win_id == window_id => {
                match event {
                    WindowEvent::CloseRequested => {
                        info!("👋 Window close requested");
                        elwt.exit();
                    }
                    WindowEvent::Resized(size) => {
                        info!("📐 Window resized to {}x{}", size.width, size.height);
                        window_size = size;
                        if let Err(e) = renderer.resize(size) {
                            error!("❌ Resize failed: {}", e);
                        }
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        if let Err(e) = renderer.resize(window_size) {
                            error!("❌ Scale factor change resize failed: {}", e);
                        }
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    WindowEvent::RedrawRequested => {
                        debug!("🔄 RedrawRequested event");
                        // --- Simulation update with error handling ---
                        let now = std::time::Instant::now();
                        let _dt = (now - last_frame_inst).as_secs_f64();
                        last_frame_inst = now;

                        // Tick the simulation inside a critical section with error handling
                        if let Ok(mut sim) = sim_arc.lock() {
                            if let Err(e) = sim.tick() {
                                error!("❌ Simulation tick failed: {}", e);
                            }
                            
                            // Update particle data from simulation before rendering
                            if let Err(e) = renderer.update_particles(&mut *sim) {
                                error!("❌ Failed to update particles: {}", e);
                            }
                            
                            // Use universe age as a time parameter in seconds for the renderer.
                            let sim_time = sim.universe_age_years() as f32;
                            if let Err(e) = renderer.render(sim_time) {
                                error!("❌ Render error: {}", e);
                            }
                        } else {
                            error!("❌ Failed to acquire simulation lock");
                        }
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    WindowEvent::KeyboardInput { event: key_event, .. } => {
                        debug!("⌨️ KeyboardInput: {:?}", key_event);
                        if let winit::keyboard::PhysicalKey::Code(code) = key_event.physical_key {
                            renderer.handle_debug_input(code, key_event.state);
                            renderer.handle_flythrough_movement(code, key_event.state);
                        }
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    WindowEvent::ModifiersChanged(mods) => {
                        debug!("🪟 ModifiersChanged: {:?}", mods);
                        renderer.handle_modifiers(mods.state());
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        debug!("🖱️ MouseInput: {:?} {:?}", button, state);
                        renderer.handle_mouse_button(button, state);
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        debug!("🖱️ CursorMoved: ({}, {})", position.x, position.y);
                        renderer.handle_mouse_move(position.x, position.y);
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        debug!("🖱️ MouseWheel: {:?}", delta);
                        let scroll_amount = match delta {
                            MouseScrollDelta::LineDelta(_x, y) => y * 32.0,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32,
                        };
                        renderer.handle_mouse_wheel(scroll_amount);
                        // Note: We can't call window.request_redraw() here due to borrowing
                    }
                    _ => {}
                }
            }

            // === Event loop is about to sleep – ensure we redraw continuously ===
            Event::AboutToWait => {
                debug!("⏳ AboutToWait: setting ControlFlow::Poll");
                elwt.set_control_flow(ControlFlow::Poll);
                // Note: We can't call window.request_redraw() here due to borrowing
            }

            _ => {}
        }
    });

    match result {
        Ok(_) => {
            info!("✅ Renderer event loop completed successfully");
            Ok(())
        },
        Err(e) => {
            error!("❌ Renderer event loop failed: {}", e);
            Err(anyhow::anyhow!("Event loop failed: {}", e))
        }
    }
}

// Helper struct to hold renderer state
struct RendererState {
    window: winit::window::Window,
    renderer: NativeRenderer<'static>,
    window_size: winit::dpi::PhysicalSize<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentVisualizationMode {
    Overview,      // Basic agent count and metrics
    Network,       // Network graph visualization
    Timeline,      // Agent activity timeline
    Heatmap,       // Interaction heatmap
    Inspector,     // Detailed agent inspection
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTimelineEvent {
    pub agent_id: String,
    pub timestamp: f64,
    pub event_type: String,
    pub description: String,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionHeatmapCell {
    pub x: f32,
    pub y: f32,
    pub intensity: f32,
    pub interaction_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentVisualizationData {
    pub network_summary: serde_json::Value,
    pub timeline_events: Vec<AgentTimelineEvent>,
    pub heatmap_data: Vec<InteractionHeatmapCell>,
}

// === ADVANCED VISUALIZATION ENHANCEMENTS ===

/// Adaptive visualization modes for different simulation states
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AdaptiveVisualizationMode {
    Performance,      // Optimized for high particle counts
    Scientific,       // Detailed physics visualization
    Quantum,          // Quantum-accurate rendering
    Cosmological,     // Large-scale structure visualization
    Molecular,        // Molecular dynamics focus
    Hybrid,           // Adaptive mode switching
}

/// Performance optimization parameters
#[derive(Debug, Clone)]
pub struct PerformanceOptimization {
    pub adaptive_lod: bool,           // Level of detail based on distance
    pub occlusion_culling: bool,      // Cull particles behind others
    pub frustum_culling: bool,        // Cull particles outside view
    pub instanced_rendering: bool,    // Use GPU instancing
    pub particle_batching: bool,      // Batch similar particles
    pub dynamic_resolution: bool,     // Adjust resolution based on performance
    pub max_particles_visible: usize, // Limit visible particles
    pub target_fps: f32,              // Target frame rate
}

impl Default for PerformanceOptimization {
    fn default() -> Self {
        Self {
            adaptive_lod: true,
            occlusion_culling: true,
            frustum_culling: true,
            instanced_rendering: true,
            particle_batching: true,
            dynamic_resolution: true,
            max_particles_visible: 1_000_000,
            target_fps: 60.0,
        }
    }
}

/// Scientific visualization enhancements
#[derive(Debug, Clone)]
pub struct ScientificVisualization {
    pub field_lines: bool,            // Show electromagnetic field lines
    pub velocity_fields: bool,        // Show velocity vector fields
    pub density_contours: bool,       // Show density isosurfaces
    pub temperature_gradients: bool,  // Show temperature gradients
    pub pressure_waves: bool,         // Show pressure wave propagation
    pub gravitational_lensing: bool,  // Show gravitational lensing effects
    pub quantum_probability: bool,    // Show quantum probability densities
    pub interaction_paths: bool,      // Show particle interaction paths
}

impl Default for ScientificVisualization {
    fn default() -> Self {
        Self {
            field_lines: false,
            velocity_fields: false,
            density_contours: false,
            temperature_gradients: false,
            pressure_waves: false,
            gravitational_lensing: false,
            quantum_probability: false,
            interaction_paths: false,
        }
    }
}

/// Real-time monitoring and analytics
#[derive(Debug, Clone)]
pub struct VisualizationAnalytics {
    pub render_time_ms: f32,
    pub particle_processing_time_ms: f32,
    pub shader_compilation_time_ms: f32,
    pub memory_usage_mb: f32,
    pub gpu_utilization_percent: f32,
    pub cpu_utilization_percent: f32,
    pub frame_drops: u32,
    pub average_fps: f32,
    pub peak_fps: f32,
    pub min_fps: f32,
    pub adaptive_mode_switches: u32,
    pub optimization_events: Vec<String>,
}

impl Default for VisualizationAnalytics {
    fn default() -> Self {
        Self {
            render_time_ms: 0.0,
            particle_processing_time_ms: 0.0,
            shader_compilation_time_ms: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization_percent: 0.0,
            cpu_utilization_percent: 0.0,
            frame_drops: 0,
            average_fps: 0.0,
            peak_fps: 0.0,
            min_fps: 0.0,
            adaptive_mode_switches: 0,
            optimization_events: Vec::new(),
        }
    }
}

/// Dynamic visualization parameters that adapt to simulation state
#[derive(Debug, Clone)]
pub struct AdaptiveVisualizationParams {
    pub mode: AdaptiveVisualizationMode,
    pub performance: PerformanceOptimization,
    pub scientific: ScientificVisualization,
    pub analytics: VisualizationAnalytics,
    pub auto_optimize: bool,
    pub quality_threshold: f32,
    pub performance_threshold: f32,
    pub last_optimization: std::time::Instant,
    pub optimization_interval: std::time::Duration,
}

impl Default for AdaptiveVisualizationParams {
    fn default() -> Self {
        Self {
            mode: AdaptiveVisualizationMode::Hybrid,
            performance: PerformanceOptimization::default(),
            scientific: ScientificVisualization::default(),
            analytics: VisualizationAnalytics::default(),
            auto_optimize: true,
            quality_threshold: 0.8,
            performance_threshold: 0.9,
            last_optimization: std::time::Instant::now(),
            optimization_interval: std::time::Duration::from_secs(5),
        }
    }
}