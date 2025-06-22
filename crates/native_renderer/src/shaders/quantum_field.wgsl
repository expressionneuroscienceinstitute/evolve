// Quantum-Accurate Field Visualization Shader
// Implements quantum superposition, entanglement, and decoherence visualization
// Based on quantum field theory and quantum mechanics principles

struct Uniforms {
    view_proj: mat4x4<f32>,
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
    
    // Quantum field data
    @location(7) quantum_amplitude_real: f32,    // Real part of quantum amplitude
    @location(8) quantum_amplitude_imag: f32,    // Imaginary part of quantum amplitude
    @location(9) quantum_phase: f32,             // Quantum phase
    @location(10) decoherence_rate: f32,         // Rate of quantum-to-classical transition
    @location(11) uncertainty_position: f32,     // Position uncertainty
    @location(12) uncertainty_momentum: f32,     // Momentum uncertainty
    @location(13) coherence_time: f32,           // Quantum coherence lifetime
    @location(14) entanglement_strength: f32,    // Strength of entanglement
    @location(15) field_type: f32,               // Quantum field type
    @location(16) scale_factor: f32,             // Multi-scale quantum effects
    
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) size: f32,
    @location(3) intensity: f32,
    @location(4) particle_class: f32,
    // Quantum-specific outputs
    @location(5) quantum_amplitude: vec2<f32>,   // Complex amplitude
    @location(6) quantum_phase: f32,             // Quantum phase
    @location(7) decoherence_factor: f32,        // Decoherence effect
    @location(8) superposition_factor: f32,      // Superposition visibility
    @location(9) entanglement_factor: f32,       // Entanglement strength
    @location(10) interference_pattern: f32,     // Interference visibility
    @location(11) tunneling_probability: f32,    // Tunneling probability
}

// Quantum constants for accurate calculations
const PLANCK_CONSTANT: f32 = 6.626e-34;
const REDUCED_PLANCK_CONSTANT: f32 = 1.055e-34;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const BOLTZMANN_CONSTANT: f32 = 1.381e-23;
const ELECTRON_MASS: f32 = 9.109e-31;

// Quantum field type constants
const FIELD_TYPE_SCALAR: f32 = 0.0;
const FIELD_TYPE_FERMION: f32 = 1.0;
const FIELD_TYPE_VECTOR: f32 = 2.0;
const FIELD_TYPE_TENSOR: f32 = 3.0;

// Quantum visualization mode constants
const QUANTUM_MODE_CLASSICAL: f32 = 0.0;
const QUANTUM_MODE_SUPERPOSITION: f32 = 1.0;
const QUANTUM_MODE_ENTANGLEMENT: f32 = 2.0;
const QUANTUM_MODE_DECOHERENCE: f32 = 3.0;
const QUANTUM_MODE_FIELD_FLUCTUATIONS: f32 = 4.0;
const QUANTUM_MODE_MULTI_SCALE: f32 = 5.0;
const QUANTUM_MODE_INTERFERENCE: f32 = 6.0;
const QUANTUM_MODE_TUNNELING: f32 = 7.0;

/// Calculate quantum superposition visualization
fn calculate_superposition_visualization(amplitude: vec2<f32>, phase: f32, threshold: f32) -> vec3<f32> {
    let amplitude_magnitude = length(amplitude);
    
    if (amplitude_magnitude < threshold) {
        return vec3<f32>(0.1, 0.1, 0.1); // Dark for collapsed states
    }
    
    // Create interference pattern based on phase
    let interference = sin(phase * 10.0) * 0.5 + 0.5;
    
    // Color based on superposition strength
    let superposition_strength = amplitude_magnitude;
    
    if (superposition_strength < 0.3) {
        // Weak superposition: blue tones
        return vec3<f32>(0.0, 0.0, 0.5 + interference * 0.5);
    } else if (superposition_strength < 0.7) {
        // Medium superposition: green tones
        return vec3<f32>(0.0, 0.5 + interference * 0.5, 0.0);
    } else {
        // Strong superposition: white with rainbow interference
        let hue = phase / (2.0 * PI);
        return hsv_to_rgb(hue, 0.8, 0.8 + interference * 0.2);
    }
}

/// Calculate quantum entanglement visualization
fn calculate_entanglement_visualization(entanglement_strength: f32, threshold: f32) -> vec3<f32> {
    if (entanglement_strength < threshold) {
        return vec3<f32>(0.2, 0.2, 0.2); // Dark for unentangled states
    }
    
    // Create correlated color patterns for entanglement
    let correlation = sin(entanglement_strength * 100.0 + uniforms.time) * 0.5 + 0.5;
    
    // Purple-red gradient for entanglement (Bell state colors)
    return vec3<f32>(0.8 + correlation * 0.2, 0.0, 0.8 + correlation * 0.2);
}

/// Calculate quantum decoherence visualization
fn calculate_decoherence_visualization(decoherence_rate: f32, coherence_time: f32) -> vec3<f32> {
    let decoherence_factor = exp(-decoherence_rate * uniforms.time);
    let coherence_factor = exp(-uniforms.time / coherence_time);
    
    let combined_factor = decoherence_factor * coherence_factor;
    
    // Transition from quantum (blue) to classical (red) as decoherence increases
    let quantum_color = vec3<f32>(0.0, 0.0, 1.0); // Blue for quantum
    let classical_color = vec3<f32>(1.0, 0.0, 0.0); // Red for classical
    
    return mix(quantum_color, classical_color, 1.0 - combined_factor);
}

/// Calculate quantum field fluctuations visualization
fn calculate_field_fluctuations_visualization(amplitude: vec2<f32>, fluctuation_amplitude: f32) -> vec3<f32> {
    let base_amplitude = length(amplitude);
    let fluctuation = sin(uniforms.time * 10.0 + base_amplitude * 100.0) * fluctuation_amplitude;
    
    // Vacuum fluctuations: subtle color variations
    let base_color = vec3<f32>(0.1, 0.1, 0.2); // Dark blue for vacuum
    let fluctuation_color = vec3<f32>(0.3, 0.3, 0.8); // Light blue for fluctuations
    
    return mix(base_color, fluctuation_color, fluctuation);
}

/// Calculate quantum interference patterns
fn calculate_interference_patterns(amplitude: vec2<f32>, phase: f32, visibility: f32) -> f32 {
    let interference = sin(phase * 20.0 + uniforms.time * 5.0) * 0.5 + 0.5;
    let amplitude_factor = length(amplitude);
    
    return interference * amplitude_factor * visibility;
}

/// Calculate quantum tunneling probability visualization
fn calculate_tunneling_visualization(mass: f32, velocity: f32, scale: f32) -> vec3<f32> {
    // Simplified tunneling probability calculation
    let energy = 0.5 * mass * velocity * velocity;
    let tunneling_probability = exp(-sqrt(mass) * scale);
    
    // Green for high tunneling probability, red for low
    let tunneling_color = vec3<f32>(0.0, tunneling_probability, 0.0);
    let barrier_color = vec3<f32>(tunneling_probability, 0.0, 0.0);
    
    return mix(barrier_color, tunneling_color, tunneling_probability);
}

/// Calculate multi-scale quantum-classical transition
fn calculate_multi_scale_transition(scale_factor: f32, transition_scale: f32) -> f32 {
    let scale_ratio = scale_factor / transition_scale;
    let quantum_factor = exp(-scale_ratio);
    let classical_factor = 1.0 - quantum_factor;
    
    return mix(quantum_factor, classical_factor, scale_ratio);
}

/// Convert HSV to RGB for quantum interference colors
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    if (h < 1.0/6.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h < 2.0/6.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h < 3.0/6.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h < 4.0/6.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h < 5.0/6.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m, m, m);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Create quantum field geometry (more complex than simple particles)
    let quad_index = input.vertex_index % 6u;
    var offset = vec2<f32>(0.0, 0.0);
    
    // Define quantum field quad vertices
    switch quad_index {
        case 0u: { offset = vec2<f32>(-0.02, -0.02); out.uv = vec2<f32>(0.0, 0.0); }
        case 1u: { offset = vec2<f32>( 0.02, -0.02); out.uv = vec2<f32>(1.0, 0.0); }
        case 2u: { offset = vec2<f32>( 0.02,  0.02); out.uv = vec2<f32>(1.0, 1.0); }
        case 3u: { offset = vec2<f32>(-0.02, -0.02); out.uv = vec2<f32>(0.0, 0.0); }
        case 4u: { offset = vec2<f32>( 0.02,  0.02); out.uv = vec2<f32>(1.0, 1.0); }
        default: { offset = vec2<f32>(-0.02,  0.02); out.uv = vec2<f32>(0.0, 1.0); }
    }
    
    // Calculate quantum field size based on uncertainty principle
    let position_uncertainty = input.uncertainty_position;
    let momentum_uncertainty = input.uncertainty_momentum;
    let quantum_size = sqrt(position_uncertainty * momentum_uncertainty) * 1000.0; // Scale for visibility
    
    // Scale offset by quantum field size
    offset = offset * quantum_size;
    
    let world_pos = vec4<f32>(input.position, 1.0);
    let clip_pos = uniforms.view_proj * world_pos;
    
    // Add quantum field offset in screen space
    out.clip_position = vec4<f32>(clip_pos.xy + offset * clip_pos.w, clip_pos.z, clip_pos.w);
    
    // Calculate quantum field properties
    let quantum_amplitude = vec2<f32>(input.quantum_amplitude_real, input.quantum_amplitude_imag);
    let amplitude_magnitude = length(quantum_amplitude);
    
    // Quantum visualization based on mode
    if (uniforms.quantum_mode == QUANTUM_MODE_SUPERPOSITION) {
        out.color = calculate_superposition_visualization(quantum_amplitude, input.quantum_phase, uniforms.superposition_threshold);
    } else if (uniforms.quantum_mode == QUANTUM_MODE_ENTANGLEMENT) {
        out.color = calculate_entanglement_visualization(input.entanglement_strength, uniforms.entanglement_threshold);
    } else if (uniforms.quantum_mode == QUANTUM_MODE_DECOHERENCE) {
        out.color = calculate_decoherence_visualization(input.decoherence_rate, input.coherence_time);
    } else if (uniforms.quantum_mode == QUANTUM_MODE_FIELD_FLUCTUATIONS) {
        out.color = calculate_field_fluctuations_visualization(quantum_amplitude, uniforms.field_fluctuation_amplitude);
    } else if (uniforms.quantum_mode == QUANTUM_MODE_INTERFERENCE) {
        let interference = calculate_interference_patterns(quantum_amplitude, input.quantum_phase, uniforms.interference_visibility);
        out.color = vec3<f32>(interference, interference, interference);
    } else if (uniforms.quantum_mode == QUANTUM_MODE_TUNNELING) {
        out.color = calculate_tunneling_visualization(input.mass, length(input.velocity), uniforms.tunneling_probability_scale);
    } else if (uniforms.quantum_mode == QUANTUM_MODE_MULTI_SCALE) {
        let transition_factor = calculate_multi_scale_transition(input.scale_factor, uniforms.multi_scale_transition);
        let quantum_color = vec3<f32>(0.0, 0.0, 1.0);
        let classical_color = vec3<f32>(1.0, 0.0, 0.0);
        out.color = mix(quantum_color, classical_color, transition_factor);
    } else {
        // Classical mode: fallback to traditional particle visualization
        out.color = vec3<f32>(amplitude_magnitude, 0.5, 1.0 - amplitude_magnitude);
    }
    
    // Set quantum-specific outputs
    out.quantum_amplitude = quantum_amplitude;
    out.quantum_phase = input.quantum_phase;
    out.decoherence_factor = exp(-input.decoherence_rate * uniforms.time);
    out.superposition_factor = amplitude_magnitude;
    out.entanglement_factor = input.entanglement_strength;
    out.interference_pattern = calculate_interference_patterns(quantum_amplitude, input.quantum_phase, uniforms.interference_visibility);
    out.tunneling_probability = exp(-sqrt(input.mass) * uniforms.tunneling_probability_scale);
    
    // Calculate quantum field intensity
    let quantum_intensity = amplitude_magnitude * (1.0 - input.decoherence_rate);
    out.intensity = quantum_intensity;
    
    // Quantum field classification
    if (input.field_type == FIELD_TYPE_SCALAR) {
        out.particle_class = 0.0; // Scalar fields
    } else if (input.field_type == FIELD_TYPE_FERMION) {
        out.particle_class = 1.0; // Fermion fields
    } else if (input.field_type == FIELD_TYPE_VECTOR) {
        out.particle_class = 2.0; // Vector fields
    } else {
        out.particle_class = 3.0; // Tensor fields
    }
    
    out.size = quantum_size;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Create quantum field geometry
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(input.uv - center);
    
    // Quantum field rendering with interference patterns
    var alpha: f32;
    var brightness: f32;
    
    // Base quantum field shape
    if (input.particle_class == 0.0) {
        // Scalar fields: smooth, spherical
        alpha = smoothstep(1.0, 0.3, dist) * exp(-dist * 2.0);
        brightness = 2.0;
    } else if (input.particle_class == 1.0) {
        // Fermion fields: point-like with quantum fuzziness
        alpha = smoothstep(0.5, 0.0, dist);
        brightness = 1.5;
    } else if (input.particle_class == 2.0) {
        // Vector fields: directional with polarization
        let direction = sin(input.quantum_phase * 10.0) * 0.5 + 0.5;
        alpha = smoothstep(1.0, 0.7, dist) * direction;
        brightness = 2.5;
    } else {
        // Tensor fields: complex geometric patterns
        let tensor_pattern = sin(input.quantum_phase * 5.0) * cos(input.quantum_phase * 3.0);
        alpha = smoothstep(1.0, 0.5, dist) * (0.5 + tensor_pattern * 0.5);
        brightness = 3.0;
    }
    
    // Apply quantum interference effects
    let interference = input.interference_pattern;
    alpha *= (0.5 + interference * 0.5);
    brightness *= (1.0 + interference * 0.5);
    
    // Apply decoherence effects
    let decoherence = input.decoherence_factor;
    alpha *= decoherence;
    brightness *= decoherence;
    
    // Apply superposition effects
    let superposition = input.superposition_factor;
    if (superposition > 0.5) {
        // Strong superposition: add interference ripples
        let ripple = sin(dist * 20.0 + input.quantum_phase * 10.0) * 0.3 + 0.7;
        alpha *= ripple;
        brightness *= ripple;
    }
    
    // Apply entanglement effects
    let entanglement = input.entanglement_factor;
    if (entanglement > 0.5) {
        // Strong entanglement: add correlated patterns
        let correlation = sin(input.quantum_phase * 15.0 + uniforms.time * 2.0) * 0.5 + 0.5;
        alpha *= correlation;
        brightness *= (1.0 + correlation * 0.5);
    }
    
    // Apply tunneling effects
    let tunneling = input.tunneling_probability;
    if (tunneling > 0.1) {
        // Significant tunneling: add barrier penetration effects
        let penetration = smoothstep(0.8, 0.2, dist) * tunneling;
        alpha = max(alpha, penetration * 0.5);
        brightness += tunneling * 2.0;
    }
    
    // Apply quantum field fluctuations
    let fluctuation = sin(uniforms.time * 5.0 + input.quantum_phase * 20.0) * 0.2 + 0.8;
    alpha *= fluctuation;
    brightness *= fluctuation;
    
    // Apply relativistic effects
    brightness *= input.intensity;
    
    // Temporal quantum evolution
    let evolution = sin(uniforms.time * 3.0 + input.quantum_phase * 5.0) * 0.1 + 0.9;
    brightness *= evolution;
    
    // Apply filtering threshold
    if (brightness < uniforms.filter_threshold) {
        discard;
    }
    
    // Final quantum field color with quantum effects
    let final_color = input.color * brightness;
    let final_alpha = alpha;
    
    return vec4<f32>(final_color, final_alpha);
} 