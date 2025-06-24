// Scientific Overlay Shader
// Advanced field line, gradient, and isosurface rendering for scientific visualization
// Publication-quality overlays for electromagnetic fields, fluid flows, and scalar fields

struct ScientificUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    // Field line parameters
    field_line_density: f32,
    field_line_length: f32,
    integration_step: f32,
    max_integration_steps: f32,
    // Gradient visualization
    gradient_scale: f32,
    gradient_threshold: f32,
    show_gradient_magnitude: f32,
    gradient_color_scale: f32,
    // Isosurface parameters
    iso_value: f32,
    iso_smoothing: f32,
    iso_transparency: f32,
    normal_smoothing: f32,
    // Scientific accuracy
    scientific_color_mode: f32,  // 0=accurate, 1=enhanced, 2=publication
    vector_field_scale: f32,
    streamline_quality: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: ScientificUniforms;

@group(0) @binding(1)
var vector_field_texture: texture_3d<f32>;  // RGB = vector field, A = magnitude

@group(0) @binding(2)
var scalar_field_texture: texture_3d<f32>;  // Scalar field for isosurfaces

@group(0) @binding(3)
var field_sampler: sampler;

@group(0) @binding(4)
var gradient_texture: texture_3d<f32>;  // Precomputed gradients

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) field_direction: vec3<f32>,
    @location(2) field_magnitude: f32,
    @location(3) field_type: f32,  // 0=electric, 1=magnetic, 2=velocity, 3=pressure gradient
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) field_direction: vec3<f32>,
    @location(2) field_magnitude: f32,
    @location(3) field_type: f32,
    @location(4) line_progress: f32,
}

@vertex
fn vs_field_lines(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    out.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.world_position = input.position;
    out.field_direction = normalize(input.field_direction);
    out.field_magnitude = input.field_magnitude;
    out.field_type = input.field_type;
    
    // Calculate progress along field line for color/thickness variation
    out.line_progress = length(input.position - uniforms.camera_pos) / uniforms.field_line_length;
    
    return out;
}

// Scientific color mapping for different field types
fn get_field_color(field_type: f32, magnitude: f32, direction: vec3<f32>) -> vec3<f32> {
    let normalized_mag = clamp(magnitude * uniforms.vector_field_scale, 0.0, 1.0);
    
    if (field_type < 0.5) {
        // Electric field - blue to red gradient
        return vec3<f32>(normalized_mag, 0.0, 1.0 - normalized_mag);
    } else if (field_type < 1.5) {
        // Magnetic field - green circular representation
        let angle = atan2(direction.y, direction.x);
        let hue = (angle + 3.14159) / (2.0 * 3.14159);
        return hsv_to_rgb(hue, 1.0, normalized_mag);
    } else if (field_type < 2.5) {
        // Velocity field - Doppler shift coloring
        let velocity_dot = dot(direction, normalize(uniforms.camera_pos));
        if (velocity_dot > 0.0) {
            return vec3<f32>(1.0, 1.0 - velocity_dot, 1.0 - velocity_dot); // Red shift
        } else {
            return vec3<f32>(1.0 + velocity_dot, 1.0 + velocity_dot, 1.0); // Blue shift
        }
    } else {
        // Pressure gradient - temperature-like coloring
        return vec3<f32>(normalized_mag, normalized_mag * 0.5, 1.0 - normalized_mag);
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(fract(h * 6.0) * 2.0 - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    let h6 = h * 6.0;
    
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m);
}

// Advanced gradient calculation with higher-order accuracy
fn calculate_gradient_enhanced(position: vec3<f32>) -> vec3<f32> {
    let eps = 0.001; // Sampling offset
    
    // Use central differences for higher accuracy
    let dx = textureSample(scalar_field_texture, field_sampler, position + vec3<f32>(eps, 0.0, 0.0)).r -
             textureSample(scalar_field_texture, field_sampler, position - vec3<f32>(eps, 0.0, 0.0)).r;
    
    let dy = textureSample(scalar_field_texture, field_sampler, position + vec3<f32>(0.0, eps, 0.0)).r -
             textureSample(scalar_field_texture, field_sampler, position - vec3<f32>(0.0, eps, 0.0)).r;
    
    let dz = textureSample(scalar_field_texture, field_sampler, position + vec3<f32>(0.0, 0.0, eps)).r -
             textureSample(scalar_field_texture, field_sampler, position - vec3<f32>(0.0, 0.0, eps)).r;
    
    return vec3<f32>(dx, dy, dz) / (2.0 * eps);
}

// Marching cubes-inspired isosurface rendering
fn render_isosurface(position: vec3<f32>) -> vec4<f32> {
    let field_value = textureSample(scalar_field_texture, field_sampler, position).r;
    let distance_to_iso = abs(field_value - uniforms.iso_value);
    
    if (distance_to_iso > uniforms.iso_smoothing) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Calculate surface normal using gradient
    let normal = normalize(calculate_gradient_enhanced(position));
    
    // Simple lighting calculation
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let ndl = max(dot(normal, light_dir), 0.0);
    
    // Scientific color based on field value
    let color = get_field_color(3.0, field_value, normal);
    
    // Alpha based on distance to isosurface (smooth falloff)
    let alpha = uniforms.iso_transparency * (1.0 - distance_to_iso / uniforms.iso_smoothing);
    
    return vec4<f32>(color * ndl, alpha);
}

@fragment
fn fs_field_lines(input: VertexOutput) -> @location(0) vec4<f32> {
    // Get field color based on type and magnitude
    let base_color = get_field_color(input.field_type, input.field_magnitude, input.field_direction);
    
    // Modulate alpha based on line progress for smooth start/end
    let line_alpha = 1.0 - smoothstep(0.0, 0.1, input.line_progress) - 
                     smoothstep(0.9, 1.0, input.line_progress);
    
    // Enhanced visibility for scientific accuracy
    let scientific_enhancement = mix(1.0, 2.0, uniforms.scientific_color_mode);
    let enhanced_color = base_color * scientific_enhancement;
    
    return vec4<f32>(enhanced_color, line_alpha * 0.8);
}

// Gradient visualization fragment shader
@fragment
fn fs_gradient_overlay(input: VertexOutput) -> @location(0) vec4<f32> {
    let gradient = calculate_gradient_enhanced(input.world_position);
    let gradient_magnitude = length(gradient);
    
    if (gradient_magnitude < uniforms.gradient_threshold) {
        discard;
    }
    
    // Normalize gradient direction for color
    let gradient_direction = normalize(gradient);
    
    // Color based on gradient direction (similar to normal mapping)
    let gradient_color = vec3<f32>(
        gradient_direction.x * 0.5 + 0.5,
        gradient_direction.y * 0.5 + 0.5,
        gradient_direction.z * 0.5 + 0.5
    );
    
    // Scale by magnitude
    let final_color = gradient_color * gradient_magnitude * uniforms.gradient_color_scale;
    
    return vec4<f32>(final_color, 0.6);
}

// Combined scientific overlay fragment shader
@fragment
fn fs_scientific_overlay(input: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = vec4<f32>(0.0);
    
    // Layer 1: Isosurface rendering
    let iso_color = render_isosurface(input.world_position);
    final_color = final_color + iso_color * (1.0 - final_color.a);
    
    // Layer 2: Gradient overlay
    let gradient = calculate_gradient_enhanced(input.world_position);
    let gradient_magnitude = length(gradient);
    
    if (gradient_magnitude > uniforms.gradient_threshold && uniforms.show_gradient_magnitude > 0.5) {
        let gradient_direction = normalize(gradient);
        let gradient_color = vec3<f32>(
            gradient_direction.x * 0.5 + 0.5,
            gradient_direction.y * 0.5 + 0.5,
            gradient_direction.z * 0.5 + 0.5
        ) * gradient_magnitude * uniforms.gradient_color_scale;
        
        let gradient_overlay = vec4<f32>(gradient_color, 0.3);
        final_color = final_color + gradient_overlay * (1.0 - final_color.a);
    }
    
    // Layer 3: Field lines (rendered separately in main pipeline)
    
    return final_color;
}

// Compute shader for field line integration (Runge-Kutta 4th order)
@compute @workgroup_size(64)
fn cs_integrate_field_lines(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let line_id = global_id.x;
    // Implementation would require separate storage buffers for field line data
    // This would be integrated with the main compute pipeline
} 