# EVOLVE Native Renderer - Heavy Mode

High-performance GPU-accelerated particle renderer for universe simulation with advanced scientific visualization capabilities.

## Features

### 🚀 Performance
- **GPU Acceleration**: Direct WGPU rendering with zero-copy particle data access
- **High Throughput**: Support for 1M+ particles with 60+ FPS rendering
- **Parallel Processing**: Rayon-based parallel particle data conversion
- **Optimized Memory**: Efficient vertex buffer management with GPU alignment

### 🔬 Scientific Visualization

#### Heavy Mode Features (--features heavy)
- **Multi-Scale Particle Rendering**: Logarithmic, linear, and energy-based scaling
- **Physics-Based Color Coding**: Scientific color palettes for particle types, charges, temperatures
- **Relativistic Effects**: Visual representation of relativistic factors and Doppler shifts
- **Advanced Shaders**: Physics-based lighting and particle appearance

#### Color Modes
1. **Particle Type**: Standard Model particle classification
   - Quarks: Red
   - Leptons: Green  
   - Gauge Bosons: Blue
   - Composite Particles: Yellow
   - Nuclei: Magenta

2. **Charge**: Electrostatic visualization
   - Positive: Red
   - Negative: Blue
   - Neutral: Gray

3. **Temperature**: Blackbody radiation colors
   - Cold (< 2500K): Black → Red
   - Warm (2500-5000K): Red → Yellow
   - Hot (5000-7500K): Yellow → White
   - Very Hot (> 7500K): White → Blue

4. **Velocity**: Doppler shift visualization
   - Approaching: Blue shift
   - Receding: Red shift
   - Intensity based on relativistic factor

5. **Interactions**: Heat map of particle interaction frequency
   - Cold: Blue → Cyan
   - Warm: Cyan → Yellow  
   - Hot: Yellow → Red

6. **Scientific**: Multi-channel composite visualization

#### Scaling Modes
- **Linear**: Direct mass-based scaling
- **Logarithmic**: Wide dynamic range support (subatomic to stellar scales)
- **Energy**: Kinetic energy-based particle sizing
- **Custom**: User-defined scaling factor

## Usage

### Basic Integration
```rust
use native_renderer::run_renderer;
use std::sync::{Arc, Mutex};

// Start the renderer with simulation
let simulation = Arc::new(Mutex::new(your_simulation));
run_renderer(simulation).await?;
```

### Standalone Renderer
```rust
use native_renderer::{NativeRenderer, Camera, ColorMode, ScaleMode};

let mut renderer = NativeRenderer::new(&window).await?;

// Configure scientific visualization
renderer.set_color_mode(ColorMode::Temperature);
renderer.set_scale_mode(ScaleMode::Logarithmic);

// Update and render
renderer.update_particles(&simulation)?;
renderer.render(simulation_time)?;
```

## Controls

### Camera Navigation
- **WASD**: Move camera horizontally
- **Q/E**: Move camera up/down
- **Mouse**: Look around (planned)

### Scientific Visualization
- **1**: Particle Type color mode
- **2**: Charge color mode
- **3**: Temperature color mode
- **4**: Velocity color mode
- **5**: Interactions color mode
- **6**: Scientific composite mode

### System
- **H**: Toggle heavy mode (if feature enabled)
- **R**: Reset camera to default position
- **ESC**: Exit renderer

## Performance

### Benchmarks
- **1M Particles**: 60+ FPS on modern GPUs
- **Frame Time**: < 16ms for smooth 60 FPS
- **Memory Usage**: Efficient GPU buffer management
- **CPU Usage**: Optimized with parallel data conversion

### Performance Monitoring
```rust
let metrics = renderer.get_metrics();
println!("FPS: {:.1}", metrics.fps);
println!("Frame Time: {:.2}ms", metrics.frame_time_ms);
println!("Particles: {}", metrics.particles_rendered);
```

## Architecture

### Shader Pipeline
- **Vertex Shader**: Particle positioning, scaling, and color calculation
- **Fragment Shader**: Physics-based particle appearance and alpha blending
- **Compute Shader** (Heavy Mode): Advanced particle effects and interactions

### Data Flow
```
Simulation → ParticleVertex → GPU Buffer → Shaders → Screen
```

### Heavy Mode Extensions
- Additional compute pipeline for advanced effects
- Extended uniform data with scientific parameters
- Interactive heat maps and temperature fields
- Real-time interaction visualization

## Scientific Accuracy

### Physical Constants
All visualizations use accurate physical constants:
- Speed of Light: 299,792,458 m/s
- Planck Constant: 6.626 × 10⁻³⁴ J⋅s
- Boltzmann Constant: 1.381 × 10⁻²³ J/K

### Relativistic Calculations
- Proper Lorentz factor: γ = 1/√(1 - v²/c²)
- Relativistic energy: E = √((pc)² + (mc²)²)
- Doppler shift visualization for high-velocity particles

### Color Science
- Wien's displacement law for temperature colors
- Accurate blackbody radiation spectrum mapping
- Scientifically based particle type classifications

## Testing

Comprehensive test suite covering:
- GPU data layout verification
- Scientific constant accuracy
- Camera matrix calculations
- Relativistic factor computations
- Color mapping validation
- Performance metrics

Run tests with:
```bash
cargo test -p native_renderer
```

## Dependencies

### Core
- `wgpu`: GPU rendering and compute
- `winit`: Window management and input
- `nalgebra`: Linear algebra and transformations
- `bytemuck`: Zero-copy GPU data conversion

### Heavy Mode
- `rayon`: Parallel data processing
- `tracing`: Performance logging and debugging

## Configuration

### Features
- `default`: Basic particle rendering
- `heavy`: Full scientific visualization suite
- `advanced-shaders`: Enhanced particle appearance
- `multi-scale-rendering`: Wide dynamic range support
- `scientific-visualization`: Physics-based color coding

### Build Examples
```bash
# Basic mode
cargo build -p native_renderer

# Heavy mode with all features
cargo build -p native_renderer --features heavy

# Specific feature set
cargo build -p native_renderer --features "advanced-shaders,scientific-visualization"
```

## Integration with EVOLVE

The native renderer seamlessly integrates with the EVOLVE Universe Simulation:

## Data Pipeline
```
PhysicsEngine → UniverseSimulation → NativeRenderer → GPU → Screen
```

## Non-Blocking Architecture
- Simulation continues running while renderer displays cached data
- Zero-copy data access where possible
- Graceful fallback when simulation is busy
- Parallel particle data conversion using Rayon

## Performance Integration
- Automatic LOD (Level of Detail) based on particle count
- Dynamic culling for particles outside view frustum
- Adaptive frame rate targeting
- Memory-efficient particle streaming

## Usage Examples

### Basic Heavy Mode Usage
```bash
# Run with heavy mode rendering
cargo run --bin evolve --features heavy -- --renderer native

# Heavy mode with specific visualization
cargo run --bin evolve --features heavy -- --renderer native --color-mode temperature
```

### Standalone Renderer Testing
```bash
# Test renderer performance
cargo run -p native_renderer --example benchmark --features heavy

# Shader development
cargo run -p native_renderer --example shader_test --features advanced-shaders
```

## Troubleshooting

### Common Issues

1. **Black Screen**: Check if particles are being generated
2. **Low FPS**: Reduce particle count or disable heavy mode
3. **Shader Compilation**: Ensure GPU supports required features
4. **Memory Issues**: Reduce max_particles in configuration

### Debug Commands
```bash
# Check shader compilation
cargo run -p native_renderer --features heavy -- --validate-shaders

# Performance profiling
cargo run -p native_renderer --features heavy -- --profile
```

## Future Enhancements

### Planned Features
- [ ] Volumetric rendering for gas clouds
- [ ] Advanced particle trails
- [ ] Real-time ray tracing for relativistic effects
- [ ] Multi-GPU support
- [ ] VR/AR integration
- [ ] Scientific measurement tools overlay

### Performance Optimizations
- [ ] GPU-based particle sorting
- [ ] Hierarchical culling
- [ ] Temporal upsampling
- [ ] Adaptive quality scaling
- [ ] Multi-threaded command buffer generation

## Technical Specifications

### Minimum Requirements
- GPU: Vulkan 1.1 or OpenGL 4.3
- RAM: 4GB available
- CPU: 4 cores for parallel processing

### Recommended Specifications
- GPU: Modern discrete GPU with 4+ GB VRAM
- RAM: 16GB+ for large simulations
- CPU: 8+ cores for optimal performance

---

For more details about the EVOLVE project, see the main README.md. 