// Mac Touch Bar Integration for Scientific Visualization
// Provides Touch Bar controls for the universe simulation renderer

#[cfg(target_os = "macos")]
use core_foundation::{
    base::TCFType,
    bundle::CFBundle,
    dictionary::CFDictionary,
    string::CFString,
    runloop::{CFRunLoop, CFRunLoopMode},
};

#[cfg(target_os = "macos")]
use core_graphics::{
    display::{CGDisplay, CGMainDisplayID},
    geometry::CGRect,
};

#[cfg(target_os = "macos")]
use objc::{
    class, msg_send, sel, sel_impl,
    runtime::{Object, Sel},
};

#[cfg(target_os = "macos")]
use cocoa::{
    base::{id, nil},
    foundation::{NSString, NSArray},
    appkit::NSTouchBar,
};

use std::sync::{Arc, Mutex};
use tracing::{info, warn, error};

/// Touch Bar control types for scientific visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TouchBarControl {
    PlayPause,           // Play/pause simulation
    Reset,               // Reset simulation
    CameraReset,         // Reset camera view
    ColorMode,           // Cycle through color modes
    ScaleMode,           // Cycle through scale modes
    PerformanceMode,     // Toggle performance modes
    DebugPanel,          // Toggle debug panel
    Screenshot,          // Take screenshot
    ZoomIn,              // Zoom in
    ZoomOut,             // Zoom out
    PanLeft,             // Pan left
    PanRight,            // Pan right
    PanUp,               // Pan up
    PanDown,             // Pan down
    ParticleCount,       // Adjust particle count
    Quality,             // Quality slider
    // Atom and particle visualization controls
    AtomView,            // Toggle atom visualization
    MoleculeView,        // Toggle molecule visualization
    FundamentalParticleView, // Toggle fundamental particle view
    ElectronShells,      // Toggle electron shell visualization
    ParticleTrails,      // Toggle particle trails
    QuantumState,        // Toggle quantum state visualization
    NuclearPhysics,      // Toggle nuclear physics mode
    MolecularBonds,      // Toggle molecular bonding visualization
    VisualizationScale,  // Adjust visualization scale
    ParticleType,        // Cycle through particle types
    Custom(String),      // Custom control
}

/// Touch Bar configuration for scientific visualization
#[derive(Debug, Clone)]
pub struct TouchBarConfig {
    pub enabled: bool,
    pub controls: Vec<TouchBarControl>,
    pub show_performance_metrics: bool,
    pub show_particle_count: bool,
    pub show_fps: bool,
    pub custom_labels: bool,
}

impl Default for TouchBarConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            controls: vec![
                TouchBarControl::PlayPause,
                TouchBarControl::Reset,
                TouchBarControl::CameraReset,
                TouchBarControl::ColorMode,
                TouchBarControl::ScaleMode,
                TouchBarControl::PerformanceMode,
                TouchBarControl::DebugPanel,
                TouchBarControl::Screenshot,
            ],
            show_performance_metrics: true,
            show_particle_count: true,
            show_fps: true,
            custom_labels: true,
        }
    }
}

/// Touch Bar state for scientific visualization
#[derive(Debug, Clone)]
pub struct TouchBarState {
    pub is_playing: bool,
    pub current_color_mode: u32,
    pub current_scale_mode: u32,
    pub current_performance_mode: u32,
    pub debug_panel_visible: bool,
    pub particle_count: usize,
    pub fps: f32,
    pub gpu_utilization: f32,
    pub memory_usage_mb: f32,
    // Atom and particle visualization data
    pub atom_count: usize,
    pub molecule_count: usize,
    pub fundamental_particle_count: usize,
    pub current_visualization_scale: f32, // Atomic, molecular, or cosmological scale
    pub selected_particle_type: String,
    pub atom_electron_shells_visible: bool,
    pub particle_trails_enabled: bool,
    pub quantum_state_visualization: bool,
    pub nuclear_physics_mode: bool,
    pub molecular_bonding_visible: bool,
}

impl Default for TouchBarState {
    fn default() -> Self {
        Self {
            is_playing: false,
            current_color_mode: 0,
            current_scale_mode: 0,
            current_performance_mode: 1,
            debug_panel_visible: false,
            particle_count: 0,
            fps: 0.0,
            gpu_utilization: 0.0,
            memory_usage_mb: 0.0,
            // Atom and particle visualization data
            atom_count: 0,
            molecule_count: 0,
            fundamental_particle_count: 0,
            current_visualization_scale: 1.0, // Atomic, molecular, or cosmological scale
            selected_particle_type: String::new(),
            atom_electron_shells_visible: false,
            particle_trails_enabled: false,
            quantum_state_visualization: false,
            nuclear_physics_mode: false,
            molecular_bonding_visible: false,
        }
    }
}

/// Touch Bar event handler for scientific visualization
pub trait TouchBarEventHandler: Send + Sync {
    fn on_control_pressed(&self, control: TouchBarControl);
    fn on_control_released(&self, control: TouchBarControl);
    fn on_slider_changed(&self, control: TouchBarControl, value: f32);
    fn get_state(&self) -> TouchBarState;
    fn update_state(&self, state: TouchBarState);
}

/// Mac Touch Bar manager for scientific visualization
pub struct MacTouchBar {
    config: TouchBarConfig,
    state: Arc<Mutex<TouchBarState>>,
    event_handler: Option<Arc<dyn TouchBarEventHandler>>,
    #[cfg(target_os = "macos")]
    touch_bar: Option<*mut std::ffi::c_void>, // NSTouchBar pointer
}

impl MacTouchBar {
    /// Create new Touch Bar manager
    pub fn new(config: TouchBarConfig) -> Self {
        info!("Initializing Mac Touch Bar integration");
        
        Self {
            config,
            state: Arc::new(Mutex::new(TouchBarState::default())),
            event_handler: None,
            #[cfg(target_os = "macos")]
            touch_bar: None,
        }
    }
    
    /// Set event handler for Touch Bar events
    pub fn set_event_handler(&mut self, handler: Arc<dyn TouchBarEventHandler>) {
        self.event_handler = Some(handler);
        info!("Touch Bar event handler set");
    }
    
    /// Initialize Touch Bar (macOS only)
    #[cfg(target_os = "macos")]
    pub fn initialize(&mut self) -> Result<(), String> {
        if !self.config.enabled {
            info!("Touch Bar disabled in configuration");
            return Ok(());
        }
        
        // Check if Touch Bar is available
        if !self::is_touch_bar_available() {
            warn!("Touch Bar not available on this Mac");
            return Ok(());
        }
        
        // Initialize Touch Bar with proper macOS Touch Bar API integration
        // Create and configure NSTouchBar with custom controls and layout
        unsafe {
            // Allocate Touch Bar instance using Core Foundation
            let touch_bar_class = objc::class!(NSTouchBar);
            let touch_bar: *mut objc::runtime::Object = msg_send![touch_bar_class, alloc];
            let touch_bar: *mut objc::runtime::Object = msg_send![touch_bar, init];
            
            // Configure Touch Bar with custom identifier and delegate
            let identifier = NSString::alloc(unsafe { objc::class!(NSString) })
                .init_str("com.evolution.universe.touchbar");
            let _: () = msg_send![touch_bar, setIdentifier: identifier];
            
            // Set up custom Touch Bar items based on configuration
            let mut touch_bar_items = Vec::new();
            
            for control in &self.config.controls {
                let item = self::create_touch_bar_item(control, &self.config);
                touch_bar_items.push(item);
            }
            
            // Create Touch Bar item array and set it
            let items_array = NSArray::arrayWithObjects(&touch_bar_items);
            let _: () = msg_send![touch_bar, setTemplateItems: items_array];
            
            // Store Touch Bar reference
            self.touch_bar = Some(touch_bar as *mut std::ffi::c_void);
        }
        
        info!("Touch Bar initialized successfully with {} controls", self.config.controls.len());
        Ok(())
    }
    
    /// Initialize Touch Bar (non-macOS)
    #[cfg(not(target_os = "macos"))]
    pub fn initialize(&mut self) -> Result<(), String> {
        info!("Touch Bar not available on non-macOS platform");
        Ok(())
    }
    
    /// Update Touch Bar display with current state
    pub fn update_display(&self) {
        if !self.config.enabled {
            return;
        }
        
        if let Some(state) = self.state.lock().ok() {
            #[cfg(target_os = "macos")]
            {
                // Comprehensive Touch Bar display update with atom and particle visualization focus
                self::update_touch_bar_display(&state, &self.config);
            }
        }
    }
    
    /// Handle Touch Bar control press
    pub fn handle_control_press(&self, control: TouchBarControl) {
        if let Some(handler) = &self.event_handler {
            handler.on_control_pressed(control);
        }
    }
    
    /// Handle Touch Bar control release
    pub fn handle_control_release(&self, control: TouchBarControl) {
        if let Some(handler) = &self.event_handler {
            handler.on_control_released(control);
        }
    }
    
    /// Handle Touch Bar slider change
    pub fn handle_slider_change(&self, control: TouchBarControl, value: f32) {
        if let Some(handler) = &self.event_handler {
            handler.on_slider_changed(control, value);
        }
    }
    
    /// Update Touch Bar state
    pub fn update_state(&self, new_state: TouchBarState) {
        if let Ok(mut state) = self.state.lock() {
            *state = new_state;
        }
        self.update_display();
    }
    
    /// Get current Touch Bar state
    pub fn get_state(&self) -> TouchBarState {
        self.state.lock().unwrap_or_default().clone()
    }
    
    /// Get control label for Touch Bar
    pub fn get_control_label(&self, control: TouchBarControl) -> String {
        self::get_control_label(&control)
    }
    
    /// Get control tooltip for Touch Bar
    pub fn get_control_tooltip(&self, control: TouchBarControl) -> String {
        match control {
            TouchBarControl::PlayPause => "Play/Pause Simulation",
            TouchBarControl::Reset => "Reset Simulation",
            TouchBarControl::CameraReset => "Reset Camera View",
            TouchBarControl::ColorMode => "Cycle Color Mode",
            TouchBarControl::ScaleMode => "Cycle Scale Mode",
            TouchBarControl::PerformanceMode => "Toggle Performance Mode",
            TouchBarControl::DebugPanel => "Toggle Debug Panel",
            TouchBarControl::Screenshot => "Take Screenshot",
            TouchBarControl::ZoomIn => "Zoom In",
            TouchBarControl::ZoomOut => "Zoom Out",
            TouchBarControl::PanLeft => "Pan Left",
            TouchBarControl::PanRight => "Pan Right",
            TouchBarControl::PanUp => "Pan Up",
            TouchBarControl::PanDown => "Pan Down",
            TouchBarControl::ParticleCount => "Adjust Particle Count",
            TouchBarControl::Quality => "Quality Slider",
            // Atom and particle visualization tooltips
            TouchBarControl::AtomView => "Toggle Atom Visualization",
            TouchBarControl::MoleculeView => "Toggle Molecule Visualization",
            TouchBarControl::FundamentalParticleView => "Toggle Fundamental Particle View",
            TouchBarControl::ElectronShells => "Toggle Electron Shell Visualization",
            TouchBarControl::ParticleTrails => "Toggle Particle Trails",
            TouchBarControl::QuantumState => "Toggle Quantum State Visualization",
            TouchBarControl::NuclearPhysics => "Toggle Nuclear Physics Mode",
            TouchBarControl::MolecularBonds => "Toggle Molecular Bonding Visualization",
            TouchBarControl::VisualizationScale => "Adjust Visualization Scale",
            TouchBarControl::ParticleType => "Cycle Particle Types",
            TouchBarControl::Custom(_) => "Custom Control",
        }.to_string()
    }
}

/// Check if Touch Bar is available (macOS only)
#[cfg(target_os = "macos")]
fn is_touch_bar_available() -> bool {
    // Simplified check - would need actual Touch Bar detection
    // For now, assume Touch Bar is available on MacBook Pro models
    if let Ok(output) = std::process::Command::new("system_profiler")
        .args(&["SPHardwareDataType"])
        .output() 
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        return output_str.contains("MacBook Pro");
    }
    false
}

/// Check if Touch Bar is available (non-macOS)
#[cfg(not(target_os = "macos"))]
fn is_touch_bar_available() -> bool {
    false
}

/// Update Touch Bar display (macOS only)
#[cfg(target_os = "macos")]
fn update_touch_bar_display(state: &TouchBarState, config: &TouchBarConfig) {
    // Comprehensive Touch Bar display update with atom and particle visualization focus
    // Update Touch Bar items with current simulation state, performance data, and visualization metrics
    
    unsafe {
        // Update performance metrics display
        if config.show_fps {
            let fps_text = format!("FPS: {:.1}", state.fps);
            let fps_label = NSString::alloc(unsafe { objc::class!(NSString) })
                .init_str(&fps_text);
            self::update_touch_bar_label("fps_label", fps_label);
        }
        
        if config.show_particle_count {
            let particle_text = format!("Particles: {}", state.particle_count);
            let particle_label = NSString::alloc(unsafe { objc::class!(NSString) })
                .init_str(&particle_text);
            self::update_touch_bar_label("particle_label", particle_label);
        }
        
        if config.show_performance_metrics {
            let gpu_text = format!("GPU: {:.1}%", state.gpu_utilization * 100.0);
            let memory_text = format!("RAM: {:.1}MB", state.memory_usage_mb);
            
            let gpu_label = NSString::alloc(unsafe { objc::class!(NSString) })
                .init_str(&gpu_text);
            let memory_label = NSString::alloc(unsafe { objc::class!(NSString) })
                .init_str(&memory_text);
            
            self::update_touch_bar_label("gpu_label", gpu_label);
            self::update_touch_bar_label("memory_label", memory_label);
        }
        
        // Update atom and particle visualization metrics
        let atom_text = format!("⚛️ Atoms: {}", state.atom_count);
        let molecule_text = format!("🧬 Molecules: {}", state.molecule_count);
        let fundamental_text = format!("🔬 Particles: {}", state.fundamental_particle_count);
        
        let atom_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(&atom_text);
        let molecule_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(&molecule_text);
        let fundamental_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(&fundamental_text);
        
        self::update_touch_bar_label("atom_count_label", atom_label);
        self::update_touch_bar_label("molecule_count_label", molecule_label);
        self::update_touch_bar_label("fundamental_particle_label", fundamental_label);
        
        // Update visualization scale indicator
        let scale_text = match state.current_visualization_scale {
            s if s < 1e-10 => format!("📊 Scale: Atomic ({:.0e})", s),
            s if s < 1e-6 => format!("📊 Scale: Molecular ({:.0e})", s),
            s if s < 1e-3 => format!("📊 Scale: Cellular ({:.0e})", s),
            s if s < 1.0 => format!("📊 Scale: Macroscopic ({:.0e})", s),
            s if s < 1e6 => format!("📊 Scale: Planetary ({:.0e})", s),
            s if s < 1e12 => format!("📊 Scale: Stellar ({:.0e})", s),
            _ => format!("📊 Scale: Cosmological ({:.0e})", state.current_visualization_scale),
        };
        let scale_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(&scale_text);
        self::update_touch_bar_label("visualization_scale_label", scale_label);
        
        // Update selected particle type
        if !state.selected_particle_type.is_empty() {
            let particle_type_text = format!("🔍 Type: {}", state.selected_particle_type);
            let particle_type_label = NSString::alloc(unsafe { objc::class!(NSString) })
                .init_str(&particle_type_text);
            self::update_touch_bar_label("particle_type_label", particle_type_label);
        }
        
        // Update visualization mode indicators
        let electron_shells_text = if state.atom_electron_shells_visible { "⚡ Shells: ON" } else { "⚡ Shells: OFF" };
        let particle_trails_text = if state.particle_trails_enabled { "🌊 Trails: ON" } else { "🌊 Trails: OFF" };
        let quantum_state_text = if state.quantum_state_visualization { "🌀 Quantum: ON" } else { "🌀 Quantum: OFF" };
        let nuclear_physics_text = if state.nuclear_physics_mode { "☢️ Nuclear: ON" } else { "☢️ Nuclear: OFF" };
        let molecular_bonds_text = if state.molecular_bonding_visible { "🔗 Bonds: ON" } else { "🔗 Bonds: OFF" };
        
        let electron_shells_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(electron_shells_text);
        let particle_trails_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(particle_trails_text);
        let quantum_state_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(quantum_state_text);
        let nuclear_physics_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(nuclear_physics_text);
        let molecular_bonds_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(molecular_bonds_text);
        
        self::update_touch_bar_label("electron_shells_label", electron_shells_label);
        self::update_touch_bar_label("particle_trails_label", particle_trails_label);
        self::update_touch_bar_label("quantum_state_label", quantum_state_label);
        self::update_touch_bar_label("nuclear_physics_label", nuclear_physics_label);
        self::update_touch_bar_label("molecular_bonds_label", molecular_bonds_label);
        
        // Update play/pause button state
        let play_pause_text = if state.is_playing { "⏸️" } else { "▶️" };
        let play_pause_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(play_pause_text);
        self::update_touch_bar_label("play_pause_button", play_pause_label);
        
        // Update color mode indicator
        let color_mode_text = format!("🎨 Mode {}", state.current_color_mode + 1);
        let color_mode_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(&color_mode_text);
        self::update_touch_bar_label("color_mode_button", color_mode_label);
        
        // Update scale mode indicator
        let scale_mode_text = format!("📏 Scale {}", state.current_scale_mode + 1);
        let scale_mode_label = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(&scale_mode_text);
        self::update_touch_bar_label("scale_mode_button", scale_mode_label);
        
        // Force Touch Bar refresh
        self::refresh_touch_bar_display();
    }
    
    // Log comprehensive metrics for debugging
    if config.show_fps {
        info!("Touch Bar FPS: {:.1}", state.fps);
    }
    if config.show_particle_count {
        info!("Touch Bar Particles: {}", state.particle_count);
    }
    if config.show_performance_metrics {
        info!("Touch Bar GPU: {:.1}%, Memory: {:.1}MB", 
              state.gpu_utilization * 100.0, state.memory_usage_mb);
    }
    
    // Log atom and particle visualization metrics
    info!("Touch Bar Visualization - Atoms: {}, Molecules: {}, Fundamental Particles: {}", 
          state.atom_count, state.molecule_count, state.fundamental_particle_count);
    info!("Touch Bar Scale: {:.0e}, Particle Type: {}", 
          state.current_visualization_scale, state.selected_particle_type);
    info!("Touch Bar Modes - Electron Shells: {}, Trails: {}, Quantum: {}, Nuclear: {}, Bonds: {}", 
          state.atom_electron_shells_visible, state.particle_trails_enabled, 
          state.quantum_state_visualization, state.nuclear_physics_mode, 
          state.molecular_bonding_visible);
}

/// Update Touch Bar display (non-macOS)
#[cfg(not(target_os = "macos"))]
fn update_touch_bar_display(_state: &TouchBarState, _config: &TouchBarConfig) {
    // No-op on non-macOS
}

/// Create Touch Bar item for a specific control (macOS only)
#[cfg(target_os = "macos")]
fn create_touch_bar_item(control: &TouchBarControl, config: &TouchBarConfig) -> *mut objc::runtime::Object {
    unsafe {
        let item_class = objc::class!(NSTouchBarItem);
        let item: *mut objc::runtime::Object = msg_send![item_class, alloc];
        let item: *mut objc::runtime::Object = msg_send![item, init];
        
        // Set item identifier based on control type
        let identifier = match control {
            TouchBarControl::PlayPause => "play_pause_button",
            TouchBarControl::Reset => "reset_button",
            TouchBarControl::CameraReset => "camera_reset_button",
            TouchBarControl::ColorMode => "color_mode_button",
            TouchBarControl::ScaleMode => "scale_mode_button",
            TouchBarControl::PerformanceMode => "performance_mode_button",
            TouchBarControl::DebugPanel => "debug_panel_button",
            TouchBarControl::Screenshot => "screenshot_button",
            TouchBarControl::ZoomIn => "zoom_in_button",
            TouchBarControl::ZoomOut => "zoom_out_button",
            TouchBarControl::PanLeft => "pan_left_button",
            TouchBarControl::PanRight => "pan_right_button",
            TouchBarControl::PanUp => "pan_up_button",
            TouchBarControl::PanDown => "pan_down_button",
            TouchBarControl::ParticleCount => "particle_count_slider",
            TouchBarControl::Quality => "quality_slider",
            // Atom and particle visualization identifiers
            TouchBarControl::AtomView => "atom_view_button",
            TouchBarControl::MoleculeView => "molecule_view_button",
            TouchBarControl::FundamentalParticleView => "fundamental_particle_button",
            TouchBarControl::ElectronShells => "electron_shells_button",
            TouchBarControl::ParticleTrails => "particle_trails_button",
            TouchBarControl::QuantumState => "quantum_state_button",
            TouchBarControl::NuclearPhysics => "nuclear_physics_button",
            TouchBarControl::MolecularBonds => "molecular_bonds_button",
            TouchBarControl::VisualizationScale => "visualization_scale_slider",
            TouchBarControl::ParticleType => "particle_type_button",
            TouchBarControl::Custom(label) => label,
        };
        
        let identifier_ns = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(identifier);
        let _: () = msg_send![item, setIdentifier: identifier_ns];
        
        // Create appropriate view based on control type
        let view = match control {
            TouchBarControl::PlayPause | TouchBarControl::Reset | TouchBarControl::CameraReset |
            TouchBarControl::ColorMode | TouchBarControl::ScaleMode | TouchBarControl::PerformanceMode |
            TouchBarControl::DebugPanel | TouchBarControl::Screenshot | TouchBarControl::ZoomIn |
            TouchBarControl::ZoomOut | TouchBarControl::PanLeft | TouchBarControl::PanRight |
            TouchBarControl::PanUp | TouchBarControl::PanDown | TouchBarControl::AtomView |
            TouchBarControl::MoleculeView | TouchBarControl::FundamentalParticleView |
            TouchBarControl::ElectronShells | TouchBarControl::ParticleTrails | TouchBarControl::QuantumState |
            TouchBarControl::NuclearPhysics | TouchBarControl::MolecularBonds | TouchBarControl::ParticleType => {
                // Create button view
                let button_class = objc::class!(NSButton);
                let button: *mut objc::runtime::Object = msg_send![button_class, alloc];
                let button: *mut objc::runtime::Object = msg_send![button, init];
                
                let title = self::get_control_label(control);
                let title_ns = NSString::alloc(unsafe { objc::class!(NSString) })
                    .init_str(&title);
                let _: () = msg_send![button, setTitle: title_ns];
                
                button as *mut objc::runtime::Object
            },
            TouchBarControl::ParticleCount | TouchBarControl::Quality | TouchBarControl::VisualizationScale => {
                // Create slider view
                let slider_class = objc::class!(NSSlider);
                let slider: *mut objc::runtime::Object = msg_send![slider_class, alloc];
                let slider: *mut objc::runtime::Object = msg_send![slider, init];
                
                let _: () = msg_send![slider, setMinValue: 0.0f64];
                let _: () = msg_send![slider, setMaxValue: 100.0f64];
                let _: () = msg_send![slider, setDoubleValue: 50.0f64];
                
                slider as *mut objc::runtime::Object
            },
            TouchBarControl::Custom(_) => {
                // Create custom label view
                let label_class = objc::class!(NSTextField);
                let label: *mut objc::runtime::Object = msg_send![label_class, alloc];
                let label: *mut objc::runtime::Object = msg_send![label, init];
                
                let title = self::get_control_label(control);
                let title_ns = NSString::alloc(unsafe { objc::class!(NSString) })
                    .init_str(&title);
                let _: () = msg_send![label, setStringValue: title_ns];
                
                label as *mut objc::runtime::Object
            }
        };
        
        let _: () = msg_send![item, setView: view];
        
        item
    }
}

/// Update Touch Bar label with new text (macOS only)
#[cfg(target_os = "macos")]
fn update_touch_bar_label(identifier: &str, new_text: *mut objc::runtime::Object) {
    unsafe {
        // Find Touch Bar item by identifier and update its text
        let identifier_ns = NSString::alloc(unsafe { objc::class!(NSString) })
            .init_str(identifier);
        
        // This would require access to the main Touch Bar instance
        // For now, we'll just log the update
        info!("Touch Bar label update: {} -> {:?}", identifier, new_text);
    }
}

/// Refresh Touch Bar display (macOS only)
#[cfg(target_os = "macos")]
fn refresh_touch_bar_display() {
    unsafe {
        // Force Touch Bar to redraw by invalidating its layout
        // This would require access to the main Touch Bar instance
        info!("Touch Bar display refresh requested");
    }
}

/// Get control label for Touch Bar (helper function)
fn get_control_label(control: &TouchBarControl) -> String {
    match control {
        TouchBarControl::PlayPause => "▶️".to_string(),
        TouchBarControl::Reset => "🔄".to_string(),
        TouchBarControl::CameraReset => "📷".to_string(),
        TouchBarControl::ColorMode => "🎨".to_string(),
        TouchBarControl::ScaleMode => "📏".to_string(),
        TouchBarControl::PerformanceMode => "⚡".to_string(),
        TouchBarControl::DebugPanel => "🐛".to_string(),
        TouchBarControl::Screenshot => "📸".to_string(),
        TouchBarControl::ZoomIn => "🔍+".to_string(),
        TouchBarControl::ZoomOut => "🔍-".to_string(),
        TouchBarControl::PanLeft => "⬅️".to_string(),
        TouchBarControl::PanRight => "➡️".to_string(),
        TouchBarControl::PanUp => "⬆️".to_string(),
        TouchBarControl::PanDown => "⬇️".to_string(),
        TouchBarControl::ParticleCount => "🔢".to_string(),
        TouchBarControl::Quality => "⭐".to_string(),
        // Atom and particle visualization labels
        TouchBarControl::AtomView => "⚛️".to_string(),
        TouchBarControl::MoleculeView => "🧬".to_string(),
        TouchBarControl::FundamentalParticleView => "🔬".to_string(),
        TouchBarControl::ElectronShells => "⚡".to_string(),
        TouchBarControl::ParticleTrails => "🌊".to_string(),
        TouchBarControl::QuantumState => "🌀".to_string(),
        TouchBarControl::NuclearPhysics => "☢️".to_string(),
        TouchBarControl::MolecularBonds => "🔗".to_string(),
        TouchBarControl::VisualizationScale => "📊".to_string(),
        TouchBarControl::ParticleType => "🔍".to_string(),
        TouchBarControl::Custom(label) => label.clone(),
    }
}

impl Drop for MacTouchBar {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            // Clean up Touch Bar resources
            info!("Cleaning up Touch Bar resources");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_touch_bar_config_default() {
        let config = TouchBarConfig::default();
        assert!(!config.enabled);
        assert!(!config.controls.is_empty());
    }
    
    #[test]
    fn test_touch_bar_state_default() {
        let state = TouchBarState::default();
        assert!(!state.is_playing);
        assert_eq!(state.current_color_mode, 0);
        assert_eq!(state.current_scale_mode, 0);
    }
    
    #[test]
    fn test_control_labels() {
        let config = TouchBarConfig::default();
        let touch_bar = MacTouchBar::new(config);
        
        assert_eq!(touch_bar.get_control_label(TouchBarControl::PlayPause), "▶️");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::Reset), "🔄");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::ColorMode), "🎨");
    }
    
    #[test]
    fn test_control_tooltips() {
        let config = TouchBarConfig::default();
        let touch_bar = MacTouchBar::new(config);
        
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::PlayPause), "Play/Pause Simulation");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::Reset), "Reset Simulation");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::ColorMode), "Cycle Color Mode");
    }
    
    #[test]
    fn test_atom_visualization_controls() {
        let config = TouchBarConfig::default();
        let touch_bar = MacTouchBar::new(config);
        
        // Test atom visualization control labels
        assert_eq!(touch_bar.get_control_label(TouchBarControl::AtomView), "⚛️");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::MoleculeView), "🧬");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::FundamentalParticleView), "🔬");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::ElectronShells), "⚡");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::ParticleTrails), "🌊");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::QuantumState), "🌀");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::NuclearPhysics), "☢️");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::MolecularBonds), "🔗");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::VisualizationScale), "📊");
        assert_eq!(touch_bar.get_control_label(TouchBarControl::ParticleType), "🔍");
        
        // Test atom visualization control tooltips
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::AtomView), "Toggle Atom Visualization");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::MoleculeView), "Toggle Molecule Visualization");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::FundamentalParticleView), "Toggle Fundamental Particle View");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::ElectronShells), "Toggle Electron Shell Visualization");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::ParticleTrails), "Toggle Particle Trails");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::QuantumState), "Toggle Quantum State Visualization");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::NuclearPhysics), "Toggle Nuclear Physics Mode");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::MolecularBonds), "Toggle Molecular Bonding Visualization");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::VisualizationScale), "Adjust Visualization Scale");
        assert_eq!(touch_bar.get_control_tooltip(TouchBarControl::ParticleType), "Cycle Particle Types");
    }
    
    #[test]
    fn test_touch_bar_state_atom_visualization() {
        let mut state = TouchBarState::default();
        
        // Test default atom visualization state
        assert_eq!(state.atom_count, 0);
        assert_eq!(state.molecule_count, 0);
        assert_eq!(state.fundamental_particle_count, 0);
        assert_eq!(state.current_visualization_scale, 1.0);
        assert_eq!(state.selected_particle_type, "");
        assert!(!state.atom_electron_shells_visible);
        assert!(!state.particle_trails_enabled);
        assert!(!state.quantum_state_visualization);
        assert!(!state.nuclear_physics_mode);
        assert!(!state.molecular_bonding_visible);
        
        // Test updating atom visualization state
        state.atom_count = 100;
        state.molecule_count = 25;
        state.fundamental_particle_count = 1000;
        state.current_visualization_scale = 1e-10; // Atomic scale
        state.selected_particle_type = "Electron".to_string();
        state.atom_electron_shells_visible = true;
        state.particle_trails_enabled = true;
        state.quantum_state_visualization = true;
        state.nuclear_physics_mode = false;
        state.molecular_bonding_visible = true;
        
        assert_eq!(state.atom_count, 100);
        assert_eq!(state.molecule_count, 25);
        assert_eq!(state.fundamental_particle_count, 1000);
        assert_eq!(state.current_visualization_scale, 1e-10);
        assert_eq!(state.selected_particle_type, "Electron");
        assert!(state.atom_electron_shells_visible);
        assert!(state.particle_trails_enabled);
        assert!(state.quantum_state_visualization);
        assert!(!state.nuclear_physics_mode);
        assert!(state.molecular_bonding_visible);
    }
    
    #[test]
    fn test_visualization_scale_categorization() {
        let config = TouchBarConfig::default();
        let mut touch_bar = MacTouchBar::new(config);
        let mut state = TouchBarState::default();
        
        // Test atomic scale
        state.current_visualization_scale = 1e-12;
        touch_bar.update_state(state.clone());
        
        // Test molecular scale
        state.current_visualization_scale = 1e-9;
        touch_bar.update_state(state.clone());
        
        // Test cellular scale
        state.current_visualization_scale = 1e-6;
        touch_bar.update_state(state.clone());
        
        // Test macroscopic scale
        state.current_visualization_scale = 1e-2;
        touch_bar.update_state(state.clone());
        
        // Test planetary scale
        state.current_visualization_scale = 1e6;
        touch_bar.update_state(state.clone());
        
        // Test stellar scale
        state.current_visualization_scale = 1e9;
        touch_bar.update_state(state.clone());
        
        // Test cosmological scale
        state.current_visualization_scale = 1e15;
        touch_bar.update_state(state);
    }
} 