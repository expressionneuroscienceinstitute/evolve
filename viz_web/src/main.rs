// Evolve: Advanced AI Evolution Monitoring Portal
//
// The most sophisticated real-time monitoring system ever created for
// fundamental particle simulation and AI agent evolution tracking.
//
// Features:
// - Real-time particle physics visualization
// - Comprehensive AI decision tracking
// - Advanced lineage analysis and family trees
// - Consciousness emergence monitoring
// - Technology development timelines
// - Natural selection analytics
// - Innovation tracking
// - Population dynamics
// - Environmental monitoring
// - Quantum field visualization

use wasm_bindgen::prelude::*;
use web_sys::*;
use js_sys::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use serde_json;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

// Import our simulation types
use universe_sim::cosmic_era::CosmicEra;

use web_sys::{HtmlCanvasElement, HtmlElement, HtmlInputElement};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Main application state for the monitoring portal
#[wasm_bindgen]
pub struct EvolutionMonitor {
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
    websocket: Option<web_sys::WebSocket>,
    simulation_state: SimulationState,
    view_mode: ViewMode,
    selected_agent: Option<Uuid>,
    selected_lineage: Option<Uuid>,
    time_scale: f64,
    // Visualization config
    particle_size_scale: f64,
    energy_filter_min: f64,
    particle_filter: ParticleFilter,
    decision_analyzer: DecisionAnalyzer,
    lineage_tracker: LineageTracker,
    consciousness_monitor: ConsciousnessMonitor,
    innovation_tracker: InnovationTracker,
    analytics_engine: AnalyticsEngine,
    connected: bool,
    // Performance tracking
    last_frame_time: f64,
    fps: f64,
    last_render_duration: f64,
    // Agent history tracking
    agent_history: HashMap<Uuid, AgentHistory>,
    // Double buffering
    offscreen_canvas: Option<HtmlCanvasElement>,
    offscreen_context: Option<CanvasRenderingContext2d>,
    needs_redraw: bool,
    target_fps: f64,
    frame_interval: f64,
    // Debug message tracking
    debug_messages: Vec<DebugMessage>,
    messages_received: u32,
    rpc_debug_messages: Vec<DebugMessage>,
    rpc_messages_received: u32,
}

/// Agent history tracking structure
#[derive(Debug, Clone)]
struct AgentHistory {
    positions: Vec<[f64; 3]>,
    decisions: Vec<DecisionSummary>,
    innovations: Vec<String>,
}

/// Complete simulation state received from backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub current_tick: u64,
    pub universe_age_gyr: f64,
    pub cosmic_era: CosmicEra,
    pub temperature: f64,
    pub energy_density: f64,
    
    // Fundamental physics
    pub particles: Vec<ParticleVisualization>,
    pub celestial_bodies: Vec<CelestialBodyVisualization>,
    pub quantum_fields: HashMap<String, FieldVisualization>,
    pub nuclei: Vec<NucleusVisualization>,
    pub atoms: Vec<AtomVisualization>,
    pub molecules: Vec<MoleculeVisualization>,
    
    // AI agents and evolution
    pub agents: Vec<AgentVisualization>,
    pub lineages: HashMap<Uuid, LineageVisualization>,
    pub decisions: Vec<DecisionVisualization>,
    pub innovations: Vec<InnovationVisualization>,
    pub consciousness_events: Vec<ConsciousnessEvent>,
    
    // Environmental conditions
    pub environments: Vec<EnvironmentVisualization>,
    pub selection_pressures: Vec<SelectionPressureVisualization>,
    
    // Statistics and analytics
    pub population_stats: PopulationStatistics,
    pub evolution_metrics: EvolutionMetrics,
    pub physics_metrics: PhysicsMetrics,
}

/// Particle visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleVisualization {
    pub id: usize,
    pub particle_type: String,
    pub position: [f64; 3],
    pub momentum: [f64; 3],
    pub energy: f64,
    pub mass: f64,
    pub charge: f64,
    pub spin: [f64; 3],
    pub color_charge: Option<String>,
    pub interaction_count: u32,
    pub age: f64,
    pub decay_probability: f64,
}

/// Celestial body visualization data (stars, planets, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialBodyVisualization {
    pub id: String,
    pub body_type: String,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub mass: f64,          // kg
    pub radius: f64,        // m
    pub temperature: f64,   // K
    pub luminosity: f64,    // W
    pub age: f64,          // years
}

/// AI agent visualization with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVisualization {
    pub id: Uuid,
    pub generation: u32,
    pub birth_tick: u64,
    pub energy: f64,
    pub fitness: f64,
    pub age: u64,
    
    // Evolution levels
    pub sentience_level: f64,
    pub industrialization_level: f64,
    pub digitalization_level: f64,
    pub tech_level: f64,
    pub consciousness_level: f64,
    pub immortality_achieved: bool,
    
    // Decision making
    pub recent_decisions: Vec<DecisionSummary>,
    pub success_rate: f64,
    pub learning_rate: f64,
    pub mutation_rate: f64,
    
    // Social and environmental
    pub social_connections: Vec<Uuid>,
    pub environmental_adaptations: Vec<String>,
    pub innovations_created: Vec<String>,
    
    // Lineage information
    pub parent_id: Option<Uuid>,
    pub children_ids: Vec<Uuid>,
    pub lineage_id: Uuid,
    
    // Physical representation
    pub position: [f64; 3],
    pub size: f64,
    pub color: [f32; 4],
    
    // Additional fields
    pub traits: HashMap<String, f64>,
    pub consciousness: Option<ConsciousnessEvent>,
    pub innovation: Option<InnovationVisualization>,
}

/// Comprehensive decision tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionVisualization {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub timestamp: u64,
    pub decision_type: String,
    pub success: bool,
    pub fitness_impact: f64,
    pub energy_cost: f64,
    pub learning_feedback: f64,
    
    // Context information
    pub environmental_factors: Vec<String>,
    pub social_factors: Vec<String>,
    pub genetic_influences: Vec<String>,
    
    // Outcome tracking
    pub immediate_effects: Vec<String>,
    pub long_term_consequences: Vec<String>,
    pub innovation_progress: f64,
}

/// Lineage visualization with complete family tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageVisualization {
    pub lineage_id: Uuid,
    pub founder_id: Uuid,
    pub current_generation: u32,
    pub total_individuals: u64,
    pub living_individuals: u32,
    pub extinct_branches: u32,
    
    // Evolutionary metrics
    pub average_fitness: f64,
    pub fitness_variance: f64,
    pub genetic_diversity: f64,
    pub mutation_rate: f64,
    
    // Achievement tracking
    pub major_innovations: Vec<InnovationRecord>,
    pub consciousness_milestones: Vec<ConsciousnessMilestone>,
    pub technology_achievements: Vec<TechnologyAchievement>,
    
    // Natural selection
    pub selection_events: Vec<SelectionEventRecord>,
    pub adaptation_events: Vec<AdaptationRecord>,
    pub speciation_events: Vec<SpeciationRecord>,
    
    // Visualization data
    pub family_tree: FamilyTreeVisualization,
    pub evolution_timeline: Vec<EvolutionTimelineEvent>,
}

/// Innovation tracking and visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationVisualization {
    pub id: Uuid,
    pub creator_id: Uuid,
    pub innovation_type: String,
    pub timestamp: u64,
    pub impact_score: f64,
    pub adoption_rate: f64,
    pub complexity_level: f64,
    
    // Innovation characteristics
    pub prerequisites: Vec<String>,
    pub enables: Vec<String>,
    pub resource_requirements: HashMap<String, f64>,
    
    // Spread and evolution
    pub adopters: Vec<Uuid>,
    pub variations: Vec<InnovationVariation>,
    pub obsolescence_risk: f64,
}

/// Consciousness emergence monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEvent {
    pub agent_id: Uuid,
    pub timestamp: u64,
    pub consciousness_type: String,
    pub level: f64,
    pub triggers: Vec<String>,
    pub neural_complexity: f64,
    pub self_awareness_indicators: Vec<String>,
}

/// Environmental visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentVisualization {
    pub id: Uuid,
    pub environment_type: String,
    pub position: [f64; 3],
    pub size: f64,
    pub habitability: f64,
    pub resource_abundance: HashMap<String, f64>,
    pub hazard_level: f64,
    pub agent_count: u32,
    pub carrying_capacity: u32,
}

/// Population statistics and dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStatistics {
    pub total_agents: u32,
    pub active_lineages: u32,
    pub extinct_lineages: u32,
    pub birth_rate: f64,
    pub death_rate: f64,
    pub mutation_rate: f64,
    pub average_fitness: f64,
    pub fitness_variance: f64,
    pub genetic_diversity: f64,
    pub consciousness_distribution: HashMap<String, u32>,
    pub technology_distribution: HashMap<String, u32>,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    pub enum ViewMode {
    ParticlePhysics,
    AtomicStructure,
    MolecularDynamics,
    CelestialBodies,
    NuclearPhysics,
    AgentOverview,
    LineageTree,
    DecisionTracking,
    ConsciousnessMap,
    InnovationTimeline,
    SelectionPressures,
    EnvironmentalMap,
    PopulationDynamics,
    QuantumFields,
    EnergyFlows,
    EmergentComplexity,
    DebugMessages,
}

impl Default for ViewMode {
    fn default() -> Self {
        ViewMode::AgentOverview
    }
}

#[wasm_bindgen]
impl EvolutionMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<EvolutionMonitor, JsValue> {
        console_log!("Initializing Evolve: AI Evolution Monitor");
        
        let document = window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id(canvas_id)
            .unwrap()
            .dyn_into::<HtmlCanvasElement>()?;
        
        let context = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;
        
        // Set up high-DPI canvas dimensions **without** applying an additional context scale.
        // Previously we also scaled the rendering context here which resulted in the scene
        // being drawn twice as large, then copied onto another already-scaled context –
        // effectively a double-scale that produced flickering and a single white dot.
        // By leaving the context transform at the identity matrix we render directly to
        // the full-resolution canvas and copy 1-to-1 onto the display canvas, which fixes
        // the flashing.
        let dpr = window().unwrap().device_pixel_ratio();
        canvas.set_width((canvas.offset_width() as f64 * dpr) as u32);
        canvas.set_height((canvas.offset_height() as f64 * dpr) as u32);

        // Create offscreen canvas for double buffering
        let offscreen_canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;
        offscreen_canvas.set_width(canvas.width());
        offscreen_canvas.set_height(canvas.height());
        
        let offscreen_context = offscreen_canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;
        // Do **not** scale the offscreen context; it already matches the device pixel ratio.

        let decision_analyzer = DecisionAnalyzer::new();
        let lineage_tracker = LineageTracker::new();
        let consciousness_monitor = ConsciousnessMonitor::new();
        let innovation_tracker = InnovationTracker::new();
        let analytics_engine = AnalyticsEngine::new();

        Ok(EvolutionMonitor {
            canvas,
            context,
            websocket: None,
            simulation_state: SimulationState::default(),
            view_mode: ViewMode::default(),
            selected_agent: None,
            selected_lineage: None,
            time_scale: 1.0,
            particle_size_scale: 1.0,
            energy_filter_min: 0.0,
            particle_filter: ParticleFilter::default(),
            decision_analyzer,
            lineage_tracker,
            consciousness_monitor,
            innovation_tracker,
            analytics_engine,
            connected: false,
            last_frame_time: 0.0,
            fps: 0.0,
            last_render_duration: 0.0,
            agent_history: HashMap::new(),
            offscreen_canvas: Some(offscreen_canvas),
            offscreen_context: Some(offscreen_context),
            needs_redraw: true,
            target_fps: 60.0,
            frame_interval: 1000.0 / 60.0,
            debug_messages: Vec::new(),
            messages_received: 0,
            rpc_debug_messages: Vec::new(),
            rpc_messages_received: 0,
        })
    }
    
    /// Main rendering loop
    #[wasm_bindgen]
    pub fn render(&mut self) -> Result<(), JsValue> {
        let current_time = web_window()
            .performance()
            .map(|p| p.now())
            .unwrap_or(js_sys::Date::now());
        let elapsed = current_time - self.last_frame_time;

        // Only render if enough time has passed or we need to redraw
        if elapsed >= self.frame_interval || self.needs_redraw {
            let start_time = current_time;
            
            // Use offscreen canvas for rendering
            if let Some(offscreen_context) = &self.offscreen_context {
                // Clear offscreen canvas
                offscreen_context.clear_rect(0.0, 0.0, self.canvas.width() as f64, self.canvas.height() as f64);
                
                // Update HTML UI elements
                self.update_html_ui()?;
                
                // Avoid E0502: collect agent positions first
                let agent_positions: Vec<(Uuid, [f64; 3])> = self.simulation_state.agents.iter().map(|a| (a.id, a.position)).collect();
                for (id, pos) in agent_positions {
                    self.update_agent_history(id, pos);
                }
                
                // Render based on current view mode
                match self.view_mode {
                    ViewMode::ParticlePhysics => self.render_particle_physics()?,
                    ViewMode::AtomicStructure => self.render_atomic_structure()?,
                    ViewMode::MolecularDynamics => self.render_particle_physics()?, // Fallback
                    ViewMode::CelestialBodies => self.render_celestial_bodies()?,
                    ViewMode::NuclearPhysics => self.render_nuclear_physics()?,
                    ViewMode::AgentOverview => self.render_agent_overview()?,
                    ViewMode::LineageTree => self.render_lineage_tree()?,
                    ViewMode::DecisionTracking => self.render_decision_tracking()?,
                    ViewMode::ConsciousnessMap => self.render_consciousness_map()?,
                    ViewMode::InnovationTimeline => self.render_innovation_timeline()?,
                    ViewMode::SelectionPressures => self.render_selection_pressures(&Vec::new())?,
                    ViewMode::EnvironmentalMap => self.render_agent_overview()?, // Fallback
                    ViewMode::PopulationDynamics => self.render_agent_overview()?, // Fallback
                    ViewMode::QuantumFields => self.render_quantum_fields()?,
                    ViewMode::EnergyFlows => self.render_particle_physics()?, // Fallback
                    ViewMode::EmergentComplexity => self.render_agent_overview()?, // Fallback
                    ViewMode::DebugMessages => self.render_debug_messages()?,
                }
                
                // Render UI overlay
                self.render_ui_overlay()?;
                
                // Copy offscreen canvas to main canvas
                self.context.clear_rect(0.0, 0.0, self.canvas.width() as f64, self.canvas.height() as f64);
                self.context.draw_image_with_html_canvas_element(
                    self.offscreen_canvas.as_ref().unwrap(),
                    0.0,
                    0.0,
                )?;
            }
            
            // Calculate and display FPS
            let end_time = web_window()
                .performance()
                .map(|p| p.now())
                .unwrap_or(js_sys::Date::now());

            // Calculate frame time, clamping to a very small value to avoid division-by-zero
            let mut frame_time = end_time - start_time;
            if frame_time <= 0.0 {
                frame_time = 0.000_001; // 1 µs minimum
            }

            // Update FPS with a sensible upper bound to reduce flicker
            self.fps = (1000.0 / frame_time).min(240.0);
            self.last_render_duration = frame_time;
            self.last_frame_time = current_time;
            self.needs_redraw = false;
        }
        
        Ok(())
    }
    
    fn update_html_ui(&self) -> Result<(), JsValue> {
        let document = web_window().document().unwrap();
        
        // Update connection status
        if let Some(elem) = document.get_element_by_id("connection-status") {
            if self.connected {
                elem.set_inner_html("🟢 Connected");
                elem.set_class_name("connection-status connected");
            } else {
                elem.set_inner_html("🔴 Disconnected");
                elem.set_class_name("connection-status disconnected");
            }
        }
        
        // Update universe stats
        if let Some(elem) = document.get_element_by_id("universe-age") {
            elem.set_inner_html(&format!("{:.2} Gyr", self.simulation_state.universe_age_gyr));
        }
        if let Some(elem) = document.get_element_by_id("temperature") {
            elem.set_inner_html(&format!("{:.2e} K", self.simulation_state.temperature));
        }
        if let Some(elem) = document.get_element_by_id("particle-count") {
            elem.set_inner_html(&format!("{}", self.simulation_state.particles.len()));
        }
        if let Some(elem) = document.get_element_by_id("agent-count") {
            elem.set_inner_html(&format!("{}", self.simulation_state.agents.len()));
        }
        
        // Update consciousness level
        let avg_consciousness: f64 = self.simulation_state.agents.iter()
            .map(|a| a.consciousness_level)
            .sum::<f64>() / self.simulation_state.agents.len() as f64;
        if let Some(elem) = document.get_element_by_id("consciousness-level") {
            elem.set_inner_html(&format!("{:.1}%", avg_consciousness * 100.0));
        }
        
        // Update performance metrics
        if let Some(elem) = document.get_element_by_id("fps") {
            elem.set_inner_html(&format!("{:.1}", self.fps));
        }
        if let Some(elem) = document.get_element_by_id("render-time") {
            elem.set_inner_html(&format!("{:.1}ms", self.last_render_duration));
        }
        if let Some(elem) = document.get_element_by_id("ups") {
            let ups_val = if self.last_render_duration > 0.0 {
                (1000.0 / self.last_render_duration).min(1000.0)
            } else { 0.0 };
            elem.set_inner_html(&format!("{:.1}", ups_val));
        }
        
        // Update current tick and cosmic era
        if let Some(elem) = document.get_element_by_id("current-tick") {
            elem.set_inner_html(&format!("{}", self.simulation_state.current_tick));
        }
        if let Some(elem) = document.get_element_by_id("cosmic-era") {
            elem.set_inner_html(&format!("{:?}", self.simulation_state.cosmic_era));
        }
        
        // Update energy density
        if let Some(elem) = document.get_element_by_id("energy-density") {
            elem.set_inner_html(&format!("{:.2e} J/m³", self.simulation_state.energy_density));
        }
        
        // Update evolution statistics
        if let Some(elem) = document.get_element_by_id("active-lineages") {
            elem.set_inner_html(&format!("{}", self.simulation_state.population_stats.active_lineages));
        }
        if let Some(elem) = document.get_element_by_id("extinct-lineages") {
            elem.set_inner_html(&format!("{}", self.simulation_state.population_stats.extinct_lineages));
        }
        if let Some(elem) = document.get_element_by_id("average-fitness") {
            elem.set_inner_html(&format!("{:.2}", self.simulation_state.population_stats.average_fitness));
        }
        if let Some(elem) = document.get_element_by_id("genetic-diversity") {
            elem.set_inner_html(&format!("{:.2}", self.simulation_state.population_stats.genetic_diversity));
        }
        
        // Update innovation rate
        let innovation_rate = self.simulation_state.innovations.len() as f64 / 
            (self.simulation_state.current_tick as f64).max(1.0);
        if let Some(elem) = document.get_element_by_id("innovation-rate") {
            elem.set_inner_html(&format!("{:.2}/tick", innovation_rate));
        }
        
        // Update atomic physics data
        if let Some(elem) = document.get_element_by_id("nuclei-count") {
            elem.set_inner_html(&format!("{}", self.simulation_state.nuclei.len()));
        }
        if let Some(elem) = document.get_element_by_id("atoms-count") {
            elem.set_inner_html(&format!("{}", self.simulation_state.atoms.len()));
        }
        if let Some(elem) = document.get_element_by_id("molecules-count") {
            elem.set_inner_html(&format!("{}", self.simulation_state.molecules.len()));
        }
        
        // Update celestial bodies data
        if let Some(elem) = document.get_element_by_id("celestial-count") {
            elem.set_inner_html(&format!("{}", self.simulation_state.celestial_bodies.len()));
        }
        
        // Count stars and planets
        let star_count = self.simulation_state.celestial_bodies.iter()
            .filter(|body| body.body_type == "Star")
            .count();
        let planet_count = self.simulation_state.celestial_bodies.iter()
            .filter(|body| body.body_type == "Planet")
            .count();
            
        if let Some(elem) = document.get_element_by_id("star-count") {
            elem.set_inner_html(&format!("{}", star_count));
        }
        if let Some(elem) = document.get_element_by_id("planet-count") {
            elem.set_inner_html(&format!("{}", planet_count));
        }
        
        // Update quantum fields list
        if let Some(elem) = document.get_element_by_id("quantum-fields-list") {
            if self.simulation_state.quantum_fields.is_empty() {
                elem.set_inner_html("<p>No fields detected</p>");
            } else {
                let mut fields_html = String::new();
                for field_name in self.simulation_state.quantum_fields.keys() {
                    fields_html.push_str(&format!("<div class=\"metric\"><span class=\"metric-label\">{}</span><span class=\"metric-value\">Active</span></div>", field_name));
                }
                elem.set_inner_html(&fields_html);
            }
        }

        // Update system performance
        if let Some(elem) = document.get_element_by_id("updates-per-second") {
            let ups_val = if self.last_render_duration > 0.0 {
                (1000.0 / self.last_render_duration).min(1000.0)
            } else { 0.0 };
            elem.set_inner_html(&format!("{:.1}", ups_val));
        }
        if let Some(elem) = document.get_element_by_id("memory-usage") {
            // TODO: Implement actual memory usage tracking
            elem.set_inner_html("0 MB");
        }
        if let Some(elem) = document.get_element_by_id("cpu-usage") {
            // TODO: Implement actual CPU usage tracking
            elem.set_inner_html("0%");
        }
        
        // Update debug section
        if let Some(elem) = document.get_element_by_id("messages-received") {
            elem.set_inner_html(&format!("{}", self.messages_received));
        }
        if let Some(elem) = document.get_element_by_id("connection-state") {
            elem.set_inner_html(if self.connected { "connected" } else { "disconnected" });
        }
        
        if let Some(elem) = document.get_element_by_id("last-message-type") {
            if let Some(last_msg) = self.debug_messages.last() {
                elem.set_inner_html(&last_msg.message_type);
            }
        }
        if let Some(elem) = document.get_element_by_id("last-message-size") {
            if let Some(last_msg) = self.debug_messages.last() {
                elem.set_inner_html(&format!("{} bytes", last_msg.size_bytes));
            }
        }
        
        // Update debug messages list
        if let Some(elem) = document.get_element_by_id("debug-messages") {
            let mut html = String::new();
            let recent_messages = self.debug_messages.iter().rev().take(20);
            for msg in recent_messages {
                let color = if msg.parse_success { "#00ff00" } else { "#ff0000" };
                let timestamp_str = format!("{:.1}s", msg.timestamp / 1000.0);
                html.push_str(&format!(
                    "<div style='color: {}; margin-bottom: 2px;'>[{}] {} ({} bytes)</div>",
                    color, timestamp_str, msg.message_type, msg.size_bytes
                ));
                if let Some(error) = &msg.error {
                    html.push_str(&format!(
                        "<div style='color: #ff6666; margin-left: 10px; font-size: 9px;'>Error: {}</div>",
                        error
                    ));
                }

                // Show full decoded JSON/content for the message
                html.push_str(&format!(
                    "<pre style='color: #cccccc; margin-left: 15px; font-size: 9px; white-space: pre-wrap;'>{}</pre>",
                    msg.content.replace('<', "&lt;").replace('>', "&gt;")
                ));
            }
            if html.is_empty() {
                html = "<div style='color: #888;'>No messages yet...</div>".to_string();
            }
            elem.set_inner_html(&html);
        }
        
        // Update raw message data
        if let Some(elem) = document.get_element_by_id("raw-message-data") {
            if let Some(last_msg) = self.debug_messages.last() {
                let content = format!("Content: {}\nRaw: {}", last_msg.content, last_msg.raw_data);
                elem.set_inner_html(&format!("<pre style='white-space: pre-wrap;'>{}</pre>", content));
            } else {
                elem.set_inner_html("<div style='color: #888;'>No raw data yet...</div>");
            }
        }
        
        // Update RPC debug messages list
        if let Some(elem) = document.get_element_by_id("rpc-debug-messages") {
            let mut html = String::new();
            let recent_messages = self.rpc_debug_messages.iter().rev().take(20);
            for msg in recent_messages {
                let color = if msg.parse_success { "#00ffff" } else { "#ff6600" };
                let timestamp_str = format!("{:.1}s", msg.timestamp / 1000.0);
                html.push_str(&format!(
                    "<div style='color: {}; margin-bottom: 2px;'>[{}] {} ({} bytes)</div>",
                    color, timestamp_str, msg.message_type, msg.size_bytes
                ));
                html.push_str(&format!(
                    "<pre style='color: #cccccc; margin-left: 15px; font-size: 9px; white-space: pre-wrap;'>{}</pre>",
                    msg.content.replace('<', "&lt;").replace('>', "&gt;")
                ));
            }
            if html.is_empty() {
                html = "<div style='color: #888;'>No RPC messages yet...</div>".to_string();
            }
            elem.set_inner_html(&html);
        }
        
        // Update RPC message count
        if let Some(elem) = document.get_element_by_id("rpc-messages-received") {
            elem.set_inner_html(&format!("{}", self.rpc_messages_received));
        }
        
        Ok(())
    }
    
    /// Render fundamental particle physics visualization
    fn render_particle_physics(&mut self) -> Result<(), JsValue> {
        // console_log!("Rendering particle physics - {} particles", 
        //             self.simulation_state.particles.len());
        
        // Pre-calculate all values that need immutable borrows
        let particle_size_scale = self.particle_size_scale;
        let energy_filter_min = self.energy_filter_min;
        let particles = if self.simulation_state.particles.is_empty() {
            // Create some demo particles for display
            vec![
                ParticleVisualization {
                    id: 1,
                    particle_type: "Electron".to_string(),
                    position: [0.0, 0.0, 0.0],
                    momentum: [1e-20, 0.0, 0.0],
                    energy: 8.187e-14,
                    mass: 9.109e-31,
                    charge: -1.602e-19,
                    spin: [0.5, 0.0, 0.0],
                    color_charge: None,
                    interaction_count: 0,
                    age: 0.0,
                    decay_probability: 0.0,
                },
                ParticleVisualization {
                    id: 2,
                    particle_type: "Proton".to_string(),
                    position: [1e-15, 0.0, 0.0],
                    momentum: [-1e-20, 0.0, 0.0],
                    energy: 1.503e-10,
                    mass: 1.673e-27,
                    charge: 1.602e-19,
                    spin: [0.5, 0.0, 0.0],
                    color_charge: None,
                    interaction_count: 2,
                    age: 1.0,
                    decay_probability: 0.0,
                },
                ParticleVisualization {
                    id: 3,
                    particle_type: "Photon".to_string(),
                    position: [2e-15, 1e-15, 0.0],
                    momentum: [3e-19, 0.0, 0.0],
                    energy: 3.3e-19,
                    mass: 0.0,
                    charge: 0.0,
                    spin: [1.0, 0.0, 0.0],
                    color_charge: None,
                    interaction_count: 1,
                    age: 0.5,
                    decay_probability: 0.0,
                },
            ]
        } else {
            self.simulation_state.particles.clone()
        };
        
        // Pre-calculate all particle rendering data including trails and interactions
        let particle_data: Vec<_> = particles.iter()
            .filter(|p| p.energy >= energy_filter_min)
            .map(|particle| {
                let (r, g, b, a) = self.get_particle_color(&particle.particle_type);
                let color_str = format!("rgba({}, {}, {}, {})", r, g, b, a);
                let (x, y) = self.map_3d_to_2d(particle.position);
                let size = self.get_particle_size(particle.mass) * particle_size_scale;
                
                // Pre-calculate trail points if needed
                let trail_points = if particle.energy > 1e-13 {
                    self.calculate_trail_points(particle)
                } else {
                    Vec::new()
                };
                
                // Pre-calculate interaction points if needed
                let interaction_points = if particle.interaction_count > 0 {
                    self.calculate_interaction_points(particle)
                } else {
                    Vec::new()
                };
                
                (particle.clone(), color_str, x, y, size, trail_points, interaction_points, (r, g, b))
            })
            .collect();
        
        // Now use the context to render everything
        let context = self.get_context_mut();
        
        // Render particles and their effects
        for (particle, color_str, x, y, size, trail_points, interaction_points, (r, g, b)) in &particle_data {
            // Draw particle
            context.set_fill_style_str(color_str);
            context.begin_path();
            context.arc(*x, *y, *size, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            // Draw trail if any
            if !trail_points.is_empty() {
                context.set_stroke_style_str(&format!("rgba({}, {}, {}, {})", r, g, b, 0.3));
                context.begin_path();
                context.move_to(trail_points[0].0, trail_points[0].1);
                for point in trail_points.iter().skip(1) {
                    context.line_to(point.0, point.1);
                }
                context.stroke();
            }
            
            // Draw interaction lines if any
            if !interaction_points.is_empty() {
                context.set_stroke_style_str(&format!("rgba({}, {}, {}, {})", r, g, b, 0.5));
                for point in interaction_points {
                    context.begin_path();
                    context.move_to(*x, *y);
                    context.line_to(point.0, point.1);
                    context.stroke();
                }
            }
        }
        
        // Render quantum field fluctuations
        self.render_quantum_field_overlay()?;
        
        // Render celestial bodies (stars, planets, etc.)
        self.render_celestial_bodies()?;
        
        Ok(())
    }
    
    /// Render AI agent overview with comprehensive tracking
    fn render_agent_overview(&mut self) -> Result<(), JsValue> {
        let context = self.get_context_mut();
        
        // Collect agent data first
        let agent_data: Vec<_> = self.simulation_state.agents.iter().map(|agent| {
            let (x, y) = self.map_3d_to_2d(agent.position);
            (agent.clone(), x, y)
        }).collect();

        // Render each agent
        for (agent, x, y) in &agent_data {
            self.draw_agent(agent, *x, *y)?;
            
            if let Some(selected) = self.selected_agent {
                if agent.id == selected {
                    self.highlight_selected_agent(agent, *x, *y)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Render comprehensive lineage tree visualization
    fn render_lineage_tree(&mut self) -> Result<(), JsValue> {
        console_log!("Rendering lineage trees for {} lineages", 
                    self.simulation_state.lineages.len());
        
        let mut y_offset = 50.0;
        
        for (lineage_id, lineage) in self.simulation_state.lineages.clone() {
            // Draw lineage header
            self.context.set_font("16px Arial");
            self.context.set_fill_style_str("white");
            self.context.fill_text(&format!(
                "Lineage {} - Gen {} - {} living", 
                lineage_id.to_string()[..8].to_string(),
                lineage.current_generation,
                lineage.living_individuals
            ), 10.0, y_offset)?;
            
            y_offset += 30.0;
            
            // Draw family tree structure
            self.render_family_tree(&lineage.family_tree, 10.0, y_offset)?;
            
            // Draw evolution timeline
            self.render_evolution_timeline(&lineage.evolution_timeline, y_offset + 100.0)?;
            
            // Draw innovation milestones
            self.render_innovation_milestones(&lineage.major_innovations, y_offset + 150.0)?;
            
            y_offset += 200.0;
        }
        
        Ok(())
    }
    
    /// Render decision tracking visualization
    fn render_decision_tracking(&mut self) -> Result<(), JsValue> {
        // Create decision flow diagram
        let mut x_pos = 50.0;
        let y_base = 200.0;
        
        for decision in self.simulation_state.decisions.clone() {
            let (x, y) = (x_pos, y_base + (decision.fitness_impact * 100.0));
            
            // Color based on success/failure
            let color = if decision.success { "green" } else { "red" };
            self.context.set_fill_style_str(color);
            
            // Draw decision point
            self.context.begin_path();
            self.context.arc(x, y, 5.0, 0.0, 2.0 * std::f64::consts::PI)?;
            self.context.fill();
            
            // Draw decision type label
            self.context.set_font("10px Arial");
            self.context.set_fill_style_str("white");
            self.context.fill_text(&decision.decision_type, x - 20.0, y - 10.0)?;
            
            // Draw impact line
            self.context.set_stroke_style_str("rgba(0,255,255,0.8)");
            self.context.set_line_width(2.0);
            self.context.begin_path();
            self.context.move_to(x, y_base);
            self.context.line_to(x, y);
            self.context.stroke();
            
            x_pos += 20.0;
        }
        
        Ok(())
    }
    
    /// Render consciousness emergence map
    fn render_consciousness_map(&mut self) -> Result<(), JsValue> {
        // Create consciousness landscape
        for agent in self.simulation_state.agents.clone() {
            let (x, y) = self.map_3d_to_2d(agent.position);
            let consciousness = agent.consciousness_level;
            
            if consciousness > 0.01 {
                // Draw consciousness "height" as vertical bars
                let height = consciousness * 100.0;
                let alpha = consciousness;
                
                self.context.set_stroke_style_str(&format!(
                    "rgba(255, 255, 0, {})", alpha
                ));
                self.context.set_line_width(2.0);
                
                self.context.begin_path();
                self.context.move_to(x, y);
                self.context.line_to(x, y - height);
                self.context.stroke();
                
                // Draw consciousness level indicator
                self.context.set_fill_style_str(&format!(
                    "rgba(255, 255, 0, {})", alpha
                ));
                self.context.begin_path();
                self.context.arc(x, y - height, 3.0, 0.0, 2.0 * std::f64::consts::PI)?;
                self.context.fill();
            }
        }
        
        Ok(())
    }
    
    /// Render innovation timeline
    fn render_innovation_timeline(&mut self) -> Result<(), JsValue> {
        let timeline_y = 400.0;
        let timeline_start = 50.0;
        let timeline_width = self.canvas.width() as f64 - 100.0;
        
        // Draw timeline base
        self.context.set_stroke_style_str("white");
        self.context.set_line_width(2.0);
        self.context.begin_path();
        self.context.move_to(timeline_start, timeline_y);
        self.context.line_to(timeline_start + timeline_width, timeline_y);
        self.context.stroke();
        
        // Plot innovations
        for innovation in self.simulation_state.innovations.clone() {
            let time_ratio = innovation.timestamp as f64 / self.simulation_state.current_tick as f64;
            let x = timeline_start + (time_ratio * timeline_width);
            let impact_height = innovation.impact_score * 50.0;
            
            // Innovation marker
            self.context.set_fill_style_str("cyan");
            self.context.begin_path();
            self.context.arc(x, timeline_y - impact_height, 4.0, 0.0, 2.0 * std::f64::consts::PI)?;
            self.context.fill();
            
            // Innovation label
            self.context.set_font("8px Arial");
            self.context.set_fill_style_str("white");
            self.context.fill_text(&innovation.innovation_type, x - 15.0, timeline_y - impact_height - 8.0)?;
        }
        
        Ok(())
    }

    /// Render atomic structure visualization
    fn render_atomic_structure(&mut self) -> Result<(), JsValue> {
        // Pre-calculate values before getting context
        let canvas_width = self.canvas.width() as f64;
        let canvas_height = self.canvas.height() as f64;
        let atoms = self.simulation_state.atoms.clone();
        let atom_count = atoms.len();
        
        let context = self.get_context_mut();
        
        // Background
        context.set_fill_style_str("#0F0F23");
        context.fill_rect(0.0, 0.0, canvas_width, canvas_height);
        
        // Title
        context.set_fill_style_str("#00FFFF");
        context.set_font("18px Arial");
        context.fill_text("Atomic Structure Visualization", 20.0, 30.0)?;
        
        // Display atom count
        context.set_fill_style_str("#FFFFFF");
        context.set_font("14px Arial");
        context.fill_text(&format!("Atoms: {}", atom_count), 20.0, 60.0)?;
        
        // Render atoms as orbital models (with fallback dummy data if no atoms)
        let atoms_to_render = if atoms.is_empty() {
            vec!["H1(e-=1, E=1.23e-18J)".to_string(), "He2(e-=2, E=2.46e-18J)".to_string(), "C6(e-=6, E=7.38e-18J)".to_string()]
        } else {
            atoms
        };
        
        for (i, atom_str) in atoms_to_render.iter().enumerate().take(20) {
            let x = 50.0 + ((i % 5) as f64) * 150.0;
            let y = 100.0 + ((i / 5) as f64) * 120.0;
            
            // Parse atom info from string (assuming format like "H1(e-=1, E=1.23e-18J)")
            let nucleus_color = if atom_str.starts_with('H') {
                "rgba(255, 100, 100, 0.8)"
            } else if atom_str.starts_with("He") {
                "rgba(100, 255, 100, 0.8)"  
            } else if atom_str.starts_with('C') {
                "rgba(100, 100, 255, 0.8)"
            } else {
                "rgba(200, 200, 200, 0.8)"
            };
            
            // Draw nucleus
            context.set_fill_style_str(nucleus_color);
            context.begin_path();
            context.arc(x, y, 8.0, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            // Draw electron orbitals
            context.set_stroke_style_str("rgba(0, 255, 255, 0.3)");
            context.set_line_width(1.0);
            for orbital in 0..3 {
                let radius = 15.0 + (orbital as f64 * 10.0);
                context.begin_path();
                context.arc(x, y, radius, 0.0, 2.0 * std::f64::consts::PI)?;
                context.stroke();
            }
            
            // Draw electrons
            context.set_fill_style_str("rgba(255, 255, 0, 0.8)");
            let electron_angle = (i as f64) * 0.5;
            for orbital in 0..2 {
                let radius = 20.0 + (orbital as f64 * 15.0);
                let ex = x + radius * (electron_angle + (orbital as f64)).cos();
                let ey = y + radius * (electron_angle + (orbital as f64)).sin();
                context.begin_path();
                context.arc(ex, ey, 3.0, 0.0, 2.0 * std::f64::consts::PI)?;
                context.fill();
            }
            
            // Label
            context.set_fill_style_str("#FFFFFF");
            context.set_font("10px Arial");
            let short_label = if atom_str.len() > 25 { &atom_str[..25] } else { atom_str };
            context.fill_text(short_label, x - 30.0, y + 50.0)?;
        }
        
        Ok(())
    }

    /// Render nuclear physics visualization
    fn render_nuclear_physics(&mut self) -> Result<(), JsValue> {
        // Pre-calculate values before getting context
        let canvas_width = self.canvas.width() as f64;
        let canvas_height = self.canvas.height() as f64;
        let nuclei = self.simulation_state.nuclei.clone();
        let nuclei_count = nuclei.len();
        
        let context = self.get_context_mut();
        
        // Background
        context.set_fill_style_str("#1A0A0A");
        context.fill_rect(0.0, 0.0, canvas_width, canvas_height);
        
        // Title
        context.set_fill_style_str("#FF6B6B");
        context.set_font("18px Arial");
        context.fill_text("Nuclear Physics", 20.0, 30.0)?;
        
        // Display nucleus count
        context.set_fill_style_str("#FFFFFF");
        context.set_font("14px Arial");
        context.fill_text(&format!("Nuclei: {}", nuclei_count), 20.0, 60.0)?;
        
        // Render nuclei (with fallback dummy data if no nuclei)
        let nuclei_to_render = if nuclei.is_empty() {
            vec!["H1(A=1, Z=1, BE=1.23e-12J)".to_string(), "He4(A=4, Z=2, BE=4.92e-12J)".to_string(), "C12(A=12, Z=6, BE=14.76e-12J)".to_string(), "Fe56(A=56, Z=26, BE=69.12e-12J)".to_string()]
        } else {
            nuclei
        };
        
        for (i, nucleus_str) in nuclei_to_render.iter().enumerate().take(15) {
            let x = 60.0 + ((i % 4) as f64) * 180.0;
            let y = 100.0 + ((i / 4) as f64) * 120.0;
            
            // Parse nucleus info (format like "H1(A=1, Z=1, BE=1.23e-12J)")
            let element_color = if nucleus_str.starts_with('H') {
                "rgba(255, 100, 100, 0.9)"
            } else if nucleus_str.starts_with("He") {
                "rgba(255, 200, 100, 0.9)"
            } else if nucleus_str.starts_with("Li") {
                "rgba(100, 255, 100, 0.9)"
            } else if nucleus_str.starts_with('C') {
                "rgba(100, 100, 255, 0.9)"
            } else if nucleus_str.starts_with('O') {
                "rgba(255, 100, 255, 0.9)"
            } else if nucleus_str.starts_with("Fe") {
                "rgba(255, 150, 50, 0.9)"
            } else {
                "rgba(150, 150, 150, 0.9)"
            };
            
            // Draw nucleus with binding energy visualization
            context.set_fill_style_str(element_color);
            context.begin_path();
            context.arc(x, y, 12.0, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            // Add glow effect for high binding energy
            context.set_fill_style_str("rgba(255, 255, 255, 0.3)");
            context.begin_path();
            context.arc(x, y, 18.0, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            // Draw proton/neutron visualization
            context.set_fill_style_str("rgba(255, 0, 0, 0.7)"); // Protons in red
            context.begin_path();
            context.arc(x - 3.0, y - 3.0, 4.0, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            context.set_fill_style_str("rgba(0, 0, 255, 0.7)"); // Neutrons in blue
            context.begin_path();
            context.arc(x + 3.0, y + 3.0, 4.0, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            // Label with nucleus info
            context.set_fill_style_str("#FFFFFF");
            context.set_font("9px Arial");
            let short_label = if nucleus_str.len() > 25 { &nucleus_str[..25] } else { nucleus_str };
            context.fill_text(short_label, x - 40.0, y + 35.0)?;
        }
        
        Ok(())
    }
    
    /// Get color for particle type
    fn get_particle_color(&self, particle_type: &str) -> (u8, u8, u8, f32) {
        match particle_type {
            "Electron" => (255, 255, 0, 1.0),     // Yellow
            "Proton" => (255, 0, 0, 1.0),         // Red  
            "Neutron" => (128, 128, 128, 1.0),    // Gray
            "Photon" => (255, 255, 255, 0.7),     // White, translucent
            "Up" | "Down" => (0, 255, 0, 1.0),    // Green (quarks)
            "Gluon" => (255, 0, 255, 0.8),        // Magenta
            "Higgs" => (255, 215, 0, 1.0),        // Gold
            _ => (128, 128, 255, 0.8),             // Light blue (other)
        }
    }
    
    /// Map 3D position to 2D screen coordinates
    fn map_3d_to_2d(&self, pos: [f64; 3]) -> (f64, f64) {
        // Convert simulation metres into on-screen pixels.
        // Empirically, initial Big-Bang positions are ~1e14 m.
        // A factor of 1e-12 maps 1e14 m → 100 px which keeps everything on-screen.
        let scale = 1e-12;
        let center_x = self.canvas.width() as f64 / 2.0;
        let center_y = self.canvas.height() as f64 / 2.0;
        
        let x = center_x + pos[0] * scale;
        let y = center_y + pos[1] * scale;
        
        (x, y)
    }
    
    /// Set view mode
    #[wasm_bindgen]
    pub fn set_view_mode(&mut self, mode: u8) {
        self.view_mode = match mode {
            0  => ViewMode::ParticlePhysics,
            1  => ViewMode::AtomicStructure,
            2  => ViewMode::MolecularDynamics,
            3  => ViewMode::CelestialBodies,
            4  => ViewMode::NuclearPhysics,
            5  => ViewMode::AgentOverview,
            6  => ViewMode::LineageTree,
            7  => ViewMode::DecisionTracking,
            8  => ViewMode::ConsciousnessMap,
            9  => ViewMode::InnovationTimeline,
            10 => ViewMode::SelectionPressures,
            11 => ViewMode::EnvironmentalMap,
            12 => ViewMode::PopulationDynamics,
            13 => ViewMode::QuantumFields,
            14 => ViewMode::EnergyFlows,
            15 => ViewMode::EmergentComplexity,
            16 => ViewMode::DebugMessages,
            _  => ViewMode::AgentOverview,
        };

        // Update debug section visibility
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(debug_section) = doc.get_element_by_id("debug-section") {
                if self.view_mode == ViewMode::DebugMessages {
                    debug_section.set_attribute("style", "display: block;").ok();
                } else {
                    debug_section.set_attribute("style", "display: none;").ok();
                }
            }
        }

        // Request a new frame so that the freshly-selected view is rendered immediately.
        self.needs_redraw = true;
        console_log!("View mode changed to: {:?}", self.view_mode);
    }
    
    /// Select an agent for detailed tracking
    #[wasm_bindgen]
    pub fn select_agent(&mut self, agent_id: &str) {
        if let Ok(uuid) = Uuid::parse_str(agent_id) {
            self.selected_agent = Some(uuid);
            console_log!("Selected agent: {}", agent_id);
        }
    }
    
    /// Get the current simulation state as JSON for JavaScript access
    #[wasm_bindgen]
    pub fn get_simulation_state_json(&self) -> String {
        match serde_json::to_string(&self.simulation_state) {
            Ok(json) => json,
            Err(_) => "{}".to_string(),
        }
    }
    
    /// Check if the WebSocket is connected
    #[wasm_bindgen]
    pub fn is_connected(&self) -> bool {
        self.connected
    }
    
    // Placeholder implementations for complex rendering methods
    fn get_particle_size(&self, mass: f64) -> f64 { 2.0 + (mass / 1e-27).log10().max(0.0) }
    fn draw_particle_trail(&mut self, particle: &ParticleVisualization) -> Result<(), JsValue> {
        let (x, y) = self.map_3d_to_2d(particle.position);
        let (r, g, b, a) = self.get_particle_color(&particle.particle_type);
        self.context.set_stroke_style(&format!("rgba({}, {}, {}, {})", r, g, b, a * 0.5).into());
        self.context.begin_path();
        self.context.move_to(x - 20.0, y);
        self.context.line_to(x, y);
        self.context.stroke();
        Ok(())
    }
    fn draw_interaction_indicators(&mut self, particle: &ParticleVisualization) -> Result<(), JsValue> {
        let (x, y) = self.map_3d_to_2d(particle.position);
        self.context.set_fill_style_str("rgba(255, 255, 255, 0.8)");
        self.context.set_font("10px Arial");
        self.context.fill_text(&format!("{}", particle.interaction_count), x + 5.0, y - 5.0)?;
        self.context.set_stroke_style_str("rgba(255, 255, 255, 0.3)");
        self.context.set_line_width(1.0);
        self.context.begin_path();
        self.context.arc(x, y, 10.0, 0.0, 2.0 * std::f64::consts::PI)?;
        self.context.stroke();
        Ok(())
    }
    fn render_quantum_field_overlay(&mut self) -> Result<(), JsValue> {
        // Draw quantum field fluctuations
        for (field_name, _field) in &self.simulation_state.quantum_fields {
            let color = match field_name.as_str() {
                "Higgs" => "rgba(255, 215, 0, 0.1)",
                "Electromagnetic" => "rgba(0, 255, 255, 0.1)",
                "Strong" => "rgba(255, 0, 0, 0.1)",
                "Weak" => "rgba(255, 0, 255, 0.1)",
                _ => "rgba(255, 255, 255, 0.1)",
            };
            self.context.set_fill_style_str(color);
            self.context.fill_rect(0.0, 0.0, self.canvas.width() as f64, self.canvas.height() as f64);
        }
        Ok(())
    }
    fn render_quantum_fields(&mut self) -> Result<(), JsValue> {
        // Draw quantum field visualization
        for (field_name, _field) in &self.simulation_state.quantum_fields {
            let color = match field_name.as_str() {
                "Higgs" => "rgba(255, 215, 0, 0.5)",
                "Electromagnetic" => "rgba(0, 255, 255, 0.5)",
                "Strong" => "rgba(255, 0, 0, 0.5)",
                "Weak" => "rgba(255, 0, 255, 0.5)",
                _ => "rgba(255, 255, 255, 0.5)",
            };
            self.context.set_fill_style_str(color);
            self.context.set_font("12px Arial");
            self.context.fill_text(field_name, 10.0, 20.0)?;
            // Draw field strength indicator
            self.context.set_stroke_style_str(color);
            self.context.set_line_width(2.0);
            self.context.begin_path();
            self.context.move_to(10.0, 30.0);
            self.context.line_to(110.0, 30.0);
            self.context.stroke();
        }
        Ok(())
    }

    fn render_debug_messages(&mut self) -> Result<(), JsValue> {
        let canvas_width = self.canvas.width() as f64;
        let canvas_height = self.canvas.height() as f64;
        
        // Set dark background for canvas
        self.context.set_fill_style_str("#000000");
        self.context.fill_rect(0.0, 0.0, canvas_width, canvas_height);
        
        // Draw main debug information on canvas
        self.context.set_fill_style_str("#00ffff");
        self.context.set_font("20px monospace");
        
        let mut y_offset = 40.0;
        let line_height = 30.0;
        
        // Header
        self.context.fill_text("🐛 Debug Messages Mode", 20.0, y_offset)?;
        y_offset += line_height * 1.5;
        
        // Connection and message stats
        self.context.set_font("16px monospace");
        self.context.set_fill_style_str("#ffffff");
        
        self.context.fill_text(&format!("Connection Status: {}", if self.connected { "✅ Connected" } else { "❌ Disconnected" }), 20.0, y_offset)?;
        y_offset += line_height;
        
        self.context.fill_text(&format!("WebSocket Messages: {}", self.messages_received), 20.0, y_offset)?;
        y_offset += line_height;
        
        self.context.fill_text(&format!("RPC Messages: {}", self.rpc_messages_received), 20.0, y_offset)?;
        y_offset += line_height;
        
        self.context.fill_text(&format!("Debug Messages Stored: {}", self.debug_messages.len()), 20.0, y_offset)?;
        y_offset += line_height * 1.5;
        
        // Instructions
        self.context.set_fill_style_str("#00ff00");
        self.context.set_font("14px monospace");
        self.context.fill_text("📋 Detailed message logs are shown in the right panel", 20.0, y_offset)?;
        y_offset += line_height;
        self.context.fill_text("🔄 Messages are updated in real-time as they arrive", 20.0, y_offset)?;
        y_offset += line_height;
        
        // Recent message summary on canvas
        if !self.debug_messages.is_empty() {
            y_offset += line_height;
            self.context.set_fill_style_str("#ffff00");
            self.context.fill_text("Latest WebSocket Messages:", 20.0, y_offset)?;
            y_offset += line_height;
            
            self.context.set_font("12px monospace");
            let recent_messages = self.debug_messages.iter().rev().take(8);
            for msg in recent_messages {
                if y_offset > canvas_height - 40.0 {
                    break;
                }
                
                let color = if msg.parse_success { "#00ff00" } else { "#ff0000" };
                self.context.set_fill_style_str(color);
                
                let timestamp_str = format!("{:.1}s", msg.timestamp / 1000.0);
                let msg_summary = format!("[{}] {} ({} bytes)", timestamp_str, msg.message_type, msg.size_bytes);
                self.context.fill_text(&msg_summary, 30.0, y_offset)?;
                y_offset += 18.0;
            }
        }
        
        if !self.rpc_debug_messages.is_empty() {
            y_offset += line_height;
            self.context.set_fill_style_str("#00ffff");
            self.context.fill_text("Latest RPC Messages:", 20.0, y_offset)?;
            y_offset += line_height;
            
            self.context.set_font("12px monospace");
            let recent_messages = self.rpc_debug_messages.iter().rev().take(5);
            for msg in recent_messages {
                if y_offset > canvas_height - 40.0 {
                    break;
                }
                
                self.context.set_fill_style_str("#00ffff");
                let timestamp_str = format!("{:.1}s", msg.timestamp / 1000.0);
                let msg_summary = format!("[{}] {} ({} bytes)", timestamp_str, msg.message_type, msg.size_bytes);
                self.context.fill_text(&msg_summary, 30.0, y_offset)?;
                y_offset += 18.0;
            }
        }
        
        // Update the HTML UI to ensure debug panel content is current
        self.update_html_ui()?;
        
        Ok(())
    }

    fn render_selection_pressures(&mut self, _: &Vec<SelectionPressureVisualization>) -> Result<(), JsValue> { Ok(()) }
    fn draw_consciousness_indicator(&mut self, agent: &AgentVisualization, x: f64, y: f64) -> Result<(), JsValue> {
        if let Some(_consciousness) = &agent.consciousness {
            self.context.set_fill_style_str("#9C27B0");
            self.context.begin_path();
            self.context.arc(x, y, 8.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.context.fill();
        }
        Ok(())
    }
    fn draw_innovation_aura(&mut self, agent: &AgentVisualization, x: f64, y: f64) -> Result<(), JsValue> {
        if let Some(_innovation) = &agent.innovation {
            self.context.set_fill_style_str("#FF9800");
            self.context.begin_path();
            self.context.arc(x, y, 10.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.context.fill();
        }
        Ok(())
    }
    fn draw_decision_trail(&mut self, agent: &AgentVisualization) -> Result<(), JsValue> {
        if let Some(history) = self.agent_history.get(&agent.id) {
            self.context.begin_path();
            self.context.set_stroke_style_str("#4CAF50");
            self.context.set_line_width(1.0);
            if let Some((first_x, first_y)) = history.positions.first().map(|p| self.map_3d_to_2d(*p)) {
                self.context.move_to(first_x, first_y);
                for pos in history.positions.iter().skip(1) {
                    let (x, y) = self.map_3d_to_2d(*pos);
                    self.context.line_to(x, y);
                }
                self.context.stroke();
            }
        }
        Ok(())
    }
    fn highlight_selected_agent(&mut self, _agent: &AgentVisualization, x: f64, y: f64) -> Result<(), JsValue> {
        self.context.set_stroke_style_str("#FF0000");
        self.context.set_line_width(2.0);
        self.context.begin_path();
        self.context.arc(x, y, 15.0, 0.0, std::f64::consts::PI * 2.0)?;
        self.context.stroke();
        Ok(())
    }
    fn render_family_tree(&mut self, _tree: &FamilyTreeVisualization, _x: f64, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_evolution_timeline(&mut self, _timeline: &Vec<EvolutionTimelineEvent>, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_innovation_milestones(&mut self, _innovations: &Vec<InnovationRecord>, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_ui_overlay(&mut self) -> Result<(), JsValue> {
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn set_particle_size_scale(&mut self, scale: f64) { self.particle_size_scale = scale; }
    #[wasm_bindgen]
    pub fn set_energy_filter_min(&mut self, min_kev: f64) { self.energy_filter_min = min_kev * 1.60218e-16; }

    pub fn reset(&mut self) {
        self.particle_size_scale = 1.0;
        self.energy_filter_min = 0.0;
        self.view_mode = ViewMode::default();
        self.selected_agent = None;
        self.selected_lineage = None;
        console_log!("EvolutionMonitor state has been reset.");
    }

    fn set_fill_style(&self, color: &str) -> Result<(), JsValue> {
        self.context.set_fill_style_str(color);
        Ok(())
    }

    fn set_stroke_style(&self, color: &str) -> Result<(), JsValue> {
        self.context.set_stroke_style(&color.into());
        Ok(())
    }

    fn draw_particle(&mut self, particle: &ParticleVisualization) -> Result<(), JsValue> {
        let (x, y) = self.map_3d_to_2d(particle.position);
        let size = self.get_particle_size(particle.mass) * self.particle_size_scale;
        let (r, g, b, a) = self.get_particle_color(&particle.particle_type);
        
        self.context.begin_path();
        self.context.arc(x, y, size, 0.0, std::f64::consts::PI * 2.0)?;
        self.set_fill_style(&format!("rgba({},{},{},{})", r, g, b, a))?;
        self.context.fill();
        
        Ok(())
    }

    fn draw_agent(&mut self, agent: &AgentVisualization, x: f64, y: f64) -> Result<(), JsValue> {
        // Draw agent body
        self.context.set_fill_style_str(&format!("rgba({},{},{},{})", 
            (agent.color[0] * 255.0) as u8,
            (agent.color[1] * 255.0) as u8,
            (agent.color[2] * 255.0) as u8,
            agent.color[3]));
        self.context.begin_path();
        self.context.arc(x, y, 5.0, 0.0, std::f64::consts::PI * 2.0)?;
        self.context.fill();

        // Draw agent name
        self.context.set_fill_style_str("white");
        self.context.set_font("12px Arial");
        self.context.fill_text(&format!("Agent {}", agent.id.to_string()[..8].to_string()), x + 10.0, y)?;

        // Draw agent stats
        self.context.set_font("10px Arial");
        self.context.fill_text(&format!("Energy: {:.1}", agent.energy), x + 10.0, y + 15.0)?;
        self.context.fill_text(&format!("Age: {}", agent.age), x + 10.0, y + 30.0)?;

        // Draw agent traits
        let mut y_offset = 45.0;
        for (trait_name, trait_value) in &agent.traits {
            self.context.fill_text(&format!("{}: {:.2}", trait_name, trait_value), x + 10.0, y + y_offset)?;
            y_offset += 15.0;
        }

        // Draw agent connections
        for connection in &agent.social_connections {
            if let Some(target_agent) = self.simulation_state.agents.iter().find(|a| a.id == *connection) {
                let (target_x, target_y) = self.map_3d_to_2d(target_agent.position);
                self.context.begin_path();
                self.context.move_to(x, y);
                self.context.line_to(target_x, target_y);
                self.context.set_stroke_style_str("#666666");
                self.context.stroke();
            }
        }

        // Draw agent history
        if let Some(history) = self.agent_history.get(&agent.id) {
            self.context.begin_path();
            self.context.set_stroke_style_str("#4CAF50");
            self.context.set_line_width(1.0);
            
            if let Some((first_x, first_y)) = history.positions.first().map(|p| self.map_3d_to_2d(*p)) {
                self.context.move_to(first_x, first_y);
                for pos in history.positions.iter().skip(1) {
                    let (x, y) = self.map_3d_to_2d(*pos);
                    self.context.line_to(x, y);
                }
                self.context.stroke();
            }
        }

        // Draw agent consciousness
        if let Some(_consciousness) = &agent.consciousness {
            self.context.set_fill_style_str("#9C27B0");
            self.context.begin_path();
            self.context.arc(x, y, 8.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.context.fill();
        }

        // Draw agent innovation
        if let Some(_innovation) = &agent.innovation {
            self.context.set_fill_style_str("#FF9800");
            self.context.begin_path();
            self.context.arc(x, y, 10.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.context.fill();
        }

        Ok(())
    }

    fn draw_ui_text(&mut self, text: &str, x: f64, y: f64) -> Result<(), JsValue> {
        self.set_fill_style("white")?;
        self.context.fill_text(text, x, y)?;
        Ok(())
    }

    fn draw_ui_box(&mut self, x: f64, y: f64, width: f64, height: f64, color: &str) -> Result<(), JsValue> {
        self.context.begin_path();
        self.context.rect(x, y, width, height);
        self.set_fill_style(color)?;
        self.context.fill();
        self.set_stroke_style("white")?;
        self.context.stroke();
        Ok(())
    }

    fn draw_ui_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: &str) -> Result<(), JsValue> {
        self.context.begin_path();
        self.context.move_to(x1, y1);
        self.context.line_to(x2, y2);
        self.set_stroke_style(color)?;
        self.context.stroke();
        Ok(())
    }

    fn draw_ui_graph(&mut self, data: &[f64], x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        if data.is_empty() {
            return Ok(());
        }
        
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max - min;
        
        self.context.begin_path();
        self.context.move_to(x, y + height);
        
        for (i, &value) in data.iter().enumerate() {
            let px = x + (i as f64 / (data.len() - 1) as f64) * width;
            let py = y + height - ((value - min) / range) * height;
            self.context.line_to(px, py);
        }
        
        self.set_stroke_style("rgba(0,255,255,0.8)")?;
        self.context.stroke();
        
        // Draw axes
        self.set_stroke_style("white")?;
        self.context.begin_path();
        self.context.move_to(x, y);
        self.context.line_to(x, y + height);
        self.context.line_to(x + width, y + height);
        self.context.stroke();
        
        Ok(())
    }

    fn draw_ui_panel(&mut self) -> Result<(), JsValue> {
        let width = self.canvas.width() as f64;
        let _height = self.canvas.height() as f64;
        
        // Draw semi-transparent background
        self.context.begin_path();
        self.context.rect(0.0, 0.0, width, 60.0);
        self.context.set_fill_style_str("rgba(0,0,0,0.7)");
        self.context.fill();
        
        // Draw stats
        self.context.set_font("12px Segoe UI");
        self.context.set_fill_style_str("white");
        self.context.fill_text(&format!("Age: {:.2} Gyr", self.simulation_state.universe_age_gyr), 10.0, 20.0)?;
        self.context.fill_text(&format!("Temp: {:.2e} K", self.simulation_state.temperature), 10.0, 40.0)?;
        self.context.fill_text(&format!("Particles: {}", self.simulation_state.particles.len()), 200.0, 20.0)?;
        self.context.fill_text(&format!("Agents: {}", self.simulation_state.agents.len()), 200.0, 40.0)?;
        self.context.fill_text(&format!("Lineages: {}", self.simulation_state.lineages.len()), 400.0, 20.0)?;
        
        Ok(())
    }

    fn apply_delta(&mut self, delta: DeltaMessage) {
        if delta.kind != "delta" { return; }

        // Update or add particles
        if let Some(new_parts) = delta.new_particles {
            self.simulation_state.particles.extend(new_parts);
            self.needs_redraw = true;
        }
        if let Some(upd_parts) = delta.updated_particles {
            for upd in upd_parts {
                if let Some(existing) = self.simulation_state.particles.iter_mut().find(|p| p.id == upd.id) {
                    *existing = upd;
                } else {
                    self.simulation_state.particles.push(upd);
                }
            }
            self.needs_redraw = true;
        }
        // Remove particles by id
        if let Some(rem) = delta.removed_particle_ids {
            self.simulation_state.particles.retain(|p| !rem.contains(&p.id));
            self.needs_redraw = true;
        }

        // Update celestial bodies from delta
        self.simulation_state.celestial_bodies = delta.celestial_bodies;
        
        // Update quantum fields from delta
        self.simulation_state.quantum_fields = delta.quantum_fields;

        self.simulation_state.current_tick = delta.current_tick;
        self.needs_redraw = true;
    }

    fn update_agent_history(&mut self, agent_id: Uuid, position: [f64; 3]) {
        let history = self.agent_history.entry(agent_id).or_insert(AgentHistory {
            positions: Vec::new(),
            decisions: Vec::new(),
            innovations: Vec::new(),
        });
        history.positions.push(position);
        // Keep only last 100 positions for performance
        if history.positions.len() > 100 {
            history.positions.remove(0);
        }
    }

    pub fn set_time_scale(&mut self, scale: f64) {
        self.time_scale = scale;
    }
    
    pub fn set_zoom_level(&mut self, scale: f64) {
        self.particle_size_scale = scale;
    }

    fn get_context(&self) -> &CanvasRenderingContext2d {
        self.offscreen_context.as_ref().unwrap()
    }

    fn get_context_mut(&mut self) -> &mut CanvasRenderingContext2d {
        self.offscreen_context.as_mut().unwrap()
    }

    // Helper methods to pre-calculate points
    fn calculate_trail_points(&self, particle: &ParticleVisualization) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        let mut pos = particle.position;
        let vel = particle.momentum;
        let dt = 0.1;
        
        for _ in 0..10 {
            // Element-wise addition and multiplication
            pos[0] += vel[0] * dt;
            pos[1] += vel[1] * dt;
            pos[2] += vel[2] * dt;
            let (x, y) = self.map_3d_to_2d(pos);
            points.push((x, y));
        }
        points
    }
    
    fn calculate_interaction_points(&self, particle: &ParticleVisualization) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        let base_pos = self.map_3d_to_2d(particle.position);
        
        // Calculate interaction points in a circle around the particle
        for i in 0..particle.interaction_count {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (particle.interaction_count as f64);
            let radius = 20.0;
            let x = base_pos.0 + radius * angle.cos();
            let y = base_pos.1 + radius * angle.sin();
            points.push((x, y));
        }
        
        points
    }

    fn render_celestial_bodies(&mut self) -> Result<(), JsValue> {
        // Pre-calculate all rendering data before getting the context (with fallback data)
        let celestial_bodies = if self.simulation_state.celestial_bodies.is_empty() {
            vec![
                CelestialBodyVisualization {
                    id: "demo_sun".to_string(),
                    body_type: "Star".to_string(),
                    position: [0.0, 0.0, 0.0],
                    velocity: [0.0, 0.0, 0.0],
                    mass: 1.989e30,
                    radius: 6.96e8,
                    temperature: 5778.0,
                    luminosity: 3.828e26,
                    age: 4.6e9,
                },
                CelestialBodyVisualization {
                    id: "demo_earth".to_string(),
                    body_type: "Planet".to_string(),
                    position: [1.496e11, 0.0, 0.0],
                    velocity: [0.0, 29780.0, 0.0],
                    mass: 5.972e24,
                    radius: 6.371e6,
                    temperature: 288.0,
                    luminosity: 0.0,
                    age: 4.54e9,
                },
            ]
        } else {
            self.simulation_state.celestial_bodies.clone()
        };
        
        let render_data: Vec<_> = celestial_bodies.iter().map(|body| {
            let (x, y) = self.map_3d_to_2d(body.position);
            
            // Determine size based on body type and radius
            let size = match body.body_type.as_str() {
                "Star" => {
                    // Stars: size based on luminosity and temperature
                    let base_size = (body.radius / 6.96e8).log10().max(0.5); // Relative to sun
                    (base_size * 8.0).max(3.0).min(20.0) // 3-20 pixel range
                },
                "Planet" => {
                    // Planets: smaller, based on radius
                    let base_size = (body.radius / 6.371e6).log10().max(0.1); // Relative to Earth
                    (base_size * 4.0).max(1.5).min(8.0) // 1.5-8 pixel range
                },
                _ => 2.0, // Default size for other bodies
            };
            
            // Determine color based on body type and properties
            let color = match body.body_type.as_str() {
                "Star" => {
                    // Color based on temperature (stellar classification)
                    if body.temperature > 30000.0 {
                        "rgba(155, 176, 255, 0.9)" // O-type: Blue
                    } else if body.temperature > 10000.0 {
                        "rgba(176, 196, 255, 0.9)" // B-type: Blue-white
                    } else if body.temperature > 7500.0 {
                        "rgba(248, 247, 255, 0.9)" // A-type: White
                    } else if body.temperature > 6000.0 {
                        "rgba(255, 244, 234, 0.9)" // F-type: Yellow-white
                    } else if body.temperature > 5200.0 {
                        "rgba(255, 255, 0, 0.9)"   // G-type: Yellow (Sun-like)
                    } else if body.temperature > 3700.0 {
                        "rgba(255, 210, 161, 0.9)" // K-type: Orange
                    } else {
                        "rgba(255, 76, 76, 0.9)"   // M-type: Red dwarf
                    }
                },
                "Planet" => "rgba(100, 149, 237, 0.8)", // Blue for planets
                "BlackHole" => "rgba(0, 0, 0, 1.0)",     // Black
                "NeutronStar" => "rgba(200, 200, 255, 0.9)", // White-blue
                "WhiteDwarf" => "rgba(255, 255, 255, 0.8)",  // White
                _ => "rgba(128, 128, 128, 0.7)", // Gray default
            };
            
            // Calculate glow info for stars
            let glow_info = if body.body_type == "Star" && body.luminosity > 0.0 {
                let glow_size = size * 2.0;
                let alpha = (body.luminosity / 3.828e26).log10().max(0.1).min(0.3); // Relative to sun
                Some((glow_size, alpha))
            } else {
                None
            };
            
            // Calculate velocity vector if significant
            let velocity_vector = if body.velocity[0].abs() > 1000.0 || body.velocity[1].abs() > 1000.0 {
                let vel_scale = 1e-6; // Scale down velocity for visualization
                let vx = body.velocity[0] * vel_scale;
                let vy = body.velocity[1] * vel_scale;
                Some((vx, vy))
            } else {
                None
            };
            
            (body.clone(), x, y, size, color.to_string(), glow_info, velocity_vector)
        }).collect();
        
        // Now use the context to render everything
        let context = self.get_context_mut();
        
        for (body, x, y, size, color, glow_info, velocity_vector) in render_data {
            // Draw the celestial body
            context.set_fill_style_str(&color);
            context.begin_path();
            context.arc(x, y, size, 0.0, 2.0 * std::f64::consts::PI)?;
            context.fill();
            
            // Add glow effect for stars
            if let Some((glow_size, alpha)) = glow_info {
                context.set_fill_style_str(&format!("rgba(255, 255, 100, {})", alpha * 0.3));
                context.begin_path();
                context.arc(x, y, glow_size, 0.0, 2.0 * std::f64::consts::PI)?;
                context.fill();
            }
            
            // Add velocity vector for moving bodies
            if let Some((vx, vy)) = velocity_vector {
                context.set_stroke_style_str("rgba(0, 255, 255, 0.6)");
                context.set_line_width(1.0);
                context.begin_path();
                context.move_to(x, y);
                context.line_to(x + vx, y + vy);
                context.stroke();
            }
        }
        
        Ok(())
    }
}

/// Establish websocket and start reading JSON messages representing SimulationState
fn connect_ws(monitor_rc: Rc<RefCell<EvolutionMonitor>>, websocket_url: String) {
    let ws = web_sys::WebSocket::new(&websocket_url).unwrap();
    ws.set_binary_type(web_sys::BinaryType::Arraybuffer);
    
    // Set up message handler
    let message_monitor_rc = monitor_rc.clone();
    let onmessage_callback = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        // First, handle text-based JSON messages (plain strings)
        if let Some(text_data) = e.data().as_string() {
            let timestamp = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(js_sys::Date::now());

            // Attempt to parse the JSON directly
            match serde_json::from_str::<serde_json::Value>(&text_data) {
                Ok(value) => {
                    let kind = value.get("kind").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let debug_msg = DebugMessage {
                        timestamp,
                        message_type: kind.to_string(),
                        size_bytes: text_data.len(),
                        content: text_data.clone(),
                        raw_data: text_data.chars().take(50).collect(),
                        parse_success: true,
                        error: None,
                    };

                    let mut monitor = message_monitor_rc.borrow_mut();
                    monitor.messages_received += 1;
                    monitor.debug_messages.push(debug_msg);
                    if monitor.debug_messages.len() > 100 {
                        monitor.debug_messages.drain(0..50);
                    }

                    if kind == "delta" {
                        if let Ok(delta) = serde_json::from_value::<DeltaMessage>(value) {
                            monitor.apply_delta(delta);
                            if let Err(e) = monitor.update_html_ui() {
                                console_log!("Error updating UI: {:?}", e);
                            }
                        }
                    } else if kind == "full" {
                        if let Ok(full) = serde_json::from_value::<FullMessage>(value) {
                            // Replace entire simulation snapshot
                            monitor.simulation_state.current_tick = full.current_tick;
                            monitor.simulation_state.universe_age_gyr = full.universe_age_gyr;
                            monitor.simulation_state.cosmic_era = full.cosmic_era;
                            monitor.simulation_state.temperature = full.temperature;
                            monitor.simulation_state.energy_density = full.energy_density;
                            monitor.simulation_state.particles = full.particles;
                            monitor.simulation_state.celestial_bodies = full.celestial_bodies;
                            monitor.simulation_state.quantum_fields = full.quantum_fields;
                            monitor.simulation_state.nuclei = full.nuclei;
                            monitor.simulation_state.atoms = full.atoms;
                            monitor.needs_redraw = true;
                            if let Err(e) = monitor.update_html_ui() {
                                console_log!("Error updating UI: {:?}", e);
                            }
                        }
                    }
                    return; // done processing text message
                }
                Err(e) => {
                    // Parsing failed; fall through to binary handling for backward compatibility
                    console_log!("Failed to parse plain text JSON: {:?}", e);
                }
            }
        }

        // Fallback: handle legacy binary-compressed messages
        if let Ok(buf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
            let array = js_sys::Uint8Array::new(&buf);
            let mut bytes = vec![0; array.length() as usize];
            array.copy_to(&mut bytes);
            
            let timestamp = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(js_sys::Date::now());
            
            // Decompress the zlib data
            if let Ok(decompressed) = miniz_oxide::inflate::decompress_to_vec_zlib(&bytes) {
                // Parse the JSON data
                if let Ok(json_str) = std::str::from_utf8(&decompressed) {
                    match serde_json::from_str::<serde_json::Value>(json_str) {
                        Ok(value) => {
                            let kind = value.get("kind").and_then(|v| v.as_str()).unwrap_or("unknown");
                            
                            // Create debug message
                            let debug_msg = DebugMessage {
                                timestamp,
                                message_type: kind.to_string(),
                                size_bytes: bytes.len(),
                                content: json_str.to_string(),
                                raw_data: format!("{:?}", &bytes[..bytes.len().min(50)]),
                                parse_success: true,
                                error: None,
                            };
                            
                            if kind == "delta" {
                                if let Ok(delta) = serde_json::from_value::<DeltaMessage>(value) {
                                    let mut monitor = message_monitor_rc.borrow_mut();
                                    monitor.messages_received += 1;
                                    monitor.debug_messages.push(debug_msg);
                                    if monitor.debug_messages.len() > 100 {
                                        monitor.debug_messages.drain(0..50);
                                    }
                                    monitor.apply_delta(delta);
                                    if let Err(e) = monitor.update_html_ui() {
                                        console_log!("Error updating UI: {:?}", e);
                                    }
                                }
                            } else if kind == "full" {
                                if let Ok(full) = serde_json::from_value::<FullMessage>(value) {
                                    let mut monitor = message_monitor_rc.borrow_mut();
                                    monitor.messages_received += 1;
                                    monitor.debug_messages.push(debug_msg);
                                    if monitor.debug_messages.len() > 100 {
                                        monitor.debug_messages.drain(0..50);
                                    }
                                    // Replace entire simulation snapshot
                                    monitor.simulation_state.current_tick = full.current_tick;
                                    monitor.simulation_state.universe_age_gyr = full.universe_age_gyr;
                                    monitor.simulation_state.cosmic_era = full.cosmic_era;
                                    monitor.simulation_state.temperature = full.temperature;
                                    monitor.simulation_state.energy_density = full.energy_density;
                                    monitor.simulation_state.particles = full.particles;
                                    monitor.simulation_state.celestial_bodies = full.celestial_bodies;
                                    monitor.simulation_state.quantum_fields = full.quantum_fields;
                                    monitor.simulation_state.nuclei = full.nuclei;
                                    monitor.simulation_state.atoms = full.atoms;
                                    monitor.needs_redraw = true;
                                    if let Err(e) = monitor.update_html_ui() {
                                        console_log!("Error updating UI: {:?}", e);
                                    }
                                }
                            } else {
                                let mut monitor = message_monitor_rc.borrow_mut();
                                monitor.messages_received += 1;
                                monitor.debug_messages.push(debug_msg);
                                if monitor.debug_messages.len() > 100 {
                                    monitor.debug_messages.drain(0..50);
                                }
                                console_log!("Unknown message kind: {}", kind);
                            }
                        },
                        Err(e) => {
                            // Create debug message for parse error
                            let debug_msg = DebugMessage {
                                timestamp,
                                message_type: "parse_error".to_string(),
                                size_bytes: bytes.len(),
                                content: json_str.to_string(),
                                raw_data: format!("{:?}", &bytes[..bytes.len().min(50)]),
                                parse_success: false,
                                error: Some(format!("{:?}", e)),
                            };
                            
                            let mut monitor = message_monitor_rc.borrow_mut();
                            monitor.messages_received += 1;
                            monitor.debug_messages.push(debug_msg);
                            if monitor.debug_messages.len() > 100 {
                                monitor.debug_messages.drain(0..50);
                            }
                            console_log!("Failed to parse websocket JSON: {:?}", e);
                        }
                    }
                } else {
                    // Create debug message for UTF-8 decode error
                    let debug_msg = DebugMessage {
                        timestamp,
                        message_type: "utf8_error".to_string(),
                        size_bytes: bytes.len(),
                        content: "Failed to decode as UTF-8".to_string(),
                        raw_data: format!("{:?}", &bytes[..bytes.len().min(50)]),
                        parse_success: false,
                        error: Some("Failed to decode decompressed data as UTF-8".to_string()),
                    };
                    
                    let mut monitor = message_monitor_rc.borrow_mut();
                    monitor.messages_received += 1;
                    monitor.debug_messages.push(debug_msg);
                    if monitor.debug_messages.len() > 100 {
                        monitor.debug_messages.drain(0..50);
                    }
                    console_log!("Failed to decode decompressed data as UTF-8");
                }
            } else {
                // Create debug message for decompression error
                let debug_msg = DebugMessage {
                    timestamp,
                    message_type: "decompress_error".to_string(),
                    size_bytes: bytes.len(),
                    content: "Failed to decompress".to_string(),
                    raw_data: format!("{:?}", &bytes[..bytes.len().min(50)]),
                    parse_success: false,
                    error: Some("Failed to decompress zlib data".to_string()),
                };
                
                let mut monitor = message_monitor_rc.borrow_mut();
                monitor.messages_received += 1;
                monitor.debug_messages.push(debug_msg);
                if monitor.debug_messages.len() > 100 {
                    monitor.debug_messages.drain(0..50);
                }
                console_log!("Failed to decompress delta message");
            }
        }
    }) as Box<dyn FnMut(web_sys::MessageEvent)>);
    
    ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
    onmessage_callback.forget();

    // Set up open handler
    let open_monitor_rc = monitor_rc.clone();
    let onopen_callback = Closure::wrap(Box::new(move |_: web_sys::Event| {
        console_log!("WebSocket connection established");
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(elem) = doc.get_element_by_id("connection-status") {
                elem.set_inner_html("🟢 Connected");
                elem.set_class_name("connection-status connected");
            }
        }
        open_monitor_rc.borrow_mut().connected = true;
    }) as Box<dyn FnMut(web_sys::Event)>);
    
    ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
    onopen_callback.forget();

    // Set up error handler
    let error_monitor_rc = monitor_rc.clone();
    let onerror_callback = Closure::wrap(Box::new(move |_: web_sys::Event| {
        console_log!("WebSocket connection error");
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(elem) = doc.get_element_by_id("connection-status") {
                elem.set_inner_html("🔴 Disconnected");
                elem.set_class_name("connection-status disconnected");
            }
        }
        error_monitor_rc.borrow_mut().connected = false;
    }) as Box<dyn FnMut(web_sys::Event)>);
    
    ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    onerror_callback.forget();

    // Set up close handler
    let close_monitor_rc = monitor_rc.clone();
    let onclose_callback = Closure::wrap(Box::new(move |_: web_sys::CloseEvent| {
        console_log!("WebSocket connection closed");
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(elem) = doc.get_element_by_id("connection-status") {
                elem.set_inner_html("🔴 Disconnected");
                elem.set_class_name("connection-status disconnected");
            }
        }
        close_monitor_rc.borrow_mut().connected = false;
    }) as Box<dyn FnMut(web_sys::CloseEvent)>);
    
    ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
    onclose_callback.forget();

    monitor_rc.borrow_mut().websocket = Some(ws);
    monitor_rc.borrow_mut().connected = true;
}

/// Start RAF loop for continuous rendering
fn start_render_loop(monitor_rc: Rc<RefCell<EvolutionMonitor>>) -> Result<(), JsValue> {
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |timestamp: f64| {
        let mut monitor = monitor_rc.borrow_mut();
        if let Err(e) = monitor.render() {
            console_log!("Error in render loop: {:?}", e);
        }
        web_window().request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref()).unwrap();
    }) as Box<dyn FnMut(f64)>));

    web_window().request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;
    Ok(())
}

// Supporting types and structures
#[derive(Debug, Default, Clone, Copy)]
pub struct ParticleFilter {
    pub show_quarks: bool,
    pub show_leptons: bool,
    pub show_bosons: bool,
    pub min_energy: f64,
    pub max_energy: f64,
}

pub struct DecisionAnalyzer;
impl DecisionAnalyzer {
    pub fn new() -> Self { Self }
}

pub struct ConsciousnessMonitor;
impl ConsciousnessMonitor {
    pub fn new() -> Self { Self }
}

pub struct InnovationTracker;
impl InnovationTracker {
    pub fn new() -> Self { Self }
}

pub struct AnalyticsEngine;
impl AnalyticsEngine {
    pub fn new() -> Self { Self }
}

pub struct LineageTracker;

impl LineageTracker {
    pub fn new() -> Self { Self }
}

// Placeholder types for visualization data
pub type DecisionSummary = String;
pub type InnovationRecord = String;
pub type ConsciousnessMilestone = String;
pub type TechnologyAchievement = String;
pub type SelectionEventRecord = String;
pub type AdaptationRecord = String;
pub type SpeciationRecord = String;
pub type FamilyTreeVisualization = String;
pub type EvolutionTimelineEvent = String;
pub type InnovationVariation = String;
pub type FieldVisualization = String;
pub type NucleusVisualization = String;
pub type AtomVisualization = String;
pub type MoleculeVisualization = String;
pub type SelectionPressureVisualization = String;
pub type EvolutionMetrics = String;
pub type PhysicsMetrics = String;

#[derive(Debug, Clone)]
pub struct DebugMessage {
    pub timestamp: f64,
    pub message_type: String,
    pub size_bytes: usize,
    pub content: String,
    pub raw_data: String,
    pub parse_success: bool,
    pub error: Option<String>,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            current_tick: 0,
            universe_age_gyr: 0.0,
            cosmic_era: CosmicEra::initial(),
            temperature: 0.0,
            energy_density: 0.0,
            particles: Vec::new(),
            celestial_bodies: Vec::new(),
            quantum_fields: HashMap::new(),
            nuclei: Vec::new(),
            atoms: Vec::new(),
            molecules: Vec::new(),
            agents: Vec::new(),
            lineages: HashMap::new(),
            decisions: Vec::new(),
            innovations: Vec::new(),
            consciousness_events: Vec::new(),
            environments: Vec::new(),
            selection_pressures: Vec::new(),
            population_stats: PopulationStatistics::default(),
            evolution_metrics: "".to_string(),
            physics_metrics: "".to_string(),
        }
    }
}

impl Default for PopulationStatistics {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_lineages: 0,
            extinct_lineages: 0,
            birth_rate: 0.0,
            death_rate: 0.0,
            mutation_rate: 0.0,
            average_fitness: 0.0,
            fitness_variance: 0.0,
            genetic_diversity: 0.0,
            consciousness_distribution: HashMap::new(),
            technology_distribution: HashMap::new(),
        }
    }
}

// Utility functions
fn window() -> Option<Window> {
    web_sys::window()
}

// Add helper to get window
fn web_window() -> web_sys::Window {
    web_sys::window().expect("no global window")
}

#[wasm_bindgen(start)]
pub fn run() -> Result<(), JsValue> {
    // This function is called when the WASM is loaded.
    // It sets up the simulation monitor and the render loop.

    // Use this to log panic messages to the browser console
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();

    let monitor = EvolutionMonitor::new("simulation-canvas")?;
    let monitor_rc = Rc::new(RefCell::new(monitor));
    
    let window = web_window();
    let location = window.location();
    let ws_protocol = if location.protocol()? == "https:" { "wss:" } else { "ws:" };
    let hostname = location.hostname()?;
    let hostname = hostname.trim_end_matches('.');
    let websocket_url = format!("{}//{}:8080/ws", ws_protocol, hostname);

    connect_ws(monitor_rc.clone(), websocket_url);
    start_render_loop(monitor_rc.clone())?;
    
    // Set up UI event handlers
    setup_ui_event_handlers(monitor_rc)?;

    Ok(())
}

fn setup_ui_event_handlers(monitor_rc: Rc<RefCell<EvolutionMonitor>>) -> Result<(), JsValue> {
    let document = web_window().document().unwrap();
    
    // Set up view mode buttons
    let buttons = document.query_selector_all(".view-btn")?;
    for i in 0..buttons.length() {
        if let Some(button_elem) = buttons.get(i) {
            let button = button_elem.dyn_into::<HtmlElement>()?;
            let monitor_rc = monitor_rc.clone();
            let button_for_closure = button.clone();
            let click_handler = Closure::wrap(Box::new(move |_: web_sys::Event| {
                // Remove active class from all buttons
                let doc = web_window().document().unwrap();
                let buttons = doc.query_selector_all(".view-btn").unwrap();
                for j in 0..buttons.length() {
                    if let Some(btn) = buttons.get(j) {
                        let btn = btn.dyn_into::<HtmlElement>().unwrap();
                        btn.class_list().remove_1("active").unwrap();
                    }
                }
                
                // Add active class to clicked button
                button_for_closure.class_list().add_1("active").unwrap();
                
                // Get view mode from data-view attribute
                if let Some(view_attr) = button_for_closure.get_attribute("data-view") {
                    if let Ok(mode_num) = view_attr.parse::<u8>() {
                        console_log!("Setting view mode to: {}", mode_num);
                        monitor_rc.borrow_mut().set_view_mode(mode_num);
                    }
                }
            }) as Box<dyn FnMut(web_sys::Event)>);
            
            button.add_event_listener_with_callback(
                "click",
                click_handler.as_ref().unchecked_ref(),
            )?;
            click_handler.forget();
        }
    }

    // Time scale slider
    let slider = document.get_element_by_id("time-scale").unwrap();
    let slider = slider.dyn_into::<HtmlInputElement>()?;
    let slider_clone = slider.clone();
    let monitor_clone = monitor_rc.clone();
    let callback = Closure::wrap(Box::new(move || {
        if let Ok(value) = slider_clone.value().parse::<f64>() {
            monitor_clone.borrow_mut().set_time_scale(value);
            // Update the display value
            if let Some(span) = web_window().document().unwrap().get_element_by_id("time-scale-value") {
                span.set_inner_html(&format!("{:.1}x", value));
            }
        }
    }) as Box<dyn FnMut()>);
    slider.set_oninput(Some(callback.as_ref().unchecked_ref()));
    callback.forget();

    // Zoom level slider
    let slider = document.get_element_by_id("zoom-level").unwrap();
    let slider = slider.dyn_into::<HtmlInputElement>()?;
    let slider_clone = slider.clone();
    let monitor_clone = monitor_rc.clone();
    let callback = Closure::wrap(Box::new(move || {
        if let Ok(value) = slider_clone.value().parse::<f64>() {
            monitor_clone.borrow_mut().set_zoom_level(value);
            // Update the display value
            if let Some(span) = web_window().document().unwrap().get_element_by_id("zoom-value") {
                span.set_inner_html(&format!("{:.1}x", value));
            }
        }
    }) as Box<dyn FnMut()>);
    slider.set_oninput(Some(callback.as_ref().unchecked_ref()));
    callback.forget();
    
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMessage {
    pub kind: String,
    pub current_tick: u64,
    pub new_particles: Option<Vec<ParticleVisualization>>,
    pub updated_particles: Option<Vec<ParticleVisualization>>,
    pub removed_particle_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub celestial_bodies: Vec<CelestialBodyVisualization>,
    #[serde(default)]
    pub quantum_fields: HashMap<String, FieldVisualization>,
}

/// Complete state snapshot sent when a client first connects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullMessage {
    pub kind: String,
    pub current_tick: u64,
    pub universe_age_gyr: f64,
    pub cosmic_era: CosmicEra,
    pub temperature: f64,
    pub energy_density: f64,
    pub particles: Vec<ParticleVisualization>,
    #[serde(default)]
    pub celestial_bodies: Vec<CelestialBodyVisualization>,
    #[serde(default)]
    pub quantum_fields: HashMap<String, FieldVisualization>,
    #[serde(default)]
    pub nuclei: Vec<NucleusVisualization>,
    #[serde(default)]
    pub atoms: Vec<AtomVisualization>,
}

thread_local! {
    static GLOBAL_MONITOR: RefCell<Option<Weak<RefCell<EvolutionMonitor>>>> = RefCell::new(None);
}

fn set_global_monitor(monitor: &Rc<RefCell<EvolutionMonitor>>) {
    GLOBAL_MONITOR.with(|slot| {
        *slot.borrow_mut() = Some(Rc::downgrade(monitor));
    });
}

#[wasm_bindgen]
pub fn log_rpc_message(message_type: &str, json: &str) {
    GLOBAL_MONITOR.with(|slot| {
        if let Some(rc) = slot.borrow().as_ref().and_then(|w| w.upgrade()) {
            let mut monitor = rc.borrow_mut();
            let timestamp = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(js_sys::Date::now());
            let debug_msg = DebugMessage {
                timestamp,
                message_type: message_type.to_string(),
                size_bytes: json.len(),
                content: json.to_string(),
                raw_data: String::new(),
                parse_success: true,
                error: None,
            };
            monitor.rpc_messages_received += 1;
            monitor.rpc_debug_messages.push(debug_msg);
            if monitor.rpc_debug_messages.len() > 100 {
                monitor.rpc_debug_messages.drain(0..50);
            }
            let _ = monitor.update_html_ui();
        }
    });
}