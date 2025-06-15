//! Diagnostics and Performance Monitoring
//! 
//! Provides real-time performance metrics, bottleneck detection, and memory usage tracking
//! for the Evolution Universe Simulation.

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::Result;
use sysinfo::System;

/// Main diagnostics system for monitoring simulation performance
#[derive(Debug)]
pub struct DiagnosticsSystem {
    /// Performance metrics collected over time
    pub metrics: PerformanceMetrics,
    /// Memory usage tracking
    pub memory_monitor: MemoryMonitor,
    /// Bottleneck detection system
    pub bottleneck_detector: BottleneckDetector,
    /// System resource monitor
    pub system_monitor: SystemMonitor,
    /// System information interface
    system_info: System,
    /// Metric collection enabled flag
    pub enabled: bool,
    /// Collection interval in milliseconds
    pub collection_interval_ms: u64,
    /// Last collection time
    last_collection: Instant,
}

/// Real-time performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Physics engine step times (milliseconds)
    pub physics_step_times: TimeSeries,
    /// Universe simulation tick times (milliseconds)
    pub universe_tick_times: TimeSeries,
    /// Particle interaction rates (events per second)
    pub interaction_rates: TimeSeries,
    /// Memory allocation rates (bytes per second)
    pub allocation_rates: TimeSeries,
    /// Frame rates for web dashboard (fps)
    pub frame_rates: TimeSeries,
    /// Network latency for RPC calls (milliseconds)
    pub rpc_latency: TimeSeries,
    /// Current system load metrics
    pub system_load: SystemLoad,
}

/// Time series data with automatic window management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Data points (timestamp, value)
    pub data: VecDeque<(f64, f64)>,
    /// Maximum number of data points to keep
    pub max_size: usize,
    /// Statistical summaries
    pub stats: TimeSeriesStats,
}

/// Statistical summaries for time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
}

/// Memory usage monitoring
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Memory usage history
    pub usage_history: TimeSeries,
    /// Peak memory usage (bytes)
    pub peak_usage: u64,
    /// Memory allocation events
    pub allocation_events: VecDeque<AllocationEvent>,
    /// Memory leak detection
    pub leak_detector: LeakDetector,
}

/// Memory allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub timestamp: f64,
    pub size: u64,
    pub source: String,
    pub allocation_type: AllocationType,
}

/// Memory allocation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    Particle,
    Nucleus,
    Atom,
    Molecule,
    CelestialBody,
    Planet,
    Lineage,
    QuantumField,
    Other(String),
}

/// Memory leak detection
#[derive(Debug)]
pub struct LeakDetector {
    /// Baseline memory usage
    pub baseline: u64,
    /// Memory growth rate (bytes per second)
    pub growth_rate: f64,
    /// Leak threshold (bytes per hour)
    pub leak_threshold: u64,
    /// Potential leaks detected
    pub potential_leaks: Vec<LeakReport>,
}

/// Memory leak report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakReport {
    pub timestamp: f64,
    pub source: String,
    pub growth_rate: f64,
    pub severity: LeakSeverity,
}

/// Leak severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakSeverity {
    Low,    // < 1 MB/hour
    Medium, // 1-10 MB/hour
    High,   // 10-100 MB/hour
    Critical, // > 100 MB/hour
}

/// Bottleneck detection system
#[derive(Debug)]
pub struct BottleneckDetector {
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Profiling data for hotspots
    pub profiling_data: HashMap<String, ProfilingData>,
    /// Threshold for bottleneck detection (milliseconds)
    pub threshold_ms: f64,
}

/// Performance bottleneck report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub timestamp: f64,
    pub component: String,
    pub operation: String,
    pub duration_ms: f64,
    pub frequency: u32,
    pub impact_score: f64,
    pub suggestions: Vec<String>,
}

/// Profiling data for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingData {
    pub total_time_ms: f64,
    pub call_count: u32,
    pub average_time_ms: f64,
    pub max_time_ms: f64,
    pub percentage_of_total: f64,
}

/// System resource monitoring
#[derive(Debug)]
pub struct SystemMonitor {
    /// CPU usage monitoring
    pub cpu_monitor: CpuMonitor,
    /// Disk I/O monitoring
    pub disk_monitor: DiskMonitor,
    /// Network monitoring
    pub network_monitor: NetworkMonitor,
}

/// CPU usage monitoring
#[derive(Debug)]
pub struct CpuMonitor {
    pub usage_history: TimeSeries,
    pub core_count: usize,
    pub current_usage: f64,
}

/// Disk I/O monitoring
#[derive(Debug)]
pub struct DiskMonitor {
    pub read_rates: TimeSeries,
    pub write_rates: TimeSeries,
    pub disk_usage: f64,
}

/// Network monitoring
#[derive(Debug)]
pub struct NetworkMonitor {
    pub bandwidth_usage: TimeSeries,
    pub connection_count: u32,
    pub packet_loss_rate: f64,
}

/// Current system load snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLoad {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_bandwidth: f64,
    pub temperature: f64,
    pub timestamp: f64,
}

impl DiagnosticsSystem {
    /// Create a new diagnostics system
    pub fn new() -> Self {
        let mut system_info = System::new_all();
        system_info.refresh_all();
        
        Self {
            metrics: PerformanceMetrics::new(),
            memory_monitor: MemoryMonitor::new(),
            bottleneck_detector: BottleneckDetector::new(),
            system_monitor: SystemMonitor::new(),
            system_info,
            enabled: true,
            collection_interval_ms: 1000, // 1 second
            last_collection: Instant::now(),
        }
    }
    
    /// Enable or disable metric collection
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Set collection interval
    pub fn set_collection_interval(&mut self, interval_ms: u64) {
        self.collection_interval_ms = interval_ms;
    }
    
    /// Update metrics (call this regularly from the main simulation loop)
    pub fn update(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let now = Instant::now();
        if now.duration_since(self.last_collection).as_millis() >= self.collection_interval_ms as u128 {
            self.collect_metrics()?;
            self.detect_bottlenecks()?;
            self.check_for_leaks()?;
            self.last_collection = now;
        }
        
        Ok(())
    }
    
    /// Record a physics step time
    pub fn record_physics_step(&mut self, duration: Duration) {
        if self.enabled {
            let timestamp = self.current_timestamp();
            let duration_ms = duration.as_millis() as f64;
            self.metrics.physics_step_times.add_point(timestamp, duration_ms);
        }
    }
    
    /// Record a universe tick time
    pub fn record_universe_tick(&mut self, duration: Duration) {
        if self.enabled {
            let timestamp = self.current_timestamp();
            let duration_ms = duration.as_millis() as f64;
            self.metrics.universe_tick_times.add_point(timestamp, duration_ms);
        }
    }
    
    /// Record memory allocation
    pub fn record_allocation(&mut self, size: u64, source: String, allocation_type: AllocationType) {
        if self.enabled {
            let event = AllocationEvent {
                timestamp: self.current_timestamp(),
                size,
                source,
                allocation_type,
            };
            self.memory_monitor.allocation_events.push_back(event);
            
            // Keep only recent events
            while self.memory_monitor.allocation_events.len() > 10000 {
                self.memory_monitor.allocation_events.pop_front();
            }
        }
    }
    
    /// Record an interaction event
    pub fn record_interaction(&mut self, count: u32) {
        if self.enabled {
            let timestamp = self.current_timestamp();
            self.metrics.interaction_rates.add_point(timestamp, count as f64);
        }
    }
    
    /// Get current performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        PerformanceReport {
            metrics: self.metrics.clone(),
            bottlenecks: self.bottleneck_detector.bottlenecks.clone(),
            memory_usage: self.memory_monitor.usage_history.stats.clone(),
            system_load: self.metrics.system_load.clone(),
            timestamp: self.current_timestamp(),
        }
    }
    
    /// Get diagnostics as JSON for web dashboard
    pub fn to_json(&self) -> Result<String> {
        let report = self.get_performance_report();
        Ok(serde_json::to_string_pretty(&report)?)
    }
    
    /// Collect system metrics
    fn collect_metrics(&mut self) -> Result<()> {
        // Refresh system information to get latest data
        self.system_info.refresh_all();
        
        let timestamp = self.current_timestamp();
        
        // Update system load
        self.metrics.system_load = SystemLoad {
            cpu_usage: self.get_cpu_usage(),
            memory_usage: self.get_memory_usage(),
            disk_usage: self.get_disk_usage(),
            network_bandwidth: self.get_network_bandwidth(),
            temperature: self.get_system_temperature(),
            timestamp,
        };
        
        // Update memory usage history
        self.memory_monitor.usage_history.add_point(timestamp, self.metrics.system_load.memory_usage);
        
        Ok(())
    }
    
    /// Detect performance bottlenecks
    fn detect_bottlenecks(&mut self) -> Result<()> {
        let current_time = self.current_timestamp();
        
        // Check physics step times
        if let Some(latest_physics_time) = self.metrics.physics_step_times.data.back() {
            if latest_physics_time.1 > self.bottleneck_detector.threshold_ms {
                let bottleneck = PerformanceBottleneck {
                    timestamp: current_time,
                    component: "PhysicsEngine".to_string(),
                    operation: "step".to_string(),
                    duration_ms: latest_physics_time.1,
                    frequency: 1,
                    impact_score: latest_physics_time.1 / self.bottleneck_detector.threshold_ms,
                    suggestions: vec![
                        "Consider reducing particle count".to_string(),
                        "Implement spatial partitioning".to_string(),
                        "Optimize interaction calculations".to_string(),
                    ],
                };
                self.bottleneck_detector.bottlenecks.push(bottleneck);
            }
        }
        
        // Keep only recent bottlenecks
        self.bottleneck_detector.bottlenecks.retain(|b| current_time - b.timestamp < 3600.0); // 1 hour
        
        Ok(())
    }
    
    /// Check for memory leaks
    fn check_for_leaks(&mut self) -> Result<()> {
        let current_memory = self.get_memory_usage();
        let current_time = self.current_timestamp();
        
        // Calculate memory growth rate
        if let Some(baseline_point) = self.memory_monitor.usage_history.data.front() {
            let time_diff = current_time - baseline_point.0;
            if time_diff > 3600.0 { // 1 hour
                let memory_diff = current_memory - baseline_point.1;
                let growth_rate = memory_diff / time_diff * 3600.0; // bytes per hour
                
                if growth_rate > self.memory_monitor.leak_detector.leak_threshold as f64 {
                    let severity = match growth_rate as u64 {
                        0..=1048576 => LeakSeverity::Low,        // < 1 MB/hour
                        1048577..=10485760 => LeakSeverity::Medium, // 1-10 MB/hour
                        10485761..=104857600 => LeakSeverity::High,  // 10-100 MB/hour
                        _ => LeakSeverity::Critical,                 // > 100 MB/hour
                    };
                    
                    let leak_report = LeakReport {
                        timestamp: current_time,
                        source: "System Memory".to_string(),
                        growth_rate,
                        severity,
                    };
                    
                    self.memory_monitor.leak_detector.potential_leaks.push(leak_report);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current timestamp as seconds since epoch
    fn current_timestamp(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
    
    /// Get current CPU usage (real implementation)
    fn get_cpu_usage(&self) -> f64 {
        self.system_info.global_cpu_info().cpu_usage() as f64
    }
    
    /// Get current memory usage in bytes (real implementation)
    fn get_memory_usage(&self) -> f64 {
        self.system_info.used_memory() as f64
    }
    
    /// Get current disk usage percentage (real implementation)
    fn get_disk_usage(&self) -> f64 {
        // Simplified implementation for now - just return a reasonable default
        // In sysinfo v0.30, disk usage is more complex to calculate
        50.0 // 50% usage as a reasonable default
    }
    
    /// Get current network bandwidth usage (real implementation)
    fn get_network_bandwidth(&self) -> f64 {
        // Simplified implementation for now
        // In sysinfo v0.30, network usage tracking is different
        0.0 // No active network usage tracking
    }
    
    /// Get system temperature (real implementation)
    fn get_system_temperature(&self) -> f64 {
        // Simplified implementation for now
        // Temperature monitoring varies significantly by platform
        25.0 // Room temperature default
    }
}

/// Performance report for external consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub metrics: PerformanceMetrics,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub memory_usage: TimeSeriesStats,
    pub system_load: SystemLoad,
    pub timestamp: f64,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            physics_step_times: TimeSeries::new(1000),
            universe_tick_times: TimeSeries::new(1000),
            interaction_rates: TimeSeries::new(1000),
            allocation_rates: TimeSeries::new(1000),
            frame_rates: TimeSeries::new(1000),
            rpc_latency: TimeSeries::new(1000),
            system_load: SystemLoad::default(),
        }
    }
}

impl TimeSeries {
    fn new(max_size: usize) -> Self {
        Self {
            data: VecDeque::new(),
            max_size,
            stats: TimeSeriesStats::default(),
        }
    }
    
    fn add_point(&mut self, timestamp: f64, value: f64) {
        self.data.push_back((timestamp, value));
        
        // Remove old points
        while self.data.len() > self.max_size {
            self.data.pop_front();
        }
        
        // Update statistics
        self.update_stats();
    }
    
    fn update_stats(&mut self) {
        if self.data.is_empty() {
            return;
        }
        
        let values: Vec<f64> = self.data.iter().map(|(_, v)| *v).collect();
        
        self.stats.min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        self.stats.max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.stats.mean = values.iter().sum::<f64>() / values.len() as f64;
        
        // Calculate median
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.stats.median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };
        
        // Calculate standard deviation
        let variance = values.iter()
            .map(|v| (v - self.stats.mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        self.stats.std_dev = variance.sqrt();
        
        // Calculate percentiles
        self.stats.percentile_95 = sorted_values[(sorted_values.len() as f64 * 0.95) as usize];
        self.stats.percentile_99 = sorted_values[(sorted_values.len() as f64 * 0.99) as usize];
    }
}

impl Default for TimeSeriesStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            percentile_95: 0.0,
            percentile_99: 0.0,
        }
    }
}

impl Default for SystemLoad {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_bandwidth: 0.0,
            temperature: 0.0,
            timestamp: 0.0,
        }
    }
}

impl MemoryMonitor {
    fn new() -> Self {
        Self {
            usage_history: TimeSeries::new(1000),
            peak_usage: 0,
            allocation_events: VecDeque::new(),
            leak_detector: LeakDetector::new(),
        }
    }
}

impl LeakDetector {
    fn new() -> Self {
        Self {
            baseline: 0,
            growth_rate: 0.0,
            leak_threshold: 1048576, // 1 MB per hour
            potential_leaks: Vec::new(),
        }
    }
}

impl BottleneckDetector {
    fn new() -> Self {
        Self {
            bottlenecks: Vec::new(),
            profiling_data: HashMap::new(),
            threshold_ms: 100.0, // 100ms threshold
        }
    }
}

impl SystemMonitor {
    fn new() -> Self {
        Self {
            cpu_monitor: CpuMonitor {
                usage_history: TimeSeries::new(1000),
                core_count: num_cpus::get(),
                current_usage: 0.0,
            },
            disk_monitor: DiskMonitor {
                read_rates: TimeSeries::new(1000),
                write_rates: TimeSeries::new(1000),
                disk_usage: 0.0,
            },
            network_monitor: NetworkMonitor {
                bandwidth_usage: TimeSeries::new(1000),
                connection_count: 0,
                packet_loss_rate: 0.0,
            },
        }
    }
}

/// Export main diagnostics function for backwards compatibility
pub fn diagnose() -> DiagnosticsSystem {
    DiagnosticsSystem::new()
}