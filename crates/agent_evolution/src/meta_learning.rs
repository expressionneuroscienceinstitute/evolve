//! # Meta-Learning Module
//!
//! Provides a lightweight, online meta-learner that dynamically tunes key
//! parameters (e.g. learning rate, exploration rate, mutation rate) of an
//! `AICore` or other adaptive systems based on recent performance trends.  The
//! design avoids hard-coded biological outcomes and instead relies purely on
//! statistical feedback signals.
//!
//! The algorithm uses an exponentially-weighted moving average (EWMA) of recent
//! performance to estimate improvement velocity and variance.  It then applies
//! a simple control-theoretic rule to adjust parameters so as to maintain a
//! target improvement rate without oscillations.
//!
//! All state is explicit; no global mutables or magic constants.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Meta-tunable parameter names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaParameter {
    LearningRate,
    ExplorationRate,
    MutationRate,
}

/// Configuration for a single meta-parameter controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaParamConfig {
    /// Desired fractional improvement per time unit (e.g. per tick)
    pub target_improvement_rate: f64,
    /// Step size used when parameter needs to increase
    pub up_step: f64,
    /// Step size used when parameter needs to decrease
    pub down_step: f64,
    /// Minimum allowed value (clamped)
    pub min: f64,
    /// Maximum allowed value (clamped)
    pub max: f64,
    /// EWMA smoothing factor for performance trend (0-1)
    pub smoothing_factor: f64,
}

impl Default for MetaParamConfig {
    fn default() -> Self {
        Self {
            target_improvement_rate: 0.001,
            up_step: 0.05,
            down_step: 0.05,
            min: 1e-4,
            max: 1.0,
            smoothing_factor: 0.2,
        }
    }
}

/// Internal statistics tracked for each `AICore`
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoreMetaStats {
    /// Exponentially weighted average of recent performance
    pub ewma_performance: f64,
    /// Exponentially weighted average of performance improvement rate
    pub ewma_velocity: f64,
    /// Last raw performance observed
    pub last_performance: Option<f64>,
}

/// Meta-learner that adapts parameters of multiple AI cores
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetaLearner {
    pub id: Uuid,
    /// Per-core statistics and parameter states
    pub cores: HashMap<Uuid, CoreMetaStats>,
    /// Per-parameter configuration shared across cores
    pub param_configs: HashMap<MetaParameter, MetaParamConfig>,
}

impl MetaLearner {
    /// Create a new meta-learner with default parameter configs
    pub fn new() -> Self {
        let mut param_configs = HashMap::new();
        param_configs.insert(MetaParameter::LearningRate, MetaParamConfig::default());
        param_configs.insert(MetaParameter::ExplorationRate, MetaParamConfig { min: 0.01, max: 1.0, ..MetaParamConfig::default() });
        param_configs.insert(MetaParameter::MutationRate, MetaParamConfig { min: 0.0, max: 0.5, ..MetaParamConfig::default() });

        Self {
            id: Uuid::new_v4(),
            cores: HashMap::new(),
            param_configs,
        }
    }

    /// Register a new core so statistics can be tracked from the next update
    pub fn register_core(&mut self, core_id: Uuid) {
        self.cores.entry(core_id).or_insert_with(CoreMetaStats::default);
    }

    /// Update meta-learner with latest performance measurement for a core and
    /// return suggested parameter adjustments.
    pub fn update_core(&mut self, core_id: Uuid, performance: f64, params: &HashMap<MetaParameter, f64>) -> Result<HashMap<MetaParameter, f64>> {
        let stats = self.cores.entry(core_id).or_default();

        // Update EWMA performance
        if let Some(last) = stats.last_performance {
            let improvement = performance - last;
            stats.ewma_velocity = (1.0 - 0.1) * stats.ewma_velocity + 0.1 * improvement;
        }
        stats.ewma_performance = (1.0 - 0.1) * stats.ewma_performance + 0.1 * performance;
        stats.last_performance = Some(performance);

        // Suggest new parameter values
        let mut new_params = HashMap::new();
        for (&param, &current_value) in params {
            if let Some(cfg) = self.param_configs.get(&param) {
                let velocity = stats.ewma_velocity;
                let desired = cfg.target_improvement_rate;
                let mut updated = current_value;

                if velocity < desired {
                    // Not improving fast enough – increase parameter
                    updated *= 1.0 + cfg.up_step;
                } else if velocity > desired * 5.0 {
                    // Improving rapidly – decrease to fine-tune
                    updated *= 1.0 - cfg.down_step;
                }

                // Clamp to [min,max]
                updated = updated.clamp(cfg.min, cfg.max);
                new_params.insert(param, updated);
            }
        }

        Ok(new_params)
    }
}

/// Convenience structure for passing around parameter maps
pub type MetaParamMap = HashMap<MetaParameter, f64>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn meta_learning_adjusts_parameters() {
        let core_id = Uuid::new_v4();
        let mut learner = MetaLearner::new();
        learner.register_core(core_id);

        let mut params = HashMap::new();
        params.insert(MetaParameter::LearningRate, 0.01);

        // Simulate stagnant performance
        for _ in 0..10 {
            let suggestions = learner.update_core(core_id, 0.5, &params).unwrap();
            // Expect learning rate to increase over time
            let new_lr = *suggestions.get(&MetaParameter::LearningRate).unwrap();
            assert!(new_lr >= 0.01);
            params.insert(MetaParameter::LearningRate, new_lr);
        }
    }
} 