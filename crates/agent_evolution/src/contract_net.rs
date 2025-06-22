//! Contract Net Protocol Implementation
//!
//! Implements the classic Contract Net Protocol for distributed task allocation
//! in multi-agent systems. Based on Reid G. Smith's original 1980 paper.
//!
//! Key concepts:
//! - Task announcements with specifications
//! - Bidding process with capability assessment
//! - Contract award based on bid evaluation
//! - Hierarchical task decomposition
//! - Distributed control without central authority
//!
//! References:
//! - Smith, R.G. "The Contract Net Protocol: High-Level Communication and Control
//!   in a Distributed Problem Solver" (1980)
//! - Modern multi-agent coordination strategies

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use std::time::{Duration, Instant};

/// Agent capabilities for task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub agent_id: String,
    pub expertise_domains: Vec<String>,
    pub current_workload: f64, // 0.0 to 1.0
    pub processing_speed: f64, // tasks per hour
    pub reliability_score: f64, // 0.0 to 1.0
    pub available_memory: f64, // MB
    pub network_latency: f64, // ms
}

/// Task specification for contract net protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpecification {
    pub task_id: Uuid,
    pub task_type: String,
    pub complexity_score: f64, // 1.0 to 10.0
    pub estimated_duration: Duration,
    pub required_expertise: Vec<String>,
    pub priority: TaskPriority,
    pub dependencies: Vec<Uuid>, // Task IDs this depends on
    pub resource_requirements: ResourceRequirements,
    pub deadline: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_memory_mb: f64,
    pub min_processing_speed: f64,
    pub required_expertise: Vec<String>,
    pub network_bandwidth_mbps: f64,
}

/// Task announcement in Contract Net Protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnnouncement {
    pub announcement_id: Uuid,
    pub manager_id: String,
    pub task_spec: TaskSpecification,
    pub expiration_time: Instant,
    pub bid_specification: BidSpecification,
    pub report_recipients: Vec<String>,
    pub related_contractors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidSpecification {
    pub required_info: Vec<String>,
    pub evaluation_criteria: Vec<BidCriterion>,
    pub minimum_bid_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BidCriterion {
    ProcessingSpeed,
    Reliability,
    Workload,
    Expertise,
    NetworkLatency,
    Cost,
}

/// Bid submitted by potential contractors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bid {
    pub bid_id: Uuid,
    pub contractor_id: String,
    pub announcement_id: Uuid,
    pub proposed_duration: Duration,
    pub cost_estimate: f64,
    pub confidence_score: f64, // 0.0 to 1.0
    pub capability_assessment: CapabilityAssessment,
    pub additional_info: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityAssessment {
    pub expertise_match: f64, // 0.0 to 1.0
    pub workload_capacity: f64, // 0.0 to 1.0
    pub processing_capability: f64, // 0.0 to 1.0
    pub reliability_estimate: f64, // 0.0 to 1.0
    pub network_performance: f64, // 0.0 to 1.0
}

/// Contract between manager and contractor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub contract_id: Uuid,
    pub manager_id: String,
    pub contractor_id: String,
    pub task_spec: TaskSpecification,
    pub bid: Bid,
    pub status: ContractStatus,
    pub start_time: Option<Instant>,
    pub completion_time: Option<Instant>,
    pub results: Option<TaskResults>,
    pub subcontracts: Vec<Subcontract>,
    pub report_recipients: Vec<String>,
    pub related_contractors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContractStatus {
    Awarded,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResults {
    pub success: bool,
    pub output_data: HashMap<String, String>,
    pub performance_metrics: PerformanceMetrics,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub actual_duration: Duration,
    pub resource_utilization: f64,
    pub quality_score: f64,
    pub efficiency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subcontract {
    pub subcontract_id: Uuid,
    pub contractor_id: String,
    pub task_spec: TaskSpecification,
    pub results: Option<TaskResults>,
    pub predecessors: Vec<Uuid>,
    pub successors: Vec<Uuid>,
}

/// Contract Net Protocol Manager
#[derive(Debug)]
pub struct ContractNetManager {
    pub agent_id: String,
    pub capabilities: AgentCapabilities,
    pub active_contracts: HashMap<Uuid, Contract>,
    pub pending_announcements: HashMap<Uuid, TaskAnnouncement>,
    pub received_bids: HashMap<Uuid, Vec<Bid>>,
    pub task_queue: Vec<TaskSpecification>,
    pub performance_history: Vec<ContractPerformance>,
}

#[derive(Debug, Clone)]
pub struct ContractPerformance {
    pub contract_id: Uuid,
    pub contractor_id: String,
    pub success_rate: f64,
    pub average_duration: Duration,
    pub quality_score: f64,
}

impl ContractNetManager {
    pub fn new(agent_id: String, capabilities: AgentCapabilities) -> Self {
        Self {
            agent_id,
            capabilities,
            active_contracts: HashMap::new(),
            pending_announcements: HashMap::new(),
            received_bids: HashMap::new(),
            task_queue: Vec::new(),
            performance_history: Vec::new(),
        }
    }

    /// Announce a task for bidding (Manager role)
    pub fn announce_task(&mut self, task_spec: TaskSpecification) -> Result<TaskAnnouncement> {
        let announcement = TaskAnnouncement {
            announcement_id: Uuid::new_v4(),
            manager_id: self.agent_id.clone(),
            task_spec: task_spec.clone(),
            expiration_time: Instant::now() + Duration::from_secs(300), // 5 minutes
            bid_specification: self.create_bid_specification(&task_spec),
            report_recipients: vec![self.agent_id.clone()],
            related_contractors: Vec::new(),
        };

        self.pending_announcements.insert(announcement.announcement_id, announcement.clone());
        Ok(announcement)
    }

    /// Evaluate received bids and award contract (Manager role)
    pub fn evaluate_bids(&mut self, announcement_id: Uuid) -> Result<Option<Contract>> {
        let bids = self.received_bids.get(&announcement_id).cloned().unwrap_or_default();
        if bids.is_empty() {
            return Ok(None);
        }

        // Rank bids based on multiple criteria
        let ranked_bids = self.rank_bids(&bids);
        
        if let Some(best_bid) = ranked_bids.first() {
            let announcement = self.pending_announcements.get(&announcement_id)
                .ok_or_else(|| anyhow::anyhow!("Announcement not found"))?;
            
            let contract = Contract {
                contract_id: Uuid::new_v4(),
                manager_id: self.agent_id.clone(),
                contractor_id: best_bid.contractor_id.clone(),
                task_spec: announcement.task_spec.clone(),
                bid: best_bid.clone(),
                status: ContractStatus::Awarded,
                start_time: None,
                completion_time: None,
                results: None,
                subcontracts: Vec::new(),
                report_recipients: announcement.report_recipients.clone(),
                related_contractors: announcement.related_contractors.clone(),
            };

            self.active_contracts.insert(contract.contract_id, contract.clone());
            self.pending_announcements.remove(&announcement_id);
            self.received_bids.remove(&announcement_id);

            Ok(Some(contract))
        } else {
            Ok(None)
        }
    }

    /// Submit a bid for a task (Contractor role)
    pub fn submit_bid(&mut self, announcement: &TaskAnnouncement) -> Result<Option<Bid>> {
        // Check if we can handle this task
        if !self.can_handle_task(&announcement.task_spec) {
            return Ok(None);
        }

        let capability_assessment = self.assess_capabilities(&announcement.task_spec);
        
        // Only bid if we meet minimum requirements
        if capability_assessment.expertise_match < 0.5 || capability_assessment.workload_capacity < 0.3 {
            return Ok(None);
        }

        let bid = Bid {
            bid_id: Uuid::new_v4(),
            contractor_id: self.agent_id.clone(),
            announcement_id: announcement.announcement_id,
            proposed_duration: self.estimate_duration(&announcement.task_spec),
            cost_estimate: self.estimate_cost(&announcement.task_spec),
            confidence_score: self.calculate_confidence(&capability_assessment),
            capability_assessment,
            additional_info: HashMap::new(),
        };

        Ok(Some(bid))
    }

    /// Execute a contracted task (Contractor role)
    pub fn execute_contract(&mut self, contract: &mut Contract) -> Result<TaskResults> {
        contract.status = ContractStatus::InProgress;
        contract.start_time = Some(Instant::now());

        // Simulate task execution
        std::thread::sleep(Duration::from_millis(100)); // Simulate work

        let results = TaskResults {
            success: true,
            output_data: HashMap::new(),
            performance_metrics: PerformanceMetrics {
                actual_duration: Duration::from_secs(1),
                resource_utilization: 0.7,
                quality_score: 0.9,
                efficiency_score: 0.8,
            },
            error_message: None,
        };

        contract.status = ContractStatus::Completed;
        contract.completion_time = Some(Instant::now());
        contract.results = Some(results.clone());

        // Update performance history
        self.update_performance_history(contract);

        Ok(results)
    }

    /// Check if agent can handle a specific task
    fn can_handle_task(&self, task_spec: &TaskSpecification) -> bool {
        // Check expertise match
        let expertise_match = task_spec.required_expertise.iter()
            .any(|exp| self.capabilities.expertise_domains.contains(exp));

        // Check workload capacity
        let workload_ok = self.capabilities.current_workload < 0.8;

        // Check resource requirements
        let resources_ok = self.capabilities.available_memory >= task_spec.resource_requirements.min_memory_mb
            && self.capabilities.processing_speed >= task_spec.resource_requirements.min_processing_speed;

        expertise_match && workload_ok && resources_ok
    }

    /// Assess capabilities for a specific task
    fn assess_capabilities(&self, task_spec: &TaskSpecification) -> CapabilityAssessment {
        let expertise_match = task_spec.required_expertise.iter()
            .map(|exp| {
                if self.capabilities.expertise_domains.contains(exp) { 1.0 } else { 0.0 }
            })
            .sum::<f64>() / task_spec.required_expertise.len() as f64;

        let workload_capacity = 1.0 - self.capabilities.current_workload;
        
        let processing_capability = (self.capabilities.processing_speed / task_spec.resource_requirements.min_processing_speed).min(1.0);

        CapabilityAssessment {
            expertise_match,
            workload_capacity,
            processing_capability,
            reliability_estimate: self.capabilities.reliability_score,
            network_performance: (1000.0 / self.capabilities.network_latency).min(1.0),
        }
    }

    /// Estimate task duration based on complexity and capabilities
    fn estimate_duration(&self, task_spec: &TaskSpecification) -> Duration {
        let base_duration = Duration::from_secs(task_spec.complexity_score as u64 * 60);
        let efficiency_factor = self.capabilities.processing_speed / 10.0; // Normalize
        Duration::from_secs((base_duration.as_secs() as f64 / efficiency_factor) as u64)
    }

    /// Estimate task cost based on complexity and duration
    fn estimate_cost(&self, task_spec: &TaskSpecification) -> f64 {
        let base_cost = task_spec.complexity_score * 10.0;
        let duration_factor = task_spec.estimated_duration.as_secs() as f64 / 3600.0; // hours
        base_cost * duration_factor
    }

    /// Calculate confidence score for bid
    fn calculate_confidence(&self, assessment: &CapabilityAssessment) -> f64 {
        let weights = [0.3, 0.2, 0.2, 0.2, 0.1]; // expertise, workload, processing, reliability, network
        let scores = [
            assessment.expertise_match,
            assessment.workload_capacity,
            assessment.processing_capability,
            assessment.reliability_estimate,
            assessment.network_performance,
        ];

        scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum()
    }

    /// Rank bids based on multiple criteria
    fn rank_bids(&self, bids: &[Bid]) -> Vec<Bid> {
        let mut ranked_bids = bids.to_vec();
        ranked_bids.sort_by(|a, b| {
            // Primary: confidence score (descending)
            b.confidence_score.partial_cmp(&a.confidence_score).unwrap()
                // Secondary: cost (ascending)
                .then(a.cost_estimate.partial_cmp(&b.cost_estimate).unwrap())
                // Tertiary: proposed duration (ascending)
                .then(a.proposed_duration.cmp(&b.proposed_duration))
        });
        ranked_bids
    }

    /// Create bid specification for task announcement
    fn create_bid_specification(&self, task_spec: &TaskSpecification) -> BidSpecification {
        BidSpecification {
            required_info: vec![
                "expertise_match".to_string(),
                "workload_capacity".to_string(),
                "processing_capability".to_string(),
                "reliability_estimate".to_string(),
            ],
            evaluation_criteria: vec![
                BidCriterion::Expertise,
                BidCriterion::Reliability,
                BidCriterion::Workload,
                BidCriterion::Cost,
            ],
            minimum_bid_quality: 0.6,
        }
    }

    /// Update performance history after contract completion
    fn update_performance_history(&mut self, contract: &Contract) {
        if let Some(results) = &contract.results {
            let performance = ContractPerformance {
                contract_id: contract.contract_id,
                contractor_id: contract.contractor_id.clone(),
                success_rate: if results.success { 1.0 } else { 0.0 },
                average_duration: results.performance_metrics.actual_duration,
                quality_score: results.performance_metrics.quality_score,
            };
            self.performance_history.push(performance);
        }
    }

    /// Get agent statistics
    pub fn get_statistics(&self) -> AgentStatistics {
        let total_contracts = self.active_contracts.len() + self.performance_history.len();
        let successful_contracts = self.performance_history.iter()
            .filter(|p| p.success_rate > 0.5)
            .count();

        AgentStatistics {
            agent_id: self.agent_id.clone(),
            total_contracts,
            success_rate: if total_contracts > 0 {
                successful_contracts as f64 / total_contracts as f64
            } else {
                0.0
            },
            average_quality: self.performance_history.iter()
                .map(|p| p.quality_score)
                .sum::<f64>() / self.performance_history.len().max(1) as f64,
            current_workload: self.capabilities.current_workload,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentStatistics {
    pub agent_id: String,
    pub total_contracts: usize,
    pub success_rate: f64,
    pub average_quality: f64,
    pub current_workload: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_net_manager_creation() {
        let capabilities = AgentCapabilities {
            agent_id: "test_agent".to_string(),
            expertise_domains: vec!["physics".to_string(), "molecular_dynamics".to_string()],
            current_workload: 0.3,
            processing_speed: 15.0,
            reliability_score: 0.9,
            available_memory: 2048.0,
            network_latency: 50.0,
        };

        let manager = ContractNetManager::new("test_agent".to_string(), capabilities);
        assert_eq!(manager.agent_id, "test_agent");
        assert_eq!(manager.active_contracts.len(), 0);
    }

    #[test]
    fn test_task_announcement() {
        let capabilities = AgentCapabilities {
            agent_id: "manager".to_string(),
            expertise_domains: vec!["coordination".to_string()],
            current_workload: 0.2,
            processing_speed: 10.0,
            reliability_score: 0.8,
            available_memory: 1024.0,
            network_latency: 30.0,
        };

        let mut manager = ContractNetManager::new("manager".to_string(), capabilities);
        
        let task_spec = TaskSpecification {
            task_id: Uuid::new_v4(),
            task_type: "molecular_dynamics".to_string(),
            complexity_score: 5.0,
            estimated_duration: Duration::from_secs(300),
            required_expertise: vec!["molecular_dynamics".to_string()],
            priority: TaskPriority::Medium,
            dependencies: Vec::new(),
            resource_requirements: ResourceRequirements {
                min_memory_mb: 512.0,
                min_processing_speed: 5.0,
                required_expertise: vec!["molecular_dynamics".to_string()],
                network_bandwidth_mbps: 10.0,
            },
            deadline: None,
        };

        let announcement = manager.announce_task(task_spec).unwrap();
        assert_eq!(announcement.manager_id, "manager");
        assert_eq!(announcement.task_spec.task_type, "molecular_dynamics");
    }

    #[test]
    fn test_bid_submission() {
        let capabilities = AgentCapabilities {
            agent_id: "contractor".to_string(),
            expertise_domains: vec!["molecular_dynamics".to_string()],
            current_workload: 0.4,
            processing_speed: 12.0,
            reliability_score: 0.85,
            available_memory: 1536.0,
            network_latency: 40.0,
        };

        let manager = ContractNetManager::new("contractor".to_string(), capabilities);
        
        let task_spec = TaskSpecification {
            task_id: Uuid::new_v4(),
            task_type: "molecular_dynamics".to_string(),
            complexity_score: 4.0,
            estimated_duration: Duration::from_secs(240),
            required_expertise: vec!["molecular_dynamics".to_string()],
            priority: TaskPriority::Medium,
            dependencies: Vec::new(),
            resource_requirements: ResourceRequirements {
                min_memory_mb: 512.0,
                min_processing_speed: 5.0,
                required_expertise: vec!["molecular_dynamics".to_string()],
                network_bandwidth_mbps: 10.0,
            },
            deadline: None,
        };

        let announcement = TaskAnnouncement {
            announcement_id: Uuid::new_v4(),
            manager_id: "manager".to_string(),
            task_spec,
            expiration_time: Instant::now() + Duration::from_secs(300),
            bid_specification: BidSpecification {
                required_info: vec!["expertise_match".to_string()],
                evaluation_criteria: vec![BidCriterion::Expertise],
                minimum_bid_quality: 0.6,
            },
            report_recipients: vec!["manager".to_string()],
            related_contractors: Vec::new(),
        };

        let bid = manager.submit_bid(&announcement).unwrap();
        assert!(bid.is_some());
        let bid = bid.unwrap();
        assert_eq!(bid.contractor_id, "contractor");
        assert!(bid.confidence_score > 0.5);
    }
} 