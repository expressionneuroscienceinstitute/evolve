//! # JSON-RPC types for Universectl

use serde::{Deserialize, Serialize};
use anyhow::Result;
use serde_json::json;
use std::time::Duration;
use tokio::time::timeout;

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    // Using serde_json::Value to allow for flexible parameters
    pub params: serde_json::Value,
    pub id: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcResponse<T> {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

// Pre-defined error codes
#[allow(dead_code)]
pub const PARSE_ERROR: i32 = -32700;
#[allow(dead_code)]
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

// Specific response payload for get_status
#[derive(Serialize, Deserialize, Debug)]
pub struct StatusResponse {
    pub status: String,
    pub tick: u64,
    pub ups: f64,
    pub universe_age_gyr: f64,
    pub universe_description: String,
    pub lineage_count: u64,
    pub save_file_age_sec: Option<u64>,
}

/// Pending resource request entry returned by `resources_queue` RPC
#[derive(Serialize, Deserialize, Debug)]
pub struct ResourceRequest {
    pub id: String,
    pub resource: String,
    pub amount: u64,
    pub requester: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>, // ISO-8601 string or "N/A"
}

/// Petition/Message sent by an in-simulation agent to the Oracle
#[derive(Serialize, Deserialize, Debug)]
pub struct Petition {
    pub id: String,
    pub agent_id: String,
    pub subject: String,
    pub body: String,
    pub received_at: String, // ISO-8601 timestamp
}

/// Aggregate resource usage vs limits returned by `resources_status` RPC
#[derive(Serialize, Deserialize, Debug)]
pub struct ResourceStatus {
    pub usage: std::collections::HashMap<String, u64>,
    pub limits: std::collections::HashMap<String, u64>,
}

/// Response from map RPC call
#[derive(Serialize, Deserialize, Debug)]
pub struct MapResponse {
    pub width: usize,
    pub height: usize,
    pub data: serde_json::Value,
}

/// Response from list_planets RPC call
#[derive(Serialize, Deserialize, Debug)]
pub struct PlanetListResponse {
    pub planets: Vec<serde_json::Value>,
}

pub struct RpcClient {
    client: reqwest::Client,
    url: String,
    timeout: Duration,
}

impl RpcClient {
    pub fn new(rpc_port: u16) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("http://127.0.0.1:{}/rpc", rpc_port),
            timeout: Duration::from_secs(5), // Default 5 second timeout
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub async fn send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let req_body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        });

        let request_future = self.client.post(&self.url).json(&req_body).send();
        
        // Apply timeout to the request
        let res = match timeout(self.timeout, request_future).await {
            Ok(res) => res?,
            Err(_) => return Err(anyhow::anyhow!("RPC request timed out after {:?}", self.timeout)),
        };
        
        if !res.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", res.status()));
        }

        let rpc_res: RpcResponse<serde_json::Value> = res.json().await?;
        
        if let Some(error) = rpc_res.error {
            return Err(anyhow::anyhow!("RPC error: {} (code: {})", error.message, error.code));
        }

        rpc_res.result.ok_or_else(|| anyhow::anyhow!("No result in RPC response"))
    }

    #[allow(dead_code)]
    pub async fn status(&self) -> Result<serde_json::Value> {
        self.send_request("status", json!({})).await
    }

    #[allow(dead_code)]
    pub async fn stop(&self) -> Result<serde_json::Value> {
        self.send_request("stop", json!({})).await
    }

    #[allow(dead_code)]
    pub async fn map(&self, zoom: f64, layer: &str) -> Result<serde_json::Value> {
        self.send_request("map", json!({ "zoom": zoom, "layer": layer })).await
    }

    #[allow(dead_code)]
    pub async fn list_planets(&self, class: Option<String>, habitable: bool) -> Result<serde_json::Value> {
        self.send_request("list_planets", json!({ "class_filter": class, "habitable_only": habitable })).await
    }

    #[allow(dead_code)]
    pub async fn inspect_planet(&self, planet_id: String) -> Result<serde_json::Value> {
        self.send_request("inspect_planet", json!({ "planet_id": planet_id })).await
    }
    
    #[allow(dead_code)]
    pub async fn inspect_lineage(&self, lineage_id: String) -> Result<serde_json::Value> {
        self.send_request("inspect_lineage", json!({ "lineage_id": lineage_id })).await
    }
    
    #[allow(dead_code)]
    pub async fn inspect_universe(&self) -> Result<serde_json::Value> {
        self.send_request("inspect_universe", json!({})).await
    }
    
    #[allow(dead_code)]
    pub async fn inspect_physics(&self) -> Result<serde_json::Value> {
        self.send_request("inspect_physics", json!({})).await
    }

    #[allow(dead_code)]
    pub async fn universe_stats(&self) -> Result<serde_json::Value> {
        self.send_request("universe_stats", json!({})).await
    }

    #[allow(dead_code)]
    pub async fn physics_diagnostics(&self) -> Result<serde_json::Value> {
        self.send_request("physics_diagnostics", json!({})).await
    }

    #[allow(dead_code)]
    pub async fn snapshot(&self, file: String) -> Result<serde_json::Value> {
        self.send_request("snapshot", json!({ "path": file })).await
    }

    #[allow(dead_code)]
    pub async fn speed(&self, factor: f64) -> Result<serde_json::Value> {
        self.send_request("speed", json!({ "factor": factor })).await
    }

    #[allow(dead_code)]
    pub async fn rewind(&self, ticks: u64) -> Result<serde_json::Value> {
        self.send_request("rewind", json!({ "ticks": ticks })).await
    }

    #[allow(dead_code)]
    pub async fn godmode_create_body(&self, mass: f64, body_type: String, pos: [f64; 3]) -> Result<serde_json::Value> {
        self.send_request("godmode_create_body", json!({
            "mass": mass,
            "body_type": body_type,
            "position": pos
        })).await
    }
    
    #[allow(dead_code)]
    pub async fn godmode_delete_body(&self, id: String) -> Result<serde_json::Value> {
        self.send_request("godmode_delete_body", json!({ "id": id })).await
    }
    
    #[allow(dead_code)]
    pub async fn godmode_set_constant(&self, name: String, value: f64) -> Result<serde_json::Value> {
        self.send_request("godmode_set_constant", json!({ "name": name, "value": value })).await
    }

    #[allow(dead_code)]
    pub async fn godmode_spawn_lineage(&self, code_hash: String, planet_id: String) -> Result<serde_json::Value> {
        self.send_request("godmode_spawn_lineage", json!({ "code_hash": code_hash, "planet_id": planet_id })).await
    }

    #[allow(dead_code)]
    pub async fn godmode_create_agent(&self, planet_id: String) -> Result<serde_json::Value> {
        self.send_request("godmode_create_agent", json!({ "planet_id": planet_id })).await
    }
    
    #[allow(dead_code)]
    pub async fn godmode_miracle(&self, planet_id: String, miracle_type: String, duration: Option<u64>, intensity: Option<f64>) -> Result<serde_json::Value> {
        self.send_request("godmode_miracle", json!({ 
            "planet_id": planet_id,
            "miracle_type": miracle_type,
            "duration": duration,
            "intensity": intensity
        })).await
    }
}

/// Make a generic RPC call to the server with timeout
pub async fn call_rpc(method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
    let client = RpcClient::new(9001).with_timeout(Duration::from_secs(3)); // 3 second timeout for CLI
    client.send_request(method, params).await
}

/// Make a generic RPC call to the server with custom timeout
pub async fn call_rpc_with_timeout(method: &str, params: serde_json::Value, timeout: Duration) -> Result<serde_json::Value> {
    let client = RpcClient::new(9001).with_timeout(timeout);
    client.send_request(method, params).await
} 