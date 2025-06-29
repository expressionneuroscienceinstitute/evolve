//! # JSON-RPC types for Universectl

use serde::{Deserialize, Serialize};
use anyhow::Result;
use serde_json::json;
use std::time::Duration;
use tokio::time::timeout;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

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

/// Enhanced RPC client with connection pooling and retry logic
#[derive(Clone)]
#[allow(dead_code)]
pub struct RpcClient {
    client: reqwest::Client,
    url: String,
    timeout: Duration,
    max_retries: u32,
    retry_delay: Duration,
    connection_pool: Arc<Mutex<HashMap<String, reqwest::Client>>>,
}

#[allow(dead_code)]
impl RpcClient {
    pub fn new(rpc_port: u16) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .pool_max_idle_per_host(10)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            url: format!("http://127.0.0.1:{}/rpc", rpc_port),
            timeout: Duration::from_secs(5), // Default 5 second timeout
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            connection_pool: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_retries(mut self, max_retries: u32, retry_delay: Duration) -> Self {
        self.max_retries = max_retries;
        self.retry_delay = retry_delay;
        self
    }

    pub async fn send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let mut last_error = None;
        
        for attempt in 0..=self.max_retries {
            match self.try_send_request(method, params.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(self.retry_delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("RPC request failed after {} attempts", self.max_retries + 1)))
    }

    async fn try_send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
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

    /// Check if the RPC server is available
    pub async fn is_available(&self) -> bool {
        match self.send_request("status", json!({})).await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Get connection statistics
    pub async fn get_connection_stats(&self) -> Result<serde_json::Value> {
        self.send_request("connection_stats", json!({})).await
    }

    /// Subscribe to real-time updates
    pub async fn subscribe(&self, event_type: &str) -> Result<serde_json::Value> {
        self.send_request("subscribe", json!({ "event_type": event_type })).await
    }

    /// Unsubscribe from real-time updates
    pub async fn unsubscribe(&self, subscription_id: &str) -> Result<serde_json::Value> {
        self.send_request("unsubscribe", json!({ "subscription_id": subscription_id })).await
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<serde_json::Value> {
        self.send_request("performance_metrics", json!({})).await
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
            "x": pos[0],
            "y": pos[1],
            "z": pos[2]
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

/// Global RPC client instance
static mut RPC_CLIENT: Option<RpcClient> = None;

/// Initialize the global RPC client
pub fn init_rpc_client(rpc_port: u16) {
    unsafe {
        RPC_CLIENT = Some(RpcClient::new(rpc_port));
    }
}

/// Get the global RPC client
pub fn get_rpc_client() -> Option<&'static RpcClient> {
    unsafe {
        RPC_CLIENT.as_ref()
    }
}

/// Call RPC with default client
pub async fn call_rpc(method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
    if let Some(client) = get_rpc_client() {
        client.send_request(method, params).await
    } else {
        // Fallback to default client
        let client = RpcClient::new(9001);
        client.send_request(method, params).await
    }
}

/// Call RPC with custom timeout
#[allow(dead_code)]
pub async fn call_rpc_with_timeout(method: &str, params: serde_json::Value, timeout: Duration) -> Result<serde_json::Value> {
    if let Some(client) = get_rpc_client() {
        let client_with_timeout = client.clone().with_timeout(timeout);
        client_with_timeout.send_request(method, params).await
    } else {
        // Fallback to default client
        let client = RpcClient::new(9001).with_timeout(timeout);
        client.send_request(method, params).await
    }
}

/// Call RPC with retry logic
#[allow(dead_code)]
pub async fn call_rpc_with_retry(method: &str, params: serde_json::Value, max_retries: u32, retry_delay: Duration) -> Result<serde_json::Value> {
    if let Some(client) = get_rpc_client() {
        let client_with_retry = client.clone().with_retries(max_retries, retry_delay);
        client_with_retry.send_request(method, params).await
    } else {
        // Fallback to default client
        let client = RpcClient::new(9001).with_retries(max_retries, retry_delay);
        client.send_request(method, params).await
    }
} 