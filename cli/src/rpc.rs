//! # JSON-RPC types for Universectl

use serde::{Deserialize, Serialize};
use anyhow::Result;

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
    pub cosmic_era: String,
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

/// Helper function to make RPC calls to the simulation server
pub async fn call_rpc(method: &str, params: &serde_json::Value) -> Result<serde_json::Value> {
    let client = reqwest::Client::new();
    let req_body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error: {}", res.status()));
    }

    let rpc_res: RpcResponse<serde_json::Value> = res.json().await?;
    
    if let Some(error) = rpc_res.error {
        return Err(anyhow::anyhow!("RPC error: {} (code: {})", error.message, error.code));
    }

    rpc_res.result.ok_or_else(|| anyhow::anyhow!("No result in RPC response"))
} 