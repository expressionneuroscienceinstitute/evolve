//! Networking
//! 
//! Handles distributed simulation networking

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use tokio::sync::mpsc; // For asynchronous message passing

// --- Core Concepts ---

/// Represents a unique identifier for a network node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    pub fn new(id: u64) -> Self {
        NodeId(id)
    }
}

/// Represents a message exchanged between network nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NetworkMessage {
    /// A command to a specific node.
    Command { target_node: NodeId, payload: String },
    /// A data packet containing simulation state updates or workload.
    Data { sender_node: NodeId, payload: Vec<u8> },
    /// A request for a specific resource or data.
    Request { requester_node: NodeId, resource_id: String },
    /// A response to a request.
    Response { responder_node: NodeId, request_id: String, payload: Vec<u8> },
    /// A heartbeat signal to indicate node liveness.
    Heartbeat { node_id: NodeId, timestamp: u64 },
}

/// Represents a unit of simulation work to be distributed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkloadPacket {
    pub task_id: u64,
    pub simulation_segment: Vec<u8>, // Serialized portion of simulation state
    pub instructions: String,
}

// --- Distributed Simulation Architecture ---

/// Manages communication, workload distribution, and state synchronization for a single network node.
#[derive(Debug)]
pub struct NetworkNode {
    pub id: NodeId,
    /// Sender for outgoing messages to other nodes.
    message_tx: mpsc::Sender<NetworkMessage>,
    /// Receiver for incoming messages from other nodes.
    message_rx: mpsc::Receiver<NetworkMessage>,
    /// Map of other connected nodes.
    connected_nodes: HashMap<NodeId, mpsc::Sender<NetworkMessage>>,
    /// Local queue for incoming workload.
    workload_queue: VecDeque<WorkloadPacket>,
    /// Placeholder for local simulation state.
    local_simulation_state: Vec<u8>,
}

impl NetworkNode {
    /// Creates a new network node.
    pub fn new(id: u64, buffer_size: usize) -> Self {
        let (message_tx, message_rx) = mpsc::channel(buffer_size);
        NetworkNode {
            id: NodeId::new(id),
            message_tx,
            message_rx,
            connected_nodes: HashMap::new(),
            workload_queue: VecDeque::new(),
            local_simulation_state: Vec::new(), // Initialize as empty
        }
    }

    /// Connects to another network node.
    pub async fn connect_to_node(&mut self, node_id: NodeId, sender: mpsc::Sender<NetworkMessage>) {
        self.connected_nodes.insert(node_id, sender);
        println!("Node {:?} connected to node {:?}", self.id, node_id);
    }

    /// Sends a message to a target node.
    pub async fn send_message(&self, message: NetworkMessage) -> Result<()> {
        // In a real distributed system, this would involve serialization and sending over TCP/UDP.
        // For now, we simulate by sending to the appropriate mpsc channel if the node is local.
        match &message {
            NetworkMessage::Command { target_node, .. } => {
                if let Some(tx) = self.connected_nodes.get(target_node) {
                    tx.send(message).await?;
                } else {
                    eprintln!("Error: Target node {:?} not found for command.", target_node);
                }
            },
            NetworkMessage::Data { sender_node: _, payload: _ } => {
                // For data, we might broadcast or send to a specific recipient based on context.
                // For simplicity, we'll assume direct send to a known node or central coordinator.
                // This part needs more detailed design based on actual data flow.
                eprintln!("Warning: Data message sent without specific recipient handling.");
            },
            _ => {
                eprintln!("Unhandled message type for sending: {:?}", message);
            }
        }
        Ok(())
    }

    /// Receives and processes incoming messages.
    pub async fn receive_and_process_messages(&mut self) -> Result<()> {
        while let Some(message) = self.message_rx.recv().await {
            match message {
                NetworkMessage::Command { target_node, payload } => {
                    println!("Node {:?} received command for {:?}: {}", self.id, target_node, payload);
                    // Process command: e.g., update local state, change behavior
                },
                NetworkMessage::Data { sender_node, payload } => {
                    println!("Node {:?} received data from {:?}: {:?}", self.id, sender_node, payload);
                    // Integrate data into local simulation state or add to workload queue
                    self.workload_queue.push_back(WorkloadPacket { task_id: 0, simulation_segment: payload, instructions: "Process data".to_string() });
                },
                NetworkMessage::Request { requester_node, resource_id } => {
                    println!("Node {:?} received request from {:?}", self.id, requester_node);
                    // Respond to request: e.g., send back part of local_simulation_state
                    let response_payload = self.local_simulation_state.clone(); // Example response
                    let response = NetworkMessage::Response {
                        responder_node: self.id,
                        request_id: resource_id,
                        payload: response_payload,
                    };
                    if let Some(tx) = self.connected_nodes.get(&requester_node) {
                        tx.send(response).await?;
                    }
                },
                NetworkMessage::Response { responder_node, request_id, payload } => {
                    println!("Node {:?} received response from {:?} for request {}: {:?}", self.id, responder_node, request_id, payload);
                    // Process response: e.g., update requested data, continue simulation
                },
                NetworkMessage::Heartbeat { node_id: _, timestamp: _ } => {
                    // println!("Node {:?} received heartbeat from {:?} at {}", self.id, node_id, timestamp);
                    // Update liveness status of node, check for failures
                },
            }
        }
        Ok(())
    }

    /// Distributes a workload packet to a connected node (example).
    pub async fn distribute_workload(&self, _target_node: NodeId, packet: WorkloadPacket) -> Result<()> {
        let message = NetworkMessage::Data { sender_node: self.id, payload: packet.simulation_segment };
        self.send_message(message).await
    }

    /// Processes a unit of workload (placeholder).
    pub fn process_workload(&mut self) -> Result<()> {
        if let Some(packet) = self.workload_queue.pop_front() {
            println!("Node {:?} processing task {}: {}", self.id, packet.task_id, packet.instructions);
            // In a real system, this would involve deserializing simulation_segment
            // and performing physics/agent evolution steps.
            self.local_simulation_state = packet.simulation_segment; // Update local state as if processed
        }
        Ok(())
    }

    /// Synchronizes a portion of the local simulation state with other nodes.
    /// This could be periodic or event-driven.
    pub async fn synchronize_state(&self, _target_node: NodeId) -> Result<()> {
        let message = NetworkMessage::Data { sender_node: self.id, payload: self.local_simulation_state.clone() };
        self.send_message(message).await
    }
}

/// Entry point for setting up and running a distributed simulation node.
pub async fn run_network_node(node_id: u64, buffer_size: usize) -> Result<()> {
    let mut node = NetworkNode::new(node_id, buffer_size);
    println!("Network node {:?} started.", node.id);

    // Example: Simulate connecting to a central coordinator or other nodes
    // In a real scenario, this would involve discovery or a known list of endpoints.
    // For demonstration, we'll just set up a dummy connection.
    let (dummy_tx, mut dummy_rx) = mpsc::channel(buffer_size);
    node.connect_to_node(NodeId::new(999), dummy_tx).await; // Connect to a dummy coordinator

    // Spawn a task to handle incoming messages
    let _node_clone = node.id;
    tokio::spawn(async move {
        while let Some(_message) = dummy_rx.recv().await {
            // Dummy processing for messages from coordinator
            // println!("Dummy coordinator received message for {:?}: {:?}", node_clone, message);
        }
    });

    // Main loop for the node to process messages and workload
    loop {
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {
                // Simulate periodic activities like sending heartbeats or processing workload
                node.process_workload()?;
                // node.send_message(NetworkMessage::Heartbeat { node_id: node.id, timestamp: chrono::Utc::now().timestamp() as u64 }).await?;
            }
            _ = node.receive_and_process_messages() => {
                // This branch will be active when messages arrive
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_node_creation() {
        let node = NetworkNode::new(1, 10);
        assert_eq!(node.id.0, 1);
        assert!(node.connected_nodes.is_empty()); // New nodes start with no connections
    }

    #[test]
    fn test_message_sending_and_receiving() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let (tx1, rx1) = mpsc::channel(10);
            let (tx2, rx2) = mpsc::channel(10);

            let mut node1 = NetworkNode::new(1, 10);
            let mut node2 = NetworkNode::new(2, 10);

            node1.connect_to_node(NodeId::new(2), tx2).await;
            node2.connect_to_node(NodeId::new(1), tx1).await;

            let msg = NetworkMessage::Command { target_node: NodeId::new(2), payload: "test command".to_string() };
            node1.send_message(msg.clone()).await.unwrap();

            let received_msg = node2.message_rx.recv().await.unwrap();
            assert_eq!(received_msg, msg);
        });
    }
}