//! Real-time WebSocket communication for universe updates

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tracing::{debug, error};

use crate::{AppState, UniverseUpdate};

/// Handle WebSocket connection for real-time updates
pub async fn handle_websocket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.update_tx.subscribe();

    // Spawn task to forward universe updates to client
    let mut send_task = tokio::spawn(async move {
        while let Ok(update) = rx.recv().await {
            let msg = match serde_json::to_string(&update) {
                Ok(json) => Message::Text(json),
                Err(e) => {
                    error!("Failed to serialize update: {}", e);
                    continue;
                }
            };

            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages from client
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    debug!("Received WebSocket message: {}", text);
                    // Handle client commands (subscriptions, filters, etc.)
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    }
}