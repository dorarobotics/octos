//! Runtime bridge between octos tool calls and dora dataflow.
//!
//! The [`DoraBridge`] holds a sender half that tool execute() calls use to
//! forward requests to the dora event loop. The event loop (in the binary)
//! receives requests, sends them as dora outputs, collects the response
//! event, and sends it back through a oneshot channel.

use std::sync::mpsc;

/// A request from a tool's execute() to the dora event loop.
pub struct ToolRequest {
    /// Dora output ID to send the request on (e.g. "skill_request").
    pub output_id: String,
    /// JSON-encoded request payload.
    pub payload: Vec<u8>,
    /// Expected input ID for the response (e.g. "skill_result").
    pub response_id: String,
    /// Timeout in seconds.
    pub timeout_secs: u64,
    /// Channel to send the response back.
    pub reply_tx: mpsc::Sender<Result<String, String>>,
}

/// Sender half — cloned into each DoraToolBridge.
pub type BridgeSender = mpsc::Sender<ToolRequest>;

/// Receiver half — held by the dora event loop.
pub type BridgeReceiver = mpsc::Receiver<ToolRequest>;

/// Create a bridge channel pair.
pub fn bridge_channel() -> (BridgeSender, BridgeReceiver) {
    mpsc::channel()
}
