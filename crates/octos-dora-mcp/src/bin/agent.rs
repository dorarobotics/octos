//! octos-dora-agent — Rust binary that runs as a dora dataflow node.
//!
//! Replaces the Python agent_node.py with the real octos-agent Rust crate.
//! Loads tool mappings from a JSON config, connects to the dora dataflow,
//! and forwards tool calls between the octos agent and dora nodes.
//!
//! # Dora dataflow YAML
//!
//! ```yaml
//! - id: agent
//!   custom:
//!     source: Local
//!     path: octos-dora-agent
//!     inputs:
//!       skill_result: nav-bridge/skill_result
//!     outputs:
//!       - skill_request
//!   env:
//!     OCTOS_TOOL_MAP: "nav_tool_map.json"
//!     USER_COMMAND: "Patrol stations A, B, home"
//! ```

use dora_node_api::DoraNode;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::arrow::array::UInt8Array;
use eyre::Result;
use octos_agent::tools::Tool;
use octos_dora_mcp::{bridge_channel, load_bridges_with_sender, BridgeConfig};
use std::env;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let tool_map_path = env::var("OCTOS_TOOL_MAP").unwrap_or_default();
    let provider_name = env::var("OCTOS_PROVIDER").unwrap_or_else(|_| "mock".to_string());
    let user_command = env::var("USER_COMMAND").unwrap_or_else(|_| "Execute pipeline".to_string());
    let safety_tier = env::var("SAFETY_TIER").unwrap_or_else(|_| "safe_motion".to_string());

    println!("==================================================");
    println!("  octos-dora-agent (Rust)");
    if !tool_map_path.is_empty() {
        println!("  tool_map:  {tool_map_path}");
    }
    println!("  provider:  {provider_name}");
    println!("  tier:      {safety_tier}");
    println!("  command:   {user_command}");
    println!("==================================================");

    // Initialize dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    println!("[agent] Dora node initialized");

    // Load tool mappings and create bridge
    let (bridge_tx, bridge_rx) = bridge_channel();

    if !tool_map_path.is_empty() {
        let config = BridgeConfig::from_file(&tool_map_path)?;
        let bridges = load_bridges_with_sender(&config, bridge_tx.clone());
        println!("[agent] Loaded {} tools from {tool_map_path}", bridges.len());
        for b in &bridges {
            println!(
                "  {} [{}] → {}/{}",
                b.name(),
                b.required_safety_tier().as_str(),
                b.mapping().dora_node_id,
                b.mapping().dora_output_id,
            );
        }
    }

    println!("[agent] Entering event loop — command: {user_command}");

    // Main event loop: forward bridge requests to dora and route responses back
    loop {
        // Check for bridge requests (from tool execute() calls)
        if let Ok(req) = bridge_rx.try_recv() {
            let output_id = DataId::from(req.output_id.clone());
            println!("[agent] → {} ({} bytes)", req.output_id, req.payload.len());

            node.send_output_raw(
                output_id,
                Default::default(),
                req.payload.len(),
                |out| {
                    out.copy_from_slice(&req.payload);
                },
            )?;

            // Wait for the response event
            let deadline = Instant::now() + Duration::from_secs(req.timeout_secs);
            let mut got_response = false;

            while Instant::now() < deadline {
                if let Some(event) = events.recv() {
                    match event {
                        dora_node_api::Event::Input { id, data, .. } => {
                            if id.as_str() == req.response_id {
                                let bytes: Vec<u8> = data
                                    .0
                                    .as_any()
                                    .downcast_ref::<UInt8Array>()
                                    .map(|arr| arr.values().to_vec())
                                    .unwrap_or_default();
                                let response = String::from_utf8_lossy(&bytes).to_string();
                                println!("[agent] ← {} ({} bytes)", req.response_id, bytes.len());
                                let _ = req.reply_tx.send(Ok(response));
                                got_response = true;
                                break;
                            }
                        }
                        dora_node_api::Event::Stop(_) => {
                            println!("[agent] Received STOP");
                            let _ = req.reply_tx.send(Err("Node stopped".to_string()));
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }

            if !got_response {
                println!("[agent] Timeout waiting for {}", req.response_id);
                let _ = req.reply_tx.send(Err(format!(
                    "Timeout: {} ({}s)",
                    req.response_id, req.timeout_secs
                )));
            }
            continue;
        }

        // Process dora events when no bridge requests pending
        if let Some(event) = events.recv() {
            match event {
                dora_node_api::Event::Stop(_) => {
                    println!("[agent] Stopped");
                    break;
                }
                dora_node_api::Event::Input { id, .. } => {
                    println!("[agent] Input: {}", id.as_str());
                }
                _ => {}
            }
        }
    }

    Ok(())
}
