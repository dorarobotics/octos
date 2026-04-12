//! octos-dora-agent — Rust dora node for bridging octos tools to dataflow.
//!
//! Phase 1: Tool bridge — loads tool mappings from JSON, forwards
//! skill_request/skill_result between the octos agent and dora nodes.
//!
//! Phase 2 (planned): Embed the full octos Agent loop with LLM provider,
//! pipeline executor, and safety tier enforcement.
//!
//! # Usage
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
//!     SAFETY_TIER: "safe_motion"
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
    let safety_tier = env::var("SAFETY_TIER").unwrap_or_else(|_| "safe_motion".to_string());

    println!("==================================================");
    println!("  octos-dora-agent v{}", env!("CARGO_PKG_VERSION"));
    if !tool_map_path.is_empty() {
        println!("  tool_map:  {tool_map_path}");
    }
    println!("  tier:      {safety_tier}");
    println!("==================================================");

    // Initialize dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    println!("[agent] Dora node initialized");

    // Create bridge channel
    let (bridge_tx, bridge_rx) = bridge_channel();

    // Load tool mappings
    if !tool_map_path.is_empty() {
        let config = BridgeConfig::from_file(&tool_map_path)?;
        let bridges = load_bridges_with_sender(&config, bridge_tx.clone());
        println!("[agent] Loaded {} tools:", bridges.len());
        for b in &bridges {
            println!(
                "  {} [{}] → {}/{}",
                b.name(),
                b.required_safety_tier().as_str(),
                b.mapping().dora_node_id,
                b.mapping().dora_output_id,
            );
        }
    } else {
        println!("[agent] No tool map — running as event forwarder");
    }

    println!("[agent] Event loop started");

    // Main event loop
    loop {
        // 1. Check for bridge requests (from tool execute() calls in agent thread)
        if let Ok(req) = bridge_rx.try_recv() {
            let output_id = DataId::from(req.output_id.clone());
            println!("[agent] → {} ({} bytes)", req.output_id, req.payload.len());

            if let Err(e) = node.send_output_raw(
                output_id,
                Default::default(),
                req.payload.len(),
                |out| out.copy_from_slice(&req.payload),
            ) {
                eprintln!("[agent] Send error: {e}");
                let _ = req.reply_tx.send(Err(format!("send failed: {e}")));
                continue;
            }

            // Wait for response
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
                            let _ = req.reply_tx.send(Err("stopped".to_string()));
                            println!("[agent] STOP");
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }

            if !got_response {
                let _ = req.reply_tx.send(Err(format!(
                    "timeout: {} ({}s)", req.response_id, req.timeout_secs
                )));
            }
            continue;
        }

        // 2. Process dora events
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
