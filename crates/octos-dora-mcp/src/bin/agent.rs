//! octos-dora-agent — Rust dora node for robot agent execution.
//!
//! Two modes:
//!   1. Pipeline mode (OCTOS_PIPELINE set): executes DOT pipeline deterministically
//!   2. Bridge mode (no pipeline): waits for external tool calls via bridge channel
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
//!     OCTOS_PIPELINE: "patrol.dot"
//! ```

use dora_node_api::DoraNode;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::arrow::array::UInt8Array;
use eyre::Result;
use octos_agent::tools::Tool;
use octos_dora_mcp::{bridge_channel, load_bridges_with_sender, BridgeConfig, BridgeSender};
use octos_dora_mcp::pipeline::DotPipeline;
use std::env;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Send a tool call through the bridge and wait for the dora response.
fn call_tool(
    tool_name: &str,
    args: &serde_json::Value,
    bridge_tx: &BridgeSender,
    output_id: &str,
    timeout_secs: u64,
) -> Result<String> {
    let payload = serde_json::json!({
        "tool": tool_name,
        "args": args,
    });

    let (reply_tx, reply_rx) = mpsc::channel();

    let request = octos_dora_mcp::ToolRequest {
        output_id: output_id.to_string(),
        payload: serde_json::to_vec(&payload)?,
        response_id: "skill_result".to_string(),
        timeout_secs,
        reply_tx,
    };

    bridge_tx
        .send(request)
        .map_err(|e| eyre::eyre!("bridge send failed: {e}"))?;

    match reply_rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(e)) => Err(eyre::eyre!("tool error: {e}")),
        Err(_) => Err(eyre::eyre!("timeout after {timeout_secs}s")),
    }
}

/// Run pipeline mode: step through DOT nodes and call tools.
fn run_pipeline(
    pipeline: &DotPipeline,
    bridge_tx: &BridgeSender,
) {
    let total = pipeline.nodes.len();
    let max_cycles = if pipeline.is_cyclic && pipeline.max_cycles > 0 {
        pipeline.max_cycles
    } else if pipeline.is_cyclic {
        1 // default 1 cycle if cyclic but no max
    } else {
        1
    };

    for cycle in 0..max_cycles {
        if pipeline.is_cyclic && max_cycles > 1 {
            println!("  [pipeline] Cycle {}/{max_cycles}", cycle + 1);
        }

        for (i, node) in pipeline.nodes.iter().enumerate() {
            let step = i + 1;

            if node.is_gate() {
                println!("  [pipeline] Gate {step}/{total}: {} — auto-approved", node.label);
                continue;
            }

            if let Some(ref tool_name) = node.tool {
                let args = node.args.clone().unwrap_or(serde_json::json!({}));
                println!("  [pipeline] Step {step}/{total}: {} → {tool_name}({args})", node.label);

                match call_tool(tool_name, &args, bridge_tx, "skill_request", 120) {
                    Ok(result) => {
                        let preview = if result.len() > 200 {
                            format!("{}...", &result[..200])
                        } else {
                            result.clone()
                        };
                        println!("  [pipeline]   Result: {preview}");
                        if let Some(ref cp) = node.checkpoint {
                            println!("  [pipeline]   Checkpoint: {cp}");
                        }
                    }
                    Err(e) => {
                        println!("  [pipeline]   FAILED: {e}");
                    }
                }
            } else {
                println!("  [pipeline] Step {step}/{total}: {} (no tool)", node.label);
            }
        }
    }

    println!("  [pipeline] Complete!");
}

fn main() -> Result<()> {
    let tool_map_path = env::var("OCTOS_TOOL_MAP").unwrap_or_default();
    let pipeline_path = env::var("OCTOS_PIPELINE").unwrap_or_default();
    let safety_tier = env::var("SAFETY_TIER").unwrap_or_else(|_| "safe_motion".to_string());

    println!("==================================================");
    println!("  octos-dora-agent v{}", env!("CARGO_PKG_VERSION"));
    if !tool_map_path.is_empty() {
        println!("  tool_map:  {tool_map_path}");
    }
    if !pipeline_path.is_empty() {
        println!("  pipeline:  {pipeline_path}");
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
                "  {} [{}]",
                b.name(),
                b.required_safety_tier().as_str(),
            );
        }
    }

    // Load pipeline
    let pipeline = if !pipeline_path.is_empty() {
        let p = DotPipeline::from_file(&pipeline_path)?;
        println!(
            "[agent] Pipeline: {} ({} nodes, {} edges, cyclic={})",
            p.name,
            p.nodes.len(),
            p.edges.len(),
            p.is_cyclic,
        );
        Some(p)
    } else {
        None
    };

    // If pipeline mode, spawn pipeline executor in background thread
    let pipeline_handle = if let Some(ref p) = pipeline {
        let p_clone = DotPipeline {
            name: p.name.clone(),
            nodes: p.nodes.clone(),
            edges: p.edges.clone(),
            is_cyclic: p.is_cyclic,
            max_cycles: p.max_cycles,
        };
        let tx = bridge_tx.clone();
        Some(std::thread::spawn(move || {
            // Small delay to let dora event loop start
            std::thread::sleep(Duration::from_secs(2));
            run_pipeline(&p_clone, &tx);
        }))
    } else {
        None
    };

    // Main thread: dora event loop — forward bridge requests
    println!("[agent] Event loop started");

    loop {
        // Check for bridge requests
        if let Ok(req) = bridge_rx.try_recv() {
            let output_id = DataId::from(req.output_id.clone());

            if let Err(e) = node.send_output_raw(
                output_id,
                Default::default(),
                req.payload.len(),
                |out| out.copy_from_slice(&req.payload),
            ) {
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
                                let _ = req.reply_tx.send(Ok(response));
                                got_response = true;
                                break;
                            }
                        }
                        dora_node_api::Event::Stop(_) => {
                            let _ = req.reply_tx.send(Err("stopped".to_string()));
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

        // Check if pipeline thread finished
        if let Some(ref h) = pipeline_handle {
            if h.is_finished() {
                println!("[agent] Pipeline finished, exiting");
                break;
            }
        }

        // Brief sleep to avoid busy-waiting
        std::thread::sleep(Duration::from_millis(10));
    }

    Ok(())
}
