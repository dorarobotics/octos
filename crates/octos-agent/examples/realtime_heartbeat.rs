//! # Realtime Heartbeat — Stall Detection and Sensor Context Injection
//!
//! ## The Problem
//!
//! A robot agent running 24/7 makes LLM calls every loop iteration. If the LLM
//! provider hangs, the network drops, or the agent deadlocks, the robot silently
//! stops responding — but the hardware is still powered and potentially in a
//! dangerous position. Nobody notices until something goes wrong.
//!
//! Meanwhile, the LLM makes decisions with zero awareness of the robot's physical
//! state. It doesn't know the battery is at 15%, the force sensor reads 50N
//! (something is stuck), or the lidar shows an obstacle 0.3m ahead. It just
//! sees text and tool results.
//!
//! **Before these features:**
//! - No stall detection — a hung LLM call means the robot freezes indefinitely.
//!   No alert, no safe-hold, no timeout.
//! - No sensor awareness — the LLM plans actions without knowing joint positions,
//!   battery level, or obstacle proximity. It's flying blind.
//! - Sensor data is only available through explicit tool calls. The LLM has to
//!   remember to call `read_sensors` — and it often doesn't.
//!
//! **After these features:**
//! - `Heartbeat` detects stalls within a configurable timeout. An external
//!   monitor reads `state()` and triggers safe-hold if the agent stops beating.
//! - `SensorContextInjector` automatically prepends live sensor data to every
//!   LLM prompt. The LLM always sees current robot state without needing to
//!   call a tool first.
//! - `RealtimeConfig` sets timing budgets so the agent never hangs indefinitely.
//!
//! ## Scenario
//!
//! A security patrol robot runs 24/7 in a warehouse. Every few seconds it loops:
//! read sensors -> ask LLM -> execute tools -> repeat. This example simulates
//! the heartbeat cycle, a stall event, recovery, and how sensor data gets
//! formatted for LLM injection.
//!
//! ```bash
//! cargo run --example realtime_heartbeat -p octos-agent
//! ```

use std::time::Duration;

use octos_agent::{
    Heartbeat, HeartbeatState, RealtimeConfig, SensorContextInjector, SensorSnapshot,
};

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[tokio::main]
async fn main() {
    // ── Step 1: Configure timing budgets ──
    //
    // These values are the difference between "the robot freezes for 30 seconds
    // while the LLM thinks" and "the robot enters safe-hold after 5 seconds of
    // no response." Each field answers a specific failure question:
    //
    // - iteration_deadline_ms: "What if tool execution hangs?"
    //   -> Log a warning and move on after 3s.
    // - heartbeat_timeout_ms: "What if the entire agent loop freezes?"
    //   -> External monitor detects stall after 5s, triggers safe-hold.
    // - llm_timeout_ms: "What if the LLM provider hangs?"
    //   -> Abort the LLM call after 4s, use fallback behavior.
    // - min_cycle_ms: "What if iterations are too fast?"
    //   -> Sleep to prevent CPU thrashing (200ms minimum cycle).
    let config = RealtimeConfig {
        iteration_deadline_ms: 3000,
        heartbeat_timeout_ms: 5000,
        llm_timeout_ms: 4000,
        min_cycle_ms: 200,
        check_estop: true,
    };
    println!("RealtimeConfig (timing budgets for failure scenarios):");
    println!("  iteration_deadline: {}ms  (max time per loop iteration)", config.iteration_deadline_ms);
    println!("  heartbeat_timeout:  {}ms  (stall detection window)", config.heartbeat_timeout_ms);
    println!("  llm_timeout:        {}ms  (abort hung LLM calls)", config.llm_timeout_ms);
    println!("  min_cycle:          {}ms  (prevent CPU thrashing)", config.min_cycle_ms);
    println!("  check_estop:        {}       (check e-stop each iteration)", config.check_estop);

    // ── Step 2: Heartbeat — detect when the agent loop stops ──
    //
    // In production, this runs in a separate monitoring thread or process.
    // The agent calls beat() every iteration. The monitor calls state() on a
    // timer. If state() returns Stalled, the monitor triggers safe-hold:
    //   - Command all joints to hold position
    //   - Activate brakes
    //   - Alert the operator
    //
    // This is like a deadman's switch: the agent must actively prove it's alive.
    let heartbeat = Heartbeat::new(Duration::from_millis(config.heartbeat_timeout_ms));

    // Normal operation: agent beats every iteration
    println!("\n--- Normal operation (agent loop is healthy) ---");
    for i in 1..=5 {
        heartbeat.beat();  // Called at the top of each agent loop iteration
        let state = heartbeat.state();
        println!("  Iteration {i}: beat #{}, state = {state:?}", heartbeat.count());
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Simulate a stall: the LLM call hangs, no beats for 5+ seconds
    println!("\n--- Stall simulation (LLM provider hangs, no beats) ---");
    println!("  Waiting {}ms with no heartbeat.beat() calls...", config.heartbeat_timeout_ms + 500);
    let _ = heartbeat.state(); // Consume current check value
    tokio::time::sleep(Duration::from_millis(config.heartbeat_timeout_ms + 500)).await;

    let stall_state = heartbeat.state();
    println!("  Heartbeat state: {stall_state:?}");
    println!("  -> In production: trigger SAFE-HOLD, alert operator");
    assert_eq!(stall_state, HeartbeatState::Stalled);

    // Recovery: agent resumes after the stall is resolved
    println!("\n--- Recovery (issue resolved, agent resumes) ---");
    heartbeat.beat();
    let recovered = heartbeat.state();
    println!("  After recovery beat: {recovered:?}");
    println!("  -> In production: release safe-hold, resume operation");
    assert_eq!(recovered, HeartbeatState::Alive);

    // ── Step 3: Sensor context injection — give the LLM physical awareness ──
    //
    // Problem: The LLM plans "move to valve V-101" but doesn't know the battery
    // is at 15%, which means the robot might die mid-motion. Or it plans a grasp
    // but doesn't know the force sensor already reads 40N (something is jammed).
    //
    // SensorContextInjector solves this by maintaining a ring buffer of recent
    // sensor readings and formatting them as a text block that gets prepended to
    // every LLM prompt. The LLM always sees current physical state.
    //
    // Ring buffer (capacity=8) means old readings are evicted automatically —
    // the LLM sees the most recent snapshot, not stale data from 5 minutes ago.
    let mut injector = SensorContextInjector::new(8);

    // Simulate sensor readings arriving from the robot's ROS/dora-rs pipeline
    let sensors = [
        ("lidar_front", serde_json::json!({"range_m": 3.2, "clear": true})),
        ("battery", serde_json::json!({"voltage": 24.1, "soc_pct": 78})),
        ("joint_positions", serde_json::json!([0.0, 0.5, -0.3, 1.2, 0.0, 0.0])),
        ("imu", serde_json::json!({"roll": 0.01, "pitch": -0.02, "yaw": 1.57})),
        ("force_torque", serde_json::json!([0.5, 0.1, 9.8, 0.0, 0.0, 0.0])),
    ];

    println!("\n--- Sensor context injection ---");
    println!("Pushing {} sensor snapshots into ring buffer (capacity=8):", sensors.len());
    for (id, value) in sensors {
        let snapshot = SensorSnapshot {
            sensor_id: id.to_string(),
            value,
            timestamp_ms: now_ms(),
        };
        injector.push(snapshot);
    }
    println!("  Buffer: {}/8 slots used", injector.len());

    // Query a specific sensor (useful for conditional logic in the agent loop)
    if let Some(battery) = injector.latest("battery") {
        let soc = battery.value.get("soc_pct").and_then(|v| v.as_u64()).unwrap_or(0);
        println!("\n  Latest battery reading: {}% SoC", soc);
        if soc < 20 {
            println!("  -> WARNING: Low battery! Agent should return to charging dock.");
        } else {
            println!("  -> Battery OK, continue patrol.");
        }
    }

    // Format the context block that gets prepended to the LLM system prompt.
    // This is what the LLM "sees" about the robot's physical state:
    let context_block = injector.to_context_block();
    println!("\n  This text block is prepended to every LLM prompt:");
    println!("  ┌────────────────────────────────────────────");
    for line in context_block.lines() {
        println!("  | {line}");
    }
    println!("  └────────────────────────────────────────────");
    println!("  The LLM now knows joint positions, battery, obstacles, etc.");
    println!("  It can reason: 'force_torque Fz=9.8N is just gravity, safe to proceed.'");

    println!("\nRealtime heartbeat demo complete.");
}
