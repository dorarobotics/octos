//! # Inspection Safety — Permissions, Hooks, and Black-Box Recording
//!
//! ## The Problem
//!
//! An LLM-driven robot agent can call any tool it discovers. Without guardrails,
//! a single hallucinated tool call like `joint_full_actuation("max_speed")` during
//! a routine camera inspection could slam a 30kg arm into a gas pipeline valve at
//! full speed — destroying equipment or injuring a nearby operator.
//!
//! **Before these features:**
//! - Every tool is equally callable. The LLM decides safety.
//! - No spatial limits — the arm can reach anywhere, including into human zones.
//! - No audit trail — when something goes wrong at 3 AM, there's no record of
//!   what the agent did or why.
//! - No hook points — you can't intercept a dangerous motion before it executes.
//!
//! **After these features:**
//! - `SafetyTier` + `RobotPermissionPolicy` enforce a hard ceiling on what the
//!   agent CAN do, regardless of what the LLM asks for.
//! - `WorkspaceBounds` reject any motion target outside a physical safe zone.
//! - `HookEvent::BeforeMotion` lets external safety systems veto commands.
//! - `BlackBoxRecorder` writes every decision to JSONL for post-incident analysis.
//!
//! ## Scenario
//!
//! A gas pipeline valve inspection robot. The operator sets a `SafeMotion` session
//! (cameras + slow moves only). The agent can photograph valves and turn them
//! slowly, but cannot use full-speed actuation — even if the LLM requests it.
//! Every authorization decision and motion command is recorded to a black-box log.
//!
//! ## What This Example Demonstrates
//!
//! 1. **Tiered permissions** — how `authorize()` gates tools by safety level
//! 2. **Workspace bounds** — how `contains()` prevents out-of-bounds motion
//! 3. **Robot payloads** — how to attach joint/force data to hook events
//! 4. **Hook events** — the 5 robot-specific lifecycle points you can intercept
//! 5. **Black-box recording** — non-blocking JSONL logging for every safety event
//!
//! ```bash
//! cargo run --example inspection_safety -p octos-agent
//! ```

use std::path::PathBuf;

use octos_agent::{
    BlackBoxRecorder, HookEvent, RobotPayload,
    RobotPermissionPolicy, SafetyTier, WorkspaceBounds,
};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // ── Step 1: Define physical workspace bounds ──
    //
    // Problem: Without bounds, the LLM could command the arm to reach behind
    // the robot into a walkway where operators pass. WorkspaceBounds defines
    // an axis-aligned box (in meters) that the arm must stay inside.
    //
    // These values come from the cell safety assessment — the physical space
    // where the arm can move without hitting the pipeline, walls, or humans.
    let workspace = WorkspaceBounds {
        x_min: -0.5,  // 0.5m left of center
        x_max: 0.5,   // 0.5m right of center
        y_min: -0.3,  // 0.3m toward robot
        y_max: 0.3,   // 0.3m away from robot
        z_min: 0.0,   // table surface
        z_max: 0.8,   // 0.8m above table
    };

    // ── Step 2: Create a permission policy for this session ──
    //
    // The key insight: the OPERATOR (not the LLM) decides the safety ceiling.
    // For a routine inspection, SafeMotion is enough — cameras and slow moves.
    // The LLM can ask for FullActuation all day; the policy will deny it.
    //
    // Compare to "Before": without this, every tool is equally callable.
    let policy = RobotPermissionPolicy::new(SafetyTier::SafeMotion)
        .with_workspace(workspace.clone());

    println!("Session safety ceiling: {}", policy.max_tier);
    println!("(Operator chose SafeMotion — cameras + slow moves only)\n");

    // ── Step 3: See how authorize() gates tool access ──
    //
    // Each tool declares what tier it needs. authorize() compares that against
    // the session ceiling. This is the core safety guarantee: even if the LLM
    // hallucinates a dangerous tool call, the runtime blocks it.
    let tool_checks = [
        ("camera_capture", SafetyTier::Observe, "read-only, always safe"),
        ("valve_slow_turn", SafetyTier::SafeMotion, "slow motion within bounds"),
        ("joint_full_actuation", SafetyTier::FullActuation, "full-speed — BLOCKED"),
        ("emergency_override", SafetyTier::EmergencyOverride, "bypass all limits — BLOCKED"),
    ];

    println!("Tool authorization results:");
    for (tool, tier, why) in &tool_checks {
        let result = match policy.authorize(tool, *tier) {
            Ok(()) => format!("ALLOWED  (requires {tier}, session allows up to {})", policy.max_tier),
            Err(e) => format!("DENIED   ({e})"),
        };
        println!("  {tool:<24} {result}");
        println!("  {:<24} ^ {why}", "");
    }

    // ── Step 4: Validate motion targets against workspace bounds ──
    //
    // Problem: The LLM might generate a valid-looking joint command that moves
    // the TCP (tool center point) outside the safe zone. WorkspaceBounds catches
    // this at the coordinate level before any motion command is sent to hardware.
    let test_points = [
        (0.0, 0.0, 0.4, "valve V-101 (center of cell)"),
        (0.6, 0.0, 0.4, "valve V-202 (outside X limit — in adjacent cell!)"),
        (0.0, 0.0, -0.1, "below table surface (impossible / collision)"),
        (0.3, 0.2, 0.7, "valve V-103 (upper corner, within bounds)"),
    ];

    println!("\nWorkspace bounds validation (is the target reachable and safe?):");
    for (x, y, z, label) in &test_points {
        let safe = workspace.contains(*x, *y, *z);
        let verdict = if safe { "SAFE — proceed" } else { "REJECTED — outside workspace" };
        println!("  ({x:+.1}, {y:+.1}, {z:+.1}) {label}");
        println!("  {:<24} -> {verdict}", "");
    }

    // ── Step 5: Attach real robot state to hook events ──
    //
    // When a BeforeMotion hook fires, the safety system needs to know the
    // robot's CURRENT state (joint positions, velocity, forces) — not just
    // which tool was called. RobotPayload carries this physical context.
    //
    // Benefit: External monitors (ROS safety node, PLC watchdog) can evaluate
    // the actual robot state and veto the motion if anything looks wrong.
    let motion_payload = RobotPayload::for_motion(
        vec![0.0, 0.5, -0.3, 1.2, 0.0, 0.0],  // 6 joint positions in radians
        Some(0.15),                               // 0.15 m/s linear velocity
    );
    println!("\nRobotPayload for BeforeMotion hook:");
    println!("  Joint positions: {:?}", motion_payload.joint_positions);
    println!("  Velocity: {:?} m/s", motion_payload.velocity);
    println!("  (External safety node receives this and can DENY the motion)");

    // Force/torque payload — fires when sensors detect unexpected resistance
    let force_payload = RobotPayload::for_force(vec![1.2, 0.3, 15.8, 0.0, 0.0, 0.0]);
    println!("\nRobotPayload for ForceLimit hook:");
    println!("  Force/torque: {:?} (Fz=15.8N — valve stuck?)", force_payload.force_torque);

    // ── Step 6: See the robot-specific hook events ──
    //
    // These are the lifecycle points where you can intercept robot actions.
    // Before-hooks can DENY operations (exit code 1 = veto).
    //
    // Benefit: You wire these to your existing safety infrastructure (ROS,
    // PLC, safety relay) without changing the agent code.
    println!("\nRobot safety hook events (wire these to your safety PLC/ROS node):");
    let events_explained = [
        (HookEvent::BeforeMotion, "Fires BEFORE any motion. Hook can DENY to prevent it."),
        (HookEvent::AfterMotion, "Fires AFTER motion completes. Log result, update state."),
        (HookEvent::ForceLimit, "Fires when force/torque exceeds limits. Trigger safe-hold."),
        (HookEvent::WorkspaceBoundary, "Fires when target is outside bounds. Auto-denied."),
        (HookEvent::EmergencyStop, "Fires on e-stop. All motion halts immediately."),
    ];
    for (event, explanation) in &events_explained {
        println!("  {event:?}");
        println!("    {explanation}");
    }

    // ── Step 7: Record everything to a black-box log ──
    //
    // Problem: At 3 AM the robot damages a valve. The operator asks "what
    // happened?" Without a recorder, the only evidence is scattered logs.
    //
    // BlackBoxRecorder writes structured JSONL — one JSON object per line —
    // that captures every safety decision, motion command, and sensor reading.
    // Non-blocking: if the file system is slow, entries are dropped rather than
    // stalling the control loop.
    //
    // After an incident, you can replay the log to see exactly:
    // - What tools were authorized/denied
    // - What motion commands were sent
    // - What force readings triggered limits
    let log_path = PathBuf::from("/tmp/inspection_safety_demo.jsonl");
    let recorder = BlackBoxRecorder::new(log_path.clone(), 256).await?;

    // Record the session setup
    recorder.log("session_start", serde_json::json!({
        "policy_tier": policy.max_tier.label(),
        "workspace": {
            "x": [workspace.x_min, workspace.x_max],
            "y": [workspace.y_min, workspace.y_max],
            "z": [workspace.z_min, workspace.z_max],
        },
    }));

    // Record authorization decisions (both allowed and denied)
    recorder.log("authorization", serde_json::json!({
        "tool": "camera_capture", "required_tier": "observe", "result": "allowed",
    }));
    recorder.log("authorization", serde_json::json!({
        "tool": "joint_full_actuation", "required_tier": "full_actuation", "result": "denied",
        "reason": "session ceiling is safe_motion",
    }));

    // Record a motion command with full robot state
    recorder.log("before_motion", serde_json::json!({
        "tool": "valve_slow_turn",
        "joint_positions": motion_payload.joint_positions,
        "velocity_ms": motion_payload.velocity,
    }));

    // Record a force limit event
    recorder.log("force_limit", serde_json::json!({
        "force_torque": force_payload.force_torque,
        "fz_newtons": 15.8,
        "threshold_newtons": 10.0,
        "action": "safe_hold",
    }));

    assert!(recorder.is_active(), "recorder should be active");

    // Drop flushes the channel. In production, the recorder lives for the
    // entire session and writes continuously.
    drop(recorder);
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    println!("\nBlack-box JSONL log written to: {}", log_path.display());
    println!("  (Each line is a self-contained JSON object for post-incident replay)");
    println!("\nInspection safety demo complete.");
    Ok(())
}
