# Release Notes

## v0.2.9 — SLAM Navigation Visual Simulation Example

- Add `slam-nav-sim` example: 9-node dora dataflow for visual SLAM navigation with MuJoCo + dora-nav + Rerun
- Hunter SE robot follows 1065-waypoint warehouse path with live 3D visualization (pointcloud, path, robot body)
- Navigation bridge node: SteeringCmd/TrqBreCmd → differential-drive wheel commands
- Optional LLM agent (robot-edge-a) for autonomous navigation via octos agent pattern
- `nav_tool_map.json` with 6 tools across 3 safety tiers (observe, safe_motion, full_actuation)
- `patrol_mission.dot` pipeline with deadlines, invariants, and checkpoints
- Vendored octos Python framework for standalone execution
- Stripped from 16-node `octos_inspection` — removes arm pipeline + cloud brain for focused nav demo

## v0.2.8 — Developer Examples for Robot Safety Features

- Add `inspection_safety` example (octos-agent): demonstrates Permissions, Hooks, and BlackBoxRecorder with a gas pipeline valve inspection scenario
- Add `realtime_heartbeat` example (octos-agent): demonstrates RealtimeConfig, Heartbeat stall detection, and SensorContextInjector ring buffer
- Add `pick_and_place_lifecycle` example (octos-plugin): demonstrates HardwareLifecycle phases with LifecycleExecutor retry/timeout handling
- Add `dora-bridge-config` config example: sample `dora_tool_map.json` and `inspection_mission.dot` DOT pipeline with HandlerKind, DeadlineAction, Invariant, and MissionCheckpoint usage
- Re-export realtime types (`RealtimeConfig`, `Heartbeat`, `HeartbeatState`, `SensorSnapshot`, `SensorContextInjector`) from `octos_agent` top-level

## v0.2.1 — Tiered Permission Model

- Add `SafetyTier` enum (`Observe`, `SafeMotion`, `FullActuation`, `EmergencyOverride`) with `PartialOrd` ordering
- Add `WorkspaceBounds` for axis-aligned safe motion validation
- Add `RobotPermissionPolicy` with `authorize()` method for tier-based tool gating
- Add `PermissionDenied` error type with descriptive messages
- Extend `Tool` trait with `required_safety_tier()` default method (returns `Observe`)
- New module: `crates/octos-agent/src/permissions.rs`
- 6 unit tests covering authorization, denial, ordering, serialization, and defaults
- Fix pre-existing `profile_name` field missing in sandbox test structs
