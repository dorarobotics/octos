# Release Notes

## v0.2.1 — Tiered Permission Model

- Add `SafetyTier` enum (`Observe`, `SafeMotion`, `FullActuation`, `EmergencyOverride`) with `PartialOrd` ordering
- Add `WorkspaceBounds` for axis-aligned safe motion validation
- Add `RobotPermissionPolicy` with `authorize()` method for tier-based tool gating
- Add `PermissionDenied` error type with descriptive messages
- Extend `Tool` trait with `required_safety_tier()` default method (returns `Observe`)
- New module: `crates/octos-agent/src/permissions.rs`
- 6 unit tests covering authorization, denial, ordering, serialization, and defaults
- Fix pre-existing `profile_name` field missing in sandbox test structs
