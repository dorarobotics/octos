//! System monitoring tools: health, metrics, logs, watchdog, provider metrics.

use std::sync::Arc;

use async_trait::async_trait;
use eyre::Result;
use serde::Deserialize;

use super::{AdminApiContext, ProfileIdInput, Tool, ToolResult, format_duration};

// ── admin_view_logs ────────────────────────────────────────────────────

pub struct ViewLogsTool {
    ctx: Arc<AdminApiContext>,
}

impl ViewLogsTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize)]
struct ViewLogsInput {
    profile_id: String,
    #[serde(default = "default_log_lines")]
    lines: usize,
}
fn default_log_lines() -> usize {
    30
}

#[async_trait]
impl Tool for ViewLogsTool {
    fn name(&self) -> &str {
        "admin_view_logs"
    }
    fn description(&self) -> &str {
        "View recent log output from a running gateway. Streams SSE log events for up to 3 seconds and returns up to N lines (default 30, max 100)."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID" },
                "lines": { "type": "integer", "description": "Number of log lines to collect (default 30, max 100)" }
            },
            "required": ["profile_id"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ViewLogsInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;
        let max_lines = input.lines.min(100);

        // Connect to SSE log stream
        let url = format!(
            "{}/api/admin/profiles/{}/logs?token={}",
            self.ctx.serve_url, input.profile_id, self.ctx.admin_token
        );

        let resp = match self.ctx.http.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    output: format!("Failed to connect to log stream: {e}"),
                    success: false,
                    ..Default::default()
                });
            }
        };

        if !resp.status().is_success() {
            return Ok(ToolResult {
                output: format!(
                    "Gateway '{}' is not running or logs unavailable.",
                    input.profile_id
                ),
                success: false,
                ..Default::default()
            });
        }

        // Read SSE events for up to 3 seconds
        let mut lines_collected = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);

        let mut stream = resp.bytes_stream();
        use futures::StreamExt;
        let mut buffer = String::new();

        loop {
            if lines_collected.len() >= max_lines {
                break;
            }
            tokio::select! {
                chunk = stream.next() => {
                    match chunk {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                            while let Some(pos) = buffer.find('\n') {
                                let line = buffer[..pos].trim().to_string();
                                buffer = buffer[pos + 1..].to_string();
                                if let Some(data) = line.strip_prefix("data:") {
                                    let data = data.trim();
                                    if !data.is_empty() {
                                        lines_collected.push(data.to_string());
                                        if lines_collected.len() >= max_lines {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(_)) | None => break,
                    }
                }
                _ = tokio::time::sleep_until(deadline) => break,
            }
        }

        if lines_collected.is_empty() {
            Ok(ToolResult {
                output: format!(
                    "No log output from '{}' in the last 3 seconds.",
                    input.profile_id
                ),
                success: true,
                ..Default::default()
            })
        } else {
            Ok(ToolResult {
                output: format!(
                    "{} log lines from '{}':\n{}",
                    lines_collected.len(),
                    input.profile_id,
                    lines_collected.join("\n")
                ),
                success: true,
                ..Default::default()
            })
        }
    }
}

// ── admin_system_health ────────────────────────────────────────────────

pub struct SystemHealthTool {
    ctx: Arc<AdminApiContext>,
}

impl SystemHealthTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for SystemHealthTool {
    fn name(&self) -> &str {
        "admin_system_health"
    }
    fn description(&self) -> &str {
        "Get system-wide health: total profiles, running/stopped counts, server uptime."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _args: &serde_json::Value) -> Result<ToolResult> {
        match self.ctx.get("/api/admin/overview").await {
            Ok(overview) => {
                let total = overview
                    .get("total_profiles")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let running = overview
                    .get("running")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let stopped = overview
                    .get("stopped")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let mut out = String::from("System Health:\n");
                out.push_str(&format!("  Profiles: {total} total\n"));
                out.push_str(&format!("  Running: {running}, Stopped: {stopped}\n"));

                if let Some(profiles) = overview.get("profiles").and_then(|p| p.as_array()) {
                    let down: Vec<_> = profiles
                        .iter()
                        .filter(|p| {
                            let enabled =
                                p.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
                            let running = p
                                .get("status")
                                .and_then(|s| s.get("running"))
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            enabled && !running
                        })
                        .collect();
                    if !down.is_empty() {
                        out.push_str("\n  WARNING: Some enabled profiles are not running!\n");
                        for p in &down {
                            let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                            let id = p.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                            out.push_str(&format!("    - {name} ({id})\n"));
                        }
                    }
                }

                Ok(ToolResult {
                    output: out,
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to get system health: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_provider_metrics ─────────────────────────────────────────────

pub struct ProviderMetricsTool {
    ctx: Arc<AdminApiContext>,
}

impl ProviderMetricsTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for ProviderMetricsTool {
    fn name(&self) -> &str {
        "admin_provider_metrics"
    }
    fn description(&self) -> &str {
        "Read provider QoS metrics (latency, error rate, token usage) for a profile."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID" }
            },
            "required": ["profile_id"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ProfileIdInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        match self
            .ctx
            .get(&format!(
                "/api/admin/profiles/{}/metrics",
                input.profile_id
            ))
            .await
        {
            Ok(metrics) => Ok(ToolResult {
                output: serde_json::to_string_pretty(&metrics).unwrap_or_else(|_| "{}".into()),
                success: true,
                ..Default::default()
            }),
            Err(e) => Ok(ToolResult {
                output: format!("No metrics available for '{}': {e}", input.profile_id),
                success: true,
                ..Default::default()
            }),
        }
    }
}

// ── admin_manage_watchdog ──────────────────────────────────────────────

pub struct ManageWatchdogTool {
    ctx: Arc<AdminApiContext>,
}

impl ManageWatchdogTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize)]
struct WatchdogInput {
    action: String,
}

#[async_trait]
impl Tool for ManageWatchdogTool {
    fn name(&self) -> &str {
        "admin_manage_watchdog"
    }
    fn description(&self) -> &str {
        "Check or toggle watchdog auto-restart and proactive alerts. Actions: status, enable, disable."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "enable", "disable"],
                    "description": "Action to perform"
                }
            },
            "required": ["action"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: WatchdogInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        let path = match input.action.as_str() {
            "status" => "/api/admin/monitor/status",
            "enable" | "disable" => {
                let endpoint = "/api/admin/monitor/watchdog";
                let body = serde_json::json!({ "enabled": input.action == "enable" });
                match self.ctx.post(endpoint, Some(&body)).await {
                    Ok(resp) => {
                        let msg = resp
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Done");
                        return Ok(ToolResult {
                            output: msg.to_string(),
                            success: true,
                            ..Default::default()
                        });
                    }
                    Err(e) => {
                        return Ok(ToolResult {
                            output: format!("Failed: {e}"),
                            success: false,
                            ..Default::default()
                        });
                    }
                }
            }
            other => {
                return Ok(ToolResult {
                    output: format!("Unknown action '{other}'. Use: status, enable, disable."),
                    success: false,
                    ..Default::default()
                });
            }
        };

        match self.ctx.get(path).await {
            Ok(status) => Ok(ToolResult {
                output: serde_json::to_string_pretty(&status).unwrap_or_else(|_| "{}".into()),
                success: true,
                ..Default::default()
            }),
            Err(e) => Ok(ToolResult {
                output: format!("Failed: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_system_metrics ──────────────────────────────────────────────

pub struct SystemMetricsTool {
    ctx: Arc<AdminApiContext>,
}

impl SystemMetricsTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for SystemMetricsTool {
    fn name(&self) -> &str {
        "admin_system_metrics"
    }
    fn description(&self) -> &str {
        "Get system resource metrics: CPU usage, memory, swap, disk storage, and platform info."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _args: &serde_json::Value) -> Result<ToolResult> {
        match self.ctx.get("/api/admin/system/metrics").await {
            Ok(data) => {
                let mut out = String::new();

                if let Some(cpu) = data.get("cpu") {
                    let usage = cpu.get("usage_percent").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let cores = cpu.get("core_count").and_then(|v| v.as_u64()).unwrap_or(0);
                    let brand = cpu.get("brand").and_then(|v| v.as_str()).unwrap_or("unknown");
                    out.push_str(&format!("CPU: {brand}\n"));
                    out.push_str(&format!("  Usage: {usage:.1}% ({cores} cores)\n"));
                }

                if let Some(mem) = data.get("memory") {
                    let total = mem.get("total_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    let used = mem.get("used_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    let avail = mem.get("available_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    let pct = if total > 0 { (used as f64 / total as f64) * 100.0 } else { 0.0 };
                    out.push_str(&format!(
                        "\nMemory: {:.1} GB used / {:.1} GB total ({pct:.0}%)\n",
                        used as f64 / 1_073_741_824.0,
                        total as f64 / 1_073_741_824.0,
                    ));
                    out.push_str(&format!(
                        "  Available: {:.1} GB\n",
                        avail as f64 / 1_073_741_824.0,
                    ));
                }

                if let Some(swap) = data.get("swap") {
                    let total = swap.get("total_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    let used = swap.get("used_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    if total > 0 {
                        out.push_str(&format!(
                            "\nSwap: {:.1} GB used / {:.1} GB total\n",
                            used as f64 / 1_073_741_824.0,
                            total as f64 / 1_073_741_824.0,
                        ));
                    }
                }

                if let Some(disks) = data.get("disks").and_then(|v| v.as_array()) {
                    out.push_str("\nStorage:\n");
                    for d in disks {
                        let mount = d.get("mount_point").and_then(|v| v.as_str()).unwrap_or("?");
                        let total = d.get("total_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                        let used = d.get("used_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                        let avail = d.get("available_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                        let pct = if total > 0 { (used as f64 / total as f64) * 100.0 } else { 0.0 };
                        out.push_str(&format!(
                            "  {mount}: {:.1} GB used / {:.1} GB total ({pct:.0}%), {:.1} GB free\n",
                            used as f64 / 1_073_741_824.0,
                            total as f64 / 1_073_741_824.0,
                            avail as f64 / 1_073_741_824.0,
                        ));
                    }
                }

                if let Some(plat) = data.get("platform") {
                    let host = plat.get("hostname").and_then(|v| v.as_str()).unwrap_or("?");
                    let os = plat.get("os").and_then(|v| v.as_str()).unwrap_or("?");
                    let ver = plat.get("os_version").and_then(|v| v.as_str()).unwrap_or("?");
                    let uptime = plat.get("uptime_secs").and_then(|v| v.as_i64()).unwrap_or(0);
                    out.push_str(&format!("\nPlatform: {os} {ver} ({host})\n"));
                    out.push_str(&format!("  Uptime: {}\n", format_duration(uptime)));
                }

                Ok(ToolResult {
                    output: out,
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to get system metrics: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}
