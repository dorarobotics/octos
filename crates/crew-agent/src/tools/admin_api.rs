//! API-based admin tools for admin mode gateways.
//!
//! These tools call the `crew serve` REST API instead of having direct
//! `Arc<ProfileStore>` / `Arc<ProcessManager>` access. This allows the
//! admin bot to run as a regular gateway profile.

use std::sync::Arc;

use async_trait::async_trait;
use eyre::Result;
use serde::Deserialize;

use super::{Tool, ToolResult};

/// Shared context for all admin API tools.
pub struct AdminApiContext {
    pub http: reqwest::Client,
    pub serve_url: String,
    pub admin_token: String,
}

impl AdminApiContext {
    /// Make an authenticated GET request.
    async fn get(&self, path: &str) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.serve_url, path);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.admin_token)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eyre::bail!("API error {}: {}", status, body);
        }
        Ok(resp.json().await?)
    }

    /// Make an authenticated POST request.
    async fn post(&self, path: &str, body: Option<&serde_json::Value>) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.serve_url, path);
        let mut req = self.http.post(&url).bearer_auth(&self.admin_token);
        if let Some(b) = body {
            req = req.json(b);
        } else {
            req = req.header("content-type", "application/json");
        }
        let resp = req.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eyre::bail!("API error {}: {}", status, body);
        }
        Ok(resp.json().await?)
    }

    /// Make an authenticated PUT request.
    async fn put(&self, path: &str, body: &serde_json::Value) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.serve_url, path);
        let resp = self
            .http
            .put(&url)
            .bearer_auth(&self.admin_token)
            .json(body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eyre::bail!("API error {}: {}", status, body);
        }
        Ok(resp.json().await?)
    }
}

// ── admin_list_profiles ────────────────────────────────────────────────

pub struct ListProfilesTool {
    ctx: Arc<AdminApiContext>,
}

impl ListProfilesTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize, Default)]
struct ListProfilesInput {
    filter: Option<String>,
}

#[async_trait]
impl Tool for ListProfilesTool {
    fn name(&self) -> &str {
        "admin_list_profiles"
    }
    fn description(&self) -> &str {
        "List all profiles with their status (running/stopped, PID, uptime). Optional filter: running, stopped, enabled, disabled."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "enum": ["running", "stopped", "enabled", "disabled"],
                    "description": "Filter profiles by status"
                }
            }
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ListProfilesInput = serde_json::from_value(args.clone()).unwrap_or_default();

        match self.ctx.get("/api/admin/overview").await {
            Ok(overview) => {
                let profiles = overview
                    .get("profiles")
                    .and_then(|p| p.as_array())
                    .cloned()
                    .unwrap_or_default();

                let mut lines = Vec::new();
                for p in &profiles {
                    let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    let id = p.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let enabled = p.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
                    let status = p.get("status").unwrap_or(&serde_json::Value::Null);
                    let running = status
                        .get("running")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let pid = status
                        .get("pid")
                        .and_then(|v| v.as_u64())
                        .map(|p| format!("PID {p}"))
                        .unwrap_or_default();
                    let uptime = status
                        .get("uptime_secs")
                        .and_then(|v| v.as_i64())
                        .map(format_duration)
                        .unwrap_or_default();

                    // Apply filter
                    match input.filter.as_deref() {
                        Some("running") if !running => continue,
                        Some("stopped") if running => continue,
                        Some("enabled") if !enabled => continue,
                        Some("disabled") if enabled => continue,
                        _ => {}
                    }

                    let state = if running { "RUNNING" } else { "STOPPED" };
                    let en = if enabled { "enabled" } else { "disabled" };
                    let provider = p
                        .get("config")
                        .and_then(|c| c.get("provider"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("-");
                    let model = p
                        .get("config")
                        .and_then(|c| c.get("model"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("-");

                    lines.push(format!(
                        "- **{name}** ({id}) [{state}] {pid} {uptime} ({en}) provider={provider} model={model}"
                    ));
                }

                if lines.is_empty() {
                    Ok(ToolResult {
                        output: "No profiles match the filter.".into(),
                        success: true,
                        ..Default::default()
                    })
                } else {
                    Ok(ToolResult {
                        output: format!("{} profiles:\n{}", lines.len(), lines.join("\n")),
                        success: true,
                        ..Default::default()
                    })
                }
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to list profiles: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_profile_status ───────────────────────────────────────────────

pub struct ProfileStatusTool {
    ctx: Arc<AdminApiContext>,
}

impl ProfileStatusTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize)]
struct ProfileIdInput {
    profile_id: String,
}

#[async_trait]
impl Tool for ProfileStatusTool {
    fn name(&self) -> &str {
        "admin_profile_status"
    }
    fn description(&self) -> &str {
        "Get detailed status for a single profile: PID, uptime, channels, provider, model, enabled."
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
            .get(&format!("/api/admin/profiles/{}", input.profile_id))
            .await
        {
            Ok(profile) => Ok(ToolResult {
                output: serde_json::to_string_pretty(&profile)
                    .unwrap_or_else(|_| "{}".into()),
                success: true,
                ..Default::default()
            }),
            Err(e) => Ok(ToolResult {
                output: format!("Failed to get profile status: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_start_profile ────────────────────────────────────────────────

pub struct StartProfileTool {
    ctx: Arc<AdminApiContext>,
}

impl StartProfileTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for StartProfileTool {
    fn name(&self) -> &str {
        "admin_start_profile"
    }
    fn description(&self) -> &str {
        "Start a gateway for the given profile."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID to start" }
            },
            "required": ["profile_id"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ProfileIdInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        match self
            .ctx
            .post(
                &format!("/api/admin/profiles/{}/start", input.profile_id),
                None,
            )
            .await
        {
            Ok(resp) => {
                let msg = resp
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Started");
                Ok(ToolResult {
                    output: format!("Gateway '{}': {msg}", input.profile_id),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to start '{}': {e}", input.profile_id),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_stop_profile ─────────────────────────────────────────────────

pub struct StopProfileTool {
    ctx: Arc<AdminApiContext>,
}

impl StopProfileTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for StopProfileTool {
    fn name(&self) -> &str {
        "admin_stop_profile"
    }
    fn description(&self) -> &str {
        "Stop a running gateway for the given profile."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID to stop" }
            },
            "required": ["profile_id"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ProfileIdInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        match self
            .ctx
            .post(
                &format!("/api/admin/profiles/{}/stop", input.profile_id),
                None,
            )
            .await
        {
            Ok(resp) => {
                let msg = resp
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Stopped");
                Ok(ToolResult {
                    output: format!("Gateway '{}': {msg}", input.profile_id),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to stop '{}': {e}", input.profile_id),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_restart_profile ──────────────────────────────────────────────

pub struct RestartProfileTool {
    ctx: Arc<AdminApiContext>,
}

impl RestartProfileTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for RestartProfileTool {
    fn name(&self) -> &str {
        "admin_restart_profile"
    }
    fn description(&self) -> &str {
        "Restart a gateway (stop then start)."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID to restart" }
            },
            "required": ["profile_id"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ProfileIdInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        match self
            .ctx
            .post(
                &format!("/api/admin/profiles/{}/restart", input.profile_id),
                None,
            )
            .await
        {
            Ok(resp) => {
                let msg = resp
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Restarted");
                Ok(ToolResult {
                    output: format!("Gateway '{}': {msg}", input.profile_id),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to restart '{}': {e}", input.profile_id),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_enable_profile ───────────────────────────────────────────────

pub struct EnableProfileTool {
    ctx: Arc<AdminApiContext>,
}

impl EnableProfileTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize)]
struct EnableInput {
    profile_id: String,
    enabled: bool,
}

#[async_trait]
impl Tool for EnableProfileTool {
    fn name(&self) -> &str {
        "admin_enable_profile"
    }
    fn description(&self) -> &str {
        "Enable or disable auto-start for a profile."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID" },
                "enabled": { "type": "boolean", "description": "true to enable auto-start, false to disable" }
            },
            "required": ["profile_id", "enabled"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: EnableInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        let body = serde_json::json!({ "enabled": input.enabled });
        match self
            .ctx
            .put(
                &format!("/api/admin/profiles/{}", input.profile_id),
                &body,
            )
            .await
        {
            Ok(_) => {
                let state = if input.enabled { "enabled" } else { "disabled" };
                Ok(ToolResult {
                    output: format!("Profile '{}' auto-start {}.", input.profile_id, state),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to update '{}': {e}", input.profile_id),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_update_profile ──────────────────────────────────────────────

pub struct UpdateProfileTool {
    ctx: Arc<AdminApiContext>,
}

impl UpdateProfileTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize)]
struct UpdateProfileInput {
    profile_id: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    max_iterations: Option<u32>,
    #[serde(default)]
    max_history: Option<usize>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    /// Email sending settings (SMTP or Feishu).
    #[serde(default)]
    email: Option<serde_json::Value>,
    /// Environment variables to set (e.g. API keys, SMTP_PASSWORD).
    #[serde(default)]
    env_vars: Option<serde_json::Map<String, serde_json::Value>>,
    /// Whether to auto-restart the gateway after updating. Default true.
    #[serde(default = "default_true")]
    restart: bool,
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Tool for UpdateProfileTool {
    fn name(&self) -> &str {
        "admin_update_profile"
    }
    fn description(&self) -> &str {
        "Update a profile's configuration: model, provider, system prompt, etc. By default restarts the gateway to apply changes."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Profile ID to update" },
                "provider": { "type": "string", "description": "LLM provider name (e.g. moonshot, deepseek, openai, anthropic, gemini)" },
                "model": { "type": "string", "description": "Model name (e.g. kimi-2.5, deepseek-chat, gpt-4o)" },
                "base_url": { "type": "string", "description": "Custom API base URL" },
                "api_key_env": { "type": "string", "description": "Environment variable name for API key" },
                "system_prompt": { "type": "string", "description": "Custom system prompt for the gateway" },
                "max_iterations": { "type": "integer", "description": "Max tool-call iterations per message" },
                "max_history": { "type": "integer", "description": "Max conversation history messages" },
                "name": { "type": "string", "description": "Display name for the profile" },
                "enabled": { "type": "boolean", "description": "Enable/disable auto-start" },
                "email": {
                    "type": "object",
                    "description": "Email sending config. For SMTP: {provider:'smtp', smtp_host, smtp_port, username, password_env, from_address}. For Feishu: {provider:'feishu', feishu_app_id, feishu_app_secret_env, feishu_from_address}.",
                    "properties": {
                        "provider": { "type": "string", "enum": ["smtp", "feishu"], "description": "Email provider type" },
                        "smtp_host": { "type": "string", "description": "SMTP server hostname (e.g. smtp.gmail.com)" },
                        "smtp_port": { "type": "integer", "description": "SMTP port (465 for TLS, 587 for STARTTLS)" },
                        "username": { "type": "string", "description": "SMTP login username" },
                        "password_env": { "type": "string", "description": "Env var name holding SMTP password" },
                        "from_address": { "type": "string", "description": "Sender email address" }
                    }
                },
                "env_vars": {
                    "type": "object",
                    "description": "Environment variables to set (e.g. SMTP_PASSWORD, API keys). Keys are var names, values are secrets.",
                    "additionalProperties": { "type": "string" }
                },
                "restart": { "type": "boolean", "description": "Auto-restart gateway after update (default true)", "default": true }
            },
            "required": ["profile_id"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: UpdateProfileInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        // Auto-detect provider from model name when provider is not explicitly set
        let provider = input.provider.clone().or_else(|| {
            input
                .model
                .as_deref()
                .and_then(crew_llm::registry::detect_provider)
                .map(String::from)
        });

        // Build the update body with only specified fields
        let mut body = serde_json::Map::new();
        let mut config = serde_json::Map::new();
        let mut gateway = serde_json::Map::new();

        if let Some(ref name) = input.name {
            body.insert("name".into(), serde_json::json!(name));
        }
        if let Some(enabled) = input.enabled {
            body.insert("enabled".into(), serde_json::json!(enabled));
        }
        if let Some(ref provider) = provider {
            config.insert("provider".into(), serde_json::json!(provider));
        }
        if let Some(ref model) = input.model {
            config.insert("model".into(), serde_json::json!(model));
        }
        if let Some(ref base_url) = input.base_url {
            config.insert("base_url".into(), serde_json::json!(base_url));
        }
        if let Some(ref api_key_env) = input.api_key_env {
            config.insert("api_key_env".into(), serde_json::json!(api_key_env));
        }
        if let Some(ref sp) = input.system_prompt {
            gateway.insert("system_prompt".into(), serde_json::json!(sp));
        }
        if let Some(mi) = input.max_iterations {
            gateway.insert("max_iterations".into(), serde_json::json!(mi));
        }
        if let Some(mh) = input.max_history {
            gateway.insert("max_history".into(), serde_json::json!(mh));
        }
        if let Some(ref email) = input.email {
            config.insert("email".into(), email.clone());
        }
        if let Some(ref env_vars) = input.env_vars {
            config.insert("env_vars".into(), serde_json::Value::Object(env_vars.clone()));
        }
        if !gateway.is_empty() {
            config.insert("gateway".into(), serde_json::Value::Object(gateway));
        }
        if !config.is_empty() {
            body.insert("config".into(), serde_json::Value::Object(config));
        }

        if body.is_empty() {
            return Ok(ToolResult {
                output: "No fields to update. Specify at least one field (provider, model, name, etc.).".into(),
                success: false,
                ..Default::default()
            });
        }

        let body_value = serde_json::Value::Object(body);
        match self
            .ctx
            .put(
                &format!("/api/admin/profiles/{}", input.profile_id),
                &body_value,
            )
            .await
        {
            Ok(_) => {
                let mut changes: Vec<String> = Vec::new();
                if let Some(ref p) = provider {
                    changes.push(format!("provider={p}"));
                }
                if let Some(ref m) = input.model {
                    changes.push(format!("model={m}"));
                }
                if let Some(ref n) = input.name {
                    changes.push(format!("name={n}"));
                }
                if input.enabled.is_some() {
                    changes.push(format!("enabled={}", input.enabled.unwrap()));
                }
                if input.system_prompt.is_some() {
                    changes.push("system_prompt=<updated>".into());
                }
                if let Some(mi) = input.max_iterations {
                    changes.push(format!("max_iterations={mi}"));
                }
                if let Some(mh) = input.max_history {
                    changes.push(format!("max_history={mh}"));
                }
                if input.base_url.is_some() {
                    changes.push("base_url=<updated>".into());
                }
                if input.api_key_env.is_some() {
                    changes.push("api_key_env=<updated>".into());
                }
                if input.email.is_some() {
                    changes.push("email=<configured>".into());
                }
                if input.env_vars.is_some() {
                    changes.push("env_vars=<updated>".into());
                }

                let mut msg = format!(
                    "Profile '{}' updated: {}",
                    input.profile_id,
                    changes.join(", ")
                );

                // Auto-restart if requested
                if input.restart {
                    match self
                        .ctx
                        .post(
                            &format!("/api/admin/profiles/{}/restart", input.profile_id),
                            None,
                        )
                        .await
                    {
                        Ok(_) => msg.push_str(". Gateway restarted."),
                        Err(e) => msg.push_str(&format!(". Restart failed: {e}")),
                    }
                }

                Ok(ToolResult {
                    output: msg,
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to update '{}': {e}", input.profile_id),
                success: false,
                ..Default::default()
            }),
        }
    }
}

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
                            // Parse SSE lines
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

                // Check for down profiles
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
                let endpoint = if input.action == "enable" {
                    "/api/admin/monitor/watchdog"
                } else {
                    "/api/admin/monitor/watchdog"
                };
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

// ── Helpers ────────────────────────────────────────────────────────────

fn format_duration(secs: i64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

// ── System Metrics Tool ──────────────────────────────────────────────

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

                // CPU
                if let Some(cpu) = data.get("cpu") {
                    let usage = cpu.get("usage_percent").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let cores = cpu.get("core_count").and_then(|v| v.as_u64()).unwrap_or(0);
                    let brand = cpu.get("brand").and_then(|v| v.as_str()).unwrap_or("unknown");
                    out.push_str(&format!("CPU: {brand}\n"));
                    out.push_str(&format!("  Usage: {usage:.1}% ({cores} cores)\n"));
                }

                // Memory
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

                // Swap
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

                // Disks
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

                // Platform
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

// ── admin_list_sub_accounts ───────────────────────────────────────────

pub struct ListSubAccountsTool {
    ctx: Arc<AdminApiContext>,
}

impl ListSubAccountsTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Tool for ListSubAccountsTool {
    fn name(&self) -> &str {
        "admin_list_sub_accounts"
    }
    fn description(&self) -> &str {
        "List all sub-accounts for a given parent profile. Returns each sub-account's ID, name, status, channels, and config."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Parent profile ID" }
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
                "/api/admin/profiles/{}/accounts",
                input.profile_id
            ))
            .await
        {
            Ok(subs) => {
                let items = subs.as_array().cloned().unwrap_or_default();
                if items.is_empty() {
                    return Ok(ToolResult {
                        output: format!(
                            "No sub-accounts found for profile '{}'.",
                            input.profile_id
                        ),
                        success: true,
                        ..Default::default()
                    });
                }

                let mut lines = Vec::new();
                for item in &items {
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    let enabled = item
                        .get("enabled")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let status = item.get("status").unwrap_or(&serde_json::Value::Null);
                    let running = status
                        .get("running")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let pid = status
                        .get("pid")
                        .and_then(|v| v.as_u64())
                        .map(|p| format!("PID {p}"))
                        .unwrap_or_default();
                    let uptime = status
                        .get("uptime_secs")
                        .and_then(|v| v.as_i64())
                        .map(format_duration)
                        .unwrap_or_default();

                    let channels = item
                        .get("channels")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|c| {
                                    c.as_object()
                                        .and_then(|o| o.keys().next().map(|k| k.as_str()))
                                })
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();

                    let state = if running { "RUNNING" } else { "STOPPED" };
                    let en = if enabled { "enabled" } else { "disabled" };
                    lines.push(format!(
                        "- **{name}** ({id}) [{state}] {pid} {uptime} ({en}) channels=[{channels}]"
                    ));
                }

                Ok(ToolResult {
                    output: format!(
                        "{} sub-accounts for '{}':\n{}",
                        items.len(),
                        input.profile_id,
                        lines.join("\n")
                    ),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to list sub-accounts: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

// ── admin_create_sub_account ──────────────────────────────────────────

pub struct CreateSubAccountTool {
    ctx: Arc<AdminApiContext>,
}

impl CreateSubAccountTool {
    pub fn new(ctx: Arc<AdminApiContext>) -> Self {
        Self { ctx }
    }
}

#[derive(Deserialize)]
struct CreateSubAccountInput {
    profile_id: String,
    name: String,
    #[serde(default)]
    channels: Vec<serde_json::Value>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    env_vars: std::collections::HashMap<String, String>,
}

#[async_trait]
impl Tool for CreateSubAccountTool {
    fn name(&self) -> &str {
        "admin_create_sub_account"
    }
    fn description(&self) -> &str {
        "Create a sub-account under a parent profile. The sub-account inherits LLM provider config from the parent but has its own channels, system prompt, and data directories."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile_id": { "type": "string", "description": "Parent profile ID" },
                "name": { "type": "string", "description": "Name for the sub-account (e.g. 'work bot', 'support')" },
                "channels": {
                    "type": "array",
                    "description": "Channel configurations (e.g. [{\"Telegram\": {\"token_env\": \"WORK_TG_TOKEN\"}}])",
                    "items": { "type": "object" }
                },
                "system_prompt": { "type": "string", "description": "Custom system prompt for this sub-account" },
                "env_vars": {
                    "type": "object",
                    "description": "Environment variables specific to this sub-account",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["profile_id", "name"]
        })
    }
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: CreateSubAccountInput =
            serde_json::from_value(args.clone()).map_err(|e| eyre::eyre!("invalid input: {e}"))?;

        let mut body = serde_json::json!({
            "name": input.name,
            "channels": input.channels,
            "env_vars": input.env_vars,
        });

        if let Some(prompt) = &input.system_prompt {
            body["gateway"] = serde_json::json!({
                "system_prompt": prompt,
            });
        }

        match self
            .ctx
            .post(
                &format!(
                    "/api/admin/profiles/{}/accounts",
                    input.profile_id
                ),
                Some(&body),
            )
            .await
        {
            Ok(resp) => {
                let sub_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let name = resp.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                Ok(ToolResult {
                    output: format!(
                        "Created sub-account '{name}' ({sub_id}) under parent '{}'.",
                        input.profile_id
                    ),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to create sub-account: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

/// Register all admin API tools into a ToolRegistry.
pub fn register_admin_api_tools(registry: &mut super::ToolRegistry, ctx: Arc<AdminApiContext>) {
    registry.register(ListProfilesTool::new(ctx.clone()));
    registry.register(ProfileStatusTool::new(ctx.clone()));
    registry.register(StartProfileTool::new(ctx.clone()));
    registry.register(StopProfileTool::new(ctx.clone()));
    registry.register(RestartProfileTool::new(ctx.clone()));
    registry.register(EnableProfileTool::new(ctx.clone()));
    registry.register(UpdateProfileTool::new(ctx.clone()));
    registry.register(ViewLogsTool::new(ctx.clone()));
    registry.register(SystemHealthTool::new(ctx.clone()));
    registry.register(SystemMetricsTool::new(ctx.clone()));
    registry.register(ProviderMetricsTool::new(ctx.clone()));
    registry.register(ListSubAccountsTool::new(ctx.clone()));
    registry.register(CreateSubAccountTool::new(ctx.clone()));
    registry.register(ManageWatchdogTool::new(ctx));
}
