//! Plugin tool: wraps a plugin executable as a Tool.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::tools::{Tool, ToolResult};

use super::manifest::PluginToolDef;

/// A tool backed by a plugin executable.
///
/// Protocol: write JSON args to stdin, read JSON result from stdout.
/// Expected output: `{ "output": "...", "success": true/false }`
pub struct PluginTool {
    plugin_name: String,
    tool_def: PluginToolDef,
    executable: PathBuf,
    /// Environment variables to strip from the plugin's environment.
    blocked_env: Vec<String>,
    /// Execution timeout.
    timeout: Duration,
}

impl PluginTool {
    /// Default timeout for plugin execution.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

    pub fn new(plugin_name: String, tool_def: PluginToolDef, executable: PathBuf) -> Self {
        Self {
            plugin_name,
            tool_def,
            executable,
            blocked_env: vec![],
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// Set environment variables to block from plugin execution.
    pub fn with_blocked_env(mut self, blocked: Vec<String>) -> Self {
        self.blocked_env = blocked;
        self
    }

    /// Set custom execution timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl Tool for PluginTool {
    fn name(&self) -> &str {
        &self.tool_def.name
    }

    fn description(&self) -> &str {
        &self.tool_def.description
    }

    fn input_schema(&self) -> serde_json::Value {
        self.tool_def.input_schema.clone()
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        tracing::info!(
            plugin = %self.plugin_name,
            tool = %self.tool_def.name,
            executable = %self.executable.display(),
            timeout_secs = self.timeout.as_secs(),
            args_size = args.to_string().len(),
            "spawning plugin process"
        );

        let mut cmd = Command::new(&self.executable);
        cmd.arg(&self.tool_def.name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Remove blocked environment variables
        for var in &self.blocked_env {
            cmd.env_remove(var);
        }

        let mut child = cmd.spawn().wrap_err_with(|| {
            format!(
                "failed to spawn plugin '{}' executable: {}",
                self.plugin_name,
                self.executable.display()
            )
        })?;

        let child_pid = child.id().unwrap_or(0);
        tracing::info!(
            plugin = %self.plugin_name,
            tool = %self.tool_def.name,
            pid = child_pid,
            "plugin process spawned"
        );

        // Write args to stdin
        if let Some(mut stdin) = child.stdin.take() {
            let data = serde_json::to_vec(args)?;
            stdin.write_all(&data).await?;
            // Drop stdin to signal EOF
        }

        // wait_with_output() takes ownership of child, so save the PID
        // for killing on timeout.
        let result = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Err(eyre::eyre!(
                    "plugin '{}' tool '{}' execution failed: {e}",
                    self.plugin_name,
                    self.tool_def.name
                ));
            }
            Err(_) => {
                // Kill the child process by PID to prevent orphaned processes.
                // child was consumed by wait_with_output(), so kill via PID.
                #[cfg(unix)]
                if child_pid > 0 {
                    // Kill the process group (catches child processes like Chrome)
                    let _ = std::process::Command::new("kill")
                        .args(["-9", &format!("-{child_pid}")])
                        .status();
                    // Also kill the process directly as fallback
                    let _ = std::process::Command::new("kill")
                        .args(["-9", &child_pid.to_string()])
                        .status();
                }
                return Err(eyre::eyre!(
                    "plugin '{}' tool '{}' timed out after {}s",
                    self.plugin_name,
                    self.tool_def.name,
                    self.timeout.as_secs()
                ));
            }
        };

        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);

        tracing::info!(
            plugin = %self.plugin_name,
            tool = %self.tool_def.name,
            pid = child_pid,
            exit_code = result.status.code().unwrap_or(-1),
            stdout_len = stdout.len(),
            stderr_len = stderr.len(),
            "plugin process completed"
        );

        // Try to parse structured output
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&stdout) {
            let output = parsed
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or(&stdout)
                .to_string();
            let success = parsed
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(result.status.success());
            return Ok(ToolResult {
                output,
                success,
                ..Default::default()
            });
        }

        // Fallback: raw stdout + stderr
        let mut output = stdout.to_string();
        if !stderr.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&stderr);
        }

        Ok(ToolResult {
            output,
            success: result.status.success(),
            ..Default::default()
        })
    }
}
