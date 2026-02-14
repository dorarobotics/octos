//! Shell tool for executing commands.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use serde::Deserialize;
use tokio::time::timeout;

use super::{Tool, ToolResult};
use crate::policy::{CommandPolicy, Decision, SafePolicy};
use crate::sandbox::{NoSandbox, Sandbox};

/// Tool for executing shell commands.
pub struct ShellTool {
    /// Timeout for command execution.
    timeout: Duration,
    /// Working directory for commands.
    cwd: std::path::PathBuf,
    /// Policy for command approval.
    policy: Arc<dyn CommandPolicy>,
    /// Sandbox for command isolation.
    sandbox: Box<dyn Sandbox>,
}

impl ShellTool {
    /// Create a new shell tool with safe defaults.
    pub fn new(cwd: impl Into<std::path::PathBuf>) -> Self {
        Self {
            timeout: Duration::from_secs(120),
            cwd: cwd.into(),
            policy: Arc::new(SafePolicy::default()),
            sandbox: Box::new(NoSandbox),
        }
    }

    /// Set the timeout for commands.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set a custom command policy.
    pub fn with_policy(mut self, policy: Arc<dyn CommandPolicy>) -> Self {
        self.policy = policy;
        self
    }

    /// Set a sandbox for command isolation.
    pub fn with_sandbox(mut self, sandbox: Box<dyn Sandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[derive(Debug, Deserialize)]
struct ShellInput {
    command: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return the output. Use this to run tests, build code, or interact with the filesystem."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ShellInput =
            serde_json::from_value(args.clone()).wrap_err("invalid shell tool input")?;

        // Check policy first
        let decision = self.policy.check(&input.command, &self.cwd);
        match decision {
            Decision::Deny => {
                tracing::warn!(command = %input.command, "command denied by policy");
                return Ok(ToolResult {
                    output: format!(
                        "Command denied by security policy: {}\n\nThis command was blocked because it matches a dangerous pattern.",
                        input.command
                    ),
                    success: false,
                    ..Default::default()
                });
            }
            Decision::Ask => {
                tracing::info!(command = %input.command, "command requires approval");
                // In a real implementation, this would prompt the user
                // For now, we log and allow (coordinator can intercept)
            }
            Decision::Allow => {}
        }

        let timeout_duration = input
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(self.timeout);

        // Execute command (through sandbox)
        let mut cmd = self.sandbox.wrap_command(&input.command, &self.cwd);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let result = timeout(timeout_duration, cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                let mut result_text = String::new();

                if !stdout.is_empty() {
                    result_text.push_str(&stdout);
                }

                if !stderr.is_empty() {
                    if !result_text.is_empty() {
                        result_text.push_str("\n--- stderr ---\n");
                    }
                    result_text.push_str(&stderr);
                }

                if result_text.is_empty() {
                    result_text = "(no output)".to_string();
                }

                // Truncate if too long (reserve space for exit code suffix)
                let exit_suffix = format!("\n\nExit code: {exit_code}");
                const MAX_OUTPUT: usize = 50000;
                crew_core::truncate_utf8(
                    &mut result_text,
                    MAX_OUTPUT - exit_suffix.len(),
                    "\n... (output truncated)",
                );

                result_text.push_str(&exit_suffix);

                Ok(ToolResult {
                    output: result_text,
                    success: output.status.success(),
                    ..Default::default()
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                output: format!("Failed to execute command: {e}"),
                success: false,
                ..Default::default()
            }),
            Err(_) => Ok(ToolResult {
                output: format!(
                    "Command timed out after {} seconds",
                    timeout_duration.as_secs()
                ),
                success: false,
                ..Default::default()
            }),
        }
    }
}
