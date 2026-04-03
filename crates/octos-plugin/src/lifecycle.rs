//! Hardware lifecycle executor for plugin manifests.
//!
//! Runs ordered lifecycle phases (preflight, init, ready_check, shutdown,
//! emergency_shutdown) with per-step timeout and retry support.

use std::time::Duration;

use eyre::{Result, bail};
use serde::{Deserialize, Serialize};

/// A single step in a hardware lifecycle phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStep {
    /// Human-readable label for logging.
    pub label: String,
    /// Shell command to execute.
    pub command: String,
    /// Timeout for this step in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Number of retry attempts on failure.
    #[serde(default)]
    pub retries: u32,
    /// If true, failure of this step aborts the entire phase.
    #[serde(default = "default_true")]
    pub critical: bool,
}

fn default_timeout_secs() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

/// Hardware lifecycle declaration for a plugin.
///
/// Each phase is a list of steps executed in order. Phases are:
/// - `preflight`: Checks before initialization (sensors connected, firmware OK)
/// - `init`: Bring hardware to operational state
/// - `ready_check`: Verify hardware is ready for operation
/// - `shutdown`: Graceful shutdown sequence
/// - `emergency_shutdown`: Fast shutdown (minimal steps, short timeouts)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareLifecycle {
    #[serde(default)]
    pub preflight: Vec<LifecycleStep>,
    #[serde(default)]
    pub init: Vec<LifecycleStep>,
    #[serde(default)]
    pub ready_check: Vec<LifecycleStep>,
    #[serde(default)]
    pub shutdown: Vec<LifecycleStep>,
    #[serde(default)]
    pub emergency_shutdown: Vec<LifecycleStep>,
}

/// Result of executing a lifecycle phase.
#[derive(Debug)]
pub struct PhaseResult {
    pub phase: String,
    pub steps_completed: usize,
    pub steps_total: usize,
    pub success: bool,
    pub error: Option<String>,
}

/// Executes lifecycle phases with timeout and retry logic.
pub struct LifecycleExecutor;

impl LifecycleExecutor {
    /// Run a lifecycle phase (list of steps) to completion.
    ///
    /// Executes steps in order. On failure:
    /// - Retries up to `step.retries` times
    /// - If `step.critical` is true, aborts the phase on final failure
    /// - Non-critical steps log a warning and continue
    pub async fn run_phase(phase_name: &str, steps: &[LifecycleStep]) -> PhaseResult {
        let total = steps.len();
        for (i, step) in steps.iter().enumerate() {
            let mut last_error = None;

            for attempt in 0..=step.retries {
                if attempt > 0 {
                    tracing::warn!(
                        phase = phase_name,
                        step = step.label,
                        attempt = attempt + 1,
                        max = step.retries + 1,
                        "retrying lifecycle step"
                    );
                }

                match Self::run_step(step).await {
                    Ok(()) => {
                        last_error = None;
                        break;
                    }
                    Err(e) => {
                        last_error = Some(e.to_string());
                    }
                }
            }

            if let Some(err) = last_error {
                if step.critical {
                    tracing::error!(
                        phase = phase_name,
                        step = step.label,
                        error = %err,
                        "critical lifecycle step failed, aborting phase"
                    );
                    return PhaseResult {
                        phase: phase_name.to_string(),
                        steps_completed: i,
                        steps_total: total,
                        success: false,
                        error: Some(format!("{}: {}", step.label, err)),
                    };
                }
                tracing::warn!(
                    phase = phase_name,
                    step = step.label,
                    error = %err,
                    "non-critical lifecycle step failed, continuing"
                );
            }
        }

        PhaseResult {
            phase: phase_name.to_string(),
            steps_completed: total,
            steps_total: total,
            success: true,
            error: None,
        }
    }

    /// Run a single lifecycle step with timeout.
    async fn run_step(step: &LifecycleStep) -> Result<()> {
        let timeout = Duration::from_secs(step.timeout_secs);

        let result = tokio::time::timeout(timeout, async {
            let output = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(&step.command)
                .output()
                .await?;

            if output.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                bail!(
                    "command exited with {}: {}",
                    output.status,
                    stderr.trim()
                )
            }
        })
        .await;

        match result {
            Ok(inner) => inner,
            Err(_) => bail!("step '{}' timed out after {}s", step.label, step.timeout_secs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_lifecycle_with_all_phases() {
        let json = r#"{
            "preflight": [
                {"label": "check_sensor", "command": "echo ok", "timeout_secs": 5, "retries": 1, "critical": true}
            ],
            "init": [
                {"label": "power_on", "command": "echo on"}
            ],
            "shutdown": [],
            "emergency_shutdown": [
                {"label": "e_stop", "command": "echo stop", "timeout_secs": 2}
            ]
        }"#;
        let lifecycle: HardwareLifecycle = serde_json::from_str(json).unwrap();
        assert_eq!(lifecycle.preflight.len(), 1);
        assert_eq!(lifecycle.init.len(), 1);
        assert!(lifecycle.shutdown.is_empty());
        assert_eq!(lifecycle.emergency_shutdown.len(), 1);
        assert_eq!(lifecycle.emergency_shutdown[0].timeout_secs, 2);
    }

    #[test]
    fn should_parse_lifecycle_without_optional_phases() {
        let json = "{}";
        let lifecycle: HardwareLifecycle = serde_json::from_str(json).unwrap();
        assert!(lifecycle.preflight.is_empty());
        assert!(lifecycle.init.is_empty());
    }

    #[tokio::test]
    async fn should_run_successful_phase() {
        let steps = vec![
            LifecycleStep {
                label: "step1".to_string(),
                command: "echo hello".to_string(),
                timeout_secs: 5,
                retries: 0,
                critical: true,
            },
            LifecycleStep {
                label: "step2".to_string(),
                command: "echo world".to_string(),
                timeout_secs: 5,
                retries: 0,
                critical: true,
            },
        ];
        let result = LifecycleExecutor::run_phase("init", &steps).await;
        assert!(result.success);
        assert_eq!(result.steps_completed, 2);
    }

    #[tokio::test]
    async fn should_abort_on_critical_failure() {
        let steps = vec![
            LifecycleStep {
                label: "will_fail".to_string(),
                command: "exit 1".to_string(),
                timeout_secs: 5,
                retries: 0,
                critical: true,
            },
            LifecycleStep {
                label: "never_reached".to_string(),
                command: "echo ok".to_string(),
                timeout_secs: 5,
                retries: 0,
                critical: true,
            },
        ];
        let result = LifecycleExecutor::run_phase("init", &steps).await;
        assert!(!result.success);
        assert_eq!(result.steps_completed, 0);
    }
}
