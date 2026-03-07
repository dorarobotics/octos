//! HTTP server types for pipeline management.
//!
//! Defines request/response types and a trait for serving pipeline
//! operations over HTTP. The actual server implementation (e.g. using axum)
//! is left to the CLI crate to avoid adding HTTP framework dependencies here.
//!
//! TODO: Implement the server trait in crew-cli with axum.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::graph::OutcomeStatus;

/// Request to submit a pipeline for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitRequest {
    /// DOT source for the pipeline.
    pub dot_source: String,
    /// Input text for the pipeline.
    pub input: String,
    /// Optional pipeline ID (auto-generated if not provided).
    pub pipeline_id: Option<String>,
    /// Additional variables for template expansion.
    #[serde(default)]
    pub variables: HashMap<String, String>,
}

/// Response after submitting a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitResponse {
    /// Assigned pipeline run ID.
    pub run_id: String,
    /// Status of the submission.
    pub status: RunStatus,
}

/// Status of a pipeline run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Pipeline is queued for execution.
    Queued,
    /// Pipeline is currently running.
    Running,
    /// Pipeline completed successfully.
    Completed,
    /// Pipeline failed.
    Failed,
    /// Pipeline was cancelled.
    Cancelled,
}

/// Request to cancel a running pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    /// Pipeline run ID to cancel.
    pub run_id: String,
}

/// Status query response for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStatusResponse {
    /// Pipeline run ID.
    pub run_id: String,
    /// Current status.
    pub status: RunStatus,
    /// Current node being executed (if running).
    pub current_node: Option<String>,
    /// Per-node outcomes (for completed nodes).
    #[serde(default)]
    pub node_outcomes: HashMap<String, NodeStatusResponse>,
    /// Final output (if completed).
    pub output: Option<String>,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// Status of a single node in a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatusResponse {
    pub status: OutcomeStatus,
    pub output: String,
}

/// Trait for pipeline server implementations.
///
/// This trait defines the API contract. Implementations handle
/// HTTP routing, serialization, and lifecycle management.
#[async_trait::async_trait]
pub trait PipelineServer: Send + Sync {
    /// Submit a pipeline for execution.
    async fn submit(&self, request: SubmitRequest) -> eyre::Result<SubmitResponse>;

    /// Get the status of a pipeline run.
    async fn status(&self, run_id: &str) -> eyre::Result<RunStatusResponse>;

    /// Cancel a running pipeline.
    async fn cancel(&self, request: CancelRequest) -> eyre::Result<()>;

    /// List all pipeline runs (optionally filtered by status).
    async fn list_runs(&self, status_filter: Option<RunStatus>) -> eyre::Result<Vec<RunStatusResponse>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_serialize_submit_request() {
        let req = SubmitRequest {
            dot_source: "digraph { a -> b }".into(),
            input: "test input".into(),
            pipeline_id: Some("my-pipe".into()),
            variables: HashMap::new(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["dot_source"], "digraph { a -> b }");
        assert_eq!(json["pipeline_id"], "my-pipe");
    }

    #[test]
    fn should_serialize_run_status() {
        let json = serde_json::to_value(&RunStatus::Running).unwrap();
        assert_eq!(json, "running");

        let json = serde_json::to_value(&RunStatus::Completed).unwrap();
        assert_eq!(json, "completed");
    }

    #[test]
    fn should_deserialize_submit_request() {
        let json = r#"{"dot_source":"digraph {}","input":"hi"}"#;
        let req: SubmitRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.dot_source, "digraph {}");
        assert_eq!(req.input, "hi");
        assert!(req.pipeline_id.is_none());
        assert!(req.variables.is_empty());
    }

    #[test]
    fn should_serialize_status_response() {
        let resp = RunStatusResponse {
            run_id: "run-1".into(),
            status: RunStatus::Running,
            current_node: Some("build".into()),
            node_outcomes: HashMap::new(),
            output: None,
            error: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "running");
        assert_eq!(json["current_node"], "build");
    }

    #[test]
    fn should_compare_run_status() {
        assert_eq!(RunStatus::Queued, RunStatus::Queued);
        assert_ne!(RunStatus::Running, RunStatus::Completed);
    }

    #[test]
    fn should_roundtrip_cancel_request() {
        let req = CancelRequest { run_id: "run-42".into() };
        let json = serde_json::to_string(&req).unwrap();
        let back: CancelRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run_id, "run-42");
    }
}
