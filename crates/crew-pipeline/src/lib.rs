//! DOT-based pipeline orchestration engine for crew-rs.
//!
//! Parse DOT graphs with typed attributes, walk the graph with async handlers,
//! and execute multi-step agent workflows with per-node model selection.

pub mod artifact;
pub mod checkpoint;
pub mod condition;
pub mod discovery;
pub mod events;
pub mod executor;
pub mod fidelity;
pub mod graph;
pub mod human_gate;
pub mod manager;
pub mod server;
pub mod thread;
pub mod handler;
pub mod parser;
pub mod run_dir;
pub mod stylesheet;
pub mod tool;
pub mod validate;

pub use events::{
    CollectingEventHandler, PipelineEvent, PipelineEventHandler, TracingEventHandler,
};
pub use executor::{PipelineExecutor, PipelineResult, PipelineStatusBridge};
pub use artifact::ArtifactStore;
pub use checkpoint::{Checkpoint, CheckpointStore};
pub use graph::{
    HandlerKind, NodeOutcome, OutcomeStatus, PipelineEdge, PipelineGraph, PipelineNode, Subgraph,
    validate_pipeline_id,
};
pub use handler::{
    CodergenHandler, GateHandler, Handler, HandlerRegistry, NoopHandler, ShellHandler,
};
pub use parser::parse_dot;
pub use run_dir::{NodeStatus, PipelineRunSummary, RunDir};
pub use stylesheet::{ModelStylesheet, ResolvedStyle, StyleRule};
pub use tool::RunPipelineTool;
pub use fidelity::FidelityMode;
pub use human_gate::{HumanInputProvider, HumanInputType, HumanRequest, HumanResponse};
pub use manager::{ChildExecutor, ChildResult, ChildSpec, ManagerOutcome, PipelineManager, SupervisionStrategy};
pub use server::{CancelRequest, PipelineServer, RunStatus, RunStatusResponse, SubmitRequest, SubmitResponse};
pub use thread::{Thread, ThreadRegistry};
pub use validate::{LintDiagnostic, Severity, validate};
