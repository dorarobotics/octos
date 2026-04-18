//! Core graph types for pipeline representation.

use std::collections::{HashMap, HashSet};

use octos_core::TokenUsage;
use serde::{Deserialize, Serialize};

/// A parsed, typed pipeline graph ready for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineGraph {
    /// Graph identifier (from `digraph name { ... }`).
    pub id: String,
    /// Human-readable description (from `label` attribute).
    pub label: Option<String>,
    /// Default model key for nodes that don't specify one.
    pub default_model: Option<String>,
    /// Nodes keyed by their ID.
    pub nodes: HashMap<String, PipelineNode>,
    /// Directed edges.
    pub edges: Vec<PipelineEdge>,
    /// Named subgraphs (clusters).
    #[serde(default)]
    pub subgraphs: Vec<Subgraph>,
}

impl PipelineGraph {
    /// Detect cycles in the pipeline graph using DFS with three-color marking.
    /// Returns `Ok(())` if the graph is acyclic, or `Err` with the cycle path.
    pub fn detect_cycles(&self) -> Result<(), String> {
        // Build adjacency list from edges
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(edge.source.as_str())
                .or_default()
                .push(edge.target.as_str());
        }

        // DFS coloring: White = unvisited, Gray = in current path, Black = fully explored
        let mut white: HashSet<&str> = self.nodes.keys().map(|s| s.as_str()).collect();
        let mut gray: HashSet<&str> = HashSet::new();
        let mut black: HashSet<&str> = HashSet::new();
        // Track the path for error reporting
        let mut path: Vec<&str> = Vec::new();

        fn dfs<'a>(
            node: &'a str,
            adj: &HashMap<&str, Vec<&'a str>>,
            white: &mut HashSet<&'a str>,
            gray: &mut HashSet<&'a str>,
            black: &mut HashSet<&'a str>,
            path: &mut Vec<&'a str>,
        ) -> Result<(), String> {
            white.remove(node);
            gray.insert(node);
            path.push(node);

            if let Some(neighbors) = adj.get(node) {
                for &next in neighbors {
                    if black.contains(next) {
                        continue;
                    }
                    if gray.contains(next) {
                        // Found a cycle: build the cycle path from `next` to `next`
                        let cycle_start = path.iter().position(|&n| n == next).unwrap_or(0);
                        let mut cycle: Vec<&str> = path[cycle_start..].to_vec();
                        cycle.push(next);
                        return Err(format!("cycle detected: {}", cycle.join(" -> ")));
                    }
                    dfs(next, adj, white, gray, black, path)?;
                }
            }

            path.pop();
            gray.remove(node);
            black.insert(node);
            Ok(())
        }

        // Visit all nodes (handles disconnected components)
        let all_nodes: Vec<&str> = self.nodes.keys().map(|s| s.as_str()).collect();
        for node in all_nodes {
            if white.contains(node) {
                dfs(node, &adj, &mut white, &mut gray, &mut black, &mut path)?;
            }
        }

        Ok(())
    }
}

/// A single node in the pipeline graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineNode {
    /// Node identifier.
    pub id: String,
    /// Handler type.
    pub handler: HandlerKind,
    /// System prompt template. Supports `{input}` and `{variable_name}` substitution.
    pub prompt: Option<String>,
    /// Human-readable label for progress reporting.
    pub label: Option<String>,
    /// Model key for `ProviderRouter::resolve()` (e.g. "cheap", "strong").
    pub model: Option<String>,
    /// Override context window size in tokens.
    pub context_window: Option<u32>,
    /// Override max output tokens per LLM call. Default 4096 is too low for
    /// nodes that write long outputs (e.g. synthesize writing full reports).
    pub max_output_tokens: Option<u32>,
    /// Allowed tool names for this node. Empty = all builtins.
    pub tools: Vec<String>,
    /// If true, a successful outcome means "pipeline goal achieved".
    pub goal_gate: bool,
    /// Retry on error (default 0).
    pub max_retries: u32,
    /// Per-node timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Hint for edge selection when no condition matches.
    pub suggested_next: Option<String>,
    /// For `Parallel` / `DynamicParallel` nodes: the node to jump to after completion.
    /// All target outputs are merged and fed as input to this convergence node.
    pub converge: Option<String>,
    /// For `DynamicParallel`: prompt template for each worker task.
    /// Supports `{task}` placeholder replaced with each planned task description.
    pub worker_prompt: Option<String>,
    /// For `DynamicParallel`: model key for the planning LLM call (optional).
    pub planner_model: Option<String>,
    /// For `DynamicParallel`: maximum number of dynamic tasks (default 8).
    pub max_tasks: Option<u32>,
    /// Deadline in seconds for this node. On expiry, `deadline_action` fires.
    pub deadline_secs: Option<u64>,
    /// Action to take when deadline expires.
    pub deadline_action: DeadlineAction,
    /// Invariants that must hold during execution.
    pub invariants: Vec<Invariant>,
    /// Whether to save a checkpoint after this node completes.
    pub checkpoint: bool,
}

impl Default for PipelineNode {
    fn default() -> Self {
        Self {
            id: String::new(),
            handler: HandlerKind::Codergen,
            prompt: None,
            label: None,
            model: None,
            context_window: None,
            max_output_tokens: None,
            tools: Vec::new(),
            goal_gate: false,
            max_retries: 0,
            timeout_secs: None,
            suggested_next: None,
            converge: None,
            worker_prompt: None,
            planner_model: None,
            max_tasks: None,
            deadline_secs: None,
            deadline_action: DeadlineAction::Abort,
            invariants: Vec::new(),
            checkpoint: false,
        }
    }
}

/// A directed edge between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEdge {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Condition expression (e.g. `outcome.status == "pass"`).
    pub condition: Option<String>,
    /// Edge weight for priority (default 1.0, must be positive).
    pub weight: f64,
}

/// Handler type for pipeline nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandlerKind {
    /// Run a full agent loop with tools.
    Codergen,
    /// Execute a shell command.
    Shell,
    /// Evaluate a condition (no LLM call).
    Gate,
    /// Pass-through.
    Noop,
    /// Fan-out: run all outgoing targets concurrently, merge results,
    /// then jump to the `converge` node.
    Parallel,
    /// Dynamic fan-out: LLM plans N sub-tasks at runtime, executes them
    /// in parallel, merges results, then jumps to the `converge` node.
    DynamicParallel,
    /// Read sensor data and evaluate condition.
    SensorCheck,
    /// Execute a robot motion command.
    Motion,
    /// Execute a grasp/release action.
    Grasp,
    /// Safety gate — requires safety condition before proceeding.
    SafetyGate,
    /// Wait for an external event (sensor trigger, timer, etc.).
    WaitForEvent,
}

impl HandlerKind {
    /// Parse from a string attribute value.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "codergen" => Some(Self::Codergen),
            "shell" => Some(Self::Shell),
            "gate" => Some(Self::Gate),
            "noop" => Some(Self::Noop),
            "parallel" => Some(Self::Parallel),
            "dynamic_parallel" => Some(Self::DynamicParallel),
            "sensor_check" => Some(Self::SensorCheck),
            "motion" => Some(Self::Motion),
            "grasp" => Some(Self::Grasp),
            "safety_gate" => Some(Self::SafetyGate),
            "wait_for_event" => Some(Self::WaitForEvent),
            _ => None,
        }
    }

    /// Resolve handler from DOT `shape` attribute (Attractor spec mapping).
    pub fn from_shape(shape: &str) -> Option<Self> {
        match shape {
            "Mdiamond" => Some(Self::Noop),             // start node
            "Msquare" => Some(Self::Noop),              // exit node
            "box" => Some(Self::Codergen),              // LLM task (default)
            "hexagon" => Some(Self::Gate),              // human gate / conditional
            "diamond" => Some(Self::Gate),              // conditional routing
            "component" => Some(Self::Parallel),        // parallel fan-out
            "parallelogram" => Some(Self::Shell),       // external tool/command
            "ellipse" => Some(Self::SensorCheck),       // sensor read/evaluate
            "house" => Some(Self::Motion),              // robot motion command
            "trapezium" => Some(Self::Grasp),           // grasp/release action
            "octagon" => Some(Self::SafetyGate),        // safety gate
            "doublecircle" => Some(Self::WaitForEvent), // wait for external event
            _ => None,
        }
    }
}

/// Action to take when a pipeline node exceeds its deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeadlineAction {
    /// Abort the node and mark as failed.
    Abort,
    /// Skip to the next node.
    Skip,
    /// Trigger emergency stop.
    EmergencyStop,
}

/// Runtime invariant that must hold true during node execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    /// Human-readable description.
    pub description: String,
    /// Condition expression to evaluate.
    pub condition: String,
    /// Action on violation.
    pub on_violation: DeadlineAction,
}

/// Checkpoint saved during mission execution for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionCheckpoint {
    /// The node ID where this checkpoint was saved.
    pub node_id: String,
    /// Timestamp when the checkpoint was created.
    pub timestamp_ms: u64,
    /// Serialized state data.
    pub state: serde_json::Value,
    /// Whether this checkpoint is resumable.
    pub resumable: bool,
}

/// The outcome of executing a single pipeline node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeOutcome {
    /// The node ID that produced this outcome.
    pub node_id: String,
    /// Whether the node succeeded.
    pub status: OutcomeStatus,
    /// Text content produced by the node.
    pub content: String,
    /// Token usage for this node.
    pub token_usage: TokenUsage,
    /// Files written by this node's agent.
    #[serde(default)]
    pub files_modified: Vec<std::path::PathBuf>,
}

/// Outcome status for a pipeline node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    Pass,
    Fail,
    Error,
}

/// A named subgraph (cluster) within a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    /// Subgraph identifier (e.g. "cluster_research").
    pub id: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Node IDs belonging to this subgraph.
    pub node_ids: Vec<String>,
}

/// Validate that a pipeline identifier (node ID, run ID, graph ID) is safe
/// for use as a filesystem path component. Rejects path separators, `..`,
/// control characters, and excessively long values.
pub fn validate_pipeline_id(id: &str) -> eyre::Result<()> {
    if id.is_empty() {
        eyre::bail!("pipeline identifier must not be empty");
    }
    if id.len() > 128 {
        eyre::bail!(
            "pipeline identifier too long (max 128 chars): {}",
            id.chars().take(32).collect::<String>()
        );
    }
    if id.contains('/') || id.contains('\\') || id.contains('\0') || id.contains("..") {
        eyre::bail!("pipeline identifier contains unsafe characters: {id}");
    }
    if id.chars().any(|c| c.is_control()) {
        eyre::bail!("pipeline identifier contains control characters: {id}");
    }
    Ok(())
}

/// Summary of a single node execution (for reporting).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    pub node_id: String,
    pub label: String,
    pub model: Option<String>,
    pub token_usage: TokenUsage,
    pub duration_ms: u64,
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn should_parse_robot_handler_kinds() {
        assert_eq!(
            HandlerKind::from_str("sensor_check"),
            Some(HandlerKind::SensorCheck)
        );
        assert_eq!(HandlerKind::from_str("motion"), Some(HandlerKind::Motion));
        assert_eq!(HandlerKind::from_str("grasp"), Some(HandlerKind::Grasp));
        assert_eq!(
            HandlerKind::from_str("safety_gate"),
            Some(HandlerKind::SafetyGate)
        );
        assert_eq!(
            HandlerKind::from_str("wait_for_event"),
            Some(HandlerKind::WaitForEvent)
        );
        // Existing variants still work
        assert_eq!(
            HandlerKind::from_str("codergen"),
            Some(HandlerKind::Codergen)
        );
        assert_eq!(HandlerKind::from_str("unknown_kind"), None);
    }

    #[test]
    fn should_parse_robot_shapes() {
        assert_eq!(
            HandlerKind::from_shape("ellipse"),
            Some(HandlerKind::SensorCheck)
        );
        assert_eq!(HandlerKind::from_shape("house"), Some(HandlerKind::Motion));
        assert_eq!(
            HandlerKind::from_shape("trapezium"),
            Some(HandlerKind::Grasp)
        );
        assert_eq!(
            HandlerKind::from_shape("octagon"),
            Some(HandlerKind::SafetyGate)
        );
        assert_eq!(
            HandlerKind::from_shape("doublecircle"),
            Some(HandlerKind::WaitForEvent)
        );
        // Existing shapes still work
        assert_eq!(HandlerKind::from_shape("box"), Some(HandlerKind::Codergen));
        assert_eq!(HandlerKind::from_shape("unknown_shape"), None);
    }

    #[test]
    fn should_default_deadline_action_to_abort() {
        let node = PipelineNode::default();
        assert_eq!(node.deadline_action, DeadlineAction::Abort);
        assert!(node.deadline_secs.is_none());
        assert!(node.invariants.is_empty());
        assert!(!node.checkpoint);
    }

    #[test]
    fn should_detect_cycle_in_graph() {
        let mut nodes = HashMap::new();
        for id in &["a", "b", "c"] {
            let mut n = PipelineNode::default();
            n.id = id.to_string();
            nodes.insert(id.to_string(), n);
        }
        let edges = vec![
            PipelineEdge {
                source: "a".into(),
                target: "b".into(),
                label: None,
                condition: None,
                weight: 1.0,
            },
            PipelineEdge {
                source: "b".into(),
                target: "c".into(),
                label: None,
                condition: None,
                weight: 1.0,
            },
            PipelineEdge {
                source: "c".into(),
                target: "a".into(),
                label: None,
                condition: None,
                weight: 1.0,
            },
        ];
        let graph = PipelineGraph {
            id: "cyclic".into(),
            label: None,
            default_model: None,
            nodes,
            edges,
            subgraphs: vec![],
        };
        let result = graph.detect_cycles();
        assert!(result.is_err(), "expected cycle detection to return Err");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("cycle detected"),
            "error message should mention cycle: {msg}"
        );
    }

    #[test]
    fn should_serialize_invariant() {
        let inv = Invariant {
            description: "Force must be below 50N".into(),
            condition: "force_sensor < 50.0".into(),
            on_violation: DeadlineAction::EmergencyStop,
        };
        let json = serde_json::to_string(&inv).expect("serialize invariant");
        let decoded: Invariant = serde_json::from_str(&json).expect("deserialize invariant");
        assert_eq!(decoded.description, inv.description);
        assert_eq!(decoded.condition, inv.condition);
        assert_eq!(decoded.on_violation, DeadlineAction::EmergencyStop);
    }
}
