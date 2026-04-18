//! Dora-RS to MCP tool bridge.
//!
//! Maps dora-rs dataflow nodes to MCP-compatible tools that can be
//! registered in the octos agent's [`ToolRegistry`].
//!
//! # Overview
//!
//! Each [`DoraToolMapping`] declares a tool name, description, the dora node
//! and output IDs that handle requests, and an optional safety tier string.
//! [`DoraToolBridge`] wraps a mapping and implements the [`Tool`] trait so it
//! can be registered directly in the agent.
//!
//! # Example
//!
//! ```no_run
//! use octos_dora_mcp::{BridgeConfig, load_bridges};
//!
//! let config = BridgeConfig::from_file("config/dora_tool_map.json").unwrap();
//! let bridges = load_bridges(&config);
//! // bridges can be registered in a ToolRegistry:
//! // for bridge in bridges { registry.register(bridge); }
//! ```

mod config;

use async_trait::async_trait;
use octos_agent::tools::{Tool, ToolResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use config::BridgeConfig;

/// Safety tiers for dora tool operations.
///
/// Stored as plain strings in config for forward compatibility.
/// Use [`SafetyTier`] variants for structured comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SafetyTier {
    /// Read-only observation, no physical effect.
    Observe,
    /// Controlled motion within pre-validated bounds.
    SafeMotion,
    /// Unrestricted actuation of joints and end-effectors.
    FullActuation,
    /// Emergency stop and override commands.
    EmergencyOverride,
}

impl SafetyTier {
    /// Parse from the string representation used in config files.
    pub fn from_config_str(s: &str) -> Self {
        match s {
            "observe" => SafetyTier::Observe,
            "safe_motion" => SafetyTier::SafeMotion,
            "full_actuation" => SafetyTier::FullActuation,
            "emergency_override" => SafetyTier::EmergencyOverride,
            _ => SafetyTier::Observe,
        }
    }

    /// Return the canonical string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            SafetyTier::Observe => "observe",
            SafetyTier::SafeMotion => "safe_motion",
            SafetyTier::FullActuation => "full_actuation",
            SafetyTier::EmergencyOverride => "emergency_override",
        }
    }
}

/// Mapping from a dora-rs node output to an MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoraToolMapping {
    /// MCP tool name exposed to the agent.
    pub tool_name: String,
    /// Description for the LLM.
    pub description: String,
    /// Dora node ID that handles this tool.
    pub dora_node_id: String,
    /// Dora output ID to send the request to.
    pub dora_output_id: String,
    /// Expected input parameters (name -> description).
    pub parameters: HashMap<String, String>,
    /// Required safety tier for this tool.
    #[serde(default = "default_tier")]
    pub safety_tier: String,
    /// Timeout in seconds for the tool call.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_tier() -> String {
    "observe".to_string()
}

fn default_timeout() -> u64 {
    30
}

/// A bridge that wraps a [`DoraToolMapping`] as an MCP-compatible [`Tool`].
///
/// In production the `execute` method would forward the request to the dora
/// dataflow via an IPC channel and await the response.  The current
/// implementation returns a placeholder describing the would-be forwarded
/// payload so the bridge can be tested without a running dora runtime.
pub struct DoraToolBridge {
    mapping: DoraToolMapping,
}

impl DoraToolBridge {
    /// Create a new bridge from the given mapping.
    pub fn new(mapping: DoraToolMapping) -> Self {
        Self { mapping }
    }

    /// Return a reference to the underlying mapping.
    pub fn mapping(&self) -> &DoraToolMapping {
        &self.mapping
    }

    /// Parse the safety tier string into the typed enum.
    pub fn required_safety_tier(&self) -> SafetyTier {
        SafetyTier::from_config_str(&self.mapping.safety_tier)
    }

    /// Build the JSON Schema object describing the tool's input parameters.
    fn build_input_schema(&self) -> serde_json::Value {
        let properties: serde_json::Map<String, serde_json::Value> = self
            .mapping
            .parameters
            .iter()
            .map(|(name, desc)| {
                (
                    name.clone(),
                    serde_json::json!({
                        "type": "string",
                        "description": desc
                    }),
                )
            })
            .collect();

        let required: Vec<serde_json::Value> = self
            .mapping
            .parameters
            .keys()
            .map(|k| serde_json::Value::String(k.clone()))
            .collect();

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        })
    }
}

#[async_trait]
impl Tool for DoraToolBridge {
    fn name(&self) -> &str {
        &self.mapping.tool_name
    }

    fn description(&self) -> &str {
        &self.mapping.description
    }

    fn input_schema(&self) -> serde_json::Value {
        self.build_input_schema()
    }

    fn tags(&self) -> &[&str] {
        &["dora", "mcp-bridge"]
    }

    async fn execute(&self, args: &serde_json::Value) -> eyre::Result<ToolResult> {
        // In a real implementation this would send args to the dora node
        // via the dora dataflow and await the response.  We return a
        // placeholder indicating the bridge would forward to dora.
        let request = serde_json::json!({
            "tool": self.mapping.tool_name,
            "node_id": self.mapping.dora_node_id,
            "output_id": self.mapping.dora_output_id,
            "args": args,
            "timeout_secs": self.mapping.timeout_secs,
        });

        Ok(ToolResult {
            success: true,
            output: format!(
                "DoraToolBridge: would forward to node '{}' output '{}': {}",
                self.mapping.dora_node_id,
                self.mapping.dora_output_id,
                serde_json::to_string_pretty(&request).unwrap_or_default()
            ),
            file_modified: None,
            files_to_send: vec![],
            tokens_used: None,
        })
    }
}

/// Load tool mappings from a [`BridgeConfig`] and create bridge tools.
pub fn load_bridges(config: &BridgeConfig) -> Vec<DoraToolBridge> {
    config
        .mappings
        .iter()
        .map(|m| DoraToolBridge::new(m.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mapping() -> DoraToolMapping {
        let mut params = HashMap::new();
        params.insert("waypoint".to_string(), "Target waypoint ID".to_string());

        DoraToolMapping {
            tool_name: "navigate_to".to_string(),
            description: "Navigate robot to a waypoint".to_string(),
            dora_node_id: "moveit-skills".to_string(),
            dora_output_id: "skill_request".to_string(),
            parameters: params,
            safety_tier: "safe_motion".to_string(),
            timeout_secs: 60,
        }
    }

    #[test]
    fn should_expose_correct_tool_name() {
        let bridge = DoraToolBridge::new(sample_mapping());
        assert_eq!(bridge.name(), "navigate_to");
    }

    #[test]
    fn should_expose_correct_description() {
        let bridge = DoraToolBridge::new(sample_mapping());
        assert_eq!(bridge.description(), "Navigate robot to a waypoint");
    }

    #[test]
    fn should_build_input_schema_with_parameters() {
        let bridge = DoraToolBridge::new(sample_mapping());
        let schema = bridge.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["waypoint"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "waypoint"));
    }

    #[test]
    fn should_build_empty_schema_when_no_parameters() {
        let mut mapping = sample_mapping();
        mapping.parameters.clear();
        let bridge = DoraToolBridge::new(mapping);
        let schema = bridge.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].as_object().unwrap().is_empty());
    }

    #[test]
    fn should_include_dora_tags() {
        let bridge = DoraToolBridge::new(sample_mapping());
        let tags = bridge.tags();
        assert!(tags.contains(&"dora"));
        assert!(tags.contains(&"mcp-bridge"));
    }

    #[test]
    fn should_parse_safety_tier_safe_motion() {
        let bridge = DoraToolBridge::new(sample_mapping());
        assert_eq!(bridge.required_safety_tier(), SafetyTier::SafeMotion);
    }

    #[test]
    fn should_default_to_observe_tier_for_unknown_string() {
        let mut mapping = sample_mapping();
        mapping.safety_tier = "unknown_tier".to_string();
        let bridge = DoraToolBridge::new(mapping);
        assert_eq!(bridge.required_safety_tier(), SafetyTier::Observe);
    }

    #[test]
    fn should_parse_all_safety_tier_variants() {
        for (s, expected) in [
            ("observe", SafetyTier::Observe),
            ("safe_motion", SafetyTier::SafeMotion),
            ("full_actuation", SafetyTier::FullActuation),
            ("emergency_override", SafetyTier::EmergencyOverride),
        ] {
            assert_eq!(SafetyTier::from_config_str(s), expected, "failed for '{s}'");
        }
    }

    #[test]
    fn should_round_trip_safety_tier_as_str() {
        for tier in [
            SafetyTier::Observe,
            SafetyTier::SafeMotion,
            SafetyTier::FullActuation,
            SafetyTier::EmergencyOverride,
        ] {
            assert_eq!(SafetyTier::from_config_str(tier.as_str()), tier);
        }
    }

    #[tokio::test]
    async fn should_execute_bridge_tool_with_args() {
        let bridge = DoraToolBridge::new(sample_mapping());
        let result = bridge
            .execute(&serde_json::json!({"waypoint": "A"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("moveit-skills"));
        assert!(result.output.contains("skill_request"));
        assert!(result.output.contains("navigate_to"));
    }

    #[tokio::test]
    async fn should_execute_bridge_tool_with_empty_args() {
        let bridge = DoraToolBridge::new(sample_mapping());
        let result = bridge.execute(&serde_json::json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.file_modified.is_none());
        assert!(result.files_to_send.is_empty());
    }

    #[test]
    fn should_load_bridges_from_config() {
        let json = r#"{
            "description": "Test config",
            "mappings": [
                {
                    "tool_name": "tool_a",
                    "description": "First tool",
                    "dora_node_id": "node-1",
                    "dora_output_id": "out-1",
                    "parameters": {},
                    "safety_tier": "observe",
                    "timeout_secs": 10
                },
                {
                    "tool_name": "tool_b",
                    "description": "Second tool",
                    "dora_node_id": "node-2",
                    "dora_output_id": "out-2",
                    "parameters": {"param": "A parameter"},
                    "safety_tier": "full_actuation",
                    "timeout_secs": 30
                }
            ]
        }"#;
        let config = BridgeConfig::from_json(json).unwrap();
        let bridges = load_bridges(&config);
        assert_eq!(bridges.len(), 2);
        assert_eq!(bridges[0].name(), "tool_a");
        assert_eq!(bridges[1].name(), "tool_b");
        assert_eq!(bridges[1].required_safety_tier(), SafetyTier::FullActuation);
    }

    #[test]
    fn should_apply_default_safety_tier_when_omitted() {
        let json = r#"{
            "mappings": [
                {
                    "tool_name": "minimal_tool",
                    "description": "A minimal tool",
                    "dora_node_id": "node-x",
                    "dora_output_id": "out-x",
                    "parameters": {}
                }
            ]
        }"#;
        let config = BridgeConfig::from_json(json).unwrap();
        let bridges = load_bridges(&config);
        assert_eq!(bridges[0].required_safety_tier(), SafetyTier::Observe);
        assert_eq!(bridges[0].mapping().timeout_secs, 30);
    }
}
