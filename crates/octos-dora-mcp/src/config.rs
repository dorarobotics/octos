//! Configuration loader for dora tool mappings.

use crate::DoraToolMapping;
use serde::{Deserialize, Serialize};

/// Top-level bridge configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Tool mappings.
    pub mappings: Vec<DoraToolMapping>,
}

impl BridgeConfig {
    /// Load from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed or missing required fields.
    pub fn from_json(json: &str) -> eyre::Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        Ok(config)
    }

    /// Load from a JSON file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is invalid.
    pub fn from_file(path: &str) -> eyre::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_config_from_json() {
        let json = r#"{
            "description": "UR5e inspection tools",
            "mappings": [
                {
                    "tool_name": "scan_station",
                    "description": "Scan station for objects",
                    "dora_node_id": "moveit-skills",
                    "dora_output_id": "skill_request",
                    "parameters": {"station": "Station ID to scan"},
                    "safety_tier": "full_actuation",
                    "timeout_secs": 120
                }
            ]
        }"#;
        let config = BridgeConfig::from_json(json).unwrap();
        assert_eq!(config.mappings.len(), 1);
        assert_eq!(config.mappings[0].tool_name, "scan_station");
        assert_eq!(config.mappings[0].safety_tier, "full_actuation");
    }

    #[test]
    fn should_default_description_to_empty_string() {
        let json = r#"{"mappings": []}"#;
        let config = BridgeConfig::from_json(json).unwrap();
        assert_eq!(config.description, "");
    }

    #[test]
    fn should_fail_on_invalid_json() {
        let result = BridgeConfig::from_json("{not valid json}");
        assert!(result.is_err());
    }
}
