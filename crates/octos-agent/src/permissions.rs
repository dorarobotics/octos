//! Tiered safety permission model for robotic tool authorization.
//!
//! Provides a four-tier safety model where each tool declares its minimum
//! required tier, and a `RobotPermissionPolicy` authorizes or denies execution
//! based on the current session's maximum allowed tier.

use serde::{Deserialize, Serialize};

/// Safety tiers ordered from least to most dangerous.
///
/// The ordering is: Observe < SafeMotion < FullActuation < EmergencyOverride.
/// A session with tier `SafeMotion` can execute tools requiring `Observe` or
/// `SafeMotion`, but not `FullActuation` or `EmergencyOverride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyTier {
    /// Read-only: cameras, sensors, status queries. No actuation.
    Observe,
    /// Low-risk motion: slow moves within verified workspace bounds.
    SafeMotion,
    /// Full-speed actuation with force control. Requires operator awareness.
    FullActuation,
    /// Bypass all safety limits. For emergency recovery only.
    EmergencyOverride,
}

impl SafetyTier {
    /// Human-readable label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::SafeMotion => "safe_motion",
            Self::FullActuation => "full_actuation",
            Self::EmergencyOverride => "emergency_override",
        }
    }
}

impl std::fmt::Display for SafetyTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Axis-aligned workspace bounds for safe motion validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl WorkspaceBounds {
    /// Check if a point (x, y, z) is within bounds.
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.x_min
            && x <= self.x_max
            && y >= self.y_min
            && y <= self.y_max
            && z >= self.z_min
            && z <= self.z_max
    }
}

/// Error returned when a tool's required tier exceeds the session's allowed tier.
#[derive(Debug, Clone)]
pub struct PermissionDenied {
    pub tool_name: String,
    pub required: SafetyTier,
    pub allowed: SafetyTier,
}

impl std::fmt::Display for PermissionDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "permission denied: tool '{}' requires tier '{}' but session allows up to '{}'",
            self.tool_name, self.required, self.allowed
        )
    }
}

impl std::error::Error for PermissionDenied {}

/// Policy that authorizes tool execution based on safety tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotPermissionPolicy {
    /// Maximum tier allowed in the current session.
    pub max_tier: SafetyTier,
    /// Optional workspace bounds for safe-motion validation.
    #[serde(default)]
    pub workspace: Option<WorkspaceBounds>,
}

impl Default for RobotPermissionPolicy {
    fn default() -> Self {
        Self {
            max_tier: SafetyTier::Observe,
            workspace: None,
        }
    }
}

impl RobotPermissionPolicy {
    /// Create a policy with the given maximum tier.
    pub fn new(max_tier: SafetyTier) -> Self {
        Self {
            max_tier,
            workspace: None,
        }
    }

    /// Create a policy with workspace bounds.
    pub fn with_workspace(mut self, bounds: WorkspaceBounds) -> Self {
        self.workspace = Some(bounds);
        self
    }

    /// Check if a tool with the given required tier is authorized.
    ///
    /// Returns `Ok(())` if allowed, `Err(PermissionDenied)` if the tool's
    /// required tier exceeds the session's maximum.
    pub fn authorize(&self, tool_name: &str, required: SafetyTier) -> Result<(), PermissionDenied> {
        if required <= self.max_tier {
            Ok(())
        } else {
            Err(PermissionDenied {
                tool_name: tool_name.to_string(),
                required,
                allowed: self.max_tier,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_authorize_lower_tier() {
        let policy = RobotPermissionPolicy::new(SafetyTier::FullActuation);
        assert!(policy.authorize("camera_read", SafetyTier::Observe).is_ok());
        assert!(policy
            .authorize("slow_move", SafetyTier::SafeMotion)
            .is_ok());
        assert!(policy
            .authorize("fast_move", SafetyTier::FullActuation)
            .is_ok());
    }

    #[test]
    fn should_deny_higher_tier() {
        let policy = RobotPermissionPolicy::new(SafetyTier::SafeMotion);
        let err = policy
            .authorize("fast_move", SafetyTier::FullActuation)
            .unwrap_err();
        assert_eq!(err.tool_name, "fast_move");
        assert_eq!(err.required, SafetyTier::FullActuation);
        assert_eq!(err.allowed, SafetyTier::SafeMotion);

        let err = policy
            .authorize("e_stop", SafetyTier::EmergencyOverride)
            .unwrap_err();
        assert_eq!(err.required, SafetyTier::EmergencyOverride);
    }

    #[test]
    fn should_order_tiers() {
        assert!(SafetyTier::Observe < SafetyTier::SafeMotion);
        assert!(SafetyTier::SafeMotion < SafetyTier::FullActuation);
        assert!(SafetyTier::FullActuation < SafetyTier::EmergencyOverride);
    }

    #[test]
    fn should_serialize_snake_case() {
        let tier = SafetyTier::SafeMotion;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"safe_motion\"");

        let deserialized: SafetyTier = serde_json::from_str("\"full_actuation\"").unwrap();
        assert_eq!(deserialized, SafetyTier::FullActuation);
    }

    #[test]
    fn should_default_observe() {
        let policy = RobotPermissionPolicy::default();
        assert_eq!(policy.max_tier, SafetyTier::Observe);
        assert!(policy.authorize("sensor", SafetyTier::Observe).is_ok());
        assert!(policy.authorize("move", SafetyTier::SafeMotion).is_err());
    }

    #[test]
    fn should_check_workspace_bounds() {
        let bounds = WorkspaceBounds {
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            z_min: 0.0,
            z_max: 2.0,
        };
        assert!(bounds.contains(0.0, 0.0, 1.0));
        assert!(!bounds.contains(2.0, 0.0, 1.0));
        assert!(!bounds.contains(0.0, 0.0, -0.1));
    }
}
