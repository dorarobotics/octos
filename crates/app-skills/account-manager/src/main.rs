//! Standalone account-manager skill binary.
//!
//! Manages sub-accounts under a parent profile by reading/writing profile JSON
//! files in `$CREW_HOME/profiles/`. Communicates via stdin/stdout JSON protocol.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ── Profile types (minimal mirror of crew-cli profiles) ──────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserProfile {
    id: String,
    name: String,
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    data_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    config: ProfileConfig,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ProfileConfig {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    api_type: Option<String>,
    #[serde(default)]
    fallback_models: Vec<serde_json::Value>,
    #[serde(default)]
    channels: Vec<serde_json::Value>,
    #[serde(default)]
    gateway: GatewaySettings,
    #[serde(default)]
    email: Option<serde_json::Value>,
    #[serde(default)]
    env_vars: HashMap<String, String>,
    // Preserve any unknown fields
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct GatewaySettings {
    #[serde(default)]
    max_history: Option<usize>,
    #[serde(default)]
    max_iterations: Option<u32>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    max_concurrent_sessions: Option<usize>,
    #[serde(default)]
    browser_timeout_secs: Option<u64>,
}

// ── Tool input ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Input {
    action: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    sub_account_id: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    telegram_token: Option<String>,
    #[serde(default)]
    enable: Option<bool>,
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
        output_error(&format!("Failed to read stdin: {e}"));
        return;
    }

    let input: Input = match serde_json::from_str(&buf) {
        Ok(v) => v,
        Err(e) => {
            output_error(&format!("Invalid JSON input: {e}"));
            return;
        }
    };

    let crew_home = match std::env::var("CREW_HOME") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => {
            // Fallback: ~/.crew
            match home_dir() {
                Some(h) => h.join(".crew"),
                None => {
                    output_error("CREW_HOME is not set and cannot determine home directory");
                    return;
                }
            }
        }
    };

    let profile_id = match std::env::var("CREW_PROFILE_ID") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            output_error("CREW_PROFILE_ID is not set — this tool must be run from a gateway");
            return;
        }
    };

    let profiles_dir = crew_home.join("profiles");
    if !profiles_dir.exists() {
        output_error(&format!(
            "Profiles directory not found: {}",
            profiles_dir.display()
        ));
        return;
    }

    match input.action.as_str() {
        "list" => action_list(&profiles_dir, &profile_id),
        "create" => action_create(&profiles_dir, &profile_id, &input),
        "delete" => action_delete(&profiles_dir, &profile_id, &input),
        "info" => action_info(&profiles_dir, &profile_id, &input),
        other => output_error(&format!(
            "Unknown action: '{other}'. Valid actions: list, create, delete, info"
        )),
    }
}

// ── Actions ──────────────────────────────────────────────────────────

fn action_list(profiles_dir: &Path, parent_id: &str) {
    let subs = match list_sub_accounts(profiles_dir, parent_id) {
        Ok(v) => v,
        Err(e) => {
            output_error(&format!("Failed to list sub-accounts: {e}"));
            return;
        }
    };

    if subs.is_empty() {
        output_ok("No sub-accounts found. You can create one by asking me to create a sub-account with a name.");
        return;
    }

    let mut lines = vec![format!("Found {} sub-account(s):", subs.len())];
    for s in &subs {
        let status = if s.enabled { "enabled" } else { "disabled" };
        let channels = channel_summary(&s.config.channels);
        let prompt_preview = s
            .config
            .gateway
            .system_prompt
            .as_deref()
            .map(|p| {
                let truncated: String = p.chars().take(60).collect();
                if p.len() > 60 {
                    format!(" | prompt: \"{truncated}...\"")
                } else {
                    format!(" | prompt: \"{truncated}\"")
                }
            })
            .unwrap_or_default();
        lines.push(format!(
            "  - {id} ({name}, {status}) [{channels}]{prompt_preview}",
            id = s.id,
            name = s.name,
        ));
    }
    output_ok(&lines.join("\n"));
}

fn action_create(profiles_dir: &Path, parent_id: &str, input: &Input) {
    let name = match &input.name {
        Some(n) if !n.trim().is_empty() => n.trim(),
        _ => {
            output_error("'name' is required for the 'create' action.");
            return;
        }
    };

    // Verify parent exists
    let parent_path = profiles_dir.join(format!("{parent_id}.json"));
    if !parent_path.exists() {
        output_error(&format!("Parent profile '{parent_id}' not found."));
        return;
    }

    let sub_id = format!("{parent_id}--{}", slugify(name));

    // Check for existing
    let sub_path = profiles_dir.join(format!("{sub_id}.json"));
    if sub_path.exists() {
        output_error(&format!("Sub-account '{sub_id}' already exists."));
        return;
    }

    // Build channel config
    let mut channels: Vec<serde_json::Value> = Vec::new();
    let mut env_vars: HashMap<String, String> = HashMap::new();

    if let Some(ref token) = input.telegram_token {
        let env_name = format!(
            "TELEGRAM_BOT_TOKEN_{}",
            slugify(name).to_uppercase().replace('-', "_")
        );
        channels.push(json!({
            "type": "telegram",
            "token_env": &env_name,
            "allowed_senders": ""
        }));
        env_vars.insert(env_name, token.clone());
    }

    let now = Utc::now();
    let profile = UserProfile {
        id: sub_id.clone(),
        name: name.to_string(),
        enabled: input.enable.unwrap_or(false),
        data_dir: None,
        parent_id: Some(parent_id.to_string()),
        config: ProfileConfig {
            channels,
            gateway: GatewaySettings {
                system_prompt: input.system_prompt.clone(),
                ..Default::default()
            },
            env_vars,
            ..Default::default()
        },
        created_at: now,
        updated_at: now,
    };

    match save_profile(profiles_dir, &profile) {
        Ok(()) => {
            let mut msg = format!("Created sub-account '{sub_id}' (name: {name}).");
            if profile.enabled {
                msg.push_str("\nThe account is enabled and will start on next gateway restart.");
            } else {
                msg.push_str(
                    "\nThe account is disabled. Enable it from the dashboard or CLI to start it.",
                );
            }
            if profile.config.channels.is_empty() {
                msg.push_str("\nNo messaging channels configured yet. Add a Telegram token or other channel via the dashboard or CLI.");
            } else {
                msg.push_str(&format!(
                    "\nChannels: {}",
                    channel_summary(&profile.config.channels)
                ));
            }
            output_ok(&msg);
        }
        Err(e) => output_error(&format!("Failed to create sub-account: {e}")),
    }
}

fn action_delete(profiles_dir: &Path, parent_id: &str, input: &Input) {
    let sub_id = match &input.sub_account_id {
        Some(id) if !id.trim().is_empty() => id.trim(),
        _ => {
            output_error("'sub_account_id' is required for the 'delete' action.");
            return;
        }
    };

    // Load and verify it's a child of this parent
    match load_profile(profiles_dir, sub_id) {
        Ok(Some(profile)) => {
            if profile.parent_id.as_deref() != Some(parent_id) {
                output_error(&format!(
                    "'{sub_id}' is not a sub-account of the current profile."
                ));
                return;
            }
        }
        Ok(None) => {
            output_error(&format!("Sub-account '{sub_id}' not found."));
            return;
        }
        Err(e) => {
            output_error(&format!("Failed to read sub-account: {e}"));
            return;
        }
    }

    let path = profiles_dir.join(format!("{sub_id}.json"));
    match std::fs::remove_file(&path) {
        Ok(()) => output_ok(&format!("Deleted sub-account '{sub_id}'.")),
        Err(e) => output_error(&format!("Failed to delete sub-account: {e}")),
    }
}

fn action_info(profiles_dir: &Path, parent_id: &str, input: &Input) {
    // Resolve sub_account_id: use explicit ID, or guess from name
    let sub_id: String = if let Some(ref id) = input.sub_account_id {
        let trimmed = id.trim();
        if trimmed.is_empty() {
            output_error("'sub_account_id' is required for the 'info' action.");
            return;
        }
        trimmed.to_string()
    } else if let Some(ref name) = input.name {
        let guessed = format!("{parent_id}--{}", slugify(name));
        if load_profile(profiles_dir, &guessed)
            .ok()
            .flatten()
            .is_some()
        {
            guessed
        } else {
            output_error(
                "'sub_account_id' is required for the 'info' action. \
                 You can get the ID from the 'list' action.",
            );
            return;
        }
    } else {
        output_error("'sub_account_id' is required for the 'info' action.");
        return;
    };

    let profile = match load_profile(profiles_dir, &sub_id) {
        Ok(Some(p)) => p,
        Ok(None) => {
            output_error(&format!("Sub-account '{sub_id}' not found."));
            return;
        }
        Err(e) => {
            output_error(&format!("Failed to read sub-account: {e}"));
            return;
        }
    };

    if profile.parent_id.as_deref() != Some(parent_id) {
        output_error(&format!(
            "'{sub_id}' is not a sub-account of the current profile."
        ));
        return;
    }

    // Load parent for inherited info
    let parent_info = load_profile(profiles_dir, parent_id)
        .ok()
        .flatten()
        .map(|p| {
            format!(
                "\nInherited provider: {} | model: {}",
                p.config.provider.as_deref().unwrap_or("none"),
                p.config.model.as_deref().unwrap_or("none"),
            )
        })
        .unwrap_or_default();

    let channels = channel_summary(&profile.config.channels);
    let status = if profile.enabled {
        "enabled"
    } else {
        "disabled"
    };
    let prompt = profile
        .config
        .gateway
        .system_prompt
        .as_deref()
        .unwrap_or("(none)");

    let msg = format!(
        "Sub-account: {id}\n\
         Name: {name}\n\
         Parent: {parent_id}\n\
         Status: {status}\n\
         Channels: [{channels}]\n\
         System prompt: {prompt}\n\
         Created: {created}{parent_info}",
        id = profile.id,
        name = profile.name,
        created = profile.created_at.format("%Y-%m-%d %H:%M UTC"),
    );
    output_ok(&msg);
}

// ── Profile I/O helpers ──────────────────────────────────────────────

fn list_sub_accounts(profiles_dir: &Path, parent_id: &str) -> Result<Vec<UserProfile>, String> {
    let entries =
        std::fs::read_dir(profiles_dir).map_err(|e| format!("cannot read profiles dir: {e}"))?;

    let mut subs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let profile: UserProfile = match serde_json::from_str(&content) {
            Ok(p) => p,
            Err(_) => continue,
        };
        if profile.parent_id.as_deref() == Some(parent_id) {
            subs.push(profile);
        }
    }
    subs.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(subs)
}

fn load_profile(profiles_dir: &Path, id: &str) -> Result<Option<UserProfile>, String> {
    let path = profiles_dir.join(format!("{id}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let content =
        std::fs::read_to_string(&path).map_err(|e| format!("cannot read profile: {e}"))?;
    let profile: UserProfile =
        serde_json::from_str(&content).map_err(|e| format!("cannot parse profile: {e}"))?;
    Ok(Some(profile))
}

fn save_profile(profiles_dir: &Path, profile: &UserProfile) -> Result<(), String> {
    let path = profiles_dir.join(format!("{}.json", profile.id));
    let content = serde_json::to_string_pretty(profile)
        .map_err(|e| format!("cannot serialize profile: {e}"))?;
    std::fs::write(&path, content).map_err(|e| format!("cannot write profile: {e}"))?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

fn slugify(s: &str) -> String {
    let slug: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    // Collapse consecutive dashes and trim
    let mut result = String::new();
    let mut prev_dash = false;
    for c in slug.trim_matches('-').chars() {
        if c == '-' {
            if !prev_dash {
                result.push(c);
            }
            prev_dash = true;
        } else {
            result.push(c);
            prev_dash = false;
        }
    }
    if result.is_empty() {
        // Non-ASCII name (e.g. Chinese): generate a deterministic short hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        format!("{:08x}", hasher.finish() as u32)
    } else {
        result
    }
}

fn channel_summary(channels: &[serde_json::Value]) -> String {
    if channels.is_empty() {
        return "none".to_string();
    }
    channels
        .iter()
        .filter_map(|c| c.get("type").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn output_ok(message: &str) {
    let out = json!({ "output": message, "success": true });
    println!("{out}");
}

fn output_error(message: &str) {
    let out = json!({ "output": message, "success": false });
    println!("{out}");
    std::process::exit(1);
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}
