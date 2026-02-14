//! Tool framework for agent tool execution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use crew_core::TokenUsage;
use crew_llm::ToolSpec;
use eyre::Result;

/// Result of executing a tool.
#[derive(Default)]
pub struct ToolResult {
    /// Output to return to the LLM.
    pub output: String,
    /// Whether the tool execution succeeded.
    pub success: bool,
    /// File modified by this tool (if any).
    pub file_modified: Option<PathBuf>,
    /// Tokens used by this tool (for subagent tools).
    pub tokens_used: Option<TokenUsage>,
}

/// Trait for implementing tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must be unique).
    fn name(&self) -> &str;

    /// Description for the LLM.
    fn description(&self) -> &str;

    /// JSON Schema for input parameters.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given arguments.
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult>;
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Provider-specific policy that filters specs() output without removing tools.
    provider_policy: Option<ToolPolicy>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            provider_policy: None,
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    /// Register a tool from an existing Arc (for keeping a separate reference).
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get tool specifications for the LLM, filtered by provider policy if set.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .filter(|t| {
                self.provider_policy
                    .as_ref()
                    .is_none_or(|p| p.is_allowed(t.name()))
            })
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Retain only tools whose names satisfy the predicate.
    pub fn retain(&mut self, f: impl Fn(&str) -> bool) {
        self.tools.retain(|name, _| f(name));
    }

    /// Remove tools not permitted by the given policy.
    pub fn apply_policy(&mut self, policy: &ToolPolicy) {
        if policy.is_empty() {
            return;
        }
        self.retain(|name| policy.is_allowed(name));
    }

    /// Set a provider-specific policy that filters `specs()` and `execute()`.
    ///
    /// Unlike `apply_policy` which permanently removes tools from the registry,
    /// this keeps tools registered but blocks both spec visibility and execution.
    pub fn set_provider_policy(&mut self, policy: ToolPolicy) {
        if policy.is_empty() {
            return;
        }
        self.provider_policy = Some(policy);
    }

    /// Return the current provider policy (if any), so callers like SpawnTool
    /// can propagate it to subagent registries.
    pub fn provider_policy(&self) -> Option<&ToolPolicy> {
        self.provider_policy.as_ref()
    }

    /// Execute a tool by name.
    ///
    /// Respects provider policy: tools hidden from `specs()` are also blocked
    /// from execution. This prevents an LLM from calling tools it shouldn't
    /// have access to.
    pub async fn execute(&self, name: &str, args: &serde_json::Value) -> Result<ToolResult> {
        if let Some(ref policy) = self.provider_policy {
            if !policy.is_allowed(name) {
                eyre::bail!("tool '{}' denied by provider policy", name);
            }
        }

        // Reject oversized arguments (1 MB limit)
        const MAX_ARGS_SIZE: usize = 1_048_576;
        let args_str = args.to_string();
        if args_str.len() > MAX_ARGS_SIZE {
            eyre::bail!(
                "tool '{}' arguments too large: {} bytes (max {})",
                name,
                args_str.len(),
                MAX_ARGS_SIZE
            );
        }

        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| eyre::eyre!("unknown tool: {}", name))?;
        tool.execute(args).await
    }
}

// Tool policy
pub mod policy;
pub use policy::ToolPolicy;

// Built-in tools
pub mod diff_edit;
pub mod edit_file;
pub mod glob_tool;
pub mod grep_tool;
pub mod list_dir;
pub mod message;
pub mod read_file;
pub mod shell;
pub mod spawn;
pub mod web_fetch;
pub mod web_search;
pub mod write_file;

#[cfg(feature = "browser")]
pub mod browser;

pub use diff_edit::DiffEditTool;
pub use edit_file::EditFileTool;
pub use glob_tool::GlobTool;
pub use grep_tool::GrepTool;
pub use list_dir::ListDirTool;
pub use message::MessageTool;
pub use read_file::ReadFileTool;
pub use shell::ShellTool;
pub use spawn::SpawnTool;
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
pub use write_file::WriteFileTool;

#[cfg(feature = "browser")]
pub use browser::BrowserTool;

use std::path::{Component, Path};

use crate::sandbox::{NoSandbox, Sandbox};

/// Resolve a user-provided path, ensuring it stays within base_dir.
///
/// Rejects absolute paths and prevents traversal via `../`.
/// Does NOT follow symlinks (normalize only, no filesystem access).
pub fn resolve_path(base_dir: &Path, user_path: &str) -> Result<PathBuf> {
    if PathBuf::from(user_path).is_absolute() {
        eyre::bail!("absolute paths are not allowed: {}", user_path);
    }

    let path = base_dir.join(user_path);
    let normalized = normalize_path(&path);
    let base_normalized = normalize_path(base_dir);

    if !normalized.starts_with(&base_normalized) {
        eyre::bail!("path outside working directory: {}", user_path);
    }

    Ok(normalized)
}

/// Normalize path by resolving `.` and `..` components without filesystem access.
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            // RootDir and Prefix reset the path (absolute path semantics)
            Component::RootDir | Component::Prefix(_) => {
                out.push(component.as_os_str());
            }
            Component::Normal(seg) => {
                out.push(seg);
            }
        }
    }
    out
}

/// Check that a path is not a symlink. Returns error message if it is.
///
/// Call AFTER `resolve_path` and before any filesystem read/write.
/// Prevents symlink-based escapes where a link inside base_dir points outside.
pub async fn reject_symlink(path: &Path) -> Option<ToolResult> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(meta) if meta.is_symlink() => Some(ToolResult {
            output: "Symlinks are not allowed".to_string(),
            success: false,
            ..Default::default()
        }),
        _ => None,
    }
}

impl ToolRegistry {
    /// Create a registry with built-in tools for the given working directory.
    pub fn with_builtins(cwd: impl AsRef<Path>) -> Self {
        Self::with_builtins_and_sandbox(cwd, Box::new(NoSandbox))
    }

    /// Create a registry with built-in tools and a custom sandbox for shell commands.
    pub fn with_builtins_and_sandbox(cwd: impl AsRef<Path>, sandbox: Box<dyn Sandbox>) -> Self {
        let cwd = cwd.as_ref();
        let mut registry = Self::new();
        registry.register(ShellTool::new(cwd).with_sandbox(sandbox));
        registry.register(ReadFileTool::new(cwd));
        registry.register(DiffEditTool::new(cwd));
        registry.register(EditFileTool::new(cwd));
        registry.register(WriteFileTool::new(cwd));
        registry.register(GlobTool::new(cwd));
        registry.register(GrepTool::new(cwd));
        registry.register(ListDirTool::new(cwd));
        registry.register(WebSearchTool::new());
        registry.register(WebFetchTool::new());
        #[cfg(feature = "browser")]
        registry.register(BrowserTool::new());
        registry
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_resolve_rejects_absolute_path() {
        let base = Path::new("/home/user/project");
        assert!(resolve_path(base, "/etc/passwd").is_err());
        assert!(resolve_path(base, "/home/user/project/../../../etc/shadow").is_err());
    }

    #[test]
    fn test_resolve_blocks_parent_traversal() {
        let base = Path::new("/home/user/project");
        assert!(resolve_path(base, "../../../etc/passwd").is_err());
        assert!(resolve_path(base, "subdir/../../..").is_err());
        assert!(resolve_path(base, "foo/../../../secret").is_err());
    }

    #[test]
    fn test_resolve_allows_valid_relative() {
        let base = Path::new("/home/user/project");
        let p = resolve_path(base, "src/main.rs").unwrap();
        assert_eq!(p, PathBuf::from("/home/user/project/src/main.rs"));
    }

    #[test]
    fn test_resolve_allows_dot_segments_within_base() {
        let base = Path::new("/home/user/project");
        let p = resolve_path(base, "src/../src/lib.rs").unwrap();
        assert_eq!(p, PathBuf::from("/home/user/project/src/lib.rs"));
    }

    #[test]
    fn test_resolve_allows_current_dir() {
        let base = Path::new("/home/user/project");
        let p = resolve_path(base, "./README.md").unwrap();
        assert_eq!(p, PathBuf::from("/home/user/project/README.md"));
    }

    #[test]
    fn test_normalize_handles_complex_paths() {
        assert_eq!(
            normalize_path(Path::new("/a/b/../c/./d")),
            PathBuf::from("/a/c/d")
        );
        assert_eq!(
            normalize_path(Path::new("/a/b/../../c")),
            PathBuf::from("/c")
        );
    }
}
