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
    /// Tokens used by this tool (for delegate_task).
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
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    /// Get tool specifications for the LLM.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Execute a tool by name.
    pub async fn execute(&self, name: &str, args: &serde_json::Value) -> Result<ToolResult> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| eyre::eyre!("unknown tool: {}", name))?;
        tool.execute(args).await
    }
}

// Built-in tools
pub mod delegate;
pub mod delegate_batch;
pub mod edit_file;
pub mod glob_tool;
pub mod grep_tool;
pub mod read_file;
pub mod shell;
pub mod web_fetch;
pub mod web_search;
pub mod write_file;

pub use delegate::DelegateTaskTool;
pub use delegate_batch::DelegateBatchTool;
pub use edit_file::EditFileTool;
pub use glob_tool::GlobTool;
pub use grep_tool::GrepTool;
pub use read_file::ReadFileTool;
pub use shell::ShellTool;
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
pub use write_file::WriteFileTool;

use std::path::Path;

use crew_llm::LlmProvider;
use crew_memory::EpisodeStore;

impl ToolRegistry {
    /// Create a registry with built-in tools for the given working directory.
    pub fn with_builtins(cwd: impl AsRef<Path>) -> Self {
        let cwd = cwd.as_ref();
        let mut registry = Self::new();
        registry.register(ShellTool::new(cwd));
        registry.register(ReadFileTool::new(cwd));
        registry.register(EditFileTool::new(cwd));
        registry.register(WriteFileTool::new(cwd));
        registry.register(GlobTool::new(cwd));
        registry.register(GrepTool::new(cwd));
        registry.register(WebSearchTool::new());
        registry.register(WebFetchTool::new());
        registry
    }

    /// Create a registry with coordinator tools (builtins + delegate + batch).
    pub fn with_coordinator_tools(
        cwd: impl AsRef<Path>,
        llm: Arc<dyn LlmProvider>,
        memory: Arc<EpisodeStore>,
    ) -> Self {
        let cwd = cwd.as_ref();
        let mut registry = Self::with_builtins(cwd);
        registry.register(DelegateTaskTool::new(
            llm.clone(),
            memory.clone(),
            cwd.to_path_buf(),
        ));
        registry.register(DelegateBatchTool::new(llm, memory, cwd.to_path_buf()));
        registry
    }
}
