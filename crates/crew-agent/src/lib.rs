//! Agent runtime, tool execution, and coordination for crew-rs.
//!
//! This crate provides:
//! - Agent struct that runs the agent loop
//! - Tool router for dispatching tool calls
//! - Command policy for approval before execution
//! - Progress reporting for real-time updates
//! - Integration with codex sandboxing (when enabled)

mod agent;
pub mod policy;
pub mod progress;
pub mod tools;

pub use agent::{Agent, AgentConfig, ConversationResponse};
pub use progress::{ConsoleReporter, ProgressEvent, ProgressReporter, SilentReporter};
pub use tools::{
    EditFileTool, GlobTool, GrepTool, ReadFileTool, ShellTool, Tool, ToolRegistry, ToolResult,
    WebFetchTool, WebSearchTool, WriteFileTool,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_dir() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("test.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/module.rs"),
            "// Module\npub struct Foo;\n",
        )
        .unwrap();
        dir
    }

    #[tokio::test]
    async fn test_glob_tool() {
        let dir = setup_test_dir();
        let tool = GlobTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"pattern": "*.rs"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("test.rs"));
        assert!(result.output.contains("lib.rs"));
    }

    #[tokio::test]
    async fn test_glob_recursive() {
        let dir = setup_test_dir();
        let tool = GlobTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"pattern": "**/*.rs"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("src/module.rs"));
    }

    #[tokio::test]
    async fn test_grep_tool() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"pattern": "println"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("test.rs"));
        assert!(result.output.contains("println"));
    }

    #[tokio::test]
    async fn test_grep_with_context() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"pattern": "add", "context": 1}))
            .await
            .unwrap();

        assert!(result.success);
        // Should include surrounding lines
        assert!(result.output.contains("pub fn"));
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"pattern": "FOO", "ignore_case": true}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("Foo"));
    }

    #[tokio::test]
    async fn test_read_file_tool() {
        let dir = setup_test_dir();
        let tool = ReadFileTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"path": "test.rs"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("fn main()"));
    }

    #[tokio::test]
    async fn test_write_file_tool() {
        let dir = setup_test_dir();
        let tool = WriteFileTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({
                "path": "new_file.rs",
                "content": "// New file\n"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.file_modified.is_some());
        assert!(dir.path().join("new_file.rs").exists());
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let dir = setup_test_dir();
        let registry = ToolRegistry::with_builtins(dir.path());

        // Should have all builtin tools
        let specs = registry.specs();
        let names: Vec<_> = specs.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"shell"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"web_fetch"));
    }

    #[tokio::test]
    async fn test_registry_execute() {
        let dir = setup_test_dir();
        let registry = ToolRegistry::with_builtins(dir.path());

        let result = registry
            .execute("read_file", &serde_json::json!({"path": "test.rs"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("fn main()"));
    }

    #[tokio::test]
    async fn test_registry_unknown_tool() {
        let dir = setup_test_dir();
        let registry = ToolRegistry::with_builtins(dir.path());

        let result = registry
            .execute("nonexistent", &serde_json::json!({}))
            .await;

        assert!(result.is_err());
    }
}
