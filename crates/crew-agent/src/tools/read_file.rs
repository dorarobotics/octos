//! Read file tool.

use std::path::PathBuf;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use serde::Deserialize;

use super::{Tool, ToolResult};

/// Tool for reading file contents.
pub struct ReadFileTool {
    /// Base directory for resolving relative paths.
    base_dir: PathBuf,
}

impl ReadFileTool {
    /// Create a new read file tool.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadFileInput {
    path: String,
    #[serde(default)]
    start_line: Option<usize>,
    #[serde(default)]
    end_line: Option<usize>,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the file content with line numbers."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to working directory)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional starting line number (1-indexed)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional ending line number (1-indexed, inclusive)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: ReadFileInput =
            serde_json::from_value(args.clone()).wrap_err("invalid read_file tool input")?;

        // Resolve path (with traversal protection)
        let path = match super::resolve_path(&self.base_dir, &input.path) {
            Ok(p) => p,
            Err(_) => {
                return Ok(ToolResult {
                    output: format!("Path outside working directory: {}", input.path),
                    success: false,
                    ..Default::default()
                })
            }
        };

        if let Some(r) = super::reject_symlink(&path).await {
            return Ok(r);
        }

        // Check if file exists
        if !path.exists() {
            return Ok(ToolResult {
                output: format!("File not found: {}", input.path),
                success: false,
                ..Default::default()
            });
        }

        // Read file
        let content = tokio::fs::read_to_string(&path)
            .await
            .wrap_err_with(|| format!("failed to read file: {}", path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Apply line range
        let start = input.start_line.unwrap_or(1).saturating_sub(1);
        let end = input.end_line.unwrap_or(total_lines).min(total_lines);

        if start >= total_lines {
            return Ok(ToolResult {
                output: format!(
                    "Start line {} is beyond file length ({} lines)",
                    start + 1,
                    total_lines
                ),
                success: false,
                ..Default::default()
            });
        }

        // Format with line numbers
        let mut output = String::new();
        let line_num_width = end.to_string().len();

        for (idx, line) in lines[start..end].iter().enumerate() {
            let line_num = start + idx + 1;
            output.push_str(&format!(
                "{:>width$}│ {}\n",
                line_num,
                line,
                width = line_num_width
            ));
        }

        // Add file info
        if start > 0 || end < total_lines {
            output.push_str(&format!(
                "\n(showing lines {}-{} of {})",
                start + 1,
                end,
                total_lines
            ));
        }

        // Truncate if too long
        const MAX_OUTPUT: usize = 100000;
        if output.len() > MAX_OUTPUT {
            let mut limit = MAX_OUTPUT;
            while limit > 0 && !output.is_char_boundary(limit) {
                limit -= 1;
            }
            output.truncate(limit);
            output.push_str("\n... (content truncated)");
        }

        Ok(ToolResult {
            output,
            success: true,
            ..Default::default()
        })
    }
}
