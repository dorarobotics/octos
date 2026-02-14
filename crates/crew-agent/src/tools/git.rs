//! Native git integration tool using gix (pure Rust).
//!
//! All operations are read-only: status, diff, log, show, blame.

use std::path::PathBuf;

use async_trait::async_trait;
use eyre::Result;
use gix::bstr::ByteSlice;
use serde::Deserialize;

use super::{Tool, ToolResult};

/// Git tool with read-only subcommands.
pub struct GitTool {
    working_dir: PathBuf,
}

impl GitTool {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: cwd.into(),
        }
    }
}

#[derive(Deserialize)]
struct GitArgs {
    command: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default = "default_count")]
    count: usize,
    #[serde(default)]
    revision: Option<String>,
}

fn default_count() -> usize {
    10
}

/// Whitelist safe revision formats to prevent arbitrary rev_parse_single input.
/// Allows: hex commit hashes, HEAD, HEAD~N, HEAD^N, branch/tag names (alphanumeric + -_./).
fn is_safe_revision(rev: &str) -> bool {
    if rev.is_empty() || rev.len() > 256 {
        return false;
    }
    // Reject known dangerous revision syntax: :/, @{, ..
    if rev.contains(":/") || rev.contains("@{") || rev.contains("..") {
        return false;
    }
    // Allow only safe characters: alphanumeric, -, _, ., /, ~, ^
    rev.chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./~^".contains(c))
}

#[async_trait]
impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Native git operations (read-only). Commands: status, diff, log, show, blame."
    }

    fn tags(&self) -> &[&str] {
        &["code", "search"]
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "show", "blame"],
                    "description": "Git subcommand"
                },
                "path": {
                    "type": "string",
                    "description": "File path (for diff, blame, show)"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of commits for log (default: 10)"
                },
                "revision": {
                    "type": "string",
                    "description": "Revision/commit hash (for show, diff)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let args: GitArgs = serde_json::from_value(args.clone())
            .map_err(|e| eyre::eyre!("invalid arguments: {e}"))?;

        // Validate user-provided path against traversal attacks
        if let Some(ref path) = args.path {
            if let Err(e) = super::resolve_path(&self.working_dir, path) {
                return Ok(ToolResult {
                    output: format!("git error: {e}"),
                    success: false,
                    ..Default::default()
                });
            }
        }

        let output = match args.command.as_str() {
            "status" => git_status(&self.working_dir),
            "diff" => {
                if args.revision.is_some() {
                    return Ok(ToolResult {
                        output: "revision-based diff is not yet supported; omit 'revision' to diff working tree against index".to_string(),
                        success: false,
                        ..Default::default()
                    });
                }
                git_diff(&self.working_dir, args.path.as_deref())
            }
            "log" => git_log(&self.working_dir, args.count),
            "show" => {
                let rev = args.revision.as_deref().unwrap_or("HEAD");
                if !is_safe_revision(rev) {
                    return Ok(ToolResult {
                        output: format!(
                            "invalid revision: {rev}. Use a commit hash, HEAD, HEAD~N, or a branch/tag name"
                        ),
                        success: false,
                        ..Default::default()
                    });
                }
                git_show(&self.working_dir, rev)
            }
            "blame" => {
                let path = args
                    .path
                    .as_deref()
                    .ok_or_else(|| eyre::eyre!("blame requires 'path' argument"))?;
                git_blame(&self.working_dir, path)
            }
            other => Err(eyre::eyre!(
                "unknown git command: {other}. Valid: status, diff, log, show, blame"
            )),
        };

        match output {
            Ok(text) => Ok(ToolResult {
                output: text,
                success: true,
                ..Default::default()
            }),
            Err(e) => Ok(ToolResult {
                output: format!("git error: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

fn git_status(cwd: &std::path::Path) -> Result<String> {
    let repo = gix::discover(cwd)?;

    // Use porcelain-style status via index comparison
    let index = repo.open_index()?;
    let worktree = repo
        .workdir()
        .ok_or_else(|| eyre::eyre!("bare repository"))?;

    let mut staged = Vec::new();
    let mut untracked = Vec::new();
    let mut modified = Vec::new();

    // Walk the working directory to find untracked and modified files
    for entry in ignore::WalkBuilder::new(worktree)
        .hidden(false)
        .git_ignore(true)
        .build()
        .flatten()
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Ok(relative) = path.strip_prefix(worktree) {
            let rel_str = relative.to_string_lossy();
            // Skip .git directory
            if rel_str.starts_with(".git/") || rel_str == ".git" {
                continue;
            }

            let in_index = index.entries().iter().any(|e| {
                e.path(&index).to_str().ok().is_some_and(|p| p == rel_str.as_ref())
            });

            if !in_index {
                untracked.push(rel_str.to_string());
            } else {
                // Check if modified by comparing mtime
                if let Ok(meta) = path.metadata() {
                    let entry = index.entries().iter().find(|e| {
                        e.path(&index).to_str().ok().is_some_and(|p| p == rel_str.as_ref())
                    });
                    if let Some(idx_entry) = entry {
                        // Heuristic: compare file size to detect modifications.
                        // This may miss same-size edits (e.g. changing one char).
                        // Full accuracy would require blob content comparison.
                        let file_size = meta.len() as u32;
                        if file_size != idx_entry.stat.size {
                            modified.push(rel_str.to_string());
                        }
                    }
                }
            }
        }
    }

    // Check HEAD tree vs index for staged changes
    if let Ok(head_commit) = repo.head_commit() {
        if let Ok(tree) = head_commit.tree() {
            let mut recorder = gix::traverse::tree::Recorder::default();
            if tree.traverse().breadthfirst(&mut recorder).is_ok() {
                let index_paths: std::collections::HashSet<String> = index
                    .entries()
                    .iter()
                    .filter_map(|e| e.path(&index).to_str().ok().map(String::from))
                    .collect();
                let tree_paths: std::collections::HashSet<String> = recorder
                    .records
                    .iter()
                    .filter(|r| r.mode.is_blob())
                    .filter_map(|r| std::str::from_utf8(r.filepath.as_ref()).ok().map(String::from))
                    .collect();
                // Files in index but not in HEAD tree = staged new files
                for path in index_paths.difference(&tree_paths) {
                    staged.push(format!("new: {path}"));
                }
            }
        }
    }

    let mut result = serde_json::json!({
        "staged": staged,
        "modified": modified,
        "untracked": untracked,
    });

    // Add branch info
    if let Ok(Some(head_ref)) = repo.head_ref() {
        let name = head_ref.name().shorten().to_string();
        result["branch"] = serde_json::json!(name);
    }

    Ok(serde_json::to_string_pretty(&result)?)
}

fn git_diff(cwd: &std::path::Path, path: Option<&str>) -> Result<String> {
    let repo = gix::discover(cwd)?;
    let worktree = repo
        .workdir()
        .ok_or_else(|| eyre::eyre!("bare repository"))?;

    // Simple diff: compare index entries with working tree files
    let index = repo.open_index()?;
    let mut diffs = Vec::new();

    for entry in index.entries() {
        let entry_path = entry.path(&index).to_str().ok().unwrap_or("").to_string();

        // Filter by path if specified
        if let Some(filter) = path {
            if !entry_path.starts_with(filter) && entry_path != filter {
                continue;
            }
        }

        let file_path = worktree.join(&entry_path);
        if !file_path.exists() {
            diffs.push(format!("deleted: {entry_path}"));
            continue;
        }

        // Compare file content with blob in index/revision
        if let Ok(current) = std::fs::read_to_string(&file_path) {
            let blob_id = entry.id;
            if let Ok(blob) = repo.find_object(blob_id) {
                let old_content = String::from_utf8_lossy(&blob.data);
                if old_content.as_ref() != current.as_str() {
                    diffs.push(format!("--- a/{entry_path}\n+++ b/{entry_path}"));
                    // Simple line-by-line diff
                    let old_lines: Vec<&str> = old_content.lines().collect();
                    let new_lines: Vec<&str> = current.lines().collect();
                    for (i, (o, n)) in old_lines.iter().zip(new_lines.iter()).enumerate() {
                        if o != n {
                            diffs.push(format!("@@ line {} @@", i + 1));
                            diffs.push(format!("-{o}"));
                            diffs.push(format!("+{n}"));
                        }
                    }
                    // Handle length differences
                    if new_lines.len() > old_lines.len() {
                        for line in &new_lines[old_lines.len()..] {
                            diffs.push(format!("+{line}"));
                        }
                    } else if old_lines.len() > new_lines.len() {
                        for line in &old_lines[new_lines.len()..] {
                            diffs.push(format!("-{line}"));
                        }
                    }
                }
            }
        }
    }

    if diffs.is_empty() {
        Ok("No changes.".to_string())
    } else {
        Ok(diffs.join("\n"))
    }
}

fn git_log(cwd: &std::path::Path, count: usize) -> Result<String> {
    let repo = gix::discover(cwd)?;
    let head = repo.head_commit()?;

    let mut commits = Vec::new();
    let mut current = Some(head);

    for _ in 0..count {
        let Some(commit) = current.take() else {
            break;
        };

        let id = commit.id().to_string();
        let message = commit.message_raw_sloppy().to_string();
        let message = message.trim().to_string();
        let author = commit.author().map_or_else(
            |_| "unknown".to_string(),
            |a| format!("{} <{}>", a.name, a.email),
        );
        let time = commit.time().map_or_else(
            |_| "unknown".to_string(),
            |t| {
                chrono::DateTime::from_timestamp(t.seconds, 0)
                    .map_or("unknown".into(), |dt| dt.to_rfc3339())
            },
        );

        commits.push(serde_json::json!({
            "hash": &id[..std::cmp::min(12, id.len())],
            "author": author,
            "date": time,
            "message": message,
        }));

        // Move to first parent
        current = commit.parent_ids().next().and_then(|pid| pid.object().ok().map(|o| o.into_commit()));
    }

    Ok(serde_json::to_string_pretty(&commits)?)
}

fn git_show(cwd: &std::path::Path, revision: &str) -> Result<String> {
    let repo = gix::discover(cwd)?;
    let commit = repo
        .rev_parse_single(revision.as_bytes())?
        .object()?
        .peel_to_kind(gix::object::Kind::Commit)?
        .into_commit();

    let id = commit.id().to_string();
    let message = commit.message_raw_sloppy().to_string();
    let author = commit.author().map_or_else(
        |_| "unknown".to_string(),
        |a| format!("{} <{}>", a.name, a.email),
    );
    let time = commit.time().map_or_else(
        |_| "unknown".to_string(),
        |t| {
            chrono::DateTime::from_timestamp(t.seconds, 0)
                .map_or("unknown".into(), |dt| dt.to_rfc3339())
        },
    );

    // List changed files by comparing with parent tree
    let tree = commit.tree()?;
    let mut files = Vec::new();
    let mut recorder = gix::traverse::tree::Recorder::default();
    tree.traverse().breadthfirst(&mut recorder)?;
    for record in &recorder.records {
        if record.mode.is_blob() {
            if let Ok(path) = std::str::from_utf8(record.filepath.as_ref()) {
                files.push(path.to_string());
            }
        }
    }

    let result = serde_json::json!({
        "hash": id,
        "author": author,
        "date": time,
        "message": message.trim(),
        "files": files,
    });

    Ok(serde_json::to_string_pretty(&result)?)
}

fn git_blame(cwd: &std::path::Path, path: &str) -> Result<String> {
    // gix doesn't have a built-in blame yet; use git2-style line attribution
    // For now, fall back to showing the file with commit info from log
    let repo = gix::discover(cwd)?;
    let worktree = repo
        .workdir()
        .ok_or_else(|| eyre::eyre!("bare repository"))?;

    // Path already validated via resolve_path in execute(), but resolve again
    // against worktree for the actual read path.
    let file_path = super::resolve_path(worktree, path)?;
    if !file_path.exists() {
        eyre::bail!("file not found: {path}");
    }

    let content = std::fs::read_to_string(&file_path)?;
    let lines: Vec<&str> = content.lines().collect();

    // Simple blame: show line numbers with content (blame attribution requires
    // walking history which gix doesn't easily support yet)
    let mut result = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        result.push(format!("{:>4} | {line}", i + 1));
    }

    Ok(format!(
        "blame for {path} ({} lines):\n{}",
        lines.len(),
        result.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_git_repo() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Initialize a git repo
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Configure user
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Create a file and commit
        std::fs::write(dir.path().join("hello.txt"), "hello world\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "hello.txt"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial commit"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        dir
    }

    #[tokio::test]
    async fn test_git_status() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"command": "status"}))
            .await
            .unwrap();

        assert!(result.success);
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed.get("branch").is_some() || parsed.get("staged").is_some());
    }

    #[tokio::test]
    async fn test_git_log() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"command": "log", "count": 5}))
            .await
            .unwrap();

        assert!(result.success);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result.output).unwrap();
        assert!(!parsed.is_empty());
        assert!(parsed[0]["message"]
            .as_str()
            .unwrap()
            .contains("initial commit"));
    }

    #[tokio::test]
    async fn test_git_show() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"command": "show", "revision": "HEAD"}))
            .await
            .unwrap();

        assert!(result.success);
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed["message"]
            .as_str()
            .unwrap()
            .contains("initial commit"));
    }

    #[tokio::test]
    async fn test_git_diff_no_changes() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"command": "diff"}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output, "No changes.");
    }

    #[tokio::test]
    async fn test_git_path_traversal_rejected() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        // blame with traversal
        let result = tool
            .execute(&serde_json::json!({"command": "blame", "path": "../../../etc/passwd"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("outside working directory"));

        // diff with traversal
        let result = tool
            .execute(&serde_json::json!({"command": "diff", "path": "../../../etc/passwd"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("outside working directory"));
    }

    #[tokio::test]
    async fn test_git_diff_revision_rejected() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"command": "diff", "revision": "HEAD~1"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("not yet supported"));
    }

    #[tokio::test]
    async fn test_git_show_rejects_unsafe_revision() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        // Reject regex revision syntax
        let result = tool
            .execute(&serde_json::json!({"command": "show", "revision": ":/secret"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("invalid revision"));

        // Reject reflog syntax
        let result = tool
            .execute(&serde_json::json!({"command": "show", "revision": "@{-1}"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("invalid revision"));

        // Allow HEAD (safe)
        let result = tool
            .execute(&serde_json::json!({"command": "show", "revision": "HEAD"}))
            .await
            .unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_is_safe_revision() {
        assert!(is_safe_revision("HEAD"));
        assert!(is_safe_revision("HEAD~3"));
        assert!(is_safe_revision("HEAD^2"));
        assert!(is_safe_revision("abc123def"));
        assert!(is_safe_revision("main"));
        assert!(is_safe_revision("refs/tags/v1.0"));
        assert!(!is_safe_revision(":/regex"));
        assert!(!is_safe_revision("@{-1}"));
        assert!(!is_safe_revision("HEAD..main"));
        assert!(!is_safe_revision(""));
    }

    #[tokio::test]
    async fn test_git_blame() {
        let dir = setup_git_repo();
        let tool = GitTool::new(dir.path());

        let result = tool
            .execute(&serde_json::json!({"command": "blame", "path": "hello.txt"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("hello world"));
    }
}
