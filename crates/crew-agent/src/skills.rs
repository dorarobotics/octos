//! Workspace skills loader.
//!
//! Loads skills from `.crew/skills/{name}/SKILL.md` with simple frontmatter.

use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr};

/// Information about a loaded skill.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub version: Option<String>,
    pub author: Option<String>,
    pub path: PathBuf,
    pub available: bool,
    pub always: bool,
    /// Kept for API compatibility; always false now (all skills are workspace-installed).
    pub builtin: bool,
    /// True if this skill package includes a manifest.json (provides tools).
    pub has_tools: bool,
}

/// Loads workspace skills from `.crew/skills/`.
pub struct SkillsLoader {
    skills_dir: PathBuf,
}

impl SkillsLoader {
    /// Create a new loader for the given data directory.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            skills_dir: data_dir.as_ref().join("skills"),
        }
    }

    /// List all installed skills from `.crew/skills/`.
    pub async fn list_skills(&self) -> Result<Vec<SkillInfo>> {
        let mut skills = Vec::new();

        let entries = match tokio::fs::read_dir(&self.skills_dir).await {
            Ok(entries) => Some(entries),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).wrap_err("failed to read skills directory"),
        };

        if let Some(mut entries) = entries {
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let skill_file = path.join("SKILL.md");
                if let Ok(content) = tokio::fs::read_to_string(&skill_file).await {
                    if let Some(info) = parse_skill(&skill_file, &content, false) {
                        skills.push(info);
                    }
                }
            }
        }

        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(skills)
    }

    /// Load a specific skill's full content (without frontmatter).
    pub async fn load_skill(&self, name: &str) -> Result<Option<String>> {
        let skill_file = self.skills_dir.join(name).join("SKILL.md");
        match tokio::fs::read_to_string(&skill_file).await {
            Ok(content) => Ok(Some(strip_frontmatter(&content))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).wrap_err_with(|| format!("failed to read skill: {name}")),
        }
    }

    /// Build an XML summary of all skills for the system prompt.
    pub async fn build_skills_summary(&self) -> Result<String> {
        let skills = self.list_skills().await?;
        if skills.is_empty() {
            return Ok(String::new());
        }

        let mut xml = String::from("<skills>\n");
        for s in &skills {
            let tools_attr = if s.has_tools { " tools=\"true\"" } else { "" };
            xml.push_str(&format!(
                "  <skill available=\"{}\"{}>\n    <name>{}</name>\n    <description>{}</description>\n    <location>{}</location>\n  </skill>\n",
                s.available, tools_attr, s.name, s.description, s.path.display()
            ));
        }
        xml.push_str("</skills>");
        Ok(xml)
    }

    /// Get names of always-on skills that meet their requirements.
    pub async fn get_always_skills(&self) -> Result<Vec<String>> {
        let skills = self.list_skills().await?;
        Ok(skills
            .into_iter()
            .filter(|s| s.always && s.available)
            .map(|s| s.name)
            .collect())
    }

    /// Load full content (minus frontmatter) for the given skill names, joined by `---`.
    pub async fn load_skills_for_context(&self, names: &[String]) -> Result<String> {
        let mut sections = Vec::new();
        for name in names {
            if let Some(content) = self.load_skill(name).await? {
                sections.push(content);
            }
        }
        Ok(sections.join("\n---\n"))
    }
}

/// Parse skill frontmatter and check requirements.
fn parse_skill(path: &Path, content: &str, builtin: bool) -> Option<SkillInfo> {
    let (fm, _) = split_frontmatter(content);
    let fm = fm?;

    let name = fm_value(&fm, "name").unwrap_or_else(|| {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    });

    let description = fm_value(&fm, "description").unwrap_or_default();
    let always = fm_value(&fm, "always")
        .map(|v| v == "true")
        .unwrap_or(false);

    let bins_ok = fm_value(&fm, "requires_bins")
        .map(|v| {
            v.split(',')
                .map(|b| b.trim())
                .filter(|b| !b.is_empty())
                .all(which_exists)
        })
        .unwrap_or(true);

    let env_ok = fm_value(&fm, "requires_env")
        .map(|v| {
            v.split(',')
                .map(|e| e.trim())
                .filter(|e| !e.is_empty())
                .all(|var| std::env::var(var).is_ok())
        })
        .unwrap_or(true);

    let version = fm_value(&fm, "version");
    let author = fm_value(&fm, "author");
    let has_tools = !builtin
        && path
            .parent()
            .map(|p| p.join("manifest.json").exists())
            .unwrap_or(false);

    Some(SkillInfo {
        name,
        description,
        version,
        author,
        path: path.to_path_buf(),
        available: bins_ok && env_ok,
        always,
        builtin,
        has_tools,
    })
}

/// Split content into (Option<frontmatter_lines>, body).
fn split_frontmatter(content: &str) -> (Option<Vec<String>>, &str) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (None, content);
    }

    // Find second ---
    let after_first = &trimmed[3..].trim_start_matches(['\r', '\n']);
    if let Some(end) = after_first.find("\n---") {
        let fm_text = &after_first[..end];
        let lines: Vec<String> = fm_text.lines().map(|l| l.to_string()).collect();
        let body_start = end + 4; // skip \n---
        let body = after_first[body_start..].trim_start_matches(['\r', '\n']);
        (Some(lines), body)
    } else {
        (None, content)
    }
}

/// Extract a key from frontmatter lines (simple `key: value` format).
fn fm_value(lines: &[String], key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    lines.iter().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            Some(trimmed[prefix.len()..].trim().to_string())
        } else {
            None
        }
    })
}

/// Strip frontmatter from content, returning only the body.
fn strip_frontmatter(content: &str) -> String {
    let (_, body) = split_frontmatter(content);
    body.to_string()
}

/// Check if a binary exists on PATH.
fn which_exists(bin: &str) -> bool {
    std::process::Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup_skills_dir(dir: &TempDir) -> PathBuf {
        let skills = dir.path().join("skills");
        tokio::fs::create_dir_all(&skills).await.unwrap();
        skills
    }

    #[tokio::test]
    async fn test_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let loader = SkillsLoader::new(dir.path());
        let skills = loader.list_skills().await.unwrap();
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_list_and_load_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = setup_skills_dir(&dir).await;

        let skill_dir = skills_dir.join("greet");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        tokio::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: greet\ndescription: Say hello\nalways: false\n---\nYou are a greeter.\n",
        )
        .await
        .unwrap();

        let loader = SkillsLoader::new(dir.path());
        let skills = loader.list_skills().await.unwrap();
        let greet = skills.iter().find(|s| s.name == "greet").unwrap();
        assert_eq!(greet.description, "Say hello");
        assert!(!greet.always);
        assert!(greet.available);
        assert!(!greet.builtin);

        let content = loader.load_skill("greet").await.unwrap().unwrap();
        assert_eq!(content, "You are a greeter.\n");
    }

    #[tokio::test]
    async fn test_xml_summary() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = setup_skills_dir(&dir).await;

        let skill_dir = skills_dir.join("test-skill");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        tokio::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: A test\n---\nBody.\n",
        )
        .await
        .unwrap();

        let loader = SkillsLoader::new(dir.path());
        let summary = loader.build_skills_summary().await.unwrap();
        assert!(summary.contains("<skills>"));
        assert!(summary.contains("<name>test-skill</name>"));
        assert!(summary.contains("<description>A test</description>"));
    }

    #[tokio::test]
    async fn test_always_filtering() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = setup_skills_dir(&dir).await;

        for (name, always) in &[("a", "true"), ("b", "false"), ("c", "true")] {
            let sd = skills_dir.join(name);
            tokio::fs::create_dir_all(&sd).await.unwrap();
            tokio::fs::write(
                sd.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: d\nalways: {always}\n---\nbody\n"),
            )
            .await
            .unwrap();
        }

        let loader = SkillsLoader::new(dir.path());
        let always = loader.get_always_skills().await.unwrap();
        assert!(always.contains(&"a".to_string()));
        assert!(always.contains(&"c".to_string()));
        assert!(!always.contains(&"b".to_string()));
    }

    #[test]
    fn test_frontmatter_parsing() {
        let content = "---\nname: foo\ndescription: bar\nalways: true\n---\nBody here.\n";
        let (fm, body) = split_frontmatter(content);
        let fm = fm.unwrap();
        assert_eq!(fm_value(&fm, "name").unwrap(), "foo");
        assert_eq!(fm_value(&fm, "description").unwrap(), "bar");
        assert_eq!(fm_value(&fm, "always").unwrap(), "true");
        assert_eq!(body, "Body here.\n");
    }

    #[test]
    fn test_no_frontmatter() {
        let content = "Just some text.";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_none());
        assert_eq!(body, content);
    }

    #[test]
    fn test_version_author_parsing() {
        let content =
            "---\nname: my-skill\ndescription: Does X\nversion: 1.2.3\nauthor: alice\n---\nBody.\n";
        let path = PathBuf::from("/tmp/fake/my-skill/SKILL.md");
        let info = parse_skill(&path, content, false).unwrap();
        assert_eq!(info.version.as_deref(), Some("1.2.3"));
        assert_eq!(info.author.as_deref(), Some("alice"));
        assert!(!info.has_tools);
    }

    #[tokio::test]
    async fn test_has_tools_detection() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = setup_skills_dir(&dir).await;

        // Skill without tools
        let plain_dir = skills_dir.join("plain");
        tokio::fs::create_dir_all(&plain_dir).await.unwrap();
        tokio::fs::write(
            plain_dir.join("SKILL.md"),
            "---\nname: plain\ndescription: No tools\n---\nBody\n",
        )
        .await
        .unwrap();

        // Skill with tools (has manifest.json)
        let tool_dir = skills_dir.join("with-tools");
        tokio::fs::create_dir_all(&tool_dir).await.unwrap();
        tokio::fs::write(
            tool_dir.join("SKILL.md"),
            "---\nname: with-tools\ndescription: Has tools\n---\nBody\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            tool_dir.join("manifest.json"),
            r#"{"name": "with-tools", "version": "1.0", "tools": []}"#,
        )
        .await
        .unwrap();

        let loader = SkillsLoader::new(dir.path());
        let skills = loader.list_skills().await.unwrap();
        let plain = skills.iter().find(|s| s.name == "plain").unwrap();
        assert!(!plain.has_tools);
        let with_tools = skills.iter().find(|s| s.name == "with-tools").unwrap();
        assert!(with_tools.has_tools);
    }
}
