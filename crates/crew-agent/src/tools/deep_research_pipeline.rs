//! Deep research pipeline: 4-phase parallel research tool.
//!
//! Phase 1: SEARCH — spawn multiple `deep_search` processes in parallel with different query angles
//! Phase 2: EXTRACT — read all sources, deduplicate, batch, extract findings per batch (parallel LLM)
//! Phase 3: SYNTHESIZE — merge all partial findings into final report (single LLM)
//! Phase 4: SAVE — write report to disk

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use crew_core::{Message, MessageRole, TokenUsage};
use crew_llm::{ChatConfig, LlmProvider};
use eyre::{Result, WrapErr};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

use super::research_utils;
use super::{Tool, ToolResult};

/// A single search angle with its own query and depth.
#[derive(Debug, Clone)]
struct SearchAngle {
    query: String,
    depth: u8,
}

/// Raw JSON deserialization target for planning agent output.
#[derive(Deserialize)]
struct RawSearchAngle {
    query: String,
    depth: Option<u8>,
}

/// Deep research pipeline tool — parallel multi-angle search + map-reduce synthesis.
pub struct DeepResearchTool {
    llm: Arc<dyn LlmProvider>,
    working_dir: PathBuf,
    #[allow(dead_code)] // reserved for future research dir resolution
    data_dir: PathBuf,
    plugin_dirs: Vec<PathBuf>,
}

impl DeepResearchTool {
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        working_dir: PathBuf,
        data_dir: PathBuf,
        plugin_dirs: Vec<PathBuf>,
    ) -> Self {
        Self {
            llm,
            working_dir,
            data_dir,
            plugin_dirs,
        }
    }

    /// Find the deep-search binary from plugin directories.
    fn find_deep_search_binary(&self) -> Option<PathBuf> {
        for dir in &self.plugin_dirs {
            let plugin_dir = dir.join("deep-search");
            if !plugin_dir.is_dir() {
                continue;
            }

            // Prefer verified copy (same as PluginLoader)
            let verified = plugin_dir.join(".main_verified");
            if verified.exists() {
                return Some(verified);
            }
            let verified2 = plugin_dir.join(".deep-search_verified");
            if verified2.exists() {
                return Some(verified2);
            }

            // Fallback to unverified
            let main = plugin_dir.join("main");
            if main.exists() {
                return Some(main);
            }
            let named = plugin_dir.join("deep-search");
            if named.exists() {
                return Some(named);
            }
        }
        None
    }

    /// Planning phase: LLM analyzes the query and decides how many angles,
    /// what each angle should search for, and the depth per angle.
    /// Falls back to a simple template if LLM fails.
    async fn plan_research(
        &self,
        query: &str,
        default_depth: u8,
    ) -> (Vec<SearchAngle>, TokenUsage) {
        match self.llm_plan_research(query, default_depth).await {
            Ok((angles, usage)) if angles.len() >= 2 => {
                info!(
                    angle_count = angles.len(),
                    "planning agent decided search strategy"
                );
                return (angles, usage);
            }
            Ok(_) => warn!("planning agent returned too few angles, using template"),
            Err(e) => warn!(error = %e, "planning agent failed, using template"),
        }

        // Template fallback
        (
            self.template_angles(query, default_depth),
            TokenUsage::default(),
        )
    }

    async fn llm_plan_research(
        &self,
        query: &str,
        default_depth: u8,
    ) -> Result<(Vec<SearchAngle>, TokenUsage)> {
        let prompt = format!(
            "[{{\"query\": \"example search\", \"depth\": 2}}]\n\n\
             Above is the output format. Now generate 4-6 search angles for this research query.\n\n\
             Query: {query}\n\n\
             Requirements:\n\
             - 4-6 angles, each a distinct subtopic (NOT rephrasing)\n\
             - Include at least one angle in Chinese (中文) and one in English\n\
             - Primary angles: depth {default_depth}. Supplementary: depth 1\n\
             - Keep each query under 60 characters\n\
             - Cover: core topic, alternatives, technical architecture, trends, cross-language\n\n\
             Respond with ONLY the JSON array. No explanation, no markdown, no text before or after.\n"
        );

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        }];

        let config = ChatConfig {
            max_tokens: Some(2000),
            // Don't set temperature — kimi-k2.5 only allows 1.0, other models vary.
            ..Default::default()
        };

        let response = self.llm.chat(&messages, &[], &config).await?;
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            ..Default::default()
        };

        // Try content first, then reasoning_content (reasoning models like kimi-k2.5
        // put their work in reasoning_content and may leave content empty)
        let content = response.content.unwrap_or_default();
        let text = if content.trim().is_empty() {
            response.reasoning_content.as_deref().unwrap_or("")
        } else {
            &content
        };

        // Extract JSON from response (handle markdown code fences, reasoning text)
        let json_str = extract_json_array(text)
            .ok_or_else(|| eyre::eyre!("no JSON array found in planning response"))?;

        let raw: Vec<RawSearchAngle> =
            serde_json::from_str(json_str).wrap_err("failed to parse planning response as JSON")?;

        let angles: Vec<SearchAngle> = raw
            .into_iter()
            .take(6) // cap at 6
            .map(|r| SearchAngle {
                query: r.query,
                depth: r.depth.unwrap_or(default_depth).clamp(1, 3),
            })
            .collect();

        Ok((angles, usage))
    }

    fn template_angles(&self, query: &str, depth: u8) -> Vec<SearchAngle> {
        // Extract core topic (first 50 chars, break at word boundary)
        let core = if query.len() > 50 {
            match query[..50].rfind(' ') {
                Some(pos) => &query[..pos],
                None => &query[..50],
            }
        } else {
            query
        };

        // Detect if query is primarily CJK (Chinese/Japanese/Korean)
        let cjk_count = query.chars().filter(|c| *c > '\u{2E80}').count();
        let is_cjk = cjk_count > query.chars().count() / 3;

        let mut angles = vec![
            // 1. Original query as-is
            SearchAngle {
                query: query.to_string(),
                depth,
            },
            // 2. Alternatives / comparison angle
            SearchAngle {
                query: format!("{core} alternatives comparison"),
                depth: 1.min(depth),
            },
            // 3. Technical deep-dive
            SearchAngle {
                query: format!("{core} architecture implementation"),
                depth: 1.min(depth),
            },
            // 4. Recent trends
            SearchAngle {
                query: format!("{core} latest trends 2025 2026"),
                depth: 1.min(depth),
            },
        ];

        // 5. Cross-language angle: add Chinese if query is English, English if query is CJK
        if is_cjk {
            angles.push(SearchAngle {
                query: format!("{core} overview comparison English"),
                depth: 1.min(depth),
            });
        } else {
            angles.push(SearchAngle {
                query: format!("{core} 技术方案 对比分析"),
                depth: 1.min(depth),
            });
        }

        angles
    }

    /// Phase 1: Spawn parallel deep_search processes with per-angle depth.
    async fn search_phase(&self, binary: &Path, angles: &[SearchAngle]) -> Vec<PathBuf> {
        let mut handles = Vec::new();

        for angle in angles {
            let binary = binary.to_path_buf();
            let query = angle.query.clone();
            let depth = angle.depth;
            let cwd = self.working_dir.clone();

            handles.push(tokio::spawn(async move {
                invoke_deep_search(&binary, &query, depth, &cwd).await
            }));
        }

        let mut research_dirs = Vec::new();
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(dir)) => {
                    info!(angle = i, dir = %dir.display(), "search completed");
                    research_dirs.push(dir);
                }
                Ok(Err(e)) => {
                    warn!(angle = i, error = %e, "search failed");
                }
                Err(e) => {
                    warn!(angle = i, error = %e, "search task panicked");
                }
            }
        }

        research_dirs
    }

    /// Phase 2+3: Read sources from all dirs, deduplicate, extract, merge.
    async fn synthesis_phase(
        &self,
        research_dirs: &[PathBuf],
        query: &str,
    ) -> Result<(String, TokenUsage)> {
        // Read all source files from all directories
        let mut all_files = Vec::new();
        for dir in research_dirs {
            match research_utils::read_sources(dir).await {
                Ok(files) => {
                    info!(dir = %dir.display(), count = files.len(), "read sources from dir");
                    all_files.extend(files);
                }
                Err(e) => {
                    warn!(dir = %dir.display(), error = %e, "failed to read sources from dir");
                }
            }
        }

        if all_files.is_empty() {
            eyre::bail!("no source files found across all research directories");
        }

        // Deduplicate by URL
        let before_dedup = all_files.len();
        all_files = dedup_by_url(all_files);
        let after_dedup = all_files.len();
        if before_dedup > after_dedup {
            info!(
                before = before_dedup,
                after = after_dedup,
                "deduplicated sources by URL"
            );
        }

        // Truncate if over total limit
        all_files = research_utils::truncate_to_limit(all_files);
        let source_count = all_files.len();

        info!(
            source_count,
            total_chars = all_files.iter().map(|(_, c)| c.len()).sum::<usize>(),
            "starting extraction phase"
        );

        let mut total_tokens = TokenUsage::default();

        // Partition into batches
        let batches = research_utils::partition_batches(&all_files);
        info!(batches = batches.len(), "partitioned into batches");

        if batches.len() == 1 {
            // Single batch: direct synthesis
            let (synthesis, usage) = research_utils::extract_findings(
                self.llm.as_ref(),
                query,
                None,
                &all_files,
                &batches[0],
                1,
                1,
            )
            .await?;

            total_tokens.input_tokens += usage.input_tokens;
            total_tokens.output_tokens += usage.output_tokens;
            return Ok((synthesis, total_tokens));
        }

        // Parallel extraction across batches
        let total_batches = batches.len();
        let mut extraction_handles = Vec::new();

        for (i, batch) in batches.iter().enumerate() {
            let llm = self.llm.clone();
            let query = query.to_string();
            let files = all_files.clone();
            let batch = batch.clone();

            extraction_handles.push(tokio::spawn(async move {
                research_utils::extract_findings(
                    llm.as_ref(),
                    &query,
                    None,
                    &files,
                    &batch,
                    i + 1,
                    total_batches,
                )
                .await
            }));
        }

        let mut partials = Vec::new();
        for (i, handle) in extraction_handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok((findings, usage))) => {
                    total_tokens.input_tokens += usage.input_tokens;
                    total_tokens.output_tokens += usage.output_tokens;
                    if !findings.is_empty() {
                        partials.push(findings);
                    }
                }
                Ok(Err(e)) => {
                    warn!(batch = i + 1, error = %e, "batch extraction failed");
                }
                Err(e) => {
                    warn!(batch = i + 1, error = %e, "extraction task panicked");
                }
            }
        }

        if partials.is_empty() {
            eyre::bail!("all batch extractions failed");
        }

        // Merge phase
        info!(partial_count = partials.len(), "merging partial analyses");

        let (synthesis, merge_usage) =
            research_utils::merge_findings(self.llm.as_ref(), query, None, &partials, source_count)
                .await?;

        total_tokens.input_tokens += merge_usage.input_tokens;
        total_tokens.output_tokens += merge_usage.output_tokens;

        Ok((synthesis, total_tokens))
    }
}

#[derive(Deserialize)]
struct Input {
    query: String,
    #[serde(default = "default_depth")]
    depth: u8,
    #[serde(default)]
    angles: Vec<RawSearchAngle>,
}

fn default_depth() -> u8 {
    2
}

/// Extract a JSON array from LLM output, handling markdown code fences.
fn extract_json_array(text: &str) -> Option<&str> {
    let text = text.trim();

    // Try direct parse first
    if text.starts_with('[') {
        return Some(text);
    }

    // Try extracting from code fence
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return Some(&text[start..=end]);
            }
        }
    }

    None
}

#[async_trait]
impl Tool for DeepResearchTool {
    fn name(&self) -> &str {
        "deep_research"
    }

    fn description(&self) -> &str {
        "Comprehensive research pipeline: runs multiple parallel web searches from different \
         angles, reads all source pages, extracts findings via map-reduce, and synthesizes \
         a detailed report. Much more thorough than a single deep_search — covers the topic \
         from multiple perspectives with deduplication and cross-referencing. Use for complex \
         research questions that need broad, multi-angle coverage."
    }

    fn tags(&self) -> &[&str] {
        &["web", "gateway"]
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or topic to investigate"
                },
                "depth": {
                    "type": "integer",
                    "description": "Search depth per angle (1=quick, 2=standard, 3=thorough). Default: 2",
                    "minimum": 1,
                    "maximum": 3
                },
                "angles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Search query for this angle" },
                            "depth": { "type": "integer", "minimum": 1, "maximum": 3, "description": "Search depth for this angle" }
                        },
                        "required": ["query"]
                    },
                    "description": "Custom search angles. If omitted, a planning agent decides the angles dynamically."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid deep_research input")?;

        let depth = input.depth.clamp(1, 3);
        let pipeline_start = Instant::now();

        info!(
            query = %input.query,
            depth,
            custom_angles = input.angles.len(),
            "starting deep research pipeline"
        );

        // Find deep-search binary
        let binary = match self.find_deep_search_binary() {
            Some(b) => b,
            None => {
                return Ok(ToolResult {
                    output: "deep-search binary not found. Ensure the deep-search app-skill \
                             is installed in .crew/skills/deep-search/."
                        .into(),
                    success: false,
                    ..Default::default()
                });
            }
        };

        // Phase 0: Planning — decide search angles
        let mut total_tokens = TokenUsage::default();

        let angles = if input.angles.is_empty() {
            info!("planning agent deciding search strategy");
            let (planned, plan_usage) = self.plan_research(&input.query, depth).await;
            total_tokens.input_tokens += plan_usage.input_tokens;
            total_tokens.output_tokens += plan_usage.output_tokens;
            planned
        } else {
            input
                .angles
                .into_iter()
                .map(|r| SearchAngle {
                    query: r.query,
                    depth: r.depth.unwrap_or(depth).clamp(1, 3),
                })
                .collect()
        };

        info!(angle_count = angles.len(), "search angles ready");
        for (i, angle) in angles.iter().enumerate() {
            info!(angle = i, query = %angle.query, depth = angle.depth, "search angle");
        }

        // Phase 1: Parallel search
        let research_dirs = self.search_phase(&binary, &angles).await;

        if research_dirs.is_empty() {
            return Ok(ToolResult {
                output: "All search attempts failed. Check network connectivity and \
                         deep-search binary."
                    .into(),
                success: false,
                ..Default::default()
            });
        }

        let search_duration = pipeline_start.elapsed();
        info!(
            dirs = research_dirs.len(),
            duration_secs = search_duration.as_secs(),
            "search phase complete"
        );

        // Phase 2+3: Extract + Synthesize
        let (report, synth_usage) = self
            .synthesis_phase(&research_dirs, &input.query)
            .await
            .wrap_err("synthesis phase failed")?;

        total_tokens.input_tokens += synth_usage.input_tokens;
        total_tokens.output_tokens += synth_usage.output_tokens;

        // Phase 4: Save report
        let slug = research_utils::slugify(&input.query);
        let report_dir = self.working_dir.join("research").join(&slug);
        tokio::fs::create_dir_all(&report_dir).await.ok();
        let report_path = report_dir.join("_deep_research_report.md");
        tokio::fs::write(&report_path, &report).await.ok();

        let total_duration = pipeline_start.elapsed();
        info!(
            duration_secs = total_duration.as_secs(),
            search_dirs = research_dirs.len(),
            tokens_in = total_tokens.input_tokens,
            tokens_out = total_tokens.output_tokens,
            report_path = %report_path.display(),
            "deep research pipeline complete"
        );

        // Build angle detail for summary
        let angle_detail: String = angles
            .iter()
            .enumerate()
            .map(|(i, a)| format!("  {}. [depth={}] {}", i + 1, a.depth, a.query))
            .collect::<Vec<_>>()
            .join("\n");

        let summary = format!(
            "\n\n---\n\
             Deep research pipeline complete:\n\
             - Search angles ({} total):\n{}\n\
             - Research dirs: {}\n\
             - Search phase: {:.0}s\n\
             - Total time: {:.0}s\n\
             - Tokens: {} input + {} output\n\
             - Report saved: {}",
            angles.len(),
            angle_detail,
            research_dirs.len(),
            search_duration.as_secs_f64(),
            total_duration.as_secs_f64(),
            total_tokens.input_tokens,
            total_tokens.output_tokens,
            report_path.display(),
        );

        Ok(ToolResult {
            output: format!("{report}{summary}"),
            success: true,
            tokens_used: Some(total_tokens),
            ..Default::default()
        })
    }
}

/// Invoke the deep-search binary for a single query angle.
async fn invoke_deep_search(binary: &Path, query: &str, depth: u8, cwd: &Path) -> Result<PathBuf> {
    let args = serde_json::json!({
        "query": query,
        "depth": depth,
        "max_results": 8,
    });

    let mut cmd = tokio::process::Command::new(binary);
    cmd.arg("deep_search")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(cwd);

    let mut child = cmd
        .spawn()
        .wrap_err_with(|| format!("failed to spawn deep-search: {}", binary.display()))?;

    let child_pid = child.id();

    if let Some(mut stdin) = child.stdin.take() {
        let data = serde_json::to_vec(&args)?;
        stdin.write_all(&data).await?;
    }

    // 10 minute timeout. wait_with_output() consumes child, so save PID
    // for killing on timeout to prevent orphaned processes.
    let result = match tokio::time::timeout(
        std::time::Duration::from_secs(600),
        child.wait_with_output(),
    )
    .await
    {
        Ok(result) => result.wrap_err("deep-search execution failed")?,
        Err(_) => {
            #[cfg(unix)]
            if let Some(pid) = child_pid {
                // Kill the process group (deep-search may spawn Chrome, etc.)
                let _ = std::process::Command::new("kill")
                    .args(["-9", &format!("-{pid}")])
                    .status();
                let _ = std::process::Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .status();
            }
            eyre::bail!("deep-search timed out after 600s");
        }
    };

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        eyre::bail!("deep-search exited with {}: {}", result.status, stderr);
    }

    // The deep-search binary saves to ./research/<slug>/ relative to cwd
    let slug = research_utils::slugify(query);
    let research_dir = cwd.join("research").join(&slug);

    if research_dir.is_dir() {
        Ok(research_dir)
    } else {
        // Try to find the directory by scanning ./research/ for recently created dirs
        let research_base = cwd.join("research");
        if research_base.is_dir() {
            // The slug might differ slightly from our slugify — find the newest dir
            let mut entries = tokio::fs::read_dir(&research_base).await?;
            let mut newest: Option<(PathBuf, std::time::SystemTime)> = None;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(meta) = tokio::fs::metadata(&path).await {
                        if let Ok(modified) = meta.modified() {
                            if newest.as_ref().is_none_or(|(_, t)| modified > *t) {
                                newest = Some((path, modified));
                            }
                        }
                    }
                }
            }
            if let Some((path, _)) = newest {
                return Ok(path);
            }
        }

        eyre::bail!(
            "research directory not found after deep-search: {}",
            research_dir.display()
        )
    }
}

/// Deduplicate source files by URL extracted from frontmatter.
fn dedup_by_url(files: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut seen_urls = HashSet::new();
    let mut result = Vec::new();

    for (name, content) in files {
        let url = research_utils::extract_url_from_frontmatter(&content);
        if let Some(ref url) = url {
            if !seen_urls.insert(url.clone()) {
                // Duplicate URL — skip
                continue;
            }
        }
        result.push((name, content));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_by_url() {
        let files = vec![
            (
                "01_example.md".into(),
                "---\nurl: https://example.com/page1\n---\nContent A".into(),
            ),
            (
                "02_example.md".into(),
                "---\nurl: https://example.com/page1\n---\nContent B (duplicate)".into(),
            ),
            (
                "03_other.md".into(),
                "---\nurl: https://other.com/page2\n---\nContent C".into(),
            ),
            ("04_no_url.md".into(), "No frontmatter here".into()),
        ];

        let result = dedup_by_url(files);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, "01_example.md");
        assert_eq!(result[1].0, "03_other.md");
        assert_eq!(result[2].0, "04_no_url.md");
    }

    #[test]
    fn test_dedup_no_duplicates() {
        let files = vec![
            ("01.md".into(), "---\nurl: https://a.com\n---\nA".into()),
            ("02.md".into(), "---\nurl: https://b.com\n---\nB".into()),
        ];

        let result = dedup_by_url(files);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_extract_json_array_direct() {
        let input = r#"[{"query": "test", "depth": 2}]"#;
        assert_eq!(extract_json_array(input), Some(input));
    }

    #[test]
    fn test_extract_json_array_code_fence() {
        let input = "```json\n[{\"query\": \"test\"}]\n```";
        assert_eq!(extract_json_array(input), Some("[{\"query\": \"test\"}]"));
    }

    #[test]
    fn test_extract_json_array_with_preamble() {
        let input = "Here are the angles:\n[{\"query\": \"a\"}, {\"query\": \"b\"}]\nDone.";
        assert_eq!(
            extract_json_array(input),
            Some("[{\"query\": \"a\"}, {\"query\": \"b\"}]")
        );
    }

    #[test]
    fn test_extract_json_array_none() {
        assert_eq!(extract_json_array("no json here"), None);
    }

    #[test]
    fn test_template_angles() {
        let tool = DeepResearchTool::new(
            Arc::new(MockLlm),
            PathBuf::from("/tmp"),
            PathBuf::from("/tmp"),
            vec![],
        );
        let angles = tool.template_angles("AI agents", 2);
        assert_eq!(angles.len(), 5);
        assert_eq!(angles[0].query, "AI agents");
        assert_eq!(angles[0].depth, 2);
        assert_eq!(angles[1].depth, 1); // supplementary = min(1, depth)
        // English query → last angle should be Chinese
        assert!(angles[4].query.contains("技术方案"));
    }

    #[test]
    fn test_template_angles_cjk() {
        let tool = DeepResearchTool::new(
            Arc::new(MockLlm),
            PathBuf::from("/tmp"),
            PathBuf::from("/tmp"),
            vec![],
        );
        let angles = tool.template_angles("人工智能代理框架对比", 2);
        assert_eq!(angles.len(), 5);
        // CJK query → last angle should be English
        assert!(angles[4].query.contains("English"));
    }

    #[test]
    fn test_find_binary_from_empty_dirs() {
        let tool = DeepResearchTool::new(
            Arc::new(MockLlm),
            PathBuf::from("/tmp"),
            PathBuf::from("/tmp"),
            vec![],
        );
        assert!(tool.find_deep_search_binary().is_none());
    }

    // Minimal mock for struct tests
    struct MockLlm;

    #[async_trait::async_trait]
    impl LlmProvider for MockLlm {
        async fn chat(
            &self,
            _messages: &[crew_core::Message],
            _tools: &[crew_llm::ToolSpec],
            _config: &ChatConfig,
        ) -> Result<crew_llm::ChatResponse> {
            Ok(crew_llm::ChatResponse {
                content: Some("mock".into()),
                reasoning_content: None,
                tool_calls: vec![],
                stop_reason: crew_llm::StopReason::EndTurn,
                usage: crew_llm::TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    ..Default::default()
                },
            })
        }

        fn model_id(&self) -> &str {
            "mock"
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }
}
