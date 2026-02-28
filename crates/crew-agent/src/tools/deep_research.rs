//! Deep research tool: spawns parallel sub-agents that autonomously search, crawl,
//! and synthesize a comprehensive report.
//!
//! Flow: split question → parallel researcher agents → merge partial reports.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use crew_core::{AgentId, Message, MessageRole, Task, TaskContext, TaskKind, TokenUsage};
use crew_llm::{ChatConfig, LlmProvider};
use crew_memory::EpisodeStore;
use eyre::{Result, WrapErr};
use serde::Deserialize;
use tracing::{info, warn};

use super::{Tool, ToolPolicy, ToolRegistry, ToolResult};
use crate::Agent;
use crate::agent::AgentConfig;

/// Notification sent when a background research task completes.
pub struct ResearchNotification {
    pub question: String,
    pub report_path: PathBuf,
    pub success: bool,
    pub summary: String,
}

/// Tool that spawns parallel research sub-agents to investigate a question.
pub struct DeepResearchTool {
    llm: Arc<dyn LlmProvider>,
    memory: Arc<EpisodeStore>,
    /// Global data directory (~/.crew/), used for research output.
    data_dir: PathBuf,
    /// Channel for background completion notifications.
    notify_tx: tokio::sync::mpsc::Sender<ResearchNotification>,
    worker_count: AtomicU32,
    config: Option<Arc<super::tool_config::ToolConfigStore>>,
}

impl DeepResearchTool {
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        memory: Arc<EpisodeStore>,
        data_dir: impl Into<PathBuf>,
        notify_tx: tokio::sync::mpsc::Sender<ResearchNotification>,
    ) -> Self {
        Self {
            llm,
            memory,
            data_dir: data_dir.into(),
            notify_tx,
            worker_count: AtomicU32::new(0),
            config: None,
        }
    }

    pub fn with_config(mut self, config: Arc<super::tool_config::ToolConfigStore>) -> Self {
        self.config = Some(config);
        self
    }

    /// Split a research question into independent sub-questions using an LLM call.
    async fn split_question(&self, question: &str) -> Result<Vec<String>> {
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You split research questions into sub-questions. \
                          Respond with ONLY a JSON array of short strings. No other text."
                    .into(),
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                timestamp: chrono::Utc::now(),
            },
            Message {
                role: MessageRole::User,
                content: format!(
                    "Split into 3 short, independent sub-questions for parallel research.\n\n\
                     Question: {question}\n\n\
                     Rules:\n\
                     - Exactly 3 sub-questions\n\
                     - Each sub-question must be under 20 words\n\
                     - Return ONLY a JSON array, nothing else\n\n\
                     Example: [\"NVIDIA 2025 AI revenue and market share\", \
                     \"AMD Intel Broadcom 2025 AI chip revenue comparison\", \
                     \"Global AI chip market size and competitive landscape 2025\"]"
                ),
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                timestamp: chrono::Utc::now(),
            },
        ];

        let config = ChatConfig {
            max_tokens: Some(4096),
            temperature: Some(0.3),
            ..Default::default()
        };

        let response = self.llm.chat(&messages, &[], &config).await?;

        // Try content first, then reasoning_content (some models put output there)
        let text = match response.content.as_deref() {
            Some(c) if !c.trim().is_empty() => c.to_string(),
            _ => response.reasoning_content.clone().unwrap_or_default(),
        };

        if text.trim().is_empty() {
            eyre::bail!("LLM returned empty response for question splitting");
        }

        // Extract JSON array from response (handle markdown code blocks, truncation)
        let json_str = extract_json_array(&text).unwrap_or_else(|| text.clone());
        let questions: Vec<String> = serde_json::from_str(&json_str)
            .wrap_err_with(|| format!("failed to parse sub-questions from: {text}"))?;

        // Clamp to 2-3 sub-questions — fewer agents with full budgets produce better results
        let n = questions.len().clamp(2, 3);
        Ok(questions.into_iter().take(n).collect())
    }

    /// Synthesize multiple partial reports into a final merged report.
    async fn synthesize_reports(
        &self,
        question: &str,
        partials: &[(String, String)], // (sub_question, report_content)
    ) -> Result<(String, TokenUsage)> {
        let mut sections = String::new();
        for (i, (sub_q, content)) in partials.iter().enumerate() {
            sections.push_str(&format!(
                "## Partial Report {} — {}\n\n{}\n\n---\n\n",
                i + 1,
                sub_q,
                content
            ));
        }

        let prompt = format!(
            "You are writing the final research report. Merge these {count} partial reports \
             into one comprehensive, well-structured report.\n\n\
             - Remove duplicates and redundancies\n\
             - Organize logically with clear section headers\n\
             - Keep ALL specific numbers, percentages, dates, and citations\n\
             - Use markdown tables where appropriate\n\
             - Include a Sources section at the end with all URLs\n\n\
             Original question: {question}\n\n{sections}\n\n\
             Write the complete merged report in markdown.",
            count = partials.len(),
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
            max_tokens: Some(8192),
            temperature: Some(0.0),
            ..Default::default()
        };

        let response = self.llm.chat(&messages, &[], &config).await?;
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        };
        Ok((response.content.unwrap_or_default(), usage))
    }

    /// Run parallel research: split → fan-out → merge.
    async fn execute_parallel(
        &self,
        question: &str,
        max_iter: u32,
        research_dir: &Path,
        report_path: &Path,
        worker_num: u32,
        custom_prompt: Option<&str>,
    ) -> Result<ToolResult> {
        // Step 1: Split question
        info!(question, "splitting question into sub-questions");
        let (sub_questions, split_usage) = match self.split_question(question).await {
            Ok(qs) => {
                info!(count = qs.len(), questions = ?qs, "split into sub-questions");
                (qs, TokenUsage::default())
            }
            Err(e) => {
                warn!(error = %e, "failed to split question, falling back to single agent");
                return self
                    .execute_single(
                        question,
                        max_iter,
                        research_dir,
                        report_path,
                        worker_num,
                        custom_prompt,
                    )
                    .await;
            }
        };

        let n = sub_questions.len() as u32;
        // Each parallel agent gets the full iteration budget — they run concurrently,
        // so total wall-clock time stays similar to a single agent.
        let per_agent_iter = max_iter.max(10);
        let researcher_prompt = match custom_prompt {
            Some(p) => p.to_string(),
            None => include_str!("../prompts/researcher.txt").to_string(),
        };

        // Step 2: Spawn parallel sub-agents
        info!(
            count = n,
            per_agent_iter, "spawning parallel research sub-agents"
        );

        let mut handles = Vec::new();
        for (i, sub_q) in sub_questions.iter().enumerate() {
            let llm = self.llm.clone();
            let memory = self.memory.clone();
            let res_dir = research_dir.to_path_buf();
            let partial_path = research_dir.join(format!("partial_{i}.md"));
            let prompt = researcher_prompt.clone();
            let question = sub_q.clone();

            // Use tokio::spawn to give each agent its own stack (avoids stack overflow)
            let handle = tokio::spawn(async move {
                let tools = build_research_tools(&res_dir);
                let agent_id = AgentId::new(format!("researcher-{worker_num}-{i}"));

                info!(
                    %agent_id,
                    question = %question,
                    "starting research sub-agent"
                );

                let worker = Agent::new(agent_id.clone(), llm, tools, memory)
                    .with_config(AgentConfig {
                        max_iterations: per_agent_iter,
                        max_timeout: Some(Duration::from_secs(600)),
                        save_episodes: false,
                        ..Default::default()
                    })
                    .with_system_prompt(prompt);

                let subtask = Task::new(
                    TaskKind::Code {
                        instruction: format!(
                            "Research this specific question and write a focused report.\n\n\
                             Question: {question}\n\n\
                             Save the report to: {}\n\n\
                             Include specific numbers, data, and cite sources with URLs.",
                            partial_path.file_name().unwrap().to_string_lossy()
                        ),
                        files: vec![],
                    },
                    TaskContext {
                        working_dir: res_dir,
                        ..Default::default()
                    },
                );

                let result = worker.run_task(&subtask).await;
                (i, question, result, partial_path)
            });
            handles.push(handle);
        }

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok()) // filter out JoinErrors
            .collect();

        // Step 3: Collect partial reports
        let mut partials = Vec::new();
        let mut total_tokens = split_usage;
        let mut all_succeeded = true;

        for (i, sub_q, result, partial_path) in &results {
            match result {
                Ok(r) => {
                    total_tokens.input_tokens += r.token_usage.input_tokens;
                    total_tokens.output_tokens += r.token_usage.output_tokens;

                    // Read from disk (preferred) or use agent output
                    let content = if partial_path.exists() {
                        tokio::fs::read_to_string(partial_path)
                            .await
                            .unwrap_or_else(|_| r.output.clone())
                    } else {
                        r.output.clone()
                    };

                    if !content.is_empty() {
                        partials.push((sub_q.clone(), content));
                    }

                    info!(agent = i, success = r.success, "sub-agent completed");
                }
                Err(e) => {
                    warn!(agent = i, error = %e, "sub-agent failed");
                    all_succeeded = false;
                }
            }
        }

        if partials.is_empty() {
            return Ok(ToolResult {
                output: "All research sub-agents failed to produce results.".into(),
                success: false,
                tokens_used: Some(total_tokens),
                ..Default::default()
            });
        }

        // Step 4: Synthesize into final report
        info!(
            partial_count = partials.len(),
            "synthesizing final report from partial reports"
        );

        let (final_report, synth_tokens) = self.synthesize_reports(question, &partials).await?;

        total_tokens.input_tokens += synth_tokens.input_tokens;
        total_tokens.output_tokens += synth_tokens.output_tokens;

        // Write final report
        tokio::fs::write(report_path, &final_report)
            .await
            .wrap_err("failed to write final report")?;

        let summary = format!(
            "{}\n\n---\n_Parallel research: {} sub-agents, {} partial reports merged. Saved to: {}_",
            final_report,
            results.len(),
            partials.len(),
            report_path.display()
        );

        Ok(ToolResult {
            output: summary,
            success: all_succeeded,
            tokens_used: Some(total_tokens),
            ..Default::default()
        })
    }

    /// Single-agent fallback (original behavior).
    async fn execute_single(
        &self,
        question: &str,
        max_iter: u32,
        research_dir: &Path,
        report_path: &Path,
        worker_num: u32,
        custom_prompt: Option<&str>,
    ) -> Result<ToolResult> {
        let worker_id = AgentId::new(format!("researcher-{worker_num}"));
        let tools = build_research_tools(research_dir);
        let system_prompt = match custom_prompt {
            Some(p) => p.to_string(),
            None => include_str!("../prompts/researcher.txt").to_string(),
        };

        let config = AgentConfig {
            max_iterations: max_iter,
            max_timeout: Some(Duration::from_secs(900)),
            save_episodes: false,
            ..Default::default()
        };

        let worker = Agent::new(worker_id, self.llm.clone(), tools, self.memory.clone())
            .with_config(config)
            .with_system_prompt(system_prompt);

        let task_prompt = format!(
            "Research the following question thoroughly and write a comprehensive report.\n\n\
             Question: {question}\n\n\
             Save the final report to: {}\n\n\
             Search from at least 2 different angles. Include tables with specific numbers. Cite all sources with URLs.",
            report_path.file_name().unwrap().to_string_lossy()
        );

        let subtask = Task::new(
            TaskKind::Code {
                instruction: task_prompt,
                files: vec![],
            },
            TaskContext {
                working_dir: research_dir.to_path_buf(),
                ..Default::default()
            },
        );

        let result = worker.run_task(&subtask).await;

        match result {
            Ok(r) => {
                let report = if report_path.exists() {
                    tokio::fs::read_to_string(report_path)
                        .await
                        .unwrap_or(r.output.clone())
                } else {
                    r.output.clone()
                };

                let summary = format!(
                    "{}\n\n---\n_Research completed. Report saved to: {}_",
                    report,
                    report_path.display()
                );

                Ok(ToolResult {
                    output: summary,
                    success: r.success,
                    tokens_used: Some(r.token_usage),
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Research sub-agent failed: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

#[derive(Deserialize)]
struct Input {
    question: String,
    #[serde(default)]
    max_iterations: Option<u32>,
    #[serde(default)]
    parallel: Option<bool>,
    #[serde(default)]
    mode: Option<String>,
    /// Optional custom system prompt for the researcher sub-agents.
    /// When provided, replaces the default researcher.txt prompt.
    /// Use this to give sub-agents domain-specific instructions.
    #[serde(default)]
    system_prompt: Option<String>,
}

#[async_trait]
impl Tool for DeepResearchTool {
    fn name(&self) -> &str {
        "deep_research"
    }

    fn description(&self) -> &str {
        "Spawn autonomous research sub-agents that search the web from multiple angles in parallel, \
         crawl sources, and synthesize a comprehensive report with tables and citations. \
         Default mode is 'background': returns immediately so the user can continue chatting, \
         and notifies when the report is ready. Use 'sync' mode only when the user needs the \
         report content inline. Use this for any research question that needs thorough investigation. \
         You can provide a custom system_prompt to give sub-agents domain-specific instructions \
         (e.g. focus on academic papers, use specific terminology, crawl certain sites)."
    }

    fn tags(&self) -> &[&str] {
        &["web"]
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question to investigate"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Total max agent iterations across all sub-agents (default: 25)"
                },
                "parallel": {
                    "type": "boolean",
                    "description": "Run multiple sub-agents in parallel (default: true). Set false for simple questions."
                },
                "mode": {
                    "type": "string",
                    "enum": ["background", "sync"],
                    "description": "background (default): return immediately, notify when done. sync: wait for report."
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Custom system prompt for researcher sub-agents. Overrides the default researcher instructions. Use to give domain-specific guidance (e.g. 'Focus on academic papers from arxiv', 'Only use Chinese-language sources', 'Crawl the official docs at https://example.com/docs/')."
                }
            },
            "required": ["question"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid deep_research input")?;

        let (cfg_max_iter, cfg_parallel, cfg_mode) = match &self.config {
            Some(c) => (
                c.get_u64("deep_research", "max_iterations")
                    .await
                    .map(|v| v as u32),
                c.get_bool("deep_research", "parallel").await,
                c.get_str("deep_research", "mode").await,
            ),
            None => (None, None, None),
        };

        let max_iter = input
            .max_iterations
            .or(cfg_max_iter)
            .unwrap_or(25)
            .clamp(5, 50);
        let parallel = input.parallel.or(cfg_parallel).unwrap_or(true);
        let mode = input
            .mode
            .or(cfg_mode)
            .unwrap_or_else(|| "background".into());

        let worker_num = self.worker_count.fetch_add(1, Ordering::SeqCst);
        let background = mode == "background";

        // Build output path
        let slug = slugify(&input.question);
        let research_dir = self.data_dir.join("research").join(&slug);
        tokio::fs::create_dir_all(&research_dir)
            .await
            .wrap_err("failed to create research directory")?;
        let report_path = research_dir.join("report.md");

        info!(
            question = %input.question,
            max_iterations = max_iter,
            parallel = parallel,
            mode = %mode,
            "starting deep research"
        );

        if background {
            // Background mode: spawn and return immediately
            let llm = self.llm.clone();
            let memory = self.memory.clone();
            let question = input.question.clone();
            let bg_parallel = parallel;
            let notify_tx = self.notify_tx.clone();
            let data_dir = self.data_dir.clone();
            let rp = report_path.clone();
            let custom_prompt = input.system_prompt.clone();

            tokio::spawn(async move {
                // Build a temporary DeepResearchTool for the background task.
                // We use a dummy notify_tx since this IS the background task.
                let (dummy_tx, _) = tokio::sync::mpsc::channel(1);
                let tool = DeepResearchTool {
                    llm,
                    memory,
                    data_dir,
                    notify_tx: dummy_tx,
                    worker_count: AtomicU32::new(worker_num),
                    config: None,
                };

                let result = if bg_parallel {
                    tool.execute_parallel(
                        &question,
                        max_iter,
                        &research_dir,
                        &rp,
                        worker_num,
                        custom_prompt.as_deref(),
                    )
                    .await
                } else {
                    tool.execute_single(
                        &question,
                        max_iter,
                        &research_dir,
                        &rp,
                        worker_num,
                        custom_prompt.as_deref(),
                    )
                    .await
                };

                let (success, summary) = match result {
                    Ok(r) => (r.success, format!("Report saved to: {}", rp.display())),
                    Err(e) => (false, format!("Research failed: {e}")),
                };

                let _ = notify_tx
                    .send(ResearchNotification {
                        question,
                        report_path: rp,
                        success,
                        summary,
                    })
                    .await;
            });

            Ok(ToolResult {
                output: format!(
                    "Research started in background. Report will be saved to: {}\n\
                     You can continue chatting — I'll notify you when it's ready.",
                    report_path.display()
                ),
                success: true,
                ..Default::default()
            })
        } else {
            // Sync mode: wait for result
            let custom_prompt = input.system_prompt.as_deref();
            if parallel {
                self.execute_parallel(
                    &input.question,
                    max_iter,
                    &research_dir,
                    &report_path,
                    worker_num,
                    custom_prompt,
                )
                .await
            } else {
                self.execute_single(
                    &input.question,
                    max_iter,
                    &research_dir,
                    &report_path,
                    worker_num,
                    custom_prompt,
                )
                .await
            }
        }
    }
}

/// Build the curated tool registry for research sub-agents.
fn build_research_tools(cwd: &Path) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(super::WebSearchTool::new());
    tools.register(super::WebFetchTool::new());
    tools.register(super::BrowserTool::new());
    tools.register(super::DeepCrawlTool::new(cwd));
    tools.register(super::DeepSearchTool::new(cwd));
    tools.register(super::ReadFileTool::new(cwd));
    tools.register(super::WriteFileTool::new(cwd));
    tools.register(super::GlobTool::new(cwd));
    tools.register(super::GrepTool::new(cwd));

    let policy = ToolPolicy {
        deny: vec![
            "shell".into(),
            "spawn".into(),
            "edit_file".into(),
            "diff_edit".into(),
            "deep_research".into(),
        ],
        ..Default::default()
    };
    tools.apply_policy(&policy);
    tools
}

/// Extract a JSON array from text that may contain markdown code blocks.
/// Also handles truncated JSON by recovering complete elements.
fn extract_json_array(text: &str) -> Option<String> {
    let raw = extract_json_array_raw(text)?;

    // Try to parse as-is first
    if serde_json::from_str::<Vec<String>>(raw).is_ok() {
        return Some(raw.to_string());
    }

    // Try to recover truncated JSON array: find last complete string element
    if raw.starts_with('[') {
        let mut depth = 0;
        let mut in_string = false;
        let mut escape = false;
        let mut last_good_end = None;

        for (i, ch) in raw.char_indices() {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                '[' if !in_string => depth += 1,
                ']' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(raw[..=i].to_string());
                    }
                }
                ',' if !in_string && depth == 1 => {
                    // Mark position after a complete element
                    last_good_end = Some(i);
                }
                _ => {}
            }
        }

        // Truncated: close the array after the last complete element
        if let Some(end) = last_good_end {
            let mut recovered = raw[..end].to_string();
            recovered.push(']');
            if serde_json::from_str::<Vec<String>>(&recovered).is_ok() {
                return Some(recovered);
            }
        }
    }

    Some(raw.to_string())
}

fn extract_json_array_raw(text: &str) -> Option<&str> {
    // Try to find ```json ... ``` block
    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            return Some(after[..end].trim());
        }
    }
    // Try to find ``` ... ``` block
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            return Some(after[..end].trim());
        }
    }
    // Try to find bare [ ... ]
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return Some(&text[start..=end]);
            }
        }
        // No closing bracket — return from [ to end (for truncation recovery)
        return Some(&text[start..]);
    }
    None
}

/// Convert a query string to a filesystem-safe slug.
fn slugify(s: &str) -> String {
    let mut slug = String::with_capacity(s.len());
    for ch in s.chars().take(60) {
        if ch.is_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if ch == ' ' || ch == '-' || ch == '_' {
            if !slug.ends_with('-') {
                slug.push('-');
            }
        }
    }
    slug.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("top AI startups 2025"), "top-ai-startups-2025");
        assert_eq!(slugify("What are the best GPUs?"), "what-are-the-best-gpus");
    }

    #[test]
    fn test_extract_json_array() {
        // Bare JSON
        let text = r#"["question 1", "question 2", "question 3"]"#;
        assert_eq!(extract_json_array(text).unwrap(), text);

        // Markdown code block
        let text = "Here are the questions:\n```json\n[\"q1\", \"q2\"]\n```\n";
        assert_eq!(extract_json_array(text).unwrap(), "[\"q1\", \"q2\"]");

        // With surrounding text
        let text = "Sure! Here: [\"a\", \"b\"] done.";
        assert_eq!(extract_json_array(text).unwrap(), "[\"a\", \"b\"]");

        // No array
        assert_eq!(extract_json_array("no array here"), None);

        // Truncated JSON — should recover complete elements
        let text = r#"["question 1", "question 2", "question 3 is trun"#;
        let recovered = extract_json_array(text).unwrap();
        let parsed: Vec<String> = serde_json::from_str(&recovered).unwrap();
        assert_eq!(parsed, vec!["question 1", "question 2"]);
    }
}
