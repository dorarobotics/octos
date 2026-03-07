//! Agent implementation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crew_core::{AgentId, Message, MessageRole, Task, TaskResult, TokenUsage};
use crew_llm::{
    ChatConfig, ChatResponse, ChatStream, EmbeddingProvider, LlmProvider, StopReason, StreamEvent,
    ToolSpec,
};
use crew_memory::{Episode, EpisodeOutcome, EpisodeStore};
use eyre::Result;
use futures::StreamExt;
use tracing::{Instrument, debug, info, info_span, warn};

use crate::hooks::{HookContext, HookEvent, HookExecutor, HookPayload, HookResult};
use crate::loop_detect::LoopDetector;
use crate::progress::{ProgressEvent, ProgressReporter, SilentReporter};
use crate::tools::ToolRegistry;

/// Compiled-in default worker prompt (from `prompts/worker.txt`).
pub const DEFAULT_WORKER_PROMPT: &str = include_str!("prompts/worker.txt");

/// Configuration for agent execution.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of iterations before stopping.
    pub max_iterations: u32,
    /// Maximum total tokens (input + output) before stopping. None = unlimited.
    pub max_tokens: Option<u32>,
    /// Wall-clock timeout for the entire agent run. None = unlimited.
    pub max_timeout: Option<Duration>,
    /// Whether to save episodes to memory.
    pub save_episodes: bool,
    /// Optional worker system prompt override (used by Agent::new as the default prompt).
    /// When None, falls back to the compiled-in prompts/worker.txt.
    pub worker_prompt: Option<String>,
    /// Maximum seconds for all parallel tool calls to complete. Default: 300.
    pub tool_timeout_secs: u64,
}

/// Default tool execution timeout in seconds.
pub const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 300;
/// Default session processing timeout in seconds.
pub const DEFAULT_SESSION_TIMEOUT_SECS: u64 = 600;

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_tokens: None,
            max_timeout: Some(Duration::from_secs(600)),
            save_episodes: true,
            worker_prompt: None,
            tool_timeout_secs: DEFAULT_TOOL_TIMEOUT_SECS,
        }
    }
}

/// Response from conversation mode (process_message).
#[derive(Debug, Clone)]
pub struct ConversationResponse {
    pub content: String,
    pub token_usage: TokenUsage,
    pub files_modified: Vec<PathBuf>,
    pub streamed: bool,
}

/// Shared atomic counters for real-time token tracking (used by status indicators).
pub struct TokenTracker {
    pub input_tokens: AtomicU32,
    pub output_tokens: AtomicU32,
}

impl TokenTracker {
    pub fn new() -> Self {
        Self {
            input_tokens: AtomicU32::new(0),
            output_tokens: AtomicU32::new(0),
        }
    }
}

impl Default for TokenTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Reason why the agent loop stopped due to budget constraints.
enum BudgetStop {
    Shutdown,
    MaxIterations,
    MaxTokens { used: u32, limit: u32 },
    WallClockTimeout { limit: Duration },
}

impl BudgetStop {
    fn message(&self) -> String {
        match self {
            Self::Shutdown => "Interrupted.".into(),
            Self::MaxIterations => "Reached max iterations.".into(),
            Self::MaxTokens { used, limit } => {
                format!("Token budget exceeded ({used} of {limit}).")
            }
            Self::WallClockTimeout { limit } => {
                format!("Wall-clock timeout ({:.0}s limit).", limit.as_secs_f64())
            }
        }
    }
}

/// An agent that can execute tasks.
pub struct Agent {
    /// Unique identifier for this agent.
    pub id: AgentId,
    /// LLM provider for generating responses.
    llm: Arc<dyn LlmProvider>,
    /// Tool registry for executing tool calls (Arc for sharing with spawned tool tasks).
    tools: Arc<ToolRegistry>,
    /// Episode store for memory.
    memory: Arc<EpisodeStore>,
    /// Embedding provider for hybrid memory search.
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    /// System prompt for this agent (RwLock for hot-reload support).
    system_prompt: RwLock<String>,
    /// Agent configuration.
    config: AgentConfig,
    /// Progress reporter.
    reporter: Arc<dyn ProgressReporter>,
    /// Lifecycle hooks executor.
    hooks: Option<Arc<HookExecutor>>,
    /// Session-level context for hook payloads.
    hook_context: std::sync::Mutex<Option<HookContext>>,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
}

impl Agent {
    /// Create a new agent.
    pub fn new(
        id: AgentId,
        llm: Arc<dyn LlmProvider>,
        tools: ToolRegistry,
        memory: Arc<EpisodeStore>,
    ) -> Self {
        let system_prompt = include_str!("prompts/worker.txt").to_string();

        Self {
            id,
            llm,
            tools: Arc::new(tools),
            memory,
            embedder: None,
            system_prompt: RwLock::new(system_prompt),
            config: AgentConfig::default(),
            reporter: Arc::new(SilentReporter),
            hooks: None,
            hook_context: std::sync::Mutex::new(None),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the agent configuration.
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        // Apply worker_prompt override if provided
        if let Some(ref wp) = config.worker_prompt {
            *self
                .system_prompt
                .write()
                .unwrap_or_else(|e| e.into_inner()) = wp.clone();
        }
        self.config = config;
        self
    }

    /// Set the progress reporter.
    pub fn with_reporter(mut self, reporter: Arc<dyn ProgressReporter>) -> Self {
        self.reporter = reporter;
        self
    }

    /// Set the shutdown signal.
    pub fn with_shutdown(mut self, shutdown: Arc<AtomicBool>) -> Self {
        self.shutdown = shutdown;
        self
    }

    /// Set the embedding provider for hybrid memory search.
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set lifecycle hooks executor.
    pub fn with_hooks(mut self, hooks: Arc<HookExecutor>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set session-level context for hook payloads.
    pub fn with_hook_context(self, ctx: HookContext) -> Self {
        *self.hook_context.lock().unwrap_or_else(|e| e.into_inner()) = Some(ctx);
        self
    }

    /// Update the session ID in the hook context (call before each message).
    pub fn set_session_id(&self, session_id: &str) {
        let mut guard = self.hook_context.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ref mut ctx) = *guard {
            ctx.session_id = Some(session_id.to_string());
        }
    }

    /// Get a snapshot of the current hook context.
    fn hook_ctx(&self) -> Option<HookContext> {
        self.hook_context
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Override the system prompt (e.g. for gateway mode).
    pub fn with_system_prompt(self, prompt: String) -> Self {
        *self.system_prompt.write().unwrap_or_else(|e| {
            tracing::warn!("system prompt lock was poisoned, recovering");
            e.into_inner()
        }) = prompt;
        self
    }

    /// Append additional content to the current system prompt (e.g. bootstrap files).
    pub fn append_system_prompt(&self, extra: &str) {
        let mut guard = self.system_prompt.write().unwrap_or_else(|e| {
            tracing::warn!("system prompt lock was poisoned, recovering");
            e.into_inner()
        });
        guard.push_str("\n\n");
        guard.push_str(extra);
    }

    /// Update the system prompt at runtime (hot-reload).
    pub fn set_system_prompt(&self, prompt: String) {
        *self.system_prompt.write().unwrap_or_else(|e| {
            tracing::warn!("system prompt lock was poisoned, recovering");
            e.into_inner()
        }) = prompt;
    }

    /// The LLM model ID in use.
    pub fn model_id(&self) -> &str {
        self.llm.model_id()
    }

    /// The LLM provider name in use.
    pub fn provider_name(&self) -> &str {
        self.llm.provider_name()
    }

    /// Process a single message in conversation mode (chat/gateway).
    /// Takes the user's message, conversation history, and optional media paths.
    pub async fn process_message(
        &self,
        user_content: &str,
        history: &[Message],
        media: Vec<String>,
    ) -> Result<ConversationResponse> {
        self.process_message_inner(user_content, history, media, None)
            .await
    }

    /// Like `process_message`, but updates a `TokenTracker` in real-time after each LLM call.
    /// Used by the gateway status indicator to show live token counts.
    pub async fn process_message_tracked(
        &self,
        user_content: &str,
        history: &[Message],
        media: Vec<String>,
        tracker: &TokenTracker,
    ) -> Result<ConversationResponse> {
        self.process_message_inner(user_content, history, media, Some(tracker))
            .await
    }

    async fn process_message_inner(
        &self,
        user_content: &str,
        history: &[Message],
        media: Vec<String>,
        tracker: Option<&TokenTracker>,
    ) -> Result<ConversationResponse> {
        let mut messages = vec![Message {
            role: MessageRole::System,
            content: self
                .system_prompt
                .read()
                .unwrap_or_else(|e| {
                    tracing::warn!("system prompt lock was poisoned, recovering");
                    e.into_inner()
                })
                .clone(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        }];

        messages.extend_from_slice(history);

        let content = if user_content.is_empty() && !media.is_empty() {
            "[User sent an image]".to_string()
        } else {
            user_content.to_string()
        };

        messages.push(Message {
            role: MessageRole::User,
            content,
            media,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        });

        let config = ChatConfig::default();
        let mut total_usage = TokenUsage::default();
        let mut files_modified = Vec::new();
        let mut iteration = 0u32;
        let start = Instant::now();
        let mut loop_detector = LoopDetector::new(12);

        loop {
            if let Some(stop) = self.check_budget(iteration, start, &total_usage) {
                return Ok(ConversationResponse {
                    content: stop.message(),
                    token_usage: total_usage,
                    files_modified,
                    streamed: false,
                });
            }

            iteration += 1;
            let tools_spec = self.tools.specs();
            self.trim_to_context_window(&mut messages);

            tracing::info!(
                iteration,
                messages = messages.len(),
                tools = tools_spec.len(),
                message_bytes = messages.iter().map(|m| m.content.len()).sum::<usize>(),
                "calling LLM"
            );
            let (response, streamed) = self
                .call_llm_with_hooks(&messages, &tools_spec, &config, iteration, &total_usage)
                .await?;
            {
                let tool_names: Vec<&str> = response
                    .tool_calls
                    .iter()
                    .map(|tc| tc.name.as_str())
                    .collect();
                let tool_ids: Vec<&str> = response
                    .tool_calls
                    .iter()
                    .map(|tc| tc.id.as_str())
                    .collect();
                tracing::info!(
                    iteration,
                    stop_reason = ?response.stop_reason,
                    tool_calls = response.tool_calls.len(),
                    tool_names = %tool_names.join(", "),
                    tool_ids = %tool_ids.join(", "),
                    response_content_len = response.content.as_ref().map(|c| c.len()).unwrap_or(0),
                    input_tokens = response.usage.input_tokens,
                    output_tokens = response.usage.output_tokens,
                    "LLM response received"
                );
            }
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;
            if let Some(t) = tracker {
                t.input_tokens
                    .store(total_usage.input_tokens, Ordering::Relaxed);
                t.output_tokens
                    .store(total_usage.output_tokens, Ordering::Relaxed);
            }

            match response.stop_reason {
                StopReason::EndTurn | StopReason::StopSequence => {
                    self.emit_cost_update(&total_usage, &response.usage);
                    return Ok(ConversationResponse {
                        content: response.content.unwrap_or_default(),
                        token_usage: total_usage,
                        files_modified,
                        streamed,
                    });
                }
                StopReason::ToolUse => {
                    // Check for loop detection before executing
                    for tc in &response.tool_calls {
                        if let Some(warning) = loop_detector.record(&tc.name, &tc.arguments) {
                            warn!("loop detected in tool calls");
                            messages.push(Message {
                                role: MessageRole::System,
                                content: warning,
                                media: vec![],
                                tool_calls: None,
                                tool_call_id: None,
                                reasoning_content: None,
                                timestamp: chrono::Utc::now(),
                            });
                        }
                    }
                    self.handle_tool_use(
                        &response,
                        &mut messages,
                        &mut files_modified,
                        &mut total_usage,
                        tracker,
                    )
                    .await?;
                }
                StopReason::MaxTokens => {
                    self.emit_cost_update(&total_usage, &response.usage);
                    return Ok(ConversationResponse {
                        content: response.content.unwrap_or_default(),
                        token_usage: total_usage,
                        files_modified,
                        streamed,
                    });
                }
            }
        }
    }

    /// Run a task to completion (used by spawn tool).
    pub async fn run_task(&self, task: &Task) -> Result<TaskResult> {
        let task_start = Instant::now();
        let span = info_span!(
            "task",
            task_id = %task.id,
            agent_id = %self.id,
        );

        async {
            info!("starting task");
            self.reporter.report(ProgressEvent::TaskStarted {
                task_id: task.id.to_string(),
            });

            let mut iteration = 0u32;
            let mut messages = self.build_initial_messages(task).await;
            let mut total_usage = TokenUsage::default();
            let mut files_modified = Vec::new();
            let config = ChatConfig::default();

            loop {
                if let Some(stop) = self.check_budget(iteration, task_start, &total_usage) {
                    self.report_budget_stop(&stop, iteration);
                    return Ok(TaskResult {
                        success: false,
                        output: stop.message(),
                        files_modified,
                        subtasks: Vec::new(),
                        token_usage: total_usage,
                    });
                }

                iteration += 1;
                let iter_start = Instant::now();
                self.reporter.report(ProgressEvent::Thinking { iteration });

                let tools_spec = self.tools.specs();
                self.trim_to_context_window(&mut messages);

                let (response, _streamed) = self
                    .call_llm_with_hooks(&messages, &tools_spec, &config, iteration, &total_usage)
                    .await?;
                total_usage.input_tokens += response.usage.input_tokens;
                total_usage.output_tokens += response.usage.output_tokens;

                debug!(
                    iteration,
                    input_tokens = response.usage.input_tokens,
                    output_tokens = response.usage.output_tokens,
                    stop_reason = ?response.stop_reason,
                    duration_ms = iter_start.elapsed().as_millis() as u64,
                    "llm response"
                );

                match response.stop_reason {
                    StopReason::EndTurn | StopReason::StopSequence => {
                        if self.config.save_episodes {
                            let summary = response.content.clone().unwrap_or_default();
                            let summary_truncated = crew_core::truncated_utf8(&summary, 500, "...");

                            let mut episode = Episode::new(
                                task.id.clone(),
                                self.id.clone(),
                                task.context.working_dir.clone(),
                                summary_truncated.clone(),
                                EpisodeOutcome::Success,
                            );
                            episode.files_modified = files_modified.clone();
                            let ep_id = episode.id.clone();

                            if let Err(e) = self.memory.store(episode).await {
                                warn!(error = %e, "failed to save episode to memory");
                            }

                            // Fire-and-forget: embed summary and store embedding
                            if let Some(ref embedder) = self.embedder {
                                let embedder = embedder.clone();
                                let memory = self.memory.clone();
                                let summary_text = summary_truncated;
                                let episode_id = ep_id;
                                tokio::spawn(async move {
                                    match embedder.embed(&[&summary_text]).await {
                                        Ok(vecs) => {
                                            if let Some(vec) = vecs.into_iter().next() {
                                                if let Err(e) =
                                                    memory.store_embedding(&episode_id, vec).await
                                                {
                                                    warn!(error = %e, "failed to store embedding");
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!(
                                                error = %e,
                                                episode_id = %episode_id,
                                                "failed to generate embedding for episode"
                                            );
                                        }
                                    }
                                });
                            }
                        }

                        self.emit_cost_update(&total_usage, &response.usage);
                        self.reporter.report(ProgressEvent::TaskCompleted {
                            success: true,
                            iterations: iteration,
                            duration: task_start.elapsed(),
                        });

                        info!(
                            total_input_tokens = total_usage.input_tokens,
                            total_output_tokens = total_usage.output_tokens,
                            iterations = iteration,
                            files_modified = files_modified.len(),
                            duration_ms = task_start.elapsed().as_millis() as u64,
                            "task completed"
                        );
                        return Ok(self.build_result(&response, total_usage, files_modified));
                    }
                    StopReason::ToolUse => {
                        self.handle_tool_use(
                            &response,
                            &mut messages,
                            &mut files_modified,
                            &mut total_usage,
                            None,
                        )
                        .await?;
                    }
                    StopReason::MaxTokens => {
                        self.emit_cost_update(&total_usage, &response.usage);
                        self.reporter.report(ProgressEvent::TaskCompleted {
                            success: false,
                            iterations: iteration,
                            duration: task_start.elapsed(),
                        });
                        return Ok(self.build_result(&response, total_usage, files_modified));
                    }
                }
            }
        }
        .instrument(span)
        .await
    }

    async fn build_initial_messages(&self, task: &Task) -> Vec<Message> {
        let mut messages = vec![Message {
            role: MessageRole::System,
            content: self
                .system_prompt
                .read()
                .unwrap_or_else(|e| {
                    tracing::warn!("system prompt lock was poisoned, recovering");
                    e.into_inner()
                })
                .clone(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        }];

        // Add working memory from context
        messages.extend(task.context.working_memory.clone());

        // Query episodic memory for relevant past experiences
        let query = match &task.kind {
            crew_core::TaskKind::Plan { goal } => goal.clone(),
            crew_core::TaskKind::Code { instruction, .. } => instruction.clone(),
            crew_core::TaskKind::Review { .. } => "code review".to_string(),
            crew_core::TaskKind::Test { command } => command.clone(),
            crew_core::TaskKind::Custom { name, .. } => name.clone(),
        };

        let episodes_result = if let Some(ref embedder) = self.embedder {
            match embedder.embed(&[query.as_str()]).await {
                Ok(vecs) => {
                    let query_emb = vecs.into_iter().next();
                    self.memory.find_relevant_hybrid(&query, query_emb, 6).await
                }
                Err(e) => {
                    warn!(error = %e, "embedding failed, falling back to keyword search");
                    self.memory.find_relevant_hybrid(&query, None, 6).await
                }
            }
        } else {
            self.memory
                .find_relevant(&task.context.working_dir, &query, 3)
                .await
        };

        if let Ok(episodes) = episodes_result {
            if !episodes.is_empty() {
                let mut context_str = String::from("## Relevant Past Experiences\n\n");
                for ep in &episodes {
                    context_str.push_str(&format!(
                        "### {} ({})\n{}\n",
                        ep.task_id,
                        match ep.outcome {
                            crew_memory::EpisodeOutcome::Success => "succeeded",
                            crew_memory::EpisodeOutcome::Failure => "failed",
                            crew_memory::EpisodeOutcome::Blocked => "blocked",
                            crew_memory::EpisodeOutcome::Cancelled => "cancelled",
                        },
                        ep.summary
                    ));
                    if !ep.key_decisions.is_empty() {
                        context_str.push_str("Key decisions:\n");
                        for decision in &ep.key_decisions {
                            context_str.push_str(&format!("- {}\n", decision));
                        }
                    }
                    context_str.push('\n');
                }

                messages.push(Message {
                    role: MessageRole::System,
                    content: context_str,
                    media: vec![],
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        // Add the task as user message
        let task_content = match &task.kind {
            crew_core::TaskKind::Plan { goal } => format!("Plan how to accomplish: {goal}"),
            crew_core::TaskKind::Code { instruction, files } => {
                let files_str = files
                    .iter()
                    .map(|f| f.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Code task: {instruction}\nFiles in scope: {files_str}")
            }
            crew_core::TaskKind::Review { diff } => format!("Review this diff:\n{diff}"),
            crew_core::TaskKind::Test { command } => format!("Run test: {command}"),
            crew_core::TaskKind::Custom { name, params } => {
                format!("Custom task '{name}': {params}")
            }
        };

        messages.push(Message {
            role: MessageRole::User,
            content: task_content,
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        });

        messages
    }

    fn response_to_message(&self, response: &ChatResponse) -> Message {
        Message {
            role: MessageRole::Assistant,
            content: response.content.clone().unwrap_or_default(),
            media: vec![],
            tool_calls: if response.tool_calls.is_empty() {
                None
            } else {
                Some(response.tool_calls.clone())
            },
            tool_call_id: None,
            reasoning_content: response.reasoning_content.clone(),
            timestamp: chrono::Utc::now(),
        }
    }

    async fn execute_tools(
        &self,
        response: &ChatResponse,
    ) -> Result<(Vec<Message>, Vec<std::path::PathBuf>, TokenUsage)> {
        // Log parallel tool execution details
        let tool_names: Vec<&str> = response
            .tool_calls
            .iter()
            .map(|tc| tc.name.as_str())
            .collect();
        tracing::info!(
            parallel_tools = response.tool_calls.len(),
            tool_names = %tool_names.join(", "),
            "executing tools in parallel"
        );

        // Spawn each tool as a separate tokio task so that if the agent-level
        // timeout fires, the tasks keep running and can perform their own cleanup
        // (e.g., browser tool kills Chrome, spawn tool finishes gracefully).
        // Without tokio::spawn, timeout would drop the futures mid-flight,
        // orphaning child processes (Chrome, shell commands, etc.).
        let handles: Vec<_> = response
            .tool_calls
            .iter()
            .map(|tool_call| {
                // Clone Arc-wrapped fields so the spawned task is 'static
                let tools = self.tools.clone();
                let reporter = self.reporter.clone();
                let hooks = self.hooks.clone();
                let hook_ctx = self.hook_ctx();
                let tc_name = tool_call.name.clone();
                let tc_id = tool_call.id.clone();
                let tc_args = tool_call.arguments.clone();

                tokio::spawn(async move {
                    let tool_start = Instant::now();
                    debug!(tool = %tc_name, tool_id = %tc_id, "executing tool");

                    reporter.report(ProgressEvent::ToolStarted {
                        name: tc_name.clone(),
                        tool_id: tc_id.clone(),
                    });

                    // Before-tool hook: may deny execution
                    if let Some(ref hooks) = hooks {
                        let payload = HookPayload::before_tool(
                            &tc_name,
                            tc_args.clone(),
                            &tc_id,
                            hook_ctx.as_ref(),
                        );
                        if let HookResult::Deny(reason) =
                            hooks.run(HookEvent::BeforeToolCall, &payload).await
                        {
                            let deny_msg = if reason.is_empty() {
                                format!("[HOOK DENIED] Tool '{}' was blocked by a lifecycle hook. Do not retry.", tc_name)
                            } else {
                                format!("[HOOK DENIED] Tool '{}' was blocked: {}. Do not retry.", tc_name, reason)
                            };
                            return (
                                Message {
                                    role: MessageRole::Tool,
                                    content: deny_msg,
                                    media: vec![],
                                    tool_calls: None,
                                    tool_call_id: Some(tc_id),
                                    reasoning_content: None,
                                    timestamp: chrono::Utc::now(),
                                },
                                None,
                                None,
                            );
                        }
                    }

                    let result = tools.execute(&tc_name, &tc_args).await;

                    let duration = tool_start.elapsed();

                    let (content, file_modified, tool_tokens, tool_success) = match result {
                        Ok(tool_result) => {
                            debug!(
                                tool = %tc_name,
                                success = tool_result.success,
                                duration_ms = duration.as_millis() as u64,
                                "tool completed"
                            );

                            if let Some(ref file) = tool_result.file_modified {
                                info!(tool = %tc_name, file = %file.display(), "file modified");
                                reporter.report(ProgressEvent::FileModified {
                                    path: file.display().to_string(),
                                });
                            }

                            let output_preview =
                                crew_core::truncated_utf8(&tool_result.output, 200, "...");

                            reporter.report(ProgressEvent::ToolCompleted {
                                name: tc_name.clone(),
                                tool_id: tc_id.clone(),
                                success: tool_result.success,
                                output_preview,
                                duration,
                            });

                            let success = tool_result.success;
                            (
                                tool_result.output,
                                tool_result.file_modified,
                                tool_result.tokens_used,
                                success,
                            )
                        }
                        Err(e) => {
                            warn!(
                                tool = %tc_name,
                                error = %e,
                                duration_ms = duration.as_millis() as u64,
                                "tool failed"
                            );

                            reporter.report(ProgressEvent::ToolCompleted {
                                name: tc_name.clone(),
                                tool_id: tc_id.clone(),
                                success: false,
                                output_preview: e.to_string(),
                                duration,
                            });

                            (format!("Error: {e}"), None, None, false)
                        }
                    };

                    // After-tool hook (fire-and-forget)
                    if let Some(ref hooks) = hooks {
                        let payload = HookPayload::after_tool(
                            &tc_name,
                            &tc_id,
                            crew_core::truncated_utf8(&content, 500, "..."),
                            tool_success,
                            duration.as_millis() as u64,
                            hook_ctx.as_ref(),
                        );
                        let _ = hooks.run(HookEvent::AfterToolCall, &payload).await;
                    }

                    // Per-tool output truncation with head/tail split
                    let limit = crew_core::tool_output_limit(&tc_name);
                    let content = crew_core::truncate_head_tail(&content, limit, 0.7);
                    let content = crate::sanitize::sanitize_tool_output(&content);

                    (
                        Message {
                            role: MessageRole::Tool,
                            content,
                            media: vec![],
                            tool_calls: None,
                            tool_call_id: Some(tc_id),
                            reasoning_content: None,
                            timestamp: chrono::Utc::now(),
                        },
                        file_modified,
                        tool_tokens,
                    )
                })
            })
            .collect();

        let tool_timeout_secs = self.config.tool_timeout_secs;
        let tool_timeout = Duration::from_secs(tool_timeout_secs);
        let results: Vec<_> =
            match tokio::time::timeout(tool_timeout, futures::future::join_all(handles)).await {
                Ok(results) => {
                    // Unwrap JoinHandle results — panics in tool tasks become errors
                    results
                        .into_iter()
                        .map(|r| {
                            r.unwrap_or_else(|e| {
                                // Task panicked — return a placeholder error tuple
                                (
                                    Message {
                                        role: MessageRole::Tool,
                                        content: format!("Tool task panicked: {e}"),
                                        media: vec![],
                                        tool_calls: None,
                                        tool_call_id: None,
                                        reasoning_content: None,
                                        timestamp: chrono::Utc::now(),
                                    },
                                    None,
                                    None,
                                )
                            })
                        })
                        .collect()
                }
                Err(_) => {
                    tracing::error!(
                        timeout_secs = tool_timeout_secs,
                        tool_count = response.tool_calls.len(),
                        tools = %tool_names.join(", "),
                        "tool execution timed out — spawned tasks continue running for cleanup"
                    );
                    // Note: spawned tasks are NOT aborted — they continue running so
                    // tools can perform their own cleanup (browser kills Chrome, etc.).
                    // They will eventually complete via their own internal timeouts.
                    let mut messages = Vec::with_capacity(response.tool_calls.len());
                    for tc in &response.tool_calls {
                        messages.push(Message {
                            role: MessageRole::Tool,
                            content: format!(
                                "Tool '{}' timed out after {} seconds",
                                tc.name, tool_timeout_secs
                            ),
                            media: vec![],
                            tool_calls: None,
                            tool_call_id: Some(tc.id.clone()),
                            reasoning_content: None,
                            timestamp: chrono::Utc::now(),
                        });
                    }
                    return Ok((messages, vec![], TokenUsage::default()));
                }
            };

        // Log completion of all parallel tools
        let result_sizes: Vec<usize> = results.iter().map(|(m, _, _)| m.content.len()).collect();
        let total_result_bytes: usize = result_sizes.iter().sum();
        tracing::info!(
            parallel_tools = results.len(),
            result_sizes = %result_sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
            total_result_bytes,
            "all parallel tools completed"
        );

        // Aggregate results — join_all preserves input order.
        let mut messages = Vec::with_capacity(results.len());
        let mut files_modified = Vec::new();
        let mut tokens_used = TokenUsage::default();

        for (message, file_modified, tool_tokens) in results {
            messages.push(message);
            if let Some(file) = file_modified {
                files_modified.push(file);
            }
            if let Some(tokens) = tool_tokens {
                tokens_used.input_tokens += tokens.input_tokens;
                tokens_used.output_tokens += tokens.output_tokens;
            }
        }

        Ok((messages, files_modified, tokens_used))
    }

    fn trim_to_context_window(&self, messages: &mut Vec<Message>) {
        use crate::compaction::{MIN_RECENT_MESSAGES, compact_messages, find_recent_boundary};
        use crew_llm::context::{estimate_message_tokens, estimate_tokens};

        if messages.len() <= 1 + MIN_RECENT_MESSAGES {
            return;
        }

        let window = self.llm.context_window();
        let budget = (window as f64 * 0.8 / crate::compaction::SAFETY_MARGIN) as u32;

        let total: u32 = messages.iter().map(estimate_message_tokens).sum();
        if total <= budget {
            return;
        }

        let system_tokens = estimate_message_tokens(&messages[0]);
        if system_tokens >= budget {
            warn!(
                system_tokens,
                budget, "system prompt exceeds context window budget, cannot trim"
            );
            return;
        }

        let split = find_recent_boundary(messages, budget, system_tokens);
        let recent_tokens: u32 = messages[split..].iter().map(estimate_message_tokens).sum();

        // If recent messages alone exceed budget, fall back to simple truncation
        if system_tokens + recent_tokens >= budget {
            self.fallback_truncate(messages, budget);
            return;
        }

        let old_messages = &messages[1..split];
        if old_messages.is_empty() {
            return;
        }

        let summary_budget = budget - system_tokens - recent_tokens;
        let summary_text = compact_messages(old_messages, summary_budget);
        let summary_tokens = estimate_tokens(&summary_text) + 4;

        let original_count = messages.len();
        let dropped = split - 1;
        messages.drain(1..split);
        messages.insert(
            1,
            Message {
                role: MessageRole::System,
                content: summary_text,
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                timestamp: chrono::Utc::now(),
            },
        );

        info!(
            original_tokens = total,
            summary_tokens,
            messages_compacted = dropped,
            messages_remaining = messages.len(),
            original_messages = original_count,
            "compacted conversation history ({} token budget)",
            budget
        );
    }

    /// Simple truncation fallback when even recent messages exceed budget.
    fn fallback_truncate(&self, messages: &mut Vec<Message>, limit: u32) {
        let system_tokens = crew_llm::context::estimate_message_tokens(&messages[0]);
        let mut kept_tokens = system_tokens;
        let mut keep_from = messages.len();

        for i in (1..messages.len()).rev() {
            let msg_tokens = crew_llm::context::estimate_message_tokens(&messages[i]);
            if kept_tokens + msg_tokens > limit {
                break;
            }
            kept_tokens += msg_tokens;
            keep_from = i;
        }

        // Keep at least 2 non-system messages
        let max_keep_from = messages.len().saturating_sub(2);
        if keep_from > max_keep_from {
            keep_from = max_keep_from;
        }

        if keep_from > 1 {
            let dropped = keep_from - 1;
            messages.drain(1..keep_from);
            warn!(
                messages_dropped = dropped,
                messages_kept = messages.len(),
                "fallback truncation ({} token limit)",
                limit
            );
        }
    }

    /// Wait until the shutdown flag is set. Used with `tokio::select!`
    /// to cancel long-running operations on Ctrl+C.
    async fn wait_for_shutdown(&self) {
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    async fn consume_stream(
        &self,
        mut stream: ChatStream,
        iteration: u32,
    ) -> Result<(ChatResponse, bool)> {
        // Clear any pending status line (e.g., "Thinking...")
        self.reporter.report(ProgressEvent::Response {
            content: String::new(),
            iteration,
        });

        let mut text = String::new();
        let mut reasoning = String::new();
        // (id, name, args_json, metadata)
        let mut tool_calls: Vec<(String, String, String, Option<serde_json::Value>)> = Vec::new();
        let mut usage = crew_llm::TokenUsage::default();
        let mut stop_reason = StopReason::EndTurn;

        loop {
            let event = tokio::select! {
                event = stream.next() => event,
                _ = self.wait_for_shutdown() => {
                    warn!("shutdown received during streaming");
                    break;
                }
            };

            let Some(event) = event else {
                tracing::debug!("stream ended (None)");
                break;
            };
            tracing::debug!(?event, "stream event received");

            match event {
                StreamEvent::ReasoningDelta(delta) => {
                    reasoning.push_str(&delta);
                }
                StreamEvent::TextDelta(delta) => {
                    self.reporter.report(ProgressEvent::StreamChunk {
                        text: delta.clone(),
                        iteration,
                    });
                    text.push_str(&delta);
                }
                StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                } => {
                    while tool_calls.len() <= index {
                        tool_calls.push((String::new(), String::new(), String::new(), None));
                    }
                    if let Some(id) = id {
                        tool_calls[index].0 = id;
                    }
                    if let Some(name) = name {
                        tool_calls[index].1 = name;
                    }
                    tool_calls[index].2.push_str(&arguments_delta);
                }
                StreamEvent::ToolCallMetadata { index, metadata } => {
                    while tool_calls.len() <= index {
                        tool_calls.push((String::new(), String::new(), String::new(), None));
                    }
                    tool_calls[index].3 = Some(metadata);
                }
                StreamEvent::Usage(u) => {
                    usage = u;
                }
                StreamEvent::Done(reason) => {
                    stop_reason = reason;
                }
                StreamEvent::Error(err) => {
                    eyre::bail!("Stream error: {}", err);
                }
            }
        }

        let streamed = !text.is_empty();
        if streamed {
            self.reporter
                .report(ProgressEvent::StreamDone { iteration });
        }

        // Strip <think> tags from accumulated streaming content (some models
        // embed chain-of-thought in <think> tags via TextDelta instead of
        // using ReasoningDelta events).
        let (text, think_extracted) = crew_llm::strip_think_tags(&text);
        if let Some(ref extracted) = think_extracted {
            if reasoning.is_empty() {
                reasoning = extracted.clone();
            }
        }

        let content = if text.is_empty() { None } else { Some(text) };
        let tool_calls: Vec<crew_core::ToolCall> = tool_calls
            .into_iter()
            .filter(|(_, name, _, _)| !name.is_empty())
            .map(|(id, name, args, metadata)| {
                let arguments = serde_json::from_str(&args).unwrap_or_else(|e| {
                    tracing::warn!(tool = %name, error = %e, raw = %args, "malformed tool call JSON");
                    // Return a String value so the tool's deserialize step fails
                    // and the error propagates back to the LLM for correction.
                    serde_json::Value::String(format!(
                        "MALFORMED_JSON: {e}. Raw input: {}",
                        crew_core::truncated_utf8(&args, 200, "...")
                    ))
                });
                crew_core::ToolCall {
                    id,
                    name,
                    arguments,
                    metadata,
                }
            })
            .collect();

        let reasoning_content = if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        };

        Ok((
            ChatResponse {
                content,
                reasoning_content,
                tool_calls,
                stop_reason,
                usage,
            },
            streamed,
        ))
    }

    fn emit_cost_update(&self, total_usage: &TokenUsage, response_usage: &crew_llm::TokenUsage) {
        let pricing = crew_llm::pricing::model_pricing(self.llm.model_id());
        let response_cost =
            pricing.map(|p| p.cost(response_usage.input_tokens, response_usage.output_tokens));
        let session_cost =
            pricing.map(|p| p.cost(total_usage.input_tokens, total_usage.output_tokens));
        self.reporter.report(ProgressEvent::CostUpdate {
            session_input_tokens: total_usage.input_tokens,
            session_output_tokens: total_usage.output_tokens,
            response_cost,
            session_cost,
        });
    }

    fn build_result(
        &self,
        response: &ChatResponse,
        usage: TokenUsage,
        files_modified: Vec<std::path::PathBuf>,
    ) -> TaskResult {
        let success = response.stop_reason != StopReason::MaxTokens;
        TaskResult {
            success,
            output: response.content.clone().unwrap_or_default(),
            files_modified,
            subtasks: Vec::new(),
            token_usage: crew_core::TokenUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                ..Default::default()
            },
        }
    }

    /// Check whether the agent loop should stop due to budget constraints.
    fn check_budget(
        &self,
        iteration: u32,
        start: Instant,
        total_usage: &TokenUsage,
    ) -> Option<BudgetStop> {
        if self.shutdown.load(Ordering::Acquire) {
            return Some(BudgetStop::Shutdown);
        }
        if iteration >= self.config.max_iterations {
            return Some(BudgetStop::MaxIterations);
        }
        if let Some(timeout) = self.config.max_timeout {
            if start.elapsed() > timeout {
                return Some(BudgetStop::WallClockTimeout { limit: timeout });
            }
        }
        if let Some(max_tokens) = self.config.max_tokens {
            let used = total_usage.input_tokens + total_usage.output_tokens;
            if used >= max_tokens {
                return Some(BudgetStop::MaxTokens {
                    used,
                    limit: max_tokens,
                });
            }
        }
        None
    }

    /// Maximum retries for transient LLM failures (empty responses, stream errors).
    const LLM_RETRY_MAX: u32 = 3;

    /// Check if an LLM response is empty (0 output tokens, no content, no tool calls).
    /// This typically indicates a transient API issue (e.g. server overload).
    fn is_empty_response(response: &ChatResponse) -> bool {
        response.usage.output_tokens == 0
            && response.content.as_ref().is_none_or(|c| c.is_empty())
            && response.tool_calls.is_empty()
    }

    /// Check if an error looks like a transient server issue worth retrying.
    fn is_retryable_stream_error(err: &eyre::Report) -> bool {
        let msg = err.to_string().to_lowercase();
        msg.contains("overloaded")
            || msg.contains("temporarily")
            || msg.contains("429")
            || msg.contains("502")
            || msg.contains("503")
            || msg.contains("1305")
            || msg.contains("rate limit")
    }

    /// Call the LLM with before/after lifecycle hooks.
    /// Automatically retries on empty responses and retryable stream errors.
    async fn call_llm_with_hooks(
        &self,
        messages: &[Message],
        tools_spec: &[ToolSpec],
        config: &ChatConfig,
        iteration: u32,
        total_usage: &TokenUsage,
    ) -> Result<(ChatResponse, bool)> {
        let ctx = self.hook_ctx();
        if let Some(ref hooks) = self.hooks {
            let payload = HookPayload::before_llm(
                self.llm.model_id(),
                messages.len(),
                iteration,
                ctx.as_ref(),
            );
            if let HookResult::Deny(reason) = hooks.run(HookEvent::BeforeLlmCall, &payload).await {
                eyre::bail!("LLM call denied by hook: {reason}");
            }
        }

        let mut last_error: Option<eyre::Report> = None;

        for attempt in 0..=Self::LLM_RETRY_MAX {
            let call_start = Instant::now();
            // Try the full LLM call (stream creation + consumption)
            let call_result = async {
                let stream = self.llm.chat_stream(messages, tools_spec, config).await?;
                self.consume_stream(stream, iteration).await
            }
            .await;

            match call_result {
                Ok((response, streamed)) => {
                    if !Self::is_empty_response(&response) || attempt == Self::LLM_RETRY_MAX {
                        // Success or final attempt — use this response
                        if let Some(ref hooks) = self.hooks {
                            let latency_ms = call_start.elapsed().as_millis() as u64;
                            let cum_in = total_usage.input_tokens + response.usage.input_tokens;
                            let cum_out = total_usage.output_tokens + response.usage.output_tokens;
                            let pricing = crew_llm::pricing::model_pricing(self.llm.model_id());
                            let session_cost = pricing.map(|p| p.cost(cum_in, cum_out));
                            let response_cost = pricing.map(|p| {
                                p.cost(response.usage.input_tokens, response.usage.output_tokens)
                            });
                            let payload = HookPayload::after_llm(
                                self.llm.model_id(),
                                iteration,
                                &format!("{:?}", response.stop_reason),
                                !response.tool_calls.is_empty(),
                                response.usage.input_tokens,
                                response.usage.output_tokens,
                                self.llm.provider_name(),
                                latency_ms,
                                cum_in,
                                cum_out,
                                session_cost,
                                response_cost,
                                ctx.as_ref(),
                            );
                            let _ = hooks.run(HookEvent::AfterLlmCall, &payload).await;
                        }
                        return Ok((response, streamed));
                    }

                    // Empty response — retry
                    let delay = Duration::from_secs(1 << attempt);
                    warn!(
                        attempt = attempt + 1,
                        max = Self::LLM_RETRY_MAX,
                        delay_s = delay.as_secs(),
                        iteration,
                        "empty (0-token) LLM response, retrying"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    if attempt < Self::LLM_RETRY_MAX && Self::is_retryable_stream_error(&e) {
                        let delay = Duration::from_secs(1 << attempt);
                        warn!(
                            attempt = attempt + 1,
                            max = Self::LLM_RETRY_MAX,
                            delay_s = delay.as_secs(),
                            error = %e,
                            iteration,
                            "retryable stream error, retrying"
                        );
                        last_error = Some(e);
                        tokio::time::sleep(delay).await;
                    } else {
                        // Non-retryable error or last attempt — propagate
                        return Err(e);
                    }
                }
            }
        }

        // All retries exhausted with errors
        Err(last_error.unwrap_or_else(|| eyre::eyre!("LLM call failed after retries")))
    }

    /// Execute tool calls from an LLM response and accumulate results.
    async fn handle_tool_use(
        &self,
        response: &ChatResponse,
        messages: &mut Vec<Message>,
        files_modified: &mut Vec<PathBuf>,
        total_usage: &mut TokenUsage,
        tracker: Option<&TokenTracker>,
    ) -> Result<()> {
        // Fix tool_call IDs — some models (e.g. qwen via dashscope) generate
        // duplicate or empty IDs which downstream providers reject with 400.
        // We fix IDs on the response clone so both the assistant message and tool result
        // messages use the same corrected IDs.
        let mut response = response.clone();
        {
            let mut seen = std::collections::HashSet::new();
            for (i, tc) in response.tool_calls.iter_mut().enumerate() {
                if tc.id.is_empty() || !seen.insert(tc.id.clone()) {
                    let new_id = format!("call_{}_{}", i, &tc.name);
                    tracing::warn!(
                        old_id = %tc.id,
                        new_id = %new_id,
                        tool = %tc.name,
                        "fixing empty/duplicate tool_call_id"
                    );
                    tc.id = new_id;
                }
            }
        }
        messages.push(self.response_to_message(&response));
        let (tool_messages, tool_files, tool_tokens) = self.execute_tools(&response).await?;
        messages.extend(tool_messages);
        files_modified.extend(tool_files);
        total_usage.input_tokens += tool_tokens.input_tokens;
        total_usage.output_tokens += tool_tokens.output_tokens;
        if let Some(t) = tracker {
            t.input_tokens
                .store(total_usage.input_tokens, Ordering::Relaxed);
            t.output_tokens
                .store(total_usage.output_tokens, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Log and report a budget stop event (used by `run_task`).
    fn report_budget_stop(&self, stop: &BudgetStop, iteration: u32) {
        match stop {
            BudgetStop::Shutdown => {
                info!(iteration, "shutdown signal received");
                self.reporter.report(ProgressEvent::TaskInterrupted {
                    iterations: iteration,
                });
            }
            BudgetStop::MaxIterations => {
                warn!(
                    iteration,
                    max = self.config.max_iterations,
                    "hit max iterations limit"
                );
                self.reporter.report(ProgressEvent::MaxIterationsReached {
                    limit: self.config.max_iterations,
                });
            }
            BudgetStop::MaxTokens { used, limit } => {
                warn!(used, max = limit, "hit token budget limit");
                self.reporter.report(ProgressEvent::TokenBudgetExceeded {
                    used: *used,
                    limit: *limit,
                });
            }
            BudgetStop::WallClockTimeout { limit } => {
                warn!(limit_s = limit.as_secs(), "hit wall-clock timeout");
                self.reporter
                    .report(ProgressEvent::WallClockTimeoutReached {
                        elapsed: *limit,
                        limit: *limit,
                    });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use crew_core::ToolCall;
    use crew_llm::{ChatResponse, StopReason, TokenUsage as LlmTokenUsage};

    // ---------- AgentConfig::default ----------

    #[test]
    fn agent_config_default_values() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.max_iterations, 50);
        assert_eq!(cfg.max_tokens, None);
        assert_eq!(cfg.max_timeout, Some(Duration::from_secs(600)));
        assert!(cfg.save_episodes);
        assert_eq!(cfg.tool_timeout_secs, 300);
        assert!(cfg.worker_prompt.is_none());
    }

    // ---------- TokenTracker ----------

    #[test]
    fn token_tracker_new_starts_at_zero() {
        let t = TokenTracker::new();
        assert_eq!(t.input_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(t.output_tokens.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn token_tracker_default_starts_at_zero() {
        let t = TokenTracker::default();
        assert_eq!(t.input_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(t.output_tokens.load(Ordering::Relaxed), 0);
    }

    // ---------- BudgetStop::message ----------

    #[test]
    fn budget_stop_shutdown_message() {
        assert_eq!(BudgetStop::Shutdown.message(), "Interrupted.");
    }

    #[test]
    fn budget_stop_max_iterations_message() {
        assert_eq!(
            BudgetStop::MaxIterations.message(),
            "Reached max iterations."
        );
    }

    #[test]
    fn budget_stop_max_tokens_message() {
        let msg = BudgetStop::MaxTokens {
            used: 1000,
            limit: 500,
        }
        .message();
        assert!(
            msg.contains("token") || msg.contains("Token") || msg.contains("TOKEN"),
            "expected 'token' in: {msg}"
        );
        assert!(msg.contains("1000"), "expected '1000' in: {msg}");
        assert!(msg.contains("500"), "expected '500' in: {msg}");
    }

    #[test]
    fn budget_stop_wall_clock_timeout_message() {
        let msg = BudgetStop::WallClockTimeout {
            limit: Duration::from_secs(120),
        }
        .message();
        assert!(
            msg.to_lowercase().contains("timeout"),
            "expected 'timeout' in: {msg}"
        );
    }

    // ---------- Agent::is_empty_response ----------

    fn make_response(
        content: Option<&str>,
        tool_calls: Vec<ToolCall>,
        output_tokens: u32,
    ) -> ChatResponse {
        ChatResponse {
            content: content.map(String::from),
            reasoning_content: None,
            tool_calls,
            stop_reason: StopReason::EndTurn,
            usage: LlmTokenUsage {
                input_tokens: 0,
                output_tokens,
                ..Default::default()
            },
        }
    }

    #[test]
    fn is_empty_response_true_when_all_empty() {
        let r = make_response(None, vec![], 0);
        assert!(Agent::is_empty_response(&r));

        // Empty string content also counts as empty
        let r2 = make_response(Some(""), vec![], 0);
        assert!(Agent::is_empty_response(&r2));
    }

    #[test]
    fn is_empty_response_false_with_content() {
        let r = make_response(Some("hello"), vec![], 0);
        assert!(!Agent::is_empty_response(&r));
    }

    #[test]
    fn is_empty_response_false_with_tool_calls() {
        let tc = ToolCall {
            id: "1".into(),
            name: "test".into(),
            arguments: serde_json::json!({}),
            metadata: None,
        };
        let r = make_response(None, vec![tc], 0);
        assert!(!Agent::is_empty_response(&r));
    }

    #[test]
    fn is_empty_response_false_with_tokens() {
        let r = make_response(None, vec![], 10);
        assert!(!Agent::is_empty_response(&r));
    }

    // ---------- Agent::is_retryable_stream_error ----------

    #[test]
    fn is_retryable_stream_error_transient_errors() {
        for keyword in ["overloaded", "429", "503", "rate limit"] {
            let err = eyre::eyre!("Server error: {}", keyword);
            assert!(
                Agent::is_retryable_stream_error(&err),
                "expected retryable for: {keyword}"
            );
        }
    }

    #[test]
    fn is_retryable_stream_error_non_retryable() {
        let err = eyre::eyre!("invalid json");
        assert!(!Agent::is_retryable_stream_error(&err));
    }

    // ---------- ConversationResponse derives ----------

    #[test]
    fn conversation_response_clone_and_debug() {
        let resp = ConversationResponse {
            content: "test".into(),
            token_usage: crew_core::TokenUsage {
                input_tokens: 10,
                output_tokens: 20,
                ..Default::default()
            },
            files_modified: vec![],
            streamed: false,
        };
        let cloned = resp.clone();
        assert_eq!(cloned.content, "test");
        assert_eq!(cloned.token_usage.input_tokens, 10);

        // Debug trait works
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("ConversationResponse"));
    }
}
