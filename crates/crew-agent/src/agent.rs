//! Agent implementation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crew_core::{AgentId, Message, MessageRole, Task, TaskResult, TokenUsage};
use crew_llm::{
    ChatConfig, ChatResponse, ChatStream, EmbeddingProvider, LlmProvider, StopReason, StreamEvent,
};
use crew_memory::{Episode, EpisodeOutcome, EpisodeStore};
use eyre::Result;
use futures::StreamExt;
use tracing::{Instrument, debug, info, info_span, warn};

use crate::hooks::{HookEvent, HookExecutor, HookPayload, HookResult};
use crate::progress::{ProgressEvent, ProgressReporter, SilentReporter};
use crate::tools::ToolRegistry;

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
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_tokens: None,
            max_timeout: Some(Duration::from_secs(600)),
            save_episodes: true,
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

/// An agent that can execute tasks.
pub struct Agent {
    /// Unique identifier for this agent.
    pub id: AgentId,
    /// LLM provider for generating responses.
    llm: Arc<dyn LlmProvider>,
    /// Tool registry for executing tool calls.
    tools: ToolRegistry,
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
            tools,
            memory,
            embedder: None,
            system_prompt: RwLock::new(system_prompt),
            config: AgentConfig::default(),
            reporter: Arc::new(SilentReporter),
            hooks: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the agent configuration.
    pub fn with_config(mut self, config: AgentConfig) -> Self {
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

    /// Override the system prompt (e.g. for gateway mode).
    pub fn with_system_prompt(self, prompt: String) -> Self {
        *self
            .system_prompt
            .write()
            .expect("system prompt lock poisoned") = prompt;
        self
    }

    /// Update the system prompt at runtime (hot-reload).
    pub fn set_system_prompt(&self, prompt: String) {
        *self
            .system_prompt
            .write()
            .expect("system prompt lock poisoned") = prompt;
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
        let mut messages = vec![Message {
            role: MessageRole::System,
            content: self
                .system_prompt
                .read()
                .expect("system prompt lock poisoned")
                .clone(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            timestamp: chrono::Utc::now(),
        }];

        messages.extend_from_slice(history);

        messages.push(Message {
            role: MessageRole::User,
            content: user_content.to_string(),
            media,
            tool_calls: None,
            tool_call_id: None,
            timestamp: chrono::Utc::now(),
        });

        let config = ChatConfig::default();
        let mut total_usage = TokenUsage::default();
        let mut files_modified = Vec::new();
        let mut iteration = 0u32;
        let start = Instant::now();

        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                return Ok(ConversationResponse {
                    content: "Interrupted.".into(),
                    token_usage: total_usage,
                    files_modified,
                    streamed: false,
                });
            }

            if iteration >= self.config.max_iterations {
                return Ok(ConversationResponse {
                    content: "Reached max iterations.".into(),
                    token_usage: total_usage,
                    files_modified,
                    streamed: false,
                });
            }

            if let Some(timeout) = self.config.max_timeout {
                if start.elapsed() > timeout {
                    return Ok(ConversationResponse {
                        content: format!(
                            "Wall-clock timeout ({:.0}s limit).",
                            timeout.as_secs_f64()
                        ),
                        token_usage: total_usage,
                        files_modified,
                        streamed: false,
                    });
                }
            }

            if let Some(max_tokens) = self.config.max_tokens {
                let used = total_usage.input_tokens + total_usage.output_tokens;
                if used >= max_tokens {
                    return Ok(ConversationResponse {
                        content: "Token budget exceeded.".into(),
                        token_usage: total_usage,
                        files_modified,
                        streamed: false,
                    });
                }
            }

            iteration += 1;
            let tools_spec = self.tools.specs();
            self.trim_to_context_window(&mut messages);

            // Before-LLM hook
            if let Some(ref hooks) = self.hooks {
                let payload = HookPayload {
                    event: HookEvent::BeforeLlmCall,
                    tool_name: None,
                    arguments: None,
                    tool_id: None,
                    result: None,
                    success: None,
                    duration_ms: None,
                    message_count: Some(messages.len()),
                    model: Some(self.llm.model_id().to_string()),
                    iteration: Some(iteration),
                    stop_reason: None,
                    has_tool_calls: None,
                    input_tokens: None,
                    output_tokens: None,
                };
                if let HookResult::Deny(reason) =
                    hooks.run(HookEvent::BeforeLlmCall, &payload).await
                {
                    eyre::bail!("LLM call denied by hook: {reason}");
                }
            }

            let stream = self
                .llm
                .chat_stream(&messages, &tools_spec, &config)
                .await?;
            let (response, streamed) = self.consume_stream(stream, iteration).await?;
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            // After-LLM hook
            if let Some(ref hooks) = self.hooks {
                let payload = HookPayload {
                    event: HookEvent::AfterLlmCall,
                    tool_name: None,
                    arguments: None,
                    tool_id: None,
                    result: None,
                    success: None,
                    duration_ms: None,
                    message_count: None,
                    model: Some(self.llm.model_id().to_string()),
                    iteration: Some(iteration),
                    stop_reason: Some(format!("{:?}", response.stop_reason)),
                    has_tool_calls: Some(!response.tool_calls.is_empty()),
                    input_tokens: Some(response.usage.input_tokens),
                    output_tokens: Some(response.usage.output_tokens),
                };
                let _ = hooks.run(HookEvent::AfterLlmCall, &payload).await;
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
                    messages.push(self.response_to_message(&response));
                    let (tool_messages, tool_files, tool_tokens) =
                        self.execute_tools(&response).await?;
                    messages.extend(tool_messages);
                    files_modified.extend(tool_files);
                    total_usage.input_tokens += tool_tokens.input_tokens;
                    total_usage.output_tokens += tool_tokens.output_tokens;
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
                if self.shutdown.load(Ordering::Relaxed) {
                    info!(iteration, "shutdown signal received");
                    self.reporter.report(ProgressEvent::TaskInterrupted {
                        iterations: iteration,
                    });
                    return Ok(TaskResult {
                        success: false,
                        output: "Task interrupted.".to_string(),
                        files_modified,
                        subtasks: Vec::new(),
                        token_usage: total_usage,
                    });
                }

                if iteration >= self.config.max_iterations {
                    warn!(
                        iteration,
                        max = self.config.max_iterations,
                        "hit max iterations limit"
                    );
                    self.reporter.report(ProgressEvent::MaxIterationsReached {
                        limit: self.config.max_iterations,
                    });
                    return Ok(TaskResult {
                        success: false,
                        output: format!("Task stopped after {} iterations (limit).", iteration),
                        files_modified,
                        subtasks: Vec::new(),
                        token_usage: total_usage,
                    });
                }

                if let Some(max_tokens) = self.config.max_tokens {
                    let used = total_usage.input_tokens + total_usage.output_tokens;
                    if used >= max_tokens {
                        warn!(used, max = max_tokens, "hit token budget limit");
                        self.reporter.report(ProgressEvent::TokenBudgetExceeded {
                            used,
                            limit: max_tokens,
                        });
                        return Ok(TaskResult {
                            success: false,
                            output: format!(
                                "Task stopped after {} tokens (budget: {}).",
                                used, max_tokens
                            ),
                            files_modified,
                            subtasks: Vec::new(),
                            token_usage: total_usage,
                        });
                    }
                }

                if let Some(timeout) = self.config.max_timeout {
                    if task_start.elapsed() > timeout {
                        warn!(
                            elapsed_s = task_start.elapsed().as_secs(),
                            limit_s = timeout.as_secs(),
                            "hit wall-clock timeout"
                        );
                        self.reporter
                            .report(ProgressEvent::WallClockTimeoutReached {
                                elapsed: task_start.elapsed(),
                                limit: timeout,
                            });
                        return Ok(TaskResult {
                            success: false,
                            output: format!(
                                "Task stopped after {:.0}s (wall-clock timeout: {:.0}s).",
                                task_start.elapsed().as_secs_f64(),
                                timeout.as_secs_f64()
                            ),
                            files_modified,
                            subtasks: Vec::new(),
                            token_usage: total_usage,
                        });
                    }
                }

                iteration += 1;
                let iter_start = Instant::now();
                self.reporter.report(ProgressEvent::Thinking { iteration });

                let tools_spec = self.tools.specs();
                self.trim_to_context_window(&mut messages);

                // Before-LLM hook
                if let Some(ref hooks) = self.hooks {
                    let payload = HookPayload {
                        event: HookEvent::BeforeLlmCall,
                        tool_name: None,
                        arguments: None,
                        tool_id: None,
                        result: None,
                        success: None,
                        duration_ms: None,
                        message_count: Some(messages.len()),
                        model: Some(self.llm.model_id().to_string()),
                        iteration: Some(iteration),
                        stop_reason: None,
                        has_tool_calls: None,
                        input_tokens: None,
                        output_tokens: None,
                    };
                    if let HookResult::Deny(reason) =
                        hooks.run(HookEvent::BeforeLlmCall, &payload).await
                    {
                        eyre::bail!("LLM call denied by hook: {reason}");
                    }
                }

                let stream = self
                    .llm
                    .chat_stream(&messages, &tools_spec, &config)
                    .await?;
                let (response, _streamed) = self.consume_stream(stream, iteration).await?;
                total_usage.input_tokens += response.usage.input_tokens;
                total_usage.output_tokens += response.usage.output_tokens;

                // After-LLM hook
                if let Some(ref hooks) = self.hooks {
                    let payload = HookPayload {
                        event: HookEvent::AfterLlmCall,
                        tool_name: None,
                        arguments: None,
                        tool_id: None,
                        result: None,
                        success: None,
                        duration_ms: None,
                        message_count: None,
                        model: Some(self.llm.model_id().to_string()),
                        iteration: Some(iteration),
                        stop_reason: Some(format!("{:?}", response.stop_reason)),
                        has_tool_calls: Some(!response.tool_calls.is_empty()),
                        input_tokens: Some(response.usage.input_tokens),
                        output_tokens: Some(response.usage.output_tokens),
                    };
                    let _ = hooks.run(HookEvent::AfterLlmCall, &payload).await;
                }

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
                            let summary_truncated =
                                crew_core::truncated_utf8(&summary, 500, "...");

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
                                    if let Ok(vecs) = embedder.embed(&[&summary_text]).await {
                                        if let Some(vec) = vecs.into_iter().next() {
                                            if let Err(e) =
                                                memory.store_embedding(&episode_id, vec).await
                                            {
                                                warn!(error = %e, "failed to store embedding");
                                            }
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
                        messages.push(self.response_to_message(&response));
                        let (tool_messages, tool_files, tool_tokens) =
                            self.execute_tools(&response).await?;
                        for msg in tool_messages {
                            messages.push(msg);
                        }
                        files_modified.extend(tool_files);
                        total_usage.input_tokens += tool_tokens.input_tokens;
                        total_usage.output_tokens += tool_tokens.output_tokens;
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
                .expect("system prompt lock poisoned")
                .clone(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
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
            timestamp: chrono::Utc::now(),
        }
    }

    async fn execute_tools(
        &self,
        response: &ChatResponse,
    ) -> Result<(Vec<Message>, Vec<std::path::PathBuf>, TokenUsage)> {
        // Execute all tool calls concurrently via join_all.
        // The LLM issued these calls in a single response, so they are independent.
        let futures: Vec<_> = response
            .tool_calls
            .iter()
            .map(|tool_call| async {
                let tool_start = Instant::now();
                debug!(tool = %tool_call.name, tool_id = %tool_call.id, "executing tool");

                self.reporter.report(ProgressEvent::ToolStarted {
                    name: tool_call.name.clone(),
                    tool_id: tool_call.id.clone(),
                });

                // Before-tool hook: may deny execution
                if let Some(ref hooks) = self.hooks {
                    let payload = HookPayload {
                        event: HookEvent::BeforeToolCall,
                        tool_name: Some(tool_call.name.clone()),
                        arguments: Some(tool_call.arguments.clone()),
                        tool_id: Some(tool_call.id.clone()),
                        result: None,
                        success: None,
                        duration_ms: None,
                        message_count: None,
                        model: None,
                        iteration: None,
                        stop_reason: None,
                        has_tool_calls: None,
                        input_tokens: None,
                        output_tokens: None,
                    };
                    if let HookResult::Deny(reason) =
                        hooks.run(HookEvent::BeforeToolCall, &payload).await
                    {
                        let deny_msg = if reason.is_empty() {
                            format!("[HOOK DENIED] Tool '{}' was blocked by a lifecycle hook. Do not retry.", tool_call.name)
                        } else {
                            format!("[HOOK DENIED] Tool '{}' was blocked: {}. Do not retry.", tool_call.name, reason)
                        };
                        return (
                            Message {
                                role: MessageRole::Tool,
                                content: deny_msg,
                                media: vec![],
                                tool_calls: None,
                                tool_call_id: Some(tool_call.id.clone()),
                                timestamp: chrono::Utc::now(),
                            },
                            None,
                            None,
                        );
                    }
                }

                let result = self
                    .tools
                    .execute(&tool_call.name, &tool_call.arguments)
                    .await;

                let duration = tool_start.elapsed();

                let (content, file_modified, tool_tokens, tool_success) = match result {
                    Ok(tool_result) => {
                        debug!(
                            tool = %tool_call.name,
                            success = tool_result.success,
                            duration_ms = duration.as_millis() as u64,
                            "tool completed"
                        );

                        if let Some(ref file) = tool_result.file_modified {
                            info!(tool = %tool_call.name, file = %file.display(), "file modified");
                            self.reporter.report(ProgressEvent::FileModified {
                                path: file.display().to_string(),
                            });
                        }

                        let output_preview =
                            crew_core::truncated_utf8(&tool_result.output, 200, "...");

                        self.reporter.report(ProgressEvent::ToolCompleted {
                            name: tool_call.name.clone(),
                            tool_id: tool_call.id.clone(),
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
                            tool = %tool_call.name,
                            error = %e,
                            duration_ms = duration.as_millis() as u64,
                            "tool failed"
                        );

                        self.reporter.report(ProgressEvent::ToolCompleted {
                            name: tool_call.name.clone(),
                            tool_id: tool_call.id.clone(),
                            success: false,
                            output_preview: e.to_string(),
                            duration,
                        });

                        (format!("Error: {e}"), None, None, false)
                    }
                };

                // After-tool hook (fire-and-forget)
                if let Some(ref hooks) = self.hooks {
                    let payload = HookPayload {
                        event: HookEvent::AfterToolCall,
                        tool_name: Some(tool_call.name.clone()),
                        arguments: None,
                        tool_id: Some(tool_call.id.clone()),
                        result: Some(crew_core::truncated_utf8(&content, 500, "...")),
                        success: Some(tool_success),
                        duration_ms: Some(duration.as_millis() as u64),
                        message_count: None,
                        model: None,
                        iteration: None,
                        stop_reason: None,
                        has_tool_calls: None,
                        input_tokens: None,
                        output_tokens: None,
                    };
                    let _ = hooks.run(HookEvent::AfterToolCall, &payload).await;
                }

                let content = crate::sanitize::sanitize_tool_output(&content);

                (
                    Message {
                        role: MessageRole::Tool,
                        content,
                        media: vec![],
                        tool_calls: None,
                        tool_call_id: Some(tool_call.id.clone()),
                        timestamp: chrono::Utc::now(),
                    },
                    file_modified,
                    tool_tokens,
                )
            })
            .collect();

        let results = futures::future::join_all(futures).await;

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
            if self.shutdown.load(Ordering::Relaxed) {
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
        let mut tool_calls: Vec<(String, String, String)> = Vec::new();
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

            let Some(event) = event else { break };

            match event {
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
                        tool_calls.push((String::new(), String::new(), String::new()));
                    }
                    if let Some(id) = id {
                        tool_calls[index].0 = id;
                    }
                    if let Some(name) = name {
                        tool_calls[index].1 = name;
                    }
                    tool_calls[index].2.push_str(&arguments_delta);
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

        let content = if text.is_empty() { None } else { Some(text) };
        let tool_calls: Vec<crew_core::ToolCall> = tool_calls
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, args)| {
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
                }
            })
            .collect();

        Ok((
            ChatResponse {
                content,
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
        TaskResult {
            success: true,
            output: response.content.clone().unwrap_or_default(),
            files_modified,
            subtasks: Vec::new(),
            token_usage: crew_core::TokenUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
            },
        }
    }
}
