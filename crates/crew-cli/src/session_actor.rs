//! Session actor: per-session tokio task that owns tools and processes messages.
//!
//! Replaces the spawn-per-message model in the gateway, eliminating the
//! `set_context()` race condition where shared tools could route messages
//! to the wrong chat.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::Utc;
use crew_agent::tools::{
    MessageTool, SendFileTool, SpawnTool, ToolPolicy, ToolRegistry, VoiceSynthesizeTool,
};
use crew_agent::{Agent, AgentConfig, HookContext, HookExecutor, TokenTracker};
use crew_bus::{ActiveSessionStore, SessionManager};
use crew_core::AgentId;
use crew_core::{InboundMessage, Message, MessageRole, OutboundMessage, SessionKey};
use crew_llm::{EmbeddingProvider, LlmProvider, ProviderRouter};
use crew_memory::EpisodeStore;
use tokio::sync::{Mutex, Semaphore, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::cron_tool::CronTool;
use crate::status_indicator::StatusIndicator;

/// Default actor inbox capacity.
const ACTOR_INBOX_SIZE: usize = 32;

/// Default idle timeout before an actor shuts down (30 minutes).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 1800;

/// Maximum number of pending messages buffered per inactive session.
const MAX_PENDING_PER_SESSION: usize = 50;

/// Shared buffer of outbound messages from inactive sessions, keyed by session key string.
/// Flushed when the user switches to that session via `/s`.
pub type PendingMessages = Arc<Mutex<HashMap<String, Vec<OutboundMessage>>>>;

// ── Messages ────────────────────────────────────────────────────────────────

/// Messages dispatched to a session actor.
pub enum ActorMessage {
    /// A user message to process.
    Inbound {
        message: InboundMessage,
        image_media: Vec<String>,
    },
    /// Cancel the current operation (future use).
    Cancel,
}

// ── ActorHandle ─────────────────────────────────────────────────────────────

/// Handle to a running session actor.
pub struct ActorHandle {
    pub tx: mpsc::Sender<ActorMessage>,
    pub created_at: Instant,
    join_handle: JoinHandle<()>,
}

impl ActorHandle {
    /// Whether the actor task has completed (idle-timeout, panic, etc.).
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }
}

// ── ActorRegistry ───────────────────────────────────────────────────────────

/// Manages the lifecycle of session actors.
pub struct ActorRegistry {
    actors: HashMap<String, ActorHandle>,
    factory: ActorFactory,
    semaphore: Arc<Semaphore>,
    out_tx: mpsc::Sender<OutboundMessage>,
    pending_messages: PendingMessages,
}

impl ActorRegistry {
    pub fn new(
        factory: ActorFactory,
        semaphore: Arc<Semaphore>,
        out_tx: mpsc::Sender<OutboundMessage>,
        pending_messages: PendingMessages,
    ) -> Self {
        Self {
            actors: HashMap::new(),
            factory,
            semaphore,
            out_tx,
            pending_messages,
        }
    }

    /// Route an inbound message to the correct actor, creating one if needed.
    pub async fn dispatch(
        &mut self,
        message: InboundMessage,
        image_media: Vec<String>,
        session_key: SessionKey,
        reply_channel: &str,
        reply_chat_id: &str,
        status_indicator: Option<Arc<StatusIndicator>>,
    ) {
        let key_str = session_key.to_string();

        // If actor exists but has finished (idle-timeout/panic), remove it
        if let Some(handle) = self.actors.get(&key_str) {
            if handle.is_finished() {
                self.actors.remove(&key_str);
            }
        }

        // Create actor if needed
        if !self.actors.contains_key(&key_str) {
            let (tx, join_handle) = self.factory.spawn(
                session_key.clone(),
                reply_channel,
                reply_chat_id,
                self.semaphore.clone(),
                status_indicator.clone(),
            );
            self.actors.insert(
                key_str.clone(),
                ActorHandle {
                    tx,
                    created_at: Instant::now(),
                    join_handle,
                },
            );
        }

        let handle = self.actors.get(&key_str).unwrap();
        let actor_msg = ActorMessage::Inbound {
            message,
            image_media,
        };

        match handle.tx.try_send(actor_msg) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(actor_msg)) => {
                // Actor inbox is full — send backpressure feedback
                let _ = self
                    .out_tx
                    .send(OutboundMessage {
                        channel: reply_channel.to_string(),
                        chat_id: reply_chat_id.to_string(),
                        content: "⏳ Still processing, your message is queued...".to_string(),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    })
                    .await;
                // Now block until space is available
                let handle = self.actors.get(&key_str).unwrap();
                let _ = handle.tx.send(actor_msg).await;
            }
            Err(mpsc::error::TrySendError::Closed(actor_msg)) => {
                // Actor died — remove and create a new one
                self.actors.remove(&key_str);
                let (tx, join_handle) = self.factory.spawn(
                    session_key,
                    reply_channel,
                    reply_chat_id,
                    self.semaphore.clone(),
                    status_indicator,
                );
                let _ = tx.send(actor_msg).await;
                self.actors.insert(
                    key_str,
                    ActorHandle {
                        tx,
                        created_at: Instant::now(),
                        join_handle,
                    },
                );
            }
        }
    }

    /// Remove actors whose tasks have completed.
    pub fn reap_dead_actors(&mut self) {
        self.actors.retain(|key, handle| {
            if handle.is_finished() {
                debug!(session = %key, "reaping completed actor");
                false
            } else {
                true
            }
        });
    }

    /// Cancel a specific session actor.
    pub async fn cancel(&self, session_key: &str) {
        if let Some(handle) = self.actors.get(session_key) {
            let _ = handle.tx.send(ActorMessage::Cancel).await;
        }
    }

    /// Shut down all actors gracefully.
    pub async fn shutdown_all(self) {
        // Drop all senders — actors will exit on recv() returning None
        let handles: Vec<_> = self
            .actors
            .into_values()
            .map(|h| {
                drop(h.tx);
                h.join_handle
            })
            .collect();

        for h in handles {
            let _ = h.await;
        }
    }

    /// Flush buffered messages for a session key (called on `/s` switch).
    /// Returns the number of messages flushed.
    pub async fn flush_pending(&self, session_key: &str) -> usize {
        let messages = self
            .pending_messages
            .lock()
            .await
            .remove(session_key)
            .unwrap_or_default();
        let count = messages.len();
        for msg in messages {
            let _ = self.out_tx.send(msg).await;
        }
        count
    }

    /// Number of active actors.
    pub fn len(&self) -> usize {
        self.actors.len()
    }

    /// Whether there are no active actors.
    pub fn is_empty(&self) -> bool {
        self.actors.is_empty()
    }
}

// ── ActorFactory ────────────────────────────────────────────────────────────

/// Shared resources needed to create per-session actors.
pub struct ActorFactory {
    pub agent_config: AgentConfig,
    pub llm: Arc<dyn LlmProvider>,
    pub llm_for_compaction: Arc<dyn LlmProvider>,
    pub memory: Arc<EpisodeStore>,
    pub system_prompt: Arc<std::sync::RwLock<String>>,
    pub hooks: Option<Arc<HookExecutor>>,
    pub hook_context_template: Option<HookContext>,
    pub session_mgr: Arc<Mutex<SessionManager>>,
    pub out_tx: mpsc::Sender<OutboundMessage>,
    pub spawn_inbound_tx: mpsc::Sender<InboundMessage>,
    pub cron_service: Option<Arc<crew_bus::CronService>>,
    pub tool_registry_factory: Arc<dyn ToolRegistryFactory + Send + Sync>,
    pub pipeline_factory: Option<Arc<dyn PipelineToolFactory + Send + Sync>>,
    pub max_history: Arc<std::sync::atomic::AtomicUsize>,
    pub idle_timeout: Duration,
    pub session_timeout: Duration,
    pub shutdown: Arc<AtomicBool>,
    /// Working directory for SpawnTool.
    pub cwd: std::path::PathBuf,
    /// Provider policy for SpawnTool and PipelineTool.
    pub provider_policy: Option<ToolPolicy>,
    /// Worker system prompt for SpawnTool subagents.
    pub worker_prompt: Option<String>,
    /// Provider router for SpawnTool and PipelineTool.
    pub provider_router: Option<Arc<ProviderRouter>>,
    /// Optional embedder for episodic memory recall.
    pub embedder: Option<Arc<dyn EmbeddingProvider>>,
    /// OminiX client for TTS (if available).
    pub ominix_client: Option<Arc<crew_llm::OminixClient>>,
    /// Active session store — used to check if a session is currently active.
    pub active_sessions: Arc<Mutex<ActiveSessionStore>>,
    /// Pending message buffer — replies from inactive sessions are held here.
    pub pending_messages: PendingMessages,
}

/// Trait for creating per-session ToolRegistry instances.
///
/// This abstracts the complex tool registration logic (builtins, plugins, MCP,
/// policies, etc.) so the actor module doesn't depend on all those details.
pub trait ToolRegistryFactory: Send + Sync {
    /// Create a base ToolRegistry with all non-session-specific tools registered.
    /// The caller will add session-specific tools (MessageTool, SendFileTool, etc.)
    fn create_base_registry(&self) -> ToolRegistry;
}

/// Trait for creating per-session pipeline tool instances.
pub trait PipelineToolFactory: Send + Sync {
    fn create(&self) -> Arc<dyn crew_agent::tools::Tool>;
}

/// ToolRegistryFactory backed by snapshot_excluding() — clones shared tools cheaply.
pub struct SnapshotToolRegistryFactory {
    base: ToolRegistry,
}

impl SnapshotToolRegistryFactory {
    pub fn new(base: ToolRegistry) -> Self {
        Self { base }
    }
}

impl ToolRegistryFactory for SnapshotToolRegistryFactory {
    fn create_base_registry(&self) -> ToolRegistry {
        // Clone all tools (Arc refcount bumps, cheap)
        self.base.snapshot_excluding(&[])
    }
}

impl ActorFactory {
    /// Spawn a new session actor, returning its inbox sender and join handle.
    fn spawn(
        &self,
        session_key: SessionKey,
        channel: &str,
        chat_id: &str,
        semaphore: Arc<Semaphore>,
        status_indicator: Option<Arc<StatusIndicator>>,
    ) -> (mpsc::Sender<ActorMessage>, JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(ACTOR_INBOX_SIZE);

        // Create a per-session proxy channel. ALL outbound messages from this
        // session (tools, final reply, errors) flow through proxy_tx. A
        // forwarding task checks whether this session is active and either
        // delivers immediately or buffers for later.
        let (proxy_tx, proxy_rx) = mpsc::channel::<OutboundMessage>(64);

        // Per-session tools — they write to proxy_tx, not the real out_tx
        let message_tool = MessageTool::with_context(proxy_tx.clone(), channel, chat_id);
        let send_file_tool = SendFileTool::with_context(proxy_tx.clone(), channel, chat_id);

        // Build tool registry with session-specific tools
        let mut tools = self.tool_registry_factory.create_base_registry();
        tools.register(message_tool);
        tools.register(send_file_tool);

        // Voice synthesize tool (if OminiX TTS is available)
        if let Some(ref ominix) = self.ominix_client {
            let tts_tool = VoiceSynthesizeTool::with_context(
                ominix.clone(),
                proxy_tx.clone(),
                channel,
                chat_id,
            );
            tools.register(tts_tool);
        }

        // Spawn tool (per-session context, fully configured)
        let mut spawn_tool = SpawnTool::with_context(
            self.llm.clone(),
            self.memory.clone(),
            self.cwd.clone(),
            self.spawn_inbound_tx.clone(),
            channel,
            chat_id,
        )
        .with_provider_policy(self.provider_policy.clone());
        if let Some(ref prompt) = self.worker_prompt {
            spawn_tool = spawn_tool.with_worker_prompt(prompt.clone());
        }
        if let Some(ref router) = self.provider_router {
            spawn_tool = spawn_tool.with_provider_router(router.clone());
        }
        tools.register(spawn_tool);

        // Cron tool (per-session context)
        if let Some(ref cron_service) = self.cron_service {
            let cron_tool = CronTool::with_context(cron_service.clone(), channel, chat_id);
            tools.register(cron_tool);
        }

        // Pipeline tool (if available)
        if let Some(ref pf) = self.pipeline_factory {
            let pt = pf.create();
            tools.register_arc(pt);
        }

        // Build per-session Agent
        let agent_id = AgentId::new(format!("session-{}", session_key));
        let system_prompt = self
            .system_prompt
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        let mut agent = Agent::new(agent_id, self.llm.clone(), tools, self.memory.clone())
            .with_config(self.agent_config.clone())
            .with_reporter(Arc::new(crew_agent::SilentReporter))
            .with_shutdown(self.shutdown.clone())
            .with_system_prompt(system_prompt);

        if let Some(ref embedder) = self.embedder {
            agent = agent.with_embedder(embedder.clone());
        }
        if let Some(ref hooks) = self.hooks {
            agent = agent.with_hooks(hooks.clone());
        }
        if let Some(ref ctx) = self.hook_context_template {
            agent = agent.with_hook_context(HookContext {
                session_id: Some(session_key.to_string()),
                profile_id: ctx.profile_id.clone(),
            });
        }

        let actor = SessionActor {
            session_key: session_key.clone(),
            channel: channel.to_string(),
            chat_id: chat_id.to_string(),
            inbox: rx,
            agent,
            session_mgr: self.session_mgr.clone(),
            llm_for_compaction: self.llm_for_compaction.clone(),
            out_tx: proxy_tx, // actor sends through proxy, not directly
            status_indicator,
            max_history: self.max_history.clone(),
            idle_timeout: self.idle_timeout,
            session_timeout: self.session_timeout,
            semaphore,
            global_shutdown: self.shutdown.clone(),
            cancelled: Arc::new(AtomicBool::new(false)),
        };

        // Spawn the outbound forwarding task — buffers messages from inactive sessions
        let fwd_session_key = session_key.clone();
        let fwd_out_tx = self.out_tx.clone();
        let fwd_active = self.active_sessions.clone();
        let fwd_pending = self.pending_messages.clone();
        let fwd_channel = channel.to_string();
        let fwd_chat_id = chat_id.to_string();
        tokio::spawn(outbound_forwarder(
            proxy_rx,
            fwd_out_tx,
            fwd_session_key,
            fwd_channel,
            fwd_chat_id,
            fwd_active,
            fwd_pending,
        ));

        let join_handle = tokio::spawn(actor.run());

        info!(session = %session_key, channel, chat_id, "spawned session actor");
        (tx, join_handle)
    }
}

/// Forwarding task: reads from the session's proxy channel and either delivers
/// messages directly (if this session is active) or buffers them.
async fn outbound_forwarder(
    mut proxy_rx: mpsc::Receiver<OutboundMessage>,
    out_tx: mpsc::Sender<OutboundMessage>,
    session_key: SessionKey,
    channel: String,
    chat_id: String,
    active_sessions: Arc<Mutex<ActiveSessionStore>>,
    pending_messages: PendingMessages,
) {
    let my_topic = session_key.topic().unwrap_or("").to_string();
    let base_key = session_key.base_key().to_string();
    let key_str = session_key.to_string();

    while let Some(msg) = proxy_rx.recv().await {
        let active_topic = active_sessions
            .lock()
            .await
            .get_active_topic(&base_key)
            .to_string();

        if my_topic == active_topic {
            // Session is active — deliver immediately
            let _ = out_tx.send(msg).await;
        } else {
            // Session is inactive — buffer the message
            let mut pending = pending_messages.lock().await;
            let buf = pending.entry(key_str.clone()).or_default();
            let is_first = buf.is_empty();
            if buf.len() < MAX_PENDING_PER_SESSION {
                buf.push(msg);
            } else {
                warn!(session = %session_key, "pending buffer full, dropping message");
            }
            drop(pending); // release lock before sending notification

            if is_first {
                let topic_label = if my_topic.is_empty() {
                    "(default)"
                } else {
                    &my_topic
                };
                let _ = out_tx
                    .send(OutboundMessage {
                        channel: channel.clone(),
                        chat_id: chat_id.clone(),
                        content: format!("📌 {topic_label} finished. /s {topic_label} to view."),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    })
                    .await;
            }
        }
    }
}

// ── SessionActor ────────────────────────────────────────────────────────────

/// Long-lived task that processes all messages for one session.
struct SessionActor {
    session_key: SessionKey,
    channel: String,
    chat_id: String,

    inbox: mpsc::Receiver<ActorMessage>,

    agent: Agent,

    session_mgr: Arc<Mutex<SessionManager>>,
    llm_for_compaction: Arc<dyn LlmProvider>,

    out_tx: mpsc::Sender<OutboundMessage>,

    status_indicator: Option<Arc<StatusIndicator>>,
    max_history: Arc<std::sync::atomic::AtomicUsize>,

    idle_timeout: Duration,
    session_timeout: Duration,
    semaphore: Arc<Semaphore>,
    /// Global shutdown flag (Ctrl+C, etc.)
    global_shutdown: Arc<AtomicBool>,
    /// Per-actor cancellation flag (only affects this session)
    cancelled: Arc<AtomicBool>,
}

impl SessionActor {
    async fn run(mut self) {
        loop {
            tokio::select! {
                msg = self.inbox.recv() => {
                    match msg {
                        Some(ActorMessage::Inbound { message, image_media }) => {
                            self.process_inbound(message, image_media).await;
                        }
                        Some(ActorMessage::Cancel) => {
                            debug!(session = %self.session_key, "cancel requested");
                            self.cancelled.store(true, Ordering::Release);
                        }
                        None => {
                            // All senders dropped
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(self.idle_timeout) => {
                    debug!(session = %self.session_key, "idle timeout, shutting down actor");
                    break;
                }
            }

            if self.global_shutdown.load(Ordering::Acquire)
                || self.cancelled.load(Ordering::Acquire)
            {
                break;
            }
        }

        debug!(session = %self.session_key, "actor exiting");
    }

    async fn process_inbound(&mut self, inbound: InboundMessage, image_media: Vec<String>) {
        // Acquire concurrency permit
        let _permit = match self.semaphore.acquire().await {
            Ok(p) => p,
            Err(_) => return, // semaphore closed
        };

        // Get conversation history
        let max_history = self.max_history.load(Ordering::Acquire);
        let history: Vec<Message> = {
            let mut mgr = self.session_mgr.lock().await;
            let session = mgr.get_or_create(&self.session_key);
            session.get_history(max_history).to_vec()
        };

        // Token tracker for status indicator
        let token_tracker = Arc::new(TokenTracker::new());

        // Start status indicator
        let status_handle = self.status_indicator.as_ref().map(|si| {
            let voice_transcript = inbound
                .metadata
                .get("voice_transcript")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            si.start(
                self.chat_id.clone(),
                &inbound.content,
                Arc::clone(&token_tracker),
                voice_transcript,
            )
        });

        // Process through agent (potentially long LLM call)
        let result = tokio::time::timeout(
            self.session_timeout,
            self.agent.process_message_tracked(
                &inbound.content,
                &history,
                image_media,
                &token_tracker,
            ),
        )
        .await;

        // Stop status indicator
        if let Some(handle) = status_handle {
            handle.stop().await;
        }

        match result {
            Ok(Ok(conv_response)) => {
                // Save messages to session
                {
                    let mut mgr = self.session_mgr.lock().await;
                    let user_msg = Message {
                        role: MessageRole::User,
                        content: inbound.content.clone(),
                        media: vec![],
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                        timestamp: Utc::now(),
                    };
                    if let Err(e) = mgr.add_message(&self.session_key, user_msg).await {
                        warn!(session = %self.session_key, error = %e, "failed to persist user message");
                    }

                    // Auto-generate summary from first user message
                    {
                        let session = mgr.get_or_create(&self.session_key);
                        if session.summary.is_none() && !inbound.content.trim().is_empty() {
                            let summary: String = inbound.content.chars().take(100).collect();
                            session.summary = Some(summary);
                        }
                    }

                    if !conv_response.content.is_empty() {
                        let assistant_msg = Message {
                            role: MessageRole::Assistant,
                            content: conv_response.content.clone(),
                            media: vec![],
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                            timestamp: Utc::now(),
                        };
                        if let Err(e) = mgr.add_message(&self.session_key, assistant_msg).await {
                            warn!(session = %self.session_key, error = %e, "failed to persist assistant message");
                        }
                    }

                    // Compact if needed
                    if let Err(e) = crate::compaction::maybe_compact(
                        &mut mgr,
                        &self.session_key,
                        &*self.llm_for_compaction,
                    )
                    .await
                    {
                        warn!("session compaction failed: {e}");
                    }
                }

                // Send reply — always goes to this actor's chat (no race!)
                let content = strip_think_tags(&conv_response.content);

                let is_cron = inbound.channel == "system" && inbound.sender_id == "cron";
                let is_silent = content.trim().is_empty()
                    || content.contains("[SILENT]")
                    || content.contains("[NO_CHANGE]");

                if !(is_cron && is_silent) {
                    let display_content = content
                        .trim_start()
                        .strip_prefix("[SILENT]")
                        .or_else(|| content.trim_start().strip_prefix("[NO_CHANGE]"))
                        .unwrap_or(&content)
                        .to_string();

                    let _ = self
                        .out_tx
                        .send(OutboundMessage {
                            channel: self.channel.clone(),
                            chat_id: self.chat_id.clone(),
                            content: display_content,
                            reply_to: None,
                            media: vec![],
                            metadata: serde_json::json!({}),
                        })
                        .await;
                }
            }
            Ok(Err(e)) => {
                tracing::error!(session = %self.session_key, error = %e, "agent processing failed");
                let _ = self
                    .out_tx
                    .send(OutboundMessage {
                        channel: self.channel.clone(),
                        chat_id: self.chat_id.clone(),
                        content: format!("Error: {e}"),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    })
                    .await;
            }
            Err(_) => {
                tracing::error!(session = %self.session_key, "session processing timed out");
                let _ = self
                    .out_tx
                    .send(OutboundMessage {
                        channel: self.channel.clone(),
                        chat_id: self.chat_id.clone(),
                        content: "Processing timed out. Please try again.".to_string(),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    })
                    .await;
            }
        }
    }
}

/// Strip `<think>...</think>` blocks that some models embed inline.
fn strip_think_tags(s: &str) -> String {
    let mut result = s.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result.replace_range(start..start + end + "</think>".len(), "");
        } else {
            result.truncate(start);
            break;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_think_tags() {
        assert_eq!(strip_think_tags("hello"), "hello");
        assert_eq!(strip_think_tags("<think>hmm</think>hello"), "hello");
        assert_eq!(
            strip_think_tags("before<think>hmm</think>after"),
            "beforeafter"
        );
        assert_eq!(strip_think_tags("<think>unclosed"), "");
    }

    // Integration tests require a real LLM provider — tested via gateway integration tests.
}
