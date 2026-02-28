//! Persona service: periodically generates a communication style guide via LLM
//! based on recent chat history. The generated persona is written to `persona.md`
//! in the data directory and injected into the agent's system prompt.
//!
//! This is a system-internal service — not exposed to users via cron or any tool.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use crew_bus::SessionManager;
use crew_core::{Message, MessageRole};
use crew_llm::{ChatConfig, LlmProvider, ToolChoice};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// Default persona refresh interval: 6 hours.
pub const DEFAULT_INTERVAL_SECS: u64 = 6 * 3600;

/// Initial delay before first persona generation (let sessions accumulate).
const INITIAL_DELAY_SECS: u64 = 60;

/// Maximum messages to sample per session.
const MAX_MSGS_PER_SESSION: usize = 20;

/// Maximum total messages to feed to the LLM.
const MAX_TOTAL_MSGS: usize = 100;

const META_PROMPT: &str = "\
Based on the recent conversations below, generate a concise communication style guide \
for an AI assistant. Analyze how users communicate (casual/formal, language preference, \
emoji usage, humor level, topics of interest) and create guidelines that match their style.

Output ONLY the style guide as bullet points (3-8 items), no preamble or explanation. Example format:
- Be casual and use humor freely
- Respond in Chinese when the user writes in Chinese
- Keep answers short and punchy
- Use emoji occasionally";

/// Background service that generates a communication persona from chat history.
pub struct PersonaService {
    data_dir: PathBuf,
    session_mgr: Arc<Mutex<SessionManager>>,
    llm: Arc<dyn LlmProvider>,
    interval_secs: u64,
    running: AtomicBool,
    timer_handle: tokio::sync::Mutex<Option<JoinHandle<()>>>,
}

impl PersonaService {
    pub fn new(
        data_dir: PathBuf,
        session_mgr: Arc<Mutex<SessionManager>>,
        llm: Arc<dyn LlmProvider>,
        interval_secs: u64,
    ) -> Self {
        Self {
            data_dir,
            session_mgr,
            llm,
            interval_secs,
            running: AtomicBool::new(false),
            timer_handle: tokio::sync::Mutex::new(None),
        }
    }

    /// Start the persona generation loop.
    ///
    /// `on_update` is called with the new persona text whenever it changes,
    /// allowing the caller to rebuild the system prompt.
    pub fn start<F>(self: &Arc<Self>, on_update: F)
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        self.running.store(true, Ordering::Relaxed);
        let this = Arc::clone(self);
        let on_update = Arc::new(on_update);

        let handle = tokio::spawn(async move {
            info!(
                interval_secs = this.interval_secs,
                "persona service started"
            );

            // Initial delay — let the gateway warm up and accumulate some messages
            tokio::time::sleep(std::time::Duration::from_secs(INITIAL_DELAY_SECS)).await;

            loop {
                if !this.running.load(Ordering::Relaxed) {
                    break;
                }

                if let Some(persona) = this.tick().await {
                    on_update(persona);
                }

                // Sleep until next interval
                tokio::time::sleep(std::time::Duration::from_secs(this.interval_secs)).await;
                if !this.running.load(Ordering::Relaxed) {
                    break;
                }
            }
        });

        let this2 = Arc::clone(self);
        tokio::spawn(async move {
            *this2.timer_handle.lock().await = Some(handle);
        });
    }

    /// Stop the persona generation loop.
    pub async fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        let mut handle = self.timer_handle.lock().await;
        if let Some(h) = handle.take() {
            h.abort();
        }
        info!("persona service stopped");
    }

    /// Single tick: collect conversations, call LLM, write persona.md.
    /// Returns the generated persona text if successful.
    async fn tick(&self) -> Option<String> {
        let conversations = self.collect_recent_conversations().await;
        if conversations.is_empty() {
            debug!("no conversations found for persona generation");
            return None;
        }

        let user_content = format!("{META_PROMPT}\n\nRecent conversations:\n\n{conversations}");

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a communication style analyzer. Output only bullet points."
                    .into(),
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                timestamp: Utc::now(),
            },
            Message {
                role: MessageRole::User,
                content: user_content,
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                timestamp: Utc::now(),
            },
        ];

        let config = ChatConfig {
            max_tokens: Some(1024),
            temperature: Some(0.7),
            tool_choice: ToolChoice::None,
            stop_sequences: vec![],
        };

        match self.llm.chat(&messages, &[], &config).await {
            Ok(response) => {
                let persona = response.content.unwrap_or_default().trim().to_string();
                if persona.is_empty() {
                    warn!("LLM returned empty persona");
                    return None;
                }

                // Write to persona.md
                let path = self.data_dir.join("persona.md");
                if let Err(e) = tokio::fs::write(&path, &persona).await {
                    warn!("failed to write persona.md: {e}");
                    return None;
                }

                info!("persona updated ({} chars)", persona.len());
                Some(persona)
            }
            Err(e) => {
                warn!("persona generation LLM call failed: {e}");
                None
            }
        }
    }

    /// Collect recent user↔assistant messages across all sessions.
    async fn collect_recent_conversations(&self) -> String {
        let mgr = self.session_mgr.lock().await;
        let session_ids: Vec<String> = mgr
            .list_sessions()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        drop(mgr);

        let mut out = String::new();
        let mut total = 0usize;

        for session_id in &session_ids {
            let mut mgr = self.session_mgr.lock().await;
            let key = crew_core::SessionKey(session_id.clone());
            let session = mgr.get_or_create(&key);
            let history = session.get_history(MAX_MSGS_PER_SESSION);

            for msg in history {
                let role = match msg.role {
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    _ => continue,
                };

                if msg.content.trim().is_empty() {
                    continue;
                }

                // Truncate very long messages
                let content = if msg.content.len() > 500 {
                    &msg.content[..500]
                } else {
                    &msg.content
                };

                out.push_str(role);
                out.push_str(": ");
                out.push_str(content);
                if msg.content.len() > 500 {
                    out.push_str("...");
                }
                out.push_str("\n\n");

                total += 1;
                if total >= MAX_TOTAL_MSGS {
                    return out;
                }
            }
        }

        out
    }

    /// Read an existing persona.md file (for startup injection).
    pub fn read_persona(data_dir: &Path) -> Option<String> {
        let path = data_dir.join("persona.md");
        match std::fs::read_to_string(&path) {
            Ok(content) if !content.trim().is_empty() => Some(content.trim().to_string()),
            _ => None,
        }
    }
}
