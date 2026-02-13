//! Session compaction: summarize old messages to keep sessions manageable.

use chrono::Utc;
use crew_bus::SessionManager;
use crew_core::{Message, MessageRole, SessionKey};
use crew_llm::{ChatConfig, LlmProvider};
use eyre::Result;
use tracing::debug;

/// Minimum messages before compaction triggers.
const COMPACTION_THRESHOLD: usize = 40;

/// Number of recent messages to keep intact (not summarized).
const KEEP_RECENT: usize = 10;

/// Compact a session if it exceeds the threshold.
///
/// Summarizes older messages into a single system message using the LLM,
/// keeping the most recent messages intact. Returns `true` if compaction occurred.
pub async fn maybe_compact(
    session_mgr: &mut SessionManager,
    key: &SessionKey,
    llm: &dyn LlmProvider,
) -> Result<bool> {
    let session = session_mgr.get_or_create(key);
    let total = session.messages.len();

    if total < COMPACTION_THRESHOLD {
        return Ok(false);
    }

    let to_summarize = total - KEEP_RECENT;
    debug!(session = %key, total, to_summarize, "compacting session");

    // Build the text to summarize
    let mut summary_input = String::new();
    for msg in &session.messages[..to_summarize] {
        let role = match msg.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::System => "System",
            MessageRole::Tool => "Tool",
        };
        summary_input.push_str(&format!("{role}: {}\n\n", msg.content));
    }

    // Ask the LLM to summarize
    let summarize_prompt = format!(
        "Summarize the following conversation concisely. \
         Preserve key facts, decisions, and context needed to continue the conversation. \
         Keep it under 500 words.\n\n{}",
        summary_input
    );

    let messages = vec![Message {
        role: MessageRole::User,
        content: summarize_prompt,
        media: vec![],
        tool_calls: None,
        tool_call_id: None,
        timestamp: Utc::now(),
    }];

    let config = ChatConfig {
        max_tokens: Some(1024),
        temperature: Some(0.0),
        ..Default::default()
    };

    let response = llm.chat(&messages, &[], &config).await?;
    let summary = response
        .content
        .unwrap_or_else(|| "[Summary unavailable]".to_string());

    // Replace old messages with a summary message + keep recent
    let session = session_mgr.get_or_create(key);
    let recent: Vec<Message> = session.messages[to_summarize..].to_vec();

    session.messages.clear();
    session.messages.push(Message {
        role: MessageRole::System,
        content: format!("[Conversation summary]\n{summary}"),
        media: vec![],
        tool_calls: None,
        tool_call_id: None,
        timestamp: Utc::now(),
    });
    session.messages.extend(recent);
    session.updated_at = Utc::now();

    // Rewrite the JSONL file
    session_mgr.rewrite(key)?;

    debug!(
        session = %key,
        before = total,
        after = KEEP_RECENT + 1,
        "session compacted"
    );

    Ok(true)
}
