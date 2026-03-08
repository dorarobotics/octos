//! Progressive streaming reporter for messaging channels.
//!
//! Bridges the synchronous `ProgressReporter` trait to async channel I/O,
//! enabling real-time LLM text streaming to Telegram, WhatsApp, etc.
//! Text is accumulated and the channel message is edited at a throttled rate.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crew_agent::progress::{ProgressEvent, ProgressReporter};
use crew_bus::Channel;
use crew_core::OutboundMessage;
use tokio::sync::mpsc;
use tracing::warn;

/// Events forwarded from the synchronous reporter to the async forwarder.
#[derive(Debug)]
pub enum StreamProgressEvent {
    /// A chunk of streaming text from the LLM.
    Chunk { text: String, iteration: u32 },
    /// Streaming finished for this iteration.
    StreamDone { iteration: u32 },
    /// A tool is about to run.
    ToolStarted { name: String },
    /// A tool completed.
    ToolCompleted { name: String, success: bool },
}

/// A `ProgressReporter` that forwards stream events through an unbounded channel.
///
/// Because `ProgressReporter::report()` is synchronous, we use `unbounded_send()`
/// which never blocks. The receiving async task handles actual channel I/O.
pub struct ChannelStreamReporter {
    tx: mpsc::UnboundedSender<StreamProgressEvent>,
}

impl ChannelStreamReporter {
    pub fn new(tx: mpsc::UnboundedSender<StreamProgressEvent>) -> Self {
        Self { tx }
    }
}

impl ProgressReporter for ChannelStreamReporter {
    fn report(&self, event: ProgressEvent) {
        let mapped = match event {
            ProgressEvent::StreamChunk { text, iteration } => {
                StreamProgressEvent::Chunk { text, iteration }
            }
            ProgressEvent::StreamDone { iteration } => {
                StreamProgressEvent::StreamDone { iteration }
            }
            ProgressEvent::ToolStarted { name, .. } => StreamProgressEvent::ToolStarted { name },
            ProgressEvent::ToolCompleted {
                name, success, ..
            } => StreamProgressEvent::ToolCompleted { name, success },
            _ => return,
        };
        let _ = self.tx.send(mapped);
    }
}

/// Minimum interval between message edits (avoids API rate limits).
const EDIT_THROTTLE: Duration = Duration::from_millis(1000);

/// Strip `<think>...</think>` blocks from streaming buffer.
/// Handles partial tags (open `<think>` not yet closed) by hiding from that point.
fn strip_think_from_buffer(buf: &str) -> String {
    let mut result = String::new();
    let mut rest = buf;

    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            rest = &after[end + "</think>".len()..];
        } else {
            // Unclosed <think> — hide everything from here (still streaming)
            return result.trim_end().to_string();
        }
    }
    result.push_str(rest);
    result
}

/// Result of the stream forwarder — returns the message ID if streaming happened.
pub struct StreamResult {
    /// The platform message ID of the streamed message, if any was sent.
    pub message_id: Option<String>,
    /// The full accumulated text that was streamed.
    pub text: String,
}

/// Run the stream forwarder task.
///
/// Receives `StreamProgressEvent`s and progressively edits a channel message
/// with accumulated text. Returns once the sender is dropped (agent completes).
///
/// `cancel_status` — if provided, stops the status indicator when first chunk arrives.
pub async fn run_stream_forwarder(
    mut rx: mpsc::UnboundedReceiver<StreamProgressEvent>,
    channel: Arc<dyn Channel>,
    chat_id: String,
    cancel_status: Option<Arc<std::sync::atomic::AtomicBool>>,
    status_msg_id: Option<Arc<tokio::sync::Mutex<Option<String>>>>,
) -> StreamResult {
    let mut buffer = String::new();
    let mut message_id: Option<String> = None;
    let mut last_edit = Instant::now() - EDIT_THROTTLE; // allow immediate first edit
    let mut first_chunk = true;

    while let Some(event) = rx.recv().await {
        match event {
            StreamProgressEvent::Chunk { text, .. } => {
                if first_chunk {
                    first_chunk = false;
                    // Cancel the "✦ Thinking..." status indicator
                    if let Some(ref cancel) = cancel_status {
                        cancel.store(true, std::sync::atomic::Ordering::Release);
                    }
                    // Delete the status message
                    if let Some(ref msg_id_lock) = status_msg_id {
                        let mid = msg_id_lock.lock().await.take();
                        if let Some(ref mid) = mid {
                            let _ = channel.delete_message(&chat_id, mid).await;
                        }
                    }
                }

                buffer.push_str(&text);

                // Throttled edit — strip <think> blocks before showing to user
                if last_edit.elapsed() >= EDIT_THROTTLE && !buffer.is_empty() {
                    let visible = strip_think_from_buffer(&buffer);
                    if !visible.is_empty() {
                        flush_to_channel(&channel, &chat_id, &visible, &mut message_id).await;
                        last_edit = Instant::now();
                    }
                }
            }
            StreamProgressEvent::StreamDone { .. } => {
                // Flush remaining buffer — strip think tags
                if !buffer.is_empty() {
                    let visible = strip_think_from_buffer(&buffer);
                    if !visible.is_empty() {
                        flush_to_channel(&channel, &chat_id, &visible, &mut message_id).await;
                    }
                }
            }
            StreamProgressEvent::ToolStarted { name } => {
                // Flush text before tool status
                if !buffer.is_empty() {
                    buffer.push_str(&format!("\n\n⚙ `{name}`..."));
                    flush_to_channel(&channel, &chat_id, &buffer, &mut message_id).await;
                    last_edit = Instant::now();
                }
            }
            StreamProgressEvent::ToolCompleted { name, success } => {
                let icon = if success { "✓" } else { "✗" };
                // Update tool status in the existing message
                if !buffer.is_empty() {
                    // Replace the "⚙ `tool`..." with the result
                    let pending = format!("⚙ `{name}`...");
                    let completed = format!("{icon} `{name}`");
                    if buffer.contains(&pending) {
                        buffer = buffer.replace(&pending, &completed);
                    }
                    flush_to_channel(&channel, &chat_id, &buffer, &mut message_id).await;
                    last_edit = Instant::now();
                }
            }
        }
    }

    // Final flush
    if !buffer.is_empty() && message_id.is_none() {
        let visible = strip_think_from_buffer(&buffer);
        if !visible.is_empty() {
            flush_to_channel(&channel, &chat_id, &visible, &mut message_id).await;
        }
    }

    StreamResult {
        message_id,
        text: buffer,
    }
}

/// Send or edit the streaming message on the channel.
async fn flush_to_channel(
    channel: &Arc<dyn Channel>,
    chat_id: &str,
    text: &str,
    message_id: &mut Option<String>,
) {
    if let Some(mid) = message_id.as_ref() {
        // Edit existing message
        if let Err(e) = channel.edit_message(chat_id, mid, text).await {
            warn!("stream edit failed: {e}");
        }
    } else {
        // Send new message and capture its ID
        let msg = OutboundMessage {
            channel: channel.name().to_string(),
            chat_id: chat_id.to_string(),
            content: text.to_string(),
            reply_to: None,
            media: vec![],
            metadata: serde_json::json!({}),
        };
        match channel.send_with_id(&msg).await {
            Ok(Some(mid)) => {
                *message_id = Some(mid);
            }
            Ok(None) => {
                // Channel doesn't support edit — won't stream further
            }
            Err(e) => {
                warn!("stream send failed: {e}");
            }
        }
    }
}
