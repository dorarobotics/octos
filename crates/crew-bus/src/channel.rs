//! Channel trait and manager for gateway message routing.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use crew_core::{InboundMessage, OutboundMessage};
use eyre::Result;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::bus::BusPublisher;
use crate::coalesce::{ChunkConfig, split_message};

/// A message channel (CLI, Telegram, Discord, etc.).
#[async_trait]
pub trait Channel: Send + Sync {
    /// Channel name used for routing (e.g. "cli", "telegram").
    fn name(&self) -> &str;

    /// Start listening for messages. Long-running — sends inbound messages via tx.
    async fn start(&self, inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()>;

    /// Send an outbound message through this channel.
    async fn send(&self, msg: &OutboundMessage) -> Result<()>;

    /// Check if a sender is allowed to use this channel.
    fn is_allowed(&self, _sender_id: &str) -> bool {
        true
    }

    /// Maximum message length in characters for this channel.
    /// Messages exceeding this limit are automatically split at natural boundaries.
    fn max_message_length(&self) -> usize {
        4000
    }

    /// Stop the channel gracefully.
    async fn stop(&self) -> Result<()> {
        Ok(())
    }

    /// Send a typing/composing indicator. Platforms that don't support it return Ok(()).
    async fn send_typing(&self, _chat_id: &str) -> Result<()> {
        Ok(())
    }

    /// Send a "listening" / recording-voice indicator (for voice transcription).
    /// Falls back to typing indicator by default.
    async fn send_listening(&self, chat_id: &str) -> Result<()> {
        self.send_typing(chat_id).await
    }

    /// Send a message and return its platform message ID (for later editing/deletion).
    /// Default: delegates to `send()` and returns None.
    async fn send_with_id(&self, msg: &OutboundMessage) -> Result<Option<String>> {
        self.send(msg).await?;
        Ok(None)
    }

    /// Edit an existing message by platform message ID.
    async fn edit_message(
        &self,
        _chat_id: &str,
        _message_id: &str,
        _new_content: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Delete a message by platform message ID.
    async fn delete_message(&self, _chat_id: &str, _message_id: &str) -> Result<()> {
        Ok(())
    }

    /// Edit a message with platform-specific metadata (e.g. inline keyboards).
    /// Default: ignores metadata and delegates to `edit_message()`.
    async fn edit_message_with_metadata(
        &self,
        chat_id: &str,
        message_id: &str,
        new_content: &str,
        _metadata: &serde_json::Value,
    ) -> Result<()> {
        self.edit_message(chat_id, message_id, new_content).await
    }
}

/// Manages registered channels and dispatches outbound messages.
pub struct ChannelManager {
    channels: HashMap<String, Arc<dyn Channel>>,
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelManager {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    pub fn register(&mut self, channel: Arc<dyn Channel>) {
        self.channels.insert(channel.name().to_string(), channel);
    }

    /// Start all channels and the outbound dispatcher.
    /// Consumes the BusPublisher to own the outbound receiver.
    pub async fn start_all(&self, publisher: BusPublisher) -> Result<()> {
        // Decompose publisher: drop its in_tx so channels are the sole senders
        let inbound_tx = publisher.inbound_sender();
        let (publisher_in_tx, mut out_rx) = publisher.into_parts();
        drop(publisher_in_tx); // Drop publisher's own in_tx to prevent leak

        // Spawn each channel's listener
        for channel in self.channels.values() {
            let ch = Arc::clone(channel);
            let tx = inbound_tx.clone();
            tokio::spawn(async move {
                let name = ch.name().to_string();
                match ch.start(tx).await {
                    Ok(()) => warn!(channel = %name, "Channel listener exited cleanly (may need restart)"),
                    Err(e) => error!(channel = %name, "Channel stopped with error: {e}"),
                }
            });
        }

        // Drop extra inbound sender so bus closes when all channels stop
        drop(inbound_tx);

        // Outbound dispatcher — routes messages to the correct channel
        let channels = self.channels.clone();
        let dispatcher_handle = tokio::spawn(async move {
            while let Some(msg) = out_rx.recv().await {
                if let Some(channel) = channels.get(&msg.channel) {
                    if !msg.media.is_empty() {
                        // File attachments: send directly without chunking
                        if let Err(e) = channel.send(&msg).await {
                            error!(
                                channel = msg.channel,
                                "Failed to send outbound file message: {e}",
                            );
                        }
                    } else {
                        // Text message: chunk and send
                        let config = ChunkConfig {
                            max_chars: channel.max_message_length(),
                        };
                        let chunks = split_message(&msg.content, &config);
                        let total = chunks.len();
                        for (i, chunk) in chunks.into_iter().enumerate() {
                            let mut chunk_msg = msg.clone();
                            chunk_msg.content = chunk;
                            if let Err(e) = channel.send(&chunk_msg).await {
                                error!(
                                    channel = msg.channel,
                                    chunk = i + 1,
                                    total = total,
                                    "Failed to send outbound chunk: {e}",
                                );
                                break;
                            }
                        }
                    }
                } else {
                    error!(
                        channel = msg.channel,
                        "No channel registered for outbound message"
                    );
                }
            }
            info!("Outbound dispatcher stopped");
        });

        // Monitor dispatcher for panics
        tokio::spawn(async move {
            if let Err(e) = dispatcher_handle.await {
                error!("CRITICAL: outbound dispatcher panicked: {e}");
            }
        });

        Ok(())
    }

    /// Get a channel by name, for direct access (typing indicators, message editing).
    pub fn get_channel(&self, name: &str) -> Option<Arc<dyn Channel>> {
        self.channels.get(name).cloned()
    }

    pub async fn stop_all(&self) -> Result<()> {
        let mut errors = Vec::new();
        for channel in self.channels.values() {
            if let Err(e) = channel.stop().await {
                errors.push(format!("{}: {e}", channel.name()));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(eyre::eyre!("failed to stop channels: {}", errors.join(", ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct MockChannel {
        channel_name: String,
        sent: Arc<Mutex<Vec<String>>>,
        start_count: Arc<AtomicUsize>,
    }

    impl MockChannel {
        fn new(name: &str) -> Self {
            Self {
                channel_name: name.to_string(),
                sent: Arc::new(Mutex::new(vec![])),
                start_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Channel for MockChannel {
        fn name(&self) -> &str {
            &self.channel_name
        }

        async fn start(&self, _inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()> {
            self.start_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn send(&self, msg: &OutboundMessage) -> Result<()> {
            self.sent.lock().await.push(msg.content.clone());
            Ok(())
        }
    }

    #[test]
    fn test_register_channels() {
        let mut mgr = ChannelManager::new();
        mgr.register(Arc::new(MockChannel::new("ch1")));
        mgr.register(Arc::new(MockChannel::new("ch2")));
        assert_eq!(mgr.channels.len(), 2);
    }

    #[tokio::test]
    async fn test_dispatch_outbound() {
        let mock = Arc::new(MockChannel::new("test"));
        let sent = Arc::clone(&mock.sent);

        let mut mgr = ChannelManager::new();
        mgr.register(mock);

        let (agent, publisher) = crate::bus::create_bus();
        mgr.start_all(publisher).await.unwrap();

        agent
            .send_outbound(OutboundMessage {
                channel: "test".into(),
                chat_id: "c1".into(),
                content: "hello from agent".into(),
                reply_to: None,
                media: vec![],
                metadata: serde_json::json!({}),
            })
            .await
            .unwrap();

        // Drop agent handle to close the outbound channel
        drop(agent);

        // Give dispatcher time to process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let messages = sent.lock().await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "hello from agent");
    }

    #[tokio::test]
    async fn test_inbound_from_channel() {
        let (mut agent, publisher) = crate::bus::create_bus();
        let tx = publisher.inbound_sender();

        tx.send(InboundMessage {
            channel: "mock".into(),
            sender_id: "user1".into(),
            chat_id: "chat1".into(),
            content: "user says hi".into(),
            timestamp: Utc::now(),
            media: vec![],
            metadata: serde_json::json!({}),
        })
        .await
        .unwrap();

        let msg = agent.recv_inbound().await.unwrap();
        assert_eq!(msg.content, "user says hi");
    }
}
