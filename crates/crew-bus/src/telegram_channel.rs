//! Telegram channel using teloxide long polling.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use chrono::Utc;
use crew_core::{InboundMessage, OutboundMessage};
use eyre::{Result, WrapErr};
use teloxide::prelude::*;
use teloxide::types::{ChatId, UpdateKind};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::channel::Channel;

pub struct TelegramChannel {
    bot: Bot,
    allowed_senders: HashSet<String>,
    shutdown: Arc<AtomicBool>,
}

impl TelegramChannel {
    pub fn new(token: &str, allowed_senders: Vec<String>, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            bot: Bot::new(token),
            allowed_senders: allowed_senders.into_iter().collect(),
            shutdown,
        }
    }

    fn check_allowed(&self, sender_id: &str) -> bool {
        if self.allowed_senders.is_empty() {
            return true;
        }
        if self.allowed_senders.contains(sender_id) {
            return true;
        }
        // Support "id|username" compound format (nanobot pattern)
        sender_id
            .split('|')
            .any(|part| self.allowed_senders.contains(part))
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn start(&self, inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        use futures::StreamExt;
        use teloxide::update_listeners::{AsUpdateStream, polling_default};

        info!("Starting Telegram channel (long polling)");

        let mut listener = polling_default(self.bot.clone()).await;
        let stream = listener.as_stream();
        tokio::pin!(stream);

        while let Some(result) = stream.next().await {
            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            let update = match result {
                Ok(update) => update,
                Err(e) => {
                    warn!("Telegram polling error: {e}");
                    continue;
                }
            };

            if let UpdateKind::Message(msg) = update.kind {
                let text = match msg.text() {
                    Some(t) => t.to_string(),
                    None => continue,
                };

                let sender_id = msg
                    .from
                    .as_ref()
                    .map(|u| {
                        let id = u.id.to_string();
                        match &u.username {
                            Some(name) => format!("{id}|{name}"),
                            None => id,
                        }
                    })
                    .unwrap_or_default();

                if !self.check_allowed(&sender_id) {
                    continue;
                }

                let inbound = InboundMessage {
                    channel: "telegram".into(),
                    sender_id,
                    chat_id: msg.chat.id.0.to_string(),
                    content: text,
                    timestamp: Utc::now(),
                    media: vec![],
                    metadata: serde_json::json!({}),
                };

                if inbound_tx.send(inbound).await.is_err() {
                    break;
                }
            }
        }

        info!("Telegram channel stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        let chat_id: i64 = msg
            .chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {}", msg.chat_id))?;

        self.bot
            .send_message(ChatId(chat_id), &msg.content)
            .await
            .wrap_err("failed to send Telegram message")?;

        Ok(())
    }

    fn is_allowed(&self, sender_id: &str) -> bool {
        self.check_allowed(sender_id)
    }

    async fn stop(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_channel(allowed: Vec<&str>) -> TelegramChannel {
        TelegramChannel {
            bot: Bot::new("test:token"),
            allowed_senders: allowed.into_iter().map(String::from).collect(),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn test_is_allowed_empty_list() {
        let ch = make_channel(vec![]);
        assert!(ch.is_allowed("anyone"));
        assert!(ch.is_allowed("12345"));
    }

    #[test]
    fn test_is_allowed_matching() {
        let ch = make_channel(vec!["12345", "67890"]);
        assert!(ch.is_allowed("12345"));
        assert!(ch.is_allowed("67890"));
        assert!(!ch.is_allowed("99999"));
    }

    #[test]
    fn test_is_allowed_compound_id() {
        let ch = make_channel(vec!["12345", "johndoe"]);
        assert!(ch.is_allowed("12345|johndoe"));
        assert!(ch.is_allowed("12345|other"));
        assert!(ch.is_allowed("99999|johndoe"));
        assert!(!ch.is_allowed("99999|other"));
    }
}
