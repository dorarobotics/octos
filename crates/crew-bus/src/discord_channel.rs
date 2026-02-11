//! Discord channel using serenity gateway + REST API.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use chrono::Utc;
use crew_core::{InboundMessage, OutboundMessage};
use eyre::{Result, WrapErr};
use serenity::all::{Context, EventHandler, GatewayIntents, Http, Message as DiscordMessage, Ready};
use serenity::Client;
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::channel::Channel;

pub struct DiscordChannel {
    token: String,
    http: Arc<Http>,
    allowed_senders: HashSet<String>,
    shutdown: Arc<AtomicBool>,
}

impl DiscordChannel {
    pub fn new(token: &str, allowed_senders: Vec<String>, shutdown: Arc<AtomicBool>) -> Self {
        let http = Arc::new(Http::new(token));
        Self {
            token: token.to_string(),
            http,
            allowed_senders: allowed_senders.into_iter().collect(),
            shutdown,
        }
    }
}

/// Internal handler that forwards Discord messages to the inbound bus.
struct Handler {
    inbound_tx: mpsc::Sender<InboundMessage>,
    allowed_senders: HashSet<String>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, _ctx: Context, msg: DiscordMessage) {
        if msg.author.bot {
            return;
        }

        let sender_id = msg.author.id.to_string();

        if !self.allowed_senders.is_empty() && !self.allowed_senders.contains(&sender_id) {
            return;
        }

        let inbound = InboundMessage {
            channel: "discord".into(),
            sender_id,
            chat_id: msg.channel_id.to_string(),
            content: msg.content.clone(),
            timestamp: Utc::now(),
            media: vec![],
            metadata: serde_json::json!({
                "message_id": msg.id.to_string(),
                "guild_id": msg.guild_id.map(|g| g.to_string()),
            }),
        };

        if let Err(e) = self.inbound_tx.send(inbound).await {
            error!("Failed to send Discord inbound message: {e}");
        }
    }

    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!(user = %ready.user.name, "Discord bot connected");
    }
}

#[async_trait]
impl Channel for DiscordChannel {
    fn name(&self) -> &str {
        "discord"
    }

    async fn start(&self, inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        info!("Starting Discord channel (gateway)");

        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let handler = Handler {
            inbound_tx,
            allowed_senders: self.allowed_senders.clone(),
        };

        let mut client = Client::builder(&self.token, intents)
            .event_handler(handler)
            .await
            .wrap_err("failed to build Discord client")?;

        client.start().await.wrap_err("Discord client error")?;

        info!("Discord channel stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        let channel_id: u64 = msg
            .chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Discord channel_id: {}", msg.chat_id))?;

        let channel = serenity::model::id::ChannelId::new(channel_id);

        channel
            .say(&*self.http, &msg.content)
            .await
            .wrap_err("failed to send Discord message")?;

        Ok(())
    }

    fn is_allowed(&self, sender_id: &str) -> bool {
        self.allowed_senders.is_empty() || self.allowed_senders.contains(sender_id)
    }

    async fn stop(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_channel(allowed: Vec<&str>) -> DiscordChannel {
        DiscordChannel {
            token: "test.token".into(),
            http: Arc::new(Http::new("test.token")),
            allowed_senders: allowed.into_iter().map(String::from).collect(),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn test_is_allowed_empty_list() {
        let ch = make_channel(vec![]);
        assert!(ch.is_allowed("anyone"));
    }

    #[test]
    fn test_is_allowed_matching() {
        let ch = make_channel(vec!["12345", "67890"]);
        assert!(ch.is_allowed("12345"));
        assert!(!ch.is_allowed("99999"));
    }

    #[test]
    fn test_is_allowed_not_matching() {
        let ch = make_channel(vec!["12345"]);
        assert!(!ch.is_allowed("other"));
    }
}
