//! Telegram channel using teloxide long polling.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use crew_core::{InboundMessage, OutboundMessage};
use eyre::{Result, WrapErr};
use reqwest::Client;
use teloxide::prelude::*;
use teloxide::types::{
    BotCommand, ChatId, FileId, InlineKeyboardButton, InlineKeyboardMarkup, InputFile,
    MessageId, ParseMode, UpdateKind,
};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::channel::Channel;
use crate::markdown_html::markdown_to_telegram_html;
use crate::media::download_media;

/// Maximum time to wait for a single media download before giving up.
const MEDIA_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30);

/// Telegram caption limit (bytes). We truncate at 1024 chars to be safe.
const CAPTION_MAX_CHARS: usize = 1024;

/// Base delay between reconnection attempts. Doubles on each consecutive failure,
/// capped at `MAX_RECONNECT_DELAY`.
const BASE_RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// Maximum delay between reconnection attempts.
const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(60);

pub struct TelegramChannel {
    bot: Bot,
    allowed_senders: HashSet<String>,
    shutdown: Arc<AtomicBool>,
    media_dir: PathBuf,
    http: Client,
}

impl TelegramChannel {
    pub fn new(
        token: &str,
        allowed_senders: Vec<String>,
        shutdown: Arc<AtomicBool>,
        media_dir: PathBuf,
    ) -> Self {
        Self {
            bot: Bot::new(token),
            allowed_senders: allowed_senders.into_iter().collect(),
            shutdown,
            media_dir,
            http: Client::new(),
        }
    }

    fn check_allowed(&self, sender_id: &str) -> bool {
        if self.allowed_senders.is_empty() {
            return true;
        }
        if self.allowed_senders.contains(sender_id) {
            return true;
        }
        sender_id
            .split('|')
            .any(|part| self.allowed_senders.contains(part))
    }

    /// Download a file from Telegram by file_id, with a timeout.
    async fn download_telegram_file(&self, file_id: &FileId, ext: &str) -> Result<PathBuf> {
        let file = self
            .bot
            .get_file(file_id.clone())
            .await
            .wrap_err("failed to get file info from Telegram")?;

        let url = format!(
            "https://api.telegram.org/file/bot{}/{}",
            self.bot.token(),
            file.path
        );

        let filename = format!("{}{}", file.meta.unique_id, ext);
        download_media(&self.http, &url, &[], &self.media_dir, &filename).await
    }

    /// Try to send an HTML message, falling back to plain text on parse error.
    async fn send_html_with_fallback(&self, chat_id: ChatId, html: &str) -> Result<Message> {
        match self
            .bot
            .send_message(chat_id, html)
            .parse_mode(ParseMode::Html)
            .await
        {
            Ok(msg) => Ok(msg),
            Err(e) => {
                // Telegram returns "Bad Request: can't parse entities" on malformed HTML.
                let err_str = e.to_string();
                if err_str.contains("can't parse entities") || err_str.contains("parse entities") {
                    warn!("HTML parse failed, falling back to plain text: {e}");
                    self.bot
                        .send_message(chat_id, html)
                        .await
                        .wrap_err("failed to send Telegram message (plain text fallback)")
                } else {
                    Err(eyre::eyre!(e).wrap_err("failed to send Telegram message"))
                }
            }
        }
    }

    /// Truncate a caption to Telegram's limit (1024 chars), appending "…" if truncated.
    fn truncate_caption(text: &str) -> String {
        if text.chars().count() <= CAPTION_MAX_CHARS {
            return text.to_string();
        }
        let truncated: String = text.chars().take(CAPTION_MAX_CHARS - 1).collect();
        format!("{truncated}…")
    }

    /// Register bot commands in Telegram's command menu.
    async fn set_commands(&self) {
        let commands = vec![
            BotCommand::new("new", "Start new session or create named session"),
            BotCommand::new("s", "Switch to a named session"),
            BotCommand::new("sessions", "List and switch sessions"),
            BotCommand::new("back", "Switch to previous session"),
            BotCommand::new("delete", "Delete a named session"),
        ];
        match self.bot.set_my_commands(commands).await {
            Ok(_) => info!("Telegram bot commands registered"),
            Err(e) => warn!("Failed to set bot commands: {e}"),
        }
    }

    /// Parse inline keyboard from OutboundMessage metadata.
    ///
    /// Expected format:
    /// ```json
    /// { "inline_keyboard": [[{"text": "Label", "callback_data": "s:topic"}]] }
    /// ```
    fn parse_inline_keyboard(metadata: &serde_json::Value) -> Option<InlineKeyboardMarkup> {
        let rows = metadata.get("inline_keyboard")?.as_array()?;

        let mut keyboard_rows: Vec<Vec<InlineKeyboardButton>> = Vec::new();
        for row in rows {
            let buttons = row.as_array()?;
            let mut row_buttons = Vec::new();
            for btn in buttons {
                let text = btn.get("text")?.as_str()?;
                let data = btn.get("callback_data")?.as_str()?;
                row_buttons.push(InlineKeyboardButton::callback(text.to_string(), data.to_string()));
            }
            if !row_buttons.is_empty() {
                keyboard_rows.push(row_buttons);
            }
        }

        if keyboard_rows.is_empty() {
            None
        } else {
            Some(InlineKeyboardMarkup::new(keyboard_rows))
        }
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    fn max_message_length(&self) -> usize {
        4000
    }

    async fn start(&self, inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        use futures::StreamExt;
        use teloxide::update_listeners::{AsUpdateStream, polling_default};

        info!("Starting Telegram channel (long polling)");

        // Register bot commands in Telegram's command menu
        self.set_commands().await;

        let mut consecutive_failures: u32 = 0;

        // Outer reconnection loop — restarts polling when the stream ends.
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                break;
            }

            let mut listener = polling_default(self.bot.clone()).await;
            let stream = listener.as_stream();
            tokio::pin!(stream);

            // Reset failure counter on successful stream creation
            consecutive_failures = 0;
            info!("Telegram polling stream connected");

            while let Some(result) = stream.next().await {
                if self.shutdown.load(Ordering::Acquire) {
                    info!("Telegram channel shutting down (shutdown flag)");
                    return Ok(());
                }

                let update = match result {
                    Ok(update) => update,
                    Err(e) => {
                        warn!("Telegram polling error: {e}");
                        continue;
                    }
                };

                match update.kind {
                    UpdateKind::Message(msg) => {
                        // Extract text: plain text or caption (for photos/documents)
                        let text = msg.text().or(msg.caption()).unwrap_or("").to_string();

                        // Download media attachments with timeout so we don't block polling
                        let mut media = Vec::new();

                        if let Some(sizes) = msg.photo() {
                            if let Some(photo) = sizes.last() {
                                match tokio::time::timeout(
                                    MEDIA_DOWNLOAD_TIMEOUT,
                                    self.download_telegram_file(&photo.file.id, ".jpg"),
                                )
                                .await
                                {
                                    Ok(Ok(path)) => media.push(path.display().to_string()),
                                    Ok(Err(e)) => warn!("failed to download photo: {e}"),
                                    Err(_) => warn!("photo download timed out after {MEDIA_DOWNLOAD_TIMEOUT:?}"),
                                }
                            }
                        }

                        if let Some(voice) = msg.voice() {
                            match tokio::time::timeout(
                                MEDIA_DOWNLOAD_TIMEOUT,
                                self.download_telegram_file(&voice.file.id, ".ogg"),
                            )
                            .await
                            {
                                Ok(Ok(path)) => media.push(path.display().to_string()),
                                Ok(Err(e)) => warn!("failed to download voice: {e}"),
                                Err(_) => warn!("voice download timed out after {MEDIA_DOWNLOAD_TIMEOUT:?}"),
                            }
                        }

                        if let Some(audio) = msg.audio() {
                            let ext = audio
                                .file_name
                                .as_ref()
                                .and_then(|n| std::path::Path::new(n).extension())
                                .map(|e| format!(".{}", e.to_string_lossy()))
                                .unwrap_or_else(|| ".mp3".to_string());
                            match tokio::time::timeout(
                                MEDIA_DOWNLOAD_TIMEOUT,
                                self.download_telegram_file(&audio.file.id, &ext),
                            )
                            .await
                            {
                                Ok(Ok(path)) => media.push(path.display().to_string()),
                                Ok(Err(e)) => warn!("failed to download audio: {e}"),
                                Err(_) => warn!("audio download timed out after {MEDIA_DOWNLOAD_TIMEOUT:?}"),
                            }
                        }

                        if let Some(doc) = msg.document() {
                            let ext = doc
                                .file_name
                                .as_ref()
                                .and_then(|n| std::path::Path::new(n).extension())
                                .map(|e| format!(".{}", e.to_string_lossy()))
                                .unwrap_or_default();
                            match tokio::time::timeout(
                                MEDIA_DOWNLOAD_TIMEOUT,
                                self.download_telegram_file(&doc.file.id, &ext),
                            )
                            .await
                            {
                                Ok(Ok(path)) => media.push(path.display().to_string()),
                                Ok(Err(e)) => warn!("failed to download document: {e}"),
                                Err(_) => warn!("document download timed out after {MEDIA_DOWNLOAD_TIMEOUT:?}"),
                            }
                        }

                        // Skip messages with no text and no media
                        if text.is_empty() && media.is_empty() {
                            continue;
                        }

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
                            media,
                            metadata: serde_json::json!({}),
                        };

                        if inbound_tx.send(inbound).await.is_err() {
                            info!("Inbound channel closed, stopping Telegram listener");
                            return Ok(());
                        }
                    }

                    UpdateKind::CallbackQuery(cb) => {
                        // Dismiss the loading spinner on the button
                        if let Err(e) = self.bot.answer_callback_query(cb.id.clone()).await {
                            warn!("Failed to answer callback query: {e}");
                        }

                        let sender_id = {
                            let id = cb.from.id.to_string();
                            match &cb.from.username {
                                Some(name) => format!("{id}|{name}"),
                                None => id,
                            }
                        };

                        if !self.check_allowed(&sender_id) {
                            continue;
                        }

                        // Extract chat_id and message_id from the callback's source message
                        let (chat_id, message_id) = match &cb.message {
                            Some(mim) => (
                                mim.chat().id.0.to_string(),
                                Some(mim.id().0.to_string()),
                            ),
                            None => continue,
                        };

                        let callback_data = cb.data.unwrap_or_default();

                        let inbound = InboundMessage {
                            channel: "telegram".into(),
                            sender_id,
                            chat_id,
                            content: callback_data.clone(),
                            timestamp: Utc::now(),
                            media: vec![],
                            metadata: serde_json::json!({
                                "callback_query": true,
                                "callback_data": callback_data,
                                "callback_message_id": message_id,
                            }),
                        };

                        if inbound_tx.send(inbound).await.is_err() {
                            info!("Inbound channel closed, stopping Telegram listener");
                            return Ok(());
                        }
                    }

                    _ => {} // Ignore other update kinds
                }
            }

            // Stream ended (returned None). Check if we should reconnect.
            if self.shutdown.load(Ordering::Acquire) {
                break;
            }

            consecutive_failures += 1;
            let delay = std::cmp::min(
                BASE_RECONNECT_DELAY * 2u32.saturating_pow(consecutive_failures - 1),
                MAX_RECONNECT_DELAY,
            );
            warn!(
                consecutive_failures,
                delay_secs = delay.as_secs(),
                "Telegram polling stream ended unexpectedly, reconnecting..."
            );
            tokio::time::sleep(delay).await;
        }

        info!("Telegram channel stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        let chat_id: i64 = msg
            .chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {}", msg.chat_id))?;

        if !msg.media.is_empty() {
            // Send files as documents
            let caption = if msg.content.is_empty() {
                None
            } else {
                let html = markdown_to_telegram_html(&msg.content);
                Some(Self::truncate_caption(&html))
            };

            for (i, path) in msg.media.iter().enumerate() {
                let file_path = std::path::PathBuf::from(path);
                let file = InputFile::file(&file_path);
                let lower = path.to_lowercase();

                if lower.ends_with(".ogg") || lower.ends_with(".oga") || lower.ends_with(".opus") {
                    // Send as voice message (shows as audio bubble in Telegram)
                    let mut req = self.bot.send_voice(ChatId(chat_id), file);
                    if i == 0 {
                        if let Some(ref cap) = caption {
                            req = req.caption(cap).parse_mode(ParseMode::Html);
                        }
                    }
                    req.await
                        .wrap_err_with(|| format!("failed to send voice: {path}"))?;
                } else if lower.ends_with(".mp3")
                    || lower.ends_with(".wav")
                    || lower.ends_with(".m4a")
                {
                    // Send as audio file (shows with player controls)
                    let mut req = self.bot.send_audio(ChatId(chat_id), file);
                    if i == 0 {
                        if let Some(ref cap) = caption {
                            req = req.caption(cap).parse_mode(ParseMode::Html);
                        }
                    }
                    req.await
                        .wrap_err_with(|| format!("failed to send audio: {path}"))?;
                } else {
                    // Send as document (generic file)
                    let mut req = self.bot.send_document(ChatId(chat_id), file);
                    if i == 0 {
                        if let Some(ref cap) = caption {
                            req = req.caption(cap).parse_mode(ParseMode::Html);
                        }
                    }
                    req.await
                        .wrap_err_with(|| format!("failed to send document: {path}"))?;
                }
            }
        } else {
            let html = markdown_to_telegram_html(&msg.content);

            // Check for inline keyboard in metadata
            if let Some(markup) = Self::parse_inline_keyboard(&msg.metadata) {
                self.bot
                    .send_message(ChatId(chat_id), &html)
                    .parse_mode(ParseMode::Html)
                    .reply_markup(markup)
                    .await
                    .wrap_err("failed to send Telegram message with keyboard")?;
            } else {
                self.send_html_with_fallback(ChatId(chat_id), &html)
                    .await?;
            }
        }

        Ok(())
    }

    fn is_allowed(&self, sender_id: &str) -> bool {
        self.check_allowed(sender_id)
    }

    async fn stop(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Release);
        Ok(())
    }

    async fn send_typing(&self, chat_id: &str) -> Result<()> {
        let id: i64 = chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {chat_id}"))?;
        self.bot
            .send_chat_action(ChatId(id), teloxide::types::ChatAction::Typing)
            .await
            .wrap_err("failed to send typing action")?;
        Ok(())
    }

    async fn send_listening(&self, chat_id: &str) -> Result<()> {
        let id: i64 = chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {chat_id}"))?;
        self.bot
            .send_chat_action(ChatId(id), teloxide::types::ChatAction::RecordVoice)
            .await
            .wrap_err("failed to send record_voice action")?;
        Ok(())
    }

    async fn send_with_id(&self, msg: &OutboundMessage) -> Result<Option<String>> {
        let chat_id: i64 = msg
            .chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {}", msg.chat_id))?;

        let html = markdown_to_telegram_html(&msg.content);

        let sent = if let Some(markup) = Self::parse_inline_keyboard(&msg.metadata) {
            self.bot
                .send_message(ChatId(chat_id), &html)
                .parse_mode(ParseMode::Html)
                .reply_markup(markup)
                .await
                .wrap_err("failed to send Telegram message with keyboard")?
        } else {
            self.send_html_with_fallback(ChatId(chat_id), &html)
                .await?
        };

        Ok(Some(sent.id.0.to_string()))
    }

    async fn edit_message(&self, chat_id: &str, message_id: &str, new_content: &str) -> Result<()> {
        let cid: i64 = chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {chat_id}"))?;
        let mid: i32 = message_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram message_id: {message_id}"))?;

        let html = markdown_to_telegram_html(new_content);
        self.bot
            .edit_message_text(ChatId(cid), MessageId(mid), &html)
            .parse_mode(ParseMode::Html)
            .await
            .wrap_err("failed to edit Telegram message")?;
        Ok(())
    }

    async fn delete_message(&self, chat_id: &str, message_id: &str) -> Result<()> {
        let cid: i64 = chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {chat_id}"))?;
        let mid: i32 = message_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram message_id: {message_id}"))?;

        self.bot
            .delete_message(ChatId(cid), MessageId(mid))
            .await
            .wrap_err("failed to delete Telegram message")?;
        Ok(())
    }

    async fn edit_message_with_metadata(
        &self,
        chat_id: &str,
        message_id: &str,
        new_content: &str,
        metadata: &serde_json::Value,
    ) -> Result<()> {
        let cid: i64 = chat_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram chat_id: {chat_id}"))?;
        let mid: i32 = message_id
            .parse()
            .wrap_err_with(|| format!("invalid Telegram message_id: {message_id}"))?;

        let html = markdown_to_telegram_html(new_content);

        if let Some(markup) = Self::parse_inline_keyboard(metadata) {
            self.bot
                .edit_message_text(ChatId(cid), MessageId(mid), &html)
                .parse_mode(ParseMode::Html)
                .reply_markup(markup)
                .await
                .wrap_err("failed to edit Telegram message with keyboard")?;
        } else {
            self.bot
                .edit_message_text(ChatId(cid), MessageId(mid), &html)
                .parse_mode(ParseMode::Html)
                .await
                .wrap_err("failed to edit Telegram message")?;
        }
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
            media_dir: PathBuf::from("/tmp/test-media"),
            http: Client::new(),
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

    #[test]
    fn test_truncate_caption_short() {
        let short = "Hello world";
        assert_eq!(TelegramChannel::truncate_caption(short), short);
    }

    #[test]
    fn test_truncate_caption_long() {
        let long: String = "x".repeat(2000);
        let truncated = TelegramChannel::truncate_caption(&long);
        assert_eq!(truncated.chars().count(), CAPTION_MAX_CHARS);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn test_truncate_caption_exact() {
        let exact: String = "a".repeat(CAPTION_MAX_CHARS);
        assert_eq!(TelegramChannel::truncate_caption(&exact), exact);
    }
}
