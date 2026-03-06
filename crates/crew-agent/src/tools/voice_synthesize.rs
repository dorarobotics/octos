//! Voice synthesize tool — TTS via OminiX API, sends WAV to chat.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use crew_core::OutboundMessage;
use crew_llm::OminixClient;
use eyre::{Result, WrapErr};
use serde::Deserialize;
use tokio::sync::mpsc;

use super::{Tool, ToolResult};

/// Tool that synthesizes text to speech and sends the audio file to chat.
pub struct VoiceSynthesizeTool {
    ominix: Arc<OminixClient>,
    out_tx: mpsc::Sender<OutboundMessage>,
    default_channel: std::sync::Mutex<String>,
    default_chat_id: std::sync::Mutex<String>,
}

impl VoiceSynthesizeTool {
    pub fn with_context(
        ominix: Arc<OminixClient>,
        out_tx: mpsc::Sender<OutboundMessage>,
        channel: impl Into<String>,
        chat_id: impl Into<String>,
    ) -> Self {
        Self {
            ominix,
            out_tx,
            default_channel: std::sync::Mutex::new(channel.into()),
            default_chat_id: std::sync::Mutex::new(chat_id.into()),
        }
    }

    pub fn set_context(&self, channel: &str, chat_id: &str) {
        *self
            .default_channel
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = channel.to_string();
        *self
            .default_chat_id
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = chat_id.to_string();
    }
}

#[derive(Deserialize)]
struct Input {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    send: Option<bool>,
}

#[async_trait]
impl Tool for VoiceSynthesizeTool {
    fn name(&self) -> &str {
        "voice_synthesize"
    }

    fn description(&self) -> &str {
        "Synthesize text to speech (TTS) and send the audio to the user. \
         Uses the on-device OminiX TTS engine. The output is a WAV audio file. \
         By default the audio is sent to the current chat. Set send=false to \
         only generate and return the file path."
    }

    fn tags(&self) -> &[&str] {
        &["gateway"]
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to synthesize into speech"
                },
                "voice": {
                    "type": "string",
                    "description": "Speaker voice name (e.g. 'vivian', 'dylan', 'serena'). Omit for server default."
                },
                "language": {
                    "type": "string",
                    "description": "Language hint (e.g. 'chinese', 'english'). Omit for server default."
                },
                "send": {
                    "type": "boolean",
                    "description": "Whether to send the audio to chat (default: true)"
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid voice_synthesize input")?;

        if input.text.is_empty() {
            return Ok(ToolResult {
                output: "Error: text is empty".into(),
                success: false,
                ..Default::default()
            });
        }

        let voice = input.voice.as_deref().unwrap_or("default");
        let language = input.language.as_deref();
        let send = input.send.unwrap_or(true);

        // Generate output path
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f");
        let media_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".crew")
            .join("media");
        tokio::fs::create_dir_all(&media_dir).await.ok();
        let wav_path = media_dir.join(format!("tts_{timestamp}.wav"));

        // Call OminiX TTS
        let duration = match self
            .ominix
            .synthesize_to_file(&input.text, voice, language, &wav_path)
            .await
        {
            Ok(d) => d,
            Err(e) => {
                return Ok(ToolResult {
                    output: format!("TTS failed: {e}"),
                    success: false,
                    ..Default::default()
                });
            }
        };

        // Convert WAV → M4A (AAC) for better Telegram/chat compatibility.
        // Falls back to WAV if afconvert is not available.
        let m4a_path = media_dir.join(format!("tts_{timestamp}.m4a"));
        let output_path = match tokio::process::Command::new("afconvert")
            .args([
                "-f",
                "m4af",
                "-d",
                "aac",
                "-s",
                "3",
                wav_path.to_str().unwrap_or_default(),
                m4a_path.to_str().unwrap_or_default(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                tokio::fs::remove_file(&wav_path).await.ok();
                m4a_path
            }
            _ => wav_path.clone(),
        };

        let output_path_str = output_path.display().to_string();

        // Log file size for debugging
        let file_size = tokio::fs::metadata(&output_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        tracing::info!(
            path = %output_path_str,
            size = file_size,
            duration_secs = duration,
            voice,
            "voice_synthesize: file ready"
        );

        if file_size == 0 {
            return Ok(ToolResult {
                output: format!("Error: TTS produced empty file at {output_path_str}"),
                success: false,
                ..Default::default()
            });
        }

        if send {
            let channel = self
                .default_channel
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let chat_id = self
                .default_chat_id
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();

            if channel.is_empty() || chat_id.is_empty() {
                return Ok(ToolResult {
                    output: format!(
                        "Audio generated at {output_path_str} ({duration:.1}s) but no target channel."
                    ),
                    success: true,
                    ..Default::default()
                });
            }

            let text_preview: String = input.text.chars().take(30).collect();
            let msg = OutboundMessage {
                channel: channel.clone(),
                chat_id: chat_id.clone(),
                content: format!("🔊 \"{text_preview}\""),
                reply_to: None,
                media: vec![output_path_str.clone()],
                metadata: serde_json::json!({}),
            };

            self.out_tx
                .send(msg)
                .await
                .map_err(|e| eyre::eyre!("failed to send TTS audio: {e}"))?;

            Ok(ToolResult {
                output: format!("Audio sent ({duration:.1}s, voice={voice}): {output_path_str}"),
                success: true,
                ..Default::default()
            })
        } else {
            Ok(ToolResult {
                output: format!(
                    "Audio generated ({duration:.1}s, voice={voice}): {output_path_str}"
                ),
                success: true,
                ..Default::default()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_text() {
        let ominix = Arc::new(OminixClient::new("http://localhost:9999"));
        let (tx, _rx) = mpsc::channel(16);
        let tool = VoiceSynthesizeTool::with_context(ominix, tx, "telegram", "123");

        let result = tool
            .execute(&serde_json::json!({"text": ""}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("empty"));
    }
}
