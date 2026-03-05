//! Async HTTP client for ominix-api (ASR/TTS).

use std::path::Path;

use eyre::{Result, WrapErr};
use reqwest::Client;
use tracing::debug;

/// Async client for ominix-api ASR/TTS endpoints.
pub struct OminixClient {
    client: Client,
    base_url: String,
    language: Option<String>,
}

impl OminixClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            language: None,
        }
    }

    /// Set default ASR language hint.
    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language;
        self
    }

    /// Check if ominix-api is reachable.
    pub async fn health(&self) -> bool {
        match self
            .client
            .get(format!("{}/health", self.base_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Transcribe an audio file to text.
    pub async fn transcribe(&self, audio_path: &Path) -> Result<String> {
        // Guard against oversized files (100MB limit matches the ASR app skill)
        let meta = tokio::fs::metadata(audio_path)
            .await
            .wrap_err_with(|| format!("failed to stat audio: {}", audio_path.display()))?;
        if meta.len() > 100_000_000 {
            eyre::bail!("audio file too large ({} bytes, max 100MB)", meta.len());
        }

        let bytes = tokio::fs::read(audio_path)
            .await
            .wrap_err_with(|| format!("failed to read audio: {}", audio_path.display()))?;

        // ominix-api expects JSON with base64-encoded audio
        use base64::Engine;
        let file_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

        let mut body = serde_json::json!({
            "file": file_b64,
            "response_format": "verbose_json",
        });

        // Only send language hint if explicitly configured (auto-detect otherwise)
        if let Some(ref lang) = self.language {
            body["language"] = serde_json::Value::String(lang.clone());
        }

        let resp = self
            .client
            .post(format!("{}/v1/audio/transcriptions", self.base_url))
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .wrap_err("failed to call ominix-api transcription")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eyre::bail!("ominix-api transcription failed: {status} - {body}");
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .wrap_err("invalid transcription response")?;

        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| eyre::eyre!("no text field in transcription response"))?;

        debug!(chars = text.len(), "audio transcribed via ominix-api");
        Ok(text.to_string())
    }

    /// Synthesize text to speech, returning raw WAV bytes.
    pub async fn synthesize(&self, text: &str, voice: &str, language: &str) -> Result<Vec<u8>> {
        let body = serde_json::json!({
            "input": text,
            "voice": voice,
            "language": language,
        });

        let resp = self
            .client
            .post(format!("{}/v1/audio/speech", self.base_url))
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .wrap_err("failed to call ominix-api TTS")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eyre::bail!("ominix-api TTS failed: {status} - {body}");
        }

        let wav_bytes = resp
            .bytes()
            .await
            .wrap_err("failed to read TTS response")?;

        debug!(size = wav_bytes.len(), "TTS audio generated via ominix-api");
        Ok(wav_bytes.to_vec())
    }

    /// Synthesize text to a WAV file. Returns audio duration in seconds.
    pub async fn synthesize_to_file(
        &self,
        text: &str,
        voice: &str,
        language: &str,
        path: &Path,
    ) -> Result<f64> {
        let wav_bytes = self.synthesize(text, voice, language).await?;

        if wav_bytes.len() < 44 {
            eyre::bail!("TTS returned invalid WAV data (too small)");
        }

        tokio::fs::write(path, &wav_bytes)
            .await
            .wrap_err_with(|| format!("failed to write TTS output: {}", path.display()))?;

        // 24kHz 16-bit mono = 48000 bytes/sec
        let duration_secs = wav_bytes.len().saturating_sub(44) as f64 / 48000.0;
        Ok(duration_secs)
    }
}
