//! LLM provider abstraction for crew-rs.
//!
//! This crate provides a unified interface for interacting with LLM providers:
//! - Anthropic (Claude)
//! - OpenAI (GPT-4)
//! - Google Gemini
//! - Ollama (local models)

pub mod adaptive;
mod config;
pub mod context;
mod context_override;
pub mod embedding;
mod failover;
pub mod pricing;
mod provider;
mod retry;
pub mod router;
pub mod sse;
mod types;
pub mod vision;

pub mod anthropic;
pub mod gemini;
pub mod ominix;
pub mod openai;
pub mod openrouter;
pub mod registry;
pub mod transcription;

pub use adaptive::{
    AdaptiveConfig, AdaptiveRouter, MetricsSnapshot, SharedMetrics, SharedPolicy,
    SharedProviderMetrics,
};
pub use config::{ChatConfig, ToolChoice};
pub use context_override::ContextWindowOverride;
pub use embedding::{EmbeddingProvider, OpenAIEmbedder};
pub use failover::ProviderChain;
pub use provider::{
    LlmProvider, build_http_client, DEFAULT_EMBEDDING_CONNECT_TIMEOUT_SECS,
    DEFAULT_EMBEDDING_TIMEOUT_SECS, DEFAULT_LLM_CONNECT_TIMEOUT_SECS, DEFAULT_LLM_TIMEOUT_SECS,
};
pub use retry::{RetryConfig, RetryProvider};
pub use router::{ProviderRouter, SubProviderMeta};
pub use ominix::OminixClient;
pub use transcription::GroqTranscriber;
pub use types::{
    ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec, strip_think_tags,
};

/// Unified trait for audio transcription backends.
#[async_trait::async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, audio_path: &std::path::Path) -> eyre::Result<String>;
}

#[async_trait::async_trait]
impl Transcriber for GroqTranscriber {
    async fn transcribe(&self, audio_path: &std::path::Path) -> eyre::Result<String> {
        self.transcribe(audio_path).await
    }
}

#[async_trait::async_trait]
impl Transcriber for OminixClient {
    async fn transcribe(&self, audio_path: &std::path::Path) -> eyre::Result<String> {
        self.transcribe(audio_path).await
    }
}
