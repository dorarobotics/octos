//! LLM provider abstraction for crew-rs.
//!
//! This crate provides a unified interface for interacting with LLM providers:
//! - Anthropic (Claude)
//! - OpenAI (GPT-4)
//! - Google Gemini
//! - Ollama (local models)

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
pub mod openai;
pub mod openrouter;
pub mod registry;
pub mod transcription;

pub use config::ChatConfig;
pub use context_override::ContextWindowOverride;
pub use embedding::{EmbeddingProvider, OpenAIEmbedder};
pub use failover::ProviderChain;
pub use provider::LlmProvider;
pub use retry::{RetryConfig, RetryProvider};
pub use router::{ProviderRouter, SubProviderMeta};
pub use transcription::GroqTranscriber;
pub use types::{ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec};
