//! LLM provider abstraction for crew-rs.
//!
//! This crate provides a unified interface for interacting with LLM providers:
//! - Anthropic (Claude)
//! - OpenAI (GPT-4)
//! - Google Gemini
//! - Ollama (local models)

mod config;
pub mod context;
mod provider;
mod retry;
pub mod sse;
mod types;
pub mod vision;

pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod openrouter;
pub mod transcription;

pub use config::ChatConfig;
pub use provider::LlmProvider;
pub use retry::{RetryConfig, RetryProvider};
pub use transcription::GroqTranscriber;
pub use types::{ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec};
