//! Types for LLM interactions.

use crew_core::ToolCall;
use serde::{Deserialize, Serialize};

/// Response from a chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Text content of the response (if any).
    pub content: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// Token usage statistics.
    pub usage: TokenUsage,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model finished naturally.
    EndTurn,
    /// Model wants to use tools.
    ToolUse,
    /// Hit max tokens limit.
    MaxTokens,
    /// Hit a stop sequence.
    StopSequence,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Tool specification for LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// Events from a streaming LLM response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text chunk.
    TextDelta(String),
    /// Incremental tool call data.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    /// Token usage (sent at stream end by most providers).
    Usage(TokenUsage),
    /// Stream finished with stop reason.
    Done(StopReason),
    /// Error during streaming.
    Error(String),
}

/// A boxed stream of StreamEvents.
pub type ChatStream = std::pin::Pin<Box<dyn futures::Stream<Item = StreamEvent> + Send>>;
