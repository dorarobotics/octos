//! Anthropic (Claude) provider implementation.

use async_trait::async_trait;
use crew_core::Message;
use eyre::{Result, WrapErr};
use futures::StreamExt;

use reqwest::Client;
use serde::Deserialize;

use secrecy::{ExposeSecret, SecretString};

use crate::vision;

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec};

/// Anthropic Claude provider.
pub struct AnthropicProvider {
    client: Client,
    api_key: SecretString,
    model: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: crate::provider::build_http_client(
                crate::provider::DEFAULT_LLM_TIMEOUT_SECS,
                crate::provider::DEFAULT_LLM_CONNECT_TIMEOUT_SECS,
            ),
            api_key: SecretString::from(api_key.into()),
            model: model.into(),
            base_url: "https://api.anthropic.com".to_string(),
        }
    }

    /// Create a provider using the ANTHROPIC_API_KEY environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .wrap_err("ANTHROPIC_API_KEY environment variable not set")?;
        Ok(Self::new(api_key, "claude-sonnet-4-20250514"))
    }

    /// Set a custom base URL (for compatible endpoints).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Replace the HTTP client with one using custom timeouts (in seconds).
    pub fn with_http_timeout(mut self, timeout_secs: u64, connect_timeout_secs: u64) -> Self {
        self.client = crate::provider::build_http_client(timeout_secs, connect_timeout_secs);
        self
    }

    /// Build the shared request struct used by both chat() and chat_stream().
    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> serde_json::Value {
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != crew_core::MessageRole::System)
            .map(|m| {
                let role = match m.role {
                    crew_core::MessageRole::User => "user",
                    crew_core::MessageRole::Assistant => "assistant",
                    crew_core::MessageRole::Tool => "user",
                    crew_core::MessageRole::System => "user",
                };
                serde_json::json!({
                    "role": role,
                    "content": build_anthropic_content_json(m),
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": &self.model,
            "max_tokens": config.max_tokens.unwrap_or(4096),
            "messages": api_messages,
        });

        // System prompt with cache_control for cost optimization
        if let Some(sys_msg) = messages.iter().find(|m| m.role == crew_core::MessageRole::System) {
            body["system"] = serde_json::json!([{
                "type": "text",
                "text": &sys_msg.content,
                "cache_control": { "type": "ephemeral" }
            }]);
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools).unwrap_or_default();
        }

        // Extended thinking support
        if let Some(effort) = &config.reasoning_effort {
            let budget = match effort {
                crate::config::ReasoningEffort::Low => 2048,
                crate::config::ReasoningEffort::Medium => 8192,
                crate::config::ReasoningEffort::High => 32768,
            };
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget
            });
        }

        body
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatResponse> {
        let body = self.build_request(messages, tools, config);

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", "2025-04-14")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .wrap_err("failed to send request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eyre::bail!(
                "Anthropic API error: {status} - {}",
                crate::provider::truncate_error_body(&body)
            );
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .wrap_err("failed to parse Anthropic response")?;

        Ok(parse_anthropic_response(api_response))
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatStream> {
        let mut body = self.build_request(messages, tools, config);
        body["stream"] = true.into();

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", "2025-04-14")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .wrap_err("failed to send streaming request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            eyre::bail!(
                "Anthropic API error: {status} - {}",
                crate::provider::truncate_error_body(&text)
            );
        }

        let sse_stream = crate::sse::parse_sse_response(response);
        let state = AnthropicStreamState::default();
        let event_stream = sse_stream
            .scan(state, |state, event| {
                let events = map_anthropic_sse(state, &event);
                futures::future::ready(Some(events))
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(event_stream))
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

/// Build message content as JSON (plain text or multipart with images).
fn build_anthropic_content_json(msg: &Message) -> serde_json::Value {
    let images: Vec<_> = msg.media.iter().filter(|p| vision::is_image(p)).collect();

    if images.is_empty() {
        return serde_json::Value::String(msg.content.clone());
    }

    let mut parts = Vec::new();
    for path in &images {
        if let Ok((mime, data)) = vision::encode_image(path) {
            parts.push(serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                }
            }));
        }
    }
    if !msg.content.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": &msg.content,
        }));
    }
    serde_json::Value::Array(parts)
}

/// Parse an Anthropic API response into our types.
fn parse_anthropic_response(api_response: AnthropicResponse) -> ChatResponse {
    let mut content = None;
    let mut reasoning_content = None;
    let mut tool_calls = Vec::new();

    for block in api_response.content {
        match block {
            ContentBlock::Text { text } => {
                content = Some(text);
            }
            ContentBlock::Thinking { thinking } => {
                reasoning_content = Some(thinking);
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(crew_core::ToolCall {
                    id,
                    name,
                    arguments: input,
                    metadata: None,
                });
            }
        }
    }

    let stop_reason = match api_response.stop_reason.as_str() {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    let usage = &api_response.usage;
    ChatResponse {
        content,
        reasoning_content,
        tool_calls,
        stop_reason,
        usage: TokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_read_tokens: usage.cache_read_input_tokens.unwrap_or(0),
            cache_write_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
            ..Default::default()
        },
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    stop_reason: String,
    usage: ApiUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

// --- Streaming SSE helpers ---

#[derive(Default)]
struct AnthropicStreamState {
    block_to_tool: std::collections::HashMap<usize, usize>,
    thinking_blocks: std::collections::HashSet<usize>,
    tool_count: usize,
    input_tokens: u32,
    cache_read_tokens: u32,
    cache_write_tokens: u32,
}

// Visible for testing
fn map_anthropic_sse(
    state: &mut AnthropicStreamState,
    event: &crate::sse::SseEvent,
) -> Vec<StreamEvent> {
    // Handle SSE-level error events (e.g. Z.AI returns `event: error` with HTTP 200)
    if event.event.as_deref() == Some("error") {
        let msg = match serde_json::from_str::<serde_json::Value>(&event.data) {
            Ok(v) => v["error"]["message"]
                .as_str()
                .unwrap_or(&event.data)
                .to_string(),
            Err(_) => event.data.clone(),
        };
        return vec![StreamEvent::Error(msg)];
    }

    let data: serde_json::Value = match serde_json::from_str(&event.data) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    // Handle error payloads without SSE event type (fallback)
    if data.get("error").is_some() {
        let msg = data["error"]["message"]
            .as_str()
            .unwrap_or("unknown API error")
            .to_string();
        return vec![StreamEvent::Error(msg)];
    }

    match data["type"].as_str().unwrap_or("") {
        "message_start" => {
            let usage = &data["message"]["usage"];
            if let Some(t) = usage["input_tokens"].as_u64() {
                state.input_tokens = t as u32;
            }
            if let Some(t) = usage["cache_read_input_tokens"].as_u64() {
                state.cache_read_tokens = t as u32;
            }
            if let Some(t) = usage["cache_creation_input_tokens"].as_u64() {
                state.cache_write_tokens = t as u32;
            }
            vec![]
        }
        "content_block_start" => {
            let idx = data["index"].as_u64().unwrap_or(0) as usize;
            match data["content_block"]["type"].as_str() {
                Some("tool_use") => {
                    let tool_idx = state.tool_count;
                    state.tool_count += 1;
                    state.block_to_tool.insert(idx, tool_idx);
                    vec![StreamEvent::ToolCallDelta {
                        index: tool_idx,
                        id: data["content_block"]["id"].as_str().map(String::from),
                        name: data["content_block"]["name"].as_str().map(String::from),
                        arguments_delta: String::new(),
                    }]
                }
                Some("thinking") => {
                    state.thinking_blocks.insert(idx);
                    vec![]
                }
                _ => vec![],
            }
        }
        "content_block_delta" => {
            let idx = data["index"].as_u64().unwrap_or(0) as usize;
            match data["delta"]["type"].as_str().unwrap_or("") {
                "text_delta" => {
                    vec![StreamEvent::TextDelta(
                        data["delta"]["text"].as_str().unwrap_or("").to_string(),
                    )]
                }
                "thinking_delta" => {
                    if state.thinking_blocks.contains(&idx) {
                        vec![StreamEvent::ReasoningDelta(
                            data["delta"]["thinking"].as_str().unwrap_or("").to_string(),
                        )]
                    } else {
                        vec![]
                    }
                }
                "input_json_delta" => {
                    if let Some(&tool_idx) = state.block_to_tool.get(&idx) {
                        vec![StreamEvent::ToolCallDelta {
                            index: tool_idx,
                            id: None,
                            name: None,
                            arguments_delta: data["delta"]["partial_json"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                        }]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }
        "message_delta" => {
            let stop_reason = match data["delta"]["stop_reason"].as_str() {
                Some("end_turn") => StopReason::EndTurn,
                Some("tool_use") => StopReason::ToolUse,
                Some("max_tokens") => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            };
            let output_tokens = data["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
            vec![
                StreamEvent::Usage(TokenUsage {
                    input_tokens: state.input_tokens,
                    output_tokens,
                    cache_read_tokens: state.cache_read_tokens,
                    cache_write_tokens: state.cache_write_tokens,
                    ..Default::default()
                }),
                StreamEvent::Done(stop_reason),
            ]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crew_core::{Message, MessageRole};

    fn msg(role: MessageRole, content: &str) -> Message {
        Message {
            role,
            content: content.to_string(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        }
    }

    // --- build_anthropic_content_json tests ---

    #[test]
    fn test_build_content_text_only() {
        let m = msg(MessageRole::User, "hello");
        let content = build_anthropic_content_json(&m);
        assert_eq!(content.as_str(), Some("hello"));
    }

    #[test]
    fn test_build_content_with_non_image_media() {
        let m = Message {
            role: MessageRole::User,
            content: "check this".into(),
            media: vec!["file.txt".into(), "data.csv".into()],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        };
        // Non-image media should fall through to plain text
        let content = build_anthropic_content_json(&m);
        assert_eq!(content.as_str(), Some("check this"));
    }

    // --- build_request tests ---

    #[test]
    fn test_build_request_filters_system() {
        let provider = AnthropicProvider::new("test-key", "claude-test");
        let messages = vec![
            msg(MessageRole::System, "system prompt"),
            msg(MessageRole::User, "hello"),
            msg(MessageRole::Assistant, "hi"),
        ];
        let config = ChatConfig::default();
        let request = provider.build_request(&messages, &[], &config);

        // System should be extracted with cache_control
        let system = &request["system"];
        assert_eq!(system[0]["text"].as_str(), Some("system prompt"));
        assert_eq!(system[0]["cache_control"]["type"].as_str(), Some("ephemeral"));
        // Messages should only have user + assistant
        assert_eq!(request["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_build_request_tool_role_mapped_to_user() {
        let provider = AnthropicProvider::new("test-key", "claude-test");
        let messages = vec![Message {
            role: MessageRole::Tool,
            content: "tool result".into(),
            media: vec![],
            tool_calls: None,
            tool_call_id: Some("tc1".into()),
            reasoning_content: None,
            timestamp: chrono::Utc::now(),
        }];
        let config = ChatConfig::default();
        let request = provider.build_request(&messages, &[], &config);
        assert_eq!(request["messages"][0]["role"].as_str(), Some("user"));
    }

    #[test]
    fn test_build_request_tools_omitted_when_empty() {
        let provider = AnthropicProvider::new("test-key", "claude-test");
        let messages = vec![msg(MessageRole::User, "hi")];
        let config = ChatConfig::default();
        let request = provider.build_request(&messages, &[], &config);
        assert!(request.get("tools").is_none());
    }

    #[test]
    fn test_build_request_default_max_tokens() {
        let provider = AnthropicProvider::new("test-key", "claude-test");
        let messages = vec![msg(MessageRole::User, "hi")];
        let config = ChatConfig::default();
        let request = provider.build_request(&messages, &[], &config);
        assert_eq!(request["max_tokens"].as_u64(), Some(4096));
    }

    #[test]
    fn test_build_request_reasoning_effort() {
        let provider = AnthropicProvider::new("test-key", "claude-test");
        let messages = vec![msg(MessageRole::User, "think hard")];
        let mut config = ChatConfig::default();
        config.reasoning_effort = Some(crate::config::ReasoningEffort::High);
        let request = provider.build_request(&messages, &[], &config);
        assert_eq!(request["thinking"]["type"].as_str(), Some("enabled"));
        assert_eq!(request["thinking"]["budget_tokens"].as_u64(), Some(32768));
    }

    #[test]
    fn test_parse_response_with_thinking() {
        let response = AnthropicResponse {
            content: vec![
                ContentBlock::Thinking { thinking: "Let me think...".into() },
                ContentBlock::Text { text: "The answer is 42.".into() },
            ],
            stop_reason: "end_turn".into(),
            usage: ApiUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_input_tokens: Some(80),
                cache_creation_input_tokens: Some(20),
            },
        };
        let result = parse_anthropic_response(response);
        assert_eq!(result.content.as_deref(), Some("The answer is 42."));
        assert_eq!(result.reasoning_content.as_deref(), Some("Let me think..."));
        assert_eq!(result.usage.cache_read_tokens, 80);
        assert_eq!(result.usage.cache_write_tokens, 20);
    }

    // --- SSE mapping tests ---

    #[test]
    fn test_sse_message_start() {
        let mut state = AnthropicStreamState::default();
        let event = crate::sse::SseEvent {
            event: None,
            data: r#"{"type": "message_start", "message": {"usage": {"input_tokens": 42}}}"#.into(),
        };
        let events = map_anthropic_sse(&mut state, &event);
        assert!(events.is_empty());
        assert_eq!(state.input_tokens, 42);
    }

    #[test]
    fn test_sse_text_delta() {
        let mut state = AnthropicStreamState::default();
        let event = crate::sse::SseEvent {
            event: None,
            data: r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#.into(),
        };
        let events = map_anthropic_sse(&mut state, &event);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hello"));
    }

    #[test]
    fn test_sse_tool_call_start() {
        let mut state = AnthropicStreamState::default();
        let event = crate::sse::SseEvent {
            event: None,
            data: r#"{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tc1", "name": "shell"}}"#.into(),
        };
        let events = map_anthropic_sse(&mut state, &event);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ToolCallDelta {
                index, id, name, ..
            } => {
                assert_eq!(*index, 0);
                assert_eq!(id.as_deref(), Some("tc1"));
                assert_eq!(name.as_deref(), Some("shell"));
            }
            _ => panic!("expected ToolCallDelta"),
        }
        assert_eq!(state.tool_count, 1);
    }

    #[test]
    fn test_sse_message_delta_end_turn() {
        let mut state = AnthropicStreamState::default();
        state.input_tokens = 100;
        let event = crate::sse::SseEvent {
            event: None,
            data: r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 50}}"#.into(),
        };
        let events = map_anthropic_sse(&mut state, &event);
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], StreamEvent::Usage(u) if u.input_tokens == 100 && u.output_tokens == 50)
        );
        assert!(matches!(&events[1], StreamEvent::Done(StopReason::EndTurn)));
    }

    #[test]
    fn test_sse_error_event() {
        let mut state = AnthropicStreamState::default();
        let event = crate::sse::SseEvent {
            event: Some("error".into()),
            data: r#"{"error": {"message": "rate limited"}}"#.into(),
        };
        let events = map_anthropic_sse(&mut state, &event);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Error(msg) if msg == "rate limited"));
    }

    #[test]
    fn test_sse_invalid_json_returns_empty() {
        let mut state = AnthropicStreamState::default();
        let event = crate::sse::SseEvent {
            event: None,
            data: "not json".into(),
        };
        let events = map_anthropic_sse(&mut state, &event);
        assert!(events.is_empty());
    }

    // --- Provider metadata tests ---

    #[test]
    fn test_provider_name_and_model() {
        let provider = AnthropicProvider::new("test-key", "claude-3-haiku");
        assert_eq!(provider.provider_name(), "anthropic");
        assert_eq!(provider.model_id(), "claude-3-haiku");
    }

    #[test]
    fn test_with_base_url() {
        let provider =
            AnthropicProvider::new("key", "model").with_base_url("https://custom.api.com");
        assert_eq!(provider.base_url, "https://custom.api.com");
    }
}
