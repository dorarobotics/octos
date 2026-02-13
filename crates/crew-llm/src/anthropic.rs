//! Anthropic (Claude) provider implementation.

use async_trait::async_trait;
use crew_core::Message;
use eyre::{Result, WrapErr};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::vision;

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec};

/// Anthropic Claude provider.
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
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
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatResponse> {
        let request = AnthropicRequest {
            model: &self.model,
            max_tokens: config.max_tokens.unwrap_or(4096),
            messages: messages
                .iter()
                .filter(|m| m.role != crew_core::MessageRole::System)
                .map(|m| {
                    let role = match m.role {
                        crew_core::MessageRole::User => "user",
                        crew_core::MessageRole::Assistant => "assistant",
                        crew_core::MessageRole::Tool => "user",
                        crew_core::MessageRole::System => "user",
                    };
                    let content = build_anthropic_content(m);
                    AnthropicMessage { role, content }
                })
                .collect(),
            system: messages
                .iter()
                .find(|m| m.role == crew_core::MessageRole::System)
                .map(|m| m.content.as_str()),
            tools: if tools.is_empty() { None } else { Some(tools) },
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .wrap_err("failed to send request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eyre::bail!("Anthropic API error: {status} - {body}");
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .wrap_err("failed to parse Anthropic response")?;

        // Convert response to our types
        let mut content = None;
        let mut tool_calls = Vec::new();

        for block in api_response.content {
            match block {
                ContentBlock::Text { text } => {
                    content = Some(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(crew_core::ToolCall {
                        id,
                        name,
                        arguments: input,
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

        Ok(ChatResponse {
            content,
            tool_calls,
            stop_reason,
            usage: TokenUsage {
                input_tokens: api_response.usage.input_tokens,
                output_tokens: api_response.usage.output_tokens,
            },
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatStream> {
        let request = AnthropicRequest {
            model: &self.model,
            max_tokens: config.max_tokens.unwrap_or(4096),
            messages: messages
                .iter()
                .filter(|m| m.role != crew_core::MessageRole::System)
                .map(|m| {
                    let role = match m.role {
                        crew_core::MessageRole::User => "user",
                        crew_core::MessageRole::Assistant => "assistant",
                        crew_core::MessageRole::Tool => "user",
                        crew_core::MessageRole::System => "user",
                    };
                    AnthropicMessage {
                        role,
                        content: build_anthropic_content(m),
                    }
                })
                .collect(),
            system: messages
                .iter()
                .find(|m| m.role == crew_core::MessageRole::System)
                .map(|m| m.content.as_str()),
            tools: if tools.is_empty() { None } else { Some(tools) },
        };

        let mut body = serde_json::to_value(&request)
            .wrap_err("failed to serialize Anthropic request")?;
        body.as_object_mut()
            .unwrap()
            .insert("stream".into(), true.into());

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .wrap_err("failed to send streaming request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            eyre::bail!("Anthropic API error: {status} - {text}");
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

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolSpec]>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: AnthropicContent,
}

/// Content can be plain text or multipart (text + images).
#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Parts(Vec<AnthropicContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
}

#[derive(Serialize)]
struct AnthropicImageSource {
    r#type: String,
    media_type: String,
    data: String,
}

fn build_anthropic_content(msg: &Message) -> AnthropicContent {
    let images: Vec<_> = msg
        .media
        .iter()
        .filter(|p| vision::is_image(p))
        .collect();

    if images.is_empty() {
        return AnthropicContent::Text(msg.content.clone());
    }

    let mut parts = Vec::new();
    for path in images {
        if let Ok((mime, data)) = vision::encode_image(path) {
            parts.push(AnthropicContentBlock::Image {
                source: AnthropicImageSource {
                    r#type: "base64".into(),
                    media_type: mime,
                    data,
                },
            });
        }
    }
    if !msg.content.is_empty() {
        parts.push(AnthropicContentBlock::Text {
            text: msg.content.clone(),
        });
    }
    AnthropicContent::Parts(parts)
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
}

// --- Streaming SSE helpers ---

#[derive(Default)]
struct AnthropicStreamState {
    block_to_tool: std::collections::HashMap<usize, usize>,
    tool_count: usize,
    input_tokens: u32,
}

fn map_anthropic_sse(
    state: &mut AnthropicStreamState,
    event: &crate::sse::SseEvent,
) -> Vec<StreamEvent> {
    let data: serde_json::Value = match serde_json::from_str(&event.data) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    match data["type"].as_str().unwrap_or("") {
        "message_start" => {
            if let Some(t) = data["message"]["usage"]["input_tokens"].as_u64() {
                state.input_tokens = t as u32;
            }
            vec![]
        }
        "content_block_start" => {
            let idx = data["index"].as_u64().unwrap_or(0) as usize;
            if data["content_block"]["type"].as_str() == Some("tool_use") {
                let tool_idx = state.tool_count;
                state.tool_count += 1;
                state.block_to_tool.insert(idx, tool_idx);
                vec![StreamEvent::ToolCallDelta {
                    index: tool_idx,
                    id: data["content_block"]["id"].as_str().map(String::from),
                    name: data["content_block"]["name"].as_str().map(String::from),
                    arguments_delta: String::new(),
                }]
            } else {
                vec![]
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
                }),
                StreamEvent::Done(stop_reason),
            ]
        }
        _ => vec![],
    }
}
