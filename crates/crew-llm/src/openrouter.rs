//! OpenRouter provider implementation (OpenAI-compatible API).

use async_trait::async_trait;
use crew_core::Message;
use eyre::{Result, WrapErr};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::vision;

use crate::config::ChatConfig;
use crate::openai::parse_openai_sse_events;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, StopReason, TokenUsage, ToolSpec};

/// OpenRouter provider (routes to many LLM providers).
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
        }
    }

    /// Create a provider using the OPENROUTER_API_KEY environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .wrap_err("OPENROUTER_API_KEY environment variable not set")?;
        Ok(Self::new(api_key, "anthropic/claude-sonnet-4-20250514"))
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatResponse> {
        let api_messages: Vec<ApiMessage> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    crew_core::MessageRole::System => "system",
                    crew_core::MessageRole::User => "user",
                    crew_core::MessageRole::Assistant => "assistant",
                    crew_core::MessageRole::Tool => "tool",
                };
                let content = build_api_content(m);
                ApiMessage {
                    role,
                    content,
                    tool_call_id: m.tool_call_id.as_deref(),
                    tool_calls: None,
                }
            })
            .collect();

        let api_tools: Option<Vec<ApiTool>> = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| ApiTool {
                        r#type: "function",
                        function: ApiFunction {
                            name: &t.name,
                            description: &t.description,
                            parameters: &t.input_schema,
                        },
                    })
                    .collect(),
            )
        };

        let request = ApiRequest {
            model: &self.model,
            messages: api_messages,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            tools: api_tools,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header(
                "HTTP-Referer",
                "https://github.com/heyong4725/crew-rs",
            )
            .header("X-Title", "crew-rs")
            .json(&request)
            .send()
            .await
            .wrap_err("failed to send request to OpenRouter")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eyre::bail!("OpenRouter API error: {status} - {body}");
        }

        let api_response: ApiResponse = response
            .json()
            .await
            .wrap_err("failed to parse OpenRouter response")?;

        let choice = api_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("no choices in OpenRouter response"))?;

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| crew_core::ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments: serde_json::from_str(&tc.function.arguments).unwrap_or_default(),
            })
            .collect();

        let stop_reason = match choice.finish_reason.as_str() {
            "stop" => StopReason::EndTurn,
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        Ok(ChatResponse {
            content: choice.message.content,
            tool_calls,
            stop_reason,
            usage: TokenUsage {
                input_tokens: api_response.usage.prompt_tokens,
                output_tokens: api_response.usage.completion_tokens,
            },
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatStream> {
        let api_messages: Vec<ApiMessage> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    crew_core::MessageRole::System => "system",
                    crew_core::MessageRole::User => "user",
                    crew_core::MessageRole::Assistant => "assistant",
                    crew_core::MessageRole::Tool => "tool",
                };
                ApiMessage {
                    role,
                    content: build_api_content(m),
                    tool_call_id: m.tool_call_id.as_deref(),
                    tool_calls: None,
                }
            })
            .collect();

        let api_tools: Option<Vec<ApiTool>> = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| ApiTool {
                        r#type: "function",
                        function: ApiFunction {
                            name: &t.name,
                            description: &t.description,
                            parameters: &t.input_schema,
                        },
                    })
                    .collect(),
            )
        };

        let request = ApiRequest {
            model: &self.model,
            messages: api_messages,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            tools: api_tools,
        };

        let mut body = serde_json::to_value(&request)
            .wrap_err("failed to serialize OpenRouter request")?;
        let obj = body.as_object_mut().unwrap();
        obj.insert("stream".into(), true.into());
        obj.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header(
                "HTTP-Referer",
                "https://github.com/heyong4725/crew-rs",
            )
            .header("X-Title", "crew-rs")
            .json(&body)
            .send()
            .await
            .wrap_err("failed to send streaming request to OpenRouter")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            eyre::bail!("OpenRouter API error: {status} - {text}");
        }

        let sse_stream = crate::sse::parse_sse_response(response);
        let event_stream = sse_stream
            .flat_map(|event| futures::stream::iter(parse_openai_sse_events(&event)));

        Ok(Box::pin(event_stream))
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openrouter"
    }
}

#[derive(Serialize)]
struct ApiRequest<'a> {
    model: &'a str,
    messages: Vec<ApiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool<'a>>>,
}

#[derive(Serialize)]
struct ApiMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<ApiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Parts(Vec<ApiContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ApiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ApiImageUrl },
}

#[derive(Serialize)]
struct ApiImageUrl {
    url: String,
}

fn build_api_content(msg: &Message) -> Option<ApiContent> {
    let images: Vec<_> = msg
        .media
        .iter()
        .filter(|p| vision::is_image(p))
        .collect();

    if images.is_empty() {
        if msg.content.is_empty() {
            return None;
        }
        return Some(ApiContent::Text(msg.content.clone()));
    }

    let mut parts = Vec::new();
    for path in images {
        if let Ok((mime, data)) = vision::encode_image(path) {
            parts.push(ApiContentPart::ImageUrl {
                image_url: ApiImageUrl {
                    url: format!("data:{mime};base64,{data}"),
                },
            });
        }
    }
    if !msg.content.is_empty() {
        parts.push(ApiContentPart::Text {
            text: msg.content.clone(),
        });
    }
    Some(ApiContent::Parts(parts))
}

#[derive(Serialize)]
struct ApiTool<'a> {
    r#type: &'a str,
    function: ApiFunction<'a>,
}

#[derive(Serialize)]
struct ApiFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

#[derive(Deserialize)]
struct ApiResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct ApiToolCall {
    id: String,
    function: FunctionCall,
}

#[derive(Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}
