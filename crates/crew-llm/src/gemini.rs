//! Google Gemini provider implementation.

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

/// Google Gemini provider.
pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GeminiProvider {
    /// Create a new Gemini provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    /// Create a provider using the GEMINI_API_KEY environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .wrap_err("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")?;
        Ok(Self::new(api_key, "gemini-2.0-flash"))
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatResponse> {
        // Build contents array from messages
        let mut contents: Vec<GeminiContent> = Vec::new();
        let mut system_instruction: Option<String> = None;

        for msg in messages {
            match msg.role {
                crew_core::MessageRole::System => {
                    system_instruction = Some(msg.content.clone());
                }
                crew_core::MessageRole::User | crew_core::MessageRole::Tool => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: build_gemini_parts(msg),
                    });
                }
                crew_core::MessageRole::Assistant => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![GeminiPart::Text {
                            text: msg.content.clone(),
                        }],
                    });
                }
            }
        }

        // Build tools array
        let gemini_tools: Option<Vec<GeminiTool>> = if tools.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(|t| GeminiFunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.input_schema.clone(),
                    })
                    .collect(),
            }])
        };

        let request = GeminiRequest {
            contents,
            system_instruction: system_instruction.map(|text| GeminiSystemInstruction {
                parts: vec![GeminiPart::Text { text }],
            }),
            tools: gemini_tools,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: config.max_tokens,
                temperature: config.temperature,
            }),
        };

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .wrap_err("failed to send request to Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eyre::bail!("Gemini API error: {status} - {body}");
        }

        let api_response: GeminiResponse = response
            .json()
            .await
            .wrap_err("failed to parse Gemini response")?;

        // Extract content from response
        let candidate = api_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("no candidates in Gemini response"))?;

        let mut content = None;
        let mut tool_calls = Vec::new();

        for part in candidate.content.parts {
            match part {
                GeminiPart::Text { text } => {
                    content = Some(text);
                }
                GeminiPart::FunctionCall { function_call } => {
                    tool_calls.push(crew_core::ToolCall {
                        id: format!("call_{}", tool_calls.len()),
                        name: function_call.name,
                        arguments: function_call.args,
                    });
                }
                GeminiPart::InlineData { .. } => {
                    // InlineData is only used in requests, not responses
                }
            }
        }

        let stop_reason = match candidate.finish_reason.as_deref() {
            Some("STOP") => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            _ if !tool_calls.is_empty() => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        };

        // Gemini doesn't always return usage in the same format
        let usage = api_response.usage_metadata.unwrap_or(GeminiUsageMetadata {
            prompt_token_count: 0,
            candidates_token_count: 0,
        });

        Ok(ChatResponse {
            content,
            tool_calls,
            stop_reason,
            usage: TokenUsage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count,
            },
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatStream> {
        let mut contents: Vec<GeminiContent> = Vec::new();
        let mut system_instruction: Option<String> = None;

        for msg in messages {
            match msg.role {
                crew_core::MessageRole::System => {
                    system_instruction = Some(msg.content.clone());
                }
                crew_core::MessageRole::User | crew_core::MessageRole::Tool => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: build_gemini_parts(msg),
                    });
                }
                crew_core::MessageRole::Assistant => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![GeminiPart::Text {
                            text: msg.content.clone(),
                        }],
                    });
                }
            }
        }

        let gemini_tools: Option<Vec<GeminiTool>> = if tools.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(|t| GeminiFunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.input_schema.clone(),
                    })
                    .collect(),
            }])
        };

        let request = GeminiRequest {
            contents,
            system_instruction: system_instruction.map(|text| GeminiSystemInstruction {
                parts: vec![GeminiPart::Text { text }],
            }),
            tools: gemini_tools,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: config.max_tokens,
                temperature: config.temperature,
            }),
        };

        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, self.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .wrap_err("failed to send streaming request to Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            eyre::bail!("Gemini API error: {status} - {text}");
        }

        let sse_stream = crate::sse::parse_sse_response(response);
        let state = GeminiStreamState::default();
        let event_stream = sse_stream
            .scan(state, |state, event| {
                let events = map_gemini_sse(state, &event);
                futures::future::ready(Some(events))
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(event_stream))
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "gemini"
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
}

#[derive(Serialize, Deserialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

fn build_gemini_parts(msg: &Message) -> Vec<GeminiPart> {
    let images: Vec<_> = msg
        .media
        .iter()
        .filter(|p| vision::is_image(p))
        .collect();

    if images.is_empty() {
        return vec![GeminiPart::Text {
            text: msg.content.clone(),
        }];
    }

    let mut parts = Vec::new();
    for path in images {
        if let Ok((mime, data)) = vision::encode_image(path) {
            parts.push(GeminiPart::InlineData {
                inline_data: GeminiInlineData {
                    mime_type: mime,
                    data,
                },
            });
        }
    }
    if !msg.content.is_empty() {
        parts.push(GeminiPart::Text {
            text: msg.content.clone(),
        });
    }
    parts
}

#[derive(Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

// --- Streaming SSE helpers ---

#[derive(Default)]
struct GeminiStreamState {
    tool_count: usize,
    has_tool_calls: bool,
}

fn map_gemini_sse(
    state: &mut GeminiStreamState,
    event: &crate::sse::SseEvent,
) -> Vec<StreamEvent> {
    let data: serde_json::Value = match serde_json::from_str(&event.data) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let mut events = Vec::new();

    if let Some(candidates) = data["candidates"].as_array() {
        if let Some(candidate) = candidates.first() {
            if let Some(parts) = candidate["content"]["parts"].as_array() {
                for part in parts {
                    if let Some(text) = part["text"].as_str() {
                        if !text.is_empty() {
                            events.push(StreamEvent::TextDelta(text.to_string()));
                        }
                    }
                    if let Some(fc) = part.get("functionCall") {
                        state.has_tool_calls = true;
                        let name = fc["name"].as_str().unwrap_or("").to_string();
                        let args = fc
                            .get("args")
                            .cloned()
                            .unwrap_or(serde_json::Value::Object(Default::default()));
                        events.push(StreamEvent::ToolCallDelta {
                            index: state.tool_count,
                            id: Some(format!("call_{}", state.tool_count)),
                            name: Some(name),
                            arguments_delta: args.to_string(),
                        });
                        state.tool_count += 1;
                    }
                }
            }

            if let Some(reason) = candidate["finishReason"].as_str() {
                let stop_reason = match reason {
                    "STOP" if state.has_tool_calls => StopReason::ToolUse,
                    "STOP" => StopReason::EndTurn,
                    "MAX_TOKENS" => StopReason::MaxTokens,
                    _ if state.has_tool_calls => StopReason::ToolUse,
                    _ => StopReason::EndTurn,
                };
                events.push(StreamEvent::Done(stop_reason));
            }
        }
    }

    if let Some(usage) = data.get("usageMetadata").filter(|u| !u.is_null()) {
        let input = usage["promptTokenCount"].as_u64().unwrap_or(0) as u32;
        let output = usage["candidatesTokenCount"].as_u64().unwrap_or(0) as u32;
        if input > 0 || output > 0 {
            events.push(StreamEvent::Usage(TokenUsage {
                input_tokens: input,
                output_tokens: output,
            }));
        }
    }

    events
}
