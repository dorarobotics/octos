//! Google Gemini provider implementation.

use async_trait::async_trait;
use crew_core::Message;
use eyre::{Result, WrapErr};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use secrecy::{ExposeSecret, SecretString};

use crate::vision;

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec};

/// Google Gemini provider.
pub struct GeminiProvider {
    client: Client,
    api_key: SecretString,
    model: String,
    base_url: String,
}

impl GeminiProvider {
    /// Create a new Gemini provider.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: SecretString::from(api_key.into()),
            model: model.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    /// Create a provider using the GEMINI_API_KEY environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .wrap_err("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")?;
        Ok(Self::new(api_key, "gemini-2.5-flash"))
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
        let (contents, system_instruction) = build_gemini_contents(messages);

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

        let url = format!("{}/models/{}:generateContent", self.base_url, self.model);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", self.api_key.expose_secret())
            .json(&request)
            .send()
            .await
            .wrap_err("failed to send request to Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eyre::bail!(
                "Gemini API error: {status} - {}",
                crate::provider::truncate_error_body(&body)
            );
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
                GeminiPart::FunctionCall {
                    function_call,
                    thought_signature,
                } => {
                    let metadata = thought_signature
                        .map(|sig| serde_json::json!({ "thought_signature": sig }));
                    tool_calls.push(crew_core::ToolCall {
                        id: format!("call_{}", tool_calls.len()),
                        name: function_call.name,
                        arguments: function_call.args,
                        metadata,
                    });
                }
                GeminiPart::InlineData { .. } | GeminiPart::FunctionResponse { .. } => {
                    // InlineData and FunctionResponse are only used in requests
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
            reasoning_content: None,
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
        let (contents, system_instruction) = build_gemini_contents(messages);

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
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, self.model
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", self.api_key.expose_secret())
            .json(&request)
            .send()
            .await
            .wrap_err("failed to send streaming request to Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            eyre::bail!(
                "Gemini API error: {status} - {}",
                crate::provider::truncate_error_body(&text)
            );
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
        /// Gemini thinking models require this signature to be echoed back.
        /// This is at the part level, NOT inside the functionCall object.
        #[serde(
            rename = "thoughtSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

/// Build the Gemini `contents` array and optional system instruction from messages.
///
/// Gemini requires:
/// - Assistant messages with tool calls → `model` role with `functionCall` parts
/// - Tool result messages → `user` role with `functionResponse` parts
/// - Consecutive same-role messages are merged (Gemini rejects adjacent same-role turns)
fn build_gemini_contents(messages: &[Message]) -> (Vec<GeminiContent>, Option<String>) {
    let mut contents: Vec<GeminiContent> = Vec::new();
    let mut system_instruction: Option<String> = None;

    // Map tool_call_id → function name so tool results can reference the right name.
    let mut call_id_to_name: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for msg in messages {
        match msg.role {
            crew_core::MessageRole::System => {
                system_instruction = Some(msg.content.clone());
            }
            crew_core::MessageRole::User => {
                let parts = build_user_parts(msg);
                push_or_merge(&mut contents, "user", parts);
            }
            crew_core::MessageRole::Assistant => {
                let mut parts = Vec::new();
                // Include text content if non-empty.
                if !msg.content.is_empty() {
                    parts.push(GeminiPart::Text {
                        text: msg.content.clone(),
                    });
                }
                // Include functionCall parts for any tool calls the model made.
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        call_id_to_name.insert(tc.id.clone(), tc.name.clone());
                        // Restore thought_signature from metadata if present.
                        let thought_signature = tc
                            .metadata
                            .as_ref()
                            .and_then(|m| m.get("thought_signature"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        parts.push(GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: tc.name.clone(),
                                args: tc.arguments.clone(),
                            },
                            thought_signature,
                        });
                    }
                }
                // Gemini requires at least one part; add empty text if everything was empty.
                if parts.is_empty() {
                    parts.push(GeminiPart::Text {
                        text: String::new(),
                    });
                }
                push_or_merge(&mut contents, "model", parts);
            }
            crew_core::MessageRole::Tool => {
                // Resolve function name from the matching tool call.
                let name = msg
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| call_id_to_name.get(id))
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());

                let part = GeminiPart::FunctionResponse {
                    function_response: GeminiFunctionResponse {
                        name,
                        response: serde_json::json!({ "content": msg.content }),
                    },
                };
                push_or_merge(&mut contents, "user", vec![part]);
            }
        }
    }
    (contents, system_instruction)
}

/// Merge parts into the last content entry if roles match (Gemini rejects adjacent same-role).
fn push_or_merge(contents: &mut Vec<GeminiContent>, role: &str, parts: Vec<GeminiPart>) {
    if let Some(last) = contents.last_mut() {
        if last.role == role {
            last.parts.extend(parts);
            return;
        }
    }
    contents.push(GeminiContent {
        role: role.to_string(),
        parts,
    });
}

fn build_user_parts(msg: &Message) -> Vec<GeminiPart> {
    let images: Vec<_> = msg.media.iter().filter(|p| vision::is_image(p)).collect();

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

fn map_gemini_sse(state: &mut GeminiStreamState, event: &crate::sse::SseEvent) -> Vec<StreamEvent> {
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
                        // Capture thought_signature for Gemini thinking models.
                        // thoughtSignature is at the part level, not inside functionCall.
                        let thought_sig = part
                            .get("thoughtSignature")
                            .and_then(|v| v.as_str())
                            .map(|s| serde_json::json!({ "thought_signature": s }));
                        events.push(StreamEvent::ToolCallDelta {
                            index: state.tool_count,
                            id: Some(format!("call_{}", state.tool_count)),
                            name: Some(name),
                            arguments_delta: args.to_string(),
                        });
                        // Emit metadata as a separate event so the agent can store it.
                        if let Some(meta) = thought_sig {
                            events.push(StreamEvent::ToolCallMetadata {
                                index: state.tool_count,
                                metadata: meta,
                            });
                        }
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
