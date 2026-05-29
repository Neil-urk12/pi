//! OpenAI-compatible API provider.
//!
//! Implements the [`Provider`] trait for OpenAI and OpenAI-compatible APIs
//! (Azure OpenAI, Ollama, vLLM, etc.).
//!
//! Supports:
//! - Chat completions (streaming and non-streaming)
//! - Function/tool calling
//! - Vision (image content blocks)

use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use rpi_core::{
    AgentConfig, ChatResponse, ChatStream, ContentBlock, FinishReason, FunctionCall, Message,
    MessageContent, PiError, Result, Role, StreamChunk, ToolCall, ToolCallDelta, ToolDefinition,
    Usage,
};

use crate::streaming::{SseEvent, sse_stream};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// An OpenAI-compatible chat completion provider.
///
/// Works with the OpenAI API directly as well as any service that implements
/// the same wire format (Azure OpenAI, Ollama `/v1/chat/completions`, vLLM, etc.).
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAiProvider {
    /// Create a new provider for the official OpenAI API.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a provider pointing at a custom base URL (Azure, Ollama, etc.).
    ///
    /// `api_key` may be `None` for services that don't require authentication.
    pub fn with_base_url(base_url: &str, api_key: Option<&str>) -> Self {
        Self {
            api_key: api_key.unwrap_or("").to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("valid header value"),
        );
        if !self.api_key.is_empty() {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key)
                    .parse()
                    .expect("valid header value"),
            );
        }
        headers
    }
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

#[async_trait]
impl rpi_core::Provider for OpenAiProvider {
    fn id(&self) -> &str {
        "openai"
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatResponse> {
        let body = build_request(model, messages, tools, config, false);

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| PiError::Provider(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(PiError::Provider(format!(
                "OpenAI API returned {status}: {text}"
            )));
        }

        let resp: OpenAiResponse = response
            .json()
            .await
            .map_err(|e| PiError::Provider(format!("Failed to parse response: {e}")))?;

        parse_full_response(resp)
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatStream> {
        let body = build_request(model, messages, tools, config, true);

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| PiError::Provider(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(PiError::Provider(format!(
                "OpenAI API returned {status}: {text}"
            )));
        }

        Ok(Box::pin(openai_sse_to_chunks(sse_stream(response))))
    }
}

// ---------------------------------------------------------------------------
// SSE → StreamChunk conversion
// ---------------------------------------------------------------------------

fn openai_sse_to_chunks(
    stream: impl Stream<Item = anyhow::Result<SseEvent>> + Send + 'static,
) -> impl Stream<Item = Result<StreamChunk>> + Send {
    stream.map(|event_result| match event_result {
        Err(e) => Err(PiError::Provider(format!("SSE error: {e}"))),
        Ok(event) => {
            if event.is_done() {
                return Ok(StreamChunk {
                    delta: None,
                    tool_calls: Vec::new(),
                    finish_reason: None,
                    usage: None,
                });
            }

            let chunk: OpenAiStreamChunk = event
                .json()
                .map_err(|e| PiError::Provider(format!("Failed to parse stream chunk: {e}")))?;

            let choice = match chunk.choices.first() {
                Some(c) => c,
                None => {
                    return Ok(StreamChunk {
                        delta: None,
                        tool_calls: Vec::new(),
                        finish_reason: None,
                        usage: None,
                    });
                }
            };

            let finish_reason = choice
                .finish_reason
                .as_deref()
                .and_then(parse_finish_reason);

            let delta = choice.delta.as_ref().and_then(|d| d.content.clone());

            let tool_calls = choice
                .delta
                .as_ref()
                .and_then(|d| d.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter()
                        .map(|tc| ToolCallDelta {
                            index: tc.index,
                            id: tc.id.clone(),
                            name: tc.function.as_ref().and_then(|f| f.name.clone()),
                            arguments_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                        })
                        .collect()
                })
                .unwrap_or_default();

            Ok(StreamChunk {
                delta,
                tool_calls,
                finish_reason,
                usage: chunk.usage.map(|u| Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                }),
            })
        }
    })
}

// ---------------------------------------------------------------------------
// Request / response types (internal)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAiFunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiToolFunction,
}

#[derive(Serialize)]
struct OpenAiToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

// -- Non-streaming response --

#[derive(Deserialize, Debug)]
struct OpenAiResponse {
    #[allow(dead_code)]
    id: String,
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChoice {
    message: Option<OpenAiMessageResponse>,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAiMessageResponse {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// -- Streaming response --

#[derive(Deserialize, Debug)]
struct OpenAiStreamChunk {
    #[allow(dead_code)]
    id: String,
    choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize, Debug)]
struct OpenAiStreamChoice {
    delta: Option<OpenAiDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAiDelta {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Deserialize, Debug)]
struct OpenAiToolCallDelta {
    index: u32,
    id: Option<String>,
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct OpenAiFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Build an [`OpenAiRequest`] from rpi-core types.
fn build_request(
    model: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    config: &AgentConfig,
    stream: bool,
) -> OpenAiRequest {
    OpenAiRequest {
        model: model.to_string(),
        messages: to_openai_messages(messages),
        tools: if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| OpenAiTool {
                        tool_type: "function".to_string(),
                        function: OpenAiToolFunction {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: t.parameters.clone(),
                        },
                    })
                    .collect(),
            )
        },
        stream,
        stream_options: if stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
        max_tokens: config.max_tokens,
        temperature: config.temperature,
    }
}

/// Convert rpi-core [`Message`]s to OpenAI message format.
fn to_openai_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            }
            .to_string();

            let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| OpenAiToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: OpenAiFunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect()
            });

            OpenAiMessage {
                role,
                content: content_to_openai_json(msg.content.as_ref()),
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
            }
        })
        .collect()
}

/// Convert rpi-core [`MessageContent`] to an OpenAI-compatible JSON value.
fn content_to_openai_json(content: Option<&MessageContent>) -> Option<serde_json::Value> {
    match content? {
        MessageContent::Text(text) => Some(serde_json::Value::String(text.clone())),
        MessageContent::Blocks(blocks) => {
            let items: Vec<serde_json::Value> = blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => {
                        Some(serde_json::json!({"type": "text", "text": text}))
                    }
                    ContentBlock::Image { media_type, data } => Some(serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{media_type};base64,{data}")
                        }
                    })),
                    // ToolUse/ToolResult are mapped via Message-level fields for OpenAI.
                    _ => None,
                })
                .collect();

            if items.is_empty() {
                None
            } else {
                Some(serde_json::Value::Array(items))
            }
        }
    }
}

/// Convert a full (non-streaming) OpenAI response into a [`ChatResponse`].
fn parse_full_response(resp: OpenAiResponse) -> Result<ChatResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| PiError::Provider("OpenAI response contained no choices".to_string()))?;

    let msg_resp = choice
        .message
        .ok_or_else(|| PiError::Provider("OpenAI choice had no message".to_string()))?;

    let finish_reason = choice
        .finish_reason
        .as_deref()
        .and_then(parse_finish_reason)
        .unwrap_or(FinishReason::Stop);

    let tool_calls = msg_resp.tool_calls.map(|tcs| {
        tcs.into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                function: FunctionCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect()
    });

    let content = match (msg_resp.content, &tool_calls) {
        (Some(text), _) if !text.is_empty() => Some(MessageContent::Text(text)),
        _ => None,
    };

    let message = Message {
        role: Role::Assistant,
        content,
        tool_calls,
        tool_call_id: None,
        name: None,
    };

    let usage = resp
        .usage
        .map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
        .unwrap_or(Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });

    Ok(ChatResponse {
        message,
        finish_reason,
        usage,
    })
}

/// Map an OpenAI finish-reason string to our enum.
fn parse_finish_reason(reason: &str) -> Option<FinishReason> {
    match reason {
        "stop" => Some(FinishReason::Stop),
        "length" => Some(FinishReason::Length),
        "tool_calls" | "function_call" => Some(FinishReason::ToolCalls),
        "content_filter" => Some(FinishReason::ContentFilter),
        _ => None,
    }
}
