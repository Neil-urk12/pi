//! Anthropic Messages API provider.
//!
//! Implements the [`Provider`] trait for the Anthropic Claude API.
//!
//! Supports:
//! - Messages API (streaming and non-streaming)
//! - Tool use / function calling
//! - Vision (image content blocks)
//! - System prompt (top-level `system` parameter)

use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

    use rpi_core::{
    AgentConfig, AnthropicCompat, ChatResponse, ChatStream, CompatFlags, ContentBlock,
    FinishReason, FunctionCall, Message, MessageContent, ModelInputKind, PiError,
    ProviderCapabilities, ProviderError, ProviderMetadata, Result, Role, StreamChunk, ToolCall,
    ToolCallDelta, ToolDefinition, Usage, normalize_messages_with_input,
};

use crate::streaming::{SseEvent, sse_stream};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const ANTHROPIC_TOOL_CALL_ID_MAX_LEN: usize = 64;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The Anthropic Claude chat provider.
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl AnthropicProvider {
    /// Create a new provider targeting the official Anthropic API.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: crate::default_http_client(),
        }
    }

    /// Create a provider with a custom base URL.
    pub fn with_base_url(base_url: &str, api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: crate::default_http_client(),
        }
    }

    fn build_headers(&self) -> Result<reqwest::header::HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("valid header value"),
        );
        headers.insert(
            "x-api-key",
            reqwest::header::HeaderValue::from_str(&self.api_key)
                .map_err(|_| ProviderError::Other {
                    message: "invalid characters in API key".to_string(),
                })?,
        );
        headers.insert(
            "anthropic-version",
            API_VERSION.parse().expect("valid header value"),
        );
        Ok(headers)
    }
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

#[async_trait]
impl rpi_core::Provider for AnthropicProvider {
    fn id(&self) -> &str {
        "anthropic"
    }

    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: "anthropic".to_string(),
            name: "Anthropic".to_string(),
            supported_apis: vec!["anthropic-messages".to_string()],
            capabilities: ProviderCapabilities {
                vision: true,
                function_calling: true,
                json_mode: false,
                streaming: true,
                thinking: true,
            },
            compat: CompatFlags {
                anthropic: Some(detect_anthropic_compat(&self.base_url)),
                ..CompatFlags::default()
            },
        }
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
            .post(format!("{}/v1/messages", self.base_url))
            .headers(self.build_headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                message: format!("HTTP request failed: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            let body_preview = if text.len() > 512 {
                format!("{}...(truncated)", &text[..text.floor_char_boundary(512)])
            } else {
                text
            };
            return Err(ProviderError::http_status(status.as_u16(), body_preview).into());
        }

        let resp: AnthropicResponse =
            response
                .json()
                .await
                .map_err(|e| ProviderError::MalformedResponse {
                    message: format!("failed to parse response: {e}"),
                })?;

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
            .post(format!("{}/v1/messages", self.base_url))
            .headers(self.build_headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                message: format!("HTTP request failed: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            let body_preview = if text.len() > 512 {
                format!("{}...(truncated)", &text[..text.floor_char_boundary(512)])
            } else {
                text
            };
            return Err(ProviderError::http_status(status.as_u16(), body_preview).into());
        }

        Ok(Box::pin(anthropic_sse_to_chunks(sse_stream(response))))
    }
}

fn detect_anthropic_compat(base_url: &str) -> AnthropicCompat {
    let base_url = base_url.to_ascii_lowercase();
    let is_fireworks = base_url.contains("fireworks");
    let is_cloudflare_ai_gateway_anthropic =
        base_url.contains("gateway.ai.cloudflare.com") && base_url.contains("anthropic");

    AnthropicCompat {
        supports_eager_tool_input_streaming: Some(!is_fireworks),
        supports_long_cache_retention: Some(!is_fireworks),
        send_session_affinity_headers: Some(is_fireworks || is_cloudflare_ai_gateway_anthropic),
        supports_cache_control_on_tools: Some(!is_fireworks),
        force_adaptive_thinking: None,
    }
}

// ---------------------------------------------------------------------------
// SSE → StreamChunk conversion
// ---------------------------------------------------------------------------

/// Internal state for reassembling Anthropic streaming events into
/// [`StreamChunk`]s. Anthropic sends fine-grained events (content_block_start,
/// content_block_delta, content_block_stop) that must be collapsed into the
/// unified chunk format.
struct AnthropicStreamState {
    /// Index of the content block currently being streamed.
    current_block_index: Option<u32>,
    /// Whether the stream reached Anthropic's terminal message_stop event.
    seen_message_stop: bool,
    /// The type of the current content block ("text" or "tool_use").
    current_block_type: Option<String>,
    /// Accumulated tool use data for the current block.
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    current_tool_args: String,
}

impl AnthropicStreamState {
    fn new() -> Self {
        Self {
            current_block_index: None,
            seen_message_stop: false,
            current_block_type: None,
            current_tool_id: None,
            current_tool_name: None,
            current_tool_args: String::new(),
        }
    }
}

fn anthropic_sse_to_chunks(
    stream: impl Stream<Item = anyhow::Result<SseEvent>> + Send + 'static,
) -> impl Stream<Item = Result<StreamChunk>> + Send {
    use async_stream::stream;

    let mut inner = Box::pin(stream);
    let mut state = AnthropicStreamState::new();

    stream! {
        while let Some(event_result) = inner.next().await {
            match event_result {
                Err(e) => {
                    yield Err(PiError::provider(format!("SSE error: {e}")));
                    return;
                }
                Ok(event) => {
                    match process_anthropic_event(&event, &mut state) {
                        Ok(Some(chunk)) => yield Ok(chunk),
                        Ok(None) => continue,
                        Err(e) => yield Err(e),
                    }
                }
            }
        }

        if !state.seen_message_stop {
            yield Err(PiError::provider("Anthropic stream ended before message_stop"));
        }
    }
}

/// Process a single Anthropic SSE event, updating stream state as needed.
///
/// Returns `Ok(None)` for events that don't produce a stream chunk
/// (e.g., `message_start`, `content_block_start`).
fn process_anthropic_event(
    event: &SseEvent,
    state: &mut AnthropicStreamState,
) -> Result<Option<StreamChunk>> {
    let event_type = event.event.as_deref().unwrap_or("");

    match event_type {
        "message_start" | "ping" => {
            // Control events — no chunk to emit.
            Ok(None)
        }
        "message_stop" => {
            state.seen_message_stop = true;
            Ok(None)
        }
        "content_block_start" => {
            let block_start: ContentBlockStartEvent = event.json().map_err(|e| {
                PiError::provider(format!("Failed to parse content_block_start: {e}"))
            })?;
            state.current_block_index = Some(block_start.index);
            match &block_start.content_block {
                AnthropicContentBlockStart::Text => {
                    state.current_block_type = Some("text".to_string());
                }
                AnthropicContentBlockStart::ToolUse { id, name } => {
                    state.current_block_type = Some("tool_use".to_string());
                    state.current_tool_id = Some(id.clone());
                    state.current_tool_name = Some(name.clone());
                    state.tool_args_start();
                }
                AnthropicContentBlockStart::Thinking => {
                    state.current_block_type = Some("thinking".to_string());
                }
            }
            Ok(None)
        }
        "content_block_delta" => {
            let delta: ContentBlockDeltaEvent = event.json().map_err(|e| {
                PiError::provider(format!("Failed to parse content_block_delta: {e}"))
            })?;

            match &delta.delta {
                AnthropicDelta::TextDelta { text } => Ok(Some(StreamChunk {
                    delta: Some(text.clone()),
                    tool_calls: Vec::new(),
                    finish_reason: None,
                    usage: None,
                })),
                AnthropicDelta::ThinkingDelta { thinking } => Ok(Some(StreamChunk {
                    delta: Some(thinking.clone()),
                    tool_calls: Vec::new(),
                    finish_reason: None,
                    usage: None,
                })),
                AnthropicDelta::InputJsonDelta { partial_json } => {
                    state.tool_args_push(partial_json);
                    let index = delta.index;
                    Ok(Some(StreamChunk {
                        delta: None,
                        tool_calls: vec![ToolCallDelta {
                            index,
                            id: None,
                            name: None,
                            arguments_delta: Some(partial_json.clone()),
                        }],
                        finish_reason: None,
                        usage: None,
                    }))
                }
            }
        }
        "content_block_stop" => {
            // If we were streaming a tool_use block, emit the complete tool call info.
            if state.current_block_type.as_deref() == Some("tool_use") {
                let index = state.current_block_index.unwrap_or(0);
                let id = state.tool_id_take();
                let name = state.tool_name_take();
                let chunk = StreamChunk {
                    delta: None,
                    tool_calls: vec![ToolCallDelta {
                        index,
                        id,
                        name,
                        // Final empty delta to signal completion.
                        arguments_delta: None,
                    }],
                    finish_reason: None,
                    usage: None,
                };
                state.tool_args_clear();
                state.current_block_type = None;
                return Ok(Some(chunk));
            }
            state.current_block_type = None;
            Ok(None)
        }
        "message_delta" => {
            let msg_delta: MessageDeltaEvent = event
                .json()
                .map_err(|e| PiError::provider(format!("Failed to parse message_delta: {e}")))?;
            let finish_reason = msg_delta
                .delta
                .stop_reason
                .as_deref()
                .and_then(parse_stop_reason);
            Ok(Some(StreamChunk {
                delta: None,
                tool_calls: Vec::new(),
                finish_reason,
                usage: msg_delta.usage.map(|u| Usage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: u.input_tokens + u.output_tokens,
                }),
            }))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Request / response types (internal)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentValue>,
}

/// A content block in an Anthropic request message.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentValue {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// -- Non-streaming response --

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    #[allow(dead_code)]
    id: String,
    content: Vec<AnthropicResponseContent>,
    #[serde(default)]
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
}

#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// -- Streaming event types --

#[derive(Deserialize, Debug)]
struct ContentBlockStartEvent {
    index: u32,
    content_block: AnthropicContentBlockStart,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlockStart {
    Text,
    ToolUse { id: String, name: String },
    Thinking,
}

#[derive(Deserialize, Debug)]
struct ContentBlockDeltaEvent {
    index: u32,
    delta: AnthropicDelta,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
}

#[derive(Deserialize, Debug)]
struct MessageDeltaEvent {
    delta: MessageDeltaInner,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize, Debug)]
struct MessageDeltaInner {
    stop_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Stream state helpers (on AnthropicStreamState)
// ---------------------------------------------------------------------------

impl AnthropicStreamState {
    fn tool_args_start(&mut self) {
        self.current_tool_args.clear();
    }

    fn tool_args_push(&mut self, fragment: &str) {
        self.current_tool_args.push_str(fragment);
    }

    fn tool_args_clear(&mut self) {
        self.current_tool_args.clear();
    }

    fn tool_id_take(&mut self) -> Option<String> {
        self.current_tool_id.take()
    }

    fn tool_name_take(&mut self) -> Option<String> {
        self.current_tool_name.take()
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Build an [`AnthropicRequest`] from rpi-core types.
fn build_request(
    model: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    config: &AgentConfig,
    stream: bool,
) -> AnthropicRequest {
    let normalized_messages = normalize_messages_with_input(
        messages,
        &[ModelInputKind::Text, ModelInputKind::Image],
        Some(&normalize_anthropic_tool_call_id),
        true,  // Preserve thinking blocks - Anthropic supports them natively
    );
    let (system, anthropic_msgs) = to_anthropic_messages(&normalized_messages);

    AnthropicRequest {
        model: model.to_string(),
        max_tokens: config.max_tokens.unwrap_or(4096),
        system: config.system_prompt.clone().or(system),
        messages: anthropic_msgs,
        tools: if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| AnthropicTool {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: t.parameters.clone(),
                    })
                    .collect(),
            )
        },
        stream,
        temperature: config.temperature,
    }
}

fn normalize_anthropic_tool_call_id(id: &str, _message: &Message) -> String {
    let normalized = id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .take(ANTHROPIC_TOOL_CALL_ID_MAX_LEN)
        .collect::<String>();

    if normalized.is_empty() {
        "tool_call".to_string()
    } else {
        normalized
    }
}

/// Convert rpi-core messages into Anthropic format.
///
/// Returns `(system_prompt, messages)` — system messages are extracted
/// separately because the Anthropic API takes `system` as a top-level field.
fn to_anthropic_messages(messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut result: Vec<AnthropicMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                if let Some(text) = extract_text_content(&msg.content) {
                    system_parts.push(text);
                }
            }
            Role::User => {
                let blocks = content_to_anthropic_blocks(&msg.content);
                if !blocks.is_empty() {
                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: blocks,
                    });
                }
            }
            Role::Assistant => {
                let mut blocks = Vec::new();

                // Text content.
                if let Some(text) = extract_text_content(&msg.content) {
                    blocks.push(AnthropicContentValue::Text { text });
                }

                // Thinking content blocks.
                if let Some(MessageContent::Blocks(content_blocks)) = &msg.content {
                    for block in content_blocks {
                        if let ContentBlock::Thinking { thinking, thinking_signature, .. } = block {
                            blocks.push(AnthropicContentValue::Thinking {
                                thinking: thinking.clone(),
                                signature: thinking_signature.clone(),
                            });
                        }
                    }
                }

                // Tool calls → tool_use blocks.
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| {
                                serde_json::Value::String(tc.function.arguments.clone())
                            });
                        blocks.push(AnthropicContentValue::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        });
                    }
                }

                if !blocks.is_empty() {
                    result.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: blocks,
                    });
                }
            }
            Role::Tool => {
                // Anthropic expects tool results in a user message with tool_result blocks.
                let (content_text, is_error) = match &msg.content {
                    Some(MessageContent::Blocks(blocks)) => {
                        let mut texts = Vec::new();
                        let mut has_error = false;
                        for block in blocks {
                            match block {
                                ContentBlock::ToolResult {
                                    content, is_error, ..
                                } => {
                                    texts.push(content.clone());
                                    if *is_error {
                                        has_error = true;
                                    }
                                }
                                ContentBlock::Text { text } => {
                                    texts.push(text.clone());
                                }
                                _ => {}
                            }
                        }
                        (texts.join("\n"), has_error)
                    }
                    Some(MessageContent::Text(text)) => (text.clone(), false),
                    None => (String::new(), false),
                };

                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![AnthropicContentValue::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: content_text,
                        is_error,
                    }],
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    (system, result)
}

/// Extract plain text from [`MessageContent`], if available.
fn extract_text_content(content: &Option<MessageContent>) -> Option<String> {
    match content.as_ref()? {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Blocks(blocks) => {
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    ContentBlock::Thinking { .. } => None,  // exclude thinking from text extraction
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
    }
}

/// Convert rpi-core [`MessageContent`] into Anthropic content blocks.
fn content_to_anthropic_blocks(content: &Option<MessageContent>) -> Vec<AnthropicContentValue> {
    match content {
        None => Vec::new(),
        Some(MessageContent::Text(text)) => {
            vec![AnthropicContentValue::Text { text: text.clone() }]
        }
        Some(MessageContent::Blocks(blocks)) => blocks
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => AnthropicContentValue::Text { text: text.clone() },
                ContentBlock::Thinking { thinking, thinking_signature, .. } => AnthropicContentValue::Thinking {
                    thinking: thinking.clone(),
                    signature: thinking_signature.clone(),
                },
                ContentBlock::Image { media_type, data } => AnthropicContentValue::Image {
                    source: AnthropicImageSource {
                        source_type: "base64".to_string(),
                        media_type: media_type.clone(),
                        data: data.clone(),
                    },
                },
                ContentBlock::ToolUse { id, name, input } => AnthropicContentValue::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                },
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => AnthropicContentValue::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                },
            })
            .collect(),
    }
}

/// Convert a full (non-streaming) Anthropic response into a [`ChatResponse`].
fn parse_full_response(resp: AnthropicResponse) -> Result<ChatResponse> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block {
            AnthropicResponseContent::Text { text } => {
                text_parts.push(text.clone());
            }
            AnthropicResponseContent::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
            }
            AnthropicResponseContent::Thinking { thinking, signature } => {
                // Thinking blocks are model-internal reasoning. Skip for user-visible content.
                // The signature is preserved when round-tripping through build_request.
                let _ = (thinking, signature);
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(MessageContent::Text(text_parts.join("")))
    };

    let tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let finish_reason = resp
        .stop_reason
        .as_deref()
        .and_then(parse_stop_reason)
        .unwrap_or(FinishReason::Stop);

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        finish_reason,
        usage: Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        },
    })
}

/// Map an Anthropic stop_reason to our [`FinishReason`].
fn parse_stop_reason(reason: &str) -> Option<FinishReason> {
    match reason {
        "end_turn" | "stop_sequence" => Some(FinishReason::Stop),
        "max_tokens" => Some(FinishReason::Length),
        "tool_use" => Some(FinishReason::ToolCalls),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpi_core::{ModelId, Provider, ProviderError};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    struct MockHttpResponse {
        status_line: &'static str,
        content_type: &'static str,
        body: &'static str,
    }

    impl MockHttpResponse {
        fn json(status_line: &'static str, body: &'static str) -> Self {
            Self {
                status_line,
                content_type: "application/json",
                body,
            }
        }
    }

    async fn spawn_mock_anthropic_server(
        response: MockHttpResponse,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock server should bind");
        let addr = listener
            .local_addr()
            .expect("mock server should have a local address");

        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener
                .accept()
                .await
                .expect("mock server should accept one request");
            let request = read_http_request(&mut socket).await;
            write_http_response(&mut socket, response).await;
            request
        });

        (format!("http://{addr}"), handle)
    }

    async fn read_http_request(socket: &mut TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];

        loop {
            let bytes_read = socket
                .read(&mut chunk)
                .await
                .expect("mock server should read request bytes");
            assert!(
                bytes_read > 0,
                "client closed connection before completing request"
            );
            buffer.extend_from_slice(&chunk[..bytes_read]);

            let Some(header_end) = find_header_end(&buffer) else {
                continue;
            };
            let headers = String::from_utf8_lossy(&buffer[..header_end]);
            let body_length = request_content_length(&headers);
            if buffer.len() >= header_end + 4 + body_length {
                break;
            }
        }

        String::from_utf8(buffer).expect("request should be valid UTF-8")
    }

    fn find_header_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn request_content_length(headers: &str) -> usize {
        headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0)
    }

    async fn write_http_response(socket: &mut TcpStream, response: MockHttpResponse) {
        let body = response.body.as_bytes();
        let raw_response = format!(
            "HTTP/1.1 {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response.status_line,
            response.content_type,
            body.len(),
            response.body
        );

        socket
            .write_all(raw_response.as_bytes())
            .await
            .expect("mock server should write response");
    }

    fn test_config() -> AgentConfig {
        AgentConfig {
            model: ModelId::new("anthropic", "claude-sonnet"),
            max_tokens: None,
            temperature: None,
            system_prompt: None,
            max_iterations: 10,
        }
    }

    fn assistant_tool_call(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: serde_json::json!({ "path": "README.md" }).to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn user_message(content: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[tokio::test]
    async fn chat_sends_messages_request_and_parses_mock_response() {
        let response_body = r#"{
            "id": "msg_mock",
            "content": [{
                "type": "text",
                "text": "hello from anthropic mock"
            }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 4
            }
        }"#;
        let (base_url, request_handle) =
            spawn_mock_anthropic_server(MockHttpResponse::json("200 OK", response_body)).await;
        let provider = AnthropicProvider::with_base_url(&base_url, "test-key");

        let response = provider
            .chat(
                "claude-sonnet",
                &[user_message("Say hello")],
                &[],
                &test_config(),
            )
            .await
            .expect("mock chat response should parse");

        let request = request_handle
            .await
            .expect("mock server task should complete");
        assert!(request.starts_with("POST /v1/messages HTTP/1.1"));
        assert!(request.to_ascii_lowercase().contains("x-api-key: test-key"));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("anthropic-version: 2023-06-01")
        );
        assert!(request.contains("\"model\":\"claude-sonnet\""));
        assert!(matches!(response.finish_reason, FinishReason::Stop));
        assert_eq!(response.usage.total_tokens, 9);
        let Some(MessageContent::Text(text)) = response.message.content else {
            panic!("expected assistant text content");
        };
        assert_eq!(text, "hello from anthropic mock");
    }

    #[tokio::test]
    async fn chat_maps_rate_limit_status_to_structured_provider_error() {
        let response_body = r#"{"type":"error","error":{"message":"slow down"}}"#;
        let (base_url, request_handle) = spawn_mock_anthropic_server(MockHttpResponse::json(
            "429 Too Many Requests",
            response_body,
        ))
        .await;
        let provider = AnthropicProvider::with_base_url(&base_url, "test-key");

        let error = provider
            .chat(
                "claude-sonnet",
                &[user_message("Say hello")],
                &[],
                &test_config(),
            )
            .await
            .expect_err("rate-limited response should fail");

        request_handle
            .await
            .expect("mock server task should complete");
        let PiError::Provider(ProviderError::RateLimited { status, body }) = error else {
            panic!("expected structured rate-limit provider error, got {error:?}");
        };
        assert_eq!(status, 429);
        assert!(body.contains("slow down"));
    }

    #[tokio::test]
    async fn chat_maps_malformed_json_to_structured_provider_error() {
        let response_body = r#"{"id":"msg_missing_usage","content":[]}"#;
        let (base_url, request_handle) =
            spawn_mock_anthropic_server(MockHttpResponse::json("200 OK", response_body)).await;
        let provider = AnthropicProvider::with_base_url(&base_url, "test-key");

        let error = provider
            .chat(
                "claude-sonnet",
                &[user_message("Say hello")],
                &[],
                &test_config(),
            )
            .await
            .expect_err("malformed response should fail");

        request_handle
            .await
            .expect("mock server task should complete");
        let PiError::Provider(ProviderError::MalformedResponse { message }) = error
        else {
            panic!("expected structured malformed provider error, got {error:?}");
        };
        assert!(message.contains("failed to parse response"));
    }

    #[test]
    fn build_request_normalizes_tool_call_ids_before_synthesizing_results() {
        let messages = vec![assistant_tool_call("call_123|fc.456/789?", "read_file")];

        let request = build_request("claude-sonnet", &messages, &[], &test_config(), false);

        let AnthropicContentValue::ToolUse { id, .. } = &request.messages[0].content[0] else {
            panic!("expected assistant tool_use block");
        };
        let AnthropicContentValue::ToolResult { tool_use_id, .. } = &request.messages[1].content[0]
        else {
            panic!("expected synthetic tool_result block");
        };

        assert_eq!(id, "call_123_fc_456_789_");
        assert_eq!(tool_use_id, "call_123_fc_456_789_");
    }

    #[test]
    fn metadata_describes_anthropic_features_and_compat() {
        let provider = AnthropicProvider::new("test-key");

        let metadata = provider.metadata();

        assert_eq!(metadata.id, "anthropic");
        assert_eq!(metadata.name, "Anthropic");
        assert_eq!(metadata.supported_apis, vec!["anthropic-messages"]);
        assert!(metadata.capabilities.vision);
        assert!(metadata.capabilities.function_calling);
        assert!(!metadata.capabilities.json_mode);
        assert!(metadata.capabilities.streaming);
        assert!(metadata.capabilities.thinking);

        let compat = metadata
            .compat
            .anthropic
            .expect("anthropic compat should be set");
        assert_eq!(compat.supports_eager_tool_input_streaming, Some(true));
        assert_eq!(compat.supports_long_cache_retention, Some(true));
        assert_eq!(compat.send_session_affinity_headers, Some(false));
        assert_eq!(compat.supports_cache_control_on_tools, Some(true));
    }

    #[test]
    fn metadata_detects_fireworks_anthropic_compat() {
        let provider =
            AnthropicProvider::with_base_url("https://api.fireworks.ai/inference/v1", "test-key");

        let metadata = provider.metadata();

        let compat = metadata
            .compat
            .anthropic
            .expect("anthropic compat should be set");
        assert_eq!(compat.supports_eager_tool_input_streaming, Some(false));
        assert_eq!(compat.supports_long_cache_retention, Some(false));
        assert_eq!(compat.send_session_affinity_headers, Some(true));
        assert_eq!(compat.supports_cache_control_on_tools, Some(false));
    }

    #[tokio::test]
    async fn anthropic_sse_to_chunks_errors_when_stream_ends_before_message_stop() {
        let events = futures::stream::iter(vec![Ok(SseEvent {
            event: Some("message_start".to_string()),
            data: "{}".to_string(),
            id: None,
        })]);

        let chunks = anthropic_sse_to_chunks(events);
        futures::pin_mut!(chunks);

        let error = chunks
            .next()
            .await
            .expect("stream should emit an error")
            .expect_err("missing message_stop should be an error");
        assert!(error.to_string().contains("ended before message_stop"));
    }

    #[test]
    fn extract_text_content_excludes_thinking_blocks() {
        let blocks = vec![
            ContentBlock::Thinking {
                thinking: "let me think...".to_string(),
                thinking_signature: None,
                redacted: false,
            },
            ContentBlock::Text {
                text: "Hello!".to_string(),
            },
        ];
        let content = Some(MessageContent::Blocks(blocks));

        let result = extract_text_content(&content);

        let text = result.expect("should extract text");
        assert_eq!(text, "Hello!");
        assert!(!text.contains("let me think"));
    }

    #[test]
    fn content_to_anthropic_blocks_serializes_thinking_as_native_block() {
        let content = Some(MessageContent::Blocks(vec![ContentBlock::Thinking {
            thinking: "I should check the weather first.".to_string(),
            thinking_signature: Some("sig123".to_string()),
            redacted: false,
        }]));

        let result = content_to_anthropic_blocks(&content);

        assert_eq!(result.len(), 1);
        match &result[0] {
            AnthropicContentValue::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "I should check the weather first.");
                assert_eq!(signature.as_deref(), Some("sig123"));
            }
            other => panic!("Expected Thinking variant, got: {:?}", other),
        }
    }

    #[test]
    fn content_to_anthropic_blocks_thinking_without_signature() {
        let content = Some(MessageContent::Blocks(vec![ContentBlock::Thinking {
            thinking: "Reasoning without signature.".to_string(),
            thinking_signature: None,
            redacted: false,
        }]));

        let result = content_to_anthropic_blocks(&content);

        assert_eq!(result.len(), 1);
        match &result[0] {
            AnthropicContentValue::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Reasoning without signature.");
                assert!(signature.is_none());
            }
            other => panic!("Expected Thinking variant, got: {:?}", other),
        }
    }

    #[test]
    fn parse_full_response_handles_thinking_content() {
        let response = AnthropicResponse {
            id: "msg_test".to_string(),
            content: vec![
                AnthropicResponseContent::Thinking {
                    thinking: "Let me reason about this.".to_string(),
                    signature: Some("sig456".to_string()),
                },
                AnthropicResponseContent::Text {
                    text: "Here is my answer.".to_string(),
                },
            ],
            stop_reason: Some("end_turn".to_string()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 20,
            },
        };

        let result = parse_full_response(response).unwrap();

        // Thinking blocks should NOT appear in user-visible message content
        match &result.message.content {
            Some(MessageContent::Text(text)) => assert_eq!(text, "Here is my answer."),
            other => panic!("Expected Text content, got: {:?}", other),
        }
        assert_eq!(result.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn build_request_round_trips_thinking_signature() {
        let messages = vec![
            Message {
                role: Role::User,
                content: Some(MessageContent::Text("What should I do?".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: Some(MessageContent::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "The user wants advice.".to_string(),
                        thinking_signature: Some("roundtrip_sig".to_string()),
                        redacted: false,
                    },
                    ContentBlock::Text {
                        text: "You should try X.".to_string(),
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let tools = vec![];
        let config = test_config();

        let request = build_request("claude-sonnet-4", &messages, &tools, &config, false);

        let anthropic_messages = request.messages;
        assert_eq!(anthropic_messages.len(), 2);
        let assistant_msg = &anthropic_messages[1];
        assert_eq!(assistant_msg.role, "assistant");
        // Verify thinking block has signature
        let has_thinking = assistant_msg.content.iter().any(|block| {
            matches!(block, AnthropicContentValue::Thinking { signature, .. }
                if signature.as_deref() == Some("roundtrip_sig"))
        });
        assert!(has_thinking, "Assistant message should contain thinking block with signature");

        // Verify text block also present
        let has_text = assistant_msg.content.iter().any(|block| {
            matches!(block, AnthropicContentValue::Text { text } if text == "You should try X.")
        });
        assert!(has_text, "Assistant message should contain text block");
    }
}
