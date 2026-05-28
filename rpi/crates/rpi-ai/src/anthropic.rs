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
    AgentConfig, ChatResponse, ChatStream, ContentBlock, FinishReason, FunctionCall, Message,
    MessageContent, PiError, Result, Role, StreamChunk, ToolCall, ToolCallDelta, ToolDefinition,
    Usage,
};

use crate::streaming::{sse_stream, SseEvent};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The Anthropic Claude chat provider.
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new provider targeting the official Anthropic API.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a provider with a custom base URL.
    pub fn with_base_url(base_url: &str, api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
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
        headers.insert(
            "x-api-key",
            self.api_key.parse().expect("valid header value"),
        );
        headers.insert(
            "anthropic-version",
            API_VERSION.parse().expect("valid header value"),
        );
        headers
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
                "Anthropic API returned {status}: {text}"
            )));
        }

        let resp: AnthropicResponse = response
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
            .post(format!("{}/v1/messages", self.base_url))
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
                "Anthropic API returned {status}: {text}"
            )));
        }

        Ok(Box::pin(anthropic_sse_to_chunks(sse_stream(response))))
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
                    yield Err(PiError::Provider(format!("SSE error: {e}")));
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
        "message_start" | "message_stop" | "ping" => {
            // Control events — no chunk to emit.
            Ok(None)
        }
        "content_block_start" => {
            let block_start: ContentBlockStartEvent = event.json().map_err(|e| {
                PiError::Provider(format!("Failed to parse content_block_start: {e}"))
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
            }
            Ok(None)
        }
        "content_block_delta" => {
            let delta: ContentBlockDeltaEvent = event.json().map_err(|e| {
                PiError::Provider(format!("Failed to parse content_block_delta: {e}"))
            })?;

            match &delta.delta {
                AnthropicDelta::TextDelta { text } => Ok(Some(StreamChunk {
                    delta: Some(text.clone()),
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
            let msg_delta: MessageDeltaEvent = event.json().map_err(|e| {
                PiError::Provider(format!("Failed to parse message_delta: {e}"))
            })?;
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
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
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
    let (system, anthropic_msgs) = to_anthropic_messages(messages);

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

                // Tool calls → tool_use blocks.
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| {
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
                                    content,
                                    is_error,
                                    ..
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

