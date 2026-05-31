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
    AgentConfig, ChatResponse, ChatStream, CompatFlags, ContentBlock, FinishReason, FunctionCall,
    Message, MessageContent, ModelInputKind, OpenAiCompat, PiError, ProviderCapabilities,
    ProviderError, ProviderMetadata, Result, Role, StreamChunk, ToolCall, ToolCallDelta,
    ToolDefinition, Usage, normalize_messages_with_input,
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
    compat: OpenAiCompat,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiProvider")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAiProvider {
    /// Create a new provider for the official OpenAI API.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com".to_string(),
            client: crate::default_http_client(),
            compat: detect_openai_compat("https://api.openai.com"),
        }
    }

    /// Create a provider pointing at a custom base URL (Azure, Ollama, etc.).
    ///
    /// `api_key` may be `None` for services that don't require authentication.
    pub fn with_base_url(base_url: &str, api_key: Option<&str>) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            api_key: api_key.unwrap_or("").to_string(),
            compat: detect_openai_compat(&base_url),
            base_url,
            client: crate::default_http_client(),
        }
    }

    fn build_headers(&self) -> Result<reqwest::header::HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("valid header value"),
        );
        if !self.api_key.is_empty() {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                    .map_err(|_| ProviderError::Other {
                        message: "invalid characters in API key".to_string(),
                    })?,
            );
        }
        Ok(headers)
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

    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            supported_apis: vec!["openai-completions".to_string()],
            capabilities: ProviderCapabilities {
                vision: true,
                function_calling: true,
                json_mode: true,
                streaming: true,
                thinking: true,
            },
            compat: CompatFlags {
                openai: Some(detect_openai_compat(&self.base_url)),
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
        let body = build_request(model, messages, tools, config, false, &self.compat);

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
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

        let resp: OpenAiResponse =
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
        let body = build_request(model, messages, tools, config, true, &self.compat);

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
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

        Ok(Box::pin(openai_sse_to_chunks(sse_stream(response))))
    }
}

fn detect_openai_compat(base_url: &str) -> OpenAiCompat {
    let base_url = base_url.to_ascii_lowercase();
    let is_zai = base_url.contains("api.z.ai");
    let is_together = base_url.contains("api.together.ai") || base_url.contains("api.together.xyz");
    let is_moonshot = base_url.contains("api.moonshot.");
    let is_cloudflare_workers_ai = base_url.contains("api.cloudflare.com");
    let is_cloudflare_ai_gateway = base_url.contains("gateway.ai.cloudflare.com");
    let is_grok = base_url.contains("api.x.ai");
    let is_deepseek = base_url.contains("deepseek.com");

    let is_non_standard = base_url.contains("cerebras.ai")
        || is_grok
        || is_together
        || base_url.contains("chutes.ai")
        || is_deepseek
        || is_zai
        || is_moonshot
        || base_url.contains("opencode.ai")
        || is_cloudflare_workers_ai
        || is_cloudflare_ai_gateway;

    let use_max_tokens =
        base_url.contains("chutes.ai") || is_moonshot || is_cloudflare_ai_gateway || is_together;

    OpenAiCompat {
        supports_store: Some(!is_non_standard),
        supports_developer_role: Some(!is_non_standard),
        supports_reasoning_effort: Some(
            !is_grok && !is_zai && !is_moonshot && !is_together && !is_cloudflare_ai_gateway,
        ),
        supports_usage_in_streaming: Some(base_url.contains("api.openai.com") || base_url.contains("openai.azure.com")),
        max_tokens_field: Some(
            if use_max_tokens {
                "max_tokens"
            } else {
                "max_completion_tokens"
            }
            .to_string(),
        ),
        requires_tool_result_name: Some(false),
        requires_assistant_after_tool_result: Some(false),
        requires_thinking_as_text: Some(false),
        supports_strict_mode: Some(!is_moonshot && !is_together && !is_cloudflare_ai_gateway),
        supports_long_cache_retention: Some(
            !(is_together || is_cloudflare_workers_ai || is_cloudflare_ai_gateway),
        ),
    }
}

// ---------------------------------------------------------------------------
// SSE → StreamChunk conversion
// ---------------------------------------------------------------------------

fn openai_sse_to_chunks(
    stream: impl Stream<Item = anyhow::Result<SseEvent>> + Send + 'static,
) -> impl Stream<Item = Result<StreamChunk>> + Send {
    stream.map(|event_result| match event_result {
        Err(e) => Err(PiError::provider(format!("SSE error: {e}"))),
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
                .map_err(|e| PiError::provider(format!("Failed to parse stream chunk: {e}")))?;

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
    compat: &OpenAiCompat,
) -> serde_json::Value {
    let normalized_messages = normalize_messages_with_input(
        messages,
        &[ModelInputKind::Text, ModelInputKind::Image],
        None,
        false,
    );

    let request = OpenAiRequest {
        model: model.to_string(),
        messages: to_openai_messages(&normalized_messages),
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
    };

    // Safe: OpenAiRequest contains only standard serde types (String, Vec, Option, u32).
    let mut value = serde_json::to_value(&request)
        .expect("OpenAiRequest should serialize");

    if let Some(field_name) = compat.max_tokens_field.as_deref() {
        if field_name != "max_tokens" {
            if let Some(obj) = value.as_object_mut() {
                if let Some(val) = obj.remove("max_tokens") {
                    obj.insert(field_name.to_string(), val);
                }
            }
        }
    }

    value
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
                    ContentBlock::Thinking { thinking, .. } => Some(serde_json::json!({
                        "type": "text",
                        "text": format!("<thinking>{thinking}</thinking>")
                    })),
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
    let choice =
        resp.choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::MalformedResponse {
                message: "OpenAI response contained no choices".to_string(),
            })?;

    let msg_resp = choice
        .message
        .ok_or_else(|| ProviderError::MalformedResponse {
            message: "OpenAI choice had no message".to_string(),
        })?;

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

    async fn spawn_mock_openai_server(
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
            model: ModelId::new("openai", "gpt-4o"),
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
    async fn chat_sends_completion_request_and_parses_mock_response() {
        let response_body = r#"{
            "id": "chatcmpl_mock",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello from mock"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "total_tokens": 7
            }
        }"#;
        let (base_url, request_handle) =
            spawn_mock_openai_server(MockHttpResponse::json("200 OK", response_body)).await;
        let provider = OpenAiProvider::with_base_url(&base_url, Some("test-key"));

        let response = provider
            .chat("gpt-4o", &[user_message("Say hello")], &[], &test_config())
            .await
            .expect("mock chat response should parse");

        let request = request_handle
            .await
            .expect("mock server task should complete");
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        assert!(request.contains("\"model\":\"gpt-4o\""));
        assert!(matches!(response.finish_reason, FinishReason::Stop));
        assert_eq!(response.usage.total_tokens, 7);
        let Some(MessageContent::Text(text)) = response.message.content else {
            panic!("expected assistant text content");
        };
        assert_eq!(text, "hello from mock");
    }

    #[tokio::test]
    async fn chat_maps_rate_limit_status_to_structured_provider_error() {
        let response_body = r#"{"error":{"message":"slow down"}}"#;
        let (base_url, request_handle) = spawn_mock_openai_server(MockHttpResponse::json(
            "429 Too Many Requests",
            response_body,
        ))
        .await;
        let provider = OpenAiProvider::with_base_url(&base_url, Some("test-key"));

        let error = provider
            .chat("gpt-4o", &[user_message("Say hello")], &[], &test_config())
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
    async fn chat_maps_missing_choices_to_structured_malformed_response_error() {
        let response_body = r#"{"id":"chatcmpl_empty","choices":[]}"#;
        let (base_url, request_handle) =
            spawn_mock_openai_server(MockHttpResponse::json("200 OK", response_body)).await;
        let provider = OpenAiProvider::with_base_url(&base_url, Some("test-key"));

        let error = provider
            .chat("gpt-4o", &[user_message("Say hello")], &[], &test_config())
            .await
            .expect_err("malformed response should fail");

        request_handle
            .await
            .expect("mock server task should complete");
        let PiError::Provider(ProviderError::MalformedResponse { message }) = error
        else {
            panic!("expected structured malformed provider error, got {error:?}");
        };
        assert!(message.contains("no choices"));
    }

    #[test]
    fn build_request_synthesizes_missing_tool_results() {
        let messages = vec![assistant_tool_call("call_123", "read_file")];

        let request = build_request("gpt-4o", &messages, &[], &test_config(), false, &OpenAiCompat::default());

        let msgs = request["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"].as_str().unwrap(), "tool");
        assert_eq!(msgs[1]["tool_call_id"].as_str().unwrap(), "call_123");
        assert_eq!(
            msgs[1]["content"],
            serde_json::Value::String("No result provided".to_string())
        );
    }

    #[test]
    fn metadata_describes_openai_features_and_compat() {
        let provider = OpenAiProvider::new("test-key");

        let metadata = provider.metadata();

        assert_eq!(metadata.id, "openai");
        assert_eq!(metadata.name, "OpenAI");
        assert_eq!(metadata.supported_apis, vec!["openai-completions"]);
        assert!(metadata.capabilities.vision);
        assert!(metadata.capabilities.function_calling);
        assert!(metadata.capabilities.json_mode);
        assert!(metadata.capabilities.streaming);
        assert!(metadata.capabilities.thinking);

        let compat = metadata.compat.openai.expect("openai compat should be set");
        assert_eq!(compat.supports_store, Some(true));
        assert_eq!(compat.supports_developer_role, Some(true));
        assert_eq!(compat.supports_reasoning_effort, Some(true));
        assert_eq!(compat.supports_usage_in_streaming, Some(true));
        assert_eq!(
            compat.max_tokens_field.as_deref(),
            Some("max_completion_tokens")
        );
        assert_eq!(compat.supports_strict_mode, Some(true));
    }

    #[test]
    fn metadata_detects_nonstandard_openai_compatible_urls() {
        let provider =
            OpenAiProvider::with_base_url("https://api.together.ai/v1", Some("test-key"));

        let metadata = provider.metadata();

        let compat = metadata.compat.openai.expect("openai compat should be set");
        assert_eq!(compat.supports_store, Some(false));
        assert_eq!(compat.supports_developer_role, Some(false));
        assert_eq!(compat.supports_reasoning_effort, Some(false));
        assert_eq!(compat.max_tokens_field.as_deref(), Some("max_tokens"));
        assert_eq!(compat.supports_strict_mode, Some(false));
        assert_eq!(compat.supports_long_cache_retention, Some(false));
    }

    #[test]
    fn unknown_provider_has_no_usage_in_streaming() {
        let provider =
            OpenAiProvider::with_base_url("https://unknown-llm.example.com/v1", Some("key"));
        let metadata = provider.metadata();
        let compat = metadata.compat.openai.expect("openai compat should be set");
        assert_ne!(
            compat.supports_usage_in_streaming,
            Some(true),
            "unknown providers should not have supports_usage_in_streaming = true"
        );
    }

    #[test]
    fn openai_native_has_usage_in_streaming() {
        let provider = OpenAiProvider::new("test-key");
        let metadata = provider.metadata();
        let compat = metadata.compat.openai.expect("openai compat should be set");
        assert_eq!(compat.supports_usage_in_streaming, Some(true));
    }

    #[test]
    fn build_request_uses_max_tokens_field_from_compat() {
        let config = AgentConfig {
            max_tokens: Some(100),
            ..test_config()
        };

        // Together uses max_tokens
        let compat_together = OpenAiCompat {
            max_tokens_field: Some("max_tokens".to_string()),
            ..OpenAiCompat::default()
        };
        let body = build_request("gpt-4o", &[], &[], &config, false, &compat_together);
        assert!(body.get("max_tokens").is_some(), "together compat should use max_tokens");
        assert!(body.get("max_completion_tokens").is_none(), "together compat should not use max_completion_tokens");

        // OpenAI native uses max_completion_tokens
        let compat_openai = OpenAiCompat {
            max_tokens_field: Some("max_completion_tokens".to_string()),
            ..OpenAiCompat::default()
        };
        let body = build_request("gpt-4o", &[], &[], &config, false, &compat_openai);
        assert!(body.get("max_completion_tokens").is_some(), "openai compat should use max_completion_tokens");
        assert!(body.get("max_tokens").is_none(), "openai compat should not use max_tokens");

        // No compat info falls back to max_tokens
        let compat_none = OpenAiCompat::default();
        let body = build_request("gpt-4o", &[], &[], &config, false, &compat_none);
        assert!(body.get("max_tokens").is_some(), "no compat should fall back to max_tokens");
    }
}
