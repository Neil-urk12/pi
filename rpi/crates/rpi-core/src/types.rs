//! Core types for the pi agent.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::str::FromStr;

/// A message role in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt messages.
    System,
    /// User input messages.
    User,
    /// Assistant (model) response messages.
    Assistant,
    /// Tool result messages.
    Tool,
}

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content (base64-encoded).
    Image {
        /// Media type (e.g., "image/png").
        media_type: String,
        /// Base64-encoded image data.
        data: String,
    },
    /// Provider reasoning/thinking content.
    Thinking {
        /// Thinking text when the provider exposes it.
        thinking: String,
        /// Provider-specific opaque signature for multi-turn continuity.
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        /// Whether the reasoning text was redacted by the provider.
        #[serde(default)]
        redacted: bool,
    },
    /// A tool use request from the model.
    ToolUse {
        /// Unique tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments as JSON.
        input: Value,
    },
    /// A tool execution result.
    ToolResult {
        /// The tool call ID this result corresponds to.
        tool_use_id: String,
        /// The result content.
        content: String,
        /// Whether the tool execution failed.
        #[serde(default)]
        is_error: bool,
    },
}

/// A tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// Details of a function call within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Function arguments as a JSON string.
    pub arguments: String,
}

/// A chat message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// The role of the message author.
    pub role: Role,
    /// Message content — can be a string or a list of content blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID — required for tool result messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Name of the tool or participant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content — either a simple string or structured content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks (text, images, tool use/result).
    Blocks(Vec<ContentBlock>),
}

/// A tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: Value,
}

/// Identifies a specific model on a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId {
    /// The provider identifier (e.g., "openai", "anthropic").
    pub provider: String,
    /// The model identifier (e.g., "gpt-4o", "claude-3.5-sonnet").
    pub model: String,
}

impl ModelId {
    /// Create a new `ModelId` from provider and model strings.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Parse a `provider/model` string into a `ModelId`.
    pub fn parse(s: &str) -> Option<Self> {
        let (provider, model) = s.split_once('/')?;
        if provider.is_empty() || model.is_empty() {
            return None;
        }
        Some(Self::new(provider, model))
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

impl FromStr for ModelId {
    type Err = &'static str;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::parse(s).ok_or("invalid model id format: expected 'provider/model'")
    }
}

/// Reasoning effort level requested for models that support thinking controls.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// Disable thinking when the provider supports an explicit off value.
    Off,
    /// Minimal reasoning effort.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    #[serde(rename = "xhigh")]
    XHigh,
}

/// Preferred transport for providers that expose multiple streaming transports.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    /// Server-sent events.
    #[serde(rename = "sse")]
    Sse,
    /// WebSocket transport.
    #[serde(rename = "websocket")]
    WebSocket,
    /// WebSocket transport with provider-side caching.
    #[serde(rename = "websocket-cached")]
    WebSocketCached,
    /// Provider chooses the best transport.
    Auto,
}

/// Prompt cache retention preference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    /// Do not request prompt caching.
    None,
    /// Short-lived prompt cache.
    Short,
    /// Long-lived prompt cache.
    Long,
}

/// Supported model input modality.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ModelInputKind {
    /// Text input.
    Text,
    /// Image input.
    Image,
}

/// Token cost metadata in dollars per million tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCost {
    /// Input token cost.
    pub input: f64,
    /// Output token cost.
    pub output: f64,
    /// Prompt-cache read token cost.
    pub cache_read: f64,
    /// Prompt-cache write token cost.
    pub cache_write: f64,
}

/// Compatibility flags for OpenAI-compatible completions APIs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiCompat {
    /// Whether the provider supports the `store` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    /// Whether the provider supports the `developer` role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Whether the provider supports reasoning effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    /// Whether streaming can include usage accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    /// Field name used for max-token limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<String>,
    /// Whether tool result messages require the tool name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    /// Whether a tool result must be followed by an assistant message before user input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    /// Whether thinking blocks must be sent as text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    /// Whether strict tool schema mode is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Whether long prompt-cache retention is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
}

/// Compatibility flags for Anthropic Messages-compatible APIs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicCompat {
    /// Whether eager tool input streaming is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// Whether long cache retention is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    /// Whether session-affinity headers should be sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Whether cache-control markers are supported on tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    /// Whether adaptive thinking format should be forced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
}

/// Compatibility flags for Google-compatible APIs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleCompat {
    /// Whether thought signatures are supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_thought_signatures: Option<bool>,
    /// Whether system instructions are supported separately from chat contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_system_instruction: Option<bool>,
}

/// Provider compatibility overrides grouped by API family.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompatFlags {
    /// OpenAI-compatible completions flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAiCompat>,
    /// Anthropic Messages-compatible flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic: Option<AnthropicCompat>,
    /// Google-compatible flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google: Option<GoogleCompat>,
}

/// Provider feature capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderCapabilities {
    /// Whether the provider supports image inputs.
    pub vision: bool,
    /// Whether the provider supports function/tool calling.
    pub function_calling: bool,
    /// Whether the provider supports JSON response mode.
    pub json_mode: bool,
    /// Whether the provider supports streaming responses.
    pub streaming: bool,
    /// Whether the provider supports thinking/reasoning controls.
    pub thinking: bool,
}

/// Provider metadata used by registries and pre-flight checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderMetadata {
    /// Stable provider identifier.
    pub id: String,
    /// Human-readable provider name.
    pub name: String,
    /// Supported API implementation identifiers.
    #[serde(default)]
    pub supported_apis: Vec<String>,
    /// Supported feature set.
    pub capabilities: ProviderCapabilities,
    /// Compatibility flags for provider-specific wire behavior.
    pub compat: CompatFlags,
}

impl ProviderMetadata {
    /// Create default metadata for a provider id.
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            supported_apis: Vec::new(),
            capabilities: ProviderCapabilities::default(),
            compat: CompatFlags::default(),
        }
    }
}

/// Provider stream request options.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamOptions {
    /// Sampling temperature override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Max output tokens override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// API key override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Preferred transport.
    pub transport: Transport,
    /// Prompt-cache retention preference.
    pub cache_retention: CacheRetention,
}

impl std::fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamOptions")
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("transport", &self.transport)
            .field("cache_retention", &self.cache_retention)
            .finish()
    }
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            api_key: None,
            transport: Transport::Auto,
            cache_retention: CacheRetention::None,
        }
    }
}

/// Provider request configuration separated from agent-loop behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProviderRequestConfig {
    /// Provider-qualified model id, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
    /// Optional context window used for pre-flight validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Stream options used by provider implementations.
    pub stream: StreamOptions,
}

/// Metadata for a model known to the Rust port.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    /// Provider-qualified model identifier.
    pub id: ModelId,
    /// Human-readable model name.
    pub name: String,
    /// API implementation identifier.
    pub api: String,
    /// Provider identifier.
    pub provider: String,
    /// Base URL used for the provider API.
    pub base_url: String,
    /// Whether the model supports reasoning controls.
    pub reasoning: bool,
    /// Optional map from pi thinking levels to provider-specific values.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub thinking_level_map: BTreeMap<ThinkingLevel, Option<String>>,
    /// Supported input modalities.
    pub input: Vec<ModelInputKind>,
    /// Token cost metadata.
    pub cost: ModelCost,
    /// Context window in tokens.
    pub context_window: u32,
    /// Maximum output tokens.
    pub max_tokens: u32,
    /// Optional headers to apply to requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Optional provider compatibility flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<CompatFlags>,
}

/// Configuration for an agent instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentConfig {
    /// Model to use.
    pub model: ModelId,
    /// Maximum tokens for the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// System prompt override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Maximum number of tool-call iterations before stopping.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
}

fn default_max_iterations() -> u32 {
    10
}

/// Provider-specific configuration.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum ProviderConfig {
    /// OpenAI-compatible API configuration.
    OpenAi {
        /// API key.
        api_key: String,
        /// Base URL override (defaults to api.openai.com).
        #[serde(skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
    },
    /// Anthropic API configuration.
    Anthropic {
        /// API key.
        api_key: String,
        /// Base URL override.
        #[serde(skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
    },
    /// Ollama local model configuration.
    Ollama {
        /// Base URL for Ollama API.
        #[serde(default = "default_ollama_url")]
        base_url: String,
    },
    /// Gitlawb OpenAI-compatible gateway configuration.
    Gitlawb {
        /// API key.
        api_key: String,
        /// Base URL override.
        #[serde(skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
    },
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi { api_key: _, base_url } => f
                .debug_struct("OpenAi")
                .field("api_key", &"<redacted>")
                .field("base_url", base_url)
                .finish(),
            Self::Anthropic { api_key: _, base_url } => f
                .debug_struct("Anthropic")
                .field("api_key", &"<redacted>")
                .field("base_url", base_url)
                .finish(),
            Self::Ollama { base_url } => f
                .debug_struct("Ollama")
                .field("base_url", base_url)
                .finish(),
            Self::Gitlawb { api_key: _, base_url } => f
                .debug_struct("Gitlawb")
                .field("api_key", &"<redacted>")
                .field("base_url", base_url)
                .finish(),
        }
    }
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop or stop sequence hit.
    Stop,
    /// Hit the token limit.
    Length,
    /// Model wants to call tools.
    ToolCalls,
    /// Content was filtered.
    ContentFilter,
}

/// Terminal reason for the assistant-message event protocol.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    /// Natural stop or stop sequence hit.
    Stop,
    /// Hit the token limit.
    Length,
    /// Model wants to call tools.
    ToolUse,
    /// Provider or runtime error.
    Error,
    /// Request was aborted.
    Aborted,
}

impl From<FinishReason> for StopReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::Stop => Self::Stop,
            FinishReason::Length => Self::Length,
            FinishReason::ToolCalls => Self::ToolUse,
            FinishReason::ContentFilter => Self::Error,
        }
    }
}

/// Token usage statistics for a request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
}

/// A complete chat response from a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    /// The response message.
    pub message: Message,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
    /// Token usage for this request.
    pub usage: Usage,
}

/// Assistant message used by the provider event protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssistantMessage {
    /// Assistant output content.
    pub content: Vec<ContentBlock>,
    /// API implementation identifier.
    pub api: String,
    /// Provider identifier.
    pub provider: String,
    /// Requested model identifier.
    pub model: String,
    /// Concrete response model when the provider returns one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-specific response identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Token usage for this message.
    pub usage: Usage,
    /// Terminal stop reason.
    pub stop_reason: StopReason,
    /// Error text for error and aborted stop reasons.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Event protocol for assistant message streams.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    /// Stream started with an initial partial message.
    Start {
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Text block started.
    TextStart {
        /// Content block index.
        content_index: usize,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Text block delta.
    TextDelta {
        /// Content block index.
        content_index: usize,
        /// Delta text.
        delta: String,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Text block completed.
    TextEnd {
        /// Content block index.
        content_index: usize,
        /// Final text content.
        content: String,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Thinking block started.
    ThinkingStart {
        /// Content block index.
        content_index: usize,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Thinking block delta.
    ThinkingDelta {
        /// Content block index.
        content_index: usize,
        /// Delta text.
        delta: String,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Thinking block completed.
    ThinkingEnd {
        /// Content block index.
        content_index: usize,
        /// Final thinking content.
        content: String,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Tool call block started.
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        /// Content block index.
        content_index: usize,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Tool call argument delta.
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        /// Content block index.
        content_index: usize,
        /// Delta JSON fragment.
        delta: String,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Tool call block completed.
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        /// Content block index.
        content_index: usize,
        /// Completed tool call.
        tool_call: ToolCall,
        /// Current partial message.
        partial: AssistantMessage,
    },
    /// Successful terminal event.
    Done {
        /// Successful terminal reason.
        reason: StopReason,
        /// Final assistant message.
        message: AssistantMessage,
    },
    /// Error terminal event.
    Error {
        /// Error terminal reason.
        reason: StopReason,
        /// Final error message.
        error: AssistantMessage,
    },
}

/// Escape XML tags in text to prevent injection into summary containers.
pub fn escape_xml_tags(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_xml_tags() {
        assert_eq!(escape_xml_tags("hello"), "hello");
        assert_eq!(escape_xml_tags("<test>"), "&lt;test&gt;");
        assert_eq!(escape_xml_tags("a & b"), "a &amp; b");
        assert_eq!(escape_xml_tags("<a>&</a>"), "&lt;a&gt;&amp;&lt;/a&gt;");
        assert_eq!(escape_xml_tags(""), "");
    }
}
