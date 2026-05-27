//! Core types for the pi agent.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// Details of a function call within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Function arguments as a JSON string.
    pub arguments: String,
}

/// A chat message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks (text, images, tool use/result).
    Blocks(Vec<ContentBlock>),
}

/// A tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        Some(Self::new(provider, model))
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

/// Configuration for an agent instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Token usage statistics for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
}

/// A complete chat response from a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The response message.
    pub message: Message,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
    /// Token usage for this request.
    pub usage: Usage,
}
