//! Core async traits for the pi agent ecosystem.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{AgentConfig, ChatResponse, Message, ToolDefinition, Usage};

/// A chunk streamed from a provider.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Text delta for this chunk.
    pub delta: Option<String>,
    /// Tool call deltas (partial tool calls being streamed).
    pub tool_calls: Vec<ToolCallDelta>,
    /// Finish reason — only set in the final chunk.
    pub finish_reason: Option<crate::types::FinishReason>,
    /// Usage stats for this chunk (typically only in the final chunk).
    pub usage: Option<Usage>,
}

/// A partial tool call being streamed.
#[derive(Debug, Clone)]
pub struct ToolCallDelta {
    /// Index of the tool call in the array.
    pub index: u32,
    /// Partial ID (only sent once).
    pub id: Option<String>,
    /// Partial function name (only sent once).
    pub name: Option<String>,
    /// Partial argument fragment.
    pub arguments_delta: Option<String>,
}

/// Stream of chat response chunks.
pub type ChatStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>,
>;

/// An LLM provider (OpenAI, Anthropic, Ollama, etc.).
///
/// Providers handle API communication, request formatting, and response parsing.
#[async_trait]
pub trait Provider: Send + Sync {
    /// The provider's identifier (e.g., "openai", "anthropic").
    fn id(&self) -> &str;

    /// Send a chat completion request and get a full response.
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatResponse>;

    /// Send a chat completion request and receive a stream of chunks.
    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatStream>;

    /// Count tokens for a set of messages (provider-specific).
    ///
    /// Default implementation returns `None` (unknown).
    fn count_tokens(&self, _messages: &[Message]) -> Option<u32> {
        None
    }
}

/// A tool that the agent can invoke during execution.
///
/// Tools provide capabilities like file I/O, shell commands, web access, etc.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's name (must match the name in `definition()`).
    fn name(&self) -> &str;

    /// Return the tool's schema definition for the provider.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given JSON arguments.
    ///
    /// Returns the tool result as a string, or an error.
    async fn execute(&self, arguments: &serde_json::Value) -> Result<String>;
}

/// The agent loop — orchestrates provider + tools to fulfill a request.
///
/// An `Agent` manages conversation state, dispatches tool calls, and handles
/// iteration limits and error recovery.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run the agent loop on the given messages.
    ///
    /// Returns the final assistant message after all tool calls are resolved,
    /// or an error if the loop fails.
    async fn run(&self, messages: &mut Vec<Message>) -> Result<ChatResponse>;

    /// Run the agent loop and stream the response.
    ///
    /// Tool calls are still handled internally; this streams the text output.
    async fn run_stream(
        &self,
        messages: &mut Vec<Message>,
    ) -> Result<ChatStream>;
}

// Re-export futures so downstream crates don't need the dep.
pub use futures;
