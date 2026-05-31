//! Shared test utilities for Rust port parity tests.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{PiError, Result};
use crate::traits::{ChatStream, Provider, StreamChunk, Tool};
use crate::types::{
    AgentConfig, ChatResponse, Message, MessageContent, Role, ToolDefinition, Usage,
};

/// Captured provider request data.
#[derive(Debug, Clone, PartialEq)]
pub struct MockProviderRequest {
    /// Requested model id.
    pub model: String,
    /// Messages passed to the provider.
    pub messages: Vec<Message>,
    /// Tool definitions passed to the provider.
    pub tools: Vec<ToolDefinition>,
    /// Agent config passed to the provider.
    pub config: AgentConfig,
}

/// Builder-style mock provider for tests.
#[derive(Debug, Clone)]
pub struct MockProvider {
    id: String,
    chat_responses: Arc<Mutex<VecDeque<ChatResponse>>>,
    stream_responses: Arc<Mutex<VecDeque<Vec<StreamChunk>>>>,
    requests: Arc<Mutex<Vec<MockProviderRequest>>>,
    token_count: Option<u32>,
}

impl MockProvider {
    /// Create a mock provider with the given provider id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            chat_responses: Arc::new(Mutex::new(VecDeque::new())),
            stream_responses: Arc::new(Mutex::new(VecDeque::new())),
            requests: Arc::new(Mutex::new(Vec::new())),
            token_count: None,
        }
    }

    /// Queue a non-streaming chat response.
    pub fn with_chat_response(self, response: ChatResponse) -> Self {
        self.chat_responses
            .lock()
            .expect("mock provider response queue poisoned")
            .push_back(response);
        self
    }

    /// Queue a streaming response represented by chunks.
    pub fn with_stream_chunks(self, chunks: Vec<StreamChunk>) -> Self {
        self.stream_responses
            .lock()
            .expect("mock provider stream queue poisoned")
            .push_back(chunks);
        self
    }

    /// Set the provider token count result.
    pub fn with_token_count(mut self, token_count: u32) -> Self {
        self.token_count = Some(token_count);
        self
    }

    /// Return captured provider requests.
    pub fn requests(&self) -> Vec<MockProviderRequest> {
        self.requests
            .lock()
            .expect("mock provider request log poisoned")
            .clone()
    }

    fn record_request(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) {
        self.requests
            .lock()
            .expect("mock provider request log poisoned")
            .push(MockProviderRequest {
                model: model.to_string(),
                messages: messages.to_vec(),
                tools: tools.to_vec(),
                config: config.clone(),
            });
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatResponse> {
        self.record_request(model, messages, tools, config);
        self.chat_responses
            .lock()
            .expect("mock provider response queue poisoned")
            .pop_front()
            .ok_or_else(|| {
                PiError::provider("mock provider has no queued chat response")
            })
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatStream> {
        self.record_request(model, messages, tools, config);
        let chunks = self
            .stream_responses
            .lock()
            .expect("mock provider stream queue poisoned")
            .pop_front()
            .ok_or_else(|| {
                PiError::provider("mock provider has no queued stream response")
            })?;
        Ok(Box::pin(crate::traits::futures::stream::iter(
            chunks.into_iter().map(Ok),
        )))
    }

    fn count_tokens(&self, _messages: &[Message]) -> Option<u32> {
        self.token_count
    }
}

/// Mock tool for tests that need deterministic tool execution.
#[derive(Debug, Clone)]
pub struct MockTool {
    definition: ToolDefinition,
    result: String,
    calls: Arc<Mutex<Vec<Value>>>,
}

impl MockTool {
    /// Create a mock tool with a definition and fixed result.
    pub fn new(definition: ToolDefinition, result: impl Into<String>) -> Self {
        Self {
            definition,
            result: result.into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return captured tool call arguments.
    pub fn calls(&self) -> Vec<Value> {
        self.calls
            .lock()
            .expect("mock tool call log poisoned")
            .clone()
    }
}

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(&self, arguments: &Value) -> Result<String> {
        self.calls
            .lock()
            .expect("mock tool call log poisoned")
            .push(arguments.clone());
        Ok(self.result.clone())
    }
}

/// Create a plain assistant response with text content and zero usage.
pub fn assistant_text_response(text: impl Into<String>) -> ChatResponse {
    ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        finish_reason: crate::types::FinishReason::Stop,
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}
