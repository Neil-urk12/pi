//! Shared test utility contract tests.

#![cfg(feature = "test-utils")]

use futures::StreamExt;
use rpi_core::test_utils::{MockProvider, MockTool};
use rpi_core::{
    AgentConfig, FinishReason, FunctionCall, Message, MessageContent, ModelId, Provider, Role,
    StreamChunk, Tool, ToolCall, ToolDefinition, Usage,
};
use serde_json::json;

fn agent_config() -> AgentConfig {
    AgentConfig {
        model: ModelId::new("mock", "test-model"),
        max_tokens: Some(1024),
        temperature: None,
        system_prompt: None,
        max_iterations: 10,
    }
}

#[tokio::test]
async fn mock_provider_returns_queued_chat_response_and_records_request() {
    let response = rpi_core::ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text("done".to_string())),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                function: FunctionCall {
                    name: "read".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        },
        finish_reason: FinishReason::ToolCalls,
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 7,
        },
    };
    let provider = MockProvider::new("mock").with_chat_response(response);
    let messages = vec![Message {
        role: Role::User,
        content: Some(MessageContent::Text("hello".to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];

    let result = provider
        .chat("test-model", &messages, &[], &agent_config())
        .await
        .unwrap();

    assert_eq!(result.finish_reason, FinishReason::ToolCalls);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn mock_provider_streams_queued_chunks() {
    let provider = MockProvider::new("mock").with_stream_chunks(vec![StreamChunk {
        delta: Some("hello".to_string()),
        tool_calls: Vec::new(),
        finish_reason: Some(FinishReason::Stop),
        usage: Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        }),
    }]);

    let chunks: Vec<_> = provider
        .chat_stream("test-model", &[], &[], &agent_config())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();

    assert_eq!(chunks[0].delta.as_deref(), Some("hello"));
}

#[tokio::test]
async fn mock_tool_records_arguments_and_returns_configured_result() {
    let tool = MockTool::new(
        ToolDefinition {
            name: "echo".to_string(),
            description: "Echo input".to_string(),
            parameters: json!({"type": "object"}),
        },
        "ok",
    );

    let result = tool.execute(&json!({"message": "hello"})).await.unwrap();

    assert_eq!(result, "ok");
    assert_eq!(tool.calls(), vec![json!({"message": "hello"})]);
}
