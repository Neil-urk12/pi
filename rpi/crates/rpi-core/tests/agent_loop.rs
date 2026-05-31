//! Integration tests for `run_agent_loop`.
//!
//! Covers: single-turn completion, tool invocation, multi-tool calls,
//! max iteration enforcement, and provider error propagation.

#![cfg(feature = "test-utils")]

use rpi_core::test_utils::{assistant_text_response, MockProvider, MockTool};
use rpi_core::{
    AgentConfig, AgentEvent, AgentLoopConfig, ChatResponse, FinishReason, FunctionCall, Message,
    MessageContent, ModelId, Role, ToolCall, ToolDefinition, Usage,
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

fn user_msg(text: &str) -> Message {
    Message {
        role: Role::User,
        content: Some(MessageContent::Text(text.to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn chat_response_with_tool_calls(calls: Vec<ToolCall>) -> ChatResponse {
    ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
            name: None,
        },
        finish_reason: FinishReason::ToolCalls,
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
    }
}

fn simple_tool_def(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: format!("A tool called {name}"),
        parameters: json!({"type": "object", "properties": {}}),
    }
}

/// Single-turn conversation with no tools: provider returns text, loop completes.
#[tokio::test]
async fn single_turn_no_tools_completes() {
    let provider = MockProvider::new("mock").with_chat_response(assistant_text_response("Hello!"));
    let mut messages = vec![user_msg("Hi")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

    // Should emit TurnStart, TurnEnd, and Done.
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::TurnStart { turn: 0 })),
        "expected TurnStart(0)"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::TurnEnd { turn: 0, .. })),
        "expected TurnEnd(0)"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Done { turns: 1, .. })),
        "expected Done with turns=1"
    );

    // Messages: original user + assistant reply.
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[1].role, Role::Assistant);
}

/// Provider returns a tool call; the tool is invoked and its result fed back.
#[tokio::test]
async fn single_tool_call_invoked() {
    let tool = MockTool::new(simple_tool_def("echo"), "echoed");
    let tool_call = ToolCall {
        id: "call_1".to_string(),
        function: FunctionCall {
            name: "echo".to_string(),
            arguments: r#"{"input":"hi"}"#.to_string(),
        },
    };

    // Turn 1: tool call. Turn 2: final text (no tool calls).
    let provider = MockProvider::new("mock")
        .with_chat_response(chat_response_with_tool_calls(vec![tool_call]))
        .with_chat_response(assistant_text_response("Done"));

    let mut messages = vec![user_msg("call echo")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[&tool],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

    // Tool should have been called once.
    assert_eq!(tool.calls().len(), 1, "tool should be invoked exactly once");

    // Events should include ToolExecutionStart and ToolExecutionEnd.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionStart { name, .. } if name == "echo")),
        "expected ToolExecutionStart for echo"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { name, is_error: false, .. } if name == "echo")),
        "expected ToolExecutionEnd for echo"
    );

    // Messages: user, assistant(tool_call), tool_result, assistant(text).
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[1].role, Role::Assistant);
    assert_eq!(messages[2].role, Role::Tool);
    assert_eq!(messages[3].role, Role::Assistant);
}

/// Provider returns multiple tool calls in one turn; all are invoked.
#[tokio::test]
async fn multiple_tool_calls_all_invoked() {
    let tool_a = MockTool::new(simple_tool_def("alpha"), "alpha_result");
    let tool_b = MockTool::new(simple_tool_def("beta"), "beta_result");

    let calls = vec![
        ToolCall {
            id: "call_a".to_string(),
            function: FunctionCall {
                name: "alpha".to_string(),
                arguments: "{}".to_string(),
            },
        },
        ToolCall {
            id: "call_b".to_string(),
            function: FunctionCall {
                name: "beta".to_string(),
                arguments: "{}".to_string(),
            },
        },
    ];

    let provider = MockProvider::new("mock")
        .with_chat_response(chat_response_with_tool_calls(calls))
        .with_chat_response(assistant_text_response("Both done"));

    let mut messages = vec![user_msg("run both")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[&tool_a, &tool_b],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

    assert_eq!(tool_a.calls().len(), 1, "alpha should be invoked once");
    assert_eq!(tool_b.calls().len(), 1, "beta should be invoked once");

    // Both tool results should appear in messages.
    let tool_msgs: Vec<_> = messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 2, "expected 2 tool result messages");
}

/// When tool rounds exceed max_tool_rounds, loop stops with an error.
#[tokio::test]
async fn max_iterations_enforced() {
    // Provider always returns a tool call — forces iteration until limit.
    let infinite_provider = {
        // Build a provider that always returns a tool call.
        // We need to queue enough responses: max_tool_rounds turns of tool calls,
        // then the error fires on turn > max_tool_rounds.
        let mut p = MockProvider::new("mock");
        for _ in 0..5 {
            let tc = ToolCall {
                id: "call_loop".to_string(),
                function: FunctionCall {
                    name: "noop".to_string(),
                    arguments: "{}".to_string(),
                },
            };
            p = p.with_chat_response(chat_response_with_tool_calls(vec![tc]));
        }
        p
    };

    let tool = MockTool::new(simple_tool_def("noop"), "ok");
    let mut messages = vec![user_msg("loop forever")];
    let config = AgentLoopConfig {
        max_tool_rounds: 3,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &infinite_provider,
        "test-model",
        &mut messages,
        &[&tool],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_err(), "expected error from max iterations");

    let err = result.unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("Exceeded maximum tool call rounds"),
        "expected max rounds error, got: {err_msg}"
    );

    // Should have emitted an Error event.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error { error } if error.contains("Exceeded maximum"))),
        "expected Error event about max rounds"
    );
}

/// Provider error propagates through the loop.
#[tokio::test]
async fn provider_error_propagates() {
    let provider = MockProvider::new("mock"); // no queued responses -> error
    let mut messages = vec![user_msg("fail")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_err(), "expected provider error to propagate");
}

/// Unknown tool name results in an error tool result, not a loop crash.
#[tokio::test]
async fn unknown_tool_name_handled() {
    let tc = ToolCall {
        id: "call_ghost".to_string(),
        function: FunctionCall {
            name: "nonexistent".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let provider = MockProvider::new("mock")
        .with_chat_response(chat_response_with_tool_calls(vec![tc]))
        .with_chat_response(assistant_text_response("Recovered"));

    let mut messages = vec![user_msg("use ghost tool")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[], // no tools registered
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok(), "expected Ok, got {:?}", result.err());

    // The tool result message should indicate "Unknown tool".
    let tool_msgs: Vec<_> = messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 1, "expected 1 tool result message");
    let content = match tool_msgs[0].content.as_ref().unwrap() {
        MessageContent::Text(s) => s.clone(),
        _ => panic!("expected text content"),
    };
    assert!(
        content.contains("Unknown tool"),
        "expected 'Unknown tool' error, got: {content}"
    );
}

/// Done event reports correct total_usage from provider responses.
#[tokio::test]
async fn done_event_reports_usage() {
    let response = ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text("Reply".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        finish_reason: FinishReason::Stop,
        usage: Usage {
            prompt_tokens: 42,
            completion_tokens: 13,
            total_tokens: 55,
        },
    };

    let provider = MockProvider::new("mock").with_chat_response(response);
    let mut messages = vec![user_msg("count tokens")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok());

    let done = events.iter().find_map(|e| match e {
        AgentEvent::Done { total_usage, .. } => Some(total_usage.clone()),
        _ => None,
    });
    let usage = done.expect("expected Done event");
    assert_eq!(usage.prompt_tokens, 42);
    assert_eq!(usage.completion_tokens, 13);
    assert_eq!(usage.total_tokens, 55);
}

/// Tool execution error is recorded as is_error=true in the result.
#[tokio::test]
async fn tool_execution_error_recorded() {
    // Create a tool that returns an error.
    struct FailTool;

    #[async_trait::async_trait]
    impl rpi_core::Tool for FailTool {
        fn name(&self) -> &str {
            "fail_tool"
        }
        fn definition(&self) -> ToolDefinition {
            simple_tool_def("fail_tool")
        }
        async fn execute(&self, _args: &serde_json::Value) -> rpi_core::Result<String> {
            Err(rpi_core::PiError::Tool { tool: "fail_tool".to_string(), message: "intentional failure".to_string() })
        }
    }

    let tc = ToolCall {
        id: "call_fail".to_string(),
        function: FunctionCall {
            name: "fail_tool".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let provider = MockProvider::new("mock")
        .with_chat_response(chat_response_with_tool_calls(vec![tc]))
        .with_chat_response(assistant_text_response("Saw error"));

    let tool = FailTool;
    let mut messages = vec![user_msg("trigger error")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[&tool],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok());

    // Should have a ToolExecutionEnd with is_error=true.
    let error_end = events.iter().find(|e| {
        matches!(
            e,
            AgentEvent::ToolExecutionEnd {
                name,
                is_error: true,
                ..
            } if name == "fail_tool"
        )
    });
    assert!(error_end.is_some(), "expected ToolExecutionEnd with is_error=true");

    // Tool result message should contain "Error:".
    let tool_msgs: Vec<_> = messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 1);
    let content = match tool_msgs[0].content.as_ref().unwrap() {
        MessageContent::Text(s) => s.clone(),
        _ => panic!("expected text content"),
    };
    assert!(content.contains("Error:"), "expected error prefix, got: {content}");
}

/// Malformed tool arguments produce an error tool result, not a panic.
#[tokio::test]
async fn malformed_tool_arguments_handled() {
    let tc = ToolCall {
        id: "call_bad".to_string(),
        function: FunctionCall {
            name: "echo".to_string(),
            arguments: "not-json".to_string(),
        },
    };

    let provider = MockProvider::new("mock")
        .with_chat_response(chat_response_with_tool_calls(vec![tc]))
        .with_chat_response(assistant_text_response("Recovered"));

    let tool = MockTool::new(simple_tool_def("echo"), "ok");
    let mut messages = vec![user_msg("bad args")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[&tool],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;

    assert!(result.is_ok());

    // Tool should NOT have been called (args couldn't be parsed).
    assert_eq!(tool.calls().len(), 0, "tool should not be called with bad args");

    // Error tool result message should mention "malformed".
    let tool_msgs: Vec<_> = messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 1);
    let content = match tool_msgs[0].content.as_ref().unwrap() {
        MessageContent::Text(s) => s.clone(),
        _ => panic!("expected text content"),
    };
    assert!(
        content.contains("malformed"),
        "expected 'malformed' error, got: {content}"
    );
}

/// Multi-turn conversation: two sequential user messages produce two assistant
/// replies, accumulating into a single messages vec with correct roles.
#[tokio::test]
async fn multi_turn_conversation_accumulates_messages() {
    let provider = MockProvider::new("mock")
        .with_chat_response(assistant_text_response("Hello!"))
        .with_chat_response(assistant_text_response("Why did the chicken cross the road?"));

    let mut messages = vec![user_msg("Hi")];
    let config = AgentLoopConfig {
        max_tool_rounds: 10,
        stream: false,
        compaction: None,
    };
    let mut events = Vec::new();

    // Turn 1: user says "Hi", assistant replies "Hello!"
    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;
    assert!(result.is_ok(), "turn 1 failed: {:?}", result.err());
    assert_eq!(messages.len(), 2, "after turn 1: expected 2 messages");
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[1].role, Role::Assistant);

    // Turn 2: user says "Tell me a joke", assistant replies with a joke.
    messages.push(user_msg("Tell me a joke"));

    let result = rpi_core::run_agent_loop(
        &provider,
        "test-model",
        &mut messages,
        &[],
        &config,
        &agent_config(),
        |e| events.push(e),
    )
    .await;
    assert!(result.is_ok(), "turn 2 failed: {:?}", result.err());

    // Messages: user -> assistant -> user -> assistant
    assert_eq!(messages.len(), 4, "after turn 2: expected 4 messages");
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[1].role, Role::Assistant);
    assert_eq!(messages[2].role, Role::User);
    assert_eq!(messages[3].role, Role::Assistant);

    // Verify message content.
    let content = |idx: usize| match messages[idx].content.as_ref().unwrap() {
        MessageContent::Text(s) => s.as_str(),
        _ => panic!("expected text content at index {idx}"),
    };
    assert_eq!(content(0), "Hi");
    assert_eq!(content(1), "Hello!");
    assert_eq!(content(2), "Tell me a joke");
    assert_eq!(content(3), "Why did the chicken cross the road?");

    // Two Done events, each reporting turns=1 (each call is one turn).
    let done_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::Done { turns: 1, .. }))
        .count();
    assert_eq!(done_count, 2, "expected 2 Done events with turns=1");

    // Two TurnStart/TurnEnd pairs.
    let turn_starts: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::TurnStart { turn } => Some(*turn),
            _ => None,
        })
        .collect();
    assert_eq!(turn_starts, vec![0, 0], "each call emits TurnStart(0)");
}
