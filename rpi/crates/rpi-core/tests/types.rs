//! Serialization and parity tests for core Rust port types.

use rpi_core::{
    AssistantMessage, AssistantMessageEvent, CacheRetention, CompatFlags, ContentBlock, Message,
    MessageContent, Model, ModelCost, ModelId, ModelInputKind, OpenAiCompat, Role, StopReason,
    ThinkingLevel, ToolCall, Transport,
};
use serde_json::json;

fn assistant_message() -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text {
            text: "hello".to_string(),
        }],
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o".to_string(),
        response_model: None,
        response_id: Some("resp_1".to_string()),
        usage: rpi_core::Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 1_700_000_000_000,
    }
}

#[test]
fn content_block_serializes_thinking_content() {
    let block = ContentBlock::Thinking {
        thinking: "chain".to_string(),
        thinking_signature: Some("sig_1".to_string()),
        redacted: true,
    };

    let value = serde_json::to_value(block).unwrap();

    assert_eq!(
        value,
        json!({
            "type": "thinking",
            "thinking": "chain",
            "thinking_signature": "sig_1",
            "redacted": true
        })
    );
}

#[test]
fn stop_reason_matches_provider_event_protocol_names() {
    let reason = StopReason::ToolUse;

    let value = serde_json::to_value(reason).unwrap();

    assert_eq!(value, json!("toolUse"));
}

#[test]
fn model_metadata_round_trips_with_reasoning_and_compat_flags() {
    let model = Model {
        id: ModelId::new("openai", "gpt-4o"),
        name: "GPT-4o".to_string(),
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        reasoning: true,
        thinking_level_map: [(ThinkingLevel::XHigh, Some("high".to_string()))].into(),
        input: vec![ModelInputKind::Text, ModelInputKind::Image],
        cost: ModelCost {
            input: 5.0,
            output: 15.0,
            cache_read: 1.0,
            cache_write: 2.0,
        },
        context_window: 128_000,
        max_tokens: 16_384,
        headers: None,
        compat: Some(CompatFlags {
            openai: Some(OpenAiCompat {
                supports_store: Some(false),
                supports_developer_role: Some(true),
                supports_reasoning_effort: Some(true),
                supports_usage_in_streaming: Some(true),
                max_tokens_field: Some("max_completion_tokens".to_string()),
                requires_tool_result_name: Some(false),
                requires_assistant_after_tool_result: Some(false),
                requires_thinking_as_text: Some(false),
                supports_strict_mode: Some(true),
                supports_long_cache_retention: Some(true),
            }),
            ..CompatFlags::default()
        }),
    };

    let encoded = serde_json::to_string(&model).unwrap();
    let decoded: Model = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, model);
}

#[test]
fn model_id_parse_accepts_provider_and_slash_containing_model_ids() {
    let model_id = ModelId::parse("openrouter/qwen/qwen3-coder").unwrap();

    assert_eq!(model_id.to_string(), "openrouter/qwen/qwen3-coder");
}

#[test]
fn model_id_parse_rejects_empty_provider_or_model() {
    let invalid_ids = ["", "openai", "/gpt-4o", "openai/"];

    assert!(invalid_ids.iter().all(|id| ModelId::parse(id).is_none()));
}

#[test]
fn assistant_message_event_serializes_stream_lifecycle_variants() {
    let event = AssistantMessageEvent::ToolCallEnd {
        content_index: 1,
        tool_call: ToolCall {
            id: "call_1".to_string(),
            function: rpi_core::FunctionCall {
                name: "read".to_string(),
                arguments: "{\"path\":\"README.md\"}".to_string(),
            },
        },
        partial: assistant_message(),
    };

    let value = serde_json::to_value(event).unwrap();

    assert_eq!(value["type"], json!("toolcall_end"));
}

#[test]
fn assistant_message_event_round_trips_terminal_error() {
    let event = AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: AssistantMessage {
            stop_reason: StopReason::Error,
            error_message: Some("provider failed".to_string()),
            ..assistant_message()
        },
    };

    let encoded = serde_json::to_string(&event).unwrap();
    let decoded: AssistantMessageEvent = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, event);
}

#[test]
fn transport_and_cache_retention_use_wire_names() {
    let value = json!({
        "transport": Transport::WebSocketCached,
        "cache_retention": CacheRetention::Long
    });

    assert_eq!(
        value,
        json!({
            "transport": "websocket-cached",
            "cache_retention": "long"
        })
    );
}

#[test]
fn existing_message_type_accepts_thinking_blocks() {
    let message = Message {
        role: Role::Assistant,
        content: Some(MessageContent::Blocks(vec![ContentBlock::Thinking {
            thinking: "model reasoning".to_string(),
            thinking_signature: None,
            redacted: false,
        }])),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    let value = serde_json::to_value(message).unwrap();

    assert_eq!(value["content"][0]["type"], json!("thinking"));
}

#[test]
fn model_id_from_str_parses_provider_slash_model() {
    let id: ModelId = "openai/gpt-4o".parse().unwrap();
    assert_eq!(id.provider, "openai");
    assert_eq!(id.model, "gpt-4o");
}

#[test]
fn model_id_from_str_rejects_invalid_format() {
    let result: std::result::Result<ModelId, _> = "invalid".parse();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid model id format"));
}
