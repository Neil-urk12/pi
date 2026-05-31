//! Message normalization parity tests.

use rpi_core::{
    ContentBlock, FunctionCall, Message, MessageContent, Model, ModelCost, ModelId, ModelInputKind,
    Role, ThinkingLevel, ToolCall, normalize_messages,
};
use serde_json::json;
use std::collections::BTreeMap;

fn model_with_input(input: Vec<ModelInputKind>) -> Model {
    Model {
        id: ModelId::new("anthropic", "claude-sonnet"),
        name: "Claude Sonnet".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        reasoning: true,
        thinking_level_map: BTreeMap::<ThinkingLevel, Option<String>>::new(),
        input,
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 128_000,
        max_tokens: 16_000,
        headers: None,
        compat: None,
    }
}

fn user_blocks(blocks: Vec<ContentBlock>) -> Message {
    Message {
        role: Role::User,
        content: Some(MessageContent::Blocks(blocks)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
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
                arguments: json!({ "path": "README.md" }).to_string(),
            },
        }]),
        tool_call_id: None,
        name: None,
    }
}

fn assistant_blocks(blocks: Vec<ContentBlock>) -> Message {
    Message {
        role: Role::Assistant,
        content: Some(MessageContent::Blocks(blocks)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn tool_result(id: &str, name: &str) -> Message {
    Message {
        role: Role::Tool,
        content: Some(MessageContent::Text("done".to_string())),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        name: Some(name.to_string()),
    }
}

#[test]
fn non_empty_thinking_blocks_become_plain_text_when_source_model_is_unknown() {
    let model = model_with_input(vec![ModelInputKind::Text]);
    let messages = vec![assistant_blocks(vec![ContentBlock::Thinking {
        thinking: "reasoning trace".to_string(),
        thinking_signature: None,
        redacted: false,
    }])];

    let normalized = normalize_messages(&messages, &model, None);

    assert_eq!(
        normalized[0].content,
        Some(MessageContent::Blocks(vec![ContentBlock::Text {
            text: "reasoning trace".to_string()
        }]))
    );
}

#[test]
fn redacted_and_empty_thinking_blocks_are_removed_when_source_model_is_unknown() {
    let model = model_with_input(vec![ModelInputKind::Text]);
    let messages = vec![assistant_blocks(vec![
        ContentBlock::Text {
            text: "before".to_string(),
        },
        ContentBlock::Thinking {
            thinking: "[Reasoning redacted]".to_string(),
            thinking_signature: Some("opaque".to_string()),
            redacted: true,
        },
        ContentBlock::Thinking {
            thinking: "   ".to_string(),
            thinking_signature: None,
            redacted: false,
        },
        ContentBlock::Text {
            text: "after".to_string(),
        },
    ])];

    let normalized = normalize_messages(&messages, &model, None);

    assert_eq!(
        normalized[0].content,
        Some(MessageContent::Blocks(vec![
            ContentBlock::Text {
                text: "before".to_string(),
            },
            ContentBlock::Text {
                text: "after".to_string(),
            },
        ]))
    );
}

#[test]
fn non_vision_models_replace_adjacent_user_images_with_one_placeholder() {
    let model = model_with_input(vec![ModelInputKind::Text]);
    let messages = vec![user_blocks(vec![
        ContentBlock::Text {
            text: "before".to_string(),
        },
        ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: "first".to_string(),
        },
        ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: "second".to_string(),
        },
        ContentBlock::Text {
            text: "after".to_string(),
        },
    ])];

    let normalized = normalize_messages(&messages, &model, None);

    assert_eq!(
        normalized[0].content,
        Some(MessageContent::Blocks(vec![
            ContentBlock::Text {
                text: "before".to_string(),
            },
            ContentBlock::Text {
                text: "(image omitted: model does not support images)".to_string(),
            },
            ContentBlock::Text {
                text: "after".to_string(),
            },
        ]))
    );
}

#[test]
fn normalizes_tool_call_ids_and_matching_tool_results() {
    let model = model_with_input(vec![ModelInputKind::Text]);
    let messages = vec![
        assistant_tool_call("call_123|fc_123", "read"),
        tool_result("call_123|fc_123", "read"),
    ];

    let normalized = normalize_messages(
        &messages,
        &model,
        Some(&|id, _message| id.replace('|', "_")),
    );

    assert_eq!(
        normalized[0].tool_calls.as_ref().unwrap()[0].id,
        "call_123_fc_123"
    );
    assert_eq!(
        normalized[1].tool_call_id.as_deref(),
        Some("call_123_fc_123")
    );
}

#[test]
fn adds_synthetic_results_for_trailing_orphaned_tool_calls() {
    let model = model_with_input(vec![ModelInputKind::Text]);
    let messages = vec![assistant_tool_call("call_123|fc_123", "read")];

    let normalized = normalize_messages(
        &messages,
        &model,
        Some(&|id, _message| id.replace('|', "_")),
    );

    assert_eq!(normalized.len(), 2);
    assert_eq!(normalized[1].role, Role::Tool);
    assert_eq!(
        normalized[1].tool_call_id.as_deref(),
        Some("call_123_fc_123")
    );
    assert_eq!(normalized[1].name.as_deref(), Some("read"));
    assert_eq!(
        normalized[1].content,
        Some(MessageContent::Text("No result provided".to_string()))
    );
}
