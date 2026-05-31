//! Provider-facing message normalization helpers.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::{ContentBlock, Message, MessageContent, Model, ModelInputKind, Role, ToolCall};
use std::borrow::Cow;

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";
const SYNTHETIC_TOOL_RESULT_TEXT: &str = "No result provided";

/// Callback used to normalize provider-specific tool call identifiers.
pub type ToolCallIdNormalizer = dyn Fn(&str, &Message) -> String + Send + Sync;

/// Normalize messages before provider-specific wire-format conversion.
///
/// This handles provider-neutral compatibility work: replacing unsupported
/// image blocks, normalizing tool call ids, updating matching tool results,
/// and synthesizing missing tool results for orphaned assistant tool calls.
pub fn normalize_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<&ToolCallIdNormalizer>,
) -> Vec<Message> {
    normalize_messages_with_input(messages, &model.input, normalize_tool_call_id, false)
}

/// Normalize messages using explicit model input capabilities.
///
/// Provider adapters can use this when they know the provider's input support
/// but do not have a full model catalog entry available at request-build time.
pub fn normalize_messages_with_input(
    messages: &[Message],
    input: &[ModelInputKind],
    normalize_tool_call_id: Option<&ToolCallIdNormalizer>,
    preserve_thinking: bool,
) -> Vec<Message> {
    let image_aware_messages = downgrade_unsupported_images(messages, input);
    let mut tool_call_id_map = BTreeMap::new();
    let transformed = image_aware_messages
        .iter()
        .map(|message| {
            let message = if preserve_thinking { message.clone() } else { normalize_message_thinking_blocks(message) };
            transform_message_tool_ids(&message, normalize_tool_call_id, &mut tool_call_id_map)
        })
        .collect::<Vec<_>>();

    synthesize_missing_tool_results(&transformed)
}

fn downgrade_unsupported_images<'a>(messages: &'a [Message], input: &[ModelInputKind]) -> Cow<'a, [Message]> {
    if input.contains(&ModelInputKind::Image) {
        return Cow::Borrowed(messages);
    }

    if !messages.iter().any(|m| {
        matches!(&m.content, Some(MessageContent::Blocks(blocks)) if blocks.iter().any(|b| matches!(b, ContentBlock::Image { .. })))
    }) {
        return Cow::Borrowed(messages);
    }

    Cow::Owned(
        messages
            .iter()
            .map(|message| {
                let placeholder = match message.role {
                    Role::User => Some(NON_VISION_USER_IMAGE_PLACEHOLDER),
                    Role::Tool => Some(NON_VISION_TOOL_IMAGE_PLACEHOLDER),
                    _ => None,
                };
                let Some(placeholder) = placeholder else {
                    return message.clone();
                };
                let Some(MessageContent::Blocks(blocks)) = &message.content else {
                    return message.clone();
                };

                let mut normalized = message.clone();
                normalized.content = Some(MessageContent::Blocks(replace_images_with_placeholder(
                    blocks,
                    placeholder,
                )));
                normalized
            })
            .collect(),
    )
}

fn replace_images_with_placeholder(
    blocks: &[ContentBlock],
    placeholder: &str,
) -> Vec<ContentBlock> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;

    for block in blocks {
        if matches!(block, ContentBlock::Image { .. }) {
            if !previous_was_placeholder {
                result.push(ContentBlock::Text {
                    text: placeholder.to_string(),
                });
            }
            previous_was_placeholder = true;
            continue;
        }

        result.push(block.clone());
        previous_was_placeholder =
            matches!(block, ContentBlock::Text { text } if text == placeholder);
    }

    result
}

fn normalize_message_thinking_blocks(message: &Message) -> Message {
    if message.role != Role::Assistant {
        return message.clone();
    }

    let Some(MessageContent::Blocks(blocks)) = &message.content else {
        return message.clone();
    };

    if !blocks.iter().any(|b| matches!(b, ContentBlock::Thinking { .. })) {
        return message.clone();
    }

    let blocks = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking {
                thinking, redacted, ..
            } if *redacted || thinking.trim().is_empty() => None,
            ContentBlock::Thinking { thinking, .. } => Some(ContentBlock::Text {
                text: thinking.clone(),
            }),
            _ => Some(block.clone()),
        })
        .collect();

    let mut normalized = message.clone();
    normalized.content = Some(MessageContent::Blocks(blocks));
    normalized
}

fn transform_message_tool_ids(
    message: &Message,
    normalize_tool_call_id: Option<&ToolCallIdNormalizer>,
    tool_call_id_map: &mut BTreeMap<String, String>,
) -> Message {
    let Some(normalize_tool_call_id) = normalize_tool_call_id else {
        return message.clone();
    };

    match message.role {
        Role::Assistant => {

            let mut normalized = message.clone();
            normalized.tool_calls = message.tool_calls.as_ref().map(|tool_calls| {
                tool_calls
                    .iter()
                    .map(|tool_call| {
                        normalize_tool_call(
                            tool_call,
                            message,
                            normalize_tool_call_id,
                            tool_call_id_map,
                        )
                    })
                    .collect()
            });
            normalized
        }
        Role::Tool => {
            let Some(tool_call_id) = &message.tool_call_id else {
                return message.clone();
            };
            let Some(normalized_id) = tool_call_id_map.get(tool_call_id) else {
                return message.clone();
            };
            let mut normalized = message.clone();
            normalized.tool_call_id = Some(normalized_id.clone());
            normalized
        }
        _ => message.clone(),
    }
}

fn normalize_tool_call(
    tool_call: &ToolCall,
    source_message: &Message,
    normalize_tool_call_id: &ToolCallIdNormalizer,
    tool_call_id_map: &mut BTreeMap<String, String>,
) -> ToolCall {
    let normalized_id = normalize_tool_call_id(&tool_call.id, source_message);
    if normalized_id == tool_call.id {
        return tool_call.clone();
    }

    tool_call_id_map.insert(tool_call.id.clone(), normalized_id.clone());
    let mut normalized_tool_call = tool_call.clone();
    normalized_tool_call.id = normalized_id;
    normalized_tool_call
}

fn synthesize_missing_tool_results(messages: &[Message]) -> Vec<Message> {
    let mut result = Vec::new();
    let mut pending_tool_calls = Vec::<ToolCall>::new();
    let mut existing_tool_result_ids = BTreeSet::<String>::new();

    for message in messages {
        match message.role {
            Role::Assistant => {
                insert_synthetic_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                );
                if let Some(tool_calls) = &message.tool_calls {
                    pending_tool_calls = tool_calls.clone();
                    existing_tool_result_ids.clear();
                }
                result.push(message.clone());
            }
            Role::Tool => {
                if let Some(tool_call_id) = &message.tool_call_id {
                    existing_tool_result_ids.insert(tool_call_id.clone());
                }
                result.push(message.clone());
            }
            Role::User => {
                insert_synthetic_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                );
                result.push(message.clone());
            }
            Role::System => result.push(message.clone()),
        }
    }

    insert_synthetic_tool_results(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_result_ids,
    );

    result
}

fn insert_synthetic_tool_results(
    result: &mut Vec<Message>,
    pending_tool_calls: &mut Vec<ToolCall>,
    existing_tool_result_ids: &mut BTreeSet<String>,
) {
    for tool_call in pending_tool_calls.drain(..) {
        if existing_tool_result_ids.contains(&tool_call.id) {
            continue;
        }

        result.push(Message {
            role: Role::Tool,
            content: Some(MessageContent::Text(SYNTHETIC_TOOL_RESULT_TEXT.to_string())),
            tool_calls: None,
            tool_call_id: Some(tool_call.id),
            name: Some(tool_call.function.name),
        });
    }
    existing_tool_result_ids.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        FunctionCall, Message, MessageContent, Model, ModelCost, ModelId, ModelInputKind, Role,
        ToolCall,
    };
    use serde_json::json;
    use std::collections::BTreeMap;

    fn text_model() -> Model {
        Model {
            id: ModelId::new("test", "text-only"),
            name: "Text Only".to_string(),
            api: "test-api".to_string(),
            provider: "test".to_string(),
            base_url: "https://api.test.com".to_string(),
            reasoning: false,
            thinking_level_map: BTreeMap::new(),
            input: vec![ModelInputKind::Text],
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

    fn vision_model() -> Model {
        let mut m = text_model();
        m.input = vec![ModelInputKind::Text, ModelInputKind::Image];
        m
    }

    fn user_text(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
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

    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
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

    fn assistant_tool_call(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: json!({ "path": "file.txt" }).to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_result(id: &str, name: &str) -> Message {
        Message {
            role: Role::Tool,
            content: Some(MessageContent::Text("result".to_string())),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            name: Some(name.to_string()),
        }
    }

    // --- Thinking block removal ---

    #[test]
    fn thinking_block_with_content_becomes_text() {
        let messages = vec![assistant_blocks(vec![ContentBlock::Thinking {
            thinking: "step by step".to_string(),
            thinking_signature: None,
            redacted: false,
        }])];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].content,
            Some(MessageContent::Blocks(vec![ContentBlock::Text {
                text: "step by step".to_string(),
            }]))
        );
    }

    #[test]
    fn redacted_thinking_block_is_removed() {
        let messages = vec![assistant_blocks(vec![
            ContentBlock::Text { text: "before".to_string() },
            ContentBlock::Thinking {
                thinking: "[redacted]".to_string(),
                thinking_signature: Some("opaque".to_string()),
                redacted: true,
            },
            ContentBlock::Text { text: "after".to_string() },
        ])];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(
            result[0].content,
            Some(MessageContent::Blocks(vec![
                ContentBlock::Text { text: "before".to_string() },
                ContentBlock::Text { text: "after".to_string() },
            ]))
        );
    }

    #[test]
    fn empty_thinking_block_is_removed() {
        let messages = vec![assistant_blocks(vec![ContentBlock::Thinking {
            thinking: "   ".to_string(),
            thinking_signature: None,
            redacted: false,
        }])];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(
            result[0].content,
            Some(MessageContent::Blocks(vec![]))
        );
    }

    #[test]
    fn thinking_blocks_only_stripped_from_assistant_role() {
        // User messages with thinking blocks should pass through unchanged
        let thinking = ContentBlock::Thinking {
            thinking: "user thought".to_string(),
            thinking_signature: None,
            redacted: false,
        };
        let messages = vec![user_blocks(vec![thinking.clone()])];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result[0].content, Some(MessageContent::Blocks(vec![thinking])));
    }

    // --- Tool call ID sanitization ---

    #[test]
    fn tool_call_id_sanitized_and_propagated_to_result() {
        let messages = vec![
            assistant_tool_call("call|pipe", "read"),
            tool_result("call|pipe", "read"),
        ];

        let result = normalize_messages(
            &messages,
            &text_model(),
            Some(&|id, _| id.replace('|', "_")),
        );

        assert_eq!(result[0].tool_calls.as_ref().unwrap()[0].id, "call_pipe");
        assert_eq!(result[1].tool_call_id.as_deref(), Some("call_pipe"));
    }

    #[test]
    fn no_normalizer_leaves_ids_unchanged() {
        let messages = vec![
            assistant_tool_call("call|123", "read"),
            tool_result("call|123", "read"),
        ];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result[0].tool_calls.as_ref().unwrap()[0].id, "call|123");
        assert_eq!(result[1].tool_call_id.as_deref(), Some("call|123"));
    }

    #[test]
    fn normalizer_returning_same_id_is_noop() {
        let messages = vec![assistant_tool_call("unchanged", "read")];

        let result = normalize_messages(
            &messages,
            &text_model(),
            Some(&|id, _| id.to_string()),
        );

        assert_eq!(result[0].tool_calls.as_ref().unwrap()[0].id, "unchanged");
    }

    // --- Orphaned tool call synthesis ---

    #[test]
    fn trailing_orphaned_tool_call_gets_synthetic_result() {
        let messages = vec![assistant_tool_call("call_a", "read")];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result.len(), 2);
        assert_eq!(result[1].role, Role::Tool);
        assert_eq!(result[1].tool_call_id.as_deref(), Some("call_a"));
        assert_eq!(result[1].name.as_deref(), Some("read"));
        assert_eq!(
            result[1].content,
            Some(MessageContent::Text(SYNTHETIC_TOOL_RESULT_TEXT.to_string()))
        );
    }

    #[test]
    fn tool_call_with_matching_result_no_synthesis() {
        let messages = vec![
            assistant_tool_call("call_a", "read"),
            tool_result("call_a", "read"),
        ];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result.len(), 2);
        assert_eq!(result[1].role, Role::Tool);
        assert_eq!(result[1].tool_call_id.as_deref(), Some("call_a"));
        // Verify it's the original tool result, not synthetic
        assert_eq!(
            result[1].content,
            Some(MessageContent::Text("result".to_string()))
        );
    }

    #[test]
    fn orphaned_tool_calls_between_user_messages_get_synthetic_results() {
        let messages = vec![
            user_text("read this file"),
            assistant_tool_call("call_x", "read"),
            user_text("now summarize"),
        ];

        let result = normalize_messages(&messages, &text_model(), None);

        // Should be: user, assistant, synthetic_tool, user
        assert_eq!(result.len(), 4);
        assert_eq!(result[2].role, Role::Tool);
        assert_eq!(result[2].tool_call_id.as_deref(), Some("call_x"));
        assert_eq!(result[2].name.as_deref(), Some("read"));
    }

    #[test]
    fn multiple_orphaned_tool_calls_all_get_synthetic_results() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![
                ToolCall {
                    id: "a".to_string(),
                    function: FunctionCall {
                        name: "read".to_string(),
                        arguments: "{}".to_string(),
                    },
                },
                ToolCall {
                    id: "b".to_string(),
                    function: FunctionCall {
                        name: "write".to_string(),
                        arguments: "{}".to_string(),
                    },
                },
            ]),
            tool_call_id: None,
            name: None,
        }];

        let result = normalize_messages(&messages, &text_model(), None);

        // 1 assistant + 2 synthetic tool results
        assert_eq!(result.len(), 3);
        assert_eq!(result[1].tool_call_id.as_deref(), Some("a"));
        assert_eq!(result[1].name.as_deref(), Some("read"));
        assert_eq!(result[2].tool_call_id.as_deref(), Some("b"));
        assert_eq!(result[2].name.as_deref(), Some("write"));
    }

    // --- Image placeholder replacement ---

    #[test]
    fn non_vision_model_replaces_images_with_placeholder() {
        let messages = vec![user_blocks(vec![
            ContentBlock::Text { text: "describe".to_string() },
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        ])];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(
            result[0].content,
            Some(MessageContent::Blocks(vec![
                ContentBlock::Text { text: "describe".to_string() },
                ContentBlock::Text {
                    text: NON_VISION_USER_IMAGE_PLACEHOLDER.to_string(),
                },
            ]))
        );
    }

    #[test]
    fn vision_model_preserves_images() {
        let messages = vec![user_blocks(vec![
            ContentBlock::Text { text: "describe".to_string() },
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        ])];

        let result = normalize_messages(&messages, &vision_model(), None);

        assert_eq!(result[0].content, Some(MessageContent::Blocks(vec![
            ContentBlock::Text { text: "describe".to_string() },
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        ])));
    }

    #[test]
    fn adjacent_images_collapsed_to_single_placeholder() {
        let messages = vec![user_blocks(vec![
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "img1".to_string(),
            },
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "img2".to_string(),
            },
        ])];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(
            result[0].content,
            Some(MessageContent::Blocks(vec![ContentBlock::Text {
                text: NON_VISION_USER_IMAGE_PLACEHOLDER.to_string(),
            }]))
        );
    }

    #[test]
    fn assistant_role_images_not_replaced() {
        let messages = vec![assistant_blocks(vec![ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: "generated".to_string(),
        }])];

        let result = normalize_messages(&messages, &text_model(), None);

        // Assistant images pass through since the function only handles User and Tool roles
        assert_eq!(result[0].content, Some(MessageContent::Blocks(vec![
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "generated".to_string(),
            },
        ])));
    }

    #[test]
    fn tool_role_image_gets_tool_placeholder() {
        let messages = vec![Message {
            role: Role::Tool,
            content: Some(MessageContent::Blocks(vec![ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "toolimg".to_string(),
            }])),
            tool_calls: None,
            tool_call_id: Some("call1".to_string()),
            name: Some("screenshot".to_string()),
        }];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(
            result[0].content,
            Some(MessageContent::Blocks(vec![ContentBlock::Text {
                text: NON_VISION_TOOL_IMAGE_PLACEHOLDER.to_string(),
            }]))
        );
    }

    // --- Empty and trivial inputs ---

    #[test]
    fn empty_message_list_returns_empty() {
        let result = normalize_messages(&[], &text_model(), None);
        assert!(result.is_empty());
    }

    #[test]
    fn single_user_message_passes_through_unchanged() {
        let messages = vec![user_text("hello")];
        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], messages[0]);
    }

    #[test]
    fn single_system_message_passes_through_unchanged() {
        let messages = vec![Message {
            role: Role::System,
            content: Some(MessageContent::Text("system prompt".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], messages[0]);
    }

    // --- Multi-turn conversation ---

    #[test]
    fn full_multi_turn_conversation_normalizes_correctly() {
        let messages = vec![
            user_text("read file"),
            assistant_tool_call("call_1", "read"),
            tool_result("call_1", "read"),
            user_text("summarize"),
            assistant_text("here is the summary"),
        ];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0].role, Role::User);
        assert_eq!(result[1].role, Role::Assistant);
        assert_eq!(result[2].role, Role::Tool);
        assert_eq!(result[3].role, Role::User);
        assert_eq!(result[4].role, Role::Assistant);
    }

    #[test]
    fn multi_turn_with_thinking_and_orphan() {
        let messages = vec![
            user_text("think about this"),
            assistant_blocks(vec![
                ContentBlock::Thinking {
                    thinking: "hmm".to_string(),
                    thinking_signature: None,
                    redacted: false,
                },
                ContentBlock::Text {
                    text: "let me call a tool".to_string(),
                },
            ]),
            assistant_tool_call("call_1", "search"),
            // No tool result — should get synthetic one before next user msg
            user_text("what did you find?"),
        ];

        let result = normalize_messages(&messages, &text_model(), None);

        // user, assistant(thinking->text, text), assistant(tool_call), synthetic_tool, user
        assert_eq!(result.len(), 5);

        // Thinking should be converted to text
        assert_eq!(
            result[1].content,
            Some(MessageContent::Blocks(vec![
                ContentBlock::Text { text: "hmm".to_string() },
                ContentBlock::Text { text: "let me call a tool".to_string() },
            ]))
        );

        // Synthetic tool result inserted before user
        assert_eq!(result[3].role, Role::Tool);
        assert_eq!(result[3].tool_call_id.as_deref(), Some("call_1"));
    }

    // --- Idempotency ---

    #[test]
    fn normalizing_twice_is_idempotent() {
        let messages = vec![
            user_text("hello"),
            assistant_blocks(vec![ContentBlock::Thinking {
                thinking: "reasoning".to_string(),
                thinking_signature: None,
                redacted: false,
            }]),
            assistant_tool_call("call_1", "read"),
            tool_result("call_1", "read"),
        ];

        let once = normalize_messages(&messages, &text_model(), None);
        let twice = normalize_messages(&once, &text_model(), None);

        assert_eq!(once, twice);
    }

    #[test]
    fn already_normalized_messages_stay_stable() {
        // Messages with no thinking, no images, no orphaned tool calls
        let messages = vec![
            user_text("hello"),
            assistant_text("hi there"),
        ];

        let result = normalize_messages(&messages, &text_model(), None);

        assert_eq!(result, messages);
    }

    // --- normalize_messages_with_input ---

    #[test]
    fn with_input_equivalent_to_model_based() {
        let model = text_model();
        let messages = vec![user_text("hello")];

        let via_model = normalize_messages(&messages, &model, None);
        let via_input = normalize_messages_with_input(&messages, &model.input, None, false);

        assert_eq!(via_model, via_input);
    }

    #[test]
    fn with_input_vision_preserves_images() {
        let messages = vec![user_blocks(vec![ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: "img".to_string(),
        }])];

        let result = normalize_messages_with_input(
            &messages,
            &[ModelInputKind::Text, ModelInputKind::Image],
            None,
            false,
        );

        assert_eq!(result[0].content, Some(MessageContent::Blocks(vec![
            ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "img".to_string(),
            },
        ])));
    }

    #[test]
    fn normalize_messages_noop_when_model_supports_images() {
        let model = vision_model(); // supports Text + Image
        let messages = vec![
            user_text("describe this"),
            user_blocks(vec![
                ContentBlock::Text { text: "look".to_string() },
                ContentBlock::Image {
                    media_type: "image/png".to_string(),
                    data: "base64data".to_string(),
                },
            ]),
            assistant_text("done"),
        ];

        let result = normalize_messages(&messages, &model, None);

        assert_eq!(result, messages);
    }
}