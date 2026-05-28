//! Context window compaction for long conversations.
//!
//! When the conversation approaches the model's context window limit, this module
//! summarizes older messages and removes them, keeping recent messages intact.
//!
//! The compaction pipeline:
//! 1. [`should_compact`] — check if context is near the limit.
//! 2. [`find_cut_point`] — walk backwards to find a safe message boundary.
//! 3. [`prepare_compaction`] — split messages into to-summarize and to-keep.
//! 4. [`compact`] — call the provider to generate summaries.
//! 5. The caller writes a `SessionEntry::Compaction` with the result.

use crate::error::Result;
use crate::token_estimation::estimate_tokens;
use crate::traits::Provider;
use crate::types::*;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Settings for the compaction system.
#[derive(Debug, Clone)]
pub struct CompactionSettings {
    /// Token budget reserved for model response (default 16384).
    pub reserve_tokens: u32,
    /// Tokens to keep from recent messages (default 20000).
    pub keep_recent_tokens: u32,
    /// Whether compaction is enabled.
    pub enabled: bool,
    /// Override the context window size. If None, falls back to max(max_tokens, 128_000).
    pub context_window: Option<u32>,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            enabled: true,
            context_window: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Preparation / result types
// ---------------------------------------------------------------------------

/// Prepared compaction data before summarization.
#[derive(Debug)]
pub struct CompactionPreparation {
    /// Index of the first message to keep (not summarize).
    pub first_kept_message_index: usize,
    /// Messages that will be summarized.
    pub messages_to_summarize: Vec<Message>,
    /// Turn prefix messages for a split turn (empty if not split).
    pub turn_prefix_messages: Vec<Message>,
    /// Whether the cut point split a tool-call turn.
    pub is_split_turn: bool,
    /// Total tokens before compaction.
    pub tokens_before: u32,
}

/// Result of a compaction operation.
#[derive(Debug)]
pub struct CompactionResult {
    /// The generated summary text.
    pub summary: String,
    /// Index of the first kept message.
    pub first_kept_message_index: usize,
    /// Token count before compaction.
    pub tokens_before: u32,
}

// ---------------------------------------------------------------------------
// Cut-point algorithm
// ---------------------------------------------------------------------------

/// Find the cut point by walking backwards from recent messages.
///
/// Returns `(first_kept_message_index, is_split_turn)`.
///
/// Walks backwards accumulating tokens until `keep_recent_tokens` is reached,
/// then finds the nearest valid boundary (user/assistant message) at or after
/// that point. Never cuts between a tool_use assistant message and its
/// corresponding tool_result.
pub fn find_cut_point(
    messages: &[Message],
    keep_recent_tokens: u32,
    start_after_index: Option<usize>,
) -> Option<(usize, bool)> {
    let start = start_after_index.unwrap_or(0);
    if messages.len() <= start {
        return None;
    }

    // Walk backwards from end, accumulating tokens.
    let mut accumulated = 0u32;
    let mut raw_cut = None;
    for i in (start..messages.len()).rev() {
        accumulated += estimate_tokens(&messages[i]);
        if accumulated >= keep_recent_tokens {
            raw_cut = Some(i);
            break;
        }
    }

    let mut cut = raw_cut?; // Not enough messages to compact

    // Advance to a valid boundary: right before a User message, or at an
    // Assistant message that is NOT immediately followed by a Tool result.
    while cut < messages.len() {
        match messages[cut].role {
            Role::User => break,
            Role::Tool | Role::System => {
                cut += 1;
            }
            Role::Assistant => {
                // If the next message is a Tool result we're mid-turn.
                if cut + 1 < messages.len() && messages[cut + 1].role == Role::Tool {
                    cut += 1;
                    continue;
                }
                break;
            }
        }
    }

    // Clamp to valid range.
    if cut >= messages.len() {
        cut = messages.len() - 1;
    }

    let is_split = messages[cut].role == Role::Tool;
    Some((cut, is_split))
}

// ---------------------------------------------------------------------------
// Summarization prompts
// ---------------------------------------------------------------------------

/// System prompt for summarization.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
You are a conversation summarizer for a coding assistant. \
Your goal is to produce a concise but information-dense summary of the conversation history. \
The summary will be used by the assistant to continue working effectively after context compaction.";

/// Initial summarization prompt (no previous summary).
pub const SUMMARIZATION_PROMPT: &str = "\
Please summarize the following conversation into a structured checkpoint. \
Include these sections:

**Goal:** What is the user trying to accomplish?
**Constraints & Preferences:** Any stated requirements, style preferences, or technical constraints.
**Progress:**
- Done: Completed tasks and changes
- In Progress: Current work
- Blocked: Any blockers or open questions
**Key Decisions:** Important technical decisions made during the conversation.
**Next Steps:** What should happen next.
**Critical Context:** File paths, function names, error messages, or other details essential for continuity.

Be specific. Include file paths, function names, error messages, and concrete details. \
Do NOT use vague language like \"various files\" or \"some changes\".";

/// Update summarization prompt (with previous summary).
pub const UPDATE_SUMMARIZATION_PROMPT: &str = "\
A previous compaction produced the following summary:

<previous_summary>
{previous_summary}
</previous_summary>

New messages have been exchanged since then. Please produce an updated summary \
that incorporates both the previous context and the new conversation. \
Use the same structured format as before.";

/// Turn-prefix summarization prompt for split turns.
pub const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "\
The conversation was split mid-turn during compaction. \
Please summarize the following early portion of the current turn. \
Focus on: what the user asked, what tools were called, and any intermediate results. \
Keep it brief but include concrete details (file paths, function names, error messages).";

// ---------------------------------------------------------------------------
// Serialization helper
// ---------------------------------------------------------------------------

/// Serialize conversation messages to text for summarization.
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut output = String::new();
    for msg in messages {
        let role_label = match msg.role {
            Role::User => "[User]",
            Role::Assistant => {
                if msg.tool_calls.is_some() {
                    "[Assistant tool calls]"
                } else {
                    "[Assistant]"
                }
            }
            Role::Tool => "[Tool result]",
            Role::System => "[System]",
        };

        let text = match &msg.content {
            Some(MessageContent::Text(t)) => t.clone(),
            Some(MessageContent::Blocks(blocks)) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    ContentBlock::ToolUse { .. } => None, // serialized via tool_calls
                    ContentBlock::ToolResult { content, .. } => Some(content.as_str()),
                    ContentBlock::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
            None => String::new(),
        };

        // Include tool-call info for assistant messages.
        let tool_info = if let Some(tool_calls) = &msg.tool_calls {
            let calls: Vec<String> = tool_calls
                .iter()
                .map(|tc| format!("{}({})", tc.function.name, tc.function.arguments))
                .collect();
            let joined = calls.join("\n");
            if text.is_empty() {
                joined
            } else {
                format!("{text}\n{joined}")
            }
        } else {
            text
        };

        if !tool_info.is_empty() {
            output.push_str(&format!("{role_label}: {tool_info}\n\n"));
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Provider-calling functions
// ---------------------------------------------------------------------------

/// Shared helper that builds messages, config, and calls the provider.
async fn call_summarizer(
    provider: &dyn Provider,
    model: &str,
    user_prompt: &str,
    max_tokens: u32,
) -> Result<String> {
    let summary_messages = vec![
        Message {
            role: Role::System,
            content: Some(MessageContent::Text(
                SUMMARIZATION_SYSTEM_PROMPT.to_string(),
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(user_prompt.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    let config = AgentConfig {
        model: ModelId::parse(model)
            .unwrap_or_else(|| ModelId::new("unknown", model)),
        max_tokens: Some(max_tokens),
        temperature: Some(0.3),
        system_prompt: None,
        max_iterations: 1,
    };

    let response = provider.chat(model, &summary_messages, &[], &config).await?;
    extract_text(response.message.content)
}

/// Generate a summary of messages using the provider.
pub async fn generate_summary(
    provider: &dyn Provider,
    model: &str,
    messages: &[Message],
    previous_summary: Option<&str>,
    max_tokens: u32,
) -> Result<String> {
    let conversation = serialize_conversation(messages);

    let user_prompt = if let Some(prev) = previous_summary {
        UPDATE_SUMMARIZATION_PROMPT.replace("{previous_summary}", prev)
    } else {
        SUMMARIZATION_PROMPT.to_string()
    };

    let full_prompt = format!("{user_prompt}\n\n{conversation}");
    call_summarizer(provider, model, &full_prompt, max_tokens).await
}

/// Generate a turn-prefix summary for split turns.
pub async fn generate_turn_prefix_summary(
    provider: &dyn Provider,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
) -> Result<String> {
    let conversation = serialize_conversation(messages);
    let full_prompt = format!("{TURN_PREFIX_SUMMARIZATION_PROMPT}\n\n{conversation}");
    call_summarizer(provider, model, &full_prompt, max_tokens).await
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Check whether compaction should trigger given current context size.
pub fn should_compact(
    context_tokens: u32,
    context_window: u32,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

/// Prepare compaction: find the cut point and collect messages into
/// to-summarize and to-keep buckets.
///
/// `previous_first_kept_index` is the index from a prior compaction (if any)
/// — messages before it are already summarized and should be skipped.
pub fn prepare_compaction(
    messages: &[Message],
    settings: &CompactionSettings,
    previous_first_kept_index: Option<usize>,
) -> Option<CompactionPreparation> {
    let (cut_index, is_split) =
        find_cut_point(messages, settings.keep_recent_tokens, previous_first_kept_index)?;

    let (messages_to_summarize, turn_prefix_messages) = if is_split {
        // Find the user message that started this turn.
        let mut turn_start = cut_index;
        while turn_start > 0 && messages[turn_start].role != Role::User {
            turn_start -= 1;
        }
        // If we walked all the way back to the start without finding a User
        // message, summarize everything before the cut.
        if turn_start == 0 && messages[0].role != Role::User {
            turn_start = 0;
        }

        let to_summarize = messages[..turn_start].to_vec();
        let prefix = messages[turn_start..cut_index].to_vec();
        (to_summarize, prefix)
    } else {
        let to_summarize = messages[..cut_index].to_vec();
        (to_summarize, Vec::new())
    };

    let tokens_before: u32 = messages.iter().map(estimate_tokens).sum();

    Some(CompactionPreparation {
        first_kept_message_index: cut_index,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: is_split,
        tokens_before,
    })
}

/// Execute compaction: generate summary (or split summaries), return result.
///
/// For split turns, the history summary and turn-prefix summary are generated
/// concurrently via [`tokio::join!`].
pub async fn compact(
    provider: &dyn Provider,
    model: &str,
    preparation: CompactionPreparation,
    previous_summary: Option<&str>,
    settings: &CompactionSettings,
) -> Result<CompactionResult> {
    let max_tokens = (settings.reserve_tokens as f64 * 0.8) as u32;

    let summary = if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
        let history_fut = generate_summary(
            provider,
            model,
            &preparation.messages_to_summarize,
            previous_summary,
            max_tokens,
        );
        let prefix_fut = generate_turn_prefix_summary(
            provider,
            model,
            &preparation.turn_prefix_messages,
            (max_tokens as f64 * 0.5) as u32,
        );

        let (history_result, prefix_result) = tokio::join!(history_fut, prefix_fut);
        let history = history_result?;
        let prefix = prefix_result?;

        format!("{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}")
    } else {
        generate_summary(
            provider,
            model,
            &preparation.messages_to_summarize,
            previous_summary,
            max_tokens,
        )
        .await?
    };

    Ok(CompactionResult {
        summary,
        first_kept_message_index: preparation.first_kept_message_index,
        tokens_before: preparation.tokens_before,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract plain text from optional message content.
fn extract_text(content: Option<MessageContent>) -> Result<String> {
    Ok(match content {
        Some(MessageContent::Text(text)) => text,
        Some(MessageContent::Blocks(blocks)) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionCall;

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_tool_call(name: &str, args: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: args.to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_result(content: &str) -> Message {
        Message {
            role: Role::Tool,
            content: Some(MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            name: None,
        }
    }

    // -- should_compact -----------------------------------------------------

    #[test]
    fn test_should_compact_below_threshold() {
        let settings = CompactionSettings::default();
        // 50k used, 200k window → 200k - 16384 = 183616 headroom
        assert!(!should_compact(50_000, 200_000, &settings));
    }

    #[test]
    fn test_should_compact_above_threshold() {
        let settings = CompactionSettings::default();
        // 190k used, 200k window → only 10k headroom < 16384 reserve
        assert!(should_compact(190_000, 200_000, &settings));
    }

    #[test]
    fn test_should_compact_disabled() {
        let settings = CompactionSettings {
            enabled: false,
            ..Default::default()
        };
        assert!(!should_compact(200_000, 200_000, &settings));
    }

    // -- find_cut_point -----------------------------------------------------

    #[test]
    fn test_find_cut_point_simple() {
        // 10 user messages, each ~5 chars → ~2 tokens each = 20 total.
        // keep_recent = 10 → raw cut after accumulating 10 tokens (5 messages
        // from the end), then advance to next User boundary.
        let messages: Vec<Message> = (0..10)
            .map(|i| user_msg(&format!("msg {i}")))
            .collect();

        let (cut, is_split) = find_cut_point(&messages, 10, None).unwrap();
        // Should cut at a User message boundary.
        assert_eq!(messages[cut].role, Role::User);
        assert!(!is_split);
        // Cut should leave at least the last few messages.
        assert!(cut < messages.len());
    }

    #[test]
    fn test_find_cut_point_with_tool_calls() {
        // Sequence: user, assistant(tool_use), tool(result), user, assistant(text)
        // Must never cut between tool_use and tool_result.
        let messages = vec![
            user_msg("first"),
            assistant_tool_call("read_file", "{\"path\":\"/tmp/a.rs\"}"),
            tool_result("file contents here"),
            user_msg("second"),
            assistant_msg("done"),
        ];

        // keep_recent = 1 → forces cut near the end, but must respect boundaries.
        let result = find_cut_point(&messages, 1, None);
        let (cut, _) = result.unwrap();
        // Cut must not land on a Tool result or between tool_use/tool_result.
        if cut < messages.len() {
            assert_ne!(messages[cut].role, Role::Tool);
        }
        // If cut is at the assistant tool call, the next message must not be Tool.
        if cut + 1 < messages.len()
            && messages[cut].role == Role::Assistant
            && messages[cut].tool_calls.is_some()
        {
            assert_ne!(messages[cut + 1].role, Role::Tool);
        }
    }

    #[test]
    fn test_find_cut_point_split_turn() {
        // user, assistant(tool_use), tool(result), assistant(text)
        // If we cut after tool_result but before final assistant → split turn.
        let messages = vec![
            user_msg("do something"),
            assistant_tool_call("run_cmd", "{}"),
            tool_result("output of command"),
            assistant_msg("here is the result"),
        ];

        // keep_recent = 2 → forces cut into the middle of the turn.
        let result = find_cut_point(&messages, 2, None);
        // It should return Some (there are enough messages).
        assert!(result.is_some());
    }

    #[test]
    fn test_find_cut_point_not_enough_messages() {
        let messages = vec![user_msg("only one")];
        // keep_recent = huge → not enough messages.
        assert!(find_cut_point(&messages, 999_999, None).is_none());
    }

    // -- serialize_conversation ---------------------------------------------

    #[test]
    fn test_serialize_conversation_basic() {
        let messages = vec![user_msg("hello"), assistant_msg("hi there")];
        let text = serialize_conversation(&messages);
        assert!(text.contains("[User]: hello"));
        assert!(text.contains("[Assistant]: hi there"));
    }

    #[test]
    fn test_serialize_conversation_tool_calls() {
        let messages = vec![assistant_tool_call("read_file", "{\"path\":\"/a\"}")];
        let text = serialize_conversation(&messages);
        assert!(text.contains("[Assistant tool calls]"));
        assert!(text.contains("read_file({\"path\":\"/a\"})"));
    }

    #[test]
    fn test_serialize_conversation_skips_empty() {
        let messages = vec![
            Message {
                role: Role::System,
                content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            user_msg("visible"),
        ];
        let text = serialize_conversation(&messages);
        assert!(!text.contains("[System]"));
        assert!(text.contains("[User]: visible"));
    }

    // -- prepare_compaction -------------------------------------------------

    #[test]
    fn test_prepare_compaction_basic() {
        // 20 user messages, each ~5 chars → ~2 tokens = 40 total.
        // keep_recent = 10 → cut somewhere in the first half.
        let messages: Vec<Message> = (0..20)
            .map(|i| user_msg(&format!("msg {i}")))
            .collect();

        let settings = CompactionSettings {
            keep_recent_tokens: 10,
            ..Default::default()
        };

        let prep = prepare_compaction(&messages, &settings, None);
        assert!(prep.is_some());
        let prep = prep.unwrap();
        assert!(prep.first_kept_message_index < messages.len());
        assert!(!prep.messages_to_summarize.is_empty());
        assert_eq!(
            prep.tokens_before,
            messages.iter().map(|m| estimate_tokens(m)).sum::<u32>()
        );
    }

    #[test]
    fn test_prepare_compaction_not_enough_messages() {
        let messages = vec![user_msg("only one")];
        let settings = CompactionSettings {
            keep_recent_tokens: 999_999,
            ..Default::default()
        };
        assert!(prepare_compaction(&messages, &settings, None).is_none());
    }

    // -- CompactionSettings defaults ----------------------------------------

    #[test]
    fn test_compaction_settings_defaults() {
        let s = CompactionSettings::default();
        assert_eq!(s.reserve_tokens, 16_384);
        assert_eq!(s.keep_recent_tokens, 20_000);
        assert!(s.enabled);
    }
}
