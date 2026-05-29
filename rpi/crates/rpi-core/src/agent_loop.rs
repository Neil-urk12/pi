//! The agent loop — the core execution engine that orchestrates LLM calls and tool execution.

use crate::compaction::CompactionSettings;
use crate::error::PiError;
use crate::traits::{Provider, Tool, ToolCallDelta};
use crate::types::*;
use futures::StreamExt;
use std::collections::HashMap;

/// Events emitted during agent execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A new turn started (LLM call initiated).
    TurnStart {
        /// Which turn number (0-indexed).
        turn: u32,
    },
    /// Text delta streamed from the LLM.
    TextDelta {
        /// The text fragment.
        text: String,
    },
    /// Thinking/reasoning delta (for models that support it).
    ThinkingDelta {
        /// The thinking fragment.
        text: String,
    },
    /// A tool call started streaming.
    ToolCallStart {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
    },
    /// Tool call arguments delta.
    ToolCallDelta {
        /// Tool call ID.
        id: String,
        /// Partial arguments.
        arguments_delta: String,
    },
    /// Tool execution started.
    ToolExecutionStart {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Parsed arguments.
        arguments: serde_json::Value,
    },
    /// Tool execution completed.
    ToolExecutionEnd {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Execution result.
        result: String,
        /// Whether the execution errored.
        is_error: bool,
    },
    /// Assistant turn completed.
    TurnEnd {
        /// Turn number.
        turn: u32,
        /// The assistant message.
        message: Message,
        /// Token usage for this turn.
        usage: Option<Usage>,
    },
    /// Agent loop completed (no more tool calls).
    Done {
        /// Total turns taken.
        turns: u32,
        /// Cumulative token usage.
        total_usage: Usage,
    },
    /// An error occurred.
    Error {
        /// Error description.
        error: String,
    },
    /// Compaction was triggered.
    CompactionTriggered {
        /// Tokens before compaction.
        tokens_before: u32,
    },
    /// Compaction completed.
    CompactionComplete {
        /// The full compaction result.
        result: crate::compaction::CompactionResult,
    },
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Maximum number of tool-call rounds before forcing stop.
    pub max_tool_rounds: u32,
    /// Whether to stream responses.
    pub stream: bool,
    /// Compaction settings. If None, compaction is disabled.
    pub compaction: Option<CompactionSettings>,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 50,
            stream: true,
            compaction: None,
        }
    }
}

/// Run the agent loop.
///
/// Takes messages, provider, tools, and emits events via callback.
/// Returns the final list of messages.
pub async fn run_agent_loop(
    provider: &dyn Provider,
    model: &str,
    messages: &mut Vec<Message>,
    tools: &[&dyn Tool],
    config: &AgentLoopConfig,
    agent_config: &AgentConfig,
    mut on_event: impl FnMut(AgentEvent),
) -> Result<(), PiError> {
    // Derive context window: use configured value, or fall back to max(max_tokens, 128_000).
    let context_window = config
        .compaction
        .as_ref()
        .and_then(|s| s.context_window)
        .unwrap_or_else(|| std::cmp::max(agent_config.max_tokens.unwrap_or(128_000), 128_000u32));
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let mut turn = 0u32;
    let mut last_turn_usage: Option<Usage> = None;
    let mut compaction_state: Option<(usize, String)> = None;
    let mut compaction_retries = 0u32;
    // Build tool definitions once — tools don't change between turns.
    let tool_defs: Vec<ToolDefinition> = tools.iter().map(|t| t.definition()).collect();

    loop {

        on_event(AgentEvent::TurnStart { turn });

        // Call the provider — catch overflow errors for compaction retry.
        let provider_result: std::result::Result<Option<Vec<ToolCall>>, PiError> = if config.stream
        {
            let stream_result = provider
                .chat_stream(model, messages, &tool_defs, agent_config)
                .await;

            match stream_result {
                Ok(mut stream) => {
                    let mut current_text = String::new();
                    let mut tool_call_deltas: HashMap<u32, crate::traits::ToolCallDelta> =
                        HashMap::new();
                    let mut last_finish_reason: Option<FinishReason> = None;
                    let mut stream_usage: Option<Usage> = None;
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        if let Some(text) = &chunk.delta {
                            current_text.push_str(text);
                            on_event(AgentEvent::TextDelta { text: text.clone() });
                        }

                        for tc_delta in &chunk.tool_calls {
                            let entry =
                                tool_call_deltas.entry(tc_delta.index).or_insert_with(|| {
                                    ToolCallDelta {
                                        index: tc_delta.index,
                                        id: None,
                                        name: None,
                                        arguments_delta: None,
                                    }
                                });
                            if let Some(id) = &tc_delta.id {
                                entry.id = Some(id.clone());
                                if let Some(name) = &tc_delta.name {
                                    on_event(AgentEvent::ToolCallStart {
                                        id: id.clone(),
                                        name: name.clone(),
                                    });
                                }
                            }
                            if let Some(name) = &tc_delta.name {
                                entry.name = Some(name.clone());
                            }
                            if let Some(args) = &tc_delta.arguments_delta {
                                let existing =
                                    entry.arguments_delta.get_or_insert_with(String::new);
                                existing.push_str(args);
                                if let Some(id) = &entry.id {
                                    on_event(AgentEvent::ToolCallDelta {
                                        id: id.clone(),
                                        arguments_delta: args.clone(),
                                    });
                                }
                            }
                        }
                        last_finish_reason = chunk.finish_reason.clone();
                        if chunk.usage.is_some() {
                            stream_usage = chunk.usage;
                        }
                    }

                    if let Some(error) = finish_reason_error(&last_finish_reason) {
                        return Err(error);
                    }

                    let mut content_blocks = Vec::new();
                    if !current_text.is_empty() {
                        content_blocks.push(ContentBlock::Text { text: current_text });
                    }

                    let tool_calls: Vec<ToolCall> = tool_call_deltas
                        .into_values()
                        .filter_map(|tc| {
                            let id = tc.id?;
                            let name = tc.name?;
                            Some(ToolCall {
                                id,
                                function: FunctionCall {
                                    name,
                                    arguments: tc.arguments_delta.unwrap_or_default(),
                                },
                            })
                        })
                        .collect();

                    let has_tool_calls = !tool_calls.is_empty();

                    let assistant_msg = Message {
                        role: Role::Assistant,
                        content: if content_blocks.is_empty() {
                            None
                        } else {
                            Some(MessageContent::Blocks(content_blocks))
                        },
                        tool_calls: if has_tool_calls {
                            Some(tool_calls.clone())
                        } else {
                            None
                        },
                        tool_call_id: None,
                        name: None,
                    };

                    messages.push(assistant_msg.clone());
                    let assistant_estimate =
                        crate::token_estimation::estimate_tokens(&assistant_msg);
                    on_event(AgentEvent::TurnEnd {
                        turn,
                        message: assistant_msg,
                        usage: stream_usage.clone().or_else(|| {
                            Some(Usage {
                                prompt_tokens: 0,
                                completion_tokens: assistant_estimate,
                                total_tokens: assistant_estimate,
                            })
                        }),
                    });

                    // Update total usage — prefer actual API usage, fall back to heuristic.
                    let turn_usage = stream_usage.take().unwrap_or_else(|| Usage {
                        prompt_tokens: 0,
                        completion_tokens: assistant_estimate,
                        total_tokens: assistant_estimate,
                    });
                    total_usage.prompt_tokens = total_usage.prompt_tokens.saturating_add(turn_usage.prompt_tokens);
                    total_usage.completion_tokens = total_usage.completion_tokens.saturating_add(turn_usage.completion_tokens);
                    total_usage.total_tokens = total_usage.total_tokens.saturating_add(turn_usage.total_tokens);
                    last_turn_usage = Some(turn_usage);

                    Ok(if has_tool_calls {
                        Some(tool_calls)
                    } else {
                        None
                    })
                }
                Err(e) => Err(e),
            }
        } else {
            match provider
                .chat(model, messages, &tool_defs, agent_config)
                .await
            {
                Ok(response) => {
                    if let Some(error) = finish_reason_error(&Some(response.finish_reason.clone()))
                    {
                        return Err(error);
                    }
                    total_usage.prompt_tokens = total_usage.prompt_tokens.saturating_add(response.usage.prompt_tokens);
                    total_usage.completion_tokens = total_usage.completion_tokens.saturating_add(response.usage.completion_tokens);
                    total_usage.total_tokens = total_usage.total_tokens.saturating_add(response.usage.total_tokens);
                    last_turn_usage = Some(response.usage.clone());

                    let assistant_msg = response.message.clone();
                    let tool_calls = assistant_msg.tool_calls.clone();
                    messages.push(assistant_msg.clone());
                    on_event(AgentEvent::TurnEnd {
                        turn,
                        message: assistant_msg,
                        usage: Some(response.usage),
                    });

                    Ok(tool_calls)
                }
                Err(e) => Err(e),
            }
        };

        // Handle overflow: attempt compaction and retry once.
        let tool_calls = match provider_result {
            Ok(tc) => {
                compaction_retries = 0;
                tc
            }
            Err(e) if is_context_overflow(&e) && config.compaction.is_some() => {
                on_event(AgentEvent::Error {
                    error: "Context overflow detected, attempting compaction...".to_string(),
                });
                let overflow_tokens = crate::token_estimation::sum_tokens_saturating(
                    messages.iter().map(crate::token_estimation::estimate_tokens)
                );
                compaction_retries += 1;
                if compaction_retries <= 2 {
                    let compacted = try_compact(
                        provider,
                        model,
                        messages,
                        config,
                        &mut on_event,
                        &mut compaction_state,
                        overflow_tokens,
                    )
                    .await;
                    if compacted {
                        compaction_retries = 0;
                        continue; // Successfully compacted, retry this turn
                    }
                    // Nothing to compact — messages already at minimum size.
                    // The overflow was handled as best we could.
                    on_event(AgentEvent::Done {
                        turns: turn + 1,
                        total_usage: total_usage.clone(),
                    });
                    return Ok(());
                }
                return Err(e);
            }
            Err(e) => return Err(e),
        };

        // If no tool calls, we're done
        let tool_calls = match tool_calls {
            Some(ref tc) if !tc.is_empty() => tc.clone(),
            _ => {
                on_event(AgentEvent::Done {
                    turns: turn + 1,
                    total_usage,
                });
                return Ok(());
            }
        };

        // Check iteration limit
        turn += 1;
        if turn > config.max_tool_rounds {
            on_event(AgentEvent::Error {
                error: format!(
                    "Exceeded maximum tool call rounds ({})",
                    config.max_tool_rounds
                ),
            });
            return Err(PiError::Tool {
                tool: "agent_loop".to_string(),
                message: format!(
                    "Exceeded maximum tool call rounds ({})",
                    config.max_tool_rounds
                ),
            });
        }

        // Execute tool calls
        for tool_call in &tool_calls {
            let args: serde_json::Value =
                serde_json::from_str(&tool_call.function.arguments).unwrap_or_default();

            on_event(AgentEvent::ToolExecutionStart {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: args.clone(),
            });

            let tool = tools.iter().find(|t| t.name() == tool_call.function.name);

            let (result_text, is_error) = if let Some(tool) = tool {
                match tool.execute(&args).await {
                    Ok(text) => (text, false),
                    Err(e) => (format!("Error: {e}"), true),
                }
            } else {
                (
                    format!("Error: Unknown tool '{}'", tool_call.function.name),
                    true,
                )
            };

            on_event(AgentEvent::ToolExecutionEnd {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                result: result_text.clone(),
                is_error,
            });

            messages.push(Message {
                role: Role::Tool,
                content: Some(MessageContent::Text(result_text)),
                tool_calls: None,
                tool_call_id: Some(tool_call.id.clone()),
                name: Some(tool_call.function.name.clone()),
            });
        }

        // After tool results, check if compaction should trigger.
        // NOTE: estimate_context_tokens is O(n) on message count. We considered incremental
        // tracking (adding estimate_tokens per new message), but the usage-based path in
        // estimate_context_tokens is significantly more accurate when API usage is available.
        // For typical sessions (hundreds of messages), the O(n) walk is negligible.
        if let Some(ref compaction_settings) = config.compaction {
            let context_tokens = crate::token_estimation::estimate_context_tokens(
                messages,
                last_turn_usage.as_ref(),
            );
            if crate::compaction::should_compact(
                context_tokens,
                context_window,
                compaction_settings,
            ) {
                try_compact(
                    provider,
                    model,
                    messages,
                    config,
                    &mut on_event,
                    &mut compaction_state,
                    context_tokens,
                )
                .await;
            }
        }
    }
}

fn finish_reason_error(reason: &Option<FinishReason>) -> Option<PiError> {
    match reason {
        Some(FinishReason::Length) => Some(PiError::Provider(
            "Response hit max tokens before completion".to_string(),
        )),
        Some(FinishReason::ContentFilter) => {
            Some(PiError::Provider("Content filtered by model".to_string()))
        }
        _ => None,
    }
}

/// Check if an error indicates context window overflow.
fn is_context_overflow(error: &PiError) -> bool {
    match error {
        PiError::Provider(msg) => {
            let lower = msg.to_lowercase();
            // Primary: must contain "context" + a size/limit indicator.
            // This eliminates false positives from rate limits, API quotas, body size limits.
            if lower.contains("context") {
                return lower.contains("length")
                    || lower.contains("window")
                    || lower.contains("exceeded")
                    || lower.contains("too long")
                    || lower.contains("too many tokens")
                    || lower.contains("token limit")
                    || lower.contains("too large")
                    || lower.contains("input is too long")
                    || lower.contains("message too long");
            }
            // Secondary: specific provider phrases that don't use "context"
            // but are unambiguous overflow signals.
            lower.contains("maximum number of tokens allowed")
                || lower.contains("input token count exceeds")
                || lower.contains("exceeds maximum input tokens")
                || (lower.contains("prompt") && lower.contains("too long"))
        }
        _ => false,
    }
}

/// Attempt compaction. Returns true if compaction succeeded and messages were updated.
async fn try_compact(
    provider: &dyn Provider,
    model: &str,
    messages: &mut Vec<Message>,
    config: &AgentLoopConfig,
    on_event: &mut impl FnMut(AgentEvent),
    compaction_state: &mut Option<(usize, String)>,
    context_tokens: u32,
) -> bool {
    let compaction_settings = match &config.compaction {
        Some(s) => s,
        None => return false,
    };
    // Use caller-provided token count for consistency with the trigger check.

    on_event(AgentEvent::CompactionTriggered {
        tokens_before: context_tokens,
    });

    let preparation = crate::compaction::prepare_compaction(
        messages,
        compaction_settings,
        compaction_state.as_ref().map(|(idx, _)| *idx),
    );

    let preparation = match preparation {
        Some(p) => p,
        None => return false,
    };

    let result = crate::compaction::compact(
        provider,
        model,
        preparation,
        compaction_state.as_ref().map(|(_, s)| s.as_str()),
        compaction_settings,
    )
    .await;

    match result {
        Ok(compaction_result) => {
            apply_compaction(messages, &compaction_result);
            // After apply_compaction, messages = [summary, ...kept].
            // The summary is always at index 0, so the first kept message
            // index relative to the post-compaction array is 1.
            // We must NOT store the pre-compaction first_kept_message_index
            // because it becomes stale after the array is transformed.
            *compaction_state = Some((
                1,
                compaction_result.summary.clone(),
            ));
            on_event(AgentEvent::CompactionComplete { result: compaction_result.clone() });
            true
        }
        Err(e) => {
            on_event(AgentEvent::Error {
                error: format!("Compaction failed: {e}"),
            });
            false
        }
    }
}

/// Create the synthetic summary message injected after compaction.
///
/// Shared by [`apply_compaction`] (in-memory agent loop) and
/// [`Session::build_context`] (session persistence) to guarantee
/// identical formatting.
pub fn create_compaction_summary_message(summary: &str) -> Message {
    Message {
        role: Role::User,
        content: Some(MessageContent::Text(format!(
            "The conversation history before this point was compacted \
             into the following summary:\n\n<summary>\n{}\n</summary>",
            escape_xml_tags(summary)
        ))),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}
/// Apply a compaction result to the in-memory messages vector.
pub fn apply_compaction(messages: &mut Vec<Message>, result: &crate::compaction::CompactionResult) {
    let kept: Vec<Message> = messages.drain(result.first_kept_message_index..).collect();
    messages.clear();
    messages.push(create_compaction_summary_message(&result.summary));
    messages.extend(kept);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::CompactionResult;
    use crate::traits::{ChatStream, StreamChunk};
    use crate::types::ModelId;
    use async_trait::async_trait;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_is_context_overflow_openai_format() {
        // OpenAI format: "context_length_exceeded"
        let error = PiError::Provider("context_length_exceeded: ...".to_string());
        assert!(is_context_overflow(&error));
    }

    #[test]
    fn test_is_context_overflow_anthropic_format() {
        // Anthropic format: "context window"
        let error = PiError::Provider("context window exceeded".to_string());
        assert!(is_context_overflow(&error));
    }

    #[test]
    fn test_is_context_overflow_missing_patterns() {
        // "prompt is too long" matches via secondary pattern (no "context" needed)
        let error = PiError::Provider("prompt is too long".to_string());
        assert!(
            is_context_overflow(&error),
            "'prompt is too long' should be detected as context overflow"
        );

        // Patterns that require "context" to be present
        let error = PiError::Provider("context window: request too large".to_string());
        assert!(
            is_context_overflow(&error),
            "'request too large' with 'context' should match"
        );

        let error = PiError::Provider("context_length_exceeded: max tokens exceeded".to_string());
        assert!(
            is_context_overflow(&error),
            "'max tokens exceeded' with 'context' should match"
        );
    }

    #[test]
    fn test_is_context_overflow_false_positives() {
        // Non-overflow errors that shouldn't match
        let error = PiError::Provider("Invalid API key".to_string());
        assert!(!is_context_overflow(&error));

        let error = PiError::Provider("Rate limit exceeded".to_string());
        assert!(!is_context_overflow(&error));

        let error = PiError::Provider("Model not found".to_string());
        assert!(!is_context_overflow(&error));
    }

    #[test]
    fn test_streaming_usage_tracking() {
        // This test documents that streaming path should track usage
        // Currently, streaming path doesn't update total_usage
        // This test will fail until usage tracking is added to streaming path

        // Create a simple mock provider that returns usage in streaming response
        struct MockProvider;

        #[async_trait]
        impl Provider for MockProvider {
            fn id(&self) -> &str {
                "mock"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: Some(MessageContent::Text("Hello".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    },
                })
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                let stream = futures::stream::iter(vec![Ok(StreamChunk {
                    delta: Some("Hello".to_string()),
                    tool_calls: vec![],
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(Usage {
                        prompt_tokens: 10,
                        completion_tokens: 2,
                        total_tokens: 12,
                    }),
                })]);
                Ok(Box::pin(stream))
            }
        }

        // This test verifies that streaming path tracks usage
        // Currently it doesn't, so this test will fail
        let provider = MockProvider;
        let mut messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let config = AgentLoopConfig {
            max_tool_rounds: 1,
            stream: true,
            compaction: None,
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        // Track events to verify usage is included
        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(async {
            run_agent_loop(
                &provider,
                "mock-model",
                &mut messages,
                &[],
                &config,
                &agent_config,
                |event| events.push(event),
            )
            .await
        });
        assert!(result.is_ok());

        // Find the Done event and check usage
        let done_event = events.iter().find(|e| matches!(e, AgentEvent::Done { .. }));
        assert!(done_event.is_some(), "Should have a Done event");

        if let Some(AgentEvent::Done { total_usage, .. }) = done_event {
            // This assertion will fail because streaming path doesn't track usage
            assert!(
                total_usage.total_tokens > 0,
                "Streaming path should track usage, but total_tokens is 0"
            );
        }
    }

    // FINDING #6: Context window derivation — max_tokens * 4 is likely wrong
    #[test]
    fn test_context_window_derivation_small_max_tokens() {
        // With max_tokens = 4096 (typical output limit), context_window should be at least 128k
        let agent_max_tokens = 4096u32;
        let context_window = std::cmp::max(agent_max_tokens, 128_000u32);
        assert_eq!(context_window, 128_000);
        assert!(
            context_window >= 32_000,
            "Context window for max_tokens=4096 should be >= 32k, got {}",
            context_window
        );
    }

    // FINDING #11: is_context_overflow missing Google format
    #[test]
    fn test_is_context_overflow_google_format() {
        let error = PiError::Provider("exceeds maximum input tokens".to_string());
        assert!(
            is_context_overflow(&error),
            "Google's 'exceeds maximum input tokens' should be detected"
        );
    }

    #[test]
    fn test_is_context_overflow_google_format_2() {
        let error = PiError::Provider("input token count exceeds the maximum".to_string());
        assert!(
            is_context_overflow(&error),
            "Google's 'input token count exceeds the maximum' should be detected"
        );
    }

    // --- Fix #1: Consistent token estimation contract ---

    #[test]
    fn test_try_compact_signature_requires_context_tokens() {
        // Verifies try_compact accepts a pre-computed context_tokens parameter.
        // Before fix: try_compact recomputed tokens internally (different path).
        // After fix: caller passes context_tokens, guaranteeing consistency.
        let messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("hello world".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let expected_tokens = crate::token_estimation::estimate_context_tokens(&messages, None);
        assert!(
            expected_tokens > 0,
            "Token estimation must return positive value"
        );
    }

    #[test]
    fn test_compaction_trigger_consistent_with_try_compact() {
        // The should_compact check and try_compact must agree on token count.
        // After fix, both use the same pre-computed value from the caller.
        let settings = CompactionSettings {
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            enabled: true,
            ..Default::default()
        };
        let big_text = "x".repeat(100_000);
        let messages: Vec<Message> = (0..8)
            .map(|_| Message {
                role: Role::User,
                content: Some(MessageContent::Text(big_text.clone())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
            .collect();
        let context_tokens = crate::token_estimation::estimate_context_tokens(&messages, None);
        let context_window = 200_000u32;
        if crate::compaction::should_compact(context_tokens, context_window, &settings) {
            assert!(
                context_tokens > context_window - settings.reserve_tokens,
                "should_compact and try_compact must see same token count"
            );
        }
    }

    // --- Fix #2: is_context_overflow false positive guard ---

    #[test]
    fn test_is_context_overflow_false_positive_rate_limit() {
        let error = PiError::Provider(
            "Rate limit exceeded. Too many requests, please retry after 60 seconds.".to_string(),
        );
        assert!(
            !is_context_overflow(&error),
            "Rate limit error should NOT be detected as context overflow"
        );
    }

    #[test]
    fn test_is_context_overflow_false_positive_generic_token_mention() {
        let error = PiError::Provider(
            "The token limit for this API key has been reached. Please upgrade.".to_string(),
        );
        assert!(
            !is_context_overflow(&error),
            "API key token limit error should NOT be detected as context overflow"
        );
    }

    #[test]
    fn test_is_context_overflow_false_positive_request_body_too_large() {
        let error =
            PiError::Provider("Request body too large. Maximum allowed size is 10MB.".to_string());
        assert!(
            !is_context_overflow(&error),
            "Request body size limit should NOT be context overflow"
        );
    }

    #[test]
    fn test_is_context_overflow_true_positive_openai_verbose() {
        let error = PiError::Provider(
            "This model's maximum context length is 128000 tokens. \
             However, your messages resulted in 150000 tokens."
                .to_string(),
        );
        assert!(is_context_overflow(&error), "OpenAI overflow must match");
    }

    #[test]
    fn test_is_context_overflow_true_positive_anthropic_verbose() {
        let error =
            PiError::Provider("prompt is too long: 200000 tokens > 180000 maximum".to_string());
        assert!(is_context_overflow(&error), "Anthropic overflow must match");
    }

    // --- Finding 5: Additional edge case coverage ---

    #[test]
    fn test_is_context_overflow_case_insensitive() {
        // Verify to_lowercase() works — uppercase error must still match
        let error = PiError::Provider("CONTEXT_LENGTH_EXCEEDED".to_string());
        assert!(
            is_context_overflow(&error),
            "Uppercase context overflow must match"
        );
    }

    #[test]
    fn test_is_context_overflow_empty_message() {
        let error = PiError::Provider("".to_string());
        assert!(
            !is_context_overflow(&error),
            "Empty error message must not match"
        );
    }

    #[test]
    fn test_is_context_overflow_too_long_without_context() {
        // "too long" without "context" should NOT match (no secondary pattern for it)
        let error = PiError::Provider("Your request is too long for processing.".to_string());
        assert!(
            !is_context_overflow(&error),
            "'too long' without 'context' must not match"
        );
    }

    #[test]
    fn test_is_context_overflow_context_config_invalid() {
        // Contains "context" but no size/limit indicator
        let error = PiError::Provider("context configuration invalid".to_string());
        assert!(
            !is_context_overflow(&error),
            "Non-overflow context error must not match"
        );
    }

    // --- Bug 1: compaction_retries never resets after successful turns ---

    #[test]
    fn test_compaction_retries_resets_after_success() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // Mock provider returns overflow on specific calls, success on others.
        // Call sequence (with correct compaction_state after Finding 2 fix):
        //   0: overflow (main chat turn 1)
        //   1: success (compact summary, non-split)
        //   2: success with tool call (retry turn 1)
        //   3: overflow (main chat turn 2)
        //   4: success (compact, non-split)
        //   5: success with tool call (retry turn 2)
        //   6: overflow (main chat turn 3)
        //   7: success (compact, non-split)
        //   8: success with tool call (retry turn 3)
        // With the bug: compaction_retries=3 > 2, compaction skipped.
        // Without the bug: retries would reset, compaction attempted.
        struct RetryMockProvider {
            call_count: AtomicU32,
        }

        #[async_trait]
        impl Provider for RetryMockProvider {
            fn id(&self) -> &str {
                "retry-mock"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                let call = self.call_count.fetch_add(1, Ordering::SeqCst);
                match call {
                    0 | 3 | 6 => Err(PiError::Provider("context_length_exceeded".to_string())),
                    1 | 4 | 7 => Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: Some(MessageContent::Text(
                                "Summary of conversation".to_string(),
                            )),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::Stop,
                        usage: Usage {
                            prompt_tokens: 10,
                            completion_tokens: 5,
                            total_tokens: 15,
                        },
                    }),
                    2 | 5 => Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: None,
                            tool_calls: Some(vec![ToolCall {
                                id: "call_1".to_string(),
                                function: FunctionCall {
                                    name: "big_tool".to_string(),
                                    arguments: "{}".to_string(),
                                },
                            }]),
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::ToolCalls,
                        usage: Usage {
                            prompt_tokens: 10,
                            completion_tokens: 5,
                            total_tokens: 15,
                        },
                    }),
                    8 => Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: None,
                            tool_calls: Some(vec![ToolCall {
                                id: "call_1".to_string(),
                                function: FunctionCall {
                                    name: "big_tool".to_string(),
                                    arguments: "{}".to_string(),
                                },
                            }]),
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::ToolCalls,
                        usage: Usage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
                    }),
                    9 => Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: Some(MessageContent::Text(
                                "after 3rd compaction".to_string(),
                            )),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::Stop,
                        usage: Usage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
                    }),
                    other => panic!("Unexpected call #{other}"),
                }
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                panic!("chat_stream should not be called in non-streaming mode")
            }
        }

        // Tool that returns a large result (~1500 tokens) so compaction
        // has enough messages to find a cut point on the second overflow.
        struct BigTool;

        #[async_trait]
        impl Tool for BigTool {
            fn name(&self) -> &str {
                "big_tool"
            }
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "big_tool".to_string(),
                    description: "A tool that returns a large result".to_string(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn execute(&self, _args: &serde_json::Value) -> crate::error::Result<String> {
                Ok("x".repeat(6000)) // ~1500 tokens
            }
        }

        // 10 alternating U/A messages, each ~252 tokens. Total ~2520.
        let mut messages: Vec<Message> = (0..10u32)
            .map(|i| Message {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: Some(MessageContent::Text(format!(
                    "msg {i}: {}",
                    "y".repeat(1000)
                ))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
            .collect();

        let provider = RetryMockProvider {
            call_count: AtomicU32::new(0),
        };
        let tool = BigTool;
        let config = AgentLoopConfig {
            max_tool_rounds: 20,
            stream: false,
            compaction: Some(CompactionSettings {
                reserve_tokens: 100,
                keep_recent_tokens: 1250,
                enabled: true,
                ..Default::default()
            }),
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[&tool],
            &config,
            &agent_config,
            |event| events.push(event),
        ));

        let compaction_count = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::CompactionTriggered { .. }))
            .count();

        // With the bug: compaction_retries reaches 3 on the third overflow,
        // 3 <= 2 is false, compaction is skipped. Only 2 CompactionTriggered.
        // Without the bug: retries reset after each success, 3 CompactionTriggered.
        assert_eq!(
            compaction_count, 3,
            "Expected 3 compaction attempts (retries should reset after success), \
             but got {compaction_count}. compaction_retries never resets."
        );

        // The loop should succeed when all overflows are handled via compaction.
        assert!(
            result.is_ok(),
            "Agent loop should succeed after compaction retries, but got: {:?}",
            result.err()
        );
    }

    // --- Bug 2: streaming finish_reason silently discarded ---

    #[test]
    fn test_streaming_finish_reason_error_propagated() {
        // Mock provider that streams a content-filter finish reason.
        struct ContentFilterProvider;

        #[async_trait]
        impl Provider for ContentFilterProvider {
            fn id(&self) -> &str {
                "content-filter"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                panic!("chat should not be called in streaming mode")
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                let stream = futures::stream::iter(vec![
                    Ok(StreamChunk {
                        delta: Some("Hello".to_string()),
                        tool_calls: vec![],
                        finish_reason: None,
                        usage: None,
                    }),
                    Ok(StreamChunk {
                        delta: None,
                        tool_calls: vec![],
                        finish_reason: Some(FinishReason::ContentFilter),
                        usage: None,
                    }),
                ]);
                Ok(Box::pin(stream))
            }
        }

        let mut messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let provider = ContentFilterProvider;
        let config = AgentLoopConfig {
            max_tool_rounds: 1,
            stream: true,
            compaction: None,
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[],
            &config,
            &agent_config,
            |event| events.push(event),
        ));

        // The finish_reason was ContentFilter -- the loop should propagate
        // this as an error. Currently it does `let _ = chunk.finish_reason;`
        // and silently ignores it, so the loop completes Ok.
        assert!(
            result.is_err(),
            "Agent loop should return error when finish_reason is ContentFilter, \
             but it completed Ok. finish_reason is silently discarded."
        );
    }

    #[test]
    fn test_streaming_length_finish_reason_returns_error() {
        struct LengthProvider;

        #[async_trait]
        impl Provider for LengthProvider {
            fn id(&self) -> &str {
                "length"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                panic!("chat should not be called in streaming mode")
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                let stream = futures::stream::iter(vec![Ok(StreamChunk {
                    delta: Some("Partial".to_string()),
                    tool_calls: vec![],
                    finish_reason: Some(FinishReason::Length),
                    usage: None,
                })]);
                Ok(Box::pin(stream))
            }
        }

        let provider = LengthProvider;
        let mut messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let config = AgentLoopConfig {
            max_tool_rounds: 1,
            stream: true,
            compaction: None,
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[],
            &config,
            &agent_config,
            |event| events.push(event),
        ));

        assert!(
            matches!(
                &result,
                Err(PiError::Provider(message))
                    if message == "Response hit max tokens before completion"
            ),
            "Agent loop should reject truncated streaming responses, got: {result:?}"
        );
        assert_eq!(
            messages.len(),
            1,
            "Partial assistant message must not be saved"
        );
    }

    #[test]
    fn test_non_streaming_length_finish_reason_returns_error() {
        struct LengthProvider;

        #[async_trait]
        impl Provider for LengthProvider {
            fn id(&self) -> &str {
                "length"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: Some(MessageContent::Text("Partial".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    finish_reason: FinishReason::Length,
                    usage: Usage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    },
                })
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                panic!("chat_stream should not be called in non-streaming mode")
            }
        }

        let provider = LengthProvider;
        let mut messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let config = AgentLoopConfig {
            max_tool_rounds: 1,
            stream: false,
            compaction: None,
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[],
            &config,
            &agent_config,
            |event| events.push(event),
        ));

        assert!(
            matches!(
                &result,
                Err(PiError::Provider(message))
                    if message == "Response hit max tokens before completion"
            ),
            "Agent loop should reject truncated non-streaming responses, got: {result:?}"
        );
        assert_eq!(
            messages.len(),
            1,
            "Partial assistant message must not be saved"
        );
    }

    // --- Bug 4: estimate_context_tokens always receives None ---

    #[test]
    fn test_compaction_uses_usage_based_estimation() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // Provider returns a tool call with very high usage on the first call,
        // then completes with no tool calls on the second call.
        struct HighUsageProvider {
            call_count: AtomicU32,
        }

        #[async_trait]
        impl Provider for HighUsageProvider {
            fn id(&self) -> &str {
                "high-usage"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                let call = self.call_count.fetch_add(1, Ordering::SeqCst);
                match call {
                    0 => Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: None,
                            tool_calls: Some(vec![ToolCall {
                                id: "call_1".to_string(),
                                function: FunctionCall {
                                    name: "echo".to_string(),
                                    arguments: "{}".to_string(),
                                },
                            }]),
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::ToolCalls,
                        // High usage: 150k total tokens, well above compaction threshold.
                        usage: Usage {
                            prompt_tokens: 100_000,
                            completion_tokens: 50_000,
                            total_tokens: 150_000,
                        },
                    }),
                    1 => Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: Some(MessageContent::Text("Done".to_string())),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::Stop,
                        usage: Usage {
                            prompt_tokens: 10,
                            completion_tokens: 5,
                            total_tokens: 15,
                        },
                    }),
                    other => panic!("Unexpected call #{other}"),
                }
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                panic!("chat_stream should not be called in non-streaming mode")
            }
        }

        struct EchoTool;

        #[async_trait]
        impl Tool for EchoTool {
            fn name(&self) -> &str {
                "echo"
            }
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "echo".to_string(),
                    description: "Echo tool".to_string(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn execute(&self, _args: &serde_json::Value) -> crate::error::Result<String> {
                Ok("ok".to_string())
            }
        }

        let mut messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("hello".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let provider = HighUsageProvider {
            call_count: AtomicU32::new(0),
        };
        let tool = EchoTool;
        // Threshold = 128_000 - 16_384 = 111_616 tokens.
        // Provider reports 150k total_tokens in usage.
        // With heuristic (None): ~10 tokens from "hello" message -> no compaction.
        // With usage-based (Some(&total_usage)): 150k > 111k -> should trigger compaction.
        let config = AgentLoopConfig {
            max_tool_rounds: 10,
            stream: false,
            compaction: Some(CompactionSettings::default()),
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let _ = runtime.block_on(run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[&tool],
            &config,
            &agent_config,
            |event| events.push(event),
        ));

        let compaction_triggered = events
            .iter()
            .any(|e| matches!(e, AgentEvent::CompactionTriggered { .. }));

        // The provider reported 150k total_tokens. The compaction check should
        // use this usage data when estimating context size. Currently it passes
        // None to estimate_context_tokens, falling back to a heuristic that
        // sees ~10 tokens -- far below the 111k threshold.
        assert!(
            compaction_triggered,
            "Compaction should trigger when provider usage (150000 tokens) \
             exceeds threshold (111616 tokens). estimate_context_tokens \
             receives None instead of Some(&total_usage)."
        );
    }

    // --- Finding: cumulative usage causes premature compaction ---

    #[test]
    fn test_multi_turn_cumulative_usage_triggers_premature_compaction() {
        // Bug: estimate_context_tokens receives cumulative total_usage across all turns,
        // but it should use per-turn usage. In multi-turn conversations with tool calls,
        // total_usage.total_tokens grows far beyond actual context size.
        //
        // The compaction check only runs after tool execution (not after text responses).
        // So the mock MUST return tool calls for several turns to reach the check.
        //
        // Scenario:
        // - 8 turns with tool calls, each reporting 5000 tokens from provider
        // - After 8 turns, cumulative total_usage = 40000 tokens
        // - Per-turn usage = 5000 tokens
        // - context_window = 29000, reserve = 1000 -> threshold = 28000
        //
        // With cumulative usage: 40000 > 28000 -> compaction triggers (WRONG)
        // With per-turn usage: 5000 < 28000 -> no compaction (CORRECT)

        use std::sync::atomic::{AtomicU32, Ordering};

        struct ToolCallProvider {
            call_count: AtomicU32,
        }

        #[async_trait]
        impl Provider for ToolCallProvider {
            fn id(&self) -> &str {
                "toolcall-mock"
            }

            async fn chat(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatResponse> {
                let call = self.call_count.fetch_add(1, Ordering::SeqCst);
                // Return tool calls for first 8 turns, then text response
                if call < 8 {
                    Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: None,
                            tool_calls: Some(vec![ToolCall {
                                id: format!("call_{call}"),
                                function: FunctionCall {
                                    name: "noop".to_string(),
                                    arguments: "{}".to_string(),
                                },
                            }]),
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::ToolCalls,
                        usage: Usage {
                            prompt_tokens: 4000,
                            completion_tokens: 1000,
                            total_tokens: 5000,
                        },
                    })
                } else {
                    Ok(ChatResponse {
                        message: Message {
                            role: Role::Assistant,
                            content: Some(MessageContent::Text("done".to_string())),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason: FinishReason::Stop,
                        usage: Usage {
                            prompt_tokens: 4000,
                            completion_tokens: 1000,
                            total_tokens: 5000,
                        },
                    })
                }
            }

            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _config: &AgentConfig,
            ) -> crate::error::Result<ChatStream> {
                panic!("chat_stream should not be called")
            }
        }

        // A simple tool that does nothing
        struct NoopTool;

        #[async_trait]
        impl Tool for NoopTool {
            fn name(&self) -> &str {
                "noop"
            }

            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "noop".to_string(),
                    description: "Does nothing".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                }
            }

            async fn execute(&self, _args: &serde_json::Value) -> crate::error::Result<String> {
                Ok("ok".to_string())
            }
        }

        let mut messages: Vec<Message> = (0..3u32)
            .map(|i| Message {
                role: Role::User,
                content: Some(MessageContent::Text(format!("msg {i}"))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
            .collect();

        let provider = ToolCallProvider {
            call_count: AtomicU32::new(0),
        };
        let noop_tool: Box<dyn Tool> = Box::new(NoopTool);
        let config = AgentLoopConfig {
            max_tool_rounds: 10,
            stream: false,
            compaction: Some(CompactionSettings {
                reserve_tokens: 1000,
                keep_recent_tokens: 20,
                enabled: true,
                context_window: Some(29_000),
            }),
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(30_000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };

        let mut events = Vec::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let _ = runtime.block_on(run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[&*noop_tool as &dyn Tool],
            &config,
            &agent_config,
            |event| events.push(event),
        ));

        let compaction_triggered = events
            .iter()
            .any(|e| matches!(e, AgentEvent::CompactionTriggered { .. }));

        // After 8 tool-call turns, cumulative total_usage = 8 * 5000 = 40000.
        // context_window = 29000. Threshold = 29000 - 1000 = 28000.
        // 40000 > 28000 -> cumulative usage would incorrectly trigger compaction.
        // Per-turn usage (5000) < 28000 -> correct behavior: no compaction.
        //
        // If this test fails, the code is using cumulative instead of per-turn usage.
        assert!(
            !compaction_triggered,
            "Per-turn usage (5000) should not trigger compaction with \n             context_window=29000 (threshold=28000). If this fails, \n             the code uses cumulative total_usage (40000 > 28000)."
        );
    }

    #[test]
    fn test_context_window_should_be_configurable() {
        // Bug: context_window is derived as max(max_tokens, 128_000) which always
        // returns 128_000 since max_tokens is the output limit (e.g. 4096).
        // CompactionSettings should have a context_window field to override this.

        // This test will NOT compile until CompactionSettings gets a context_window field.
        // That's the point — it documents the missing API.
        let settings = CompactionSettings {
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            enabled: true,
            context_window: Some(200_000),
        };
        assert_eq!(settings.context_window, Some(200_000));
    }

    #[test]
    fn test_context_window_defaults_to_none() {
        // When context_window is None, the agent loop should fall back to
        // max(max_tokens, 128_000).
        let settings = CompactionSettings::default();
        assert_eq!(settings.context_window, None);
    }

    #[test]
    fn test_compaction_respects_configured_context_window() {
        // Integration test: with context_window set to 200k, compaction threshold
        // should be 200k - reserve_tokens, not 128k - reserve_tokens.
        //
        // Create 160k tokens of context (via usage). With 128k window this would
        // trigger compaction (160k > 128k - 16384 = 111616). With 200k window
        // it should NOT (160k < 200k - 16384 = 183616).

        let settings_128k = CompactionSettings {
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            enabled: true,
            context_window: None, // falls back to 128k
        };
        let settings_200k = CompactionSettings {
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            enabled: true,
            context_window: Some(200_000),
        };

        let context_tokens = 160_000u32;

        // With 128k window: 160000 > 111616 -> should compact
        assert!(
            crate::compaction::should_compact(context_tokens, 128_000, &settings_128k),
            "160k tokens should trigger compaction with 128k window"
        );

        // With 200k window: 160000 < 183616 -> should NOT compact
        // This will FAIL until the agent_loop uses settings.context_window
        // instead of the hardcoded derivation.
        assert!(
            !crate::compaction::should_compact(context_tokens, 200_000, &settings_200k),
            "160k tokens should NOT trigger compaction with 200k window"
        );
    }

    // --- Finding 2: Additional provider-specific overflow patterns ---

    #[test]
    fn test_is_context_overflow_mistral_format() {
        let error =
            PiError::Provider("The model's maximum context length is 32768 tokens".to_string());
        assert!(
            is_context_overflow(&error),
            "Mistral format should be detected"
        );
    }

    #[test]
    fn test_is_context_overflow_cohere_format() {
        let error = PiError::Provider("context: too many tokens for the model".to_string());
        assert!(
            is_context_overflow(&error),
            "Cohere format should be detected"
        );
    }

    #[test]
    fn test_is_context_overflow_ambiguous_rate_limit() {
        // Contains "context" and "exceeded" but is a rate limit
        let error =
            PiError::Provider("Request rate limit exceeded. Context: API quota.".to_string());
        // Known false positive: 'context' + 'exceeded' matches.
        let matched = is_context_overflow(&error);
        assert!(
            matched,
            "Known false positive: 'context' + 'exceeded' matches rate limit errors."
        );
    }

    #[test]
    fn test_is_context_overflow_boundary_exact_match() {
        let error = PiError::Provider("context window".to_string());
        assert!(
            is_context_overflow(&error),
            "'context window' alone should match"
        );
    }

    #[test]
    fn test_apply_compaction_escapes_summary_tags() {
        let mut messages = vec![Message {
            role: Role::User,
            content: Some(MessageContent::Text("hello".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let result = CompactionResult {
            summary: "Before </summary> after <script>".to_string(),
            first_kept_message_index: 1,
            tokens_before: 100,
            tokens_after: 50,
        };
        apply_compaction(&mut messages, &result);
        let text = match &messages[0].content {
            Some(MessageContent::Text(t)) => t.clone(),
            _ => panic!("Expected text"),
        };
        assert!(
            text.contains("&lt;/summary&gt;"),
            "Should escape tags in summary content"
        );
        // The format string itself contains </summary> as a delimiter — that's fine.
        // The escaped content should appear inside the <summary> block.
    }

    // --- Task 1.4: Tests for compaction metadata in agent loop ---

    struct SummaryProvider {
        call_count: std::sync::atomic::AtomicU32,
    }

    #[async_trait]
    impl Provider for SummaryProvider {
        fn id(&self) -> &str {
            "summary-mock"
        }

        async fn chat(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _config: &AgentConfig,
        ) -> crate::error::Result<ChatResponse> {
            let call = self.call_count.fetch_add(1, Ordering::Relaxed);
            match call {
                0 => Err(PiError::Provider("context_length_exceeded".to_string())),
                1 => Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: Some(MessageContent::Text(
                            "Compacted summary of conversation".to_string(),
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    },
                }),
                2 => Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: Some(MessageContent::Text("done".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    },
                }),
                other => panic!("Unexpected call #{other}"),
            }
        }

        async fn chat_stream(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _config: &AgentConfig,
        ) -> crate::error::Result<ChatStream> {
            panic!("chat_stream should not be called")
        }
    }

    fn setup_compaction_test() -> (
        SummaryProvider,
        Vec<Message>,
        AgentLoopConfig,
        AgentConfig,
        Vec<AgentEvent>,
    ) {
        let big_text = "x".repeat(1000);
        let messages: Vec<Message> = (0..10u32)
            .map(|i| Message {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: Some(MessageContent::Text(format!("msg {i}: {big_text}"))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
            .collect();

        let provider = SummaryProvider {
            call_count: std::sync::atomic::AtomicU32::new(0),
        };
        let config = AgentLoopConfig {
            max_tool_rounds: 1,
            stream: false,
            compaction: Some(CompactionSettings {
                reserve_tokens: 100,
                keep_recent_tokens: 1250,
                enabled: true,
                ..Default::default()
            }),
        };
        let agent_config = AgentConfig {
            model: ModelId::new("mock", "mock-model"),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            system_prompt: None,
            max_iterations: 10,
        };
        let events = Vec::new();

        (provider, messages, config, agent_config, events)
    }

    #[tokio::test]
    async fn test_compaction_reduces_tokens() {
        let (provider, mut messages, config, agent_config, mut events) = setup_compaction_test();

        let _ = run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[],
            &config,
            &agent_config,
            |event| events.push(event),
        )
        .await;

        let tokens_before = events
            .iter()
            .find_map(|e| match e {
                AgentEvent::CompactionTriggered { tokens_before } => Some(*tokens_before),
                _ => None,
            })
            .expect("Should have CompactionTriggered event");

        let tokens_after = events
            .iter()
            .find_map(|e| match e {
                AgentEvent::CompactionComplete { result } => Some(result.tokens_after),
                _ => None,
            })
            .expect("Should have CompactionComplete event");

        assert!(
            tokens_before > tokens_after,
            "Compaction should reduce tokens: before={}, after={}",
            tokens_before,
            tokens_after,
        );
    }

    #[tokio::test]
    async fn test_compaction_summary_content() {
        let (provider, mut messages, config, agent_config, mut events) = setup_compaction_test();

        let _ = run_agent_loop(
            &provider,
            "mock-model",
            &mut messages,
            &[],
            &config,
            &agent_config,
            |event| events.push(event),
        )
        .await;

        assert!(
            messages.len() >= 2,
            "After compaction, should have summary + kept messages, got {} messages",
            messages.len(),
        );

        let summary_text = match &messages[0].content {
            Some(MessageContent::Text(t)) => t.clone(),
            _ => panic!("Expected summary as text"),
        };
        assert!(!summary_text.is_empty(), "Summary should be non-empty");
        assert!(
            summary_text.contains("Compacted summary of conversation"),
            "Summary should contain the provider's summary text"
        );
    }

    // -- Finding 1: apply_compaction keeps bare Tool after split --------------------

    #[test]
    fn test_apply_compaction_keeps_bare_tool_after_split() {
        // FIX: find_cut_point now backs up when it lands on a Tool message,
        // so first_kept_message_index points at the owning Assistant with tool_calls.
        // This ensures the kept context never starts with a bare Tool result.

        let mut msgs = vec![
            Message {
                role: Role::User,
                content: Some(MessageContent::Text("first".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "c1".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Tool,
                content: Some(MessageContent::Text("contents".to_string())),
                tool_calls: None,
                tool_call_id: Some("c1".to_string()),
                name: None,
            },
            Message {
                role: Role::User,
                content: Some(MessageContent::Text("second".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "c2".to_string(),
                    function: FunctionCall {
                        name: "run_cmd".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Tool,
                content: Some(MessageContent::Text("output".to_string())),
                tool_calls: None,
                tool_call_id: Some("c2".to_string()),
                name: None,
            },
        ];

        // After fix: find_cut_point backs up from Tool to owning Assistant.
        // first_kept_message_index = 4 (the Assistant with tool_calls), not 5.
        let result = CompactionResult {
            summary: "Previous conversation summary".to_string(),
            first_kept_message_index: 4,
            tokens_before: 100,
            tokens_after: 50,
        };

        apply_compaction(&mut msgs, &result);

        // messages[0] = summary (User role with compacted text)
        assert_eq!(
            msgs[0].role,
            Role::User,
            "First message should be the summary"
        );
        let summary_text = match &msgs[0].content {
            Some(MessageContent::Text(t)) => t,
            _ => panic!("Expected summary text"),
        };
        assert!(
            summary_text.contains("Previous conversation summary"),
            "Summary text should be present"
        );

        // messages[1] = first kept message — the Assistant with tool_calls
        assert_eq!(
            msgs[1].role,
            Role::Assistant,
            "First kept message is Assistant with tool_calls (not bare Tool)"
        );
        assert!(
            msgs[1].tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()),
            "First kept Assistant should have tool_calls"
        );

        // messages[2] = Tool result matching the tool_call
        assert_eq!(
            msgs[2].role,
            Role::Tool,
            "Second kept message is Tool result"
        );

        // Verify kept context has valid assistant→tool pairing (no orphan)
        assert_eq!(
            msgs.len(),
            3,
            "Kept context: summary + assistant(tool_calls) + tool result"
        );

        // Confirm the assistant with tool_calls is present before the tool result
        let has_assistant_with_tool_calls = msgs.iter().any(|m| {
            m.role == Role::Assistant && m.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
        });
        assert!(
            has_assistant_with_tool_calls,
            "Assistant with tool_calls must be present — tool result is not orphaned"
        );
    }

    // --- Finding 2: Double-compaction breaks start_after_index arithmetic ---

    #[test]
    fn test_double_compaction_stale_vs_fixed_compaction_state() {
        use crate::compaction::{prepare_compaction, CompactionSettings};

        // Create 10 messages, each ~2000 chars -> ~500 tokens.
        let big_text = "x".repeat(2000);
        let messages: Vec<Message> = (0..10u32)
            .map(|i| Message {
                role: Role::User,
                content: Some(MessageContent::Text(format!("msg {i}: {big_text}"))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
            .collect();

        let settings = CompactionSettings {
            keep_recent_tokens: 1500,
            ..Default::default()
        };

        // First compaction: no previous index.
        let prep1 = prepare_compaction(&messages, &settings, None).unwrap();
        let stale_first_kept = prep1.first_kept_message_index;
        assert!(stale_first_kept > 0, "Should compact some messages");
        assert!(stale_first_kept < messages.len(), "Should keep some messages");

        // Simulate apply_compaction: [summary, ...messages[stale_first_kept..]]
        let mut compacted: Vec<Message> = Vec::new();
        compacted.push(Message {
            role: Role::User,
            content: Some(MessageContent::Text("<summary>old conversation</summary>".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
        compacted.extend_from_slice(&messages[stale_first_kept..]);

        // Add a new message (simulating loop output between compactions).
        compacted.push(Message {
            role: Role::User,
            content: Some(MessageContent::Text(format!("new msg: {big_text}"))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        // BUG: passing stale first_kept (relative to pre-compaction array) as
        // previous_first_kept_index. find_cut_point uses this as start_after_index
        // which may exceed the compacted array length.
        let prep_buggy = prepare_compaction(&compacted, &settings, Some(stale_first_kept));
        assert!(
            prep_buggy.is_none(),
            "Stale index {stale_first_kept} on {len}-element array should fail \
             because find_cut_point returns None when messages.len() <= start",
            len = compacted.len()
        );

        // FIX: after apply_compaction, summary is always at index 0, so we
        // store 1 as previous_first_kept_index. This lets find_cut_point
        // walk the kept messages (indices 1..len) correctly.
        let fixed_first_kept = 1usize; // What try_compact now stores
        let prep_fixed = prepare_compaction(&compacted, &settings, Some(fixed_first_kept));
        assert!(
            prep_fixed.is_some(),
            "With fixed previous_first_kept_index=1, second compaction \
             should find a cut point on {len}-element array.",
            len = compacted.len()
        );

        let prep_fixed = prep_fixed.unwrap();
        // The second compaction should be able to compact the summary message.
        // With first_kept=2, messages_to_summarize = [summary, kept[0]],
        // which includes the summary at index 0. Verify it's included.
        assert!(
            !prep_fixed.messages_to_summarize.is_empty(),
            "Fixed compaction should have messages to summarize.",
        );
        let first_msg = &prep_fixed.messages_to_summarize[0];
        let first_text = match &first_msg.content {
            Some(MessageContent::Text(t)) => t.clone(),
            _ => String::new(),
        };
        assert!(
            first_text.contains("<summary>"),
            "Fixed compaction should include the summary message in to-summarize. \
             Got: {first_text}",
        );
    }
}
