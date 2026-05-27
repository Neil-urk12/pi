//! The agent loop — the core execution engine that orchestrates LLM calls and tool execution.

use crate::error::PiError;
use crate::traits::{ChatStream, Provider, StreamChunk, Tool, ToolCallDelta};
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
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Maximum number of tool-call rounds before forcing stop.
    pub max_tool_rounds: u32,
    /// Whether to stream responses.
    pub stream: bool,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 50,
            stream: true,
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
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let mut turn = 0u32;

    loop {
        // Build tool definitions
        let tool_defs: Vec<ToolDefinition> = tools.iter().map(|t| t.definition()).collect();

        on_event(AgentEvent::TurnStart { turn });

        // Call the provider
        let tool_calls = if config.stream {
            let mut stream = provider
                .chat_stream(model, messages, &tool_defs, agent_config)
                .await?;

            let mut current_text = String::new();
            let mut tool_call_deltas: HashMap<u32, crate::traits::ToolCallDelta> = HashMap::new();
            let mut finish_reason = None;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if let Some(text) = &chunk.delta {
                    current_text.push_str(text);
                    on_event(AgentEvent::TextDelta {
                        text: text.clone(),
                    });
                }
                for tc_delta in &chunk.tool_calls {
                    let entry = tool_call_deltas
                        .entry(tc_delta.index)
                        .or_insert_with(|| ToolCallDelta {
                            index: tc_delta.index,
                            id: None,
                            name: None,
                            arguments_delta: None,
                        });
                    if let Some(id) = &tc_delta.id {
                        entry.id = Some(id.clone());
                        // Emit ToolCallStart when we first see the ID
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
                        let existing = entry.arguments_delta.get_or_insert_with(String::new);
                        existing.push_str(args);
                        if let Some(id) = &entry.id {
                            on_event(AgentEvent::ToolCallDelta {
                                id: id.clone(),
                                arguments_delta: args.clone(),
                            });
                        }
                    }
                }
                if chunk.finish_reason.is_some() {
                    finish_reason = chunk.finish_reason;
                }
            }

            // Build assistant message
            let mut content_blocks = Vec::new();
            if !current_text.is_empty() {
                content_blocks.push(ContentBlock::Text {
                    text: current_text,
                });
            }

            // Build tool calls from deltas
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
            on_event(AgentEvent::TurnEnd {
                turn,
                message: assistant_msg,
                usage: None,
            });

            if has_tool_calls {
                Some(tool_calls)
            } else {
                None
            }
        } else {
            // Non-streaming
            let response = provider
                .chat(model, messages, &tool_defs, agent_config)
                .await?;

            total_usage.prompt_tokens += response.usage.prompt_tokens;
            total_usage.completion_tokens += response.usage.completion_tokens;
            total_usage.total_tokens += response.usage.total_tokens;

            let assistant_msg = response.message.clone();
            let tool_calls = assistant_msg.tool_calls.clone();
            messages.push(assistant_msg.clone());
            on_event(AgentEvent::TurnEnd {
                turn,
                message: assistant_msg,
                usage: Some(response.usage),
            });

            tool_calls
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

            // Add tool result message
            messages.push(Message {
                role: Role::Tool,
                content: Some(MessageContent::Text(result_text)),
                tool_calls: None,
                tool_call_id: Some(tool_call.id.clone()),
                name: Some(tool_call.function.name.clone()),
            });
        }
    }
}
