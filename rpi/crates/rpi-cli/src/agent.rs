use anyhow::{Context, Result};
use rpi_ai::create_provider;
use rpi_core::*;
use rpi_tools::ToolRegistry;
use rpi_tui::{DefaultTheme, Frame, TerminalBackend, TuiEvent, TurnView};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::config::Config;
use rpi_cli::safe_tool_summary;

pub struct AgentRunner {
    config: Config,
    model: String,
    session: Session,
    session_manager: SessionManager,
    context: Vec<ContextMessage>,
    system_prompt: Option<String>,
    tool_registry: ToolRegistry,
    compaction_settings: Option<CompactionSettings>,
}

impl AgentRunner {
    pub fn new(config: Config, model: Option<String>) -> Result<Self> {
        let model = model.unwrap_or_else(|| "openai/gpt-4o".to_string());
        let system_prompt = config.system_prompt.clone();
        let working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let tool_registry = ToolRegistry::with_defaults_in(working_dir.clone());
        let session_manager = SessionManager::new()?;
        let session = Session::new(model.clone());
        let compaction_settings = config.compaction.as_ref().map(|c| c.to_core_settings());

        Ok(Self {
            config,
            model,
            session,
            session_manager,
            context: Vec::new(),
            system_prompt,
            tool_registry,
            compaction_settings,
        })
    }

    pub fn with_session(config: Config, model: Option<String>, session: Session) -> Result<Self> {
        let model = model.unwrap_or_else(|| session.model.clone());
        let system_prompt = config.system_prompt.clone();
        let working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let tool_registry = ToolRegistry::with_defaults_in(working_dir.clone());
        let session_manager = SessionManager::new()?;
        let compaction_settings = config.compaction.as_ref().map(|c| c.to_core_settings());

        let context = session.build_context_with_entry_ids();

        Ok(Self {
            config,
            model,
            session,
            session_manager,
            context,
            system_prompt,
            tool_registry,
            compaction_settings,
        })
    }

    pub async fn run_prompt(&mut self, prompt: &str) -> Result<String> {
        self.run_prompt_with_observer(prompt, |event| match event {
            AgentEvent::TextDelta { text } => {
                print!("{text}");
                let _ = io::stdout().flush();
            }
            AgentEvent::TurnStart { .. } => {}
            AgentEvent::TurnEnd { .. } => {
                println!();
            }
            AgentEvent::ToolExecutionStart { name, .. } => {
                eprintln!("\n[Tool: {name}]");
            }
            AgentEvent::ToolExecutionEnd {
                result, is_error, ..
            } => {
                if *is_error {
                    eprintln!("[Error: {result}]");
                } else {
                    let preview = truncate_chars(result, 200);
                    eprintln!("[Result: {preview}]");
                }
            }
            AgentEvent::Done { turns, total_usage } => {
                eprintln!(
                    "\n[Completed in {turns} turns, {} tokens]",
                    total_usage.total_tokens
                );
            }
            AgentEvent::Error { error } => {
                eprintln!("\n[Error: {error}]");
            }
            _ => {}
        })
        .await
    }

    pub async fn run_prompt_tui(&mut self, prompt: &str) -> Result<String> {
        let width = TerminalBackend::terminal_width().unwrap_or(80);
        let mut view = TurnView::new(DefaultTheme::default(), width);
        let mut previous_frame = Frame::default();
        let mut tool_summaries: HashMap<String, (String, Option<String>)> = HashMap::new();
        let color = TerminalBackend::should_color();

        self.run_prompt_with_observer(prompt, |event| {
            let tui_event = match event {
                AgentEvent::TextDelta { text } => Some(TuiEvent::AssistantDelta(text.clone())),
                AgentEvent::ToolExecutionStart {
                    id,
                    name,
                    arguments,
                } => {
                    let summary = safe_tool_summary(name, arguments);
                    tool_summaries.insert(id.clone(), (name.clone(), summary.clone()));
                    Some(TuiEvent::ToolStarted {
                        name: name.clone(),
                        summary,
                    })
                }
                AgentEvent::ToolExecutionEnd {
                    id, name, is_error, ..
                } => {
                    let summary = tool_summaries.remove(id).and_then(|(_, summary)| summary);
                    Some(TuiEvent::ToolFinished {
                        name: name.clone(),
                        summary,
                        is_error: *is_error,
                    })
                }
                AgentEvent::TurnEnd { .. } | AgentEvent::Done { .. } => {
                    Some(TuiEvent::TurnFinished)
                }
                AgentEvent::Error { error } => Some(TuiEvent::RendererWarning(error.clone())),
                _ => None,
            };

            if let Some(tui_event) = tui_event {
                let decision = view.apply_event(tui_event);
                if decision.should_render {
                    let next_frame = view.render_frame();
                    let encoded = TerminalBackend::encode_active_region_update(
                        &previous_frame,
                        &next_frame,
                        color,
                    );
                    print!("{encoded}");
                    let _ = io::stdout().flush();
                    previous_frame = next_frame;
                }
            }
        })
        .await
    }

    async fn run_prompt_with_observer<F>(&mut self, prompt: &str, mut observer: F) -> Result<String>
    where
        F: FnMut(&AgentEvent),
    {
        // Persist user message to session
        let user_message = Message {
            role: Role::User,
            content: Some(MessageContent::Text(prompt.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let entry_id = self.session.append_message(user_message.clone());
        self.session_manager.save(&self.session)?;

        // Incremental context update — push the new message instead of
        // rebuilding the entire context from scratch (Finding 7).
        self.context.push(ContextMessage {
            message: user_message,
            entry_id: Some(entry_id),
        });

        // Build provider
        let (provider, model_name) = self.build_provider()?;
        let agent_config = self.build_agent_config(&model_name);

        // Run agent loop with streaming
        let mut final_text = String::new();
        // Save entry_ids before draining context (needed for compaction
        // first_kept_entry_id lookup). Clone is cheap (UUIDs only).
        let context_entry_ids: Vec<Option<String>> =
            self.context.iter().map(|cm| cm.entry_id.clone()).collect();
        // Drain messages from context — takes ownership without cloning
        // Message content (Finding 8).
        let mut messages: Vec<Message> =
            self.context.drain(..).map(|cm| cm.message).collect();
        let messages_before = messages.len();
        let mut compaction_results: Vec<CompactionResult> = Vec::new();

        let config = AgentLoopConfig {
            max_tool_rounds: 20,
            stream: true,
            compaction: self.compaction_settings.clone(),
        };

        match run_agent_loop(
            provider.as_ref(),
            &model_name,
            &mut messages,
            &self.tool_registry.tools(),
            &config,
            &agent_config,
            |event| {
                observer(&event);
                if let AgentEvent::TextDelta { text } = &event {
                    final_text.push_str(text);
                }
                if let AgentEvent::CompactionComplete { result } = &event {
                    compaction_results.push(result.clone());
                }
            },
        )
        .await
        {
            Ok(()) => {
                // Persist turn results to session
                if compaction_results.is_empty() {
                    // No compaction — incremental context update (common path).
                    // Drain new messages to avoid extra clones.
                    let new_messages: Vec<Message> = messages.drain(messages_before..).collect();
                    for msg in new_messages {
                        let entry_id = self.session.append_message(msg.clone());
                        self.context.push(ContextMessage {
                            message: msg,
                            entry_id: Some(entry_id),
                        });
                    }
                } else {
                    // Compaction(s) happened — persist all compaction entries and new messages.
                    // Each CompactionResult.first_kept_message_index is relative to the
                    // messages array at the time that compaction ran.  After the first
                    // compaction the array is reshuffled to [summary, kept…], so for
                    // compaction i > 0 the offset into the original context_entry_ids
                    // is: first_kept_0 + Σ(first_kept_j − 1) for j in 1..=i.
                    let mut cumulative_first_kept = 0usize;
                    for (i, result) in compaction_results.iter().enumerate() {
                        if i == 0 {
                            cumulative_first_kept = result.first_kept_message_index;
                        } else {
                            cumulative_first_kept = cumulative_first_kept
                                .checked_add(
                                    result.first_kept_message_index.checked_sub(1)
                                        .ok_or_else(|| anyhow::anyhow!(
                                            "first_kept_message_index is 0 for compaction {i}"
                                        ))?
                                )
                                .ok_or_else(|| anyhow::anyhow!(
                                    "cumulative_first_kept overflow at compaction {i}"
                                ))?;
                        }

                        let first_kept_entry_id = context_entry_ids
                            .get(cumulative_first_kept)
                            .and_then(|eid| eid.as_deref())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "first_kept_entry_id missing for index {} \
                                     (context_entry_ids len: {})",
                                    cumulative_first_kept,
                                    context_entry_ids.len()
                                )
                            })?
                            .to_string();

                        self.session.append_compaction(
                            result.summary.clone(),
                            first_kept_entry_id,
                            result.tokens_before,
                        );
                    }

                    // After all compactions the final messages array is
                    // [summary_last, original_kept…, new_from_loop…].
                    // original_kept count = messages_before − cumulative_first_kept.
                    let num_original_kept = messages_before
                        .checked_sub(cumulative_first_kept)
                        .ok_or_else(|| anyhow::anyhow!(
                            "cumulative_first_kept ({}) > messages_before ({})",
                            cumulative_first_kept, messages_before
                        ))?;
                    let new_messages_start = 1 + num_original_kept; // skip summary + original kept
                    for msg in messages.get(new_messages_start..).unwrap_or(&[]) {
                        self.session.append_message(msg.clone());
                    }

                    // Compaction reshuffles context structure — must rebuild.
                    self.context = self.session.build_context_with_entry_ids();
                }
                self.session_manager.save(&self.session)?;
            }
            Err(e) => {
                // Rebuild context from session — preserves persisted messages
                self.context = self.session.build_context_with_entry_ids();
                return Err(e).context("Agent loop failed");
            }
        }
        Ok(final_text)
    }

    pub async fn list_models(&self) -> Result<()> {
        println!("Available models:");
        println!("  openai/gpt-4o");
        println!("  openai/gpt-4o-mini");
        println!("  openai/gpt-4-turbo");
        println!("  openai/o1-preview");
        println!("  openai/o1-mini");
        println!("  anthropic/claude-sonnet-4-20250514");
        println!("  anthropic/claude-3-5-sonnet-20241022");
        println!("  anthropic/claude-3-opus-20240229");
        println!("\nUse --model provider/model to select.");
        Ok(())
    }

    fn build_provider(&self) -> Result<(Box<dyn Provider>, String)> {
        let model_id = ModelId::parse(&self.model)
            .context("Model must be in provider/model format (e.g. openai/gpt-4o)")?;

        let api_key = self
            .config
            .get_api_key(&model_id.provider)
            .map(String::from)
            .or_else(|| std::env::var(format!("{}_API_KEY", model_id.provider.to_uppercase())).ok())
            .with_context(|| {
                format!(
                    "No API key for provider '{}'. Set via config or {}_API_KEY env var.",
                    model_id.provider,
                    model_id.provider.to_uppercase()
                )
            })?;

        let base_url = self
            .config
            .get_base_url(&model_id.provider)
            .map(String::from);

        let provider_config = match model_id.provider.as_str() {
            "openai" | "azure" => ProviderConfig::OpenAi { api_key, base_url },
            "anthropic" => ProviderConfig::Anthropic { api_key, base_url },
            "ollama" => ProviderConfig::Ollama {
                base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            },
            "gitlawb" => ProviderConfig::Gitlawb { api_key, base_url },
            _ => anyhow::bail!("Unknown provider: {}", model_id.provider),
        };

        let provider = create_provider(&provider_config)?;
        Ok((provider, model_id.model))
    }

    fn build_agent_config(&self, model_name: &str) -> AgentConfig {
        AgentConfig {
            model: ModelId::new(self.model.split('/').next().unwrap_or("openai"), model_name),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature.map(|t| t as f64),
            system_prompt: self.system_prompt.clone(),
            max_iterations: 20,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn clear_messages(&mut self) {
        self.context.clear();
        self.session = Session::new(self.model.clone());
    }

    pub fn model_display(&self) -> &str {
        &self.model
    }

}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            output.push_str("...");
            return output;
        }
        output.push(ch);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpi_core::{CompactionResult, Message, MessageContent, Role, Session, SessionEntry, SessionManager};
    use tempfile::tempdir;

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

    fn persist_compaction(
        runner: &mut AgentRunner,
        result: &CompactionResult,
    ) -> anyhow::Result<()> {
        let first_kept_entry_id = runner
            .context
            .get(result.first_kept_message_index)
            .and_then(|cm| cm.entry_id.as_deref())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "first_kept_entry_id missing for index {} (context len: {})",
                    result.first_kept_message_index,
                    runner.context.len()
                )
            })?
            .to_string();
        runner.session.append_compaction(
            result.summary.clone(),
            first_kept_entry_id,
            result.tokens_before,
        );
        // Persist new messages (those with no entry_id after first_kept).
        for cm in &runner.context[result.first_kept_message_index..] {
            if cm.entry_id.is_none() {
                runner.session.append_message(cm.message.clone());
            }
        }
        runner.context = runner.session.build_context_with_entry_ids();
        Ok(())
    }

    /// Mirror of the multi-compaction persistence logic in `run_prompt_with_observer`.
    /// Takes pre-loop `context_entry_ids` and post-loop `runner.context`.
    fn persist_compaction_results(
        runner: &mut AgentRunner,
        compaction_results: &[CompactionResult],
        context_entry_ids: &[Option<String>],
        messages_before: usize,
    ) -> anyhow::Result<()> {
        let mut cumulative_first_kept = 0usize;
        for (i, result) in compaction_results.iter().enumerate() {
            if i == 0 {
                cumulative_first_kept = result.first_kept_message_index;
            } else {
                cumulative_first_kept += result.first_kept_message_index.checked_sub(1)
                    .ok_or_else(|| anyhow::anyhow!(
                        "first_kept_message_index is 0 for compaction {i}"
                    ))?;
            }

            let first_kept_entry_id = context_entry_ids
                .get(cumulative_first_kept)
                .and_then(|eid| eid.as_deref())
                .ok_or_else(|| anyhow::anyhow!(
                    "first_kept_entry_id missing for index {} \
                     (context_entry_ids len: {})",
                    cumulative_first_kept,
                    context_entry_ids.len()
                ))?
                .to_string();

            runner.session.append_compaction(
                result.summary.clone(),
                first_kept_entry_id,
                result.tokens_before,
            );
        }

        let num_original_kept = messages_before
            .checked_sub(cumulative_first_kept)
            .ok_or_else(|| anyhow::anyhow!(
                "cumulative_first_kept ({}) > messages_before ({})",
                cumulative_first_kept, messages_before
            ))?;
        let new_messages_start = 1 + num_original_kept;
        for cm in runner.context.get(new_messages_start..).unwrap_or(&[]) {
            if cm.entry_id.is_none() {
                runner.session.append_message(cm.message.clone());
            }
        }

        runner.context = runner.session.build_context_with_entry_ids();
        Ok(())
    }

    /// Build an AgentRunner with a pre-populated session so we can test
    /// compaction persistence without a real provider.
    fn build_runner(msg_count: usize) -> (AgentRunner, Vec<String>) {
        let dir = tempdir().unwrap();
        let session_manager = SessionManager::with_dir(dir.path().to_path_buf());
        let mut session = Session::new("test-model");

        let entry_ids: Vec<String> = (0..msg_count)
            .map(|i| session.append_message(user_msg(&format!("msg {i}"))))
            .collect();

        let context = session.build_context_with_entry_ids();

        let runner = AgentRunner {
            config: Config::default(),
            model: "test-model".to_string(),
            session,
            session_manager,
            context,
            system_prompt: None,
            tool_registry: ToolRegistry::with_defaults_in(PathBuf::from(".")),
            compaction_settings: None,
        };
        (runner, entry_ids)
    }

    #[test]
    fn compaction_persists_all_kept_messages() {
        // After compaction, self.messages = [summary, kept..., new_from_loop...].
        // Kept messages are already in session via old branch (compaction parent
        // points to old tip). Only NEW messages should be appended.
        let (mut runner, _entry_ids) = build_runner(6);
        let _messages_before = runner.context.len(); // 6
        let initial_entry_count = runner.session.entries.len(); // meta + leaf + 6*(msg+leaf) = 14

        // Simulate post-compaction state: 3 kept + 2 new.
        runner.context = vec![
            ContextMessage {
                message: user_msg("<summary>compacted</summary>"),
                entry_id: None,
            },
            ContextMessage {
                message: assistant_msg("kept 0"),
                entry_id: Some("fake-0".to_string()),
            },
            ContextMessage {
                message: assistant_msg("kept 1"),
                entry_id: Some("fake-1".to_string()),
            },
            ContextMessage {
                message: assistant_msg("kept 2"),
                entry_id: Some("fake-2".to_string()),
            },
            ContextMessage {
                message: user_msg("new from loop 1"),
                entry_id: None,
            },
            ContextMessage {
                message: assistant_msg("new from loop 2"),
                entry_id: None,
            },
        ];

        let result = CompactionResult {
            summary: "compacted".to_string(),
            first_kept_message_index: 3, // kept = original[3..6]
            tokens_before: 100,
            tokens_after: 50,
        };

        persist_compaction(&mut runner, &result).unwrap();

        // Verify new messages ARE persisted.
        let context = runner.session.build_context();
        let texts: Vec<String> = context
            .iter()
            .filter_map(|m| match &m.content {
                Some(MessageContent::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect();

        assert!(
            texts.iter().any(|t| t.contains("new from loop 1")),
            "new message 1 missing from session after compaction"
        );
        assert!(
            texts.iter().any(|t| t.contains("new from loop 2")),
            "new message 2 missing from session after compaction"
        );

        // Verify kept messages are NOT duplicated.
        // Expected: +2 entries for compaction+leaf, +4 for 2 new messages = 20 total.
        // If kept were re-appended: +6 for 3 kept = 26 total (wrong).
        let expected_entries = initial_entry_count + 2 + 4; // compaction + 2 new msgs
        assert_eq!(
            runner.session.entries.len(),
            expected_entries,
            "kept messages were duplicated in session"
        );
    }

    #[test]
    fn compaction_errors_on_missing_entry_id() {
        // H1: if the entry ID lookup yields None, return an error
        // instead of panicking.
        let (mut runner, _entry_ids) = build_runner(2);
        let _messages_before = runner.context.len(); // 2

        // Build context with a valid first_kept_message_index but no entry_id at that index.
        runner.context = vec![
            ContextMessage {
                message: user_msg("<summary>compacted</summary>"),
                entry_id: None,
            },
            ContextMessage {
                message: assistant_msg("kept 0"),
                entry_id: None,
            }, // entry_id is None!
        ];

        let result = CompactionResult {
            summary: "compacted".to_string(),
            first_kept_message_index: 1, // Valid for messages slice, OOB for context_entry_ids
            tokens_before: 100,
            tokens_after: 50,
        };

        let err = persist_compaction(&mut runner, &result).unwrap_err();
        assert!(
            err.to_string().contains("first_kept_entry_id missing"),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string().contains("context len: 2"),
            "error should include context length: {err}"
        );
    }

    #[test]
    fn multiple_compactions_all_persisted() {
        // Two compactions in one loop run. Both entries must be persisted to the
        // session, and all new messages from the loop must appear.
        //
        // Original messages: [m0, m1, m2, m3, m4, m5]
        // Compaction 0: first_kept=3 → keeps [m3, m4, m5]
        // Loop adds m6, m7
        // Compaction 1: first_kept=2 (relative to [s0, m3, m4, m5, m6, m7])
        //             → keeps [m4, m5, m6, m7]
        // Loop adds m8
        // Final messages: [summary1, m4, m5, m6, m7, m8]

        let (mut runner, entry_ids) = build_runner(6);

        // Capture context_entry_ids BEFORE mutating runner.context (mirrors production).
        let context_entry_ids: Vec<Option<String>> =
            entry_ids.iter().map(|eid| Some(eid.clone())).collect();

        // Set runner.context to the final post-loop state.
        runner.context = vec![
            ContextMessage {
                message: user_msg("<summary>summary1</summary>"),
                entry_id: None,
            },
            ContextMessage {
                message: assistant_msg("kept m4"),
                entry_id: Some(entry_ids[4].clone()),
            },
            ContextMessage {
                message: assistant_msg("kept m5"),
                entry_id: Some(entry_ids[5].clone()),
            },
            ContextMessage {
                message: user_msg("new m6"),
                entry_id: None,
            },
            ContextMessage {
                message: assistant_msg("new m7"),
                entry_id: None,
            },
            ContextMessage {
                message: user_msg("new m8"),
                entry_id: None,
            },
        ];

        let compaction_results = vec![
            CompactionResult {
                summary: "summary0".to_string(),
                first_kept_message_index: 3,
                tokens_before: 100,
                tokens_after: 50,
            },
            CompactionResult {
                summary: "summary1".to_string(),
                first_kept_message_index: 2,
                tokens_before: 80,
                tokens_after: 40,
            },
        ];

        persist_compaction_results(
            &mut runner,
            &compaction_results,
            &context_entry_ids,
            6,
        )
        .unwrap();

        // Both compaction entries must exist in the session.
        let compaction_entries: Vec<_> = runner
            .session
            .entries
            .iter()
            .filter(|e| matches!(e, SessionEntry::Compaction { .. }))
            .collect();
        assert_eq!(
            compaction_entries.len(),
            2,
            "expected 2 compaction entries, found {}",
            compaction_entries.len()
        );

        // Verify first compaction points to m3 (entry_ids[3]).
        if let SessionEntry::Compaction {
            first_kept_entry_id,
            ..
        } = &compaction_entries[0]
        {
            assert_eq!(first_kept_entry_id, &entry_ids[3]);
        } else {
            unreachable!();
        }

        // Verify second compaction points to m4 (entry_ids[4]).
        if let SessionEntry::Compaction {
            first_kept_entry_id,
            ..
        } = &compaction_entries[1]
        {
            assert_eq!(first_kept_entry_id, &entry_ids[4]);
        } else {
            unreachable!();
        }

        // New messages (m6, m7, m8) must appear in the built context.
        let context = runner.session.build_context();
        let texts: Vec<String> = context
            .iter()
            .filter_map(|m| match &m.content {
                Some(MessageContent::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect();

        for label in ["new m6", "new m7", "new m8"] {
            assert!(
                texts.iter().any(|t| t.contains(label)),
                "{label} missing from session after multi-compaction"
            );
        }
    }
}
