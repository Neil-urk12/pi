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
    messages: Vec<Message>,
    system_prompt: Option<String>,
    tool_registry: ToolRegistry,
}

impl AgentRunner {
    pub fn new(config: Config, model: Option<String>) -> Result<Self> {
        let model = model.unwrap_or_else(|| "openai/gpt-4o".to_string());
        let system_prompt = config.system_prompt.clone();
        let working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let tool_registry = ToolRegistry::with_defaults_in(working_dir.clone());

        Ok(Self {
            config,
            model,
            messages: Vec::new(),
            system_prompt,
            tool_registry,
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

    async fn run_prompt_with_observer<F>(
        &mut self,
        prompt: &str,
        mut observer: F,
    ) -> Result<String>
    where
        F: FnMut(&AgentEvent),
    {
        // Add user message
        self.messages.push(Message {
            role: Role::User,
            content: Some(MessageContent::Text(prompt.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        // Build provider
        let (provider, model_name) = self.build_provider()?;
        let agent_config = self.build_agent_config(&model_name);

        // Run agent loop with streaming
        let mut final_text = String::new();
        let config = AgentLoopConfig {
            max_tool_rounds: 20,
            stream: true,
            compaction: None,
        };

        run_agent_loop(
            provider.as_ref(),
            &model_name,
            &mut self.messages,
            &self.tool_registry.tools(),
            &config,
            &agent_config,
            |event| {
                observer(&event);
                if let AgentEvent::TextDelta { text } = event {
                    final_text.push_str(&text);
                }
            },
        )
        .await
        .context("Agent loop failed")?;

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
            _ => anyhow::bail!("Unknown provider: {}", model_id.provider),
        };

        let provider = create_provider(&provider_config)?;
        Ok((provider, model_id.model))
    }

    fn build_agent_config(&self, model_name: &str) -> AgentConfig {
        AgentConfig {
            model: ModelId::new(
                self.model.split('/').next().unwrap_or("openai"),
                model_name,
            ),
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
        self.messages.clear();
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
