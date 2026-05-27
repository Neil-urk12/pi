use anyhow::{Context, Result};
use rpi_core::*;
use rpi_ai::create_provider;
use rpi_tools::ToolRegistry;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::Config;

pub struct AgentRunner {
    config: Config,
    model: String,
    messages: Vec<Message>,
    system_prompt: Option<String>,
    tool_registry: ToolRegistry,
    working_dir: PathBuf,
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
            working_dir,
        })
    }

    pub async fn run_prompt(&mut self, prompt: &str) -> Result<String> {
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

        // Get tool definitions from registry
        let tool_defs: Vec<ToolDefinition> = self.tool_registry.definitions();

        // Run agent loop with streaming
        let mut final_text = String::new();
        let config = AgentLoopConfig {
            max_tool_rounds: 20,
            stream: true,
        };

        run_agent_loop(
            provider.as_ref(),
            &model_name,
            &mut self.messages,
            &self.tool_registry.tools(),
            &config,
            &agent_config,
            |event| match event {
                AgentEvent::TextDelta { text } => {
                    print!("{text}");
                    io::stdout().flush().ok();
                    final_text.push_str(&text);
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
                    if is_error {
                        eprintln!("[Error: {result}]");
                    } else {
                        // Show truncated result
                        let preview = if result.len() > 200 {
                            format!("{}...", &result[..200])
                        } else {
                            result.clone()
                        };
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

        let base_url = self.config.get_base_url(&model_id.provider).map(String::from);

        let provider_config = match model_id.provider.as_str() {
            "openai" | "azure" => ProviderConfig::OpenAi {
                api_key,
                base_url,
            },
            "anthropic" => ProviderConfig::Anthropic {
                api_key,
                base_url,
            },
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
                self.model.splitn(2, '/').next().unwrap_or("openai"),
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

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    pub fn model_display(&self) -> &str {
        &self.model
    }
}
