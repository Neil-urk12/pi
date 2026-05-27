use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod agent;
mod config;
mod interactive;

use agent::AgentRunner;
use config::Config;

#[derive(Parser)]
#[command(name = "rpi", about = "Pi coding agent - Rust port", version)]
struct Cli {
    /// Run a single prompt and exit
    #[arg(short, long)]
    prompt: Option<String>,

    /// Model to use (provider/model format)
    #[arg(short, long)]
    model: Option<String>,

    /// List available models
    #[arg(long)]
    list_models: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive mode (default)
    Chat,
    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set { key: String, value: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // Load configuration
    let config = Config::load()?;

    match cli.command {
        Some(Commands::Config { action }) => match action {
            ConfigAction::Show => {
                println!("{}", serde_json::to_string_pretty(&config)?);
            }
            ConfigAction::Set { key, value } => {
                config.set(&key, &value)?;
                println!("Set {key} = {value}");
            }
        },
        _ => {
            // Default: run agent
            let model = cli.model.or(config.default_model.clone());
            let mut runner = AgentRunner::new(config, model)?;

            if let Some(prompt) = cli.prompt {
                // Single prompt mode
                let response = runner.run_prompt(&prompt).await?;
                println!("{response}");
            } else if cli.list_models {
                // List models
                runner.list_models().await?;
            } else {
                // Interactive mode
                interactive::run(&mut runner).await?;
            }
        }
    }

    Ok(())
}
