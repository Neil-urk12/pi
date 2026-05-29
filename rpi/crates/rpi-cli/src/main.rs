use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod agent;
mod config;
mod interactive;

use agent::AgentRunner;
use config::Config;
use rpi_cli::validate_tui_mode;

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

    /// Enable the experimental Rust TUI renderer in interactive mode
    #[arg(long)]
    tui: bool,

    /// Resume an existing session by ID
    #[arg(long)]
    resume: Option<String>,

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
    /// Session management
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
}

#[derive(Subcommand)]
enum SessionAction {
    /// List all sessions
    List,
    /// Delete a session by ID
    Delete { session_id: String },
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
    let interactive_mode = cli.prompt.is_none()
        && !cli.list_models
        && !matches!(cli.command, Some(Commands::Config { .. }))
        && !matches!(cli.command, Some(Commands::Session { .. }));
    validate_tui_mode(interactive_mode, cli.tui)?;

    // Initialize tracing
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // Load configuration
    let mut config = Config::load()?;

    match cli.command {
        Some(Commands::Config { action }) => match action {
            ConfigAction::Show => {
                println!("{}", serde_json::to_string_pretty(&config)?);
            }
            ConfigAction::Set { key, value } => {
                config = config.set(&key, &value)?;
                config.save()?;
                println!("Set {key} = {value}");
            }
        },
        Some(Commands::Session { action }) => {
            use rpi_core::SessionManager;
            let session_manager = SessionManager::new()?;
            match action {
                SessionAction::List => {
                    let sessions = session_manager.list()?;
                    if sessions.is_empty() {
                        println!("No sessions found.");
                    } else {
                        println!("Sessions:");
                        for s in &sessions {
                            println!(
                                "  {}  model={}  messages={}  created={}",
                                s.session_id, s.model, s.message_count, s.created_at
                            );
                            if let Some(preview) = &s.last_message_preview {
                                println!("    Preview: {preview}");
                            }
                        }
                    }
                }
                SessionAction::Delete { session_id } => {
                    session_manager.delete(&session_id)?;
                    println!("Deleted session: {session_id}");
                }
            }
        }
        _ => {
            // Default: run agent
            let model = cli.model.or(config.default_model.clone());
            let mut runner = if let Some(ref session_id) = cli.resume {
                use rpi_core::SessionManager;
                let session_manager = SessionManager::new()?;
                match session_manager.load(session_id) {
                    Ok(session) => AgentRunner::with_session(config, model, session)?,
                    Err(e) => anyhow::bail!("Session not found: {session_id}: {e}"),
                }
            } else {
                AgentRunner::new(config, model)?
            };

            if let Some(prompt) = cli.prompt {
                // Single prompt mode
                let response = runner.run_prompt(&prompt).await?;
                println!("{response}");
            } else if cli.list_models {
                // List models
                runner.list_models().await?;
            } else {
                // Interactive mode
                if cli.tui {
                    interactive::run_tui(&mut runner).await?;
                } else {
                    interactive::run(&mut runner).await?;
                }
            }
        }
    }

    Ok(())
}
