use anyhow::Result;
use std::io::{self, BufRead, Write};

use crate::agent::AgentRunner;

pub async fn run(runner: &mut AgentRunner) -> Result<()> {
    println!("rpi - Pi Coding Agent (Rust)");
    println!("Type your message, /help for commands, or /quit to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" | "/q" => {
                    println!("Goodbye!");
                    break;
                }
                "/help" | "/h" => {
                    print_help();
                    continue;
                }
                "/clear" | "/c" => {
                    runner.clear_messages();
                    println!("Conversation cleared.");
                    continue;
                }
                "/config" => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(runner.config())?
                    );
                    continue;
                }
                "/model" => {
                    println!("Current model: {}", runner.model_display());
                    continue;
                }
                _ => {
                    println!("Unknown command: {input}. Type /help for available commands.");
                    continue;
                }
            }
        }

        // Process message
        match runner.run_prompt(input).await {
            Ok(response) => {
                println!("\n{response}\n");
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }
    }

    Ok(())
}

fn print_help() {
    println!("Commands:");
    println!("  /help, /h     - Show this help");
    println!("  /quit, /q     - Exit");
    println!("  /clear, /c    - Clear conversation");
    println!("  /config       - Show configuration");
    println!("  /model        - Show current model");
    println!();
    println!("Just type a message to chat with the AI.");
}
