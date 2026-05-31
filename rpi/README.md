# rpi - Pi Coding Agent (Rust Port)

A Rust port of the [pi](https://github.com/earendil-works/pi-mono) coding agent. Provides a fast, memory-safe CLI for interacting with LLM providers with tool-use capabilities.

## Architecture

```
rpi/
├── Cargo.toml          # Workspace manifest
└── crates/
    ├── rpi-core/       # Core types, traits, agent loop
    ├── rpi-ai/         # LLM provider implementations
    ├── rpi-tools/      # Built-in tool implementations
    ├── rpi-tui/        # Experimental terminal UI renderer
    └── rpi-cli/        # CLI entry point
```

### Crate Dependency Graph

```
rpi-cli
  ├── rpi-core
  ├── rpi-ai
  │   └── rpi-core
  ├── rpi-tools
  │   └── rpi-core
  └── rpi-tui
```

### Crate Descriptions

| Crate | Purpose |
|-------|---------|
| **rpi-core** | Core types (`Message`, `ToolCall`, `ChatResponse`), traits (`Provider`, `Tool`, `Agent`), error types, and the agent loop implementation |
| **rpi-ai** | LLM provider implementations for OpenAI (and compatible APIs), Anthropic, and Ollama. Includes SSE streaming support |
| **rpi-tools** | Built-in tools: `read_file`, `write_file`, `edit_file`, `bash`, `grep`, `ls` |
| **rpi-tui** | Experimental terminal UI rendering primitives for interactive mode |
| **rpi-cli** | CLI binary with argument parsing, configuration management, and interactive mode |

### Workspace Boundaries

Phase 0 intentionally keeps the current five-crate workspace. New crates should
wait until at least two existing crates need the same stable abstraction. Likely
future extraction candidates are config/auth storage, JSONL session persistence,
and the extensions system.

## Features

- **Multi-provider LLM support**: OpenAI, Anthropic, Ollama (and any OpenAI-compatible API)
- **Streaming responses**: Real-time SSE streaming with delta events
- **Built-in tools**: File read/write/edit, bash execution, grep search, directory listing
- **Interactive and single-shot modes**: Chat interactively or run one-off prompts
- **Configuration management**: JSON config file with env var overrides
- **Extensible tool system**: Implement the `Tool` trait to add custom tools
- **Agent loop**: Automatic tool call execution with iteration limits
- **Experimental Rust TUI**: `rpi --tui` enables rich streaming markdown rendering in interactive mode

## Building

```bash
cd rpi
cargo build --release
```

The binary will be at `target/release/rpi`.

## Usage

### Single Prompt Mode

```bash
# Simple prompt
rpi -p "Explain how Rust lifetimes work"

# With specific model
rpi -m anthropic/claude-sonnet-4-20250514 -p "Write a hello world in Rust"

# Verbose output
rpi -v -p "What is 2+2?"
```

### Interactive Mode

```bash
# Start interactive session
rpi

# Start interactive session with experimental rich TUI rendering
rpi --tui

# With specific model
rpi -m openai/gpt-4o
```

Interactive commands:
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/config` - Show current configuration
- `/model` - Show current model
- `/quit` - Exit

### List Models

```bash
rpi --list-models
```

### Configuration

```bash
# Show current config
rpi config show

# Set default model
rpi config set default_model openai/gpt-4o

# Set API key
rpi config set api_key.openai sk-...

# Set custom base URL (for Azure, proxies, etc.)
rpi config set base_url.openai https://my-proxy.example.com/v1

# Set system prompt
rpi config set system_prompt "You are a Rust expert."

# Set max tokens
rpi config set max_tokens 8192

# Set temperature
rpi config set temperature 0.5
```

## Configuration

Config file location: `~/.config/rpi/config.json`

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OLLAMA_BASE_URL` | Ollama base URL (default: `http://localhost:11434`) |

### Config File Format

```json
{
  "default_model": "openai/gpt-4o",
  "api_keys": {
    "openai": "sk-...",
    "anthropic": "sk-ant-..."
  },
  "base_urls": {},
  "system_prompt": "You are a helpful coding assistant.",
  "max_tokens": 4096,
  "temperature": 0.7
}
```

## Development

### Adding a New Provider

1. Create `crates/rpi-ai/src/your_provider.rs`
2. Implement the `Provider` trait from `rpi-core`
3. Register in `crates/rpi-ai/src/provider.rs`
4. Add `ProviderConfig` variant in `rpi-core/src/types.rs`

### Adding a New Tool

1. Create `crates/rpi-tools/src/builtin/your_tool.rs`
2. Implement the `Tool` trait from `rpi-core`
3. Register in `crates/rpi-tools/src/builtin/mod.rs`

### Running Tests

```bash
cargo test
```

### Checking for Warnings

```bash
cargo clippy
```

## Comparison with TypeScript pi

| Feature | TypeScript pi | rpi (Rust) |
|---------|--------------|------------|
| Language | TypeScript/Node.js | Rust |
| Binary size | ~50MB (Node.js runtime) | ~30MB (static binary) |
| Startup time | ~200ms | ~10ms |
| Memory usage | ~100MB baseline | ~10MB baseline |
| TUI | Full differential rendering | Experimental `--tui` first slice |
| Extensions | TypeScript plugins | Not yet implemented |
| Providers | 9 providers | 3 providers (extensible) |
| Tools | 7 built-in tools | 6 built-in tools |
| Sessions | JSONL-based with branching | Not yet implemented |

## License

MIT
