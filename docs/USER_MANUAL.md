# User Manual: crew-rs

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Commands Reference](#commands-reference)
6. [Working with Providers](#working-with-providers)
7. [Gateway Mode](#gateway-mode)
8. [Memory & Skills](#memory--skills)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

crew-rs is a Rust-native AI agent framework that operates in two modes:

- **Chat mode** (`crew chat`): Interactive multi-turn conversation with tools (or single-message via `--message`)
- **Gateway mode** (`crew gateway`): Persistent daemon serving multiple messaging channels

### Key Concepts

| Term | Description |
|------|-------------|
| **Agent** | AI that executes tasks using tools |
| **Tool** | A capability (shell, file ops, search, messaging) |
| **Provider** | LLM API service (Anthropic, OpenAI, etc.) |
| **Channel** | Messaging platform (CLI, Telegram, Slack, etc.) |
| **Session** | Conversation history per channel:chat_id |
| **Sandbox** | Isolated execution environment (bwrap, macOS sandbox-exec, Docker) |
| **Tool Policy** | Allow/deny rules controlling which tools are available |
| **Skill** | Reusable instruction template (SKILL.md) |
| **Bootstrap** | Context files loaded into system prompt (AGENTS.md, SOUL.md, etc.) |

---

## Installation

### Prerequisites

- Rust 1.85.0 or later
- An API key from at least one supported provider

### From Source

```bash
git clone https://github.com/heyong4725/crew-rs
cd crew-rs

# Basic (CLI, chat, run, gateway with CLI channel)
cargo install --path crates/crew-cli

# With messaging channels
cargo install --path crates/crew-cli --features telegram,discord,slack,whatsapp,feishu,email

# With browser automation (requires Chrome/Chromium)
cargo install --path crates/crew-cli --features browser

# Verify
crew --version
```

### Docker

```bash
docker compose --profile gateway up -d
```

### API Keys

```bash
# Anthropic (Claude) - recommended
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Or any other supported provider (see Providers section)
# Or use OAuth: crew auth login --provider openai
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

---

## Quick Start

```bash
# 1. Initialize workspace
cd your-project
crew init

# 2. Check setup
crew status

# 3. Start chatting
crew chat

# 4. Or send a single message
crew chat --message "Add a hello function to lib.rs"
```

---

## Configuration

### Config Files

Loaded in order (first found wins):
1. `.crew/config.json` (project-local)
2. `~/.config/crew/config.json` (global)

### Basic Config

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "api_key_env": "ANTHROPIC_API_KEY"
}
```

### Gateway Config

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "gateway": {
    "channels": [
      {"type": "cli"},
      {"type": "telegram", "allowed_senders": ["123456789"]},
      {"type": "discord", "settings": {"token_env": "DISCORD_BOT_TOKEN"}},
      {"type": "slack", "settings": {"bot_token_env": "SLACK_BOT_TOKEN", "app_token_env": "SLACK_APP_TOKEN"}},
      {"type": "whatsapp", "settings": {"bridge_url": "ws://localhost:3001"}},
      {"type": "feishu", "settings": {"app_id_env": "FEISHU_APP_ID", "app_secret_env": "FEISHU_APP_SECRET"}}
    ],
    "max_history": 50,
    "system_prompt": "You are a helpful assistant."
  }
}
```

### Environment Variable Expansion

Use `${VAR_NAME}` syntax:

```json
{
  "base_url": "${ANTHROPIC_BASE_URL}",
  "model": "${CREW_MODEL}"
}
```

---

## Commands Reference

### `crew chat`

Interactive multi-turn conversation with readline history.

```bash
crew chat [OPTIONS]

Options:
  -c, --cwd <PATH>         Working directory
      --config <PATH>      Config file path
      --provider <NAME>    LLM provider
      --model <NAME>       Model name
      --base-url <URL>     Custom API endpoint
  -m, --message <MSG>      Single message (non-interactive)
      --max-iterations <N> Max tool iterations per message (default: 20)
  -v, --verbose            Show tool outputs
      --no-retry           Disable retry
```

Features:
- Arrow keys, line editing (rustyline)
- Persistent history at `.crew/history/chat_history`
- Exit: `/exit`, `/quit`, `exit`, `quit`, `:q`, Ctrl+C, Ctrl+D
- Full tool access (shell, files, search, web)

```bash
crew chat                              # Interactive (default)
crew chat --provider deepseek          # Use DeepSeek
crew chat --model glm-4-plus           # Auto-detects Zhipu
crew chat --message "Fix auth bug"     # Single message, exit
```

---

### `crew gateway`

Run as a persistent multi-channel daemon.

```bash
crew gateway [OPTIONS]

Options:
  -c, --cwd <PATH>         Working directory
      --config <PATH>      Config file path
      --provider <NAME>    Override provider
      --model <NAME>       Override model
      --base-url <URL>     Override API endpoint
  -v, --verbose            Verbose logging
      --no-retry           Disable retry
```

Requires `gateway` section in config with `channels` array. Runs continuously until Ctrl+C.

---

### `crew init`

Initialize workspace with config and bootstrap files.

```bash
crew init [OPTIONS]

Options:
  -c, --cwd <PATH>    Working directory
      --defaults       Skip prompts, use defaults
```

Creates:
- `.crew/config.json` - Provider/model config
- `.crew/.gitignore` - Ignores state files
- `.crew/AGENTS.md` - Agent instructions template
- `.crew/SOUL.md` - Personality template
- `.crew/USER.md` - User info template
- `.crew/memory/` - Memory storage directory
- `.crew/sessions/` - Session history directory
- `.crew/skills/` - Custom skills directory

---

### `crew status`

Show system status.

```bash
crew status [OPTIONS]

Options:
  -c, --cwd <PATH>    Working directory
```

```
crew-rs Status
══════════════════════════════════════════════════

Config:    .crew/config.json (found)
Workspace: .crew/            (found)
Provider:  anthropic
Model:     claude-sonnet-4-20250514

API Keys
──────────────────────────────────────────────────
  Anthropic    ANTHROPIC_API_KEY         set
  OpenAI       OPENAI_API_KEY           not set
  ...

Bootstrap Files
──────────────────────────────────────────────────
  AGENTS.md        found
  SOUL.md          found
  USER.md          found
  TOOLS.md         missing
  IDENTITY.md      missing
```

---

### Other Commands

```bash
crew clean [--all] [--dry-run] # Clean database files
crew completions <shell>       # Generate completions (bash/zsh/fish/powershell)
crew docs                      # Generate tool + provider documentation
crew cron list [--all]         # List cron jobs
crew cron add [OPTIONS]        # Add a cron job
crew cron remove <job-id>      # Remove a cron job
crew cron enable <job-id>      # Enable/disable a cron job
crew channels status           # Show channel compile/config status
crew channels login            # WhatsApp QR code login
```

### `crew auth`

OAuth login and API key management.

```bash
crew auth login --provider openai           # PKCE browser OAuth
crew auth login --provider openai --device-code  # Device code flow
crew auth login --provider anthropic        # Paste-token (stdin)
crew auth logout --provider openai          # Remove stored credential
crew auth status                            # Show authenticated providers
```

Credentials are stored in `~/.crew/auth.json` (file mode 0600). The auth store is checked before environment variables when resolving API keys.

### `crew skills`

Manage skills.

```bash
crew skills list                            # List installed skills
crew skills install user/repo/skill-name    # Install from GitHub
crew skills remove skill-name               # Remove a skill
```

Fetches `SKILL.md` from the GitHub repo's main branch and installs to `.crew/skills/`.

---

## Working with Providers

### Supported Providers

| Provider | Env Var | Default Model | Notes |
|----------|---------|---------------|-------|
| anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 | Native API |
| openai | `OPENAI_API_KEY` | gpt-4o | Native API |
| gemini | `GEMINI_API_KEY` | gemini-2.0-flash | Native API |
| openrouter | `OPENROUTER_API_KEY` | anthropic/claude-sonnet-4-20250514 | Aggregator |
| deepseek | `DEEPSEEK_API_KEY` | deepseek-chat | OpenAI-compatible |
| groq | `GROQ_API_KEY` | llama-3.3-70b-versatile | OpenAI-compatible |
| moonshot | `MOONSHOT_API_KEY` | kimi-k2.5 | Also: `kimi` |
| dashscope | `DASHSCOPE_API_KEY` | qwen-max | Also: `qwen` |
| minimax | `MINIMAX_API_KEY` | MiniMax-Text-01 | OpenAI-compatible |
| zhipu | `ZHIPU_API_KEY` | glm-4-plus | Also: `glm` |
| ollama | (none) | llama3.2 | Local, no key needed |
| vllm | `VLLM_API_KEY` | (required) | Requires --base-url |

### Provider Auto-Detection

When `--provider` is omitted, crew-rs detects from model name:

```bash
crew chat --model gpt-4o           # -> openai
crew chat --model claude-sonnet-4-20250514  # -> anthropic
crew chat --model deepseek-chat    # -> deepseek
crew chat --model glm-4-plus       # -> zhipu
crew chat --model qwen-max         # -> dashscope
```

### Custom Endpoints

```bash
# Azure OpenAI
crew chat --provider openai --base-url "https://your.openai.azure.com/v1" --message "Task"

# Local Ollama
crew chat --provider ollama --model llama3.2

# vLLM server
crew chat --provider vllm --base-url "http://localhost:8000/v1" --model "meta-llama/Llama-3-70b" --message "Task"
```

---

## Gateway Mode

### Channel Setup

#### Telegram

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC..."
```

Config:
```json
{"type": "telegram", "allowed_senders": ["your_user_id"], "settings": {"token_env": "TELEGRAM_BOT_TOKEN"}}
```

#### Slack

Requires Socket Mode app with bot token + app-level token.

```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
```

Config:
```json
{"type": "slack", "settings": {"bot_token_env": "SLACK_BOT_TOKEN", "app_token_env": "SLACK_APP_TOKEN"}}
```

#### Discord

```bash
export DISCORD_BOT_TOKEN="..."
```

Config:
```json
{"type": "discord", "settings": {"token_env": "DISCORD_BOT_TOKEN"}}
```

#### WhatsApp

Requires Node.js bridge (Baileys) running at `ws://localhost:3001`.

Config:
```json
{"type": "whatsapp", "settings": {"bridge_url": "ws://localhost:3001"}}
```

#### Feishu/Lark

```bash
export FEISHU_APP_ID="cli_..."
export FEISHU_APP_SECRET="..."
```

Config:
```json
{"type": "feishu", "settings": {"app_id_env": "FEISHU_APP_ID", "app_secret_env": "FEISHU_APP_SECRET"}}
```

#### Email (IMAP/SMTP)

Requires IMAP for inbound and SMTP for outbound. Feature-gated behind `email`.

Config:
```json
{
  "type": "email",
  "allowed_senders": ["trusted@example.com"],
  "settings": {
    "imap_host": "imap.gmail.com",
    "imap_port": 993,
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 465,
    "username_env": "EMAIL_USERNAME",
    "password_env": "EMAIL_PASSWORD",
    "from_address": "bot@example.com",
    "poll_interval_secs": 30,
    "max_body_chars": 10000
  }
}
```

```bash
export EMAIL_USERNAME="bot@example.com"
export EMAIL_PASSWORD="app-specific-password"
```

### Voice Transcription

When `GROQ_API_KEY` is set, voice and audio messages from channels are automatically transcribed using Groq Whisper before being sent to the agent. The transcription is prepended as `[transcription: ...]`.

```bash
export GROQ_API_KEY="gsk_..."
```

### Access Control

Use `allowed_senders` to restrict access. Empty list = allow all.

```json
{"type": "telegram", "allowed_senders": ["123456", "789012"]}
```

### Cron Jobs

The agent can schedule recurring tasks via the `cron` tool:

- `add` - Create a scheduled job with `every_seconds`, `cron_expr`, or `at` timestamp
- `list` - Show all scheduled jobs
- `remove` - Delete a job by ID
- `enable` / `disable` - Toggle job active state

Cron expressions use standard syntax (e.g., `"0 0 9 * * * *"` for daily at 9am).

Cron jobs send messages through the bus and can deliver responses to any channel.

#### Cron CLI

Manage jobs directly from the command line (no running gateway needed):

```bash
crew cron list                          # List active jobs
crew cron list --all                    # Include disabled
crew cron add --name "report" --message "Generate daily report" --cron "0 0 9 * * * *"
crew cron add --name "check" --message "Check status" --every 3600
crew cron add --name "once" --message "Run migration" --at "2025-03-01T09:00:00Z"
crew cron remove <job-id>
crew cron enable <job-id>               # Enable
crew cron enable <job-id> --disable     # Disable
```

### Channel Status

Check configured channels and their compile/config status:

```bash
crew channels status
```

Shows a table with channel name, compile status (feature flags), and config summary (env vars set/missing).

### Heartbeat

The heartbeat service periodically reads `.crew/HEARTBEAT.md` and sends its content to the agent if non-empty. Default interval: 30 minutes. Use this for background task instructions.

---

## Memory & Skills

### Bootstrap Files

Loaded into the system prompt at startup:

| File | Purpose |
|------|---------|
| `.crew/AGENTS.md` | Agent instructions and guidelines |
| `.crew/SOUL.md` | Personality and values |
| `.crew/USER.md` | User information and preferences |
| `.crew/TOOLS.md` | Tool-specific guidance |
| `.crew/IDENTITY.md` | Custom identity definition |

Create via `crew init` (creates AGENTS, SOUL, USER templates).

### Memory System

- **Long-term**: `.crew/memory/MEMORY.md` - Persistent notes
- **Daily**: `.crew/memory/YYYY-MM-DD.md` - Auto-created daily logs
- **Recent**: Last 7 days of daily notes included in context

### Skills

Place custom skills in `.crew/skills/{name}/SKILL.md`:

```markdown
---
name: my-skill
description: My custom skill
always: true
requires_bins: curl
requires_env: MY_API_KEY
---

# Skill Instructions

Your skill content here...
```

Frontmatter fields: `name`, `description`, `always` (auto-load), `requires_bins` (comma-separated binaries checked via `which`), `requires_env` (comma-separated env vars). Availability is derived from requirement checks — not a frontmatter field.

Skills with `always: true` are included in every prompt. Others are available for the agent to read on demand.

### Built-in Skills

crew-rs bundles 6 skills at compile time. These are available without any setup:

| Skill | Description | Requirements |
|-------|-------------|-------------|
| cron | Cron tool usage examples | (none, always-on) |
| github | gh CLI patterns (PR, issue, API) | `gh` binary |
| skill-creator | How to create custom skills | (none) |
| summarize | URL/file summarization | `summarize` binary |
| tmux | tmux session automation | `tmux` binary |
| weather | Weather via wttr.in | `curl` binary |

Workspace skills in `.crew/skills/` override built-in skills with the same name.

---

### Files

```
.crew/
├── config.json          # Configuration (versioned, auto-migrated)
├── cron.json            # Cron job store
├── AGENTS.md            # Agent instructions
├── SOUL.md              # Personality
├── USER.md              # User info
├── HEARTBEAT.md         # Background tasks
├── sessions/            # Chat history (JSONL)
├── memory/              # Memory files
│   ├── MEMORY.md        # Long-term
│   └── 2025-02-10.md    # Daily
├── skills/              # Custom skills
├── episodes.redb        # Episodic memory DB
└── history/
    └── chat_history     # Readline history
```

---

## Advanced Usage

### Verbose Mode

```bash
crew chat -v                 # Shows tool execution details
```

### Tool Policies

Control which tools are available to the agent via `tools` in config:

```json
{
  "tools": {
    "allow": ["group:fs", "group:web", "shell"],
    "deny": ["spawn"]
  }
}
```

**Named groups** expand to tool sets:
- `group:fs` -> read_file, write_file, edit_file, diff_edit
- `group:runtime` -> shell
- `group:search` -> glob, grep, list_dir
- `group:web` -> web_search, web_fetch, browser
- `group:sessions` -> spawn

**Wildcard matching**: `exec*` matches `exec`, `exec_bg`, etc.

**Deny-wins semantics**: If a tool appears in both allow and deny, it is denied.

**Provider-specific policies**: Different tool sets per LLM model:

```json
{
  "tools": {
    "byProvider": {
      "openai/gpt-4o-mini": {
        "deny": ["shell", "write_file"]
      }
    }
  }
}
```

### Sandbox

Shell commands run inside a sandbox for isolation. Three backends are supported:

| Backend | Platform | Notes |
|---------|----------|-------|
| bwrap | Linux | Bubblewrap namespace isolation |
| macOS | macOS | sandbox-exec with SBPL profiles |
| Docker | Any | Container isolation with resource limits |

Configure in `config.json`:

```json
{
  "sandbox": {
    "enabled": true,
    "mode": "auto",
    "allow_network": false,
    "docker": {
      "image": "alpine:3.21",
      "mount_mode": "rw",
      "cpu_limit": "1.0",
      "memory_limit": "512m",
      "pids_limit": 100
    }
  }
}
```

**Modes**: `auto` (detect best available), `bwrap`, `macos`, `docker`, `none`.

**Mount modes**: `rw` (read-write), `ro` (read-only), `none` (no workspace mount).

**Environment sanitization**: 18 dangerous environment variables (LD_PRELOAD, NODE_OPTIONS, etc.) are automatically cleared in all sandbox backends.

### Session Forking

In gateway mode, send `/new` to create a branched conversation:

```
/new
```

This creates a new session that copies the last 10 messages from the current conversation. The child session has a `parent_key` reference to the original. Each fork gets a unique key namespaced by sender and timestamp.

### Config Hot-Reload

The gateway automatically detects config file changes:

- **Hot-reloaded** (no restart): system prompt, AGENTS.md, SOUL.md, USER.md
- **Restart required**: provider, model, API keys, gateway channels

Changes are detected via SHA-256 hashing with debounce.

### Message Coalescing

Long responses are automatically split into channel-safe chunks before sending:

| Channel | Max chars per message |
|---------|-----------------------|
| Telegram | 4000 |
| Discord | 1900 |
| Slack | 3900 |

Split preference: paragraph boundary > newline > sentence end > space > hard cut. Messages exceeding 50 chunks are truncated with a marker.

### Context Compaction

When the conversation exceeds the LLM's context window, older messages are automatically compacted:

- Tool arguments are stripped (replaced with `"[stripped]"`)
- Messages are summarized to first lines
- Recent tool call/result pairs are preserved intact
- The agent continues seamlessly without losing critical context

### Hooks (Lifecycle Events)

Run shell commands before/after tool calls and LLM calls. Configure in `config.json`:

```json
{
  "hooks": [
    {
      "event": "before_tool_call",
      "command": ["python3", "~/.crew/hooks/audit.py"],
      "timeout_ms": 3000,
      "tool_filter": ["shell", "write_file"]
    }
  ]
}
```

**Events**: `before_tool_call`, `after_tool_call`, `before_llm_call`, `after_llm_call`.

**Protocol**: Command receives JSON payload on stdin. Exit code 0 = allow, 1 = deny (before-hooks only), 2+ = error. Before-hooks can block operations; after-hooks are informational.

**Circuit breaker**: Hooks auto-disable after 3 consecutive failures. Commands use argv arrays (no shell interpretation). Environment sanitized via `BLOCKED_ENV_VARS`.

### Message Queue Modes

Control how messages arriving during an active agent run are handled:

```json
{
  "gateway": {
    "queue_mode": "followup"
  }
}
```

- **`followup`** (default): Process queued messages one at a time (FIFO)
- **`collect`**: Merge queued messages by session, concatenating content before processing

### Web UI (`crew serve`)

The REST API server (feature: `api`) includes an embedded web UI:

```bash
cargo install --path crates/crew-cli --features api
crew serve
# Open http://localhost:8080
```

Features: session sidebar, chat interface, SSE streaming, dark theme. A `/metrics` endpoint provides Prometheus-format metrics (`crew_tool_calls_total`, `crew_tool_call_duration_seconds`, `crew_llm_tokens_total`).

### Hybrid Memory Search

Memory search combines BM25 (keyword) and vector (semantic) scoring:

- **Ranking**: `alpha * vector_score + (1 - alpha) * bm25_score` (default alpha: 0.7)
- **Index**: HNSW via `hnsw_rs` with L2-normalized embeddings
- **Fallback**: BM25-only when no embedding provider is configured

Configure an embedding provider to enable vector search:

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small"
  }
}
```

---

## Troubleshooting

### API Key Not Set

```
Error: ANTHROPIC_API_KEY environment variable not set
```

Fix: `export ANTHROPIC_API_KEY="your-key"` or check with `crew status`.

### Rate Limited (429)

Retry mechanism handles this automatically (3 attempts with backoff). If persistent, try a different provider or wait.

### Debug Logging

```bash
RUST_LOG=debug crew chat
RUST_LOG=crew_agent=trace crew chat --message "task"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `GROQ_API_KEY` | Groq API key |
| `MOONSHOT_API_KEY` | Moonshot API key |
| `DASHSCOPE_API_KEY` | DashScope API key |
| `MINIMAX_API_KEY` | MiniMax API key |
| `ZHIPU_API_KEY` | Zhipu API key |
| `RUST_LOG` | Log level (error/warn/info/debug/trace) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `SLACK_BOT_TOKEN` | Slack bot token |
| `SLACK_APP_TOKEN` | Slack app-level token |
| `FEISHU_APP_ID` | Feishu app ID |
| `FEISHU_APP_SECRET` | Feishu app secret |
| `EMAIL_USERNAME` | Email account username |
| `EMAIL_PASSWORD` | Email account password |
