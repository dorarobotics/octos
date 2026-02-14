# Architecture Document: crew-rs

## Overview

crew-rs is a 6-crate Rust workspace (Edition 2024, rust-version 1.85.0) providing both a coding agent CLI and a multi-channel messaging gateway. Pure Rust TLS via rustls (no OpenSSL). Error handling via `eyre`/`color-eyre`.

```
┌─────────────────────────────────────────────────────────────┐
│                        crew-cli                             │
│           (CLI: chat, gateway, init, status)                │
├──────────────────────────┬──────────────────────────────────┤
│       crew-agent         │           crew-bus               │
│  (Agent, Tools, Skills)  │  (Channels, Sessions, Cron)     │
├──────────┬───────────────┴──────────────────────────────────┤
│crew-memory│           crew-llm                              │
│(Episodes) │      (LLM Providers)                            │
├──────────┴──────────────────────────────────────────────────┤
│                       crew-core                             │
│            (Types, Messages, Gateway Protocol)              │
└─────────────────────────────────────────────────────────────┘
```

---

## crew-core — Foundation Types

Shared types with no internal dependencies. Only depends on serde, chrono, uuid, eyre.

### Task Model

```rust
pub struct Task {
    pub id: TaskId,                   // UUID v7 (temporal ordering)
    pub parent_id: Option<TaskId>,    // For subtasks
    pub status: TaskStatus,
    pub kind: TaskKind,
    pub context: TaskContext,
    pub result: Option<TaskResult>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

**TaskId**: Newtype over `Uuid`. Generates UUID v7 via `Uuid::now_v7()`. Implements Display, FromStr, Default.

**TaskStatus** (tagged enum, `"state"` discriminant):
- `Pending` — awaiting assignment
- `InProgress { agent_id: AgentId }` — executing
- `Blocked { reason: String }` — waiting for dependency
- `Completed` — success
- `Failed { error: String }` — failure with message

**TaskKind** (tagged enum, `"type"` discriminant):
- `Plan { goal: String }`
- `Code { instruction: String, files: Vec<PathBuf> }`
- `Review { diff: String }`
- `Test { command: String }`
- `Custom { name: String, params: serde_json::Value }`

**TaskContext**:
- `working_dir: PathBuf`, `git_state: Option<GitState>`, `working_memory: Vec<Message>`, `episodic_refs: Vec<EpisodeRef>`, `files_in_scope: Vec<PathBuf>`

**TaskResult**:
- `success: bool`, `output: String`, `files_modified: Vec<PathBuf>`, `subtasks: Vec<TaskId>`, `token_usage: TokenUsage`

**TokenUsage**: `input_tokens: u32`, `output_tokens: u32` (defaults to 0/0)

### Message Types

```rust
pub struct Message {
    pub role: MessageRole,           // System | User | Assistant | Tool
    pub content: String,
    pub media: Vec<String>,          // File paths (images, audio)
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}
```

### Gateway Protocol

```rust
pub struct InboundMessage {       // channel → agent
    pub channel: String,          // "telegram", "cli", "discord", etc.
    pub sender_id: String,
    pub chat_id: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub media: Vec<String>,
    pub metadata: serde_json::Value,
}

pub struct OutboundMessage {      // agent → channel
    pub channel: String,
    pub chat_id: String,
    pub content: String,
    pub reply_to: Option<String>,
    pub media: Vec<String>,
    pub metadata: serde_json::Value,
}
```

`InboundMessage::session_key()` derives `SessionKey::new(channel, chat_id)` — format `"{channel}:{chat_id}"`.

### Inter-Agent Coordination

```rust
pub enum AgentMessage {           // tagged: "type", snake_case
    TaskAssign { task: Box<Task> },
    TaskUpdate { task_id: TaskId, status: TaskStatus },
    TaskComplete { task_id: TaskId, result: TaskResult },
    ContextRequest { task_id: TaskId, query: String },
    ContextResponse { task_id: TaskId, context: Vec<Message> },
}
```

### Error System

```rust
pub struct Error {
    pub kind: ErrorKind,
    pub context: Option<String>,      // Chained context
    pub suggestion: Option<String>,   // Actionable fix hint
}
```

**ErrorKind variants**: TaskNotFound, AgentNotFound, InvalidStateTransition, LlmError, ApiError (status-aware: 401→check key, 429→rate limit), ToolError, ConfigError, ApiKeyNotSet, UnknownProvider, Timeout, ChannelError, SessionError, IoError, SerializationError, Other(eyre::Report).

### Utilities

`truncate_utf8(s: &mut String, max_len: usize, suffix: &str)` — in-place truncation at UTF-8 char boundaries. Appends suffix after truncation. Used across all tool outputs.

---

## crew-llm — LLM Provider Abstraction

### Provider Trait

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn chat(&self, messages: &[Message], tools: &[ToolSpec], config: &ChatConfig) -> Result<ChatResponse>;
    async fn chat_stream(&self, messages: &[Message], tools: &[ToolSpec], config: &ChatConfig) -> Result<ChatStream>;  // default: falls back to chat()
    fn context_window(&self) -> u32;  // default: context_window_tokens(self.model_id())
    fn model_id(&self) -> &str;
    fn provider_name(&self) -> &str;
}
```

### Configuration

```rust
pub struct ChatConfig {
    pub max_tokens: Option<u32>,        // default: Some(4096)
    pub temperature: Option<f32>,       // default: Some(0.0)
    pub tool_choice: ToolChoice,        // Auto | Required | None | Specific { name }
    pub stop_sequences: Vec<String>,
}
```

### Response Types

```rust
pub struct ChatResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub stop_reason: StopReason,       // EndTurn | ToolUse | MaxTokens | StopSequence
    pub usage: TokenUsage,
}

pub enum StreamEvent {
    TextDelta(String),
    ToolCallDelta { index, id, name, arguments_delta },
    Usage(TokenUsage),
    Done(StopReason),
    Error(String),
}

pub type ChatStream = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;
```

### Native Providers

| Provider | Base URL | Auth Header | Image Format | Default Model |
|----------|----------|-------------|--------------|---------------|
| Anthropic | api.anthropic.com | x-api-key | Base64 blocks | claude-sonnet-4-20250514 |
| OpenAI | api.openai.com/v1 | Authorization: Bearer | Data URI | gpt-4o |
| Gemini | generativelanguage.googleapis.com/v1beta | x-goog-api-key | Base64 inline | gemini-2.0-flash |
| OpenRouter | openrouter.ai/api/v1 | Authorization: Bearer | Data URI | anthropic/claude-sonnet-4-20250514 |

**OpenAI-compatible** (all via `OpenAIProvider::with_base_url()`):

| Provider | Base URL | Default Model | API Key Env |
|----------|----------|---------------|-------------|
| DeepSeek | api.deepseek.com/v1 | deepseek-chat | DEEPSEEK_API_KEY |
| Groq | api.groq.com/openai/v1 | llama-3.3-70b-versatile | GROQ_API_KEY |
| Moonshot | api.moonshot.ai/v1 | kimi-k2.5 | MOONSHOT_API_KEY |
| DashScope | dashscope.aliyuncs.com/compatible-mode/v1 | qwen-max | DASHSCOPE_API_KEY |
| MiniMax | api.minimax.io/v1 | MiniMax-Text-01 | MINIMAX_API_KEY |
| Zhipu | open.bigmodel.cn/api/paas/v4 | glm-4-plus | ZHIPU_API_KEY |
| Ollama | localhost:11434/v1 | llama3.2 | (none) |
| vLLM | (user-provided) | (user-provided) | (user-provided) |

### SSE Streaming

`parse_sse_response(response) -> impl Stream<Item = SseEvent>` — stateful unfold-based parser. Max buffer: 1 MB. Handles `\n\n` and `\r\n\r\n` separators. Each provider maps SSE events to `StreamEvent`:

- **Anthropic**: `message_start` → input tokens, `content_block_start/delta` → text/tool chunks, `message_delta` → stop reason. Custom SSE state machine.
- **OpenAI/OpenRouter**: Standard OpenAI SSE with `[DONE]` sentinel. `delta.content` for text, `delta.tool_calls[]` for tools. Shared parser: `parse_openai_sse_events()`.
- **Gemini**: `alt=sse` endpoint. `candidates[0].content.parts[]` with function call data.

### RetryProvider

Wraps any `Arc<dyn LlmProvider>` with exponential backoff. Wrapped by `ProviderChain` for multi-provider failover.

```rust
pub struct RetryConfig {
    pub max_retries: u32,           // default: 3
    pub initial_delay: Duration,    // default: 1s
    pub max_delay: Duration,        // default: 60s
    pub backoff_multiplier: f64,    // default: 2.0
}
```

**Delay formula**: `initial_delay * backoff_multiplier^attempt`, capped at max_delay.

**Retryable errors** (three-tier detection):
1. HTTP status: 429, 500, 502, 503, 504, 529
2. reqwest: `is_connect()` or `is_timeout()`
3. String fallback: "connection refused", "timed out", "overloaded"

### Provider Failover Chain

`ProviderChain` wraps multiple `Arc<dyn LlmProvider>` and transparently fails over on retriable errors. Configured via `fallback_models` in config.

```rust
pub struct ProviderChain {
    slots: Vec<ProviderSlot>,       // provider + AtomicU32 failure count
    failure_threshold: u32,         // default: 3
}
```

**Behavior**: Tries providers in order, skipping degraded ones (failures >= threshold). On retriable error, moves to the next. On success, resets failure count. If all degraded, picks the one with fewest failures.

**Retryable**: Same criteria as RetryProvider (429, 5xx, connect/timeout errors).

### Token Estimation

```rust
pub fn estimate_tokens(text: &str) -> u32  // ~4 chars/token ASCII, ~1.5 chars/token CJK
pub fn estimate_message_tokens(msg: &Message) -> u32  // content + tool_calls + 4 overhead
```

### Context Windows

| Model Family | Tokens |
|---|---|
| Claude 3/4 | 200,000 |
| GPT-4o/4-turbo | 128,000 |
| o1/o3/o4 | 200,000 |
| Gemini 2.0/1.5 | 1,000,000 |
| Default (unknown) | 128,000 |

### Pricing

`model_pricing(model_id) -> Option<ModelPricing>` — case-insensitive substring match. Cost = `(input/1M) * input_rate + (output/1M) * output_rate`.

| Model | Input $/1M | Output $/1M |
|---|---|---|
| claude-opus-4 | 15.00 | 75.00 |
| claude-sonnet-4 | 3.00 | 15.00 |
| claude-haiku | 0.80 | 4.00 |
| gpt-4o | 2.50 | 10.00 |
| gpt-4o-mini | 0.15 | 0.60 |
| o3/o4 | 10.00 | 40.00 |

### Embedding

```rust
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
```

**OpenAIEmbedder**: Default model `text-embedding-3-small` (1536 dims). `text-embedding-3-large` = 3072 dims.

### Transcription

**GroqTranscriber**: Whisper `whisper-large-v3` via `https://api.groq.com/openai/v1/audio/transcriptions`. Multipart form. 60s timeout. MIME detection: ogg/opus→audio/ogg, mp3→audio/mpeg, m4a→audio/mp4, wav→audio/wav.

### Vision

`encode_image(path) -> (mime_type, base64_data)` — JPEG/PNG/GIF/WebP. `is_image(path) -> bool`.

---

## crew-memory — Persistence & Search

### EpisodeStore

redb database at `.crew/episodes.redb` with three tables:

| Table | Key | Value | Purpose |
|---|---|---|---|
| episodes | &str (episode_id) | &str (JSON) | Full episode records |
| cwd_index | &str (working_dir) | &str (JSON array of IDs) | Directory-scoped lookup |
| embeddings | &str (episode_id) | &[u8] (bincode Vec<f32>) | Vector embeddings |

```rust
pub struct Episode {
    pub id: String,                   // UUID v7
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub working_dir: PathBuf,
    pub summary: String,              // LLM-generated, truncated to 500 chars
    pub outcome: EpisodeOutcome,      // Success | Failure | Blocked | Cancelled
    pub key_decisions: Vec<String>,
    pub files_modified: Vec<PathBuf>,
    pub created_at: DateTime<Utc>,
}
```

**Operations**:
- `store(episode)` — serialize to JSON, update cwd_index, insert into in-memory HybridIndex
- `get(id)` — direct lookup by episode_id
- `find_relevant(cwd, query, limit)` — keyword matching scoped to directory
- `recent_for_cwd(cwd, n)` — N most recent by created_at descending
- `store_embedding(id, Vec<f32>)` — bincode serialize, store in embeddings table, update HybridIndex
- `find_relevant_hybrid(query, query_embedding, limit)` — global hybrid search across all episodes

**Initialization**: On `open()`, rebuilds in-memory HybridIndex by iterating all episodes and loading embeddings from DB.

### MemoryStore

File-based persistent memory at `{data_dir}/memory/`:

- `MEMORY.md` — long-term memory (full overwrite)
- `YYYY-MM-DD.md` — daily notes (append with date header)

**`get_memory_context()`** builds system prompt injection:
1. `## Long-term Memory` — full MEMORY.md
2. `## Recent Activity` — 7-day rolling window of daily notes
3. `## Today's Notes` — current day

### HybridIndex — BM25 + Vector Search

```rust
pub struct HybridIndex {
    inverted: HashMap<String, Vec<(usize, f32)>>,  // term → [(doc_idx, tf)]
    doc_lengths: Vec<usize>,
    avg_dl: f64,
    ids: Vec<String>,
    texts: Vec<String>,
    hnsw: Option<Hnsw<'static, f32, DistCosine>>,
    has_embedding: Vec<bool>,
    dimension: usize,                               // default: 1536
}
```

**BM25 scoring** (constants: K1=1.2, B=0.75):
- Tokenization: lowercase, split on non-alphanumeric, filter tokens < 2 chars
- IDF: `ln((N - df + 0.5) / (df + 0.5) + 1.0)`
- Score: `IDF * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl/avg_dl))`
- Normalized to [0, 1] range

**HNSW vector index** (via `hnsw_rs`):
- Parameters: M=16, max_nodes=10000, ef_construction=16, ef=200, DistCosine
- L2 normalization before insertion/search
- Cosine similarity = `1 - distance` (DistCosine returns 1-cos_sim)

**Hybrid ranking** — fetches `limit * 4` candidates from each:
- With vectors: `0.7 * vector_score + 0.3 * bm25_score`
- Without vectors: BM25 only (graceful fallback)

---

## crew-agent — Agent Runtime

### Agent Core

```rust
pub struct Agent {
    id: AgentId,
    llm: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    memory: Arc<EpisodeStore>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    system_prompt: RwLock<String>,
    config: AgentConfig,
    reporter: Arc<dyn ProgressReporter>,
    shutdown: Arc<AtomicBool>,
}

pub struct AgentConfig {
    pub max_iterations: u32,          // default: 50 (CLI overrides to 20)
    pub max_tokens: Option<u32>,      // None = unlimited
    pub max_timeout: Option<Duration>,// default: 600s wall-clock timeout
    pub save_episodes: bool,          // default: true
}
```

### Execution Loop (`run_task` / `process_message`)

```
1. Build messages: system prompt + history + memory context + input
2. Loop (up to max_iterations):
   a. Check shutdown flag and token budget
   b. trim_to_context_window() — compact if needed
   c. Call LLM via chat_stream()
   d. Consume stream → accumulate text, tool_calls, tokens
   e. Match stop_reason:
      - EndTurn/StopSequence → save episode, return result
      - ToolUse → execute_tools() → append results → continue
      - MaxTokens → return result
```

**ConversationResponse**: `content: String`, `token_usage: TokenUsage`, `files_modified: Vec<PathBuf>`, `streamed: bool`

**Episode saving**: After task completion, fires-and-forgets embedding generation if embedder present.

**Wall-clock timeout**: Agent aborts after `max_timeout` (default 600s) regardless of iteration count.

### Tool Output Sanitization

Before feeding tool results back to the LLM, `sanitize_tool_output()` (in `sanitize.rs`) strips noise:
- **Base64 data URIs**: `data:...;base64,<payload>` → `[base64-data-redacted]`
- **Long hex strings**: 64+ contiguous hex chars (SHA-256, raw keys) → `[hex-redacted]`

### Context Compaction

Triggered when estimated tokens exceed 80% of context window / 1.2 safety margin.

**Algorithm**:
1. Keep MIN_RECENT_MESSAGES (6) most recent non-system messages
2. Don't split inside tool call/result pairs
3. Summarize old messages: first line (200 chars), strip tool arguments, drop media
4. Budget: 40% of total for summary (BASE_CHUNK_RATIO = 0.4)
5. Replace: `[System, CompactionSummary, Recent1, Recent2, ...]`

**Format**:
- User: `> User: first line [media omitted]`
- Assistant: `> Assistant: content` or `- Called tool_name`
- Tool: `-> tool_name: ok|error - first 100 chars`

### Tool System

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;
    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult>;
}

pub struct ToolResult {
    pub output: String,
    pub success: bool,
    pub file_modified: Option<PathBuf>,
    pub tokens_used: Option<TokenUsage>,
}
```

**ToolRegistry**: `HashMap<String, Arc<dyn Tool>>` with `provider_policy: Option<ToolPolicy>` for soft filtering.

### Built-in Tools (14)

| Tool | Parameters | Key Behavior |
|---|---|---|
| **read_file** | path, start_line?, end_line? | Line numbers (NNN\|), 100KB truncation, symlink rejection |
| **write_file** | path, content | Creates parent dirs, returns file_modified |
| **edit_file** | path, old_string, new_string | Exact match required, error on 0 or >1 occurrences |
| **diff_edit** | path, diff | Unified diff with fuzzy matching (+-3 lines), reverse hunk application |
| **glob** | pattern, limit=100 | Rejects absolute paths and `..`, relative results |
| **grep** | pattern, file_pattern?, limit=50, context=0, ignore_case=false | .gitignore-aware via `ignore::WalkBuilder`, regex with `(?i)` flag |
| **list_dir** | path | Sorted, `[dir]`/`[file]` prefix |
| **shell** | command, timeout_secs=120 | SafePolicy check, 50KB output truncation, sandbox-wrapped |
| **web_search** | query, count=5 | Brave Search API (BRAVE_API_KEY) |
| **web_fetch** | url, extract_mode="markdown", max_chars=50000 | SSRF protection, htmd HTML→markdown, 30s timeout |
| **message** | content, channel?, chat_id? | Cross-channel messaging via OutboundMessage. **Gateway-only** |
| **spawn** | task, label?, mode="background", allowed_tools, context? | Subagent with inherited provider policy. sync=inline, background=async. **Gateway-only** |
| **cron** | action, message, schedule params | Schedule add/list/remove/enable/disable. **Gateway-only** |
| **browser** | action, url?, selector?, text?, expression? | Feature-gated (`browser`). Headless Chrome via CDP. Actions: navigate (SSRF + scheme check), get_text, get_html, click, type, screenshot, evaluate, close. 5min idle timeout, env sanitization, 10s JS timeout, early action validation |

**Registration**: 10 core tools registered in `ToolRegistry::with_builtins()` (all modes). Browser is feature-gated. Message, spawn, and cron are registered only in gateway mode (`gateway.rs`).

### Tool Policies

```rust
pub struct ToolPolicy {
    pub allow: Vec<String>,   // empty = allow all
    pub deny: Vec<String>,    // deny-wins
}
```

**Groups**: `group:fs` (read_file, write_file, edit_file, diff_edit), `group:runtime` (shell), `group:web` (web_search, web_fetch, browser), `group:search` (glob, grep, list_dir), `group:sessions` (spawn).

**Wildcards**: `exec*` matches prefix. Provider-specific policies via config `tools.byProvider`.

### Command Policy (ShellTool)

```rust
pub enum Decision { Allow, Deny, Ask }
```

**SafePolicy deny patterns**: `rm -rf /`, `rm -rf /*`, `dd if=`, `mkfs`, `:(){:|:&};:`, `chmod -R 777 /`

**SafePolicy ask patterns**: `sudo`, `rm -rf`, `git push --force`, `git reset --hard`

### Sandbox

```rust
pub enum SandboxMode { Auto, Bwrap, Macos, Docker, None }
```

**BLOCKED_ENV_VARS** (18 vars, shared across all backends + MCP):
`LD_PRELOAD, LD_LIBRARY_PATH, LD_AUDIT, DYLD_INSERT_LIBRARIES, DYLD_LIBRARY_PATH, DYLD_FRAMEWORK_PATH, DYLD_FALLBACK_LIBRARY_PATH, DYLD_VERSIONED_LIBRARY_PATH, NODE_OPTIONS, PYTHONSTARTUP, PYTHONPATH, PERL5OPT, RUBYOPT, RUBYLIB, JAVA_TOOL_OPTIONS, BASH_ENV, ENV, ZDOTDIR`

| Backend | Isolation | Network | Path Validation |
|---|---|---|---|
| **Bwrap** (Linux) | RO bind /usr,/lib,/bin,/sbin,/etc; RW bind workdir; tmpfs /tmp; unshare-pid | `--unshare-net` if !allow_network | N/A |
| **Macos** (sandbox-exec) | SBPL profile: process-exec/fork, file-read*, writes to workdir+/private/tmp | `(allow network*)` or `(deny network*)` | Rejects control chars, `(`, `)`, `\`, `"` |
| **Docker** | `--rm --security-opt no-new-privileges --cap-drop ALL` | `--network none` | Rejects `:`, `\0`, `\n`, `\r` |

**Docker resource limits**: `--cpus`, `--memory`, `--pids-limit`. Mount modes: None (/tmp workdir), ReadOnly, ReadWrite.

### Hooks System

Lifecycle hooks run shell commands at agent events. Configured via `hooks` array in config.

```rust
pub enum HookEvent { BeforeToolCall, AfterToolCall, BeforeLlmCall, AfterLlmCall }

pub struct HookConfig {
    pub event: HookEvent,
    pub command: Vec<String>,       // argv array (no shell interpretation)
    pub timeout_ms: u64,            // default: 5000
    pub tool_filter: Vec<String>,   // tool events only; empty = all
}
```

**Shell protocol**: JSON payload on stdin. Exit code semantics: 0=allow, 1=deny (before-hooks only), 2+=error. Before-hooks can deny operations; after-hook exit codes only count as errors.

**Circuit breaker**: `HookExecutor` auto-disables a hook after 3 consecutive failures (configurable via `with_threshold()`). Resets on success.

**Environment**: Commands sanitized via `BLOCKED_ENV_VARS`. Tilde expansion supports `~/` and `~username/`.

**Integration**: Wired into `chat.rs`, `gateway.rs`, `serve.rs`. Hook config changes trigger restart via config watcher.

### MCP Integration

JSON-RPC transport for Model Context Protocol servers. Two transport modes:

**Transports**:
1. **Stdio**: Spawns server as child process (command + args + env). Line limit: 1MB. Env sanitized via `BLOCKED_ENV_VARS`.
2. **HTTP/SSE**: Connects to remote server via `url` field. POST JSON, SSE response handling.

**Lifecycle** (stdio):
1. Spawn server (command + args + env, filtering BLOCKED_ENV_VARS)
2. Initialize: `protocolVersion: "2024-11-05"`
3. Discover tools: `tools/list` RPC
4. Register McpTool wrappers (30s timeout, 1MB max response)

**McpTool execution**: `tools/call` with name + arguments. Extracts `content[].text` from response.

### Skills System

Skills are markdown instruction files that extend agent capabilities. Two sources: built-in (compiled into binary) and workspace (user-installed).

#### Skill File Format (SKILL.md)

```yaml
---
name: skill_name
description: What it does
requires_bins: binary1, binary2    # comma-separated, checked via `which`
requires_env: ENV_VAR1, ENV_VAR2   # comma-separated, checked via std::env::var()
always: true|false                 # auto-load into system prompt when available
---
Skill instructions here (markdown). This body is injected into the agent's
system prompt when the skill is activated.
```

**Frontmatter parsing**: Simple `key: value` line matching (not full YAML). `split_frontmatter()` finds content between `---` delimiters. `strip_frontmatter()` returns body only.

#### SkillInfo

```rust
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub path: PathBuf,          // filesystem path or "(built-in)/name/SKILL.md"
    pub available: bool,        // bins_ok && env_ok
    pub always: bool,           // auto-load into system prompt
    pub builtin: bool,          // true if from BUILTIN_SKILLS, false if workspace
}
```

**Availability check**: `available = requires_bins all found on PATH AND requires_env all set`. Missing requirements make the skill unavailable but still listed.

#### SkillsLoader

```rust
pub struct SkillsLoader {
    skills_dir: PathBuf,        // {data_dir}/skills/
}
```

**Methods**:
- `list_skills()` — scans workspace dir + built-ins. Workspace skills override built-ins with same name (checked via HashSet). Results sorted alphabetically.
- `load_skill(name)` — returns body (frontmatter stripped). Checks workspace first, falls back to built-in.
- `build_skills_summary()` — generates XML for system prompt injection:
  ```xml
  <skills>
    <skill available="true">
      <name>skill_name</name>
      <description>What it does</description>
      <location>/path/to/SKILL.md</location>
    </skill>
  </skills>
  ```
- `get_always_skills()` — filters skills where `always: true` AND `available: true`.
- `load_skills_for_context(names)` — loads multiple skills, joins with `\n---\n`.

#### Built-in Skills (6, compile-time `include_str!()`)

```rust
pub struct BuiltinSkill {
    pub name: &'static str,
    pub content: &'static str,  // full SKILL.md including frontmatter
}
pub const BUILTIN_SKILLS: &[BuiltinSkill] = &[...];
```

| Skill | Purpose |
|---|---|
| cron | Task scheduling instructions |
| github | GitHub integration (issues, PRs) |
| skill-creator | Create new skills |
| summarize | Conversation summarization |
| tmux | Terminal multiplexer control |
| weather | Weather information retrieval |

#### CLI Management (`crew skills`)

- `list` — shows built-in skills (with override status) + workspace skills
- `install <user/repo/skill-name>` — fetches `SKILL.md` from `https://raw.githubusercontent.com/{repo}/main/SKILL.md` (15s timeout), saves to `.crew/skills/{name}/SKILL.md`. Fails if skill already exists.
- `remove <name>` — deletes `.crew/skills/{name}/` directory

#### Integration with Gateway

In the gateway command, skills are loaded during system prompt construction:
1. `get_always_skills()` — collects auto-load skill names
2. `load_skills_for_context(names)` — loads and joins skill bodies
3. `build_skills_summary()` — appends XML skill index to system prompt
4. Always-on skill content is prepended to the system prompt

### Plugin System

Plugins extend the agent with external tools via standalone executables. Each plugin is a directory containing a `manifest.json` and an executable file.

#### Directory Layout

```
.crew/plugins/           # local (project-level)
~/.crew/plugins/         # global (user-level)
  └── my-plugin/
      ├── manifest.json  # plugin metadata + tool definitions
      └── my-plugin      # executable (or "main" as fallback)
```

**Discovery order**: local `.crew/plugins/` first, then global `~/.crew/plugins/`. Both are scanned by `Config::plugin_dirs()`.

#### PluginManifest

```rust
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub tools: Vec<PluginToolDef>,    // default: empty vec
}

pub struct PluginToolDef {
    pub name: String,                 // must be unique across all plugins
    pub description: String,
    pub input_schema: serde_json::Value,  // default: {"type": "object"}
}
```

**Example manifest.json**:
```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "tools": [
    {
      "name": "greet",
      "description": "Greet someone by name",
      "input_schema": {
        "type": "object",
        "properties": { "name": { "type": "string" } }
      }
    }
  ]
}
```

#### PluginLoader

```rust
pub struct PluginLoader;  // stateless, all methods are associated functions
```

**`load_into(registry, dirs)`**:
1. Scan each directory for subdirectories
2. For each subdirectory, look for `manifest.json`
3. Parse manifest, find executable (try directory name first, then `main`)
4. Validate executable permissions (Unix: `mode & 0o111 != 0`; non-Unix: existence check)
5. Wrap each tool definition as a `PluginTool` implementing the `Tool` trait
6. Register into `ToolRegistry`
7. Log warning: `"loaded unverified plugin (no signature check)"`
8. Return total tool count. Failed plugins are skipped with warning, not fatal.

#### PluginTool — Execution Protocol

```rust
pub struct PluginTool {
    plugin_name: String,
    tool_def: PluginToolDef,
    executable: PathBuf,
}
```

**Invocation**: `executable <tool_name>` (tool name passed as first argument).

**stdin/stdout protocol**:
1. Spawn executable with tool name as arg, piped stdin/stdout/stderr
2. Write JSON-serialized arguments to stdin, close (EOF signals end of input)
3. Wait for exit with 30s timeout (`PLUGIN_TIMEOUT`)
4. Parse stdout as JSON:
   - **Structured**: `{"output": "...", "success": true/false}` → use parsed values
   - **Fallback**: raw stdout + stderr concatenated, success from exit code
5. Return `ToolResult` (no `file_modified` tracking for plugins)

**Error handling**:
- Spawn failure → eyre error with plugin name and executable path
- Timeout → eyre error with plugin name, tool name, and duration
- JSON parse failure → graceful fallback to raw output

### Progress Reporting

The agent emits structured events during execution via a trait-based observer pattern. Consumers (CLI, REST API) implement the trait to render progress in their own format.

#### ProgressReporter Trait

```rust
pub trait ProgressReporter: Send + Sync {
    fn report(&self, event: ProgressEvent);
}
```

Agent holds `reporter: Arc<dyn ProgressReporter>`. Events are fired synchronously during the execution loop (non-blocking — implementations must not block).

#### ProgressEvent Enum

```rust
pub enum ProgressEvent {
    TaskStarted { task_id: String },
    Thinking { iteration: u32 },
    Response { content: String, iteration: u32 },
    ToolStarted { name: String, tool_id: String },
    ToolCompleted { name: String, tool_id: String, success: bool,
                    output_preview: String, duration: Duration },
    FileModified { path: String },
    TokenUsage { input_tokens: u32, output_tokens: u32 },
    TaskCompleted { success: bool, iterations: u32, duration: Duration },
    TaskInterrupted { iterations: u32 },
    MaxIterationsReached { limit: u32 },
    TokenBudgetExceeded { used: u32, limit: u32 },
    StreamChunk { text: String, iteration: u32 },
    StreamDone { iteration: u32 },
    CostUpdate { session_input_tokens: u32, session_output_tokens: u32,
                 response_cost: Option<f64>, session_cost: Option<f64> },
}
```

#### Implementations (3)

**SilentReporter** — no-op, used as default when no reporter is configured.

**ConsoleReporter** — CLI output with ANSI colors and streaming support:

```rust
pub struct ConsoleReporter {
    use_colors: bool,
    verbose: bool,
    stdout: Mutex<BufWriter<Stdout>>,  // buffered for streaming chunks
}
```

| Event | Output |
|---|---|
| Thinking | `\r⟳ Thinking... (iteration N)` (overwrites line, yellow) |
| Response | `◆ first 3 lines...` (cyan, clears Thinking line) |
| ToolStarted | `\r⚙ Running tool_name...` (overwrites line, yellow) |
| ToolCompleted | `✓ tool_name (duration)` green or `✗ tool_name` red; verbose: 5 lines of output + `...` |
| FileModified | `📝 Modified: path` (green) |
| TokenUsage | `Tokens: N in, N out` (verbose only, dim) |
| TaskCompleted | `✓ Completed N iterations, Xs` or `✗ Failed after N iterations` |
| TaskInterrupted | `⚠ Interrupted after N iterations.` (yellow) |
| MaxIterationsReached | `⚠ Reached max iterations limit (N).` (yellow) |
| TokenBudgetExceeded | `⚠ Token budget exceeded (used, limit).` (yellow) |
| StreamChunk | Write to buffered stdout; flush only on `\n` (reduces syscalls) |
| StreamDone | Flush + newline |
| CostUpdate | `Tokens: N in / N out \| Cost: $X.XXXX` |
| TaskStarted | `▶ Task: id` (verbose only, dim) |

**Duration formatting**: >1s → `{:.1}s`, ≤1s → `{N}ms`.

**SseBroadcaster** (REST API, feature: `api`) — converts events to JSON and broadcasts via `tokio::sync::broadcast` channel:

```rust
pub struct SseBroadcaster {
    tx: broadcast::Sender<String>,  // JSON-serialized events
}
```

| ProgressEvent | JSON `type` field | Additional fields |
|---|---|---|
| ToolStarted | `"tool_start"` | `tool` |
| ToolCompleted | `"tool_end"` | `tool`, `success` |
| StreamChunk | `"token"` | `text` |
| StreamDone | `"stream_end"` | — |
| CostUpdate | `"cost_update"` | `input_tokens`, `output_tokens`, `session_cost` |
| Thinking | `"thinking"` | `iteration` |
| Response | `"response"` | `iteration` |
| (other) | `"other"` | — (logged at debug level) |

Subscribers receive events via `SseBroadcaster::subscribe() -> broadcast::Receiver<String>`. Send errors (no subscribers) are silently ignored.

---

## crew-bus — Gateway Infrastructure

### Message Bus

`create_bus() -> (AgentHandle, BusPublisher)` linked by mpsc channels (capacity 256). AgentHandle receives InboundMessages; BusPublisher dispatches OutboundMessages.

**Queue Modes** (configured via `gateway.queue_mode`):
- `Followup` (default): FIFO — process queued messages one at a time
- `Collect`: Merge queued messages by session, concatenating content before processing

### Channel Trait

```rust
#[async_trait]
pub trait Channel: Send + Sync {
    fn name(&self) -> &str;
    async fn start(&self, inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()>;
    async fn send(&self, msg: &OutboundMessage) -> Result<()>;
    fn is_allowed(&self, sender_id: &str) -> bool;
    async fn stop(&self) -> Result<()>;
}
```

### Channel Implementations

| Channel | Transport | Feature Flag | Auth | Dedup |
|---|---|---|---|---|
| **CLI** | stdin/stdout | (always) | N/A | N/A |
| **Telegram** | teloxide long-poll | `telegram` | Bot token (env) | teloxide built-in |
| **Discord** | serenity gateway | `discord` | Bot token (env) | serenity built-in |
| **Slack** | Socket Mode (tokio-tungstenite) | `slack` | Bot token + App token | message_ts |
| **WhatsApp** | WebSocket bridge (ws://localhost:3001) | `whatsapp` | Baileys bridge | HashSet (10K cap, clear on overflow) |
| **Feishu** | WebSocket (tokio-tungstenite) | `feishu` | App ID + Secret → tenant token (TTL 6000s) | HashSet (10K cap, clear on overflow) |
| **Email** | IMAP poll + SMTP send | `email` | Username/password, rustls TLS | IMAP UNSEEN flag |

**Email specifics**: IMAP `async-imap` with rustls for inbound (poll unseen, mark \Seen). SMTP `lettre` for outbound (port 465=implicit TLS, other=STARTTLS). `mailparse` for RFC822 body extraction. Body truncated via `truncate_utf8(max_body_chars)`.

**Feishu specifics**: Tenant access token with TTL cache (6000s). WebSocket gateway URL from `/callback/ws/endpoint`. Message type detection via `header.event_type == "im.message.receive_v1"`. Supports `oc_*` (chat_id) vs `ou_*` (open_id) routing.

**Media**: `download_media()` helper downloads photos/voice/audio/documents to `.crew/media/`.

**Transcription**: Voice/audio auto-transcribed via GroqTranscriber before agent processing.

### Message Coalescing

Splits oversized messages into channel-safe chunks:

| Channel | Max Chars |
|---|---|
| Telegram | 4000 |
| Discord | 1900 |
| Slack | 3900 |

**Break preference**: paragraph (`\n\n`) > newline (`\n`) > sentence (`. `) > space (` `) > hard cut.

MAX_CHUNKS = 50 (DoS limit). UTF-8 safe boundary detection via `char_indices()`.

### Session Manager

JSONL persistence at `.crew/sessions/{key}.jsonl`.

- **In-memory cache**: HashMap with disk sync on write
- **Filenames**: Percent-encoded SessionKey for collision-free mapping
- **Crash safety**: Atomic write-then-rename
- **Forking**: `fork()` creates child session with `parent_key` tracking, copies last N messages

### Cron Service

JSON persistence at `.crew/cron.json`.

**Schedule types**:
- `Every { seconds: u64 }` — recurring interval
- `Cron { expr: String }` — cron expression via `cron` crate
- `At { timestamp_ms: i64 }` — one-shot (auto-delete after run)

**CronJob fields**: id (8-char hex from UUIDv7), name, enabled, schedule, payload (message + deliver flag + channel + chat_id), state (next_run_at_ms, run_count), delete_after_run.

### Heartbeat Service

Periodic check of `HEARTBEAT.md` (default: 30 min interval). Sends content to agent if non-empty.

---

## crew-cli — CLI & Configuration

### Commands

| Command | Description |
|---|---|
| `chat` | Interactive multi-turn chat. Readline with history. Exit: exit/quit/:q |
| `gateway` | Persistent multi-channel daemon with session management |
| `init` | Initialize .crew/ with config, templates, directories |
| `status` | Show config, provider, API keys, bootstrap files |
| `auth login/logout/status` | OAuth PKCE (OpenAI), device code, paste-token |
| `cron list/add/remove/enable` | CLI cron job management |
| `channels status/login` | Channel compilation status, WhatsApp bridge setup |
| `skills list/install/remove` | Skill management, GitHub fetch |
| `clean` | Remove .redb files with dry-run support |
| `completions` | Shell completion generation (bash/zsh/fish) |
| `docs` | Generate tool + provider documentation |
| `serve` | REST API server (feature: api) — axum on port 8080 |

### Configuration

Loaded from `.crew/config.json` (local) or `~/.config/crew/config.json` (global). Local takes precedence.

- **`${VAR}` expansion**: Environment variable substitution in string values
- **Versioned config**: Version field with automatic `migrate_config()` framework
- **Provider auto-detect** (`detect_provider(model)`): claude→anthropic, gpt/o1/o3/o4→openai, gemini→gemini, deepseek→deepseek, kimi/moonshot→moonshot, qwen→dashscope, glm→zhipu, llama/mixtral→groq

**API key resolution order**: Auth store (`~/.crew/auth.json`) → environment variable.

### Auth Module

**OAuth PKCE** (OpenAI):
1. Generate 64-char verifier (two UUIDv4 hex)
2. SHA-256 challenge, base64-URL encode (no padding)
3. TCP listener on port 1455
4. Browser → `auth.openai.com` with PKCE + state
5. Callback validates state (CSRF), exchanges code+verifier for tokens

**Device Code Flow** (OpenAI): POST `deviceauth/usercode`, poll `deviceauth/token` every 5s+.

**Paste Token**: Prompt for API key from stdin, store as `auth_method: "paste_token"`.

**AuthStore**: `~/.crew/auth.json` (mode 0600). `{credentials: {provider: AuthCredential}}`.

### Config Watcher

Polls every 5 seconds. SHA-256 hash comparison of file contents.

**Hot-reloadable**: system_prompt, max_history (applied live).

**Restart-required**: provider, model, base_url, api_key_env, sandbox, mcp_servers, hooks, gateway.queue_mode, channels.

### REST API (feature: `api`)

| Route | Method | Description |
|---|---|---|
| `/api/chat` | POST | Send message → response |
| `/api/chat/stream` | GET | SSE stream of ProgressEvents |
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/{id}/messages` | GET | Paginated history (?limit=100&offset=0, max 500) |
| `/api/status` | GET | Version, model, provider, uptime |
| `/metrics` | GET | Prometheus text exposition format (unauthenticated) |
| `/*` (fallback) | GET | Embedded web UI (static files via rust-embed) |

**Auth**: Optional bearer token with constant-time comparison (API routes only; `/metrics` and static files are public). **CORS**: localhost:3000/8080. **Max message**: 1MB.

**Web UI**: Embedded SPA via `rust-embed` served as the fallback handler. Session sidebar, chat interface, SSE streaming, dark theme. Vanilla HTML/CSS/JS (no build tools).

**Prometheus Metrics**: `crew_tool_calls_total` (counter, labels: tool, success), `crew_tool_call_duration_seconds` (histogram, label: tool), `crew_llm_tokens_total` (counter, label: direction). Powered by `metrics` + `metrics-exporter-prometheus` crates.

### Session Compaction (Gateway)

Triggered when message count > 40 (threshold). Keeps 10 recent messages. Summarizes older messages via LLM to <500 words. Rewrites JSONL session file.

---

## Data Flows

### Chat Mode

```
User Input → readline → Agent.process_message(input, history)
                              │
                              ├─ Build messages (system + history + memory + input)
                              ├─ trim_to_context_window() if needed
                              ├─ Call LLM via chat_stream() with tool specs
                              ├─ Execute tools if ToolUse (loop)
                              └─ Return ConversationResponse
                                    │
                              Print response, append to history
```

### Gateway Mode

```
Channel → InboundMessage → MessageBus → [transcribe audio] → [load session]
                                              │
                                    Agent.process_message()
                                              │
                                        OutboundMessage
                                              │
                                   ChannelManager.dispatch()
                                              │
                                    coalesce() → Channel.send()
```

System messages (cron, heartbeat, spawn results) flow through the same bus with `channel: "system"` and metadata routing.

---

## Feature Flags

```toml
# crew-bus
telegram = ["teloxide"]
discord  = ["serenity"]
slack    = ["tokio-tungstenite"]
whatsapp = ["tokio-tungstenite"]
feishu   = ["tokio-tungstenite"]
email    = ["async-imap", "tokio-rustls", "rustls", "webpki-roots", "lettre", "mailparse"]

# crew-agent
browser  = ["tokio-tungstenite", "which", "tempfile", "base64"]

# crew-cli
api      = ["axum", "tower-http", "futures"]
telegram = ["crew-bus/telegram"]
discord  = ["crew-bus/discord"]
slack    = ["crew-bus/slack"]
whatsapp = ["crew-bus/whatsapp"]
feishu   = ["crew-bus/feishu"]
email    = ["crew-bus/email"]
browser  = ["crew-agent/browser"]
```

---

## File Layout

```
crates/
├── crew-core/src/
│   ├── lib.rs, task.rs, types.rs, error.rs, gateway.rs, message.rs, utils.rs
├── crew-llm/src/
│   ├── lib.rs, provider.rs, config.rs, types.rs, retry.rs, failover.rs, sse.rs
│   ├── embedding.rs, pricing.rs, context.rs, transcription.rs, vision.rs
│   ├── anthropic.rs, openai.rs, gemini.rs, openrouter.rs
├── crew-memory/src/
│   ├── lib.rs, episode.rs, store.rs, memory_store.rs, hybrid_search.rs
├── crew-agent/src/
│   ├── lib.rs, agent.rs, progress.rs, policy.rs, compaction.rs, sanitize.rs, hooks.rs
│   ├── sandbox.rs, mcp.rs, skills.rs, builtin_skills.rs
│   ├── plugins/ (mod.rs, loader.rs, manifest.rs, tool.rs)
│   ├── skills/ (cron, github, skill-creator, summarize, tmux, weather SKILL.md)
│   └── tools/ (mod, policy, shell, read_file, write_file, edit_file, diff_edit,
│               list_dir, glob_tool, grep_tool, web_search, web_fetch,
│               message, spawn, browser)
├── crew-bus/src/
│   ├── lib.rs, bus.rs, channel.rs, session.rs, coalesce.rs, media.rs
│   ├── cli_channel.rs, telegram_channel.rs, discord_channel.rs
│   ├── slack_channel.rs, whatsapp_channel.rs, feishu_channel.rs, email_channel.rs
│   ├── cron_service.rs, cron_types.rs, heartbeat.rs
└── crew-cli/src/
    ├── main.rs, config.rs, config_watcher.rs, cron_tool.rs, compaction.rs
    ├── auth/ (mod.rs, store.rs, oauth.rs, token.rs)
    ├── api/ (mod.rs, router.rs, handlers.rs, sse.rs, metrics.rs, static_files.rs)
    └── commands/ (mod, chat, init, status, gateway, clean,
                   completions, cron, channels, auth, skills, docs, serve)
```

---

## Security

### Workspace-Level Safety
- `#![deny(unsafe_code)]` — workspace-wide lint via `[workspace.lints.rust]`
- `secrecy::SecretString` — all provider API keys are wrapped; prevents accidental logging/display

### Authentication & Credentials
- API keys: auth store (`~/.crew/auth.json`, mode 0600) checked before env vars
- OAuth PKCE with SHA-256 challenges, state parameter (CSRF protection)
- Constant-time byte comparison for API bearer tokens (timing attack prevention)

### Execution Sandbox
- Three backends: bwrap (Linux), sandbox-exec (macOS), Docker — `SandboxMode::Auto` detection
- 18 BLOCKED_ENV_VARS shared across all sandbox backends, MCP server spawning, and browser tool
- Path injection prevention per backend (Docker: `:`, `\0`, `\n`, `\r`; macOS: control chars, `(`, `)`, `\`, `"`)
- Docker: `--cap-drop ALL`, `--security-opt no-new-privileges`, `--network none`

### Tool Safety
- ShellTool SafePolicy: deny `rm -rf /`, `dd`, `mkfs`, fork bombs; ask for `sudo`, `git push --force`
- Tool policies: allow/deny with deny-wins semantics, group support, provider-specific filtering
- Path traversal prevention + symlink rejection in all file tools
- SSRF protection in web_fetch and browser: blocks private IPs (10/8, 172.16/12, 192.168/16, 169.254/16, IPv6 ULA/link-local)
- Browser: URL scheme allowlist (http/https only), 10s JS execution timeout, zombie process reaping, secure tempfiles for screenshots

### Data Safety
- Tool output sanitization: strips base64 data URIs and long hex strings (`sanitize.rs`)
- UTF-8 safe truncation via `truncate_utf8()` across all tool outputs and email bodies
- Session file collision prevention via percent-encoded filenames
- Atomic write-then-rename for session persistence (crash safety)
- Channel access control via `allowed_senders` lists
- MCP response limit: 1MB per JSON-RPC line (DoS prevention)
- Message coalescing: MAX_CHUNKS=50 DoS limit
- API message limit: 1MB per request

---

## Testing

253+ tests across all crates:
- **Unit**: type serde round-trips, tool arg parsing, config validation, provider detection, tool policies, compaction, coalescing, BM25 scoring, L2 normalization, SSE parsing
- **Integration**: CLI commands, file tools, session persistence, cron jobs, session forking, plugin loading
- **Security**: sandbox path injection, env sanitization, SSRF blocking, symlink rejection, private IP detection, dedup overflow
- **Channel**: allowed_senders, message parsing, dedup logic, email address extraction
