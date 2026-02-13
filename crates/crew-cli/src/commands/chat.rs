//! Chat command: interactive multi-turn conversation with an agent.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Args;
use colored::Colorize;
use crew_agent::{Agent, AgentConfig, ConsoleReporter, ToolRegistry};
use crew_core::{AgentId, Message, MessageRole};
use crew_llm::{
    LlmProvider, RetryProvider, anthropic::AnthropicProvider, gemini::GeminiProvider,
    openai::OpenAIProvider, openrouter::OpenRouterProvider,
};
use crew_memory::EpisodeStore;
use eyre::{Result, WrapErr};
use rustyline::DefaultEditor;

use super::Executable;
use crate::config::Config;

/// Interactive multi-turn chat with an agent.
#[derive(Debug, Args)]
pub struct ChatCommand {
    /// Working directory (defaults to current directory).
    #[arg(short, long)]
    pub cwd: Option<PathBuf>,

    /// Path to config file.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// LLM provider to use (overrides config).
    #[arg(long)]
    pub provider: Option<String>,

    /// Model to use (overrides config).
    #[arg(long)]
    pub model: Option<String>,

    /// Custom base URL for the API endpoint (overrides config).
    #[arg(long)]
    pub base_url: Option<String>,

    /// Maximum tool-call iterations per message (default: 20).
    #[arg(long, default_value = "20")]
    pub max_iterations: u32,

    /// Verbose output (show tool outputs).
    #[arg(short, long)]
    pub verbose: bool,

    /// Disable automatic retry on transient errors.
    #[arg(long)]
    pub no_retry: bool,

    /// Send a single message and exit (non-interactive mode).
    #[arg(short, long)]
    pub message: Option<String>,
}

/// Exit commands.
const EXIT_COMMANDS: &[&str] = &["exit", "quit", "/exit", "/quit", ":q"];

impl Executable for ChatCommand {
    fn execute(self) -> Result<()> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .wrap_err("failed to create tokio runtime")?
            .block_on(self.run_async())
    }
}

impl ChatCommand {
    async fn run_async(self) -> Result<()> {
        let cwd = self.cwd.unwrap_or_else(|| std::env::current_dir().unwrap());

        // Load config
        let config = if let Some(config_path) = &self.config {
            Config::from_file(config_path)?
        } else {
            Config::load(&cwd)?
        };

        let model = self.model.or(config.model.clone());
        let base_url = self.base_url.or(config.base_url.clone());
        let provider_name = self
            .provider
            .or(config.provider.clone())
            .or_else(|| {
                model
                    .as_deref()
                    .and_then(crate::config::detect_provider)
                    .map(String::from)
            })
            .unwrap_or_else(|| "anthropic".to_string());

        // Create LLM provider
        let base_provider: Arc<dyn LlmProvider> =
            create_provider(&provider_name, &config, model, base_url)?;

        let llm: Arc<dyn LlmProvider> = if self.no_retry {
            base_provider
        } else {
            Arc::new(RetryProvider::new(base_provider))
        };

        // Create stores
        let data_dir = cwd.join(".crew");
        let memory = Arc::new(
            EpisodeStore::open(&data_dir)
                .await
                .wrap_err("failed to open episode store")?,
        );

        // Create tool registry
        let tools = ToolRegistry::with_builtins(&cwd);

        // Set up Ctrl+C handler
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        tokio::spawn(async move {
            if let Ok(()) = tokio::signal::ctrl_c().await {
                shutdown_clone.store(true, Ordering::Relaxed);
            }
        });

        // Create agent
        let reporter = Arc::new(ConsoleReporter::new().with_verbose(self.verbose));
        let agent_config = AgentConfig {
            max_iterations: self.max_iterations,
            max_tokens: None,
            save_episodes: false,
        };
        let agent = Agent::new(
            AgentId::new("chat"),
            llm,
            tools,
            memory,
        )
        .with_config(agent_config)
        .with_reporter(reporter)
        .with_shutdown(shutdown.clone());

        // Single-message mode: send one message and exit
        if let Some(msg) = self.message {
            let response = agent.process_message(&msg, &[], vec![]).await?;
            if !response.streamed {
                println!("{}", response.content);
            }
            return Ok(());
        }

        // Set up readline
        let history_dir = data_dir.join("history");
        std::fs::create_dir_all(&history_dir).ok();
        let history_path = history_dir.join("chat_history");

        let mut rl = DefaultEditor::new().wrap_err("failed to initialize readline")?;
        let _ = rl.load_history(&history_path);

        // Banner
        println!("{}", "crew-rs chat".cyan().bold());
        println!(
            "{}",
            "(type /exit or Ctrl+C to quit)".dimmed()
        );
        println!();

        // Conversation history
        let mut history: Vec<Message> = Vec::new();

        // Interactive loop
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let line = match rl.readline("you> ") {
                Ok(line) => line,
                Err(rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof) => {
                    break;
                }
                Err(e) => {
                    eprintln!("Input error: {e}");
                    break;
                }
            };

            let input = line.trim();
            if input.is_empty() {
                continue;
            }

            rl.add_history_entry(input).ok();

            if EXIT_COMMANDS.contains(&input.to_lowercase().as_str()) {
                break;
            }

            // Process message
            let response = agent.process_message(input, &history, vec![]).await?;

            // Append to history
            history.push(Message {
                role: MessageRole::User,
                content: input.to_string(),
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                timestamp: chrono::Utc::now(),
            });
            history.push(Message {
                role: MessageRole::Assistant,
                content: response.content.clone(),
                media: vec![],
                tool_calls: None,
                tool_call_id: None,
                timestamp: chrono::Utc::now(),
            });

            // Print response (skip if already streamed to console)
            if !response.streamed {
                println!();
                println!("{}: {}", "assistant".blue().bold(), response.content);
            }
            println!();
        }

        // Save history
        let _ = rl.save_history(&history_path);
        println!("{}", "Goodbye!".dimmed());

        Ok(())
    }
}

/// Create an LLM provider from name and config.
fn create_provider(
    name: &str,
    config: &Config,
    model: Option<String>,
    base_url: Option<String>,
) -> Result<Arc<dyn LlmProvider>> {
    let provider: Arc<dyn LlmProvider> = match name {
        "anthropic" => {
            let api_key = config.get_api_key("anthropic")?;
            let model_name = model.unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());
            let mut p = AnthropicProvider::new(&api_key, &model_name);
            if let Some(url) = &base_url {
                p = p.with_base_url(url);
            }
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "openai" => {
            let api_key = config.get_api_key("openai")?;
            let model_name = model.unwrap_or_else(|| "gpt-4o".to_string());
            let mut p = OpenAIProvider::new(&api_key, &model_name);
            if let Some(url) = &base_url {
                p = p.with_base_url(url);
            }
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "gemini" | "google" => {
            let api_key = config.get_api_key("gemini")?;
            let model_name = model.unwrap_or_else(|| "gemini-2.0-flash".to_string());
            let mut p = GeminiProvider::new(&api_key, &model_name);
            if let Some(url) = &base_url {
                p = p.with_base_url(url);
            }
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "openrouter" => {
            let api_key = config.get_api_key("openrouter")?;
            let model_name =
                model.unwrap_or_else(|| "anthropic/claude-sonnet-4-20250514".to_string());
            let mut p = OpenRouterProvider::new(&api_key, &model_name);
            if let Some(url) = &base_url {
                p = p.with_base_url(url);
            }
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "deepseek" => {
            let api_key = config.get_api_key("deepseek")?;
            let model_name = model.unwrap_or_else(|| "deepseek-chat".to_string());
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(
                base_url.as_deref().unwrap_or("https://api.deepseek.com/v1"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "groq" => {
            let api_key = config.get_api_key("groq")?;
            let model_name = model.unwrap_or_else(|| "llama-3.3-70b-versatile".to_string());
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(
                base_url.as_deref().unwrap_or("https://api.groq.com/openai/v1"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "moonshot" | "kimi" => {
            let api_key = config.get_api_key("moonshot")?;
            let model_name = model.unwrap_or_else(|| "kimi-k2.5".to_string());
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(
                base_url.as_deref().unwrap_or("https://api.moonshot.ai/v1"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "dashscope" | "qwen" => {
            let api_key = config.get_api_key("dashscope")?;
            let model_name = model.unwrap_or_else(|| "qwen-max".to_string());
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(
                base_url
                    .as_deref()
                    .unwrap_or("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "minimax" => {
            let api_key = config.get_api_key("minimax")?;
            let model_name = model.unwrap_or_else(|| "MiniMax-Text-01".to_string());
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(
                base_url.as_deref().unwrap_or("https://api.minimax.io/v1"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "zhipu" | "glm" => {
            let api_key = config.get_api_key("zhipu")?;
            let model_name = model.unwrap_or_else(|| "glm-4-plus".to_string());
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(
                base_url
                    .as_deref()
                    .unwrap_or("https://open.bigmodel.cn/api/paas/v4"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "ollama" => {
            let model_name = model.unwrap_or_else(|| "llama3.2".to_string());
            let p = OpenAIProvider::new("ollama", &model_name).with_base_url(
                base_url.as_deref().unwrap_or("http://localhost:11434/v1"),
            );
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        "vllm" => {
            let api_key = config
                .get_api_key("vllm")
                .unwrap_or_else(|_| "token".to_string());
            let model_name = model.ok_or_else(|| {
                eyre::eyre!("vllm provider requires --model to be specified")
            })?;
            let url = base_url.ok_or_else(|| {
                eyre::eyre!("vllm provider requires --base-url to be specified")
            })?;
            let p = OpenAIProvider::new(&api_key, &model_name).with_base_url(&url);
            println!("{}: {}", "Model".green(), p.model_id());
            Arc::new(p)
        }
        other => {
            eyre::bail!(
                "unknown provider: {other}. Valid: anthropic, openai, gemini, openrouter, \
                 deepseek, groq, moonshot, dashscope, minimax, zhipu, ollama, vllm"
            );
        }
    };
    Ok(provider)
}
