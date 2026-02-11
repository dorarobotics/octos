//! Gateway command: run as a persistent messaging daemon.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use clap::Args;
use colored::Colorize;
use crew_agent::{Agent, AgentConfig, SilentReporter, ToolRegistry};
use crew_bus::{ChannelManager, CliChannel, SessionManager, create_bus};
use crew_core::{AgentId, AgentRole, Message, MessageRole, OutboundMessage};
use crew_llm::{
    LlmProvider, RetryProvider, anthropic::AnthropicProvider, gemini::GeminiProvider,
    openai::OpenAIProvider,
};
use crew_memory::EpisodeStore;
use eyre::{Result, WrapErr};
use tracing::info;

use super::Executable;
use crate::config::Config;

/// Run as a persistent gateway daemon.
#[derive(Debug, Args)]
pub struct GatewayCommand {
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

    /// Maximum agent iterations per message (default: 50).
    #[arg(long, default_value = "50")]
    pub max_iterations: u32,

    /// Disable automatic retry on transient errors.
    #[arg(long)]
    pub no_retry: bool,
}

impl Executable for GatewayCommand {
    fn execute(self) -> Result<()> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .wrap_err("failed to create tokio runtime")?
            .block_on(self.run_async())
    }
}

impl GatewayCommand {
    async fn run_async(self) -> Result<()> {
        println!("{}", "crew gateway".cyan().bold());
        println!();

        let cwd = self.cwd.unwrap_or_else(|| std::env::current_dir().unwrap());

        let config = if let Some(config_path) = &self.config {
            Config::from_file(config_path)?
        } else {
            Config::load(&cwd)?
        };

        let provider_name = self
            .provider
            .or(config.provider.clone())
            .unwrap_or_else(|| "anthropic".to_string());
        let model = self.model.or(config.model.clone());
        let base_url = self.base_url.or(config.base_url.clone());

        let gw_config = config
            .gateway
            .clone()
            .unwrap_or_else(|| crate::config::GatewayConfig {
                channels: vec![crate::config::ChannelEntry {
                    channel_type: "cli".into(),
                    allowed_senders: vec![],
                    settings: serde_json::json!({}),
                }],
                max_history: 50,
                system_prompt: None,
            });

        println!("{}: {}", "Provider".green(), provider_name);

        // Create LLM provider (same pattern as RunCommand)
        let base_provider: Arc<dyn LlmProvider> = match provider_name.as_str() {
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
            other => {
                eyre::bail!(
                    "unknown provider: {}. Use 'anthropic', 'openai', or 'gemini'",
                    other
                );
            }
        };

        let llm: Arc<dyn LlmProvider> = if self.no_retry {
            base_provider
        } else {
            Arc::new(RetryProvider::new(base_provider))
        };

        let data_dir = cwd.join(".crew");
        let memory = Arc::new(
            EpisodeStore::open(&data_dir)
                .await
                .wrap_err("failed to open episode store")?,
        );

        let tools = ToolRegistry::with_builtins(&cwd);

        // Build the agent
        let agent_config = AgentConfig {
            max_iterations: self.max_iterations,
            max_tokens: None,
            save_episodes: false,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let mut agent = Agent::new(
            AgentId::new("gateway"),
            AgentRole::Worker,
            llm,
            tools,
            memory,
        )
        .with_config(agent_config)
        .with_reporter(Arc::new(SilentReporter))
        .with_shutdown(shutdown.clone());

        if let Some(ref prompt) = gw_config.system_prompt {
            agent = agent.with_system_prompt(prompt.clone());
        }

        // Create message bus
        let (mut agent_handle, publisher) = create_bus();

        // Create session manager
        let mut session_mgr =
            SessionManager::open(&data_dir).wrap_err("failed to open session manager")?;

        // Create channel manager and register channels
        let mut channel_mgr = ChannelManager::new();
        for entry in &gw_config.channels {
            match entry.channel_type.as_str() {
                "cli" => {
                    channel_mgr.register(Arc::new(CliChannel::new(shutdown.clone())));
                }
                #[cfg(feature = "telegram")]
                "telegram" => {
                    let env = settings_str(&entry.settings, "token_env", "TELEGRAM_BOT_TOKEN");
                    let token = std::env::var(&env)
                        .wrap_err_with(|| format!("{env} environment variable not set"))?;
                    channel_mgr.register(Arc::new(crew_bus::TelegramChannel::new(
                        &token,
                        entry.allowed_senders.clone(),
                        shutdown.clone(),
                    )));
                }
                #[cfg(feature = "discord")]
                "discord" => {
                    let env = settings_str(&entry.settings, "token_env", "DISCORD_BOT_TOKEN");
                    let token = std::env::var(&env)
                        .wrap_err_with(|| format!("{env} environment variable not set"))?;
                    channel_mgr.register(Arc::new(crew_bus::DiscordChannel::new(
                        &token,
                        entry.allowed_senders.clone(),
                        shutdown.clone(),
                    )));
                }
                other => {
                    println!(
                        "{}: channel '{}' not supported, skipping",
                        "Warning".yellow(),
                        other
                    );
                }
            }
        }

        // Start channels and dispatcher
        channel_mgr.start_all(publisher).await?;

        // Set up Ctrl+C handler
        tokio::spawn(async move {
            if let Ok(()) = tokio::signal::ctrl_c().await {
                println!();
                println!("{}", "Shutting down gateway...".yellow());
                shutdown_clone.store(true, Ordering::Relaxed);
            }
        });

        println!("{}: {}", "Max history".green(), gw_config.max_history);
        println!();
        println!(
            "{}",
            "Gateway ready. Type a message or /quit to exit.".dimmed()
        );
        println!();

        // Main loop: process inbound messages
        while let Some(inbound) = agent_handle.recv_inbound().await {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let session_key = inbound.session_key();
            info!(
                channel = %inbound.channel,
                sender = %inbound.sender_id,
                session = %session_key,
                "processing message"
            );

            // Get conversation history
            let session = session_mgr.get_or_create(&session_key);
            let history: Vec<Message> = session.get_history(gw_config.max_history).to_vec();

            // Process message through agent
            let response = agent.process_message(&inbound.content, &history).await;

            match response {
                Ok(conv_response) => {
                    // Save user message to session
                    let user_msg = Message {
                        role: MessageRole::User,
                        content: inbound.content.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        timestamp: Utc::now(),
                    };
                    let _ = session_mgr.add_message(&session_key, user_msg);

                    // Save assistant response to session
                    let assistant_msg = Message {
                        role: MessageRole::Assistant,
                        content: conv_response.content.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        timestamp: Utc::now(),
                    };
                    let _ = session_mgr.add_message(&session_key, assistant_msg);

                    // Send response back through channel
                    let outbound = OutboundMessage {
                        channel: inbound.channel.clone(),
                        chat_id: inbound.chat_id.clone(),
                        content: conv_response.content,
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };

                    if agent_handle.send_outbound(outbound).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let error_msg = OutboundMessage {
                        channel: inbound.channel.clone(),
                        chat_id: inbound.chat_id.clone(),
                        content: format!("Error: {e}"),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    if agent_handle.send_outbound(error_msg).await.is_err() {
                        break;
                    }
                }
            }
        }

        channel_mgr.stop_all().await?;
        println!("{}", "Gateway stopped.".dimmed());
        Ok(())
    }
}

/// Extract a string value from channel settings JSON, with a default fallback.
#[cfg(any(feature = "telegram", feature = "discord"))]
fn settings_str(settings: &serde_json::Value, key: &str, default: &str) -> String {
    settings
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}
