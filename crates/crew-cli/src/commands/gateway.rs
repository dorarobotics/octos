//! Gateway command: run as a persistent messaging daemon.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use clap::Args;
use colored::Colorize;
use crew_agent::{Agent, AgentConfig, MessageTool, SilentReporter, SkillsLoader, SpawnTool, ToolRegistry};
use crew_bus::{ChannelManager, CliChannel, CronService, HeartbeatService, SessionManager, create_bus};
use crew_core::{AgentId, Message, MessageRole, OutboundMessage};
use crew_llm::{
    GroqTranscriber, LlmProvider, RetryProvider, anthropic::AnthropicProvider,
    gemini::GeminiProvider, openai::OpenAIProvider, openrouter::OpenRouterProvider,
};
use crew_memory::{EpisodeStore, MemoryStore};
use eyre::{Result, WrapErr};
use tracing::{info, warn};

use std::path::Path;

use super::Executable;
use crate::config::{Config, detect_provider};
use crate::cron_tool::CronTool;

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

        let model = self.model.or(config.model.clone());
        let base_url = self.base_url.or(config.base_url.clone());
        let provider_name = self
            .provider
            .or(config.provider.clone())
            .or_else(|| {
                model.as_deref().and_then(detect_provider).map(String::from)
            })
            .unwrap_or_else(|| "anthropic".to_string());

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
            // OpenAI-compatible providers (same API, different base URL)
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
                let model_name =
                    model.unwrap_or_else(|| "llama-3.3-70b-versatile".to_string());
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
            other => {
                eyre::bail!(
                    "unknown provider: {other}. Valid: anthropic, openai, gemini, openrouter, \
                     deepseek, groq, moonshot, dashscope, minimax, zhipu, ollama, vllm"
                );
            }
        };

        let llm: Arc<dyn LlmProvider> = if self.no_retry {
            base_provider
        } else {
            Arc::new(RetryProvider::new(base_provider))
        };

        let data_dir = cwd.join(".crew");
        let media_dir = data_dir.join("media");
        let _ = &media_dir; // used by channel feature gates below

        // Create voice transcriber if GROQ_API_KEY is set
        let transcriber = std::env::var("GROQ_API_KEY")
            .ok()
            .map(|key| {
                println!("{}: Groq Whisper", "Transcriber".green());
                GroqTranscriber::new(key)
            });

        let memory = Arc::new(
            EpisodeStore::open(&data_dir)
                .await
                .wrap_err("failed to open episode store")?,
        );

        // Initialize memory store
        let memory_store = MemoryStore::open(&data_dir)
            .await
            .wrap_err("failed to open memory store")?;

        // Initialize skills loader
        let skills_loader = SkillsLoader::new(&data_dir);

        // Create message bus (before publisher is consumed by channel manager)
        let (mut agent_handle, publisher) = create_bus();

        // Clone senders before publisher is consumed
        let cron_inbound_tx = publisher.inbound_sender();
        let heartbeat_inbound_tx = publisher.inbound_sender();
        let spawn_inbound_tx = publisher.inbound_sender();
        let out_tx = agent_handle.outbound_sender();

        // Initialize cron service
        let cron_service = Arc::new(CronService::new(
            data_dir.join("cron.json"),
            cron_inbound_tx,
        ));
        cron_service.start();

        // Initialize heartbeat service
        let heartbeat_service = Arc::new(HeartbeatService::new(
            &cwd,
            heartbeat_inbound_tx,
            crew_bus::heartbeat::DEFAULT_INTERVAL_SECS,
        ));
        heartbeat_service.start();

        // Build tool registry
        let mut tools = ToolRegistry::with_builtins(&cwd);
        tools.register(CronTool::new(cron_service.clone()));

        // Message tool (cross-channel messaging)
        let message_tool = Arc::new(MessageTool::new(out_tx));
        tools.register_arc(message_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Spawn tool (background subagents)
        let spawn_tool = Arc::new(SpawnTool::new(
            llm.clone(),
            memory.clone(),
            cwd.clone(),
            spawn_inbound_tx,
        ));
        tools.register_arc(spawn_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Build enhanced system prompt
        let system_prompt = build_system_prompt(
            gw_config.system_prompt.as_deref(),
            &data_dir,
            &memory_store,
            &skills_loader,
        )
        .await;

        // Build the agent
        let agent_config = AgentConfig {
            max_iterations: self.max_iterations,
            max_tokens: None,
            save_episodes: false,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let llm_for_compaction = llm.clone();
        let agent = Agent::new(
            AgentId::new("gateway"),
            llm,
            tools,
            memory,
        )
        .with_config(agent_config)
        .with_reporter(Arc::new(SilentReporter))
        .with_shutdown(shutdown.clone())
        .with_system_prompt(system_prompt);

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
                        media_dir.clone(),
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
                        media_dir.clone(),
                    )));
                }
                #[cfg(feature = "slack")]
                "slack" => {
                    let bot_env =
                        settings_str(&entry.settings, "bot_token_env", "SLACK_BOT_TOKEN");
                    let app_env =
                        settings_str(&entry.settings, "app_token_env", "SLACK_APP_TOKEN");
                    let bot_token = std::env::var(&bot_env)
                        .wrap_err_with(|| format!("{bot_env} environment variable not set"))?;
                    let app_token = std::env::var(&app_env)
                        .wrap_err_with(|| format!("{app_env} environment variable not set"))?;
                    channel_mgr.register(Arc::new(crew_bus::SlackChannel::new(
                        &bot_token,
                        &app_token,
                        entry.allowed_senders.clone(),
                        shutdown.clone(),
                        media_dir.clone(),
                    )));
                }
                #[cfg(feature = "whatsapp")]
                "whatsapp" => {
                    let url =
                        settings_str(&entry.settings, "bridge_url", "ws://localhost:3001");
                    channel_mgr.register(Arc::new(crew_bus::WhatsAppChannel::new(
                        &url,
                        entry.allowed_senders.clone(),
                        shutdown.clone(),
                    )));
                }
                #[cfg(feature = "email")]
                "email" => {
                    let imap_host = settings_str(&entry.settings, "imap_host", "");
                    let imap_port = entry
                        .settings
                        .get("imap_port")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(993) as u16;
                    let smtp_host = settings_str(&entry.settings, "smtp_host", "");
                    let smtp_port = entry
                        .settings
                        .get("smtp_port")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(465) as u16;
                    let user_env =
                        settings_str(&entry.settings, "username_env", "EMAIL_USERNAME");
                    let pass_env =
                        settings_str(&entry.settings, "password_env", "EMAIL_PASSWORD");
                    let username = std::env::var(&user_env)
                        .wrap_err_with(|| format!("{user_env} not set"))?;
                    let password = std::env::var(&pass_env)
                        .wrap_err_with(|| format!("{pass_env} not set"))?;
                    let from_address =
                        settings_str(&entry.settings, "from_address", &username);
                    let poll_interval = entry
                        .settings
                        .get("poll_interval_secs")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(30);
                    let max_body_chars = entry
                        .settings
                        .get("max_body_chars")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10000) as usize;

                    let email_config = crew_bus::email_channel::EmailConfig {
                        imap_host,
                        imap_port,
                        smtp_host,
                        smtp_port,
                        username,
                        password,
                        from_address,
                        poll_interval_secs: poll_interval,
                        allowed_senders: entry.allowed_senders.clone(),
                        max_body_chars,
                    };
                    channel_mgr.register(Arc::new(crew_bus::EmailChannel::new(
                        email_config,
                        shutdown.clone(),
                    )));
                }
                #[cfg(feature = "feishu")]
                "feishu" | "lark" => {
                    let id_env =
                        settings_str(&entry.settings, "app_id_env", "FEISHU_APP_ID");
                    let secret_env =
                        settings_str(&entry.settings, "app_secret_env", "FEISHU_APP_SECRET");
                    let app_id = std::env::var(&id_env)
                        .wrap_err_with(|| format!("{id_env} environment variable not set"))?;
                    let app_secret = std::env::var(&secret_env)
                        .wrap_err_with(|| {
                            format!("{secret_env} environment variable not set")
                        })?;
                    channel_mgr.register(Arc::new(crew_bus::FeishuChannel::new(
                        &app_id,
                        &app_secret,
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
        while let Some(mut inbound) = agent_handle.recv_inbound().await {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Transcribe audio media and separate images
            let mut image_media = Vec::new();
            if let Some(ref transcriber) = transcriber {
                for path in &inbound.media {
                    if crew_bus::media::is_audio(path) {
                        match transcriber.transcribe(std::path::Path::new(path)).await {
                            Ok(text) => {
                                let prefix = format!("[Voice transcription: {text}]\n\n");
                                inbound.content = format!("{prefix}{}", inbound.content);
                            }
                            Err(e) => warn!("transcription failed: {e}"),
                        }
                    } else if crew_bus::media::is_image(path) {
                        image_media.push(path.clone());
                    }
                }
            } else {
                // No transcriber: just keep image media
                image_media = inbound
                    .media
                    .iter()
                    .filter(|p| crew_bus::media::is_image(p))
                    .cloned()
                    .collect();
            }

            // Route cron-triggered messages to their target channel
            let (reply_channel, reply_chat_id) = if inbound.channel == "system" {
                let ch = inbound
                    .metadata
                    .get("deliver_to_channel")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cli")
                    .to_string();
                let cid = inbound
                    .metadata
                    .get("deliver_to_chat_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&inbound.chat_id)
                    .to_string();
                (ch, cid)
            } else {
                (inbound.channel.clone(), inbound.chat_id.clone())
            };

            // Update per-message context for tools
            message_tool.set_context(&reply_channel, &reply_chat_id);
            spawn_tool.set_context(&reply_channel, &reply_chat_id);

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

            // Process message through agent (with images for vision)
            let response = agent
                .process_message(&inbound.content, &history, image_media)
                .await;

            match response {
                Ok(conv_response) => {
                    // Save user message to session
                    let user_msg = Message {
                        role: MessageRole::User,
                        content: inbound.content.clone(),
                        media: vec![],
                        tool_calls: None,
                        tool_call_id: None,
                        timestamp: Utc::now(),
                    };
                    let _ = session_mgr.add_message(&session_key, user_msg);

                    // Save assistant response to session
                    let assistant_msg = Message {
                        role: MessageRole::Assistant,
                        content: conv_response.content.clone(),
                        media: vec![],
                        tool_calls: None,
                        tool_call_id: None,
                        timestamp: Utc::now(),
                    };
                    let _ = session_mgr.add_message(&session_key, assistant_msg);

                    // Compact session if it's grown too large
                    if let Err(e) =
                        crate::compaction::maybe_compact(&mut session_mgr, &session_key, &*llm_for_compaction)
                            .await
                    {
                        warn!("session compaction failed: {e}");
                    }

                    // Send response back through channel
                    let outbound = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
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
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
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

        heartbeat_service.stop().await;
        cron_service.stop().await;
        channel_mgr.stop_all().await?;
        println!("{}", "Gateway stopped.".dimmed());
        Ok(())
    }
}

/// Build the system prompt with bootstrap files, memory context, and skills.
async fn build_system_prompt(
    base: Option<&str>,
    data_dir: &Path,
    memory_store: &MemoryStore,
    skills_loader: &SkillsLoader,
) -> String {
    let mut prompt = base.unwrap_or("You are a helpful AI assistant.").to_string();

    // Append bootstrap files (AGENTS.md, SOUL.md, USER.md, etc.)
    let bootstrap = load_bootstrap_files(data_dir);
    if !bootstrap.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&bootstrap);
    }

    // Append memory context
    let memory_ctx = memory_store.get_memory_context().await;
    if !memory_ctx.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&memory_ctx);
    }

    // Append always-on skills
    if let Ok(always_names) = skills_loader.get_always_skills().await {
        if !always_names.is_empty() {
            if let Ok(skills_content) = skills_loader.load_skills_for_context(&always_names).await {
                if !skills_content.is_empty() {
                    prompt.push_str("\n\n## Active Skills\n\n");
                    prompt.push_str(&skills_content);
                }
            }
        }
    }

    // Append skills summary
    if let Ok(summary) = skills_loader.build_skills_summary().await {
        if !summary.is_empty() {
            prompt.push_str("\n\n## Available Skills\n\n");
            prompt.push_str(&summary);
        }
    }

    prompt
}

/// Extract a string value from channel settings JSON, with a default fallback.
#[cfg(any(
    feature = "telegram",
    feature = "discord",
    feature = "slack",
    feature = "whatsapp",
    feature = "email",
    feature = "feishu"
))]
fn settings_str(settings: &serde_json::Value, key: &str, default: &str) -> String {
    settings
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

/// Load optional bootstrap/personality files from the .crew/ directory.
fn load_bootstrap_files(data_dir: &Path) -> String {
    const FILES: &[&str] = &[
        "AGENTS.md",
        "SOUL.md",
        "USER.md",
        "TOOLS.md",
        "IDENTITY.md",
    ];
    let mut parts = Vec::new();
    for filename in FILES {
        let path = data_dir.join(filename);
        if let Ok(content) = std::fs::read_to_string(&path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                parts.push(format!("## {filename}\n\n{trimmed}"));
            }
        }
    }
    parts.join("\n\n")
}
