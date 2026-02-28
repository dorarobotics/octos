//! Gateway command: run as a persistent messaging daemon.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use clap::Args;
use colored::Colorize;
use crew_agent::{
    Agent, AgentConfig, HookExecutor, MessageTool, SendFileTool, SilentReporter, SkillsLoader,
    SpawnTool, TakePhotoTool, ToolRegistry,
};
use crew_bus::{
    ChannelManager, CliChannel, CronService, HeartbeatService, SessionManager, create_bus,
};
use crew_core::{AgentId, Message, MessageRole, OutboundMessage, SessionKey};
use crew_llm::{GroqTranscriber, LlmProvider, ProviderChain, ProviderRouter, RetryProvider};
use crew_memory::{EpisodeStore, MemoryStore};
use eyre::{Result, WrapErr};
use tokio::sync::{Mutex, Semaphore};
use tracing::{info, warn};

use std::path::Path;

use super::Executable;
use crate::commands::chat::{create_embedder, resolve_provider_policy};
use crate::config::{Config, QueueMode, detect_provider};
use crate::config_watcher::{ConfigChange, ConfigWatcher};
use crate::persona_service::PersonaService;
use crate::cron_tool::CronTool;

/// Run as a persistent gateway daemon.
#[derive(Debug, Args)]
pub struct GatewayCommand {
    /// Working directory (defaults to current directory).
    #[arg(short, long)]
    pub cwd: Option<PathBuf>,

    /// Data directory for episodes, memory, sessions (defaults to $CREW_HOME or ~/.crew).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Path to config file.
    #[arg(long, conflicts_with = "profile")]
    pub config: Option<PathBuf>,

    /// Path to a profile JSON file (used by managed gateways).
    #[arg(long, conflicts_with = "config")]
    pub profile: Option<PathBuf>,

    /// Override WhatsApp bridge URL (used by managed gateways).
    #[arg(long, hide = true)]
    pub bridge_url: Option<String>,

    /// Override Feishu webhook port (used by managed gateways).
    #[arg(long, hide = true)]
    pub feishu_port: Option<u16>,

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
    #[arg(long)]
    pub max_iterations: Option<u32>,

    /// Disable automatic retry on transient errors.
    #[arg(long)]
    pub no_retry: bool,
}

impl Executable for GatewayCommand {
    fn execute(self) -> Result<()> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(8 * 1024 * 1024) // 8MB stack for deep agent futures
            .build()
            .wrap_err("failed to create tokio runtime")?
            .block_on(self.run_async())
    }
}

impl GatewayCommand {
    async fn run_async(self) -> Result<()> {
        println!("{}", "crew gateway".cyan().bold());
        println!();

        let cwd = match self.cwd {
            Some(p) => p,
            None => std::env::current_dir().wrap_err("failed to get current directory")?,
        };

        let config = if let Some(ref profile_path) = self.profile {
            // Load config from profile JSON (single source of truth)
            let content = std::fs::read_to_string(profile_path)
                .wrap_err_with(|| format!("failed to read profile: {}", profile_path.display()))?;
            let profile: crate::profiles::UserProfile = serde_json::from_str(&content)
                .wrap_err_with(|| format!("failed to parse profile: {}", profile_path.display()))?;
            crate::profiles::config_from_profile(
                &profile,
                self.bridge_url.as_deref(),
                self.feishu_port,
            )
        } else if let Some(config_path) = &self.config {
            Config::from_file(config_path)?
        } else {
            Config::load(&cwd)?
        };

        let model = self.model.or(config.model.clone());
        let base_url = self.base_url.or(config.base_url.clone());
        let provider_name = self
            .provider
            .or(config.provider.clone())
            .or_else(|| model.as_deref().and_then(detect_provider).map(String::from))
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
                queue_mode: QueueMode::default(),
                max_sessions: 1000,
                max_concurrent_sessions: 10,
                browser_timeout_secs: None,
            });

        println!("{}: {}", "Provider".green(), provider_name);

        // Create LLM provider (reuses the shared create_provider from chat.rs)
        use super::chat::create_provider;
        let base_provider = create_provider(&provider_name, &config, model, base_url)?;

        let model_id = base_provider.model_id().to_string();
        let llm: Arc<dyn LlmProvider> = if self.no_retry {
            base_provider
        } else if config.fallback_models.is_empty() {
            Arc::new(RetryProvider::new(base_provider))
        } else {
            let mut providers: Vec<Arc<dyn LlmProvider>> =
                vec![Arc::new(RetryProvider::new(base_provider))];
            for fb in &config.fallback_models {
                let fb_config = if fb.api_key_env.is_some() {
                    let mut c = config.clone();
                    c.api_key_env = fb.api_key_env.clone();
                    c
                } else {
                    config.clone()
                };
                match super::chat::create_provider_with_api_type(
                    &fb.provider,
                    &fb_config,
                    fb.model.clone(),
                    fb.base_url.clone(),
                    fb.api_type.as_deref(),
                ) {
                    Ok(p) => providers.push(Arc::new(RetryProvider::new(p))),
                    Err(e) => {
                        warn!(provider = %fb.provider, error = %e, "skipping fallback provider");
                    }
                }
            }
            Arc::new(ProviderChain::new(providers))
        };

        // Resolve data directory (--data-dir > $CREW_HOME > ~/.crew)
        let data_dir = super::resolve_data_dir(self.data_dir)?;
        let media_dir = data_dir.join("media");
        let _ = &media_dir; // used by channel feature gates below

        // Create voice transcriber if GROQ_API_KEY is set
        let transcriber = std::env::var("GROQ_API_KEY").ok().map(|key| {
            println!("{}: Groq Whisper", "Transcriber".green());
            GroqTranscriber::new(key)
        });

        let memory = Arc::new(
            EpisodeStore::open(&data_dir)
                .await
                .wrap_err("failed to open episode store")?,
        );

        // Initialize memory store
        let memory_store = Arc::new(
            MemoryStore::open(&data_dir)
                .await
                .wrap_err("failed to open memory store")?,
        );

        // Initialize skills loader (project-level, from cwd/.crew/)
        let project_dir = cwd.join(".crew");
        let skills_loader = SkillsLoader::new(&project_dir);

        // Create message bus (before publisher is consumed by channel manager)
        let (mut agent_handle, publisher) = create_bus();

        // Clone senders before publisher is consumed
        let cron_inbound_tx = publisher.inbound_sender();
        let heartbeat_inbound_tx = publisher.inbound_sender();
        let spawn_inbound_tx = publisher.inbound_sender();
        let collect_inbound_tx = publisher.inbound_sender();
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

        // Build tool registry (with sandbox if configured)
        let sandbox = crew_agent::create_sandbox(&config.sandbox);
        let mut tools = ToolRegistry::with_builtins_and_sandbox(&cwd, sandbox);

        // Open tool config store for user-customizable tool defaults
        let tool_config = Arc::new(
            crew_agent::ToolConfigStore::open(&data_dir)
                .await
                .wrap_err("failed to open tool config store")?,
        );
        tools.inject_tool_config(tool_config.clone());

        tools.register(crew_agent::DeepSearchTool::new(data_dir.join("research")));
        tools.register(
            crew_agent::DeepCrawlTool::new(data_dir.join("research"))
                .with_config(tool_config.clone()),
        );
        tools.register(
            crew_agent::NewsDigestTool::new(llm.clone(), &data_dir)
                .with_config(tool_config.clone()),
        );

        // Override browser tool with configured timeout (replaces default 300s)
        if let Some(secs) = gw_config.browser_timeout_secs {
            tools.register(
                crew_agent::BrowserTool::with_timeout(std::time::Duration::from_secs(secs))
                    .with_config(tool_config.clone()),
            );
        }

        // Register MCP tools
        if !config.mcp_servers.is_empty() {
            match crew_agent::McpClient::start(&config.mcp_servers).await {
                Ok(client) => client.register_tools(&mut tools),
                Err(e) => warn!("MCP initialization failed: {e}"),
            }
        }

        // Load plugins
        let plugin_dirs = crate::config::Config::plugin_dirs(&cwd);
        if !plugin_dirs.is_empty() {
            if let Err(e) = crew_agent::PluginLoader::load_into(&mut tools, &plugin_dirs) {
                warn!("plugin loading failed: {e}");
            }
        }

        // Apply tool policy from config
        if let Some(ref policy) = config.tool_policy {
            tools.apply_policy(policy);
        }

        // Apply context-based tag filter
        if !config.context_filter.is_empty() {
            tools.set_context_filter(config.context_filter.clone());
        }

        // Apply provider-specific tool policy
        if let Some(policy) = resolve_provider_policy(&config, &provider_name, &model_id) {
            tools.set_provider_policy(policy);
        }

        let cron_tool = Arc::new(CronTool::new(cron_service.clone()));
        tools.register_arc(cron_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Message tool (cross-channel messaging)
        let message_tool = Arc::new(MessageTool::new(out_tx.clone()));
        tools.register_arc(message_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Send file tool (document attachments)
        let send_file_tool = Arc::new(SendFileTool::new(out_tx.clone()));
        tools.register_arc(send_file_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Take photo tool (camera capture + send)
        let take_photo_tool = Arc::new(TakePhotoTool::new(out_tx));
        tools.register_arc(take_photo_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Build sub-provider router from config
        let provider_router = if !config.sub_providers.is_empty() {
            let router = Arc::new(ProviderRouter::new());
            for sp in &config.sub_providers {
                // Clone config and override api_key_env if the sub-provider specifies one
                let sp_config = if sp.api_key_env.is_some() {
                    let mut c = config.clone();
                    c.api_key_env = sp.api_key_env.clone();
                    c
                } else {
                    config.clone()
                };
                match super::chat::create_provider_with_api_type(
                    &sp.provider,
                    &sp_config,
                    sp.model.clone(),
                    sp.base_url.clone(),
                    sp.api_type.as_deref(),
                ) {
                    Ok(p) => {
                        router.register_with_meta(
                            &sp.key,
                            Arc::new(RetryProvider::new(p)),
                            sp.description.clone(),
                            sp.default_context_window,
                        );
                        println!(
                            "  {}: {}/{}",
                            "Sub-provider".green(),
                            sp.key,
                            sp.model.as_deref().unwrap_or("default")
                        );
                    }
                    Err(e) => {
                        warn!(key = %sp.key, provider = %sp.provider, error = %e, "skipping sub-provider");
                    }
                }
            }
            Some(router)
        } else {
            None
        };

        // Spawn tool (background subagents, inherits provider policy + router)
        let mut spawn = SpawnTool::new(llm.clone(), memory.clone(), cwd.clone(), spawn_inbound_tx)
            .with_provider_policy(tools.provider_policy().cloned());
        if let Some(ref router) = provider_router {
            spawn = spawn.with_provider_router(router.clone());
        }
        let spawn_tool = Arc::new(spawn);
        tools.register_arc(spawn_tool.clone() as Arc<dyn crew_agent::Tool>);

        // Deep research tool with background notification channel
        let (research_tx, _research_rx) =
            tokio::sync::mpsc::channel::<crew_agent::ResearchNotification>(8);
        tools.register(
            crew_agent::DeepResearchTool::new(
                llm.clone(),
                memory.clone(),
                data_dir.clone(),
                research_tx,
            )
            .with_config(tool_config.clone()),
        );

        // Memory bank tools (recall/save entity pages)
        tools.register(crew_agent::RecallMemoryTool::new(memory_store.clone()));
        tools.register(crew_agent::SaveMemoryTool::new(memory_store.clone()));

        // Send email tool (configured via config.email)
        if let Some(ref email_cfg) = config.email {
            match build_email_sender(email_cfg) {
                Ok(sender) => {
                    tools.register(crew_agent::SendEmailTool::new(sender));
                    info!(
                        "send_email tool registered (provider: {})",
                        email_cfg.provider
                    );
                }
                Err(e) => warn!("skipping send_email tool: {e}"),
            }
        }

        // Build enhanced system prompt
        let system_prompt = build_system_prompt(
            gw_config.system_prompt.as_deref(),
            &data_dir,
            &project_dir,
            &memory_store,
            &skills_loader,
            &tool_config,
        )
        .await;

        // Build the agent
        let max_iterations = self.max_iterations.or(config.max_iterations).unwrap_or(50);
        let agent_config = AgentConfig {
            max_iterations,
            save_episodes: false,
            ..Default::default()
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let llm_for_compaction = llm.clone();
        let mut agent = Agent::new(AgentId::new("gateway"), llm, tools, memory)
            .with_config(agent_config)
            .with_reporter(Arc::new(SilentReporter))
            .with_shutdown(shutdown.clone())
            .with_system_prompt(system_prompt);

        if !config.hooks.is_empty() {
            agent = agent.with_hooks(Arc::new(HookExecutor::new(config.hooks.clone())));
        }

        if let Some(embedder) = create_embedder(&config) {
            agent = agent.with_embedder(embedder);
        }

        // Start config watcher for hot-reload
        let watch_paths = {
            let mut paths = Vec::new();
            if let Some(ref p) = self.profile {
                paths.push(p.clone());
            } else if let Some(ref p) = self.config {
                paths.push(p.clone());
            } else {
                let local = cwd.join(".crew").join("config.json");
                if local.exists() {
                    paths.push(local);
                }
                if let Some(global) = Config::global_config_path() {
                    if global.exists() {
                        paths.push(global);
                    }
                }
            }
            paths
        };
        let (config_tx, mut config_rx) = tokio::sync::watch::channel(None);
        let _watcher_handle = ConfigWatcher::new(watch_paths, config.clone(), config_tx).spawn();
        let max_history = gw_config.max_history;

        // Create session manager with LRU eviction (shared for concurrent access)
        let session_mgr = Arc::new(Mutex::new(
            SessionManager::open(&data_dir)
                .wrap_err("failed to open session manager")?
                .with_max_sessions(gw_config.max_sessions),
        ));

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
                    let bot_env = settings_str(&entry.settings, "bot_token_env", "SLACK_BOT_TOKEN");
                    let app_env = settings_str(&entry.settings, "app_token_env", "SLACK_APP_TOKEN");
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
                    let url = settings_str(&entry.settings, "bridge_url", "ws://localhost:3001");
                    channel_mgr.register(Arc::new(crew_bus::WhatsAppChannel::new(
                        &url,
                        entry.allowed_senders.clone(),
                        shutdown.clone(),
                        media_dir.clone(),
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
                    let user_env = settings_str(&entry.settings, "username_env", "EMAIL_USERNAME");
                    let pass_env = settings_str(&entry.settings, "password_env", "EMAIL_PASSWORD");
                    let username =
                        std::env::var(&user_env).wrap_err_with(|| format!("{user_env} not set"))?;
                    let password =
                        std::env::var(&pass_env).wrap_err_with(|| format!("{pass_env} not set"))?;
                    let from_address = settings_str(&entry.settings, "from_address", &username);
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
                    let id_env = settings_str(&entry.settings, "app_id_env", "FEISHU_APP_ID");
                    let secret_env =
                        settings_str(&entry.settings, "app_secret_env", "FEISHU_APP_SECRET");
                    let region = settings_str(&entry.settings, "region", "cn");
                    let app_id = std::env::var(&id_env)
                        .wrap_err_with(|| format!("{id_env} environment variable not set"))?;
                    let app_secret = std::env::var(&secret_env)
                        .wrap_err_with(|| format!("{secret_env} environment variable not set"))?;
                    let mode = settings_str(&entry.settings, "mode", "ws");
                    let webhook_port: u16 = entry
                        .settings
                        .get("webhook_port")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(9321) as u16;
                    let encrypt_key = entry
                        .settings
                        .get("encrypt_key")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let verification_token = entry
                        .settings
                        .get("verification_token")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    channel_mgr.register(Arc::new(
                        crew_bus::FeishuChannel::new(
                            &app_id,
                            &app_secret,
                            entry.allowed_senders.clone(),
                            shutdown.clone(),
                            &region,
                            media_dir.clone(),
                        )
                        .with_mode(&mode)
                        .with_webhook_port(webhook_port)
                        .with_encrypt_key(encrypt_key)
                        .with_verification_token(verification_token),
                    ));
                }
                #[cfg(feature = "twilio")]
                "twilio" => {
                    let sid_env =
                        settings_str(&entry.settings, "account_sid_env", "TWILIO_ACCOUNT_SID");
                    let token_env =
                        settings_str(&entry.settings, "auth_token_env", "TWILIO_AUTH_TOKEN");
                    let from_number = settings_str(&entry.settings, "from_number", "");
                    let account_sid = std::env::var(&sid_env)
                        .wrap_err_with(|| format!("{sid_env} environment variable not set"))?;
                    let auth_token = std::env::var(&token_env)
                        .wrap_err_with(|| format!("{token_env} environment variable not set"))?;
                    let webhook_port: u16 = entry
                        .settings
                        .get("webhook_port")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(8090) as u16;
                    channel_mgr.register(Arc::new(crew_bus::TwilioChannel::new(
                        &account_sid,
                        &auth_token,
                        &from_number,
                        entry.allowed_senders.clone(),
                        shutdown.clone(),
                        media_dir.clone(),
                        webhook_port,
                    )));
                }
                #[cfg(feature = "wecom")]
                "wecom" => {
                    let corp_id_env =
                        settings_str(&entry.settings, "corp_id_env", "WECOM_CORP_ID");
                    let secret_env =
                        settings_str(&entry.settings, "agent_secret_env", "WECOM_AGENT_SECRET");
                    let corp_id = std::env::var(&corp_id_env)
                        .wrap_err_with(|| format!("{corp_id_env} environment variable not set"))?;
                    let agent_secret = std::env::var(&secret_env)
                        .wrap_err_with(|| format!("{secret_env} environment variable not set"))?;
                    let agent_id = settings_str(&entry.settings, "agent_id", "");
                    let verification_token =
                        settings_str(&entry.settings, "verification_token", "");
                    let encoding_aes_key =
                        settings_str(&entry.settings, "encoding_aes_key", "");
                    let webhook_port: u16 = entry
                        .settings
                        .get("webhook_port")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(9322) as u16;
                    channel_mgr.register(Arc::new(
                        crew_bus::WeComChannel::new(
                            &corp_id,
                            &agent_id,
                            &agent_secret,
                            &verification_token,
                            &encoding_aes_key,
                            entry.allowed_senders.clone(),
                            shutdown.clone(),
                            media_dir.clone(),
                        )
                        .with_webhook_port(webhook_port),
                    ));
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

        // Determine default channel and chat_id for cron delivery fallback
        let default_cron_channel: String = gw_config
            .channels
            .iter()
            .map(|e| e.channel_type.as_str())
            .find(|t| *t != "cli")
            .unwrap_or("cli")
            .to_string();

        // Default chat_id: first allowed_sender from the first non-CLI channel
        let default_cron_chat_id: String = gw_config
            .channels
            .iter()
            .find(|e| e.channel_type != "cli")
            .and_then(|e| e.allowed_senders.first())
            .cloned()
            .unwrap_or_default();

        // Start channels and dispatcher
        channel_mgr.start_all(publisher).await?;

        // Set up Ctrl+C handler
        tokio::spawn(async move {
            if let Ok(()) = tokio::signal::ctrl_c().await {
                println!();
                println!("{}", "Shutting down gateway...".yellow());
                shutdown_clone.store(true, Ordering::Release);
            }
        });

        println!("{}: {}", "Max history".green(), gw_config.max_history);
        println!(
            "{}: {}",
            "Max concurrent".green(),
            gw_config.max_concurrent_sessions
        );
        println!();
        println!(
            "{}",
            "Gateway ready. Type a message or /quit to exit.".dimmed()
        );
        println!();

        // Wrap agent in Arc for sharing across spawned tasks
        let agent = Arc::new(agent);

        // Start persona service (generates communication style from chat history)
        let persona_service = Arc::new(PersonaService::new(
            data_dir.clone(),
            session_mgr.clone(),
            llm_for_compaction.clone(),
            crate::persona_service::DEFAULT_INTERVAL_SECS,
        ));
        {
            let agent_for_persona = agent.clone();
            let base_prompt = gw_config.system_prompt.clone();
            let data_dir_p = data_dir.clone();
            let project_dir_p = project_dir.clone();
            let memory_store_p = memory_store.clone();
            let tool_config_p = tool_config.clone();
            persona_service.start(move |_persona_text| {
                // Rebuild the full system prompt with the new persona and hot-update
                let base = base_prompt.clone();
                let dd = data_dir_p.clone();
                let pd = project_dir_p.clone();
                let ms = memory_store_p.clone();
                let tc = tool_config_p.clone();
                let agent = agent_for_persona.clone();
                tokio::spawn(async move {
                    let sl = SkillsLoader::new(&pd);
                    let new_prompt = build_system_prompt(
                        base.as_deref(),
                        &dd,
                        &pd,
                        &ms,
                        &sl,
                        &tc,
                    )
                    .await;
                    agent.set_system_prompt(new_prompt);
                    info!("system prompt updated with new persona");
                });
            });
        }

        // Per-session locks to serialize messages within the same session.
        // Pruned periodically to prevent unbounded growth.
        let session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Semaphore to bound concurrent session processing
        let concurrency_semaphore = Arc::new(Semaphore::new(gw_config.max_concurrent_sessions));

        // Track monitoring JoinHandles so we can await them on shutdown
        let mut monitor_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();

        // Shared max_history behind Arc<Mutex<>> for hot-reload
        let max_history = Arc::new(std::sync::atomic::AtomicUsize::new(max_history));

        // Main loop: dispatch inbound messages to concurrent tasks
        while let Some(mut inbound) = agent_handle.recv_inbound().await {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Apply hot-reload config changes (stays on main task)
            if config_rx.has_changed().unwrap_or(false) {
                if let Some(change) = config_rx.borrow_and_update().clone() {
                    match change {
                        ConfigChange::HotReload {
                            system_prompt,
                            max_history: new_max,
                        } => {
                            if let Some(prompt) = system_prompt {
                                agent.set_system_prompt(prompt);
                                info!("System prompt updated via hot-reload");
                            }
                            if let Some(new_max) = new_max {
                                max_history.store(new_max, Ordering::Relaxed);
                                info!("Max history updated to {new_max} via hot-reload");
                            }
                        }
                        ConfigChange::RestartRequired(_) => {
                            // Already logged by ConfigWatcher
                        }
                    }
                }
            }

            // Transcribe audio media and separate images (stays on main task)
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
                    .and_then(|s| if s.is_empty() { None } else { Some(s) })
                    .unwrap_or(&default_cron_channel)
                    .to_string();
                let cid = inbound
                    .metadata
                    .get("deliver_to_chat_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| if s.is_empty() { None } else { Some(s) })
                    .unwrap_or_else(|| {
                        if !default_cron_chat_id.is_empty() {
                            &default_cron_chat_id
                        } else {
                            &inbound.chat_id
                        }
                    })
                    .to_string();
                (ch, cid)
            } else {
                (inbound.channel.clone(), inbound.chat_id.clone())
            };

            let session_key = inbound.session_key();

            // Handle /new command inline (quick operation, no concurrency needed)
            if inbound.content.trim() == "/new" {
                let new_id = format!(
                    "{}_{}_{}",
                    inbound.sender_id,
                    inbound.chat_id,
                    chrono::Utc::now().timestamp_millis(),
                );
                match session_mgr
                    .lock()
                    .await
                    .fork(&session_key, &new_id, 10)
                    .await
                {
                    Ok(new_key) => {
                        let msg = OutboundMessage {
                            channel: reply_channel.clone(),
                            chat_id: reply_chat_id.clone(),
                            content: format!("Session forked. New session: {new_key}"),
                            reply_to: None,
                            media: vec![],
                            metadata: serde_json::json!({}),
                        };
                        let _ = agent_handle.send_outbound(msg).await;
                    }
                    Err(e) => {
                        warn!("session fork failed: {e}");
                    }
                }
                continue;
            }

            // Handle /config command inline
            if inbound.content.trim() == "/config" || inbound.content.trim().starts_with("/config ")
            {
                let args = inbound
                    .content
                    .trim()
                    .strip_prefix("/config")
                    .unwrap_or("")
                    .trim();
                let response = tool_config.handle_config_command(args).await;
                let msg = OutboundMessage {
                    channel: reply_channel.clone(),
                    chat_id: reply_chat_id.clone(),
                    content: response,
                    reply_to: None,
                    media: vec![],
                    metadata: serde_json::json!({}),
                };
                let _ = agent_handle.send_outbound(msg).await;
                continue;
            }

            info!(
                channel = %inbound.channel,
                sender = %inbound.sender_id,
                session = %session_key,
                "dispatching message to concurrent handler"
            );

            // Clone shared state for the spawned task
            let agent = agent.clone();
            let session_mgr = session_mgr.clone();
            let session_locks = session_locks.clone();
            let semaphore = concurrency_semaphore.clone();
            let message_tool = message_tool.clone();
            let send_file_tool = send_file_tool.clone();
            let take_photo_tool = take_photo_tool.clone();
            let spawn_tool = spawn_tool.clone();
            let cron_tool = cron_tool.clone();
            let llm_for_compaction = llm_for_compaction.clone();
            let out_tx = agent_handle.outbound_sender();
            let max_history = max_history.clone();
            let shutdown = shutdown.clone();
            let queue_mode = gw_config.queue_mode.clone();
            let collect_inbound_tx = collect_inbound_tx.clone();

            let session_key_str = session_key.to_string();
            let locks_for_prune = session_locks.clone();
            let handle = tokio::spawn(async move {
                // Acquire concurrency permit (blocks if at max)
                let _permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => return, // semaphore closed
                };

                if shutdown.load(Ordering::Relaxed) {
                    return;
                }

                // Get or create per-session lock
                let session_lock = {
                    let mut locks = session_locks.lock().await;
                    locks
                        .entry(session_key.to_string())
                        .or_insert_with(|| Arc::new(Mutex::new(())))
                        .clone()
                };

                // Serialize processing within the same session
                let _session_guard = session_lock.lock().await;

                process_session_message(
                    &agent,
                    &session_mgr,
                    &message_tool,
                    &send_file_tool,
                    &take_photo_tool,
                    &spawn_tool,
                    &cron_tool,
                    &llm_for_compaction,
                    &out_tx,
                    &inbound,
                    &session_key,
                    &reply_channel,
                    &reply_chat_id,
                    image_media,
                    max_history.load(Ordering::Relaxed),
                    &queue_mode,
                    &collect_inbound_tx,
                )
                .await;
            });

            // Monitor spawned task for panics; prune session lock after completion
            let session_key_for_log = session_key_str;
            let mh = tokio::spawn(async move {
                if let Err(e) = handle.await {
                    tracing::error!(
                        session = %session_key_for_log,
                        error = %e,
                        "session task panicked"
                    );
                }
                // Prune session lock if no other task holds a reference
                let mut locks = locks_for_prune.lock().await;
                if let Some(lock) = locks.get(&session_key_for_log) {
                    // Arc::strong_count == 1 means only the HashMap holds it
                    if Arc::strong_count(lock) == 1 {
                        locks.remove(&session_key_for_log);
                    }
                }
            });
            monitor_handles.push(mh);

            // Periodically clean up completed monitor handles to avoid Vec growth
            if monitor_handles.len() > 100 {
                monitor_handles.retain(|h| !h.is_finished());
            }
        }

        // Wait for all in-flight tasks to complete before shutdown
        for h in monitor_handles {
            let _ = h.await;
        }

        persona_service.stop().await;
        heartbeat_service.stop().await;
        cron_service.stop().await;
        channel_mgr.stop_all().await?;
        println!("{}", "Gateway stopped.".dimmed());
        Ok(())
    }
}

/// Process a single inbound message for a session (runs inside a spawned task).
#[allow(clippy::too_many_arguments)]
async fn process_session_message(
    agent: &Agent,
    session_mgr: &Mutex<SessionManager>,
    message_tool: &MessageTool,
    send_file_tool: &SendFileTool,
    take_photo_tool: &TakePhotoTool,
    spawn_tool: &SpawnTool,
    cron_tool: &CronTool,
    llm: &Arc<dyn LlmProvider>,
    out_tx: &tokio::sync::mpsc::Sender<OutboundMessage>,
    inbound: &crew_core::InboundMessage,
    session_key: &SessionKey,
    reply_channel: &str,
    reply_chat_id: &str,
    image_media: Vec<String>,
    max_history: usize,
    queue_mode: &QueueMode,
    collect_tx: &tokio::sync::mpsc::Sender<crew_core::InboundMessage>,
) {
    // Set tool context for this session's reply routing
    message_tool.set_context(reply_channel, reply_chat_id);
    send_file_tool.set_context(reply_channel, reply_chat_id);
    take_photo_tool.set_context(reply_channel, reply_chat_id);
    spawn_tool.set_context(reply_channel, reply_chat_id);
    cron_tool.set_context(reply_channel, reply_chat_id);

    // Get conversation history (hold session_mgr lock briefly)
    let history: Vec<Message> = {
        let mut mgr = session_mgr.lock().await;
        let session = mgr.get_or_create(session_key);
        session.get_history(max_history).to_vec()
    };

    // Send a quick acknowledgment so the user knows we're working on it
    let ack = OutboundMessage {
        channel: reply_channel.to_string(),
        chat_id: reply_chat_id.to_string(),
        content: "Thinking...".to_string(),
        reply_to: None,
        media: vec![],
        metadata: serde_json::json!({}),
    };
    let _ = out_tx.send(ack).await;

    // Process message through agent (potentially long LLM call, no lock held)
    let response = agent
        .process_message(&inbound.content, &history, image_media)
        .await;

    match response {
        Ok(conv_response) => {
            // Save user + assistant messages (hold lock briefly)
            {
                let mut mgr = session_mgr.lock().await;
                let user_msg = Message {
                    role: MessageRole::User,
                    content: inbound.content.clone(),
                    media: vec![],
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                    timestamp: Utc::now(),
                };
                let _ = mgr.add_message(session_key, user_msg).await;

                // Only save non-empty assistant messages to session history
                if !conv_response.content.is_empty() {
                    let assistant_msg = Message {
                        role: MessageRole::Assistant,
                        content: conv_response.content.clone(),
                        media: vec![],
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                        timestamp: Utc::now(),
                    };
                    let _ = mgr.add_message(session_key, assistant_msg).await;
                }

                // Compact session if it's grown too large
                if let Err(e) =
                    crate::compaction::maybe_compact(&mut mgr, session_key, &**llm).await
                {
                    warn!("session compaction failed: {e}");
                }
            }

            // Send response back through channel
            // Strip <think>...</think> blocks from models that embed reasoning inline
            let content = strip_think_tags(&conv_response.content);

            // For cron-triggered messages: suppress delivery if the agent response
            // is empty or starts with [SILENT] (allows conditional-notify jobs).
            let is_cron = inbound.channel == "system" && inbound.sender_id == "cron";
            let is_silent = content.trim().is_empty()
                || content.trim_start().starts_with("[SILENT]")
                || content.trim_start().starts_with("[NO_CHANGE]");

            if is_cron && is_silent {
                tracing::debug!("cron job response suppressed (silent/empty)");
            } else {
                let display_content = content
                    .trim_start()
                    .strip_prefix("[SILENT]")
                    .or_else(|| content.trim_start().strip_prefix("[NO_CHANGE]"))
                    .unwrap_or(&content)
                    .to_string();
                let outbound = OutboundMessage {
                    channel: reply_channel.to_string(),
                    chat_id: reply_chat_id.to_string(),
                    content: display_content,
                    reply_to: None,
                    media: vec![],
                    metadata: serde_json::json!({}),
                };
                let _ = out_tx.send(outbound).await;
            }

            // Collect mode: not applicable in concurrent processing
            // (would require access to agent_handle which stays on main task)
            let _ = (queue_mode, collect_tx);
        }
        Err(e) => {
            let error_msg = OutboundMessage {
                channel: reply_channel.to_string(),
                chat_id: reply_chat_id.to_string(),
                content: format!("Error: {e}"),
                reply_to: None,
                media: vec![],
                metadata: serde_json::json!({}),
            };
            let _ = out_tx.send(error_msg).await;
        }
    }
}

/// Strip `<think>...</think>` blocks that some models (e.g. MiniMax) embed inline.
fn strip_think_tags(s: &str) -> String {
    let mut result = s.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result.replace_range(start..start + end + "</think>".len(), "");
        } else {
            // Unclosed <think> — strip from tag to end
            result.truncate(start);
            break;
        }
    }
    result.trim().to_string()
}

/// Build the system prompt with bootstrap files, memory context, and skills.
async fn build_system_prompt(
    base: Option<&str>,
    data_dir: &Path,
    project_dir: &Path,
    memory_store: &MemoryStore,
    skills_loader: &SkillsLoader,
    tool_config: &crew_agent::ToolConfigStore,
) -> String {
    let default_prompt = "You are Crew, a smart and resourceful AI assistant. \
        You're direct, warm, and occasionally witty — like a capable friend who happens to \
        know how to code, research, and get things done.\n\
        \n\
        ## How to work\n\
        - Be conversational and natural — not robotic or overly formal\n\
        - Keep responses concise. Don't over-explain. Get to the point.\n\
        - Use your tools proactively to get things done — don't just describe what you could do\n\
        - When a task takes time (web_search, deep_search, deep_research, spawn, take_photo, \
        or multi-step tool work), FIRST use the `message` tool to immediately acknowledge \
        the user and briefly say what you're about to do. Never leave the user waiting in silence.\n\
        - When you hit a wall, say so honestly and suggest alternatives\n\
        - When the user shares preferences, personal info, or important facts, \
        save them using `save_memory` so you remember next time\n\
        \n\
        ## Formatting\n\
        - You may use markdown: **bold**, *italic*, `code`, ```code blocks```, \
        ~~strikethrough~~, [text](url), > blockquotes, and - bullet lists\n\
        - But don't over-format. Plain conversational text is often better than \
        a wall of bullet points.";
    let mut prompt = base.unwrap_or(default_prompt).to_string();

    // Inject dynamically generated persona (from persona.md) if available
    if let Some(persona) = PersonaService::read_persona(data_dir) {
        prompt.push_str("\n\n## Communication Style\n\n");
        prompt.push_str(&persona);
    }

    // Append bootstrap files (AGENTS.md, SOUL.md, USER.md, etc.)
    let bootstrap = super::load_bootstrap_files(project_dir);
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

    // Append memory bank summary (entity abstracts)
    let bank_summary = memory_store.get_bank_summary().await;
    if !bank_summary.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&bank_summary);
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

    // Append tool preferences summary
    let config_summary = tool_config.summary().await;
    if !config_summary.is_empty() {
        prompt.push_str("\n\n## Tool Preferences\n\n");
        prompt.push_str(&config_summary);
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
    feature = "feishu",
    feature = "twilio",
    feature = "wecom"
))]
fn settings_str(settings: &serde_json::Value, key: &str, default: &str) -> String {
    settings
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

/// Build an email sender from the config.
pub(crate) fn build_email_sender(
    cfg: &crate::config::EmailConfig,
) -> Result<Arc<dyn crew_agent::EmailSender>> {
    match cfg.provider.as_str() {
        "smtp" => Ok(Arc::new(crew_agent::SmtpEmailSender::new(
            cfg.smtp_host.clone().unwrap_or_default(),
            cfg.smtp_port.unwrap_or(465),
            cfg.username.clone().unwrap_or_default(),
            cfg.password_env.clone().unwrap_or_default(),
            cfg.from_address.clone().unwrap_or_default(),
        ))),
        "feishu" | "lark" => Ok(Arc::new(crew_agent::FeishuEmailSender::new(
            cfg.feishu_app_id.clone().unwrap_or_default(),
            cfg.feishu_app_secret_env.clone().unwrap_or_default(),
            cfg.feishu_from_address.clone().unwrap_or_default(),
            cfg.feishu_region.as_deref().unwrap_or("cn"),
        ))),
        other => eyre::bail!("unknown email provider: {other}"),
    }
}

/// Merge queued inbound messages by session key.
/// Messages from the same session are concatenated with `\n\n`.
/// Used by Collect queue mode (reserved for future concurrent collect support).
#[allow(dead_code)]
fn merge_queued_by_session(
    messages: Vec<crew_core::InboundMessage>,
) -> Vec<crew_core::InboundMessage> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<crew_core::InboundMessage>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for msg in messages {
        let key = msg.session_key().to_string();
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(msg);
    }
    order
        .into_iter()
        .filter_map(|key| {
            let mut msgs = groups.remove(&key)?;
            if msgs.len() == 1 {
                return msgs.pop();
            }
            let mut base = msgs.remove(0);
            for m in &msgs {
                base.content.push_str("\n\n");
                base.content.push_str(&m.content);
            }
            Some(base)
        })
        .collect()
}
