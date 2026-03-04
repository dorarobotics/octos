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
    SpawnTool, TokenTracker, /* TakePhotoTool, */ ToolRegistry,
};
use crew_bus::{
    ActiveSessionStore, ChannelManager, CliChannel, CronService, HeartbeatService, SessionManager,
    create_bus, validate_topic_name,
};
use crew_core::{AgentId, Message, MessageRole, OutboundMessage, SessionKey};
use crew_llm::{
    AdaptiveConfig, AdaptiveRouter, GroqTranscriber, LlmProvider, ProviderChain, ProviderRouter,
    RetryProvider,
};
use crew_memory::{EpisodeStore, MemoryStore};
use eyre::{Result, WrapErr};
use tokio::sync::{Mutex, Semaphore};
use tracing::{info, warn};

use std::path::Path;

use super::Executable;
use crate::commands::chat::{create_embedder, resolve_provider_policy};
use crate::config::{Config, QueueMode, detect_provider};
use crate::config_watcher::{ConfigChange, ConfigWatcher};
use crate::cron_tool::CronTool;
use crate::persona_service::PersonaService;
use crate::status_indicator::StatusIndicator;

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

    /// Path to parent profile JSON (sub-accounts inherit provider config).
    #[arg(long, hide = true)]
    pub parent_profile: Option<PathBuf>,

    /// Crew home directory for ProfileStore access (used by managed gateways).
    #[arg(long, hide = true)]
    pub crew_home: Option<PathBuf>,
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
        // Use eprintln! for the startup banner so it reaches the server's stderr
        // reader immediately (stderr is unbuffered, unlike piped stdout).
        eprintln!("[gateway] starting");
        println!("{}", "crew gateway".cyan().bold());
        println!();

        let cwd = match self.cwd {
            Some(p) => p,
            None => std::env::current_dir().wrap_err("failed to get current directory")?,
        };

        let mut profile_id: Option<String> = None;
        eprintln!("[gateway] loading config (profile={:?})", self.profile.as_deref().map(|p| p.display().to_string()));
        let mut admin_mode = false;
        let config = if let Some(ref profile_path) = self.profile {
            // Load config from profile JSON (single source of truth)
            let content = std::fs::read_to_string(profile_path)
                .wrap_err_with(|| format!("failed to read profile: {}", profile_path.display()))?;
            let mut profile: crate::profiles::UserProfile = serde_json::from_str(&content)
                .wrap_err_with(|| format!("failed to parse profile: {}", profile_path.display()))?;
            profile_id = Some(profile.id.clone());
            admin_mode = profile.config.admin_mode;

            // Sub-account: merge LLM provider config from parent profile
            if let Some(ref parent_path) = self.parent_profile {
                if let Ok(parent_content) = std::fs::read_to_string(parent_path) {
                    if let Ok(parent) =
                        serde_json::from_str::<crate::profiles::UserProfile>(&parent_content)
                    {
                        info!(
                            parent = %parent.id,
                            sub_account = %profile.id,
                            "inheriting provider config from parent profile"
                        );
                        profile.config.provider = parent.config.provider;
                        profile.config.model = parent.config.model;
                        profile.config.base_url = parent.config.base_url;
                        profile.config.api_key_env = parent.config.api_key_env;
                        profile.config.api_type = parent.config.api_type;
                        profile.config.fallback_models = parent.config.fallback_models;
                        if profile.config.email.is_none() {
                            profile.config.email = parent.config.email;
                        }
                    }
                }
            }

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
                ..Default::default()
            });

        eprintln!("[gateway] provider={provider_name}");
        println!("{}: {}", "Provider".green(), provider_name);

        // Create LLM provider (reuses the shared create_provider from chat.rs)
        use super::chat::create_provider;
        let base_provider = create_provider(&provider_name, &config, model, base_url)?;
        eprintln!("[gateway] LLM provider created, model={}", base_provider.model_id());

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
            // Auto-enable adaptive routing when multiple providers exist
            if providers.len() > 1 {
                let adaptive_config = config
                    .adaptive_routing
                    .as_ref()
                    .map(|ar| AdaptiveConfig::from(ar))
                    .unwrap_or_default();
                info!("adaptive routing enabled ({} providers)", providers.len());
                Arc::new(AdaptiveRouter::new(providers, adaptive_config))
            } else {
                Arc::new(ProviderChain::new(providers))
            }
        };

        // Resolve data directory (--data-dir > $CREW_HOME > ~/.crew)
        let data_dir = super::resolve_data_dir(self.data_dir)?;

        // Open ProfileStore for /account commands (if crew-home is available)
        let profile_store: Option<Arc<crate::profiles::ProfileStore>> =
            if let Some(ref crew_home) = self.crew_home {
                crate::profiles::ProfileStore::open(crew_home)
                    .ok()
                    .map(Arc::new)
            } else {
                None
            };

        // Export CREW_HOME and CREW_PROFILE_ID so plugin tools (e.g. account-manager)
        // can access the profile store and know which profile is running.
        // SAFETY: gateway is single-threaded at this point (before tokio tasks spawn).
        #[allow(unsafe_code)]
        unsafe {
            if let Some(ref crew_home) = self.crew_home {
                std::env::set_var("CREW_HOME", crew_home);
            }
            if let Some(ref pid) = profile_id {
                std::env::set_var("CREW_PROFILE_ID", pid);
            }
        }

        // Spawn periodic metrics exporter (writes provider_metrics.json every 30s)
        if llm.export_metrics().is_some() {
            let metrics_llm = llm.clone();
            let metrics_path = data_dir.join("provider_metrics.json");
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
                loop {
                    interval.tick().await;
                    if let Some(value) = metrics_llm.export_metrics() {
                        if let Ok(json) = serde_json::to_string_pretty(&value) {
                            let _ = tokio::fs::write(&metrics_path, json).await;
                        }
                    }
                }
            });
        }

        let media_dir = data_dir.join("media");
        let _ = &media_dir; // used by channel feature gates below

        // Create voice transcriber if GROQ_API_KEY is set
        let transcriber = std::env::var("GROQ_API_KEY").ok().map(|key| {
            println!("{}: Groq Whisper", "Transcriber".green());
            GroqTranscriber::new(key)
        });

        eprintln!("[gateway] opening episode store at {}", data_dir.display());
        let memory = Arc::new(
            EpisodeStore::open(&data_dir)
                .await
                .wrap_err("failed to open episode store")?,
        );
        eprintln!("[gateway] episode store opened");

        // Initialize memory store
        eprintln!("[gateway] opening memory store");
        let memory_store = Arc::new(
            MemoryStore::open(&data_dir)
                .await
                .wrap_err("failed to open memory store")?,
        );
        eprintln!("[gateway] memory store opened");

        // Initialize skills loader (project-level, from cwd/.crew/)
        let project_dir = cwd.join(".crew");

        // Bootstrap bundled app-skill binaries into .crew/skills/
        let skills_dir = project_dir.join("skills");
        std::fs::create_dir_all(&skills_dir).ok();
        let n = crew_agent::bootstrap::bootstrap_bundled_skills(&skills_dir);
        if n > 0 {
            info!(count = n, "bootstrapped bundled app-skills");
        }

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

        // Build tool registry — admin mode gets only admin API tools + messaging
        let tool_config = Arc::new(
            crew_agent::ToolConfigStore::open(&data_dir)
                .await
                .wrap_err("failed to open tool config store")?,
        );

        // Create messaging tools before the if/else (needed in both modes)
        let message_tool = Arc::new(MessageTool::new(out_tx.clone()));
        let send_file_tool = Arc::new(SendFileTool::new(out_tx.clone()));
        drop(out_tx);

        // Optional tools (only exist in normal mode)
        let mut spawn_tool: Option<Arc<SpawnTool>> = None;
        let mut cron_tool_arc: Option<Arc<CronTool>> = None;
        let mut pipeline_tool: Option<Arc<crew_pipeline::RunPipelineTool>> = None;

        let mut tools;
        if admin_mode {
            // Admin mode: register only admin API tools
            tools = ToolRegistry::new();

            // Register admin API tools (calls REST API on crew serve)
            let serve_url = std::env::var("CREW_SERVE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
            let admin_token =
                std::env::var("CREW_ADMIN_TOKEN").unwrap_or_default();
            let admin_ctx = Arc::new(crew_agent::AdminApiContext {
                http: reqwest::Client::new(),
                serve_url,
                admin_token,
            });
            crew_agent::register_admin_api_tools(&mut tools, admin_ctx);

            // Cron tool (scheduled tasks, reminders)
            let ct = Arc::new(CronTool::new(cron_service.clone()));
            tools.register_arc(ct.clone() as Arc<dyn crew_agent::Tool>);
            cron_tool_arc = Some(ct);

            // Register messaging tools
            tools.register_arc(message_tool.clone() as Arc<dyn crew_agent::Tool>);
            tools.register_arc(send_file_tool.clone() as Arc<dyn crew_agent::Tool>);

            // App-skill tools (send_email, news, etc.) via plugin loader
            let plugin_dirs = crate::config::Config::plugin_dirs(&cwd);
            if !plugin_dirs.is_empty() {
                if let Err(e) = crew_agent::PluginLoader::load_into(&mut tools, &plugin_dirs) {
                    warn!("plugin loading failed: {e}");
                }
            }

            // Memory bank tools
            tools.register(crew_agent::RecallMemoryTool::new(memory_store.clone()));
            tools.register(crew_agent::SaveMemoryTool::new(memory_store.clone()));

            info!("admin mode: registered admin API + cron + skills tools");
        } else {
            // Normal mode: full tool registration
            let sandbox = crew_agent::create_sandbox(&config.sandbox);
            tools = ToolRegistry::with_builtins_and_sandbox(&cwd, sandbox);
            tools.inject_tool_config(tool_config.clone());

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

            let ct = Arc::new(CronTool::new(cron_service.clone()));
            tools.register_arc(ct.clone() as Arc<dyn crew_agent::Tool>);
            cron_tool_arc = Some(ct);

            // Register messaging tools
            tools.register_arc(message_tool.clone() as Arc<dyn crew_agent::Tool>);
            tools.register_arc(send_file_tool.clone() as Arc<dyn crew_agent::Tool>);

            // Build sub-provider router from config
            let provider_router = if !config.sub_providers.is_empty() {
                let router = Arc::new(ProviderRouter::new());
                for sp in &config.sub_providers {
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

            // Spawn tool (background subagents)
            let worker_prompt = super::load_prompt("worker", crew_agent::DEFAULT_WORKER_PROMPT);
            let mut spawn = SpawnTool::new(llm.clone(), memory.clone(), cwd.clone(), spawn_inbound_tx)
                .with_provider_policy(tools.provider_policy().cloned())
                .with_worker_prompt(worker_prompt);
            if let Some(ref router) = provider_router {
                spawn = spawn.with_provider_router(router.clone());
            }
            let st = Arc::new(spawn);
            tools.register_arc(st.clone() as Arc<dyn crew_agent::Tool>);
            spawn_tool = Some(st);

            // Research synthesis tool
            tools.register(crew_agent::SynthesizeResearchTool::new(
                llm.clone(),
                data_dir.clone(),
            ));

            // Deep research pipeline
            tools.register(crew_agent::DeepResearchTool::new(
                llm.clone(),
                cwd.clone(),
                data_dir.clone(),
                plugin_dirs.clone(),
            ));

            // Pipeline tool
            let mut pt = crew_pipeline::RunPipelineTool::new(
                llm.clone(),
                memory.clone(),
                cwd.clone(),
                data_dir.clone(),
            )
            .with_provider_policy(tools.provider_policy().cloned())
            .with_plugin_dirs(plugin_dirs.clone());
            if let Some(ref router) = provider_router {
                pt = pt.with_provider_router(router.clone());
            }
            let pt = Arc::new(pt);
            tools.register_arc(pt.clone() as Arc<dyn crew_agent::Tool>);
            pipeline_tool = Some(pt);

            // Memory bank tools
            tools.register(crew_agent::RecallMemoryTool::new(memory_store.clone()));
            tools.register(crew_agent::SaveMemoryTool::new(memory_store.clone()));
        }

        // Note: send_email tool is now provided by the system-skills package

        // Build system prompt — admin mode uses a built-in admin prompt
        let system_prompt = if admin_mode {
            let custom = gw_config.system_prompt.as_deref();
            if let Some(custom_prompt) = custom {
                custom_prompt.to_string()
            } else {
                let compiled = include_str!("../prompts/admin_default.txt");
                super::load_prompt("admin", compiled)
            }
        } else {
            build_system_prompt(
                gw_config.system_prompt.as_deref(),
                &data_dir,
                &project_dir,
                &memory_store,
                &skills_loader,
                &tool_config,
            )
            .await
        };

        // Build the agent
        let max_iterations = self.max_iterations.or(config.max_iterations).unwrap_or(50);
        let agent_config = AgentConfig {
            max_iterations,
            save_episodes: false,
            tool_timeout_secs: gw_config
                .tool_timeout_secs
                .unwrap_or(crew_agent::DEFAULT_TOOL_TIMEOUT_SECS),
            ..Default::default()
        };
        let session_timeout_secs = gw_config
            .session_timeout_secs
            .unwrap_or(crew_agent::DEFAULT_SESSION_TIMEOUT_SECS);

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

        // Set hook context with profile_id (session_id updated per message)
        if profile_id.is_some() || !config.hooks.is_empty() {
            agent = agent.with_hook_context(crew_agent::HookContext {
                session_id: None,
                profile_id: profile_id.clone(),
            });
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

        // Active session store for multi-session support
        let active_sessions = Arc::new(Mutex::new(
            ActiveSessionStore::open(&data_dir)
                .wrap_err("failed to open active session store")?,
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
                    let corp_id_env = settings_str(&entry.settings, "corp_id_env", "WECOM_CORP_ID");
                    let secret_env =
                        settings_str(&entry.settings, "agent_secret_env", "WECOM_AGENT_SECRET");
                    let corp_id = std::env::var(&corp_id_env)
                        .wrap_err_with(|| format!("{corp_id_env} environment variable not set"))?;
                    let agent_secret = std::env::var(&secret_env)
                        .wrap_err_with(|| format!("{secret_env} environment variable not set"))?;
                    let agent_id = settings_str(&entry.settings, "agent_id", "");
                    let verification_token =
                        settings_str(&entry.settings, "verification_token", "");
                    let encoding_aes_key = settings_str(&entry.settings, "encoding_aes_key", "");
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
        eprintln!("[gateway] starting channels");
        channel_mgr.start_all(publisher).await?;
        eprintln!("[gateway] channels started");

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
        eprintln!("[gateway] ready");
        println!(
            "{}",
            "Gateway ready. Type a message or /quit to exit.".dimmed()
        );
        println!();

        // Wrap agent in Arc for sharing across spawned tasks
        let agent = Arc::new(agent);

        // Create status indicators for each channel (used for typing + dynamic status)
        let status_words = PersonaService::read_status_words(&data_dir);
        let status_indicators: Arc<HashMap<String, Arc<StatusIndicator>>> = {
            let mut map = HashMap::new();
            for entry in &gw_config.channels {
                if let Some(ch) = channel_mgr.get_channel(&entry.channel_type) {
                    map.insert(
                        entry.channel_type.clone(),
                        Arc::new(StatusIndicator::new(ch, status_words.clone())),
                    );
                }
            }
            Arc::new(map)
        };

        // Start persona service (generates communication style from chat history)
        let persona_service = Arc::new(PersonaService::new(
            data_dir.clone(),
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
            let indicators = status_indicators.clone();
            persona_service.start(
                move |_persona_text| {
                    // Rebuild the full system prompt with the new persona and hot-update
                    let base = base_prompt.clone();
                    let dd = data_dir_p.clone();
                    let pd = project_dir_p.clone();
                    let ms = memory_store_p.clone();
                    let tc = tool_config_p.clone();
                    let agent = agent_for_persona.clone();
                    tokio::spawn(async move {
                        let sl = SkillsLoader::new(&pd);
                        let new_prompt =
                            build_system_prompt(base.as_deref(), &dd, &pd, &ms, &sl, &tc).await;
                        agent.set_system_prompt(new_prompt);
                        info!("system prompt updated with new persona");
                    });
                },
                move |words| {
                    // Update status word pools in all indicators
                    for indicator in indicators.values() {
                        indicator.set_words(words.clone());
                    }
                    info!("status words updated in indicators");
                },
            );
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
            if shutdown.load(Ordering::Acquire) {
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
                                max_history.store(new_max, Ordering::Release);
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

            // Resolve session key with active topic
            let base_session_key = inbound.session_key();
            let base_key_str = base_session_key.0.clone();
            let session_key = {
                let store = active_sessions.lock().await;
                store.resolve_session_key(&base_key_str)
            };

            let cmd = inbound.content.trim();

            // Handle /new command — clear current session or create named session
            if cmd == "/new" || cmd.starts_with("/new ") {
                let name = cmd.strip_prefix("/new").unwrap_or("").trim();
                if name.is_empty() {
                    // Clear current session (existing behavior)
                    match session_mgr.lock().await.clear(&session_key).await {
                        Ok(()) => {
                            let msg = OutboundMessage {
                                channel: reply_channel.clone(),
                                chat_id: reply_chat_id.clone(),
                                content: "Session cleared.".to_string(),
                                reply_to: None,
                                media: vec![],
                                metadata: serde_json::json!({}),
                            };
                            let _ = agent_handle.send_outbound(msg).await;
                        }
                        Err(e) => {
                            warn!("session clear failed: {e}");
                        }
                    }
                } else {
                    // Create/switch to named session
                    if let Err(reason) = validate_topic_name(name) {
                        let msg = OutboundMessage {
                            channel: reply_channel.clone(),
                            chat_id: reply_chat_id.clone(),
                            content: format!("Invalid session name: {reason}"),
                            reply_to: None,
                            media: vec![],
                            metadata: serde_json::json!({}),
                        };
                        let _ = agent_handle.send_outbound(msg).await;
                    } else {
                        active_sessions
                            .lock()
                            .await
                            .switch_to(&base_key_str, name)
                            .unwrap_or_else(|e| warn!("switch_to failed: {e}"));
                        let msg = OutboundMessage {
                            channel: reply_channel.clone(),
                            chat_id: reply_chat_id.clone(),
                            content: format!("Switched to session: {name}"),
                            reply_to: None,
                            media: vec![],
                            metadata: serde_json::json!({}),
                        };
                        let _ = agent_handle.send_outbound(msg).await;
                    }
                }
                continue;
            }

            // Handle /s command — switch to a named session
            if cmd == "/s" || cmd.starts_with("/s ") {
                let name = cmd.strip_prefix("/s").unwrap_or("").trim();
                if name.is_empty() {
                    // Switch to default session
                    active_sessions
                        .lock()
                        .await
                        .switch_to(&base_key_str, "")
                        .unwrap_or_else(|e| warn!("switch_to failed: {e}"));
                    let msg = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
                        content: "Switched to default session.".to_string(),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    let _ = agent_handle.send_outbound(msg).await;
                } else if let Err(reason) = validate_topic_name(name) {
                    let msg = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
                        content: format!("Invalid session name: {reason}"),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    let _ = agent_handle.send_outbound(msg).await;
                } else {
                    active_sessions
                        .lock()
                        .await
                        .switch_to(&base_key_str, name)
                        .unwrap_or_else(|e| warn!("switch_to failed: {e}"));

                    // Show last 2 messages as context preview
                    let new_key = SessionKey::with_topic(
                        &inbound.channel,
                        &inbound.chat_id,
                        name,
                    );
                    let preview = {
                        let mut mgr = session_mgr.lock().await;
                        let session = mgr.get_or_create(&new_key);
                        let history = session.get_history(2);
                        if history.is_empty() {
                            String::new()
                        } else {
                            let mut lines = String::from("\n---\n");
                            for m in history {
                                let role = m.role.as_str();
                                let text: String = m.content.chars().take(100).collect();
                                lines.push_str(&format!("[{role}] {text}\n"));
                            }
                            lines
                        }
                    };

                    let msg = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
                        content: format!("Switched to session: {name}{preview}"),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    let _ = agent_handle.send_outbound(msg).await;
                }
                continue;
            }

            // Handle /sessions command — list all sessions for this chat
            if cmd == "/sessions" {
                let entries = session_mgr
                    .lock()
                    .await
                    .list_sessions_for_chat(&base_key_str);
                let active_topic = active_sessions
                    .lock()
                    .await
                    .get_active_topic(&base_key_str)
                    .to_string();

                if entries.is_empty() {
                    let msg = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
                        content: "No sessions found.".to_string(),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    let _ = agent_handle.send_outbound(msg).await;
                } else {
                    let mut lines = format!("{} session(s):\n\n", entries.len());
                    for entry in &entries {
                        let name = entry
                            .topic
                            .as_deref()
                            .unwrap_or("(default)");
                        let is_active = match &entry.topic {
                            Some(t) => t == &active_topic,
                            None => active_topic.is_empty(),
                        };
                        let marker = if is_active { " *" } else { "" };
                        let summary = entry
                            .summary
                            .as_deref()
                            .unwrap_or("");
                        let time = entry
                            .updated_at
                            .format("%m/%d %H:%M")
                            .to_string();
                        lines.push_str(&format!(
                            "{name}{marker}  ({} msgs, {time})",
                            entry.message_count,
                        ));
                        if !summary.is_empty() {
                            lines.push_str(&format!(" - {summary}"));
                        }
                        lines.push('\n');
                    }
                    let msg = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
                        content: lines,
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    let _ = agent_handle.send_outbound(msg).await;
                }
                continue;
            }

            // Handle /back command — switch to previous session
            if cmd == "/back" {
                let result = active_sessions
                    .lock()
                    .await
                    .go_back(&base_key_str);
                match result {
                    Ok(Some(topic)) => {
                        let label = if topic.is_empty() {
                            "(default)".to_string()
                        } else {
                            topic
                        };
                        let msg = OutboundMessage {
                            channel: reply_channel.clone(),
                            chat_id: reply_chat_id.clone(),
                            content: format!("Switched back to session: {label}"),
                            reply_to: None,
                            media: vec![],
                            metadata: serde_json::json!({}),
                        };
                        let _ = agent_handle.send_outbound(msg).await;
                    }
                    Ok(None) => {
                        let msg = OutboundMessage {
                            channel: reply_channel.clone(),
                            chat_id: reply_chat_id.clone(),
                            content: "No previous session to switch to.".to_string(),
                            reply_to: None,
                            media: vec![],
                            metadata: serde_json::json!({}),
                        };
                        let _ = agent_handle.send_outbound(msg).await;
                    }
                    Err(e) => {
                        warn!("go_back failed: {e}");
                    }
                }
                continue;
            }

            // Handle /delete command — delete a named session
            if cmd.starts_with("/delete ") {
                let name = cmd.strip_prefix("/delete").unwrap_or("").trim();
                if name.is_empty() {
                    let msg = OutboundMessage {
                        channel: reply_channel.clone(),
                        chat_id: reply_chat_id.clone(),
                        content: "Usage: /delete <session-name>".to_string(),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    };
                    let _ = agent_handle.send_outbound(msg).await;
                } else {
                    let del_key = SessionKey::with_topic(
                        &inbound.channel,
                        &inbound.chat_id,
                        name,
                    );
                    match session_mgr.lock().await.clear(&del_key).await {
                        Ok(()) => {
                            active_sessions
                                .lock()
                                .await
                                .remove_topic(&base_key_str, name)
                                .unwrap_or_else(|e| warn!("remove_topic failed: {e}"));
                            let msg = OutboundMessage {
                                channel: reply_channel.clone(),
                                chat_id: reply_chat_id.clone(),
                                content: format!("Deleted session: {name}"),
                                reply_to: None,
                                media: vec![],
                                metadata: serde_json::json!({}),
                            };
                            let _ = agent_handle.send_outbound(msg).await;
                        }
                        Err(e) => {
                            warn!("delete session failed: {e}");
                        }
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

            // Handle /account command inline — sub-account management
            if inbound.content.trim() == "/account"
                || inbound.content.trim().starts_with("/account ")
            {
                let args = inbound
                    .content
                    .trim()
                    .strip_prefix("/account")
                    .unwrap_or("")
                    .trim();
                let response =
                    handle_account_command(args, profile_id.as_deref(), &profile_store).await;
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
            // let take_photo_tool = take_photo_tool.clone();
            let spawn_tool = spawn_tool.clone();
            let cron_tool_arc = cron_tool_arc.clone();
            let pipeline_tool = pipeline_tool.clone();
            let llm_for_compaction = llm_for_compaction.clone();
            let out_tx = agent_handle.outbound_sender();
            let max_history = max_history.clone();
            let shutdown = shutdown.clone();
            let queue_mode = gw_config.queue_mode.clone();
            let collect_inbound_tx = collect_inbound_tx.clone();
            // Skip status indicator for cron/heartbeat messages — they're background tasks
            let status_indicator = if inbound.channel == "system" {
                None
            } else {
                status_indicators.get(&reply_channel).cloned()
            };

            let session_key_str = session_key.to_string();
            let locks_for_prune = session_locks.clone();
            let handle = tokio::spawn(async move {
                // Acquire concurrency permit (blocks if at max)
                let _permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => return, // semaphore closed
                };

                if shutdown.load(Ordering::Acquire) {
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

                let session_timeout = std::time::Duration::from_secs(session_timeout_secs);
                let result = tokio::time::timeout(
                    session_timeout,
                    process_session_message(
                        &agent,
                        &session_mgr,
                        &message_tool,
                        &send_file_tool,
                        // &take_photo_tool,
                        spawn_tool.as_deref(),
                        cron_tool_arc.as_deref(),
                        pipeline_tool.as_deref(),
                        &llm_for_compaction,
                        &out_tx,
                        &inbound,
                        &session_key,
                        &reply_channel,
                        &reply_chat_id,
                        image_media,
                        max_history.load(Ordering::Acquire),
                        &queue_mode,
                        &collect_inbound_tx,
                        status_indicator.as_deref(),
                    ),
                )
                .await;

                if result.is_err() {
                    tracing::error!(
                        session = %session_key,
                        timeout_secs = session_timeout_secs,
                        "session processing timed out"
                    );
                    if out_tx.send(OutboundMessage {
                        channel: reply_channel.to_string(),
                        chat_id: reply_chat_id.to_string(),
                        content: "Processing timed out. Please try again.".to_string(),
                        reply_to: None,
                        media: vec![],
                        metadata: serde_json::json!({}),
                    }).await.is_err() {
                        tracing::error!(session = %session_key, "outbound channel closed (timeout msg)");
                    }
                }
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
    // take_photo_tool: &TakePhotoTool,
    spawn_tool: Option<&SpawnTool>,
    cron_tool: Option<&CronTool>,
    pipeline_tool: Option<&crew_pipeline::RunPipelineTool>,
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
    status_indicator: Option<&StatusIndicator>,
) {
    // Set tool context for this session's reply routing
    // TODO: Race condition — these tools are shared Arc across all sessions.
    // If two sessions run concurrently, set_context() calls can interleave,
    // causing messages to be sent to the wrong chat. Fix requires per-session
    // ToolRegistry or Agent instances. Use MessageTool::with_context() /
    // SendFileTool::with_context() when per-session tool support is added.
    message_tool.set_context(reply_channel, reply_chat_id);
    send_file_tool.set_context(reply_channel, reply_chat_id);
    // take_photo_tool.set_context(reply_channel, reply_chat_id);
    if let Some(st) = spawn_tool {
        st.set_context(reply_channel, reply_chat_id); // TODO: same race condition
    }
    if let Some(ct) = cron_tool {
        ct.set_context(reply_channel, reply_chat_id); // TODO: same race condition
    }

    // Get conversation history (hold session_mgr lock briefly)
    let history: Vec<Message> = {
        let mut mgr = session_mgr.lock().await;
        let session = mgr.get_or_create(session_key);
        session.get_history(max_history).to_vec()
    };

    // Shared token tracker for real-time status updates
    let token_tracker = Arc::new(TokenTracker::new());

    // Start dynamic status indicator (typing + rotating status message + token counts)
    let status_handle = status_indicator.map(|si| {
        // Set up pipeline status bridge so pipeline nodes update the status words
        if let Some(pt) = pipeline_tool {
            let bridge = crew_pipeline::PipelineStatusBridge::new(
                si.status_words_handle(),
                Arc::clone(&token_tracker),
            );
            pt.set_status_bridge(bridge);
        }

        si.start(
            reply_chat_id.to_string(),
            &inbound.content,
            Arc::clone(&token_tracker),
        )
    });

    // Update hook context with current session ID
    agent.set_session_id(&session_key.to_string());

    // Process message through agent (potentially long LLM call, no lock held)
    let response = agent
        .process_message_tracked(&inbound.content, &history, image_media, &token_tracker)
        .await;

    // Stop status indicator and clean up status message
    if let Some(handle) = status_handle {
        handle.stop().await;
    }

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
                if let Err(e) = mgr.add_message(session_key, user_msg).await {
                    warn!(session = %session_key, error = %e, "failed to persist user message");
                }

                // Auto-generate summary from first user message
                {
                    let session = mgr.get_or_create(session_key);
                    if session.summary.is_none() && !inbound.content.trim().is_empty() {
                        let summary: String =
                            inbound.content.chars().take(100).collect();
                        session.summary = Some(summary);
                    }
                }

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
                    if let Err(e) = mgr.add_message(session_key, assistant_msg).await {
                        warn!(session = %session_key, error = %e, "failed to persist assistant message");
                    }
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
                if out_tx.send(outbound).await.is_err() {
                    tracing::error!(session = %session_key, "outbound channel closed");
                }
            }

            // Collect mode: not applicable in concurrent processing
            // (would require access to agent_handle which stays on main task)
            let _ = (queue_mode, collect_tx);
        }
        Err(e) => {
            tracing::error!(session = %session_key, error = %e, "agent processing failed");
            let error_msg = OutboundMessage {
                channel: reply_channel.to_string(),
                chat_id: reply_chat_id.to_string(),
                content: format!("Error: {e}"),
                reply_to: None,
                media: vec![],
                metadata: serde_json::json!({}),
            };
            if out_tx.send(error_msg).await.is_err() {
                tracing::error!(session = %session_key, "outbound channel closed (error msg)");
            }
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
    let compiled = include_str!("../prompts/gateway_default.txt");
    let runtime = super::load_prompt("gateway", compiled);
    let mut prompt = base.unwrap_or(&runtime).to_string();

    // Inject current date so the model knows "今年" = which year
    let today = chrono::Local::now().format("%Y-%m-%d");
    prompt.push_str(&format!("\n\nCurrent date: {today}"));

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

// ── /account command handler ─────────────────────────────────────────

async fn handle_account_command(
    args: &str,
    parent_profile_id: Option<&str>,
    profile_store: &Option<Arc<crate::profiles::ProfileStore>>,
) -> String {
    let parent_id = match parent_profile_id {
        Some(id) => id,
        None => return "Account management requires a profile-based gateway.".to_string(),
    };

    let store = match profile_store {
        Some(s) => s,
        None => {
            return "Account management is not available (no crew-home configured).".to_string();
        }
    };

    let parts: Vec<&str> = args.splitn(2, ' ').collect();
    match parts.first().copied().unwrap_or("list") {
        "" | "list" => match store.list_sub_accounts(parent_id) {
            Ok(subs) if subs.is_empty() => {
                format!("No sub-accounts.\nCreate one with: /account create <name>")
            }
            Ok(subs) => {
                let mut lines = vec!["Sub-accounts:".to_string()];
                for s in &subs {
                    let status = if s.enabled { "enabled" } else { "disabled" };
                    let ch_types: Vec<&str> = s
                        .config
                        .channels
                        .iter()
                        .map(|c| match c {
                            crate::profiles::ChannelCredentials::Telegram { .. } => "telegram",
                            crate::profiles::ChannelCredentials::Discord { .. } => "discord",
                            crate::profiles::ChannelCredentials::Slack { .. } => "slack",
                            crate::profiles::ChannelCredentials::WhatsApp { .. } => "whatsapp",
                            crate::profiles::ChannelCredentials::Feishu { .. } => "feishu",
                            crate::profiles::ChannelCredentials::Email { .. } => "email",
                            crate::profiles::ChannelCredentials::Twilio { .. } => "twilio",
                        })
                        .collect();
                    lines.push(format!(
                        "  {} — {} ({}) [{}]",
                        s.id,
                        s.name,
                        status,
                        ch_types.join(", ")
                    ));
                }
                lines.join("\n")
            }
            Err(e) => format!("Error: {e}"),
        },

        "create" => {
            let name = parts.get(1).copied().unwrap_or("").trim();
            if name.is_empty() {
                return "Usage: /account create <name>".to_string();
            }
            match store.create_sub_account(
                parent_id,
                name,
                vec![],
                crate::profiles::GatewaySettings::default(),
            ) {
                Ok(sub) => format!(
                    "Created sub-account: {}\nAdd channels via dashboard or CLI:\n  crew account create --profile {} {} --telegram-token <token>",
                    sub.id, parent_id, name
                ),
                Err(e) => format!("Error: {e}"),
            }
        }

        "delete" => {
            let sub_id = parts.get(1).copied().unwrap_or("").trim();
            if sub_id.is_empty() {
                return "Usage: /account delete <sub-id>".to_string();
            }
            // Safety: verify it's a sub-account of this parent
            match store.get(sub_id) {
                Ok(Some(sub)) if sub.parent_id.as_deref() == Some(parent_id) => {
                    match store.delete(sub_id) {
                        Ok(true) => format!("Deleted sub-account: {sub_id}"),
                        Ok(false) => format!("Sub-account '{sub_id}' not found"),
                        Err(e) => format!("Error: {e}"),
                    }
                }
                Ok(Some(_)) => format!("'{sub_id}' is not a sub-account of this profile."),
                Ok(None) => format!("Sub-account '{sub_id}' not found."),
                Err(e) => format!("Error: {e}"),
            }
        }

        // /account update <sub-id> key=value key=value ...
        // Keys: telegram-token, telegram-senders, whatsapp (true/false),
        //       feishu-app-id, feishu-app-secret, system-prompt, enabled (true/false)
        "update" => {
            let rest = parts.get(1).copied().unwrap_or("").trim();
            let mut tokens = rest.splitn(2, ' ');
            let sub_id = tokens.next().unwrap_or("").trim();
            let kv_str = tokens.next().unwrap_or("").trim();
            if sub_id.is_empty() || kv_str.is_empty() {
                return "Usage: /account update <sub-id> key=value [key=value ...]\n\
                    Keys: telegram-token, telegram-senders, whatsapp, \
                    feishu-app-id, feishu-app-secret, system-prompt, enabled\n\
                    Example: /account update my--bot telegram-token=123:ABC enabled=true"
                    .to_string();
            }
            // Verify it's a sub-account of this parent
            let mut profile = match store.get(sub_id) {
                Ok(Some(p)) if p.parent_id.as_deref() == Some(parent_id) => p,
                Ok(Some(_)) => {
                    return format!("'{sub_id}' is not a sub-account of this profile.")
                }
                Ok(None) => return format!("Sub-account '{sub_id}' not found."),
                Err(e) => return format!("Error: {e}"),
            };

            let mut changed = Vec::new();

            // Parse key=value pairs (simple split on '=')
            for pair in kv_str.split_whitespace() {
                let mut kv = pair.splitn(2, '=');
                let key = kv.next().unwrap_or("");
                let val = kv.next().unwrap_or("");
                match key {
                    "telegram-token" => {
                        let env_name = format!(
                            "TELEGRAM_BOT_TOKEN_{}",
                            profile.name.to_uppercase().replace(' ', "_").replace('-', "_")
                        );
                        profile
                            .config
                            .channels
                            .retain(|ch| !matches!(ch, crate::profiles::ChannelCredentials::Telegram { .. }));
                        profile
                            .config
                            .channels
                            .push(crate::profiles::ChannelCredentials::Telegram {
                                token_env: env_name.clone(),
                                allowed_senders: String::new(),
                            });
                        profile.config.env_vars.insert(env_name, val.to_string());
                        changed.push("telegram channel");
                    }
                    "telegram-senders" => {
                        let mut found = false;
                        for ch in &mut profile.config.channels {
                            if let crate::profiles::ChannelCredentials::Telegram {
                                allowed_senders, ..
                            } = ch
                            {
                                *allowed_senders = val.to_string();
                                found = true;
                            }
                        }
                        if found {
                            changed.push("telegram senders");
                        } else {
                            return "No Telegram channel to update senders on. Set telegram-token first.".to_string();
                        }
                    }
                    "whatsapp" => {
                        profile
                            .config
                            .channels
                            .retain(|ch| !matches!(ch, crate::profiles::ChannelCredentials::WhatsApp { .. }));
                        if val == "true" || val == "1" {
                            profile
                                .config
                                .channels
                                .push(crate::profiles::ChannelCredentials::WhatsApp {
                                    bridge_url: String::new(),
                                });
                            changed.push("whatsapp enabled");
                        } else {
                            changed.push("whatsapp disabled");
                        }
                    }
                    "feishu-app-id" | "feishu-app-secret" => {
                        // Collect both if provided; create/replace Feishu channel
                        let id_env = format!(
                            "LARK_APP_ID_{}",
                            profile.name.to_uppercase().replace(' ', "_").replace('-', "_")
                        );
                        let secret_env = format!(
                            "LARK_APP_SECRET_{}",
                            profile.name.to_uppercase().replace(' ', "_").replace('-', "_")
                        );
                        if key == "feishu-app-id" {
                            profile.config.env_vars.insert(id_env.clone(), val.to_string());
                        } else {
                            profile.config.env_vars.insert(secret_env.clone(), val.to_string());
                        }
                        // Ensure Feishu channel exists
                        if !profile.config.channels.iter().any(|ch| {
                            matches!(ch, crate::profiles::ChannelCredentials::Feishu { .. })
                        }) {
                            profile.config.channels.push(
                                crate::profiles::ChannelCredentials::Feishu {
                                    app_id_env: id_env,
                                    app_secret_env: secret_env,
                                    mode: "webhook".to_string(),
                                    region: String::new(),
                                    webhook_port: None,
                                    verification_token_env: String::new(),
                                    encrypt_key_env: String::new(),
                                },
                            );
                        }
                        changed.push("feishu channel");
                    }
                    "system-prompt" => {
                        profile.config.gateway.system_prompt = if val.is_empty() {
                            None
                        } else {
                            Some(val.to_string())
                        };
                        changed.push("system prompt");
                    }
                    "enabled" => {
                        let en = val == "true" || val == "1";
                        profile.enabled = en;
                        changed.push(if en { "enabled" } else { "disabled" });
                    }
                    _ => {
                        return format!("Unknown key: {key}\nValid keys: telegram-token, telegram-senders, whatsapp, feishu-app-id, feishu-app-secret, system-prompt, enabled");
                    }
                }
            }

            if changed.is_empty() {
                return "Nothing to update.".to_string();
            }

            profile.updated_at = chrono::Utc::now();
            match store.save(&profile) {
                Ok(()) => {
                    let mut msg = format!("Updated sub-account: {sub_id}");
                    for c in &changed {
                        msg.push_str(&format!("\n  - {c}"));
                    }
                    msg.push_str("\nGateway will auto-restart to pick up changes.");
                    msg
                }
                Err(e) => format!("Error saving: {e}"),
            }
        }

        // /account start <sub-id> — enable and trigger gateway start
        "start" | "enable" => {
            let sub_id = parts.get(1).copied().unwrap_or("").trim();
            if sub_id.is_empty() {
                return "Usage: /account start <sub-id>".to_string();
            }
            let mut profile = match store.get(sub_id) {
                Ok(Some(p)) if p.parent_id.as_deref() == Some(parent_id) => p,
                Ok(Some(_)) => {
                    return format!("'{sub_id}' is not a sub-account of this profile.")
                }
                Ok(None) => return format!("Sub-account '{sub_id}' not found."),
                Err(e) => return format!("Error: {e}"),
            };
            profile.enabled = true;
            profile.updated_at = chrono::Utc::now();
            match store.save(&profile) {
                Ok(()) => format!("Enabled sub-account: {sub_id}\nGateway will start within ~5 seconds."),
                Err(e) => format!("Error saving: {e}"),
            }
        }

        // /account stop <sub-id> — disable and trigger gateway stop
        "stop" | "disable" => {
            let sub_id = parts.get(1).copied().unwrap_or("").trim();
            if sub_id.is_empty() {
                return "Usage: /account stop <sub-id>".to_string();
            }
            let mut profile = match store.get(sub_id) {
                Ok(Some(p)) if p.parent_id.as_deref() == Some(parent_id) => p,
                Ok(Some(_)) => {
                    return format!("'{sub_id}' is not a sub-account of this profile.")
                }
                Ok(None) => return format!("Sub-account '{sub_id}' not found."),
                Err(e) => return format!("Error: {e}"),
            };
            profile.enabled = false;
            profile.updated_at = chrono::Utc::now();
            match store.save(&profile) {
                Ok(()) => format!("Disabled sub-account: {sub_id}\nGateway will stop within ~5 seconds."),
                Err(e) => format!("Error saving: {e}"),
            }
        }

        // /account restart <sub-id> — touch profile to trigger gateway restart
        "restart" => {
            let sub_id = parts.get(1).copied().unwrap_or("").trim();
            if sub_id.is_empty() {
                return "Usage: /account restart <sub-id>".to_string();
            }
            let mut profile = match store.get(sub_id) {
                Ok(Some(p)) if p.parent_id.as_deref() == Some(parent_id) => p,
                Ok(Some(_)) => {
                    return format!("'{sub_id}' is not a sub-account of this profile.")
                }
                Ok(None) => return format!("Sub-account '{sub_id}' not found."),
                Err(e) => return format!("Error: {e}"),
            };
            if !profile.enabled {
                return format!("Sub-account '{sub_id}' is disabled. Use /account start {sub_id} first.");
            }
            profile.updated_at = chrono::Utc::now();
            match store.save(&profile) {
                Ok(()) => format!("Restarting sub-account: {sub_id}\nGateway will restart within ~5 seconds."),
                Err(e) => format!("Error saving: {e}"),
            }
        }

        other => format!(
            "Unknown sub-command: {other}\nUsage: /account [list|create|update|delete|start|stop|restart]\n  /account list — list sub-accounts\n  /account create <name> — create sub-account\n  /account update <sub-id> key=value ... — update config\n  /account start <sub-id> — enable & start gateway\n  /account stop <sub-id> — disable & stop gateway\n  /account restart <sub-id> — restart gateway\n  /account delete <sub-id> — delete sub-account"
        ),
    }
}
