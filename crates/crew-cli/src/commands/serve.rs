//! Serve command: start the REST API server.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use colored::Colorize;
use crew_agent::{Agent, AgentConfig, HookExecutor, ToolRegistry};
use crew_bus::SessionManager;
use crew_core::AgentId;
use crew_llm::{LlmProvider, RetryProvider};
use crew_memory::EpisodeStore;
use eyre::{Result, WrapErr};

use super::Executable;
use super::chat::create_provider;
use crate::api::metrics::MetricsReporter;
use crate::api::{AppState, SseBroadcaster, build_router, init_metrics};
use crate::config::Config;

/// Start the REST API server.
#[derive(Debug, Args)]
pub struct ServeCommand {
    /// Port to listen on.
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Host address to bind to. Defaults to localhost for security.
    /// Use 0.0.0.0 to accept connections from all interfaces.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Working directory (defaults to current directory).
    #[arg(short, long)]
    pub cwd: Option<PathBuf>,

    /// Data directory for episodes, memory, sessions (defaults to $CREW_HOME or ~/.crew).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Path to config file.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// LLM provider to use (overrides config).
    #[arg(long)]
    pub provider: Option<String>,

    /// Model to use (overrides config).
    #[arg(long)]
    pub model: Option<String>,

    /// Auth token for API access (overrides config).
    #[arg(long)]
    pub auth_token: Option<String>,

    /// Disable automatic retry on transient errors.
    #[arg(long)]
    pub no_retry: bool,
}

impl Executable for ServeCommand {
    fn execute(self) -> Result<()> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .wrap_err("failed to create tokio runtime")?
            .block_on(self.run_async())
    }
}

impl ServeCommand {
    async fn run_async(self) -> Result<()> {
        let cwd = match &self.cwd {
            Some(p) => p.clone(),
            None => std::env::current_dir().wrap_err("failed to get current directory")?,
        };

        let (config, resolved_config_path) = if let Some(config_path) = &self.config {
            (Config::from_file(config_path)?, Some(config_path.clone()))
        } else {
            // Resolve config path the same way Config::load does
            let local_config = cwd.join(".crew").join("config.json");
            if local_config.exists() {
                (Config::from_file(&local_config)?, Some(local_config))
            } else if let Some(global_config) = Config::global_config_path() {
                if global_config.exists() {
                    (Config::from_file(&global_config)?, Some(global_config))
                } else {
                    (Config::default(), None)
                }
            } else {
                (Config::default(), None)
            }
        };

        // Resolve data directory (--data-dir > $CREW_HOME > ~/.crew)
        let data_dir = super::resolve_data_dir(self.data_dir.clone())?;
        tracing::info!(data_dir = %data_dir.display(), "data directory resolved");

        // Try to create the LLM provider + agent, but don't fail if no API key.
        // The admin dashboard works without it.
        let agent_and_sessions = self.try_create_agent(&config, &cwd, &data_dir).await;

        let (agent, sessions) = match agent_and_sessions {
            Ok((a, s)) => (Some(Arc::new(a)), Some(s)),
            Err(e) => {
                tracing::warn!("LLM agent not available: {e}");
                tracing::info!("Admin dashboard will still work. Configure profiles via /admin/");
                (None, None)
            }
        };

        let broadcaster = Arc::new(SseBroadcaster::new(256));
        let metrics_handle = Some(init_metrics());

        // Security: warn if binding to non-localhost without auth token
        let auth_token = if self.auth_token.is_some() {
            self.auth_token
        } else if self.host != "127.0.0.1" && self.host != "localhost" && self.host != "::1" {
            tracing::warn!(
                "Binding to {} without --auth-token is dangerous! \
                 Generating a random token for this session.",
                self.host
            );
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut h);
            std::process::id().hash(&mut h);
            let token = format!(
                "{:016x}{:016x}",
                h.finish(),
                h.finish().wrapping_mul(6364136223846793005)
            );
            println!(
                "{}: {} (auto-generated, pass --auth-token to set your own)",
                "Auth token".yellow(),
                token
            );
            Some(token)
        } else {
            None
        };

        // Initialize profile store and process manager for admin dashboard
        tracing::info!("initializing profile store and process manager");
        let profile_store = Arc::new(
            crate::profiles::ProfileStore::open(&data_dir)
                .wrap_err("failed to open profile store")?,
        );
        let bridge_js_path = data_dir.join("whatsapp-bridge").join("bridge.js");
        let process_manager = Arc::new(
            crate::process_manager::ProcessManager::new(profile_store.clone())
                .with_bridge_js(bridge_js_path),
        );

        // Initialize user store and auth manager for multi-user support
        let user_store = Arc::new(
            crate::user_store::UserStore::open(&data_dir).wrap_err("failed to open user store")?,
        );
        let auth_manager = {
            let auth_config = config.dashboard_auth.clone();
            if auth_config.is_none() {
                tracing::warn!("no dashboard_auth.smtp configured — OTP codes will be logged to console only");
            }
            Some(Arc::new(crate::otp::AuthManager::new(
                auth_config,
                user_store.clone(),
            )))
        };

        // Spawn auth cleanup task if auth manager is active
        if let Some(ref am) = auth_manager {
            let am_clone = am.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
                loop {
                    interval.tick().await;
                    am_clone.cleanup().await;
                }
            });
        }

        let state = Arc::new(AppState {
            agent,
            sessions,
            broadcaster,
            started_at: chrono::Utc::now(),
            auth_token,
            metrics_handle,
            profile_store: Some(profile_store.clone()),
            process_manager: Some(process_manager.clone()),
            user_store: Some(user_store),
            auth_manager,
            http_client: reqwest::Client::new(),
            config_path: resolved_config_path,
        });

        // Auto-start enabled profiles
        let profiles = profile_store.list().unwrap_or_default();
        let enabled_count = profiles.iter().filter(|p| p.enabled).count();
        tracing::info!(total = profiles.len(), enabled = enabled_count, "loaded profiles");
        if enabled_count > 0 {
            for p in &profiles {
                if p.enabled {
                    tracing::info!(profile = %p.id, "auto-starting gateway");
                    if let Err(e) = process_manager.start(p).await {
                        tracing::warn!(profile = %p.id, error = %e, "failed to auto-start gateway");
                    }
                }
            }
        }

        // Profile file watcher: auto-restart gateways when profile JSON changes.
        {
            let ps = profile_store.clone();
            let pm = process_manager.clone();
            tokio::spawn(async move {
                use crate::profiles::{ProfileChange, UserProfile, diff_profiles};
                use sha2::{Digest, Sha256};
                use std::collections::HashMap;

                // Snapshot of known profile states: (hash, profile)
                let mut known: HashMap<String, ([u8; 32], UserProfile)> = HashMap::new();
                // Seed with current profiles
                if let Ok(list) = ps.list() {
                    for p in list {
                        if let Ok(bytes) = std::fs::read(ps.profile_path(&p.id)) {
                            let hash: [u8; 32] = Sha256::digest(&bytes).into();
                            known.insert(p.id.clone(), (hash, p));
                        }
                    }
                }

                let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
                loop {
                    interval.tick().await;
                    let current = match ps.list() {
                        Ok(list) => list,
                        Err(_) => continue,
                    };
                    for profile in &current {
                        let bytes = match std::fs::read(ps.profile_path(&profile.id)) {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                        let hash: [u8; 32] = Sha256::digest(&bytes).into();

                        if let Some((old_hash, old_profile)) = known.get(&profile.id) {
                            if hash == *old_hash {
                                continue; // no change
                            }
                            // File changed — check if gateway is running
                            let status = pm.status(&profile.id).await;
                            if status.running {
                                match diff_profiles(old_profile, profile) {
                                    ProfileChange::RestartRequired(fields) => {
                                        tracing::info!(
                                            profile = %profile.id,
                                            fields = ?fields,
                                            "profile changed (restart-required fields), restarting gateway"
                                        );
                                        if let Err(e) = pm.restart(profile).await {
                                            tracing::warn!(
                                                profile = %profile.id,
                                                error = %e,
                                                "failed to restart gateway after profile change"
                                            );
                                        }
                                    }
                                    ProfileChange::HotReloadable => {
                                        // Gateway's own ConfigWatcher handles hot-reload
                                        tracing::debug!(
                                            profile = %profile.id,
                                            "profile changed (hot-reloadable only), gateway watcher will handle"
                                        );
                                    }
                                    ProfileChange::Unchanged => {}
                                }
                            }
                        }
                        known.insert(profile.id.clone(), (hash, profile.clone()));
                    }
                }
            });
        }

        // Start admin bot if configured
        #[cfg(feature = "admin-bot")]
        if let Some(ref admin_config) = config.admin_bot {
            let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
            match crate::admin_bot::AdminBot::new(
                admin_config,
                profile_store.clone(),
                process_manager.clone(),
                shutdown,
                &data_dir,
            )
            .await
            {
                Ok(admin_bot) => {
                    tokio::spawn(async move {
                        if let Err(e) = admin_bot.run().await {
                            tracing::error!(error = %e, "admin bot error");
                        }
                    });
                    tracing::info!("admin bot spawned");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to start admin bot (continuing without it)");
                }
            }
        }

        let app = build_router(state);
        let addr = format!("{}:{}", self.host, self.port);

        tracing::info!(address = %addr, "crew-rs API server starting");
        tracing::info!(dashboard = %format!("http://{}/admin/", addr), "dashboard available");
        if enabled_count > 0 {
            tracing::info!(count = enabled_count, "gateway profiles auto-started");
        }

        println!("{}", "crew-rs API server".cyan().bold());
        println!("{}: http://{}", "Listening".green(), addr);
        println!("{}: http://{}/admin/", "Dashboard".green(), addr);
        if enabled_count > 0 {
            println!(
                "{}: {} profiles auto-started",
                "Gateways".green(),
                enabled_count
            );
        }
        println!();

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                println!();
                println!("{}", "Shutting down server...".yellow());
            })
            .await?;

        // Stop all gateway child processes before exiting
        tracing::info!("stopping all gateway child processes");
        println!("{}", "Stopping gateways...".yellow());
        let stopped = process_manager.stop_all().await;
        if stopped > 0 {
            tracing::info!(count = stopped, "gateways stopped");
            println!("  stopped {} gateway(s)", stopped);
        }

        // Force exit — background tokio tasks (profile watcher, auth cleanup,
        // admin bot) have no shutdown signal and would hang indefinitely.
        std::process::exit(0);
    }

    /// Try to create an Agent + SessionManager. Returns Err if API key is missing etc.
    async fn try_create_agent(
        &self,
        config: &Config,
        cwd: &std::path::Path,
        data_dir: &std::path::Path,
    ) -> Result<(Agent, Arc<tokio::sync::Mutex<SessionManager>>)> {
        let model = self.model.clone().or(config.model.clone());
        let base_url = config.base_url.clone();
        let provider_name = self
            .provider
            .clone()
            .or(config.provider.clone())
            .or_else(|| {
                model
                    .as_deref()
                    .and_then(crate::config::detect_provider)
                    .map(String::from)
            })
            .unwrap_or_else(|| "anthropic".to_string());

        let base_provider: Arc<dyn LlmProvider> =
            create_provider(&provider_name, config, model, base_url)?;

        let llm: Arc<dyn LlmProvider> = if self.no_retry {
            base_provider
        } else {
            Arc::new(RetryProvider::new(base_provider))
        };

        let memory = Arc::new(
            EpisodeStore::open(data_dir)
                .await
                .wrap_err("failed to open episode store")?,
        );

        let sandbox = crew_agent::create_sandbox(&config.sandbox);
        let mut tools = ToolRegistry::with_builtins_and_sandbox(cwd, sandbox);

        // Open tool config store for user-customizable tool defaults
        let tool_config = std::sync::Arc::new(
            crew_agent::ToolConfigStore::open(data_dir)
                .await
                .wrap_err("failed to open tool config store")?,
        );
        tools.inject_tool_config(tool_config);

        // MCP tools
        if !config.mcp_servers.is_empty() {
            match crew_agent::McpClient::start(&config.mcp_servers).await {
                Ok(client) => client.register_tools(&mut tools),
                Err(e) => tracing::warn!("MCP initialization failed: {e}"),
            }
        }

        // Plugins
        let plugin_dirs = Config::plugin_dirs(cwd);
        if !plugin_dirs.is_empty() {
            let _ = crew_agent::PluginLoader::load_into(&mut tools, &plugin_dirs);
        }

        let reporter: Arc<dyn crew_agent::ProgressReporter> =
            Arc::new(MetricsReporter::new(Arc::new(crew_agent::SilentReporter)));

        let mut agent = Agent::new(AgentId::new("api"), llm, tools, memory)
            .with_config(AgentConfig {
                max_iterations: 20,
                save_episodes: false,
                ..Default::default()
            })
            .with_reporter(reporter);

        if !config.hooks.is_empty() {
            agent = agent.with_hooks(Arc::new(HookExecutor::new(config.hooks.clone())));
        }

        let sessions = Arc::new(tokio::sync::Mutex::new(
            SessionManager::open(data_dir).wrap_err("failed to open session manager")?,
        ));

        Ok((agent, sessions))
    }
}
