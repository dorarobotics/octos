//! CLI commands for crew-rs.

mod auth;
mod channels;
mod chat;
mod clean;
mod completions;
mod cron;
mod docs;
mod gateway;
mod init;
mod office;
#[cfg(feature = "api")]
mod serve;
mod skills;
mod status;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use eyre::Result;

pub use auth::AuthCommand;
pub use channels::ChannelsCommand;
pub use chat::ChatCommand;
pub use clean::CleanCommand;
pub use completions::CompletionsCommand;
pub use cron::CronCommand;
pub use docs::DocsCommand;
pub use gateway::GatewayCommand;
pub use init::InitCommand;
pub use office::OfficeCommand;
#[cfg(feature = "api")]
pub use serve::ServeCommand;
pub use skills::SkillsCommand;
pub use status::StatusCommand;

/// crew-rs: Rust-native coding agent orchestration.
#[derive(Debug, Parser)]
#[command(name = "crew")]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

/// Available commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Manage authentication for LLM providers.
    Auth(AuthCommand),
    /// Manage messaging channels.
    Channels(ChannelsCommand),
    /// Interactive multi-turn chat with an agent.
    Chat(ChatCommand),
    /// Manage scheduled cron jobs.
    Cron(CronCommand),
    /// Generate documentation for tools and providers.
    Docs(DocsCommand),
    /// Initialize a new .crew configuration.
    Init(InitCommand),
    /// Start the REST API server (requires --features api).
    #[cfg(feature = "api")]
    Serve(ServeCommand),
    /// Manage agent skills (list, install, remove).
    Skills(SkillsCommand),
    /// Show system status.
    Status(StatusCommand),
    /// Run as a persistent messaging gateway.
    Gateway(GatewayCommand),
    /// Clean up stale state and cache files.
    Clean(CleanCommand),
    /// Generate shell completions.
    Completions(CompletionsCommand),
    /// Office file manipulation (extract, unpack, pack, clean, add-slide, validate).
    Office(OfficeCommand),
}

/// Trait for executable commands (following dora-rs pattern).
pub trait Executable {
    fn execute(self) -> Result<()>;
}

/// Resolve the data directory for episodes, memory, sessions, etc.
///
/// Priority: `--data-dir` CLI flag > `CREW_HOME` env var > `~/.crew` default.
pub(crate) fn resolve_data_dir(cli_override: Option<PathBuf>) -> eyre::Result<PathBuf> {
    let dir = if let Some(d) = cli_override {
        d
    } else if let Ok(env_dir) = std::env::var("CREW_HOME") {
        PathBuf::from(env_dir)
    } else {
        dirs::home_dir()
            .ok_or_else(|| eyre::eyre!("cannot determine home directory"))?
            .join(".crew")
    };
    std::fs::create_dir_all(&dir).ok();
    Ok(dir)
}

/// Load optional bootstrap/personality files from the .crew/ directory.
/// Used by both chat and gateway to build the system prompt from AGENTS.md, SOUL.md, etc.
pub(crate) fn load_bootstrap_files(data_dir: &std::path::Path) -> String {
    const FILES: &[&str] = &["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"];
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

impl Executable for Command {
    fn execute(self) -> Result<()> {
        match self {
            Self::Auth(cmd) => cmd.execute(),
            Self::Channels(cmd) => cmd.execute(),
            Self::Chat(cmd) => cmd.execute(),
            Self::Cron(cmd) => cmd.execute(),
            Self::Docs(cmd) => cmd.execute(),
            Self::Init(cmd) => cmd.execute(),
            #[cfg(feature = "api")]
            Self::Serve(cmd) => cmd.execute(),
            Self::Skills(cmd) => cmd.execute(),
            Self::Status(cmd) => cmd.execute(),
            Self::Gateway(cmd) => cmd.execute(),
            Self::Clean(cmd) => cmd.execute(),
            Self::Completions(cmd) => cmd.execute(),
            Self::Office(cmd) => cmd.execute(),
        }
    }
}
