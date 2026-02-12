//! CLI commands for crew-rs.

mod auth;
mod channels;
mod chat;
mod clean;
mod completions;
mod cron;
mod gateway;
mod init;
mod skills;
mod status;

use clap::{Parser, Subcommand};
use eyre::Result;

pub use auth::AuthCommand;
pub use channels::ChannelsCommand;
pub use chat::ChatCommand;
pub use clean::CleanCommand;
pub use completions::CompletionsCommand;
pub use cron::CronCommand;
pub use gateway::GatewayCommand;
pub use init::InitCommand;
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
    /// Initialize a new .crew configuration.
    Init(InitCommand),
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
}

/// Trait for executable commands (following dora-rs pattern).
pub trait Executable {
    fn execute(self) -> Result<()>;
}

impl Executable for Command {
    fn execute(self) -> Result<()> {
        match self {
            Self::Auth(cmd) => cmd.execute(),
            Self::Channels(cmd) => cmd.execute(),
            Self::Chat(cmd) => cmd.execute(),
            Self::Cron(cmd) => cmd.execute(),
            Self::Init(cmd) => cmd.execute(),
            Self::Skills(cmd) => cmd.execute(),
            Self::Status(cmd) => cmd.execute(),
            Self::Gateway(cmd) => cmd.execute(),
            Self::Clean(cmd) => cmd.execute(),
            Self::Completions(cmd) => cmd.execute(),
        }
    }
}
