//! crew-rs CLI entry point.

use clap::Parser;
use color_eyre::eyre::Result;

#[cfg(feature = "api")]
pub mod api;
pub mod auth;
mod commands;
pub mod compaction;
pub mod config;
pub mod config_watcher;
pub mod cron_tool;
#[cfg(feature = "api")]
pub mod otp;
pub mod persona_service;
#[cfg(feature = "api")]
pub mod process_manager;
pub mod profiles;
#[cfg(feature = "api")]
pub mod user_store;

use commands::{Args, Executable};

fn main() -> Result<()> {
    // Initialize error handling
    color_eyre::install()?;

    // Initialize tracing
    init_tracing()?;

    // Parse arguments and execute command
    let args = Args::parse();
    args.command.execute()
}

fn init_tracing() -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"))
        // Suppress noisy HTML5 parser warnings ("foster parenting not implemented")
        .add_directive("html5ever=error".parse().unwrap());

    // Check if JSON format is requested via environment
    let json_logs = std::env::var("CREW_LOG_JSON").is_ok();

    if json_logs {
        // JSON format for structured logging (good for log aggregation)
        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .json()
                    .with_target(true)
                    .with_span_list(true)
                    .with_current_span(true),
            )
            .with(filter)
            .init();
    } else {
        // Human-readable format (default)
        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .with_target(false)
                    .with_thread_ids(false)
                    .compact(),
            )
            .with(filter)
            .init();
    }

    Ok(())
}
