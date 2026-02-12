//! Auth command: login, logout, status for LLM providers.

use clap::{Args, Subcommand};
use colored::Colorize;
use eyre::Result;

use super::Executable;
use crate::auth::{AuthStore, oauth, token};

/// Manage authentication for LLM providers.
#[derive(Debug, Args)]
pub struct AuthCommand {
    #[command(subcommand)]
    pub action: AuthAction,
}

#[derive(Debug, Subcommand)]
pub enum AuthAction {
    /// Log in to an LLM provider.
    Login {
        /// Provider name (openai, anthropic, gemini, etc.).
        #[arg(long, short)]
        provider: String,

        /// Use device code flow instead of browser (OpenAI only).
        #[arg(long)]
        device_code: bool,
    },
    /// Log out from a provider.
    Logout {
        /// Provider name.
        #[arg(long, short)]
        provider: String,
    },
    /// Show authentication status for all providers.
    Status,
}

impl Executable for AuthCommand {
    fn execute(self) -> Result<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(self.run_async())
    }
}

impl AuthCommand {
    async fn run_async(self) -> Result<()> {
        match self.action {
            AuthAction::Login {
                provider,
                device_code,
            } => login(&provider, device_code).await,
            AuthAction::Logout { provider } => logout(&provider),
            AuthAction::Status => status(),
        }
    }
}

async fn login(provider: &str, device_code: bool) -> Result<()> {
    let cred = match provider {
        "openai" => {
            if device_code {
                oauth::device_code_flow().await?
            } else {
                oauth::browser_oauth_flow().await?
            }
        }
        // All other providers use paste-token flow.
        _ => token::paste_token_flow(provider)?,
    };

    let mut store = AuthStore::load()?;
    store.set(provider, cred)?;

    println!(
        "{} Logged in to {} (credentials saved)",
        "OK".green().bold(),
        provider
    );
    Ok(())
}

fn logout(provider: &str) -> Result<()> {
    let mut store = AuthStore::load()?;
    if store.remove(provider)? {
        println!(
            "{} Logged out from {}",
            "OK".green().bold(),
            provider
        );
    } else {
        println!("No credentials found for {provider}");
    }
    Ok(())
}

fn status() -> Result<()> {
    let store = AuthStore::load()?;
    let creds: Vec<_> = store.list().collect();

    if creds.is_empty() {
        println!("No saved credentials. Use {} to log in.", "crew auth login".cyan());
        return Ok(());
    }

    println!("{}", "Authenticated providers:".bold());
    for (name, cred) in creds {
        let method = &cred.auth_method;
        let status = if cred.is_expired() {
            "expired".red().to_string()
        } else {
            "active".green().to_string()
        };
        let expiry = cred
            .expires_at
            .map(|t| format!(" (expires {})", t.format("%Y-%m-%d %H:%M UTC")))
            .unwrap_or_default();
        println!("  {name}: {status} [{method}]{expiry}");
    }
    Ok(())
}
