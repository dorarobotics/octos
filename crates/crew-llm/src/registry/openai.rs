use std::sync::Arc;

use eyre::Result;

use crate::openai::OpenAIProvider;
use crate::provider::LlmProvider;

use super::{CreateParams, ProviderEntry};

pub const ENTRY: ProviderEntry = ProviderEntry {
    name: "openai",
    aliases: &[],
    default_model: Some("gpt-4o"),
    api_key_env: Some("OPENAI_API_KEY"),
    default_base_url: Some("https://api.openai.com/v1"),
    requires_api_key: true,
    requires_base_url: false,
    requires_model: false,
    detect_patterns: &["gpt"],
    create,
};

fn create(p: CreateParams) -> Result<Arc<dyn LlmProvider>> {
    let key = p
        .api_key
        .ok_or_else(|| eyre::eyre!("OPENAI_API_KEY not set"))?;
    let model = p.model.unwrap_or_else(|| "gpt-4o".into());
    let mut provider = OpenAIProvider::new(&key, &model);
    if let Some(url) = p.base_url {
        provider = provider.with_base_url(&url);
    }
    if let Some(hints) = p.model_hints {
        provider = provider.with_hints(hints);
    }
    Ok(Arc::new(provider))
}
