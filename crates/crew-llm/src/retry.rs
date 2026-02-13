//! Retry wrapper for LLM providers with exponential backoff.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use crew_core::Message;
use eyre::Result;
use tracing::{debug, warn};

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, ToolSpec};

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay between retries.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier for exponential backoff.
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }
}

/// Wrapper that adds retry logic to any LLM provider.
pub struct RetryProvider {
    inner: Arc<dyn LlmProvider>,
    config: RetryConfig,
}

impl RetryProvider {
    /// Create a new retry provider wrapping an existing provider.
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            inner: provider,
            config: RetryConfig::default(),
        }
    }

    /// Set custom retry configuration.
    pub fn with_config(mut self, config: RetryConfig) -> Self {
        self.config = config;
        self
    }

    /// Check if an error is retryable based on the error message.
    fn is_retryable_error(error: &eyre::Report) -> bool {
        let error_str = error.to_string().to_lowercase();

        // Rate limiting
        if error_str.contains("429") || error_str.contains("rate limit") {
            return true;
        }

        // Server errors (5xx)
        if error_str.contains("500")
            || error_str.contains("502")
            || error_str.contains("503")
            || error_str.contains("504")
            || error_str.contains("internal server error")
            || error_str.contains("bad gateway")
            || error_str.contains("service unavailable")
            || error_str.contains("gateway timeout")
        {
            return true;
        }

        // Network errors
        if error_str.contains("connection")
            || error_str.contains("timeout")
            || error_str.contains("network")
        {
            return true;
        }

        // Overloaded
        if error_str.contains("overloaded") {
            return true;
        }

        false
    }

    fn calculate_delay(&self, attempt: u32) -> Duration {
        let delay = self.config.initial_delay.as_secs_f64()
            * self.config.backoff_multiplier.powi(attempt as i32);
        let delay = Duration::from_secs_f64(delay);
        std::cmp::min(delay, self.config.max_delay)
    }
}

#[async_trait]
impl LlmProvider for RetryProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatResponse> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.inner.chat(messages, tools, config).await {
                Ok(response) => {
                    if attempt > 0 {
                        debug!(attempt, "request succeeded after retry");
                    }
                    return Ok(response);
                }
                Err(e) => {
                    if attempt < self.config.max_retries && Self::is_retryable_error(&e) {
                        let delay = self.calculate_delay(attempt);
                        warn!(
                            attempt = attempt + 1,
                            max_retries = self.config.max_retries,
                            delay_secs = delay.as_secs_f64(),
                            error = %e,
                            "retryable error, backing off"
                        );
                        tokio::time::sleep(delay).await;
                        last_error = Some(e);
                    } else {
                        // Non-retryable error or max retries exceeded
                        return Err(e);
                    }
                }
            }
        }

        // Should only get here if we exhausted retries
        Err(last_error.unwrap_or_else(|| eyre::eyre!("unknown error after retries")))
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatStream> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.inner.chat_stream(messages, tools, config).await {
                Ok(stream) => {
                    if attempt > 0 {
                        debug!(attempt, "stream request succeeded after retry");
                    }
                    return Ok(stream);
                }
                Err(e) => {
                    if attempt < self.config.max_retries && Self::is_retryable_error(&e) {
                        let delay = self.calculate_delay(attempt);
                        warn!(
                            attempt = attempt + 1,
                            max_retries = self.config.max_retries,
                            delay_secs = delay.as_secs_f64(),
                            error = %e,
                            "retryable stream error, backing off"
                        );
                        tokio::time::sleep(delay).await;
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| eyre::eyre!("unknown error after retries")))
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_retryable_429() {
        let err = eyre::eyre!("API error: 429 - rate limited");
        assert!(RetryProvider::is_retryable_error(&err));
    }

    #[test]
    fn test_is_retryable_500() {
        let err = eyre::eyre!("API error: 500 - internal server error");
        assert!(RetryProvider::is_retryable_error(&err));
    }

    #[test]
    fn test_is_retryable_503() {
        let err = eyre::eyre!("API error: 503 - service unavailable");
        assert!(RetryProvider::is_retryable_error(&err));
    }

    #[test]
    fn test_not_retryable_401() {
        let err = eyre::eyre!("API error: 401 - unauthorized");
        assert!(!RetryProvider::is_retryable_error(&err));
    }

    #[test]
    fn test_not_retryable_400() {
        let err = eyre::eyre!("API error: 400 - bad request");
        assert!(!RetryProvider::is_retryable_error(&err));
    }

    #[test]
    fn test_calculate_delay() {
        let provider = RetryProvider {
            inner: Arc::new(MockProvider),
            config: RetryConfig {
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 2.0,
                ..Default::default()
            },
        };

        assert_eq!(provider.calculate_delay(0), Duration::from_secs(1));
        assert_eq!(provider.calculate_delay(1), Duration::from_secs(2));
        assert_eq!(provider.calculate_delay(2), Duration::from_secs(4));
        assert_eq!(provider.calculate_delay(3), Duration::from_secs(8));
        // Should cap at max_delay
        assert_eq!(provider.calculate_delay(10), Duration::from_secs(60));
    }

    struct MockProvider;

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _config: &ChatConfig,
        ) -> Result<ChatResponse> {
            unimplemented!()
        }

        fn model_id(&self) -> &str {
            "mock"
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }
}
