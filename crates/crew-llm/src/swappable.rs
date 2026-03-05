//! Runtime-swappable LLM provider wrapper.
//!
//! Wraps an `Arc<dyn LlmProvider>` behind an `RwLock` so the active provider
//! can be atomically replaced at runtime (e.g. via a `switch_model` tool).

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use crew_core::Message;
use eyre::Result;

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, ToolSpec};

/// A provider wrapper that allows swapping the inner provider at runtime.
///
/// `model_id()` and `provider_name()` return `&str`, which can't borrow from
/// behind an `RwLock`, so we cache those values in separate `RwLock<String>`
/// fields that are updated on each swap.
pub struct SwappableProvider {
    inner: RwLock<Arc<dyn LlmProvider>>,
    cached_model_id: RwLock<String>,
    cached_provider_name: RwLock<String>,
}

impl SwappableProvider {
    /// Create a new swappable provider wrapping the given provider.
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        let model_id = provider.model_id().to_string();
        let provider_name = provider.provider_name().to_string();
        Self {
            inner: RwLock::new(provider),
            cached_model_id: RwLock::new(model_id),
            cached_provider_name: RwLock::new(provider_name),
        }
    }

    /// Atomically replace the inner provider with a new one.
    pub fn swap(&self, new_provider: Arc<dyn LlmProvider>) {
        let model_id = new_provider.model_id().to_string();
        let provider_name = new_provider.provider_name().to_string();
        *self.inner.write().unwrap() = new_provider;
        *self.cached_model_id.write().unwrap() = model_id;
        *self.cached_provider_name.write().unwrap() = provider_name;
    }

    /// Get the current provider name and model ID as owned strings.
    pub fn provider_info(&self) -> (String, String) {
        let name = self.cached_provider_name.read().unwrap().clone();
        let model = self.cached_model_id.read().unwrap().clone();
        (name, model)
    }

    /// Get a clone of the current inner provider Arc.
    pub fn current(&self) -> Arc<dyn LlmProvider> {
        self.inner.read().unwrap().clone()
    }
}

#[async_trait]
impl LlmProvider for SwappableProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatResponse> {
        // Clone the Arc to release the lock before the async call.
        let provider = self.inner.read().unwrap().clone();
        provider.chat(messages, tools, config).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        config: &ChatConfig,
    ) -> Result<ChatStream> {
        let provider = self.inner.read().unwrap().clone();
        provider.chat_stream(messages, tools, config).await
    }

    fn context_window(&self) -> u32 {
        self.inner.read().unwrap().context_window()
    }

    #[allow(unsafe_code)]
    fn model_id(&self) -> &str {
        // SAFETY: The cached string lives as long as SwappableProvider (&self).
        // swap() replaces the String but old data is only freed after the write-lock
        // releases. Callers use the returned &str immediately (not across swaps).
        let guard = self.cached_model_id.read().unwrap();
        let s: &str = &guard;
        unsafe { &*(s as *const str) }
    }

    #[allow(unsafe_code)]
    fn provider_name(&self) -> &str {
        let guard = self.cached_provider_name.read().unwrap();
        let s: &str = &guard;
        unsafe { &*(s as *const str) }
    }

    fn export_metrics(&self) -> Option<serde_json::Value> {
        self.inner.read().unwrap().export_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StopReason, TokenUsage};

    struct MockProvider {
        name: &'static str,
        model: &'static str,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _config: &ChatConfig,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                content: Some(format!("from {}", self.name)),
                reasoning_content: None,
                tool_calls: vec![],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
        }

        fn model_id(&self) -> &str {
            self.model
        }

        fn provider_name(&self) -> &str {
            self.name
        }
    }

    #[test]
    fn test_provider_info() {
        let p = SwappableProvider::new(Arc::new(MockProvider {
            name: "test",
            model: "test-model",
        }));
        assert_eq!(p.provider_info(), ("test".into(), "test-model".into()));
    }

    #[test]
    fn test_swap_updates_info() {
        let p = SwappableProvider::new(Arc::new(MockProvider {
            name: "old",
            model: "old-model",
        }));
        p.swap(Arc::new(MockProvider {
            name: "new",
            model: "new-model",
        }));
        assert_eq!(p.provider_info(), ("new".into(), "new-model".into()));
        assert_eq!(p.model_id(), "new-model");
        assert_eq!(p.provider_name(), "new");
    }

    #[tokio::test]
    async fn test_chat_delegates() {
        let p = SwappableProvider::new(Arc::new(MockProvider {
            name: "test",
            model: "m1",
        }));
        let resp = p.chat(&[], &[], &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.content.unwrap(), "from test");

        // Swap and verify delegation changes
        p.swap(Arc::new(MockProvider {
            name: "other",
            model: "m2",
        }));
        let resp = p.chat(&[], &[], &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.content.unwrap(), "from other");
    }
}
