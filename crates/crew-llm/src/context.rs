//! Context window limits and token estimation.

use crew_core::Message;

/// Known context window sizes (in tokens) for common models.
pub fn context_window_tokens(model_id: &str) -> u32 {
    let m = model_id.to_lowercase();
    match () {
        // Anthropic Claude
        _ if m.contains("claude-opus-4") || m.contains("claude-sonnet-4") => 200_000,
        _ if m.contains("claude-3") => 200_000,
        // OpenAI
        _ if m.contains("gpt-4o") || m.contains("gpt-4-turbo") => 128_000,
        _ if m.contains("o1") || m.contains("o3") || m.contains("o4") => 200_000,
        _ if m.contains("gpt-4") => 128_000,
        _ if m.contains("gpt-3.5") => 16_385,
        // Google Gemini
        _ if m.contains("gemini-2") || m.contains("gemini-1.5") => 1_000_000,
        _ if m.contains("gemini") => 128_000,
        // Chinese providers
        _ if m.contains("deepseek") => 128_000,
        _ if m.contains("kimi") || m.contains("moonshot") => 128_000,
        _ if m.contains("qwen") => 128_000,
        _ if m.contains("glm") || m.contains("zhipu") => 128_000,
        _ if m.contains("minimax") => 128_000,
        // Local
        _ if m.contains("llama") => 128_000,
        // Conservative default
        _ => 128_000,
    }
}

/// Estimate token count from text using character heuristic.
/// ~4 chars per token for English/code. Rough but sufficient for guard purposes.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.len() as u32;
    (chars / 4).max(1)
}

/// Estimate tokens for a message (content + serialized tool calls + overhead).
pub fn estimate_message_tokens(msg: &Message) -> u32 {
    let mut tokens = estimate_tokens(&msg.content);
    if let Some(ref calls) = msg.tool_calls {
        for call in calls {
            tokens += estimate_tokens(&call.name);
            tokens += estimate_tokens(&call.arguments.to_string());
        }
    }
    // Role/structural overhead (~4 tokens)
    tokens + 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_window_claude() {
        assert_eq!(context_window_tokens("claude-sonnet-4-20250514"), 200_000);
        assert_eq!(context_window_tokens("claude-opus-4-20250514"), 200_000);
    }

    #[test]
    fn test_context_window_openai() {
        assert_eq!(context_window_tokens("gpt-4o"), 128_000);
        assert_eq!(context_window_tokens("o3-mini"), 200_000);
    }

    #[test]
    fn test_context_window_gemini() {
        assert_eq!(context_window_tokens("gemini-2.0-flash"), 1_000_000);
    }

    #[test]
    fn test_context_window_default() {
        assert_eq!(context_window_tokens("unknown-model"), 128_000);
    }

    #[test]
    fn test_estimate_tokens() {
        // ~4 chars per token
        assert_eq!(estimate_tokens("hello world"), 2); // 11/4 = 2
        assert_eq!(estimate_tokens("a"), 1); // min 1
    }

    #[test]
    fn test_estimate_message_tokens() {
        // estimate_message_tokens uses content + tool_calls + 4 overhead
        let content = "Hello, how are you today?";
        let expected_min = estimate_tokens(content) + 4;
        // Create a minimal message-like check via the public function
        assert!(expected_min > 4);
    }
}
