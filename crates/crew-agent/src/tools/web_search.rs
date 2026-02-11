//! Web search tool using Brave Search API.

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use reqwest::Client;
use serde::Deserialize;

use super::{Tool, ToolResult};

pub struct WebSearchTool {
    client: Client,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct Input {
    query: String,
    #[serde(default = "default_count")]
    count: u8,
}

fn default_count() -> u8 {
    5
}

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Deserialize)]
struct BraveWebResults {
    results: Vec<BraveWebResult>,
}

#[derive(Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    description: String,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web using Brave Search API. Requires BRAVE_API_KEY environment variable."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results (1-10, default: 5)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid web_search input")?;

        let api_key = match std::env::var("BRAVE_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                return Ok(ToolResult {
                    output: "BRAVE_API_KEY environment variable not set.".to_string(),
                    success: false,
                    ..Default::default()
                });
            }
        };

        let count = input.count.clamp(1, 10);

        let response = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("X-Subscription-Token", &api_key)
            .header("Accept", "application/json")
            .query(&[("q", &input.query), ("count", &count.to_string())])
            .send()
            .await
            .wrap_err("failed to call Brave Search API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Ok(ToolResult {
                output: format!("Brave Search API error ({status}): {body}"),
                success: false,
                ..Default::default()
            });
        }

        let brave: BraveResponse = response
            .json()
            .await
            .wrap_err("failed to parse Brave Search response")?;

        let results = brave.web.map(|w| w.results).unwrap_or_default();

        if results.is_empty() {
            return Ok(ToolResult {
                output: format!("No results found for: {}", input.query),
                success: true,
                ..Default::default()
            });
        }

        let mut output = format!("Results for: {}\n\n", input.query);
        for (i, r) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                r.title,
                r.url,
                r.description
            ));
        }

        Ok(ToolResult {
            output,
            success: true,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_missing_api_key() {
        // Ensure BRAVE_API_KEY is not set for this test
        let was_set = std::env::var("BRAVE_API_KEY").ok();
        if was_set.is_some() {
            unsafe { std::env::remove_var("BRAVE_API_KEY") };
        }

        let tool = WebSearchTool::new();
        let result = tool
            .execute(&serde_json::json!({"query": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("BRAVE_API_KEY"));

        // Restore if it was set
        if let Some(key) = was_set {
            unsafe { std::env::set_var("BRAVE_API_KEY", key) };
        }
    }

    #[tokio::test]
    async fn test_invalid_input() {
        let tool = WebSearchTool::new();
        let result = tool.execute(&serde_json::json!({})).await;
        assert!(result.is_err());
    }
}
