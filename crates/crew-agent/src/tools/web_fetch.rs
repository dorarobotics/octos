//! Web fetch tool for retrieving URL content.

use std::time::Duration;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use reqwest::Client;
use serde::Deserialize;

use super::{Tool, ToolResult};

pub struct WebFetchTool {
    client: Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("crew-rs/0.1 (web-fetch-tool)")
                .build()
                .expect("failed to build HTTP client"),
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct Input {
    url: String,
    #[serde(default = "default_extract_mode")]
    extract_mode: String,
    #[serde(default = "default_max_chars")]
    max_chars: usize,
}

fn default_extract_mode() -> String {
    "markdown".to_string()
}

fn default_max_chars() -> usize {
    50_000
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL and extract its content as markdown or plain text."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "extract_mode": {
                    "type": "string",
                    "enum": ["markdown", "text"],
                    "description": "Output format: 'markdown' (default) or 'text'"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 50000)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid web_fetch input")?;

        if !input.url.starts_with("http://") && !input.url.starts_with("https://") {
            return Ok(ToolResult {
                output: "URL must start with http:// or https://".to_string(),
                success: false,
                ..Default::default()
            });
        }

        let response = match self.client.get(&input.url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    output: format!("Failed to fetch URL: {e}"),
                    success: false,
                    ..Default::default()
                });
            }
        };

        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let final_url = response.url().to_string();

        if !status.is_success() {
            return Ok(ToolResult {
                output: format!("HTTP {status} for {}", input.url),
                success: false,
                ..Default::default()
            });
        }

        let body = response
            .text()
            .await
            .wrap_err("failed to read response body")?;

        let is_html = content_type.contains("text/html");
        let mut content = if is_html {
            match input.extract_mode.as_str() {
                "text" => extract_text(&body),
                _ => extract_markdown(&body),
            }
        } else {
            body
        };

        if content.len() > input.max_chars {
            content.truncate(input.max_chars);
            content.push_str("\n\n... (content truncated)");
        }

        let mut output = format!("URL: {final_url}\n");
        if final_url != input.url {
            output.push_str(&format!("Redirected from: {}\n", input.url));
        }
        output.push_str(&format!("Content-Type: {content_type}\n"));
        output.push_str(&format!("Length: {} chars\n\n", content.len()));
        output.push_str(&content);

        Ok(ToolResult {
            output,
            success: true,
            ..Default::default()
        })
    }
}

fn extract_markdown(html: &str) -> String {
    htmd::convert(html).unwrap_or_else(|_| extract_text(html))
}

fn extract_text(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;

    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            continue;
        }
        if c == '>' {
            in_tag = false;
            result.push(' ');
            continue;
        }
        if !in_tag {
            result.push(c);
        }
    }

    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text() {
        let html = "<h1>Hello</h1><p>World <b>bold</b></p>";
        let text = extract_text(html);
        assert_eq!(text, "Hello World bold");
    }

    #[test]
    fn test_extract_text_with_whitespace() {
        let html = "<div>\n  <p>  spaced  </p>\n</div>";
        let text = extract_text(html);
        assert_eq!(text, "spaced");
    }

    #[tokio::test]
    async fn test_invalid_url_scheme() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(&serde_json::json!({"url": "ftp://example.com"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("http://"));
    }

    #[tokio::test]
    async fn test_invalid_input() {
        let tool = WebFetchTool::new();
        let result = tool.execute(&serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_markdown() {
        let html = "<h1>Title</h1><p>Paragraph</p>";
        let md = extract_markdown(html);
        assert!(md.contains("Title"));
        assert!(md.contains("Paragraph"));
    }
}
