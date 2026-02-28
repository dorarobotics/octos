//! Web fetch tool for retrieving URL content.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use reqwest::Client;
use serde::Deserialize;

use super::{Tool, ToolResult};

pub struct WebFetchTool {
    client: Client,
    config: Option<Arc<super::tool_config::ToolConfigStore>>,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("crew-rs/0.1 (web-fetch-tool)")
                .build()
                .expect("failed to build HTTP client"),
            config: None,
        }
    }

    pub fn with_config(mut self, config: Arc<super::tool_config::ToolConfigStore>) -> Self {
        self.config = Some(config);
        self
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
    #[serde(default)]
    extract_mode: Option<String>,
    #[serde(default)]
    max_chars: Option<usize>,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL and extract its content as markdown or plain text."
    }

    fn tags(&self) -> &[&str] {
        &["web"]
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

        let (cfg_extract_mode, cfg_max_chars) = match &self.config {
            Some(c) => (
                c.get_str("web_fetch", "extract_mode").await,
                c.get_usize("web_fetch", "max_chars").await,
            ),
            None => (None, None),
        };
        let extract_mode = input
            .extract_mode
            .or(cfg_extract_mode)
            .unwrap_or_else(|| "markdown".to_string());
        let max_chars = input.max_chars.or(cfg_max_chars).unwrap_or(50_000);

        if !input.url.starts_with("http://") && !input.url.starts_with("https://") {
            return Ok(ToolResult {
                output: "URL must start with http:// or https://".to_string(),
                success: false,
                ..Default::default()
            });
        }

        // Block requests to private/internal hosts (SSRF protection)
        if let Ok(url) = reqwest::Url::parse(&input.url) {
            if let Some(host) = url.host_str() {
                // Check hostname string first (catches literal IPs and "localhost")
                if super::ssrf::is_private_host(host) {
                    return Ok(ToolResult {
                        output: "Requests to private/internal hosts are not allowed".to_string(),
                        success: false,
                        ..Default::default()
                    });
                }

                // Resolve DNS and check resolved IPs (prevents DNS rebinding)
                let port = url.port_or_known_default().unwrap_or(443);
                if let Ok(addrs) = tokio::net::lookup_host(format!("{host}:{port}")).await {
                    for addr in addrs {
                        if super::ssrf::is_private_ip(&addr.ip()) {
                            return Ok(ToolResult {
                                output: "Requests to private/internal hosts are not allowed (DNS resolved to private IP)".to_string(),
                                success: false,
                                ..Default::default()
                            });
                        }
                    }
                }
            }
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
            match extract_mode.as_str() {
                "text" => extract_text(&body),
                _ => extract_markdown(&body),
            }
        } else {
            body
        };

        crew_core::truncate_utf8(&mut content, max_chars, "\n\n... (content truncated)");

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

    #[tokio::test]
    async fn test_dns_rebind_localhost() {
        // "localhost" should be caught by hostname check before DNS
        let tool = WebFetchTool::new();
        let result = tool
            .execute(&serde_json::json!({"url": "http://localhost/test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("private"));
    }

    #[test]
    fn test_extract_markdown() {
        let html = "<h1>Title</h1><p>Paragraph</p>";
        let md = extract_markdown(html);
        assert!(md.contains("Title"));
        assert!(md.contains("Paragraph"));
    }
}
