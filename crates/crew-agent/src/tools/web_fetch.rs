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

        // Block requests to private/internal hosts (SSRF protection)
        if let Ok(url) = reqwest::Url::parse(&input.url) {
            if let Some(host) = url.host_str() {
                if is_private_host(host) {
                    return Ok(ToolResult {
                        output: "Requests to private/internal hosts are not allowed".to_string(),
                        success: false,
                        ..Default::default()
                    });
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
            match input.extract_mode.as_str() {
                "text" => extract_text(&body),
                _ => extract_markdown(&body),
            }
        } else {
            body
        };

        if content.len() > input.max_chars {
            let mut limit = input.max_chars;
            while limit > 0 && !content.is_char_boundary(limit) {
                limit -= 1;
            }
            content.truncate(limit);
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

/// Check if a hostname resolves to a private/internal address.
fn is_private_host(host: &str) -> bool {
    let lower = host.to_ascii_lowercase();
    // Localhost variants
    if lower == "localhost" || lower == "localhost." {
        return true;
    }
    // Parse as IP and check private ranges
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback()           // 127.0.0.0/8
                    || v4.is_private()     // 10/8, 172.16/12, 192.168/16
                    || v4.is_link_local()  // 169.254/16 (AWS metadata)
                    || v4.is_unspecified() // 0.0.0.0
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()           // ::1
                    || v6.is_unspecified() // ::
                    || v6.is_multicast()   // ff00::/8
                    // ULA fc00::/7
                    || matches!(v6.segments()[0], 0xfc00..=0xfdff)
                    // Link-local fe80::/10
                    || (v6.segments()[0] & 0xffc0) == 0xfe80
                    // Site-local fec0::/10 (deprecated RFC 3879, still routable)
                    || (v6.segments()[0] & 0xffc0) == 0xfec0
                    // IPv4-mapped ::ffff:x.x.x.x
                    || v6.to_ipv4_mapped().is_some_and(|v4| {
                        v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
                    })
                    // IPv4-compatible ::x.x.x.x (deprecated RFC 4291)
                    || v6.to_ipv4().is_some_and(|v4| {
                        v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
                    })
            }
        };
    }
    false
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
    fn test_private_host_localhost() {
        assert!(is_private_host("localhost"));
        assert!(is_private_host("LOCALHOST"));
        assert!(is_private_host("localhost."));
    }

    #[test]
    fn test_private_host_ipv4() {
        assert!(is_private_host("127.0.0.1"));
        assert!(is_private_host("10.0.0.1"));
        assert!(is_private_host("172.16.0.1"));
        assert!(is_private_host("192.168.1.1"));
        assert!(is_private_host("169.254.169.254"));
        assert!(is_private_host("0.0.0.0"));
    }

    #[test]
    fn test_private_host_ipv6() {
        assert!(is_private_host("::1"));              // loopback
        assert!(is_private_host("::"));               // unspecified
        assert!(is_private_host("fc00::1"));           // ULA
        assert!(is_private_host("fd12:3456::1"));      // ULA
        assert!(is_private_host("fe80::1"));           // link-local
        assert!(is_private_host("::ffff:127.0.0.1")); // IPv4-mapped loopback
        assert!(is_private_host("::ffff:192.168.1.1"));// IPv4-mapped private
        assert!(is_private_host("ff02::1"));            // multicast
        assert!(is_private_host("fec0::1"));           // site-local (deprecated)
        assert!(is_private_host("::192.168.1.1"));     // IPv4-compatible (deprecated)
    }

    #[test]
    fn test_public_host_allowed() {
        assert!(!is_private_host("8.8.8.8"));
        assert!(!is_private_host("1.1.1.1"));
        assert!(!is_private_host("example.com"));
        assert!(!is_private_host("2001:4860:4860::8888"));
    }

    #[test]
    fn test_extract_markdown() {
        let html = "<h1>Title</h1><p>Paragraph</p>";
        let md = extract_markdown(html);
        assert!(md.contains("Title"));
        assert!(md.contains("Paragraph"));
    }
}
