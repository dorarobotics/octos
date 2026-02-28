//! Send email tool for agent-initiated email delivery.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use serde::Deserialize;

use super::{Tool, ToolResult};

/// Email payload passed to senders.
pub struct EmailPayload {
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    pub html: Option<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
}

/// Trait for email sending backends.
#[async_trait]
pub trait EmailSender: Send + Sync {
    async fn send(&self, email: &EmailPayload) -> Result<()>;
}

// ---------------------------------------------------------------------------
// SMTP sender (lettre)
// ---------------------------------------------------------------------------

/// Sends email via SMTP using lettre (Gmail, etc.).
pub struct SmtpEmailSender {
    host: String,
    port: u16,
    username: String,
    password_env: String,
    from_address: String,
}

impl SmtpEmailSender {
    pub fn new(
        host: String,
        port: u16,
        username: String,
        password_env: String,
        from_address: String,
    ) -> Self {
        Self {
            host,
            port,
            username,
            password_env,
            from_address,
        }
    }
}

#[async_trait]
impl EmailSender for SmtpEmailSender {
    async fn send(&self, email: &EmailPayload) -> Result<()> {
        use lettre::message::header::ContentType;
        use lettre::message::{Mailbox, MultiPart, SinglePart};
        use lettre::transport::smtp::authentication::Credentials;
        use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

        let password = std::env::var(&self.password_env)
            .map_err(|_| eyre::eyre!("SMTP password env var '{}' not set", self.password_env))?;

        let from: Mailbox = self
            .from_address
            .parse()
            .map_err(|e| eyre::eyre!("invalid from address: {e}"))?;

        let mut builder = Message::builder().from(from);

        for addr in &email.to {
            builder = builder.to(addr
                .parse()
                .map_err(|e| eyre::eyre!("invalid to address '{addr}': {e}"))?);
        }
        for addr in &email.cc {
            builder = builder.cc(addr
                .parse()
                .map_err(|e| eyre::eyre!("invalid cc address '{addr}': {e}"))?);
        }
        for addr in &email.bcc {
            builder = builder.bcc(
                addr.parse()
                    .map_err(|e| eyre::eyre!("invalid bcc address '{addr}': {e}"))?,
            );
        }

        builder = builder.subject(&email.subject);

        let message = if let Some(ref html) = email.html {
            builder
                .multipart(
                    MultiPart::alternative()
                        .singlepart(
                            SinglePart::builder()
                                .header(ContentType::TEXT_PLAIN)
                                .body(email.body.clone()),
                        )
                        .singlepart(
                            SinglePart::builder()
                                .header(ContentType::TEXT_HTML)
                                .body(html.clone()),
                        ),
                )
                .map_err(|e| eyre::eyre!("failed to build email: {e}"))?
        } else {
            builder
                .header(ContentType::TEXT_PLAIN)
                .body(email.body.clone())
                .map_err(|e| eyre::eyre!("failed to build email: {e}"))?
        };

        let creds = Credentials::new(self.username.clone(), password);

        let mailer = if self.port == 465 {
            AsyncSmtpTransport::<Tokio1Executor>::relay(&self.host)
                .map_err(|e| eyre::eyre!("SMTP relay error: {e}"))?
                .credentials(creds)
                .port(self.port)
                .build()
        } else {
            AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&self.host)
                .map_err(|e| eyre::eyre!("SMTP STARTTLS error: {e}"))?
                .credentials(creds)
                .port(self.port)
                .build()
        };

        mailer
            .send(message)
            .await
            .map_err(|e| eyre::eyre!("failed to send email: {e}"))?;

        tracing::info!(to = ?email.to, subject = %email.subject, "email sent via SMTP");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Feishu/Lark Mail sender
// ---------------------------------------------------------------------------

const FEISHU_TOKEN_TTL_SECS: u64 = 7000;

/// Sends email via Feishu/Lark Mail API.
pub struct FeishuEmailSender {
    app_id: String,
    app_secret_env: String,
    from_address: String,
    base_url: String,
    http: reqwest::Client,
    token_cache: Arc<tokio::sync::Mutex<Option<(String, Instant)>>>,
}

impl FeishuEmailSender {
    pub fn new(app_id: String, app_secret_env: String, from_address: String, region: &str) -> Self {
        let base_url = match region {
            "global" | "lark" => "https://open.larksuite.com/open-apis".to_string(),
            _ => "https://open.feishu.cn/open-apis".to_string(),
        };
        Self {
            app_id,
            app_secret_env,
            from_address,
            base_url,
            http: reqwest::Client::new(),
            token_cache: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    async fn get_token(&self) -> Result<String> {
        let mut cache = self.token_cache.lock().await;
        if let Some((ref token, ref created)) = *cache {
            if created.elapsed().as_secs() < FEISHU_TOKEN_TTL_SECS {
                return Ok(token.clone());
            }
        }

        let app_secret = std::env::var(&self.app_secret_env).map_err(|_| {
            eyre::eyre!(
                "Feishu app secret env var '{}' not set",
                self.app_secret_env
            )
        })?;

        let resp: serde_json::Value = self
            .http
            .post(format!(
                "{}/auth/v3/tenant_access_token/internal",
                self.base_url
            ))
            .json(&serde_json::json!({
                "app_id": self.app_id,
                "app_secret": app_secret,
            }))
            .send()
            .await
            .wrap_err("failed to get Feishu tenant token")?
            .json()
            .await?;

        let token = resp
            .get("tenant_access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                let msg = resp
                    .get("msg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                eyre::eyre!("Feishu token error: {msg}")
            })?
            .to_string();

        *cache = Some((token.clone(), Instant::now()));
        Ok(token)
    }
}

#[async_trait]
impl EmailSender for FeishuEmailSender {
    async fn send(&self, email: &EmailPayload) -> Result<()> {
        let token = self.get_token().await?;

        let to: Vec<serde_json::Value> = email
            .to
            .iter()
            .map(|a| serde_json::json!({"mail_address": a}))
            .collect();
        let cc: Vec<serde_json::Value> = email
            .cc
            .iter()
            .map(|a| serde_json::json!({"mail_address": a}))
            .collect();
        let bcc: Vec<serde_json::Value> = email
            .bcc
            .iter()
            .map(|a| serde_json::json!({"mail_address": a}))
            .collect();

        // Use HTML if provided, otherwise wrap plain text in <p> tags
        let content = email
            .html
            .clone()
            .unwrap_or_else(|| format!("<p>{}</p>", html_escape(&email.body)));

        let body = serde_json::json!({
            "subject": email.subject,
            "to": to,
            "cc": cc,
            "bcc": bcc,
            "body": {
                "content": content
            },
            "head_from": {
                "mail_address": self.from_address
            }
        });

        let url = format!(
            "{}/mail/v1/user_mailboxes/{}/messages/send",
            self.base_url, self.from_address
        );

        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .json(&body)
            .send()
            .await
            .wrap_err("Feishu mail API request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            eyre::bail!("Feishu mail API error (HTTP {status}): {text}");
        }

        let result: serde_json::Value = resp.json().await.unwrap_or_default();
        let code = result.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            let msg = result
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            eyre::bail!("Feishu mail API error (code {code}): {msg}");
        }

        tracing::info!(to = ?email.to, subject = %email.subject, "email sent via Feishu");
        Ok(())
    }
}

/// Minimal HTML escaping for plain text in Feishu mail body.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Tool that sends emails via configured provider (SMTP or Feishu).
pub struct SendEmailTool {
    sender: Arc<dyn EmailSender>,
}

impl SendEmailTool {
    pub fn new(sender: Arc<dyn EmailSender>) -> Self {
        Self { sender }
    }
}

#[derive(Deserialize)]
struct Input {
    to: StringOrVec,
    subject: String,
    body: String,
    #[serde(default)]
    html: Option<String>,
    #[serde(default)]
    cc: Option<StringOrVec>,
    #[serde(default)]
    bcc: Option<StringOrVec>,
}

/// Accepts either a single string or an array of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    Single(String),
    Multiple(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::Single(s) => vec![s],
            StringOrVec::Multiple(v) => v,
        }
    }
}

#[async_trait]
impl Tool for SendEmailTool {
    fn name(&self) -> &str {
        "send_email"
    }

    fn description(&self) -> &str {
        "Send an email to one or more recipients. Supports plain text and optional HTML body."
    }

    fn tags(&self) -> &[&str] {
        &["gateway", "communication"]
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to": {
                    "description": "Recipient email address(es). A single string or an array of strings.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Plain text email body"
                },
                "html": {
                    "type": "string",
                    "description": "Optional HTML email body (sent alongside plain text as multipart/alternative)"
                },
                "cc": {
                    "description": "CC recipient(s). A single string or an array of strings.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "bcc": {
                    "description": "BCC recipient(s). A single string or an array of strings.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid send_email input")?;

        let to = input.to.into_vec();
        if to.is_empty() {
            return Ok(ToolResult {
                output: "Error: 'to' must contain at least one recipient.".into(),
                success: false,
                ..Default::default()
            });
        }

        let payload = EmailPayload {
            to: to.clone(),
            subject: input.subject.clone(),
            body: input.body,
            html: input.html,
            cc: input.cc.map(|v| v.into_vec()).unwrap_or_default(),
            bcc: input.bcc.map(|v| v.into_vec()).unwrap_or_default(),
        };

        match self.sender.send(&payload).await {
            Ok(()) => {
                let recipients = to.join(", ");
                Ok(ToolResult {
                    output: format!(
                        "Email sent successfully to {recipients} with subject \"{}\"",
                        input.subject
                    ),
                    success: true,
                    ..Default::default()
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to send email: {e}"),
                success: false,
                ..Default::default()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSender {
        should_fail: bool,
    }

    #[async_trait]
    impl EmailSender for MockSender {
        async fn send(&self, _email: &EmailPayload) -> Result<()> {
            if self.should_fail {
                eyre::bail!("mock SMTP error");
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_send_email_success() {
        let tool = SendEmailTool::new(Arc::new(MockSender { should_fail: false }));
        let result = tool
            .execute(&serde_json::json!({
                "to": "test@example.com",
                "subject": "Hello",
                "body": "Test body"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("test@example.com"));
    }

    #[tokio::test]
    async fn test_send_email_multiple_recipients() {
        let tool = SendEmailTool::new(Arc::new(MockSender { should_fail: false }));
        let result = tool
            .execute(&serde_json::json!({
                "to": ["a@b.com", "c@d.com"],
                "subject": "Hello",
                "body": "Test body"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("a@b.com"));
        assert!(result.output.contains("c@d.com"));
    }

    #[tokio::test]
    async fn test_send_email_failure() {
        let tool = SendEmailTool::new(Arc::new(MockSender { should_fail: true }));
        let result = tool
            .execute(&serde_json::json!({
                "to": "test@example.com",
                "subject": "Hello",
                "body": "Test body"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("mock SMTP error"));
    }

    #[tokio::test]
    async fn test_send_email_empty_to() {
        let tool = SendEmailTool::new(Arc::new(MockSender { should_fail: false }));
        let result = tool
            .execute(&serde_json::json!({
                "to": [],
                "subject": "Hello",
                "body": "Test body"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.output.contains("at least one recipient"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(
            html_escape("<b>foo</b> & bar"),
            "&lt;b&gt;foo&lt;/b&gt; &amp; bar"
        );
    }
}
