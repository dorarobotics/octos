//! MCP (Model Context Protocol) client for external tool integration.
//!
//! Spawns MCP servers via stdio, discovers tools via JSON-RPC,
//! and wraps them as `Tool` implementations for the agent.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::tools::{Tool, ToolRegistry, ToolResult};

/// Maximum size for a single JSON-RPC response line (1MB).
const MAX_LINE_BYTES: usize = 1_048_576;

use crate::sandbox::BLOCKED_ENV_VARS;

/// Configuration for a single MCP server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// A running MCP server connection.
struct McpConnection {
    stdin: tokio::process::ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
    child: tokio::process::Child,
    next_id: u64,
}

impl McpConnection {
    async fn rpc_call(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let id = self.next_id;
        self.next_id += 1;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };

        let mut line = serde_json::to_string(&request)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;

        let buf = read_line_limited(&mut self.reader, MAX_LINE_BYTES).await?;

        let response: JsonRpcResponse =
            serde_json::from_str(&buf).wrap_err("invalid JSON-RPC response from MCP server")?;

        if let Some(err) = response.error {
            eyre::bail!("MCP error {}: {}", err.code, err.message);
        }

        response
            .result
            .ok_or_else(|| eyre::eyre!("MCP response missing result"))
    }
}

/// Read a single line with a size limit to prevent memory exhaustion.
async fn read_line_limited(
    reader: &mut BufReader<tokio::process::ChildStdout>,
    limit: usize,
) -> Result<String> {
    let mut buf = Vec::with_capacity(4096);
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            eyre::bail!("MCP server closed connection");
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            buf.extend_from_slice(&available[..=pos]);
            reader.consume(pos + 1);
            break;
        }
        // Check BEFORE extending to enforce strict limit
        if buf.len() + available.len() > limit {
            eyre::bail!("MCP response exceeds {}KB limit", limit / 1024);
        }
        let len = available.len();
        buf.extend_from_slice(available);
        reader.consume(len);
    }
    String::from_utf8(buf).wrap_err("MCP response is not valid UTF-8")
}

impl Drop for McpConnection {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        // Reap child to avoid zombie processes
        let _ = self.child.try_wait();
    }
}

/// MCP client that manages server connections and tool registration.
pub struct McpClient {
    /// Kept alive so Drop kills child processes.
    #[allow(dead_code)]
    connections: Vec<(String, Arc<Mutex<McpConnection>>)>,
    tools: Vec<McpToolSpec>,
}

struct McpToolSpec {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    connection: Arc<Mutex<McpConnection>>,
}

impl McpClient {
    /// Start all configured MCP servers and discover their tools.
    pub async fn start(configs: &[McpServerConfig]) -> Result<Self> {
        let mut connections = Vec::new();
        let mut tools = Vec::new();

        for config in configs {
            match Self::start_server(config).await {
                Ok((conn, server_tools)) => {
                    let server_name = &config.command;
                    let conn = Arc::new(Mutex::new(conn));
                    info!(
                        server = server_name,
                        tools = server_tools.len(),
                        "MCP server started"
                    );
                    for tool in server_tools {
                        tools.push(McpToolSpec {
                            name: tool.name,
                            description: tool.description.unwrap_or_default(),
                            input_schema: tool.input_schema.unwrap_or(serde_json::json!({"type": "object"})),
                            connection: conn.clone(),
                        });
                    }
                    connections.push((server_name.clone(), conn));
                }
                Err(e) => {
                    warn!(server = config.command, error = %e, "failed to start MCP server, skipping");
                }
            }
        }

        Ok(Self { connections, tools })
    }

    async fn start_server(config: &McpServerConfig) -> Result<(McpConnection, Vec<McpToolDef>)> {
        let mut cmd = tokio::process::Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // Forward stderr for debugging

        for (k, v) in &config.env {
            if BLOCKED_ENV_VARS.iter().any(|blocked| k.eq_ignore_ascii_case(blocked)) {
                warn!(key = k, "blocked dangerous MCP environment variable, skipping");
                continue;
            }
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().wrap_err_with(|| {
            format!("failed to spawn MCP server: {}", config.command)
        })?;

        let stdin = child.stdin.take().ok_or_else(|| eyre::eyre!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| eyre::eyre!("no stdout"))?;
        let reader = BufReader::new(stdout);

        let mut conn = McpConnection {
            stdin,
            reader,
            child,
            next_id: 1,
        };

        // Initialize
        let _init = conn
            .rpc_call(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "crew-rs", "version": env!("CARGO_PKG_VERSION")}
                }),
            )
            .await
            .wrap_err("MCP initialize failed")?;

        // Discover tools
        let tools_result = conn
            .rpc_call("tools/list", serde_json::json!({}))
            .await
            .wrap_err("MCP tools/list failed")?;

        let tool_list: McpToolListResponse = serde_json::from_value(tools_result)
            .wrap_err("failed to parse MCP tools/list response")?;

        Ok((conn, tool_list.tools))
    }

    /// Register all discovered MCP tools into the given registry.
    pub fn register_tools(self, registry: &mut ToolRegistry) {
        for spec in self.tools {
            registry.register(McpTool {
                name: spec.name,
                description: spec.description,
                input_schema: spec.input_schema,
                connection: spec.connection,
            });
        }
    }
}

/// A tool backed by an MCP server.
struct McpTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    connection: Arc<Mutex<McpConnection>>,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        self.input_schema.clone()
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            let mut conn = self.connection.lock().await;
            conn.rpc_call(
                "tools/call",
                serde_json::json!({
                    "name": self.name,
                    "arguments": args,
                }),
            )
            .await
        })
        .await
        .wrap_err("MCP tool call timed out after 30s")?
        .wrap_err_with(|| format!("MCP tool '{}' call failed", self.name))?;

        // Parse MCP tool result: { "content": [{"type": "text", "text": "..."}] }
        let output = if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            content
                .iter()
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            result.to_string()
        };

        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(ToolResult {
            output,
            success: !is_error,
            ..Default::default()
        })
    }
}

// --- JSON-RPC types ---

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: serde_json::Value,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: u64,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Deserialize)]
struct McpToolListResponse {
    tools: Vec<McpToolDef>,
}

#[derive(Deserialize)]
struct McpToolDef {
    name: String,
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Option<serde_json::Value>,
}
