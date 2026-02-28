//! Gateway and bridge child process lifecycle management.
//!
//! Spawns `crew gateway` and optionally `node bridge.js` (WhatsApp) as child
//! processes, monitors their output, and provides start/stop/status/log-streaming
//! capabilities. Managed WhatsApp bridges are auto-spawned when a profile with
//! a WhatsApp channel (no explicit `bridge_url`) is started.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use crew_agent::sandbox::BLOCKED_ENV_VARS;
use eyre::{Result, bail};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock, broadcast, watch};

use crate::profiles::{ChannelCredentials, ProfileStore, UserProfile};

/// Base port for managed WhatsApp bridge WebSocket servers.
/// HTTP media port = WS port + 1.
const BRIDGE_BASE_WS_PORT: u16 = 3101;

/// Base port for auto-assigned Feishu/Twilio webhook servers.
const WEBHOOK_BASE_PORT: u16 = 9321;

/// Manages gateway and bridge child processes — one of each per user profile.
pub struct ProcessManager {
    processes: Arc<RwLock<HashMap<String, GatewayProcess>>>,
    bridges: Arc<RwLock<HashMap<String, BridgeProcess>>>,
    profile_store: Arc<ProfileStore>,
    /// Path to bridge.js. If None, managed bridges are disabled.
    bridge_js_path: Option<PathBuf>,
}

struct GatewayProcess {
    pid: u32,
    started_at: DateTime<Utc>,
    log_tx: broadcast::Sender<String>,
    stop_tx: watch::Sender<bool>,
    /// Feishu/Twilio webhook port this gateway is listening on (if any).
    webhook_port: Option<u16>,
}

#[allow(dead_code)]
struct BridgeProcess {
    pid: u32,
    ws_port: u16,
    http_port: u16,
    started_at: DateTime<Utc>,
    qr_code: Arc<Mutex<Option<String>>>,
    status: Arc<Mutex<BridgeStatus>>,
    phone_number: Arc<Mutex<Option<String>>>,
    lid: Arc<Mutex<Option<String>>>,
    log_tx: broadcast::Sender<String>,
    stop_tx: watch::Sender<bool>,
}

/// WhatsApp bridge connection status.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BridgeStatus {
    /// Bridge started, waiting for QR scan.
    Waiting,
    /// WhatsApp connected.
    Connected,
    /// Disconnected (may auto-reconnect).
    Disconnected,
    /// Logged out — needs re-pairing.
    LoggedOut,
}

/// Status of a gateway process.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub started_at: Option<String>,
    pub uptime_secs: Option<i64>,
}

/// WhatsApp bridge QR + status info returned to the dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct BridgeQrInfo {
    pub qr: Option<String>,
    pub status: BridgeStatus,
    pub ws_port: u16,
    pub http_port: u16,
    /// Phone number of the connected WhatsApp account (e.g. "14088882719").
    pub phone_number: Option<String>,
    /// WhatsApp LID of the connected account (e.g. "197061790171194").
    pub lid: Option<String>,
}

impl ProcessManager {
    /// Create a new process manager backed by the given profile store.
    pub fn new(profile_store: Arc<ProfileStore>) -> Self {
        Self {
            processes: Arc::new(RwLock::new(HashMap::new())),
            bridges: Arc::new(RwLock::new(HashMap::new())),
            profile_store,
            bridge_js_path: None,
        }
    }

    /// Set the path to bridge.js for managed WhatsApp bridges.
    pub fn with_bridge_js(mut self, path: PathBuf) -> Self {
        if path.exists() {
            self.bridge_js_path = Some(path);
        } else {
            tracing::warn!(path = %path.display(), "bridge.js not found, managed WhatsApp bridges disabled");
        }
        self
    }

    // ── Gateway lifecycle ──────────────────────────────────────────────

    /// Start the gateway for a profile. Returns an error if already running.
    /// If the profile has a managed WhatsApp channel, the bridge is started first.
    pub async fn start(&self, profile: &UserProfile) -> Result<()> {
        // Hold the write lock for the entire operation to prevent TOCTOU races.
        let mut procs = self.processes.write().await;
        if procs.contains_key(&profile.id) {
            bail!("gateway for '{}' is already running", profile.id);
        }

        // Check if profile needs a managed WhatsApp bridge
        let bridge_url_override = if self.needs_managed_bridge(profile) {
            match self.start_bridge_inner(profile).await {
                Ok(ws_port) => Some(format!("ws://localhost:{ws_port}")),
                Err(e) => {
                    tracing::warn!(
                        profile = %profile.id,
                        error = %e,
                        "failed to start managed WhatsApp bridge, continuing without it"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Auto-assign Feishu webhook port if needed
        let feishu_port = match crate::profiles::feishu_webhook_port(profile) {
            Some(Some(port)) => Some(port),
            Some(None) => Some(self.allocate_webhook_port(&procs)),
            None => None,
        };

        // Resolve data directory and ensure subdirs exist
        let data_dir = self.profile_store.resolve_data_dir(profile);
        for sub in ["memory", "sessions", "research", "skills", "history"] {
            std::fs::create_dir_all(data_dir.join(sub))?;
        }

        // Spawn the gateway as a child process, pointing at the profile JSON directly
        let exe = std::env::current_exe()?;
        let profile_path = self.profile_store.profile_path(&profile.id);
        let mut cmd = Command::new(&exe);
        cmd.arg("gateway")
            .arg("--profile")
            .arg(&profile_path)
            .arg("--data-dir")
            .arg(&data_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        if let Some(ref url) = bridge_url_override {
            cmd.arg("--bridge-url").arg(url);
        }
        if let Some(port) = feishu_port {
            cmd.arg("--feishu-port").arg(port.to_string());
        }

        // Pass env vars from profile config, filtering out dangerous ones.
        for (key, value) in &profile.config.env_vars {
            if BLOCKED_ENV_VARS
                .iter()
                .any(|blocked| key.eq_ignore_ascii_case(blocked))
            {
                tracing::warn!(
                    profile = %profile.id,
                    var = %key,
                    "skipping blocked environment variable"
                );
                continue;
            }
            cmd.env(key, value);
        }

        let mut child = cmd.spawn()?;

        let pid = child.id().unwrap_or(0);
        let (log_tx, _) = broadcast::channel::<String>(1024);
        let (stop_tx, stop_rx) = watch::channel(false);

        let process = GatewayProcess {
            pid,
            started_at: Utc::now(),
            log_tx: log_tx.clone(),
            stop_tx,
            webhook_port: feishu_port,
        };

        procs.insert(profile.id.clone(), process);
        // Release the write lock now that the entry is inserted.
        drop(procs);

        // Spawn task to read stdout and forward to log channel
        if let Some(stdout) = child.stdout.take() {
            let tx = log_tx.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(line);
                }
            });
        }

        // Spawn task to read stderr and forward to log channel
        if let Some(stderr) = child.stderr.take() {
            let tx = log_tx.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("[stderr] {line}"));
                }
            });
        }

        // Spawn task to wait for process exit or stop signal.
        let profile_id = profile.id.clone();
        let processes = Arc::clone(&self.processes);
        tokio::spawn(async move {
            let mut stop_rx = stop_rx;
            tokio::select! {
                status = child.wait() => {
                    processes.write().await.remove(&profile_id);
                    match status {
                        Ok(s) => tracing::info!(profile = %profile_id, exit = %s, "gateway exited"),
                        Err(e) => tracing::error!(profile = %profile_id, error = %e, "gateway error"),
                    }
                }
                _ = stop_rx.changed() => {
                    let _ = child.kill().await;
                    tracing::info!(profile = %profile_id, "gateway stopped");
                }
            }
        });

        tracing::info!(profile = %profile.id, pid = pid, "gateway started");
        Ok(())
    }

    /// Stop the gateway for a profile. Also stops managed bridge if running.
    pub async fn stop(&self, profile_id: &str) -> Result<bool> {
        let process = {
            let mut procs = self.processes.write().await;
            procs.remove(profile_id)
        };

        // Also stop the managed bridge
        self.stop_bridge(profile_id).await;

        match process {
            Some(proc) => {
                let _ = proc.stop_tx.send(true);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Restart a gateway (stop then start).
    pub async fn restart(&self, profile: &UserProfile) -> Result<()> {
        let _ = self.stop(&profile.id).await;
        // Small delay to let the process clean up
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        self.start(profile).await
    }

    /// Get the status of a gateway process.
    pub async fn status(&self, profile_id: &str) -> ProcessStatus {
        let procs = self.processes.read().await;
        match procs.get(profile_id) {
            Some(proc) => {
                let uptime = Utc::now() - proc.started_at;
                ProcessStatus {
                    running: true,
                    pid: Some(proc.pid),
                    started_at: Some(proc.started_at.to_rfc3339()),
                    uptime_secs: Some(uptime.num_seconds()),
                }
            }
            None => ProcessStatus {
                running: false,
                pid: None,
                started_at: None,
                uptime_secs: None,
            },
        }
    }

    /// Subscribe to log output for a profile. Returns None if not running.
    pub async fn subscribe_logs(&self, profile_id: &str) -> Option<broadcast::Receiver<String>> {
        let procs = self.processes.read().await;
        procs.get(profile_id).map(|p| p.log_tx.subscribe())
    }

    /// Get the status of all profiles.
    pub async fn all_statuses(&self) -> HashMap<String, ProcessStatus> {
        let procs = self.processes.read().await;
        let mut statuses = HashMap::new();
        for (id, proc) in procs.iter() {
            let uptime = Utc::now() - proc.started_at;
            statuses.insert(
                id.clone(),
                ProcessStatus {
                    running: true,
                    pid: Some(proc.pid),
                    started_at: Some(proc.started_at.to_rfc3339()),
                    uptime_secs: Some(uptime.num_seconds()),
                },
            );
        }
        statuses
    }

    /// Stop all running gateways (and their bridges).
    pub async fn stop_all(&self) -> usize {
        // Stop all bridges first
        let bridge_ids: Vec<String> = {
            let bridges = self.bridges.read().await;
            bridges.keys().cloned().collect()
        };
        for id in &bridge_ids {
            self.stop_bridge(id).await;
        }

        let processes: HashMap<String, GatewayProcess> = {
            let mut procs = self.processes.write().await;
            std::mem::take(&mut *procs)
        };
        let count = processes.len();
        for (id, proc) in processes {
            let _ = proc.stop_tx.send(true);
            tracing::info!(profile = %id, "gateway stopped");
        }
        count
    }

    /// Get the data directory path for a profile.
    pub fn resolve_data_dir(&self, profile: &UserProfile) -> PathBuf {
        self.profile_store.resolve_data_dir(profile)
    }

    /// Get the webhook port for a running gateway (if any).
    pub async fn webhook_port(&self, profile_id: &str) -> Option<u16> {
        let procs = self.processes.read().await;
        procs.get(profile_id).and_then(|p| p.webhook_port)
    }

    /// Allocate the next available webhook port.
    fn allocate_webhook_port(&self, procs: &HashMap<String, GatewayProcess>) -> u16 {
        let used: std::collections::HashSet<u16> =
            procs.values().filter_map(|p| p.webhook_port).collect();
        let mut port = WEBHOOK_BASE_PORT;
        while used.contains(&port) || !port_available(port) {
            port += 1;
        }
        port
    }

    // ── Bridge lifecycle ───────────────────────────────────────────────

    /// Check if a profile has a WhatsApp channel that needs a managed bridge.
    /// A managed bridge is needed when bridge_url is empty or "auto".
    fn needs_managed_bridge(&self, profile: &UserProfile) -> bool {
        if self.bridge_js_path.is_none() {
            return false;
        }
        profile.config.channels.iter().any(|ch| {
            matches!(ch,
                ChannelCredentials::WhatsApp { bridge_url }
                if bridge_url.is_empty() || bridge_url == "auto"
            )
        })
    }

    /// Internal: start a bridge process. Returns the WS port.
    async fn start_bridge_inner(&self, profile: &UserProfile) -> Result<u16> {
        let mut bridges = self.bridges.write().await;
        if bridges.contains_key(&profile.id) {
            let existing = bridges.get(&profile.id).unwrap();
            return Ok(existing.ws_port);
        }

        let bridge_js = self
            .bridge_js_path
            .as_ref()
            .ok_or_else(|| eyre::eyre!("bridge.js path not configured"))?;

        let (ws_port, http_port) = self.allocate_bridge_ports(&bridges);

        let data_dir = self.profile_store.resolve_data_dir(profile);
        let auth_dir = data_dir.join("whatsapp-auth");
        let media_dir = data_dir.join("whatsapp-media");
        std::fs::create_dir_all(&auth_dir)?;
        std::fs::create_dir_all(&media_dir)?;

        let node = find_node()?;
        let mut cmd = Command::new(&node);
        cmd.arg(bridge_js)
            .env("BRIDGE_PORT", ws_port.to_string())
            .env("MEDIA_PORT", http_port.to_string())
            .env("AUTH_DIR", &auth_dir)
            .env("MEDIA_DIR", &media_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn()?;
        let pid = child.id().unwrap_or(0);

        let (log_tx, _) = broadcast::channel::<String>(256);
        let (stop_tx, stop_rx) = watch::channel(false);
        let qr_code: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let status: Arc<Mutex<BridgeStatus>> = Arc::new(Mutex::new(BridgeStatus::Waiting));
        let phone_number: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let lid: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let bridge = BridgeProcess {
            pid,
            ws_port,
            http_port,
            started_at: Utc::now(),
            qr_code: Arc::clone(&qr_code),
            status: Arc::clone(&status),
            phone_number: Arc::clone(&phone_number),
            lid: Arc::clone(&lid),
            log_tx: log_tx.clone(),
            stop_tx,
        };

        bridges.insert(profile.id.clone(), bridge);
        drop(bridges);

        // Spawn task to read stdout — parse JSON events for QR/status
        if let Some(stdout) = child.stdout.take() {
            let tx = log_tx.clone();
            let qr = Arc::clone(&qr_code);
            let st = Arc::clone(&status);
            let ph = Arc::clone(&phone_number);
            let li = Arc::clone(&lid);
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // Try to parse structured JSON events from bridge
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        match json.get("type").and_then(|t| t.as_str()) {
                            Some("qr") => {
                                if let Some(qr_str) = json.get("qr").and_then(|q| q.as_str()) {
                                    *qr.lock().await = Some(qr_str.to_string());
                                    *st.lock().await = BridgeStatus::Waiting;
                                }
                            }
                            Some("status") => {
                                if let Some(s) = json.get("status").and_then(|s| s.as_str()) {
                                    match s {
                                        "connected" => {
                                            *qr.lock().await = None;
                                            *st.lock().await = BridgeStatus::Connected;
                                            if let Some(phone) =
                                                json.get("phone").and_then(|p| p.as_str())
                                            {
                                                *ph.lock().await = Some(phone.to_string());
                                            }
                                            if let Some(lid_val) =
                                                json.get("lid").and_then(|l| l.as_str())
                                            {
                                                *li.lock().await = Some(lid_val.to_string());
                                            }
                                        }
                                        "disconnected" => {
                                            *st.lock().await = BridgeStatus::Disconnected;
                                        }
                                        "logged_out" => {
                                            *qr.lock().await = None;
                                            *st.lock().await = BridgeStatus::LoggedOut;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    let _ = tx.send(line);
                }
            });
        }

        // Spawn task to read stderr
        if let Some(stderr) = child.stderr.take() {
            let tx = log_tx.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(format!("[stderr] {line}"));
                }
            });
        }

        // Spawn task to wait for exit or stop
        let profile_id = profile.id.clone();
        let bridges_ref = Arc::clone(&self.bridges);
        tokio::spawn(async move {
            let mut stop_rx = stop_rx;
            tokio::select! {
                exit_status = child.wait() => {
                    bridges_ref.write().await.remove(&profile_id);
                    match exit_status {
                        Ok(s) => tracing::info!(profile = %profile_id, exit = %s, "bridge exited"),
                        Err(e) => tracing::error!(profile = %profile_id, error = %e, "bridge error"),
                    }
                }
                _ = stop_rx.changed() => {
                    let _ = child.kill().await;
                    tracing::info!(profile = %profile_id, "bridge stopped");
                }
            }
        });

        tracing::info!(
            profile = %profile.id,
            pid = pid,
            ws_port = ws_port,
            http_port = http_port,
            "managed WhatsApp bridge started"
        );

        // Small delay to let bridge start listening before gateway connects
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        Ok(ws_port)
    }

    /// Stop a managed bridge for a profile.
    async fn stop_bridge(&self, profile_id: &str) {
        let bridge = {
            let mut bridges = self.bridges.write().await;
            bridges.remove(profile_id)
        };
        if let Some(b) = bridge {
            let _ = b.stop_tx.send(true);
            tracing::info!(profile = %profile_id, "bridge stopped");
        }
    }

    /// Get the WhatsApp QR code and status for a profile.
    pub async fn bridge_qr(&self, profile_id: &str) -> Option<BridgeQrInfo> {
        let bridges = self.bridges.read().await;
        let bridge = bridges.get(profile_id)?;
        let qr = bridge.qr_code.lock().await.clone();
        let status = *bridge.status.lock().await;
        let phone_number = bridge.phone_number.lock().await.clone();
        let lid = bridge.lid.lock().await.clone();
        Some(BridgeQrInfo {
            qr,
            status,
            ws_port: bridge.ws_port,
            http_port: bridge.http_port,
            phone_number,
            lid,
        })
    }

    /// Allocate the next available port pair for a bridge.
    /// Checks both the in-memory bridge map and actual port availability.
    fn allocate_bridge_ports(&self, bridges: &HashMap<String, BridgeProcess>) -> (u16, u16) {
        let used_ports: std::collections::HashSet<u16> =
            bridges.values().map(|b| b.ws_port).collect();
        let mut ws_port = BRIDGE_BASE_WS_PORT;
        loop {
            if !used_ports.contains(&ws_port)
                && port_available(ws_port)
                && port_available(ws_port + 1)
            {
                return (ws_port, ws_port + 1);
            }
            ws_port += 2;
        }
    }
}

/// Check if a TCP port is available by attempting to bind it.
fn port_available(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// Find the `node` binary on PATH.
fn find_node() -> Result<PathBuf> {
    let candidates = ["node", "/opt/homebrew/bin/node", "/usr/local/bin/node"];
    for name in candidates {
        let p = Path::new(name);
        if p.is_absolute() && p.exists() {
            return Ok(p.to_path_buf());
        }
        // Try which
        if let Ok(output) = std::process::Command::new("which").arg(name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }
    }
    bail!("node not found — install Node.js to use managed WhatsApp bridges")
}
