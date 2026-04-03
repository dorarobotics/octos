//! Real-time agent loop extensions for robotic operation.
//!
//! Provides timing guarantees, heartbeat monitoring, and sensor context
//! injection for safe 24/7 robotic agent operation.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Configuration for real-time agent loop behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Maximum time per agent loop iteration (LLM call + tool execution).
    /// If exceeded, the loop logs a warning and continues.
    #[serde(default = "default_iteration_deadline_ms")]
    pub iteration_deadline_ms: u64,

    /// Heartbeat interval. If no `beat()` within this period, the agent
    /// is considered stalled and should enter safe-hold.
    #[serde(default = "default_heartbeat_timeout_ms")]
    pub heartbeat_timeout_ms: u64,

    /// LLM call timeout. Aborts the LLM request if it exceeds this.
    #[serde(default = "default_llm_timeout_ms")]
    pub llm_timeout_ms: u64,

    /// Minimum cycle time. The loop sleeps to fill remaining time,
    /// preventing busy-spinning on fast iterations.
    #[serde(default = "default_min_cycle_ms")]
    pub min_cycle_ms: u64,

    /// Whether to check e-stop state before each iteration.
    #[serde(default = "default_true")]
    pub check_estop: bool,
}

fn default_iteration_deadline_ms() -> u64 {
    5000
}
fn default_heartbeat_timeout_ms() -> u64 {
    10000
}
fn default_llm_timeout_ms() -> u64 {
    8000
}
fn default_min_cycle_ms() -> u64 {
    100
}
fn default_true() -> bool {
    true
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            iteration_deadline_ms: default_iteration_deadline_ms(),
            heartbeat_timeout_ms: default_heartbeat_timeout_ms(),
            llm_timeout_ms: default_llm_timeout_ms(),
            min_cycle_ms: default_min_cycle_ms(),
            check_estop: true,
        }
    }
}

/// Atomic heartbeat counter for monitoring agent liveness.
///
/// The agent loop calls `beat()` each iteration. External monitors
/// read `state()` to detect stalls.
pub struct Heartbeat {
    counter: AtomicU32,
    last_check_value: AtomicU32,
    timeout: Duration,
    last_beat: std::sync::Mutex<Instant>,
}

/// Heartbeat health states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatState {
    /// Agent is actively beating.
    Alive,
    /// No beat received within the timeout period.
    Stalled,
}

impl Heartbeat {
    pub fn new(timeout: Duration) -> Self {
        Self {
            counter: AtomicU32::new(0),
            last_check_value: AtomicU32::new(0),
            timeout,
            last_beat: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Record a heartbeat (called each agent loop iteration).
    pub fn beat(&self) {
        self.counter.fetch_add(1, Ordering::Relaxed);
        *self.last_beat.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    /// Get the current beat count.
    pub fn count(&self) -> u32 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Check the heartbeat state. Returns `Stalled` if no beat since last check
    /// and the timeout has elapsed.
    pub fn state(&self) -> HeartbeatState {
        let current = self.counter.load(Ordering::Relaxed);
        let prev = self.last_check_value.swap(current, Ordering::Relaxed);

        if current != prev {
            return HeartbeatState::Alive;
        }

        let last = *self.last_beat.lock().unwrap_or_else(|e| e.into_inner());
        if last.elapsed() > self.timeout {
            HeartbeatState::Stalled
        } else {
            HeartbeatState::Alive
        }
    }
}

/// A timestamped snapshot of sensor data for LLM context injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSnapshot {
    /// Sensor identifier (e.g., "joint_positions", "force_torque").
    pub sensor_id: String,
    /// Sensor value as JSON.
    pub value: serde_json::Value,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl SensorSnapshot {
    /// Format as a compact text line for LLM context injection.
    pub fn to_context_line(&self) -> String {
        let age_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
            .saturating_sub(self.timestamp_ms);
        format!(
            "[{}] {} ({}ms ago)",
            self.sensor_id,
            self.value,
            age_ms
        )
    }
}

/// Ring buffer that accumulates sensor snapshots and formats them
/// for injection into the LLM system prompt.
pub struct SensorContextInjector {
    buffer: VecDeque<SensorSnapshot>,
    capacity: usize,
}

impl SensorContextInjector {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new sensor snapshot, evicting the oldest if at capacity.
    pub fn push(&mut self, snapshot: SensorSnapshot) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(snapshot);
    }

    /// Get the number of snapshots in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Format all snapshots as a text block for LLM context injection.
    pub fn to_context_block(&self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }
        let mut lines = vec!["## Live Sensor Data".to_string()];
        for snap in &self.buffer {
            lines.push(snap.to_context_line());
        }
        lines.join("\n")
    }

    /// Get the latest snapshot for a given sensor ID.
    pub fn latest(&self, sensor_id: &str) -> Option<&SensorSnapshot> {
        self.buffer.iter().rev().find(|s| s.sensor_id == sensor_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_track_heartbeat_alive() {
        let hb = Heartbeat::new(Duration::from_millis(100));
        hb.beat();
        assert_eq!(hb.state(), HeartbeatState::Alive);
        assert_eq!(hb.count(), 1);
    }

    #[test]
    fn should_detect_stalled_heartbeat() {
        let hb = Heartbeat::new(Duration::from_millis(1));
        // First check consumes the initial 0 value
        let _ = hb.state();
        // Sleep past timeout with no beats
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(hb.state(), HeartbeatState::Stalled);
    }

    #[test]
    fn should_use_ring_buffer_for_sensors() {
        let mut injector = SensorContextInjector::new(3);
        for i in 0..5 {
            injector.push(SensorSnapshot {
                sensor_id: format!("sensor_{i}"),
                value: serde_json::json!(i),
                timestamp_ms: 1000 + i as u64,
            });
        }
        assert_eq!(injector.len(), 3);
        // Oldest two evicted, should have sensor_2, sensor_3, sensor_4
        assert!(injector.latest("sensor_0").is_none());
        assert!(injector.latest("sensor_4").is_some());
    }

    #[test]
    fn should_format_sensor_context() {
        let mut injector = SensorContextInjector::new(10);
        injector.push(SensorSnapshot {
            sensor_id: "joint_positions".to_string(),
            value: serde_json::json!([0.0, 1.0, 2.0]),
            timestamp_ms: 999_999_000,
        });
        let block = injector.to_context_block();
        assert!(block.contains("## Live Sensor Data"));
        assert!(block.contains("joint_positions"));
    }

    #[test]
    fn should_use_default_config() {
        let config = RealtimeConfig::default();
        assert_eq!(config.iteration_deadline_ms, 5000);
        assert_eq!(config.heartbeat_timeout_ms, 10000);
        assert_eq!(config.llm_timeout_ms, 8000);
        assert_eq!(config.min_cycle_ms, 100);
        assert!(config.check_estop);
    }
}
