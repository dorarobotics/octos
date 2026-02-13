//! Session management with JSONL persistence.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use crew_core::{Message, SessionKey};
use eyre::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Metadata stored as the first line of each JSONL session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionMeta {
    session_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_key: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

/// A conversation session with message history.
#[derive(Debug, Clone)]
pub struct Session {
    pub key: SessionKey,
    /// Parent session key if this session was forked.
    pub parent_key: Option<SessionKey>,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    fn new(key: SessionKey) -> Self {
        let now = Utc::now();
        Self {
            key,
            parent_key: None,
            messages: vec![],
            created_at: now,
            updated_at: now,
        }
    }

    /// Get the most recent N messages from history.
    pub fn get_history(&self, max: usize) -> &[Message] {
        let len = self.messages.len();
        if len <= max {
            &self.messages
        } else {
            &self.messages[len - max..]
        }
    }
}

/// Manages sessions with in-memory cache and JSONL disk persistence.
pub struct SessionManager {
    sessions_dir: PathBuf,
    cache: HashMap<String, Session>,
}

impl SessionManager {
    pub fn open(data_dir: &Path) -> Result<Self> {
        let sessions_dir = data_dir.join("sessions");
        std::fs::create_dir_all(&sessions_dir)?;
        Ok(Self {
            sessions_dir,
            cache: HashMap::new(),
        })
    }

    /// List all sessions (ID + message count) from disk.
    pub fn list_sessions(&self) -> Vec<(String, usize)> {
        let mut result = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.sessions_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                        let count = std::fs::read_to_string(&path)
                            .map(|c| c.lines().count())
                            .unwrap_or(0);
                        result.push((name.to_string(), count));
                    }
                }
            }
        }
        result
    }

    /// Get or create a session. Loads from disk on first access.
    pub fn get_or_create(&mut self, key: &SessionKey) -> &mut Session {
        let key_str = key.0.clone();
        if !self.cache.contains_key(&key_str) {
            let session = self
                .load_from_disk(key)
                .unwrap_or_else(|| Session::new(key.clone()));
            self.cache.insert(key_str.clone(), session);
        }
        self.cache.get_mut(&key_str).expect("session must exist: inserted above")
    }

    /// Add a message to a session and persist it.
    pub fn add_message(&mut self, key: &SessionKey, message: Message) -> Result<()> {
        let session = self.get_or_create(key);
        session.messages.push(message.clone());
        session.updated_at = Utc::now();
        self.append_to_disk(key, &message)
    }

    /// Get the JSONL file path for a session key.
    ///
    /// Uses percent-encoding for non-safe characters to ensure different keys
    /// always produce different filenames (no collisions).
    fn session_path(&self, key: &SessionKey) -> PathBuf {
        let safe_name: String = key
            .0
            .chars()
            .flat_map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    vec![c]
                } else {
                    // Percent-encode: ':' -> '%3A', '.' -> '%2E', etc.
                    format!("%{:02X}", c as u32).chars().collect()
                }
            })
            .collect();
        self.sessions_dir.join(format!("{safe_name}.jsonl"))
    }

    /// Load a session from its JSONL file.
    fn load_from_disk(&self, key: &SessionKey) -> Option<Session> {
        let path = self.session_path(key);
        let content = std::fs::read_to_string(&path).ok()?;
        let mut lines = content.lines();

        // First line is metadata
        let meta_line = lines.next()?;
        let meta: SessionMeta = serde_json::from_str(meta_line).ok()?;

        // Remaining lines are messages
        let messages: Vec<Message> = lines
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        debug!(key = %key, messages = messages.len(), "Loaded session from disk");

        Some(Session {
            key: key.clone(),
            parent_key: meta.parent_key.map(SessionKey),
            messages,
            created_at: meta.created_at,
            updated_at: meta.updated_at,
        })
    }

    /// Append a message to the JSONL file. Creates the file with metadata if new.
    fn append_to_disk(&self, key: &SessionKey, message: &Message) -> Result<()> {
        use std::io::Write;

        let path = self.session_path(key);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        // Check file size after open to avoid TOCTOU race with exists() check
        let is_new = file.metadata()?.len() == 0;
        if is_new {
            let parent_key = self.cache.get(&key.0).and_then(|s| s.parent_key.as_ref().map(|k| k.0.clone()));
            let meta = SessionMeta {
                session_key: key.0.clone(),
                parent_key,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            writeln!(file, "{}", serde_json::to_string(&meta)?)?;
        }

        writeln!(file, "{}", serde_json::to_string(message)?)?;
        Ok(())
    }

    /// Rewrite a session's JSONL file from the in-memory state.
    /// Uses atomic write-then-rename to avoid corruption on crash.
    pub fn rewrite(&self, key: &SessionKey) -> Result<()> {
        use std::io::Write;

        let session = self
            .cache
            .get(&key.0)
            .ok_or_else(|| eyre::eyre!("session not in cache: {}", key))?;

        let path = self.session_path(key);
        let tmp_path = path.with_extension("jsonl.tmp");

        let mut file = std::fs::File::create(&tmp_path)?;

        let meta = SessionMeta {
            session_key: key.0.clone(),
            parent_key: session.parent_key.as_ref().map(|k| k.0.clone()),
            created_at: session.created_at,
            updated_at: session.updated_at,
        };
        writeln!(file, "{}", serde_json::to_string(&meta)?)?;

        for msg in &session.messages {
            writeln!(file, "{}", serde_json::to_string(msg)?)?;
        }
        file.flush()?;

        // Atomic rename (on same filesystem)
        std::fs::rename(&tmp_path, &path)?;

        debug!(key = %key, messages = session.messages.len(), "Rewrote session to disk");
        Ok(())
    }

    /// Fork a session: create a new session that copies the last N messages from the parent.
    ///
    /// The new session's channel is taken from the parent key; `new_chat_id` becomes the chat ID.
    /// Returns the new session's key.
    pub fn fork(
        &mut self,
        parent_key: &SessionKey,
        new_chat_id: &str,
        copy_messages: usize,
    ) -> Result<SessionKey> {
        let parent = self.get_or_create(parent_key);
        let messages: Vec<Message> = parent.get_history(copy_messages).to_vec();
        // Derive channel from parent key (format: "channel:chat_id")
        let channel = parent_key
            .0
            .split(':')
            .next()
            .unwrap_or("cli");
        let new_key = SessionKey::new(channel, new_chat_id);

        let now = Utc::now();
        let session = Session {
            key: new_key.clone(),
            parent_key: Some(parent_key.clone()),
            messages,
            created_at: now,
            updated_at: now,
        };
        self.cache.insert(new_key.0.clone(), session);
        self.rewrite(&new_key)?;

        debug!(
            parent = %parent_key,
            child = %new_key,
            copied = copy_messages,
            "Forked session"
        );
        Ok(new_key)
    }

    /// Clear a session's history (both in-memory and on disk).
    pub fn clear(&mut self, key: &SessionKey) -> Result<()> {
        self.cache.remove(&key.0);
        let path = self.session_path(key);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crew_core::MessageRole;
    use tempfile::TempDir;

    fn make_message(role: MessageRole, content: &str) -> Message {
        Message {
            role,
            content: content.into(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_session_get_history() {
        let mut session = Session::new(SessionKey::new("cli", "test"));
        for i in 0..10 {
            session
                .messages
                .push(make_message(MessageRole::User, &format!("msg{i}")));
        }
        let history = session.get_history(3);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].content, "msg7");
        assert_eq!(history[2].content, "msg9");
    }

    #[test]
    fn test_session_get_history_all() {
        let mut session = Session::new(SessionKey::new("cli", "test"));
        session.messages.push(make_message(MessageRole::User, "a"));
        session.messages.push(make_message(MessageRole::User, "b"));
        let history = session.get_history(10);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_session_manager_create_and_retrieve() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SessionManager::open(tmp.path()).unwrap();
        let key = SessionKey::new("cli", "default");

        let session = mgr.get_or_create(&key);
        assert_eq!(session.messages.len(), 0);

        mgr.add_message(&key, make_message(MessageRole::User, "hello"))
            .unwrap();
        mgr.add_message(&key, make_message(MessageRole::Assistant, "hi"))
            .unwrap();

        let session = mgr.get_or_create(&key);
        assert_eq!(session.messages.len(), 2);
    }

    #[test]
    fn test_session_manager_persistence() {
        let tmp = TempDir::new().unwrap();
        let key = SessionKey::new("cli", "persist");

        // Write session
        {
            let mut mgr = SessionManager::open(tmp.path()).unwrap();
            mgr.add_message(&key, make_message(MessageRole::User, "saved"))
                .unwrap();
            mgr.add_message(&key, make_message(MessageRole::Assistant, "reply"))
                .unwrap();
        }

        // New manager should load from disk
        {
            let mut mgr = SessionManager::open(tmp.path()).unwrap();
            let session = mgr.get_or_create(&key);
            assert_eq!(session.messages.len(), 2);
            assert_eq!(session.messages[0].content, "saved");
            assert_eq!(session.messages[1].content, "reply");
        }
    }

    #[test]
    fn test_session_manager_clear() {
        let tmp = TempDir::new().unwrap();
        let key = SessionKey::new("cli", "clear-me");
        let mut mgr = SessionManager::open(tmp.path()).unwrap();

        mgr.add_message(&key, make_message(MessageRole::User, "temp"))
            .unwrap();
        assert_eq!(mgr.get_or_create(&key).messages.len(), 1);

        mgr.clear(&key).unwrap();

        // After clear, should be empty
        let session = mgr.get_or_create(&key);
        assert_eq!(session.messages.len(), 0);
    }

    #[test]
    fn test_concurrent_sessions() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SessionManager::open(tmp.path()).unwrap();

        let k1 = SessionKey::new("telegram", "chat1");
        let k2 = SessionKey::new("telegram", "chat2");

        mgr.add_message(&k1, make_message(MessageRole::User, "from chat1"))
            .unwrap();
        mgr.add_message(&k2, make_message(MessageRole::User, "from chat2"))
            .unwrap();

        assert_eq!(mgr.get_or_create(&k1).messages.len(), 1);
        assert_eq!(mgr.get_or_create(&k2).messages.len(), 1);
        assert_eq!(mgr.get_or_create(&k1).messages[0].content, "from chat1");
        assert_eq!(mgr.get_or_create(&k2).messages[0].content, "from chat2");
    }

    #[test]
    fn test_session_rewrite() {
        let tmp = TempDir::new().unwrap();
        let key = SessionKey::new("cli", "rewrite");
        let mut mgr = SessionManager::open(tmp.path()).unwrap();

        // Add 5 messages
        for i in 0..5 {
            mgr.add_message(&key, make_message(MessageRole::User, &format!("msg{i}")))
                .unwrap();
        }

        // Mutate in-memory: keep only last 2
        let session = mgr.get_or_create(&key);
        session.messages.drain(0..3);
        assert_eq!(session.messages.len(), 2);

        // Rewrite to disk
        mgr.rewrite(&key).unwrap();

        // Load fresh from disk — should have only 2 messages
        let mut mgr2 = SessionManager::open(tmp.path()).unwrap();
        let session2 = mgr2.get_or_create(&key);
        assert_eq!(session2.messages.len(), 2);
        assert_eq!(session2.messages[0].content, "msg3");
        assert_eq!(session2.messages[1].content, "msg4");
    }

    #[test]
    fn test_fork_creates_child() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SessionManager::open(tmp.path()).unwrap();
        let parent = SessionKey::new("telegram", "chat1");

        for i in 0..5 {
            mgr.add_message(&parent, make_message(MessageRole::User, &format!("msg{i}")))
                .unwrap();
        }

        let child_key = mgr.fork(&parent, "chat1_fork", 3).unwrap();
        assert_eq!(child_key, SessionKey::new("telegram", "chat1_fork"));

        let child = mgr.get_or_create(&child_key);
        assert_eq!(child.parent_key, Some(parent.clone()));
        assert_eq!(child.messages.len(), 3);
        assert_eq!(child.messages[0].content, "msg2");
        assert_eq!(child.messages[2].content, "msg4");
    }

    #[test]
    fn test_fork_persists_to_disk() {
        let tmp = TempDir::new().unwrap();
        let parent = SessionKey::new("cli", "main");

        {
            let mut mgr = SessionManager::open(tmp.path()).unwrap();
            mgr.add_message(&parent, make_message(MessageRole::User, "hello"))
                .unwrap();
            mgr.fork(&parent, "branch", 1).unwrap();
        }

        // Reload from disk
        let mut mgr2 = SessionManager::open(tmp.path()).unwrap();
        let child_key = SessionKey::new("cli", "branch");
        let child = mgr2.get_or_create(&child_key);
        assert_eq!(child.parent_key, Some(parent));
        assert_eq!(child.messages.len(), 1);
        assert_eq!(child.messages[0].content, "hello");
    }
}
