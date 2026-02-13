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
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

/// A conversation session with message history.
#[derive(Debug, Clone)]
pub struct Session {
    pub key: SessionKey,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    fn new(key: SessionKey) -> Self {
        let now = Utc::now();
        Self {
            key,
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

    /// Get or create a session. Loads from disk on first access.
    pub fn get_or_create(&mut self, key: &SessionKey) -> &mut Session {
        let key_str = key.0.clone();
        if !self.cache.contains_key(&key_str) {
            let session = self
                .load_from_disk(key)
                .unwrap_or_else(|| Session::new(key.clone()));
            self.cache.insert(key_str.clone(), session);
        }
        self.cache.get_mut(&key_str).unwrap()
    }

    /// Add a message to a session and persist it.
    pub fn add_message(&mut self, key: &SessionKey, message: Message) -> Result<()> {
        let session = self.get_or_create(key);
        session.messages.push(message.clone());
        session.updated_at = Utc::now();
        self.append_to_disk(key, &message)
    }

    /// Get the JSONL file path for a session key.
    fn session_path(&self, key: &SessionKey) -> PathBuf {
        // Replace characters that are invalid in filenames
        let safe_name = key.0.replace(['/', '\\', ':'], "_");
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
            messages,
            created_at: meta.created_at,
            updated_at: meta.updated_at,
        })
    }

    /// Append a message to the JSONL file. Creates the file with metadata if new.
    fn append_to_disk(&self, key: &SessionKey, message: &Message) -> Result<()> {
        use std::io::Write;

        let path = self.session_path(key);
        let is_new = !path.exists();

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        if is_new {
            let meta = SessionMeta {
                session_key: key.0.clone(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            writeln!(file, "{}", serde_json::to_string(&meta)?)?;
        }

        writeln!(file, "{}", serde_json::to_string(message)?)?;
        Ok(())
    }

    /// Rewrite a session's JSONL file from the in-memory state.
    /// This replaces the file entirely (meta + all current messages).
    pub fn rewrite(&self, key: &SessionKey) -> Result<()> {
        use std::io::Write;

        let session = self
            .cache
            .get(&key.0)
            .ok_or_else(|| eyre::eyre!("session not in cache: {}", key))?;

        let path = self.session_path(key);
        let mut file = std::fs::File::create(&path)?;

        let meta = SessionMeta {
            session_key: key.0.clone(),
            created_at: session.created_at,
            updated_at: session.updated_at,
        };
        writeln!(file, "{}", serde_json::to_string(&meta)?)?;

        for msg in &session.messages {
            writeln!(file, "{}", serde_json::to_string(msg)?)?;
        }

        debug!(key = %key, messages = session.messages.len(), "Rewrote session to disk");
        Ok(())
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
}
