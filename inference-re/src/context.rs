//! Context storage and retrieval using SQLite.
//!
//! This module provides the SQLite-based context memory described in the PRD.
//! It stores conversation history and provides simple retrieval mechanisms.

use rusqlite::{Connection, params, Result};
use std::path::Path;

/// A conversation turn consisting of user input and assistant response.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub id: i64,
    pub timestamp: String,
    pub user_message: String,
    pub assistant_response: String,
    pub token_count: i32,
}

/// SQLite-based context storage for conversation history.
pub struct ContextStore {
    conn: Connection,
}

impl ContextStore {
    /// Create a new context store, initializing the database if needed.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        
        // Create the conversation table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        
        Ok(Self { conn })
    }
    
    /// Store a conversation turn in the database.
    pub fn save_conversation(&self, user_message: &str, assistant_response: &str, token_count: i32) -> Result<i64> {
        let timestamp = chrono::Utc::now().to_rfc3339();
        
        let _rows_affected = self.conn.execute(
            "INSERT INTO conversations (timestamp, user_message, assistant_response, token_count)
             VALUES (?1, ?2, ?3, ?4)",
            params![timestamp, user_message, assistant_response, token_count],
        )?;
        
        Ok(self.conn.last_insert_rowid())
    }
    
    /// Retrieve the last N conversation turns.
    pub fn get_recent_conversations(&self, limit: i32) -> Result<Vec<ConversationTurn>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, user_message, assistant_response, token_count
             FROM conversations 
             ORDER BY id DESC 
             LIMIT ?1"
        )?;
        
        let conversation_iter = stmt.query_map([limit], |row| {
            Ok(ConversationTurn {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                user_message: row.get(2)?,
                assistant_response: row.get(3)?,
                token_count: row.get(4)?,
            })
        })?;
        
        let mut conversations = Vec::new();
        for conversation in conversation_iter {
            conversations.push(conversation?);
        }
        
        // Reverse to get chronological order (oldest first)
        conversations.reverse();
        
        Ok(conversations)
    }
    
    /// Get total number of stored conversations.
    pub fn get_conversation_count(&self) -> Result<i32> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM conversations",
            [],
            |row| row.get(0)
        )?;
        
        Ok(count)
    }
    
    /// Search for conversations containing specific text.
    pub fn search_conversations(&self, query: &str, limit: i32) -> Result<Vec<ConversationTurn>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, user_message, assistant_response, token_count
             FROM conversations 
             WHERE user_message LIKE ?1 OR assistant_response LIKE ?1
             ORDER BY id DESC 
             LIMIT ?2"
        )?;
        
        let search_pattern = format!("%{}%", query);
        let conversation_iter = stmt.query_map(params![search_pattern, limit], |row| {
            Ok(ConversationTurn {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                user_message: row.get(2)?,
                assistant_response: row.get(3)?,
                token_count: row.get(4)?,
            })
        })?;
        
        let mut conversations = Vec::new();
        for conversation in conversation_iter {
            conversations.push(conversation?);
        }
        
        Ok(conversations)
    }
    
    /// Clear all conversation history.
    pub fn clear_history(&self) -> Result<()> {
        self.conn.execute("DELETE FROM conversations", [])?;
        Ok(())
    }
    
    /// Get database statistics for monitoring.
    pub fn get_stats(&self) -> Result<ContextStats> {
        let total_conversations: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM conversations",
            [],
            |row| row.get(0)
        )?;
        
        let total_tokens: i32 = self.conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0) FROM conversations",
            [],
            |row| row.get(0)
        )?;
        
        let db_size = if let Some(path) = self.conn.path() {
            std::fs::metadata(path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };
        
        Ok(ContextStats {
            total_conversations,
            total_tokens,
            db_size_bytes: db_size,
        })
    }
}

/// Statistics about the context store.
#[derive(Debug)]
pub struct ContextStats {
    pub total_conversations: i32,
    pub total_tokens: i32,
    pub db_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_context_store_creation() {
        let db_path = "/tmp/test_context.db";
        let _ = fs::remove_file(db_path); // Clean up any existing test db
        
        let store = ContextStore::new(db_path).unwrap();
        assert_eq!(store.get_conversation_count().unwrap(), 0);
        
        // Clean up
        drop(store);
        let _ = fs::remove_file(db_path);
    }
    
    #[test]
    fn test_save_and_retrieve_conversation() {
        let db_path = "/tmp/test_save_retrieve.db";
        let _ = fs::remove_file(db_path);
        
        let store = ContextStore::new(db_path).unwrap();
        
        // Save a conversation
        let id = store.save_conversation("Hello", "Hi there!", 5).unwrap();
        assert!(id > 0);
        
        // Retrieve it
        let conversations = store.get_recent_conversations(1).unwrap();
        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].user_message, "Hello");
        assert_eq!(conversations[0].assistant_response, "Hi there!");
        assert_eq!(conversations[0].token_count, 5);
        
        // Clean up
        drop(store);
        let _ = fs::remove_file(db_path);
    }
    
    #[test]
    fn test_multiple_conversations() {
        let db_path = "/tmp/test_multiple.db";
        let _ = fs::remove_file(db_path);
        
        let store = ContextStore::new(db_path).unwrap();
        
        // Save multiple conversations
        store.save_conversation("First", "Response 1", 3).unwrap();
        store.save_conversation("Second", "Response 2", 4).unwrap();
        store.save_conversation("Third", "Response 3", 5).unwrap();
        
        // Get all conversations
        let all = store.get_recent_conversations(10).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].user_message, "First"); // Should be in chronological order
        assert_eq!(all[2].user_message, "Third");
        
        // Get limited conversations
        let recent = store.get_recent_conversations(2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].user_message, "Second"); // Last 2 in chronological order
        assert_eq!(recent[1].user_message, "Third");
        
        // Clean up
        drop(store);
        let _ = fs::remove_file(db_path);
    }
}