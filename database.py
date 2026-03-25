import sqlite3
from pathlib import Path
import uuid
import datetime

DB_PATH = Path(__file__).parent / "data" / "chat_history.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def create_session(title="Chat mới"):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO sessions (id, title) VALUES (?, ?)', (session_id, title))
    conn.commit()
    conn.close()
    return session_id

def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, created_at FROM sessions ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA foreign_keys = ON') # Enable cascading deletes
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()

def update_session_title(session_id, new_title):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE sessions SET title = ? WHERE id = ?', (new_title, session_id))
    conn.commit()
    conn.close()

def get_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC', (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_message(session_id, role, content):
    msg_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO messages (id, session_id, role, content) 
        VALUES (?, ?, ?, ?)
    ''', (msg_id, session_id, role, content))
    conn.commit()
    conn.close()
    return msg_id

# Initialize DB on load
init_db()
