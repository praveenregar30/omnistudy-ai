"""SQLite Persistence Layer for OmniStudy AI.

Provides per-user private data isolation for profiles, BKT mastery states,
chat histories, custom uploads, and quiz attempt logs.
"""

from __future__ import annotations
import sqlite3
import json
import time
import os
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "omnistudy.db")

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        display_name TEXT,
        academic_level TEXT DEFAULT 'Undergraduate',
        created_at REAL
    )
    """)
    # BKT Mastery per user
    cur.execute("""
    CREATE TABLE IF NOT EXISTS topic_mastery (
        username TEXT,
        topic TEXT,
        mastery_prob REAL,
        interactions_count INTEGER DEFAULT 0,
        correct_count INTEGER DEFAULT 0,
        stability_days REAL DEFAULT 2.0,
        last_updated REAL,
        PRIMARY KEY (username, topic)
    )
    """)
    # Chat history per user
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        role TEXT,
        content TEXT,
        citations_json TEXT,
        is_outside_knowledge INTEGER DEFAULT 0,
        timestamp REAL
    )
    """)
    # Quiz attempts per user
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question_id TEXT,
        question_text TEXT,
        question_type TEXT,
        topic TEXT,
        user_answer TEXT,
        correct_answer TEXT,
        is_correct INTEGER,
        misconception_tag TEXT,
        timestamp REAL
    )
    """)
    # User uploaded custom documents
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        title TEXT,
        source_type TEXT,
        topic TEXT,
        location TEXT,
        content_text TEXT,
        timestamp REAL
    )
    """)
    conn.commit()
    conn.close()

# Initialize DB upon import
init_db()

class UserStorage:
    @staticmethod
    def get_or_create_user(username: str, display_name: str, level: str = "Undergraduate") -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            now = time.time()
            cur.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (username, display_name, level, now))
            conn.commit()
            row_dict = {"username": username, "display_name": display_name, "academic_level": level, "created_at": now}
        else:
            row_dict = dict(row)
        conn.close()
        return row_dict

    @staticmethod
    def load_mastery(username: str) -> Dict[str, Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM topic_mastery WHERE username = ?", (username,))
        rows = cur.fetchall()
        conn.close()
        return {r["topic"]: dict(r) for r in rows}

    @staticmethod
    def save_mastery(username: str, topic: str, mastery_prob: float, interactions: int, correct: int, stability: float) -> None:
        conn = get_db_connection()
        cur = conn.cursor()
        now = time.time()
        cur.execute("""
        INSERT INTO topic_mastery (username, topic, mastery_prob, interactions_count, correct_count, stability_days, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username, topic) DO UPDATE SET
            mastery_prob = excluded.mastery_prob,
            interactions_count = excluded.interactions_count,
            correct_count = excluded.correct_count,
            stability_days = excluded.stability_days,
            last_updated = excluded.last_updated
        """, (username, topic, mastery_prob, interactions, correct, stability, now))
        conn.commit()
        conn.close()

    @staticmethod
    def save_chat_message(username: str, role: str, content: str, citations: Optional[List[Dict[str, Any]]] = None, is_outside: bool = False) -> None:
        conn = get_db_connection()
        cur = conn.cursor()
        c_json = json.dumps(citations or [])
        cur.execute("INSERT INTO chat_messages (username, role, content, citations_json, is_outside_knowledge, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (username, role, content, c_json, 1 if is_outside else 0, time.time()))
        conn.commit()
        conn.close()

    @staticmethod
    def get_chat_history(username: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM chat_messages WHERE username = ? ORDER BY id ASC LIMIT ?", (username, limit))
        rows = cur.fetchall()
        conn.close()
        res = []
        for r in rows:
            d = dict(r)
            try:
                d["citations"] = json.loads(d["citations_json"])
            except Exception:
                d["citations"] = []
            res.append(d)
        return res

    @staticmethod
    def clear_chat_history(username: str) -> None:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_messages WHERE username = ?", (username,))
        conn.commit()
        conn.close()

    @staticmethod
    def log_quiz_attempt(username: str, q_id: str, text: str, q_type: str, topic: str, user_ans: str, correct_ans: str, is_corr: bool, misconception: str = "") -> None:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO quiz_attempts (username, question_id, question_text, question_type, topic, user_answer, correct_answer, is_correct, misconception_tag, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, q_id, text, q_type, topic, user_ans, correct_ans, 1 if is_corr else 0, misconception, time.time()))
        conn.commit()
        conn.close()

    @staticmethod
    def get_answered_question_ids(username: str) -> List[str]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT question_id FROM quiz_attempts WHERE username = ?", (username,))
        rows = cur.fetchall()
        conn.close()
        return [r["question_id"] for r in rows]

    @staticmethod
    def save_user_document(username: str, title: str, source_type: str, topic: str, location: str, text: str) -> None:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO user_documents (username, title, source_type, topic, location, content_text, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (username, title, source_type, topic, location, text, time.time()))
        conn.commit()
        conn.close()

    @staticmethod
    def get_user_documents(username: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM user_documents WHERE username = ? ORDER BY id DESC", (username,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_user_documents_grouped(username: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
        SELECT title, topic, source_type, MIN(timestamp) as uploaded_at, COUNT(*) as page_count
        FROM user_documents
        WHERE username = ?
        GROUP BY title, topic, source_type
        ORDER BY uploaded_at DESC
        """, (username,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def delete_user_document(username: str, title: str) -> None:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM user_documents WHERE username = ? AND title = ?", (username, title))
        conn.commit()
        conn.close()

