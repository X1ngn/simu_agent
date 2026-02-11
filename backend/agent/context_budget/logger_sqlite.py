from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage

@dataclass
class CompressionEvent:
    thread_id: str
    start_idx: int
    end_idx: int
    raw_messages: List[Dict[str, Any]]
    summary_after: Dict[str, Any]
    token_before: int
    token_after: int

class SQLiteCompressionLogger:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compression_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    thread_id TEXT NOT NULL,
                    start_idx INTEGER NOT NULL,
                    end_idx INTEGER NOT NULL,
                    raw_json TEXT NOT NULL,
                    summary_after_json TEXT NOT NULL,
                    token_before INTEGER NOT NULL,
                    token_after INTEGER NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ce_thread ON compression_events(thread_id, id);")
            conn.commit()
        finally:
            conn.close()

    def log(self, ev: CompressionEvent) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO compression_events
                (thread_id, start_idx, end_idx, raw_json, summary_after_json, token_before, token_after)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev.thread_id,
                    ev.start_idx,
                    ev.end_idx,
                    json.dumps(ev.raw_messages, ensure_ascii=False),
                    json.dumps(ev.summary_after, ensure_ascii=False),
                    ev.token_before,
                    ev.token_after,
                ),
            )
            conn.commit()
        finally:
            conn.close()

def _msg_to_dict(m: BaseMessage, idx: int) -> Dict[str, Any]:
    # BaseMessage 没有统一 id，这里用 index + type/role 记录，足够追溯
    return {
        "idx": idx,
        "type": getattr(m, "type", m.__class__.__name__),
        "content": getattr(m, "content", "") or "",
    }
