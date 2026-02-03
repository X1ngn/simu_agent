from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal
import time
import uuid

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str
    ts: float = field(default_factory=lambda: time.time())

class MemoryStore:
    """
    最简内存会话存储：
    - 单进程有效
    - 重启丢失
    - 多 worker 不共享
    生产环境请替换为 Redis/Postgres 等
    """
    def __init__(self) -> None:
        self._sessions: Dict[str, List[Message]] = {}

    def new_session(self) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = []
        return sid

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get(self, session_id: str) -> List[Message]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: Role, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(Message(role=role, content=content))

    def reset(self, session_id: str) -> None:
        self._sessions[session_id] = []
