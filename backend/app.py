# backend/app.py
from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from .memory_store import MemoryStore
from .agent.graph import build_graph  # 你的真实 agent 图构建函数


# ========== App & Middleware ==========
app = FastAPI(title="Agent App (FastAPI + POST SSE, LangGraph invoke)")

# 开发期允许本地直接打开 html 调用；生产请改成你的域名白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = MemoryStore()

# 确保 data/runs 存在（你原 main() 里做的那段）
BASE_RUN_DIR = os.path.join(os.path.dirname(__file__), "data", "runs")
os.makedirs(BASE_RUN_DIR, exist_ok=True)

# 复用同一个 app(graph) 实例，避免每次请求重复 build（通常更快）
GRAPH_APP = build_graph()


# ========== Models ==========
class CreateSessionResp(BaseModel):
    session_id: str


class ChatReq(BaseModel):
    session_id: Optional[str] = Field(default=None, description="不传则自动创建")
    message: str = Field(..., description="用户意图/输入")
    debug: bool = Field(default=False, description="debug 开关")
    max_retries: int = Field(default=1, ge=0, le=10, description="最大重试次数（写入 init_state）")


class ChatResp(BaseModel):
    session_id: str
    report: Dict[str, Any]


class ChatStreamReq(BaseModel):
    session_id: Optional[str] = Field(default=None, description="不传则自动创建")
    message: str = Field(..., description="用户意图/输入")
    debug: bool = Field(default=False, description="debug 开关")
    max_retries: int = Field(default=1, ge=0, le=10, description="最大重试次数（写入 init_state）")


# ========== Helpers ==========
def _history_as_dicts(session_id: str) -> List[Dict[str, str]]:
    msgs = store.get(session_id)
    return [{"role": m.role, "content": m.content} for m in msgs]


def sse_pack(data: Any, event: str = "message") -> str:
    """
    SSE 格式：
      event: <name>
      data: <json>
      <空行>
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def run_real_agent(user_intent: str, *, thread_id: str, debug: bool, max_retries: int) -> Dict[str, Any]:
    """
    直接复刻你 main() 里 app.invoke 的逻辑，并返回 report（即 final_state['analyst_report']）。
    """
    init_state: Dict[str, Any] = {
        "debug": debug,
        "stage": "init",
        "user_intent": user_intent,
        "retries": 0,
        "max_retries": max_retries,
    }

    # thread_id 用于 checkpoint / 多会话并行
    config = {"configurable": {"thread_id": thread_id}}

    final_state = GRAPH_APP.invoke(init_state, config=config)
    report = final_state.get("analyst_report", {})

    # 保底：确保是 dict
    if report is None:
        report = {}
    if not isinstance(report, dict):
        report = {"value": report}

    return report


# ========== Routes ==========
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=CreateSessionResp)
def create_session() -> CreateSessionResp:
    sid = store.new_session()
    return CreateSessionResp(session_id=sid)


@app.post("/api/chat", response_model=ChatResp)
def chat(req: ChatReq) -> ChatResp:
    sid = req.session_id or store.new_session()

    # 记录用户消息
    store.append(sid, "user", req.message)

    # （可选）如果你的 graph 内部会用到历史消息，可以从这里取：
    # history = _history_as_dicts(sid)

    # 调用真实 agent（thread_id 直接用 session_id，保证多会话隔离）
    report = run_real_agent(
        req.message,
        thread_id=sid,
        debug=req.debug,
        max_retries=req.max_retries,
    )

    # 记录助手消息（把 report 序列化存一下，便于回看/审计）
    store.append(sid, "assistant", json.dumps(report, ensure_ascii=False))

    return ChatResp(session_id=sid, report=report)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamReq) -> StreamingResponse:
    """
    流式接口：POST + SSE
    注意：你的当前 invoke 调用本身不是 token-streaming。
    这里会用“阶段事件 + 最终 done”让前端仍可流式更新 UI。
    若你后续改为 GRAPH_APP.stream / astream，可在这里把中间产物逐步 yield 出去。
    """
    sid = req.session_id or store.new_session()

    # 记录用户消息
    store.append(sid, "user", req.message)

    async def gen():
        # 先发 session_id
        yield sse_pack({"session_id": sid}, event="meta")

        # 发一个开始事件（前端可显示“处理中...”）
        yield sse_pack({"status": "started"}, event="delta")
        await asyncio.sleep(0)

        try:
            # 因为 invoke 是同步阻塞的，为避免阻塞事件循环，用线程池跑
            report = await asyncio.to_thread(
                run_real_agent,
                req.message,
                thread_id=sid,
                debug=req.debug,
                max_retries=req.max_retries,
            )
        except Exception as e:
            yield sse_pack({"error": str(e)}, event="error")
            return

        # 存储助手消息
        store.append(sid, "assistant", json.dumps(report, ensure_ascii=False))

        # 最终返回
        yield sse_pack({"report": report}, event="done")

    return StreamingResponse(gen(), media_type="text/event-stream")
