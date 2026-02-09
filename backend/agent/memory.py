from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mem0 import Memory
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from backend.agent.prompts.custom_fact_extraction_prompt import custom_fact_extraction_prompt
from backend.agent.prompts.custom_update_memory_prompt import custom_update_memory_prompt

@dataclass(frozen=True)
class Mem0MilvusStoreConfig:
    # Milvus / Zilliz
    url: str                      # "./milvus.db" for milvus-lite, or "http://host:19530"
    token: Optional[str]          # optional
    db_name: Optional[str]        # optional
    collection_name: str
    embedding_model_dims: int

    # mem0
    version: str                  # e.g. "v1.1"

    # behavior
    top_k: int
    max_payload_chars: int


class Mem0MilvusMemoryStore:
    """
    Standalone memory store for:
      - recording human review feedback (accept/edit/reject)
      - searching memories for designer

    No global variables. Lazy-initializes mem0 Memory on first use.
    """

    def __init__(self, config: Optional[Mem0MilvusStoreConfig] = None) -> None:
        self._cfg = config or self._load_config_from_env()
        self._mem: Optional[Memory] = None  # lazy init

    # -------------------------
    # Public APIs
    # -------------------------

    def record_from_human_review(
        self,
        *,
        state: Dict[str, Any],
        action: str,
        human_feedback: str,
        experiments_plan: Any,
        experiments: Any = None,
    ) -> None:
        """
        accept:
          - record user_intent, experiments_plan, human_feedback
          - mark plan as correct (accepted_plan)

        edit:
          - record user_intent, experiments(correct), experiments_plan(wrong), human_feedback
          - mark experiments as correct, experiments_plan as wrong

        reject:
          - record user_intent, experiments_plan(wrong), human_feedback
          - mark plan as wrong (rejected_plan)
        """
        action = (action or "").strip().lower()
        if action not in ("accept", "edit", "reject"):
            return

        user_id = self._extract_user_id(state)
        user_intent = str(state.get("user_intent") or "").strip()
        run_id = str(state.get("run_id") or "")

        base_meta = self._base_metadata(run_id=run_id, action=action)

        if action == "accept":
            meta = self._mark_label(base_meta, label="accepted_plan", is_correct=True)
            text = self._build_text_accept(
                user_intent=user_intent,
                human_feedback=human_feedback,
                experiments_plan=experiments_plan,
            )
            self._add(user_id=user_id, text=text, metadata=meta)
            return

        if action == "edit":
            # Correct experiments (edited)
            meta_ok = self._mark_label(base_meta, label="edited_correct_experiments", is_correct=True)
            text_ok = self._build_text_edit_correct(
                user_intent=user_intent,
                human_feedback=human_feedback,
                experiments=experiments,
            )
            self._add(user_id=user_id, text=text_ok, metadata=meta_ok)

            # Wrong original plan
            meta_bad = self._mark_label(base_meta, label="edited_incorrect_plan", is_correct=False)
            text_bad = self._build_text_edit_wrong(
                user_intent=user_intent,
                human_feedback=human_feedback,
                experiments_plan=experiments_plan,
            )
            self._add(user_id=user_id, text=text_bad, metadata=meta_bad)
            return

        if action == "reject":
            meta = self._mark_label(base_meta, label="rejected_plan", is_correct=False)
            text = self._build_text_reject(
                user_intent=user_intent,
                human_feedback=human_feedback,
                experiments_plan=experiments_plan,
            )
            self._add(user_id=user_id, text=text, metadata=meta)
            return

    def search_for_designer(
        self,
        *,
        state: Dict[str, Any],
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search by semantic query (typically user_intent).
        Returns compact memories for injecting into llm_req context.
        """
        user_id = self._extract_user_id(state)
        q = (query or "").strip()
        if not q:
            return []

        limit = int(top_k or self._cfg.top_k)
        mem = self._get_mem()

        res = mem.search(query=q, user_id=user_id, limit=limit)
        items: List[Dict[str, Any]]
        if isinstance(res, dict) and "results" in res:
            items = res["results"]  # type: ignore
        elif isinstance(res, list):
            items = res
        else:
            items = []

        out: List[Dict[str, Any]] = []
        for it in items[:limit]:
            if not isinstance(it, dict):
                continue
            out.append(
                {
                    "id": it.get("id"),
                    "memory": it.get("memory"),
                    "metadata": it.get("metadata"),
                    "created_at": it.get("created_at"),
                    "updated_at": it.get("updated_at"),
                }
            )
        return out

    # -------------------------
    # Internal: mem0 init
    # -------------------------

    def _get_mem(self):
        if self._mem is not None:
            return self._mem

        cfg = self._cfg
        vector_cfg: Dict[str, Any] = {
            "collection_name": cfg.collection_name,
            "embedding_model_dims": cfg.embedding_model_dims,
            "url": cfg.url,
        }

        # ✅ 关键：只有在有值时才传 token/db_name，否则不要传这个 key
        if cfg.token:  # 非空字符串才传
            vector_cfg["token"] = cfg.token
        if cfg.db_name:
            vector_cfg["db_name"] = cfg.db_name

        config = {
            "vector_store": {
                "provider": "milvus",
                "config": vector_cfg,
            },
            "version": cfg.version,
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen-max",
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": "sk-03911389b5724651a484b91acfe10e09"
                }
            },
            "custom_fact_extraction_prompt": custom_fact_extraction_prompt,
            "custom_update_memory_prompt": custom_update_memory_prompt
        }

        self._mem = Memory.from_config(config)
        return self._mem

    # -------------------------
    # Internal: write
    # -------------------------

    def _add(self, *, user_id: str, text: str, metadata: Dict[str, Any]) -> None:
        mem = self._get_mem()

        # ✅ 强制写入：用 messages 形式 + infer=False，避免抽取为空导致 results=[]
        messages = [{"role": "user", "content": text}]
        resp = mem.add(messages, user_id=user_id, metadata=metadata, infer=True)

        print(f"[mem0][add] user_id={user_id} resp={resp}")

    # -------------------------
    # Internal: helpers (high cohesion)
    # -------------------------

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _base_metadata(self, *, run_id: str, action: str) -> Dict[str, Any]:
        return {
            "app": "simu_agent",
            "agent": "designer",
            "event": "human_review",
            "action": action,
            "run_id": run_id,
            "ts_ms": self._now_ms(),
        }

    @staticmethod
    def _mark_label(meta: Dict[str, Any], *, label: str, is_correct: bool) -> Dict[str, Any]:
        # "标记xx为正确/错误结果" 封装点
        out = dict(meta)
        out["label"] = label
        out["is_correct"] = bool(is_correct)
        return out

    def _extract_user_id(self, state: Dict[str, Any]) -> str:
        # 你可以在 API 层把 state["mem_user_id"] 固定写入；否则 fallback 到 session/run
        return str(
            state.get("mem_user_id")
            or state.get("user_id")
            or state.get("session_id")
            or "default_user"
        )

    def _safe_json(self, obj: Any) -> str:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        if len(s) > self._cfg.max_payload_chars:
            return s[: self._cfg.max_payload_chars - 20] + "...(truncated)"
        return s

    # ---- text builders (record fields) ----

    def _build_text_accept(self, *, user_intent: str, human_feedback: str, experiments_plan: Any) -> str:
        return (
            "[ACCEPTED] user accepted experiment plan.\n"
            f"user_intent: {user_intent}\n"
            f"human_feedback: {human_feedback}\n"
            f"experiments_plan: {self._safe_json(experiments_plan)}\n"
        )

    def _build_text_edit_correct(self, *, user_intent: str, human_feedback: str, experiments: Any) -> str:
        return (
            "[EDIT-CORRECT] user edited the plan; this is the CORRECT experiments.\n"
            f"user_intent: {user_intent}\n"
            f"human_feedback: {human_feedback}\n"
            f"experiments(correct): {self._safe_json(experiments)}\n"
        )

    def _build_text_edit_wrong(self, *, user_intent: str, human_feedback: str, experiments_plan: Any) -> str:
        return (
            "[EDIT-WRONG] the original experiments_plan was WRONG (user edited it).\n"
            f"user_intent: {user_intent}\n"
            f"human_feedback: {human_feedback}\n"
            f"experiments_plan(wrong): {self._safe_json(experiments_plan)}\n"
        )

    def _build_text_reject(self, *, user_intent: str, human_feedback: str, experiments_plan: Any) -> str:
        return (
            "[REJECTED] user rejected experiment plan.\n"
            f"user_intent: {user_intent}\n"
            f"human_feedback: {human_feedback}\n"
            f"experiments_plan(wrong): {self._safe_json(experiments_plan)}\n"
        )

    # -------------------------
    # Config loader
    # -------------------------

    @staticmethod
    def _load_config_from_env() -> Mem0MilvusStoreConfig:
        """
        Default config from env. Keep this file standalone.
        """
        load_dotenv(override=True)
        return Mem0MilvusStoreConfig(
            url=os.getenv("MEM0_MILVUS_URL", "./milvus.db"),
            token=os.getenv("MEM0_MILVUS_TOKEN") or None,
            db_name=os.getenv("MEM0_MILVUS_DB_NAME") or None,
            collection_name=os.getenv("MEM0_MILVUS_COLLECTION", "simu_agent_mem0"),
            embedding_model_dims=int(os.getenv("MEM0_EMBED_DIMS", "1536")),  # ([docs.mem0.ai](https://docs.mem0.ai/components/vectordbs/dbs/milvus))
            version=os.getenv("MEM0_VERSION", "v1.1"),
            top_k=int(os.getenv("MEM0_TOP_K", "5")),
            max_payload_chars=int(os.getenv("MEM0_MAX_PAYLOAD_CHARS", "12000")),
        )

def _get_mem_store(config: Optional[RunnableConfig]) -> Optional[Mem0MilvusMemoryStore]:
    if not config:
        return None
    cfg = (config.get("configurable") or {})  # type: ignore
    store = cfg.get("mem_store")
    return store if isinstance(store, Mem0MilvusMemoryStore) else None