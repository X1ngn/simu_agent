# tests/helpers/trace.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, List, Optional
import time
import inspect

from langchain_core.runnables import RunnableConfig


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    start: float
    end: float
    ok: bool
    error: Optional[str] = None


@dataclass
class NodeCall:
    node: str
    start: float
    end: float
    duration: float
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    diff: Dict[str, Any]


class TraceRecorder:
    """
    TraceRecorder 负责记录：
    - nodes: 每个 node 的调用耗时 + state diff
    - tools: 工具调用耗时（可选）
    - current_config: ✅ 关键兜底。LangGraph 有时不会把 config 注入 node 函数；
      所以测试在 invoke/ainvoke 前写入 current_config，wrap_node 会优先使用它。
    """

    def __init__(self):
        self.nodes: List[NodeCall] = []
        self.tools: List[ToolCall] = []
        self.current_config: RunnableConfig | None = None

    def record_node(
        self,
        node: str,
        start: float,
        end: float,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        diff: Dict[str, Any],
    ):
        self.nodes.append(
            NodeCall(
                node=node,
                start=start,
                end=end,
                duration=end - start,
                state_before=state_before,
                state_after=state_after,
                diff=diff,
            )
        )


def shallow_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if a.get(k) != b.get(k):
            diff[k] = {"before": a.get(k), "after": b.get(k)}
    return diff


def _pick_config(trace: TraceRecorder, config: RunnableConfig | None, kwargs: Dict[str, Any]) -> RunnableConfig:
    """
    统一获取 config：
    1) kwargs["config"]
    2) 显式参数 config
    3) trace.current_config（测试兜底）
    4) 空 RunnableConfig
    """
    cfg = kwargs.get("config", None) or config
    if cfg is None:
        cfg = trace.current_config
    if cfg is None:
        cfg = RunnableConfig(configurable={})
    return cfg


def wrap_node(trace: TraceRecorder, name: str, fn):
    """
    ✅ 同时兼容：
      - sync node: def fn(state) / def fn(state, config)
      - async node: async def fn(state) / async def fn(state, config)

    ✅ config 注入不可靠时，优先使用 trace.current_config（由测试在 invoke/ainvoke 前设置）
    """
    sig = inspect.signature(fn)
    needs_config = len(sig.parameters) >= 2
    is_async = inspect.iscoroutinefunction(fn)

    # -------------------------
    # async wrapper
    # -------------------------
    async def _wrapped_async(
        state: Dict[str, Any],
        config: RunnableConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        state_before = dict(state)
        t0 = time.perf_counter()

        cfg = _pick_config(trace, config, kwargs)

        if needs_config:
            out = await fn(state, cfg)
        else:
            out = await fn(state)

        t1 = time.perf_counter()

        state_after = dict(state)
        if isinstance(out, dict):
            state_after.update(out)

        diff = shallow_diff(state_before, state_after)

        trace.record_node(
            node=name,
            start=t0,
            end=t1,
            state_before=state_before,
            state_after=state_after,
            diff=diff,
        )
        return out

    # -------------------------
    # sync wrapper
    # -------------------------
    def _wrapped_sync(
        state: Dict[str, Any],
        config: RunnableConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        state_before = dict(state)
        t0 = time.perf_counter()

        cfg = _pick_config(trace, config, kwargs)

        if needs_config:
            out = fn(state, cfg)
        else:
            out = fn(state)

        t1 = time.perf_counter()

        state_after = dict(state)
        if isinstance(out, dict):
            state_after.update(out)

        diff = shallow_diff(state_before, state_after)

        trace.record_node(
            node=name,
            start=t0,
            end=t1,
            state_before=state_before,
            state_after=state_after,
            diff=diff,
        )
        return out

    return _wrapped_async if is_async else _wrapped_sync


class ToolStub:
    """带 trace 的工具桩，可注入延迟；同时支持 sync/async fn"""

    def __init__(
        self,
        tr: TraceRecorder,
        name: str,
        fn: Callable[[Dict[str, Any]], Any],
        delay_s: float = 0.0,
    ):
        self.tr = tr
        self.name = name
        self.fn = fn
        self.delay_s = delay_s
        self.is_async = inspect.iscoroutinefunction(fn)

    async def _call_async(self, args: Dict[str, Any]) -> Any:
        start = time.perf_counter()
        try:
            if self.delay_s > 0:
                # 这里用 time.sleep 也行，但会阻塞；若你未来要更真实 async，可换 asyncio.sleep
                time.sleep(self.delay_s)
            out = await self.fn(args)
            end = time.perf_counter()
            self.tr.tools.append(ToolCall(tool=self.name, args=args, start=start, end=end, ok=True))
            return out
        except Exception as e:
            end = time.perf_counter()
            self.tr.tools.append(ToolCall(tool=self.name, args=args, start=start, end=end, ok=False, error=str(e)))
            raise

    def __call__(self, args: Dict[str, Any]) -> Any:
        """
        若 fn 是 async，这里返回 coroutine（让调用方 await）。
        若 fn 是 sync，直接返回结果。
        """
        if self.is_async:
            return self._call_async(args)

        start = time.perf_counter()
        try:
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            out = self.fn(args)
            end = time.perf_counter()
            self.tr.tools.append(ToolCall(tool=self.name, args=args, start=start, end=end, ok=True))
            return out
        except Exception as e:
            end = time.perf_counter()
            self.tr.tools.append(ToolCall(tool=self.name, args=args, start=start, end=end, ok=False, error=str(e)))
            raise
