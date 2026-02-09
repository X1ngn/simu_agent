from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, List, Optional
import time
import inspect

# ✅ 用 langchain 的 RunnableConfig（和你生产代码一致）
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
    - tools: 将来你可以把工具调用也塞进来
    - current_config: ✅ 关键兜底。因为 LangGraph 版本/签名规则可能不稳定注入 config，
      所以我们让测试在 invoke 前写入 current_config，wrap_node 必须优先使用它。
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
    diff = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if a.get(k) != b.get(k):
            diff[k] = {"before": a.get(k), "after": b.get(k)}
    return diff


def wrap_node(trace: TraceRecorder, name: str, fn):
    """
    ✅ 兼容真实 agent 的签名：
      - fn(state, config)
    也兼容 stub 的签名：
      - fn(state)

    ✅ 关键点：config 注入不可靠时，优先使用 trace.current_config（由测试在 invoke 前设置）
    """
    sig = inspect.signature(fn)
    needs_config = len(sig.parameters) >= 2

    def _wrapped(
        state: Dict[str, Any],
        config: RunnableConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        state_before = dict(state)
        t0 = time.perf_counter()

        # 有些版本会把 config 放到 kwargs["config"]
        cfg = kwargs.get("config", None) or config

        # ✅ 最强兜底：LangGraph 不传 config 或传 None 时，用测试预先写入的 current_config
        if cfg is None:
            cfg = trace.current_config

        # 仍然 None 就给一个空 RunnableConfig，避免生产代码 type/attr 崩
        if cfg is None:
            cfg = RunnableConfig(configurable={})

        if needs_config:
            out = fn(state, cfg)
        else:
            out = fn(state)

        t1 = time.perf_counter()

        # 合成 state_after：仅 dict 输出 merge（Command 等不 merge，避免依赖内部结构）
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

    return _wrapped


class ToolStub:
    """带 trace 的工具桩，可注入延迟"""

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

    def __call__(self, args: Dict[str, Any]) -> Any:
        start = time.perf_counter()
        try:
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            out = self.fn(args)
            end = time.perf_counter()
            self.tr.tools.append(
                ToolCall(tool=self.name, args=args, start=start, end=end, ok=True)
            )
            return out
        except Exception as e:
            end = time.perf_counter()
            self.tr.tools.append(
                ToolCall(
                    tool=self.name, args=args, start=start, end=end, ok=False, error=str(e)
                )
            )
            raise
