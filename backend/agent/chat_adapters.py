from __future__ import annotations

from typing import List, Optional, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _messages_to_text(messages: List[BaseMessage]) -> str:
    # 简单、安全、可预期：把 system + history 拼成一个纯文本 prompt
    parts = []
    for m in messages:
        if isinstance(m, SystemMessage):
            parts.append(f"[SYSTEM]\n{m.content}")
        elif isinstance(m, HumanMessage):
            parts.append(f"[USER]\n{m.content}")
        else:
            # AIMessage / ToolMessage 等，你也可以扩展更精细的格式
            parts.append(f"[ASSISTANT]\n{m.content}")
    return "\n\n".join(parts)


class LLMAsChatModel(BaseChatModel):
    """
    Adapter: wraps a (non-chat) LLM (e.g., langchain_community.llms.VLLM)
    into a BaseChatModel so LangGraph nodes can depend only on BaseChatModel.
    """

    def __init__(self, llm: Any, model_name: str = "vllm_community_adapter"):
        super().__init__()
        self._llm = llm
        self._model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "llm_as_chat_model"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_text(messages)
        # LLM 返回字符串
        text = self._llm.invoke(prompt, stop=stop, **kwargs)
        ai = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai)])
