"""模型接口层，负责对接聊天模型服务。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Iterator, Optional

from .models import PromptContext
from .prompt import PromptBuilder

try:  # pragma: no cover - 提供可选依赖的降级路径
    from openai import OpenAI
except Exception:  # pragma: no cover - 测试环境缺少 openai 时使用占位符
    OpenAI = None  # type: ignore[misc]


@dataclass
class ModelConfig:
    """模型调用的基础配置。"""

    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_output_tokens: Optional[int] = None
    extra_headers: Optional[Dict[str, str]] = None
    stream: bool = False


class ModelInterface:
    """根据提示词上下文生成聊天回复的高层封装。"""

    def __init__(
        self,
        config: ModelConfig,
        *,
        builder: Optional[PromptBuilder] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.config = config
        self.builder = builder or PromptBuilder()
        if client is not None:
            self.client = client
        else:
            if OpenAI is None:  # pragma: no cover - 缺少依赖时抛错
                raise RuntimeError(
                    "openai 库未安装，无法创建默认模型客户端。请先安装依赖或传入自定义 client。"
                )
            kwargs = {"api_key": config.api_key}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self.client = OpenAI(**kwargs)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # 对外方法
    # ------------------------------------------------------------------
    def complete(
        self,
        context: PromptContext,
        *,
        user_message: Optional[str] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> str | Iterator[str]:
        """向模型发送提示词并返回回复。

        当 ``stream`` 为 True 时，返回一个生成器按块产出回复文本；否则直接返回完整回复。
        """

        effective_context = replace(context, user_message=user_message) if user_message is not None else context
        messages = list(self.builder.iter_messages(effective_context))

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": stream if stream is not None else self.config.stream,
        }

        max_tokens = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        headers = extra_headers if extra_headers is not None else self.config.extra_headers
        if headers:
            payload["extra_headers"] = headers

        response = self.client.chat.completions.create(**payload)

        if payload["stream"]:
            return self._consume_stream(response)
        return self._extract_text(response)

    def complete_stream(
        self,
        context: PromptContext,
        *,
        user_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """始终以流式返回模型回复。"""

        result = self.complete(
            context,
            user_message=user_message,
            stream=True,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            extra_headers=extra_headers,
        )
        if isinstance(result, str):
            yield result
        else:
            yield from result

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(response: object) -> str:
        """从标准 ChatCompletion 响应中提取文本。"""

        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is not None and getattr(message, "content", None):
            return message.content  # type: ignore[return-value]
        if getattr(choice, "text", None):
            return choice.text  # type: ignore[return-value]
        return ""

    @staticmethod
    def _consume_stream(stream_response: Iterable[object]) -> Iterator[str]:
        """将流式事件转为纯文本生成器。"""

        for event in stream_response:
            choices = getattr(event, "choices", None) or []
            for choice in choices:
                delta = getattr(choice, "delta", None)
                if delta is not None and getattr(delta, "content", None):
                    yield delta.content  # type: ignore[misc]
                    continue
                message = getattr(choice, "message", None)
                if message is not None and getattr(message, "content", None):
                    yield message.content  # type: ignore[misc]
                    continue
                if getattr(choice, "text", None):
                    yield choice.text  # type: ignore[misc]

