"""Conversation management primitives."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Sequence

from .memory import MemoryCompressor
from .models import CharacterCard, ConversationTurn, MemoryEntry, PromptContext, WorldInfo
from .prompt import PromptBuilder, limit_history, limit_memories


class ConversationManager:
    """Manages conversation state for SillyTavern-like experiences."""

    def __init__(
        self,
        *,
        history_limit: int = 12,
        memory_limit: int = 5,
        retention_limit: int = 50,
        archive_threshold: float = 0.5,
    ) -> None:
        self.history_limit = history_limit
        self.memory_limit = memory_limit
        self.history: Deque[ConversationTurn] = deque(maxlen=history_limit)
        self.memories: List[MemoryEntry] = []
        self.prompt_builder = PromptBuilder(history_limit=history_limit, memory_limit=memory_limit)
        self.memory_compressor = MemoryCompressor(
            retention_limit=retention_limit, archive_threshold=archive_threshold
        )
        self.character: CharacterCard | None = None
        self.world_info: List[WorldInfo] = []
        self.system_prompt: str | None = None

    def set_character(self, card: CharacterCard) -> None:
        self.character = card

    def set_world_info(self, entries: Sequence[WorldInfo]) -> None:
        self.world_info = list(entries)

    def set_system_prompt(self, prompt: str | None) -> None:
        self.system_prompt = prompt

    def load_memories(self, memories: Iterable[MemoryEntry]) -> None:
        self.memories = list(memories)

    def remember(self, entry: MemoryEntry) -> None:
        self.memories.append(entry)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.history.append(turn)

    def build_context(self, user_message: str | None = None) -> PromptContext:
        return PromptContext(
            system_prompt=self.system_prompt,
            character=self.character,
            world_info=self.world_info,
            memories=limit_memories(self.memories, self.memory_limit),
            history=limit_history(list(self.history), self.history_limit),
            user_message=user_message,
        )

    def build_prompt(self, user_message: str | None = None) -> str:
        context = self.build_context(user_message)
        return self.prompt_builder.build_prompt(context)

    def iter_messages(self, user_message: str | None = None) -> Iterable[dict]:
        context = self.build_context(user_message)
        return self.prompt_builder.iter_messages(context)

    def compress_memories(self) -> None:
        result = self.memory_compressor.compress(self.memories)
        self.memories = result.retained

    def reset(self) -> None:
        self.history.clear()
        self.memories.clear()
        self.character = None
        self.world_info = []
        self.system_prompt = None


def conversation_from_transcript(transcript: Sequence[tuple[str, str]]) -> List[ConversationTurn]:
    """Create conversation turns from a raw transcript."""

    return [ConversationTurn(role=role, content=content) for role, content in transcript]
