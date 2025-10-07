"""High level API surface for the SillyTavern core toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .character import load_character_card
from .conversation import ConversationManager, conversation_from_transcript
from .memory import MemoryCompressor
from .model_interface import ModelConfig, ModelInterface
from .models import ConversationTurn, MemoryEntry, PromptContext
from .prompt import PromptBuilder
from .world import load_world_info


def build_prompt_from_files(
    *,
    character_card: Path | str | None = None,
    world_info_files: Sequence[Path | str] = (),
    system_prompt: str | None = None,
    history: Sequence[ConversationTurn] = (),
    memories: Sequence[MemoryEntry] = (),
    user_message: str | None = None,
    history_limit: int = 12,
    memory_limit: int = 5,
) -> str:
    """Load resources from disk and assemble a prompt."""

    card = load_character_card(character_card) if character_card else None
    world_entries: List = []
    for path in world_info_files:
        world_entries.extend(load_world_info(path))

    context = PromptContext(
        system_prompt=system_prompt,
        character=card,
        world_info=world_entries,
        memories=list(memories),
        history=list(history),
        user_message=user_message,
    )
    builder = PromptBuilder(history_limit=history_limit, memory_limit=memory_limit)
    return builder.build_prompt(context)


def create_conversation_manager(
    *,
    character_card: Path | str | None = None,
    world_info_files: Sequence[Path | str] = (),
    system_prompt: str | None = None,
    history_limit: int = 12,
    memory_limit: int = 5,
) -> ConversationManager:
    """Create a conversation manager with optional resources pre-loaded."""

    manager = ConversationManager(history_limit=history_limit, memory_limit=memory_limit)
    if character_card:
        manager.set_character(load_character_card(character_card))
    if world_info_files:
        world_entries: List = []
        for path in world_info_files:
            world_entries.extend(load_world_info(path))
        manager.set_world_info(world_entries)
    manager.set_system_prompt(system_prompt)
    return manager


def compress_memories(
    memories: Iterable[MemoryEntry], *, retention_limit: int = 50, archive_threshold: float = 0.5
):
    """Compress a collection of memories and return the compression result."""

    compressor = MemoryCompressor(retention_limit=retention_limit, archive_threshold=archive_threshold)
    return compressor.compress(list(memories))


def turns_from_transcript(transcript: Sequence[tuple[str, str]]) -> List[ConversationTurn]:
    return conversation_from_transcript(transcript)


def create_model_interface(
    config: ModelConfig,
    *,
    builder: PromptBuilder | None = None,
    client: object | None = None,
) -> ModelInterface:
    """Convenience helper for constructing :class:`ModelInterface`."""

    return ModelInterface(config, builder=builder, client=client)


def generate_model_reply(
    config: ModelConfig,
    context: PromptContext,
    *,
    user_message: str | None = None,
    stream: bool | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str | Iterable[str]:
    """Build messages from ``context`` and request a model response."""

    interface = ModelInterface(config)
    return interface.complete(
        context,
        user_message=user_message,
        stream=stream,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        extra_headers=extra_headers,
    )
