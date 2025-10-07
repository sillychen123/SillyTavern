from __future__ import annotations

import json
import base64
import struct
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from types import SimpleNamespace

from sillytavern_core.character import load_character_card
from sillytavern_core.conversation import ConversationManager, conversation_from_transcript
from sillytavern_core.memory import MemoryCompressor
from sillytavern_core.model_interface import ModelConfig, ModelInterface
from sillytavern_core.models import ConversationTurn, MemoryEntry, PromptContext
from sillytavern_core.prompt import PromptBuilder
from sillytavern_core.world import filter_world_info, load_world_info


@pytest.fixture()
def sample_world_file(tmp_path: Path) -> Path:
    path = tmp_path / "world.json"
    payload = {
        "entries": {
            "0": {
                "uid": 0,
                "key": ["city"],
                "comment": "City Overview",
                "content": "A bustling city",
                "displayIndex": 0,
            },
            "1": {
                "uid": 1,
                "key": ["forest"],
                "keysecondary": ["woods"],
                "comment": "Mystic Forest",
                "content": "A quiet forest",
                "displayIndex": 1,
                "position": 1,
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def sample_character_file(tmp_path: Path) -> Path:
    path = tmp_path / "char.json"
    payload = {
        "name": "Alice",
        "description": "Alice is an intrepid explorer.",
        "personality": "Curious and adventurous",
        "scenario": "Exploring a new world",
        "system_prompt": "Stay in character and be helpful.",
        "post_history_instructions": "Always ask follow-up questions.",
        "first_message": "Hello there!",
        "example_dialogue": ["Alice: Let's go!", "User: Sure!"],
        "world_info_refs": ["City"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def sample_character_png(tmp_path: Path) -> Path:
    path = tmp_path / "char.png"

    payload = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": "Alice",
            "description": "Alice is an intrepid explorer.",
            "personality": "Curious and adventurous",
            "scenario": "Exploring a new world",
            "system_prompt": "Stay in character and be helpful.",
            "post_history_instructions": "Always ask follow-up questions.",
            "first_mes": "Hello there!",
            "mes_example": ["Alice: Let's go!", "User: Sure!"],
            "character_book_refs": ["City"],
            "extensions": {"talkativeness": {"value": "medium"}},
        },
        "metadata": {"creator": "tests"},
    }

    json_blob = json.dumps(payload, ensure_ascii=False)
    encoded = base64.b64encode(json_blob.encode("utf-8")).decode("ascii")

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    png.extend(chunk(b"IHDR", ihdr))
    raw = b"\x00\x00\x00\x00\x00"
    png.extend(chunk(b"IDAT", zlib.compress(raw)))
    text_data = b"ccv3" + b"\x00" + encoded.encode("latin-1")
    png.extend(chunk(b"tEXt", text_data))
    png.extend(chunk(b"IEND", b""))

    path.write_bytes(bytes(png))
    return path


def test_load_world_info(sample_world_file: Path) -> None:
    entries = load_world_info(sample_world_file)
    assert len(entries) == 2
    assert entries[0].comment == "City Overview"


def test_filter_world_info(sample_world_file: Path) -> None:
    entries = load_world_info(sample_world_file)
    filtered = filter_world_info(entries, ["forest"])
    assert len(filtered) == 1
    assert filtered[0].comment == "Mystic Forest"


def test_load_character_card(sample_character_file: Path) -> None:
    card = load_character_card(sample_character_file)
    assert card.name == "Alice"
    assert card.description.startswith("Alice is")
    assert card.system_prompt == "Stay in character and be helpful."
    assert card.post_history_instructions == "Always ask follow-up questions."
    assert card.world_info_refs == ["City"]


def test_load_character_card_from_png(sample_character_png: Path) -> None:
    card = load_character_card(sample_character_png)
    assert card.name == "Alice"
    assert card.description.startswith("Alice is")
    assert card.system_prompt == "Stay in character and be helpful."
    assert card.post_history_instructions == "Always ask follow-up questions."
    assert card.world_info_refs == ["City"]
    assert card.extensions.get("talkativeness", {}).get("value") == "medium"


def test_prompt_builder_sections(sample_world_file: Path, sample_character_file: Path) -> None:
    entries = load_world_info(sample_world_file)
    card = load_character_card(sample_character_file)
    context = PromptContext(
        system_prompt="Behave like a helpful guide",
        character=card,
        world_info=entries,
        memories=[MemoryEntry(id="1", content="Met Alice", weight=1.5)],
        history=[ConversationTurn(role="user", content="Hi"), ConversationTurn(role="assistant", content="Hello!")],
        user_message="Where are we headed?",
        persona="An excited traveler eager to learn.",
    )
    builder = PromptBuilder(history_limit=1, memory_limit=1)
    prompt = builder.build_prompt(context)
    assert "Behave like a helpful guide" in prompt
    assert "Stay in character and be helpful." in prompt
    assert "Alice's personality" in prompt
    assert "World Info (↑Char)" in prompt
    assert "World Info (↓Char)" in prompt
    assert "Mystic Forest" in prompt
    assert "Example dialogue" in prompt
    assert "User: Where are we headed?" in prompt


def test_conversation_manager_flow(sample_world_file: Path, sample_character_file: Path) -> None:
    manager = ConversationManager(history_limit=2, memory_limit=1)
    manager.set_world_info(load_world_info(sample_world_file))
    manager.set_character(load_character_card(sample_character_file))
    manager.set_system_prompt("You are helpful")
    manager.add_turn(ConversationTurn(role="user", content="Hello"))
    manager.add_turn(ConversationTurn(role="assistant", content="Hi there"))
    manager.remember(MemoryEntry(id="m1", content="Met user"))
    prompt = manager.build_prompt("What's next?")
    assert "What's next?" in prompt
    messages = list(manager.iter_messages("Tell me more"))
    assert messages[-1]["content"] == "Tell me more"


def test_memory_compressor_prioritises_recent_entries() -> None:
    now = datetime.utcnow()
    memories = [
        MemoryEntry(id="old", content="Old memory", weight=0.2, timestamp=now - timedelta(days=5)),
        MemoryEntry(id="recent", content="Recent memory", weight=0.3, timestamp=now - timedelta(minutes=5)),
        MemoryEntry(id="heavy", content="Important memory", weight=2.0, timestamp=now - timedelta(days=1)),
    ]
    compressor = MemoryCompressor(retention_limit=2, archive_threshold=0.5)
    result = compressor.compress(memories)
    retained_ids = {mem.id for mem in result.retained}
    assert "heavy" in retained_ids
    assert "recent" in retained_ids
    assert result.summary.startswith("Key memories")


def test_conversation_from_transcript() -> None:
    transcript = [("user", "Hi"), ("assistant", "Hello")]
    turns = conversation_from_transcript(transcript)
    assert turns[0].role == "user"
    assert turns[1].content == "Hello"


class DummyCompletions:
    def __init__(self, factory):
        self._factory = factory
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._factory()


class DummyClient:
    def __init__(self, factory):
        self.completions = DummyCompletions(factory)
        self.chat = SimpleNamespace(completions=self.completions)


def _chat_response(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _stream_events(chunks):
    def generator():
        for chunk in chunks:
            delta = SimpleNamespace(content=chunk)
            choice = SimpleNamespace(delta=delta)
            yield SimpleNamespace(choices=[choice])

    return generator


def test_model_interface_complete_aggregates_response() -> None:
    context = PromptContext(history=[ConversationTurn(role="user", content="Hello")])
    config = ModelConfig(model="demo", api_key="key")
    client = DummyClient(lambda: _chat_response("你好"))

    interface = ModelInterface(config, client=client)
    result = interface.complete(context, user_message="在吗？")

    assert result == "你好"
    call = client.chat.completions.calls[0]
    assert call["model"] == "demo"
    assert call["messages"][-1]["role"] == "user"
    assert call["messages"][-1]["content"] == "在吗？"


def test_model_interface_streaming_returns_chunks() -> None:
    context = PromptContext(history=[ConversationTurn(role="user", content="测试")])
    config = ModelConfig(model="demo", api_key="key", stream=True)
    client = DummyClient(_stream_events(["你", "好"]))

    interface = ModelInterface(config, client=client)
    chunks = list(interface.complete_stream(context, user_message="say"))

    assert "".join(chunks) == "你好"
    call = client.chat.completions.calls[0]
    assert call["stream"] is True
