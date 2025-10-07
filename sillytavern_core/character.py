"""Character card loading utilities aligned with SillyTavern."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .models import CharacterCard

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TEXT_CHUNK = b"tEXt"


def load_character_card(path: Path | str) -> CharacterCard:
    """Load a SillyTavern character card from JSON or PNG metadata."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Character card not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        return CharacterCard.from_dict(data)

    if suffix == ".png":
        data = _load_card_from_png(path.read_bytes())
        return CharacterCard.from_dict(data)

    raise ValueError(f"Unsupported character card format: {path.suffix}")


def _load_card_from_png(payload: bytes) -> Dict[str, object]:
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError("PNG card is missing PNG signature")

    text_chunks = list(_iter_text_chunks(payload))
    if not text_chunks:
        raise ValueError("PNG metadata does not contain any text chunks")

    chunk_map = {keyword.lower(): text for keyword, text in text_chunks}

    encoded = chunk_map.get("ccv3") or chunk_map.get("chara")
    if not encoded:
        raise ValueError("PNG metadata missing SillyTavern character data")

    try:
        json_text = base64.b64decode(encoded).decode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("Failed to decode PNG character metadata") from exc

    return json.loads(json_text)


def _iter_text_chunks(payload: bytes) -> Iterable[Tuple[str, str]]:
    """Yield (keyword, text) pairs from PNG tEXt chunks."""

    offset = len(PNG_SIGNATURE)
    total = len(payload)

    while offset + 8 <= total:
        length = struct.unpack_from(">I", payload, offset)[0]
        offset += 4
        chunk_type = payload[offset : offset + 4]
        offset += 4
        data = payload[offset : offset + length]
        offset += length
        # Skip CRC
        offset += 4

        if chunk_type != _TEXT_CHUNK:
            continue

        try:
            keyword, text = data.split(b"\x00", 1)
        except ValueError:
            continue

        try:
            yield keyword.decode("latin-1"), text.decode("latin-1")
        except UnicodeDecodeError:
            continue


def merge_with_defaults(card: CharacterCard, defaults: Dict[str, str]) -> CharacterCard:
    """Fill missing metadata fields in a character card."""

    merged = CharacterCard(
        name=card.name,
        description=card.description,
        personality=card.personality,
        scenario=card.scenario,
        first_message=card.first_message,
        example_dialogue=list(card.example_dialogue),
        world_info_refs=list(card.world_info_refs),
        metadata={**defaults, **card.metadata},
        creator_notes=card.creator_notes,
        system_prompt=card.system_prompt,
        post_history_instructions=card.post_history_instructions,
        alternate_greetings=list(card.alternate_greetings),
        tags=list(card.tags),
        creator=card.creator,
        character_version=card.character_version,
        extensions=dict(card.extensions),
    )
    return merged
