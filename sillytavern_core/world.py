"""Utilities for loading world information datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from .models import WorldInfo

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fall back when yaml isn't available
    yaml = None


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"World info file not found: {path}")
    return path.read_text(encoding="utf-8")


def _normalize_entry_container(data: Any) -> Iterable[Mapping[str, Any]]:
    """Extract the iterable of world info dicts from a payload."""

    if isinstance(data, Mapping):
        entries = data.get("entries")
        if entries is None:
            entries = data.get("world_info")
        if entries is None:
            entries = data
    else:
        entries = data

    if isinstance(entries, Mapping):
        return entries.values()
    if isinstance(entries, Iterable):
        return entries
    raise ValueError("World info file must contain an iterable of entries")


def _parse_world_records(data: Iterable[Mapping[str, Any]]) -> List[WorldInfo]:
    return [WorldInfo.from_dict(item) for item in data]


def load_world_info(path: Path | str) -> List[WorldInfo]:
    """Load world information entries from JSON or YAML files."""

    path = Path(path)
    raw = _read_text(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML world info files")
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    records = _parse_world_records(_normalize_entry_container(data))
    records.sort(key=lambda entry: (entry.display_index if entry.display_index is not None else entry.uid))
    return records


def filter_world_info(entries: Sequence[WorldInfo], tags: Sequence[str] | None = None) -> List[WorldInfo]:
    """Filter world info entries by tags."""

    if not tags:
        return list(entries)
    wanted = {tag.lower() for tag in tags}

    def _entry_terms(entry: WorldInfo) -> set[str]:
        terms = [*entry.key, *entry.keysecondary]
        if entry.comment:
            terms.append(entry.comment)
        return {term.lower() for term in terms}

    return [entry for entry in entries if wanted & _entry_terms(entry)]
