"""Lightweight dataclasses for the SillyTavern backend toolkit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_dict(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    raise TypeError(f"Expected mapping for metadata, received {type(value)!r}")


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Invalid datetime string: {value!r}") from exc
    raise TypeError(f"Unsupported datetime value: {value!r}")


@dataclass
class WorldInfo:
    """Represents a lorebook entry matching SillyTavern's world info schema."""

    uid: int
    key: List[str] = field(default_factory=list)
    keysecondary: List[str] = field(default_factory=list)
    comment: str = ""
    content: str = ""
    constant: bool = False
    selective: bool = True
    selective_logic: Optional[int] = None
    add_memo: bool = False
    order: int = 0
    position: int = 0
    disable: bool = False
    ignore_budget: bool = False
    display_index: Optional[int] = None
    group: str = ""
    group_override: bool = False
    group_weight: int = 0
    sticky: Optional[int] = None
    cooldown: Optional[int] = None
    delay: Optional[int] = None
    probability: Optional[float] = None
    depth: Optional[int] = None
    use_probability: Optional[bool] = None
    role: Optional[int] = None
    vectorized: bool = False
    exclude_recursion: bool = False
    prevent_recursion: bool = False
    delay_until_recursion: Optional[int] = None
    scan_depth: Optional[int] = None
    case_sensitive: Optional[bool] = None
    match_whole_words: Optional[bool] = None
    use_group_scoring: Optional[bool] = None
    automation_id: str = ""
    match_persona_description: bool = False
    match_character_description: bool = False
    match_character_personality: bool = False
    match_character_depth_prompt: bool = False
    match_scenario: bool = False
    match_creator_notes: bool = False
    triggers: List[str] = field(default_factory=list)
    character_filter_names: List[str] = field(default_factory=list)
    character_filter_tags: List[str] = field(default_factory=list)
    character_filter_exclude: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldInfo":
        def _optional_bool(value: Any) -> Optional[bool]:
            if value is None:
                return None
            return bool(value)

        def _optional_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            return int(value)

        uid = data.get("uid", data.get("id"))
        if uid is None:
            raise KeyError("World info entry must contain a 'uid' or 'id' field")

        return cls(
            uid=int(uid),
            key=_coerce_list(data.get("key") or data.get("keys")),
            keysecondary=_coerce_list(data.get("keysecondary") or data.get("secondary_keys")),
            comment=str(data.get("comment", "")),
            content=str(data.get("content", "")),
            constant=bool(data.get("constant", False)),
            selective=bool(data.get("selective", True)),
            selective_logic=_optional_int(data.get("selectiveLogic")),
            add_memo=bool(data.get("addMemo", False)),
            order=int(data.get("order", 0)),
            position=int(data.get("position", 0)),
            disable=bool(data.get("disable", False)),
            ignore_budget=bool(data.get("ignoreBudget", False)),
            display_index=_optional_int(data.get("displayIndex")),
            group=str(data.get("group", "")),
            group_override=bool(data.get("groupOverride", False)),
            group_weight=int(data.get("groupWeight", 0)),
            sticky=_optional_int(data.get("sticky")),
            cooldown=_optional_int(data.get("cooldown")),
            delay=_optional_int(data.get("delay")),
            probability=(
                float(data["probability"]) if "probability" in data and data["probability"] is not None else None
            ),
            depth=_optional_int(data.get("depth")),
            use_probability=_optional_bool(data.get("useProbability")),
            role=_optional_int(data.get("role")),
            vectorized=bool(data.get("vectorized", False)),
            exclude_recursion=bool(data.get("excludeRecursion", False)),
            prevent_recursion=bool(data.get("preventRecursion", False)),
            delay_until_recursion=_optional_int(data.get("delayUntilRecursion")),
            scan_depth=_optional_int(data.get("scanDepth")),
            case_sensitive=_optional_bool(data.get("caseSensitive")),
            match_whole_words=_optional_bool(data.get("matchWholeWords")),
            use_group_scoring=_optional_bool(data.get("useGroupScoring")),
            automation_id=str(data.get("automationId", "")),
            match_persona_description=bool(data.get("matchPersonaDescription", False)),
            match_character_description=bool(data.get("matchCharacterDescription", False)),
            match_character_personality=bool(data.get("matchCharacterPersonality", False)),
            match_character_depth_prompt=bool(data.get("matchCharacterDepthPrompt", False)),
            match_scenario=bool(data.get("matchScenario", False)),
            match_creator_notes=bool(data.get("matchCreatorNotes", False)),
            triggers=_coerce_list(data.get("triggers")),
            character_filter_names=_coerce_list(data.get("characterFilterNames")),
            character_filter_tags=_coerce_list(data.get("characterFilterTags")),
            character_filter_exclude=bool(data.get("characterFilterExclude", False)),
        )


@dataclass
class CharacterCard:
    """Represents a SillyTavern character card."""

    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_message: str = ""
    example_dialogue: List[str] = field(default_factory=list)
    creator_notes: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    creator: str = ""
    character_version: str = ""
    world_info_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CharacterCard":
        """Create a card from either a raw ST card or a simplified mapping."""

        if "data" in data and isinstance(data["data"], Mapping):
            payload = data["data"]
            return cls(
                name=str(payload.get("name", data.get("name", ""))),
                description=str(payload.get("description", "")),
                personality=str(payload.get("personality", "")),
                scenario=str(payload.get("scenario", "")),
                first_message=str(payload.get("first_mes", "")),
                example_dialogue=_coerce_list(_split_example_dialogue(payload.get("mes_example"))),
                creator_notes=str(payload.get("creator_notes", "")),
                system_prompt=str(payload.get("system_prompt", "")),
                post_history_instructions=str(payload.get("post_history_instructions", "")),
                alternate_greetings=_coerce_list(payload.get("alternate_greetings")),
                tags=_coerce_list(payload.get("tags")),
                creator=str(payload.get("creator", "")),
                character_version=str(payload.get("character_version", "")),
                world_info_refs=_coerce_list(payload.get("character_book_refs")),
                metadata=_coerce_dict(data.get("metadata")),
                extensions=_coerce_extensions(payload.get("extensions")),
            )

        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            personality=str(data.get("personality", "")),
            scenario=str(data.get("scenario", "")),
            first_message=str(data.get("first_message", data.get("first_mes", ""))),
            example_dialogue=_coerce_list(
                _split_example_dialogue(data.get("example_dialogue", data.get("mes_example")))
            ),
            creator_notes=str(data.get("creator_notes", "")),
            system_prompt=str(data.get("system_prompt", "")),
            post_history_instructions=str(data.get("post_history_instructions", "")),
            alternate_greetings=_coerce_list(data.get("alternate_greetings")),
            tags=_coerce_list(data.get("tags")),
            creator=str(data.get("creator", "")),
            character_version=str(data.get("character_version", "")),
            world_info_refs=_coerce_list(data.get("world_info_refs")),
            metadata=_coerce_dict(data.get("metadata")),
            extensions=_coerce_extensions(data.get("extensions")),
        )


def _split_example_dialogue(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    if not text:
        return []
    return [segment.strip() for segment in text.split("\n") if segment.strip()]


def _coerce_extensions(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping for extensions, received {type(value)!r}")


@dataclass
class MemoryEntry:
    id: str
    content: str
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryEntry":
        timestamp = data.get("timestamp")
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            weight=float(data.get("weight", 1.0)),
            timestamp=_coerce_datetime(timestamp) if timestamp else datetime.utcnow(),
            metadata=_coerce_dict(data.get("metadata")),
        )

    def copy_with(self, **updates: Any) -> "MemoryEntry":
        return replace(self, **updates)


@dataclass
class ConversationTurn:
    role: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationTurn":
        return cls(role=str(data["role"]), content=str(data["content"]), metadata=_coerce_dict(data.get("metadata")))


@dataclass
class PromptContext:
    system_prompt: Optional[str] = None
    character: Optional[CharacterCard] = None
    world_info: List[WorldInfo] = field(default_factory=list)
    memories: List[MemoryEntry] = field(default_factory=list)
    history: List[ConversationTurn] = field(default_factory=list)
    user_message: Optional[str] = None
    persona: Optional[str] = None
    user_name: str = "User"
    character_name: Optional[str] = None
    story_template: Optional[str] = None
    story_in_chat: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "character": asdict(self.character) if self.character else None,
            "world_info": [asdict(entry) for entry in self.world_info],
            "memories": [asdict(memory) for memory in self.memories],
            "history": [asdict(turn) for turn in self.history],
            "user_message": self.user_message,
            "persona": self.persona,
            "user_name": self.user_name,
            "character_name": self.character_name,
            "story_template": self.story_template,
            "story_in_chat": self.story_in_chat,
        }
