"""Prompt assembly utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence

from .models import CharacterCard, ConversationTurn, MemoryEntry, PromptContext, WorldInfo

DEFAULT_STORY_TEMPLATE = (
    "{{#if system}}{{system}}\n{{/if}}"
    "{{#if description}}{{description}}\n{{/if}}"
    "{{#if personality}}{{char}}'s personality: {{personality}}\n{{/if}}"
    "{{#if scenario}}Scenario: {{scenario}}\n{{/if}}"
    "{{#if persona}}{{persona}}\n{{/if}}"
)

WORLD_INFO_POSITION_BEFORE = 0
WORLD_INFO_POSITION_AFTER = 1


@dataclass
class PromptSections:
    story: List[str] = field(default_factory=list)
    world_before: List[str] = field(default_factory=list)
    world_after: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    story_in_chat: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    post_history: List[str] = field(default_factory=list)

    def join(self, separator: str = "\n\n") -> str:
        segments: List[str] = []
        for block in [
            self.story,
            self.world_before,
            self.world_after,
            self.memory,
            self.examples,
            self.story_in_chat,
            self.history,
            self.post_history,
        ]:
            for section in block:
                if section:
                    segments.append(section)
        return separator.join(segments)


class PromptBuilder:
    """Builds prompts from structured context."""

    def __init__(self, history_limit: int = 12, memory_limit: int = 5) -> None:
        self.history_limit = history_limit
        self.memory_limit = memory_limit

    # ------------------------------------------------------------------
    # High level assembly
    # ------------------------------------------------------------------
    def build_sections(self, context: PromptContext) -> PromptSections:
        sections = PromptSections()
        card = context.character
        assistant_name = context.character_name or (card.name if card else "Assistant")
        user_name = context.user_name or "User"

        before_entries, after_entries = self._categorise_world_info(context.world_info)
        formatted_before = self._format_world_entries(before_entries)
        formatted_after = self._format_world_entries(after_entries)

        story_text = self._render_story_string(
            context,
            assistant_name=assistant_name,
            user_name=user_name,
            world_before="\n\n".join(formatted_before),
            world_after="\n\n".join(formatted_after),
            examples=self._join_examples(card),
        )

        if story_text:
            if context.story_in_chat:
                sections.story_in_chat.append(
                    f"System: {story_text.strip()}"
                )
            else:
                sections.story.append(story_text)

        world_before_block = self._build_world_block(formatted_before, "World Info (↑Char)")
        if world_before_block:
            sections.world_before.append(world_before_block)

        world_after_block = self._build_world_block(formatted_after, "World Info (↓Char)")
        if world_after_block:
            sections.world_after.append(world_after_block)

        memory_block = self._build_memory_block(context.memories)
        if memory_block:
            sections.memory.append(memory_block)

        examples_block = self._build_examples_block(card)
        if examples_block:
            sections.examples.append(examples_block)

        history_block = self._build_history_block(context.history, user_name, assistant_name)
        if history_block:
            sections.history.append(history_block)

        post_history_block = self._build_post_history_block(card)
        if post_history_block:
            sections.post_history.append(post_history_block)

        return sections

    def build_prompt(self, context: PromptContext) -> str:
        sections = self.build_sections(context)
        prompt = sections.join()
        if context.user_message:
            user_line = f"{context.user_name or 'User'}: {context.user_message}"
            prompt = f"{prompt}\n\n{user_line}" if prompt else user_line
        return prompt

    def iter_messages(self, context: PromptContext) -> Iterable[dict]:
        """Yield chat-completion compatible messages."""

        card = context.character
        assistant_name = context.character_name or (card.name if card else "assistant")
        user_name = context.user_name or "user"

        before_entries, after_entries = self._categorise_world_info(context.world_info)
        formatted_before = self._format_world_entries(before_entries)
        formatted_after = self._format_world_entries(after_entries)

        story_text = self._render_story_string(
            context,
            assistant_name=assistant_name,
            user_name=user_name,
            world_before="\n\n".join(formatted_before),
            world_after="\n\n".join(formatted_after),
            examples=self._join_examples(card),
        ).strip()

        if story_text and not context.story_in_chat:
            yield {"role": "system", "content": story_text}

        if formatted_before:
            yield {
                "role": "system",
                "content": self._build_world_block(formatted_before, "World Info (↑Char)").strip(),
            }

        if formatted_after:
            yield {
                "role": "system",
                "content": self._build_world_block(formatted_after, "World Info (↓Char)").strip(),
            }

        for memory in context.memories[-self.memory_limit :]:
            yield {"role": "system", "content": f"Memory: {memory.content}"}

        if card and card.example_dialogue:
            yield {"role": "system", "content": self._build_examples_block(card).strip()}

        if story_text and context.story_in_chat:
            yield {"role": "system", "content": story_text}

        for turn in context.history[-self.history_limit :]:
            yield {"role": turn.role, "content": turn.content}

        if card and card.post_history_instructions:
            yield {"role": "system", "content": card.post_history_instructions.strip()}

        if context.user_message:
            yield {"role": "user", "content": context.user_message}

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    def _render_story_string(
        self,
        context: PromptContext,
        *,
        assistant_name: str,
        user_name: str,
        world_before: str,
        world_after: str,
        examples: str,
    ) -> str:
        template = context.story_template or DEFAULT_STORY_TEMPLATE
        card = context.character

        system_prompt = self._combine_system_prompts(context)
        description = card.description if card else ""
        personality = card.personality if card else ""
        scenario = card.scenario if card else ""

        params = {
            "system": system_prompt,
            "description": description,
            "personality": personality,
            "scenario": scenario,
            "persona": context.persona or "",
            "char": assistant_name,
            "user": user_name,
            "wiBefore": world_before,
            "wiAfter": world_after,
            "loreBefore": world_before,
            "loreAfter": world_after,
            "anchorBefore": "",
            "anchorAfter": "",
            "mesExamples": examples,
            "mesExamplesRaw": examples,
        }

        rendered = _render_template(template, params)
        rendered = rendered.replace("\r", "")
        rendered = rendered.lstrip("\n")
        if rendered and not rendered.endswith("\n") and not context.story_in_chat:
            rendered += "\n"
        return rendered

    def _build_world_block(self, entries: Sequence[str], heading: str) -> str:
        if not entries:
            return ""
        body = "\n\n".join(entries)
        return f"{heading}:\n{body}" if heading else body

    def _build_memory_block(self, memories: Sequence[MemoryEntry]) -> str:
        if not memories:
            return ""
        trimmed = list(memories)[-self.memory_limit :]
        lines = [f"- {mem.content} (weight: {mem.weight:.2f})" for mem in trimmed]
        return "Memories:\n" + "\n".join(lines)

    def _build_examples_block(self, card: CharacterCard | None) -> str:
        if not card or not card.example_dialogue:
            return ""
        lines = ["Example dialogue:"] + card.example_dialogue
        return "\n".join(lines)

    def _build_history_block(
        self, history: Sequence[ConversationTurn], user_name: str, assistant_name: str
    ) -> str:
        trimmed = list(history)[-self.history_limit :]
        if not trimmed:
            return ""
        lines = [
            f"{self._display_name_for_role(turn.role, user_name, assistant_name)}: {turn.content}"
            for turn in trimmed
        ]
        return "Conversation:\n" + "\n".join(lines)

    def _build_post_history_block(self, card: CharacterCard | None) -> str:
        if not card or not card.post_history_instructions:
            return ""
        text = card.post_history_instructions.strip()
        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _combine_system_prompts(self, context: PromptContext) -> str:
        card_prompt = context.character.system_prompt if context.character else ""
        system_prompt = context.system_prompt or ""
        prompts = [prompt for prompt in [system_prompt, card_prompt] if prompt]
        return "\n".join(prompts)

    def _join_examples(self, card: CharacterCard | None) -> str:
        if not card or not card.example_dialogue:
            return ""
        return "\n".join(card.example_dialogue)

    def _categorise_world_info(self, entries: Sequence[WorldInfo]) -> tuple[List[WorldInfo], List[WorldInfo]]:
        before: List[WorldInfo] = []
        after: List[WorldInfo] = []
        for entry in entries:
            if entry.position == WORLD_INFO_POSITION_BEFORE:
                before.append(entry)
            elif entry.position == WORLD_INFO_POSITION_AFTER:
                after.append(entry)
            else:
                after.append(entry)
        return before, after

    def _format_world_entries(self, entries: Sequence[WorldInfo]) -> List[str]:
        formatted: List[str] = []
        for entry in entries:
            title = self._world_entry_title(entry)
            block = f"[{title}]\n{entry.content.strip()}" if entry.content else f"[{title}]"
            formatted.append(block)
        return formatted

    @staticmethod
    def _world_entry_title(entry: WorldInfo) -> str:
        if entry.comment:
            return entry.comment
        if entry.key:
            return ", ".join(entry.key)
        return f"Entry {entry.uid}"

    @staticmethod
    def _display_name_for_role(role: str, user_name: str, assistant_name: str) -> str:
        role_lower = role.lower()
        if role_lower == "user":
            return user_name
        if role_lower in {"assistant", "bot"}:
            return assistant_name
        if role_lower == "system":
            return "System"
        return role


def _render_template(template: str, params: Mapping[str, str]) -> str:
    """Render a limited Handlebars-like template supporting {{#if}} sections."""

    def replace_section(match: re.Match[str]) -> str:
        key = match.group(1)
        body = match.group(2)
        value = params.get(key)
        if value:
            return _render_template(body, params)
        return ""

    section_pattern = re.compile(r"{{#if\s+([\w]+)}}(.*?){{/if}}", re.DOTALL)
    rendered = section_pattern.sub(replace_section, template)

    def replace_var(match: re.Match[str]) -> str:
        key = match.group(1)
        return params.get(key, "")

    variable_pattern = re.compile(r"{{([\w]+)}}")
    return variable_pattern.sub(replace_var, rendered)


def merge_world_info(base: Sequence[WorldInfo], additional: Sequence[WorldInfo]) -> List[WorldInfo]:
    """Merge world info entries ensuring unique identifiers."""

    merged = {entry.uid: entry for entry in base}
    for entry in additional:
        merged[entry.uid] = entry
    return list(merged.values())


def limit_history(history: Sequence[ConversationTurn], limit: int) -> List[ConversationTurn]:
    """Helper for trimming conversation history."""

    if limit <= 0:
        return []
    return list(history)[-limit:]


def limit_memories(memories: Sequence[MemoryEntry], limit: int) -> List[MemoryEntry]:
    if limit <= 0:
        return []
    return list(memories)[-limit:]
