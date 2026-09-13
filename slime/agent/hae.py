"""Reference HiPER response contract and logical-turn annotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


REFERENCE_HAE_PROMPT = """Return exactly these three XML fields in order:
<switch>KEEP|SWITCH</switch>
<subgoal>the current high-level subgoal</subgoal>
<action>the next executable action</action>
Use KEEP to continue the prior subgoal and SWITCH to begin a new one.
"""

_TAGS = {
    "switch": ("<switch>", "</switch>"),
    "subgoal": ("<subgoal>", "</subgoal>"),
    "action": ("<action>", "</action>"),
}


@dataclass(frozen=True)
class TaggedField:
    """One exact tagged field mapped to a tag-inclusive token span."""

    content: str
    token_span: tuple[int, int]
    char_end: int


def _decode_pieces(tokenizer: Any, output_ids: Sequence[int]) -> list[str]:
    try:
        return tokenizer.batch_decode(
            [[int(token_id)] for token_id in output_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except (AttributeError, TypeError):
        pieces = []
        for token_id in output_ids:
            try:
                pieces.append(tokenizer.decode([int(token_id)], skip_special_tokens=False))
            except TypeError:
                pieces.append(tokenizer.decode([int(token_id)]))
        return pieces


def _token_span(offsets: Sequence[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    overlapping = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1] + 1


def parse_tagged_fields(
    output_ids: Sequence[int],
    tokenizer: Any,
    tags: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, TaggedField], tuple[tuple[int, int], ...]] | None:
    """Parse exactly one occurrence of each ordered XML-like field."""

    if not output_ids:
        return None
    pieces = _decode_pieces(tokenizer, output_ids)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    text = "".join(pieces)

    fields: dict[str, TaggedField] = {}
    previous_end = 0
    for role, opening, closing in tags:
        if text.count(opening) != 1 or text.count(closing) != 1:
            return None
        start = text.find(opening)
        content_start = start + len(opening)
        close_start = text.find(closing, content_start)
        end = close_start + len(closing)
        if start < previous_end or close_start < content_start:
            return None
        token_span = _token_span(offsets, start, end)
        if token_span is None:
            return None
        fields[role] = TaggedField(
            content=text[content_start:close_start].strip(),
            token_span=token_span,
            char_end=end,
        )
        previous_end = end
    return fields, tuple(offsets)


def token_position_after(offsets: Sequence[tuple[int, int]], char_end: int) -> int | None:
    """Return the first token beginning after a parsed closing tag."""

    return next((index for index, (start, _) in enumerate(offsets) if start >= char_end), None)


def _invalid_annotation() -> dict[str, Any]:
    return {
        "switch": None,
        "role_spans": {
            "switch": [],
            "subgoal": [],
            "high_subgoal": [],
            "low_subgoal": [],
            "action": [],
        },
        "value_positions": {"high": None, "low": None},
        "format_valid": False,
    }


def parse_reference_hae_response(
    output_ids: Sequence[int],
    tokenizer: Any,
    *,
    turn_idx: int,
) -> dict[str, Any]:
    """Parse tag-inclusive reference HiPER roles into trajectory metadata.

    Character matches are mapped back to complete overlapping token spans so
    tags remain actor-trained even when a tokenizer splits their delimiters.
    Malformed output returns an explicit fail-closed annotation.
    """

    if isinstance(turn_idx, bool) or not isinstance(turn_idx, int) or turn_idx < 0 or not output_ids:
        return {"multi_turn": _invalid_annotation()}

    parsed = parse_tagged_fields(
        output_ids,
        tokenizer,
        [(role, *_TAGS[role]) for role in ("switch", "subgoal", "action")],
    )
    if parsed is None:
        return {"multi_turn": _invalid_annotation()}
    fields, offsets = parsed
    switch = fields["switch"].content
    if switch not in {"KEEP", "SWITCH"}:
        return {"multi_turn": _invalid_annotation()}

    low_position = token_position_after(offsets, fields["subgoal"].char_end)
    if low_position is None:
        low_position = token_position_after(offsets, fields["switch"].char_end) or 0
    high_position = 0 if turn_idx == 0 or switch == "SWITCH" else None
    subgoal_span = list(fields["subgoal"].token_span)
    annotation = {
        "switch": switch,
        "role_spans": {
            "switch": [list(fields["switch"].token_span)],
            "subgoal": [subgoal_span],
            "high_subgoal": [subgoal_span],
            "low_subgoal": [],
            "action": [list(fields["action"].token_span)],
        },
        "value_positions": {"high": high_position, "low": low_position},
        "format_valid": True,
    }
    return {"multi_turn": annotation}


__all__ = [
    "REFERENCE_HAE_PROMPT",
    "TaggedField",
    "parse_reference_hae_response",
    "parse_tagged_fields",
    "token_position_after",
]
