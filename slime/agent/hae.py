"""Reference HiPER response contract and logical-turn annotations."""

from __future__ import annotations

from collections.abc import Sequence
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


def _token_span(offsets: Sequence[tuple[int, int]], start: int, end: int) -> list[int] | None:
    overlapping = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not overlapping:
        return None
    return [overlapping[0], overlapping[-1] + 1]


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

    pieces = _decode_pieces(tokenizer, output_ids)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    text = "".join(pieces)

    char_spans: dict[str, tuple[int, int]] = {}
    contents: dict[str, str] = {}
    previous_end = 0
    for role in ("switch", "subgoal", "action"):
        opening, closing = _TAGS[role]
        if text.count(opening) != 1 or text.count(closing) != 1:
            return {"multi_turn": _invalid_annotation()}
        start = text.find(opening)
        content_start = start + len(opening)
        close_start = text.find(closing, content_start)
        end = close_start + len(closing)
        if start < previous_end or close_start < content_start:
            return {"multi_turn": _invalid_annotation()}
        char_spans[role] = (start, end)
        contents[role] = text[content_start:close_start].strip()
        previous_end = end

    switch = contents["switch"]
    if switch not in {"KEEP", "SWITCH"}:
        return {"multi_turn": _invalid_annotation()}
    token_spans = {role: _token_span(offsets, start, end) for role, (start, end) in char_spans.items()}
    if any(span is None for span in token_spans.values()):
        return {"multi_turn": _invalid_annotation()}

    subgoal_close = char_spans["subgoal"][1]
    switch_close = char_spans["switch"][1]
    low_position = next((index for index, (start, _) in enumerate(offsets) if start >= subgoal_close), None)
    if low_position is None:
        low_position = next((index for index, (start, _) in enumerate(offsets) if start >= switch_close), 0)
    high_position = 0 if turn_idx == 0 or switch == "SWITCH" else None
    subgoal_span = token_spans["subgoal"]
    annotation = {
        "switch": switch,
        "role_spans": {
            "switch": [token_spans["switch"]],
            "subgoal": [subgoal_span],
            "high_subgoal": [subgoal_span],
            "low_subgoal": [],
            "action": [token_spans["action"]],
        },
        "value_positions": {"high": high_position, "low": low_position},
        "format_valid": True,
    }
    return {"multi_turn": annotation}


__all__ = ["REFERENCE_HAE_PROMPT", "parse_reference_hae_response"]
