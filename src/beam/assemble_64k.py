"""Assemble exactly 1,000 draft probes from generated BEAM-64K candidates.

This is intentionally a draft builder: it performs structural and provenance
checks, but it does not claim to replace the human validation described in the
BEAM paper.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.beam.generation_settings import CHAT_TOKEN_MINIMUMS, get_token_limit


QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
ANSWER_FIELDS = (
    "answer",
    "ideal_answer",
    "ideal_response",
    "ideal_summary",
    "expected_compliance",
)


def _flatten_dicts(value: Any):
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_dicts(item)


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _source_ids(value: Any) -> list[int]:
    ids = []
    if isinstance(value, bool):
        return ids
    if isinstance(value, int):
        ids.append(value)
    elif isinstance(value, str) and value.strip().isdigit():
        ids.append(int(value.strip()))
    elif isinstance(value, dict):
        for item in value.values():
            ids.extend(_source_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_source_ids(item))
    return ids


def _candidate_score(candidate: dict, question_type: str, valid_ids: set[int]) -> int:
    question = candidate.get("question")
    if not isinstance(question, str) or not question.strip():
        return -1

    score = 10
    rubric = candidate.get("rubric")
    if isinstance(rubric, list) and any(str(item).strip() for item in rubric):
        score += 4
    if any(candidate.get(field) for field in ANSWER_FIELDS):
        score += 3
    if candidate.get("difficulty"):
        score += 1

    ids = _source_ids(candidate.get("source_chat_ids"))
    if ids:
        if all(source_id in valid_ids for source_id in ids):
            score += 4
        else:
            return -1
    elif question_type != "abstention":
        return -1
    return score


def _ensure_draft_rubric(candidate: dict) -> None:
    rubric = candidate.get("rubric")
    if isinstance(rubric, list) and any(str(item).strip() for item in rubric):
        return

    for field in ("key_facts_tested", "ordering_tested", "key_elements_tested", "compliance_indicators"):
        value = candidate.get(field)
        if isinstance(value, list) and value:
            candidate["rubric"] = [f"Response should include: {item}" for item in value]
            return

    for field in ANSWER_FIELDS:
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            candidate["rubric"] = [f"Response should convey: {value.strip()}"]
            return

    raise ValueError(f"Cannot derive a draft rubric for question: {candidate.get('question')}")


def _load_chat_ids(chat_directory: Path) -> set[int]:
    chat_path = (
        chat_directory / "chat_trunecated.json"
        if (chat_directory / "chat_trunecated.json").exists()
        else chat_directory / "chat.json"
    )
    with chat_path.open(encoding="utf-8") as handle:
        chat = json.load(handle)

    ids = set()
    for batch in chat:
        for turn in batch.get("turns", []):
            for message in turn:
                if isinstance(message.get("id"), int):
                    ids.add(message["id"])
    return ids


def _load_candidates(chat_directory: Path, question_type: str) -> list[dict]:
    path = chat_directory / "probing_questions" / question_type / f"{question_type}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate file: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(_flatten_dicts(json.load(handle)))


def _select_two(candidates: list[dict], question_type: str, valid_ids: set[int]):
    ranked = sorted(
        (
            (_candidate_score(candidate, question_type, valid_ids), index, candidate)
            for index, candidate in enumerate(candidates)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    ranked = [item for item in ranked if item[0] >= 0]

    selected = []
    seen_questions = set()
    seen_difficulties = set()
    # Prefer different difficulty levels when quality scores are comparable.
    for prefer_new_difficulty in (True, False):
        for _, _, candidate in ranked:
            normalized = _normalize_question(candidate["question"])
            difficulty = str(candidate.get("difficulty", "unspecified")).lower()
            if normalized in seen_questions:
                continue
            if prefer_new_difficulty and difficulty in seen_difficulties:
                continue
            selected.append(dict(candidate))
            seen_questions.add(normalized)
            seen_difficulties.add(difficulty)
            if len(selected) == 2:
                return selected

    raise ValueError(
        f"Need two structurally valid, distinct {question_type} candidates; found {len(selected)}"
    )


def _read_length_metadata(chat_directory: Path, allow_short: bool) -> dict:
    path = chat_directory / "length_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing length metadata: {path}")
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    measured = metadata.get("measured_tokens")
    minimum = CHAT_TOKEN_MINIMUMS["64K"]
    maximum = get_token_limit("64K")
    if not isinstance(measured, int) or measured > maximum:
        raise ValueError(f"Invalid 64K token measurement in {path}: {measured}")
    if measured < minimum and not allow_short:
        raise ValueError(
            f"Conversation {chat_directory.name} has only {measured} tokens; minimum is {minimum}"
        )
    return metadata


def assemble_dataset(
    chats_directory: Path,
    start_index: int = 0,
    end_index: int = 50,
    allow_short: bool = False,
) -> dict:
    with (chats_directory / "topics.json").open(encoding="utf-8") as handle:
        topics = json.load(handle)
    selected_topics = topics[start_index:end_index]

    manifest = {
        "benchmark": "BEAM-64K draft",
        "validation_status": "needs_human_review",
        "conversation_token_minimum": CHAT_TOKEN_MINIMUMS["64K"],
        "conversation_token_limit": get_token_limit("64K"),
        "questions_per_ability_per_conversation": 2,
        "question_types": list(QUESTION_TYPES),
        "conversations": [],
    }
    total_questions = 0

    for topic in selected_topics:
        conversation_id = str(topic["id"])
        chat_directory = chats_directory / conversation_id
        length_metadata = _read_length_metadata(chat_directory, allow_short)
        valid_ids = _load_chat_ids(chat_directory)
        if not valid_ids:
            raise ValueError(f"No message IDs found for conversation {conversation_id}")

        final_questions = {}
        for question_type in QUESTION_TYPES:
            selected = _select_two(
                _load_candidates(chat_directory, question_type),
                question_type,
                valid_ids,
            )
            for item_index, item in enumerate(selected, start=1):
                _ensure_draft_rubric(item)
                item["question_id"] = f"beam64k_{int(conversation_id):04d}_{question_type}_{item_index:02d}"
                item["validation_status"] = "needs_human_review"
            final_questions[question_type] = selected
            total_questions += len(selected)

        output = chat_directory / "probing_questions" / "probing_questions.draft.json"
        with output.open("w", encoding="utf-8") as handle:
            json.dump(final_questions, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        manifest["conversations"].append(
            {
                "conversation_id": conversation_id,
                "topic": topic["title"],
                "category": topic["category"],
                "measured_tokens": length_metadata["measured_tokens"],
                "question_count": sum(len(items) for items in final_questions.values()),
                "questions_path": str(output.relative_to(chats_directory)),
            }
        )

    expected = len(selected_topics) * len(QUESTION_TYPES) * 2
    if total_questions != expected:
        raise AssertionError(f"Expected {expected} questions, assembled {total_questions}")
    manifest["conversation_count"] = len(selected_topics)
    manifest["question_count"] = total_questions

    output = chats_directory / "benchmark_manifest.draft.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chats-dir", type=Path, default=Path("chats/64K"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=50)
    parser.add_argument("--allow-short", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = assemble_dataset(
        chats_directory=args.chats_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        allow_short=args.allow_short,
    )
    print(json.dumps({
        "conversations": result["conversation_count"],
        "questions": result["question_count"],
        "validation_status": result["validation_status"],
    }, indent=2))
