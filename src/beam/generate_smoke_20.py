"""Generate and validate one compact, 20-row BEAM-style Gemini smoke dataset.

This command intentionally uses Google's native ``generateContent`` REST API
and the ``x-goog-api-key`` header. It makes exactly one API request and has no
automatic retry, so a failed run cannot silently spend additional quota.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_OUTPUT = Path("outputs/smoke_20/gemini_beam_smoke_20.json")
HF_BEAM_FIELDS = (
    "conversation_id",
    "conversation_seed",
    "narratives",
    "user_profile",
    "conversation_plan",
    "user_questions",
    "chat",
    "probing_questions",
)
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

HF_ANSWER_FIELDS = {
    "abstention": "ideal_response",
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "information_extraction": "answer",
    "instruction_following": "expected_compliance",
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "preference_following": "expected_compliance",
    "summarization": "ideal_summary",
    "temporal_reasoning": "answer",
}


SMOKE_PROMPT = """Create one compact, wholly synthetic BEAM-style benchmark smoke dataset.

Return one JSON object with exactly three top-level fields: metadata,
conversation, and rows.

The metadata object must contain:
- dataset: "BEAM-style smoke test"
- conversation_id: "smoke_0001"
- requested_rows: 20
- generation_note: "Synthetic smoke-test data; not an official BEAM release."

The conversation must contain 26-34 coherent messages across at least five
sessions about planning a community science fair. Every message must contain a
unique sequential message_id (m001 onward), session_id, ISO-8601 timestamp,
role, and content. Include named people and responsibilities, dates and times,
a venue, a budget, participant preferences, a standing formatting instruction,
a plan that is later changed, an explicit contradiction that is later
resolved, and events spanning multiple sessions.

Create exactly 20 rows: exactly two rows for each of these exact ability labels:
- abstention
- contradiction_resolution
- event_ordering
- information_extraction
- instruction_following
- knowledge_update
- multi_session_reasoning
- preference_following
- summarization
- temporal_reasoning

Every row must contain:
- row_id, sequentially numbered smoke_0001_q01 through smoke_0001_q20
- conversation_id: "smoke_0001"
- ability: one of the exact labels above
- difficulty: easy, medium, or hard
- question
- reference_answer
- source_message_ids: a JSON array of existing message IDs
- rubric
- validation_status: "machine_generated_smoke_test"

Each non-abstention answer must be unambiguously grounded in the cited
messages. Each abstention question must request a genuinely absent fact; its
answer must explicitly state that the conversation does not provide it, while
its cited messages establish the nearby topic. Questions must not mention
message IDs, ability labels, or benchmark mechanics. Answers must be concise
and directly gradeable. Rubrics must identify the indispensable facts. Keep
dates internally consistent except for the deliberate contradiction that is
explicitly resolved later. Do not copy released BEAM examples.
"""


def _request_payload(
    max_output_tokens: int,
    temperature: float,
    prompt: str = SMOKE_PROMPT,
) -> dict[str, Any]:
    return {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You generate strict JSON benchmark data. Return valid "
                        "JSON only, without Markdown fences."
                    )
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }


def _call_gemini(
    api_key: str,
    model: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
    prompt: str = SMOKE_PROMPT,
) -> dict[str, Any]:
    encoded_model = urllib.parse.quote(model, safe="-._")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model}:generateContent"
    )
    body = json.dumps(
        _request_payload(max_output_tokens, temperature, prompt=prompt),
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "x-goog-api-client": "beam-64k-smoke/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini API returned HTTP {exc.code}: {response_text[:2000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the Gemini API: {exc.reason}") from exc


def _candidate_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = response.get("promptFeedback", "not provided")
        raise ValueError(f"Gemini returned no candidates. Prompt feedback: {feedback}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    text = "".join(text_parts).strip()
    if not text:
        raise ValueError("Gemini returned a candidate without text content")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text


def validate_dataset(
    dataset: Any, expected_conversation_id: str = "smoke_0001"
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(dataset, dict):
        raise ValueError("Generated content must be a JSON object")

    metadata = dataset.get("metadata")
    conversation = dataset.get("conversation")
    rows = dataset.get("rows")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}
    if not isinstance(conversation, list):
        errors.append("conversation must be an array")
        conversation = []
    if not isinstance(rows, list):
        errors.append("rows must be an array")
        rows = []

    if metadata.get("conversation_id") != expected_conversation_id:
        errors.append(
            f"metadata.conversation_id must be {expected_conversation_id}"
        )
    if metadata.get("requested_rows") != 20:
        errors.append("metadata.requested_rows must be 20")
    if not 26 <= len(conversation) <= 34:
        errors.append(f"conversation must contain 26-34 messages; found {len(conversation)}")

    message_ids: list[str] = []
    session_ids: set[str] = set()
    for index, message in enumerate(conversation, start=1):
        if not isinstance(message, dict):
            errors.append(f"conversation item {index} is not an object")
            continue
        required = ("message_id", "session_id", "timestamp", "role", "content")
        missing = [field for field in required if not message.get(field)]
        if missing:
            errors.append(f"conversation item {index} is missing: {', '.join(missing)}")
        message_id = message.get("message_id")
        if isinstance(message_id, str):
            message_ids.append(message_id)
        session_id = message.get("session_id")
        if isinstance(session_id, str) and session_id:
            session_ids.add(session_id)

    expected_message_ids = [f"m{index:03d}" for index in range(1, len(conversation) + 1)]
    if message_ids != expected_message_ids:
        errors.append("message IDs must be unique and sequential from m001")
    if len(session_ids) < 5:
        errors.append(f"conversation must span at least five sessions; found {len(session_ids)}")

    if len(rows) != 20:
        errors.append(f"rows must contain exactly 20 items; found {len(rows)}")

    valid_message_ids = set(message_ids)
    row_ids: list[str] = []
    abilities: list[str] = []
    required_row_fields = (
        "row_id",
        "conversation_id",
        "ability",
        "difficulty",
        "question",
        "reference_answer",
        "source_message_ids",
        "rubric",
        "validation_status",
    )
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"row {index} is not an object")
            continue
        missing = [field for field in required_row_fields if not row.get(field)]
        if missing:
            errors.append(f"row {index} is missing: {', '.join(missing)}")
        row_id = row.get("row_id")
        if isinstance(row_id, str):
            row_ids.append(row_id)
        ability = row.get("ability")
        if isinstance(ability, str):
            abilities.append(ability)
        if row.get("conversation_id") != expected_conversation_id:
            errors.append(f"row {index} has the wrong conversation_id")
        if row.get("difficulty") not in {"easy", "medium", "hard"}:
            errors.append(f"row {index} has an invalid difficulty")
        if row.get("validation_status") != "machine_generated_smoke_test":
            errors.append(f"row {index} has an invalid validation_status")
        source_ids = row.get("source_message_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"row {index} must cite at least one source message")
        elif any(source_id not in valid_message_ids for source_id in source_ids):
            errors.append(f"row {index} cites a missing source message")

    expected_row_ids = [
        f"{expected_conversation_id}_q{index:02d}" for index in range(1, 21)
    ]
    if row_ids != expected_row_ids:
        errors.append(
            "row IDs must be unique and sequential from "
            f"{expected_conversation_id}_q01"
        )
    ability_counts = Counter(abilities)
    if set(ability_counts) != set(QUESTION_TYPES):
        errors.append("rows must use exactly the ten required ability labels")
    for ability in QUESTION_TYPES:
        if ability_counts[ability] != 2:
            errors.append(
                f"ability {ability} must have exactly two rows; found {ability_counts[ability]}"
            )

    normalized_questions = [
        re.sub(r"\s+", " ", str(row.get("question", ""))).strip().lower()
        for row in rows
        if isinstance(row, dict)
    ]
    if len(normalized_questions) != len(set(normalized_questions)):
        errors.append("all questions must be distinct")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    return {
        "messages": len(conversation),
        "sessions": len(session_ids),
        "rows": len(rows),
        "ability_counts": dict(sorted(ability_counts.items())),
    }


def _numeric_message_id(message_id: str) -> int:
    match = re.fullmatch(r"m(\d+)", message_id)
    if not match:
        raise ValueError(f"Cannot convert message ID to BEAM chat ID: {message_id}")
    return int(match.group(1)) - 1


def _rubric_items(rubric: Any) -> list[str]:
    if isinstance(rubric, list):
        return [str(item).strip() for item in rubric if str(item).strip()]
    if isinstance(rubric, str) and rubric.strip():
        return [rubric.strip()]
    return []


def _conversation_references(source_ids: list[int]) -> list[str]:
    return [f"chat_id: {source_id}" for source_id in source_ids]


def _to_hf_probe(
    row: dict[str, Any], messages_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ability = row["ability"]
    source_message_ids = row["source_message_ids"]
    source_chat_ids = [
        _numeric_message_id(message_id) for message_id in source_message_ids
    ]
    source_sessions = list(
        dict.fromkeys(
            messages_by_id[message_id]["session_id"]
            for message_id in source_message_ids
        )
    )
    rubric = _rubric_items(row["rubric"])
    reference_answer = row["reference_answer"]

    probe: dict[str, Any] = {
        "question_id": row["row_id"],
        "question": row["question"],
        HF_ANSWER_FIELDS[ability]: reference_answer,
        "difficulty": row["difficulty"],
    }

    if ability == "abstention":
        probe.update(
            {
                "abstention_type": "missing_detail",
                "why_unanswerable": reference_answer,
                "plan_reference": ", ".join(_conversation_references(source_chat_ids)),
            }
        )
    elif ability == "contradiction_resolution":
        probe.update(
            {
                "contradiction_type": "resolved_in_conversation",
                "topic_questioned": row["question"],
                "conversation_references": _conversation_references(source_chat_ids),
                "tests_for": rubric[0] if rubric else "Resolve the conflicting statements.",
            }
        )
    elif ability == "event_ordering":
        probe.update(
            {
                "ordering_type": "chronological_sequence",
                "total_mentions": len(source_chat_ids),
                "conversation_references": _conversation_references(source_chat_ids),
                "ordering_tested": rubric,
            }
        )
    elif ability == "information_extraction":
        probe.update(
            {
                "question_type": "conversation_fact",
                "conversation_reference": ", ".join(_conversation_references(source_chat_ids)),
                "key_facts_tested": rubric,
            }
        )
    elif ability == "instruction_following":
        probe.update(
            {
                "instruction_being_tested": rubric[0] if rubric else reference_answer,
                "compliance_indicators": rubric,
                "non_compliance_signs": ["The response does not satisfy the stated instruction."],
                "instruction_type": "persistent_instruction",
            }
        )
    elif ability == "knowledge_update":
        probe.update(
            {
                "update_type": "conversation_update",
                "tests_retention_of": reference_answer,
                "conversation_references": _conversation_references(source_chat_ids),
                "potential_confusion": (
                    "Earlier information may be recalled instead of the later update."
                ),
            }
        )
    elif ability == "multi_session_reasoning":
        probe.update(
            {
                "reasoning_type": "cross_session_facts",
                "sessions_required": len(source_sessions),
                "conversation_references": _conversation_references(source_chat_ids),
                "reasoning_steps": [
                    f"Use the evidence in chat_id {source_id}."
                    for source_id in source_chat_ids
                ],
            }
        )
    elif ability == "preference_following":
        probe.update(
            {
                "preference_being_tested": rubric[0] if rubric else reference_answer,
                "compliance_indicators": rubric,
                "non_compliance_signs": ["The response ignores the stated preference."],
                "preference_type": "stated_preference",
            }
        )
    elif ability == "summarization":
        probe.update(
            {
                "summarization_type": "conversation_summary",
                "bullet_points_covered": len(source_chat_ids),
                "conversation_sessions": source_sessions,
                "key_elements_tested": rubric,
                "synthesis_required": reference_answer,
            }
        )
    elif ability == "temporal_reasoning":
        probe.update(
            {
                "temporal_type": "time_relation",
                "time_points": [
                    messages_by_id[message_id]["timestamp"]
                    for message_id in source_message_ids
                ],
                "conversation_references": _conversation_references(source_chat_ids),
                "calculation_required": rubric[0] if rubric else reference_answer,
            }
        )

    probe["source_chat_ids"] = source_chat_ids
    probe["rubric"] = rubric
    probe["validation_status"] = row["validation_status"]
    return probe


def _hf_conversation_seed(conversation_seed: dict[str, Any] | None) -> dict[str, Any]:
    source = conversation_seed or {
        "category": "Event Planning",
        "id": 1,
        "subtopics": [
            "Community science fair planning",
            "Venue and capacity management",
            "Budget allocation and updates",
            "Scheduling and logistics",
            "Participant preferences and accommodations",
        ],
        "theme": "Coordinating a community science fair across multiple sessions",
        "title": "Planning a Community Science Fair",
    }
    return {
        "category": str(source["category"]),
        "id": int(source["id"]),
        "subtopics": [str(item) for item in source.get("subtopics", [])],
        "theme": str(source["theme"]),
        "title": str(source["title"]),
    }


def to_hf_beam_record(
    dataset: dict[str, Any], conversation_seed: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convert the smoke dataset to Mohammadta/BEAM's one-row column schema."""

    conversation_id = dataset.get("metadata", {}).get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("metadata.conversation_id must be a non-empty string")
    validate_dataset(dataset, expected_conversation_id=conversation_id)
    conversation = dataset["conversation"]
    messages_by_id = {message["message_id"]: message for message in conversation}
    seed = _hf_conversation_seed(conversation_seed)

    session_batches = [
        (session_id, list(messages))
        for session_id, messages in groupby(
            conversation, key=lambda message: message["session_id"]
        )
    ]

    chat: list[list[dict[str, Any]]] = []
    user_questions: list[dict[str, Any]] = []
    plan_sections: list[str] = []
    for batch_number, (session_id, messages) in enumerate(session_batches, start=1):
        user_messages = [message for message in messages if message["role"] == "user"]
        question_number = 0
        batch: list[dict[str, Any]] = []
        for message in messages:
            is_user = message["role"] == "user"
            if is_user:
                question_number += 1
            is_main_question = is_user and question_number == 1
            batch.append(
                {
                    "content": message["content"],
                    "id": _numeric_message_id(message["message_id"]),
                    "index": f"{batch_number},{question_number}" if is_user else None,
                    "question_type": (
                        "main_question"
                        if is_main_question
                        else "follow_up_question" if is_user else None
                    ),
                    "role": message["role"],
                    "time_anchor": message["timestamp"] if is_main_question else None,
                }
            )
        chat.append(batch)
        user_questions.append(
            {
                "messages": [[message["content"] for message in user_messages]],
                "time_anchor": messages[0]["timestamp"],
            }
        )
        plan_sections.append(
            "\n".join(
                (
                    f"BATCH {batch_number} PLAN",
                    f"- Session ID: {session_id}",
                    f"- Time Anchor: {messages[0]['timestamp']}",
                    f"- Messages: {len(messages)}",
                    f"- Opening topic: {messages[0]['content']}",
                )
            )
        )

    probing_questions = {ability: [] for ability in QUESTION_TYPES}
    for row in dataset["rows"]:
        probing_questions[row["ability"]].append(_to_hf_probe(row, messages_by_id))

    record = {
        "conversation_id": conversation_id,
        "conversation_seed": seed,
        "narratives": (
            "1. LABEL CATEGORY: Project Planning & Coordination\n"
            "   LABEL DESCRIPTION: Roles, venue logistics, budget, scheduling, "
            "and accommodations.\n\n"
            "2. LABEL CATEGORY: Long-Term Memory\n"
            "   LABEL DESCRIPTION: Updates, preferences, instructions, "
            "contradictions, and temporal facts."
        ),
        "user_profile": {
            "user_info": (
                "USER PROFILE:\n"
                "- Role: Participant in an ongoing synthetic conversation\n"
                f"- Goal: Explore {seed['title']} across multiple sessions."
            ),
            "user_relationships": (
                "RELATIONSHIPS:\n"
                "- Named collaborators and their responsibilities are recorded in the chat."
            ),
        },
        "conversation_plan": "\n\n".join(plan_sections),
        "user_questions": user_questions,
        "chat": chat,
        "probing_questions": repr(probing_questions),
    }
    validate_hf_beam_record(record)
    return record


def validate_hf_beam_record(record: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(record, dict):
        raise ValueError("HF BEAM record must be an object")
    if tuple(record) != HF_BEAM_FIELDS:
        errors.append(f"top-level fields must be exactly: {', '.join(HF_BEAM_FIELDS)}")

    conversation_seed = record.get("conversation_seed")
    if not isinstance(conversation_seed, dict):
        errors.append("conversation_seed must be an object")
    elif (
        not isinstance(conversation_seed.get("category"), str)
        or not isinstance(conversation_seed.get("id"), int)
        or not isinstance(conversation_seed.get("subtopics"), list)
        or not all(isinstance(item, str) for item in conversation_seed.get("subtopics", []))
        or not isinstance(conversation_seed.get("theme"), str)
        or not isinstance(conversation_seed.get("title"), str)
    ):
        errors.append("conversation_seed does not match the BEAM feature types")

    user_profile = record.get("user_profile")
    if not isinstance(user_profile, dict) or not all(
        isinstance(user_profile.get(field), str) for field in ("user_info", "user_relationships")
    ):
        errors.append("user_profile must contain user_info and user_relationships strings")

    chat = record.get("chat")
    chat_ids: list[int] = []
    if not isinstance(chat, list) or not chat:
        errors.append("chat must be a non-empty nested array")
    else:
        for batch in chat:
            if not isinstance(batch, list) or not batch:
                errors.append("each chat batch must be a non-empty array")
                continue
            for message in batch:
                if not isinstance(message, dict):
                    errors.append("each chat message must be an object")
                    continue
                if tuple(message) != (
                    "content",
                    "id",
                    "index",
                    "question_type",
                    "role",
                    "time_anchor",
                ):
                    errors.append("chat message fields do not match the BEAM schema")
                if isinstance(message.get("id"), int):
                    chat_ids.append(message["id"])
                else:
                    errors.append("chat message id must be an integer")
                if message.get("role") not in {"user", "assistant"}:
                    errors.append("chat message role must be user or assistant")

    if chat_ids != list(range(len(chat_ids))):
        errors.append("chat message IDs must be zero-based and sequential")

    user_questions = record.get("user_questions")
    if not isinstance(user_questions, list) or not user_questions:
        errors.append("user_questions must be a non-empty array")
    else:
        for batch in user_questions:
            messages = batch.get("messages") if isinstance(batch, dict) else None
            if (
                not isinstance(messages, list)
                or not all(
                    isinstance(group, list) and all(isinstance(item, str) for item in group)
                    for group in messages
                )
                or not isinstance(batch.get("time_anchor"), str)
            ):
                errors.append("user_questions does not match the BEAM nested feature type")

    serialized_probes = record.get("probing_questions")
    if not isinstance(serialized_probes, str):
        errors.append("probing_questions must be a string")
        probes = {}
    else:
        try:
            probes = ast.literal_eval(serialized_probes)
        except (SyntaxError, ValueError):
            errors.append("probing_questions must contain a Python-literal dictionary")
            probes = {}
    if not isinstance(probes, dict) or tuple(probes) != QUESTION_TYPES:
        errors.append("probing_questions must contain the ten BEAM ability keys in order")
    else:
        for ability in QUESTION_TYPES:
            values = probes[ability]
            if not isinstance(values, list) or len(values) != 2:
                errors.append(f"probing_questions.{ability} must contain exactly two items")
                continue
            for probe in values:
                if not isinstance(probe, dict) or not probe.get("question"):
                    errors.append(f"probing_questions.{ability} contains an invalid item")
                    continue
                if not probe.get(HF_ANSWER_FIELDS[ability]):
                    errors.append(
                        f"probing_questions.{ability} is missing {HF_ANSWER_FIELDS[ability]}"
                    )
                source_ids = probe.get("source_chat_ids")
                if not isinstance(source_ids, list) or any(
                    source_id not in chat_ids for source_id in source_ids
                ):
                    errors.append(f"probing_questions.{ability} has invalid source_chat_ids")
                if not isinstance(probe.get("rubric"), list) or not probe["rubric"]:
                    errors.append(f"probing_questions.{ability} must have a rubric list")

    if errors:
        raise ValueError("HF BEAM validation failed:\n- " + "\n- ".join(dict.fromkeys(errors)))

    return {
        "conversations": 1,
        "messages": len(chat_ids),
        "sessions": len(chat) if isinstance(chat, list) else 0,
        "probes": (
            sum(len(values) for values in probes.values())
            if isinstance(probes, dict)
            else 0
        ),
    }


def _write_hf_output(dataset: dict[str, Any], output: Path) -> Path:
    record = to_hf_beam_record(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _write_outputs(
    dataset: dict[str, Any], output: Path, hf_output: Path | None = None
) -> tuple[Path, Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output = output.with_name(f"{output.stem}.rows.jsonl")
    hf_output = hf_output or output.with_name(f"{output.stem}.hf.jsonl")
    output.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in dataset["rows"]
        ),
        encoding="utf-8",
    )
    _write_hf_output(dataset, hf_output)
    return output, rows_output, hf_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--hf-output",
        type=Path,
        help=(
            "Optional path for the one-row Mohammadta/BEAM-compatible JSONL export; "
            "defaults to <output stem>.hf.jsonl"
        ),
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--confirm-generation",
        action="store_true",
        help="Required acknowledgment that this command makes one model API call",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_generation:
        raise SystemExit(
            "This command makes one Gemini API call. Re-run with --confirm-generation."
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY environment variable")

    try:
        response = _call_gemini(
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
        dataset = json.loads(_candidate_text(response))
        validation = validate_dataset(dataset)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    dataset["generation"] = {
        "provider": "google_ai_studio_native_api",
        "model": args.model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
    }
    full_output, rows_output, hf_output = _write_outputs(
        dataset, args.output, hf_output=args.hf_output
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "model": args.model,
                **validation,
                "dataset_output": str(full_output),
                "rows_output": str(rows_output),
                "hf_dataset_output": str(hf_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
