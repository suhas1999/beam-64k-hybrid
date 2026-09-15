"""Exhaustively verify an assembled OpenRouter BEAM-style dataset package."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.beam.generate_smoke_20 import QUESTION_TYPES, validate_hf_beam_record


DEFAULT_HUB_DIR = Path("outputs/openrouter_50k/hub_50k")
DEFAULT_PROBES = Path("outputs/openrouter_50k/probes.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def verify(
    hub_dir: Path,
    probes_path: Path,
    expected_conversations: int,
    expected_probes: int,
) -> dict[str, Any]:
    if expected_probes != expected_conversations * 20:
        raise ValueError("expected_probes must equal expected_conversations * 20")

    shard_paths = sorted((hub_dir / "data").glob("train-*.jsonl"))
    if not shard_paths:
        raise ValueError(f"no data shards found under {hub_dir / 'data'}")

    expected_conversation_ids = [
        f"beam50k_{index:06d}" for index in range(1, expected_conversations + 1)
    ]
    conversation_ids: list[str] = []
    nested_probe_pairs: list[tuple[str, str]] = []
    nested_question_texts: list[str] = []
    message_counts: list[int] = []
    session_counts: list[int] = []

    for shard_path in shard_paths:
        for record in _read_jsonl(shard_path):
            validation = validate_hf_beam_record(record)
            conversation_id = record.get("conversation_id")
            if not isinstance(conversation_id, str):
                raise ValueError(f"{shard_path}: conversation_id must be a string")
            conversation_ids.append(conversation_id)
            message_counts.append(validation["messages"])
            session_counts.append(validation["sessions"])
            if not 26 <= validation["messages"] <= 34:
                raise ValueError(
                    f"{conversation_id}: expected 26-34 messages; "
                    f"found {validation['messages']}"
                )
            if validation["sessions"] < 5:
                raise ValueError(
                    f"{conversation_id}: expected at least five sessions; "
                    f"found {validation['sessions']}"
                )

            probes = ast.literal_eval(record["probing_questions"])
            for ability in QUESTION_TYPES:
                for probe in probes[ability]:
                    question_id = probe.get("question_id")
                    question = probe.get("question")
                    if not isinstance(question_id, str) or not isinstance(question, str):
                        raise ValueError(
                            f"{conversation_id}: invalid nested question ID or text"
                        )
                    nested_probe_pairs.append((question_id, question))
                    nested_question_texts.append(" ".join(question.split()).casefold())

    if conversation_ids != expected_conversation_ids:
        raise ValueError("conversation IDs are not complete, unique, and sequential")
    if len(nested_probe_pairs) != expected_probes:
        raise ValueError(
            f"expected {expected_probes} nested probes; found {len(nested_probe_pairs)}"
        )
    nested_question_ids = [question_id for question_id, _ in nested_probe_pairs]
    if len(nested_question_ids) != len(set(nested_question_ids)):
        raise ValueError("nested question IDs are not globally unique")

    flat_probes = _read_jsonl(probes_path)
    if len(flat_probes) != expected_probes:
        raise ValueError(
            f"expected {expected_probes} flat probes; found {len(flat_probes)}"
        )
    flat_probe_pairs = [(row.get("row_id"), row.get("question")) for row in flat_probes]
    if flat_probe_pairs != nested_probe_pairs:
        raise ValueError("flat probes do not exactly match nested probes in shard order")

    ability_counts = Counter(row.get("ability") for row in flat_probes)
    expected_ability_count = expected_conversations * 2
    if ability_counts != Counter(
        {ability: expected_ability_count for ability in QUESTION_TYPES}
    ):
        raise ValueError(f"ability distribution is incorrect: {dict(ability_counts)}")

    topics = json.loads((hub_dir / "metadata" / "topics.json").read_text())
    if not isinstance(topics, list) or len(topics) != expected_conversations:
        raise ValueError("topics metadata count does not match conversation count")
    topic_ids = [topic.get("id") for topic in topics if isinstance(topic, dict)]
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError("topic IDs are not globally unique")

    usage_rows = _read_jsonl(
        hub_dir / "metadata" / "openrouter_usage_by_conversation.jsonl"
    )
    if len(usage_rows) != expected_conversations:
        raise ValueError("usage row count does not match conversation count")
    if [row.get("conversation_id") for row in usage_rows] != expected_conversation_ids:
        raise ValueError("usage rows do not match sequential conversation IDs")
    response_ids = [row.get("response_id") for row in usage_rows]
    if not all(isinstance(response_id, str) and response_id for response_id in response_ids):
        raise ValueError("one or more usage rows is missing a response ID")
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("successful response IDs are not globally unique")

    summary = json.loads(
        (hub_dir / "metadata" / "openrouter_usage_summary.json").read_text()
    )
    if summary.get("completed_conversations") != expected_conversations:
        raise ValueError("usage summary conversation count is incorrect")
    if summary.get("completed_questions") != expected_probes:
        raise ValueError("usage summary probe count is incorrect")
    if summary.get("missing_conversations"):
        raise ValueError("usage summary reports missing conversations")

    successful_expected = summary.get("successful_responses_only")
    successful_actual = {
        "input_tokens": sum(int(row["prompt_tokens"]) for row in usage_rows),
        "completion_tokens_including_reasoning": sum(
            int(row["completion_tokens"]) for row in usage_rows
        ),
        "visible_output_tokens": sum(
            int(row["visible_output_tokens"]) for row in usage_rows
        ),
        "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in usage_rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in usage_rows),
        "cost_usd": sum(float(row["cost_usd"]) for row in usage_rows),
    }
    for key, actual in successful_actual.items():
        expected = successful_expected.get(key) if isinstance(successful_expected, dict) else None
        tolerance = 1e-9 if key == "cost_usd" else 0
        if expected is None or abs(actual - expected) > tolerance:
            raise ValueError(
                f"successful usage total mismatch for {key}: "
                f"expected {expected}, found {actual}"
            )

    duplicate_text_count = len(nested_question_texts) - len(set(nested_question_texts))
    if summary.get("question_text_duplicates_across_conversations") != duplicate_text_count:
        raise ValueError("question-text duplicate count does not match usage summary")

    category_counts = Counter(
        topic.get("category") for topic in topics if isinstance(topic, dict)
    )
    return {
        "status": "ok",
        "shards": len(shard_paths),
        "conversations": len(conversation_ids),
        "probes": len(flat_probes),
        "abilities": dict(ability_counts),
        "message_count_min": min(message_counts),
        "message_count_max": max(message_counts),
        "session_count_min": min(session_counts),
        "session_count_max": max(session_counts),
        "topic_categories": len(category_counts),
        "question_text_duplicates": duplicate_text_count,
        "input_tokens": summary["totals"]["input_tokens"],
        "completion_tokens_including_reasoning": summary["totals"][
            "completion_tokens_including_reasoning"
        ],
        "cost_usd": summary["totals"]["cost_usd"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-dir", type=Path, default=DEFAULT_HUB_DIR)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--expected-conversations", type=int, default=2500)
    parser.add_argument("--expected-probes", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify(
                args.hub_dir,
                args.probes,
                args.expected_conversations,
                args.expected_probes,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
