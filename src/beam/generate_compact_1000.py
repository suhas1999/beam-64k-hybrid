"""Generate 50 compact conversations and 1,000 BEAM-style probes with Gemini.

The run is resumable at the conversation level. Each missing conversation makes
exactly one API request per invocation. Gemini usage metadata is saved with each
checkpoint and aggregated into a scaling report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.beam.generate_smoke_20 import (
    DEFAULT_MODEL,
    QUESTION_TYPES,
    _call_gemini,
    _candidate_text,
    to_hf_beam_record,
    validate_dataset,
    validate_hf_beam_record,
)


DEFAULT_TOPICS = Path("topics/64k/64k_topics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/compact_1000")
EXPECTED_CONVERSATIONS = 50
QUESTIONS_PER_CONVERSATION = 20
EXPECTED_PROBES = EXPECTED_CONVERSATIONS * QUESTIONS_PER_CONVERSATION
USAGE_FIELDS = (
    "promptTokenCount",
    "candidatesTokenCount",
    "thoughtsTokenCount",
    "cachedContentTokenCount",
    "toolUsePromptTokenCount",
    "totalTokenCount",
)


def compact_prompt(conversation_id: str, topic: dict[str, Any]) -> str:
    topic_json = json.dumps(
        {
            "category": topic["category"],
            "title": topic["title"],
            "theme": topic["theme"],
            "subtopics": topic.get("subtopics", []),
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""Create one compact, wholly synthetic BEAM-style benchmark conversation.

Use this topic seed:
{topic_json}

Return one JSON object with exactly three top-level fields: metadata,
conversation, and rows.

The metadata object must contain:
- dataset: "BEAM-style compact scaling sample"
- conversation_id: "{conversation_id}"
- requested_rows: 20
- generation_note: "Synthetic machine-generated data; not an official BEAM release."

The conversation must contain 26-34 coherent messages across at least five
chronological sessions. Every message must contain a unique sequential
message_id (m001 onward), session_id, ISO-8601 timestamp, role, and content.
Keep each session's messages together and keep timestamps chronological.

Build a realistic, evolving conversation around the supplied topic. Include
named people and responsibilities, concrete dates or times, a location or
technical environment, a budget or resource constraint, a user preference, a
standing instruction, a plan or fact that is later updated, an explicit
contradiction that is later resolved, and events spanning multiple sessions.
Also leave a few plausible adjacent details genuinely unstated so abstention
can be tested. Keep all non-deliberate facts internally consistent.

Create exactly 20 rows: exactly two rows for each exact ability label below,
in this order:
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
- row_id, sequentially numbered {conversation_id}_q01 through {conversation_id}_q20
- conversation_id: "{conversation_id}"
- ability: one of the exact labels above
- difficulty: easy, medium, or hard
- question
- reference_answer
- source_message_ids: a non-empty JSON array of existing message IDs
- rubric
- validation_status: "machine_generated_smoke_test"

Each non-abstention answer must be unambiguously grounded in the cited
messages. Each abstention question must request a genuinely absent fact; its
answer must explicitly state that the conversation does not provide it, while
its cited messages establish the nearby topic. Questions must not mention
message IDs, ability labels, or benchmark mechanics. Answers must be concise
and directly gradeable. Rubrics must identify the indispensable facts. Do not
copy released BEAM examples.

CRITICAL STRUCTURE CHECK: Repeat the exact conversation_id value on every one
of the 20 rows. Do not omit it from later rows, even when it is repetitive.
"""


def _usage_metadata(response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usageMetadata", {})
    return {
        field: int(raw.get(field, 0) or 0)
        for field in USAGE_FIELDS
    }


def _checkpoint_path(output_dir: Path, conversation_id: str) -> Path:
    return output_dir / "checkpoints" / f"{conversation_id}.json"


def _normalize_redundant_fields(
    dataset: dict[str, Any], conversation_id: str
) -> dict[str, Any]:
    """Fill deterministic identifiers without changing generated content."""

    metadata = dataset.get("metadata")
    if isinstance(metadata, dict):
        metadata.setdefault("conversation_id", conversation_id)
        metadata.setdefault("requested_rows", QUESTIONS_PER_CONVERSATION)
    rows = dataset.get("rows")
    if isinstance(rows, list):
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            row.setdefault("row_id", f"{conversation_id}_q{index:02d}")
            row.setdefault("conversation_id", conversation_id)
            row.setdefault(
                "validation_status", "machine_generated_smoke_test"
            )
    return dataset


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _generate_one(
    topic_index: int,
    topic: dict[str, Any],
    output_dir: Path,
    api_key: str,
    model: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
    attempt: int,
) -> dict[str, Any]:
    conversation_id = f"compact_{topic_index + 1:04d}"
    started_at = datetime.now(timezone.utc).isoformat()
    response: dict[str, Any] | None = None
    try:
        response = _call_gemini(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            prompt=compact_prompt(conversation_id, topic),
        )
        usage = _usage_metadata(response)
        dataset = _normalize_redundant_fields(
            json.loads(_candidate_text(response)), conversation_id
        )
        validation = validate_dataset(
            dataset, expected_conversation_id=conversation_id
        )
        dataset["generation"] = {
            "provider": "google_ai_studio_native_api",
            "model": model,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "topic_index": topic_index,
            "topic_seed": topic,
            "usage_metadata": usage,
            "validation": validation,
        }
        _write_json_atomic(
            _checkpoint_path(output_dir, conversation_id), dataset
        )
        return {
            "conversation_id": conversation_id,
            "topic_index": topic_index,
            "attempt": attempt,
            "status": "ok",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "usage_metadata": usage,
        }
    except Exception as exc:
        return {
            "conversation_id": conversation_id,
            "topic_index": topic_index,
            "attempt": attempt,
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "usage_metadata": _usage_metadata(response) if response else {},
            "error": str(exc)[:2000],
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _next_attempts(entries: list[dict[str, Any]]) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for entry in entries:
        conversation_id = entry.get("conversation_id")
        attempt = entry.get("attempt")
        if isinstance(conversation_id, str) and isinstance(attempt, int):
            attempts[conversation_id] = max(attempts.get(conversation_id, 0), attempt)
    return attempts


def _usage_summary(
    usage_rows: list[dict[str, Any]],
    model: str,
    generated_at_utc: str,
    attempt_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    successful_totals = {
        field: sum(int(row.get(field, 0) or 0) for row in usage_rows)
        for field in USAGE_FIELDS
    }
    normalized_attempts = []
    if attempt_rows is not None:
        for row in attempt_rows:
            usage = row.get("usage_metadata", {})
            if isinstance(usage, dict) and any(usage.values()):
                normalized_attempts.append(usage)
    billed_rows = normalized_attempts or usage_rows
    totals = {
        field: sum(int(row.get(field, 0) or 0) for row in billed_rows)
        for field in USAGE_FIELDS
    }
    count = len(usage_rows)
    averages = {
        field: (totals[field] / count if count else 0.0)
        for field in USAGE_FIELDS
    }
    projections = {}
    for conversations in (100, 500, 1_000, 10_000):
        projections[str(conversations)] = {
            "input_tokens": round(averages["promptTokenCount"] * conversations),
            "candidate_output_tokens": round(
                averages["candidatesTokenCount"] * conversations
            ),
            "thinking_tokens": round(
                averages["thoughtsTokenCount"] * conversations
            ),
            "total_tokens": round(averages["totalTokenCount"] * conversations),
            "estimated_questions": conversations * QUESTIONS_PER_CONVERSATION,
        }

    return {
        "generated_at_utc": generated_at_utc,
        "provider": "google_ai_studio_native_api",
        "model": model,
        "source": "Gemini GenerateContentResponse.usageMetadata",
        "api_attempts": len(attempt_rows) if attempt_rows is not None else count,
        "token_counted_attempts": len(billed_rows),
        "failed_attempts": (
            sum(row.get("status") != "ok" for row in attempt_rows)
            if attempt_rows is not None
            else 0
        ),
        "completed_conversations": count,
        "questions_per_conversation": QUESTIONS_PER_CONVERSATION,
        "completed_questions": count * QUESTIONS_PER_CONVERSATION,
        "totals": {
            "input_tokens": totals["promptTokenCount"],
            "candidate_output_tokens": totals["candidatesTokenCount"],
            "thinking_tokens": totals["thoughtsTokenCount"],
            "cached_input_tokens": totals["cachedContentTokenCount"],
            "tool_use_prompt_tokens": totals["toolUsePromptTokenCount"],
            "total_tokens": totals["totalTokenCount"],
        },
        "averages_per_conversation": {
            "input_tokens": averages["promptTokenCount"],
            "candidate_output_tokens": averages["candidatesTokenCount"],
            "thinking_tokens": averages["thoughtsTokenCount"],
            "total_tokens": averages["totalTokenCount"],
        },
        "successful_responses_only": {
            "input_tokens": successful_totals["promptTokenCount"],
            "candidate_output_tokens": successful_totals[
                "candidatesTokenCount"
            ],
            "thinking_tokens": successful_totals["thoughtsTokenCount"],
            "total_tokens": successful_totals["totalTokenCount"],
        },
        "linear_projections": projections,
        "projection_note": (
            "Linear estimates based on these compact conversations. Actual usage varies "
            "with prompt size, response length, model behavior, and failed attempts."
        ),
    }


def _dataset_card(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    averages = summary["averages_per_conversation"]
    return f"""---
license: cc-by-sa-4.0
language:
- en
pretty_name: BEAM Compact Gemini Scaling Sample
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# BEAM Compact Gemini Scaling Sample

This is a wholly synthetic, machine-generated scaling sample inspired by the
schema of `Mohammadta/BEAM`. It is not an official BEAM release and has not
received BEAM's human validation.

## Contents

- Conversations: {summary['completed_conversations']}
- Probing questions: {summary['completed_questions']}
- Questions per conversation: {summary['questions_per_conversation']}
- Model: `{summary['model']}`

Each dataset row contains the eight published BEAM columns:
`conversation_id`, `conversation_seed`, `narratives`, `user_profile`,
`conversation_plan`, `user_questions`, `chat`, and `probing_questions`.
The `probing_questions` value is a string-serialized dictionary with two
questions for each of ten memory abilities.

## Observed Gemini token usage

Usage comes directly from Gemini `GenerateContentResponse.usageMetadata`.
Totals include all {summary['api_attempts']} API attempts required to produce
the {summary['completed_conversations']} validated conversations, including
{summary['failed_attempts']} rejected attempts. Per-conversation averages are
therefore amortized generation costs including retries.

| Measure | Total | Average per conversation |
|---|---:|---:|
| Input tokens | {totals['input_tokens']:,} | {averages['input_tokens']:,.2f} |
| Candidate output tokens | {totals['candidate_output_tokens']:,} | {averages['candidate_output_tokens']:,.2f} |
| Thinking tokens | {totals['thinking_tokens']:,} | {averages['thinking_tokens']:,.2f} |
| Total tokens | {totals['total_tokens']:,} | {averages['total_tokens']:,.2f} |

See `metadata/gemini_usage_summary.json` for linear scaling projections and
`metadata/gemini_usage_by_conversation.jsonl` for per-conversation counts.

## Limitations

These are compact conversations rather than 100K/500K/1M-token BEAM samples.
All probes have `machine_generated_smoke_test` validation status and should be
human-reviewed before use as a benchmark.
"""


def assemble_outputs(
    topics: list[dict[str, Any]], output_dir: Path, model: str
) -> dict[str, Any]:
    datasets = []
    hf_records = []
    rows = []
    usage_rows = []
    missing = []
    for topic_index, topic in enumerate(topics[:EXPECTED_CONVERSATIONS]):
        conversation_id = f"compact_{topic_index + 1:04d}"
        path = _checkpoint_path(output_dir, conversation_id)
        if not path.exists():
            missing.append(conversation_id)
            continue
        dataset = json.loads(path.read_text(encoding="utf-8"))
        validate_dataset(dataset, expected_conversation_id=conversation_id)
        record = to_hf_beam_record(dataset, conversation_seed=topic)
        validate_hf_beam_record(record)
        datasets.append(dataset)
        hf_records.append(record)
        rows.extend(dataset["rows"])
        usage_rows.append(
            {
                "conversation_id": conversation_id,
                "topic_id": topic["id"],
                "category": topic["category"],
                "model": dataset["generation"]["model"],
                **dataset["generation"]["usage_metadata"],
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    attempt_rows = _load_jsonl(output_dir / "generation_attempts.jsonl")
    summary = _usage_summary(
        usage_rows,
        model=model,
        generated_at_utc=generated_at,
        attempt_rows=attempt_rows,
    )
    summary["expected_conversations"] = EXPECTED_CONVERSATIONS
    summary["expected_questions"] = EXPECTED_PROBES
    summary["missing_conversations"] = missing

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "gemini_usage_summary.json", summary)
    if missing:
        raise RuntimeError(
            f"Cannot assemble: {len(missing)} conversations are missing: "
            + ", ".join(missing)
        )
    if len(datasets) != EXPECTED_CONVERSATIONS or len(rows) != EXPECTED_PROBES:
        raise RuntimeError(
            f"Expected {EXPECTED_CONVERSATIONS} conversations and {EXPECTED_PROBES} "
            f"probes; found {len(datasets)} and {len(rows)}"
        )

    hub_dir = output_dir / "hub"
    data_dir = hub_dir / "data"
    metadata_dir = hub_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in hf_records
        ),
        encoding="utf-8",
    )
    (output_dir / "probes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (metadata_dir / "gemini_usage_by_conversation.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in usage_rows
        ),
        encoding="utf-8",
    )
    (metadata_dir / "generation_attempts.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in attempt_rows
        ),
        encoding="utf-8",
    )
    _write_json_atomic(metadata_dir / "gemini_usage_summary.json", summary)
    (hub_dir / "README.md").write_text(_dataset_card(summary), encoding="utf-8")

    return {
        "status": "ok",
        "conversations": len(hf_records),
        "probes": len(rows),
        "hub_directory": str(hub_dir),
        "dataset_output": str(data_dir / "train.jsonl"),
        "probes_output": str(output_dir / "probes.jsonl"),
        "usage_summary_output": str(
            metadata_dir / "gemini_usage_summary.json"
        ),
        "usage": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument(
        "--confirm-generation",
        action="store_true",
        help="Required acknowledgment that missing chats each make one API request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8"))
    if len(topics) < EXPECTED_CONVERSATIONS:
        raise SystemExit(
            f"Need at least {EXPECTED_CONVERSATIONS} topics; found {len(topics)}"
        )
    topics = topics[:EXPECTED_CONVERSATIONS]

    attempts_path = args.output_dir / "generation_attempts.jsonl"
    attempts = _load_jsonl(attempts_path)
    next_attempts = _next_attempts(attempts)
    pending = []
    for topic_index, topic in enumerate(topics):
        conversation_id = f"compact_{topic_index + 1:04d}"
        if not _checkpoint_path(args.output_dir, conversation_id).exists():
            pending.append((topic_index, topic, conversation_id))

    if pending and not args.assemble_only:
        if not args.confirm_generation:
            raise SystemExit(
                f"{len(pending)} conversations are missing and will each make one "
                "Gemini API request. Re-run with --confirm-generation."
            )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Missing GEMINI_API_KEY environment variable")
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")

        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _generate_one,
                    topic_index,
                    topic,
                    args.output_dir,
                    api_key,
                    args.model,
                    args.timeout,
                    args.max_output_tokens,
                    args.temperature,
                    next_attempts.get(conversation_id, 0) + 1,
                ): conversation_id
                for topic_index, topic, conversation_id in pending
            }
            for future in as_completed(futures):
                result = future.result()
                _append_jsonl(attempts_path, result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["status"] != "ok":
                    failures.append(result["conversation_id"])
        if failures:
            print(
                f"Generation failed for {len(failures)} conversations. "
                "Re-run the same command to resume: " + ", ".join(failures),
                file=sys.stderr,
            )

    try:
        result = assemble_outputs(topics, args.output_dir, args.model)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
