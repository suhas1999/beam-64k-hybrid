"""Generate 2,500 chats / 50,000 probes through OpenRouter.

The generator is resumable at conversation granularity. A completed chat is
stored as an atomic checkpoint, so later runs only request missing chats. The
same output directory can first be run with ``--target-conversations 25`` as a
500-probe smoke test and then with the default 2,500-chat target.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.beam.generate_compact_1000 import compact_prompt
from src.beam.generate_smoke_20 import (
    QUESTION_TYPES,
    to_hf_beam_record,
    validate_dataset,
    validate_hf_beam_record,
)


DEFAULT_MODEL = "google/gemini-3.8-flash"
DEFAULT_TOPICS = Path("topics/50k_rows/2500_topics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/openrouter_50k")
EXPECTED_CONVERSATIONS = 2_500
QUESTIONS_PER_CONVERSATION = 20
EXPECTED_PROBES = EXPECTED_CONVERSATIONS * QUESTIONS_PER_CONVERSATION
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "total_tokens",
    "cost_usd",
)


def conversation_id_for(topic_index: int) -> str:
    return f"beam50k_{topic_index + 1:06d}"


def openrouter_prompt(conversation_id: str, topic: dict[str, Any]) -> str:
    return compact_prompt(conversation_id, topic) + """

QUALITY CHECKS BEFORE RETURNING JSON:
- Every cited message must directly support the reference answer, not merely
  discuss the same broad topic.
- An abstention target must be wholly absent and not logically deducible. If
  the conversation supplies a value that answers the question, even indirectly
  (for example, a zero total that necessarily implies a zero component), use a
  different genuinely missing detail.
- The two questions within each ability must test different facts.
- Do not invent facts in reference answers that are absent from the chat.
- For health, legal, financial, privacy, or security topics, keep advice
  educational, organizational, or defensive and recommend qualified review
  where an individualized high-stakes decision is involved.
"""


def _json_schema() -> dict[str, Any]:
    message = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message_id": {"type": "string"},
            "session_id": {"type": "string"},
            "timestamp": {"type": "string"},
            "role": {"type": "string", "enum": ["user", "assistant"]},
            "content": {"type": "string"},
        },
        "required": [
            "message_id",
            "session_id",
            "timestamp",
            "role",
            "content",
        ],
    }
    row = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "row_id": {"type": "string"},
            "conversation_id": {"type": "string"},
            "ability": {"type": "string", "enum": list(QUESTION_TYPES)},
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
            },
            "question": {"type": "string"},
            "reference_answer": {"type": "string"},
            "source_message_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "rubric": {"type": "string"},
            "validation_status": {"type": "string"},
        },
        "required": [
            "row_id",
            "conversation_id",
            "ability",
            "difficulty",
            "question",
            "reference_answer",
            "source_message_ids",
            "rubric",
            "validation_status",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dataset": {"type": "string"},
                    "conversation_id": {"type": "string"},
                    "requested_rows": {"type": "integer"},
                    "generation_note": {"type": "string"},
                },
                "required": [
                    "dataset",
                    "conversation_id",
                    "requested_rows",
                    "generation_note",
                ],
            },
            "conversation": {
                "type": "array",
                "minItems": 26,
                "maxItems": 34,
                "items": message,
            },
            "rows": {
                "type": "array",
                "minItems": QUESTIONS_PER_CONVERSATION,
                "maxItems": QUESTIONS_PER_CONVERSATION,
                "items": row,
            },
        },
        "required": ["metadata", "conversation", "rows"],
    }


def _request_payload(
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate strict, wholly synthetic benchmark JSON. Follow "
                    "the supplied schema and return no Markdown or commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort, "exclude": True},
        # Gemini accepts the individual JSON-Schema features used above, but
        # rejects the complete deeply nested schema at this output size. JSON
        # mode plus local structural validation is reliable for this workload.
        "response_format": {"type": "json_object"},
        "plugins": [{"id": "response-healing"}],
        "provider": {"require_parameters": True},
        "stream": False,
    }


def _call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> dict[str, Any]:
    body = json.dumps(
        _request_payload(
            model,
            prompt,
            max_output_tokens,
            temperature,
            reasoning_effort,
        ),
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://huggingface.co/datasets/vm2825/BEAM_50K",
            "X-Title": "BEAM 50K synthetic dataset generation",
            "X-OpenRouter-Metadata": "enabled",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        retry_after = exc.headers.get("Retry-After")
        detail = f"OpenRouter API returned HTTP {exc.code}"
        if retry_after:
            detail += f" (Retry-After: {retry_after})"
        safe_error: dict[str, Any] = {}
        try:
            parsed_error = json.loads(response_text).get("error", {})
            metadata = parsed_error.get("metadata", {})
            safe_error = {
                "message": parsed_error.get("message"),
                "code": parsed_error.get("code"),
                "provider_name": metadata.get("provider_name"),
                "provider_error_code": metadata.get("provider_error_code"),
            }
        except (AttributeError, json.JSONDecodeError):
            safe_error = {"message": response_text[:500]}
        raise RuntimeError(
            f"{detail}: {json.dumps(safe_error, ensure_ascii=False)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"OpenRouter returned no choices: {str(response)[:1000]}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter returned an empty assistant message")
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    return content


def _usage_metadata(response: dict[str, Any] | None) -> dict[str, int | float]:
    raw = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    completion_details = raw.get("completion_tokens_details", {})
    prompt_details = raw.get("prompt_tokens_details", {})
    if not isinstance(completion_details, dict):
        completion_details = {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    completion_tokens = int(raw.get("completion_tokens", 0) or 0)
    reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
    return {
        "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
        "completion_tokens": completion_tokens,
        "visible_output_tokens": max(0, completion_tokens - reasoning_tokens),
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": int(prompt_details.get("cached_tokens", 0) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
        "cost_usd": float(raw.get("cost", 0.0) or 0.0),
    }


def _normalize_generated_dataset(
    dataset: dict[str, Any], conversation_id: str
) -> dict[str, Any]:
    """Canonicalize deterministic identifiers while preserving generated prose."""

    metadata = dataset.get("metadata")
    if isinstance(metadata, dict):
        metadata["dataset"] = "BEAM-style OpenRouter 50K compact dataset"
        metadata["conversation_id"] = conversation_id
        metadata["requested_rows"] = QUESTIONS_PER_CONVERSATION
        metadata["generation_note"] = (
            "Synthetic machine-generated data; not an official BEAM release."
        )

    conversation = dataset.get("conversation")
    id_map: dict[str, str] = {}
    if isinstance(conversation, list):
        for index, message in enumerate(conversation, start=1):
            if not isinstance(message, dict):
                continue
            new_id = f"m{index:03d}"
            old_id = message.get("message_id")
            if isinstance(old_id, str):
                id_map[old_id] = new_id
            message["message_id"] = new_id
        existing_sessions = {
            message.get("session_id")
            for message in conversation
            if isinstance(message, dict) and message.get("session_id")
        }
        # Occasionally Gemini repeats s01 even though its timestamps clearly
        # form five separate days. Recover those semantic sessions from the
        # chronological date groups rather than paying to regenerate prose.
        if len(existing_sessions) < 5:
            date_order: dict[str, str] = {}
            for message in conversation:
                if not isinstance(message, dict):
                    continue
                timestamp = str(message.get("timestamp", ""))
                date_key = timestamp[:10]
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_key):
                    date_order.setdefault(
                        date_key, f"s{len(date_order) + 1:02d}"
                    )
            if len(date_order) >= 5:
                for message in conversation:
                    if isinstance(message, dict):
                        date_key = str(message.get("timestamp", ""))[:10]
                        if date_key in date_order:
                            message["session_id"] = date_order[date_key]

    rows = dataset.get("rows")
    if isinstance(rows, list):
        counts = Counter(
            row.get("ability") for row in rows if isinstance(row, dict)
        )
        if all(counts[ability] == 2 for ability in QUESTION_TYPES):
            ability_order = {
                ability: index for index, ability in enumerate(QUESTION_TYPES)
            }
            rows.sort(key=lambda row: ability_order.get(row.get("ability"), 999))
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            row["row_id"] = f"{conversation_id}_q{index:02d}"
            row["conversation_id"] = conversation_id
            row.setdefault("difficulty", "medium")
            row["validation_status"] = "machine_generated_smoke_test"
            source_ids = row.get("source_message_ids")
            if isinstance(source_ids, list):
                normalized_sources = []
                for source_id in source_ids:
                    source_text = str(source_id)
                    normalized = id_map.get(source_text)
                    if normalized is None:
                        numeric = re.fullmatch(r"m0*(\d+)", source_text)
                        if numeric and 1 <= int(numeric.group(1)) <= len(
                            conversation or []
                        ):
                            normalized = f"m{int(numeric.group(1)):03d}"
                    normalized_sources.append(normalized or source_text)
                row["source_message_ids"] = normalized_sources
    return dataset


def _parse_iso8601(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _validate_quality(dataset: dict[str, Any]) -> dict[str, Any]:
    messages = dataset["conversation"]
    timestamps = []
    session_sequence = []
    for index, message in enumerate(messages, start=1):
        try:
            timestamps.append(_parse_iso8601(message["timestamp"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"message {index} has invalid ISO timestamp") from exc
        if len(message["content"].strip()) < 8:
            raise ValueError(f"message {index} content is too short")
        session_sequence.append(message["session_id"])
    if timestamps != sorted(timestamps):
        raise ValueError("conversation timestamps are not chronological")
    closed_sessions: set[str] = set()
    active = None
    for session_id in session_sequence:
        if session_id != active:
            if session_id in closed_sessions:
                raise ValueError("session messages are not grouped contiguously")
            if active is not None:
                closed_sessions.add(active)
            active = session_id

    difficulties = Counter(row["difficulty"] for row in dataset["rows"])
    for index, row in enumerate(dataset["rows"], start=1):
        if len(row["question"].strip()) < 10:
            raise ValueError(f"row {index} question is too short")
        if not str(row["rubric"]).strip():
            raise ValueError(f"row {index} rubric is empty")
    if len(difficulties) < 2:
        raise ValueError("questions must use at least two difficulty levels")
    return {
        "chronological_timestamps": True,
        "contiguous_sessions": True,
        "difficulty_counts": dict(sorted(difficulties.items())),
    }


def _checkpoint_path(output_dir: Path, conversation_id: str) -> Path:
    return output_dir / "checkpoints" / f"{conversation_id}.json"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _next_attempts(entries: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in entries:
        conversation_id = entry.get("conversation_id")
        attempt = entry.get("attempt")
        if isinstance(conversation_id, str) and isinstance(attempt, int):
            result[conversation_id] = max(result.get(conversation_id, 0), attempt)
    return result


def _sanitized_attempt_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove account identifiers that can occur in legacy HTTP errors."""

    clean = dict(row)
    error = clean.get("error")
    if isinstance(error, str):
        error = re.sub(
            r'"user_id"\s*:\s*"[^"]+"',
            '"user_id":"[redacted]"',
            error,
        )
        error = re.sub(
            r'"org_id"\s*:\s*"[^"]+"',
            '"org_id":"[redacted]"',
            error,
        )
        clean["error"] = error
    return clean


def _save_failed_response(
    output_dir: Path,
    conversation_id: str,
    attempt: int,
    response: dict[str, Any],
) -> None:
    safe_response = {
        key: value
        for key, value in response.items()
        if key not in {"debug", "request", "headers"}
    }
    _write_json_atomic(
        output_dir
        / "failed_responses"
        / f"{conversation_id}_attempt_{attempt:02d}.json",
        safe_response,
    )


def _generate_one(
    topic_index: int,
    topic: dict[str, Any],
    output_dir: Path,
    api_key: str,
    model: str,
    timeout: int,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str,
    attempt: int,
) -> dict[str, Any]:
    conversation_id = conversation_id_for(topic_index)
    started_at = datetime.now(timezone.utc).isoformat()
    response: dict[str, Any] | None = None
    try:
        response = _call_openrouter(
            api_key=api_key,
            model=model,
            prompt=openrouter_prompt(conversation_id, topic),
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        usage = _usage_metadata(response)
        dataset = json.loads(_response_text(response))
        if not isinstance(dataset, dict):
            raise ValueError("generated JSON must be an object")
        dataset = _normalize_generated_dataset(dataset, conversation_id)
        validation = validate_dataset(
            dataset, expected_conversation_id=conversation_id
        )
        validation.update(_validate_quality(dataset))
        dataset["generation"] = {
            "provider": "openrouter",
            "requested_model": model,
            "response_model": response.get("model", model),
            "upstream_provider": response.get("provider"),
            "response_id": response.get("id"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "topic_index": topic_index,
            "topic_seed": topic,
            "attempt": attempt,
            "reasoning_effort": reasoning_effort,
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
            "response_id": response.get("id"),
            "response_model": response.get("model", model),
            "upstream_provider": response.get("provider"),
            "usage_metadata": usage,
        }
    except Exception as exc:
        if response:
            _save_failed_response(
                output_dir, conversation_id, attempt, response
            )
        return {
            "conversation_id": conversation_id,
            "topic_index": topic_index,
            "attempt": attempt,
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "response_id": response.get("id") if response else None,
            "response_model": response.get("model") if response else None,
            "upstream_provider": response.get("provider") if response else None,
            "usage_metadata": _usage_metadata(response),
            "error": str(exc)[:2000],
        }


def _run_generation_pass(
    pending: list[tuple[int, dict[str, Any], str]],
    args: argparse.Namespace,
    api_key: str,
    attempts_path: Path,
    next_attempts: dict[str, int],
    progress_start: int,
    target: int,
) -> tuple[int, list[dict[str, Any]]]:
    processed = 0
    failures: list[dict[str, Any]] = []
    for batch_start in range(0, len(pending), args.request_batch_size):
        batch = pending[batch_start : batch_start + args.request_batch_size]
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
                    args.reasoning_effort,
                    next_attempts.get(conversation_id, 0) + 1,
                ): conversation_id
                for topic_index, topic, conversation_id in batch
            }
            batch_failures = []
            for future in as_completed(futures):
                result = future.result()
                _append_jsonl(attempts_path, result)
                next_attempts[result["conversation_id"]] = result["attempt"]
                processed += 1
                if result["status"] != "ok":
                    failures.append(result)
                    batch_failures.append(result)
                usage = result.get("usage_metadata", {})
                print(
                    json.dumps(
                        {
                            "progress": f"{progress_start + processed}/{target}",
                            "conversation_id": result["conversation_id"],
                            "attempt": result["attempt"],
                            "status": result["status"],
                            "tokens": usage.get("total_tokens", 0),
                            "cost_usd": usage.get("cost_usd", 0.0),
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if batch and len(batch_failures) / len(batch) >= args.abort_failure_ratio:
            raise RuntimeError(
                f"Aborting after systemic failures in a batch: "
                f"{len(batch_failures)}/{len(batch)} failed"
            )
    return processed, failures


def _pending_topics(
    topics: list[dict[str, Any]], output_dir: Path
) -> list[tuple[int, dict[str, Any], str]]:
    pending = []
    for topic_index, topic in enumerate(topics):
        conversation_id = conversation_id_for(topic_index)
        if not _checkpoint_path(output_dir, conversation_id).exists():
            pending.append((topic_index, topic, conversation_id))
    return pending


def _usage_summary(
    usage_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    model: str,
    target: int,
) -> dict[str, Any]:
    successful_usage = [
        {field: row.get(field, 0) for field in OPENROUTER_USAGE_FIELDS}
        | {"visible_output_tokens": row.get("visible_output_tokens", 0)}
        for row in usage_rows
    ]
    # A forced interruption can occur after a worker atomically saves a valid
    # checkpoint but before the main thread appends its attempt log. Combine
    # both sources and deduplicate paid responses by OpenRouter response ID.
    billed_by_response: dict[str, dict[str, Any]] = {}
    attempt_response_ids = set()
    for row in attempt_rows:
        response_id = row.get("response_id")
        if isinstance(response_id, str) and response_id:
            attempt_response_ids.add(response_id)
        usage = row.get("usage_metadata", {})
        if not isinstance(usage, dict) or not any(usage.values()):
            continue
        key = (
            response_id
            if isinstance(response_id, str) and response_id
            else f"attempt:{row.get('conversation_id')}:{row.get('attempt')}"
        )
        billed_by_response.setdefault(key, usage)
    unlogged_successes = 0
    for row in usage_rows:
        response_id = row.get("response_id")
        if isinstance(response_id, str) and response_id:
            key = response_id
            if response_id not in attempt_response_ids:
                unlogged_successes += 1
        else:
            key = f"success:{row.get('conversation_id')}"
        usage = {
            field: row.get(field, 0)
            for field in (*OPENROUTER_USAGE_FIELDS, "visible_output_tokens")
        }
        billed_by_response.setdefault(key, usage)
    counted_attempts = list(billed_by_response.values())

    def total(rows: list[dict[str, Any]], field: str) -> int | float:
        value = sum(row.get(field, 0) or 0 for row in rows)
        return float(value) if field == "cost_usd" else int(value)

    fields = (*OPENROUTER_USAGE_FIELDS, "visible_output_tokens")
    totals = {field: total(counted_attempts, field) for field in fields}
    successful_totals = {
        field: total(successful_usage, field) for field in fields
    }
    completed = len(usage_rows)
    averages = {
        field: (totals[field] / completed if completed else 0.0)
        for field in fields
    }
    projections = {}
    for conversations in (25, 50, 500, 2_500, 10_000):
        projections[str(conversations)] = {
            "input_tokens": round(averages["prompt_tokens"] * conversations),
            "completion_tokens_including_reasoning": round(
                averages["completion_tokens"] * conversations
            ),
            "reasoning_tokens": round(
                averages["reasoning_tokens"] * conversations
            ),
            "total_tokens": round(averages["total_tokens"] * conversations),
            "estimated_cost_usd": round(
                averages["cost_usd"] * conversations, 6
            ),
            "estimated_questions": conversations * QUESTIONS_PER_CONVERSATION,
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "openrouter",
        "model": model,
        "source": "OpenRouter chat completion response usage",
        "target_conversations": target,
        "target_questions": target * QUESTIONS_PER_CONVERSATION,
        "api_attempts": len(attempt_rows) + unlogged_successes,
        "successful_checkpoints_missing_attempt_log": unlogged_successes,
        "token_counted_attempts": len(counted_attempts),
        "failed_attempts": sum(
            row.get("status") != "ok" for row in attempt_rows
        ),
        "completed_conversations": completed,
        "questions_per_conversation": QUESTIONS_PER_CONVERSATION,
        "completed_questions": completed * QUESTIONS_PER_CONVERSATION,
        "totals": {
            "input_tokens": totals["prompt_tokens"],
            "completion_tokens_including_reasoning": totals[
                "completion_tokens"
            ],
            "visible_output_tokens": totals["visible_output_tokens"],
            "reasoning_tokens": totals["reasoning_tokens"],
            "cached_input_tokens": totals["cached_input_tokens"],
            "total_tokens": totals["total_tokens"],
            "cost_usd": totals["cost_usd"],
        },
        "averages_per_completed_conversation": {
            "input_tokens": averages["prompt_tokens"],
            "completion_tokens_including_reasoning": averages[
                "completion_tokens"
            ],
            "visible_output_tokens": averages["visible_output_tokens"],
            "reasoning_tokens": averages["reasoning_tokens"],
            "total_tokens": averages["total_tokens"],
            "cost_usd": averages["cost_usd"],
        },
        "successful_responses_only": {
            "input_tokens": successful_totals["prompt_tokens"],
            "completion_tokens_including_reasoning": successful_totals[
                "completion_tokens"
            ],
            "visible_output_tokens": successful_totals[
                "visible_output_tokens"
            ],
            "reasoning_tokens": successful_totals["reasoning_tokens"],
            "total_tokens": successful_totals["total_tokens"],
            "cost_usd": successful_totals["cost_usd"],
        },
        "linear_projections": projections,
        "projection_note": (
            "Projections amortize paid failed attempts over completed chats. "
            "Actual cost depends on routing, discounts, output length, and retries."
        ),
    }


def _dataset_card(summary: dict[str, Any], shard_count: int) -> str:
    totals = summary["totals"]
    averages = summary["averages_per_completed_conversation"]
    return f"""---
license: cc-by-sa-4.0
language:
- en
pretty_name: BEAM 50K Synthetic Compact Dataset
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.jsonl
---

# BEAM 50K Synthetic Compact Dataset

This is a wholly synthetic, machine-generated dataset inspired by the column
schema of `Mohammadta/BEAM`. It is not an official BEAM release and has not
received BEAM's human validation.

## Contents

- HF conversation records: {summary['completed_conversations']:,}
- Probing questions nested in those records: {summary['completed_questions']:,}
- Probing questions per conversation: {summary['questions_per_conversation']}
- Data shards: {shard_count}
- Generation route: OpenRouter
- Requested model: `{summary['model']}`

Each row contains the eight published BEAM columns: `conversation_id`,
`conversation_seed`, `narratives`, `user_profile`, `conversation_plan`,
`user_questions`, `chat`, and `probing_questions`. `probing_questions` is a
string-serialized dictionary containing two questions for each of ten memory
abilities.

## Generation usage

Usage is taken directly from OpenRouter response metadata. Totals include all
paid attempts, including responses rejected by local validation.

| Measure | Total | Amortized per completed conversation |
|---|---:|---:|
| Input tokens | {totals['input_tokens']:,} | {averages['input_tokens']:,.2f} |
| Completion tokens (including reasoning) | {totals['completion_tokens_including_reasoning']:,} | {averages['completion_tokens_including_reasoning']:,.2f} |
| Reasoning tokens | {totals['reasoning_tokens']:,} | {averages['reasoning_tokens']:,.2f} |
| Total tokens | {totals['total_tokens']:,} | {averages['total_tokens']:,.2f} |
| OpenRouter-reported cost | ${totals['cost_usd']:,.6f} | ${averages['cost_usd']:,.6f} |

See `metadata/openrouter_usage_summary.json` and
`metadata/openrouter_usage_by_conversation.jsonl` for details.

## Limitations

These are compact synthetic conversations, not BEAM's long-context samples.
All probes are machine-generated and should be human-reviewed before being
used for high-stakes evaluation. Health, legal, finance, privacy, and security
topics are framed as educational, organizational, or defensive assistance.
"""


def _customize_hf_record(
    dataset: dict[str, Any], topic: dict[str, Any]
) -> dict[str, Any]:
    record = to_hf_beam_record(dataset, conversation_seed=topic)
    record["narratives"] = (
        f"1. LABEL CATEGORY: {topic['category']}\n"
        f"   LABEL DESCRIPTION: {topic['theme']}\n\n"
        "2. LABEL CATEGORY: Long-Term Conversational Memory\n"
        "   LABEL DESCRIPTION: Updates, preferences, instructions, "
        "contradictions, event order, and temporal facts across sessions."
    )
    record["user_profile"] = {
        "user_info": (
            "USER PROFILE:\n"
            "- Role: Participant in an ongoing synthetic conversation\n"
            f"- Domain: {topic['category']}\n"
            f"- Goal: Work through {topic['title']} across multiple sessions."
        ),
        "user_relationships": (
            "RELATIONSHIPS:\n"
            "- Named collaborators and their responsibilities are recorded "
            "only in the synthetic chat."
        ),
    }
    validate_hf_beam_record(record)
    return record


def assemble_outputs(
    topics: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    shard_size: int,
) -> dict[str, Any]:
    if shard_size < 1:
        raise ValueError("shard_size must be at least 1")
    hf_records = []
    rows = []
    usage_rows = []
    missing = []
    for topic_index, topic in enumerate(topics):
        conversation_id = conversation_id_for(topic_index)
        path = _checkpoint_path(output_dir, conversation_id)
        if not path.exists():
            missing.append(conversation_id)
            continue
        dataset = json.loads(path.read_text(encoding="utf-8"))
        validate_dataset(dataset, expected_conversation_id=conversation_id)
        _validate_quality(dataset)
        hf_records.append(_customize_hf_record(dataset, topic))
        rows.extend(dataset["rows"])
        generation = dataset["generation"]
        usage_rows.append(
            {
                "conversation_id": conversation_id,
                "topic_id": topic["id"],
                "category": topic["category"],
                "requested_model": generation["requested_model"],
                "response_model": generation["response_model"],
                "upstream_provider": generation.get("upstream_provider"),
                "response_id": generation.get("response_id"),
                **generation["usage_metadata"],
            }
        )

    attempts = _load_jsonl(output_dir / "generation_attempts.jsonl")
    target_ids = {conversation_id_for(index) for index in range(len(topics))}
    target_attempts = [
        row for row in attempts if row.get("conversation_id") in target_ids
    ]
    summary = _usage_summary(usage_rows, target_attempts, model, len(topics))
    summary["missing_conversations"] = missing
    _write_json_atomic(output_dir / "openrouter_usage_summary.json", summary)
    if missing:
        raise RuntimeError(
            f"Cannot assemble: {len(missing)} conversations are missing"
        )
    if len(hf_records) != len(topics) or len(rows) != len(topics) * 20:
        raise RuntimeError(
            f"Expected {len(topics)} conversations and {len(topics) * 20} "
            f"probes; found {len(hf_records)} and {len(rows)}"
        )

    normalized_questions = [
        re.sub(r"\s+", " ", row["question"]).strip().casefold() for row in rows
    ]
    summary["question_text_duplicates_across_conversations"] = (
        len(normalized_questions) - len(set(normalized_questions))
    )

    hub_dir = output_dir / (
        "hub_50k"
        if len(topics) == EXPECTED_CONVERSATIONS
        else f"hub_smoke_{len(topics) * QUESTIONS_PER_CONVERSATION}"
    )
    data_dir = hub_dir / "data"
    metadata_dir = hub_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    shard_count = (len(hf_records) + shard_size - 1) // shard_size
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        records = hf_records[start : start + shard_size]
        shard_path = data_dir / (
            f"train-{shard_index + 1:05d}-of-{shard_count:05d}.jsonl"
        )
        shard_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        shard_paths.append(str(shard_path))

    (output_dir / "probes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (metadata_dir / "openrouter_usage_by_conversation.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in usage_rows
        ),
        encoding="utf-8",
    )
    (metadata_dir / "generation_attempts.jsonl").write_text(
        "".join(
            json.dumps(_sanitized_attempt_row(row), ensure_ascii=False) + "\n"
            for row in target_attempts
        ),
        encoding="utf-8",
    )
    _write_json_atomic(metadata_dir / "openrouter_usage_summary.json", summary)
    (metadata_dir / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (hub_dir / "README.md").write_text(
        _dataset_card(summary, shard_count), encoding="utf-8"
    )
    return {
        "status": "ok",
        "conversations": len(hf_records),
        "probes": len(rows),
        "shards": len(shard_paths),
        "hub_directory": str(hub_dir),
        "probes_output": str(output_dir / "probes.jsonl"),
        "usage_summary_output": str(
            metadata_dir / "openrouter_usage_summary.json"
        ),
        "usage": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--target-conversations", type=int, default=EXPECTED_CONVERSATIONS
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
    )
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--abort-failure-ratio", type=float, default=0.8)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument(
        "--confirm-generation",
        action="store_true",
        help="Required acknowledgment that each missing chat makes a paid request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8"))
    if not isinstance(topics, list):
        raise SystemExit("Topics file must contain a JSON array")
    if not 1 <= args.target_conversations <= EXPECTED_CONVERSATIONS:
        raise SystemExit(
            f"--target-conversations must be between 1 and {EXPECTED_CONVERSATIONS}"
        )
    if len(topics) < args.target_conversations:
        raise SystemExit(
            f"Need {args.target_conversations} topics; found {len(topics)}"
        )
    if args.workers < 1 or args.request_batch_size < 1:
        raise SystemExit("--workers and --request-batch-size must be at least 1")
    if args.max_passes < 1:
        raise SystemExit("--max-passes must be at least 1")
    topics = topics[: args.target_conversations]

    attempts_path = args.output_dir / "generation_attempts.jsonl"
    attempts = _load_jsonl(attempts_path)
    next_attempts = _next_attempts(attempts)
    pending = _pending_topics(topics, args.output_dir)
    if pending and not args.assemble_only:
        if not args.confirm_generation:
            raise SystemExit(
                f"{len(pending)} chats are missing and require paid OpenRouter "
                "requests. Re-run with --confirm-generation."
            )
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Missing OPENROUTER_API_KEY environment variable")
        initial_completed = len(topics) - len(pending)
        for pass_number in range(1, args.max_passes + 1):
            pending = _pending_topics(topics, args.output_dir)
            if not pending:
                break
            print(
                json.dumps(
                    {
                        "event": "generation_pass_started",
                        "pass": pass_number,
                        "pending": len(pending),
                        "target": len(topics),
                    }
                ),
                flush=True,
            )
            try:
                _run_generation_pass(
                    pending,
                    args,
                    api_key,
                    attempts_path,
                    next_attempts,
                    initial_completed,
                    len(topics),
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr, flush=True)
                break
            initial_completed = len(topics) - len(
                _pending_topics(topics, args.output_dir)
            )
            if _pending_topics(topics, args.output_dir) and pass_number < args.max_passes:
                time.sleep(args.retry_delay)

    pending = _pending_topics(topics, args.output_dir)
    if pending:
        summary = _usage_summary(
            [],
            [
                row
                for row in _load_jsonl(attempts_path)
                if row.get("conversation_id")
                in {conversation_id_for(index) for index in range(len(topics))}
            ],
            args.model,
            len(topics),
        )
        summary["missing_conversations"] = [item[2] for item in pending]
        _write_json_atomic(args.output_dir / "openrouter_usage_summary.json", summary)
        raise SystemExit(
            f"{len(pending)} conversations remain missing after this run; resume "
            "with the same command"
        )

    try:
        result = assemble_outputs(
            topics, args.output_dir, args.model, args.shard_size
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
