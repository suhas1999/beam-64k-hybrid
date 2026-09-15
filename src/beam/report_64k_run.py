"""Build JSON and Markdown timing/token/cost reports for BEAM-64K runs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.beam.assemble_64k import QUESTION_TYPES
from src.beam.generation_settings import CHAT_TOKEN_MINIMUMS, get_token_limit


PAID_STAGES = ("plan", "question", "answer", "probes")
USAGE_FIELDS = (
    "request_count",
    "successful_request_count",
    "failed_request_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "cost_reported_request_count",
)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def _stage_records(timings: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    attempts = timings.get("stage_attempts", {}).get(stage)
    if isinstance(attempts, list) and attempts:
        return [record for record in attempts if isinstance(record, dict)]
    record = timings.get("stages", {}).get(stage)
    return [record] if isinstance(record, dict) else []


def _probe_counts(chat_directory: Path) -> dict[str, int]:
    draft = _read_json(
        chat_directory / "probing_questions" / "probing_questions.draft.json",
        {},
    )
    if not isinstance(draft, dict):
        return {ability: 0 for ability in QUESTION_TYPES}
    return {
        ability: len(draft.get(ability, []))
        if isinstance(draft.get(ability), list)
        else 0
        for ability in QUESTION_TYPES
    }


def _empty_usage() -> dict[str, int | float]:
    return {**{field: 0 for field in USAGE_FIELDS}, "cost_usd": 0.0}


def _add_usage(target: dict[str, Any], api: dict[str, Any]) -> None:
    for field in USAGE_FIELDS:
        target[field] += int(api.get(field, 0) or 0)
    target["cost_usd"] += float(api.get("cost_usd", 0.0) or 0.0)


def _matching_batch_wall(
    chats_directory: Path, start_index: int, end_index: int
) -> float | None:
    pipeline = _read_json(chats_directory / "pipeline_timings.json", {})
    runs = pipeline.get("runs", []) if isinstance(pipeline, dict) else []
    for run in reversed(runs):
        if (
            isinstance(run, dict)
            and run.get("kind") == "complete_chats"
            and run.get("start_index") == start_index
            and run.get("end_index") == end_index
        ):
            return float(run.get("wall_seconds", 0.0) or 0.0)
    return None


def build_report(
    chats_directory: Path, start_index: int, end_index: int
) -> dict[str, Any]:
    topics = _read_json(chats_directory / "topics.json")
    if not isinstance(topics, list):
        raise FileNotFoundError(f"Missing or invalid {chats_directory / 'topics.json'}")
    if not 0 <= start_index < end_index <= len(topics):
        raise ValueError(
            f"Index range must satisfy 0 <= start < end <= {len(topics)}"
        )

    generation_config = _read_json(chats_directory / "generation_config.json", {})
    stage_totals = {stage: _empty_usage() for stage in PAID_STAGES}
    stage_wall = {stage: 0.0 for stage in PAID_STAGES}
    stage_models: dict[str, set[tuple[str, str]]] = {
        stage: set() for stage in PAID_STAGES
    }
    conversations = []

    for topic_index, topic in enumerate(topics[start_index:end_index], start_index):
        conversation_id = str(topic["id"])
        chat_directory = chats_directory / conversation_id
        timings = _read_json(chat_directory / "timings.json", {})
        length = _read_json(chat_directory / "length_metadata.json", {})
        conversation_usage = _empty_usage()

        for stage in PAID_STAGES:
            for record in _stage_records(timings, stage):
                api = record.get("api", {})
                if not isinstance(api, dict):
                    continue
                _add_usage(stage_totals[stage], api)
                _add_usage(conversation_usage, api)
                stage_wall[stage] += float(record.get("wall_seconds", 0.0) or 0.0)
                provider = str(record.get("provider", "unknown"))
                model = str(record.get("model", "unknown"))
                stage_models[stage].add((provider, model))

        stages = timings.get("stages", {}) if isinstance(timings, dict) else {}
        pipeline_record = (
            stages.get("complete_pipeline") or stages.get("optimized_pipeline") or {}
        )
        ability_question_counts = _probe_counts(chat_directory)
        question_count = sum(ability_question_counts.values())
        measured = length.get("measured_tokens")
        abilities_complete = all(
            ability_question_counts[ability] == 2 for ability in QUESTION_TYPES
        )
        conversations.append({
            "topic_index": topic_index,
            "conversation_id": conversation_id,
            "category": topic.get("category"),
            "title": topic.get("title"),
            "measured_conversation_tokens": measured,
            "question_count": question_count,
            "ability_question_counts": ability_question_counts,
            "complete": (
                isinstance(measured, int)
                and CHAT_TOKEN_MINIMUMS["64K"]
                <= measured
                <= get_token_limit("64K")
                and abilities_complete
            ),
            "end_to_end_wall_seconds": pipeline_record.get("wall_seconds"),
            "api": conversation_usage,
        })

    totals = _empty_usage()
    for stage in PAID_STAGES:
        _add_usage(totals, stage_totals[stage])
    completed = sum(conversation["complete"] for conversation in conversations)
    count = len(conversations)
    measured_tokens = sum(
        int(conversation["measured_conversation_tokens"] or 0)
        for conversation in conversations
    )
    questions = sum(conversation["question_count"] for conversation in conversations)
    ability_question_counts = {
        ability: sum(
            conversation["ability_question_counts"][ability]
            for conversation in conversations
        )
        for ability in QUESTION_TYPES
    }
    successful = int(totals["successful_request_count"])
    cost_reported = int(totals["cost_reported_request_count"])
    cost_complete = successful == cost_reported

    average_divisor = count or 1
    averages = {
        "measured_conversation_tokens": measured_tokens / average_divisor,
        "questions": questions / average_divisor,
        "wall_seconds": (
            sum(
                float(conversation["end_to_end_wall_seconds"] or 0.0)
                for conversation in conversations
            )
            / average_divisor
        ),
        **{
            field: float(totals[field]) / average_divisor
            for field in (
                "request_count",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
            )
        },
    }
    projection_100 = {
        field: value * 100
        for field, value in averages.items()
        if field != "wall_seconds"
    }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(chats_directory),
        "range": {"start_index": start_index, "end_index": end_index},
        "routing": generation_config.get("routing"),
        "configured_models": generation_config.get("models"),
        "status": {
            "requested_conversations": count,
            "completed_conversations": completed,
            "questions": questions,
            "expected_questions": count * 20,
            "ability_question_counts": ability_question_counts,
            "expected_questions_per_ability": count * 2,
            "all_complete": completed == count,
        },
        "timing": {
            "batch_wall_seconds": _matching_batch_wall(
                chats_directory, start_index, end_index
            ),
            "sum_of_stage_wall_seconds": sum(stage_wall.values()),
            "stage_wall_seconds": stage_wall,
        },
        "usage": {
            "totals": totals,
            "cost_is_complete": cost_complete,
            "missing_cost_request_count": max(0, successful - cost_reported),
            "stage_totals": stage_totals,
            "stage_routes": {
                stage: [
                    {"provider": provider, "model": model}
                    for provider, model in sorted(stage_models[stage])
                ]
                for stage in PAID_STAGES
            },
        },
        "conversation_artifact_tokens": measured_tokens,
        "averages_per_conversation": averages,
        "projection_for_100_from_observed_average": projection_100,
        "conversations": conversations,
        "notes": [
            "cost_usd is the provider-reported billed OpenRouter cost; it is not a static estimate.",
            "Parallel batch wall time is not projected because provider quotas and throttling are nonlinear.",
            "Draft probes still require BEAM-style human validation before benchmark use.",
        ],
    }


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    status = report["status"]
    timing = report["timing"]
    totals = report["usage"]["totals"]
    projection = report["projection_for_100_from_observed_average"]
    cost_label = "complete" if report["usage"]["cost_is_complete"] else "partial"
    lines = [
        "# BEAM-64K generation report",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Result",
        "",
        "| Metric | Actual | 100-chat projection from observed average |",
        "|---|---:|---:|",
        f"| Completed conversations | {status['completed_conversations']} / {status['requested_conversations']} | 100 |",
        f"| Selected probe questions | {_number(status['questions'])} | {_number(projection['questions'])} |",
        f"| Conversation artifact tokens | {_number(report['conversation_artifact_tokens'])} | {_number(projection['measured_conversation_tokens'])} |",
        f"| API requests | {_number(totals['request_count'])} | {_number(projection['request_count'])} |",
        f"| Input tokens | {_number(totals['input_tokens'])} | {_number(projection['input_tokens'])} |",
        f"| Output tokens | {_number(totals['output_tokens'])} | {_number(projection['output_tokens'])} |",
        f"| Total API tokens | {_number(totals['total_tokens'])} | {_number(projection['total_tokens'])} |",
        f"| Provider-billed cost (USD, {cost_label}) | ${_number(totals['cost_usd'], 6)} | ${_number(projection['cost_usd'], 6)} |",
        f"| Parallel batch wall time | {_number(timing['batch_wall_seconds'])} s | not projected |",
        "",
        "## Stage routing and usage",
        "",
        "| Stage | Provider / model | Wall seconds | Requests | Input tokens | Output tokens | Cost USD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for stage in PAID_STAGES:
        usage = report["usage"]["stage_totals"][stage]
        routes = report["usage"]["stage_routes"][stage]
        route_text = "<br>".join(
            f"{route['provider']} / `{route['model']}`" for route in routes
        ) or "n/a"
        lines.append(
            f"| {stage} | {route_text} | {_number(timing['stage_wall_seconds'][stage])} "
            f"| {_number(usage['request_count'])} | {_number(usage['input_tokens'])} "
            f"| {_number(usage['output_tokens'])} | ${_number(usage['cost_usd'], 6)} |"
        )
    lines.extend([
        "",
        "## Probe category checks",
        "",
        "| Ability | Selected | Expected |",
        "|---|---:|---:|",
    ])
    for ability in QUESTION_TYPES:
        lines.append(
            f"| {ability} | {_number(status['ability_question_counts'][ability])} "
            f"| {_number(status['expected_questions_per_ability'])} |"
        )
    lines.extend([
        "",
        "## Conversation checks",
        "",
        "| ID | 64K artifact tokens | Probe questions | End-to-end seconds | Complete |",
        "|---|---:|---:|---:|:---:|",
    ])
    for conversation in report["conversations"]:
        lines.append(
            f"| {conversation['conversation_id']} "
            f"| {_number(conversation['measured_conversation_tokens'])} "
            f"| {_number(conversation['question_count'])} "
            f"| {_number(conversation['end_to_end_wall_seconds'])} "
            f"| {'yes' if conversation['complete'] else 'no'} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_report(
    chats_directory: Path,
    start_index: int,
    end_index: int,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict[str, str]:
    stem = f"run_{start_index:06d}_{end_index:06d}"
    report_directory = chats_directory / "reports"
    output_json = output_json or report_directory / f"{stem}.json"
    output_markdown = output_markdown or report_directory / f"{stem}.md"
    report = build_report(chats_directory, start_index, end_index)
    _write_atomic(
        output_json,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    _write_atomic(output_markdown, _markdown(report))
    return {
        "json_report": str(output_json),
        "markdown_report": str(output_markdown),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chats-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_report(
        args.chats_dir,
        args.start_index,
        args.end_index,
        args.output_json,
        args.output_markdown,
    ), indent=2))


if __name__ == "__main__":
    main()
