"""Safe orchestration for a 50-conversation, 1,000-probe BEAM-64K draft.

Examples:
    python -m src.beam.generate_64k --stage prepare
    python -m src.beam.generate_64k --stage preflight --provider gemini
    python -m src.beam.generate_64k --stage all --provider gemini --confirm-generation

Paid stages never run unless --confirm-generation is present. API keys are read
from environment variables and are never accepted as command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.beam.assemble_64k import QUESTION_TYPES, assemble_dataset
from src.beam.generation_settings import (
    CHAT_TOKEN_MINIMUMS,
    get_plan_settings,
    get_question_settings,
    get_token_limit,
    normalize_domain,
)
from src.beam.prepare_64k_topics import prepare_topics


DEFAULT_CHATS_DIR = Path("chats/64K")
EXPECTED_CONVERSATIONS = 50
EXPECTED_FINAL_QUESTIONS = 1_000
CHAT_TOKEN_TARGET = 64_000
PAID_STAGES = {"smoke-test", "plan", "question", "answer", "probes"}
PIPELINE_STAGES = ("plan", "question", "answer", "convert", "truncate", "probes", "assemble")
HARD_STAGES = frozenset({"plan", "answer"})
EASY_STAGES = frozenset({"question", "probes"})
DEFAULT_WORKER_MODEL = "deepseek/deepseek-v4.1-flash"


class _TimedLLM:
    """Thread-safe timing and token-usage proxy around a LangChain chat model."""

    def __init__(self, llm):
        self._llm = llm
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._llm, name)

    @staticmethod
    def _usage(response) -> dict[str, Any]:
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        input_tokens = (
            usage.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )
        total_tokens = (
            usage.get("total_tokens")
            or token_usage.get("total_tokens")
            or input_tokens + output_tokens
        )
        input_details = (
            token_usage.get("prompt_tokens_details")
            or token_usage.get("input_token_details")
            or usage.get("input_token_details")
            or {}
        )
        output_details = (
            token_usage.get("completion_tokens_details")
            or token_usage.get("output_token_details")
            or usage.get("output_token_details")
            or {}
        )
        raw_cost = token_usage.get("cost")
        if raw_cost is None:
            raw_cost = metadata.get("cost")
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "cached_input_tokens": int(
                input_details.get("cached_tokens")
                or input_details.get("cache_read")
                or 0
            ),
            "reasoning_tokens": int(
                output_details.get("reasoning_tokens")
                or output_details.get("reasoning")
                or 0
            ),
            "cost_usd": float(raw_cost) if raw_cost is not None else None,
            "response_id": metadata.get("id"),
            "response_model": metadata.get("model_name"),
            "upstream_provider": metadata.get("provider"),
            "service_tier": metadata.get("service_tier"),
        }

    def invoke(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            response = self._llm.invoke(*args, **kwargs)
        except Exception as exc:
            record = {
                "seconds": time.perf_counter() - started,
                "ok": False,
                "error_type": type(exc).__name__,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": None,
                "response_id": None,
                "response_model": None,
                "upstream_provider": None,
                "service_tier": None,
            }
            with self._lock:
                self._records.append(record)
            raise
        record = {
            "seconds": time.perf_counter() - started,
            "ok": True,
            **self._usage(response),
        }
        with self._lock:
            self._records.append(record)
        return response

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        durations = [record["seconds"] for record in records]
        successful = [record for record in records if record["ok"]]
        cost_records = [
            record for record in successful if record.get("cost_usd") is not None
        ]
        response_models = sorted({
            str(record["response_model"])
            for record in successful
            if record.get("response_model")
        })
        upstream_providers = sorted({
            str(record["upstream_provider"])
            for record in successful
            if record.get("upstream_provider")
        })
        service_tiers = sorted({
            str(record["service_tier"])
            for record in successful
            if record.get("service_tier")
        })
        return {
            "request_count": len(records),
            "successful_request_count": len(successful),
            "failed_request_count": len(records) - len(successful),
            "request_seconds_sum": sum(durations),
            "request_seconds_mean": sum(durations) / len(durations) if durations else 0.0,
            "request_seconds_max": max(durations, default=0.0),
            "input_tokens": sum(record["input_tokens"] for record in successful),
            "output_tokens": sum(record["output_tokens"] for record in successful),
            "total_tokens": sum(record["total_tokens"] for record in successful),
            "cached_input_tokens": sum(
                record["cached_input_tokens"] for record in successful
            ),
            "reasoning_tokens": sum(
                record["reasoning_tokens"] for record in successful
            ),
            "cost_usd": sum(record["cost_usd"] for record in cost_records),
            "cost_reported_request_count": len(cost_records),
            "response_models": response_models,
            "upstream_providers": upstream_providers,
            "service_tiers": service_tiers,
        }


def _is_transient_provider_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in (
        "429",
        "rate limit",
        "rate-limit",
        "temporarily",
        "timeout",
        "timed out",
        "502",
        "503",
        "504",
        "connection reset",
    ))


class _RetryingLLM:
    """Retry transient provider failures without losing the current BEAM turn."""

    def __init__(self, llm, max_attempts: int = 6):
        self._llm = llm
        self._max_attempts = max_attempts

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def invoke(self, *args, **kwargs):
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as exc:
                if (
                    not _is_transient_provider_error(exc)
                    or attempt == self._max_attempts
                ):
                    raise
                delay = min(2 ** (attempt - 1), 12)
                print(
                    f"Transient provider error; retrying request "
                    f"{attempt + 1}/{self._max_attempts} in {delay}s"
                )
                time.sleep(delay)


def _stage_routes(
    generation_mode: str,
    primary_config: dict[str, Any],
    worker_config: dict[str, Any],
    primary_reasoning_effort: str | None,
    worker_reasoning_effort: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    """Route globally dependent work to Gemini and routine fan-out to DeepSeek."""

    if generation_mode == "single":
        return (
            {stage: primary_config for stage in HARD_STAGES | EASY_STAGES},
            {
                stage: primary_reasoning_effort
                for stage in HARD_STAGES | EASY_STAGES
            },
        )
    if generation_mode != "hybrid":
        raise ValueError(f"Unknown generation mode: {generation_mode}")
    return (
        {
            "plan": primary_config,
            "answer": primary_config,
            "question": worker_config,
            "probes": worker_config,
        },
        {
            "plan": primary_reasoning_effort,
            "answer": primary_reasoning_effort,
            "question": worker_reasoning_effort,
            "probes": worker_reasoning_effort,
        },
    )


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _prepare_requested_topics(args, reuse_existing: bool = False) -> list[dict]:
    """Install a supplied manifest or build the default 50-chat pilot."""

    installed_path = args.chats_dir / "topics.json"
    if args.topics_file is None and reuse_existing and installed_path.exists():
        with installed_path.open(encoding="utf-8") as handle:
            topics = json.load(handle)
        if isinstance(topics, list) and topics:
            return topics
    if args.topics_file is None:
        return prepare_topics(chat_output=args.chats_dir / "topics.json")
    with args.topics_file.open(encoding="utf-8") as handle:
        topics = json.load(handle)
    if not isinstance(topics, list) or not topics:
        raise ValueError("--topics-file must contain a non-empty JSON array")
    required = {"id", "category", "title", "theme", "subtopics"}
    for index, topic in enumerate(topics):
        missing = required - set(topic)
        if missing:
            raise ValueError(
                f"Topic index {index} is missing: {', '.join(sorted(missing))}"
            )
    ids = [str(topic["id"]) for topic in topics]
    if len(ids) != len(set(ids)):
        raise ValueError("Topic IDs must be unique")
    _atomic_json_dump(topics, installed_path)
    return topics


def _record_chat_stage(chat_directory: Path, stage: str, record: dict[str, Any]) -> None:
    path = chat_directory / "timings.json"
    report = {
        "schema_version": 2,
        "conversation_id": chat_directory.name,
        "stages": {},
        "stage_attempts": {},
    }
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            pass
    report["schema_version"] = 2
    stages = report.setdefault("stages", {})
    attempts = report.setdefault("stage_attempts", {})
    stage_attempts = attempts.setdefault(stage, [])
    existing = stages.get(stage)
    if existing and not stage_attempts:
        stage_attempts.append(existing)
    stage_attempts.append(record)

    # A resumability check must not erase the API usage from the generation it
    # discovered. Keep the latest substantive attempt as the canonical stage.
    if record.get("status") != "already complete" or existing is None:
        stages[stage] = record
    else:
        stages[stage]["last_resume_check"] = {
            "status": record.get("status"),
            "wall_seconds": record.get("wall_seconds", 0.0),
        }
    report["wall_seconds_sum"] = sum(
        item.get("wall_seconds", 0.0)
        for name, item in report["stages"].items()
        if name not in {"optimized_pipeline", "complete_pipeline"}
    )
    _atomic_json_dump(report, path)


def _record_dataset_run(chats_directory: Path, record: dict[str, Any]) -> None:
    path = chats_directory / "pipeline_timings.json"
    report = {"schema_version": 1, "runs": []}
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            pass
    report.setdefault("runs", []).append(record)
    _atomic_json_dump(report, path)


def _seed_everything(seed: int):
    import numpy as np
    from faker import Faker
    from src.beam import profile_creation

    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    try:
        profile_creation.fake.seed_instance(seed)
    except AttributeError:
        pass


def _configure_generation_modules(llm):
    import src.llm as llm_module
    from src.beam import main as beam_main
    from src.beam import profile_creation

    for module in (llm_module, beam_main, profile_creation):
        module.llama_llm = llm
        module.qwen_llm = llm
        module.gpt_llm = llm


def _topic_at(chats_directory: str, index: int) -> dict:
    with open(os.path.join(chats_directory, "topics.json"), encoding="utf-8") as handle:
        topics = json.load(handle)
    return topics[index]


def _chat_paths(chats_directory: str, index: int) -> tuple[Path, Path, Path]:
    topic = _topic_at(chats_directory, index)
    chat_directory = Path(chats_directory) / str(topic["id"])
    plan_path = (
        chat_directory / "plan_new_trunecated.pickle"
        if (chat_directory / "plan_new_trunecated.pickle").exists()
        else chat_directory / "plan_new.pickle"
    )
    chat_path = (
        chat_directory / "chat_trunecated.json"
        if (chat_directory / "chat_trunecated.json").exists()
        else chat_directory / "chat.json"
    )
    return chat_directory, plan_path, chat_path


def _probe_type_valid(chat_directory: Path, question_type: str) -> bool:
    """Use the final assembler's rules to validate a probe checkpoint."""

    from src.beam.assemble_64k import (
        _load_candidates,
        _load_chat_ids,
        _select_two,
    )

    try:
        candidates = _load_candidates(chat_directory, question_type)
        _select_two(candidates, question_type, _load_chat_ids(chat_directory))
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _probe_files_complete(chat_directory: Path) -> bool:
    return all(
        _probe_type_valid(chat_directory, question_type)
        for question_type in QUESTION_TYPES
    )


def _plan_file_complete(chats_directory: str, index: int) -> bool:
    """Verify the structural contract needed by downstream BEAM stages."""

    from src.beam.main import extract_plan_bullets

    topic = _topic_at(chats_directory, index)
    domain = normalize_domain(topic["category"])
    expected_batches, expected_bullets = get_plan_settings("64K", domain)
    plan_path = (
        Path(chats_directory) / str(topic["id"]) / "plan_new.pickle"
    )
    if not plan_path.exists():
        return False
    try:
        with plan_path.open("rb") as handle:
            plans = pickle.load(handle)
        return (
            len(plans) == expected_batches
            and all(
                len(extract_plan_bullets(plan)) == expected_bullets + 3
                for plan in plans
            )
        )
    except (OSError, EOFError, pickle.PickleError, TypeError):
        return False


def _finished_chat_artifact_exists(chats_directory: str, index: int) -> bool:
    topic = _topic_at(chats_directory, index)
    chat_directory = Path(chats_directory) / str(topic["id"])
    metadata_path = chat_directory / "length_metadata.json"
    if not (chat_directory / "chat.json").exists() or not metadata_path.exists():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        return (
            CHAT_TOKEN_MINIMUMS["64K"]
            <= metadata["measured_tokens"]
            <= get_token_limit("64K")
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


def _raw_chat_reached_target(chats_directory: str, index: int) -> bool:
    """Recognize a finished answer checkpoint before local conversion."""

    from src.beam.main import _count_saved_chat_tokens

    topic = _topic_at(chats_directory, index)
    chat_path = Path(chats_directory) / str(topic["id"]) / "chat.pickle"
    if not chat_path.exists():
        return False
    try:
        with chat_path.open("rb") as handle:
            batches = pickle.load(handle)
        return _count_saved_chat_tokens(
            batches, "meta-llama/llama-3.1-8b-instruct"
        ) >= CHAT_TOKEN_TARGET
    except (OSError, EOFError, pickle.PickleError, TypeError, ValueError):
        return False


def _run_probes_one(chats_directory: str, index: int, probe_threads: int):
    from src.beam import main as beam_main

    topic = _topic_at(chats_directory, index)
    chat_directory, plan_path, chat_path = _chat_paths(chats_directory, index)
    invalid_types = [
        question_type
        for question_type in QUESTION_TYPES
        if not _probe_type_valid(chat_directory, question_type)
    ]
    if not invalid_types:
        return "already complete"
    if not plan_path.exists() or not chat_path.exists():
        raise FileNotFoundError(
            f"Conversation {topic['id']} must be converted and truncated before probes"
        )
    beam_main.run_probing_question_parallel(
        plan_address=str(plan_path),
        model="gpt",
        chat_address=str(chat_path),
        save_directory=str(chat_directory),
        domain=normalize_domain(topic["category"]),
        max_workers=probe_threads,
        question_types=[
            question_type
            for question_type in invalid_types
            if question_type != "abstention"
        ],
        generate_abstention="abstention" in invalid_types,
    )
    remaining = [
        question_type
        for question_type in QUESTION_TYPES
        if not _probe_type_valid(chat_directory, question_type)
    ]
    if remaining:
        raise RuntimeError(
            "Probe files remain structurally invalid: " + ", ".join(remaining)
        )
    return "generated"


def _paid_worker(
    stage: str,
    chats_directory: str,
    index: int,
    provider_config: dict[str, Any],
    reasoning_effort: str | None,
    seed: int,
    probe_threads: int,
    question_threads: int,
    answer_max_tokens: int,
) -> dict[str, Any]:
    from src.beam.run_pipeline import (
        get_chat_directory_for_index,
        run_answer_generation,
        run_plans_generation,
        run_question_generation,
    )
    from src.llm import build_provider_llm

    _seed_everything(seed + index)
    started = time.perf_counter()
    chat_directory = Path(get_chat_directory_for_index(chats_directory, index))
    timed_llm = None
    details = None
    try:
        temperature = 0.1 if stage == "answer" else 0.0
        base_llm = build_provider_llm(
            provider=provider_config["provider"],
            model_name=provider_config["model_name"],
            model_url=provider_config["model_url"],
            api_key_env=provider_config["api_key_env"],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_enabled=provider_config.get("reasoning_enabled"),
            provider_preferences=provider_config.get("provider_preferences"),
            max_retries=provider_config.get("max_retries", 6),
            request_timeout=provider_config.get("request_timeout", 300),
        )
        timed_llm = _TimedLLM(base_llm)
        generation_llm = (
            _RetryingLLM(timed_llm)
            if stage == "answer"
            else timed_llm
        )
        _configure_generation_modules(generation_llm)

        if stage == "plan":
            if _plan_file_complete(chats_directory, index):
                result = "already complete"
            else:
                run_plans_generation(chats_directory, "64K", index)
                result = "generated"
        elif stage == "question":
            if (chat_directory / "user_messages.pickle").exists():
                result = "already complete"
            else:
                details = run_question_generation(
                    chats_directory,
                    "64K",
                    index,
                    question_workers=question_threads,
                )
                result = "generated"
        elif stage == "answer":
            if (
                _finished_chat_artifact_exists(chats_directory, index)
                or _raw_chat_reached_target(chats_directory, index)
            ):
                result = "already complete"
            else:
                # Upstream answer_generation resumes from complete batches in
                # chat.pickle; the 64K adaptation stops at the retained target.
                run_answer_generation(
                    chats_directory,
                    "64K",
                    index,
                    generation_llm,
                    answer_max_tokens=answer_max_tokens,
                )
                result = "generated or resumed"
        elif stage == "probes":
            result = _run_probes_one(chats_directory, index, probe_threads)
        else:
            raise ValueError(f"Unknown paid stage: {stage}")
    except Exception:
        _record_chat_stage(chat_directory, stage, {
            "status": "failed",
            "wall_seconds": time.perf_counter() - started,
            "provider": provider_config["provider"],
            "model": provider_config["model_name"],
            "api": timed_llm.summary() if timed_llm is not None else {},
        })
        raise

    record = {
        "status": result,
        "wall_seconds": time.perf_counter() - started,
        "provider": provider_config["provider"],
        "model": provider_config["model_name"],
        "api": timed_llm.summary() if timed_llm is not None else {},
    }
    if details:
        record["details"] = details
    _record_chat_stage(chat_directory, stage, record)
    return {"index": index, "stage": stage, **record}


def _run_paid_stage(args,
                    stage: str,
                    provider_config: dict[str, Any],
                    reasoning_effort: str | None = None):
    if not args.confirm_generation:
        raise RuntimeError(
            f"Stage '{stage}' calls a paid model API. Re-run with --confirm-generation "
            "only after reviewing the preflight report."
        )
    key_name = provider_config["api_key_env"]
    if not os.getenv(key_name):
        raise RuntimeError(f"Missing API key. Set {key_name} in the environment.")

    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _paid_worker,
                stage,
                str(args.chats_dir),
                index,
                provider_config,
                reasoning_effort,
                args.seed,
                args.probe_threads,
                args.question_threads,
                args.answer_max_tokens,
            ): index
            for index in range(args.start_index, args.end_index)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                print(json.dumps(result))
            except Exception as exc:
                failures.append((index, exc))
                print(f"Stage {stage} failed at topic index {index}: {exc}", file=sys.stderr)
    if failures:
        indexes = ", ".join(str(index) for index, _ in failures)
        raise RuntimeError(f"Stage '{stage}' failed for topic indexes: {indexes}")
    wall_seconds = time.perf_counter() - started
    _record_dataset_run(args.chats_dir, {
        "kind": "stage",
        "stage": stage,
        "provider": provider_config["provider"],
        "model": provider_config["model_name"],
        "start_index": args.start_index,
        "end_index": args.end_index,
        "conversation_workers": args.workers,
        "question_threads_per_conversation": args.question_threads,
        "wall_seconds": wall_seconds,
    })
    print(json.dumps({"stage": stage, "stage_wall_seconds": wall_seconds}))


def _optimized_chat_worker(
    chats_directory: str,
    index: int,
    stage_configs: dict[str, dict[str, Any]],
    stage_reasoning_efforts: dict[str, str | None],
    seed: int,
    question_threads: int,
    probe_threads: int,
    answer_max_tokens: int,
    tokenizer_model: str,
    include_probes: bool,
    allow_short: bool,
) -> dict[str, Any]:
    """Run one complete chat pipeline while other chats progress independently."""

    started = time.perf_counter()
    paid_results = []
    for stage in ("plan", "question", "answer"):
        paid_results.append(
            _paid_worker(
                stage=stage,
                chats_directory=chats_directory,
                index=index,
                provider_config=stage_configs[stage],
                reasoning_effort=stage_reasoning_efforts[stage],
                seed=seed,
                probe_threads=1,
                question_threads=question_threads,
                answer_max_tokens=answer_max_tokens,
            )
        )

    chat_root = Path(chats_directory)
    local_results = []
    for stage, operation in (
        ("convert", lambda: _convert_one(chat_root, index)),
        ("truncate", lambda: _truncate_one(chat_root, index, tokenizer_model)),
    ):
        stage_started = time.perf_counter()
        result = operation()
        chat_directory, _, _ = _chat_paths(chats_directory, index)
        _record_chat_stage(chat_directory, stage, {
            "status": result.get("status", result.get("truncated", "complete")),
            "wall_seconds": time.perf_counter() - stage_started,
            "details": result,
        })
        local_results.append(result)

    topic = _topic_at(chats_directory, index)
    chat_directory = chat_root / str(topic["id"])
    with (chat_directory / "length_metadata.json").open(encoding="utf-8") as handle:
        length_metadata = json.load(handle)
    measured_tokens = length_metadata["measured_tokens"]
    if include_probes:
        if (
            measured_tokens < CHAT_TOKEN_MINIMUMS["64K"]
            or measured_tokens > get_token_limit("64K")
        ) and not allow_short:
            raise RuntimeError(
                f"Conversation {topic['id']} has {measured_tokens} tokens; "
                "probe generation requires the 64K acceptance band"
            )
        paid_results.append(
            _paid_worker(
                stage="probes",
                chats_directory=chats_directory,
                index=index,
                provider_config=stage_configs["probes"],
                reasoning_effort=stage_reasoning_efforts["probes"],
                seed=seed,
                probe_threads=probe_threads,
                question_threads=1,
                answer_max_tokens=answer_max_tokens,
            )
        )

    total_wall_seconds = time.perf_counter() - started
    pipeline_stage = "complete_pipeline" if include_probes else "optimized_pipeline"
    _record_chat_stage(chat_directory, pipeline_stage, {
        "status": "complete",
        "wall_seconds": total_wall_seconds,
        "question_threads": question_threads,
        "probe_threads": probe_threads if include_probes else 0,
        "measured_tokens": measured_tokens,
        "models": {
            stage: stage_configs[stage]["model_name"]
            for stage in ("plan", "question", "answer", "probes")
            if include_probes or stage != "probes"
        },
    })
    return {
        "index": index,
        "conversation_id": topic["id"],
        "status": "complete",
        "wall_seconds": total_wall_seconds,
        "measured_tokens": measured_tokens,
        "probes_generated": include_probes,
        "paid_stages": paid_results,
        "local_stages": local_results,
    }


def _run_optimized_chats(args,
                         stage_configs: dict[str, dict[str, Any]],
                         stage_reasoning_efforts: dict[str, str | None],
                         include_probes: bool = False) -> list[dict[str, Any]]:
    if not args.confirm_generation:
        raise RuntimeError(
            "Optimized chat generation calls a paid model API. Re-run with "
            "--confirm-generation only after reviewing preflight."
        )
    for key_name in {
        config["api_key_env"] for config in stage_configs.values()
    }:
        if not os.getenv(key_name):
            raise RuntimeError(f"Missing API key. Set {key_name} in the environment.")

    started = time.perf_counter()
    failures = []
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _optimized_chat_worker,
                str(args.chats_dir),
                index,
                stage_configs,
                stage_reasoning_efforts,
                args.seed,
                args.question_threads,
                args.probe_threads,
                args.answer_max_tokens,
                args.tokenizer_model,
                include_probes,
                args.allow_short,
            ): index
            for index in range(args.start_index, args.end_index)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({
                    "index": index,
                    "conversation_id": result["conversation_id"],
                    "status": result["status"],
                    "wall_seconds": result["wall_seconds"],
                    "measured_tokens": result["measured_tokens"],
                }))
            except Exception as exc:
                failures.append((index, exc))
                print(
                    f"Optimized chat pipeline failed at topic index {index}: {exc}",
                    file=sys.stderr,
                )

    wall_seconds = time.perf_counter() - started
    record = {
        "kind": "complete_chats" if include_probes else "optimized_chats",
        "start_index": args.start_index,
        "end_index": args.end_index,
        "conversation_workers": args.workers,
        "question_threads_per_conversation": args.question_threads,
        "maximum_in_flight_question_requests": args.workers * args.question_threads,
        "maximum_in_flight_probe_requests": (
            args.workers * args.probe_threads if include_probes else 0
        ),
        "models": {
            stage: stage_configs[stage]["model_name"]
            for stage in ("plan", "question", "answer", "probes")
            if include_probes or stage != "probes"
        },
        "wall_seconds": wall_seconds,
        "completed_conversations": len(results),
        "failed_indexes": [index for index, _ in failures],
    }
    _record_dataset_run(args.chats_dir, record)
    print(json.dumps(record, indent=2))
    if failures:
        indexes = ", ".join(str(index) for index, _ in failures)
        raise RuntimeError(f"Chat generation failed for topic indexes: {indexes}")
    return sorted(results, key=lambda result: result["index"])


def _run_smoke_test(
    args,
    provider_config: dict[str, Any],
    reasoning_effort: str | None = None,
):
    if not args.confirm_generation:
        raise RuntimeError(
            "The smoke test makes one model API call. Re-run with --confirm-generation."
        )
    key_name = provider_config["api_key_env"]
    if not os.getenv(key_name):
        raise RuntimeError(f"Missing API key. Set {key_name} in the environment.")

    from src.llm import build_provider_llm

    llm = build_provider_llm(
        provider=provider_config["provider"],
        model_name=provider_config["model_name"],
        model_url=provider_config["model_url"],
        api_key_env=provider_config["api_key_env"],
        temperature=0,
        reasoning_effort=reasoning_effort,
        reasoning_enabled=provider_config.get("reasoning_enabled"),
        provider_preferences=provider_config.get("provider_preferences"),
        max_retries=provider_config.get("max_retries", 6),
        request_timeout=provider_config.get("request_timeout", 300),
    )
    response = llm.invoke("Reply with exactly: BEAM_PROVIDER_OK")
    content = response.content if isinstance(response.content, str) else str(response.content)
    if "BEAM_PROVIDER_OK" not in content:
        raise RuntimeError(f"Provider responded, but smoke-test marker was missing: {content[:200]}")
    print(json.dumps({
        "provider": provider_config["provider"],
        "model": provider_config["model_name"],
        "status": "ok",
    }))


def _convert_one(chats_directory: Path, index: int) -> dict:
    from src.beam.utils import add_ids_to_chats, convert_chats_pickle_to_json

    topic = _topic_at(str(chats_directory), index)
    chat_directory = chats_directory / str(topic["id"])
    source = chat_directory / "chat.pickle"
    output = chat_directory / "chat.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing generated conversation: {source}")
    convert_chats_pickle_to_json(str(source), str(output))
    add_ids_to_chats(str(output), str(output))
    return {"index": index, "stage": "convert", "status": "converted"}


def _truncate_one(chats_directory: Path, index: int, tokenizer_model: str) -> dict:
    from src.beam.adjust_chats_length import count_chats_tokens

    topic = _topic_at(str(chats_directory), index)
    chat_directory = chats_directory / str(topic["id"])
    result = count_chats_tokens(
        chat_directory=str(chat_directory),
        model_name=tokenizer_model,
        token_limit=get_token_limit("64K"),
    )
    return {"index": index, "stage": "truncate", **result}


def _run_local_stage(args, stage: str):
    stage_started = time.perf_counter()
    for index in range(args.start_index, args.end_index):
        started = time.perf_counter()
        if stage == "convert":
            result = _convert_one(args.chats_dir, index)
        elif stage == "truncate":
            result = _truncate_one(args.chats_dir, index, args.tokenizer_model)
        else:
            raise ValueError(f"Unknown local stage: {stage}")
        chat_directory, _, _ = _chat_paths(str(args.chats_dir), index)
        _record_chat_stage(chat_directory, stage, {
            "status": result.get("status", result.get("truncated", "complete")),
            "wall_seconds": time.perf_counter() - started,
            "details": result,
        })
        print(json.dumps(result))
    wall_seconds = time.perf_counter() - stage_started
    _record_dataset_run(args.chats_dir, {
        "kind": "stage",
        "stage": stage,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "conversation_workers": 1,
        "wall_seconds": wall_seconds,
    })
    print(json.dumps({"stage": stage, "stage_wall_seconds": wall_seconds}))


def _length_report(chats_directory: Path, start_index: int, end_index: int) -> dict:
    records = []
    for index in range(start_index, end_index):
        topic = _topic_at(str(chats_directory), index)
        chat_directory = chats_directory / str(topic["id"])
        path = chat_directory / "length_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing length metadata: {path}")
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        records.append({"conversation_id": chat_directory.name, **metadata})

    minimum = CHAT_TOKEN_MINIMUMS["64K"]
    maximum = get_token_limit("64K")
    short = [item for item in records if item["measured_tokens"] < minimum]
    oversized = [item for item in records if item["measured_tokens"] > maximum]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "conversations": len(records),
        "short_conversation_ids": [item["conversation_id"] for item in short],
        "oversized_conversation_ids": [item["conversation_id"] for item in oversized],
        "ready_for_probes": not short and not oversized,
    }


def _preflight(
    args,
    stage_configs: dict[str, dict[str, Any]],
) -> dict:
    from src.llm import provider_preflight

    topics_path = args.chats_dir / "topics.json"
    if not topics_path.exists():
        raise FileNotFoundError(
            f"Missing {topics_path}. Run --stage prepare before preflight."
        )
    with topics_path.open(encoding="utf-8") as handle:
        topics = json.load(handle)
    if not 0 <= args.start_index < args.end_index <= len(topics):
        raise ValueError(
            f"Index range must satisfy 0 <= start < end <= {len(topics)}"
        )

    categories = {"general": 0, "coding": 0, "math": 0}
    for topic in topics:
        categories[normalize_domain(topic["category"])] += 1

    plan_settings = {
        domain: get_plan_settings("64K", domain) for domain in categories
    }
    question_settings = {
        domain: get_question_settings("64K", domain) for domain in categories
    }
    model_reports = {
        stage: provider_preflight(**stage_configs[stage])
        for stage in ("plan", "question", "answer", "probes")
    }
    report = {
        "schema_version": 1,
        # Retain the legacy field for older report consumers.
        "provider": model_reports["plan"],
        "models": model_reports,
        "routing": {
            "mode": args.generation_mode,
            "hard_stages": sorted(HARD_STAGES),
            "easy_stages": sorted(EASY_STAGES),
            "openrouter_worker_strategy": (
                "highest rolling throughput"
                if args.generation_mode == "hybrid"
                and args.worker_provider == "openrouter"
                and args.worker_provider_sort == "throughput"
                else "standard OpenRouter routing"
                if args.generation_mode == "hybrid"
                and args.worker_provider == "openrouter"
                else None
            ),
        },
        "dataset": {
            "chat_size": "64K",
            "conversation_token_minimum": CHAT_TOKEN_MINIMUMS["64K"],
            "conversation_token_limit": get_token_limit("64K"),
            "conversation_count": len(topics),
            "category_counts": categories,
            "final_questions": len(topics) * 20,
            "questions_per_conversation": 20,
            "questions_per_ability": len(topics) * 2,
            "plan_settings": plan_settings,
            "question_settings": question_settings,
        },
        "execution": {
            "start_index": args.start_index,
            "end_index": args.end_index,
            "process_workers": args.workers,
            "question_threads_per_process": args.question_threads,
            "maximum_in_flight_question_requests": args.workers * args.question_threads,
            "probe_threads_per_process": args.probe_threads,
            "maximum_in_flight_probe_requests": args.workers * args.probe_threads,
            "seed": args.seed,
            "confirmed": args.confirm_generation,
        },
    }
    with (args.chats_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "preflight",
            "smoke-test",
            *PIPELINE_STAGES,
            "length-check",
            "optimized-chats",
            "complete-chats",
            "chats",
            "all",
        ),
    )
    parser.add_argument("--provider", choices=("gemini", "openrouter", "openai-compatible"), default="gemini")
    parser.add_argument("--model-name")
    parser.add_argument("--model-url")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--generation-mode",
        choices=("single", "hybrid"),
        default="single",
        help=(
            "Use one model for every stage, or route hard planning/answering "
            "to the primary model and easy question/probe fan-out to a worker."
        ),
    )
    parser.add_argument(
        "--worker-provider",
        choices=("gemini", "openrouter", "openai-compatible"),
        default="openrouter",
        help="Provider for question and probe generation in hybrid mode",
    )
    parser.add_argument(
        "--worker-model-name",
        default=DEFAULT_WORKER_MODEL,
        help="Model for question and probe generation in hybrid mode",
    )
    parser.add_argument("--worker-model-url")
    parser.add_argument("--worker-api-key-env")
    parser.add_argument(
        "--worker-endpoint-provider",
        help=(
            "Pin hybrid worker calls to one OpenRouter endpoint provider, "
            "for example Groq"
        ),
    )
    parser.add_argument(
        "--worker-provider-sort",
        choices=("standard", "price", "throughput", "latency"),
        default="standard",
        help=(
            "OpenRouter endpoint sorting when the worker is not pinned; "
            "standard leaves OpenRouter's normal routing enabled"
        ),
    )
    parser.add_argument(
        "--worker-request-timeout",
        type=float,
        default=90,
        help="Per-attempt timeout in seconds for hybrid worker requests",
    )
    parser.add_argument(
        "--worker-max-retries",
        type=int,
        default=2,
        help="Transport-level retries for hybrid worker requests",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        default="low",
        help="Use 'none' for non-reasoning models such as Llama 3.1 Instruct",
    )
    parser.add_argument(
        "--worker-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        default="none",
        help="Reasoning effort for the hybrid question/probe worker model",
    )
    parser.add_argument("--chats-dir", type=Path, default=DEFAULT_CHATS_DIR)
    parser.add_argument(
        "--topics-file",
        type=Path,
        help=(
            "Optional JSON topic manifest to install into --chats-dir; use "
            "topics/50k_rows/2500_topics.json for the 50K-probe run."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=EXPECTED_CONVERSATIONS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--question-threads",
        type=int,
        default=8,
        help=(
            "Independent plan-indexed question requests per conversation. "
            "Peak question concurrency is --workers multiplied by this value."
        ),
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=1_000,
        help="Maximum tokens for each assistant answer or follow-up response",
    )
    parser.add_argument("--probe-threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--tokenizer-model",
        default="meta-llama/llama-3.1-8b-instruct",
        help="Tokenizer used to enforce the final 64K serialized-chat length",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow draft assembly below the configured 64K acceptance minimum",
    )
    parser.add_argument(
        "--confirm-generation",
        action="store_true",
        help="Explicitly authorize model API calls for paid generation stages",
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.question_threads < 1
        or args.probe_threads < 1
        or args.answer_max_tokens < 1
    ):
        parser.error(
            "--workers, --question-threads, --probe-threads, and "
            "--answer-max-tokens must be positive"
        )
    if args.worker_request_timeout <= 0 or args.worker_max_retries < 0:
        parser.error("--worker-request-timeout must be positive and retries non-negative")
    return args


def main():
    args = parse_args()

    if args.stage == "prepare":
        topics = _prepare_requested_topics(args)
        print(json.dumps({"prepared_topics": len(topics), "path": str(args.chats_dir / "topics.json")}, indent=2))
        return

    from src.llm import resolve_provider_config

    planner_config = resolve_provider_config(
        provider=args.provider,
        model_name=args.model_name,
        model_url=args.model_url,
        api_key_env=args.api_key_env,
    )
    if args.generation_mode == "hybrid":
        worker_config = resolve_provider_config(
            provider=args.worker_provider,
            model_name=args.worker_model_name,
            model_url=args.worker_model_url,
            api_key_env=args.worker_api_key_env,
        )
        provider_preferences = {}
        if args.worker_endpoint_provider:
            provider_preferences.update({
                "only": [args.worker_endpoint_provider],
                "allow_fallbacks": False,
            })
        elif args.worker_provider_sort != "standard":
            provider_preferences.update({
                "sort": args.worker_provider_sort,
                "allow_fallbacks": True,
            })
        worker_config.update({
            "max_retries": args.worker_max_retries,
            "request_timeout": args.worker_request_timeout,
        })
        if args.worker_reasoning_effort == "none" and args.worker_provider == "openrouter":
            worker_config["reasoning_enabled"] = False
        if provider_preferences:
            worker_config["provider_preferences"] = provider_preferences
    else:
        worker_config = planner_config

    planner_reasoning_effort = (
        None if args.reasoning_effort == "none" else args.reasoning_effort
    )
    selected_worker_effort = (
        args.worker_reasoning_effort
        if args.generation_mode == "hybrid"
        else args.reasoning_effort
    )
    worker_reasoning_effort = (
        None if selected_worker_effort == "none" else selected_worker_effort
    )
    stage_configs, stage_reasoning_efforts = _stage_routes(
        args.generation_mode,
        planner_config,
        worker_config,
        planner_reasoning_effort,
        worker_reasoning_effort,
    )
    if args.stage == "preflight":
        print(json.dumps(_preflight(args, stage_configs), indent=2))
        return
    if args.stage == "smoke-test":
        _preflight(args, stage_configs)
        _run_smoke_test(args, planner_config, planner_reasoning_effort)
        if worker_config != planner_config:
            _run_smoke_test(
                args, worker_config, worker_reasoning_effort
            )
        return
    if args.stage in PAID_STAGES:
        _preflight(args, stage_configs)
        _run_paid_stage(
            args,
            args.stage,
            stage_configs[args.stage],
            stage_reasoning_efforts[args.stage],
        )
        return
    if args.stage in {"convert", "truncate"}:
        _preflight(args, stage_configs)
        _run_local_stage(args, args.stage)
        return
    if args.stage == "length-check":
        print(json.dumps(_length_report(args.chats_dir, args.start_index, args.end_index), indent=2))
        return
    if args.stage == "assemble":
        manifest = assemble_dataset(
            chats_directory=args.chats_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            allow_short=args.allow_short,
        )
        print(json.dumps({
            "conversations": manifest["conversation_count"],
            "questions": manifest["question_count"],
            "validation_status": manifest["validation_status"],
        }, indent=2))
        return

    if args.stage in {"optimized-chats", "complete-chats"}:
        _prepare_requested_topics(args, reuse_existing=True)
        preflight = _preflight(args, stage_configs)
        print(json.dumps(preflight, indent=2))
        include_probes = args.stage == "complete-chats"
        _run_optimized_chats(
            args,
            stage_configs,
            stage_reasoning_efforts,
            include_probes=include_probes,
        )
        length_report = _length_report(
            args.chats_dir, args.start_index, args.end_index
        )
        print(json.dumps(length_report, indent=2))
        if include_probes:
            manifest = assemble_dataset(
                chats_directory=args.chats_dir,
                start_index=args.start_index,
                end_index=args.end_index,
                allow_short=args.allow_short,
            )
            from src.beam.report_64k_run import write_report

            report_paths = write_report(
                args.chats_dir, args.start_index, args.end_index
            )
            print(json.dumps({
                "conversations": manifest["conversation_count"],
                "questions": manifest["question_count"],
                "validation_status": manifest["validation_status"],
                **report_paths,
            }, indent=2))
        return

    if args.stage in {"chats", "all"}:
        _prepare_requested_topics(args, reuse_existing=True)
        preflight = _preflight(args, stage_configs)
        print(json.dumps(preflight, indent=2))
        for stage in ("plan", "question", "answer"):
            _run_paid_stage(
                args,
                stage,
                stage_configs[stage],
                stage_reasoning_efforts[stage],
            )
        _run_local_stage(args, "convert")
        _run_local_stage(args, "truncate")
        length_report = _length_report(args.chats_dir, args.start_index, args.end_index)
        print(json.dumps(length_report, indent=2))
        if not length_report["ready_for_probes"] and not args.allow_short:
            raise RuntimeError(
                "Length check failed. No probe API calls were made. Review the short conversations, "
                "then extend/regenerate them or explicitly use --allow-short for a non-final draft."
            )
        if args.stage == "chats":
            print(json.dumps({
                "status": "chat_generation_complete",
                **length_report,
            }, indent=2))
            return
        _run_paid_stage(
            args,
            "probes",
            stage_configs["probes"],
            stage_reasoning_efforts["probes"],
        )
        manifest = assemble_dataset(
            chats_directory=args.chats_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            allow_short=args.allow_short,
        )
        print(json.dumps({
            "conversations": manifest["conversation_count"],
            "questions": manifest["question_count"],
            "validation_status": manifest["validation_status"],
        }, indent=2))


if __name__ == "__main__":
    main()
