"""Benchmark one OpenRouter endpoint with real BEAM question/answer prompts."""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.beam.generate_64k import _TimedLLM
from src.beam.main import (
    _build_parallel_question_tasks,
    _invoke_parallel_question_task,
    ai_assistant_llm_template,
)
from src.llm import build_provider_llm


def _evenly_spaced(items: list, count: int) -> list:
    count = min(count, len(items))
    if count == 1:
        return [items[0]]
    indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indexes]


def _assess_question(candidate: str) -> dict:
    text = candidate.rsplit("->->", 1)[0].strip().strip("*")
    lowered = text.lower()
    concrete_markers = (
        "```", "`", ".py", "flask", "sql", "html", "css", "route",
        "database", "error", "function", "class", "api", "session", "cookie",
    )
    reasons = []
    if len(text) < 120 or len(text.split()) < 25:
        reasons.append("too_short")
    if lowered.startswith(("here is the question", "here are")):
        reasons.append("meta_preface")
    if not any(marker in lowered for marker in concrete_markers):
        reasons.append("not_concrete_coding")
    return {
        "pass": not reasons,
        "characters": len(text),
        "words": len(text.split()),
        "reasons": reasons,
    }


def _assess_answer(answer: str, finish_reason: str | None = None) -> dict:
    lowered = answer.lower()
    reasons = []
    if len(answer) < 500 or len(answer.split()) < 100:
        reasons.append("too_short")
    if not any(marker in answer for marker in ("```", "`")):
        reasons.append("no_code_or_inline_code")
    if answer.count("```") % 2:
        reasons.append("unclosed_code_fence")
    if finish_reason == "length":
        reasons.append("token_limit_truncation")
    if any(marker in lowered for marker in ("as an ai language model", "i cannot assist")):
        reasons.append("boilerplate_refusal")
    return {
        "pass": not reasons,
        "characters": len(answer),
        "words": len(answer.split()),
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="meta-llama/llama-3.1-8b-instruct"
    )
    parser.add_argument("--endpoint-provider")
    parser.add_argument(
        "--provider-sort",
        choices=("price", "throughput", "latency"),
        default="throughput",
    )
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument("--question-count", type=int, default=10)
    parser.add_argument("--answer-count", type=int, default=3)
    parser.add_argument("--answer-max-tokens", type=int, default=1_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    topic_data = json.loads((args.chat_directory / "topic.json").read_text())
    with (args.chat_directory / "plan_new.pickle").open("rb") as handle:
        plans = pickle.load(handle)
    topic = f"{topic_data['category']} -> {topic_data['title']}"
    theme = f"{topic_data['theme']} -> {', '.join(topic_data['subtopics'])}"
    tasks, _ = _build_parallel_question_tasks(
        topic, theme, plans, 3, 23, 1, "coding", True
    )
    selected = _evenly_spaced(tasks, args.question_count)

    provider_preferences = (
        {"only": [args.endpoint_provider], "allow_fallbacks": False}
        if args.endpoint_provider
        else {"sort": args.provider_sort, "allow_fallbacks": True}
    )
    base = build_provider_llm(
        provider="openrouter",
        model_name=args.model_name,
        reasoning_effort=None,
        reasoning_enabled=False if args.disable_reasoning else None,
        provider_preferences=provider_preferences,
        max_retries=0,
        request_timeout=args.timeout,
    )
    question_llm = _TimedLLM(base)
    question_started = time.perf_counter()
    results = {}
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(selected))) as executor:
        futures = {
            executor.submit(
                _invoke_parallel_question_task, task, question_llm, None, 2
            ): task
            for task in selected
        }
        for future in as_completed(futures):
            task = futures[future]
            key = f"{task['batch_idx']},{task['sub_batch_idx']}"
            try:
                results[key] = {"ok": True, **future.result()}
            except Exception as exc:
                results[key] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    question_wall = time.perf_counter() - question_started

    questions = []
    for key, result in results.items():
        if not result["ok"]:
            questions.append({"key": key, "pass": False, "reasons": ["request_failed"]})
            continue
        for candidate in result["candidates"]:
            assessment = _assess_question(candidate)
            questions.append({
                "key": key,
                "candidate": candidate,
                "pass": assessment["pass"],
                "assessment": assessment,
            })

    answer_inputs = [
        item for item in questions if item["pass"] and item.get("candidate")
    ][:args.answer_count]
    answer_system = (
        ai_assistant_llm_template
        .replace("<topic>", "N/A")
        .replace("<theme>", "N/A")
        .replace("<previous_plans_summary>", "")
        .replace("<previous_batches>", "N/A")
    )
    answers = []
    answer_llm = _TimedLLM(base)
    answer_started = time.perf_counter()
    for item in answer_inputs:
        started = time.perf_counter()
        raw_response = answer_llm.invoke(
            [
                SystemMessage(content=answer_system),
                HumanMessage(content=item["candidate"] + "\nPlease respond only in English."),
            ],
            max_tokens=args.answer_max_tokens,
        )
        response = raw_response.content
        finish_reason = (raw_response.response_metadata or {}).get("finish_reason")
        assessment = _assess_answer(response, finish_reason)
        answers.append({
            "question_key": item["key"],
            "question": item["candidate"],
            "answer": response,
            "seconds": time.perf_counter() - started,
            "finish_reason": finish_reason,
            "pass": assessment["pass"],
            "assessment": assessment,
        })
    answer_wall = time.perf_counter() - answer_started

    successful_latencies = [
        sum(result["request_seconds"])
        for result in results.values()
        if result["ok"]
    ]
    report = {
        "schema_version": 1,
        "model": args.model_name,
        "provider_preferences": provider_preferences,
        "reasoning_enabled": not args.disable_reasoning,
        "concurrency": args.concurrency,
        "question_benchmark": {
            "task_count": len(selected),
            "wall_seconds": question_wall,
            "latency_mean": statistics.mean(successful_latencies) if successful_latencies else 0,
            "latency_max": max(successful_latencies, default=0),
            "quality_pass_count": sum(item["pass"] for item in questions),
            "quality_total": len(questions),
            "results": results,
            "questions": questions,
        },
        "answer_benchmark": {
            "sample_count": len(answers),
            "wall_seconds": answer_wall,
            "quality_pass_count": sum(item["pass"] for item in answers),
            "answers": answers,
        },
        "api": {
            "questions": question_llm.summary(),
            "answers": answer_llm.summary(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "model": report["model"],
        "provider_preferences": report["provider_preferences"],
        "question_benchmark": {
            key: value for key, value in report["question_benchmark"].items()
            if key not in {"results", "questions"}
        },
        "answer_benchmark": {
            key: value for key, value in report["answer_benchmark"].items()
            if key != "answers"
        },
        "api": report["api"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
