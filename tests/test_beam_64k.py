import argparse
import json
import os
import pickle
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from src.beam.assemble_64k import QUESTION_TYPES, assemble_dataset
from src.beam.generate_64k import (
    _TimedLLM,
    _record_chat_stage,
    _run_paid_stage,
    _stage_routes,
)
from src.beam.report_64k_run import build_report
from src.beam.generation_settings import (
    get_plan_settings,
    get_question_settings,
    get_token_limit,
    normalize_domain,
)
from src.beam.prepare_64k_topics import build_topics
from src.llm import build_provider_llm, provider_preflight


class SettingsTests(unittest.TestCase):
    def test_64k_settings_reserve_context(self):
        self.assertEqual(get_token_limit("64K"), 65_536)
        self.assertEqual(get_plan_settings("64K", "general"), (5, 20))
        self.assertEqual(get_question_settings("64K", "coding"), (3, 23, 23))

    def test_topic_manifest_is_40_5_5(self):
        source = []
        for topic_id, category in enumerate(
            ["Coding", "Coding", "Math", "Math", "Lifestyle", "Learning"],
            start=1,
        ):
            source.append({
                "id": topic_id,
                "category": category,
                "title": f"Topic {topic_id}",
                "theme": "Theme",
                "subtopics": ["A", "B"],
            })
        topics = build_topics(source)
        counts = {"coding": 0, "math": 0, "general": 0}
        for topic in topics:
            category = topic["category"].lower()
            counts[category if category in {"coding", "math"} else "general"] += 1
        self.assertEqual(len(topics), 50)
        self.assertEqual(counts, {"coding": 5, "math": 5, "general": 40})
        self.assertEqual([topic["id"] for topic in topics], list(range(1, 51)))

    def test_expanded_topic_categories_route_to_specialized_prompts(self):
        self.assertEqual(normalize_domain("Software Development"), "coding")
        self.assertEqual(normalize_domain("Data Science and Analytics"), "coding")
        self.assertEqual(normalize_domain("Mathematics and Statistics"), "math")
        self.assertEqual(normalize_domain("Writing and Editing"), "general")


class ProviderTests(unittest.TestCase):
    def test_provider_defaults_do_not_expose_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "secret-value"}, clear=False):
            report = provider_preflight("gemini")
        self.assertEqual(report["model_name"], "gemini-3.8-flash")
        self.assertEqual(report["api_key_env"], "GEMINI_API_KEY")
        self.assertTrue(report["api_key_present"])
        self.assertNotIn("secret-value", json.dumps(report))

    def test_direct_gemini_and_openrouter_build_with_expected_routes(self):
        captured = []

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
        env = {
            "GEMINI_API_KEY": "gemini-secret",
            "OPENROUTER_API_KEY": "router-secret",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(
            sys.modules, {"langchain_openai": fake_module}
        ):
            build_provider_llm("gemini")
            build_provider_llm("openrouter")

        self.assertEqual(captured[0]["model"], "gemini-3.8-flash")
        self.assertIn("generativelanguage.googleapis.com", captured[0]["base_url"])
        self.assertEqual(captured[0]["reasoning_effort"], "low")
        self.assertEqual(captured[1]["model"], "google/gemini-3.8-flash")
        self.assertEqual(captured[1]["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(captured[1]["extra_body"], {"reasoning": {"effort": "low"}})
        self.assertEqual(
            captured[1]["default_headers"]["X-OpenRouter-Title"],
            "BEAM 64K Dataset Generator",
        )

    def test_paid_stage_requires_explicit_confirmation(self):
        args = argparse.Namespace(confirm_generation=False)
        with self.assertRaisesRegex(RuntimeError, "confirm-generation"):
            _run_paid_stage(
                args,
                "plan",
                {"api_key_env": "GEMINI_API_KEY"},
            )

    def test_timed_llm_collects_latency_and_token_usage(self):
        response = types.SimpleNamespace(
            content="ok",
            usage_metadata={
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "cost": 0.00125,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
                "model_name": "deepseek/deepseek-v4.1-flash",
            },
        )
        timed = _TimedLLM(types.SimpleNamespace(invoke=lambda *args, **kwargs: response))
        self.assertEqual(timed.invoke("hello").content, "ok")
        summary = timed.summary()
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["input_tokens"], 11)
        self.assertEqual(summary["output_tokens"], 7)
        self.assertEqual(summary["cached_input_tokens"], 3)
        self.assertEqual(summary["reasoning_tokens"], 2)
        self.assertEqual(summary["cost_usd"], 0.00125)
        self.assertEqual(summary["cost_reported_request_count"], 1)
        self.assertGreaterEqual(summary["request_seconds_sum"], 0)

    def test_hybrid_routes_hard_and_easy_stages(self):
        primary = {"model_name": "google/gemini-3.8-flash"}
        worker = {"model_name": "deepseek/deepseek-v4.1-flash"}
        configs, efforts = _stage_routes("hybrid", primary, worker, "low", None)
        self.assertIs(configs["plan"], primary)
        self.assertIs(configs["answer"], primary)
        self.assertIs(configs["question"], worker)
        self.assertIs(configs["probes"], worker)
        self.assertEqual(efforts["answer"], "low")
        self.assertIsNone(efforts["probes"])

    def test_resume_check_preserves_original_usage(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            chat = Path(temp_directory)
            original = {
                "status": "generated",
                "wall_seconds": 3.0,
                "api": {"total_tokens": 123, "cost_usd": 0.01},
            }
            _record_chat_stage(chat, "plan", original)
            _record_chat_stage(
                chat,
                "plan",
                {
                    "status": "already complete",
                    "wall_seconds": 0.1,
                    "api": {"total_tokens": 0, "cost_usd": 0.0},
                },
            )
            report = json.loads((chat / "timings.json").read_text())

        self.assertEqual(report["stages"]["plan"]["api"]["total_tokens"], 123)
        self.assertEqual(len(report["stage_attempts"]["plan"]), 2)

    def test_transient_provider_errors_are_retried(self):
        from src.beam.generate_64k import _RetryingLLM

        class FlakyLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, *args, **kwargs):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("429 temporarily rate-limited")
                return types.SimpleNamespace(content="recovered")

        flaky = FlakyLLM()
        with mock.patch("src.beam.generate_64k.time.sleep") as sleep:
            response = _RetryingLLM(flaky, max_attempts=4).invoke("hello")
        self.assertEqual(response.content, "recovered")
        self.assertEqual(flaky.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])


class ParallelQuestionTests(unittest.TestCase):
    @staticmethod
    def _plan(batch_number, date, marker):
        return f"""BATCH {batch_number} PLAN
• **Time Anchor:** {date}
• **Distinct Topic:** {marker} ordinary detail
• **Information Update:** {marker} updated value
• **User Instruction:** Always mention {marker} when summarizing
• **Logical Contradiction:** {marker} impossible event
"""

    def test_parallel_questions_restore_plan_order_and_exact_counts(self):
        from src.beam import main as beam_main

        class DelayedFakeLLM:
            def invoke(self, messages, **kwargs):
                prompt = messages[0].content
                focus = prompt.split("CURRENT FOCUS AREAS", 1)[1].split(
                    "## AVOID", 1
                )[0]
                numbers = [
                    int(match)
                    for match in __import__("re").findall(
                        r"(?:^|:\s|\n)(\d+)\)", focus
                    )
                ]
                # Force completion order to differ from plan order.
                time.sleep(0.002 * (6 - min(numbers)))
                signature = sum(ord(character) for character in focus)
                lines = [
                    " ".join(
                        f"sig{signature}n{number}unique{unique}"
                        for unique in range(20)
                    )
                    + f" bullet {number} token{number} detail{number} "
                    "realistic user request with enough specific context to "
                    "explain the problem, constraints, desired result, and "
                    f"important background clearly? ->-> {number}"
                    for number in numbers
                ]
                return types.SimpleNamespace(content="\n".join(lines))

        plans = [
            self._plan(1, "January 01, 2026", "alpha"),
            self._plan(2, "February 01, 2026", "beta"),
        ]
        with tempfile.TemporaryDirectory() as temp_directory, mock.patch.object(
            beam_main, "llama_llm", DelayedFakeLLM()
        ):
            output = Path(temp_directory) / "user_messages.pickle"
            metrics = beam_main.user_messages_generation_parallel(
                topic="Test topic",
                theme="Test theme",
                plans=plans,
                num_batches=2,
                sub_batches_per_batch=2,
                sub_batch_size=1,
                batch_size=2,
                llm_name="llama",
                save_address=str(output),
                domain="general",
                sepcial_bullets=True,
                max_workers=4,
            )
            with output.open("rb") as handle:
                messages = pickle.load(handle)

        self.assertEqual(metrics["task_count"], 6)
        self.assertEqual(metrics["message_count"], 10)
        self.assertEqual([len(batch["messages"][0]) for batch in messages], [5, 5])
        for batch_number, batch in enumerate(messages, start=1):
            suffixes = [message.rsplit("->->", 1)[1].strip() for message in batch["messages"][0]]
            self.assertEqual(
                suffixes,
                [
                    f"{batch_number},1",
                    f"{batch_number},2",
                    f"{batch_number},3",
                    f"{batch_number},4",
                    f"{batch_number},5",
                ],
            )

    def test_plan_parser_normalizes_header_dash_and_trailing_bullet(self):
        from src.beam.main import _parse_single_plan_batch

        parsed = _parse_single_plan_batch(
            """- **Time Anchor:** January 01, 2026
- **First Event:** one
- **Second Event:** two
- **Unrequested Extra:** discard me
""",
            batch_number=1,
            num_bullets=3,
        )
        self.assertEqual(
            parsed,
            """BATCH 1 PLAN
• **Time Anchor:** January 01, 2026
• **First Event:** one
• **Second Event:** two""",
        )

    def test_question_validator_rejects_marker_only_and_prompt_leaks(self):
        from src.beam.main import _question_candidate_errors

        task = {
            "domain": "coding",
            "focus": "1) **Implementation:** Build a Flask route",
        }
        self.assertTrue(_question_candidate_errors("->-> 1", task))
        self.assertTrue(_question_candidate_errors(
            "CURRENT FOCUS AREAS BULLET TYPE IDENTIFICATION "
            + "word " * 30
            + "```python\nprint('x')\n``` ->-> 1",
            task,
        ))

    def test_question_validator_requires_concrete_technical_detail(self):
        from src.beam.main import _question_candidate_errors

        task = {
            "domain": "coding",
            "focus": "1) **Implementation:** Build a Flask route",
        }
        without_detail = (
            "I'm working through this part of the project and need help deciding "
            "what to do next while respecting the constraints, schedule, user "
            "expectations, and desired result described above. ->-> 1"
        )
        with_detail = without_detail.replace(
            "this part of the project", "this Flask database route"
        )
        self.assertIn(
            "coding request lacks concrete technical detail",
            _question_candidate_errors(without_detail, task),
        )
        self.assertEqual(_question_candidate_errors(with_detail, task), [])

    def test_question_parser_recovers_messages_without_separator(self):
        from src.beam.main import _extract_parallel_question_candidates

        task = {"domain": "coding"}
        raw = (
            "First substantive Flask question with enough context to parse. ->-> 24\n\n"
            "Second substantive SQL question with enough context to parse. ->-> 26"
        )
        recovered = _extract_parallel_question_candidates(raw, task, [])
        self.assertEqual(len(recovered), 2)
        self.assertTrue(recovered[0].endswith("->-> 24"))
        self.assertTrue(recovered[1].endswith("->-> 26"))

    def test_question_marker_normalization_uses_known_plan_number(self):
        from src.beam.main import _normalize_question_plan_marker

        substantive = (
            "I'm updating my Flask dashboard fetch code and need help handling "
            "timeouts, JSON errors, and visible retry state for unreliable links."
        )
        self.assertEqual(
            _normalize_question_plan_marker(substantive + " ->-> 1", 18),
            substantive + " ->-> 18",
        )
        self.assertEqual(
            _normalize_question_plan_marker(substantive + " ->-> N", 18),
            substantive + " ->-> 18",
        )

    def test_question_task_plan_numbers_include_both_special_items(self):
        from src.beam.main import _task_plan_numbers

        task = {
            "expected": 2,
            "focus": "24) Update fact\n\n26) Contradictory fact",
        }
        self.assertEqual(_task_plan_numbers(task), [24, 26])


class AssemblyTests(unittest.TestCase):
    def test_assembly_selects_exactly_twenty_questions(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "topics.json").write_text(
                json.dumps([{"id": 1, "title": "Test", "category": "Lifestyle"}]),
                encoding="utf-8",
            )
            chat_directory = root / "1"
            chat_directory.mkdir()
            (chat_directory / "chat.json").write_text(
                json.dumps([{
                    "batch_number": 1,
                    "time_anchor": "January-01-2026",
                    "turns": [[
                        {"role": "user", "id": 0, "content": "A"},
                        {"role": "assistant", "id": 1, "content": "B"},
                    ]],
                }]),
                encoding="utf-8",
            )
            (chat_directory / "length_metadata.json").write_text(
                json.dumps({"measured_tokens": 64_000}), encoding="utf-8"
            )
            for question_type in QUESTION_TYPES:
                output_directory = chat_directory / "probing_questions" / question_type
                output_directory.mkdir(parents=True)
                candidates = []
                for index, difficulty in enumerate(("easy", "medium", "hard"), start=1):
                    candidate = {
                        "question": f"{question_type} question {index}?",
                        "answer": f"Answer {index}",
                        "difficulty": difficulty,
                        "rubric": [f"Criterion {index}"],
                    }
                    if question_type != "abstention":
                        candidate["source_chat_ids"] = [0]
                    candidates.append(candidate)
                (output_directory / f"{question_type}.json").write_text(
                    json.dumps(candidates), encoding="utf-8"
                )

            manifest = assemble_dataset(root, start_index=0, end_index=1)
            self.assertEqual(manifest["question_count"], 20)
            draft = json.loads(
                (chat_directory / "probing_questions" / "probing_questions.draft.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(all(len(draft[question_type]) == 2 for question_type in QUESTION_TYPES))
            self.assertTrue(
                all(
                    item["validation_status"] == "needs_human_review"
                    for values in draft.values()
                    for item in values
                )
            )


class ReportTests(unittest.TestCase):
    def test_report_aggregates_actual_usage_and_cost(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "topics.json").write_text(
                json.dumps([{"id": 1, "category": "General", "title": "Test"}]),
                encoding="utf-8",
            )
            chat = root / "1"
            (chat / "probing_questions").mkdir(parents=True)
            (chat / "length_metadata.json").write_text(
                json.dumps({"measured_tokens": 64_000}), encoding="utf-8"
            )
            (chat / "probing_questions" / "probing_questions.draft.json").write_text(
                json.dumps({ability: [{}, {}] for ability in QUESTION_TYPES}),
                encoding="utf-8",
            )
            api = {
                "request_count": 2,
                "successful_request_count": 2,
                "failed_request_count": 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cached_input_tokens": 4,
                "reasoning_tokens": 3,
                "cost_usd": 0.005,
                "cost_reported_request_count": 2,
            }
            (chat / "timings.json").write_text(
                json.dumps({
                    "stages": {
                        "plan": {
                            "status": "generated",
                            "wall_seconds": 2.5,
                            "provider": "openrouter",
                            "model": "google/gemini-3.8-flash",
                            "api": api,
                        },
                        "complete_pipeline": {"wall_seconds": 9.0},
                    }
                }),
                encoding="utf-8",
            )

            report = build_report(root, 0, 1)

        self.assertTrue(report["status"]["all_complete"])
        self.assertEqual(report["status"]["questions"], 20)
        self.assertEqual(report["usage"]["totals"]["total_tokens"], 120)
        self.assertEqual(report["usage"]["totals"]["cost_usd"], 0.005)
        self.assertTrue(report["usage"]["cost_is_complete"])
        self.assertEqual(
            report["projection_for_100_from_observed_average"]["cost_usd"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
