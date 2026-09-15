import tempfile
import unittest
from pathlib import Path

from src.beam.generate_openrouter_50k import (
    QUESTION_TYPES,
    _json_schema,
    _normalize_generated_dataset,
    _request_payload,
    _sanitized_attempt_row,
    _usage_summary,
    _usage_metadata,
    conversation_id_for,
    openrouter_prompt,
)


class OpenRouter50KTests(unittest.TestCase):
    def test_conversation_ids_are_stable_and_six_digit(self):
        self.assertEqual(conversation_id_for(0), "beam50k_000001")
        self.assertEqual(conversation_id_for(2499), "beam50k_002500")

    def test_prompt_strengthens_abstention_and_safety_rules(self):
        prompt = openrouter_prompt(
            "beam50k_000001",
            {
                "category": "Health and Wellness",
                "title": "Planning an appointment",
                "theme": "Organize questions",
                "subtopics": ["notes"],
            },
        )
        self.assertIn("not logically deducible", prompt)
        self.assertIn("qualified review", prompt)

    def test_schema_requires_exact_array_sizes_and_ability_enum(self):
        schema = _json_schema()
        self.assertEqual(schema["properties"]["conversation"]["minItems"], 26)
        self.assertEqual(schema["properties"]["conversation"]["maxItems"], 34)
        self.assertEqual(schema["properties"]["rows"]["minItems"], 20)
        row_schema = schema["properties"]["rows"]["items"]
        self.assertEqual(
            row_schema["properties"]["ability"]["enum"],
            list(QUESTION_TYPES),
        )

    def test_request_uses_json_output_and_minimal_reasoning(self):
        payload = _request_payload(
            "google/gemini-test", "prompt", 20_000, 0.2, "minimal"
        )
        self.assertEqual(payload["response_format"]["type"], "json_object")
        self.assertEqual(payload["plugins"], [{"id": "response-healing"}])
        self.assertEqual(payload["reasoning"]["effort"], "minimal")
        self.assertTrue(payload["provider"]["require_parameters"])

    def test_openrouter_usage_parsing(self):
        usage = _usage_metadata(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 250,
                    "total_tokens": 350,
                    "cost": 0.0042,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 50},
                }
            }
        )
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 250)
        self.assertEqual(usage["visible_output_tokens"], 200)
        self.assertEqual(usage["reasoning_tokens"], 50)
        self.assertEqual(usage["cached_input_tokens"], 20)
        self.assertEqual(usage["cost_usd"], 0.0042)

    def test_attempt_sanitizer_redacts_account_ids(self):
        row = {
            "error": 'bad {"user_id":"org_private","org_id": "another"}'
        }
        clean = _sanitized_attempt_row(row)
        self.assertNotIn("org_private", clean["error"])
        self.assertNotIn("another", clean["error"])

    def test_usage_summary_adds_unlogged_checkpoint_without_double_count(self):
        checkpoint_usage = {
            "conversation_id": "beam50k_000001",
            "response_id": "response-1",
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "visible_output_tokens": 200,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "total_tokens": 300,
            "cost_usd": 0.01,
        }
        summary = _usage_summary(
            [checkpoint_usage], [], "google/test", 1
        )
        self.assertEqual(summary["api_attempts"], 1)
        self.assertEqual(
            summary["successful_checkpoints_missing_attempt_log"], 1
        )
        self.assertEqual(summary["totals"]["total_tokens"], 300)

        attempt = {
            "conversation_id": "beam50k_000001",
            "attempt": 1,
            "status": "ok",
            "response_id": "response-1",
            "usage_metadata": checkpoint_usage,
        }
        summary = _usage_summary(
            [checkpoint_usage], [attempt], "google/test", 1
        )
        self.assertEqual(summary["api_attempts"], 1)
        self.assertEqual(
            summary["successful_checkpoints_missing_attempt_log"], 0
        )
        self.assertEqual(summary["totals"]["total_tokens"], 300)

    def test_normalization_preserves_source_references(self):
        rows = []
        for ability in QUESTION_TYPES:
            for number in range(2):
                rows.append(
                    {
                        "row_id": "wrong",
                        "conversation_id": "wrong",
                        "ability": ability,
                        "question": f"Question {ability} {number}?",
                        "reference_answer": "Answer",
                        "source_message_ids": ["old-1", "m0001"],
                        "rubric": "Answer",
                        "validation_status": "wrong",
                    }
                )
        dataset = {
            "metadata": {},
            "conversation": [
                {
                    "message_id": "old-1",
                    "session_id": "s1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "role": "user",
                    "content": "A message",
                }
            ],
            "rows": list(reversed(rows)),
        }
        result = _normalize_generated_dataset(dataset, "beam50k_000001")
        self.assertEqual(result["conversation"][0]["message_id"], "m001")
        self.assertEqual(result["rows"][0]["ability"], QUESTION_TYPES[0])
        self.assertEqual(
            result["rows"][0]["source_message_ids"], ["m001", "m001"]
        )
        self.assertEqual(result["rows"][0]["row_id"], "beam50k_000001_q01")
        self.assertEqual(
            result["metadata"]["conversation_id"], "beam50k_000001"
        )
        self.assertEqual(result["rows"][0]["difficulty"], "medium")

    def test_normalization_recovers_sessions_from_distinct_dates(self):
        dataset = {
            "metadata": {},
            "conversation": [
                {
                    "message_id": f"m{index:03d}",
                    "session_id": "s01",
                    "timestamp": f"2026-01-{index:02d}T09:00:00Z",
                    "role": "user",
                    "content": "A useful message",
                }
                for index in range(1, 6)
            ],
            "rows": [],
        }
        result = _normalize_generated_dataset(dataset, "beam50k_000001")
        self.assertEqual(
            [message["session_id"] for message in result["conversation"]],
            ["s01", "s02", "s03", "s04", "s05"],
        )


if __name__ == "__main__":
    unittest.main()
