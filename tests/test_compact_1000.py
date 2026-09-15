import unittest

from src.beam.generate_compact_1000 import (
    QUESTION_TYPES,
    _normalize_redundant_fields,
    _usage_metadata,
    _usage_summary,
    compact_prompt,
)


class CompactDatasetTests(unittest.TestCase):
    def test_prompt_uses_topic_and_unique_ids(self):
        topic = {
            "category": "Math",
            "id": 7,
            "title": "Triangle Geometry",
            "theme": "Working through proofs",
            "subtopics": ["Pythagorean theorem", "similarity"],
        }
        prompt = compact_prompt("compact_0007", topic)
        self.assertIn("Triangle Geometry", prompt)
        self.assertIn("compact_0007_q01", prompt)
        self.assertIn("compact_0007_q20", prompt)
        for ability in QUESTION_TYPES:
            self.assertIn(f"- {ability}", prompt)

    def test_usage_metadata_defaults_missing_counts_to_zero(self):
        usage = _usage_metadata(
            {
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 200,
                    "thoughtsTokenCount": 50,
                    "totalTokenCount": 350,
                }
            }
        )
        self.assertEqual(usage["promptTokenCount"], 100)
        self.assertEqual(usage["cachedContentTokenCount"], 0)
        self.assertEqual(usage["totalTokenCount"], 350)

    def test_normalizes_only_missing_redundant_fields(self):
        dataset = {
            "metadata": {},
            "rows": [
                {"question": "one"},
                {
                    "row_id": "custom",
                    "conversation_id": "custom_conversation",
                    "validation_status": "custom_status",
                },
            ],
        }
        _normalize_redundant_fields(dataset, "compact_0001")
        self.assertEqual(dataset["metadata"]["conversation_id"], "compact_0001")
        self.assertEqual(dataset["rows"][0]["row_id"], "compact_0001_q01")
        self.assertEqual(dataset["rows"][0]["conversation_id"], "compact_0001")
        self.assertEqual(dataset["rows"][1]["row_id"], "custom")
        self.assertEqual(
            dataset["rows"][1]["conversation_id"], "custom_conversation"
        )

    def test_usage_summary_calculates_totals_averages_and_projections(self):
        rows = [
            {
                "promptTokenCount": 100,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 50,
                "totalTokenCount": 350,
            },
            {
                "promptTokenCount": 120,
                "candidatesTokenCount": 180,
                "thoughtsTokenCount": 40,
                "totalTokenCount": 340,
            },
        ]
        summary = _usage_summary(rows, "gemini-test", "2026-01-01T00:00:00Z")
        self.assertEqual(summary["totals"]["input_tokens"], 220)
        self.assertEqual(summary["totals"]["candidate_output_tokens"], 380)
        self.assertEqual(summary["totals"]["thinking_tokens"], 90)
        self.assertEqual(summary["totals"]["total_tokens"], 690)
        self.assertEqual(
            summary["linear_projections"]["100"]["total_tokens"],
            34_500,
        )

    def test_usage_summary_includes_failed_attempt_tokens(self):
        successful = [
            {
                "promptTokenCount": 100,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 50,
                "totalTokenCount": 350,
            }
        ]
        attempts = [
            {"status": "ok", "usage_metadata": successful[0]},
            {
                "status": "failed",
                "usage_metadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 100,
                    "thoughtsTokenCount": 25,
                    "totalTokenCount": 225,
                },
            },
        ]
        summary = _usage_summary(
            successful,
            "gemini-test",
            "2026-01-01T00:00:00Z",
            attempt_rows=attempts,
        )
        self.assertEqual(summary["api_attempts"], 2)
        self.assertEqual(summary["failed_attempts"], 1)
        self.assertEqual(summary["totals"]["total_tokens"], 575)
        self.assertEqual(
            summary["successful_responses_only"]["total_tokens"], 350
        )


if __name__ == "__main__":
    unittest.main()
