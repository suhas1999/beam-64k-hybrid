import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.beam.generate_smoke_20 import (
    HF_ANSWER_FIELDS,
    HF_BEAM_FIELDS,
    QUESTION_TYPES,
    _call_gemini,
    _write_outputs,
    to_hf_beam_record,
    validate_dataset,
    validate_hf_beam_record,
)


def valid_dataset():
    conversation = []
    for index in range(1, 27):
        conversation.append(
            {
                "message_id": f"m{index:03d}",
                "session_id": f"s{min(((index - 1) // 6) + 1, 5):02d}",
                "timestamp": f"2026-01-{index:02d}T09:00:00Z",
                "role": "user" if index % 2 else "assistant",
                "content": f"Synthetic message {index}",
            }
        )

    rows = []
    for index in range(20):
        rows.append(
            {
                "row_id": f"smoke_0001_q{index + 1:02d}",
                "conversation_id": "smoke_0001",
                "ability": QUESTION_TYPES[index // 2],
                "difficulty": "easy" if index % 2 == 0 else "medium",
                "question": f"Distinct question {index + 1}?",
                "reference_answer": f"Answer {index + 1}",
                "source_message_ids": [f"m{index + 1:03d}"],
                "rubric": f"Must include answer {index + 1}",
                "validation_status": "machine_generated_smoke_test",
            }
        )
    return {
        "metadata": {
            "conversation_id": "smoke_0001",
            "requested_rows": 20,
        },
        "conversation": conversation,
        "rows": rows,
    }


class SmokeDatasetValidationTests(unittest.TestCase):
    def test_native_request_uses_header_and_not_url_query(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"candidates": []}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _call_gemini(
                api_key="test-secret",
                model="gemini-3.8-flash",
                timeout=17,
                max_output_tokens=100,
                temperature=0.2,
            )

        request = captured["request"]
        self.assertEqual(request.get_header("X-goog-api-key"), "test-secret")
        self.assertNotIn("test-secret", request.full_url)
        self.assertEqual(captured["timeout"], 17)

    def test_accepts_exactly_two_rows_for_each_ability(self):
        report = validate_dataset(valid_dataset())
        self.assertEqual(report["messages"], 26)
        self.assertEqual(report["sessions"], 5)
        self.assertEqual(report["rows"], 20)
        self.assertTrue(all(count == 2 for count in report["ability_counts"].values()))

    def test_rejects_missing_source_message(self):
        dataset = valid_dataset()
        dataset["rows"][0]["source_message_ids"] = ["m999"]
        with self.assertRaisesRegex(ValueError, "missing source message"):
            validate_dataset(dataset)

    def test_rejects_wrong_ability_distribution(self):
        dataset = valid_dataset()
        dataset["rows"][0]["ability"] = "summarization"
        with self.assertRaisesRegex(ValueError, "exactly two rows"):
            validate_dataset(dataset)

    def test_converts_to_hugging_face_beam_schema(self):
        record = to_hf_beam_record(valid_dataset())

        self.assertEqual(tuple(record), HF_BEAM_FIELDS)
        self.assertEqual(record["conversation_id"], "smoke_0001")
        self.assertEqual(len(record["chat"]), 5)
        self.assertEqual(
            [message["id"] for batch in record["chat"] for message in batch],
            list(range(26)),
        )
        self.assertEqual(record["chat"][0][0]["question_type"], "main_question")
        self.assertEqual(record["chat"][0][0]["index"], "1,1")
        self.assertIsNone(record["chat"][0][1]["index"])

        probes = ast.literal_eval(record["probing_questions"])
        self.assertEqual(tuple(probes), QUESTION_TYPES)
        for ability in QUESTION_TYPES:
            self.assertEqual(len(probes[ability]), 2)
            self.assertIn(HF_ANSWER_FIELDS[ability], probes[ability][0])
            self.assertIsInstance(probes[ability][0]["rubric"], list)

        report = validate_hf_beam_record(record)
        self.assertEqual(report["messages"], 26)
        self.assertEqual(report["sessions"], 5)
        self.assertEqual(report["probes"], 20)

    def test_writes_hugging_face_export_with_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.json"
            full_output, rows_output, hf_output = _write_outputs(valid_dataset(), output)

            self.assertTrue(full_output.exists())
            self.assertEqual(len(rows_output.read_text(encoding="utf-8").splitlines()), 20)
            hf_lines = hf_output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(hf_lines), 1)
            validate_hf_beam_record(json.loads(hf_lines[0]))


if __name__ == "__main__":
    unittest.main()
