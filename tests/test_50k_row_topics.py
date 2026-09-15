import json
import tempfile
import unittest
from pathlib import Path

from src.beam.prepare_50k_row_topics import (
    DOMAIN_SPECS,
    EXPECTED_PROBES,
    EXPECTED_TOPICS,
    SCENARIO_VARIANTS,
    build_topics,
    prepare_topics,
    validate_topics,
)


class FiftyThousandRowTopicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topics = build_topics()

    def test_builds_exact_topic_and_probe_counts(self):
        report = validate_topics(self.topics)
        self.assertEqual(report["topics"], EXPECTED_TOPICS)
        self.assertEqual(report["projected_probe_rows"], EXPECTED_PROBES)
        self.assertEqual(report["base_topics"], 250)

    def test_balances_domains_and_scenario_variants(self):
        report = validate_topics(self.topics)
        self.assertEqual(report["categories"], len(DOMAIN_SPECS))
        self.assertTrue(
            all(
                count in {80, 90}
                for count in report["category_counts"].values()
            )
        )
        self.assertEqual(sum(report["category_counts"].values()), 2_500)
        self.assertEqual(report["variants"], len(SCENARIO_VARIANTS))
        self.assertTrue(
            all(count == 250 for count in report["variant_counts"].values())
        )

    def test_topics_are_unique_and_have_eight_subtopics(self):
        self.assertEqual(len({topic["title"] for topic in self.topics}), 2_500)
        for topic in self.topics:
            self.assertEqual(len(topic["subtopics"]), 8)
            self.assertEqual(len(set(topic["subtopics"])), 8)

    def test_writes_valid_manifest_and_report_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "topics.json"
            report_output = Path(directory) / "report.json"
            topics, report = prepare_topics(output, report_output)
            self.assertEqual(json.loads(output.read_text()), topics)
            self.assertEqual(json.loads(report_output.read_text()), report)
            self.assertFalse(report["generation_started"])


if __name__ == "__main__":
    unittest.main()
