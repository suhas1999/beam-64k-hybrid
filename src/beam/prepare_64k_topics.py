"""Create the deterministic 50-conversation topic manifest for BEAM-64K."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


DEFAULT_SOURCE = Path("topics/100k/100k_topics.json")
DEFAULT_TOPIC_OUTPUT = Path("topics/64k/64k_topics.json")
DEFAULT_CHAT_OUTPUT = Path("chats/64K/topics.json")

VARIANT_FRAMINGS = (
    "Start from first principles and build a practical plan with explicit constraints.",
    "Revisit the topic over several sessions as the user's circumstances and priorities evolve.",
    "Emphasize trade-offs, corrections, and decisions that depend on earlier details.",
    "Treat the conversation as a realistic ongoing project with milestones and setbacks.",
)


def _category_group(topic: dict) -> str:
    category = topic["category"].strip().lower()
    if category in {"coding", "math"}:
        return category
    return "general"


def _expand_group(topics: list[dict], count: int, start_id: int) -> list[dict]:
    expanded = []
    for index in range(count):
        source = copy.deepcopy(topics[index % len(topics)])
        cycle = index // len(topics)
        source_topic_id = source.get("source_topic_id", source.get("id"))
        source["id"] = start_id + index
        source["source_topic_id"] = source_topic_id
        source["variant"] = cycle + 1
        framing = VARIANT_FRAMINGS[cycle % len(VARIANT_FRAMINGS)]
        source["theme"] = f"{source['theme']}. {framing}"
        subtopics = source.get("subtopics", [])
        if subtopics:
            rotation = cycle % len(subtopics)
            source["subtopics"] = subtopics[rotation:] + subtopics[:rotation]
        expanded.append(source)
    return expanded


def build_topics(source_topics: list[dict]) -> list[dict]:
    grouped = {"general": [], "coding": [], "math": []}
    for topic in source_topics:
        grouped[_category_group(topic)].append(topic)

    missing = [group for group, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"Source topics are missing categories: {', '.join(missing)}")

    # Recommended pilot mix: 40 general, 5 coding, and 5 math conversations.
    result = []
    result.extend(_expand_group(grouped["coding"], count=5, start_id=1))
    result.extend(_expand_group(grouped["math"], count=5, start_id=6))
    result.extend(_expand_group(grouped["general"], count=40, start_id=11))
    return result


def prepare_topics(
    source: Path = DEFAULT_SOURCE,
    topic_output: Path = DEFAULT_TOPIC_OUTPUT,
    chat_output: Path = DEFAULT_CHAT_OUTPUT,
) -> list[dict]:
    with source.open(encoding="utf-8") as handle:
        source_topics = json.load(handle)

    topics = build_topics(source_topics)
    for output in (topic_output, chat_output):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(topics, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return topics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--topic-output", type=Path, default=DEFAULT_TOPIC_OUTPUT)
    parser.add_argument("--chat-output", type=Path, default=DEFAULT_CHAT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generated = prepare_topics(args.source, args.topic_output, args.chat_output)
    counts = {"general": 0, "coding": 0, "math": 0}
    for item in generated:
        counts[_category_group(item)] += 1
    print(json.dumps({"topics": len(generated), "categories": counts}, indent=2))
