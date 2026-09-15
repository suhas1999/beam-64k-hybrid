"""Audit a generated chat against BEAM's published conversation structure."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.beam.adjust_chats_length import count_message_tokens
from src.beam.main import extract_plan_bullets


def load_reference(url: str) -> list:
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def audit_chat(chat: list, tokenizer_model: str) -> dict:
    messages = [
        message
        for batch in chat
        for turn in batch.get("turns", [])
        for message in turn
    ]
    turns = [turn for batch in chat for turn in batch.get("turns", [])]
    ids = [message.get("id") for message in messages]
    roles = Counter(message.get("role") for message in messages)
    question_types = Counter(
        message.get("question_type")
        for message in messages
        if message.get("question_type")
    )
    turn_token_counts = [
        count_message_tokens(turn, tokenizer_model) for turn in turns
    ]
    measured_tokens = sum(turn_token_counts)
    main_questions = [
        message
        for message in messages
        if message.get("question_type") == "main_question"
    ]
    return {
        "batch_count": len(chat),
        "turn_count": len(turns),
        "message_count": len(messages),
        "roles": dict(roles),
        "question_types": dict(question_types),
        "llama_tokens": measured_tokens,
        "tokens_per_main_turn": {
            "mean": round(mean(turn_token_counts), 2),
            "median": median(turn_token_counts),
            "minimum": min(turn_token_counts),
            "maximum": max(turn_token_counts),
        },
        "batch_schema_valid": all(
            set(batch) == {"batch_number", "turns", "time_anchor"}
            for batch in chat
        ),
        "turns_start_with_main_user": all(
            turn
            and turn[0].get("role") == "user"
            and turn[0].get("question_type") == "main_question"
            for turn in turns
        ),
        "roles_alternate": all(
            all(
                turn[position].get("role") != turn[position - 1].get("role")
                for position in range(1, len(turn))
            )
            for turn in turns
        ),
        "ids_are_contiguous": ids == list(range(len(ids))),
        "contents_nonempty": all(
            isinstance(message.get("content"), str)
            and bool(message["content"].strip())
            for message in messages
        ),
        "main_questions_plan_indexed": all(
            re.fullmatch(r"\d+,(?:\d+|N/A)", str(message.get("index")))
            and "->->" in message.get("content", "")
            for message in main_questions
        ),
    }


def audit_directory(chat_directory: Path, tokenizer_model: str) -> dict:
    with (chat_directory / "chat.json").open(encoding="utf-8") as handle:
        chat = json.load(handle)
    report = {"chat": audit_chat(chat, tokenizer_model)}

    with (chat_directory / "plan_new.pickle").open("rb") as handle:
        plans = pickle.load(handle)
    plan_bullets = [extract_plan_bullets(plan) for plan in plans]
    report["plan"] = {
        "batch_count": len(plans),
        "bullets_per_batch": [len(bullets) for bullets in plan_bullets],
        "special_labels_present": all(
            [bullet.split(":", 1)[0] for bullet in bullets[-3:]]
            == [
                "Information Update",
                "User Instruction",
                "Logical Contradiction",
            ]
            for bullets in plan_bullets
        ),
    }

    with (chat_directory / "user_messages.pickle").open("rb") as handle:
        user_batches = pickle.load(handle)
    report["user_questions"] = {
        "batch_count": len(user_batches),
        "questions_per_batch": [
            len(batch.get("messages", [[]])[0]) for batch in user_batches
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat_directory", type=Path)
    parser.add_argument(
        "--tokenizer-model",
        default="meta-llama/llama-3.1-8b-instruct",
    )
    parser.add_argument("--reference-url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_directory(args.chat_directory, args.tokenizer_model)
    if args.reference_url:
        report["reference_chat"] = audit_chat(
            load_reference(args.reference_url), args.tokenizer_model
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
