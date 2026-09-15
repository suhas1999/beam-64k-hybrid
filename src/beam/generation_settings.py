"""Central generation settings, including the 64K benchmark preset."""

from __future__ import annotations


CHAT_TOKEN_LIMITS = {
    # The user-facing requirement is for the serialized chat itself to occupy a
    # 64K Llama context.  As in upstream BEAM, generation deliberately runs
    # long and complete turns are retained until the next one would cross the
    # limit.
    "64K": 65_536,
    "100K": 126_000,
    "500K": 499_000,
    "1M": 1_040_000,
    "10M": 9_990_000,
}

CHAT_TOKEN_MINIMUMS = {
    # A complete-turn boundary will almost never land on exactly 65,536.  Keep
    # the accepted result within roughly the final 5% of the requested window.
    "64K": 62_000,
}


PLAN_SETTINGS = {
    "64K": {
        # Reuse the published BEAM 100K generation density verbatim, then
        # truncate at the 64K complete-turn boundary.  Over-generation is what
        # makes the final length reliable across models with variable answers.
        "general": (5, 20),
        "coding": (3, 23),
        "math": (3, 25),
    },
    "100K": {
        "general": (5, 20),
        "coding": (3, 23),
        "math": (3, 25),
    },
    "500K": {
        "general": (10, 30),
        "coding": (10, 30),
        "math": (10, 30),
    },
    "1M": {
        "general": (10, 30),
        "coding": (10, 30),
        "math": (10, 30),
    },
    "10M": {
        "general": (10, 30),
        "coding": (10, 30),
        "math": (10, 30),
    },
}


QUESTION_SETTINGS = {
    "64K": {
        # (number of batches, ordinary bullets/messages per batch,
        #  sub-batches per batch). Special bullets are generated separately.
        "general": (5, 20, 10),
        "coding": (3, 23, 23),
        "math": (3, 25, 25),
    },
    "100K": {
        "general": (5, 20, 10),
        "coding": (3, 23, 23),
        "math": (3, 25, 25),
    },
    "500K": {
        "general": (10, 40, 10),
        "coding": (10, 30, 10),
        "math": (10, 40, 10),
    },
    "1M": {
        "general": (10, 90, 10),
        "coding": (10, 60, 10),
        "math": (10, 60, 10),
    },
    "10M": {
        "general": (10, 90, 10),
        "coding": (10, 60, 10),
        "math": (10, 60, 10),
    },
}


def normalize_domain(category: str) -> str:
    category = category.strip().lower()
    if category in {
        "coding",
        "software development",
        "data science and analytics",
        "it cloud and devops",
        "cybersecurity and privacy",
    }:
        return "coding"
    if category in {"math", "mathematics", "mathematics and statistics"}:
        return "math"
    return "general"


def _get_setting(table: dict, chat_size: str, category: str):
    if chat_size not in table:
        supported = ", ".join(table)
        raise ValueError(f"Unsupported chat size '{chat_size}'. Choose one of: {supported}")
    return table[chat_size][normalize_domain(category)]


def get_plan_settings(chat_size: str, category: str) -> tuple[int, int]:
    return _get_setting(PLAN_SETTINGS, chat_size, category)


def get_question_settings(chat_size: str, category: str) -> tuple[int, int, int]:
    return _get_setting(QUESTION_SETTINGS, chat_size, category)


def get_token_limit(chat_size: str) -> int:
    try:
        return CHAT_TOKEN_LIMITS[chat_size]
    except KeyError as exc:
        supported = ", ".join(CHAT_TOKEN_LIMITS)
        raise ValueError(
            f"Unsupported chat size '{chat_size}'. Choose one of: {supported}"
        ) from exc
