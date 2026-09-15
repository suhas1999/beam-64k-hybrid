"""LLM construction helpers used by BEAM generation and evaluation.

The upstream project eagerly constructed three clients at import time. That made
the generation modules impossible to import without populating every legacy
configuration entry. This module keeps the old global names for compatibility,
but creates them only when their configuration is complete. New generation
commands should use :func:`build_provider_llm` and environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderDefaults:
    model_name: str
    model_url: str
    api_key_env: str


PROVIDER_DEFAULTS = {
    "gemini": ProviderDefaults(
        model_name="gemini-3.8-flash",
        model_url=GEMINI_BASE_URL,
        api_key_env="GEMINI_API_KEY",
    ),
    "openrouter": ProviderDefaults(
        model_name="google/gemini-3.8-flash",
        model_url=OPENROUTER_BASE_URL,
        api_key_env="OPENROUTER_API_KEY",
    ),
    "openai-compatible": ProviderDefaults(
        model_name="",
        model_url="",
        api_key_env="LLM_API_KEY",
    ),
}


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    aliases = {
        "google": "gemini",
        "google-gemini": "gemini",
        "open-router": "openrouter",
        "openai": "openai-compatible",
        "custom": "openai-compatible",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROVIDER_DEFAULTS:
        supported = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ValueError(f"Unsupported provider '{provider}'. Choose one of: {supported}")
    return normalized


def resolve_provider_config(
    provider: str,
    model_name: str | None = None,
    model_url: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, str]:
    provider = normalize_provider(provider)
    defaults = PROVIDER_DEFAULTS[provider]
    resolved = {
        "provider": provider,
        "model_name": model_name or defaults.model_name,
        "model_url": model_url or defaults.model_url,
        "api_key_env": api_key_env or defaults.api_key_env,
    }
    if not resolved["model_name"]:
        raise ValueError(f"--model-name is required for provider '{provider}'")
    if not resolved["model_url"]:
        raise ValueError(f"--model-url is required for provider '{provider}'")
    return resolved


class BuildLLm:
    """Backward-compatible wrapper around LangChain's OpenAI-compatible client."""

    def __init__(
        self,
        model_url,
        model_name,
        api_key,
        temperature,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        n=None,
        extra_body=None,
        default_headers=None,
        model_kwargs=None,
        reasoning_effort=None,
        max_retries=6,
        request_timeout=300,
    ):
        self.model_url = model_url
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.llm = None
        self.top_p = top_p
        self.n = n
        self.extra_body = extra_body
        self.default_headers = default_headers
        self.model_kwargs = model_kwargs
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def build_llm(self):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
        }
        if self.model_url:
            kwargs["base_url"] = self.model_url
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.n is not None:
            kwargs["n"] = self.n
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.model_kwargs:
            kwargs["model_kwargs"] = self.model_kwargs
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        self.llm = ChatOpenAI(**kwargs)
        return self.llm

    def set_paramaters(self, model_url, model_name, api_key, temperature):
        self.model_url = model_url
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature

    def get_llm(self):
        return self.llm


def build_provider_llm(
    provider: str,
    model_name: str | None = None,
    model_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.1,
    reasoning_effort: str | None = "low",
    reasoning_enabled: bool | None = None,
    provider_preferences: dict[str, Any] | None = None,
    max_retries: int = 6,
    request_timeout: float = 300,
):
    """Build a direct Gemini, OpenRouter, or custom compatible client.

    Secrets are read from the selected environment variable at runtime. Their
    values are never returned or logged.
    """

    config = resolve_provider_config(
        provider=provider,
        model_name=model_name,
        model_url=model_url,
        api_key_env=api_key_env,
    )
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {config['api_key_env']} before running generation."
        )

    default_headers = None
    if config["provider"] == "openrouter":
        default_headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE", "BEAM 64K Dataset Generator")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-OpenRouter-Title"] = title

    extra_body: dict[str, Any] = {}
    direct_reasoning_effort = None
    if reasoning_enabled is False:
        if config["provider"] != "openrouter":
            raise ValueError("Explicit reasoning disable is supported only for OpenRouter")
        extra_body["reasoning"] = {"enabled": False}
    elif reasoning_effort:
        if config["provider"] == "openrouter":
            extra_body["reasoning"] = {"effort": reasoning_effort}
        else:
            direct_reasoning_effort = reasoning_effort
    if provider_preferences:
        if config["provider"] != "openrouter":
            raise ValueError(
                "Provider endpoint preferences are supported only for OpenRouter"
            )
        extra_body["provider"] = provider_preferences

    return BuildLLm(
        model_url=config["model_url"],
        model_name=config["model_name"],
        api_key=api_key,
        temperature=temperature,
        extra_body=extra_body or None,
        default_headers=default_headers,
        reasoning_effort=direct_reasoning_effort,
        max_retries=max_retries,
        request_timeout=request_timeout,
    ).build_llm()


def provider_preflight(
    provider: str,
    model_name: str | None = None,
    model_url: str | None = None,
    api_key_env: str | None = None,
    provider_preferences: dict[str, Any] | None = None,
    reasoning_enabled: bool | None = None,
    max_retries: int | None = None,
    request_timeout: float | None = None,
) -> dict[str, Any]:
    """Return non-secret provider details for a dry-run check."""

    config = resolve_provider_config(provider, model_name, model_url, api_key_env)
    report = {
        **config,
        "api_key_present": bool(os.getenv(config["api_key_env"])),
    }
    if provider_preferences:
        report["provider_preferences"] = provider_preferences
    if reasoning_enabled is not None:
        report["reasoning_enabled"] = reasoning_enabled
    if max_retries is not None:
        report["max_retries"] = max_retries
    if request_timeout is not None:
        report["request_timeout"] = request_timeout
    return report


english_only_regex = (
    r"^[\t\n\r -~"
    r"\u00A0-\u00FF"
    r"\u0370-\u03FF"
    r"\u2070-\u209F"
    r"\u2190-\u21FF"
    r"\u2200-\u22FF"
    r"]*$"
)


def _load_legacy_config() -> dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "llms_config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _optional_legacy_client(name: str, temperature: float, extra_body=None):
    config = _load_legacy_config().get(name, {})
    api_key = config.get("api_key")
    model_name = config.get("model_name")
    model_url = config.get("model_url")
    if name == "gpt":
        model_name = model_name or "gpt-4.1-mini"
    if not api_key or not model_name:
        return None
    return BuildLLm(
        model_url=model_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        extra_body=extra_body,
    ).build_llm()


# Compatibility names used throughout the original codebase. The new runner
# replaces these in each worker before invoking generation functions.
llama_llm = _optional_legacy_client("llama", temperature=0)
qwen_llm = _optional_legacy_client(
    "qwen", temperature=0, extra_body={"guided_regex": english_only_regex}
)
gpt_llm = _optional_legacy_client("gpt", temperature=0)
