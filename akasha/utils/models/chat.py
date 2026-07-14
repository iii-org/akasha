"""LangChain ChatModel factory used by the public Akasha model selectors."""

from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from akasha.utils.models.thinking import (
    ThinkingBudget,
    normalize_thinking_budget,
    normalize_thinking_level,
)


def _normalize_openai_base_url(base_url: str) -> str:
    """Convert a full OpenAI endpoint URL into an SDK base URL."""
    if not base_url:
        return base_url
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")] or "/"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return base_url


def build_chat_model(
    provider: str,
    model_name: str,
    env: Mapping[str, Any],
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    thinking: bool = False,
    thinking_budget: ThinkingBudget = None,
):
    """Build a LangChain 1.3+ chat model for a supported provider."""

    provider = provider.lower()
    normalized_budget = normalize_thinking_budget(
        thinking_budget,
        thinking=thinking,
        max_output_tokens=max_output_tokens,
    )
    thinking_level = normalize_thinking_level(thinking_budget)

    if provider in {"openai", "azure"}:
        from langchain_openai import ChatOpenAI

        azure_key = env.get("AZURE_OPENAI_API_KEY")
        azure_base_url = env.get("AZURE_OPENAI_BASE_URL")
        if bool(azure_key) != bool(azure_base_url):
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL must be provided together."
            )

        # Azure is selected explicitly by the ``azure:`` model alias.  Keep
        # the old implicit path only when no public OpenAI key is present.
        use_azure = provider == "azure" or (
            bool(azure_key) and bool(azure_base_url) and not env.get("OPENAI_API_KEY")
        )
        if use_azure:
            kwargs = {
                "model": model_name,
                "api_key": azure_key,
                "base_url": _normalize_openai_base_url(azure_base_url),
                "temperature": temperature,
            }
            if thinking:
                kwargs["max_completion_tokens"] = max_output_tokens
                kwargs["reasoning_effort"] = thinking_level or "medium"
            else:
                kwargs["max_tokens"] = max_output_tokens
            return ChatOpenAI(**kwargs)

        if provider == "azure":
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL are required for azure models."
            )

        if not env.get("OPENAI_API_KEY"):
            raise ValueError("can not find the OPENAI_API_KEY in environment variable.\n\n")
        kwargs = {
            "model": model_name,
            "api_key": env["OPENAI_API_KEY"],
            "temperature": temperature,
        }
        if env.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = _normalize_openai_base_url(env["OPENAI_BASE_URL"])
        if thinking:
            kwargs["max_completion_tokens"] = max_output_tokens
            kwargs["reasoning_effort"] = thinking_level or "medium"
        else:
            kwargs["max_tokens"] = max_output_tokens
        return ChatOpenAI(
            **kwargs,
        )

    if provider in {"google", "gemini", "gemi"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not env.get("GEMINI_API_KEY"):
            raise ValueError("can not find the GEMINI_API_KEY in environment variable.\n\n")
        kwargs = {
            "model": model_name,
            "google_api_key": env["GEMINI_API_KEY"],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if thinking:
            kwargs["include_thoughts"] = True
            if normalized_budget is not None:
                kwargs["thinking_budget"] = normalized_budget
        return ChatGoogleGenerativeAI(**kwargs)

    if provider in {"anthropic", "anthropicai", "claude", "anthro"}:
        from langchain_anthropic import ChatAnthropic

        if not env.get("ANTHROPIC_API_KEY"):
            raise ValueError("can not find the ANTHROPIC_API_KEY in environment variable.\n\n")
        kwargs = {
            "model": model_name,
            "api_key": env["ANTHROPIC_API_KEY"],
            "temperature": temperature,
            "max_tokens_to_sample": max_output_tokens,
        }
        if thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": normalized_budget or max(1024, max_output_tokens // 2),
            }
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=env.get("OLLAMA_API_BASE", "http://localhost:11434"),
            temperature=temperature,
            num_predict=(
                max(2048, max_output_tokens + (normalized_budget or max_output_tokens))
                if thinking
                else max_output_tokens
            ),
            reasoning=thinking,
        )

    raise ValueError(f"Unsupported ChatModel provider: {provider}")
