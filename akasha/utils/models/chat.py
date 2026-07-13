"""LangChain ChatModel factory used by the public Akasha model selectors."""

from typing import Any, Mapping


def build_chat_model(
    provider: str,
    model_name: str,
    env: Mapping[str, Any],
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    thinking: bool = False,
    thinking_budget: int | None = None,
):
    """Build a LangChain 1.3+ chat model for a supported provider."""

    provider = provider.lower()
    # A budget has no meaning when thinking is disabled. Ignore it so callers
    # can pass a common configuration to models with different capabilities.
    if not thinking:
        thinking_budget = None

    if thinking_budget is not None and thinking_budget <= 0:
        raise ValueError("thinking_budget must be greater than zero when provided.")

    if provider == "openai":
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        if env.get("AZURE_API_TYPE") == "azure" or env.get("OPENAI_API_TYPE") == "azure":
            if thinking:
                raise ValueError(
                    "thinking=True is not yet supported for Azure OpenAI model strings; "
                    "pass a configured AzureChatOpenAI model instead."
                )
            return AzureChatOpenAI(
                azure_endpoint=env["AZURE_API_BASE"],
                azure_deployment=model_name.replace(".", ""),
                api_key=env["AZURE_API_KEY"],
                api_version=env.get("AZURE_API_VERSION", "2023-05-15"),
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

        if not env.get("OPENAI_API_KEY"):
            raise ValueError("can not find the OPENAI_API_KEY in environment variable.\n\n")
        kwargs = {
            "model": model_name,
            "api_key": env["OPENAI_API_KEY"],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if thinking:
            raise ValueError(
                "thinking=True is not yet supported for OpenAI model strings; "
                "pass a configured ChatOpenAI model instead."
            )
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
            if thinking_budget is not None:
                kwargs["thinking_budget"] = thinking_budget
        return ChatGoogleGenerativeAI(**kwargs)

    if provider in {"anthropic", "anthropicai", "claude", "anthro"}:
        from langchain_anthropic import ChatAnthropic

        if not env.get("ANTHROPIC_API_KEY"):
            raise ValueError("can not find the ANTHROPIC_API_KEY in environment variable.\n\n")
        if thinking:
            raise ValueError(
                "thinking=True is not yet supported for Anthropic model strings; "
                "pass a configured ChatAnthropic model instead."
            )
        return ChatAnthropic(
            model=model_name,
            api_key=env["ANTHROPIC_API_KEY"],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if thinking:
            raise ValueError(
                "thinking=True is not yet supported for Ollama model strings; "
                "pass a configured ChatOllama model instead."
            )
        return ChatOllama(
            model=model_name,
            base_url=env.get("OLLAMA_API_BASE", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_output_tokens,
        )

    raise ValueError(f"Unsupported ChatModel provider: {provider}")
