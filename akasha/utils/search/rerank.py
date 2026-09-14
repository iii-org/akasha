"""Second-stage document reranking adapters.

Retrieval chooses candidates. This module reorders and optionally limits them,
so callers can change a reranker without changing their first-stage search mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, TypeAlias

from akasha.utils.optional_dependencies import require_optional_dependency

Reranker: TypeAlias = str | Callable[[str, list[Any]], Sequence[Any]] | None


def validate_reranker_dependencies(reranker: Reranker) -> None:
    """Fail before retrieval when a selected local reranker is unavailable."""

    if isinstance(reranker, str) and _local_model_name(reranker) is not None:
        require_optional_dependency(
            "torch",
            feature="Local BGE reranking",
            extra="full",
        )
        require_optional_dependency(
            "transformers",
            feature="Local BGE reranking",
            extra="full",
        )


def rerank_documents(
    query: str,
    documents: Sequence[Any],
    reranker: Reranker,
    *,
    model_obj: Any = None,
    top_k: int | None = None,
    verbose: bool = False,
    keep_logs: bool = False,
) -> list[Any]:
    """Return ``documents`` in relevance order, optionally limited by ``top_k``."""

    validate_reranker_dependencies(reranker)
    candidates = list(documents)
    if reranker is None or not candidates:
        return candidates

    if callable(reranker):
        ranked = list(reranker(query, candidates))
        return _limit_ranked_documents(
            _validate_permutation(candidates, ranked), top_k
        )

    reranker_name = reranker.strip().lower()
    if reranker_name == "llm":
        if model_obj is None:
            raise ValueError("The 'llm' reranker requires a configured model.")
        ranked = _rerank_with_llm(
            query,
            candidates,
            model_obj=model_obj,
            verbose=verbose,
            keep_logs=keep_logs,
        )
        return _limit_ranked_documents(
            _validate_permutation(candidates, ranked), top_k
        )

    local_model = _local_model_name(reranker)
    if local_model is not None:
        pairs = [[query, document.page_content] for document in candidates]
        scores = _score_with_local_bge(local_model, pairs)
        if len(scores) != len(candidates):
            raise ValueError("The local reranker returned an invalid score count.")
        ranked = [
            document
            for _, document in sorted(
                zip(scores, candidates), key=lambda item: item[0], reverse=True
            )
        ]
        return _limit_ranked_documents(
            _validate_permutation(candidates, ranked), top_k
        )

    raise ValueError(f"Unsupported reranker: {reranker!r}")


def _rerank_with_llm(
    query: str,
    candidates: list[Any],
    *,
    model_obj: Any,
    verbose: bool,
    keep_logs: bool,
) -> list[Any]:
    payload = [
        {"id": index, "text": document.page_content}
        for index, document in enumerate(candidates)
    ]
    prompt = (
        "Rank the candidate documents by relevance to the user question. "
        "Return JSON only in this exact shape: {\"order\": [id, ...]}. "
        "Include every candidate id exactly once.\n\n"
        f"Question: {query}\n\n"
        f"Candidates: {json.dumps(payload, ensure_ascii=False)}"
    )
    response = _call_model(model_obj, prompt, verbose, keep_logs)
    order = _parse_order(response, len(candidates))
    return [candidates[index] for index in order]


def _parse_order(response: str, candidate_count: int) -> list[int]:
    try:
        import json_repair

        parsed = json_repair.loads(response)
    except (ImportError, TypeError, ValueError):
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("The LLM reranker did not return valid JSON.") from error

    order = parsed.get("order") if isinstance(parsed, dict) else None
    expected = list(range(candidate_count))
    if (
        not isinstance(order, list)
        or not all(type(candidate_id) is int for candidate_id in order)
        or sorted(order) != expected
    ):
        raise ValueError(
            "The LLM reranker must return every candidate id exactly once."
        )
    return order


def _call_model(model_obj: Any, prompt: str, verbose: bool, keep_logs: bool) -> str:
    from akasha.helper.run_llm import call_model

    return call_model(
        model_obj,
        prompt,
        verbose=verbose,
        keep_logs=keep_logs,
    )


def _local_model_name(reranker: str) -> str | None:
    prefix, separator, model_name = reranker.strip().partition(":")
    if prefix.lower() not in {"local", "bge"}:
        return None
    if not separator:
        return "BAAI/bge-reranker-base"
    if not model_name.strip():
        raise ValueError("A local reranker model name cannot be empty.")
    return model_name.strip()


def _score_with_local_bge(model_name: str, pairs: list[list[str]]) -> list[float]:
    torch, tokenizer, model, device = _load_local_bge_model(model_name)
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, return_dict=True).logits.view(-1)
    return logits.float().cpu().tolist()


@lru_cache(maxsize=4)
def _load_local_bge_model(model_name: str):
    torch = require_optional_dependency(
        "torch",
        feature="Local BGE reranking",
        extra="full",
    )
    transformers = require_optional_dependency(
        "transformers",
        feature="Local BGE reranking",
        extra="full",
    )
    AutoModelForSequenceClassification = (
        transformers.AutoModelForSequenceClassification
    )
    AutoTokenizer = transformers.AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _validate_permutation(candidates: list[Any], ranked: list[Any]) -> list[Any]:
    if len(ranked) != len(candidates):
        raise ValueError("A reranker must return every candidate document exactly once.")

    expected_ids = sorted(id(document) for document in candidates)
    ranked_ids = sorted(id(document) for document in ranked)
    if ranked_ids != expected_ids:
        raise ValueError("A reranker must only reorder the candidate documents.")

    return ranked


def _limit_ranked_documents(ranked: list[Any], top_k: int | None) -> list[Any]:
    if top_k is None:
        return ranked
    if type(top_k) is not int or top_k < 1:
        raise ValueError("rerank_top_k must be a positive integer.")
    return ranked[:top_k]
