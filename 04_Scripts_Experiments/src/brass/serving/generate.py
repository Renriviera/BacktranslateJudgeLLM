"""Generation and scoring helpers built on :class:`brass.serving.vllm_engine.VLLMEngine`.

Three capabilities the BRASS pipeline needs:

- ``sample_completions``  - draw ``n`` completions per prompt (the empirical ``E_p(.)`` samples).
- ``generate_once``       - one deterministic completion per prompt (judge prompts, greedy probes).
- ``sequence_logprob``    - total log-probability a model assigns to a (prompt, completion) pair,
  used for base log-likelihood recovery in BRASS.

Chat models apply the tokenizer chat template; the base model uses raw continuation prompting.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from brass.serving.vllm_engine import VLLMEngine

logger = logging.getLogger(__name__)

# FastChat Vicuna v1.1. lmsys/vicuna-7b-v1.5 ships no tokenizer.chat_template, and vLLM's
# tokenizer wrapper ignores attribute assignment -- pass this via chat_template=.
VICUNA_V11_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    '{% set system_message = "A chat between a curious user and an artificial intelligence '
    "assistant. The assistant gives helpful, detailed, and polite answers to the user's "
    'questions." %}'
    "{% endif %}"
    "{{ system_message }}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ ' USER: ' + message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ASSISTANT: ' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ ' ASSISTANT:' }}"
    "{% endif %}"
)


def fallback_chat_template(hf_id: str | None) -> str | None:
    """Return a chat template to pass to ``apply_chat_template`` when the tokenizer has none."""
    hid = (hf_id or "").lower()
    if "vicuna" in hid:
        return VICUNA_V11_CHAT_TEMPLATE
    return None


def _apply_chat_template(
    engine: VLLMEngine,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict | None = None,
) -> str:
    kwargs = dict(chat_template_kwargs or {})
    fallback = fallback_chat_template(engine.spec.hf_id)
    if fallback is not None:
        kwargs.setdefault("chat_template", fallback)
    return engine.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


@dataclass
class CompletionSet:
    """Completions for a single prompt: ``prompt_id`` -> list of generated strings."""

    prompt_id: str
    prompt: str
    completions: list[str]


def _format_prompt(
    engine: VLLMEngine,
    prompt: str,
    system: str | None = None,
    chat_template_kwargs: dict | None = None,
) -> str:
    """Apply the chat template for chat models, else return the raw prompt.

    ``chat_template_kwargs`` is forwarded to ``apply_chat_template`` for templates that accept
    extra flags, e.g. ``{"enable_thinking": False}`` to put Qwen3 in non-thinking mode.
    """
    if not engine.spec.is_chat:
        return prompt
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return _apply_chat_template(engine, messages, chat_template_kwargs)
    except Exception as exc:  # noqa: BLE001 - fall back to raw prompt
        logger.warning("chat template failed (%s); using raw prompt", exc)
        return prompt


def sample_completions(
    engine: VLLMEngine,
    prompts: Sequence[tuple[str, str]],
    *,
    n: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    seed: int | None = None,
    system: str | None = None,
    chat_template_kwargs: dict | None = None,
) -> list[CompletionSet]:
    """Sample ``n`` completions for each ``(prompt_id, prompt)`` pair.

    Returns one :class:`CompletionSet` per input prompt, preserving input order.

    ``chat_template_kwargs`` is forwarded to the chat template (e.g.
    ``{"enable_thinking": False}`` for Qwen3 non-thinking generation).
    """
    from vllm import SamplingParams

    formatted = [
        _format_prompt(engine, p, system=system, chat_template_kwargs=chat_template_kwargs)
        for _, p in prompts
    ]
    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    outputs = engine.llm.generate(formatted, params)
    results: list[CompletionSet] = []
    for (pid, ptext), out in zip(prompts, outputs, strict=False):
        results.append(
            CompletionSet(
                prompt_id=pid,
                prompt=ptext,
                completions=[o.text for o in out.outputs],
            )
        )
    return results


def sample_chat(
    engine: VLLMEngine,
    conversations: Sequence[tuple[str, list[dict[str, str]]]],
    *,
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> list[CompletionSet]:
    """Sample completions for pre-built multi-turn chat conversations.

    Unlike :func:`sample_completions` (which wraps a single user turn, optionally with a system
    message), this accepts a full message list per item -- ``[{"role": ..., "content": ...}, ...]``
    with arbitrary ``system``/``user``/``assistant`` turns. This is what iterative attacks like PAIR
    need: the attacker LLM is re-queried with the running conversation history.

    Each item is ``(prompt_id, messages)``. Returns one :class:`CompletionSet` per item, preserving
    input order. ``chat_template_kwargs`` is forwarded to the chat template (e.g.
    ``{"enable_thinking": False}`` for Qwen3 non-thinking generation).
    """
    from vllm import SamplingParams

    formatted: list[str] = []
    for _, messages in conversations:
        try:
            formatted.append(_apply_chat_template(engine, messages, chat_template_kwargs))
        except Exception as exc:  # noqa: BLE001 - fall back to the last user content
            logger.warning("chat template failed (%s); using last message content", exc)
            formatted.append(messages[-1]["content"] if messages else "")

    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    outputs = engine.llm.generate(formatted, params)
    results: list[CompletionSet] = []
    for (pid, _), ftext, out in zip(conversations, formatted, outputs, strict=False):
        results.append(
            CompletionSet(
                prompt_id=pid,
                prompt=ftext,
                completions=[o.text for o in out.outputs],
            )
        )
    return results


def generate_once(
    engine: VLLMEngine,
    formatted_prompts: Sequence[str],
    *,
    max_tokens: int = 4,
    temperature: float = 0.0,
    logprobs: int | None = None,
):
    """Greedy single generation for pre-formatted prompts (e.g. judge templates).

    When ``logprobs`` is set, per-step top-logprobs are returned for answer-token scoring.
    Returns the raw vLLM ``RequestOutput`` list so callers can inspect tokens/logprobs.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )
    kwargs = {}
    if getattr(engine, "lora_request", None) is not None:
        kwargs["lora_request"] = engine.lora_request
    return engine.llm.generate(list(formatted_prompts), params, **kwargs)


def sequence_logprob(
    engine: VLLMEngine,
    pairs: Sequence[tuple[str, str]],
) -> list[float]:
    """Total log-prob a model assigns to each ``(context, continuation)`` pair.

    Uses ``prompt_logprobs`` over the concatenated text and sums the logprobs of the continuation
    tokens only. This powers base log-likelihood recovery: scoring instruct / attacked
    completions under the base model ``M``.
    """
    from vllm import SamplingParams

    tok = engine.tokenizer
    full_texts: list[str] = []
    cont_token_counts: list[int] = []
    for context, continuation in pairs:
        ctx_ids = tok(context, add_special_tokens=False)["input_ids"]
        full_ids = tok(context + continuation, add_special_tokens=False)["input_ids"]
        full_texts.append(context + continuation)
        cont_token_counts.append(max(len(full_ids) - len(ctx_ids), 0))

    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = engine.llm.generate(full_texts, params)

    scores: list[float] = []
    for out, n_cont in zip(outputs, cont_token_counts, strict=False):
        plp = out.prompt_logprobs or []
        total = 0.0
        # prompt_logprobs[i] is a dict {token_id: Logprob} for position i (first is None).
        for entry in plp[-n_cont:] if n_cont > 0 else []:
            if not entry:
                continue
            # The realized token's logprob is the max-rank-0 entry; sum them.
            total += min((lp.logprob for lp in entry.values()), key=lambda x: -x)
        scores.append(total)
    return scores
