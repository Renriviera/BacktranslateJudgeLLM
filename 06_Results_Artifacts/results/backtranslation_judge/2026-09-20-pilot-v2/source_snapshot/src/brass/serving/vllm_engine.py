"""vLLM engine lifecycle management.

A single 96 GB GPU holds one 32B model in bf16 at a time, so the pipeline loads and *fully tears
down* each engine before loading the next. :class:`VLLMEngine` is a context manager that owns one
``vllm.LLM`` and best-effort frees GPU memory on exit (vLLM's teardown surface shifts between
releases, so cleanup is defensive).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Declarative description of a servable model (mirrors the Hydra ``model``/``judge`` group).

    Attributes:
        name: Short logical name (e.g. ``olmo31_instruct``).
        hf_id: Hugging Face repo id or local path.
        is_chat: Whether to apply the tokenizer chat template (instruct/judge) vs raw continuation
            prompting (base model).
        dtype: vLLM dtype; ``bfloat16`` for the OLMo 32B checkpoints.
        max_model_len: Optional context length cap (keeps KV cache bounded).
        gpu_memory_utilization: Fraction of VRAM vLLM may reserve.
        tensor_parallel_size: TP degree (1 on a single GPU).
        revision: Optional HF revision/branch (e.g. a checkpoint stage).
        trust_remote_code: Forwarded to vLLM/HF.
        enable_lora: Serve a LoRA adapter on top of ``hf_id`` (the base model).
        lora_path: HF id or local path of the LoRA adapter (e.g. the StrongREJECT judge adapter).
        max_lora_rank: Max LoRA rank vLLM should accommodate (>= the adapter's ``r``).
        extra: Any additional kwargs forwarded verbatim to ``vllm.LLM``.
    """

    name: str
    hf_id: str
    is_chat: bool = True
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    revision: str | None = None
    trust_remote_code: bool = True
    enable_lora: bool = False
    lora_path: str | None = None
    max_lora_rank: int = 16
    extra: dict[str, Any] = field(default_factory=dict)


class VLLMEngine:
    """Context-managed wrapper around a single ``vllm.LLM``."""

    def __init__(self, spec: ModelSpec, *, seed: int | None = None) -> None:
        self.spec = spec
        self.seed = seed
        self._llm = None
        self._tokenizer = None
        self._lora_request = None

    # -- lifecycle ---------------------------------------------------------------------------- #
    def __enter__(self) -> VLLMEngine:
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def load(self) -> None:
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            model=self.spec.hf_id,
            dtype=self.spec.dtype,
            gpu_memory_utilization=self.spec.gpu_memory_utilization,
            tensor_parallel_size=self.spec.tensor_parallel_size,
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.spec.max_model_len is not None:
            kwargs["max_model_len"] = self.spec.max_model_len
        if self.spec.revision is not None:
            kwargs["revision"] = self.spec.revision
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.spec.enable_lora:
            kwargs["enable_lora"] = True
            kwargs["max_loras"] = 1
            kwargs["max_lora_rank"] = self.spec.max_lora_rank
        kwargs.update(self.spec.extra)

        logger.info("Loading vLLM engine for %s (%s)", self.spec.name, self.spec.hf_id)
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        if self.spec.enable_lora and self.spec.lora_path:
            self._lora_request = self._build_lora_request(self.spec.lora_path)

    @staticmethod
    def _build_lora_request(lora_path: str):
        """Build a vLLM ``LoRARequest``, resolving an HF id to a local snapshot when possible."""
        from vllm.lora.request import LoRARequest

        resolved = lora_path
        if "/" in lora_path and not Path(lora_path).exists():
            try:
                from huggingface_hub import snapshot_download

                resolved = snapshot_download(lora_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not resolve LoRA '%s' locally (%s); passing id.", lora_path, exc
                )
        return LoRARequest("adapter", 1, resolved)

    @property
    def lora_request(self):
        """The active LoRA request, or ``None`` if the engine serves a plain model."""
        return self._lora_request

    def unload(self) -> None:
        """Tear down the engine and free GPU memory (best-effort across vLLM versions)."""
        if self._llm is None:
            return
        logger.info("Unloading vLLM engine for %s", self.spec.name)
        try:
            del self._llm
        finally:
            self._llm = None
            self._tokenizer = None
        self._free_gpu()

    @staticmethod
    def _free_gpu() -> None:
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort
            logger.debug("vLLM distributed teardown skipped: %s", exc)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: %s", exc)

    # -- accessors ---------------------------------------------------------------------------- #
    @property
    def llm(self):
        if self._llm is None:
            raise RuntimeError("Engine is not loaded. Use `with VLLMEngine(spec) as eng:` ...")
        return self._llm

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Engine is not loaded.")
        return self._tokenizer
