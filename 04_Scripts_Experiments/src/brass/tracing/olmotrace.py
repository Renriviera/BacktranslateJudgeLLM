"""OLMoTrace client (infini-gram backed).

OLMoTrace links model outputs back to the open pretraining corpus (Dolma), which lets BRASS check
whether base-model completions are supported by genuine pretraining content rather than artifacts
(``main.tex``, "Why OLMo 3 and Open Weights"). This client talks to the public infini-gram API
that powers OLMoTrace.

It is intentionally lightweight and *degrades gracefully*: if the API is unset, unreachable, or
errors, tracing is skipped with a warning so the rest of the pipeline (and the smoke test) still
completes. For the smoke test we record, per completion, corpus counts for the full text and for a
handful of n-gram spans, which is enough to confirm the integration works end to end.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"


@dataclass
class TraceResult:
    """Tracing outcome for one completion.

    Attributes:
        text: The traced completion (possibly truncated for the query).
        full_count: Corpus occurrence count of the full text (``-1`` if unavailable).
        spans: List of ``{"ngram": str, "count": int}`` for probed spans.
        ok: Whether the trace completed against a live API.
        error: Error message if tracing failed / was skipped.
    """

    text: str
    full_count: int = -1
    spans: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = False
    error: str | None = None


class OLMoTraceClient:
    """Minimal infini-gram / OLMoTrace HTTP client."""

    def __init__(
        self,
        api_url: str | None = None,
        index: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.api_url = (api_url or os.environ.get("OLMOTRACE_API_URL") or DEFAULT_API_URL).rstrip(
            "/"
        ) + "/"
        self.index = index or os.environ.get("OLMOTRACE_INDEX") or DEFAULT_INDEX
        self.api_key = api_key or os.environ.get("OLMOTRACE_API_KEY")
        self.timeout = timeout
        self.enabled = enabled

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.api_url, data=data, headers=headers)  # noqa: S310
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def count(self, query: str) -> int:
        """Return the corpus occurrence count of ``query`` (``-1`` on failure)."""
        try:
            resp = self._post({"index": self.index, "query_type": "count", "query": query})
            return int(resp.get("count", -1))
        except Exception as exc:  # noqa: BLE001
            logger.debug("infini-gram count failed: %s", exc)
            return -1

    def trace_completion(
        self, text: str, *, max_chars: int = 2000, n_spans: int = 5
    ) -> TraceResult:
        """Trace one completion: full-text count + a few n-gram span counts."""
        if not self.enabled:
            return TraceResult(text=text[:max_chars], ok=False, error="tracing disabled")
        text = (text or "").strip()
        if not text:
            return TraceResult(text="", ok=False, error="empty completion")
        probe = text[:max_chars]
        try:
            full_count = self.count(probe)
            words = probe.split()
            spans: list[dict[str, Any]] = []
            if len(words) >= 5:
                step = max(len(words) // max(n_spans, 1), 1)
                for start in range(0, max(len(words) - 5, 1), step):
                    ngram = " ".join(words[start : start + 5])
                    spans.append({"ngram": ngram, "count": self.count(ngram)})
                    if len(spans) >= n_spans:
                        break
            return TraceResult(text=probe, full_count=full_count, spans=spans, ok=True)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning("OLMoTrace failed (%s); skipping.", exc)
            return TraceResult(text=probe, ok=False, error=str(exc))
