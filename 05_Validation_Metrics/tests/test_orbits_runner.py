"""Orbits: leakage, reproducibility, censoring, parsing, and compute accounting."""

import json
from dataclasses import replace

import pytest

from brass.orbits.io import read_jsonl, stable_seed
from brass.orbits.runner import (
    Budget,
    Generation,
    OrbitConfig,
    inverse_messages,
    parse_inverse,
    run_orbits,
)


class FakeBackend:
    fingerprint = {"model": "test-double-not-real-inference"}

    def __init__(self, *, fail_after=None, truncate=False):
        self.calls = []
        self.fail_after = fail_after
        self.truncate = truncate

    def generate(self, requests):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls.append(requests)
        return [
            Generation(
                (
                    json.dumps({"prompt": f"Reconstructed task {r.seed}"})
                    if "assistant_response" in r.messages[0]["content"]
                    else f"Response {r.seed}"
                ),
                5,
                12,
                "length" if self.truncate else "stop",
            )
            for r in requests
        ]


MANIFEST = [
    {
        "id": "sample:a",
        "group_id": "task:a",
        "cohort": "benign",
        "prompt": "ORIGINAL_PROMPT_SENTINEL",
        "evaluation": {"answer": "ANSWER_SECRET"},
    }
]


def test_inverse_only_receives_response():
    messages = inverse_messages("Observed answer")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert "Observed answer" in messages[0]["content"]
    assert "ORIGINAL_PROMPT_SENTINEL" not in str(messages)


def test_cycle_counts_fresh_context_and_no_metadata_leak(tmp_path):
    backend = FakeBackend()
    run_orbits(
        MANIFEST,
        OrbitConfig(round_trips=2, sampled_trajectories=2),
        backend,
        tmp_path / "out",
        tmp_path / "budget.jsonl",
    )
    events = read_jsonl(tmp_path / "out/events.jsonl")
    assert len(events) == 3 * 5
    assert sum(e["direction"] == "forward" for e in events) == 9
    assert all(len(r.messages) == 1 for batch in backend.calls for r in batch)
    assert all("ANSWER_SECRET" not in str(r.messages) for batch in backend.calls for r in batch)
    assert all(
        "ORIGINAL_PROMPT_SENTINEL" not in str(e["input_messages"]) for e in events if e["stage"] > 0
    )
    assert len({e["seed"] for e in events}) == len(events)


def test_resume_performs_no_duplicate_inference(tmp_path):
    backend = FakeBackend()
    cfg = OrbitConfig(round_trips=1, sampled_trajectories=1)
    run_orbits(MANIFEST, cfg, backend, tmp_path / "out", tmp_path / "budget.jsonl")
    resumed = FakeBackend()
    run_orbits(MANIFEST, cfg, resumed, tmp_path / "out", tmp_path / "budget.jsonl")
    assert resumed.calls == []
    assert Budget(tmp_path / "budget.jsonl", 1000).used() == 30


def test_interrupted_resume_matches_uninterrupted_events(tmp_path):
    cfg = OrbitConfig(round_trips=2, sampled_trajectories=1, greedy=False)
    with pytest.raises(RuntimeError, match="interruption"):
        run_orbits(
            MANIFEST, cfg, FakeBackend(fail_after=2), tmp_path / "resume", tmp_path / "budget.jsonl"
        )
    run_orbits(MANIFEST, cfg, FakeBackend(), tmp_path / "resume", tmp_path / "budget.jsonl")
    run_orbits(MANIFEST, cfg, FakeBackend(), tmp_path / "full", tmp_path / "other_budget.jsonl")
    resumed = read_jsonl(tmp_path / "resume/events.jsonl")
    full = read_jsonl(tmp_path / "full/events.jsonl")
    assert [(r["seed"], r["text"], r["request_sha256"]) for r in resumed] == [
        (r["seed"], r["text"], r["request_sha256"]) for r in full
    ]
    # Failed calls remain conservatively reserved, rather than disappearing from accounting.
    assert Budget(tmp_path / "budget.jsonl", 10000).used() == 1024 + 25


def test_resume_rejects_changed_generation_settings(tmp_path):
    cfg = OrbitConfig(round_trips=0, sampled_trajectories=1)
    run_orbits(MANIFEST, cfg, FakeBackend(), tmp_path / "out", tmp_path / "budget.jsonl")
    with pytest.raises(ValueError, match="Resume refused"):
        run_orbits(
            MANIFEST,
            replace(cfg, forward_temperature=0.5),
            FakeBackend(),
            tmp_path / "out",
            tmp_path / "budget.jsonl",
        )


def test_truncation_is_terminal_not_convergence(tmp_path):
    backend = FakeBackend(truncate=True)
    run_orbits(
        MANIFEST,
        OrbitConfig(round_trips=4, sampled_trajectories=1, greedy=False),
        backend,
        tmp_path / "out",
        tmp_path / "budget.jsonl",
    )
    events = read_jsonl(tmp_path / "out/events.jsonl")
    assert len(events) == 1 and events[0]["status"] == "generation_truncated"


@pytest.mark.parametrize(
    "text,status",
    [
        ('{"prompt":"A task"}', "ok"),
        ('```json\n{"prompt":"A task"}\n```', "ok"),
        ('{"prompt":""}', "empty_prompt"),
        ('{"prompt":["bad"]}', "invalid_schema"),
        ('{"prompt":"A task", "answer":"leak"}', "invalid_schema"),
        ("I cannot reconstruct that.", "inverse_refusal_heuristic"),
        ("not json", "invalid_json"),
    ],
)
def test_inverse_parser(text, status):
    assert parse_inverse(text)[1] == status


def test_quoted_refusal_is_not_itself_an_inverse_refusal():
    text = json.dumps({"prompt": "Explain the phrase 'I cannot answer' in this poem."})
    assert parse_inverse(text)[1] == "ok"


def test_budget_enforced_before_model_call(tmp_path):
    backend = FakeBackend()
    with pytest.raises(RuntimeError, match="budget"):
        run_orbits(
            MANIFEST,
            OrbitConfig(global_generated_token_limit=10),
            backend,
            tmp_path / "out",
            tmp_path / "budget.jsonl",
        )
    assert not backend.calls


def test_seed_independent_of_batch_order():
    ids = ["a", "b", "c"]
    forward = {k: stable_seed(235711, k, "sample:0", 1) for k in ids}
    reverse = {k: stable_seed(235711, k, "sample:0", 1) for k in reversed(ids)}
    assert forward == reverse


def test_backend_requests_flat_token_ids_with_new_tokenizer_default(monkeypatch):
    import sys
    from types import SimpleNamespace

    from brass.orbits.runner import Request, VLLMBackend

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            # Transformers 5 defaults to a dictionary unless explicitly disabled.
            assert kwargs["return_dict"] is False
            return [1, 2, 3]

    class LLM:
        def generate(self, prompts, params, **kwargs):
            assert prompts == [{"prompt_token_ids": [1, 2, 3]}]
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        SimpleNamespace(
                            text="ok", token_ids=[4], finish_reason="stop", stop_reason=None
                        )
                    ],
                )
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    backend = VLLMBackend(SimpleNamespace(tokenizer=Tokenizer(), llm=LLM()), {}, 20)
    result = backend.generate(
        [Request("a", [{"role": "user", "content": "hello"}], 7, 0.0, 10, 1.0)]
    )
    assert result[0].text == "ok" and result[0].input_tokens == 3
