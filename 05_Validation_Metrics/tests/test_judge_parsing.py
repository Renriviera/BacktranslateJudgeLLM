"""Tests for judge output parsing (no model / engine needed)."""

from dataclasses import dataclass

from brass.metrics.judge_asr import _expected_score_from_logprobs


@dataclass
class _LP:
    logprob: float


def test_expected_score_all_mass_on_five():
    digit_ids = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    # Heavy mass on token id 5 -> score near 1.0.
    token_logprobs = {1: _LP(-20.0), 2: _LP(-20.0), 3: _LP(-20.0), 4: _LP(-20.0), 5: _LP(0.0)}
    assert _expected_score_from_logprobs(token_logprobs, digit_ids) > 0.95


def test_expected_score_all_mass_on_one():
    digit_ids = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    token_logprobs = {1: _LP(0.0), 2: _LP(-20.0), 3: _LP(-20.0), 4: _LP(-20.0), 5: _LP(-20.0)}
    assert _expected_score_from_logprobs(token_logprobs, digit_ids) < 0.05


def test_expected_score_uniform_is_half():
    digit_ids = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    token_logprobs = {i: _LP(0.0) for i in range(1, 6)}
    # Uniform over {0, .25, .5, .75, 1} -> mean 0.5.
    assert abs(_expected_score_from_logprobs(token_logprobs, digit_ids) - 0.5) < 1e-6


def test_expected_score_missing_digits_handled():
    digit_ids = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    # Only token id 5 present in the top-k logprobs.
    token_logprobs = {5: _LP(-1.0)}
    assert _expected_score_from_logprobs(token_logprobs, digit_ids) == 1.0
