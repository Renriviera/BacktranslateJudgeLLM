"""Tests for the dataset loader registry and unified schema (no network)."""

from brass.data import BehaviorPrompt
from brass.data.loaders import _REGISTRY


def test_all_expected_loaders_registered():
    for name in ("strongreject", "harmbench", "orbench", "xstest", "jailjudge"):
        assert name in _REGISTRY


def test_behavior_prompt_schema():
    p = BehaviorPrompt(
        id="strongreject:0", prompt="example", source="strongreject", is_harmful=True
    )
    assert p.id == "strongreject:0"
    assert p.is_harmful is True
    assert p.context is None
    assert isinstance(p.metadata, dict)
