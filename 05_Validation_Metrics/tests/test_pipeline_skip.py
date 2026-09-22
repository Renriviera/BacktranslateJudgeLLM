"""Judges-only pipeline skip: BRASS/OLMoTrace flags must not load the base model."""

from omegaconf import OmegaConf

from brass.pipeline.run_experiment import brass_enabled, plan_phases


def test_brass_enabled_false_skips_base_and_clean():
    cfg = OmegaConf.create(
        {
            "metrics": {
                "brass": {"enabled": False},
                "judge": {"enabled": True},
            },
            "olmotrace": {"enabled": False},
        }
    )
    assert brass_enabled(cfg) is False
    phases = plan_phases(cfg)
    assert phases["clean_completions"] is False
    assert phases["base"] is False
    assert phases["embedding_brass"] is False
    assert phases["judges"] is True
    assert phases["prefix_asr"] is True
    assert phases["olmotrace"] is False


def test_brass_enabled_true_keeps_full_pipeline():
    cfg = OmegaConf.create(
        {
            "metrics": {
                "brass": {"enabled": True},
                "judge": {"enabled": True},
            },
            "olmotrace": {"enabled": True},
        }
    )
    assert brass_enabled(cfg) is True
    phases = plan_phases(cfg)
    assert phases["clean_completions"] is True
    assert phases["base"] is True
    assert phases["embedding_brass"] is True
    assert phases["olmotrace"] is True


def test_brass_enabled_defaults_true_when_flag_absent():
    cfg = OmegaConf.create({"metrics": {"brass": {"eps": 1e-8}, "judge": {"enabled": True}}})
    assert brass_enabled(cfg) is True
    assert plan_phases(cfg)["olmotrace"] is False
