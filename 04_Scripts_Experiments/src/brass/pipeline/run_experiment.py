"""End-to-end BRASS experiment driver (Hydra entry point).

Phases (one model resident at a time on a single GPU):

  A. Instruct engine  -> clean completions E_p(I) and attacked completions E_{a(p)}(I).
  B. Base engine      -> base completions E_p(M) + base log-likelihood recovery distances.
  C. Embeddings       -> embedding-distribution BRASS (MMD / energy / sliced-W / centroid-cosine).
  D. Judges           -> StrongREJECT-ft + HarmBench ASR (each engine loaded then freed).
  E. Prefix ASR       -> refusal-string matching (CPU).
  F. OLMoTrace        -> corpus lookups for a sample of completions.
  G. Aggregate        -> write results JSON + log to W&B.

Run: ``python -m brass.pipeline.run_experiment experiment=smoke_test``
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from brass import attacks  # noqa: F401 - registers attacks on import
from brass.attacks.base import AttackResult, get_attack
from brass.data import BehaviorPrompt, load_dataset_prompts
from brass.metrics.brass.base import BrassScore
from brass.metrics.prefix_asr import PrefixASR
from brass.serving.generate import CompletionSet, sample_completions
from brass.serving.vllm_engine import ModelSpec, VLLMEngine
from brass.utils.io import write_json
from brass.utils.logging_wandb import WandbRun
from brass.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------- #
# Helpers                                                                                         #
# --------------------------------------------------------------------------------------------- #
def _model_spec(node: DictConfig) -> ModelSpec:
    d = OmegaConf.to_container(node, resolve=True)
    return ModelSpec(**d)  # type: ignore[arg-type]


def _select_prompts(cfg: DictConfig) -> list[BehaviorPrompt]:
    prompts = load_dataset_prompts(
        cfg.dataset.loader, **OmegaConf.to_container(cfg.dataset.kwargs, resolve=True)
    )
    n = int(cfg.sampling.n_prompts)
    rng = random.Random(cfg.seed)
    if n < len(prompts):
        prompts = rng.sample(prompts, n)
    logger.info("Selected %d/%d prompts from %s", len(prompts), len(prompts), cfg.dataset.name)
    return prompts


def _completion_map(sets: list[CompletionSet]) -> dict[str, list[str]]:
    return {cs.prompt_id: cs.completions for cs in sets}


def brass_enabled(cfg: DictConfig) -> bool:
    """Return True when base + embedding BRASS phases should run."""
    metrics = cfg.get("metrics")
    if not metrics:
        return False
    brass = metrics.get("brass")
    if not brass:
        return False
    return bool(brass.get("enabled", True))


def plan_phases(cfg: DictConfig) -> dict[str, bool]:
    """Which pipeline phases to run. Extracted so judges-only configs are unit-testable."""
    brass = brass_enabled(cfg)
    judge_on = bool(OmegaConf.select(cfg, "metrics.judge.enabled", default=True))
    olmo_on = bool(OmegaConf.select(cfg, "olmotrace.enabled", default=False))
    return {
        "clean_completions": brass,
        "base": brass,
        "embedding_brass": brass,
        "judges": judge_on,
        "prefix_asr": True,
        "olmotrace": olmo_on,
    }


# --------------------------------------------------------------------------------------------- #
# Phases                                                                                          #
# --------------------------------------------------------------------------------------------- #
def _phase_instruct(
    cfg: DictConfig, prompts: list[BehaviorPrompt], attacked: list[AttackResult]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Generate clean E_p(I) and attacked E_{a(p)}(I) completions with the instruct engine.

    When BRASS is disabled, skip clean completions (judges/prefix ASR only need attacked).
    """
    spec = _model_spec(cfg.subject_model)
    s = cfg.sampling
    clean_inputs = [(p.id, p.prompt) for p in prompts]
    attacked_inputs = [(a.prompt_id, a.attacked_prompt) for a in attacked]
    is_identity = all(a.metadata.get("identity") for a in attacked)
    skip_clean = not plan_phases(cfg)["clean_completions"]

    with VLLMEngine(spec, seed=cfg.seed) as eng:
        if skip_clean:
            logger.info("BRASS disabled; skipping clean instruct completions.")
            clean_map = {p.id: [] for p in prompts}
            clean = None
        else:
            clean = sample_completions(
                eng,
                clean_inputs,
                n=s.n_completions,
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens,
                seed=cfg.seed,
                system=cfg.get("system_prompt"),
            )
            clean_map = _completion_map(clean)
        if is_identity and clean is not None:
            attacked_sets = clean  # E_{a(p)}(I) == E_p(I) for the identity attack
        else:
            attacked_sets = sample_completions(
                eng,
                attacked_inputs,
                n=s.n_completions,
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens,
                seed=cfg.seed,
                system=cfg.get("system_prompt"),
            )
    return clean_map, _completion_map(attacked_sets)


def _phase_base(
    cfg: DictConfig,
    prompts: list[BehaviorPrompt],
    clean: dict[str, list[str]],
    attacked: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, BrassScore]]:
    """Generate base E_p(M) completions and compute base log-likelihood recovery BRASS."""
    from brass.metrics.brass.likelihood import compute_likelihood_brass

    spec = _model_spec(cfg.base_model)
    s = cfg.sampling
    base_inputs = [(p.id, p.prompt) for p in prompts]
    likelihood_scores: dict[str, BrassScore] = {}
    compute_ll = bool(cfg.metrics.brass.get("compute_likelihood", True))

    with VLLMEngine(spec, seed=cfg.seed) as eng:
        base_sets = sample_completions(
            eng,
            base_inputs,
            n=s.n_completions,
            temperature=s.temperature,
            top_p=s.top_p,
            max_tokens=s.get("base_max_tokens", s.max_tokens),
            seed=cfg.seed,
        )
        base = _completion_map(base_sets)
        if compute_ll:
            for p in prompts:
                try:
                    likelihood_scores[p.id] = compute_likelihood_brass(
                        eng,
                        context=p.prompt,
                        base_completions=base.get(p.id, []),
                        clean_completions=clean.get(p.id, []),
                        attacked_completions=attacked.get(p.id, []),
                        eps=float(cfg.metrics.brass.eps),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("likelihood BRASS failed for %s: %s", p.id, exc)
    return base, likelihood_scores


def _phase_embedding_brass(
    cfg: DictConfig,
    prompts: list[BehaviorPrompt],
    base: dict[str, list[str]],
    clean: dict[str, list[str]],
    attacked: dict[str, list[str]],
) -> dict[str, dict[str, BrassScore]]:
    """Per-prompt embedding-distribution BRASS under each requested distance."""
    import numpy as np

    from brass.metrics.brass.embedding import compute_embedding_brass, embed_texts

    distances = list(cfg.metrics.brass.distances)
    per_prompt: dict[str, dict[str, BrassScore]] = {}
    for p in prompts:
        b, c, a = base.get(p.id, []), clean.get(p.id, []), attacked.get(p.id, [])
        if not (b and c and a):
            logger.warning("Skipping embedding BRASS for %s (missing completions).", p.id)
            continue
        emb_b = embed_texts(
            b,
            model_name=cfg.embedding.model_name,
            device=cfg.embedding.device,
            batch_size=cfg.embedding.batch_size,
        )
        emb_c = embed_texts(
            c,
            model_name=cfg.embedding.model_name,
            device=cfg.embedding.device,
            batch_size=cfg.embedding.batch_size,
        )
        emb_a = embed_texts(
            a,
            model_name=cfg.embedding.model_name,
            device=cfg.embedding.device,
            batch_size=cfg.embedding.batch_size,
        )
        per_prompt[p.id] = compute_embedding_brass(
            np.asarray(emb_b),
            np.asarray(emb_c),
            np.asarray(emb_a),
            distances=distances,
            eps=float(cfg.metrics.brass.eps),
        )
    return per_prompt


def _phase_judges(
    cfg: DictConfig,
    prompts: list[BehaviorPrompt],
    attacked: dict[str, list[str]],
) -> dict[str, Any]:
    """Run each configured judge engine sequentially over attacked-instruct completions."""
    from brass.metrics.judge_asr import (
        HarmBenchJudge,
        StrongRejectFinetunedJudge,
        aggregate_judge,
    )

    prompt_text = {p.id: p.prompt for p in prompts}
    # (prompt_id, forbidden_prompt/behavior, response) triples, one per completion.
    triples = [(p.id, prompt_text[p.id], comp) for p in prompts for comp in attacked.get(p.id, [])]
    results: dict[str, Any] = {}

    for key, jcfg in cfg.judges.items():
        kind = jcfg.kind
        spec = _model_spec(jcfg.model)
        logger.info("Running judge '%s' (%s)", key, kind)
        try:
            with VLLMEngine(spec, seed=cfg.seed) as eng:
                if kind == "strongreject_finetuned":
                    judge = StrongRejectFinetunedJudge(
                        eng,
                        threshold=float(jcfg.threshold),
                        max_response_length=int(jcfg.max_response_length),
                    )
                elif kind == "harmbench":
                    judge = HarmBenchJudge(eng)
                else:
                    logger.warning("Unknown judge kind '%s'; skipping.", kind)
                    continue
                jres = judge.score(triples)
        except Exception as exc:  # noqa: BLE001 - one judge failing must not abort the run
            logger.warning(
                "Judge '%s' (%s) failed and was skipped: %s "
                "(StrongREJECT needs HF_TOKEN for the gated google/gemma-2b base).",
                key,
                kind,
                exc,
            )
            results[str(key)] = {
                "judge": kind,
                "asr": float("nan"),
                "skipped": True,
                "error": str(exc),
            }
            continue
        results[str(key)] = {
            "judge": judge.name,
            "asr": aggregate_judge(jres),
            "per_prompt": {
                r.prompt_id: {"asr": r.asr, "threshold_asr": r.threshold_asr} for r in jres
            },
        }
    return results


# --------------------------------------------------------------------------------------------- #
# Aggregation                                                                                     #
# --------------------------------------------------------------------------------------------- #
def _aggregate_embedding(
    per_prompt: dict[str, dict[str, BrassScore]], distances: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in distances:
        vals = [pp[name].brass for pp in per_prompt.values() if name in pp]
        out[name] = sum(vals) / len(vals) if vals else float("nan")
    return out


# --------------------------------------------------------------------------------------------- #
# Main                                                                                            #
# --------------------------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # vLLM spawns its engine core in a subprocess. Because this driver initializes CUDA (seeding,
    # sentence-transformers embeddings) before later engine loads, the engine subprocess must use
    # the 'spawn' start method or CUDA re-init fails. vLLM is imported lazily (inside the serving
    # layer), so setting this here, before any engine is constructed, takes effect.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    seed_everything(int(cfg.seed), deterministic_torch=bool(cfg.deterministic_torch))
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    results_dir = Path(cfg.paths.results_dir) / cfg.experiment_name
    prompts = _select_prompts(cfg)

    # Attack transform (identity for the smoke test).
    attack = get_attack(cfg.attack.name, **OmegaConf.to_container(cfg.attack.kwargs, resolve=True))
    attacked_results = attack.transform_many(prompts)

    with WandbRun(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or f"{cfg.experiment_name}-{cfg.dataset.name}-{cfg.attack.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        enabled=bool(cfg.wandb.enabled),
    ) as run:
        # Phase A/B: generation.
        phases = plan_phases(cfg)
        clean, attacked = _phase_instruct(cfg, prompts, attacked_results)
        if phases["base"]:
            base, likelihood_scores = _phase_base(cfg, prompts, clean, attacked)
        else:
            logger.info("BRASS disabled; skipping base-model completions and likelihood BRASS.")
            base, likelihood_scores = {}, {}

        # Phase C: embedding BRASS.
        if phases["embedding_brass"]:
            emb_per_prompt = _phase_embedding_brass(cfg, prompts, base, clean, attacked)
            distances = list(cfg.metrics.brass.distances)
            emb_agg = _aggregate_embedding(emb_per_prompt, distances)
            ll_agg = (
                sum(s.brass for s in likelihood_scores.values()) / len(likelihood_scores)
                if likelihood_scores
                else float("nan")
            )
        else:
            emb_per_prompt = {}
            emb_agg = {}
            ll_agg = float("nan")

        # Phase D: judges.
        judge_results = _phase_judges(cfg, prompts, attacked) if phases["judges"] else {}

        # Phase E: prefix ASR.
        prefix = PrefixASR()
        prefix_results = [prefix.score_completions(p.id, attacked.get(p.id, [])) for p in prompts]
        prefix_asr = prefix.aggregate(prefix_results)

        # Phase F: OLMoTrace.
        traces = []
        if phases["olmotrace"]:
            from brass.tracing import OLMoTraceClient

            client = OLMoTraceClient(api_url=cfg.olmotrace.api_url, index=cfg.olmotrace.index)
            sample = [(p.id, c) for p in prompts for c in attacked.get(p.id, [])[:1]][
                : int(cfg.olmotrace.max_completions)
            ]
            for pid, text in sample:
                tr = client.trace_completion(text)
                traces.append({"prompt_id": pid, **tr.__dict__})

        # Phase G: aggregate, log, write.
        summary = {
            "experiment": cfg.experiment_name,
            "dataset": cfg.dataset.name,
            "attack": cfg.attack.name,
            "n_prompts": len(prompts),
            "n_completions": int(cfg.sampling.n_completions),
            "prefix_asr": prefix_asr,
            "judge_asr": {k: v["asr"] for k, v in judge_results.items()},
            "brass_embedding": emb_agg,
            "brass_likelihood": ll_agg,
        }
        logger.info("SUMMARY:\n%s", OmegaConf.to_yaml(OmegaConf.create(summary)))

        run.log({"prefix_asr": prefix_asr})
        run.log({f"judge_asr/{k}": v["asr"] for k, v in judge_results.items()})
        run.log({f"brass_embedding/{k}": v for k, v in emb_agg.items()})
        if ll_agg == ll_agg:  # not NaN
            run.log({"brass_likelihood": ll_agg})

        # Per-completion table for auditability.
        rows = []
        for p in prompts:
            for i, comp in enumerate(attacked.get(p.id, [])):
                rows.append([p.id, p.prompt, i, comp])
        run.log_table(
            "completions", ["prompt_id", "prompt", "idx", "attacked_instruct_completion"], rows
        )

        write_json(results_dir / "summary.json", summary)
        write_json(
            results_dir / "details.json",
            {
                "prompts": [p.__dict__ for p in prompts],
                "completions": {
                    "base": base,
                    "clean_instruct": clean,
                    "attacked_instruct": attacked,
                },
                "prefix": [r.__dict__ for r in prefix_results],
                "judges": judge_results,
                "brass_embedding_per_prompt": {
                    pid: {k: v.__dict__ for k, v in d.items()} for pid, d in emb_per_prompt.items()
                },
                "brass_likelihood_per_prompt": {
                    k: v.__dict__ for k, v in likelihood_scores.items()
                },
                "olmotrace": traces,
            },
        )
        logger.info("Wrote results to %s", results_dir)


if __name__ == "__main__":
    main()
