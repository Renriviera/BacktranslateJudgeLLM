"""Deterministic sampling and conservative behavior-group splits."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from brass.orbits.io import digest, file_digest, read_jsonl, stable_seed, write_json

VARIANTS = {
    "pap_authority": "pap/olmo3_7b_instruct.authority_endorsement.json",
    "pap_logic": "pap/olmo3_7b_instruct.logical_appeal.json",
    "pap_misrep": "pap/olmo3_7b_instruct.json",
    "pair": "pair/olmo3_7b_instruct.json",
    "slotgcg": "slotgcg/olmo3_7b_instruct.json",
}
DIAGNOSTICS = {
    "pair": [210, 185],
    "pap_misrep": [248],
    "pap_authority": [64, 155],
    "slotgcg": [203, 174, 278, 7],
}


def group_behaviors(prompts, candidates, threshold=0.82):
    parent = {p["id"]: p["id"] for p in prompts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_text = {}
    for p in prompts:
        key = " ".join(p["prompt"].lower().split())
        if key in by_text:
            parent[find(p["id"])] = find(by_text[key])
        by_text[key] = p["id"]
    for pair in candidates:
        if pair["cosine"] >= threshold:
            a, b = find(pair["a"]), find(pair["b"])
            parent[max(a, b)] = min(a, b)
    return {p["id"]: find(p["id"]) for p in prompts}


def split_groups(prompts, group_map, known, seed=235711):
    forced = {group_map[p] for p in known}
    categories = {}
    for p in prompts:
        categories.setdefault(group_map[p["id"]], p["category"])
    bins = defaultdict(list)
    for group, category in categories.items():
        if group not in forced:
            bins[category].append(group)
    for category, vals in bins.items():
        vals.sort()
        random.Random(stable_seed(seed, category)).shuffle(vals)
    order = []
    while any(bins.values()):
        for category in sorted(bins):
            if bins[category]:
                order.append(bins[category].pop())
    if len(forced) > 60 or len(categories) < 140:
        raise ValueError("Grouping requires a new split allocation")
    dev = forced | set(order[: 60 - len(forced)])
    remaining = [g for g in order if g not in dev]
    calibration = set(remaining[:60])
    return {
        g: "development" if g in dev else "calibration" if g in calibration else "test"
        for g in categories
    }


def prepare(repo: Path, out: Path, seed=235711):
    if (out / "frozen.json").exists():
        verify(out)
        return json.loads((out / "design_summary.json").read_text())
    inventory = json.loads(
        (repo / "06_Results_Artifacts/results/backtranslation_judge/design_inventory.json").read_text()
    )
    candidates = json.loads(
        (repo / "06_Results_Artifacts/results/backtranslation_judge/near_duplicate_candidates.json").read_text()
    )
    baseline = json.loads((repo / "06_Results_Artifacts/results/pair_strongreject_olmo3_7b/details.json").read_text())
    prompts = baseline["prompts"]
    group_map = group_behaviors(prompts, candidates)
    known = set(inventory["known_development_behavior_ids"])
    for ids in DIAGNOSTICS.values():
        known.update(f"strongreject:{i}" for i in ids)
    splits = split_groups(prompts, group_map, known, seed)
    recovered = {
        r["id"]: r
        for r in read_jsonl(repo / "06_Results_Artifacts/results/orbits/2026-09-18-pilot/archive_strongreject.jsonl")
    }
    selected = json.loads((repo / "01_Datasets_Benchmarks/provenance/audit_selected.json").read_text())["rows"]
    labels = {
        "PAP Authority Endorsement": "pap_authority",
        "PAP Logical Appeal": "pap_logic",
        "PAP Misrepresentation": "pap_misrep",
        "PAIR": "pair",
        "SlotGCG": "slotgcg",
    }
    audited = {(labels[r["attack"]], r["id"], r["completion_index"]) for r in selected}
    manifest = []
    for variant, cache_path in VARIANTS.items():
        path = repo / f"06_Results_Artifacts/results/{variant}_strongreject_olmo3_7b/details.json"
        source_hash = file_digest(path)
        data = json.loads(path.read_text())
        cache = json.loads((repo / "06_Results_Artifacts/results/attacks" / cache_path).read_text())
        for prompt in data["prompts"]:
            pid = prompt["id"]
            indices = set(
                random.Random(stable_seed(seed, "sample", variant, pid)).sample(range(16), 4)
            )
            diag = int(pid.split(":")[1]) in DIAGNOSTICS.get(variant, [])
            scores = data["judges"]["strongreject"]["per_prompt"][pid].get("scores")
            for index, response in enumerate(data["completions"]["attacked_instruct"][pid]):
                diagnostic = diag or (variant, pid, index) in audited
                if index not in indices and not diagnostic:
                    continue
                rid = f"{variant}:{pid}:{index}"
                pair_hash = digest([prompt["prompt"], response])
                if scores is None:
                    cached = recovered[rid]
                    if cached["pair_sha256"] != pair_hash:
                        raise ValueError("Recovered score mismatch")
                    score = cached["score"]
                else:
                    score = scores[index]
                manifest.append(
                    dict(
                        id=rid,
                        behavior_id=pid,
                        group_id=group_map[pid],
                        variant=variant,
                        category=prompt["category"],
                        split=splits[group_map[pid]],
                        behavior=prompt["prompt"],
                        response=response,
                        pair_sha256=pair_hash,
                        annotation_id=digest(["annotation", pair_hash])[:24],
                        completion_index=index,
                        probability_sample=index in indices,
                        diagnostic=diagnostic,
                        inclusion_probability=0.25 if index in indices else None,
                        archived_sr_ft=score,
                        attack_status=cache[pid].get(
                            "pair_status", cache[pid].get("pap_status", "unrecorded")
                        ),
                        details_path=str(path.relative_to(repo)),
                        source_sha256=source_hash,
                    )
                )
    manifest.sort(key=lambda r: (stable_seed(seed, "execution", r["id"]), r["id"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest)
    )
    write_json(
        out / "groups.json",
        dict(
            group_map=group_map,
            splits=splits,
            known_development_ids=sorted(known),
            method="Conservative connected components: normalized exact matches and MiniLM cosine >=0.82; not human-certified semantic equivalence",
            candidates=candidates,
        ),
    )
    summary = dict(
        seed=seed,
        records=len(manifest),
        probability_records=sum(r["probability_sample"] for r in manifest),
        diagnostic_records=sum(r["diagnostic"] for r in manifest),
        behavior_groups=len(splits),
        groups_by_split=dict(Counter(splits.values())),
        probability_records_by_split=dict(
            Counter(r["split"] for r in manifest if r["probability_sample"])
        ),
        human_labels_present=False,
        split_status="retrospective conservative semantic grouping; historical exposure incompletely known",
    )
    write_json(out / "design_summary.json", summary)
    write_json(
        out / "frozen.json",
        {
            name: file_digest(out / name)
            for name in ["manifest.jsonl", "groups.json", "design_summary.json"]
        },
    )
    return summary


def verify(out):
    for name, expected in json.loads((out / "frozen.json").read_text()).items():
        if file_digest(out / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
