#!/usr/bin/env python3
"""Build a human-readable audit of every target response in a PAP run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def md(value: object) -> str:
    if value is None or value == "":
        return "_(not recorded)_"
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def code_block(value: str) -> str:
    longest = max((len(run) for run in __import__("re").findall(r"`+", value)), default=0)
    fence = "`" * max(4, longest + 1)
    return f"{fence}text\n{value}\n{fence}"


def resolve_run_dir(base_dir: Path, run_id: str) -> Path:
    candidates = [base_dir / f"pap-{run_id}"]
    candidates.extend(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / f"pap_strongreject_0_{run_id}.report.json").exists()
    )
    for candidate in candidates:
        if (candidate / f"pap_strongreject_0_{run_id}.report.json").exists():
            return candidate
    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=313)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    base_dir = repo / "06_Results_Artifacts/new_runs/semantic_reconstruction"
    out_dir = resolve_run_dir(base_dir, args.run_id)
    entries: list[str] = []
    counts = {"valid": 0, "refused": 0, "distorted": 0, "semantic_saved_valid": 0,
              "valid_recovered": 0, "mutation_refusal_original_prompt": 0}

    for index in range(args.count):
        report_path = out_dir / f"pap_strongreject_{index}_{args.run_id}.report.json"
        semantic_path = out_dir / f"pap_semantic_{index}_{args.run_id}.json"
        recovery_path = out_dir / f"pap_target_recovery_{index}_{args.run_id}.json"
        if not report_path.exists():
            raise SystemExit(f"Missing PAP report: {report_path}")
        report = json.loads(report_path.read_text())
        row = report["per_behavior"][0]
        status = row.get("pap_status", "unknown")
        counts[status] = counts.get(status, 0) + 1

        semantic = json.loads(semantic_path.read_text()) if semantic_path.exists() else None
        recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
        if semantic is not None:
            response = semantic.get("target_response", "")
            counts["semantic_saved_valid" if status == "valid" else "mutation_refusal_original_prompt"] += 1
            sent_prompt = semantic.get("attacked_prompt") or row.get("behavior")
            response_provenance = "Original saved semantic pipeline output"
        elif recovery is not None:
            if status != "valid":
                raise SystemExit(f"Unexpected recovery for non-valid PAP index {index}")
            response = recovery.get("target_response", "")
            classification = {}
            sent_prompt = recovery.get("attacked_prompt") or row.get("attacked_prompt")
            response_provenance = recovery.get("recovery_note", "Regenerated for audit; not classified")
            counts["valid_recovered"] += 1
        else:
            raise SystemExit(f"No target response found for PAP index {index}")

        if not response:
            raise SystemExit(f"Empty target response for PAP index {index}")
        if status == "refused":
            input_note = "PAP did not produce a mutated prompt; the semantic pipeline sent the original prompt to the target. This is not a response to a PAP attack prompt."
        elif status == "valid":
            input_note = "PAP-mutated prompt"
        elif status == "distorted":
            input_note = "Prompt produced by PAP but marked distorted; see the exact recorded input below."
        else:
            input_note = "See exact recorded input below."

        jailbreak_status = semantic.get("jailbreak_status", "not scored") if semantic else "not scored"
        entries.append(
            f"## {index + 1}. {md(row.get('id') or f'strongreject:{index}')}\n\n"
            f"- **PAP status:** `{md(status)}` ({md(row.get('fail_reason') or 'no failure reason')})\n"
            f"- **Input used for target response:** {input_note}\n"
            f"- **Pipeline decision:** `{md(jailbreak_status)}`\n"
            f"- **Response provenance:** {md(response_provenance)}\n\n"
            f"### Original prompt\n\n{code_block(row.get('behavior', ''))}\n\n"
            f"### Prompt sent to target\n\n{code_block(sent_prompt)}\n\n"
            f"### Full target response\n\n{code_block(response)}"
        )

    if sum(counts.get(k, 0) for k in ("valid", "refused", "distorted")) != args.count:
        raise SystemExit(f"PAP status counts do not sum to {args.count}: {counts}")
    if counts["valid"] != counts["semantic_saved_valid"] + counts["valid_recovered"]:
        raise SystemExit(f"Valid PAP cases do not all have either original or recovered target responses: {counts}")
    if counts["refused"] != counts["mutation_refusal_original_prompt"]:
        raise SystemExit(f"Not every mutation-refusal case has its recorded original-prompt target response: {counts}")

    content = (
        f"# PAP Target Response Audit — run {args.run_id}\n\n"
        "This file contains the original behavior, exact prompt sent to the target, saved classifier rationale when available, and the complete target response for every benchmark row.\n\n"
        "## Run summary\n\n"
        f"- PAP rows: **{args.count}**\n"
        f"- Valid PAP mutations: **{counts['valid']}**\n"
        f"- Valid mutations with original saved classifier output: **{counts['semantic_saved_valid']}**\n"
        f"- Valid mutations regenerated for this audit after the original pipeline failed to persist output: **{counts['valid_recovered']}**\n"
        f"- PAP mutation refusals: **{counts['refused']}**\n"
        f"- Distorted mutations: **{counts['distorted']}**\n\n"
        "The recovered responses use the original per-row generation seed and settings, but are new generations. They have no saved classifier decision and were not reclassified or pairwise-scored. For PAP mutation-refusal rows, the saved pipeline nevertheless queried the target with the original, unmutated behavior; those responses are shown for completeness and are not attack-prompt responses.\n\n"
        "## Cases\n\n" + "\n\n---\n\n".join(entries) + "\n"
    )
    output = out_dir / f"pap_responses_audit_{args.run_id}.md"
    output.write_text(content)
    print(json.dumps({"output": str(output), "counts": counts, "bytes": len(content.encode())}, indent=2))


if __name__ == "__main__":
    main()
