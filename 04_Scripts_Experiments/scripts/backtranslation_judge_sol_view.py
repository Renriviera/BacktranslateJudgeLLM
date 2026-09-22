#!/usr/bin/env python3
"""Create a self-contained results canvas and append Sol assessments to the 20 reviewed traces."""

import argparse
import json
import sys
from pathlib import Path
from brass.paths import recorded_path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))
from brass.orbits.io import file_digest, read_jsonl, write_json  # noqa: E402
from brass.sol_reference.codex_runner import DEFAULT_OUT  # noqa: E402

TEMPLATE = REPO / "04_Scripts_Experiments/scripts/templates/sol_reference_canvas.txt"
DEFAULT_CANVAS = REPO / "06_Results_Artifacts/new_runs/views/backtranslation-sol-reference.canvas.tsx"


def export(run, canvas, partial=False):
    suffix = ".partial" if partial else ""
    summary = json.loads((run / f"analysis{suffix}.json").read_text())
    if not partial and not summary["inference_complete"]:
        raise ValueError("Full results require complete inference")
    assessments = read_jsonl(run / f"assessments{suffix}.jsonl")
    mapping = {r["id"]: r for r in assessments}
    spec = json.loads((run / "run_spec.json").read_text())
    parent = recorded_path(spec["parent_run"])
    packets = []
    for folder, direction in [
        ("manual_review_10", "S-positive to H-negative"),
        ("manual_review_10_negative_to_positive", "S-negative to H-positive"),
    ]:
        for row in read_jsonl(parent / folder / "full_traces.jsonl"):
            packets.append(row | {"direction": direction, "sol_assessment": mapping[row["id"]]})
    packet_path = run / f"reviewed_20_with_sol{suffix}.jsonl"
    packet_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packets))
    lines = [
        "# The 20 reviewed cases with Sol assessments",
        "",
        "Reference: Sol medium via Codex, unchanged StrongREJECT rubric; labels below are relative to Sol.",
        "",
    ]
    for row in packets:
        sol = row["sol_assessment"]
        lines += [
            f"## {row['direction']} — {sol['id']}",
            "",
            f"Sol score: {sol['sol_parsed']['score']}; status: {sol['sol_parsed']['status']}. S-rubric: {sol['labels_relative_to_sol']['S-rubric']}; H: {sol['labels_relative_to_sol']['H']}.",
            "",
        ]
        sections = [
            ("Original benchmark request", sol["original_benchmark_request"]),
            ("Attacked request", row["attacked_request"]),
            ("Original OLMo response", sol["original_olmo_response"]),
        ]
        for kind, title in [
            ("supported_inverse", "Inverted / backtranslated prompt — full output"),
            ("target_supported", "Auxiliary OLMo reply to the reconstructed request"),
            ("H", "Final hybrid H reply"),
            ("H+R", "Final hybrid H+R reply"),
            ("S-rubric", "Original StrongREJECT rubric reply"),
        ]:
            if kind in row["events"]:
                sections.append((title, row["events"][kind]["event"]["raw"]))
        sections.append(
            ("Sol assessment", sol["sol_raw"] or f"Unavailable: {sol['sol_parsed']['status']}")
        )
        for title, value in sections:
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            fence = "~~~~"
            while fence in value:
                fence += "~"
            lines += [f"### {title}", "", fence + "text", value, fence, ""]
    (run / f"REVIEWED_20_SOL{suffix}.md").write_text("\n".join(lines))
    payload = dict(
        summary=summary,
        cases=assessments,
        reviewed_ids=[r["id"] for r in packets],
        result_note=(
            (run / "FINDINGS.md").read_text().split("\n\n")[1].replace("**", "")
            if (run / "FINDINGS.md").exists()
            else None
        ),
        paths=dict(
            findings=str(run / "FINDINGS.md"),
            report=str(run / f"REPORT{suffix}.md"),
            assessments=str(run / f"assessments{suffix}.jsonl"),
            reviewed=str(run / f"REVIEWED_20_SOL{suffix}.md"),
            full_review_traces=str(packet_path),
            methods=str(REPO / "04_Scripts_Experiments/docs/backtranslation_sol_reference.md"),
        ),
    )
    literal = json.dumps(json.dumps(payload, ensure_ascii=False), ensure_ascii=False)
    canvas.write_text(TEMPLATE.read_text().replace("__INLINE_DATA__", literal))
    write_json(
        run / f"view_provenance{suffix}.json",
        dict(
            canvas=str(canvas),
            canvas_sha256=file_digest(canvas),
            analysis_sha256=file_digest(run / f"analysis{suffix}.json"),
            assessments_sha256=file_digest(run / f"assessments{suffix}.jsonl"),
            template_sha256=file_digest(TEMPLATE),
            exporter_sha256=file_digest(Path(__file__)),
            reviewed_packet_sha256=file_digest(packet_path),
        ),
    )
    print(
        json.dumps(
            dict(
                canvas=str(canvas),
                records=len(assessments),
                reviewed=len(packets),
                canvas_bytes=canvas.stat().st_size,
            )
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--canvas", type=Path, default=DEFAULT_CANVAS)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    export(a.run_dir.resolve(), a.canvas.resolve(), a.partial)


if __name__ == "__main__":
    main()
