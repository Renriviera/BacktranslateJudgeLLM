#!/usr/bin/env python3
"""Export the already-selected ten review cases with their full recorded inference traces."""

import argparse
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())
sys.path.insert(0, str(REPO / "04_Scripts_Experiments/src"))

from brass.backtranslation_judge.data import VARIANTS, verify  # noqa: E402
from brass.backtranslation_judge.prompts import messages  # noqa: E402
from brass.orbits.io import digest, file_digest, read_jsonl  # noqa: E402


def build(root, examples_path):
    verify(root)
    spec = json.loads((root / "run_spec.json").read_text())
    for name, expected in spec["source_sha256"].items():
        if file_digest(REPO / name) != expected:
            raise ValueError(f"Frozen inference source changed: {name}")
    selected = read_jsonl(examples_path)
    ids = {r["id"] for r in selected}
    if len(ids) != 10 or len(selected) != 10:
        raise ValueError("Expected the same ten distinct review examples")
    manifest = {r["id"]: r for r in read_jsonl(root / "manifest.jsonl")}
    predictions = {r["id"]: r for r in read_jsonl(root / "predictions.jsonl")}
    inventory = json.loads(
        (REPO / "06_Results_Artifacts/results/backtranslation_judge/design_inventory.json").read_text()
    )
    events, sources = {}, {}
    for path in sorted((root / "events").glob("*.jsonl")):
        sources[str(path.relative_to(REPO))] = file_digest(path)
        for line_number, e in enumerate(read_jsonl(path), 1):
            if e["id"] not in ids:
                continue
            key = (e["id"], e["kind"])
            if key in events:
                raise ValueError(f"Duplicate event: {key}")
            events[key] = dict(event=e, file=str(path.relative_to(REPO)), line=line_number)
    archives, caches = {}, {}
    for variant in {r["variant"] for r in selected}:
        cache_path = REPO / "06_Results_Artifacts/results/attacks" / VARIANTS[variant]
        cache_relative = str(cache_path.relative_to(REPO))
        if file_digest(cache_path) != inventory["source_sha256"][cache_relative]:
            raise ValueError(f"Attack cache differs from the audited archive: {cache_path}")
        sources[cache_relative] = file_digest(cache_path)
        caches[variant] = json.loads(cache_path.read_text())
        path = REPO / f"06_Results_Artifacts/results/{variant}_strongreject_olmo3_7b/details.json"
        sources[str(path.relative_to(REPO))] = file_digest(path)
        archives[variant] = json.loads(path.read_text())
    result = []
    for example in selected:
        row = manifest[example["id"]]
        variant, pid = row["variant"], row["behavior_id"]
        archive = archives[variant]
        prompt = next(p for p in archive["prompts"] if p["id"] == pid)
        original = archive["completions"]["attacked_instruct"][pid][row["completion_index"]]
        if (
            sources[row["details_path"]] != row["source_sha256"]
            or prompt["prompt"] != row["behavior"]
            or original != row["response"]
            or row["behavior"] != example["behavior"]
            or row["response"] != example["response"]
            or digest([row["behavior"], row["response"]]) != row["pair_sha256"]
        ):
            raise ValueError(f"Original archive identity mismatch: {row['id']}")
        record = dict(example)
        record["attacked_request"] = caches[variant][pid]["attacked_prompt"]
        record["attack_cache_path"] = str(Path("06_Results_Artifacts/results/attacks") / VARIANTS[variant])
        record["events"] = {}
        for kind in [
            "supported_inverse",
            "H",
            "H+R",
            "S-rubric",
            "E",
            "target_supported",
            "wang_inverse",
            "target_wang",
        ]:
            entry = events.get((row["id"], kind))
            if entry is None:
                continue
            if entry["event"]["pair_sha256"] != row["pair_sha256"]:
                raise ValueError("Event response identity mismatch")
            record["events"][kind] = entry
        for kind in ["supported_inverse", "H", "H+R", "S-rubric", "E"]:
            if kind not in record["events"]:
                raise ValueError(f"Required event missing: {row['id']} / {kind}")
        inv = record["events"]["supported_inverse"]["event"]["parsed"]
        target = record["events"].get("target_supported")
        for kind, entry in record["events"].items():
            extra = None
            if kind in ("H", "H+R"):
                extra = {"reconstruction": inv}
                if kind == "H+R":
                    extra["target_check"] = (
                        target["event"]["parsed"] if target else {"status": "missing"}
                    )
            if kind in ("target_supported", "target_wang"):
                inverse_kind = "supported_inverse" if kind == "target_supported" else "wang_inverse"
                source = record["events"][inverse_kind]["event"]["parsed"]
                query = (
                    source["requests"][0]["request"]
                    if kind == "target_supported"
                    else source["request"]
                )
                msg = [{"role": "user", "content": query}]
            else:
                msg = messages(kind, row, extra)
            if digest(msg) != entry["event"]["request_sha256"]:
                raise ValueError(f"Reconstructed input hash mismatch: {row['id']} / {kind}")
            entry["input_messages"] = msg
        outcomes = predictions[row["id"]]["outcomes"]
        if (
            outcomes["S-rubric"]["score"] != example["S_rubric"]
            or outcomes["H"]["score"] != example["H"]
        ):
            raise ValueError("Review score mismatch")
        record["H_plus_R"] = outcomes["H+R"]["score"]
        record["wang_decision"] = outcomes["W"]
        if not target:
            record["supported_requery_skip_reason"] = (
                f"No re-query ran: supported inverse parser status was {inv['status']}. "
                "The final judges received that status, not the rejected raw reconstruction."
            )
        if "wang_inverse" not in record["events"]:
            if record["wang_decision"].get("gate") != "initial_refusal":
                raise ValueError("Unexplained missing Wang reconstruction")
            record["wang_skip_reason"] = (
                "W stopped at its initial refusal-string gate, so no Wang reconstruction or re-query ran."
            )
        result.append(record)
    for name in ["manifest.jsonl", "predictions.jsonl", "run_spec.json"]:
        sources[str((root / name).relative_to(REPO))] = file_digest(root / name)
    sources[str(examples_path.relative_to(REPO))] = file_digest(examples_path)
    return result, sources


def block(value, language="text"):
    fence = "````" if "```" in value else "```"
    return f"{fence}{language}\n{value}\n{fence}\n"


def event_section(record, kind, heading):
    entry = record["events"][kind]
    event = entry["event"]
    return [
        f"### {heading}\n",
        f"Model: `{event['model']['hf_id']}`. Parser status: `{event['parsed']['status']}`. "
        f"Finish reason: `{event.get('finish_reason', 'unrecorded')}`.\n",
        block(event["raw"], "json" if kind in ("H", "H+R", "supported_inverse", "E") else "text"),
    ]


def markdown(records, direction):
    text = [
        f"# Ten {direction} cases: complete recorded traces\n",
        "The cases and numbering match this packet's examples.jsonl. All prompts and outputs below "
        "are copied from the archive or recorded inference events, without excerpts or new generation. "
        "Abrupt endings and typos are preserved. All ten are development cases.\n",
        "**H:** original behavior + original response + supported response-only reconstruction → "
        "final Qwen judgment. H does not re-query OLMo. **H+R:** adds OLMo's reply to the first "
        "accepted reconstruction as auxiliary evidence. The original baseline judgment came from "
        "the StrongREJECT rubric. The separate Wang W trace is also included where it ran.\n",
        "Re-query replies were capped at 256 tokens; `finish_reason=length` identifies a capped reply. "
        "A stored refusal flag is a string-matcher result, not a human verdict. Evidence-line numbers "
        "refer to the original response including blank lines; exact judge inputs and parsed evidence "
        "are preserved in `full_traces.jsonl`.\n",
    ]
    for r in records:
        text += [
            f"## {r['number']}. {r['id']}\n",
            f"S-rubric **{r['S_rubric']:g}** → H **{r['H']:.3f}**; H+R **{r['H_plus_R']:.3f}**. "
            "Threshold 0.5. Completion index is zero-based.\n",
            "### 1. Original benchmark request\n",
            block(r["behavior"]),
            "### 2. Attacked request sent to OLMo (archived attack cache)\n",
            block(r["attacked_request"]),
            "### 3. Original OLMo model output\n",
            block(r["response"]),
        ]
        text += event_section(
            r, "supported_inverse", "4. Supported inverted/backtranslated prompt — full raw output"
        )
        parsed = r["events"]["supported_inverse"]["event"]["parsed"]
        if parsed["status"] != "ok":
            text += [
                "**This raw reconstruction was rejected by the parser.** What H/H+R received:\n",
                block(json.dumps(parsed, indent=2), "json"),
            ]
        text += event_section(r, "H", "5. Final reply from hybrid judge H")
        if "target_supported" in r["events"]:
            query = r["events"]["target_supported"]["input_messages"][0]["content"]
            text += ["### 6a. Exact backtranslated request re-queried for H+R\n", block(query)]
            text += event_section(
                r,
                "target_supported",
                "6b. OLMo final reply to that request — H+R auxiliary evidence",
            )
        else:
            text += ["### 6. OLMo reply for H+R\n", r["supported_requery_skip_reason"] + "\n"]
        text += event_section(r, "H+R", "7. Final reply from hybrid judge H+R")
        text += event_section(r, "S-rubric", "8. Original StrongREJECT-rubric judgment")
        text += [
            "### 9. Separate Wang baseline W\n",
            "Recorded W decision:\n",
            block(json.dumps(r["wang_decision"], indent=2), "json"),
        ]
        if "wang_inverse" in r["events"]:
            text += event_section(r, "wang_inverse", "9a. Wang harm-maximizing inverted request")
            if "target_wang" in r["events"]:
                text += event_section(r, "target_wang", "9b. OLMo reply to the Wang request")
        else:
            text += [r["wang_skip_reason"] + "\n"]
        text += [
            f"Attack cache: `{r['attack_cache_path']}`. Behavior: `{r['behavior_id']}`. "
            f"Original pair SHA-256: `{r['pair_sha256']}`.\n"
        ]
    return "\n".join(text)


CANVAS = """import {Stack,Row,H1,H2,Text,Select,Button,Link,CollapsibleSection,useCanvasState,useHostTheme} from "cursor/canvas";
const cases: any[] = __DATA__;
const documentPath = __DOCUMENT__;
const jsonPath = __JSON__;
export default function FullJudgeTraces(){
 const theme=useHostTheme();
 const [index,setIndex]=useCanvasState("ten-flips-review-index",0);
 const i=Math.max(0,Math.min(cases.length-1,index));const r=cases[i];
 const e=(kind:string)=>r.events[kind]?.event;
 const verbatim=(value:string)=><Text style={{whiteSpace:"pre-wrap",overflowWrap:"anywhere",lineHeight:1.6}}>{value}</Text>;
 const raw=(kind:string)=><Stack gap={8}>
  <Text size="small" tone="secondary">{e(kind).model.hf_id} · parser: {e(kind).parsed.status} · finish: {e(kind).finish_reason}</Text>
  {verbatim(e(kind).raw)}
 </Stack>;
 return <Stack gap={20} style={{padding:24,maxWidth:1080,color:theme.text.primary}}>
  <H1>Ten __DIRECTION__ cases: complete prompts and replies</H1>
  <Text>Same ten development examples, with every stored prompt and output shown in full. Original OLMo responses are fixed. No new responses were generated for this export.</Text>
  <Text>H uses the original behavior, original response and supported reconstruction. H+R also uses OLMo's reply to the first accepted reconstruction. Both final judge replies are shown below. Wang's separate detector appears at the end.</Text>
  <Row wrap gap={12}>
   <Button disabled={i===0} onClick={()=>setIndex(i-1)}>Previous</Button>
   <Select value={String(i)} onChange={v=>setIndex(Number(v))} options={cases.map((x,j)=>({value:String(j),label:`${x.number}. ${x.variant} · ${x.behavior_id}`}))}/>
   <Button disabled={i===cases.length-1} onClick={()=>setIndex(i+1)}>Next</Button>
  </Row>
  <Row wrap gap={16}><Link href={documentPath}>All ten complete traces in one document</Link><Link href={jsonPath}>Raw events, evidence and exact judge inputs</Link></Row>
  <Text weight="medium">Case {r.number} of 10 · S-rubric {r.S_rubric.toFixed(3)} → H {r.H.toFixed(3)} · H+R {r.H_plus_R.toFixed(3)} · threshold 0.5</Text>
  <H2>1. Original benchmark request</H2>{verbatim(r.behavior)}
  <H2>2. Attacked request sent to OLMo</H2>{verbatim(r.attacked_request)}
  <Text size="small" tone="secondary">Verbatim archived attack-cache text, including any noise or literal placeholders.</Text>
  <H2>3. Original OLMo model output</H2>{verbatim(r.response)}
  <Text size="small" tone="secondary">End of archived response. Abrupt endings and typos are preserved.</Text>
  <H2>4. Supported inverted/backtranslated prompt</H2>{raw("supported_inverse")}
  {e("supported_inverse").parsed.status!=="ok" && <Stack gap={6}>
   <Text weight="medium">This raw reconstruction failed validation and was not supplied as a valid prompt to the final judges.</Text>
   {verbatim(JSON.stringify(e("supported_inverse").parsed,null,2))}
  </Stack>}
  <H2>5. Final reply from hybrid judge H</H2>{raw("H")}
  <H2>6. OLMo reply to the backtranslated request · H+R only</H2>
  {e("target_supported") ? <Stack gap={12}>
   <Text weight="medium">Exact request re-queried</Text>{verbatim(r.events.target_supported.input_messages[0].content)}
   <Text weight="medium">OLMo reply</Text>{raw("target_supported")}
   <Text size="small" tone="secondary">A length finish means the 256-token generation cap was reached; this is the whole stored reply.</Text>
  </Stack>:<Text>{r.supported_requery_skip_reason}</Text>}
  <H2>7. Final reply from hybrid judge H+R</H2>{raw("H+R")}
  <H2>8. Original StrongREJECT-rubric judgment</H2>{raw("S-rubric")}
  <H2>9. Separate Wang baseline</H2>
  {verbatim(JSON.stringify(r.wang_decision,null,2))}
  {e("wang_inverse") ? <Stack gap={12}>
   <Text weight="medium">Wang inverted request</Text>{raw("wang_inverse")}
   {e("target_wang") && <Stack gap={8}><Text weight="medium">OLMo reply to the Wang request</Text>{raw("target_wang")}</Stack>}
  </Stack>:<Text>{r.wang_skip_reason}</Text>}
  <CollapsibleSection title="Exact final H judge input and evidence" defaultOpen={false}>
   {verbatim(JSON.stringify(r.events.H.input_messages,null,2))}
  </CollapsibleSection>
  <Text size="small" tone="secondary">Source: September 20, 2026 frozen run; attack-cache hashes, original response identities and all displayed event input hashes verified. Case ID: {r.id}. Evidence-line IDs count blank lines too. No human error label has been assigned.</Text>
 </Stack>;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=REPO / "06_Results_Artifacts/results/backtranslation_judge/2026-09-20"
    )
    parser.add_argument(
        "--examples", type=Path, help="Existing ten-case selection; outputs go beside this file"
    )
    parser.add_argument("--canvas-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    examples_path = (args.examples or root / "manual_review_10/examples.jsonl").resolve()
    records, sources = build(root, examples_path)
    if all(r["S_rubric"] >= 0.5 and r["H"] < 0.5 for r in records):
        direction = "positive-to-negative"
    elif all(r["S_rubric"] < 0.5 and r["H"] >= 0.5 for r in records):
        direction = "negative-to-positive"
    else:
        raise ValueError("All ten cases must have the same direction of disagreement")
    out = examples_path.parent
    doc = out / "EXAMPLES.md"
    json_path = out / "full_traces.jsonl"
    json_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    doc.write_text(markdown(records, direction))
    canvas = CANVAS.replace("__DATA__", json.dumps(records, ensure_ascii=False))
    canvas = canvas.replace("__DOCUMENT__", json.dumps(str(doc)))
    canvas = canvas.replace("__JSON__", json.dumps(str(json_path)))
    canvas = canvas.replace("__DIRECTION__", direction)
    canvas = canvas.replace("ten-flips-review-index", f"ten-{direction}-review-index")
    args.canvas_output.write_text(canvas)
    provenance = dict(
        examples=len(records),
        direction=direction,
        source_sha256=sources,
        exporter_sha256=file_digest(Path(__file__)),
        raw_input_hashes_verified=sum(len(r["events"]) for r in records),
        valid_supported_inversions=sum(
            r["events"]["supported_inverse"]["event"]["parsed"]["status"] == "ok" for r in records
        ),
        supported_requeries=sum("target_supported" in r["events"] for r in records),
        wang_requeries=sum("target_wang" in r["events"] for r in records),
        document_sha256=file_digest(doc),
        traces_sha256=file_digest(json_path),
        canvas_sha256=file_digest(args.canvas_output),
        no_new_inference=True,
    )
    (out / "trace_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
