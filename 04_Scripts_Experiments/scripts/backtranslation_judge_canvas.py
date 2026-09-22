#!/usr/bin/env python3
"""Render a self-contained Codex canvas from completed experiment summaries."""

import argparse
import json
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "01_Datasets_Benchmarks").is_dir())

TEMPLATE = """import {Stack, Row, Grid, H1, H2, Text, Table, Select, Callout, Link, BarChart, useCanvasState, useHostTheme} from "cursor/canvas";
const data: any = __DATA__;
const root = __ROOT__;
const definitions = [
 ["S-ft-archived", "Historical StrongREJECT Gemma + adapter scores"],
 ["S-ft", "Pinned replay, with all five digit logits explicitly available"],
 ["S-rubric", "Unchanged official rubric on Qwen3-32B"],
 ["E", "Direct evidence judge on original behavior and response"],
 ["W", "Wang defense trigger adapted as a judge, including original gates"],
 ["H", "Supported response-only reconstruction + evidence judge"],
 ["H+R", "H with OLMo re-query and likelihood as auxiliary evidence"],
 ["E-budget", "Mean of two direct judgments; call-count control"],
];
const fmt = (n: number) => n.toLocaleString();
export default function BacktranslationRun() {
 const theme = useHostTheme();
 const [cohort, setCohort] = useCanvasState("bt-results-cohort", "probability");
 const [variant, setVariant] = useCanvasState("bt-results-variant", "all");
 const group = data.analysis.cohorts[cohort];
 const selected = group[variant] || group.all;
 const arms = selected.arms;
 const shown = definitions.map(d => d[0]);
 const changes = selected.H_disagreements;
 const primaryChanges = data.analysis.cohorts.probability.all.H_disagreements["S-rubric"];
 const case155 = data.diagnostics.diagnostic_case_predictions["pap_authority / strongreject:155"];
 const case210 = data.diagnostics.diagnostic_case_predictions["pair / strongreject:210"];
 const replay = data.diagnostics.S_ft_replay;
 const requery = [data.diagnostics.by_kind.target_wang, data.diagnostics.by_kind.target_supported];
 return <Stack gap={18} style={{padding:24,maxWidth:1100,color:theme.text.primary}}>
  <H1>Backtranslation as an ASR judge: executed comparison</H1>
  <Text>Fixed OLMo-3-7B-Instruct responses · Qwen3-32B judging · PAP, PAIR, SlotGCG · September 20, 2026</Text>
  <Callout tone="warning" title="Inference complete; accuracy remains unvalidated">
   These are predictions and disagreements. Without independent human reference labels,
   a lower predicted success rate cannot establish fewer false positives or preserved recall.
  </Callout>
  <Grid columns="repeat(auto-fit,minmax(200px,1fr))" gap={20}>
   <Stack gap={4}><H2>{fmt(data.diagnostics.records)} responses</H2><Text>6,260 probability records plus diagnostic additions; 6,373 unique pairs overall.</Text></Stack>
   <Stack gap={4}><H2>{fmt(data.analysis.inference.calls)} events</H2><Text>{fmt(data.analysis.inference.output_tokens)} logged output tokens. All scheduled events verified.</Text></Stack>
   <Stack gap={4}><H2>313 behavior groups</H2><Text>60 development, 60 calibration, 193 retrospective test. Siblings stay together.</Text></Stack>
  </Grid>
  <Row wrap gap={16}>
   <Link href={root+"/REPORT.md"}>Execution report</Link>
   <Link href={root+"/execution_diagnostics.json"}>Machine-readable diagnostics</Link>
   <Link href={root+"/annotations/annotate.html"}>Human annotation editor</Link>
  </Row>
  <H2>What this run establishes</H2>
  <Text>The hybrid changes {fmt(primaryChanges.baseline_positive_H_negative || 0)} rubric-positive responses to negative and {fmt(primaryChanges.baseline_negative_H_positive || 0)} rubric-negative responses to positive in the probability sample. Human labels must determine which changes help.</Text>
  {case155 && case210 && <Text>On PAP authority #155, the archived fine-tuned judge predicts {case155["S-ft-archived"].positive}/{case155.H.selected} positives, compared with {case155["S-rubric"].positive} for the rubric, {case155.E.positive} for the direct judge and {case155.H.positive} for the hybrid. On PAIR #210, the direct judge predicts {case210.E.positive}/{case210.H.selected} positives and the hybrid {case210.H.positive}/{case210.H.selected}. Reconstruction has not yet demonstrated an accuracy advantage over direct judging.</Text>}
  <H2>Predicted success at the uncalibrated 0.5 threshold</H2>
  <Row wrap gap={14}>
   <Select value={cohort} onChange={value=>{setCohort(value);setVariant("all");}} options={[{value:"probability",label:"Probability sample"},{value:"diagnostic",label:"Enriched diagnostics"}]}/>
   <Select value={variant} onChange={setVariant} options={Object.keys(group).map(k=>({value:k,label:k}))}/>
  </Row>
  <Text size="small" tone="secondary">Horizontal axis: judge arm. Vertical axis: known positive predictions / all selected responses (%), the lower end of the missingness bounds. Missing judgments remain unresolved and are listed separately below. Source: frozen run analysis.json, September 20, 2026.</Text>
  <BarChart categories={shown} series={[{name:"Predicted positive / selected",tone:"info",data:shown.map(a=>100*arms[a].positive/arms[a].n)}]} yMin={0} yMax={100} valueSuffix="%" height={250}/>
  <Table framed={false} headers={["Arm","Positive","Negative","Missing","Selected"]} rows={Object.entries(arms).map(([a,v]:[string,any])=>[a,fmt(v.positive),fmt(v.negative),fmt(v.missing),fmt(v.n)])}/>
  <Text>Counts measure completion-level predictions, not best-of-16 attack success. The diagnostic cohort is enriched and cannot estimate population error rates. E/H scores are normalized ordinal assistance, not success probabilities.</Text>
  <H2>Where the hybrid changes a verdict</H2>
  <Table framed={false} headers={["Baseline","Baseline + / H −","Baseline − / H +","Both +","Both −","Unpaired"]} rows={Object.entries(changes).map(([b,c]:[string,any])=>[b,c.baseline_positive_H_negative||0,c.baseline_negative_H_positive||0,c.both_positive||0,c.both_negative||0,c.unpaired||0])}/>
  <Text>Both directions require human review. Baseline-positive/H-negative cases may be corrected false positives or newly missed successes. Baseline-negative/H-positive cases may be rescued successes or new false positives.</Text>
  <H2>Backtranslation's gates need separate scrutiny</H2>
  <Table framed={false} headers={["W gate · all variants, probability sample","Responses"]} rows={Object.entries(data.diagnostics.wang_gates).map(([k,v])=>[k,String(v)])}/>
  <Text>W's initial string matcher and likelihood threshold can suppress actual assistance. H verifies the original response and gives neither reconstruction nor re-query an automatic veto. Re-queries are capped at 256 tokens; cap status is recorded.</Text>
  <H2>Previously identified diagnostic cases · all variants</H2>
  <Text size="small" tone="secondary">Positive predictions / selected diagnostic responses at 0.5. These development cases are enriched; counts are not human-confirmed errors or population error rates. A cell can contain both benign and useful-assistance siblings.</Text>
  <Table framed={false} headers={["Variant / behavior ID","Selected","S-ft archived +","S-rubric +","E +","H +","H+R +"]} rows={Object.entries(data.diagnostics.diagnostic_case_predictions).map(([k,v]:[string,any])=>[k,v.H.selected,...["S-ft-archived","S-rubric","E","H","H+R"].map(a=>String(v[a].positive)+(v[a].missing?` (${v[a].missing} missing)`:""))])}/>
  <H2>Comparison arms and model control</H2>
  <Table framed={false} headers={["Arm","Definition"]} rows={definitions}/>
  <Text>The rubric, direct evidence judge and hybrid use the same pinned Qwen3-32B in bf16 with thinking disabled. E-budget matches calls, not exact token usage. The W arm adapts the paper's target-likelihood algorithm to Qwen and OLMo; it does not reproduce the paper's model results.</Text>
  <Text>StrongREJECT replay changes {fmt((replay.threshold_flips.replay_positive || 0) + (replay.threshold_flips.archive_positive || 0))} of {fmt(replay.paired)} archived thresholded verdicts ({replay.threshold_flips.replay_positive || 0} become positive, {replay.threshold_flips.archive_positive || 0} negative). Both score versions are preserved. Target re-queries hit the 256-token cap in {fmt(requery.reduce((n,v)=>n+(v.statuses.requery_truncated || 0),0))} of {fmt(requery.reduce((n,v)=>n+v.calls,0))} events, so their finite outputs are limited auxiliary evidence.</Text>
  <H2>What remains before an error-reduction claim</H2>
  <Text>Two independent human ratings per unique pair; third-person adjudication for disagreements. Calibrate on the 60 reserved behavior groups, then compare FPR/FNR on the 193 test groups with paired group bootstrap intervals and explicit missingness bounds. Keep the enriched diagnostics separate.</Text>
  <Text>Calibration minimizes FPR subject to point-estimated 99% recall. This cannot guarantee no false negatives. Even zero misses in 193 independent positive tasks leaves a one-sided 95% upper bound above 1%; a stronger claim requires additional independent behaviors.</Text>
  <Text size="small" tone="secondary">No human labels were supplied to this automated run. The retrospective holdout is not an untouched prospective benchmark. Source/config snapshots and raw event outputs are preserved in the run directory.</Text>
 </Stack>;
}
"""


def render(root, output):
    analysis = json.loads((root / "analysis.json").read_text())
    diagnostics = json.loads((root / "execution_diagnostics.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    if (
        progress.get("status") != "inference_complete"
        or not diagnostics["all_scheduled_events_present"]
    ):
        raise ValueError("Canvas requires a completed full run")
    if analysis["human_gold_present"] or analysis["threshold_status"] != "uncalibrated 0.5":
        raise ValueError("This descriptive template is only for the uncalibrated no-gold report")
    source = TEMPLATE.replace(
        "__DATA__", json.dumps(dict(analysis=analysis, diagnostics=diagnostics))
    )
    source = source.replace("__ROOT__", json.dumps(str(root.resolve())))
    output.write_text(source)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=REPO / "06_Results_Artifacts/results/backtranslation_judge/2026-09-20"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.run_dir.resolve(), args.output))


if __name__ == "__main__":
    main()
