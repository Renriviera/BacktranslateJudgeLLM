# Models

[`registry.json`](registry.json) indexes the configured target and judge models; the
authoritative inference configurations live in
[`configs/model`](../04_Scripts_Experiments/configs/model) and
[`configs/judge`](../04_Scripts_Experiments/configs/judge).

```bash
python bt.py models --list
python bt.py models --only instruct_7b qwen3_32b
```

Model weights are excluded. `HF_HOME` and `HF_HUB_CACHE` select the local cache; otherwise
standard Hugging Face cache locations are used. `BACKTRANSLATION_QWEN_PATH` may point to
a separately downloaded Qwen3-32B snapshot. Record and verify its revision yourself when
using that override. The judge runner's default Qwen revision is pinned in its source.

OLMo-3-7B, Vicuna-7B-v1.5, and Llama-2-7B-Chat are the archived attack targets.
OLMo-3.1-32B-Instruct supplies the matched two-cycle comparison. Qwen3-32B supplies
the open-model rubric and reconstruction-assisted judges. StrongREJECT-ft uses the
Gemma-2B base with its adapter; it is separate from rubric prompting. Sol's historical
model alias and medium reasoning are recorded in its run specifications.

Historical `preflight.json` files preserve old cache paths and revisions. Create a fresh
preflight for new inference; do not edit a frozen run merely to match your workstation.
