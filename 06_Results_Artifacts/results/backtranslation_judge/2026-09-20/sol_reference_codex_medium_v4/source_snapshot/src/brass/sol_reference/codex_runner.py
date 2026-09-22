"""Run the frozen StrongREJECT rubric using ChatGPT-funded Codex exec sessions."""

import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from brass.orbits.io import digest, file_digest, read_jsonl, write_json
from brass.sol_reference.runner import DEFAULT_PARENT, MODEL, REPO, completed, now, parse_reference

DEFAULT_OUT = DEFAULT_PARENT / "sol_reference_codex_medium_v4"
SOURCE_PATHS = [
    "src/brass/sol_reference/codex_runner.py",
    "src/brass/sol_reference/runner.py",
    "src/brass/backtranslation_judge/prompts.py",
    "scripts/backtranslation_judge_sol_codex.py",
    "src/brass/backtranslation_judge/vendor/strongreject_strong_reject_eval_files_judge_templates.json",
]


def config(out):
    values = dict(
        model_reasoning_effort="medium",
        forced_login_method="chatgpt",
        model_instructions_file=str(out / "rubric_system.txt"),
        project_doc_max_bytes=0,
        web_search="disabled",
        include_environment_context=False,
        include_permissions_instructions=False,
        include_apps_instructions=False,
        include_collaboration_mode_instructions=False,
        suppress_unstable_features_warning=True,
    )
    for name in [
        "shell_tool",
        "apps",
        "plugins",
        "hooks",
        "multi_agent",
        "image_generation",
        "browser_use",
        "computer_use",
        "view_image",
        "goals",
        "sleep_tool",
        "workspace_dependencies",
        "tool_suggest",
        "skill_search",
        "code_mode_host",
        "in_app_browser",
        "memories",
        "unbounded_connection_retries",
    ]:
        values[f"features.{name}"] = False
    values["features.skip_host_skill_discovery"] = True
    return values


def clean_env():
    # Use the CLI's existing ChatGPT sign-in. Do not read .env or provide an API key.
    env = os.environ.copy()
    for name in [
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "OPENAI_BASE_URL",
        "OA_Key",
        "OA_key",
        "OA_KEY",
    ]:
        env.pop(name, None)
    return env


def command(out):
    result = [
        shutil.which("codex"),
        "exec",
        "--ignore-user-config",
        "--strict-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "-C",
        str(out / "empty_workspace"),
        "-s",
        "read-only",
        "-m",
        MODEL,
    ]
    for key, value in config(out).items():
        result += ["-c", f"{key}={json.dumps(value)}"]
    return result + ["--json", "-"]


def prepare(out=DEFAULT_OUT, api_run=DEFAULT_PARENT / "sol_reference_medium_v2"):
    out, api_run = out.resolve(), api_run.resolve()
    api_spec = json.loads((api_run / "run_spec.json").read_text())
    requests = read_jsonl(api_run / "requests.jsonl")
    if digest(requests) != api_spec["request_set_sha256"]:
        raise ValueError("Original request set changed")
    units = {r["pair_sha256"]: r for r in requests}
    old_events = completed(api_run, units)
    blocks = {p: e for p, e in old_events.items() if e["parsed"]["status"] == "api_policy_blocked"}
    system = requests[0]["messages"][0]["content"]
    for r in requests:
        if (
            r["messages"][0] != {"role": "system", "content": system}
            or r["messages"][1]["role"] != "user"
            or len(r["messages"]) != 2
        ):
            raise ValueError("Unexpected official rubric messages")
    cli = Path(shutil.which("codex")).resolve()
    version = subprocess.run(
        [str(cli), "--version"], text=True, capture_output=True, check=True, env=clean_env()
    ).stdout.strip()
    spec = {
        k: api_spec[k]
        for k in [
            "parent_run",
            "parent_manifest_sha256",
            "parent_predictions_sha256",
            "parent_run_spec_sha256",
            "selected_records",
            "unique_pairs",
            "request_set_sha256",
            "score_formula",
            "threshold",
            "reference_type",
        ]
    }
    spec.update(
        model=MODEL,
        reasoning_effort="medium",
        runtime="codex_exec_chatgpt_subscription",
        cli_version=version,
        cli_entrypoint=str(cli),
        cli_entrypoint_sha256=file_digest(cli),
        configuration=config(out),
        command=command(out),
        source_sha256={p: file_digest(REPO / p) for p in SOURCE_PATHS},
        api_run=str(api_run),
        api_run_spec_sha256=file_digest(api_run / "run_spec.json"),
        api_events_sha256=file_digest(api_run / "events.jsonl"),
        policy_blocked_pairs_not_retried=sorted(blocks),
        inference_unit="One independent ephemeral Codex session per exact benchmark-request/response pair; no cross-example context.",
        judge_inputs="Official StrongREJECT system text supplied as model_instructions_file; exact official rubric user text on stdin. No candidate scores, inversion or attacked prompt supplied.",
        model_verification="Exact model and medium reasoning configured in CLI; CLI header verified in benign probe. API-returned snapshot/fingerprint unavailable through exec JSON.",
        runtime_difference="Codex runtime may add platform instructions/tool schemas; medium reasoning requested, no controllable API max_completion_tokens. This is a uniform Codex reference, not a claim of identical API execution.",
        billing="Existing ChatGPT subscription login only; API-key environment variables removed; forced_login_method=chatgpt.",
        retry_policy="No retry of completed verdicts, malformed outputs or policy blocks. Runtime errors halt for inspection; no model fallback.",
        warnings=api_spec["warnings"]
        + [
            "Uniform Codex runtime reference; earlier successful API scores are retained separately and not mixed into reference labels.",
            "Policy-blocked inputs remain unscored and are not resubmitted through Codex.",
            "Codex CLI does not expose an API-returned model snapshot ID; configured alias and CLI version are recorded.",
        ],
    )
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "run_spec.json"
    if spec_path.exists():
        if (
            json.loads(spec_path.read_text()) != spec
            or (out / "requests.jsonl").read_bytes() != (api_run / "requests.jsonl").read_bytes()
            or (out / "rubric_system.txt").read_text() != system
        ):
            raise ValueError("Frozen Codex run changed")
    else:
        write_json(spec_path, spec)
        (out / "requests.jsonl").write_bytes((api_run / "requests.jsonl").read_bytes())
        (out / "rubric_system.txt").write_text(system)
        (out / "empty_workspace").mkdir()
        (out / "traces").mkdir()
        with (out / "events.jsonl").open("w") as handle:
            for event in blocks.values():
                handle.write(
                    json.dumps(
                        event | {"runtime": "inherited_api_policy_block_not_retried"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for source in SOURCE_PATHS:
            destination = out / "source_snapshot" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / source).read_bytes())
    return spec


def parse_cli(stdout, response, returncode):
    trace = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    finished = [e for e in trace if e["type"] == "turn.completed"]
    # The CLI reports transient reconnects as errors even when its built-in
    # transport fallback succeeds. A successful terminal turn is authoritative.
    failures = [e for e in trace if e["type"] == "turn.failed"]
    items = [e["item"] for e in trace if e["type"] == "item.completed"]
    tools = [i for i in items if i["type"] not in {"agent_message", "reasoning", "error"}]
    answers = [i["text"] for i in items if i["type"] == "agent_message"]
    if returncode or failures or not finished:
        raise RuntimeError("Codex execution failed; inspect saved trace (no score recorded)")
    if tools or len(answers) != 1 or len(finished) != 1:
        raise RuntimeError(
            "Codex used a tool or returned multiple turns/messages; do not treat as blinded reference"
        )
    usage = finished[0].get("usage", {})
    return dict(
        raw=answers[0],
        parsed=parse_reference(answers[0], response),
        thread_id=next((e.get("thread_id") for e in trace if e["type"] == "thread.started"), None),
        cli_usage=usage,
        intermediate_transport_errors=[e for e in trace if e["type"] == "error"],
        usage={
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": usage.get("cached_input_tokens", 0)},
            "completion_tokens_details": {
                "reasoning_tokens": usage.get("reasoning_output_tokens", 0)
            },
        },
    )


def policy_blocked(text):
    return any(
        marker in text
        for marker in [
            "bio_policy",
            "content_policy_violation",
            "content_filter",
            "Invalid prompt: we've limited access to this content for safety reasons.",
            "This content was flagged for possible cybersecurity risk.",
        ]
    )


async def evaluate(unit, out):
    start = time.monotonic()
    pair = unit["pair_sha256"]
    stem = out / "traces" / pair
    process = await asyncio.create_subprocess_exec(
        *command(out),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=clean_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(unit["messages"][1]["content"].encode()), timeout=240
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    stdout, stderr = stdout.decode(), stderr.decode()
    stem.with_suffix(".jsonl").write_text(stdout)
    stem.with_suffix(".stderr.txt").write_text(stderr)
    try:
        result = parse_cli(stdout, unit["response"], process.returncode)
    except Exception:
        # Never move a denied input to another runtime or reinterpret failure as a negative.
        if policy_blocked(stdout + stderr):
            result = dict(
                raw="", parsed={"status": "api_policy_blocked", "score": None}, usage=None
            )
        else:
            raise
    return dict(
        pair_sha256=pair,
        row_ids=unit["row_ids"],
        request_sha256=unit["request_sha256"],
        model_requested=MODEL,
        model_returned=None,
        reasoning_effort="medium",
        runtime="codex_exec_chatgpt_subscription",
        created_at_utc=now(),
        attempts=1,
        latency_seconds=round(time.monotonic() - start, 3),
        finish_reason="stop",
        judge_refusal=None,
        trace_sha256=file_digest(stem.with_suffix(".jsonl")),
        cli_return_code=process.returncode,
        **result,
    )


async def execute(out, api_run, limit=0, concurrency=8):
    spec = prepare(out, api_run)
    auth = subprocess.run(
        [shutil.which("codex"), "login", "status"], text=True, capture_output=True, env=clean_env()
    )
    if auth.returncode or "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise RuntimeError("ChatGPT login required; will not use an API key")
    requests = {r["pair_sha256"]: r for r in read_jsonl(out / "requests.jsonl")}
    done = completed(out, requests)
    selected = list(requests.values())
    if limit:
        selected = [r for r in selected if r["split"] == "development"][:limit]
    queue = asyncio.Queue()
    for unit in selected:
        if unit["pair_sha256"] not in done:
            queue.put_nowait(unit)
    stopped = asyncio.Event()
    errors = []
    start = time.monotonic()
    new = 0

    def update(status):
        p = dict(
            status=status,
            unique_pairs_completed=len(done),
            unique_pairs_total=len(requests),
            selected_records_covered=sum(len(e["row_ids"]) for e in done.values()),
            newly_completed=new,
            statuses=dict(Counter(e["parsed"]["status"] for e in done.values())),
            elapsed_seconds=round(time.monotonic() - start, 2),
            errors=errors,
            updated_at_utc=now(),
        )
        write_json(out / "progress.json", p)
        print(json.dumps(p), flush=True)

    update("running")
    with (out / "events.jsonl").open("a") as handle:

        async def worker():
            nonlocal new
            while not stopped.is_set():
                try:
                    unit = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    event = await evaluate(unit, out)
                except Exception as exc:
                    errors.append(
                        dict(
                            pair_sha256=unit["pair_sha256"],
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                    stopped.set()
                    return
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                done[unit["pair_sha256"]] = event
                new += 1
                if new % 25 == 0:
                    update("running")

        await asyncio.gather(*[worker() for _ in range(concurrency)])
    update(
        "runtime_blocked"
        if errors
        else "inference_complete" if len(done) == len(requests) else "pilot_complete"
    )
    if errors:
        raise RuntimeError(
            "Codex reference halted; inspect progress and saved trace before resuming"
        )
    return spec


def run(
    out=DEFAULT_OUT, api_run=DEFAULT_PARENT / "sol_reference_medium_v2", limit=0, concurrency=8
):
    out, api_run = out.resolve(), api_run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return asyncio.run(execute(out, api_run, limit, concurrency))
