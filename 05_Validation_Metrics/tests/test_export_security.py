"""Security gates must inspect exported/indexed bytes, including compressed LFS content."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
scan = runpy.run_path(str(ROOT / "05_Validation_Metrics/credential_scan.py"))


@pytest.mark.parametrize("prefix", ["sk-" + "proj-", "hf" + "_", "ghp" + "_"])
def test_provider_tokens_detected_without_echoing_values(prefix):
    secret = (prefix + "Q" * 45).encode()
    findings = scan["scan_bytes"](b"credential=" + secret, "example.txt")
    assert findings
    assert secret.decode() not in json.dumps(findings)


def test_scan_decompresses_npz_and_rejects_private_environment_files(tmp_path):
    secret = ("hf" + "_" + "K" * 40).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("keys.npy", secret)
    (tmp_path / "embedding.npz").write_bytes(buffer.getvalue())
    (tmp_path / ".env").write_text("DEBUG=1\n")
    findings, count, _ = scan["scan_tree"](tmp_path)
    assert count == 2
    assert {x["rule"] for x in findings} >= {"huggingface_token", "credential_container"}


def test_index_scan_catches_secret_hidden_by_unstaged_cleanup(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    secret = "sk-" + "proj-" + "R" * 45
    path = tmp_path / "settings.txt"
    path.write_text(secret)
    subprocess.run(["git", "add", "settings.txt"], cwd=tmp_path, check=True)
    path.write_text("clean working copy")
    findings, count, _ = scan["scan_staged"](tmp_path)
    assert count == 1
    assert any(x["rule"] == "openai_or_anthropic_key" for x in findings)


def test_baseline_cannot_hide_new_content_or_provider_tokens():
    data = b"a synthetic example"
    file_hash = hashlib.sha256(data).hexdigest()
    finding = {"path": "a.json", "rule": "credential_assignment", "match_sha256": "reviewed"}
    baseline = {("a.json", file_hash, "credential_assignment", "reviewed")}
    assert not scan["apply_reviewed_examples"]([finding], data, "a.json", baseline)
    assert scan["apply_reviewed_examples"]([finding], data + b"new content", "a.json", baseline)
    finding = finding | {"rule": "huggingface_token"}
    baseline = {("a.json", file_hash, "huggingface_token", "reviewed")}
    assert scan["apply_reviewed_examples"]([finding], data, "a.json", baseline)


def test_unquoted_environment_assignment_detected():
    payload = ("WANDB_API_KEY=" + "e" * 40 + "\n").encode()
    assert any(x["rule"] == "unquoted_credential_assignment" for x in scan["scan_bytes"](payload, "launch.sh"))
    assert not scan["scan_bytes"](b"OPENAI_API_KEY=\nOLMOTRACE_API_KEY=\nWANDB_API_KEY=\n", ".env.example")


def test_staged_lfs_object_scanned_instead_of_pointer_only(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    payload = ("hf" + "_" + "Z" * 40).encode()
    digest = hashlib.sha256(payload).hexdigest()
    obj = tmp_path / ".git/lfs/objects" / digest[:2] / digest[2:4] / digest
    obj.parent.mkdir(parents=True)
    obj.write_bytes(payload)
    (tmp_path / "events.jsonl").write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(payload)}\n")
    subprocess.run(["git", "-c", "filter.lfs.clean=cat", "-c", "filter.lfs.process=", "add", "events.jsonl"], cwd=tmp_path, check=True)
    findings, _, _ = scan["scan_staged"](tmp_path)
    assert any(x["rule"] == "huggingface_token" for x in findings)
    obj.unlink()
    findings, _, _ = scan["scan_staged"](tmp_path)
    assert any(x["rule"] == "lfs_object_unavailable" for x in findings)
