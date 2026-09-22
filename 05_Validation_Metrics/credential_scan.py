#!/usr/bin/env python3
"""Offline publication gate. Reports paths/rules only, never matching text."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "openai_or_anthropic_key": rb"\bsk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{20,}",
    "huggingface_token": rb"\bhf_[A-Za-z0-9]{20,}",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
    "aws_access_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "google_api_key": rb"\bAIza[0-9A-Za-z_-]{30,}",
    "slack_token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
    "stripe_key": rb"\b[rs]k_live_[A-Za-z0-9]{16,}",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "jwt": rb"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}",
    "authenticated_url": rb"[a-zA-Z][a-zA-Z0-9+.-]{0,20}://[^\s/@:'\"]{1,80}:[^\s/@'\"]{8,}@",
    "credential_assignment": rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|password|hf_token|oa_key)[\"']?\s*[:=]\s*[\"']([^\"'\r\n]{8,})[\"']",
    "bearer_authorization": rb"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?Bearer\s+([A-Za-z0-9_.-]{16,})",
    "unquoted_credential_assignment": rb"(?im)^[ \t]*(?:export[ \t]+)?(?:[A-Z][A-Z0-9_]*_)?(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET_KEY|CLIENT_SECRET|PASSWORD|HF_TOKEN|OA_KEY)[ \t]*[:=][ \t]*([A-Za-z0-9_./+@=-]{16,})[ \t]*(?:#.*)?$",
}
PATTERNS = {name: re.compile(value) for name, value in PATTERNS.items()}
MARKERS = {
    "openai_or_anthropic_key": (b"sk-",),
    "huggingface_token": (b"hf_",),
    "github_token": (b"ghp_", b"gho_", b"ghu_", b"ghs_", b"ghr_", b"github_pat_"),
    "aws_access_key": (b"AKIA", b"ASIA"),
    "google_api_key": (b"AIza",),
    "slack_token": (b"xox",),
    "stripe_key": (b"k_live_",),
    "private_key": (b"PRIVATE KEY-----",),
    "jwt": (b"eyJ",),
    "authenticated_url": (b"://",),
    "credential_assignment": (b"key", b"token", b"secret", b"password"),
    "bearer_authorization": (b"authorization",),
    "unquoted_credential_assignment": (b"key", b"token", b"secret", b"password"),
}
SKIP_DIRS = {".git", ".venv", ".venv-attacks", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules"}
FORBIDDEN_NAMES = {"auth.json", "credentials.json", "credentials", "stored_tokens", "token", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}
SAFE_VALUES = {b"", b"EMPTY", b"REDACTED_CREDENTIAL", b"YOUR_API_KEY", b"your-api-key", b"your_api_key", b"your-password", b"your_password", b"your_hf_token", b"placeholder", b"test-api-key", b"dummy-api-key", b"not-a-real-key"}
REVIEWED_MATCHES = []


def placeholder(value):
    return value in SAFE_VALUES or value.startswith((b"${", b"<", b"os.environ", b"os.getenv", b"{", b"sk-...", b"hf_..."))


def scan_bytes(data, label):
    findings = []
    lower = data.lower()
    for rule, pattern in PATTERNS.items():
        haystack = lower if rule in {"credential_assignment", "bearer_authorization", "unquoted_credential_assignment"} else data
        if not any(marker in haystack for marker in MARKERS[rule]):
            continue
        for match in pattern.finditer(data):
            if rule in {"credential_assignment", "bearer_authorization", "unquoted_credential_assignment"} and placeholder(match[1]):
                continue
            findings.append({"path": label, "rule": rule, "line": data.count(b"\n", 0, match.start()) + 1, "match_sha256": hashlib.sha256(match[0]).hexdigest()})
    return findings


def reviewed_baseline(raw):
    entries = json.loads(raw)["reviewed_examples"]
    return {(e["path"], e["file_sha256"], e["rule"], e["match_sha256"]) for e in entries}


def apply_reviewed_examples(findings, data, label, baseline):
    file_hash = hashlib.sha256(data).hexdigest()
    retained = []
    for finding in findings:
        key = (label, file_hash, finding["rule"], finding.get("match_sha256"))
        # Provider tokens and private keys can never be baseline-exempted.
        if finding["rule"] in {"credential_assignment", "authenticated_url"} and key in baseline:
            REVIEWED_MATCHES.append({"path": label, "rule": finding["rule"]})
        else:
            retained.append(finding)
    return retained


def scan_payload(data, label):
    findings = scan_bytes(data, label)
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    # npz arrays are compressed ZIP entries and must also be inspected.
                    if member.file_size > 1024**3:
                        findings.append({"path": label, "rule": "archive_member_too_large_to_audit"})
                    else:
                        findings.extend(scan_bytes(archive.read(member), label + "::" + member.filename))
        except (zipfile.BadZipFile, RuntimeError):
            findings.append({"path": label, "rule": "unreadable_archive"})
    return findings


def check_name(path):
    return path.name in FORBIDDEN_NAMES or (path.name.startswith(".env") and path.name != ".env.example") or path.suffix in {".pem", ".p12", ".pfx", ".key"}


def scan_tree(root):
    findings, count, total = [], 0, 0
    baseline_path = root / "05_Validation_Metrics/reviewed_credential_examples.json"
    baseline = reviewed_baseline(baseline_path.read_text()) if baseline_path.is_file() else set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.is_symlink():
            findings.append({"path": str(relative), "rule": "symlink_requires_review"})
            continue
        if not path.is_file():
            continue
        count += 1
        if check_name(path):
            findings.append({"path": str(relative), "rule": "credential_container"})
        try:
            data = path.read_bytes()
            total += len(data)
            findings.extend(apply_reviewed_examples(scan_payload(data, str(relative)), data, str(relative), baseline))
        except OSError:
            findings.append({"path": str(relative), "rule": "unreadable_file"})
    return findings, count, total


def scan_staged(root):
    top = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root).decode().strip()).resolve()
    if top != root.resolve():
        raise ValueError("Initialize this folder as its own repository before scanning the index")
    names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=root).split(b"\0")
    findings, count, total = [], 0, 0
    recorded = subprocess.run(["git", "show", ":05_Validation_Metrics/reviewed_credential_examples.json"], cwd=root, capture_output=True)
    baseline = reviewed_baseline(recorded.stdout) if recorded.returncode == 0 else set()
    for raw in filter(None, names):
        name = raw.decode()
        count += 1
        mode = subprocess.check_output(["git", "ls-files", "--stage", "--", name], cwd=root).split(b" ", 1)[0]
        if mode == b"120000":
            findings.append({"path": name, "rule": "symlink_requires_review"})
        data = subprocess.check_output(["git", "show", ":" + name], cwd=root)
        if check_name(Path(name)):
            findings.append({"path": name, "rule": "credential_container"})
        total += len(data)
        findings.extend(apply_reviewed_examples(scan_payload(data, name), data, name, baseline))
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            oid = re.search(rb"oid sha256:([0-9a-f]{64})", data)
            if not oid:
                findings.append({"path": name, "rule": "invalid_lfs_pointer"})
                continue
            digest = oid[1].decode()
            gitdir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=root).decode().strip())
            obj = gitdir / "lfs/objects" / digest[:2] / digest[2:4] / digest
            if not obj.is_file():
                findings.append({"path": name, "rule": "lfs_object_unavailable"})
            else:
                payload = obj.read_bytes()
                if hashlib.sha256(payload).hexdigest() != digest:
                    findings.append({"path": name, "rule": "lfs_object_hash_mismatch"})
                total += len(payload)
                findings.extend(apply_reviewed_examples(scan_payload(payload, name), payload, name, baseline))
    return findings, count, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    findings, count, total = (scan_staged if args.staged else scan_tree)(args.root.resolve())
    report = {"status": "fail" if findings else "pass", "files_scanned": count, "bytes_scanned": total, "findings": findings, "reviewed_example_matches": len(REVIEWED_MATCHES), "reviewed_example_files": len({r["path"] for r in REVIEWED_MATCHES}), "scope": "staged blobs and LFS payloads" if args.staged else "filesystem excluding VCS, environments and caches", "limitation": "Pattern-based checks are not proof that every possible credential format is absent."}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
