# Credentials and release hygiene

Only the new Backtranslation export is audited. The original BRASS checkout, its Git
objects, model caches, local login state, and execution environment remain outside scope.

The migration excluded real environment files, Git history, model weights, account/session
files, virtual environments, runtime logs, compiled files, notebooks with saved output,
and opaque source archives. It compared included bytes with credential values read locally
from the source `.env`, without printing or exporting those values. `.env.example` was
written from scratch with empty credential fields.

`05_Validation_Metrics/credential_scan.py` checks OpenAI/Anthropic, Hugging Face, GitHub,
AWS, Google, Slack, Stripe, private-key, JWT, authorization, authenticated-URL, and generic
credential-assignment patterns. It also checks decompressed ZIP/NPZ members and rejects
credential containers, unexpected symlinks, unreadable payloads, and unavailable staged
LFS objects. Reports contain paths, rules, and line numbers only; matched text is omitted.

The whole-tree check scans files regardless of `.gitignore`, except VCS internals,
environments, and caches. The pre-commit mode scans actual index blobs and their LFS
objects; it does not trust the working copy as a replacement for staged content.

Pattern checks cannot prove the absence of every possible secret format. The saved audit
records the exact scope and results, and every publication should rerun the checks.
If a genuine key is discovered, revoke it with the provider and remove it from the export
and any published history. Do not paste it into a report, issue, or pull request.

## Preserving scientific evidence

Generated responses may contain synthetic example passwords or credential-shaped code.
Such matches require classification; research directories are never blanket-exempted.
`05_Validation_Metrics/reviewed_credential_examples.json` records reviewed synthetic
examples by exact file hash, rule, and match hash, without storing the matched values.
Any change to the file invalidates its exceptions. Provider tokens and private keys
cannot be exempted through this baseline. The scanner reports the reviewed-match count
separately from unresolved findings.
Any redaction must be recorded in the migration inventory with changed export hashes.
Original recorded source hashes describe the historical experiment; they must not be
silently changed to match refactored code.

Archived absolute paths are provenance. Active helpers resolve known BRASS paths to the
new checkout without rewriting the original artifact or reading the old checkout.
