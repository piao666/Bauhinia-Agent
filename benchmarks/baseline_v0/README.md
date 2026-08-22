# Bauhinia-Agent Evo offline baseline v0

This directory is the tracked, public synthetic P0 corpus. It is a fixture-health
baseline, not a model evaluation, held-out corpus, or promotion result. It never
requires a network connection, credentials, or an external model.

Validate the immutable corpus and run all twelve initial scenarios:

```powershell
.\.venv\Scripts\python.exe benchmarks\baseline_v0\validate_manifest.py --run-baseline
```

The command first verifies `corpus.lock.json`, then checks the manifest and runs
the deterministic verifier from a second, integrity-checked private snapshot.
Python starts with `-I -S -B`, so user/site packages and bytecode caches cannot
silently become verifier inputs. Eight scenarios intentionally pass and four
intentionally fail; the command succeeds only when every observed result matches
the manifest.

Each run writes a new JSON report under `benchmarks/baseline_v0/runs/`. The
directory is ignored by Git and files are created exclusively, so a later run
cannot overwrite an earlier report. Use `--no-report` for a read-only health run,
or `--report-dir <path>` to store reports elsewhere.

The older ignored `evaluation/baseline_v0/` directory may remain in an existing
workspace as local historical evidence. New clones and new runs use this tracked
directory; the historical data is not modified or imported automatically.

## Integrity and scope

- `corpus.lock.json` records normalized-LF SHA-256 digests for the manifest,
  validators, and every workspace fixture.
- Corpus changes require a new corpus version and a deliberately regenerated
  lock; do not edit an existing published version in place.
- Scenario verification supports fixed observed values and named Python calls.
  It does not evaluate arbitrary expressions.
- Symlinks, Windows reparse points, bytecode, unlocked workspace files, and
  unexpected root entries fail integrity validation before execution.
- Runtime reports contain corpus/version/hash, repository commit, a safe
  environment summary, verifier version, separate metrics, and per-case results.
- Reports deliberately state that P0-005 is not satisfied: this fixture runner
  does not execute the Coding Agent or measure model tokens, interventions,
  dangerous actions, or run-to-run Agent reproducibility.
- The expected values are public synthetic fixture data. They are not P8 private
  held-out answers and cannot be used as promotion evidence.
