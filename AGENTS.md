# Working agreement

This is a research repository. Prefer scientific accuracy, inspectable code, and reproducible
evidence over framework completeness.

## Communication

- State the result first. Keep prose short and concrete.
- Separate observed, derived, assumed, hypothesized, and unknown claims.
- Report failed checks, missing data, and negative results.
- Do not claim significance, robustness, or reproducibility without evidence.

## Research

- Freeze the question, baseline, data version, split, metric, seeds, and compute limit before a
  confirmatory run.
- Never tune on test data. Record contamination and use a new holdout when required.
- Validate schemas, identifiers, row counts, uniqueness, missingness, and join cardinality.
- Never silently drop, coerce, pad, retry with changed parameters, or reuse partial artifacts.
- Persist exact Git state, configuration, input versions and hashes, hardware, precision, runtime,
  cost, seeds, outputs, and failures.
- Keep W&B runs for training and evaluation. Keep large artifacts in S3 or Hugging Face, not Git.

## Code

- Simple and direct beats generic. Do not add an abstraction before two real uses need it.
- Put reusable domain logic in `src/vec2vec/lib/`; keep Kedro nodes small.
- Raise specific errors with useful context. Never swallow an exception.
- Comments explain scientific intent or invariants, not syntax.
- Match existing style. Ruff uses 100-character lines and double quotes.
- Do not commit data, checkpoints, logs, or notebook outputs.

## Workflow

- Read relevant code and configuration before editing.
- Use `uv` and the existing virtual environment.
- Do not commit unless asked.
- Ask before destructive Git history changes, paid compute, publishing, or deleting remote data.
- Run `pytest`, `ruff check .`, and `ruff format --check .` after non-trivial changes.

The compact result tables and append-only experiment log are the active scientific record. Older
experiment implementations and full reports remain in Git history and immutable S3 artifacts.
