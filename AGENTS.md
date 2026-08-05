# Working Agreement for Research Agents

## Purpose

This repository is a research project. The primary goal is to understand the system and
produce trustworthy evidence. A working program is useful only when we can explain what it
does and why its results are credible.

Use this priority order when goals conflict:

1. Scientific accuracy and data integrity.
2. Clear reasoning and visible uncertainty.
3. Reproducibility and provenance.
4. Simple code that a researcher can inspect.
5. Delivery speed.
6. Performance and abstraction.

Never hide an error to make a run complete. Never describe an assumption, estimate, or
plausible interpretation as a measured fact.

## Communication

Use ASD-STE100 Simplified Technical English as the default writing style. Extend it when a
technical term is necessary. Define uncommon terms on first use. Prefer short sentences,
concrete nouns, and active voice. Use one term for one concept. Do not create new terminology
when a standard term exists.

State the result before implementation details. Separate these categories when they matter:

- **Observed:** Directly measured or read from a source.
- **Derived:** Computed from observed values. State the method.
- **Assumed:** Treated as true without direct evidence.
- **Hypothesized:** A claim that the experiment will test.
- **Unknown:** Not established by the available evidence.

Report failed checks, missing data, conflicting evidence, and negative results. Do not omit
them because they complicate the conclusion. Use calibrated language. Do not use words such
as "proves," "significant," "robust," or "state of the art" unless the evidence supports the
specific meaning.

## Research Method

Before a new experiment, record:

- the question or hypothesis;
- the comparison or baseline;
- the input dataset and exact version;
- the split and exclusion rules;
- the primary metric and expected direction of change;
- important secondary metrics and failure checks;
- the stopping rule or compute budget;
- known limitations and possible confounders.

Keep exploratory analysis distinct from confirmatory evaluation. Do not tune a method on the
test set. Do not use test results to choose checkpoints, prompts, thresholds, or hyperparameters.
If exploration touches a held-out set, record this fact and treat that set as contaminated.

Use a meaningful baseline before a complex method. Change one important factor at a time when
the goal is causal interpretation. When several factors change, state that the cause of the
result is not isolated.

Report uncertainty when it can affect a conclusion. Use confidence intervals, variation across
seeds, or another suitable measure. Do not compare only the best run. Record all planned runs,
including failed and stopped runs. Distinguish technical failures from valid poor results.

## Accuracy and Failure Policy

Fail early and with a specific error when an input violates an invariant. Do not silently:

- drop rows, samples, labels, or features;
- coerce invalid values;
- replace missing values;
- truncate or pad data;
- change units, coordinate systems, encodings, or data types;
- fall back to a different model, dataset, device, or algorithm;
- retry with changed parameters;
- continue from a partial artifact;
- weaken a validation rule.

If one of these actions is scientifically justified, make it explicit in configuration, log the
count and reason, and test it. Preserve raw source values when practical.

Validate data at boundaries. Check schema, units, allowed values, identifiers, row counts,
uniqueness, missingness, and join cardinality as applicable. For joins, report matched,
unmatched, duplicated, and expanded rows. Treat unexpected many-to-many joins as errors.

Use exact comparisons for identifiers and discrete values. For floating-point results, define
and justify a tolerance. Do not choose a tolerance only to make a test pass.

Inspect source code, local data, logs, or primary documentation before making a factual claim
about them. If evidence is unavailable, say so. Do not invent citations, results, file contents,
API behavior, or run status.

## Code Design

Write research code for inspection and change. Prefer a direct implementation over a generic
framework. Some repetition is better than a premature abstraction. Optimize only after a
measurement shows that performance blocks the research.

Keep reusable domain logic in `src/vec2vec/lib/`. It should be pure when practical, independent
of Kedro, and directly unit tested. Keep custom data access in `src/vec2vec/datasets/`. Keep
Kedro nodes small: nodes coordinate reusable functions and should not contain hidden global
state.

Create a separate pipeline or module for each distinct experiment. Create a separate analysis
file for each distinct research question. Do not add experiment-specific branches to shared
logic when a parameter, small wrapper, or separate node makes the comparison clearer. Give
experiments stable, descriptive names. Do not use names such as `final`, `new`, or `test2`.

Notebooks are for exploration and presentation. Do not make a production pipeline depend on
notebook state. Move reusable logic from a notebook into `src/vec2vec/lib/`, test it, and call it
from the notebook. Keep notebooks restartable from a clean kernel. Do not commit large notebook
outputs.

Add comments for scientific intent, invariants, source conventions, and non-obvious choices.
Do not comment on syntax that the code already states. Use docstrings for public functions and
for functions whose assumptions or units are not obvious.

## Code Style Guide

Use Ruff as the source of truth for formatting, import order, and lint rules. Use a line length of
100 characters and double quotes. Do not reformat unrelated code during a focused change.

Prefer code that exposes the scientific operation directly:

- Give functions and variables names that state their scientific role.
- Keep one main responsibility in each function. Split a function when it mixes distinct
  scientific rules, validation phases, or side effects. Do not split code only to meet a line
  count.
- Prefer explicit inputs and return values. Avoid mutable global state and hidden defaults.
- Keep reusable calculations pure when practical. Keep network, file, catalog, and tracking
  operations at clear boundaries.
- Do not add an abstraction until at least two real uses need the same behavior and the shared
  concept has a stable meaning.

Use types to make boundaries inspectable:

- Add type annotations to public functions and to functions that cross module boundaries.
- Use Pydantic models for records that cross a process, API, configuration, or serialization
  boundary. Validate a deserialized record before use.
- Use a dataclass or a small built-in type for simple internal records.
- Limit `Any` to external or genuinely dynamic boundaries. Convert it to a specific type as soon
  as practical.
- State units, coordinate conventions, and allowed values in names, models, or docstrings.

Handle tables as scientific data, not as untyped containers:

- Validate required columns, row identifiers, uniqueness, and expected missingness before a
  transformation.
- Use explicit join validation. Record unmatched, duplicated, and expanded row counts.
- Copy a DataFrame before mutation when ownership is not local and clear.
- Use a stable sort when row order contributes to an identifier, sample, hash, or report.
- Preserve source columns beside normalized or derived columns when practical.

Make failures useful. Raise a specific error that names the violated rule and includes useful
counts or identifiers. Catch a broad exception only at an explicit system boundary where the
failure is recorded as data. Preserve raw provider responses when parsing or validation fails.

Keep configuration and documentation close to scientific intent. Put scientific choices in
Kedro parameters. Use constants only for fixed contracts, such as schema or prompt versions.
Comments and docstrings must explain intent, evidence, invariants, units, or provenance. Use the
communication rules in this file for user-facing text and documentation.

Tests must name one behavior and include its important failure case. Keep unit tests offline and
deterministic. Mock network and paid services. Use realistic small fixtures when the edge case is
part of the scientific claim.

Delete generated caches, logs, and code paths that no longer serve an active contract. Do not
delete failed-run records, superseded experiment plans, or contrary results. These files are
research provenance. If a public or persisted contract changes, add an explicit migration or a
new version instead of silently reusing the old name.

## Kedro Conventions

Use Kedro for pipeline composition, configuration, dataset access, and run organization.

- Put reusable configuration in `conf/base/`.
- Put local secrets and machine-specific overrides in `conf/local/`. Never commit secrets.
- Put test configuration in `conf/test/` and keep tests offline.
- Put data locations and serialization rules in the Data Catalog. Do not embed paths in nodes.
- Pass scientific choices as explicit parameters. Do not hide them in module constants.
- Use modular pipelines with clear inputs and outputs.
- Use dataset names that describe content, not storage format.
- Use Kedro data layers consistently. Raw data is immutable.
- Version important model, feature, prediction, and reporting outputs.
- Run a small end-to-end fixture before a full or costly run.

Preserve the current catalog contracts. In particular, Kedro transcoded dataset names in this
project carry dependency meaning. Do not rename or split them without checking the graph. Kedro
configuration replaces some mappings rather than deep-merging them. Restate a complete parameter
block in an environment override and access required parameters without silent defaults.

Do not place paid, destructive, remote-publishing, or large-compute pipelines in `__default__`.
Require an explicit command and a configured limit for each such run.

## Research Layers

Extend the Kedro workflow with research layers. Keep the boundary between machine-produced
evidence and human interpretation clear:

```text
data preparation -> feature construction -> model training -> evaluation
                 -> experiment comparison -> interpretation -> report
```

- **Model training** fits a model and writes a versioned model artifact and training diagnostics.
- **Evaluation** applies a frozen protocol to a frozen model and dataset. It writes predictions,
  metrics, uncertainty estimates, and validation results. Evaluation is Kedro pipeline code.
- **Experiment comparison** combines evaluation outputs across variants or seeds. It writes tidy
  comparison tables and plot-ready data. It is also Kedro pipeline code when it supports a claim.
- **Interpretation** explains what the evidence means, checks examples, and develops new
  hypotheses. It belongs in a study notebook. It must not silently redefine primary metrics.
- **Report** is the short, durable account of the question, method, evidence, conclusion, and
  limitations. It links to immutable run and artifact identifiers.

Use Kedro data layers for artifacts, not for scientific meaning. In the standard data layout,
trained models belong in `06_models`, predictions and other model outputs belong in
`07_model_output`, and metrics, comparison tables, and plot data belong in `08_reporting`.
Notebooks and prose reports do not belong in `data/`.

## Annotation Source Policy

Use pLannotate as the primary source for feature identities, coordinates, coverage measurements,
sequence masking, and putative edit intervals in this research study. Select pLannotate rows
explicitly by their `source` value. Do not read the combined `addgene_annotation_features` product
for a measurement that claims to be pLannotate-based.

Do not fall back to plasmidkit when pLannotate has no annotation. Record the result as missing
pLannotate evidence. The plasmidkit source can be used only in a separately named concordance,
sensitivity, or error analysis. Keep its results separate from the primary estimate.

The existing retrieval artifact and generated descriptions can contain features collected from
both annotation sources. Preserve this provenance. Do not describe an existing mixed-source
artifact as pLannotate-only. A new pLannotate-only product needs a distinct catalog name and
version.

Pin and record the pLannotate software version, database versions, run configuration, circular
sequence setting, and coordinate convention. Validate coordinate normalization before calculating
interval overlap, coverage, masking, or edits. Preserve the original coordinates beside every
derived interval.

## Studies, Experiments, Notebooks, and Reports

Use **study** for one research question or coherent set of questions. This avoids confusion with
the Kedro project, which is this complete repository. Keep human-facing research work grouped by
study:

```text
studies/
└── <study_id>/
    ├── README.md
    ├── experiments/
    │   └── <experiment_id>.md
    ├── notebooks/
    │   ├── 00_overview.ipynb
    │   └── <ordered_analysis>.ipynb
    └── reports/
        ├── report.md
        ├── figures/
        └── tables/
```

Use a stable study identifier such as `sequence_text_baselines`. Use an experiment identifier
that states the comparison, such as `encoder_pooling_mean_vs_cls`. Add a date only when it helps
distinguish planned rounds. Do not put the result in the identifier.

The study `README.md` is the entry point. A researcher should understand the study in five
minutes. It must contain:

- the research question and why it matters;
- the current answer in plain language;
- the status: planned, active, complete, superseded, or inconclusive;
- the dataset, split, model, and evaluation protocol;
- a table of experiments with run links and outcomes;
- the main limitations and open questions;
- links to the best notebook and current report.

Each experiment file is a preregistration-sized specification, not a run log. Record the
hypothesis, controlled change, baseline, fixed factors, metrics, seeds, stopping rule, and
acceptance or interpretation rule before the main run. Link all resulting W&B runs and Kedro
artifact versions after execution. If the protocol changes, add a dated amendment. Do not rewrite
the original plan.

Notebooks must consume versioned catalog outputs from evaluation or experiment comparison. Do not
copy data-loading, metric, or model code into them. Use the Kedro IPython integration to load the
catalog when useful. A notebook may calculate exploratory summaries, but any number used as
primary evidence in a report must come from tested library or pipeline code.

Make each interpretation notebook quick to review:

1. Title, study identifier, purpose, and artifact or run identifiers.
2. **Conclusion first:** three to seven short bullets that state what the notebook shows.
3. Method and scope, including exclusions and known contamination.
4. Quality checks and data coverage before result plots.
5. Results, with readable labels, units, uncertainty, and informative captions.
6. Examples and failure cases when they help explain aggregate metrics.
7. Limitations, unresolved questions, and the next proposed test.

Keep one notebook focused on one analysis question. Prefer several short ordered notebooks over
one long chronological notebook. Remove dead cells and incidental debugging before review. Restart
the kernel and run all cells before marking a notebook complete. A notebook is not evidence of a
successful pipeline run unless it cites the exact persisted outputs that it reads.

Reports are curated conclusions, not dumps of notebook output. Keep report figures and tables in
the same study directory. Every figure and table must have a caption that states the population,
metric, direction, units, uncertainty where applicable, and relevant sample count. Generated
figures and tables must be reproducible from named catalog artifacts. Clearly mark a manually
edited figure or table.

## Reproducibility and Provenance

Every reported result must be traceable to:

- the Git commit and whether the worktree was dirty;
- the full resolved configuration;
- input dataset identifiers, versions, revisions, or content hashes;
- code and prompt versions;
- package and runtime versions;
- random seeds and determinism settings;
- hardware and precision when they can change the result;
- produced artifacts, metrics, logs, and validation reports.

Set seeds for Python, NumPy, and each machine-learning framework in use. Record known sources of
nondeterminism. Do not claim exact reproducibility when the runtime cannot provide it.

Do not overwrite a result that supports a conclusion. Produce an immutable or versioned artifact.
Derived artifacts must identify their inputs. A cached artifact is valid only when its input and
configuration identity match the current run.

## Weights & Biases

Use Weights & Biases (W&B) for experiment runs when tracking adds research value. Log the full
resolved scientific configuration, stable run name, group, tags, seed, commit, dataset revision,
model revision, summary metrics, and useful diagnostic metrics. Log artifacts that are needed to
interpret or reproduce a run.

Do not put secrets, personal data, restricted data, or large raw datasets in W&B. Do not make core
computation depend on a live W&B service. A run must still produce local Kedro outputs if tracking
is disabled or temporarily unavailable. Record tracking failure visibly.

Use one W&B run for one experimental unit. Use groups for repeated seeds or a planned comparison.
Do not edit past metrics to improve a result. If a run is invalid, mark it invalid and record why.

## Hugging Face Hub

Use the Hugging Face Hub for versioned datasets, models, tokenizers, and other research artifacts
that need shared access. Pin an exact repository revision when consuming an artifact. Do not rely
on a moving branch name for a reported result.

Each published dataset needs a dataset card. Each published model needs a model card. Cards must
state provenance, license, intended use, excluded use, known limitations, data splits, evaluation
method, metrics, and the code or W&B run that produced the artifact. Check privacy, consent,
license, and access restrictions before upload.

Do not publish or overwrite a remote artifact without explicit user approval. Prefer a new commit,
revision, or pull request to a destructive update.

## Testing and Review

Test scientific invariants, not only code execution. Include tests for edge cases, leakage,
schema drift, ordering assumptions, missing values, duplicate identifiers, and expected join
cardinality when relevant.

Use small fixed fixtures for fast unit and pipeline tests. A fixture must retain the feature that
the test studies. Do not replace a realistic edge case with a convenient synthetic case without
recording the limitation.

Before presenting work as complete, run the relevant checks. The normal local checks are:

```bash
pytest
ruff check .
ruff format --check .
```

Also run the affected Kedro pipeline with the test environment when pipeline composition changes.
Report exactly what ran, what did not run, and why. A passing test suite is evidence of tested
properties; it is not evidence that every scientific claim is correct.

Review research changes in this order:

1. Is the question well defined?
2. Are the data and labels valid for the question?
3. Is leakage prevented?
4. Are assumptions and transformations explicit?
5. Do metrics measure the stated claim?
6. Can another researcher reproduce the result?
7. Is the code as simple as the problem permits?

## Agent Behavior

Before editing, inspect the relevant code, configuration, tests, and data contracts. For a large
change, state the intended scientific behavior and validation plan first.

Ask for direction when a choice changes the research question, data inclusion rule, evaluation
protocol, privacy posture, publication scope, or material compute cost. Make a reasonable explicit
assumption for low-risk implementation details.

Do not change an experiment definition after seeing its result without recording the change as a
new experiment. Do not delete or rewrite contrary evidence. Correct errors openly and state which
prior results they affect.

When finishing a task, summarize:

- what changed;
- what evidence supports it;
- which checks ran and their results;
- remaining uncertainty, limitations, and untested paths;
- any action that still needs user approval.
