# ADR-001: Separate Reproducible Evidence from Interpretation

## Status

Proposed.

## Context

The repository already has small Kedro pipelines for data processing, dataset construction, and
audit. The new research study needs constraint construction, model training, evaluation,
experiment comparison, notebooks, and reports.

Two failures are possible:

- notebooks can become hidden computation pipelines whose results are hard to reproduce;
- a custom experiment registry and runner can duplicate Kedro and W&B while making the code harder
  to inspect.

## Decision

Use Kedro for every computation that produces evidence for a scientific claim. Use versioned
catalog artifacts at the boundary between computation and interpretation.

Use `studies/<study_id>/` for research questions, experiment specifications, interpretation
notebooks, and reports. A notebook consumes persisted evidence. It does not define the primary
metric or train the primary model.

Use six reusable core pipelines:

```text
constraint_semantics -> benchmark -> encoder_features -> training -> evaluation -> comparison
```

Represent experiments as explicit configurations and study specifications. Use W&B for run records.
Do not add a second experiment registry or custom runner.

Keep sequence geometry and edit mining as optional pipelines behind data-validation gates.

## Alternatives Considered

### Put each experiment in a separate pipeline

This gives clear names but duplicates training and evaluation code. Objective comparisons can drift
silently. Reject this option. Use one parameterized pipeline and explicit experiment specifications.

### Run experiments and metrics in notebooks

This is quick for exploration but hides state, configuration, and failed cells. Reject it for
primary evidence. Keep notebooks for interpretation of cataloged results.

### Add a custom experiment registry and runner

This can provide one command interface, but Kedro already provides pipeline execution and W&B
provides run grouping. Reject it until a measured limitation appears.

## Consequences

Positive:

- each reported result has a catalog artifact and reproducible pipeline;
- objective comparisons share code and differ through visible configuration;
- notebooks stay short and readable;
- experimental intent remains beside reports for the same study.

Negative:

- exploratory notebook calculations must be promoted into tested code before they become primary
  evidence;
- study documents and Kedro configuration must link to each other explicitly;
- pipeline interfaces need stable artifact schemas.

## Revisit When

- Kedro configuration cannot express a required experiment matrix clearly;
- W&B and catalog identities cannot be linked reliably;
- several studies need the same report-generation workflow;
- geometry or edit work passes its gate and materially changes the pipeline boundaries.
