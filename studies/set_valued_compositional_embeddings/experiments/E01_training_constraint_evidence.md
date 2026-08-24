# E01 — Rule-Derived Training Constraint Evidence

## Status

Complete. The offline evidence, hand check, smoke, and fixed accuracy benchmark completed. The
specification below was fixed before the first complete pipeline run.

## Question

Can exact metadata rules produce a useful volume of training constraints while a compact,
independent validation sample measures their accuracy?

## Hypothesis

Exact mappings for controlled Addgene metadata will provide broad training coverage with fewer
semantic errors than literal description matching. Values outside the enabled mappings will remain
unlabeled.

## Fixed inputs

- Retrieval dataset version: `2026-08-04T09.02.10.007Z`.
- Training labels use only the grouped training split.
- Benchmark candidates use only the grouped validation split.
- The catalog does not load test metadata for this pipeline.
- pLannotate is the only annotation source included in benchmark evidence. Missing pLannotate
  evidence stays missing. There is no plasmidkit fallback.

## Enabled mappings

- Copy class: configured `included` mappings.
- Growth temperature: configured `included` mappings for 30 and 37 degrees Celsius. The unresolved
  stored value `23` is not enabled.
- Bacterial selection: configured `included` mappings and the five reviewed exact mappings from
  rule version 0.2.
- Intended use: exact controlled expression and use-category mappings. Free text is not mapped.

Each training claim preserves its raw source value, rule, relation, canonical value, split, leakage
component, and content-derived identifier. These are noisy training labels, not benchmark truth.

## Benchmark protocol

Select 240 mapping applications from the validation split. Keep at most one application per facet
and leakage component. Select at least 30 applications per available facet, then fill the remaining
sample by deterministic hash order. Include all pLannotate features for each selected sequence.

The planned judge is a strong model with a fixed structured response. It will assess the complete
mapping application, not every generated training claim. Review only uncertain responses and
systematic disagreements. Do not run the paid judge until the sample and estimated cost are
inspected.

## Metrics

The primary metric is claim precision on the fixed validation sample, with a binomial confidence
interval. Report overall and per-facet precision. Secondary metrics are training-claim count,
mapping coverage by source field, unlabeled-value counts, pLannotate coverage, invalid-response
rate, and judge uncertainty rate.

## Interpretation rule

Use the measured benchmark accuracy to describe training-label noise. Do not require every claim to
be reviewed. If errors cluster by an exact mapping, disable or revise that mapping as a new rule
version. Do not tune the protocol on the test split.

## Stopping rule and compute

The offline evidence and sampling pipeline has no paid calls. Estimate the paid validation cost from
the inspected packet size and prior model runs. Obtain approval before a material paid run. Stop a
paid run at its configured cost cap and preserve all failures.

## Known limitations

- Addgene metadata can be incomplete or internally inconsistent.
- pLannotate feature absence is not evidence that a metadata claim is false.
- The validation benchmark estimates mapping accuracy, not downstream retrieval quality.
- Equal minimum sampling by facet changes the sample distribution. Overall estimates must account
  for the sampling design or be described as sample accuracy rather than population accuracy.

## Offline run 1

- Git state: dirty worktree based on `fa02968`; the unchanged production implementation and
  configuration from the run were committed as `7e66cf4`.
- Output version: `2026-08-06T08.44.42.865Z`.
- Runtime: 80.6 seconds.
- Training claims: 375,819 across 92,097 training sequences and 11,456 leakage components.
- Validation sample: 240 mapping applications across 237 sequences and 178 leakage components.
- Sample allocation: 47 copy-class, 54 expression-context, 61 growth-temperature, 30 use-category,
  and 48 bacterial-selection applications.
- Test rows loaded: zero.
- Generated descriptions used: no.
- Plasmidkit fallback: no.
- Paid model calls: zero.

The representative sample contains no `reviewed_mappings` rows because those mappings are rare. The
earlier fixed 16-packet targeted gate covers all five reviewed bacterial mappings. Keep that targeted
result separate from the population-oriented accuracy sample.

## Accuracy benchmark

- Packet output version: `2026-08-06T08.58.54.091Z`.
- Prompt: `constraint-benchmark-judge-v1`.
- Model and provider: `openai/gpt-5.6-sol`, provider `OpenAI`.
- Five-facet smoke output: `2026-08-06T08.59.59.220Z`; 5/5 passed; cost USD 0.117088.
- Complete decision output: `2026-08-06T09.01.02.445Z`.
- Runtime: 1,073.7 seconds.
- Valid responses: 240/240.
- Semantic support: 240 `supported`.
- Benchmark scope: 240 `in_scope`.
- Model-reference pass fraction: 1.0; 95% Wilson interval 0.984246 to 1.0.
- Manual-review rows: zero under the fixed decision rule.
- Complete-run cost: USD 4.670470 under the USD 7.50 cap.
- Accepted benchmark labels created by the judge pipeline: zero.

The hand check and model benchmark support using this rule contract for noisy training supervision.
They do not convert the metadata into biological ground truth.
