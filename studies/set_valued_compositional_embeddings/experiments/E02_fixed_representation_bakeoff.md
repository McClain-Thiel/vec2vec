# E02 Fixed-Representation Bake-off

- **Status:** planned; protocol frozen before feature extraction or model evaluation
- **Gate:** 1
- **Protocol version:** `fixed_representation_bakeoff_v0.1`
- **Frozen:** 2026-08-18 Europe/London

## Question

Which frozen DNA encoder and frozen text encoder provide the best validation-set representation
for recorded-constraint retrieval under a fixed alignment probe and a bounded compute budget?

## Hypothesis

At least one prokaryote-specific or larger Carbon-family DNA encoder will improve validation
`utility@10` by at least 0.01 over the pinned Carbon-500M incumbent. A modern text encoder can
improve top-rank retrieval, but the DNA representation will remain the larger source of variance.

This is an exploratory selection hypothesis. The test split cannot confirm it because Gate 1 does
not read test outcomes.

## Baselines and controlled comparisons

Use these DNA representations:

1. a train-fitted 6-mer TF-IDF plus 512-dimensional truncated-SVD baseline;
2. `HuggingFaceBio/Carbon-500M@106e36ff51b5dfbfe0b078ad18ad37a6956c5714`;
3. `HuggingFaceBio/Carbon-3B@95c3c68fc77fdf70b1582031bacf9d7753f72cf2`;
4. `GenerTeam/GENERanno-prokaryote-0.5b-base@d02db0f24f2c62fa1efde760217cdf75771b0228`;
5. `GenerTeam/GENERator-v2-prokaryote-1.2b-base@8b2f768b0d293953518ff91d34600f9322ef1f94`.

Use these text representations:

1. the BGE-base incumbent above;
2. `Alibaba-NLP/gte-modernbert-base@e7f32e3c00f91d699e8c43b53106206bcc72bb22`;
3. `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

Evaluate the full five-by-three DNA and text factorial. Extract each frozen representation once,
then fit all 15 alignment configurations. This controls tower interaction. Do not select the DNA
tower under one text model and assume that its ranking transfers to another text model.

Do not add a candidate after inspecting validation results. A new candidate requires a new
experiment version. Evo 2 7B is not part of version 0.1.

## Fixed data

- Retrieval dataset: `2026-08-04T09.02.10.007Z`.
- Population SHA-256:
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Constraint-state artifact: `2026-08-06T13.27.47.937Z`.
- Global graph artifact: `2026-08-17T22.59.04.326Z`.
- `split_grouped_v2`: `2026-08-17T23.49.47.355Z`.
- Frozen query benchmark v0.1: `2026-08-17T23.51.35.629Z`.

Use `split_grouped_v2` only. Do not read `split_grouped` for sampling, fitting, checkpoint
selection, or evaluation.

## Sampling and exclusions

Create one content-hashed sample manifest before encoder extraction.

- **Alignment training panel:** 20,000 rows from the v2 training split. Sample components with
  inverse-size weights. Add one hash-ordered row per component per pass until the panel is full.
  No component can supply more than 100 rows.
- **Validation panel:** all 11,344 v2 validation rows.
- **Invariance panel:** 512 v2 training rows, stratified across sequence-length deciles, with one
  row per primary component.
- **Numerical smoke panel:** 32 rows from the invariance panel, including the shortest and longest
  eligible sequence in each represented length stratum.

Do not use test rows. Do not select rows by a model score. Preserve every sampled identifier,
component identifier, sequence length, and description hash in the manifest.

Exact duplicate descriptions and exact duplicate sequences can create multiple valid paired
targets. The alignment loss must use explicit many-to-many positive masks. It cannot force a known
duplicate to act as a negative.

## Sequence coverage and pooling contract

No encoder can silently drop, truncate, or replace a base.

For each pinned tokenizer, measure the largest input window that fits after all required model
tags and special tokens. The window length must respect the tokenizer unit. Use deterministic
circular windows with 25% overlap when one window cannot cover a plasmid. A window that crosses
the recorded origin wraps to the start of the same sequence. Add a final wrapped window when the
regular step leaves any base uncovered.

Record these values for every plasmid:

```text
sequence_id
encoder_id
sequence_length_bp
window_length_bp
window_count
covered_base_count
coverage_fraction
content_token_count
pooling_rule_id
```

`coverage_fraction` must equal 1.0. Fail on an out-of-vocabulary base, an empty content-token set,
an invalid mask, or a non-finite value.

Use mean pooling over non-padding, non-special DNA content tokens. Use the mean of the last four
hidden layers for the Carbon family to preserve the incumbent PlasmidCLIP extraction contract.
Use the last hidden layer for GENERanno and GENERator, as documented by their embedding examples.
Pool multiple windows with the number of newly covered bases as weights. L2-normalize the final
plasmid vector.

This layer rule is a known model-family confound. Do not tune it on validation in this experiment.

## Numerical and invariance checks

Use bfloat16 for the main neural extraction. Use float32 on the numerical smoke panel as the
reference. IEEE float16 is prohibited because PlasmidCLIP observed invalid Carbon outputs.

Stop one candidate before alignment fitting if:

- any pooled vector is non-finite;
- bfloat16 versus float32 cosine similarity is below 0.99 for any smoke row;
- any row has coverage below 1.0;
- effective rank is below 1% of the available vector dimension on the invariance panel; or
- median circular-rotation or reverse-complement cosine similarity is below 0.90.

Report the complete similarity distributions, effective rank, mean pairwise cosine, length
correlation, GC-content correlation, window count, throughput, peak memory, and output bytes. A
passing diagnostic does not select the model.

## Fixed alignment probe

Freeze both encoders. Fit train-only centering and whitening with zero removed principal
components. Map both towers to 512 dimensions with one linear projection per tower. L2-normalize
the projected vectors.

Train the projections with the same symmetric many-to-many contrastive loss for every candidate.
Use three seeds: 13, 42, and 20260818. Use AdamW with learning rate `1e-3`, weight decay `0.01`,
and the PyTorch default beta values. Use an effective batch size of 4,096 for 60 epochs with no
learning-rate schedule. Initialize the learned logit scale to `log(1 / 0.07)` and cap its
exponential at 100. Use the last epoch. Do not use early stopping or candidate-specific
hyperparameters.

The probe trains on paired sequence-description rows only. It does not use query-benchmark labels.
This keeps the Gate 2 set-supervision comparison separate from encoder selection.

## Metrics

The primary selection metric is validation closed-gallery, query-macro:

```text
utility@10 = verified@10 - contradicted@10
```

Higher is better. Calculate it on measurement-usable atomic and pair queries from query benchmark
v0.1. Report atomic, pair, and combined results separately. Also report verified, contradicted,
unknown, and known fractions at K = 1, 5, 10, and 50.

Secondary metrics are:

- paired sequence-to-description and description-to-sequence R@1, R@10, and median rank;
- component-macro query metrics;
- performance by verified-set size, contradiction support, sequence-length stratum, and primary
  component size;
- effective rank, mean pairwise cosine, length and GC correlations;
- circular-rotation and reverse-complement cosine distributions;
- GPU-hours, peak device memory, plasmids per second, and persisted bytes.

Use whole-component bootstrap resampling for 95% confidence intervals. Keep all three seeds. Do
not report only the best seed.

## Selection rule

Remove a candidate only when it fails a fixed coverage, numerical, collapse, or invariance check.
Among passing DNA-by-text pairs, select the highest mean validation `utility@10` across the three
seeds.

Treat two candidates as a practical tie when their mean difference is less than 0.01 and their
whole-component bootstrap intervals overlap. Select the candidate with lower measured extraction
GPU-hours. If cost also differs by less than 10%, select the smaller persisted feature product.

If no pair improves the Carbon-500M plus BGE-base incumbent by at least 0.01, retain the incumbent
pair. Freeze the selected revisions, pooling rules, window rules, and feature hashes before any
Gate 2 run. Report DNA main effects, text main effects, and pair interactions as descriptive
results. Do not give them a causal interpretation.

Do not inspect test metrics during selection. A test read before the configuration freeze
contaminates the confirmatory evaluation and requires a new test protocol.

## Compute and stopping budget

Version 0.1 permits at most 40 A100-equivalent GPU-hours for smoke checks, feature extraction, and
alignment probes. It permits at most 250 GB of persisted artifacts. Paid GPU work requires
explicit user approval before launch.

Stop the experiment when one of these events occurs:

- all planned candidates and three seeds complete;
- the 40-GPU-hour budget is consumed;
- persisted output reaches 250 GB;
- a fixed validation or scientific invariant fails; or
- two exact technical retries fail for the same candidate and configuration.

A technical retry must use the identical model revision, code commit, sample manifest,
configuration, and seed. Do not change a parameter to make a failed run complete. Record every
failed and stopped run.

Full-population feature extraction for the selected pair is a separate post-selection action. It
requires a new resolved cost estimate and approval. It cannot overwrite screening features.

## Required outputs

- a resolved protocol and candidate manifest;
- the frozen sample manifest;
- per-row sequence-coverage reports;
- pooled frozen features with input and model hashes;
- numerical, invariance, collapse, confound, throughput, and storage reports;
- per-seed projection checkpoints and training diagnostics;
- tidy validation metric components and bootstrap draws;
- a selection report that records the rule outcome and remaining uncertainty.

Store paths and serialization rules in the Kedro Data Catalog. Record the Git commit, dirty state,
resolved configuration, package versions, model and tokenizer revisions, hardware, precision,
seeds, and feature hashes.

## Known limitations

- Gate 1 selects on one validation split, so its validation estimate is optimistic after model
  selection.
- The paired alignment probe can favor representations that match descriptions, even when another
  representation could work better under set supervision.
- Model pretraining contamination is unknown. Public model cards do not provide an Addgene-level
  membership audit.
- Model-family layer extraction differs by documented architecture. The comparison therefore
  evaluates complete frozen representation recipes, not model weights alone.
- Recorded metadata is incomplete and is not biological ground truth.
- The largest training component contains 40.62% of training rows. The component-aware sample and
  component bootstrap reduce this imbalance but do not remove collection bias.
