# E02 Fixed-Representation Bake-off

- **Status:** active; E02b harness implemented before feature extraction or model evaluation
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

## 2026-08-18 numerical smoke execution amendment before results

Run the 32-row numerical smoke panel for the four neural DNA candidates before any invariance or
retrieval run. This execution does not run the 6-mer baseline or the text encoders because those
representations do not have the DNA coverage and bfloat16 risks that this check measures.

Build the 512-row invariance panel from `split_grouped_v2` training rows. Use ten length deciles,
one row per primary similarity component, and deterministic SHA-256 ordering. Reserve the shortest
and longest eligible row in each decile before filling the remaining component-balanced quota.
Build the 32-row numerical panel from that frozen panel. Use four rows in each of the first two
deciles and three rows in each remaining decile. Include the shortest and longest panel row in
each decile.

For a 6-mer tokenizer, wrap the final one to five input bases across the recorded circular origin
to create a complete 6-mer. Record the number of wrapped input bases. This rule adds no unknown
base, does not drop a source base, and keeps every model input aligned to the tokenizer unit.

For each candidate, extract the same 32 original sequences once in bfloat16 and once in float32.
Do not run rotation, reverse-complement, collapse, or retrieval checks in this smoke execution.
Those checks still use the full 512-row invariance panel in the later run. Pass this smoke check
only if every vector is finite, every source base has coverage 1.0, the tokenizer emits no
out-of-vocabulary token, and every paired bfloat16-to-float32 cosine is at least 0.99.

Use the existing project-dedicated EC2 `g6.4xlarge` host in `us-east-1`. AWS reported an on-demand
Linux price of $1.3232 per instance-hour on 2026-08-18. Stop this smoke attempt after six instance
hours, for a maximum observed instance charge of $7.94 before storage and data-transfer charges.
The original 40 A100-equivalent GPU-hour experiment budget remains unchanged. Stop the instance
after the run or after the six-hour limit, whichever happens first.

## 2026-08-18 EC2 capacity failure and host amendment before model results

Two exact attempts to start the project-dedicated `g6.4xlarge` host in `us-east-1b` failed with
`InsufficientInstanceCapacity`. AWS left the instance stopped. The failed requests incurred no
instance charge and produced no model result.

Run the same smoke code and scientific configuration on the existing stopped `g6.2xlarge` host
in `us-east-1d`. This host has the same single NVIDIA L4 GPU with 22,888 MiB of device memory. It
has 8 virtual CPUs and 32 GiB of system memory. AWS reported an on-demand Linux price of $0.9776
per instance-hour on 2026-08-18. Keep the six-hour limit, which reduces the maximum observed
instance charge to $5.87 before storage and data-transfer charges. Stop on a memory failure. Do
not reduce the model, precision, sequence length, or panel to make a failed candidate complete.

## 2026-08-18 GENERanno runtime failure and dependency amendment before model results

The first GENERanno load attempt used Transformers 5.12.1, the version that passed the prior
Carbon run. The pinned GENERanno remote model code failed before it created the model because the
Transformers 5 rope registry no longer contains the `default` key that the model requests. The
run produced no feature, coverage, diagnostic, or manifest artifact.

The official GENERanno and GENERator repositories both pin `transformers[torch]==4.49.0`. Run
both GenerTeam candidates in a separate environment with Transformers 4.49.0. Keep Torch 2.11.0,
CUDA 13.0, the model revisions, tokenization rules, precision, sequence panel, and all scientific
thresholds unchanged. Keep Carbon candidates on the already tested Transformers 5.12.1 runtime.
Record the resolved runtime in each candidate manifest.

## 2026-08-18 Carbon-3B memory failure and 48 GB host amendment before retry results

The first Carbon-3B attempt used the approved `g6.2xlarge` host and unchanged full-length
deterministic scaled dot-product attention. It failed during model inference. The process used
14.73 GiB of the 22.03 GiB L4 device and then requested 14.27 GiB more. The attempt produced no
feature, coverage, diagnostic, or manifest artifact. Treat this as a technical memory failure,
not as a model result.

Retry Carbon-3B once on a task-specific `g6e.xlarge` host with one NVIDIA L40S and 45,776 MiB of
device memory. Keep the model revision, code commit, sample manifest, tokenizer, sequence length,
windowing, precision, attention implementation, deterministic settings, and seed unchanged. AWS
reported an on-demand Linux price of $1.861 per instance-hour on 2026-08-18. Stop the retry after
two instance-hours, for a maximum observed instance charge of $3.72 before storage and data
transfer. Stop and remove the task-specific host after the attempt.

## 2026-08-19 Carbon-3B 80 GB host amendment before second retry results

The first Carbon-3B retry exceeded the measured 44.39 GiB on the task-specific NVIDIA L40S.
The process used 43.38 GiB and requested 3.57 GiB more. It produced no feature, coverage,
diagnostic, or manifest artifact. Treat this as a second technical memory failure, not as a model
result.

The user approved another exact retry and paid AWS compute on 2026-08-19. Run Carbon-3B once on a
task-specific `p5.4xlarge` host with one NVIDIA H100 and 81,920 MiB of device memory. Use the
scientific code from commit `66163b161fdd064da3926bf55d8d6853f25cf305`. Keep the model revision,
sample manifest, tokenizer, sequence length, windowing, precision, attention implementation,
deterministic settings, and seed unchanged. AWS Pricing reported an on-demand Linux price of
$6.88 per instance-hour in `us-east-1` on 2026-08-19. Stop the retry after one instance-hour, for
a maximum instance charge of $6.88 before storage and data transfer. Stop and remove the
task-specific host after the attempt.

## Observed numerical smoke outcomes

The 2026-08-18 smoke execution accepted Carbon-500M, GENERanno prokaryote 500M, and GENERator-v2
prokaryote 1.2B. Carbon-3B exceeded both the 22.03 GiB L4 and 44.39 GiB L40S devices. Treat this as
a technical memory failure. Fourteen approved `p5.4xlarge` launch requests failed before instance
creation on 2026-08-19 because AWS had insufficient capacity. Carbon-3B remains unresolved. The
run did not measure retrieval and did not select an encoder.

See [the numerical smoke report](../reports/14_gate1_numerical_smoke.md) for exact S3 versions,
hashes, runtime provenance, failed attempts, cost, and independent read-back results.

## 2026-08-22 invariance implementation amendment before results

Use the same four perturbations as the prior PlasmidCLIP encoder comparison: circular rotations at
25%, 50%, and 75% of sequence length, plus the reverse complement. Calculate each rotation offset
as `round(length_bp * fraction) modulo length_bp`. Require the median cosine for each perturbation,
not a pooled rotation median, to meet the fixed 0.90 threshold. This amendment resolves an
implementation detail that version 0.1 did not state. No invariance feature or model outcome had
been produced when it was added.

Calculate effective rank from the singular values of the centered original-sequence feature
matrix: normalize the singular values to sum to one, calculate their entropy, and exponentiate it.
Divide by the encoder output dimension for the fixed 1% collapse rule. Calculate mean and median
pairwise cosine over all original-sequence vectors. Report length and G+C confounding as the
Pearson correlation between pairwise cosine and, respectively, absolute log2 length ratio and
absolute G+C-fraction difference. Preserve the full per-row perturbation cosine table.

The invariance run must load the exact accepted numerical-smoke version for each candidate and
verify canonical SHA-256 hashes for its panel and numerical-smoke manifests, and recheck the frozen
panel hash before model loading. It runs only in bfloat16 and writes separate versioned features,
coverage, perturbation similarities, diagnostics, and manifest artifacts. The runner executes each
complete Kedro candidate run in a child process under the remaining batch deadline, so input
loading, encoder inference, diagnostics, and artifact writes share one cap. Reserve 30 seconds for
child termination and runner exit. One command uses one total hour cap across all requested
candidates; completed candidates reduce the time available to later candidates. The command must
receive a durable approval reference, AWS region, current instance type, total hour cap, and
observed hourly price. Preparing this harness does not authorize paid execution. A command must not
mix candidate Transformers versions. Run Carbon in the pinned 5.12.1 environment and the two
GenerTeam candidates together in their pinned 4.49.0 environment, with a separate explicit cap for
each command.

## 2026-08-23 A/C/G/T-only panel amendment before retry results

The first Carbon-500M invariance attempt stopped before writing a feature because 23 of the frozen
512 panel rows contained one or more IUPAC ambiguity symbols. The three accepted tokenizers do not
handle these symbols consistently. Carbon-500M and GENERator-v2 map an ambiguity-bearing 6-mer to
an out-of-vocabulary token. GENERanno maps an unsupported symbol to `N`. Native tokenizer behavior
would therefore change the amount and form of information available to each candidate.

Version 0.2 makes an unambiguous source sequence an eligibility condition for the invariance and
nested numerical-smoke panels. An eligible sequence contains only uppercase `A`, `C`, `G`, and
`T`. Do not replace a base, infer a base, discard part of a sequence, or use a tokenizer fallback.
The complete retrieval input and its IUPAC source sequences remain unchanged.

Preserve the version 0.1 selection salt, length deciles, and every eligible version 0.1 panel row.
For each ineligible version 0.1 panel row, select one replacement from the same length decile by
the original rule: preserve an eligible shortest or longest row when the removed row supplied that
extreme, then use deterministic SHA-256 order. Each replacement must use a primary similarity
component that is not already represented. The amended panel must contain 512 rows, ten unchanged
length-stratum quotas, 512 distinct primary components, and no symbol outside `A/C/G/T`.

Bind the amendment to version 0.1 panel SHA-256
`6dddbc33e0bb07ffcd3a2bebfcbf58f8c07573da976d0ef02e62c252e6e1593b`. Record the eligible and
excluded training counts, ambiguity-symbol counts, an exact exclusion-set hash, and the identities
and source hashes of all removed and replacement panel rows. A changed version 0.1 panel hash is an
error.

A read-only derivation from the pinned retrieval and `split_grouped_v2` inputs found 88,474
eligible and 3,805 excluded training rows. It preserved 489 panel rows, replaced 23 rows, retained
all 32 numerical-smoke identities, and produced amended panel SHA-256
`2516a415c7040e4ef75805294c8c9d5693749033c1cd196de24a79f14b5a30a0`. This derivation did not run
an encoder or read validation outcomes or test rows.

The nested 32-row identities did not change, but their parent-panel contract and protocol version
did. Write new versioned numerical-smoke artifacts for Carbon-500M, GENERanno prokaryote 500M, and
GENERator-v2 prokaryote 1.2B before the version 0.2 invariance run. Keep model revisions,
tokenization, precision, pooling, thresholds, and runtimes unchanged.

The user approved this amendment and at most three additional `g6.2xlarge` instance-hours for the
new smoke and invariance commands at the recorded price of $0.9776 per instance-hour. The maximum
approved command cost is $2.9328 before storage and transfer. The user requested that the existing
instance remain running after the commands, so the scientific command timeouts remain mandatory
but no automatic post-run instance stop is part of this amendment.

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

## 2026-08-23 E02b retrieval-recipe amendment before feature extraction

Apply the approved uppercase `A`/`C`/`G`/`T` eligibility rule before every representation. Build
the 20,000-row training panel from eligible training rows. Filter validation rows without
replacement. Evaluate the 6-mer baseline and the three DNA models that passed protocol v0.2
invariance. Record Carbon-3B as technically ineligible within the approved hardware envelope.

Freeze these text recipes from the pinned model cards:

- BGE-base uses the first-token (`[CLS]`) vector, L2 normalization, no document prefix, and
  `Represent this sentence for searching relevant passages: ` before retrieval queries.
- GTE-ModernBERT uses the first-token vector and L2 normalization, with no role prefix.
- Qwen3-Embedding uses left padding, the last non-padding token, L2 normalization, no document
  prefix, and
  `Instruct: Given a plasmid constraint query, retrieve plasmid descriptions that satisfy the recorded constraint\nQuery:`
  before retrieval queries.

Do not truncate a text. Stop if BGE exceeds 512 tokens, GTE exceeds 8,192 tokens, or Qwen exceeds
32,768 tokens. Use bfloat16 and Transformers 5.12.1 for all three text models. Extract each unique
source sequence, paired description, and query-role text once per candidate. Reuse its frozen
vector for rows with the same source hash.

The 6-mer baseline uses lowercase-disabled character 6-mers, standard smoothed inverse-document
frequency, L2-normalized TF-IDF, and randomized truncated SVD with 512 components, seven power
iterations, and seed 20260818. Fit the vocabulary, inverse-document frequencies, and SVD only on
the 20,000 training rows. L2-normalize the SVD vectors. Persist the fitted state and its hashes.

Fit full-rank train-only centering and principal-component whitening with epsilon `1e-6`. Remove
zero components. Use bias-free projection heads. Shuffle training rows once per seed and epoch.
Use every row: the final batch can contain fewer than 4,096 rows and cannot be discarded. Resolve
equal retrieval scores by the stable, ascending frozen gallery order. Use 2,000 whole-component
bootstrap draws with seed 20260818 and persist every draw.

This amendment is preregistered after the input-eligibility audit and before any E02b feature,
probe, validation ranking, or candidate result. The current test split was read during that audit
and is contaminated for later confirmatory use. E02b remains validation-only.

Run paid neural feature extraction and alignment only through
`scripts/run_fixed_representation_bakeoff.py`. The runner requires the exact stage, candidate when
applicable, durable approval reference, region, instance type, total instance-hour limit, and
observed hourly price. It binds every accepted input, invariance, and feature artifact version. It
runs Kedro input loading, computation, and output persistence in one externally timed child
process. The library code also checks the same deadline between model batches, sequences, probe
batches, and bootstrap draws. A configured time limit is therefore an enforced stop rule, not only
provenance text.

Before a later stage can load an artifact, perform an independent persisted read-back, recompute
its table and manifest hashes, record its physical bytes, and freeze its exact version and hashes
in `parameters_fixed_representation_bakeoff.yml`. Feature acceptance must also record measured
extraction GPU-hours. The train-fitted TF-IDF baseline records zero extraction GPU-hours.

## 2026-08-24 host-provenance correction and retry authorization before retry results

AWS identified instance `i-0cda00ffb3cacfc12`, hostname `ip-172-31-90-236`, as an on-demand
`g6.4xlarge` in `us-east-1b`. The 2026-08-23 feature manifests recorded this host as a
`g6.2xlarge` at `$0.9776` per hour. The current AWS price record gives `$1.3232` per on-demand
Linux instance-hour for the actual `g6.4xlarge`. This changes execution provenance and derived
costs. It does not change feature values, feature hashes, model revisions, or scientific
configuration. Do not rewrite the immutable manifests. Store the correction beside their
accepted-artifact records.

The rejected GENERanno version contains a complete-looking feature table with 30,821 unique
sequence hashes, but it lacks the required coverage and manifest artifacts. The current pipeline
has no persisted extraction checkpoint. Do not load or complete this partial version. Repeat the
complete GENERanno stage under a new version.

The retry uses the merged repository after the cleanup review. Comparison with failed-run commit
`3afc292f85222f22d477bbac40be37c55d7dac56` found no scientific change in the neural feature
extraction path: the JSON hash helper moved to the shared serialization module, and unused code
was deleted. The retry must keep the same input and invariance versions, model and tokenizer
revisions, precision, pooling, windowing, seed, and seven-hour stage deadline. Run it as a detached
system service so SSH session removal cannot terminate the process.

The user approved the retry and alignment completion on 2026-08-24 with approval reference
`chat-2026-08-24-e02b-finish-benchmark`. Use the observed `g6.4xlarge` price of `$1.3232` per hour.
The GENERanno limit is seven hours (`$9.2624` maximum), and the alignment limit is three hours
(`$3.9696` maximum). The total additional command limit is ten instance-hours and `$13.2320`
before storage and transfer. No other paid stage is authorized by this approval.

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
