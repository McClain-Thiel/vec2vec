# E00: Split Similarity and Concentration Audit

## Status

Active. This protocol was fixed before the full cross-split similarity result was inspected.
It audits the existing `split_grouped` assignment. It does not change or overwrite that split.

## Question

Does the current declared-family and exact-sequence grouped split prevent near-duplicate plasmid
sequences from crossing train, validation, and test? Is each evaluation split too concentrated in
a small number of leakage components for stable component-macro estimates?

## Fixed input

```text
s3://plasmidclip/kedro/04_feature/retrieval_dataset.parquet/
2026-08-04T09.02.10.007Z/retrieval_dataset.parquet
```

The audit must record the input content identity, Git commit, dirty-worktree state, complete
configuration, tool versions, and generated artifact versions.

## Existing split contract

The current split unions rows that share a declared `family_key` or an exact `sequence_sha256`.
It then assigns each complete component to one split. The audit first verifies these invariants.
Approximate sequence similarity is an additional test and is not assumed from the family labels.

## Cross-split search

Run an indexed nucleotide search for all three cross-split pairs:

1. validation query against the train database;
2. test query against the train database;
3. test query against the validation database.

Use NCBI BLAST+ `megablast` with both strands enabled. Repeat each query sequence twice in its
FASTA record. A complete circular rotation can then align as one continuous segment across the
query's original endpoint. Classify each sequence pair from its best high-scoring segment. Report
query and subject coverage separately. Require a compatible length ratio so the doubled query
cannot make a tandem duplication look like complete mutual coverage. Do not interpret a common
local cassette as a whole-plasmid near duplicate.

The search must fail if a configured target cap is reached. It must not silently truncate
candidates. Preserve the BLAST parameters and version in the audit manifest.

## Thresholds fixed before the full run

The primary near-duplicate definition is:

- aggregate nucleotide identity at least 99%;
- query coverage at least 95%;
- subject coverage at least 95%.
- shorter-to-longer complete-sequence length ratio at least 95%.

The sensitivity definition is:

- aggregate nucleotide identity at least 95%;
- query coverage at least 90%;
- subject coverage at least 90%.
- shorter-to-longer complete-sequence length ratio at least 90%.

Coverage is the aligned coordinate span divided by the complete, undoubled sequence length and is
capped at 100%. Identity is identical aligned bases divided by aligned columns in the selected
segment. Threshold comparisons are inclusive. The candidate-search coverage floor is 80% of the
undoubled query, below both reporting thresholds.

## Validation before the full run

A deterministic synthetic fixture must show these behaviors:

- an exact copy is detected;
- a circular rotation is detected;
- a reverse complement is detected;
- a sequence with approximately 1% substitutions meets the primary rule when coverage permits;
- two otherwise unrelated plasmids that share only one cassette do not meet either whole-plasmid
  rule;
- invalid lengths, coordinates, missing values, duplicate rows, and target-cap saturation fail
  with specific errors.

Run a bounded sample through the external search before the complete input. The sample validates
the tool boundary and estimates runtime. It is not evidence about full-dataset leakage.

## Concentration measurements

For each split, report:

- rows and leakage components;
- singleton components;
- largest-component and ten-largest-component row fractions;
- effective component count from row weights;
- median, 90th percentile, and maximum component size.

The effective component count is `1 / sum(weight_i ** 2)`, where `weight_i` is a component's
fraction of rows in its split. This is a descriptive concentration measure. It is not an estimate
of model performance.

The later benchmark comparison must report both query-macro and component-macro estimates. This
audit cannot test whether those estimands reverse a model conclusion because no model outcomes are
inspected during split selection.

## Decision rule

The current split fails the strict near-duplicate rule if one or more primary cross-split edges
exist. In that case, preserve it and create a separately named split version that unions the new
near-duplicate edges before assignment. Do not modify the existing `split_grouped` column in
place.

The sensitivity result is diagnostic. Sensitivity-only edges do not automatically fail the strict
rule, but the audit must report their counts, involved rows, and augmented component sizes.

Concentration does not have a defensible universal pass threshold before model metrics exist.
Report it as a limitation. If one component contains at least 25% of an evaluation split, prepare
component-macro evaluation and grouped resampling as required analyses before a benchmark can pass.

## Stopping rule and compute budget

Use at most 10 local CPU threads. Stop on an external-tool error, invalid input, candidate-cap
saturation, or a failed synthetic invariant. Do not change thresholds after seeing the full
result. A changed threshold requires a dated protocol amendment and a separately named output.

## 2026-08-06 execution amendment

The initial complete run used BLAST automatic threading and a cap of 1,000 target sequences per
query. It was stopped as a technical failure after approximately 90 minutes. The first of three
search pairs had not completed, its temporary result file was empty, and no similarity result was
read or saved.

The rerun keeps all scientific thresholds, masking, candidate identity, coverage, and HSP rules
unchanged. It fixes BLAST to query-level threading (`-mt_mode 1`) across the same 10 threads and
sets the candidate cap to 100 targets per query. A query that returns 100 targets fails the run.
Thus, the smaller cap cannot be interpreted as a complete edge count when truncation occurs.

### Second execution amendment: global lower-bound search

The amended direct-BLAST benchmark was also stopped as a technical failure. A stable sample of
100 validation queries had not completed after more than three minutes against the complete train
database. No alignment result was produced.

Two Mash prefilters were measured and rejected before use as scientific evidence. At k=12, 100
queries produced 3,848,503 pairs at distance at most 0.10. At k=21, the same queries produced
2,598,344 pairs at distance at most 0.05 before the complete-sequence length gate and 486,607 after
the 0.90 length-ratio gate. These candidate sets were too broad for pair confirmation. The k=11
prototype was rejected after Mash warned that its random-match probability exceeded the configured
warning threshold for the longest sequences.

The complete rerun uses minimap2 2.31 with the `asm20` assembly preset. It retains the doubled
query, both strands, 10 threads, and all fixed identity, coverage, and length thresholds. A frozen
synthetic run detects the exact copy, circular rotation, reverse complement, and 39-substitution
case, while the shared-cassette case fails whole-plasmid coverage.

Minimap2 retains 10 secondary alignments with a minimum secondary-to-primary score ratio of 0.5.
All validation and test queries are searched against the relevant complete target split, but the
edge table is a lower bound because secondary alignments can be truncated. A nonzero primary edge
count is a valid counterexample and rejects the current split. A zero count would be inconclusive
and cannot pass it. The lower-bound graph must not be used to claim that a repaired v2 split is
leak-free; v2 needs a complete closure search and a new audit.

## Known limitations

- BLAST is a heuristic candidate search, not a mathematical all-pairs proof of absence.
- Rearranged plasmids can need several local segments. The aggregation rule can miss a complex
  rearrangement even when much sequence content is shared.
- Sequence similarity does not by itself establish functional or biological independence.
- The audit measures data structure only. It does not measure downstream model leakage effects.
