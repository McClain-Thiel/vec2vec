# Set-Valued and Compositional DNA-Language Embeddings

## Current answer

**Gate 0 is complete.** The global similarity graph is accepted (4,450,238 primary and 4,676,653
sensitivity edges over all 115,120 plasmids), the leak-free `split_grouped_v2` is built and
independently validated with **zero primary cross-split edges** (92,279 / 11,344 / 11,497
train/val/test, and far less concentrated than the old split: test's largest component fell from
29.23% of rows to 2.86%), and the frozen query benchmark v0.1 (131 semantic queries, 4 candidate
galleries, 5,740,247 sparse verified/contradicted states) passed its own independent read-back
validation, including the oracle and contradiction-first controls. The Gate 0 data-support flag
passed for both the validation and test closed evaluations.

The frozen evidence contract underlying the benchmark contains 35 stable constraints and 684,987
unique verified or contradicted plasmid-constraint states. A 30-application hand check and fixed
240-application strong-model benchmark passed without an unsupported, uncertain, or out-of-scope
positive mapping. These remain narrow Addgene metadata states, not biological ground truth — the
benchmark measures recorded-metadata consistency, not plasmid function.

**Status:** Gate 0 complete. Not yet ready for model training — Gate 1 (fixed representations) is
next.

**Next decision:** select and pin one DNA encoder and one text encoder (Gate 1). Validate
sequence-length coverage and the long-sequence policy, circular-rotation sensitivity,
reverse-complement sensitivity, pooling behavior, and storage/compute cost for frozen features
before any training run.

**Current report:** [Split similarity and concentration audit v0.1](reports/10_split_audit_v01.md)
covers the near-duplicate finding that motivated `split_grouped_v2`; a Gate 0 completion report
covering the accepted graph, v2 split, and query benchmark has not been written yet. The study has
no interpretation notebook because model evaluation has not started. See the
[experiment log](EXPERIMENT_LOG.md) for the chronological record.

## Research questions

Use separate questions so that a failure has a clear meaning.

1. **Set supervision:** Does verified multi-positive supervision improve retrieval of verified
   answer sets over paired identity supervision?
2. **Symbolic composition:** When training uses atomic constraints only, does adding learned atomic
   query vectors retrieve the verified intersection for an unseen conjunction?
3. **Language composition:** Does a frozen text representation map direct conjunction language to
   the same retrieval behavior as the symbolic sum?
4. **Geometry:** Does a validated sequence or scaffold prior improve generalization across held-out
   declared families?
5. **Modification, later:** Does adding or subtracting a text direction retrieve putative
   minimal-change variants of a source plasmid?

Questions 4 and 5 proceed only after their data products pass separate validation gates. Do not
describe subtraction as removal before question 5 succeeds.

## Claims that this study can and cannot make

The first study can test retrieval against recorded Addgene metadata. A verified match means that
the available metadata supports the query. It does not prove that a plasmid works in a biological
system.

Use these labels:

- **verified:** all query constraints have positive recorded evidence;
- **contradicted:** at least one constraint has explicit, facet-specific conflict evidence;
- **unknown:** the record does not establish either state.

Do not call `verified` the complete set of acceptable plasmids. The data is incomplete. Report
verified, contradicted, and unknown retrieval fractions separately.

Do not call a learned constraint interaction biological epistasis. Addgene co-occurrence reflects
biology, engineering practice, contributor choices, and collection bias. Use **non-additivity** or
**constraint interaction** unless a causal interpretation becomes justified.

## Validated input

The initial review used this catalog artifact:

```text
s3://plasmidclip/kedro/04_feature/retrieval_dataset.parquet/
2026-08-04T09.02.10.007Z/retrieval_dataset.parquet
```

Observed properties:

| Property | Observed value |
| --- | ---: |
| Rows | 115,120 |
| Distinct exact sequences | 114,968 |
| Declared families | 14,202 |
| Leakage components | 14,157 |
| Train / validation / test rows | 92,097 / 11,515 / 11,508 |
| Components crossing the grouped split | 0 |
| Median sequence length | 6,943 bp |
| Maximum sequence length | 98,922 bp |

The split guarantee covers `family_key` and exact sequence hashes. The global lower-bound audit
found 7,624 strict near-duplicate cross-split edges, so the current split fails the approximate
similarity rule. Preserve it as **declared-family-disjoint but near-duplicate-contaminated**. Do not
use it for model evaluation.

**Superseded for model evaluation.** `split_grouped_v2` (output version `2026-08-17T23.49.47.355Z`,
built from the accepted complete similarity graph) closes this leak: zero primary cross-split edges,
92,279 / 11,344 / 11,497 train/val/test, and substantially better concentration than the table
above. Use `split_grouped_v2`, not `split_grouped`, for any evaluation. See the
[experiment log](EXPERIMENT_LOG.md) entry for 2026-08-18.

## Study gates

### Gate 0: benchmark feasibility

Run E00 before encoder selection or model training.

Required outputs:

- a versioned constraint vocabulary with stable content identifiers;
- explicit facet rules for positive, conflict, and unknown evidence;
- a stratified accuracy benchmark of included facets and targeted review of ambiguous mappings;
- a frozen query catalog and candidate galleries;
- split concentration and cross-split near-duplicate audits;
- oracle, contradiction, prevalence, and random baselines;
- an estimate of usable atomic and conjunction queries by split and component.

Pass only if the benchmark measures recorded constraint satisfaction with acceptable label error
and enough independent components. If it fails, refine the data or narrow the claim. Do not change
the model to compensate for a weak benchmark.

### Gate 1: fixed representations

Select one DNA encoder and one text encoder after Gate 0. Pin their Hugging Face repository
revisions. Validate:

- sequence-length coverage and the long-sequence policy;
- circular rotation sensitivity;
- reverse-complement sensitivity;
- pooling behavior;
- storage and compute cost for frozen features.

The source data ranges from 201 to 98,922 bp. Encoder context and pooling are therefore part of the
experimental definition, not an implementation detail.

### Gate 2: set supervision

Compare paired identity and verified-set supervision with the same controlled query texts,
candidate sampler, frozen features, projection size, update budget, and seeds. A separate
description-retrieval track can follow. This control prevents query wording from being confused
with the objective.

### Gate 3: composition

Train on atomic queries. Freeze the eligible atoms and held-out conjunctions before the main run.
Test whether the vector sum retrieves the verified intersection. Then test whether direct natural
language conjunctions induce similar behavior.

### Gate 4: optional geometry

Build geometry only after cross-split similarity and annotation-coordinate semantics are valid.
Do not construct a global sequence graph by blocking only within existing families; that design
cannot detect cross-family similarity.

Use pLannotate alone for the primary annotation-derived geometry and coverage measurements. The
plasmidkit source can provide a separate sensitivity analysis, but it must not be merged into the
primary graph or used as a fallback for missing pLannotate rows.

### Gate 5: optional modification

Treat mined pairs as **putative variant pairs**, not known engineering edits. Addgene and predicted
annotations do not establish edit history or direction. Proceed only if coordinate normalization,
outside-interval similarity, semantic differences, and manual review all pass.

Use pLannotate coordinates and features for the primary edit benchmark. Keep plasmidkit-derived
results in a separately named concordance analysis.

## Kedro pipeline layout

Keep experiment identity in configuration and study specifications. Do not create a new pipeline
implementation for every experiment.

```text
existing data pipelines
  processing -> dataset -> audit

new core pipelines
  constraint_semantics
    -> constraint_evidence
    -> benchmark
    -> encoder_features
    -> training
    -> evaluation
    -> comparison

optional gated pipelines
  sequence_geometry
  edit_benchmark
```

Responsibilities:

| Pipeline | Responsibility | Main layer |
| --- | --- | --- |
| `constraint_semantics` | Canonical constraints and evidence rules | `04_feature` |
| `constraint_evidence` | Rule-derived training claims and validation sample | `05_model_input`, `08_reporting` |
| `benchmark` | Queries, exclusions, galleries, and frozen labels | `05_model_input` |
| `encoder_features` | Pinned frozen DNA and text features | `05_model_input` |
| `training` | Parameterized objectives and model fitting | `06_models` |
| `evaluation` | Per-query scores, predictions, and metric components | `07_model_output` |
| `comparison` | Seed summaries, paired contrasts, intervals, and plot data | `08_reporting` |
| `sequence_geometry` | Validated sequence or scaffold graph products | `04_feature` |
| `edit_benchmark` | Validated putative variant pairs and edit queries | `05_model_input` |

Put reusable pure logic in `src/vec2vec/lib/`. Add a package only when one module becomes hard to
read. Let Kedro provide orchestration and let W&B record runs. Do not add a second experiment
registry or a custom runner.

## Data products

Start with simple Parquet products. Do not add compressed bitmap infrastructure until measurement
shows that sorted integer arrays or factorized constraint sets are too slow.

### Constraint vocabulary

Each constraint needs a content-derived string identifier. An incrementing integer is not stable
when a new value is inserted.

Minimum fields:

```text
constraint_id
facet
relation
canonical_value
rule_id
rule_version
train_row_support
train_component_support
```

Relations must reflect the stored field. For example:

- `insert_species derived_from homo_sapiens`, not host compatibility;
- `growth_strain propagated_in dh5alpha`, not host range;
- `growth_temperature propagated_at 37_c`, not guaranteed growth capability;
- `bacterial_resistance selectable_with ampicillin`;
- `vector_type tagged_as mammalian_expression`.

### Constraint evidence

Use a sparse table with evidence, not an unexplained confidence score:

```text
sequence_id
constraint_id
state                 # verified or contradicted; absence means unknown
evidence_type
source_field
source_value
rule_id
```

Only materialize contradicted states for reviewed conflict rules. A different recorded value is not
automatically a contradiction for multi-valued or incomplete facets.

### Query catalog

Derive `query_id` from the constraint set, wording revision, gallery, and exclusion policy. Do not
derive it from a row position. Store symbolic constraints and text as separate views of one query.

For source descriptions, record and exclude the source and exact duplicates. Atomic canonical
queries do not require an artificial source plasmid. Define any same-family exclusion as a separate
gallery because it changes the scientific question.

### Annotation-derived products

Create an explicit pLannotate-only catalog view before producing feature coverage, masked sequence,
geometry, or putative edit products. The reviewed artifact contains pLannotate rows for 115,072 of
115,120 retrieval plasmids (99.958%). Treat the other 48 plasmids as missing annotation evidence.
Do not fill them from plasmidkit in the primary analysis.

Every derived interval must retain:

```text
sequence_id
source                    # must be plannotate for primary products
source_start
source_end
source_strand
normalized_start
normalized_end
wraps_origin
normalization_rule
plannotate_version
database_version
```

The existing `annotations` column and `addgene_annotation_features` dataset are mixed-source
products. They remain valid with that provenance, but they are not inputs to pLannotate-only
measurements.

## Metrics and estimands

Primary retrieval outputs:

- Verified@K.
- Contradicted@K.
- Unknown@K.
- Per-constraint verified and contradicted fractions.
- Coverage: the fraction of a ranked list with known relevance.

Use the three fractions together. Do not interpret unknown as a low relevance grade. Do not use the
proposed four-level nDCG until the relevance grades have an evidence-based meaning.

For composition, report:

- retrieval of the verified atomic-set intersection by `q_A + q_B`;
- the change from the best atomic query to the sum;
- the distributional gap between direct conjunction text and the symbolic sum;
- results by conjunction support and held-out declared family.

For the exponential-tilt score, `p(q_A + q_B)` equals the normalized product of `p(q_A)` and
`p(q_B)` divided by the same base measure. This is an algebraic identity. Do not report it as
learned compositional behavior.

Define both estimands before evaluation:

- **query-macro:** each eligible query has equal weight;
- **component-macro:** calculate within each leakage component, then give each component equal
  weight.

Report row- or query-weighted results as secondary where large components can dominate. Bootstrap
whole components and state how resampled components are weighted.

## Experiment sequence

| ID | Experiment | Gate | Status or outcome |
| --- | --- | --- | --- |
| E00 | Benchmark feasibility and oracles | 0 | [Complete — Gate 0 accepted](EXPERIMENT_LOG.md) |
| [E00-S](experiments/E00_split_grouped_v2.md) | Strict similarity-closed grouped split v2 | 0 | Complete; zero primary cross-split edges |
| [E00-Q](experiments/E00_query_benchmark_v0.1.md) | Frozen symbolic queries, galleries, and controls | 0 | Complete; Gate 0 data-support flag passed both splits |
| E00-J | Agent-assisted facet review pilot | 0 | Complete |
| [E01](experiments/E01_training_constraint_evidence.md) | Rule-derived training constraint evidence | 0 | [Accepted for noisy supervision](reports/08_constraint_accuracy_benchmark.md) |
| E02 | Paired identity control on controlled queries | 2 | Planned |
| E03 | Verified-set supervision | 2 | Planned |
| E04 | Atomic-only symbolic addition | 3 | Planned |
| E05a | Compound supervision without additivity regularization | 3 | Planned |
| E05b | The same compound supervision with additivity regularization | 3 | Planned |
| E06 | Validated geometry ablation | 4 | Gated |
| E07 | Query norm and information analysis | after E03 | Planned |
| E08 | Descriptive non-additivity map | after E05 | Planned |
| E09 | Putative source-conditioned modification | 5 | Gated |

E05a and E05b must use the same compound queries. The initial plan called this comparison E04. It
changed both the training data and the regularizer, which would not isolate the effect of the
regularizer. Unstarted experiment identifiers shifted when the implemented constraint-evidence
work received identifier E01. The historical plan report keeps its original identifiers.

## Study directory

```text
studies/set_valued_compositional_embeddings/
├── README.md
├── EXPERIMENT_LOG.md
├── experiments/
│   ├── E00_benchmark_feasibility.md
│   ├── E00_agent_judge_pilot.md
│   └── E01_training_constraint_evidence.md
├── notebooks/
│   └── README.md
└── reports/
    ├── 00_plan_validation.md
    └── 08_constraint_accuracy_benchmark.md
```

Add one experiment specification per controlled comparison. Add one notebook per interpretation
question. Keep final figures and tables under `reports/` when they exist.

Append every planned, completed, failed, or changed run to `EXPERIMENT_LOG.md`. Never rewrite an
earlier log entry to match a later interpretation.
