# E00: Benchmark Feasibility

## Status

Active. The profiling pipeline and facet audit v0.2 have run. The current deterministic review
sample is output version `2026-08-05T11.42.05.452Z`. Growth-temperature and bacterial-selection
rules are now version 0.2. A two-axis model-assisted protocol separates semantic support from
benchmark scope, and its 16 deterministic packets are frozen at output version
`2026-08-05T11.45.45.519Z`. The one-row diagnostic and complete 16-row GPT-5.6 Sol gate passed the
structural contract. All 16 rows were semantically supported. Thirteen rows were in scope, two were
out of scope, and one intended-use row was scope-uncertain. No benchmark labels exist. This
experiment is a data and measurement gate. It does not train an embedding model.

## Question

Can the available Addgene fields support a reproducible benchmark of verified, contradicted, and
unknown plasmid-query relationships across declared-family-disjoint data?

## Hypothesis

A small reviewed subset of facets will support reliable atomic and two-facet conjunction queries
across enough independent leakage components. Payload facets with free-text vocabularies will need
more canonicalization and may not pass the first version.

## Fixed input

Initial feasibility evidence was measured from:

```text
s3://plasmidclip/kedro/04_feature/retrieval_dataset.parquet/
2026-08-04T09.02.10.007Z/retrieval_dataset.parquet
```

The implementation must pin the final input through the Kedro catalog. Record its version and
content identity in every output.

## Candidate facets

Audit these first:

1. `plasmid_copy`;
2. `growth_temp`;
3. `bacterial_resistance`, after splitting and reviewing combination values;
4. a curated subset of `vector_types`.

Do not include `insert_genes`, `insert_mutations`, `insert_tags`, `insert_promoters`, or
`insert_species` in benchmark version 1 only because they have high support. Their vocabularies are
large, sparse, and often multi-valued. They need separate alias, completeness, and contradiction
audits.

The proposed semantics are indexed in [Current Facet Semantics](../semantics/README.md). The rules
use narrow Addgene metadata claims. In particular, antibiotic selection and intended-use tags are
positive-only in benchmark version 1.

## Observed feasibility counts

The initial read-only scan found:

| Field | Row coverage | Distinct normalized values | Values with at least 10 rows and 2 components |
| --- | ---: | ---: | ---: |
| `plasmid_copy` | 84.44% | 2 | 2 |
| `growth_temp` | 100.00% | 3 | 3 |
| `bacterial_resistance` | 99.99% | 51 | 28 |
| `vector_types` | 98.97% | 1,294 | 154 |
| `insert_species` | 86.09% | 2,970 | 316 |
| `insert_genes` | 44.12% | 11,404 | 1,064 |
| `insert_mutations` | 25.85% | 21,761 | 378 |
| `insert_tags` | 37.92% | 5,372 | 471 |
| `insert_promoters` | 48.86% | 3,994 | 358 |

High support is necessary but is not evidence that a value is canonical or that missing values are
negative.

## Procedure

### 1. Define facet rules

For each included facet, document:

- what the source field measures;
- how raw values map to canonical values;
- whether multiple values can be true at once;
- what counts as positive evidence;
- what counts as explicit conflict evidence;
- why absence remains unknown or can be interpreted otherwise;
- known ambiguous and excluded values.

Assign each rule a stable version. Unit test every closed-world or conflict rule.

### 2. Manual audit

Draw a deterministic, stratified sample for each facet and state. Include common values, rare
eligible values, multi-valued rows, missing rows, and proposed contradictions. Review the raw field,
source description, generated description, and relevant annotations where applicable.

Set the sample size before review. Report an interval for estimated label error. Do not pass a facet
only because aggregate counts look useful.

### 3. Build stable evidence products

Build the constraint vocabulary and sparse evidence table. Verify:

- content identifiers are invariant to row order;
- canonicalization is deterministic;
- no sequence-constraint pair has both states;
- every contradicted state cites a reviewed rule and source value;
- raw values remain traceable.

Constraint evidence in benchmark version 1 comes from reviewed Addgene fields. If annotation-derived
constraints are added later, use an explicit pLannotate-only view. Do not merge plasmidkit evidence
or use it as an automatic fallback.

### 4. Audit the split

The current split has 1,199 test components, but its largest component has 3,364 rows. This is
29.2% of the test rows. The ten largest components contain 44.8% of test rows.

Before freezing the benchmark:

- measure exact and approximate cross-split sequence similarity;
- report family-key source (`backbone`, name prefix, or isolated ID);
- report component-size concentration in each split;
- compare query-macro and component-macro estimates;
- design a revised split or grouped resampling scheme if conclusions are unstable.

Do not inspect model outcomes while selecting the benchmark split.

### 5. Freeze queries and galleries

Select constraints using training support only. Freeze atomic and conjunction query sets before
model experiments. Record test support for reporting, not selection.

Start with canonical symbolic queries. Add natural-language templates as a separate version. Add
paraphrases only after a human review confirms semantic equivalence.

### 6. Run measurement controls

Implement:

- exact verified-set oracle;
- explicit contradiction-first ranking;
- constraint-prevalence ranking;
- deterministic random ranking;
- paired-origin ranking where a source exists.

Each control must produce analytically expected metric behavior on synthetic fixtures before it
runs on the full artifact.

## Primary outputs

- Included facets and rule versions.
- Manual audit results and estimated label error.
- Atomic and conjunction counts by split and independent component.
- Verified, contradicted, and unknown set-size distributions.
- Split similarity and concentration report.
- Oracle and negative-control metrics.
- A pass, narrow, or stop decision for model work.

Any annotation coverage reported beside E00 must be pLannotate-only and must report the 48 retrieval
plasmids without pLannotate rows as missing annotation evidence.

## Acceptance rule

Pass a benchmark version only when:

- every included facet has a reviewed evidence rule;
- manual audit error is within a threshold fixed before review;
- oracle and negative controls behave as specified;
- query sets have adequate support across independent components;
- split concentration does not reverse the main result under the predefined estimands;
- cross-split near-duplicate leakage is measured and acceptably bounded;
- all artifacts are deterministic from pinned inputs and configuration.

The [manual audit protocol v0.2](../semantics/manual_audit_v0.2.md) records the current revised-rule
sample and stopping conditions. Protocol v0.1 and its outputs remain unchanged as research history.

## Current audit sample

The active v0.2 sample contains 918 rows from 768 leakage components. It contains train and
validation rows only. It contains 169 deterministic second-review rows. The sample preserves each
raw source cell, both descriptions, the Addgene URL, the rule version, the proposed symbolic claim,
and a readable mapping note for every revised non-obvious mapping.

The revised-rule strata include all 41 eligible `room_temperature` components and all 34 eligible
components across the five revised bacterial-selection cells.

Three support gates remain unresolved before rule acceptance:

- residual excluded resistance syntax has 15 independent components, below the target of 75;
- the ordinary resistance-combination stratum has no apramycin or erythromycin components and only
  two hygromycin components, below the minimum of five;
- the intended-use stratum has only three independent affinity-reagent components.

Do not lower these thresholds. Report the affected categories as unresolved or exclude them from a
later accepted rule version.
