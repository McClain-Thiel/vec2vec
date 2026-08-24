# Validation of the Initial Implementation Plan

## Scope

This review compared the initial plan with commit `3c445dd`, the Kedro catalog, the implemented
constraint and split logic, the current reporting artifacts, and the versioned retrieval dataset
from `2026-08-04T09.02.10.007Z`.

The review was read-only. `AGENTS.md` was an uncommitted documentation file during the scan.

## Overall assessment

The plan has a strong research objective, conservative treatment of missing metadata, useful
decision gates, and a good preference for fixed encoders during objective comparisons. It should
not be implemented in its original order.

Benchmark validity, split validity, and the mathematical interpretation of additivity must be
corrected first. Geometry and edit mining are later feasibility studies, not core data engineering
that can be assumed to work.

## Claims confirmed by the data and code

- The retrieval artifact contains 115,120 rows.
- The grouped split contains 92,097 train, 11,515 validation, and 11,508 test rows.
- It contains 14,157 constructed leakage components and 14,202 family keys.
- No constructed component, family key, or exact sequence hash crosses the grouped split.
- The current train audit contains 276,168 controlled queries.
- The median query has five metadata-derived hard negatives within its same-backbone pool.
- Metadata absence is kept separate from a recorded field mismatch in the current index.
- The catalog carries the fields named in the plan and preserves versioned retrieval data.

## Corrections required

### The answer sets are verified sets, not complete acceptable sets

The current relevance index uses recorded metadata. It cannot establish all plasmids that would
biologically satisfy a natural-language request. A candidate without evidence is unknown. Therefore
`A_Q` must mean a verified-positive set, not the complete set of acceptable plasmids.

This changes metric names and interpretation. `Validity@K` becomes `Verified@K`, and it must be
reported with `Contradicted@K` and `Unknown@K`.

### Current contradictions are not valid for every facet

The current implementation treats a candidate as a mismatch when it records a different value in
the same field. This is suitable only after a facet-specific completeness and exclusivity review.
It is unsafe as a general rule for genes, promoters, tags, species, and other multi-valued fields.

The source satisfying its surfaced constraints is mostly true by construction: surfaced values are
extracted from that source row's own metadata. This check detects implementation drift but does not
validate biological correctness or LLM description quality.

### Canonicalization is not yet implemented

Current normalization performs Unicode, case, punctuation, placeholder, and delimiter handling. It
does not implement a reviewed alias ontology. The bacterial-resistance field alone includes values
such as `chloramphenicol and ampicillin`, `bleocin zeocin`, and dosage-bearing free text. Stable
constraint identifiers and alias rules are new work.

### The current audit is narrower than the proposed benchmark

The 276,168-query audit builds nested controlled queries from source metadata and searches for hard
negatives among same-backbone train peers. It does not construct full-gallery verified,
contradicted, and unknown sets. Its query IDs contain row positions and are not stable under row
reordering.

### The grouped split needs a concentration and similarity revision

The split is pure with respect to its constructed keys, but the test set is concentrated:

| Split | Components | Largest component share | Ten largest component share |
| --- | ---: | ---: | ---: |
| Train | 11,456 | 2.83% | 15.25% |
| Validation | 1,502 | 16.23% | 30.54% |
| Test | 1,199 | 29.23% | 44.84% |

The family key is based on declared backbone for 114,024 rows, a name prefix for 742 rows, and an
isolated ID for 354 rows. It is not a measured sequence-similarity cluster. The phrase "unseen
plasmid family" is too broad. Use "held-out declared family" and audit cross-split near-similarity.

### Product-of-experts agreement is an identity

With

```text
p(x | q) proportional to mu(x) * exp(q dot z_x / tau),
```

the distribution induced by `q_A + q_B` is exactly the normalized product of the two atomic
distributions divided by `mu`. No training result is needed. This equality validates an
implementation, not compositional semantics.

The scientific test is whether `q_A + q_B` retrieves the verified intersection for a held-out
conjunction. A second test asks whether direct conjunction text maps to the same retrieval behavior
as the symbolic sum.

### Query norms are not identified without scale controls

The score is invariant to compensating rescaling of query vectors, plasmid vectors, and
temperature. If temperature is learned and both projections can change scale, query norm does not
have a unique scientific meaning. Fix the scale or treat norm analysis as descriptive. Weight decay
alone does not establish identifiability.

### Unknown candidates need an explicit training interpretation

Normalizing a loss only over verified and contradicted candidates defines a conditional labeled-set
objective. It is not the full-gallery distribution stated earlier in the plan. Unknown candidates
can receive arbitrary full-gallery scores unless another term constrains them.

Keep the primary loss conditional on labeled candidates if it works, but name it accurately. Test
how it behaves when unknown candidates enter the evaluation gallery. Do not give unknown candidates
an arbitrary low nDCG grade.

### Geometry cannot use within-family blocking for the primary graph

A raw-sequence graph intended to measure cross-family relationships cannot generate candidates only
inside current family or leakage components. That construction would reproduce the split assumption
and miss the similarities that need auditing. Use a scalable sequence sketch or another global
candidate method, then validate exact similarities for proposed neighbors.

### Annotation coordinates are not ready for masking or edits

The annotation table has 5,936,251 rows from two sources. The source coordinate conventions remain
intentionally unmodified. The review observed 49,303 pLannotate rows with `start > end` and 29,407
plasmidkit rows whose `end` exceeds the recorded length among rows matched to the retrieval data.
These can include circular wraparound and source-specific conventions, but their meaning has not
been resolved.

Backbone masking and edit interval comparison therefore require a coordinate-semantics pipeline,
tests, and manual review first. Annotation overlap does not prove actual edit history.

The project has now selected pLannotate as the primary source for new annotation-derived
measurements. In the retrieval population, pLannotate covers 115,072 of 115,120 plasmids and
provides 3,233,114 annotation rows. The 48 plasmids without pLannotate rows will remain missing in
the primary analysis. The plasmidkit source will be used only for separately reported concordance
or sensitivity checks.

This policy does not change the provenance of existing artifacts. The retrieval dataset's feature
list and some generated descriptions were constructed from both sources. They must remain labelled
as mixed-source.

### E04 needs a factorial control

The initial E04 adds both compound-query training data and an additivity loss. Compare the same
compound training data with and without the additivity loss. Otherwise the cause of a change cannot
be isolated.

### Objective comparisons need matched query inputs

Paired CLIP on free descriptions and set supervision on controlled atomic text differ in both loss
and language input. The primary objective comparison must use matched controlled query texts and
candidate batches. Free-description retrieval is a separate external-validity experiment.

### Edit subtraction has no guaranteed removal meaning

Subtracting `q_A` lowers scores in the learned direction relative to the base distribution. It does
not logically guarantee absence or removal of `A`. The modification study must test this claim; it
must not assume it in labels or naming.

## Architecture decision

Use six small core Kedro pipelines: constraint semantics, benchmark construction, frozen encoder
features, parameterized training, frozen evaluation, and experiment comparison. Keep geometry and
edit work behind gates. Use Kedro configuration and W&B groups as the experiment registry. This is
simpler than adding a second custom registry and runner.

## Recommended first milestone

Complete E00 and publish a benchmark audit report. Do not choose a final DNA encoder, train E01, or
implement edit mining until E00 passes. The first useful research result may be that only a narrow
subset of the metadata supports defensible set-valued evaluation. That is an acceptable result.
