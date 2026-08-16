# E00: Frozen Query Benchmark v0.1

## Status

Preregistered on 2026-08-13 before the complete similarity graph or `split_grouped_v2` result
was available. This experiment constructs data and measurement controls. It does not train or
select a model.

## Question

Does the frozen three-state Addgene metadata contract support a deterministic set-retrieval
benchmark after strict sequence-similarity split closure?

## Fixed inputs

- Retrieval dataset `2026-08-04T09.02.10.007Z`, population SHA-256
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Constraint-state artifact `2026-08-06T13.27.47.937Z`.
- One independently accepted global similarity-graph artifact.
- One independently accepted `split_grouped_v2` artifact built from that graph.

The graph and split artifact versions must be pinned at execution time. The pipeline must fail
before writing if either accepted manifest or a content hash differs.

## Query definitions

Version 0.1 uses canonical symbolic queries only. It does not include paraphrases, generated
descriptions, triples, or source-conditioned queries.

An atomic constraint is eligible when its v2 training split contains at least 20 verified rows
from at least five v2 leakage components. A lack of reviewed contradiction evidence does not
remove a positive-only atomic constraint. Instead, the catalog marks that its contradiction
control is not eligible.

Construct every two-constraint conjunction that meets all of these rules in the v2 training
split:

- both atoms are eligible;
- the atoms use different facets;
- their verified-set intersection contains at least 20 rows from at least five v2 components;
- their explicit contradiction-set union contains at least 20 rows from at least five v2
  components.

These thresholds and the different-facet rule are fixed before the v2 split is inspected. Do not
discard a frozen query because validation or test support is small. Report that support instead.
A later change requires a new benchmark version.

For the Gate 0 data-support decision, a query is measurement-usable in a closed gallery when it
has at least 10 verified candidates from at least two v2 leakage components. A contradiction
control is usable when it has at least 10 contradicted candidates from at least two v2 components.
Each closed validation and test gallery must contain at least 10 usable atomic queries, at least
20 usable pair queries, and at least 20 pair queries with usable contradiction controls. These
counts determine pass versus narrow/stop. They do not remove rows from the frozen catalog.

Create canonical text from the recorded facet, relation, and value. This text is an auditable
symbolic rendering, not a reviewed natural-language paraphrase. Derive identifiers from the
benchmark version, ordered constraint identifiers, text revision, gallery, evaluation split, and
exclusion policy.

## Answer sets

For constraint set Q and candidate x:

- verified: every constraint in Q has a verified state for x;
- contradicted: at least one constraint in Q has an explicit contradicted state for x;
- unknown: neither rule applies.

Unknown is implicit. Persist only verified and contradicted query-candidate rows. Atomic verified
sets must equal their source constraint sets. Conjunction verified sets must equal the exact
intersection of their atomic verified sets. Verified and contradicted sets must be disjoint.

## Galleries and exclusions

Produce validation and test views of two galleries:

- `closed_grouped_v2`: candidates from the same v2 split only;
- `open_all`: every retrieval plasmid.

Version 0.1 has no source plasmid because it contains only canonical symbolic queries. Therefore
source, identical-sequence, and same-backbone exclusions are not applicable. Record this fact in
the manifest. Do not silently generalize it to later source-description or edit queries.

Use answer-set buckets `zero`, `singleton`, `2_to_10`, `11_to_100`, `101_to_1000`, and
`over_1000`. Retain zero-answer rows as feasibility evidence. They are not valid retrieval metric
units.

## Base measures

Normalize and store log mass inside each gallery for:

1. uniform plasmid;
2. uniform v2 leakage component;
3. uniform declared family;
4. primary-graph degree correction with unnormalized weight `1 / (1 + primary_degree)`.

The graph-degree measure is descriptive. It does not claim phylogenetic distance.

## Measurement controls

Use `K = 1, 5, 10, 50`.

- `verified_first_oracle`: verified, then unknown, then contradicted; stable sequence-ID ties.
- `contradiction_first`: contradicted, then unknown, then verified; stable sequence-ID ties.
- `metadata_prevalence_prior`: query-independent candidate ranking by the sum of
  `log1p(v2 train verified row support)` over the candidate's verified frozen constraints.
- `deterministic_random`: a stable 64-bit keyed permutation with seed 42.

Report verified, contradicted, unknown, and known fractions at K. Also report mean per-candidate
fractions of query constraints verified, contradicted, and unknown. The random control includes
the analytic gallery prevalence for comparison. A paired-origin control is not applicable because
version 0.1 has no source queries.

Do not calculate four-grade nDCG. The available evidence does not justify ordered relevance among
unknown candidates.

## Acceptance and stopping rules

The data product is accepted only if:

- all input versions are pinned and accepted;
- all joins are one-to-one where required and preserve the 115,120-row population;
- no v2 component crosses a split;
- every query has disjoint verified, contradicted, and unknown sets;
- atomic and conjunction set identities pass exact checks;
- each base measure sums to one within numerical tolerance in every gallery;
- the verified-first oracle has verified@K equal to one whenever at least K verified candidates
  exist;
- the contradiction-first control has contradicted@K equal to one whenever at least K
  contradicted candidates exist;
- all identifiers, rankings, buckets, hashes, and controls are deterministic under row reorder;
- an independent read-back validation passes.

The Gate 0 data-support flag passes only when the preregistered usable-query counts above pass in
both closed galleries. The artifact can still be structurally valid when this support flag fails.
In that case, record a narrow or stop decision and do not start model experiments.

If no conjunction passes the fixed training rules, or validation and test contain too few usable
queries or independent components, record a narrow or stop decision. Do not lower the thresholds
in this version. A concentration warning in the v2 split remains a reporting limitation rather
than a hidden exclusion.

## Known limitations

- The states measure narrow Addgene metadata claims, not biological function.
- Positive-only facets cannot support a contradiction control by themselves.
- Canonical symbolic text does not test language understanding.
- Test support is reported after freezing and does not select query definitions.
- The open gallery contains training rows and measures collection-wide retrieval, not an unseen-
  family-only task.
