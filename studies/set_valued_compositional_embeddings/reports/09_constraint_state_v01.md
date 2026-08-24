# Plasmid-Constraint State v0.1

## Conclusion

The frozen exact-rule contract now produces a stable constraint vocabulary and a sparse
plasmid-constraint state table over the complete retrieval population. The build contains 35
constraints and 684,987 unique states. No plasmid-constraint pair has both verified and
contradicted evidence.

This completes the first Gate 0 data product. It does not complete Gate 0 because the split audit,
frozen queries, candidate galleries, and measurement controls remain outstanding.

## Fixed identities

| Item | Identity |
| --- | --- |
| Retrieval input | `2026-08-04T09.02.10.007Z` |
| State protocol | `e00-plasmid-constraint-state-v0.1` |
| Accepted positive-rule contract | `aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2` |
| Output version | `2026-08-06T13.27.47.937Z` |
| Input population SHA-256 | `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5` |
| State-source input SHA-256 | `65feac686141b7c9a22179324a9c84a87e28e14f2d044aa56f6b92f147c2d376` |
| Code state | Dirty worktree based on `c3b26ae` |

The run loaded the complete fixed retrieval artifact and applied rules accepted before test-state
construction. Test metadata did not select mappings or conflict rules. The pipeline did not use
generated descriptions, annotations, plasmidkit, or model decisions.

## State policy

- An enabled exact mapping creates a `verified` state.
- Only an explicit reviewed conflict rule creates a `contradicted` state.
- Absence from the sparse table means `unknown`.
- Constraint IDs are SHA-256 identities of facet, relation, canonical value, and versioned rule ID.
- Every state preserves its exact source field, raw value, observed constraint, and evidence rule.

The reviewed conflict groups are:

1. Addgene copy class: `high` versus `low`.
2. Addgene reported propagation temperature: `30_c` versus `37_c`.

Bacterial selection, expression context, and intended-use categories remain positive-only.

## Results

| Result | Count |
| --- | ---: |
| Constraints | 35 |
| Constraints with reviewed conflict rules | 4 |
| Verified states | 472,765 |
| Contradicted states | 212,222 |
| Total states | 684,987 |
| Sequences with verified evidence | 115,120 |
| Sequences with contradicted evidence | 115,089 |
| Pair-state conflicts | 0 |

| Split | Verified | Contradicted |
| --- | ---: | ---: |
| Train | 375,816 | 169,221 |
| Validation | 46,317 | 21,108 |
| Test | 50,632 | 21,893 |

The prior training artifact contained 375,819 positive evidence rows. The new state table contains
375,816 unique training plasmid-constraint pairs. Three training pairs had two case-variant source
values for the same constraint and are correctly consolidated. One validation pair is also
consolidated. The source variants remain preserved inside each state's evidence list.

## Primary conflict support in training data

| Constraint | Verified rows | Verified components | Contradicted rows | Contradicted components |
| --- | ---: | ---: | ---: | ---: |
| Copy class `high` | 68,043 | 7,875 | 9,160 | 1,703 |
| Copy class `low` | 9,160 | 1,703 | 68,043 | 7,875 |
| Temperature `30_c` | 5,760 | 1,060 | 86,258 | 10,729 |
| Temperature `37_c` | 86,258 | 10,729 | 5,760 | 1,060 |

These four constraints have nontrivial verified and contradicted support. They are candidates for
the first atomic benchmark, subject to the split audit and frozen eligibility rules.

## Validation

- Production pipeline completed successfully in 56.0 seconds with the retrieval input version,
  population SHA-256, and constraint-source content SHA-256 enforced before state construction.
- The saved vocabulary has 35 rows and 29 columns.
- The saved state table has 684,987 rows and unique `state_id` values.
- The saved table has zero plasmid-constraint pairs with multiple states.
- The maximum evidence count for one state is two; four states contain two preserved case variants.
- `pytest -q`: 124 passed.
- `ruff check .`: passed.
- `ruff format --check .`: 79 files formatted.
- `git diff --check`: passed.

Earlier successful outputs at `2026-08-06T13.19.13.181Z` and
`2026-08-06T13.24.44.354Z` have identical reported counts. The first did not enforce the expected
population hash. The second enforced row identity and split membership but did not yet hash the
four constraint source fields. Preserve both as provisional artifacts. Use
`2026-08-06T13.27.47.937Z` for later work.

## Limits and next step

The state means that Addgene metadata supports or conflicts with a narrow recorded claim. It does
not establish plasmid function. Positive-only facets do not yet supply valid atomic negatives.

The next Gate 0 task is the global approximate-sequence and split-concentration audit. Query
construction must use training support only and must not start until the split decision is frozen.
