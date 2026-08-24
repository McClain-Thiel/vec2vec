# Rule-Derived Constraint Evidence v0.1

## Conclusion

The enabled exact rules produced enough labels for an initial training signal without claim-by-claim
review. The output contains 375,819 training claims. A separate 240-application validation sample is
ready for an accuracy benchmark. It contains no benchmark decisions or test rows.

## Fixed artifacts

| Item | Version |
| --- | --- |
| Retrieval input | `2026-08-04T09.02.10.007Z` |
| Evidence protocol | `e00-constraint-evidence-v0.1` |
| Benchmark sample protocol | `e00-constraint-benchmark-sample-v0.1` |
| Output | `2026-08-06T08.44.42.865Z` |
| Production implementation | Git commit `7e66cf4` |
| Rule contract hash | `aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2` |

The run used a dirty worktree based on `fa02968`. The production implementation and configuration
did not change between the run and commit `7e66cf4`; later edits were tests and documentation.

## Training evidence

| Facet | Claims | Canonical values |
| --- | ---: | ---: |
| Copy class | 77,203 | 2 |
| Expression context | 69,244 | 6 |
| Growth temperature | 92,018 | 2 |
| Use category | 43,681 | 11 |
| Bacterial selection | 93,673 | 14 |
| **Total** | **375,819** | — |

The claims cover 92,097 training sequences and 11,456 leakage components. Evidence identifiers are
unique. The output contains no `room_temperature` claim and no DAP canonical value.

## Mapping coverage

| Source field | Mapped units | Unlabeled units | Mapping coverage |
| --- | ---: | ---: | ---: |
| `plasmid_copy` | 86,812 | 16,800 | 83.79% |
| `growth_temp` | 103,517 | 95 | 99.91% |
| `bacterial_resistance` | 103,573 | 39 | 99.96% |
| `vector_types` | 126,379 | 11,955 | 91.36% |

The 95 unlabeled growth values are stored `23` values. The 16,800 unlabeled copy values are
`Unknown`. The vector-type remainder contains 1,351 distinct values, mainly free text and categories
outside version 1. These values remain unlabeled rather than being forced into a class.

## Validation sample

| Facet | Applications |
| --- | ---: |
| Copy class | 47 |
| Expression context | 54 |
| Growth temperature | 61 |
| Use category | 30 |
| Bacterial selection | 48 |
| **Total** | **240** |

The sample contains 237 sequences and 178 leakage components. It has at most one application per
facet and leakage component. All selected sequences have pLannotate evidence. The sample includes
no generated description, plasmidkit fallback, model decision, or benchmark label.

The sample is representative of enabled mapping applications, so rare reviewed mappings do not
appear. The earlier targeted model gate tested those mappings directly. Do not expand this sample
only to repeat that review.

## Next measurement

Prepare one fixed structured judge packet per sampled mapping application. Run a small diagnostic
before the complete paid benchmark. The prior targeted run cost USD 0.0245 per application on
average. A direct 240-application extrapolation is USD 5.88, but the new packets differ in size. Set
the actual cost cap only after inspecting the packets and diagnostic cost.
