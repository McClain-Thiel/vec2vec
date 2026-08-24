# Gate 0 Completion

## Conclusion first

- **Gate 0 is complete and accepted.** The graph, strict grouped split, and frozen query
  benchmark exist as versioned artifacts and pass independent S3 read-back validation.
- **The study can start Gate 1.** Gate 1 selects frozen DNA and text representations. It does not
  train the study models or read the test benchmark.
- **The data is not biological ground truth.** The benchmark measures consistency with recorded
  Addgene metadata. Missing evidence remains unknown.
- **The main remaining data risk is training concentration.** The largest training component has
  37,486 rows, or 40.62% of the training split. Training and uncertainty estimates must control
  component weighting.

## Accepted inputs and outputs

| Product | Accepted version | Observed result |
| --- | --- | --- |
| Retrieval dataset | `2026-08-04T09.02.10.007Z` | 115,120 plasmids |
| Constraint-state contract | `2026-08-06T13.27.47.937Z` | 35 constraints; 684,987 explicit states |
| Global similarity graph | `2026-08-17T22.59.04.326Z` | 4,450,238 primary; 4,676,653 sensitivity edges |
| `split_grouped_v2` | `2026-08-17T23.49.47.355Z` | 92,279 / 11,344 / 11,497 train/val/test |
| Query benchmark v0.1 | `2026-08-17T23.51.35.629Z` | 131 semantic queries; 5,740,247 sparse states |

The graph covers all 115,120 plasmids. It contains no saturated exact cap-10,000 query. The
accepted run used 164.98 CPU-hours, below the fixed 500-CPU-hour limit.

The strict split contains 11,764 primary similarity components. No declared family, exact
sequence group, old leakage component, primary similarity component, or primary graph edge
crosses the split. The separate 95%-similarity sensitivity graph has 6,259 cross-split edges.
This count does not invalidate the preregistered 99%-identity split rule. It remains a required
sensitivity analysis.

## Versioned artifact locations

The catalog resolves these data products. The version suffixes below identify the accepted
artifacts.

```text
s3://plasmidclip/kedro/08_reporting/e00/
├── similarity_graph_edges.parquet/2026-08-17T22.59.04.326Z/
├── similarity_graph_nodes.parquet/2026-08-17T22.59.04.326Z/
├── similarity_graph_components.parquet/2026-08-17T22.59.04.326Z/
└── similarity_graph_manifest.json/2026-08-17T22.59.04.326Z/

s3://plasmidclip/kedro/04_feature/e00/
└── split_grouped_v2.parquet/2026-08-17T23.49.47.355Z/

s3://plasmidclip/kedro/05_model_input/e00/
├── query_catalog.parquet/2026-08-17T23.51.35.629Z/
├── candidate_galleries.parquet/2026-08-17T23.51.35.629Z/
├── query_candidate_state.parquet/2026-08-17T23.51.35.629Z/
└── candidate_base_mass.parquet/2026-08-17T23.51.35.629Z/
```

## Independent validation

The validators reloaded the saved products before acceptance. They did not rely on in-memory
pipeline outputs.

The graph validator confirmed population identity, edge rules, component assignments, cap
coverage, non-saturation, content hashes, and manifest identity. The split validator confirmed
115,120 unique sequence identifiers, exact input-set equality, deterministic assignments, and
zero strict crossings.

The query validator recomputed every atomic and conjunction answer set from the frozen source
states. It confirmed disjoint verified and contradicted sets, normalized base measures, exact set
identities, deterministic controls, and both oracle rules.

The preregistered support gate passed:

| Closed split | Usable atomic | Usable pair | Pair with contradiction control |
| --- | ---: | ---: | ---: |
| Validation | 28 | 80 | 80 |
| Test | 32 | 90 | 90 |

The required floors were 10 atomic queries, 20 pair queries, and 20 pair contradiction controls
in each closed split.

## Data coverage for encoder selection

The source sequences range from 201 to 98,922 base pairs (bp). The observed cumulative coverage
is:

| Maximum length | Population covered |
| ---: | ---: |
| 4,096 bp | 11.9580% |
| 8,192 bp | 66.2474% |
| 16,384 bp | 98.9698% |
| 32,768 bp | 99.9105% |
| 65,536 bp | 99.9983% |

An encoder cannot silently truncate the remaining sequence. Gate 1 must record tokenization,
window coverage, circular wrapping, pooled-token counts, and the fraction of bases represented.

## Claims and limitations

The accepted benchmark supports a narrow claim: a ranked plasmid is verified when every query
constraint has positive recorded evidence. It is contradicted when at least one reviewed conflict
rule applies. Every other candidate is unknown.

This evidence does not establish biological function, host compatibility, or experimental
success. Only copy class and the 30 °C versus 37 °C propagation-temperature pair have reviewed
contradiction rules. Positive-only facets cannot support a contradiction control by themselves.

The strict graph is a measured minimap2 result, not a mathematical all-pairs proof. Single-linkage
components can connect dissimilar endpoints through chains. Component-macro reporting and
whole-component bootstrap resampling remain mandatory because the training split is concentrated.

## Next action

Run the validation-only [fixed-representation bake-off](../experiments/E02_fixed_representation_bakeoff.md).
Use the [PlasmidCLIP prior and current model review](13_encoder_prior_and_candidates.md) as the
candidate rationale. Do not read test outcomes or start paid GPU work until the fixed protocol and
compute budget receive approval.
