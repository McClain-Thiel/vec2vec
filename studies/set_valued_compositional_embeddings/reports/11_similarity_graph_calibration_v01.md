# E00 Similarity Graph Calibration v0.1

## Conclusion first

- **Proceed with a bounded adaptive full run.** The calibrated CPU and raw-output projections are
  below 500 CPU-hours and 250 GB.
- A cap of 1,000 is not complete. It saturated 67 of 1,024 candidate queries and one of 32 exact
  queries. A cap of 10,000 saturated none of the 64 candidate stress-tail queries.
- Use candidate cap 1,000 to route queries. Run exact cap 1,000 for ordinary queries and exact cap
  10,000 for saturated queries. Fail if any exact cap-10,000 query remains saturated.
- Base-level alignment is necessary for reported identity. Approximate PAF output is candidate
  evidence only.

## Fixed inputs and output

- Retrieval dataset: `2026-08-04T09.02.10.007Z`.
- Split-audit edges: `2026-08-06T16.02.42.779Z`.
- Population SHA-256:
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Accepted calibration output version: `2026-08-10T09.34.59.159Z`.

```text
s3://plasmidclip/kedro/08_reporting/e00/
├── similarity_calibration_runs.parquet/2026-08-10T09.34.59.159Z/
├── similarity_calibration_query_profile.parquet/2026-08-10T09.34.59.159Z/
├── similarity_calibration_exact_edges.parquet/2026-08-10T09.34.59.159Z/
└── similarity_calibration_manifest.json/2026-08-10T09.34.59.159Z/
```

## Observed calibration

| Mode | Cap | Queries | CPU seconds | PAF MB | Saturated queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate | 10 | 1,024 | 72.44 | 1.40 | 824 |
| Candidate | 100 | 1,024 | 68.56 | 6.81 | 377 |
| Candidate | 1,000 | 1,024 | 73.37 | 21.34 | 67 |
| Candidate stress tail | 10,000 | 64 | 2.85 | 21.31 | 0 |
| Exact | 10 | 32 | 5.17 | 0.05 | 17 |
| Exact | 1,000 | 32 | 33.53 | 0.48 | 1 |

The exact cap-10 output contained 28 primary and 50 sensitivity edges. The exact cap-1,000 output
contained 496 primary and 522 sensitivity edges. Every cap-10 edge was also present at cap 1,000;
472 additional sensitivity edges appeared at the larger cap.

## Derived full-population projections

| Mode | Cap | Projected CPU-hours | Projected raw PAF |
| --- | ---: | ---: | ---: |
| Candidate | 1,000 | 2.29 | 2.40 GB |
| Candidate stress-tail extrapolation | 10,000 | 1.42 | 38.34 GB |
| Exact | 1,000 | 33.51 | 1.73 GB |

The stress-tail byte projection treats every plasmid as a dense tail query and is deliberately
conservative. The planned full run applies cap 10,000 only to queries flagged by the candidate
stage.

## Validation and technical failures

The synthetic gate detected an exact copy, circular rotation, reverse complement, and 39
substitutions in 4,000 bases. A shared 1,000-base cassette failed both whole-plasmid stages.

Two package installation commands failed because the existing virtual environment has neither a
`pip` executable nor the `pip` module. They changed no environment state. Ray 2.55.1 was then
installed successfully with `uv` against the explicit project interpreter.

The first executable synthetic gate failed because the CIGAR assertion was wired to the candidate
parser. No real data was searched. The assertion was moved to the exact parser, focused tests
passed, and the executable gate then passed.

An earlier optional CIGAR timing diagnostic was stopped after more than one minute. It produced no
accepted result. It motivated the preregistered separation between fast candidate search and
base-level exact measurement.

## Limitations

- The 1,024-query sample can miss a denser family.
- Minimap2 is a heuristic candidate generator, even when no result cap is reached.
- The exact benchmark intentionally overrepresents stress queries.
- The graph will measure sequence similarity, not functional independence.
