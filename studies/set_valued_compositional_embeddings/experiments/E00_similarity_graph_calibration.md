# E00 Global Similarity Graph Calibration

**Status:** complete; accepted
**Protocol version:** `similarity_graph_calibration_v0.1`  
**Fixed before the calibration run:** 2026-08-10 Europe/London

## Question

What compute, memory, temporary storage, and candidate density should we expect when constructing
a global plasmid similarity graph for split repair and later graph analysis?

## Hypothesis

An indexed two-stage search can build the graph within 500 CPU-hours and 250 GB of persisted
artifacts. Fast approximate mapping will produce many local candidates, but exact whole-plasmid
confirmation will retain a much smaller set.

## Baselines and controlled comparisons

The primary comparison changes only the minimap2 secondary-alignment cap: 10, 100, and 1,000.
An adaptive cap of 10,000 is applied only to at most 64 queries that remain saturated at 1,000.
All candidate runs use the same sequences, minimap2 preset, query repetition, strands, target
index, worker count, and filters.

The exact-cost comparison uses a fixed 32-query subset at caps 10 and 1,000. These runs request
base-level alignment. Candidate runs do not request base-level alignment and must not be used as
measured identity evidence.

## Fixed inputs

- Retrieval dataset: `2026-08-04T09.02.10.007Z`.
- Population SHA-256:
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Lower-bound split-audit edges: `2026-08-06T16.02.42.779Z`.
- No model outcome is an input.

## Deterministic query sample

The calibration contains 1,024 unique queries:

- 512 queries sampled across ten sequence-length strata;
- up to 256 queries sampled across the largest existing leakage components, with at most four
  queries from one component;
- up to 256 queries with the highest degree in the saved lower-bound candidate graph;
- deterministic hash-ordered fill queries if overlap makes the sample smaller than 1,024.

The 32-query exact subset balances representative and stress queries. The adaptive tail subset is
selected only from queries saturated at cap 1,000. Selection uses seed `20260810` and does not use
model results.

## Metrics

Primary calibration metrics are:

- wall seconds and child-process CPU seconds per query;
- raw alignments and unique targets per query;
- queries saturated at each cap;
- candidates remaining after approximate coverage, length, and divergence filters;
- temporary PAF bytes per query;
- projected full-population CPU-hours and bytes.

The exact benchmark also reports primary and sensitivity whole-plasmid edges using the thresholds
fixed in `E00_split_similarity_audit.md`.

## Execution and stopping rules

Run local Ray 2.55.1 with two workers and four minimap2 threads per worker. Each normal shard has
at most 128 queries. Stop and record failure if any of these limits is reached:

- 1,800 seconds for one task;
- 2 GB of PAF output for one task;
- 7,200 seconds for the complete calibration;
- less than 40 GB of free local disk.

Do not start the full graph job if the calibrated projection exceeds 500 CPU-hours or 250 GB.
Create a new protocol amendment before changing these limits.

## Interpretation rule

Proceed to a bounded full run only if:

1. candidate and exact stages pass their synthetic controls;
2. cap convergence and saturation are measured rather than assumed;
3. the full-run projection stays within both fixed limits; and
4. the chosen full-run method cannot silently truncate saturated queries.

The calibration cannot certify a leak-free split and cannot establish complete edge enumeration.
It selects an execution design only.

## Known limitations

- Minimap2 candidate generation is heuristic.
- Approximate PAF coordinates and divergence are filters, not exact identity measurements.
- A 1,024-query sample can miss an unusually dense family.
- A complete edge list can be much larger than the component structure needed for split repair.
- Publication of a graph dataset needs a separate license, provenance, and dataset-card review.

## Outcome observed after execution

The accepted calibration version is `2026-08-10T09.34.59.159Z`. Candidate cap 1,000 saturated 67
of 1,024 queries. Candidate cap 10,000 saturated none of the 64 fixed stress-tail queries. Exact
cap 1,000 saturated one of 32 queries. These results selected the adaptive full-run design: route
with candidate cap 1,000, then run exact cap 1,000 for ordinary queries and cap 10,000 for routed
dense queries.

The projected full-run cost remained below both fixed limits. The complete result and limitations
are in [similarity graph calibration v0.1](../reports/11_similarity_graph_calibration_v01.md).
