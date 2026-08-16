# E00 Split Similarity and Concentration Audit v0.1

## Conclusion first

- **The current `split_grouped` assignment fails the strict near-duplicate rule.** The global
  lower-bound search found 7,624 cross-split alignments at at least 99% identity, 95% coverage of
  both complete plasmids, and a 0.95 length ratio. The allowed count was zero.
- The saved graph is incomplete by design. Minimap2's secondary-alignment limit truncates many
  query result lists. The true primary and sensitivity edge counts can only be higher.
- The lower-bound primary graph already involves 4,310 plasmids and 1,459 current components. It
  joins current components into at least 357 cross-split augmented components.
- Test concentration is also poor. Its largest current component contains 3,364 of 11,508 rows
  (29.23%). Its ten largest components contain 44.84% of rows. Its row-weighted effective component
  count is only 11.12, despite 1,199 nominal components.
- Preserve the current split as provenance. Do not use it for model evaluation. Do not construct a
  v2 split directly from the lower-bound graph; a v2 needs complete similarity closure and a new
  cross-split audit.

## Status and decision

**Observed:** All 115,120 query sequences were searched in the three required cross-split
directions. The output contains a nonzero strict primary lower bound.

**Derived:** The strict decision is `fail_current_split_requires_v2`, because the protocol allowed
zero primary cross-split edges and one counterexample is sufficient to reject leak-free status.

**Unknown:** The complete number of near-duplicate edges and the complete augmented-component
sizes are not established. The saved edge graph cannot certify a repaired split.

## Fixed input and output

Input retrieval artifact:

```text
s3://plasmidclip/kedro/04_feature/retrieval_dataset.parquet/
2026-08-04T09.02.10.007Z/retrieval_dataset.parquet
```

Input population SHA-256:

```text
7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5
```

Accepted output version for all three artifacts:

```text
2026-08-06T16.02.42.779Z
```

```text
s3://plasmidclip/kedro/08_reporting/e00/
├── split_audit_edges.parquet/2026-08-06T16.02.42.779Z/
├── split_audit_component_profile.parquet/2026-08-06T16.02.42.779Z/
└── split_audit_manifest.json/2026-08-06T16.02.42.779Z/
```

The artifacts contain 333,686 lower-bound candidate edges, 14,157 current component rows, and the
complete resolved manifest. Their stored object sizes are 8,202,573 bytes, 64,661 bytes, and
16,054 bytes, respectively.

## Existing-split validation

The audit checked the complete sequence string, IUPAC alphabet, `length_bp`, sequence SHA-256,
population identity, and split purity before search.

| Existing grouping rule | Groups crossing a split |
| --- | ---: |
| Declared `family_key` | 0 |
| Exact `sequence_sha256` | 0 |
| Existing `leakage_component` | 0 |

The failure therefore comes from approximate whole-plasmid similarity that the declared family
and exact hash do not capture.

## Cross-split lower bounds

| Search pair | Candidate edges | Primary edges | Sensitivity edges |
| --- | ---: | ---: | ---: |
| Validation vs train | 104,879 | 4,626 | 7,198 |
| Test vs train | 115,240 | 2,272 | 5,049 |
| Test vs validation | 113,567 | 726 | 1,504 |
| **Total** | **333,686** | **7,624** | **13,751** |

The primary edges include 795 reverse-complement alignments. Primary identity ranges from exactly
0.9900 to 1.0000, with median 0.9981. An identity of 1.0 does not imply an exact stored sequence:
circular origin changes and small terminal or length differences can preserve complete aligned
identity while changing the raw sequence hash.

The primary lower bound involves 4,310 plasmids and 1,459 current components. Unioning only these
observed edges produces at least 357 augmented components that cross the original splits. The
largest contains 12,483 rows. The sensitivity lower bound involves 8,354 plasmids and 2,318 current
components; its largest observed augmented component contains 30,341 rows.

These augmented values are diagnostic lower bounds. They are not a valid v2 component assignment.

## Concentration

| Split | Rows | Components | Effective components | Largest component | Ten largest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 92,097 | 11,456 | 289.35 | 2,607 (2.83%) | 14,042 (15.25%) |
| Validation | 11,515 | 1,502 | 32.52 | 1,869 (16.23%) | 3,517 (30.54%) |
| Test | 11,508 | 1,199 | 11.12 | 3,364 (29.23%) | 5,160 (44.84%) |

Effective components equal `1 / sum(weight_i ** 2)`, where each weight is a component's fraction
of rows in its split. This is a concentration description, not a performance estimate. The test
split crosses the preregistered 25% largest-component trigger. Any later evaluation must report
component-macro estimates and grouped resampling, even after a v2 split is built.

## Method and validation

Minimap2 2.31-r1302 used the `asm20` assembly preset, 10 threads, both strands, and a query formed by
repeating each circular plasmid twice. Classification used the undoubled length. The fixed primary
rule was at least 99% aligned identity, at least 95% coverage of each plasmid, and at least 0.95
shorter-to-longer length ratio. The sensitivity rule used 95%, 90%, 90%, and 0.90.

The synthetic tool check detected an exact copy, a circular rotation, a reverse complement, and 39
substitutions in 4,000 bases. A pair sharing only a 1,000-base cassette did not pass whole-plasmid
coverage.

The saved run searched every query globally but retained at most 10 secondary alignments per query.
It used a secondary-to-primary score ratio of 0.5. Between 10,512 and 11,051 queries per search pair
had enough returned alignments to be potentially truncated. Thus all edge counts are lower bounds.
This limitation cannot invalidate the failure decision because observed qualifying alignments are
direct counterexamples.

## Technical negative results

- Direct BLAST was stopped after approximately 90 minutes without completing the first full pair.
- A query-threaded BLAST benchmark was stopped after more than three minutes without completing
  100 queries against the full train database.
- Mash k=11 was rejected after its random-match warning for the longest sequences.
- Mash k=12 and k=21 were rejected because their candidate sets remained too broad after measured
  distance and length filters.
- Two secondary LSH designs were rejected after explicit candidate ceilings or bucket-size checks.

No stopped or rejected method produced a split decision.

## Provenance and limitations

The accepted run used Git commit `c3b26aee1cb90ec943c50385d987ae34abafe9dc` with a dirty
worktree. The manifest records the changed paths, resolved configuration, runtime, tool logs,
synthetic result, input identity, search durations, and saved decision. No model outcome was read.

Important limitations:

- Minimap2 is heuristic and the secondary limit makes the graph incomplete.
- The result establishes that leakage exists. It does not enumerate every leak.
- Sequence similarity does not establish functional or biological independence.
- The current code and report changes are local and uncommitted. The versioned S3 evidence is
  backed up, but the source changes need a Git commit and push for equivalent remote protection.

## Next action

Build a separately named `split_grouped_v2` from a complete global similarity-closure method. Keep
the current split and all audit artifacts unchanged. Re-run exact, family, approximate-similarity,
and concentration checks on v2 before freezing queries or starting model experiments.
