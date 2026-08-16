# E00 Global Plasmid Similarity Graph

**Status:** active  
**Protocol version:** `global_similarity_graph_v0.1`  
**Fixed before the full run:** 2026-08-10 Europe/London

## Question and hypothesis

Can an adaptive, global whole-plasmid search produce a non-truncated 99% and 95% similarity graph
for all 115,120 pinned plasmids within 500 CPU-hours and 250 GB?

The hypothesis is that candidate cap 1,000 identifies every query that needs exact cap 10,000 and
that no exact cap-10,000 query remains saturated.

## Fixed evidence and input

- Retrieval dataset: `2026-08-04T09.02.10.007Z`.
- Population SHA-256:
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Accepted calibration: `2026-08-10T09.34.59.159Z`.
- Calibration report: `reports/11_similarity_graph_calibration_v01.md`.
- No model outcome is an input.

## Search and controlled stages

1. Search every doubled circular query against the complete undoubled target index without
   base-level alignment at cap 1,000.
2. Re-search candidate-saturated queries at cap 10,000. Fail if any remains saturated.
3. Run base-level exact alignment at cap 1,000 for ordinary queries.
4. Run base-level exact alignment at cap 10,000 for candidate-saturated queries.
5. If an ordinary exact query unexpectedly saturates, re-run it at cap 10,000 and replace its
   lower-cap evidence. Fail if any replacement remains saturated.

All stages use minimap2 2.31 `asm20`, both strands, doubled queries, and a 0.5 minimum
secondary-to-primary score ratio. Candidate divergence is approximate and cannot classify an
edge.

## Edge definitions

Primary edges require at least 99% base-aligned identity, 95% coverage of both complete plasmids,
and a 0.95 shorter-to-longer length ratio. Sensitivity edges require 95%, 90%, 90%, and 0.90.

Store one canonical undirected edge per qualifying pair. Preserve identity, coverage of each
endpoint, orientation, alignment size, direction count, and exact-search cap. Do not materialize
transitive-closure pairs.

## Components and outputs

Union the existing exact-sequence and declared-family leakage components with primary edges to
form stable primary components. Build a separate sensitivity component assignment. Save:

- canonical sensitivity edges with primary flags;
- one node row per plasmid with both component identifiers;
- component summaries;
- per-query cap and saturation evidence;
- a resolved run manifest.

The graph does not assign `split_grouped_v2`. Split assignment is a separate planned operation
after graph validation.

## Validation and stopping rules

Stop on any schema, input hash, external-tool, timeout, output-size, disk, or high-cap saturation
failure. Stop if projected or observed use exceeds 500 CPU-hours, 250 GB persisted output, 12
hours wall time, or the configured free-disk floor. Do not weaken a threshold or cap after seeing
the result.

The graph is accepted only if all 115,120 queries have one final exact-search record, no final
query is saturated, all canonical edges satisfy the sensitivity rule, primary edges are a subset,
endpoints are valid, and current leakage components remain whole inside the new components.

## Known limitations

- No result-cap saturation does not turn minimap2 into a mathematical all-pairs proof.
- Complex rearrangements can require several local segments and can be missed by the single best
  alignment rule.
- A 95% single-linkage component can be biologically broad because similarity is not transitive.
- Publication needs a separate license, provenance, and dataset-card review.

## 2026-08-10 execution amendment before the full result

The restricted AWS operator could describe existing instances but could not inspect the required
network and IAM controls. A non-mutating EC2 launch dry run then failed because the operator lacks
`iam:PassRole` for the existing S3 role. It created no resource. The default AWS profile resolves
to the account root identity and will not be used to bypass this control.

Run the full graph on the local research host with Ray 2.55.1, two workers, and four minimap2
threads per worker. Reuse the content-addressed target cache validated by the calibration. Keep
all search thresholds, caps, tool versions, 12-hour wall limit, 500-CPU-hour limit, and 250-GB
byte limit unchanged. The local free-disk floor is 40 GB. This amendment changes execution
capacity only and was recorded before the full graph result.

## 2026-08-10 failed-run record and retry amendment

The first local full run started at 10:51:30 BST and failed at 17:58:58 BST. Candidate cap
1,000 completed for the population, candidate cap 10,000 completed for the dense subset, and
exact cap 1,000 completed for the ordinary subset. In the exact cap-10,000 stage, shard 0
completed near the 30-minute boundary and shard 1 reached the fixed 1,800-second task limit.
The pipeline raised `exact cap 10000 shard 1 reached its time limit`. It saved no final graph
artifact. This is a technical execution failure, not a negative similarity result.

Before the retry, change only the execution plan:

- use 32 queries per adaptive shard instead of 128;
- use eight one-thread Ray workers instead of two four-thread workers on the same 10-core host;
- write a content-identified, hash-validated checkpoint for each parsed shard before returning
  its result to the Ray driver;
- reuse a checkpoint only when its query FASTA, target-index identity, search settings, parser
  inputs, and similarity rules match exactly.

Keep the 1,800-second per-shard timeout, 12-hour run limit, 40 GB free-disk floor, caps,
thresholds, dataset, minimap2 version, and graph acceptance rules unchanged. Thread count can
change alignment output order, so canonical output sorting and content hashes remain required.
The checkpoint change preserves failed-run evidence and permits an exact restart after a later
technical failure. This amendment was recorded before any retry result was inspected.

## 2026-08-10 interrupted retry and exact resume

The checkpointed retry started at 18:06:48 BST. It completed all 900 candidate cap-1,000 shards,
all 152 candidate cap-10,000 shards, and 170 of 862 ordinary exact cap-1,000 shards. At 21:34:40
BST, the Ray driver called its intended shutdown path while 674 exact shards were still waiting and
eight were active. Ray recorded no failed task, object error, or worker creation exception. The
captured Kedro terminal session was no longer available, so the initiating exception or external
cause is unknown. No final graph artifact was saved or accepted.

Resume the same protocol without a scientific or execution-parameter change. Start it as a
detached local job so a desktop terminal-session loss cannot end the driver. Reuse a completed
shard only after the existing checkpoint identity, file SHA-256, and row-count checks pass. The
12-hour wall limit applies to the resumed process; completed checkpoint reuse is recorded in its
run table. Preserve the interrupted checkpoints and Ray logs as failed-run evidence.

## 2026-08-12 failed exact resume, backup, and unfinished-shard amendment

The detached resume started at 22:10:11 BST on 2026-08-10 and failed at 22:41:16 BST. It reused
the earlier candidate and exact checkpoints, completed 51 additional ordinary exact shards, and
then raised `exact cap 1000 shard 141 reached its time limit`. The failed process saved no final
graph artifact. This is a technical execution failure. It does not change a similarity result.

Before another run, validate every checkpoint identity, file SHA-256, and persisted row count.
Back up the validated checkpoints, partial raw output, Kedro and Ray failure logs, current code,
local execution configuration, and target cache under the unique S3 prefix
`s3://plasmidclip/research-backups/vec2vec/e00/global_similarity_graph_v0.1/2026-08-12T12-26-40Z/`.
The pinned retrieval and accepted calibration remain in their existing versioned S3 locations.

Change only unfinished exact-shard scheduling. Keep a completed 128-query exact shard under its
existing identity. Deterministically divide each unfinished exact shard into parts of at most 32
queries. Give each part a stable execution identifier and preserve its exact FASTA bytes. Keep all
search caps, thresholds, input identities, worker resources, 1,800-second task limit, 12-hour run
limit, 500-CPU-hour limit, 250-GB persisted-output limit, and 40-GB free-disk floor unchanged.
This amendment was recorded before the next result was inspected.

## 2026-08-13 dense exact failure and isolated-query retry amendment

The amended run started at 13:43:19 BST on 2026-08-12. It completed exact cap 1,000 for all
110,276 ordinary queries and exact cap 10,000 for 1,472 of 4,844 routed dense queries. At
17:16:42 BST, unfinished exact cap-10,000 shard 34 reached the fixed 1,800-second task limit.
The pipeline stopped without writing a final graph artifact. This is a technical execution
failure, not a negative similarity result.

Before retry, validate every retained checkpoint and create a new self-contained S3 backup that
includes the target FASTA and minimap2 index, deterministic query files, all checkpoint files,
partial raw output, failure logs, local execution configuration, current research files, and Git
provenance. Correct the earlier backup record: the 2026-08-12 prefix contains the target FASTA but
not a completed target-index object, so it is evidence backup rather than a standalone restore.

Preserve all 46 completed 32-query dense exact shards under their existing identities. Divide
only each unfinished dense exact shard into deterministic one-query units. This isolates any slow
query and checkpoints every completed query. Keep the pinned dataset, target index, search preset,
caps, similarity thresholds, parser, output schema, worker count, threads, 1,800-second task
limit, 12-hour run limit, 500-CPU-hour limit, 250-GB output limit, and 40-GB free-disk floor
unchanged. This amendment was recorded before the retry result was inspected.

The required backup completed before the retry at
`s3://plasmidclip/research-backups/vec2vec/e00/global_similarity_graph_v0.1/2026-08-13T15-45-08Z/`.
It contains seven AES-256-encrypted objects totaling 1,233,658,681 bytes. The restore manifest
records local SHA-256 values, target-cache byte sizes and nanosecond timestamps, checkpoint
validation counts, archive contents, and the restore command. The S3 object length matches the
1,228,867,644-byte local resume archive, and AWS recorded a SHA-256 transport checksum.

## 2026-08-13 free-disk stop and guarded-resume amendment

The isolated-query retry started at 16:59:11 BST. It retained the 46 completed dense parent
shards and completed 930 additional one-query checkpoints. The retained dense exact evidence now
covers 2,402 of 4,844 routed queries. No query reached the 1,800-second task limit, and no search,
parser, checkpoint, or checksum error was observed.

During the run, unrelated active Docker workloads on the same host consumed disk space. The
available space fell below the fixed 40,000,000,000-byte floor and continued down to approximately
24.6 GB. Stop the run as required by the fixed safety rule. The driver did not react because its
disk check ran only between major search stages. A terminal interrupt and direct `SIGINT` did not
stop the Ray wait, so terminate the exact Kedro process group with `SIGTERM`. This discarded only
eight unfinished one-query tasks. It did not remove a completed checkpoint. No final graph table
was written or accepted.

The final checkpoint sync completed at 20:28:04 UTC. The local and S3 prefixes each contain 976
dense exact checkpoint manifests: 46 completed parent shards and 930 isolated queries. The
standalone backup also now includes a verified 334,407,789-byte Zstandard archive of the earlier
split-audit scratch directory and a per-file SHA-256 manifest. The archive contains all 26 source
files and has SHA-256
`f928f50dd0ee1cf08cf87183283f8c111b1b2f2de3f9fb4b76425a7ba120a4cd`.

Before another retry, change only the execution guard. Poll free disk while Ray tasks are pending,
cancel unfinished tasks immediately if available space crosses the existing 40-GB floor, and
retain completed content-validated checkpoints. Do not start the next local Ray process until the
host has at least 60 GB free for ten consecutive minutes. Keep the dataset, target index, search
preset, thresholds, caps, parser, worker count, threads, task timeout, wall limit, CPU limit,
output limit, and 40-GB running floor unchanged. This amendment was recorded before the next retry
result was inspected.

## 2026-08-13 technical-retry supervisor amendment

The 12-hour wall limit applies to each resumed process. A process can therefore save additional
valid query checkpoints and then stop at the wall limit before it writes final graph tables. Rearm
the identical command only after either of these exact technical failures:

- `global graph reached its fixed wall-time limit`;
- free disk is below the fixed calibration minimum.

Do not rearm after a search, timeout, parser, checksum, manifest, identity, join, graph-validation,
or other failure. Before each technical retry, sync completed checkpoints to the existing encrypted
backup prefix. Apply the same 60-GB, ten-minute start gate and the same runtime disk guard. Keep a
separate immutable log for each attempt. Permit at most four technical retries after the active
attempt. Stop visibly after that limit. This amendment changes orchestration only; it does not
change a scientific or search setting.
