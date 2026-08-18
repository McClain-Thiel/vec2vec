# Gate 1 DNA Numerical Smoke Report

## Conclusion

Three of four neural DNA candidates passed the preregistered 32-plasmid numerical smoke check:
Carbon-500M, GENERanno prokaryote 500M, and GENERator-v2 prokaryote 1.2B. Each accepted run had
complete source-base coverage, no out-of-vocabulary token, finite embeddings, and a minimum
bfloat16-to-float32 cosine greater than 0.99999.

Carbon-3B did not complete. It exceeded both a 22.03 GiB NVIDIA L4 and a 44.39 GiB NVIDIA L40S
under the fixed full-length deterministic attention protocol. This is a technical memory failure,
not evidence about retrieval quality. Do not include Carbon-3B in the L4 invariance run. A future
retry needs a GPU with more than 48 GB of memory and separate compute approval.

This smoke check did not measure invariance or retrieval. It did not select an encoder. Validation
outcomes and test rows remained unread.

## Accepted results

All runs used the same 32 training plasmids from a frozen 512-plasmid panel. The panel contains
one row per primary similarity component and covers ten length strata. Smoke sequences range from
201 to 65,630 base pairs and contain 272,297 base pairs in total.

| Candidate | S3 output version | Minimum BF16/FP32 cosine | BF16 peak | FP32 peak | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| Carbon-500M | `2026-08-18T15.02.18.875Z` | 0.99999355 | 1.20 GiB | 11.48 GiB | Passed |
| GENERanno prokaryote 500M | `2026-08-18T15.15.56.524Z` | 0.99999595 | 1.17 GiB | 2.40 GiB | Passed |
| GENERator-v2 prokaryote 1.2B | `2026-08-18T15.19.31.423Z` | 0.99999383 | 2.69 GiB | 5.37 GiB | Passed |
| Carbon-3B | No accepted output | Unknown | More than 22.03 GiB | Not reached | Technical memory failure |

The accepted manifests are under this versioned prefix:

```text
s3://plasmidclip/kedro/08_reporting/e02/fixed_representation_smoke_manifest.json/
```

The persisted feature hashes are:

| Candidate | Feature SHA-256 |
| --- | --- |
| Carbon-500M | `3e290f8ce3c533735daa001527c241b831ab93d0a59d53a1f56b4472da04bbd1` |
| GENERanno prokaryote 500M | `9ea2701d32fa65036c0930f353907e4b82126e818be59eece24939b8a29412ec` |
| GENERator-v2 prokaryote 1.2B | `c6a35ff6da3b7a42a1b3c858f01f15028ff086078befaa0fe27cf840b53a82df` |

## Fixed inputs and runtimes

- Retrieval input version: `2026-08-04T09.02.10.007Z`.
- Split input version: `2026-08-17T23.49.47.355Z`.
- Input population SHA-256:
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Frozen panel SHA-256:
  `6dddbc33e0bb07ffcd3a2bebfcbf58f8c07573da976d0ef02e62c252e6e1593b`.
- Seed: `20260818`.
- Deterministic algorithms: enabled. TensorFloat-32: disabled.
- Carbon runtime: Python 3.13.9, Torch 2.11.0, CUDA 13.0, Transformers 5.12.1.
- GenerTeam runtime: Python 3.13.9, Torch 2.11.0, CUDA 13.0, Transformers 4.49.0.

The official GenerTeam repositories pin Transformers 4.49.0. An initial GENERanno load with
Transformers 5.12.1 failed before model construction. The experiment amendment records this
dependency boundary. The accepted GenerTeam runs used the pinned 4.49.0 runtime.

## Independent S3 read-back

A separate local read-back loaded the exact persisted features, coverage tables, diagnostics, and
manifests for all three accepted versions. It recomputed each BF16-to-FP32 cosine, checked finite
vectors, summed newly covered bases by sequence and precision, checked out-of-vocabulary counts,
and recomputed all three table hashes.

Observed read-back results:

- 32 diagnostic rows and 64 feature rows per candidate;
- zero coverage failures;
- zero out-of-vocabulary tokens;
- all recomputed cosines matched the persisted values within `1e-12`;
- all feature, coverage, and diagnostic hashes matched their manifests;
- all manifests state that validation outcomes and test rows were not read.

## Technical failures

The run history includes these failures. None produced an accepted model result.

1. Two attempts to start the existing `g6.4xlarge` failed because AWS had no capacity. AWS left
   the instance stopped.
2. Carbon-3B on the L4 used 14.73 GiB and then requested 14.27 GiB more. The 22.03 GiB device
   could not satisfy the request.
3. GENERanno failed to construct under Transformers 5.12.1 because its pinned remote code expects
   the Transformers 4.49 rope registry. The accepted rerun used the official dependency version.
4. One GENERator command used an unconfigured candidate identifier. The command-line validator
   rejected it before data or model execution. The corrected identifier produced the accepted run.
5. Three task-specific `g6e.xlarge` launch attempts failed for capacity in `us-east-1d`,
   `us-east-1a`, and `us-east-1c`. No instance was created for these requests.
6. Carbon-3B on the L40S used 43.38 GiB and then requested 3.57 GiB more. The 44.39 GiB device
   could not satisfy the request.

## Compute and cost

The accepted runs used one on-demand `g6.2xlarge` L4 host at $0.9776 per instance-hour. The host
ran from 14:57:52 UTC until the stop request at 15:28:01 UTC. The Carbon-3B retry used one
task-specific `g6e.xlarge` L40S host at $1.861 per instance-hour from 15:23:02 UTC until the
termination request at 15:28:00 UTC.

The launch-to-stop estimate is $0.49 for the L4 and $0.15 for the L40S, or $0.65 total. This is a
derived estimate, not an AWS invoice. It excludes storage and data transfer. The L4 host was
stopped. The task-specific L40S host and its ephemeral files were removed. Accepted outputs remain
in versioned S3 locations.

## Next action

Run the full 512-row rotation, reverse-complement, collapse, and throughput checks for the three
candidates that passed. Keep Carbon-500M as the incumbent. Do not run retrieval or select a model
until the invariance results and projected full-run costs are recorded.
