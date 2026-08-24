# Gate 1 E02b Benchmark Status

## Conclusion first

- E02b feature extraction is complete. No DNA-and-text pair has been selected.
- Seven representation products passed independent persisted-artifact read-back: TF-IDF/SVD,
  Carbon-500M, GENERanno, GENERator-v2, BGE-base, GTE-ModernBERT, and Qwen3-Embedding.
- GENERanno version `2026-08-23T16.23.06.763Z` is a rejected partial artifact. It contains the
  feature table but lacks the required coverage and manifest products.
- GENERanno retry version `2026-08-24T11.03.45.874Z` is complete and accepted. It did not reuse
  the rejected feature table.
- The complete 4-by-3-by-3 alignment factorial did not start. No validation ranking or old-test
  model outcome was read.
- The alignment factorial is authorized and ready to run from the seven accepted feature
  products.

## Question and scope

E02b asks which frozen DNA and text representation pair gives the best validation query-macro
`utility@10` under the fixed many-positive alignment probe. The approved reduced population uses
20,000 training rows, 10,852 validation-gallery rows, and 108 queries. It excludes rows outside
the uppercase `A`/`C`/`G`/`T` alphabet.

E02b is validation-only. An earlier eligibility diagnostic read the old test artifacts, so that
test split is contaminated for confirmatory evaluation.

## Accepted representation products

| Kind | Candidate | Version | Dimension | GPU-hours | Persisted bytes |
| --- | --- | --- | ---: | ---: | ---: |
| DNA | 6-mer TF-IDF/SVD | `2026-08-23T14.21.58.754Z` | 512 | 0 | 93,002,965 |
| DNA | Carbon-500M | `2026-08-23T14.54.08.128Z` | 1,024 | 0.303184 | 162,999,200 |
| DNA | GENERanno prokaryote 500M | `2026-08-24T11.03.45.874Z` | 1,280 | 4.931926 | 209,933,751 |
| DNA | GENERator-v2 prokaryote 1.2B | `2026-08-23T15.30.18.164Z` | 2,048 | 0.781670 | 313,980,591 |
| Text | BGE-base-en-v1.5 | `2026-08-23T15.16.22.538Z` | 768 | 0.010603 | 92,373,563 |
| Text | GTE-ModernBERT-base | `2026-08-23T15.19.38.226Z` | 768 | 0.023440 | 92,409,461 |
| Text | Qwen3-Embedding-0.6B | `2026-08-23T15.23.41.974Z` | 1,024 | 0.057724 | 118,110,567 |

These accepted products use 1,082,810,098 persisted bytes. Their exact manifest and table hashes are
frozen in `parameters_fixed_representation_bakeoff.yml` and the experiment log.

The accepted GENERanno retry ran as detached systemd service
`vec2vec-e02b-generanno-20260824.service` at clean Git commit
`6534e2eda05776218e4f61979f6b3d729496c957`. It completed in 17,835.84 seconds. The measured
command charge was `$6.555663` at `$1.3232` per `g6.4xlarge` instance-hour. Independent read-back
matched 30,852 source rows, 30,821 unique sequences, 42,700 coverage-window rows, complete source
coverage, a 1,280-dimensional feature table, and these hashes:

- manifest: `cdbedeeee110900d602d399968a1be0b0d614f65007dc07d689fa4e492a3d13b`;
- features: `8bd9b216632bbc2d225001ff225910e65a76228f72f8adbcf1d8129bed1d5c37`;
- coverage: `6d721dea4a3ac83b9866ff4ead7d6fe3ef5fad601dd8107ce50b48889e6f5eb6`.

## Rejected GENERanno partial version

The exact approved command started at 2026-08-23 16:23:05 UTC on the existing host. It
used clean Git commit `3afc292f85222f22d477bbac40be37c55d7dac56`, Transformers 4.49.0, the
accepted E02b input, and the accepted GENERanno invariance artifact.

Observed evidence:

- the remote login session ended at 2026-08-23 21:20:37 UTC after 4 hours 57 minutes;
- system logs record session removal and 4 hours 57 minutes of consumed CPU time;
- no kernel out-of-memory event was recorded in the inspected interval;
- S3 contains only
  `04_feature/e02b/dna_features.parquet/2026-08-23T16.23.06.763Z/`;
- the feature object uses 208,302,597 bytes and contains 30,821 rows keyed by unique sequence
  hashes, with 1,280-dimensional GENERanno vectors;
- its read-back content hash is
  `7b6f6cbf5d0fa495599a50b3a3cbfc75e1727ccfa656145922ad69fd3773f0ed`;
- the matching coverage and manifest objects do not exist;
- the independent validator fails with `FileNotFoundError` on the coverage product.

The causal reason for the login-session removal is unknown. The feature object exists, and the next
two required outputs do not. The available evidence does not establish the exact sequence between
feature upload and session removal. This is a technical orchestration failure, not a
retrieval-quality result.

## Host-provenance correction

AWS identified the host as on-demand `g6.4xlarge` instance `i-0cda00ffb3cacfc12` in
`us-east-1b`, not the `g6.2xlarge` recorded in the 2026-08-23 manifests. The hostname and private
IP match the run manifests and SSH target. AWS's current price record gives `$1.3232` per hour for
the actual instance, not the recorded `$0.9776` rate. The immutable manifests remain unchanged;
the correction is frozen in configuration. No scientific artifact hash changes.

## Compute and stopping decision

The failed session lasted approximately 17,852 seconds. At the corrected price of `$1.3232` per
instance-hour, its derived command charge is approximately `$6.561602` before storage and
transfer. Scaling the previously reported total command time to the corrected rate gives
approximately `$8.230789`. Complete EC2 host cost is higher because the host remained running
outside the measured commands.

The no-partial-artifact and no-silent-retry rules still apply to the rejected version. The
accepted feature configuration is now complete. No alignment compute had started when this status
was recorded.

## Limitations and next action

- The rejected GENERanno feature table did not receive coverage or manifest validation and remains
  unaccepted. The complete retry is a separate accepted version.
- No alignment result, bootstrap interval, main effect, interaction, or selection exists.
- The user authorized the retry and alignment on 2026-08-24. The retry used 4.9544 wrapper hours
  and `$6.555663` of the additional command budget. The alignment has a separate three-hour and
  `$3.9696` cap.
- Run the frozen 36-configuration alignment factorial from a clean commit that contains all seven
  accepted artifact identities. Validate every persisted output before selection.
