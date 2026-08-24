# Gate 1 E02b Benchmark Result

## Conclusion first

- E02b is complete. The selected validation pair is 6-mer TF-IDF/SVD DNA plus
  Qwen3-Embedding-0.6B text.
- Seven representation products passed independent persisted-artifact read-back: TF-IDF/SVD,
  Carbon-500M, GENERanno, GENERator-v2, BGE-base, GTE-ModernBERT, and Qwen3-Embedding.
- GENERanno version `2026-08-23T16.23.06.763Z` is a rejected partial artifact. It contains the
  feature table but lacks the required coverage and manifest products.
- GENERanno retry version `2026-08-24T11.03.45.874Z` is complete and accepted. It did not reuse
  the rejected feature table.
- All 36 configurations in the 4-by-3-by-3 alignment factorial completed and passed independent
  read-back. The selected mean validation `utility@10` is `0.153086`, with whole-component 95%
  interval `[0.076227, 0.188279]`.
- The incumbent Carbon-500M plus BGE-base mean is `-0.041049`. The selected improvement is
  `0.194136`, so the incumbent-retention guard did not apply.
- Atomic-query utility is `0.602381`, but pair-conjunction utility is `-0.004167`. The selected
  representation does not by itself solve compositional retrieval.
- No test row was read. E02b remains validation-only.

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

## Accepted alignment and selection

Alignment version `2026-08-24T16.34.48.358Z` completed all 36 planned configurations at clean Git
commit `410fa42c280716dde5461535d7a1baef109bec57`. It used seeds 13, 42, and 20260818. Independent
read-back verified 194,400 ranking rows, 15,552 query-metric rows, 72,000 bootstrap rows, the
whitening refit, every checkpoint and training history, and the frozen selection rule.

| DNA | Text | Mean utility@10 | Whole-component 95% interval |
| --- | --- | ---: | ---: |
| TF-IDF/SVD | Qwen3 | 0.153086 | [0.076227, 0.188279] |
| GENERanno | Qwen3 | 0.141049 | [0.057701, 0.157716] |
| Carbon-500M | Qwen3 | 0.071296 | [0.025926, 0.124691] |
| TF-IDF/SVD | GTE | 0.047531 | [-0.016674, 0.060185] |
| GENERator-v2 | Qwen3 | 0.043827 | [-0.001543, 0.077778] |
| GENERanno | BGE | 0.031173 | [-0.025309, 0.055556] |
| TF-IDF/SVD | BGE | 0.017284 | [-0.037037, 0.054938] |
| GENERanno | GTE | -0.001543 | [-0.058958, 0.037353] |
| Carbon-500M | GTE | -0.026543 | [-0.068835, 0.007724] |
| Carbon-500M | BGE | -0.041049 | [-0.088580, 0.008642] |
| GENERator-v2 | GTE | -0.045370 | [-0.080864, -0.024383] |
| GENERator-v2 | BGE | -0.050309 | [-0.094761, -0.015733] |

The selected and runner-up mean difference is `0.012037`. It exceeds the fixed `0.01` practical
tie threshold, so overlapping bootstrap intervals do not trigger the cost tie-break. The selected
pair is also the cheapest top candidate: its feature extraction used `0.057724` GPU-hours because
the DNA baseline used no GPU.

| Pair | Seed 13 | Seed 42 | Seed 20260818 | Mean |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF/SVD + Qwen3 | 0.155556 | 0.158333 | 0.145370 | 0.153086 |
| Carbon-500M + BGE | -0.039815 | -0.039815 | -0.043519 | -0.041049 |

For the selected pair at K=10, the seed-mean verified, contradicted, and unknown fractions are
`0.425617`, `0.272531`, and `0.301852`. Atomic-query utility is `0.602381`. Pair-conjunction
utility is `-0.004167`, with verified and contradicted fractions `0.346250` and `0.350417`. This
difference is the main scientific limitation of the selection result.

The descriptive mean across text encoders ranks DNA as TF-IDF/SVD `0.072634`, GENERanno
`0.056893`, Carbon-500M `0.001235`, and GENERator-v2 `-0.017284`. The descriptive mean across DNA
encoders ranks text as Qwen3 `0.102315`, GTE `-0.006481`, and BGE `-0.010725`. These are factorial
summaries, not causal effects.

Across seeds, the selected pair's sequence-to-description R@1/R@10 are approximately
`0.1279`/`0.3711`; description-to-sequence R@1/R@10 are approximately `0.1434`/`0.3926`. The
incumbent values are approximately `0.0810`/`0.2815` and `0.0813`/`0.2825`.

The alignment outputs use 245,589,153 bytes. The selection-report hash is
`a675a3a3fac1b87827749764caeea07a395debf86c0ee886998417fd9a5b8d25`. All eight table hashes are
frozen in `parameters_fixed_representation_bakeoff.yml` and the experiment log.

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

The accepted alignment wrapper used 1,085.08 seconds and `$0.398828`. The GENERanno retry and
alignment together used 5.2558 wrapper hours and `$6.954491` of the additional ten-hour and
`$13.2320` cap. Adding the corrected previously recorded commands gives an approximate E02b
measured-command total of `$15.185280`. Complete EC2 host cost is higher because the host remained
running outside the measured commands.

## Limitations and next action

- The rejected GENERanno feature table did not receive coverage or manifest validation and remains
  unaccepted. The complete retry is a separate accepted version.
- The primary neural-DNA hypothesis was not supported. The train-fitted TF-IDF/SVD baseline won
  the frozen validation selection.
- Selection on one validation split makes the selected estimate optimistic. The old test split is
  contaminated and cannot provide a confirmatory claim.
- Positive atomic retrieval does not transfer to pair conjunctions under this paired-identity
  probe. Gate 2 must test set supervision and composition directly.
- Freeze TF-IDF/SVD plus Qwen3 for the Gate 2 protocol. Full-population extraction is a separate
  post-selection action and requires a resolved cost estimate and approval.
