# Gate 1 E02b Benchmark Status

## Conclusion first

- E02b is incomplete. No DNA-and-text pair has been selected.
- Six representation products passed independent persisted-artifact read-back: TF-IDF/SVD,
  Carbon-500M, GENERator-v2, BGE-base, GTE-ModernBERT, and Qwen3-Embedding.
- GENERanno version `2026-08-23T16.23.06.763Z` is a rejected partial artifact. It contains the
  feature table but lacks the required coverage and manifest products.
- The complete 4-by-3-by-3 alignment factorial did not start. No validation ranking or old-test
  model outcome was read.
- A new exact GENERanno attempt needs user authorization. The failed version cannot be reused or
  silently completed.

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
| DNA | GENERator-v2 prokaryote 1.2B | `2026-08-23T15.30.18.164Z` | 2,048 | 0.781670 | 313,980,591 |
| Text | BGE-base-en-v1.5 | `2026-08-23T15.16.22.538Z` | 768 | 0.010603 | 92,373,563 |
| Text | GTE-ModernBERT-base | `2026-08-23T15.19.38.226Z` | 768 | 0.023440 | 92,409,461 |
| Text | Qwen3-Embedding-0.6B | `2026-08-23T15.23.41.974Z` | 1,024 | 0.057724 | 118,110,567 |

These accepted products use 872,876,347 persisted bytes. Their exact manifest and table hashes are
frozen in `parameters_fixed_representation_bakeoff.yml` and the experiment log.

## Rejected GENERanno partial version

The exact approved command started at 2026-08-23 16:23:05 UTC on the existing `g6.2xlarge`. It
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

## Compute and stopping decision

The failed session lasted approximately 17,852 seconds. At the approved observed price of
$0.9776 per instance-hour, its derived command charge is approximately $4.847810 before storage
and transfer. Completed E02b neural feature commands and this failed command total approximately
$6.081030 in derived instance charges. Complete EC2 host cost is higher because the host remained
running outside the measured commands.

The no-partial-artifact and no-silent-retry rules apply. The accepted feature configuration remains
incomplete, and the alignment pipeline fails closed without GENERanno. No alignment compute was
started.

## Limitations and next action

- The GENERanno feature table did not receive coverage or manifest validation and is not accepted.
- No alignment result, bootstrap interval, main effect, interaction, or selection exists.
- A retry would consume additional paid compute and requires explicit user authorization.
- A retry must use the same model revision, input version, scientific parameters, and seed. Its
  orchestration must survive terminal or login-session closure.
