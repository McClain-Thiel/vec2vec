# E00-J: Agent-Judge Pilot

## Status

Completed as a provisional review job. The fixed prompt-v3 packet version is
`2026-08-05T09.44.22.296Z`. The full GPT-5.6 Luna decision version is
`2026-08-05T09.54.34.956Z`.

Twenty-nine of 30 responses passed the structural contract. The valid responses still contained
domain errors, scope leakage, and inconsistent verdicts. No accepted benchmark label exists. The
model output is a scientist review queue only.

A later GPT-5.6 Sol validation gate produced 30 valid decisions at version
`2026-08-05T11.04.17.330Z`. Sol corrected the clearest DAP and Trimethoprim failures. Comparison with
Luna exposed unresolved growth-temperature rules and a conflation of semantic meaning with version 1
benchmark scope. The current prompt is stopped at 30 rows and will not run over all 918 rows.

The replacement prompt is `agent-judge-v4-semantic-scope`. It uses separate `semantic_support` and
`benchmark_scope` fields and separate reasons. Sixteen targeted packets from facet audit v0.2 are
frozen at output version `2026-08-05T11.45.45.519Z`. The first paid diagnostic stopped before a
provider request because the API key was absent from the process environment. No v0.2 decision or
cost exists yet.

## Question

Can one language model help a scientist find weak or wrong proposed metadata treatments without
hiding uncertainty or replacing human review?

## Fixed input

Use only the facet-audit sample at output version `2026-08-04T14.39.25.320Z`. It came from retrieval
dataset version `2026-08-04T09.02.10.007Z`. Pin the audit version with Kedro when the job runs.

The evidence packet contains the frozen Addgene metadata field, its raw value, the source
description, the proposed treatment, and the Addgene URL. The URL is provenance only. The judge
does not browse it. The packet excludes the generated plasmid description to reduce confirmation
bias. It also excludes annotations because the current pLannotate artifact has incomplete run
provenance.

## Procedure

1. Select 30 rows with fixed counts from six strata.
2. Serialize each evidence packet and its exact prompt. Store SHA-256 identities for both.
3. Ask one configured OpenRouter model for one Pydantic `JudgeDecision`.
4. Permit only `supported`, `not_supported`, or `uncertain`.
5. Preserve invalid text and request failures. Do not convert them into decisions.
6. Require a scientist to review all 30 pilot rows.

`supported` means that the evidence supports the complete proposed treatment. `not_supported`
means that the treatment is wrong or excludes a clearly supported direct mapping. `uncertain`
means that the evidence does not resolve the question.

## Checks and stopping rule

The pilot passes as a useful aid only if a scientist can understand each valid reason, packet
identities match, invalid-output rates are acceptable, and disagreements reveal specific rule
problems. Measure agreement by verdict and stratum. Report all disagreements; do not use one total
agreement score as proof of label quality.

Run at most 30 requests. Stop after reported cumulative cost reaches USD 0.10. One final request can
cause a small overshoot because the provider reports cost after completion. Do not retry with a
different model or prompt within this pilot.

## Interpretation boundary

Agent output is a review proposal. It is not accepted constraint evidence and it does not determine
whether E00 passes. A scientist must resolve each row against the source record. Any prompt or rule
change starts a new version.

## Observed result

The complete run took 82.6 seconds and reported USD 0.006769. It produced 21 `supported`, five
`not_supported`, three `uncertain`, and one `invalid_response` rows. Packet identities matched for
all rows. All rows require human review, and no accepted labels were created.

The most important findings are:

- three Trimethoprim records received three different verdicts;
- DAP was incorrectly proposed as a bacterial resistance label in multiple responses;
- one DAP response failed validation because it returned uppercase `DAP`;
- one growth-temperature packet contains a 23 versus 20 degree source conflict; and
- one intended-use packet changed from three `supported` smoke verdicts to `not_supported` in the
  full run and produced suggestions outside the narrow classified value.

See `../reports/03_luna_full_pilot.md` for the review order and full interpretation boundary.
