# GPT-5.6 Luna Judge Smoke Test

## Result

GPT-5.6 Luna passed the structural smoke test. All 18 responses across three identical six-row runs
passed the Pydantic contract and matched the requested packet identity.

Four of six packets had the same verdict in all three runs. The two disagreements crossed an
uncertainty boundary; no packet changed directly between `supported` and `not_supported`.

This is good enough to use Luna as a scientist-reviewed triage aid. It is not evidence that Luna is
an accurate autonomous judge.

## Fixed protocol

- Model: `openai/gpt-5.6-luna`
- Provider: `OpenAI`, pinned with fallbacks disabled
- Prompt: `agent-judge-v3`
- Prompt hash: `6882f1feff24e80b83b47d896093ea8d3cd9d6bbb09f81ce18d95f9230486104`
- Packet version: `2026-08-05T09.44.22.296Z`
- Packet indices: 1, 5, 9, 17, 22, and 27
- Seed: 17
- Reasoning: disabled
- Temperature: not sent because the model does not advertise this parameter
- Maximum output tokens: 1,000
- Structured output: enabled

## Runs

| Repeat | Kedro output version | Valid | Reported cost | Runtime |
| ---: | --- | ---: | ---: | ---: |
| 1 | `2026-08-05T09.45.12.796Z` | 6/6 | USD 0.001471 | 17.6 s |
| 2 | `2026-08-05T09.46.38.725Z` | 6/6 | USD 0.000683 | 22.4 s |
| 3 | `2026-08-05T09.47.03.757Z` | 6/6 | USD 0.000742 | 24.9 s |

Total reported cost was USD 0.002896. Mean runtime was 21.6 seconds and the range was 7.3 seconds.

## Verdict stability

| Packet | Treatment | Repeat 1 | Repeat 2 | Repeat 3 |
| ---: | --- | --- | --- | --- |
| 1 | Copy class `Unknown` as missing | supported | supported | supported |
| 5 | Growth temperature `23` held out | uncertain | supported | uncertain |
| 9 | Trimethoprim excluded | not_supported | uncertain | not_supported |
| 17 | Spectinomycin and tetracycline included | supported | supported | supported |
| 22 | Free-text intended-use value excluded | supported | supported | supported |
| 27 | Lentiviral category included | supported | supported | supported |

Unanimous verdict agreement was 4/6, or 66.7%. Pairwise agreement was 4/6, 6/6, and 4/6. The
temperature and Trimethoprim rows need human resolution. Their disagreement is scientifically
meaningful because it concerns whether limited evidence supports a treatment or should remain
uncertain.

## Interpretation

Luna fixed the Qwen integration failures: response structure, packet identity, provider identity,
latency, and cost were all stable enough for a small review job. The fixed seed did not force exact
verdict agreement on ambiguous rows.

Use Luna to prioritize or explain review rows. Do not let one Luna verdict create a benchmark label.
For the 30-row pilot, retain all raw responses, require human review, and mark ambiguous model
verdicts as a review signal rather than an adjudication.
