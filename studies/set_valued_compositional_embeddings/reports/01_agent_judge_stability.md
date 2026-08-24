# Agent-Judge Stability Check

## Result

Increasing the Qwen judge budget from 500 to 2,000 output tokens and enabling reasoning did not
improve reliability. Temperature was already 0.0 and remained 0.0.

Only 2 of 18 responses produced valid Pydantic decisions. No packet produced a valid verdict in all
three identical repeats. Do not use this configuration for the 30-row pilot.

## Fixed protocol

- Model: `qwen/qwen3.5-397b-a17b`
- Prompt: `agent-judge-v2`
- Prompt hash: `226ddc98f62be91f1579f185eff59b93a85fe5e82a186c8fbd8cebbf4f9fe4b5`
- Packet version: `2026-08-04T15.38.00.322Z`
- Packet indices: 1, 5, 9, 17, 22, and 27
- Temperature: 0.0
- Reasoning: enabled
- Maximum output tokens: 2,000
- Structured output: enabled
- Provider: not pinned

## Repeats

| Repeat | Kedro output version | Valid | Invalid response | Request error | Reported cost | Runtime |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2026-08-05T09.09.18.054Z` | 1 | 0 | 5 | USD 0.009517 | 223.6 s |
| 2 | `2026-08-05T09.13.04.562Z` | 1 | 1 | 4 | USD 0.027894 | 750.7 s |
| 3 | `2026-08-05T09.25.37.235Z` | 0 | 0 | 6 | USD 0.000000 | 399.2 s |

The valid fraction was 2/18, or 11.1%. Fifteen responses had request or response-extraction errors.
One response failed Pydantic validation. Four rows had the same status in all three repeats only
because all three attempts failed. Exact verdict agreement is not estimable because no row had
three valid verdicts.

The mean runtime was 457.8 seconds. The median was 399.2 seconds, and the range was 527.1 seconds.

## Cost limitation

The artifacts report USD 0.037411. After the runs, the key reported USD 0.127318443 of daily usage.
The difference of USD 0.089907443 includes charged responses that the old client could not parse and
any other use of the same key that day. It cannot be assigned exactly. Treat the artifact total as
a lower bound.

The client now retains reported cost and generation ID when a charged JSON response lacks usable
final content. This fix applies only to future runs.

## Interpretation

Temperature zero does not guarantee identical hosted inference. The provider was not pinned, and
hosted model execution can remain nondeterministic. However, verdict variance is not the main
failure here: final-answer production failed on 16 of 18 calls.

The 2,000-token budget was still consumed without a usable final answer on most calls. A larger
reasoning budget would increase cost and latency without evidence that it would solve the protocol
failure. Keep reasoning disabled for this structured classification task.

If another stability test is justified, change one factor at a time. Use a non-reasoning model or
disable reasoning, pin one OpenRouter provider, request a fixed seed when supported, and retain the
same six packet hashes. Do not infer judge accuracy from output consistency alone.
