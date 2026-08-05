# GPT-5.6 Luna Full Agent-Judge Pilot

## Result

The fixed 30-row pilot completed. Twenty-nine responses passed the structural contract. One
response failed validation. The valid responses contained important scientific errors and
inconsistencies. Therefore, this run is useful as a review queue, but it is not a source of accepted
labels.

No model decision has been accepted as constraint evidence.

## Fixed protocol

- Model: `openai/gpt-5.6-luna`
- Provider: `OpenAI`, pinned with fallbacks disabled
- Prompt: `agent-judge-v3`
- Prompt hash: `6882f1feff24e80b83b47d896093ea8d3cd9d6bbb09f81ce18d95f9230486104`
- Packet version: `2026-08-05T09.44.22.296Z`
- Decision and summary version: `2026-08-05T09.54.34.956Z`
- Facet-audit input version: `2026-08-04T14.39.25.320Z`
- Seed: 17
- Reasoning: disabled
- Temperature: not sent
- Maximum output tokens: 1,000
- Structured output: enabled
- Cost cap: USD 0.10

The run took 82.6 seconds. OpenRouter reported a total cost of USD 0.006769.

## Structural result

| Measure | Count |
| --- | ---: |
| Packets | 30 |
| Valid responses | 29 |
| Invalid responses | 1 |
| `supported` | 21 |
| `not_supported` | 5 |
| `uncertain` | 3 |

All 30 decision rows matched the stored packet identities. Every row retained
`human_review_required = true`, and every row retained `accepted_label_created = false`.

## Result by stratum

| Stratum | Supported | Not supported | Uncertain | Invalid |
| --- | ---: | ---: | ---: | ---: |
| Copy class: missing | 4 | 0 | 0 | 0 |
| Growth temperature: held out | 3 | 0 | 1 | 0 |
| Bacterial selection: proposed exclusion | 1 | 4 | 2 | 1 |
| Bacterial selection: proposed combination | 5 | 0 | 0 | 0 |
| Intended use: proposed exclusion | 4 | 1 | 0 | 0 |
| Intended use: controlled category | 4 | 0 | 0 | 0 |

These counts describe model outputs. They are not accuracy results.

## Scientist review queue

Review the rows in this order. The links identify the source records, but the stored packet remains
the fixed evidence used by the model.

| Priority | Rows | Source records | Why review is required |
| --- | --- | --- | --- |
| 1 | 12, 13, 15, 16 | [187394](https://www.addgene.org/187394/), [187386](https://www.addgene.org/187386/), [202274](https://www.addgene.org/202274/), [187385](https://www.addgene.org/187385/) | The proposed rule treats DAP as a growth requirement. Luna proposed `dap` as a resistance label in rows 12, 13, and 16. Row 12 also failed the response contract because it returned uppercase `DAP`. |
| 1 | 9, 10, 11 | [209743](https://www.addgene.org/209743/), [111608](https://www.addgene.org/111608/), [112917](https://www.addgene.org/112917/) | The same classified value, `Trimethoprim`, received all three verdicts. Row 11 also says that exclusion is not supported while its verdict is `uncertain`. Resolve the treatment as one explicit rule. |
| 1 | 22 | [121780](https://www.addgene.org/121780/) | The full run rejected the exclusion, but all three smoke repeats supported it. The suggestions `gateway_assembly` and `puromycin_selection` may exceed the intended scope of the classified source value. |
| 2 | 7 | [172199](https://www.addgene.org/172199/) | The structured field says 23, but the source description says optimal growth at 20 degrees. Preserve this conflict until a scientist defines which field controls the benchmark claim. |
| 2 | 14 | [223180](https://www.addgene.org/223180/) | The source value contains erythromycin and kanamycin with doses. Luna proposed direct mappings for both. Confirm that the parser should retain antibiotic identity and discard dose syntax. |

After these rows, review the proposed combination rows 17–21 as positive controls, then review the
remaining missing, held-out, and intended-use rows. A `supported` verdict is not a reason to skip
human review.

## Interpretation

Luna is reliable enough to produce structured triage notes at low cost. It is not reliable enough
to adjudicate metadata rules. Structural validity did not prevent domain errors, scope leakage, or
verdict instability.

The next decision must be made by a scientist. Record one human verdict and reason for each row.
Only after that review may a separate process propose changes to controlled values or benchmark
labels.
