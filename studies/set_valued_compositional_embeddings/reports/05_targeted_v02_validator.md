# Targeted v0.2 Validator

## Conclusion

- The two-axis response contract worked on all 16 frozen packets.
- GPT-5.6 Sol supported every proposed biological interpretation.
- All growth and bacterial-selection mappings were judged inside benchmark scope.
- Free-text intended use remained biologically meaningful while two of three cases were outside
  benchmark version 1.
- One `Gateway Destination` case was scope-uncertain and remains unresolved.
- This run validates the model contract and revised exact mappings. It does not replace source-page
  review or create accepted labels.

## Fixed inputs

| Item | Identity |
| --- | --- |
| Retrieval dataset | `2026-08-04T09.02.10.007Z` |
| Facet audit | `e00-facet-audit-v0.2`; output `2026-08-05T11.42.05.452Z` |
| Packet output | `2026-08-05T11.45.45.519Z` |
| Prompt | `agent-judge-v4-semantic-scope` |
| Model | `openai/gpt-5.6-sol` |
| Provider | `OpenAI`, with fallbacks disabled |
| Decision output | `2026-08-05T14.08.54.949Z` |

The run used high reasoning, no temperature parameter, strict structured output, zero retries, and
a USD 0.40 reported-cost cap. The OpenRouter key was injected into the Kedro process from a
1Password concealed field. It was not written to the repository or logs.

## Results

| Packet group | Rows | Semantic support | Benchmark scope |
| --- | ---: | --- | --- |
| Stored `23` as `room_temperature` | 3 | 3 supported | 3 in scope |
| Trimethoprim | 2 | 2 supported | 2 in scope |
| `Kan + DAP` | 2 | 2 supported | 2 in scope |
| `Amp + Kan + Dap` | 2 | 2 supported | 2 in scope |
| `Amp + Chl + Dap` | 1 | supported | in scope |
| Dose-qualified erythromycin and kanamycin | 1 | supported | in scope |
| Intended-use free text | 3 | 3 supported | 2 out of scope; 1 uncertain |
| Unchanged controls | 2 | 2 supported | 2 in scope |

All 16 responses were valid and bound to the requested packet identities. Reported cost was USD
0.392059. No accepted labels were created.

## Interpretation

The judge consistently treated DAP as a growth requirement rather than a resistance marker. It
retained the explicitly named antibiotic markers and did not propose a DAP canonical value. The
ordinary selection control and controlled intended-use control did not regress.

The intended-use cases demonstrate the purpose of the two-axis contract. The model recognized a
direct biological or engineering meaning without automatically expanding benchmark version 1. The
scope-uncertain `Gateway Destination` case shows that the packet needs a clearer statement that only
the listed exact controlled values are in scope.

The three growth results are not independent confirmation of the upstream transformation. Each
packet includes a mapping note based on earlier source-page review. Complete source verification or
the transformation's upstream provenance is still required before the rule can support a final
benchmark label.

## Decision

The targeted model-contract gate passes. Do not run the judge over all 918 audit rows. Keep the
growth rule proposed until its source transformation is resolved. Keep the five revised selection
mappings proposed but eligible for the next explicit rule-acceptance step. Clarify the intended-use
scope boundary before materializing constraint evidence.
