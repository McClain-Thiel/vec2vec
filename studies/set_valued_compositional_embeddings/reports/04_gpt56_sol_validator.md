# GPT-5.6 Sol Validator Gate

## Result

GPT-5.6 Sol passed the 30-row structural gate and corrected the clearest Luna biology errors. It did
not make the current agent-judge prompt suitable for a 918-row run. The prompt combines two
different questions: whether a source value has a defensible biological meaning, and whether that
meaning belongs in benchmark version 1.

Do not scale this prompt. Revise the facet rules and split semantic support from benchmark scope
before another validator run.

## Final fixed run

- Judge version: `e00-agent-validator-gpt56-sol-v0.6`
- Model: `openai/gpt-5.6-sol`
- Provider: `OpenAI`, pinned with fallbacks disabled
- Packet version: `2026-08-05T09.44.22.296Z`
- Decision version: `2026-08-05T11.04.17.330Z`
- Prompt: `agent-judge-v3`
- Prompt hash: `6882f1feff24e80b83b47d896093ea8d3cd9d6bbb09f81ce18d95f9230486104`
- Reasoning effort: high
- Seed: 17
- Temperature: not sent
- Maximum output tokens: 3,000
- Structured output: full JSON Schema
- Transport retries: zero
- Cost cap: USD 0.75

All 30 responses passed the complete Pydantic contract. Every response returned model
`openai/gpt-5.6-sol` and provider `OpenAI`. The run took 240.0 seconds and reported USD 0.536997.
It produced 14 `supported`, 15 `not_supported`, and one `uncertain` verdict.

No accepted label was created.

## Integration history

The failed attempts remain part of the experiment record:

| Version | Result | Cost status |
| --- | --- | --- |
| Opus v0.1 | Amazon Bedrock rejected unsupported JSON Schema keywords before generation | Reported zero |
| Opus v0.2 | Portable schema cleared HTTP 400, but the request stalled and entered a generic retry loop; run stopped | No result artifact |
| Opus v0.3 | Azure reported that structured outputs were not enabled for the workspace | No generation |
| Opus v0.4 | Azure generated text but omitted required `evidence_used`; local Pydantic rejected it | Cost unknown |
| Sol v0.5 | One-row diagnostic passed full structured and local validation | USD 0.0154225 |
| Sol v0.6 | Complete 30-row run passed | USD 0.536997 |

## Comparison with Luna

The version-pinned comparison is `2026-08-05T11.08.45.901Z`.

| Measure | Result |
| --- | ---: |
| Rows | 30 |
| Both responses valid | 29 |
| Exact verdict agreements | 18 |
| Exact verdict disagreements | 11 |
| Luna-invalid rows | 1 |
| Manual or rule resolution rows | 12 |
| Agreement among valid pairs | 62.1% |

Agreement is not accuracy. The useful result is the pattern of corrections and unresolved rule
questions.

## Scientific findings

### Bacterial selection

Sol handled the DAP rows coherently. It retained the antibiotic markers and did not propose DAP as a
resistance marker:

- `Amp + Kan + Dap` became ampicillin and kanamycin;
- `Kan + DAP` became kanamycin;
- `Amp + Chl + Dap` became ampicillin and chloramphenicol.

It also mapped all three Trimethoprim rows to `trimethoprim` and mapped the dose-qualified
erythromycin and kanamycin value to both antibiotics. These results show that
`bacterial_selection_marker.v0_1` excludes complete source cells too coarsely. A revised exact rule
should retain reviewed antibiotic parts while ignoring reviewed non-marker parts.

### Growth temperature

Sol did not give one treatment to the four stored `23` rows. It returned one `uncertain`, two
`not_supported`, and one `supported` verdict.

A direct source check is more informative. All four Addgene pages currently display `Room
Temperature` in the Growth Temperature field:

- [Addgene 216749](https://www.addgene.org/216749/)
- [Addgene 200280](https://www.addgene.org/200280/)
- [Addgene 172199](https://www.addgene.org/172199/)
- [Addgene 164724](https://www.addgene.org/164724/)

The stored raw value `23` is therefore consistent with an upstream encoding of Addgene's controlled
`Room Temperature` display. Addgene 172199 also says “optimal growth at 20 degrees” in its purpose.
Keep that statement as a source conflict note, but use the explicit Growth Temperature field for the
narrow `reported_for_propagation_at` facet.

Do not map `23` to a canonical value named `23`. Propose `23 -> room_temperature` in a new rule
version and audit that transformation separately.

### Intended use

Sol rejected exclusion of all five free-text values and proposed meanings such as Gateway cloning,
barcode labeling, entry clone, insertional mutagenesis, and reporter use. Those meanings are
plausible, but the current version 1 rule intentionally limits the benchmark to exact controlled
Addgene tags.

The current judge verdict cannot express both statements:

1. the free text has a defensible biological meaning; and
2. the value remains outside the frozen version 1 benchmark scope.

Treat this as a measurement-design error, not a model error or a reason to expand the vocabulary
automatically.

### Stable controls

Both models supported all four missing copy-class rows, all five accepted antibiotic combinations,
and all four controlled intended-use categories. These are useful controls, but they remain
proposals until the rule-level audit is complete.

## Gate decision

Stop the current agent-judge prompt at 30 rows. Do not run it over all 918 audit rows.

Before another model-assisted audit:

1. propose `addgene_growth_temperature.v0_2` with `23 -> room_temperature` and explicit source-page
   provenance;
2. propose `bacterial_selection_marker.v0_2` with reviewed partial mappings for Trimethoprim, DAP
   compounds, and dose-qualified values;
3. keep free-text intended uses outside version 1 while recording that exclusion as a scope choice,
   not lack of semantic meaning;
4. replace the single judge verdict with separate semantic-support and benchmark-scope fields; and
5. generate a new deterministic audit version and smoke-test only the revised edge cases before any
   larger paid run.
