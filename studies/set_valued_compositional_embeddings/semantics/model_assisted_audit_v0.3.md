# Model-Assisted Facet Audit Protocol v0.3

## Status

The v0.2 audit sample and exact packet list are persisted and inspected. Protocol v0.2 and its
30-row results remain unchanged.

## Purpose

Use GPT-5.6 Sol for a small synthetic-biology knowledge check of the revised rules. The model makes
proposals. It does not write accepted labels.

## Two-axis response contract

Each response must make two independent judgments:

- `semantic_support`: `supported`, `not_supported`, or `uncertain`;
- `benchmark_scope`: `in_scope`, `out_of_scope`, or `uncertain`.

The response must include separate `semantic_reason` and `scope_reason` fields. A biologically
meaningful free-text intended use can therefore be semantically supported and still out of scope
for benchmark version 1.

The local Pydantic model is the complete response contract. It forbids extra fields, binds both
packet identities, validates list contents, and permits alternative canonical values only when the
proposed semantic mapping is `not_supported`. Preserve invalid raw responses without repair.

## Planned targeted packet groups

- reviewed `23 -> room_temperature` cases;
- every new bacterial-selection exact mapping;
- intended-use free text that is meaningful but outside version 1;
- unchanged controlled mappings as drift controls.

The targeted set contains 16 packets. It uses three `room_temperature` rows; eight rows across the
five revised selection cells; three out-of-scope intended-use rows; and two unchanged controls.
Packet output version is `2026-08-05T11.45.45.519Z`.

Run one packet first with a USD 0.08 cap. Run the complete set only if the response is valid and
uses both axes correctly. The complete run has a USD 0.40 cap. Use provider `OpenAI`, model
`openai/gpt-5.6-sol`, high reasoning, temperature omitted, strict structured output, no fallbacks,
and zero transport retries. Pin the packet load version in each Kedro command.

## Gate

Proceed only if every response is schema-valid, the two axes are used independently, revised
mappings are interpreted consistently, and unchanged controls do not regress. Do not scale to the
complete audit sample merely because the targeted gate passes.
