# F003: Bacterial Selection Marker v0.2

## Status and identity

- **Status:** Proposed. No labels exist.
- **Rule ID:** `bacterial_selection_marker.v0_2`
- **Source field:** `bacterial_resistance`
- **Facet:** `bacterial_selection_marker`
- **Relation:** `reported_selection_includes`

This rule supersedes `bacterial_selection_marker.v0_1` for new audit outputs. It does not rewrite
the v0.1 sample or decisions.

## Meaning

The field records the bacterial selection condition reported by Addgene. A mapped antibiotic means
that the reported selection includes that antibiotic. It does not mean that the cell contains only
that resistance marker or that resistance works in every host and condition.

## Mapping form

Ordinary controlled antibiotic names use the simple exact mapping table in Kedro configuration.
An ambiguous or compound source cell uses a `reviewed_mappings` entry with:

- the unchanged exact source cell;
- full canonical antibiotic names;
- one plain-language interpretation of abbreviations, doses, and ignored components.

The interpretation is saved as `mapping_note` in the vocabulary, audit sample, review export, and
agent evidence packet. Code must not split arbitrary text on `and`, `+`, commas, or parentheses.

## New reviewed mappings

| Exact source cell | Full mapped markers | Component interpretation |
| --- | --- | --- |
| `Trimethoprim` | `trimethoprim` | The antibiotic name is already written in full. |
| `Kan + DAP` | `kanamycin` | `Kan` means kanamycin. DAP is diaminopimelic acid and is a growth requirement, not an antibiotic-resistance marker. |
| `Amp + Kan + Dap` | `ampicillin`, `kanamycin` | `Amp` means ampicillin. `Kan` means kanamycin. DAP is not mapped. |
| `Amp + Chl + Dap` | `ampicillin`, `chloramphenicol` | `Amp` means ampicillin. `Chl` means chloramphenicol. DAP is not mapped. |
| `Erythromycin (200 μg/mL), Kanamycin (50 μg/mL)` | `erythromycin`, `kanamycin` | Map both named antibiotics. Preserve both doses only in the raw value. |

These are complete-cell exact mappings. They do not authorize a general abbreviation parser.
`Amp + DAP`, spelling variants, supplements, and other dose-qualified cells remain excluded unless
they receive their own reviewed entry.

## Evidence rules

- A reviewed exact mapping is proposed verified evidence for each mapped antibiotic.
- DAP, glucose, arabinose, sucrose, and dose text do not become canonical resistance markers.
- A different antibiotic is unknown for the queried antibiotic.
- Missing, excluded, free-text, and unreviewed values are unknown.
- Version 0.2 remains positive-only. It defines no contradicted evidence for this facet.

## Required v0.2 review

Review every available train and validation leakage component for each new exact mapping. Report
results separately by exact source cell. Do not combine the five cells into one accuracy number
without also showing their individual counts.

See [manual audit version 0.2](manual_audit_v0.2.md).
