# Manual Facet Audit Protocol v0.2

## Status

Proposed before generation of the v0.2 sample. This protocol creates no accepted labels. Version
0.1 and all of its outputs remain unchanged.

## Purpose

Test the revised growth-temperature and bacterial-selection mappings while retaining the broader
E00 audit structure. The audit tests narrow Addgene metadata claims. It does not test universal
biological function or missing-marker recall.

## Fixed inputs and identity

- Retrieval dataset version: `2026-08-04T09.02.10.007Z`.
- Eligible row splits: train and validation only.
- Sampling unit: leakage component, with at most one row per component and stratum.
- Audit version and sampling key: `e00-facet-audit-v0.2`.
- Unchanged rules: `addgene_copy_class.v0_1` and `addgene_intended_use.v0_1`.
- Revised rules: `addgene_growth_temperature.v0_2` and
  `bacterial_selection_marker.v0_2`.

Calculate deterministic selection order as:

```text
sha256("e00-facet-audit-v0.2|<facet>|<stratum>|<component_id>|<sequence_id>")
```

Do not inspect test-row metadata while changing rules or selecting review rows.

## Changes from v0.1

| Area | Version 0.2 change |
| --- | --- |
| Growth temperature | Map stored `23` to categorical `room_temperature`; do not call it exactly 23 degrees Celsius. |
| Bacterial selection | Add five reviewed complete-cell mappings. Use full antibiotic names and preserve an interpretation note. |
| Sampling | Put changed exact mappings in dedicated strata and retain every available train/validation component in those strata. |
| Judge contract | Judge semantic support separately from frozen benchmark scope. |

All ordinary v0.1 strata keep their previous target and minimum rules. The new strata are:

- `growth_temperature:reviewed_exact_mapping`;
- `bacterial_selection:reviewed_exact_mapping`.

Both new strata sample all available eligible leakage components. The manifest must report support
by canonical value and whether ordinary targets were met.

## Review view

Show exact raw source values, full canonical values, `mapping_note`, source description, Addgene
page URL, rule identity, leakage component, and proposed narrow claim. Do not show generated
descriptions, retrieval scores, model conclusions, or accepted labels.

For the first v0.2 check, inspect:

1. all changed exact source cells as separate groups;
2. source-page category agreement for stored growth value `23`;
3. DAP treatment as a non-resistance growth requirement;
4. preservation of antibiotic doses without turning doses into canonical values;
5. unchanged controlled-value examples as negative controls for code drift.

## Decision and stopping rule

Use `supported`, `not_supported`, `ambiguous`, or `source_unavailable`. Require a reason except for
`supported`. Preserve every decision and any later adjudication.

Stop and revise the rule if any changed mapping makes a stronger claim than its source, loses a
named antibiotic, maps DAP as resistance, treats `room_temperature` as an exact numeric
measurement, or changes an unchanged control unexpectedly. A failed v0.2 result must remain in the
record. A new rule needs a new identifier and sampling key.

The targeted check is a rule-validation gate, not a population accuracy estimate. Decide whether a
larger precision audit is needed only after this gate passes.
