# F002: Addgene Growth Temperature v0.2

## Status and identity

- **Status:** Exact `30` and `37` positive mappings are accepted for noisy supervision. State
  protocol `e00-plasmid-constraint-state-v0.1` materializes their narrow reported-temperature
  conflict. Stored value `23` remains unlabeled in this protocol.
- **Rule ID:** `addgene_growth_temperature.v0_2`
- **Source field:** `growth_temp`
- **Facet:** `addgene_growth_temperature`
- **Relation:** `reported_for_propagation_at`

This rule supersedes `addgene_growth_temperature.v0_1` for new audit outputs. It does not rewrite
the v0.1 sample or decisions.

## Meaning

The field records the propagation temperature shown by Addgene for the shipped bacterial strain.
It is not a complete plasmid growth range. A different reported value does not prove that growth is
impossible at the query temperature.

## Exact mappings

| Exact stored value | Canonical value | Interpretation |
| --- | --- | --- |
| `30` | `30_c` | Addgene reports propagation at 30 degrees Celsius. |
| `37` | `37_c` | Addgene reports propagation at 37 degrees Celsius. |
| `23` | `room_temperature` | Reviewed source pages display `Room Temperature` for records stored locally as `23`. |

The `23` mapping is categorical. Do not report it as an exact measurement of 23 degrees Celsius.
Preserve the stored `23` beside `room_temperature`. Preserve source-description conflicts, such as
a separate statement that names an optimal temperature of 20 degrees Celsius.

Only case and outer whitespace normalization are allowed before an exact match. Do not infer a
temperature from strain, copy class, origin, description text, pLannotate, or plasmidkit.

## Evidence rules

- A reviewed exact mapping is proposed verified evidence for its narrow reported-propagation claim.
- A different reviewed numeric value is proposed contradicted evidence for `30_c` versus `37_c`.
- `room_temperature` remains a category and does not contradict either exact numeric value.
- Missing and unrecognized values are unknown.

## Required v0.2 review

Review every available train and validation leakage component with stored value `23`. Confirm the
displayed source category and retain any conflicting temperature statement. Report the number of
source pages checked, unavailable pages, category matches, and conflicts.

See [manual audit version 0.2](manual_audit_v0.2.md).
