# F002: Addgene Growth Temperature

## Status and identity

- **Status:** Proposed. No labels exist.
- **Rule ID:** `addgene_growth_temperature.v0_1`
- **Source field:** `growth_temp`
- **Facet:** `addgene_growth_temperature`
- **Relation:** `reported_for_propagation_at`

## Meaning

The field records the temperature that Addgene reports for growth of the shipped bacterial strain.
It is part of the propagation protocol. It is not a complete plasmid growth range.

Allowed benchmark wording:

```text
Addgene reports propagation at 30 degrees Celsius.
Addgene reports propagation at 37 degrees Celsius.
```

Do not use wording such as `cannot grow at 37 degrees Celsius`.

## Proposed mapping

| Exact raw value | Canonical value | Decision |
| --- | --- | --- |
| `30` | `30_c` | Include after audit |
| `37` | `37_c` | Include after audit |
| `23` | `room_temperature` | Hold out until provenance is confirmed |

Addgene documents `30`, `37`, and `room temperature` as the controlled choices. The local flattening
code preserves source text and does not convert room temperature to `23`. Therefore, the current
evidence does not yet show where raw `23` entered the source artifact. Do not silently treat `23` as
an exact Celsius measurement or as room temperature.

Only case and outer whitespace normalization are allowed before the exact match. Do not infer a
temperature from strain, copy class, origin, description text, or annotation features.

## Evidence rules

For a query about `30_c`:

- reviewed raw `30` is verified evidence;
- reviewed raw `37` is proposed contradicted evidence;
- raw `23`, missing values, and unrecognized values are unknown.

Apply the inverse rule for `37_c`. Do not create evidence for `room_temperature` until the raw `23`
provenance check passes.

A different reviewed value conflicts with the recorded protocol. It does not establish failure to
grow at the query temperature.

## Observed support

| Raw value | Rows | Leakage components | Train rows | Train components |
| --- | ---: | ---: | ---: | ---: |
| `37` | 107,754 | 13,285 | 86,258 | 10,729 |
| `30` | 7,260 | 1,294 | 5,760 | 1,060 |
| `23` | 106 | 45 | 79 | 33 |

## Exclusions and audit questions

- Hold out raw `23` until its source transformation is located and checked.
- Check original source pages for the displayed temperature and any special growth instructions.
- Record whether growth notes change the meaning of a bare temperature value.
- Do not infer a contradiction from an unrecognized or missing value.

See [manual audit version 0.1](manual_audit_v0.1.md) for the frozen sample rule.
