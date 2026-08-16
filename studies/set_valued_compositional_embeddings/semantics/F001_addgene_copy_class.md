# F001: Addgene Copy Class

## Status and identity

- **Status:** Exact positive mappings are accepted for noisy supervision. State protocol
  `e00-plasmid-constraint-state-v0.1` also materializes the opposite recorded class as a narrow
  metadata contradiction. This is not biological ground truth.
- **Rule ID:** `addgene_copy_class.v0_1`
- **Source field:** `plasmid_copy`
- **Facet:** `addgene_copy_class`
- **Relation:** `reported_as`

## Meaning

The field is an operational Addgene category. Addgene asks whether a standard miniprep gives
enough DNA. `Low Copy` can also mean that the plasmid needs special growth conditions or is hard to
grow.

The field does not report an exact copies-per-cell measurement. It does not establish the same copy
number in a different strain, medium, temperature, or growth phase.

Allowed benchmark wording:

```text
Addgene records this plasmid as high copy.
Addgene records this plasmid as low copy.
```

Do not use wording such as `has exactly N copies per cell` or `is always high copy`.

## Proposed mapping

| Exact raw value | Canonical value |
| --- | --- |
| `High Copy` | `high` |
| `Low Copy` | `low` |

Only case and outer whitespace normalization are allowed before the exact match. Do not infer a
class from sequence origin, yield, description text, or pLannotate features in this rule.

## Evidence rules

For a query about canonical value `high`:

- `High Copy` is verified evidence.
- `Low Copy` is proposed contradicted evidence.
- A missing or unrecognized value is unknown.

Apply the inverse rule for `low`.

The contradiction applies only to the recorded Addgene class. It does not claim that the plasmid
cannot have a different copy number under another condition.

## Observed support

The fixed E00 profile contains:

| Canonical value | Rows | Leakage components | Train rows | Train components |
| --- | ---: | ---: | ---: | ---: |
| `high` | 85,992 | 9,741 | 68,043 | 7,875 |
| `low` | 11,216 | 2,101 | 9,160 | 1,703 |

The other 17,912 rows have no usable copy-class value.

## Exclusions and audit questions

- Treat exact source value `Unknown` as missing. Do not impute it from origin or backbone.
- Check that raw source pages use the same two-class meaning.
- Check examples with special growth notes. The class can be operational rather than mechanistic.
- Check that the two values are not artifacts of an earlier conversion step.

See [manual audit version 0.1](manual_audit_v0.1.md) for the frozen sample rule.
