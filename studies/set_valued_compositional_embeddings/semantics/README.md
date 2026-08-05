# Current Facet Semantics

## Status

Proposed. These rules do not create benchmark labels.

## Purpose

This directory defines what each candidate field can support. A rule describes a narrow metadata
claim. It does not turn depositor metadata into a universal biological claim.

The rules use this fixed profile input:

```text
s3://plasmidclip/kedro/08_reporting/e00/constraint_value_profile.parquet/
2026-08-04T14.05.27.282Z/constraint_value_profile.parquet
```

## Proposed facets

| Rule sheet | Source field | Proposed claim | Negative evidence |
| --- | --- | --- | --- |
| [Addgene copy class](F001_addgene_copy_class.md) | `plasmid_copy` | Recorded operational copy class | Proposed for the opposite class |
| [Addgene growth temperature v0.2](F002_addgene_growth_temperature_v0.2.md) | `growth_temp` | Recorded propagation temperature or room-temperature category | Proposed for a different reviewed value |
| [Bacterial selection marker v0.2](F003_bacterial_selection_marker_v0.2.md) | `bacterial_resistance` | Recorded bacterial selection includes an antibiotic | None in version 1 |
| [Addgene intended use](F004_addgene_intended_use.md) | `vector_types` | Addgene controlled intended-use tag | None in version 1 |

The [manual audit protocol v0.2](manual_audit_v0.2.md) defines the current sample and pass rule. A
facet remains proposed until that audit passes. The original growth-temperature and
bacterial-selection sheets and [manual audit v0.1](manual_audit_v0.1.md) remain unchanged as
research history.

## Shared evidence states

- `verified`: the reviewed source value directly supports the narrow claim.
- `contradicted`: a reviewed rule directly conflicts with the narrow claim.
- `unknown`: there is no reviewed positive or conflict rule.

Absence is unknown. A different value is also unknown unless the rule sheet defines it as an
explicit conflict.

## Source limits

Addgene states that depositors provide the plasmid data. Addgene validates deposited sequences,
but this does not mean that it independently tests each growth or intended-use claim. The benchmark
must therefore use wording such as `Addgene-recorded` or `reported`.

These rules do not use annotation features. If a later rule needs sequence feature evidence, it
must use a separately versioned pLannotate-only product. It must not fall back to plasmidkit.

## Primary documentation

- [Addgene deposit spreadsheet field definitions](https://blog.addgene.org/quick-way-to-deposit-plasmids-the-deposit-spreadsheet), updated February 2024.
- [Anatomy of an Addgene plasmid page](https://blog.addgene.org/anatomy-of-a-plasmid-page-at-addgene).
- [Addgene plasmid catalog filters](https://www.addgene.org/search/catalog/plasmids/).
- [Addgene deposit overview](https://www.addgene.org/deposit/).

Documentation was accessed on 2026-08-04.
