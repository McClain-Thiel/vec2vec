# F004: Addgene Intended Use

## Status and identity

- **Status:** Proposed. No labels exist.
- **Rule ID:** `addgene_intended_use.v0_1`
- **Source field:** `vector_types`
- **Relation:** `addgene_tagged_for`

## Meaning

Addgene calls this source field `Primary Vector Type`. Its deposit instructions say that it
describes the intended use of the plasmid. The observed field also contains free text from the
`Other` option. It is therefore not one clean biological type system.

The current Addgene catalog presents expression filters separately from other vector-type filters.
Version 1 follows this distinction and derives two facets from exact controlled values.

Allowed benchmark wording:

```text
Addgene tags this plasmid for mammalian expression.
Addgene tags this plasmid for CRISPR use.
```

Do not use wording such as `works in every mammalian cell` or `is a complete lentiviral system`.

## Proposed controlled mappings

### Expression context

Use facet `addgene_expression_context` with relation `tagged_for_expression_in`.

| Exact controlled value | Canonical value |
| --- | --- |
| `Mammalian Expression` | `mammalian` |
| `Bacterial Expression` | `bacterial` |
| `Yeast Expression` | `yeast` |
| `Worm Expression` | `worm` |
| `Insect Expression` | `insect` |
| `Plant Expression` | `plant` |

### Use category

Use facet `addgene_use_category` with relation `tagged_for`.

| Exact controlled value | Canonical value |
| --- | --- |
| `Lentiviral` | `lentiviral` |
| `Retroviral` | `retroviral` |
| `AAV` | `aav` |
| `RNAi` | `rnai` |
| `Luciferase` | `luciferase` |
| `Cre/Lox` | `cre_lox` |
| `Mouse Targeting` | `mouse_targeting` |
| `CRISPR` | `crispr` |
| `TALEN` | `talen` |
| `Synthetic Biology` | `synthetic_biology` |
| `Affinity Reagent/ Antibody` | `affinity_reagent_antibody` |

Only case, outer whitespace, and the shown punctuation normalization are allowed. Do not map free
text by keyword in version 1.

## Evidence rules

- An exact reviewed controlled value is verified evidence for its derived facet.
- Version 1 has no contradicted evidence for either derived facet.
- A different tag, an absent tag, or free text is unknown.

Multiple tags can be present. Different intended uses are not mutually exclusive. A tag states
intended use. It does not by itself establish every component or function needed for that use.

## Observed support

The fixed E00 profile contains 113,940 rows with at least one value and 1,294 normalized values.
Only 122 values meet the initial train support screen. Many supported values are still free text.

Large exact controlled values include:

| Controlled value | Rows |
| --- | ---: |
| `Mammalian Expression` | 42,274 |
| `Bacterial Expression` | 16,710 |
| `Lentiviral` | 13,454 |
| `CRISPR` | 11,572 |
| `Synthetic Biology` | 7,859 |
| `AAV` | 5,470 |
| `Yeast Expression` | 4,218 |
| `Plant Expression` | 2,933 |
| `Insect Expression` | 2,215 |
| `Retroviral` | 2,142 |

The profile also contains `Unspecified`, `Other`, Gateway terms, cloning-vector descriptions,
promoter descriptions, donor-template text, and other free text.

## Exclusions and audit questions

- Treat exact source value `N/A` and an empty list as missing.
- Exclude `Unspecified` and `Other` because they do not state a useful controlled category.
- Exclude all free text in version 1, even when a keyword looks clear.
- Check whether the source artifact keeps controlled and free-text values distinguishable.
- Check co-occurrence between controlled tags. Do not make values mutually exclusive.
- For viral tags, check examples of transfer, packaging, helper, and complete constructs. Keep the
  claim at the Addgene-tag level.
- Do not use pLannotate or plasmidkit to infer an intended-use tag.

A later rule version can add reviewed free-text mappings. It must have a new rule ID and a separate
audit.

See [manual audit version 0.1](manual_audit_v0.1.md) for the frozen sample rule.
