# F003: Bacterial Selection Marker

## Status and identity

- **Status:** Proposed. No labels exist.
- **Rule ID:** `bacterial_selection_marker.v0_1`
- **Source field:** `bacterial_resistance`
- **Facet:** `bacterial_selection_marker`
- **Relation:** `reported_selection_includes`

## Meaning

Addgene defines this field as antibiotic resistance encoded by the plasmid and used to maintain the
glycerol stock. The field can contain more than one antibiotic. Some observed cells also contain a
concentration, a supplement, a growth requirement, or free text.

Allowed benchmark wording:

```text
Addgene reports bacterial selection that includes ampicillin.
```

Do not shorten this to `the plasmid is only ampicillin resistant`. Do not claim resistance in an
unrecorded host or condition.

## Proposed canonical values

Start with the controlled antibiotic names in the Addgene deposit instructions:

```text
ampicillin
apramycin
blasticidin
bleocin_zeocin
chloramphenicol
erythromycin
gentamicin
hygromycin
kanamycin
nourseothricin_clonnat
spectinomycin
streptomycin
tetracycline
```

The aliases `Zeocin` and `clonNat` must remain visible in the raw-value trace. Correct a spelling
variant only through an explicit, reviewed mapping.

## Proposed mapping rules

1. Map an exact controlled single-antibiotic value to one canonical value.
2. Map a reviewed controlled combination to each antibiotic that it names.
3. Use a frozen table for compound values. Do not split arbitrary text on `and`, `+`, commas, or
   parentheses.
4. Keep the exact raw cell with every derived evidence row.
5. Do not map a concentration or supplement unless the complete raw pattern has a reviewed rule.

Examples:

| Raw value | Proposed result |
| --- | --- |
| `Ampicillin` | `ampicillin` |
| `Chloramphenicol and Ampicillin` | `chloramphenicol`, `ampicillin` |
| `Bleocin (Zeocin)` | `bleocin_zeocin` |
| `Kan + DAP` | Exclude pending review |
| `Ampicillin + 1% Glucose` | Exclude pending review |
| `RNA-OUT (6% sucrose)` | Exclude |

`DAP` is a growth requirement, not an antibiotic-resistance marker. Glucose, arabinose, and sucrose
are not antibiotic-resistance markers.

## Evidence rules

- A reviewed exact mapping is verified evidence for each antibiotic in the mapped set.
- Version 1 has no contradicted evidence for this facet.
- An alternative recorded antibiotic is unknown for the query antibiotic.
- Missing, free-text, excluded, and unreviewed values are unknown.

This positive-only rule does not assume that the field is a complete list. The metadata can record
the selection condition used by Addgene without listing every functional marker under every
condition.

## Observed support and edge cases

The fixed E00 profile contains 115,114 known rows and 51 normalized values. The largest values are:

| Normalized value | Rows |
| --- | ---: |
| `ampicillin` | 80,404 |
| `kanamycin` | 21,521 |
| `spectinomycin` | 4,879 |
| `chloramphenicol` | 3,961 |
| `chloramphenicol and ampicillin` | 866 |
| `gentamicin` | 585 |
| `tetracycline` | 543 |
| `ampicillin and kanamycin` | 361 |
| `bleocin zeocin` | 356 |
| `chloramphenicol and kanamycin` | 300 |
| `apramycin` | 298 |

Observed rare values include doses, supplements, `DAP`, spelling variants, and antibiotics that
Addgene treats as free text. Support alone does not make these values eligible.

## Exclusions and audit questions

- Treat exact source value `None` as missing.
- Exclude `other` content until an explicit mapping is reviewed.
- Exclude growth supplements and growth requirements from the marker set.
- Review every one of the 51 normalized values before a mapping table is frozen.
- Check whether simple combination values describe encoded markers or only the selected growth
  condition.
- Do not use pLannotate or plasmidkit as an automatic fallback for this metadata rule.

A later sequence-feature concordance study can compare the reported marker with pLannotate-only
features. It must be a separate analysis and cannot silently change these labels.

See [manual audit version 0.1](manual_audit_v0.1.md) for the frozen sample rule.
