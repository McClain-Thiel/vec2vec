# Hand Check of 30 Validation Applications

## Conclusion

All 30 selected mappings pass the hand check. No row changes the meaning of its source value, adds
an unsupported canonical value, or overstates the relation. Two rows have weak short-description
corroboration, but their complete metadata and relation remain consistent. No systematic problem
blocks the model accuracy benchmark.

This is a sanity check, not the accuracy benchmark. It checks 30 deliberately varied applications
and does not estimate population precision.

## Frozen selection

- Input sample version: `2026-08-06T08.44.42.865Z`.
- Selection: six applications per facet.
- Within each facet: select distinct raw-to-canonical mappings first, then fill by stable hash order.
- Selection identity SHA-256: `ba823d9de942f1ebd2c73592b1830eb8e063e2f062d1149dbff1c3a3db99cbb8`.
- Generated descriptions were not inspected.
- pLannotate was supplementary evidence. Its absence or ambiguous feature name was not treated as a
  contradiction to an exact Addgene metadata field.

## Decisions

| Index | Addgene | Facet | Raw value | Canonical value | Decision | Review note |
| ---: | ---: | --- | --- | --- | --- | --- |
| 1 | 167334 | Copy class | High Copy | high | Pass | Exact controlled value; relation is `reported_as`. |
| 5 | 221517 | Copy class | High Copy | high | Pass | Exact controlled value; relation is `reported_as`. |
| 9 | 175045 | Copy class | High Copy | high | Pass | Exact controlled value; relation is `reported_as`. |
| 10 | 89757 | Copy class | High Copy | high | Pass | Exact controlled value; relation is `reported_as`. |
| 12 | 118538 | Copy class | High Copy | high | Pass | Exact controlled value; relation is `reported_as`. |
| 19 | 202248 | Copy class | Low Copy | low | Pass | Exact controlled value; relation is `reported_as`. |
| 7 | 1164 | Expression context | Yeast Expression | yeast | Pass | Source description is missing; the controlled tag and yeast elements are consistent. |
| 8 | 163701 | Expression context | Mammalian Expression | mammalian | Pass | Controlled tag; AAVS1 knock-in description is consistent. |
| 36 | 123417 | Expression context | Bacterial Expression | bacterial | Pass | Exact controlled tag; short description gives no conflict. |
| 62 | 215197 | Expression context | Plant Expression | plant | Pass | Barley promoter description corroborates the tag. |
| 72 | 154328 | Expression context | Worm Expression | worm | Pass | Exact controlled tag; generic CRISPR description gives no conflict. |
| 113 | 125003 | Expression context | Insect Expression | insect | Pass | Exact controlled tag; guide targets are consistent with insect use. |
| 3 | 161733 | Growth temperature | 30 | 30_c | Pass | Exact numeric value; relation is limited to reported propagation. |
| 4 | 111284 | Growth temperature | 37 | 37_c | Pass | Exact numeric value; relation is limited to reported propagation. |
| 27 | 177965 | Growth temperature | 37 | 37_c | Pass | Exact numeric value; relation is limited to reported propagation. |
| 31 | 204404 | Growth temperature | 37 | 37_c | Pass | Exact numeric value; relation is limited to reported propagation. |
| 44 | 190028 | Growth temperature | 30 | 30_c | Pass | Exact numeric value; relation is limited to reported propagation. |
| 48 | 114598 | Growth temperature | 37 | 37_c | Pass | Exact numeric value; relation is limited to reported propagation. |
| 11 | 124771 | Use category | Lentiviral | lentiviral | Pass | Description explicitly states third-generation lentiviral plasmid. |
| 16 | 196417 | Use category | AAV | aav | Pass | Description explicitly states AAV use. |
| 24 | 225336 | Use category | RNAi | rnai | Pass | shRNAmir and RNA-interference description corroborate the tag. |
| 28 | 87519 | Use category | TALEN | talen | Pass | Complete tags include TALE DBD library preparation; relation is only `tagged_for`. |
| 34 | 179919 | Use category | CRISPR | crispr | Pass | Cas9 and guide-RNA description corroborate the tag. |
| 38 | 22539 | Use category | Retroviral | retroviral | Pass | Description is missing; retroviral tag, LTR, gag, and env evidence are consistent. |
| 2 | 200063 | Bacterial selection | Ampicillin | ampicillin | Pass | Exact drug name; pLannotate includes AmpR. |
| 6 | 210637 | Bacterial selection | Ampicillin | ampicillin | Pass | Exact drug name; pLannotate includes AmpR. |
| 23 | 139762 | Bacterial selection | Kanamycin | kanamycin | Pass | Exact drug name; pLannotate includes KanR and kanMX. |
| 129 | 217677 | Bacterial selection | Bleocin (Zeocin) | bleocin_zeocin | Pass | Exact synonym mapping; pLannotate includes `ble` and `bleMX6`. |
| 155 | 51833 | Bacterial selection | Spectinomycin | spectinomycin | Pass | Exact drug name; pLannotate has `SmR` and AAC/AAD evidence but is not used as the primary label. |
| 211 | 188983 | Bacterial selection | Chloramphenicol | chloramphenicol | Pass | Exact drug name; pLannotate includes CmR and a `cat` promoter. |

## Decision

Proceed to fixed judge-packet preparation. Run a paid diagnostic before the complete 240-application
benchmark. Preserve invalid and uncertain responses; review only those responses and any repeated
error pattern.
