# Manual Facet Audit Protocol v0.1

## Status

Frozen before row-level review. The deterministic sample now exists at output version
`2026-08-04T14.39.25.320Z`: 918 rows from 773 leakage components. No human decision or benchmark
label has been accepted. The operational model-assisted extension is defined separately in
`model_assisted_audit_v0.2.md`; this version 0.1 protocol remains unchanged as the original human
audit design.

## Question

Do the proposed mappings support their narrow Addgene metadata claims with an acceptably low error
rate?

This audit tests mapping precision. It does not test whether the plasmid works in every biological
context. It also does not estimate missing-marker recall.

## Fixed inputs

- Retrieval dataset version: `2026-08-04T09.02.10.007Z`.
- Value profile version: `2026-08-04T14.05.27.282Z`.
- Rule sheets: `addgene_copy_class.v0_1`, `addgene_growth_temperature.v0_1`,
  `bacterial_selection_marker.v0_1`, and `addgene_intended_use.v0_1`.
- Sampling key: `e00-facet-audit-v0.1`.

The aggregate E00 profile included all three splits. This was exploratory data validation and is
recorded in the experiment log. The row-level audit will use only train and validation rows. Do not
inspect test row content while changing rules. Keep the test split sealed until the rules, queries,
and pass criteria are frozen.

## Sampling unit

Sample leakage components, not only rows. Select at most one row from a component in each stratum.
This reduces duplicate evidence from related plasmids.

For each eligible row, calculate:

```text
sha256("e00-facet-audit-v0.1|<facet>|<stratum>|<component_id>|<sequence_id>")
```

Sort the hexadecimal hashes in ascending order. Take the first required number. Do not replace a
sample after review. Record a source-unavailable result instead.

For a stratum that has a minimum per canonical value, first take the required minimum within each
canonical value by the same hash order. Remove duplicate rows and components. Then fill the rest of
the stratum by the common hash order. Process canonical values in lexical order. A selected compound
row counts for each canonical antibiotic that it contains. Store all mapped antibiotics in the
audit row.

## Fixed sample

| Rule family or stratum | Required sample |
| --- | ---: |
| Copy class `high` | 75 components |
| Copy class `low` | 75 components |
| Copy class missing | 75 components |
| Growth temperature `37` | 75 components |
| Growth temperature `30` | 75 components |
| Growth temperature `23` | All available train and validation components |
| Growth temperature missing | All available train and validation components |
| Resistance: accepted single-antibiotic rules | 75 components total, with at least 5 per included canonical antibiotic when available |
| Resistance: accepted combination rules | 75 components total, with at least 5 per included canonical antibiotic when available |
| Resistance: excluded or unresolved syntax | 75 components total |
| Resistance: missing | All available train and validation components |
| Intended use: controlled expression-context values | 75 components total, with at least 5 per included value when available |
| Intended use: controlled use-category values | 75 components total, with at least 5 per included value when available |
| Intended use: `Other`, `Unspecified`, and free text | 75 components total |
| Intended use: missing | 75 components |

Also review all 51 distinct normalized resistance values once as a vocabulary audit. Review every
proposed controlled intended-use mapping once. These vocabulary reviews do not replace the sampled
row audit.

If a stratum has fewer components than required, review all available components and report that
the precision threshold is not resolved for that stratum. Do not lower the sample target.

## Audit view

Show the reviewer:

- `sequence_id`, Addgene ID, split, and leakage component;
- the exact raw source field and its canonical mapping;
- the complete source record field when a flattened value came from a list;
- the Addgene plasmid page URL and displayed growth information;
- the proposed narrow claim and evidence state;
- the rule ID and version.

Do not show retrieval ranks, model scores, or model-generated conclusions. pLannotate is not needed
to validate these four metadata claims. Do not substitute plasmidkit evidence.

## Reviewer decision

Use one of these values:

- `supported`: the source supports the exact narrow claim;
- `not_supported`: the mapping or claim is wrong;
- `ambiguous`: the source has more than one reasonable meaning;
- `source_unavailable`: the source cannot be checked.

Require a short reason for all results except `supported`. Keep the first decision. If a decision
changes during adjudication, store both decisions and the reason.

One reviewer checks all rows. A second reviewer checks a deterministic 20% subsample and all rows
that are not `supported`. Report raw agreement and Cohen's kappa, a chance-adjusted agreement
measure, for the double-reviewed rows. Do not use agreement as a substitute for source correctness.

## Pass rule fixed before review

A proposed rule family passes only when all these conditions hold:

1. No reviewed mapping makes a stronger biological claim than its source.
2. The sample has at least 75 independent components.
3. There are zero `not_supported` or `ambiguous` results among included mappings.
4. There is no systematic source-transformation error.
5. Every non-pass result is retained in the audit table and experiment log.

With zero errors in 75 independent observations, the two-sided exact 95% upper confidence bound on
the error rate is approximately 4.8%. Report the Clopper-Pearson exact binomial confidence interval
from the realized sample. Component sampling reduces dependence but does not prove independence.

The raw `23` growth-temperature mapping has an additional gate: locate and validate its source
transformation. It cannot pass only from row examples.

Contradicted evidence has a separate gate. It can pass only for copy class and the reviewed `30`
versus `37` growth-temperature rule. Resistance and intended-use rules remain positive-only even if
the positive mapping audit passes.

## Failure action

If a rule fails, do not edit the audit result. Mark the rule version failed. Write a new rule
version, draw a new deterministic sample with a new sampling key, and record the change in the
experiment log. Do not create benchmark labels from a failed or unresolved rule.

## Planned outputs

- `facet_audit_sample.parquet`: immutable sampled rows and proposed claims;
- `facet_audit_vocabulary.parquet`: every observed raw value and its proposed classification;
- `facet_audit_manifest.json`: sample counts, target checks, input identity, and configuration;
- `facet_audit_decisions.csv`: later human decisions with reviewer and timestamp;
- a short report that lists accepted mappings and unresolved exclusions.

The sample-generation and manifest steps belong in Kedro. Human decisions remain a reviewed study
artifact that Kedro reads as an explicit input. Sample generation must not write accepted labels.
