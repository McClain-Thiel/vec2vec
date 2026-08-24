# ADR-002: Use pLannotate as the Primary Annotation Source

## Status

Accepted.

## Context

The current processing pipeline normalizes pLannotate and plasmidkit into one long annotation
table. It also creates a combined feature list. This is useful for broad data preservation, but a
merged source is hard to interpret in coordinate, coverage, geometry, and edit measurements.

The research study needs one defined measurement source. The current retrieval population has
pLannotate rows for 115,072 of 115,120 plasmids, or 99.958% coverage.

## Decision

Use pLannotate as the only primary source for new annotation-derived research measurements,
including:

- feature identity and coverage;
- coordinate intervals;
- cargo or scaffold masking;
- sequence-geometry features that use annotations;
- putative edit intervals and semantic differences.

Select source rows explicitly. A missing pLannotate result stays missing. Do not fill it from
plasmidkit.

Use plasmidkit only in a separately named concordance, sensitivity, or error analysis. Never merge
its result into the primary estimate.

Keep the existing mixed-source annotation table and generated artifacts unchanged. Preserve their
mixed-source provenance. Create distinct, versioned pLannotate-only catalog products for new work.

Pin pLannotate software and database versions. Record whether input sequences are treated as
circular. Preserve source coordinates and strand beside normalized intervals.

## Alternatives Considered

### Merge both sources

This increases apparent feature coverage but makes disagreements and coordinate conventions part of
the measurement without a clear rule. Reject it for primary research measurements.

### Use plasmidkit when pLannotate is missing

This reduces missingness but changes the measurement method for a small, non-random subset. Reject
silent fallback. A sensitivity analysis can measure the effect explicitly.

### Remove plasmidkit from ingestion

This would discard useful provenance and break the historical meaning of existing artifacts.
Reject deletion. Keep the raw and normalized source available but outside primary measurements.

## Consequences

Positive:

- primary measurements have one interpretable annotation method;
- missingness is visible;
- source disagreement can be studied instead of hidden;
- coordinate normalization needs only one primary convention.

Negative:

- 48 retrieval plasmids have no primary annotation evidence;
- existing generated descriptions are not pLannotate-only;
- pLannotate database limitations remain measurement limitations;
- a separate catalog view and provenance fields are required.

## Revisit When

- a manual validation shows that another source is more accurate for a defined feature class;
- a curated annotation source becomes available;
- pLannotate coverage or database provenance changes materially;
- source-specific ensembles can be evaluated without obscuring the primary estimate.
