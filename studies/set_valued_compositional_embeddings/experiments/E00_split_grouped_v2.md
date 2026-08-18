# E00 Similarity-Closed Grouped Split v2

**Status:** complete; accepted
**Protocol version:** `split_grouped_v2_v0.1`  
**Fixed before the global graph result:** 2026-08-13 Europe/London

## Question and hypothesis

Can the accepted strict whole-plasmid similarity components be assigned to train, validation, and
test without a declared family, exact sequence, old leakage component, or strict 99% similarity
component crossing a split?

The hypothesis is that whole-component assignment can retain approximately 80% train, 10%
validation, and 10% test rows without making either evaluation split unusably concentrated.

## Fixed inputs

- Retrieval dataset: `2026-08-04T09.02.10.007Z`.
- Population SHA-256:
  `7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.
- Accepted graph protocol: `global_similarity_graph_v0.1`.
- The graph artifact version must be pinned after its independent read-back validation passes.
- No model output or benchmark score is an input.

## Component and assignment rule

Use `similarity_component_primary` from the accepted graph. This component already unions the
existing leakage components with measured whole-plasmid edges that meet all strict rules: at
least 99% identity, 95% coverage of each plasmid, and a 0.95 shorter-to-longer length ratio.

Preserve the existing `leakage_component` and `split_grouped` fields as provenance. Create a
separate mapping with:

```text
sequence_id
similarity_component_primary
leakage_component_v2
split_grouped_v2
```

Set `leakage_component_v2` equal to the stable primary similarity-component identifier. Sort the
component identifiers, shuffle them once with NumPy seed 42, and assign whole components in that
order. Use row-count targets of 80% train, 10% validation, and 10% test. Move to the next split
only at a component boundary. Do not split or repair a component after assignment.

## Independent audit

After assignment, reload the versioned mapping, graph nodes, graph edges, and pinned retrieval
dataset. Check:

- exactly 115,120 unique sequence IDs and exact set equality across inputs;
- one split label for each primary component;
- zero declared families, exact-sequence groups, or old leakage components crossing v2;
- zero strict primary graph edges crossing v2;
- deterministic row counts and content hashes after reload;
- split row counts, component counts, effective component counts, largest-component fraction, and
  ten-largest-component fraction;
- the count of sensitivity-only 95% edges that cross v2, reported as a sensitivity result rather
  than silently added to the strict grouping rule.

Fail on an unexpected join expansion, unmatched row, duplicate identifier, invalid split label,
or strict crossing. Do not overwrite the current retrieval artifact or current split.

## Acceptance and interpretation rules

Accept leakage closure only if every strict crossing count is zero. Treat a validation or test
largest-component fraction above 25% as a concentration warning that requires component-macro
reporting and may make that evaluation split unsuitable. Report target deviation caused by whole
components; do not repair it by moving individual rows.

The 95% sensitivity graph is not the primary split rule. A nonzero sensitivity-only crossing count
does not invalidate strict closure, but it must remain visible in the audit and later sensitivity
analysis.

## Outputs

- a versioned v2 sequence-to-component-and-split mapping;
- a versioned primary-component profile;
- a versioned cross-split edge audit;
- a resolved manifest with input versions, hashes, configuration, Git provenance, and decision;
- a short report that links the exact artifact versions.

## Known limitations

- Strict closure is conditional on the accepted minimap2 graph and is not a mathematical all-pairs
  proof.
- Single-linkage components can connect rows through chains of pairwise similarity.
- A large component can prevent close agreement with the target split fractions.
- Sequence dissimilarity does not establish functional independence.

## Outcome observed after execution

The accepted mapping version is `2026-08-17T23.49.47.355Z`. It contains 115,120 rows and 11,764
primary components. The split contains 92,279 training rows, 11,344 validation rows, and 11,497
test rows.

The independent audit found zero declared-family, exact-sequence, old-component, primary-component,
or primary-edge crossings. It found 6,259 sensitivity-only 95%-similarity edges that cross the v2
split, as required by the protocol. Neither validation nor test reached the 25% concentration
warning. Validation's largest component contains 3.92% of rows. Test's largest component contains
2.86%.
