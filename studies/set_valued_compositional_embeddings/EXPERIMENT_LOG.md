# Experiment Log

This file is an append-only record of study actions and decisions. Add a new timestamped entry for
each planned run, completed run, failed run, protocol change, or interpretation decision. Do not
rewrite earlier entries. Add a correction entry when an earlier entry is wrong.

Each entry records:

- the local timestamp and time zone;
- the action and its purpose;
- the exact input artifact or source;
- the Git commit and whether the worktree was dirty;
- the configuration and commands;
- outputs and validation results;
- failures, deviations, and unresolved questions;
- the next decision.

## 2026-08-04 15:00:26 BST — E00 implementation started

**Status:** started.

**Question:** Can we reproduce the initial field, split, and pLannotate observations through a
tested Kedro pipeline before defining benchmark labels?

**Planned action:** Implement the first `constraint_semantics` profiling pipeline. It will measure
raw and normalized field support, component concentration, and pLannotate coverage and coordinate
quality. It will not create constraint labels or change the split.

**Primary input:** Latest catalog-resolved version of `retrieval_dataset` during development. The
first full run must record the exact resolved version. The review target is:

```text
s3://plasmidclip/kedro/04_feature/retrieval_dataset.parquet/
2026-08-04T09.02.10.007Z/retrieval_dataset.parquet
```

**Annotation policy:** Use only rows whose source is exactly `plannotate`. Do not fall back to
plasmidkit. Existing mixed-source artifacts retain their original provenance.

**Split policy:** Inspect the existing grouped split without changing it. This is exploratory data
validation, not test-set model evaluation. No model outputs exist or will be inspected.

**Code state:** Git commit `3c445dd`. The worktree was dirty with uncommitted research-policy and
study-layout documentation before implementation began.

**Planned validation:** Unit tests, a real Kedro session with the local fixture, `pytest`, Ruff
lint, Ruff format check, and one full read-only run against the cataloged S3 inputs.

**Known limitation:** The normalized annotation artifact does not currently record the pLannotate
software version, database version, or circular-sequence setting. The profiling output must report
this missing provenance. Later annotation-derived scientific measurements must not proceed until
the provenance is recovered or the annotations are regenerated with pinned versions.

## 2026-08-04 15:07:29 BST — E00 profiling run completed

**Status:** completed successfully.

**Command:**

```bash
.venv/bin/kedro run \
  --pipelines constraint_semantics \
  --load-versions 'retrieval_dataset@semantics:2026-08-04T09.02.10.007Z'
```

**Input retrieval version:** `2026-08-04T09.02.10.007Z`.

**Output version:** `2026-08-04T14.05.27.282Z` for all five E00 outputs.

**Runtime:** 87.0 seconds, reported by Kedro.

**Produced artifacts:**

```text
s3://plasmidclip/kedro/08_reporting/e00/
├── constraint_field_profile.parquet/2026-08-04T14.05.27.282Z/
├── constraint_value_profile.parquet/2026-08-04T14.05.27.282Z/
├── split_component_profile.parquet/2026-08-04T14.05.27.282Z/
├── split_profile.json/2026-08-04T14.05.27.282Z/
└── plannotate_profile.json/2026-08-04T14.05.27.282Z/
```

**Observed retrieval and split checks:**

- 115,120 rows and unique `sequence_id` values.
- 14,157 leakage components.
- Zero components cross the grouped split.
- The largest test component has 3,364 rows, or 29.23% of test rows.
- The ten largest test components contain 5,160 rows, or 44.84% of test rows.

**Observed pLannotate checks:**

- The catalog view contained only `source == "plannotate"` rows.
- 3,856,316 pLannotate rows exist in the normalized annotation artifact.
- 3,233,114 rows belong to the retrieval population.
- 115,072 of 115,120 retrieval plasmids have at least one pLannotate row.
- 48 retrieval plasmids have no pLannotate row and remain missing.
- 42,481 in-population rows have raw `start > end`.
- 427 in-population rows have raw `start < 0`.
- 22,517 in-population rows have raw `start == 0`.
- No coordinates were normalized or interpreted in this run.
- Software version, database version, and circular setting remain unknown.

**Observed field coverage:**

| Field | Known rows | Coverage | Normalized values | Values meeting train support |
| --- | ---: | ---: | ---: | ---: |
| `vector_types` | 113,940 | 98.98% | 1,294 | 122 |
| `bacterial_resistance` | 115,114 | 99.99% | 51 | 28 |
| `insert_species` | 99,105 | 86.09% | 2,970 | 264 |
| `plasmid_copy` | 97,208 | 84.44% | 2 | 2 |
| `growth_strain` | 115,111 | 99.99% | 203 | 42 |
| `growth_temp` | 115,120 | 100.00% | 3 | 3 |
| `insert_genes` | 50,790 | 44.12% | 11,404 | 812 |
| `insert_mutations` | 29,761 | 25.85% | 21,761 | 303 |
| `insert_tags` | 43,650 | 37.92% | 5,372 | 387 |
| `insert_promoters` | 56,250 | 48.86% | 3,994 | 301 |

Train support means at least 10 train rows and two train leakage components. It is a profiling
threshold, not an acceptance rule for benchmark inclusion.

**Validation:**

- Focused tests: 11 passed.
- Full suite before the full data run: 87 passed.
- Ruff lint: passed after one 103-character line was corrected.
- Ruff format: two new files initially required mechanical formatting; the formatter was applied.
- Ruff format check and `git diff --check`: passed after formatting.
- Independent post-run assertions checked row counts, component counts, source purity, coverage,
  artifact consistency, and the known missing provenance.

**Deviations and failures:** No pipeline node or scientific validation failed. The two local style
failures above changed formatting only and did not change an experiment definition.

**Interpretation:** The pipeline reproduces the initial exploratory observations. This does not
validate constraint semantics. High field coverage and value support do not establish biological
completeness, exclusivity, or label accuracy.

**Next decision:** Inspect the raw-value profile for the four version-1 candidate facets. Then write
facet rule sheets before creating any verified or contradicted labels.

## 2026-08-04 15:08:14 BST — Post-run repository validation

**Status:** passed.

After the full E00 run and log update:

- `pytest -q`: 87 passed;
- `ruff check .`: passed;
- `ruff format --check .`: 52 files already formatted;
- `git diff --check`: passed.

This validation checks the implemented behavior and repository composition. It does not validate
the biological meaning of any candidate constraint.

## 2026-08-04 15:13:17 BST — Candidate facet review started

**Status:** started.

**Question:** What do the four version-1 candidate fields measure, and which raw values can support
proposed verified, contradicted, or unknown evidence rules?

**Input profile version:** `2026-08-04T14.05.27.282Z`.

**Candidate fields:** `plasmid_copy`, `growth_temp`, `bacterial_resistance`, and `vector_types`.

**Method:** Inspect all normalized values and their raw cell variants. Compare the field behavior
with primary Addgene documentation where available. Write proposed facet rule sheets. Do not create
plasmid-level labels in this step.

**Bench interpretation rule:** Describe recorded propagation, selection, and depositor tags as
metadata claims. Do not promote them to universal biological capability claims.

**Stopping rule:** Stop after the four proposed rule sheets, an exclusion list, and a manual-audit
plan exist. Each rule remains proposed until a frozen manual audit passes.

## 2026-08-04 15:16:24 BST — Candidate facet rule sheets proposed

**Status:** documentation gate completed. No benchmark labels were created.

**Code state:** Git commit `3c445dd`. The worktree remained dirty with the active E00 implementation
and research documentation.

**Inputs:** Retrieval dataset version `2026-08-04T09.02.10.007Z` and E00 value-profile version
`2026-08-04T14.05.27.282Z`.

**Primary source review:** Addgene deposit instructions define `Primary Vector Type` as intended use,
`Bacterial Resistance` as the encoded antibiotic resistance used to maintain the glycerol stock,
copy class as an operational miniprep or special-growth category, and growth temperature as one of
30 degrees Celsius, 37 degrees Celsius, or room temperature. Addgene also states that depositors
provide plasmid data. The rule wording therefore stays at the recorded metadata level.

**Observed value decisions:**

- `plasmid_copy` has only `High Copy` and `Low Copy`. The proposed conflict is valid only for the
  narrow Addgene-recorded class.
- `growth_temp` has raw `37`, `30`, and `23`. The local flattening code preserves the source string.
  It does not convert room temperature to `23`. Raw `23` is held out until its upstream provenance
  is located.
- `bacterial_resistance` has 51 normalized values. Compound cells include antibiotics, doses,
  supplements, growth requirements, and spelling variants. Version 1 uses a reviewed mapping table
  and has no negative evidence for this facet.
- `vector_types` has 1,294 normalized values and mixes controlled intended-use tags with free text.
  Version 1 keeps exact controlled values, separates expression context from other use categories,
  excludes `Other`, `Unspecified`, and free text, and has no negative evidence for either facet.

**Annotation decision:** These four rules describe Addgene metadata, so pLannotate is not used to
validate them. A later sequence-feature concordance analysis must use pLannotate only and remain
separate. Plasmidkit is not a fallback.

**Audit design:** The proposed audit samples leakage components by a fixed SHA-256 order. It uses
train and validation row content only. It fixes a zero-error threshold over at least 75 independent
components for each rule family, which gives an approximate two-sided exact 95% upper error bound
of 4.8%. Raw `23` also requires a source-transformation audit.

**Outputs:** Four proposed facet rule sheets, one semantics index, and manual audit protocol v0.1 in
`studies/set_valued_compositional_embeddings/semantics/`.

**Unresolved questions:** Locate the origin of raw growth temperature `23`. Build and review the
complete resistance mapping table. Confirm that controlled and free-text intended-use values remain
distinguishable in the upstream record.

**Next decision:** Implement the deterministic audit-sample pipeline without generating accepted
constraint evidence. Then review the sample and record decisions before label construction.

## 2026-08-04 15:19:04 BST — Facet documentation validation

**Status:** passed.

- `pytest -q`: 87 passed.
- `ruff check .`: passed.
- `ruff format --check .`: 52 files already formatted.
- `git diff --check`: passed before the final wording correction.

The validation confirms repository consistency. It does not make any proposed facet rule
scientifically accepted.

## 2026-08-04 15:19:29 BST — Final documentation diff check

**Status:** passed. `git diff --check` passed after the final terminology and audit-sampling wording
changes. No data pipeline ran and no scientific output changed.

## 2026-08-04 15:22:57 BST — Deterministic facet-audit sampling started

**Status:** started.

**Question:** Can Kedro create the frozen manual-review sample without creating accepted benchmark
labels or inspecting test-row content?

**Code state:** Git commit `3c445dd`. The worktree is dirty with the active E00 pipeline and study
documentation.

**Fixed retrieval input:** `2026-08-04T09.02.10.007Z`.

**Fixed profile input:** `2026-08-04T14.05.27.282Z`.

**Sampling key:** `e00-facet-audit-v0.1`.

**Eligible splits:** train and validation only. Test-row content is excluded from the audit sample.

**Planned outputs:** An immutable row-review sample, a complete observed-value classification table,
and a manifest with target checks and a content-derived input identity. No decision table and no
accepted constraint evidence will be produced.

**Cleanup decisions:** Remove the stale fixed test count from the project README. Give the sampler
its own narrow retrieval view. Keep the sampler separate from the descriptive profiling pipeline.
Add missing-value strata so unknown evidence is checked rather than assumed.

**Stopping rule:** Stop after the fixture pipeline and repository checks pass, the pinned full run
completes, and the persisted sample and manifest pass independent validation.

### Pre-run amendment: separate antibiotic syntax strata

The first focused test showed that one pooled accepted-antibiotic sample can meet per-marker
coverage without selecting a compound source value. Before any full run or human review, the
protocol was amended to draw 75 single-antibiotic components and 75 combination components. This
keeps the riskier compound syntax visible.

## 2026-08-04 15:35:34 BST — First full facet-audit run stopped on a placeholder

**Status:** failed safely. No output artifact was written.

**Command:**

```bash
.venv/bin/kedro run --pipelines facet_audit_sample \
  --load-versions 'retrieval_dataset@facet_audit:2026-08-04T09.02.10.007Z'
```

**Failure:** The exact resistance classifier found literal source value `None`, which the proposed
mapping did not classify. The node raised `ValueError` before saving any output.

**Observed placeholder audit:**

- `plasmid_copy`: 17,912 rows contain exact `Unknown`;
- `growth_temp`: no unknown rows;
- `bacterial_resistance`: 6 rows contain exact `None`;
- `vector_types`: 1,179 rows contain an empty list and one row contains exact `N/A`.

**Correction:** Add the exact placeholder strings to explicit `missing_exact` configuration. Do not
add a general placeholder coercion. Keep unrecognized copy, temperature, and resistance values as
errors. The vector rule continues to exclude unrecognized free text by its stated version-1 policy.

**Earlier focused-test failures:** A synthetic numeric growth-temperature column first produced
`23.0`; the fixture was corrected to match the real string-valued schema. A pooled resistance
sample could omit combination syntax; this caused the pre-run amendment above. Neither failure
produced a scientific artifact.

**Next action:** Add placeholder tests, rerun the fixture and repository checks, then repeat the
same pinned full command.

## 2026-08-04 15:39:11 BST — First persisted facet-audit sample superseded

**Status:** run completed, but its artifacts are superseded before human review.

**Output version:** `2026-08-04T14.36.13.505Z`.

**Runtime:** 32.6 seconds.

**Observed outputs:** 918 sample rows across 772 leakage components. All rows were train or
validation. No stratum contained the same component twice. The manifest stated
`accepted_labels_created: false`.

**Validation failure:** The persisted vocabulary showed 51 audit-population rows with exact source
value `Affinity Reagent/ Antibody`. The proposed mapping used `Affinity Reagent/Antibody`, so the
exact classifier correctly treated those rows as excluded. No reviewer has used this sample.

**Correction:** Change the proposed mapping to the observed punctuation and whitespace. Also make
every configured canonical value explicit in the manifest, including values with zero available
components. This prevents a missing configured category from making `minimum_met` appear true.

**Other unresolved targets in this version:** Only 49 components were available for excluded
resistance syntax, below the target of 75. The combination stratum had only one hygromycin
component, below its minimum of five. These are data limitations, not pipeline errors, and must
remain visible in the next manifest.

**Artifact policy:** Do not delete or overwrite version `2026-08-04T14.36.13.505Z`. Do not use it for
review or label decisions.

## 2026-08-04 15:41:08 BST — Corrected facet-audit sample completed

**Status:** sample generation and persisted-output validation passed. Human review has not started.
No benchmark labels were created.

**Command:**

```bash
.venv/bin/kedro run --pipelines facet_audit_sample \
  --load-versions 'retrieval_dataset@facet_audit:2026-08-04T09.02.10.007Z'
```

**Input retrieval version:** `2026-08-04T09.02.10.007Z`.

**Output version:** `2026-08-04T14.39.25.320Z` for all three outputs.

**Runtime:** 30.3 seconds, reported by Kedro.

**Produced artifacts:**

```text
s3://plasmidclip/kedro/08_reporting/e00/
├── facet_audit_sample.parquet/2026-08-04T14.39.25.320Z/
├── facet_audit_vocabulary.parquet/2026-08-04T14.39.25.320Z/
└── facet_audit_manifest.json/2026-08-04T14.39.25.320Z/
```

**Manifest identity:** The input population SHA-256 is
`8d073a0570df1cb03d9f1ae43da4fe93e9cd45cd0ae4266f70a2d1e78896d94f`.

**Observed sample:**

- 918 review rows across 773 distinct leakage components;
- 819 train rows and 99 validation rows;
- zero test rows;
- zero repeated component-stratum pairs;
- 174 rows selected for deterministic second review;
- 1,298 vocabulary rows;
- 52 bacterial-resistance vocabulary rows: 33 proposed includes, 18 proposed exclusions, and one
  configured missing value;
- `accepted_labels_created` is false;
- `test_metadata_used_for_sampling` is false.

**Unresolved sample targets:**

- Excluded resistance syntax has 49 components, below the target of 75. All 49 were selected.
- The resistance-combination minimum failed for apramycin and erythromycin, with zero eligible
  components, and hygromycin, with one eligible component.
- The intended-use minimum failed for `affinity_reagent_antibody`, with three eligible components.

These shortfalls are properties of the eligible population. The configuration was not relaxed.

**Persisted validation:** Recomputed every selection hash and second-review assignment. Confirmed
unique audit row IDs, component uniqueness within strata, train-and-validation-only membership,
sample and manifest count agreement, the corrected affinity-reagent mapping, and the complete
resistance vocabulary. The first validation command had a technical `fsspec.OpenFile` read error;
the corrected read used `s3fs` and all assertions passed.

**Repository validation:** `pytest -q` passed with 92 tests. Ruff lint passed. Ruff formatting found
one test file that needed mechanical formatting; the formatter was applied and the final format,
lint, and `git diff --check` checks passed.

**Code state:** Git commit `3c445dd`; the worktree remains dirty with the active E00 implementation
and documentation.

**Next decision:** Prepare the review decision table from this exact sample version. A scientist
must review the source claims before any mapping can become accepted constraint evidence.

## 2026-08-04 16:23:12 BST — Agent-judge pilot preregistered and implemented

**Status:** implementation complete; offline validation is in progress. No paid judge request has
run. No accepted benchmark label was created.

**Question:** Can one language model help a scientist find weak or wrong proposed metadata
treatments while retaining exact evidence, uncertainty, failures, and human authority?

**Fixed input:** facet-audit output version `2026-08-04T14.39.25.320Z`. The pilot selects 30 rows
from six configured strata. It does not inspect test rows.

**Design:** One Pydantic `JudgeDecision` permits `supported`, `not_supported`, or `uncertain`.
Responses must repeat the audit-row and evidence-packet SHA-256 identities. Extra fields and invalid
JSON fail validation. The decisions table retains raw invalid responses and request errors instead
of inventing a result. Every pilot row requires human review and `accepted_labels_created` remains
false.

**Evidence boundary:** Packets contain frozen Addgene metadata, the source description, the proposed
treatment, and a provenance URL. The model cannot browse the URL. Generated descriptions are
excluded. pLannotate evidence is deferred because the current artifact has incomplete run
provenance.

**Stopping rule:** At most 30 requests. Stop after reported cumulative cost reaches USD 2.00. The
last completed request can cause a small overshoot because cost is reported after completion.

**Model configuration:** `moonshotai/kimi-k2` through OpenRouter, temperature 0, maximum 500 output
tokens. This reuses the repository's existing model choice. Confirm it before the paid run; changing
it creates a new configured pilot version.

**Offline checks so far:** Eight focused tests passed. They cover Pydantic serialization, strict
fields, deterministic packet selection, exclusion of generated descriptions and test rows, packet
identity binding, explicit invalid responses, spend accounting, and cost-cap stopping.

**Next action:** Build and inspect the 30 free evidence packets from the pinned audit artifact. Do
not run the paid judge node without an explicit decision to spend.

## 2026-08-04 16:24:43 BST — Agent-judge evidence packets persisted

**Status:** free packet-building node completed and its persisted output passed validation. The paid
judge node did not run. No accepted benchmark label was created.

**Command:**

```bash
.venv/bin/kedro run --pipeline agent_judge_pilot \
  --to-nodes build_e00_agent_judge_pilot_packets \
  --load-versions 'e00_facet_audit_sample:2026-08-04T14.39.25.320Z'
```

**Output version:** `2026-08-04T15.23.46.275Z`.

**Observed output:** 30 unique audit rows and 30 unique evidence-packet hashes. The configured
counts were eight proposed bacterial-selection exclusions, five bacterial-selection combinations,
four missing copy-class rows, four held-out growth-temperature rows, five proposed intended-use
exclusions, and four intended-use categories.

**Persisted validation:** Recomputed all evidence-packet and message SHA-256 values. Confirmed one
prompt version and hash, the pinned input version recorded on every row, valid system and user
message roles, no generated descriptions in the evidence, and zero accepted labels. Prompt hash:
`a2df379e8d98c009921c7ba7766065f8b7f4f77cd3078d6e34650ed16c61e67e`.

**Next action:** Inspect the packet contents and confirm the configured model before any paid call.

## 2026-08-04 16:26:52 BST — Agent-judge implementation validation completed

**Status:** offline implementation checks passed. No paid judge request ran.

**Observed packet content:** The 30 rows include direct and mixed cases that can expose rule errors.
Examples include `Trimethoprim`, resistance combinations with DAP or dose qualifiers, controlled
values mixed with free-text intended uses, and the held-out `23` temperature value.

**Paid-node boundary:** The judge node now recomputes each evidence and message hash, reconstructs
the exact prompt, checks the prompt and input versions, rejects accepted labels, and checks the row
count before the first API request. The persisted version `2026-08-04T15.23.46.275Z` passed these
checks for all 30 rows.

**Technical validation failure:** The first standalone boundary-check command used Kedro's
configuration loader without one selected environment. It stopped because `conf/test` correctly
overrides keys from `conf/base`. The corrected command loaded only
`conf/base/parameters_agent_judge.yml`; all packet checks then passed.

**Model availability:** OpenRouter's current model page still lists `moonshotai/kimi-k2` under that
exact slug. Availability was checked without making a completion request.

**Repository validation:** Ruff formatting and lint passed. The complete offline test suite passed
with 101 tests. `git diff --check` passed.

**Next action:** Decide whether to run the 30 paid calls. If approved, use the persisted packet
version and then review every result as a scientist. Do not accept labels automatically.

## 2026-08-04 16:29:05 BST — Qwen agent-judge smoke run preregistered

**Status:** ready to run with user authorization. No request in this smoke run has yet completed.

**Purpose:** Test instruction following, Pydantic validity, reason readability, and reported cost
before the complete 30-row pilot.

**Fixed input:** Agent-judge packet version `2026-08-04T15.23.46.275Z`. Select pilot indices 1, 5,
9, 17, 22, and 27. This gives one fixed row from each pilot stratum. Do not replace a failed row.

**Model:** `qwen/qwen3.5-397b-a17b`. OpenRouter currently describes this as a large open-weight Qwen
model. The model was selected before reading any judge output.

**Settings:** Temperature 0, maximum 500 output tokens, six requests, and a USD 0.25 stopping cap.
The API key is process input only and must not be stored in the repository, Kedro parameters,
artifacts, or this log.

**Decision rule:** Continue to the full pilot only after inspecting all six raw responses. Invalid
JSON, identity mismatch, unclear reasons, or unsupported inferences remain failures. The smoke run
does not create accepted labels.

## 2026-08-04 16:33:48 BST — Qwen smoke v0.1 completed; v0.2 preregistered

**Smoke v0.1 output version:** `2026-08-04T15.29.32.555Z` for packets, decisions, and summary.

**Observed v0.1 result:** Four of six responses passed the Pydantic contract. Two requests returned
no string in the normal answer field and were recorded as `request_error`. The four valid verdicts
were two `supported`, one `not_supported`, and one `uncertain`. The valid rows reported USD
0.010306. No accepted labels were created.

**Spend limitation:** After the run, the API key reported USD 0.01339858 of total daily usage. The
USD 0.00309258 difference from the artifact total contains the two failed requests and any earlier
same-day use of the same key. The exact failed-request cost cannot be recovered because the client
did not retain the upstream generation identifiers. Treat USD 0.010306 as a lower bound, not the
complete run cost.

**Preliminary content inspection:** The copy-class and lentiviral decisions used direct metadata and
were readable. The temperature result stayed uncertain. The Trimethoprim result correctly detected
that the direct metadata supports a canonical selection value. These are observations about four
examples, not an accuracy estimate.

**Failure diagnosis:** OpenRouter identifies this Qwen model as reasoning-enabled but not reasoning
mandatory. Reasoning tokens are output tokens. The 500-token cap can end before final JSON. The
model also supports structured outputs. This explains the response shape but is not proven because
the failed upstream payloads and generation identifiers were not retained.

**Smoke v0.2 change:** Repeat the same six fixed packets with the same model, temperature, token cap,
and USD 0.25 stopping cap. Disable reasoning and request strict JSON-schema output derived from the
same Pydantic model. Change the judge version to `e00-agent-judge-smoke-qwen35-397b-v0.2`. Preserve
v0.1 unchanged. Do not replace or hide its failures.

**Offline validation:** Eleven focused tests passed after adding explicit reasoning and structured
output fields to the OpenRouter request boundary.

## 2026-08-04 16:37:40 BST — Qwen smoke v0.2 completed; v0.3 preregistered

**Smoke v0.2 output version:** `2026-08-04T15.34.09.807Z` for packets, decisions, and summary.

**Observed v0.2 result:** Five of six responses passed the current Pydantic contract. One valid JSON
object repeated the correct audit-row ID but an incorrect evidence-packet hash. The binding check
rejected it as `invalid_response`. The five accepted structures contained four `supported` and one
`not_supported` verdict. The artifact reported USD 0.006639.

**Spend check:** API-key daily usage increased from USD 0.01339858 to USD 0.02003843 across the v0.2
run and immediate checks. The USD 0.00663985 increase agrees with the artifact after its six-decimal
rounding. This comparison assumes no concurrent use of the same key during that short interval.

**Content check:** The four decisions that were valid in both v0.1 and v0.2 kept the same verdict,
except the held-out temperature row failed identity binding in v0.2 and cannot be compared as a
valid decision. The v0.2 free-text explanation said the whole record was excluded when the proposal
applied only to one classified free-text value. One supported decision also repeated its existing
canonical value as a suggestion, contrary to the prompt. These are contract and wording defects,
not accepted scientific decisions.

**Smoke v0.3 changes:** Use prompt `agent-judge-v2`. State that the treatment applies to
`classified_source_values_json`, while the complete source field is context. Add packet-specific
`const` values for both identity fields in the API JSON schema. Make the Pydantic model reject
canonical suggestions unless the verdict is `not_supported`, and require snake_case suggestions.
Keep the model, six rows, temperature, output-token cap, and cost cap fixed.

**Stopping rule:** Build new prompt-v2 packets and rerun the same six examples once. Do not continue
to the 30-row pilot unless all six bind to the correct packet and their explanations respect the
treatment scope.

## 2026-08-04 16:40:07 BST — Qwen smoke v0.3 completed; full pilot stopped

**Smoke v0.3 output version:** `2026-08-04T15.38.33.193Z` for packets, decisions, and summary.

**Prompt-v2 packet version:** `2026-08-04T15.38.00.322Z`. All 30 free packets passed identity and
prompt reconstruction checks before the smoke run.

**Observed v0.3 result:** Five of six responses passed the tightened Pydantic contract. One response
used the correct packet identities but attached `lentiviral` as a canonical suggestion to a
`supported` decision. The model validator rejected it. Three valid decisions were `supported` and
two were `not_supported`. The artifact reported USD 0.007472. No accepted labels were created.

**Cross-run instability:** The Trimethoprim row was `not_supported` in v0.1 and v0.2, then
`supported` in v0.3. The copy-class `Unknown` row changed from `supported` to `not_supported`. These
changes followed prompt-contract revisions and show that the verdict is sensitive to procedural
wording. They are not independent repeats under one fixed protocol.

**Scope error:** On the mixed intended-use row, v0.3 stated that the proposal discarded the known
`Synthetic Biology` category. The packet's `classified_source_values_json` contained only the free
text `pMVP Gateway Entry Plasmid`; the controlled category is handled separately. The explanation
therefore misunderstood the treatment scope despite the prompt-v2 clarification.

**Conclusion:** Do not run the complete 30-row pilot with this judge contract and model
configuration. The API and Pydantic safeguards expose failures correctly, but the model is not
stable or exact enough to act as a judge. It can remain a fallible triage aid if every result is
checked by a scientist. A later test should define missing-value and exclusion semantics more
directly and compare another model on the same fixed packets.

**Repository validation:** Ruff formatting and lint passed. The complete offline suite passed with
104 tests. `git diff --check` passed.

## 2026-08-05 10:08:53 BST — Fixed-configuration judge stability test preregistered

**Status:** ready to run with user authorization. No stability repeat has completed.

**Question:** With one frozen prompt and packet set, how often does the judge return the same valid
verdict at temperature zero when it has a larger reasoning and completion budget?

**Fixed input:** Prompt-v2 packet version `2026-08-04T15.38.00.322Z`, pilot indices 1, 5, 9, 17,
22, and 27. These are the same six strata used in the earlier smoke runs.

**Fixed settings:** `qwen/qwen3.5-397b-a17b`, temperature 0.0, reasoning enabled, maximum 2,000
combined output tokens, strict structured output, and a USD 0.25 cap per six-row run.

**Replicates:** Run three repeats without changing code, prompt, packets, model settings, routing,
or validation. Each repeat must have its own Kedro output version. Do not replace failed rows.

**Primary checks:** Pydantic-valid fraction; exact verdict agreement across repeats; packet identity
agreement; and reported cost. Review reasons for semantic disagreement. This is an exploratory
stability measurement and does not create accepted labels.

## 2026-08-05 10:33:59 BST — Fixed-configuration stability test completed

**Status:** all three preregistered repeats completed. The configuration failed the stability gate.
No accepted labels were created, and the 30-row pilot did not run.

**Output versions:**

1. `2026-08-05T09.09.18.054Z`;
2. `2026-08-05T09.13.04.562Z`;
3. `2026-08-05T09.25.37.235Z`.

Each version contains its own smoke packets, decisions, and summary.

**Observed validity:** Repeat 1 produced one valid response and five request errors. Repeat 2
produced one valid response, one invalid response, and four request errors. Repeat 3 produced six
request errors. Across all repeats, 2 of 18 responses were valid. No packet had valid decisions in
all three repeats, so exact verdict agreement cannot be estimated.

**Observed runtime:** 223.6, 750.7, and 399.2 seconds. Mean runtime was 457.8 seconds; median runtime
was 399.2 seconds; the range was 527.1 seconds.

**Observed artifact cost:** USD 0.009517, USD 0.027894, and USD 0.000000, for a recorded total of USD
0.037411. The API key reported USD 0.127318443 of daily use after the runs. The USD 0.089907443
difference includes charged responses that the old client could not extract and any concurrent
same-day use of the key. It cannot be allocated exactly.

**Conclusion:** A 2,000-token output budget, reasoning enabled, and temperature 0.0 did not make this
judge reliable. Most calls consumed reasoning or returned an unusable upstream payload without a
normal final answer. Do not increase the reasoning budget again without a new hypothesis.

**Corrective code change:** Future extraction failures retain the upstream-reported cost and
generation ID when those values exist. This prevents a charged response with missing final content
from appearing free in the decision artifact. It cannot repair the historical runs.

**Next method if justified:** Disable reasoning, pin one provider, and use a fixed seed if the model
and provider honor it. Change one factor at a time. Consistency would establish repeatability, not
scientific correctness.

## 2026-08-05 10:42:08 BST — GPT-5.6 Luna smoke test preregistered

**Status:** ready to run with user authorization. No Luna decision has completed.

**Question:** Does a non-reasoning, provider-pinned GPT-5.6 Luna configuration produce valid and
scientifically readable decisions on the same six fixed packets?

**Fixed input:** Prompt-v2 packet version `2026-08-04T15.38.00.322Z`, pilot indices 1, 5, 9, 17,
22, and 27.

**Model metadata observed before the run:** OpenRouter lists the exact slug
`openai/gpt-5.6-luna`. It supports structured outputs, a seed, and configurable reasoning. Its
reported supported-parameter list does not include temperature.

**Fixed settings:** Provider `OpenAI` with fallbacks disabled, seed 17, reasoning disabled, maximum
1,000 output tokens, strict structured output, and a USD 0.25 cost cap. Do not send a temperature
parameter because this model does not advertise support for it.

**Stopping rule:** Run one six-row smoke version. Require six valid Pydantic responses with matching
packet identities before considering exact repeats. Review scientific scope separately from
structural validity. No decision becomes an accepted label.

**Provenance change:** Future decision rows record requested provider, seed, temperature, and token
limit, plus returned generation ID, model, and provider when OpenRouter supplies them.

## 2026-08-05 10:44:04 BST — Luna v0.1 rejected before generation; v0.2 preregistered

**Luna v0.1 output version:** `2026-08-05T09.42.30.706Z`.

**Observed result:** The pipeline completed in 7.7 seconds, but OpenRouter returned HTTP 400 for all
six requests. No completion ran, all six rows were `request_error`, and reported cost was zero. The
old HTTP boundary retained only status and URL, not the provider error body, so the exact upstream
message is unavailable for v0.1.

**Diagnosis:** The Pydantic schema had six properties but only five required properties because
`suggested_canonical_values` had a default. OpenAI strict structured outputs require every property
to be required. This diagnosis is based on the local schema and API contract; the discarded v0.1
error body cannot confirm it directly.

**Luna v0.2 change:** Make `suggested_canonical_values` required while still allowing an empty list.
This changes the response schema, so increment the prompt contract to `agent-judge-v3` and rebuild
the packet artifact. Keep the six rows, Luna model, provider, seed, reasoning, token limit, and cost
cap unchanged.

**HTTP boundary change:** Future non-retryable OpenRouter errors retain a bounded provider message,
HTTP status, and request ID. This change does not reinterpret v0.1.

**Stopping rule:** Run one corrected six-row version. Stop if any request is rejected or any response
fails the Pydantic contract. No decision becomes an accepted label.

## 2026-08-05 10:46:03 BST — Luna v0.2 passed smoke gate; repeats preregistered

**Luna v0.2 output version:** `2026-08-05T09.45.12.796Z`.

**Observed structure:** Six of six responses were valid. All packet identities matched. OpenRouter
returned model `openai/gpt-5.6-luna` and provider `OpenAI` on every row. The run took 17.6 seconds
and reported USD 0.001471.

**Observed decisions:** Four `supported`, one `not_supported`, and one `uncertain`. The explanations
respected the intended packet scope on the six reviewed examples. In particular, the agent treated
`Unknown` as the configured missing sentinel, kept temperature `23` uncertain, rejected the
Trimethoprim exclusion, and excluded only the classified free-text intended-use value.

**Interpretation limit:** This content check covers six selected examples. It is not an accuracy
estimate and does not accept any mapping.

**Repeat protocol:** Run two more exact repeats without changing model, provider, seed, reasoning,
token limit, prompt, schema, packets, or cost cap. Require three valid decisions per packet and
compare exact verdicts. Preserve each repeat as a separate Kedro version.

## 2026-08-05 10:48:38 BST — Luna v0.2 repeatability test completed

**Output versions:** `2026-08-05T09.45.12.796Z`, `2026-08-05T09.46.38.725Z`, and
`2026-08-05T09.47.03.757Z`.

**Structural result:** All 18 responses were valid. All packet identities matched. Every returned
model and provider identifier was `openai/gpt-5.6-luna` and `OpenAI`. No accepted labels were
created.

**Verdict result:** Four of six packets had unanimous verdicts. Growth temperature `23` produced
`uncertain`, `supported`, and `uncertain`. Trimethoprim exclusion produced `not_supported`,
`uncertain`, and `not_supported`. The other four packets had the same verdict in every repeat. No
packet changed directly between `supported` and `not_supported`.

**Agreement:** Unanimous verdict agreement was 4/6. Pairwise agreement was 4/6 between repeats 1
and 2, 6/6 between repeats 1 and 3, and 4/6 between repeats 2 and 3. Do not calculate an accuracy
score because no accepted reference labels exist.

**Cost and runtime:** Total reported cost was USD 0.002896. Runtimes were 17.6, 22.4, and 24.9
seconds.

**Conclusion:** Luna passes the structural gate for a scientist-reviewed triage job. It does not
pass as an autonomous judge because two ambiguous rows were not verdict-stable. If the 30-row pilot
runs, preserve every result, require human review, and use model disagreement or uncertainty only
to prioritize review.

## 2026-08-05 10:54:10 BST — Full 30-row Luna pilot authorized and preregistered

**Status:** ready to run with explicit user authorization. No full-pilot completion has run.

**Purpose:** Produce one provisional Luna decision for each of the 30 fixed prompt-v3 packets so a
scientist can prioritize and conduct manual review.

**Fixed input:** Agent-judge packet version `2026-08-05T09.44.22.296Z`. It contains 30 unique
prompt-v3 packets from facet-audit version `2026-08-04T14.39.25.320Z`. Do not rebuild or resample
packets in the paid run.

**Fixed model settings:** `openai/gpt-5.6-luna`, provider `OpenAI` with fallbacks disabled, seed 17,
reasoning disabled, no temperature parameter, maximum 1,000 output tokens, and strict structured
output. Judge version: `e00-agent-judge-pilot-gpt56-luna-v0.1`.

**Stopping rule:** At most 30 calls and USD 0.10 of reported cost. Preserve request errors, invalid
responses, raw valid responses, generation identifiers, and per-row cost. Do not replace a failed
row or change parameters during this run.

**Interpretation boundary:** Every row requires scientist review. Model output can prioritize review
but cannot accept a mapping, create a benchmark label, or decide whether E00 passes.

## 2026-08-05 10:58:15 BST — Full 30-row Luna pilot completed

**Status:** completed as a provisional scientist-review artifact. No label was accepted.

**Command:**

```bash
.venv/bin/kedro run --pipelines agent_judge_pilot \
  --from-nodes judge_e00_agent_judge_pilot_packets \
  --load-versions \
  'e00_agent_judge_pilot_packets:2026-08-05T09.44.22.296Z'
```

The OpenRouter credential was supplied through a silent terminal prompt. It was not written to a
configuration file, command argument, log entry, or output artifact.

**Code state:** Git commit `3c445dd`. The worktree was dirty with the current uncommitted research
project implementation and documentation.

**Output version:** `2026-08-05T09.54.34.956Z` for the decision and summary artifacts.

**Fixed settings:** `openai/gpt-5.6-luna`, provider `OpenAI` with fallbacks disabled, seed 17,
reasoning disabled, no temperature parameter, maximum 1,000 output tokens, strict structured
output, and USD 0.10 cost cap. The packet and prompt versions did not change during the run.

**Observed structure:** The pipeline completed 30 calls in 82.6 seconds. Twenty-nine responses were
valid and one was `invalid_response`. Packet identities matched on all 30 rows. All rows retained
`human_review_required = true` and `accepted_label_created = false`.

**Observed verdicts:** 21 `supported`, five `not_supported`, three `uncertain`, and one invalid.
OpenRouter reported a total cost of USD 0.006769.

**Scientific review findings:**

- Three rows with classified value `Trimethoprim` produced `not_supported`, `supported`, and
  `uncertain`.
- In DAP-containing bacterial-selection values, Luna proposed `dap` as a canonical resistance value
  in multiple rows. This conflicts with the proposed rule, which treats DAP as a growth requirement.
- The invalid response was row 12, Addgene 187394. Its otherwise structured response proposed
  uppercase `DAP`, which failed the snake-case response contract. The raw response and USD 0.000236
  cost were preserved.
- Row 7, Addgene 172199, retained a conflict between structured growth temperature 23 and a source
  description that says optimal growth at 20 degrees.
- Row 22, Addgene 121780, changed from `supported` in all three smoke repeats to `not_supported` in
  the full run. Its suggestions may exceed the classified-value scope.
- The five proposed bacterial-selection combination controls and four controlled intended-use
  controls were all structurally valid and returned `supported`.

**Interpretation:** Structural success is not scientific correctness. Luna can help prioritize a
small manual review, but it cannot adjudicate these treatments. Do not calculate accuracy because
there is no accepted human reference label.

**Next decision:** A scientist reviews the DAP, Trimethoprim, intended-use row 22, temperature row
7, and dose-qualified antibiotic row 14 first. Then the scientist reviews the remaining rows and
records an independent verdict and reason before any rule or label changes.

## 2026-08-05 11:49:57 BST — Independent stronger-model validator preregistered

**Status:** implementation started before any Claude Opus 5 request. No validator decision exists.

**Reason for change:** The user stated that a strong language model is likely to have more relevant
synthetic-biology knowledge than the available human reviewer. Human status will not be treated as
privileged evidence. The source record and narrow facet rule remain authoritative.

**Design:** Reuse the exact 30 prompt-v3 packets at version `2026-08-05T09.44.22.296Z`. Run
`anthropic/claude-opus-5` independently, without Luna decisions or generated descriptions. Compare
it later with Luna decision version `2026-08-05T09.54.34.956Z`. Do not change the sample, prompt,
response schema, or packet order.

**Fixed settings:** Provider `Amazon Bedrock` with fallbacks disabled and required-parameter routing,
reasoning enabled, no temperature, no seed, maximum 2,000 output tokens, strict structured output,
180-second request timeout, 30 calls, and USD 0.50 reported-cost cap.

**Decision policy:** Claude Opus 5 is the primary validator proposal. Luna remains an independent
first opinion. Model disagreement, validator uncertainty, invalid output, and source conflict require
manual resolution. Agreement is not accuracy, and no pipeline in this experiment creates accepted
labels.

**Scale gate:** Inspect the 30 rows, especially DAP, Trimethoprim, growth temperature, and intended
use. Do not run the validator over all 918 audit rows if it repeats an obvious domain error, produces
invalid output, or shows unexplained rule inconsistency.

**Additional artifact:** Build a versioned model-free CSV for the 918-row audit. It will retain exact
source evidence and blank human-decision fields while excluding generated descriptions and all model
conclusions.

**Code state:** Git commit `3c445dd`. The worktree is dirty with the current uncommitted research
implementation and documentation.

## 2026-08-05 11:55:21 BST — Opus validator v0.1 failed before generation; v0.2 preregistered

**Validator v0.1 output version:** `2026-08-05T10.52.22.189Z`.

**Observed result:** The Kedro pipeline completed in 42.7 seconds, but all 30 requests were
`request_error`. No model response or verdict was produced, and reported cost was zero. The failed
rows and summary were preserved.

**Diagnostic request:** One additional packet-1 request reproduced HTTP 400 at zero observed model
output. OpenRouter metadata identified the pinned provider as Amazon Bedrock. One endpoint reported
that `maxItems` is not supported in `output_config.format.schema`; another endpoint rejected
`output_config.format`. This was a response-schema transport failure, not a scientific model result.

**Error-boundary correction:** Future OpenRouter HTTP errors retain bounded provider name, typed
error, provider code, and provider detail when returned. This prevents a generic provider message
from hiding the actionable cause.

**Validator v0.2 change:** Keep the packet messages, prompt hash, semantic response fields, model,
provider, reasoning, token limit, row order, and cost cap unchanged. Send a portable JSON Schema
that contains only field types, required fields, the verdict enum, and
`additionalProperties: false`. Apply the complete `JudgeDecision` Pydantic contract locally after
generation. It still checks exact packet identities, list bounds, reason length, snake case, and
conditional suggestions.

**Version:** Change the judge version to `e00-agent-validator-claude-opus5-v0.2`. This transport
change starts a new validator version. Run the exact 30 rows once; do not reinterpret v0.1.

## 2026-08-05 12:00:07 BST — Opus validator v0.2 stopped; v0.3 preregistered

**Validator v0.2 result:** The portable schema cleared the immediate HTTP 400 failure. The first
request then exceeded the configured 180-second timeout and entered the generic client's retry loop.
The run was stopped manually before the node returned. Kedro wrote no v0.2 decision or summary
artifact. The prior v0.1 artifacts remain unchanged.

**Reason for stopping:** A generic retry can repeat a possibly charged scientific request and can
delay one row for more than 20 minutes. That behavior is not appropriate for a fixed, auditable job.
No partial model result was available to interpret.

**Validator v0.3 changes:** Pin the OpenRouter provider to `Azure`, which currently advertises
Claude Opus 5 reasoning and structured-output support. Keep the portable transport schema. Set
`max_retries: 0`, so each packet permits one request and any transport failure becomes an explicit
row. Keep the model, packet version, prompt, local Pydantic contract, reasoning setting, output-token
limit, timeout, packet order, and USD 0.50 cost cap unchanged.

**Version:** Change the judge version to `e00-agent-validator-claude-opus5-v0.3`. Before the complete
run, make one packet-1 transport diagnostic with the exact v0.3 request envelope. It is not a pilot
decision. Run the 30-row version only if that request returns a Pydantic-valid response and the
returned provider is Azure.

## 2026-08-05 12:01:03 BST — Opus validator v0.3 rejected; v0.4 preregistered

**Validator v0.3 diagnostic:** The exact packet-1 diagnostic returned HTTP 400 before generation.
Azure reported `structured_outputs not supported in your workspace`. No verdict was produced.

**Validator v0.4 change:** Keep Azure, the exact packet content, prompt and full embedded response
schema, model, reasoning, token limit, zero retries, timeout, row order, and cost cap. Do not send
OpenRouter's provider-side `response_format`. Require the returned text to pass the complete local
`JudgeDecision` Pydantic model without repair. Preserve malformed text as `invalid_response`.

**Reason:** Provider-side enforcement is an integration feature, not scientific evidence. Removing
it does not relax the accepted local decision contract. It exposes the stronger model's actual
instruction following and avoids adding a repair layer.

**Version:** Change the judge version to `e00-agent-validator-claude-opus5-v0.4`. Make one packet-1
diagnostic before the complete run. Continue only if it returns from Azure and passes Pydantic.

## 2026-08-05 12:02:25 BST — Opus validator v0.4 failed contract; Sol v0.5 preregistered

**Validator v0.4 diagnostic:** Azure generated packet-1 text without provider-side structured output.
The text failed the unchanged local Pydantic contract because `evidence_used` was absent. Under the
fixed gate, the Opus configuration will not scale to 30 rows. The diagnostic script did not persist
the raw text or completion cost before raising the validation error, so the diagnostic cost is
unknown. Do not silently treat it as zero.

**Interpretation:** This is an instruction-following failure, not evidence about the biological
verdict. Adding repair logic or weakening the required decision fields would make the validator less
auditable.

**Validator v0.5 design:** Use `openai/gpt-5.6-sol`, the flagship GPT-5.6 tier, through provider
`OpenAI` with fallbacks disabled. Use high reasoning effort, seed 17, no temperature, maximum 3,000
output tokens, strict full-schema structured output, 180-second timeout, zero retries, 30 rows, and
the existing USD 0.50 cost cap.

**Reason for model-family change:** Independence from Luna is useful, but contract-valid output and
an auditable transport are prerequisites. Sol is substantially stronger than Luna and uses the same
structured-output route that already passed this exact schema. The comparison will therefore test a
stronger model, but not independent model-family agreement.

**Version:** Change the judge version to `e00-agent-validator-gpt56-sol-v0.5`. Make one exact
packet-1 diagnostic that always records provider, generation ID, cost, raw response, and local
validation status. Run the complete 30-row gate only if it is Pydantic-valid.

## 2026-08-05 12:03:38 BST — Sol diagnostic passed; v0.6 full run preregistered

**Validator v0.5 diagnostic result:** Packet 1 returned from provider `OpenAI` as model
`openai/gpt-5.6-sol`. Generation ID was `gen-1785927797-aEJD7ia3xN6Hpl1NDJgC`. The response matched
both packet identities, passed the complete Pydantic contract, and returned `supported`. Reported
cost was USD 0.0154225.

**Cost projection:** Thirty calls at the observed diagnostic cost would total about USD 0.462675.
The preregistered USD 0.50 cap leaves too little allowance for packet-length and output variation and
could truncate the fixed sample.

**Validator v0.6 change:** Increase only the reported-cost cap from USD 0.50 to USD 0.75. Keep the
model, provider, packet version, prompt, schema, high reasoning effort, seed, temperature omission,
token limit, timeout, zero retries, and row order unchanged. The new cap is a stopping-rule change,
so increment the judge version to `e00-agent-validator-gpt56-sol-v0.6` before the full run.

## 2026-08-05 12:10:38 BST — Sol validator and comparison completed; scale gate stopped

**Validator output version:** `2026-08-05T11.04.17.330Z`.

**Observed validator result:** All 30 responses were valid and returned provider `OpenAI` and model
`openai/gpt-5.6-sol`. The run took 240.0 seconds and reported USD 0.536997. Verdicts were 14
`supported`, 15 `not_supported`, and one `uncertain`. Packet identities matched. No accepted labels
were created.

**Comparison output version:** `2026-08-05T11.08.45.901Z`, using Luna decision version
`2026-08-05T09.54.34.956Z` and the unchanged packet version
`2026-08-05T09.44.22.296Z`.

**Comparison result:** Twenty-nine rows had two valid model decisions. Eighteen verdicts agreed and
11 disagreed; Luna's invalid DAP row was incomplete. Exact agreement among valid pairs was 62.1%.
Twelve rows require rule or source resolution. Agreement is not an accuracy estimate.

**Bacterial-selection finding:** Sol mapped all three Trimethoprim rows consistently. For DAP
compounds, it retained ampicillin, kanamycin, or chloramphenicol as present and did not map DAP as a
resistance marker. It also retained erythromycin and kanamycin from the dose-qualified source cell.
This exposes overly coarse whole-cell exclusions in `bacterial_selection_marker.v0_1`.

**Growth-temperature source check:** The current Addgene pages for pilot rows 5 through 8 all show
`Room Temperature` in the Growth Temperature field, although the stored artifact contains `23`.
Addgene 172199 also says optimal growth at 20 degrees in its purpose. This supports proposing the
upstream transformation `23 -> room_temperature` while retaining the row-7 conflict note. It does
not support a canonical value named `23`.

**Intended-use finding:** Sol treated all reviewed free-text intended-use values as semantically
meaningful. The current rule excludes them because benchmark version 1 is limited to exact
controlled tags. The single judge verdict cannot distinguish semantic meaning from an intentional
scope boundary. This is a prompt and measurement-design problem.

**Scale decision:** Stop `agent-judge-v3` at 30 rows. Do not run it over all 918 audit rows. First
revise the growth-temperature and bacterial-selection rules, represent intended-use scope separately
from semantic support, and create a new protocol and deterministic audit version. The detailed report
is `reports/04_gpt56_sol_validator.md`.

## 2026-08-05 12:12:35 BST — Blinded audit export validated

**Output version:** `2026-08-05T10.51.25.201Z` at
`08_reporting/e00/facet_audit_decisions.csv`.

**Observed result:** The CSV contains 918 unique audit rows from the exact facet-audit sample version
`2026-08-04T14.39.25.320Z`. It contains no test rows, generated descriptions, Luna conclusions, Sol
conclusions, filled human verdicts, or accepted labels. `model_outputs_visible` and
`accepted_label_created` are false for every row.

**Decision schema:** `HumanAuditDecision` requires a named reviewer, timezone-aware timestamp, fixed
verdict vocabulary, and a reason for each non-supported decision. This artifact remains available
for conflict resolution, but human status is not treated as stronger subject-matter evidence than a
validated model.

## 2026-08-05 12:21:53 BST — Repository quality pass completed

**Scope:** This was an implementation-maintenance pass. It did not change source data, audit rules,
judge prompts, model settings, decisions, accepted labels, or reported scientific results.

**Code cleanup:** Removed the unused portable response-schema branch that existed for the stopped
Anthropic transport trials. The active judge still requires the complete packet-bound Pydantic JSON
schema, and its summary still records `response_schema_profile: full`. Preserve the failed-trial
reports as research provenance.

**Repository cleanup:** Moved the generated root log, Ruff cache, pytest cache, and Python bytecode
caches under `src/` and `tests/` to the system Trash. Added the test and Ruff cache paths to
`.gitignore`. Do not remove any study plan, protocol, report, failed-run record, or versioned result.

**Working agreement:** Added a practical code style guide to `AGENTS.md`. It defines direct research
code, typed and validated boundaries, table invariants, explicit failures, offline tests, and the
distinction between disposable generated files and research provenance.

**Validation:** `ruff check --no-cache .`, `ruff format --check .`, `pytest -q -p no:cacheprovider`,
`git diff --check`, and a repository credential-pattern scan passed. The test result was 112 passed.

## 2026-08-05 12:48:24 BST — Facet audit v0.2 and targeted packets completed

**Rule changes:** Created `addgene_growth_temperature.v0_2` and
`bacterial_selection_marker.v0_2`. The growth rule maps stored `23` to the categorical value
`room_temperature` and forbids reporting it as exactly 23 degrees Celsius. The selection rule adds
five complete-cell exact mappings. Each non-obvious mapping stores full antibiotic names and a
plain-language `mapping_note`. DAP remains a growth requirement and is not mapped as resistance.

**Audit output version:** `2026-08-05T11.42.05.452Z`. The deterministic v0.2 audit contains 918
unique rows and no test rows or accepted labels. The changed strata contain all 41 eligible
`room_temperature` leakage components and all 34 revised bacterial-selection components. The
selection components comprise 19 Trimethoprim, 11 `Kan + DAP`, two `Amp + Kan + Dap`, one
`Amp + Chl + Dap`, and one dose-qualified erythromycin-plus-kanamycin component.

**Blinded review export:** Version `2026-08-05T11.43.33.650Z` contains the new `mapping_note` column.
It excludes generated descriptions and model conclusions.

**Judge contract:** Replaced the single model verdict with `semantic_support` and
`benchmark_scope`, with a separate reason for each axis. Prompt version is
`agent-judge-v4-semantic-scope`. The intended-use scope cases can now be biologically meaningful
and still remain outside benchmark version 1.

**Targeted packet version:** `2026-08-05T11.45.45.519Z`. The 16 unique packets contain three growth
rows, eight revised selection rows across every exact mapping, three intended-use scope rows, and two
unchanged controls. Packet hashes and prompt hashes are fixed. No generated descriptions or
accepted labels are present.

**Paid diagnostic stop:** The one-row diagnostic stopped before a provider request because
`OPENROUTER_API_KEY` was not present in the process environment. No decision artifact was written
and no model cost was incurred. The selected one-row smoke packet was persisted as version
`2026-08-05T11.47.51.082Z`. Do not infer a model failure from this configuration failure.

**Validation:** The complete local suite passed with 114 tests. Ruff lint, Ruff format, and
`git diff --check` also passed.

## 2026-08-05 15:13:28 BST — 1Password injection and targeted v0.2 gate completed

**Credential handling:** Authenticated the installed 1Password CLI through the desktop app. Used
the existing `OpenRouter` API Credential item's concealed field to inject `OPENROUTER_API_KEY` only
into the Kedro subprocess. No plaintext key was printed, written to Git, or stored in project
configuration. An empty placeholder item created during setup was archived after the existing valid
item was identified.

**Diagnostic output version:** `2026-08-05T14.08.01.359Z`. The exact first packet returned one valid
GPT-5.6 Sol response from provider `OpenAI`. It judged the categorical `23 -> room_temperature`
mapping semantically supported and in scope without calling it exactly 23 degrees Celsius. Reported
cost was USD 0.023482. No accepted label was created.

**Complete decision output version:** `2026-08-05T14.08.54.949Z`, using the unchanged 16-packet
version `2026-08-05T11.45.45.519Z`. The run took 207.2 seconds and reported USD 0.392059, within the
USD 0.40 cap. All 16 responses were valid and returned model `openai/gpt-5.6-sol` from provider
`OpenAI`. All packet identities matched. No accepted label was created.

**Semantic result:** All 16 rows were `supported`. This includes all growth and revised
bacterial-selection packets, all three intended-use free-text packets, and both unchanged controls.

**Scope result:** Thirteen rows were `in_scope`, two intended-use rows were `out_of_scope`, and the
`Gateway Destination` intended-use row was `uncertain`. Preserve the uncertainty. The reason states
that the supplied packet did not clearly establish whether that categorical cloning role belongs to
the frozen rule.

**Interpretation:** The two-axis contract worked and the revised exact selection mappings were
consistent. This is not independent source-page verification of the growth transformation because
the packets include the reviewed mapping note. Do not scale the judge to 918 rows or accept labels
from this output. Resolve the growth-source provenance and clarify the exact controlled-value scope
before materializing accepted constraint evidence.

## 2026-08-05 15:31:05 BST — Local `.env` credential loading enabled

**Scope:** This was a local credential-handling change. It did not call a model, change scientific
parameters, or create a data artifact.

**Local secret:** Created the ignored root `.env` from the active 1Password OpenRouter item. The file
contains `OPENROUTER_API_KEY`, has owner-only permissions (`0600`), and is not tracked by Git. The
key value was not printed or added to a tracked file.

**Project behavior:** Added `python-dotenv` as a direct dependency. Kedro startup now loads the root
`.env` before OmegaConf resolves `OPENROUTER_API_KEY`. The loader uses `override=False`, so an
existing shell, continuous-integration, or job-runner value remains authoritative.

**Validation:** A process without `OPENROUTER_API_KEY` loaded the local value and passed a structural
key check. A process with a sentinel value preserved the sentinel. Ruff lint, Ruff format, and the
complete offline test suite passed. The test result was 114 passed.

## 2026-08-06 09:49:12 BST — Rule-derived training evidence v0.1 completed

**Question:** Can enabled exact metadata rules produce a useful training signal while a compact
validation sample measures accuracy without reviewing every claim?

**Protocol:** Added pipeline `constraint_evidence`. It loads only train and validation metadata from
retrieval version `2026-08-04T09.02.10.007Z`. It enables ordinary copy, 30/37-degree growth,
bacterial-selection, controlled expression, and controlled use mappings. It also enables the five
reviewed bacterial mappings. It does not enable stored growth value `23` or free-text intended-use
values. Training labels use only train. Benchmark applications use only validation. pLannotate is
the sole annotation source; there is no plasmidkit fallback.

**Output version:** `2026-08-06T08.44.42.865Z`. The offline run completed in 80.6 seconds. It created
375,819 training claims across 92,097 sequences and 11,456 leakage components. It created a fixed
240-application validation sample across 237 sequences and 178 components. No model call ran. No
benchmark label was created. The catalog filter loaded zero test rows.

**Git state:** The run used a dirty worktree based on `fa02968`. The production implementation and
configuration did not change after the run and were committed as `7e66cf4`. Later changes were tests
and research documentation.

**Coverage:** Exact mappings covered 83.79% of non-null copy units, 99.91% of growth-temperature
units, 99.96% of bacterial-resistance units, and 91.36% of vector-type units. The unmatched values
remain explicit in the manifest. The output contains no `room_temperature` claim and no DAP
canonical value.

**Sampling note:** The representative validation sample contains no rare `reviewed_mappings` rows.
The fixed 16-packet targeted gate already covers all five reviewed bacterial mappings. Do not enlarge
the accuracy sample only to repeat that targeted check.

**Validation:** The focused library tests passed. The complete suite passed with 117 tests before
the final test-environment pipeline check. The full S3 pipeline and a local Kedro test-environment
composition run both completed. Ruff lint, Ruff format, and `git diff --check` passed before the
documentation update.

**Next decision:** Build and inspect fixed judge packets for the 240 validation applications. The
previous mean cost gives a rough USD 5.88 extrapolation, but packet size differs. Run one small paid
diagnostic and set a capped complete-run budget only after measuring its cost.

## 2026-08-06 09:54:18 BST — Thirty validation applications hand-checked

**Selection:** Selected six applications per facet from validation sample output
`2026-08-06T08.44.42.865Z`. Within each facet, distinct raw-to-canonical mappings were selected
first, then stable hash order filled the remainder. The selection identity SHA-256 is
`ba823d9de942f1ebd2c73592b1830eb8e063e2f062d1149dbff1c3a3db99cbb8`.

**Review:** All 30 mappings passed. The review checked the exact raw value, canonical value,
relation, source description, complete local metadata where needed, and supplementary pLannotate
features for bacterial-selection claims. Generated descriptions were not used.

**Cautions:** Addgene 1164 and 22539 have missing short source descriptions, but their controlled
tags and feature evidence are consistent. Addgene 87519 has a generic short description, but its
complete tags include TALE DBD library preparation. Addgene 51833 has pLannotate `SmR` rather than a
specific spectinomycin name; the exact Addgene selection field remains the primary evidence. None of
these cases contradicts the limited relation used by the rule.

**Decision:** The sanity check passes. It does not estimate population precision. Proceed to fixed
judge-packet preparation and a paid diagnostic. Review only uncertain responses and systematic
errors after the complete benchmark.

## 2026-08-06 10:19:58 BST — Constraint accuracy benchmark completed

**Packet preparation:** Production judge implementation is Git commit `400a47d`. Fixed packet output
version is `2026-08-06T08.58.54.091Z`. The 240 unique packets use prompt
`constraint-benchmark-judge-v1` with hash
`491cd43a849cb74d624f0c00c4ab1b6b740d6d3107f1f43c7108f728140019c4`. No packet contains a
generated description or accepted label. Mean message size was 7,852 characters.

**Smoke:** Output version `2026-08-06T08.59.59.220Z` contains one application per facet. All five
responses were valid, `supported`, and `in_scope`. Reported cost was USD 0.117088. The smoke passed
the preregistered gate.

**Complete run:** Decision output version is `2026-08-06T09.01.02.445Z`. The run used a clean
worktree, model `openai/gpt-5.6-sol`, provider `OpenAI`, high reasoning, strict structured output,
zero retries, and a USD 7.50 cap. It completed in 1,073.7 seconds and cost USD 4.670470.

**Result:** All 240 responses were valid. All 240 semantic decisions were `supported`, and all 240
scope decisions were `in_scope`. There were no uncertain, unsupported, out-of-scope, invalid, or
manual-review rows. The model-reference pass fraction is 1.0 with a 95% Wilson interval of 0.984246
to 1.0. The judge pipeline created no accepted benchmark labels.

**Evidence check:** Every `evidence_used` base field resolves to a supplied packet field. The model
sometimes serialized a reference as `field: value`; record this as a formatting variation rather
than unsupported evidence.

**Decision:** Accept rule contract
`aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2` for noisy training
supervision. This is not biological ground truth. Keep disabled and unknown values unlabeled. Do not
add claim-by-claim review. Proceed to query and candidate-gallery construction.

## 2026-08-06 13:48:45 BST — Repository maintenance after constraint validation

**Scope:** Refactor repeated stable-JSON, SHA-256, and exact metadata-key helpers into shared library
functions. Remove the registered v3 pilot, smoke, validator, and comparison code paths because they
expect the superseded one-axis response contract. Preserve their configuration, catalog entries,
persisted outputs, reports, and earlier Git implementation for research provenance. Keep the v4
targeted check and the current constraint benchmark registered.

**Identity checks:** The constraint benchmark prompt hash remains
`491cd43a849cb74d624f0c00c4ab1b6b740d6d3107f1f43c7108f728140019c4`. The accepted rule
contract hash remains `aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2`.
No data pipeline or paid request ran during this maintenance.

## 2026-08-06 14:21:36 BST — Frozen plasmid-constraint state v0.1 built

**Status:** complete data product within active E00. Gate 0 remains active.

**Question:** Can the accepted exact positive mappings and the two reviewed conflict rules produce
a stable sparse verified/contradicted state table without converting missing metadata into a
negative label?

**Input:** retrieval dataset version `2026-08-04T09.02.10.007Z`. Positive rule contract
`aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2` was accepted before applying
it to test metadata. Test data did not select mappings or conflict groups.

**Implementation:** added a separate `constraint_state` pipeline. It writes a content-addressed
constraint vocabulary, one unique row per materialized plasmid-constraint state, and a validation
manifest. Unknown is represented by absence. Only high/low copy class and numeric 30/37 degree
propagation temperature create contradictions. The completed E01 outputs were not changed.

**Run:** output version `2026-08-06T13.19.13.181Z`; dirty worktree based on `c3b26ae`; runtime 68.4
seconds. The build produced 35 constraints, 472,765 verified states, 212,222 contradicted states,
and 684,987 total states. All 115,120 sequences have verified evidence; 115,089 have at least one
contradicted state. No pair has both states. Four states preserve two exact case-variant evidence
values rather than duplicating the state.

**Checks:** 124 tests passed. Ruff lint and formatting passed. `git diff --check` passed. The saved
S3 artifacts were loaded back and checked for row identity, support counts, conflicts, and evidence
provenance.

**Decision:** accept state protocol `e00-plasmid-constraint-state-v0.1` as the input to later query
construction. Do not yet freeze queries. Run the global near-duplicate and split-concentration
audit first. Positive-only facets remain ineligible for a primary atomic query that requires a
nontrivial contradiction set.

## 2026-08-06 14:25:50 BST — State build input identity enforced and rerun

**Correction:** The successful output `2026-08-06T13.19.13.181Z` recorded the input population
SHA-256 but did not reject a future mismatched population before construction. Preserve that output
as provisional. Do not use it for later query work.

**Change:** Added the expected retrieval population SHA-256
`7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5` to configuration and made a
mismatch a hard error. The local fixture overrides this value with its own measured identity.

**Final run:** Pinned input version `2026-08-04T09.02.10.007Z`; output version
`2026-08-06T13.24.44.354Z`; runtime 65.3 seconds. Counts and invariants match the provisional run:
35 constraints, 684,987 states, and zero pair-state conflicts. Use this output version for later
work.

## 2026-08-06 14:28:44 BST — Constraint source content added to input identity

**Correction:** Population identity covers sequence IDs, sequence hashes, components, and splits.
It does not detect a changed constraint source value when those identities remain fixed. Preserve
output `2026-08-06T13.24.44.354Z` as a second provisional artifact. Do not use it for query work.

**Change:** Added a second deterministic SHA-256 over row identity plus `plasmid_copy`,
`growth_temp`, `bacterial_resistance`, and `vector_types`. The production identity is
`65feac686141b7c9a22179324a9c84a87e28e14f2d044aa56f6b92f147c2d376`. A source-value change is now
a hard error even when the population and split are unchanged.

**Accepted run:** Pinned retrieval version `2026-08-04T09.02.10.007Z`; output version
`2026-08-06T13.27.47.937Z`; runtime 56.0 seconds. Counts and invariants remain unchanged. Use this
version for later work.

## 2026-08-06 — E00 split audit technical stop

The first complete `split_audit` attempt was stopped after approximately 90 minutes. BLAST had not
completed `val_vs_train`, the first of three search pairs. The temporary result file remained
empty. Kedro saved no audit output, and no sequence-similarity result was inspected. This was a
technical runtime stop, not a valid poor result or a split decision.

The dated execution amendment in
[`E00_split_similarity_audit.md`](experiments/E00_split_similarity_audit.md) changes only BLAST
thread scheduling and the explicit fail-on-saturation target cap. Scientific thresholds remain
unchanged.

The amended direct-BLAST benchmark was also stopped after more than three minutes without
completing 100 queries against the full train database. Mash k=11, k=12, and k=21 prefilters were
rejected for recorded warning or candidate-selectivity failures. The active execution uses
minimap2 2.31 to search every query globally and reports a lower-bound edge table. This change was
made before the full minimap2 result. Identity, coverage, length-ratio, circular-query, and split
decision thresholds remain unchanged.

## 2026-08-06 17:51:32 BST — E00 split audit completed; current split rejected

**Status:** completed audit with a failed current-split decision. The edge table is a lower bound.

**Input:** retrieval dataset `2026-08-04T09.02.10.007Z`, population SHA-256
`7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.

**Run:** minimap2 2.31-r1302, `asm20`, 10 threads, doubled circular queries, 10 secondary
alignments, and a 0.5 minimum secondary score ratio. Kedro runtime was 2,926.3 seconds. Git commit
was `c3b26ae` with a dirty worktree. No model outcome was inspected.

**Output version:** `2026-08-06T16.02.42.779Z` for the edge table, component profile, and manifest.

**Observed lower bounds:** 333,686 candidate edges, 7,624 primary edges, and 13,751 sensitivity
edges, involving 4,310 plasmids and 1,459 current components. The primary edges alone form at
least 357 augmented components that cross original splits; the largest has 12,483 rows. The test
split's largest current component is already 29.23% of its rows (44.84% for the ten largest), with
a row-weighted effective component count of only 11.12.

**Decision:** reject the current grouped split for model evaluation. Preserve it and the audit as
provenance. Do not build v2 directly from the lower-bound edge graph. Build complete global
similarity closure, write a separately named v2 split, and re-audit it before query freezing.

## 2026-08-10 10:39:48 BST — E00 similarity-graph calibration completed

**Status:** completed successfully; proceed to a bounded adaptive full design.

**Inputs:** retrieval dataset `2026-08-04T09.02.10.007Z`, split-audit edges
`2026-08-06T16.02.42.779Z`, population SHA-256
`7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`.

**Run:** local Ray 2.55.1, two workers, four minimap2 threads per worker, deterministic
1,024-query sample, approximate caps 10/100/1,000, a 64-query adaptive cap-10,000 tail, and a
32-query exact benchmark at caps 10 and 1,000. Runtime 287.9 seconds. Git commit `c3b26ae`, dirty
worktree. Accepted output: `2026-08-10T09.34.59.159Z`.

**Observed:** candidate cap 1,000 saturated 67 of 1,024 queries; the adaptive cap-10,000 tail
saturated none of 64. Exact cap 1,000 saturated 1 of 32 stress-balanced queries; sensitivity edges
rose from 50 at cap 10 to 522 at cap 1,000 with no cap-10 edge lost.

**Derived:** candidate cap 1,000 projects to 2.29 CPU-hours / 2.40 GB raw PAF for the full
population; exact cap 1,000 projects to 33.51 CPU-hours / 1.73 GB. Both stay well under the
preregistered 500 CPU-hour / 250 GB ceiling.

**Decision:** freeze an adaptive full protocol — candidate cap 1,000 to route dense queries, exact
cap 1,000 for ordinary queries, exact cap 10,000 for flagged queries, fail on any final high-cap
saturation. Do not assign a v2 split until the full graph and components pass validation. (One
misrouted CIGAR assertion was caught and fixed by the synthetic gate before this run; no other
technical failures.)

## 2026-08-10 17:58:58 BST — E00 first full similarity-graph run failed

**Status:** technical failure; no graph output saved.

**Run:** `kedro run --pipeline similarity_graph` against pinned retrieval `2026-08-04T09.02.10.007Z`
and calibration `2026-08-10T09.34.59.159Z`; local Ray 2.55.1, two workers, four minimap2
threads/worker. Started 10:51:30 BST, failed 17:58:58 BST: candidate caps 1,000 and 10,000 and
ordinary exact cap 1,000 all completed; in exact cap 10,000, shard 1 hit the fixed 1,800-second
task limit.

**Decision:** keep all data, caps, thresholds, and acceptance rules fixed. Shrink adaptive shards
from 128 to 32 queries, run eight one-thread workers instead of two four-thread workers, and add
hash-validated per-shard checkpoints so a future failure loses at most one shard. Full amendment:
[`E00_global_similarity_graph.md`](experiments/E00_global_similarity_graph.md).

## 2026-08-10 21:34:40 BST — E00 checkpointed retry interrupted

**Status:** technical failure; no graph output saved.

**Observed:** the checkpointed retry finished all 900 candidate cap-1,000 shards, all 152
candidate cap-10,000 shards, and 170 of 862 ordinary exact cap-1,000 shards, then the Ray driver
called its own `ray.shutdown()` at 21:34:40 BST with zero failed tasks recorded. The Kedro
terminal session was lost, so the trigger is unknown; 95 GiB was free on the data volume.

**Decision:** resume unchanged as a detached job, so a lost terminal session cannot end the driver
again. Reuse only checkpoints that pass identity, file-hash, and row-count validation. Amendment:
[`E00_global_similarity_graph.md`](experiments/E00_global_similarity_graph.md).

## 2026-08-12 — E00 checkpoint backup and unfinished exact-shard amendment

**Observed:** the detached resume failed after 30.6 minutes when exact cap-1,000 shard 141 hit the
1,800-second limit, after adding 51 more completed shards. All 1,273 retained checkpoints (900
candidate cap-1,000, 152 candidate cap-10,000, 221 exact cap-1,000) passed independent identity,
hash, and row-count validation — 148,252 query-profile rows and 263,573 parsed edge rows, no
failure.

**Backup:** validated checkpoints, partial output, logs, code, and configuration archived to a new
encrypted S3 prefix; local byte counts and MD5 values matched every uploaded object. Full manifest:
[`E00_global_similarity_graph.md`](experiments/E00_global_similarity_graph.md).

**Decision:** preserve completed 128-query exact checkpoints. Split only unfinished exact shards
into parts of at most 32 queries. No scientific parameter changes.

## 2026-08-13 — E00 dense exact failure and isolated-query retry decision

**Status:** technical failure; no graph output saved.

**Observed:** the shard-32 run (started 13:43:19 BST on 2026-08-12) completed exact cap 1,000 for
all 110,276 ordinary queries and 46 of 152 dense cap-10,000 parent shards (1,472 of 4,844 routed
queries) before shard 34 hit the 1,800-second limit at 17:16:42 BST. All 3,882 checkpoints later
passed hash and row-count validation.

**Backup correction:** the 2026-08-12 backup has the target FASTA but no completed target-index
object — treat it as evidence, not a standalone restore. A fresh, self-contained backup (target
index, checkpoints, code, and Git provenance; manifest in
[`E00_global_similarity_graph.md`](experiments/E00_global_similarity_graph.md)) completed before
this retry.

**Decision:** preserve every completed checkpoint. Split only unfinished dense cap-10,000 shards
into one-query units. No scientific parameter changes.

## 2026-08-13 — E00 free-disk stop and guarded-resume decision

**Status:** technical stop; no graph output saved.

**Observed:** the isolated-query retry (started 16:59:11 BST) added 930 one-query checkpoints on
top of the 46 completed dense parent shards, covering 2,402 of 4,844 routed dense queries with no
search, parser, or checksum failure. Unrelated Docker workloads on the same host dropped free disk
below the fixed 40-GB floor to roughly 24.6 GB, so the process was stopped under the preregistered
safety rule (`SIGTERM`, since `SIGINT` did not stop the Ray wait); only 8 unfinished one-query
tasks were lost. Checkpoints synced to S3 by 20:28:04 UTC.

**Decision:** add a driver-side disk check while Ray tasks are pending, cancelling unfinished tasks
if free space crosses the 40-GB floor, and require 60 GB free for ten consecutive minutes before
starting the next retry. All other scientific and execution limits unchanged. Full amendment:
[`E00_global_similarity_graph.md`](experiments/E00_global_similarity_graph.md).

## 2026-08-13 — E00 similarity-closed split v2 protocol and implementation fixed

**Status:** implemented and tested; not run because no accepted final graph artifact exists.

**Decision before graph result:** use stable primary 99% similarity components as indivisible v2
leakage components. Preserve the old split. Assign whole components with the original 80/10/10
targets and seed 42. Report 95% sensitivity-only crossings separately rather than adding them to
the primary split rule after seeing results.

**Implementation evidence:** the `similarity_split` Kedro pipeline writes a versioned mapping,
component profile, cross-split sensitivity table, and manifest. Its second node independently
rejoins the pinned graph and retrieval population and fails on identifier mismatch, join expansion,
group crossing, or strict edge crossing. Independent S3 read-back validators are prepared for both
the graph and the v2 split. The full local suite passes with 157 tests; Ruff lint, format, Git
diff, and Kedro registry checks also pass.

**Protocol:** [`E00_split_grouped_v2.md`](experiments/E00_split_grouped_v2.md).

## 2026-08-13 — E00 frozen query benchmark protocol and finalizer implemented

**Status:** implemented and tested; not run because the graph and v2 split are not complete.

**Decision before graph and v2 results:** freeze version 0.1 as canonical symbolic atomic and
two-facet conjunction queries, selected only from v2 training evidence. Keep positive-only atoms
but mark their contradiction control unavailable. Use fixed row and v2 component support floors
for both query construction and the separate Gate 0 data-support decision. No paraphrases, triples,
generated descriptions, or source-conditioned queries in this version.

**Implementation evidence:** the `query_benchmark` pipeline requires pinned retrieval,
constraint-state, graph, and split artifact versions, and writes a query catalog, four candidate
gallery views, sparse verified/contradicted query-candidate states, four normalized base measures,
control top ranks, metrics, and a manifest. An independent validator recomputes answer sets from
the frozen source states and checks hashes, gallery counts, base-mass normalization, and
verified-first/contradiction-first behavior. Five query tests and three finalizer tests pass. The
post-graph finalizer now chains graph read-back, v2 construction and read-back, and query benchmark
construction and read-back as one failure-visible run.

The detached retry supervisor now rearms only on the exact wall-limit or disk-floor technical
message, keeps a separate log per attempt, syncs checkpoints before each retry, reapplies the
60-GB/ten-minute start gate, and stops after four retries. It replaces the earlier one-shot
watcher without disturbing the active graph or checkpoint mirror. Two classification tests pass.

**Backup:** encrypted S3 object `code-and-records-gate0-ready.tar.zst` (326 source, test,
configuration, script, and study paths; 693,072 bytes) — read back and hash-verified at SHA-256
`17f5518b60e32f0b4d7411e9ac1c74b926426d9a9bdfad8926ea2ecfe96a1ddf`.

**Protocol:** [`E00_query_benchmark_v0.1.md`](experiments/E00_query_benchmark_v0.1.md).

## 2026-08-17 — E00 graph execution moved to a dedicated host; restored from backup

**Status:** execution-only change; no scientific parameter changed.

**Observed:** the prior four graph attempts (2026-08-10 through 2026-08-13) all ran on the
10-core Mac research host and repeatedly hit its local disk floor. That host currently has 49 GB
free at its root, which is why every retry kept crossing the 40 GB minimum.

**Decision:** move execution to a dedicated 16-vCPU AWS EC2 host (`g6-big`, us-east-1), not
previously used for this study. Cloned the repository fresh at `d38d838`, installed minimap2
`2.31-r1302` (identical to the pinned protocol version) and `ray==2.55.1` via the project's
`similarity-graph` extra, and confirmed 164 tests pass. The project checkout lives on the host's
root EBS volume (74 GB free at setup), not its separate, unrelated, and nearly-full ephemeral NVMe
volume that hosts other active research projects — no cleanup of that volume was needed or
performed.

**Restore:** downloaded `resume-state.tar.zst` from the 2026-08-13T15-45-08Z backup prefix,
verified its SHA-256 against `backup-manifest.json`, and extracted it into
`data/09_scratch/similarity_graph_calibration` per the manifest's own restore command. This
recovered all 900 candidate cap-1,000, all 152 candidate cap-10,000, and 46 of 152 exact
cap-10,000 shard checkpoints, plus the 858 MB target FASTA and 1.75 GB minimap2 index (both
SHA-256-verified against the manifest). Then synced the newer
`in-progress-checkpoints/exact-cap10000/` prefix on top, bringing dense exact coverage from 46 to
1,093 checkpointed shard directories (2,402 of 4,844 routed dense queries, matching the last
free-disk-stop entry). Root disk free space after restore: 65 GB.

**Execution amendment:** raised `execution.ray_workers` from 8 to 12 in
`conf/local/parameters_similarity_graph.yml` to use more of the new host's 16 vCPUs, leaving 4 for
OS/Ray driver/S3 sync overhead. No cap, threshold, timeout, or acceptance rule changed. Updated
`scripts/supervise_similarity_data.py` and `scripts/finalize_similarity_data.py` in place, per
existing project convention, to point at a new session (`vec2vec-graph-autoresume-20260817`) and
backup prefix (`2026-08-17T08-48-41Z`) rather than the exhausted 2026-08-13 one.

**Next:** resume `kedro run --pipeline similarity_graph` under the retry supervisor on `g6-big`.
