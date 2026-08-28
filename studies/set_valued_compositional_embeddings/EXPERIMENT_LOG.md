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

## 2026-08-17 — E00 global similarity graph accepted; four execution bugs found and fixed

**Status:** graph accepted. Four non-scientific bugs found and fixed along the way; no cap,
threshold, timeout, or acceptance rule changed.

**Run:** resumed on `g6-big` at `ray_workers: 12`. The dense exact cap-10,000 stage (4,844 routed
queries) reached 98% (4,758/4,844) before hitting the fixed 12-hour `full_run_wall_limit_seconds`.
The supervisor correctly classified this as a retryable technical stop, synced checkpoints, and
launched a retry once free disk held above 60 GB for ten consecutive minutes. The retry reused the
prior checkpoints (same host, same absolute paths this time) and completed in 396.9 seconds:
`Pipeline execution completed successfully`, 4,450,238 primary and 4,676,653 sensitivity edges over
all 115,120 plasmids, 164.98 CPU-hours total, output version `2026-08-17T22.59.04.326Z`.

**Bug 1 — success misclassified as failure:** `scripts/supervise_similarity_data.py` matched a
literal `"Pipeline execution completed successfully"` substring, but Kedro's Rich console renderer
hard-wraps that exact phrase across two lines (with an unrelated `module.py:line` gutter injected
at the break) when output is redirected to a file. The supervisor read the wrapped success log,
classified it `non_retryable_failure`, and aborted instead of chaining into the finalizer. Fixed by
requiring the two halves as independent, non-adjacent substrings; added a regression test using the
real wrapped text.

**Bug 2 — validator checked the wrong manifest key:** `similarity_graph_validation.py` checked
`input_validation.population_sha256`, but every writer in the codebase (`constraint_evidence`,
`facet_audit`, `constraint_state`, `split_audit`, and the graph pipeline itself) writes
`input_population_sha256`. The mismatch rejected an otherwise-valid, passing manifest. The unit
test fixture had the same wrong key, so it never caught this. Fixed the key name in both.

**Bug 3 — content hash crashed on a list column:** `_json_scalar` (used by
`dataframe_content_sha256` for read-back provenance hashing) checked `isinstance(value, list |
tuple)` before falling through to a scalar `pd.isna(value)` call. `similarity_graph_runs`' captured
per-shard minimap2 stderr lines (`tool_log`) deserialize from Parquet as `numpy.ndarray`, which
`pd.isna` cannot evaluate as a single boolean. Added `np.ndarray` to the isinstance check; added a
regression test with an array-valued column.

**Bug 4 — `--params` silently dropped sibling keys:** `scripts/finalize_similarity_data.py` ran
`similarity_split` and `query_benchmark` via `kedro run --params block.leaf_key=value`. Kedro
replaces a whole top-level parameter block on a runtime override rather than deep-merging into it
— the same non-deep-merge behavior this README already documents for environment config files —
so passing only the changed leaf key silently dropped `train_fraction`, `val_fraction`, `seed`, and
every other sibling, producing a bare `KeyError: 'train_fraction'`. It was also fundamentally
unsafe for list-valued params (`query_benchmark.top_k`, `evaluation_splits`): a naive comma-join
cannot distinguish a list literal's internal commas from the next `key=value` pair. Replaced both
invocations with the Kedro Python session API: read the base params block, merge the override leaf
keys onto a full copy of it, pass the complete block as `runtime_params`. Added unit tests for the
merge logic.

All four fixes: 168 tests pass, ruff clean, verified independently on `g6-big` before each rerun.
Commits `2217f0a`, `a94413b`, `8cd831e`.

**Next:** run the finalizer (`split_grouped_v2` then `query_benchmark`) against the accepted graph.

## 2026-08-18 — Gate 0 complete: split_grouped_v2 and frozen query benchmark accepted

**Status:** `gate0_data_complete`. All required Gate 0 outputs exist and pass independent S3
read-back validation. This closes E00.

**split_grouped_v2** — output version `2026-08-17T23.49.47.355Z`, built from accepted graph
`2026-08-17T22.59.04.326Z`: 115,120 rows, 11,764 primary components, split
92,279 / 11,344 / 11,497 (train/val/test). **Zero primary (99%-identity) cross-split edges** — the
strict near-duplicate leak the old split failed on is closed. 6,259 sensitivity-only (95%) edges
still cross, reported as required but not a failure per protocol. Concentration improved sharply
over the old split: test's largest component fell from 29.23% of rows to 2.86%, its effective
component count rose from 11.12 to 206.6; val's largest component fell from 16.23% to 3.92%,
effective components rose from 32.52 to 132.9. `concentration_warning_splits` is empty (both val
and test stay under the 25% largest-component trigger).

**Frozen query benchmark v0.1** — output version `2026-08-17T23.51.35.629Z`: 131 semantic queries
(atomic and two-facet conjunctions), 524 catalog rows across 4 candidate galleries, 5,740,247
sparse verified/contradicted query-candidate states. Independent validation recomputed every
answer set from the frozen source states (not read from the persisted table), confirmed the four
base measures are normalized, confirmed verified/contradicted state pairs are disjoint, and passed
both the verified-first-oracle and contradiction-first control checks.

**Gate 0 data-support flag: passed for both closed evaluation splits**, against the preregistered
floors (10 usable atomic queries, 20 usable pair queries, 20 usable pair contradiction controls
per split):

| Split | Usable atomic | Usable pair | Usable pair w/ contradiction control |
| --- | ---: | ---: | ---: |
| Validation | 28 | 80 | 80 |
| Test | 32 | 90 | 90 |

**Decision:** Gate 0 is accepted. Proceed to Gate 1 (fixed representations: select and pin one DNA
encoder and one text encoder; validate length coverage, circular-rotation sensitivity,
reverse-complement sensitivity, and pooling behavior) before any model training.

## 2026-08-18 — Gate 1 encoder prior reviewed and fixed-representation bake-off preregistered

**Status:** documentation and protocol complete. No encoder feature extraction, alignment probe,
validation scoring, test read, or paid GPU run occurred.

Reviewed the PlasmidCLIP encoder evidence before selecting new candidates. Its first 100-plasmid
geometry benchmark favored Mistral-DNA-bacteria, but its later leakage-safe target-retrieval
comparison favored Carbon-500M: R@1 8.1%, R@10 28.8%, and median rank 41, compared with Mistral's
6.0%, 24.5%, and 57. Carbon also had effective rank 475/1,024, while Mistral had about 6/768. This
means rotation and reverse-complement invariance are diagnostics, not the primary selector. The
same prior work found BGE-base healthy at effective rank 472/768 and found Carbon float16 inference
invalid. Gate 1 uses bfloat16 with a float32 check.

Reviewed current official model cards, repositories, papers, licenses, contexts, and Hugging Face
revisions. The frozen DNA panel is:

- incumbent `HuggingFaceBio/Carbon-500M@106e36ff51b5dfbfe0b078ad18ad37a6956c5714`;
- `HuggingFaceBio/Carbon-3B@95c3c68fc77fdf70b1582031bacf9d7753f72cf2`;
- `GenerTeam/GENERanno-prokaryote-0.5b-base@d02db0f24f2c62fa1efde760217cdf75771b0228`;
- `GenerTeam/GENERator-v2-prokaryote-1.2b-base@8b2f768b0d293953518ff91d34600f9322ef1f94`.

The frozen text panel is BGE-base revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, GTE-ModernBERT-base revision
`e7f32e3c00f91d699e8c43b53106206bcc72bb22`, and Qwen3-Embedding-0.6B revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

Evo 2 7B is a high-cost reserve. NTv3 is excluded because the current weights are non-commercial
and the long-context post-training targets animal and plant functional tracks. CENO remains a
watchlist model because its July 2026 release is too new for the primary gate. Bacformer requires
protein calls and belongs in a later annotation-aware comparison. The fine-tuned PlasmidCLIP
checkpoint is not a confirmatory candidate because its old training split can overlap v2
validation and test rows.

Preregistered `E02_fixed_representation_bakeoff.md`. It fixes component-aware training and
invariance panels, full validation evaluation, the five-by-three DNA and text factorial, complete
circular-window coverage, pooling and precision rules, three 60-epoch projection seeds, query-macro
`verified@10 - contradicted@10` as the primary metric, component bootstrap intervals, a
40-A100-equivalent-GPU-hour limit, and a 250-GB artifact limit. Test outcomes remain unread. Paid
GPU work requires explicit approval.

The required scientist-workflow consult did not return a protocol decision. The advisor could not
read its required Notion UX Constitution because `NOTION_API_KEY` was unset. It made no edits.
This leaves the protocol without that product-workflow review; it does not change the recorded
scientific checks.

**Decision:** start Gate 1 with smoke-loading and cost measurement only after approval. Do not
start Gate 2 or inspect test metrics until the selected encoder revisions, extraction rules, and
feature hashes are frozen.

## 2026-08-18 — Gate 1 DNA numerical smoke executed on AWS

**Status:** three candidates passed. Carbon-3B had a technical memory failure. No encoder was
selected. Validation outcomes and test rows remained unread.

Built a deterministic 512-row training-only invariance panel with one row per primary similarity
component and a nested 32-row numerical panel. The input population hash was
`7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5`; the panel hash was
`6dddbc33e0bb07ffcd3a2bebfcbf58f8c07573da976d0ef02e62c252e6e1593b`.

Accepted S3 output versions:

- Carbon-500M: `2026-08-18T15.02.18.875Z`, minimum BF16/FP32 cosine `0.99999355`;
- GENERanno prokaryote 500M: `2026-08-18T15.15.56.524Z`, minimum cosine `0.99999595`;
- GENERator-v2 prokaryote 1.2B: `2026-08-18T15.19.31.423Z`, minimum cosine `0.99999383`.

Every accepted run had 32 finite diagnostic rows, complete source-base coverage, and zero
out-of-vocabulary tokens. Independent S3 read-back recomputed the cosines and table hashes. All
values and hashes matched the manifests. The accepted runs used commits `6a19a66` and `81768b9`.

Carbon-3B exceeded the 22.03 GiB L4, then exceeded a task-specific 44.39 GiB L40S on the unchanged
protocol. The second process used 43.38 GiB and requested 3.57 GiB more. Neither attempt wrote an
accepted feature, coverage, diagnostic, or manifest artifact. This is a technical memory failure,
not a retrieval result.

The L4 host was stopped. The task-specific L40S host was terminated. The derived launch-to-stop
compute estimate is $0.65 before storage and data transfer. See
[`reports/14_gate1_numerical_smoke.md`](reports/14_gate1_numerical_smoke.md) for the complete run
history, exact hashes, runtime versions, and limitations.

**Decision:** continue the 512-row invariance checks with Carbon-500M, GENERanno prokaryote 500M,
and GENERator-v2 prokaryote 1.2B. Keep Carbon-3B out of the L4 run. Do not select an encoder until
the invariance and validation-only retrieval stages complete.

## 2026-08-19 — Carbon-3B exact 80 GiB retry approved before execution

**Status:** approved and not yet run. No model result was available when this amendment was
recorded.

The user approved another exact Carbon-3B numerical smoke retry and paid AWS compute. The prior
L40S attempt used 43.38 GiB and then requested 3.57 GiB more, so the new host must provide more
than 48 GiB on one device. AWS describes `p5.4xlarge` as one NVIDIA H100 with 80 GiB. AWS Pricing
reported $6.88 per on-demand Linux instance-hour in `us-east-1` on 2026-08-19. The account quota
allows this instance type, and AWS lists it in all six `us-east-1` Availability Zones.

Run one task-specific `p5.4xlarge` for at most one instance-hour. Keep the scientific code at
commit `66163b161fdd064da3926bf55d8d6853f25cf305` and keep the model revision, sample manifest,
tokenizer, sequence length, windowing, precision, deterministic attention, and seed unchanged.
Terminate the host after success, technical failure, or the one-hour limit. The maximum instance
charge is $6.88 before storage and data transfer. This retry cannot inspect validation outcomes or
test rows.

**Decision:** run the exact Carbon-3B numerical smoke retry. Treat a capacity failure as an AWS
launch failure. Treat an inference failure as a technical result, not as retrieval evidence.

### 2026-08-19 launch outcome

**Observed:** AWS rejected six zone-specific On-Demand requests and one regional On-Demand
request with `InsufficientInstanceCapacity`. AWS also rejected seven regional one-time Spot
requests with the same error. No request created an instance or incurred an instance charge. The
account retains quota for 192 On-Demand P-instance virtual CPUs and 192 P Spot-instance virtual
CPUs in `us-east-1`. Other checked regions have zero On-Demand P-instance quota.

**Decision:** Carbon-3B remains unresolved. Do not change the scientific configuration to fit a
smaller device. Retry the approved `p5.4xlarge` job when AWS capacity is available. Continue the
512-row invariance stage for the three candidates that passed the numerical smoke check.

## 2026-08-22 22:32:02 BST — Gate 1 invariance harness implementation started

**Status:** implementation started. No paid GPU process, encoder inference, validation-outcome
read, or test-row read occurred.

**Question:** Can the three accepted DNA candidates run the frozen 512-row rotation,
reverse-complement, collapse, confound, coverage, throughput, memory, and cost checks through a
failure-visible Kedro pipeline?

**Base code:** commit `246f017da60168b0c6eadeef1c38a322d66dabc8` on a new
`agent/gate1-invariance-harness` branch. The implementation worktree became dirty only with this
active change.

**Fixed inputs:** Use the exact accepted numerical-smoke artifact version for each candidate and
require panel SHA-256 `6dddbc33e0bb07ffcd3a2bebfcbf58f8c07573da976d0ef02e62c252e6e1593b`.
Only the 512 v2 training rows are eligible.

**Pre-run amendment:** Match the prior PlasmidCLIP comparison with 25%, 50%, and 75% circular
rotations plus the reverse complement. Test each perturbation median against 0.90. Define effective
rank from entropy over centered singular values. Define the two confound correlations from
pairwise cosine versus absolute log2 length ratio and absolute G+C-fraction difference. The E02
specification records the exact formulas.

**Compute boundary:** The runner must reject a command without a durable approval reference, AWS
region, explicit instance type, instance-hour limit, and current observed price. It must stop
before starting another sequence after the remaining cap. One command has one total hour cap across
all requested candidates; the budget does not reset per candidate. This implementation action does
not authorize paid execution. Reject a mixed Transformers-version candidate batch. Carbon uses one
pinned 5.12.1 command; the two GenerTeam candidates can share one pinned 4.49.0 command.

**Planned validation:** Deterministic transform and geometry unit tests, a fake-encoder node test,
pipeline registration and catalog checks, the full offline test suite, Ruff lint, Ruff format
check, and `git diff --check`.

**Next decision:** Complete local validation and independent review. Then prepare a bounded cost
proposal before any paid invariance run.

## 2026-08-23 00:26:07 BST — Gate 1 invariance harness validated locally

**Status:** implementation and offline validation complete. No paid GPU process, encoder
inference, validation-outcome read, or test-row read occurred.

**Observed input check:** A read-only S3 check loaded all three exact accepted versions. Each
version contained the expected 512-row training panel, panel-manifest SHA-256, smoke-manifest
SHA-256, model revision, and Transformers runtime. Carbon-500M's accepted manifest predates the
recipe-level `transformers_version` field, so its exact hash-bound runtime provenance supplies
5.12.1. Both GenerTeam manifests supply 4.49.0 in the recipe itself.

**Review corrections:** Independent code and architecture reviews found that the first runner
timed only inference, direct Kedro invocation did not prove exact accepted input content, the
read-back omitted confound recomputation, and Git provenance depended on the caller's working
directory. The corrected runner places the complete candidate session, including catalog reads and
writes, in an externally timed child process and reserves 30 seconds for termination. The node
checks immutable input-manifest hashes before model loading. The read-back compares the frozen
configuration and recomputes every geometry field. Git commands now run against this worktree.

**Validation:** `pytest -q` passed 200 tests. `ruff check src tests scripts` passed.
`ruff format --check src tests scripts` reported 132 formatted files. Kedro catalog and registry
inspection resolved only the frozen smoke panel and manifests as inputs and the five versioned
invariance products as outputs. `git diff HEAD --check` passed.

**Known limitation:** Offline fake-encoder tests do not establish real GPU memory, throughput,
S3 output size, or model-runtime behavior. The child-process timeout bounds the complete candidate
command. The EC2 job still needs an independent host-termination deadline because an
uninterruptible kernel or driver failure can defeat an in-process operating-system timeout.

**Next decision:** Commit and publish the reviewed harness. Then prepare a current price and
instance-cap proposal. Do not start a paid invariance run without a new explicit approval
reference and host-termination plan.

## 2026-08-23 00:46:44 BST — Gate 1 invariance run stopped on IUPAC ambiguity gate

**Status:** technical protocol failure before the first feature row. No invariance feature,
coverage, perturbation, diagnostic, or run-manifest artifact was written. Validation outcomes and
test rows remained unread.

The user approved a maximum of three `g6.2xlarge` instance-hours at the observed
`us-east-1` Linux price of $0.9776 per hour, with approval reference
`chat-2026-08-23-start-benchmarking`. The Carbon-500M command had a separate one-hour cap. The run
used clean commit `f3227a79ea7df17c8e070f5a252218483a830dbf` on task-specific instance
`i-07e521b0268df674b` in `us-east-1a`, with an external AWS stop deadline at
2026-08-23 02:35:48 UTC.

The runner loaded the pinned Carbon-500M weights, then stopped on the first panel row,
`addgene_87671`, because its sequence contains the IUPAC ambiguity symbol `W`. The fixed encoder
boundary accepts only `A`, `C`, `G`, and `T`; E02 requires failure on an out-of-vocabulary base and
prohibits silent base replacement. This is the intended fail-early behavior, not a model outcome.

A read-only audit of the exact frozen 512-row training panel found 23 affected rows and 258
ambiguity symbols: `B=1`, `K=19`, `M=25`, `N=43`, `R=36`, `S=64`, `V=1`, `W=22`, and `Y=47`.
The accepted source contract permits the complete IUPAC DNA alphabet. The numerical smoke panel
did not expose this full-panel condition.

The exact pinned tokenizer implementations do not supply a candidate-neutral fallback. Carbon
maps any ambiguity-bearing 6-mer to `<oov>`. GENERator-v2 also maps an ambiguity-bearing 6-mer to
`<oov>`. GENERanno maps an unsupported source symbol to its `N` token. Allowing these native
behaviors would therefore apply different information loss and token granularity across
candidates.

The instance entered `stopping` at 2026-08-22 23:46:44 UTC after 6 minutes 36 seconds from launch.
At the approved hourly rate, the derived instance charge is $0.107536 before EBS storage and data
transfer. The instance remained in `stopping` after both normal and forced stop requests. AWS does
not charge instance usage in this state. The encrypted 200 GB root volume remains recoverable, and
the external stop schedule remains enabled.

The scientist-advisor consult did not return a policy recommendation. It could not read its
required Notion UX Constitution because `NOTION_API_KEY` was unset, and it made no changes.

**Decision:** do not retry or change the panel from this result. First preregister one explicit
IUPAC policy. The proposed candidate-neutral amendment is to make `A`/`C`/`G`/`T`-only sequence
eligibility explicit and deterministically replace the 23 affected panel rows under the original
length-stratum and primary-component selection rules. Recompute the nested numerical-smoke
manifest and rerun that smoke only if its row identities or parent-panel contract changes.

**Post-stop update:** AWS subsequently confirmed the instance as `stopped`. The encrypted root
volume remains attached. The now-redundant one-time stop schedule and its task-scoped IAM role were
deleted. The instance and volume were not terminated or deleted.

## 2026-08-23 01:13:29 BST — Gate 1 A/C/G/T-only panel amendment approved before retry

**Status:** protocol version 0.2 approved and implemented locally. No retry, encoder inference,
validation-outcome read, test-row read, or new remote artifact had occurred when this entry was
recorded.

**Question:** Can one candidate-neutral input rule remove the tokenizer-specific IUPAC behavior
without changing the source sequences or resampling unaffected panel rows?

**Approved rule:** An invariance-panel row is eligible only when its complete source sequence uses
uppercase `A`, `C`, `G`, and `T`. Preserve all eligible version 0.1 panel rows. Replace each
ineligible row within its original length decile, under the original extreme-preservation,
deterministic SHA-256 ordering, and primary-component uniqueness rules. Do not replace or infer a
base. Bind the amendment to the version 0.1 panel hash.

**Observed input audit:** A read-only calculation from retrieval version
`2026-08-04T09.02.10.007Z` and `split_grouped_v2` version `2026-08-17T23.49.47.355Z` found 88,474
A/C/G/T-only training rows and 3,805 excluded training rows. The excluded rows contain these
ambiguity-symbol counts: `B=64`, `D=64`, `H=21`, `K=2,640`, `M=1,667`, `N=9,100`, `R=6,755`,
`S=12,032`, `V=75`, `W=1,614`, and `Y=2,205`. The exact excluded-row identity, source-sequence
hash, and per-row symbol-count records have SHA-256
`de65f169e71a087e190f1539faeddf216551d911d5842cdd0dcbad624c1f325b`.

**Derived panel:** The amended selection preserves 489 version 0.1 rows and replaces the 23
ineligible panel rows. It retains 512 rows, 512 primary components, the fixed length-stratum
quotas, and only `A/C/G/T` sequences. The 32 nested numerical-smoke identities do not change. The
new panel SHA-256 is
`2516a415c7040e4ef75805294c8c9d5693749033c1cd196de24a79f14b5a30a0`. This was an in-memory
derivation only; no new artifact was written.

**Provenance rule:** Because the parent-panel contract and protocol version changed, create new
versioned numerical-smoke artifacts for the three accepted candidates before invariance. Keep all
model and scientific settings fixed. The panel manifest records the complete eligibility audit,
removed-row identities, replacement-row identities, source hashes, length deciles, and components.

**Compute authorization:** The user approved protocol version 0.2 and at most three additional
`g6.2xlarge` instance-hours for the smoke and invariance commands at $0.9776 per instance-hour,
approval reference `chat-2026-08-23-acgt-panel-and-benchmark`. The maximum approved command cost is
$2.9328 before storage and transfer. The user requested that the preserved EC2 instance remain
running after the commands. Each scientific command keeps an external timeout, but there is no
automatic post-run instance stop.

**Planned checks:** Run the focused selection regressions, complete offline suite, Ruff lint and
format checks, Kedro catalog and pipeline registration, exact pinned-input panel derivation, and
`git diff --check`. Commit and push the preregistered implementation before any paid retry.

**Pre-commit validation:** `uv run --extra dev pytest -q` passed 203 tests. `uv run --extra dev
ruff check .`, `uv run --extra dev ruff format --check .`, and `git diff --check` passed. A fresh
read-only derivation from the two exact pinned inputs passed both panel-hash gates, catalog and
pipeline registration, the 512-row and 512-component checks, all ten stratum quotas, and the
A/C/G/T-only check. The removed and replacement counts match in every affected stratum. The nested
smoke identity difference is empty.

**Architecture consult:** The required one-round platform consult could not read the live Notion
ADR register because `NOTION_API_KEY` is unset. It made no edits. It confirmed that the diff is
confined to seven research files and does not touch authentication, the biolake gateway,
infrastructure, platform lineage migrations, or storage paths. Record this live-ADR review gap in
the draft PR. Do not describe the unavailable review as approval.

## 2026-08-23 01:48:01 BST — Gate 1 protocol v0.2 numerical smoke passed

**Status:** all three accepted DNA candidates passed the new versioned protocol v0.2 numerical
smoke check. Validation outcomes and test rows remained unread. No retrieval result or candidate
selection was produced.

The preserved `g6.2xlarge` instance `i-07e521b0268df674b` initially remained stopped after three
manual and two monitored `StartInstances` requests failed with `InsufficientInstanceCapacity`.
The third monitored request succeeded without changing the instance, availability zone, instance
type, panel, or model configuration. The failed requests incurred no compute charge. The instance
started in `us-east-1a` and remains running by user request.

The exact code was clean commit `bc29344cf8a8240b36ebc9264c74de73c9f94ab3`. Carbon used Python
3.13.9, Torch 2.11.0+cu130, Transformers 5.12.1, and Accelerate 1.14.0. Both GenerTeam candidates
used the same Python, Torch, and Accelerate versions with Transformers 4.49.0. All runs used one
NVIDIA L4 with 23,659,151,360 reported device bytes. The scientific command clock started at
2026-08-23 00:33:16 UTC. The absolute three-hour deadline is 2026-08-23 03:33:16 UTC.

| Candidate | S3 version | Minimum BF16/FP32 cosine | Model-node seconds | Outcome |
| --- | --- | ---: | ---: | --- |
| Carbon-500M | `2026-08-23T00.33.38.853Z` | 0.9999935476 | 257.54 | Passed |
| GENERanno prokaryote 500M | `2026-08-23T00.41.52.864Z` | 0.9999959534 | 165.62 | Passed |
| GENERator-v2 prokaryote 1.2B | `2026-08-23T00.45.09.745Z` | 0.9999938263 | 68.10 | Passed |

All three exact artifacts contain the 512-row, 512-component, A/C/G/T-only panel with SHA-256
`2516a415c7040e4ef75805294c8c9d5693749033c1cd196de24a79f14b5a30a0`. Their canonical panel
manifest SHA-256 is
`dc88d9ba0b8d2c8680874f6d90265d2b3e7f990d13f91943a0d0afeeca005f91`.

Canonical numerical-smoke manifest SHA-256 values are:

- Carbon-500M: `3d3ef20cd3695ff40933a5e436af74eb035363bcc37ead5b06da04926744d8d1`;
- GENERanno prokaryote 500M:
  `51ae97c4b9c007cb45dba2a0b470cccda58aeae29a85a62bc76d5b91f5160389`;
- GENERator-v2 prokaryote 1.2B:
  `692bca0587c2d4fcb3647ef81771d2c46613174bc705d9bc8920ef2a24c6f65a`.

**Independent read-back:** Reloaded all six products for each exact version. It checked 64 finite
feature rows per candidate, recomputed each embedding SHA-256, required zero OOV tokens, recomputed
coverage and all 32 BF16/FP32 cosines, matched the persisted DataFrame hashes, and checked the
model revision, runtime, clean Git provenance, decisions, and no-test/no-validation flags. Exact
cosine equality differed by at most `1.55e-15` after Parquet reload. The read-back therefore used
an absolute tolerance of `1e-12` and zero relative tolerance, which is more than twelve orders of
magnitude below the 0.99 pass boundary. Identifiers, counts, decisions, and content hashes remained
exact.

**Decision:** freeze these three version and manifest-hash tuples as the only accepted protocol
v0.2 smoke inputs. Commit and push the frozen configuration before starting invariance. Continue
the absolute three-hour command deadline; do not reset it after smoke validation.

## 2026-08-23 02:23:04 BST — Gate 1 protocol v0.2 DNA invariance benchmark passed

**Status:** all three accepted DNA candidates passed the frozen coverage, transform-invariance,
and effective-rank gates. Independent read-back passed for every persisted artifact. Validation
outcomes and test rows remained unread. No retrieval metric or candidate selection was produced.

The commands used clean commit `130d3feda38d08623d2dc0a26d1db7f806b52d9d` and the three exact
protocol v0.2 numerical-smoke versions frozen above. Each candidate encoded the original sequence,
25%, 50%, and 75% circular rotations, and the reverse complement for all 512 A/C/G/T-only training
rows. Every run used BF16, deterministic algorithms, disabled TF32, and one NVIDIA L4. Carbon used
Transformers 5.12.1. Both GenerTeam candidates used Transformers 4.49.0.

| Candidate | Invariance artifact version | Reverse-complement median cosine | Lowest rotation median cosine | Effective-rank fraction | Median pairwise cosine | Original throughput (bp/s) | Peak device memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Carbon-500M | `2026-08-23T00.50.18.556Z` | 0.983288 | 0.991919 | 0.237089 | 0.974671 | 211,301 | 1,286,269,952 bytes |
| GENERanno prokaryote 500M | `2026-08-23T00.53.22.430Z` | 0.998770 | 0.999920 | 0.031039 | 0.995220 | 12,903 | 1,259,836,416 bytes |
| GENERator-v2 prokaryote 1.2B | `2026-08-23T01.18.17.964Z` | 0.999442 | 0.999688 | 0.107953 | 0.999315 | 84,809 | 2,887,549,440 bytes |

The minimum accepted transform median is 0.90, and the minimum accepted effective-rank fraction is
0.01. All twelve transform medians and all three effective-rank fractions passed. Coverage passed
for every candidate, and persisted read-back found 2,560 feature rows and 2,048 similarity rows per
candidate with zero recomputation error.

The persisted feature, coverage, and similarity hashes are:

- Carbon-500M: features
  `4f56a1357d3b9fa0555944cd24364cecba42840a04d76b917a2285045cb34139`, coverage
  `2c3f730b525ddee71cdc50a306190ab2c6689c6fa57826186b794d876174815f`, similarities
  `425c91bc2352f5676c2e2e630018ea0bab106e8251de168c2df45db9744a5c63`;
- GENERanno prokaryote 500M:
  features `ab5da24bdc71fcd73755e425189e4ba39026cbf4fba0386b55c26eb67e7a5754`,
  coverage `fd0373151c93b3c3a61989e888770fab2cf6c6e779aabde2b16456c4aa2ce39d`,
  similarities `a1d096baa8e278d8f82ed5a0bd9cdb4f123011d73a20526af774aa8863252899`;
- GENERator-v2 prokaryote 1.2B:
  features `8cbeacb326470561467f83e5230b2a8f7e282e9344e0d51ef03fbf1c293cce82`,
  coverage `4b9f5e0fccfc29edfcc7dfc7ad46959c53431ec1ef19b5747e79b51760ab1555`,
  similarities `4d0739328226f9b350b3939bc9bfe72e5ec2305d3e90c1fc182b236cf6dac742`.

**Derived execution cost:** Carbon completed in 104.45 wall seconds, including setup and
persistence. The GenerTeam batch completed both candidates in 1,738.53 wall seconds. At the
approved $0.9776 per instance-hour, the two benchmark commands used 1,842.98 seconds and cost
$0.500472 before storage and transfer. This value measures the scientific commands, not complete
EC2 uptime. The user requested that instance `i-07e521b0268df674b` remain running after completion,
so its total host charge continues to increase at the observed hourly rate.

**Interpretation:** invariance is a failure gate, not a selection metric. Carbon has the highest
effective-rank fraction and the fastest measured throughput. GENERanno has the lowest
effective-rank fraction and the strongest measured length and G+C correlations. GENERator-v2 has
an effective-rank fraction between the other candidates and near-zero measured G+C correlation,
despite a high median pairwise cosine. These observations do not establish retrieval utility.

**Decision:** retain all three candidates for the fixed Gate 1 retrieval evaluation. Freeze the
three exact invariance artifact versions and hashes. Do not choose an encoder from invariance,
throughput, or geometry alone. The next scientific step is frozen feature extraction and the fixed
alignment-probe validation comparison. Keep the test split unread.

## 2026-08-23 11:44:26 BST — E02b reduced-population retrieval amendment approved

**Status:** approved before feature extraction, alignment-probe fitting, validation scoring, or
candidate selection. No new model artifact or result existed when this entry was recorded.

**Observed incompatibility:** the complete retrieval population retains valid IUPAC ambiguity
symbols. Carbon-500M and GENERator-v2 map an ambiguity-bearing 6-mer to an out-of-vocabulary token,
while GENERanno maps unsupported symbols to `N`. Native behavior would therefore change the input
meaning by candidate. Replacement, truncation, and candidate-specific fallback remain prohibited.

A read-only eligibility audit found 3,805 of 92,279 `split_grouped_v2` training rows and 492 of
11,344 validation rows outside the common uppercase `A`/`C`/`G`/`T` alphabet. The reduced
validation gallery contains 10,852 rows. All 108 previously measurement-usable closed-validation
queries retain at least ten verified targets after filtering: 28 atomic and 80 pair-conjunction
queries.

**Approved E02b rule:** apply one `A`/`C`/`G`/`T`-only eligibility rule before model-specific
processing. Rebuild the 20,000-row component-aware training panel from eligible training rows.
Filter the validation gallery without replacement. Recompute query support and every metric on
that reduced gallery. Preserve and report excluded row identities, source hashes, symbol counts,
length strata, and component coverage. Do not describe E02b as the original full-population E02.

Evaluate the train-fitted 6-mer TF-IDF/SVD baseline and the three neural DNA candidates with
accepted smoke and invariance artifacts against all three frozen text candidates, for four-by-three
factorial coverage and all three preregistered seeds. Record Carbon-3B as **not evaluable: technically
ineligible within the E02 hardware and compute envelope**. Its 22 GiB and 44 GiB memory failures
and unavailable 80 GiB host are not evidence about retrieval quality.

The interpretation notebook must show the reduced-population warning, population flow, exclusion
bias, query coverage, candidate disposition, factorial completeness, and numerical/invariance
checks before its leaderboard. A selected pair is eligible only for this reduced-population Gate 1
decision. Ambiguity-bearing rows need a separate frozen policy before full-population Gate 2 use.

**Approval:** the user approved this E02b amendment in chat on 2026-08-23 and instructed that the
reasoning be recorded in this log.

**Test-contamination record:** during the pre-implementation eligibility audit, a local diagnostic
loaded the complete split mapping, query catalog, and sparse state table before filtering its
calculation to validation. It printed only aggregate test alphabet/component counts and produced no
test feature, model score, ranking, or test metric, but the process read test row and state data.
Under the frozen protocol this access contaminates the current test set for confirmatory use. E02b
selection remains validation-only. Any later confirmatory evaluation requires a new test protocol
or a demonstrably untouched holdout; do not report the current test split as unread.

## 2026-08-23 13:17:25 BST — E02b representation and probe recipes frozen

**Status:** preregistered before E02b feature extraction, probe fitting, validation ranking, or
candidate selection. The only E02b work at this point was deterministic input auditing and code
under test. No model feature or validation result existed.

The three pinned text-model cards specify different representation contracts. Freeze BGE-base to
normalized first-token pooling, no document prefix, and its documented English retrieval-query
prefix. Freeze GTE-ModernBERT to normalized first-token pooling without role prefixes. Freeze
Qwen3-Embedding to left padding, normalized last-token pooling, no document prefix, and this
task-specific query instruction:
`Given a plasmid constraint query, retrieve plasmid descriptions that satisfy the recorded constraint`.
The serialized Qwen query form is
`Instruct: <task>\nQuery:<canonical query>`. Use the exact pinned revisions and Transformers 5.12.1.
Do not truncate. Stop above 512, 8,192, and 32,768 tokens for BGE, GTE, and Qwen, respectively.

The TF-IDF baseline was underspecified in E02 v0.1. Freeze character 6-mers, case preservation,
standard smoothed inverse-document frequency, L2 TF-IDF normalization, randomized 512-dimensional
truncated SVD with seven power iterations and seed 20260818, then L2-normalize the SVD output. Fit
all state on the approved 20,000 training rows and persist it.

Freeze full-rank train-only principal-component whitening with epsilon `1e-6`, zero removed
components, and bias-free 512-dimensional heads. Every epoch uses all training rows; the final
batch can be smaller than 4,096 and cannot be dropped. Stable frozen gallery order resolves score
ties. Use 2,000 whole-component bootstrap draws with seed 20260818 and persist every draw.

These choices resolve implementation details before outcomes. They do not change the approved
E02b population, candidate set, three seeds, optimizer, 60 epochs, primary metric, selection rule,
or validation-only scope. The earlier test-contamination record remains in force.

## 2026-08-23 15:14:06 BST — E02b input artifact read-back failure

**Status:** invalid technical run; do not accept or reuse version
`2026-08-23T14.12.10.259Z`. No model loaded, no feature or ranking was calculated, and no candidate
was selected.

The deterministic input pipeline completed and wrote all five versioned products. Independent
read-back matched the exclusion, query, and query-state SHA-256 values. The pairs table did not
match: the pre-write manifest recorded
`19f062c3ad0bedf5ac855001e7ce749bde5954c00e1190ca3645e60269033b6c`, while the reloaded table
produced `35c202ecc95fd714b7b283857d761966a8b2a8a80f01409b3add531348a1b972`.

**Observed cause:** the training-panel selection pass was an integer before persistence and a
nullable float after Parquet read-back. The scientific values were unchanged, but the canonical
JSON hash distinguished `0` from `0.0`. Store this provenance field as nullable decimal text and
add an explicit Parquet round-trip regression test. Rerun the input pipeline from a new clean Git
commit. Preserve this failed version as technical-failure evidence.

## 2026-08-23 15:17:44 BST — E02b input artifact accepted

**Status:** passed independent persisted-artifact read-back. This run did not load a model, extract
a feature, fit a probe, calculate a ranking, or select a candidate.

The clean worktree at Git commit `25847aad6ec15edb410c4067cccbf8bd7ccc2c3d` produced E02b input
version `2026-08-23T14.16.15.778Z`. Independent read-back observed 20,000 training rows across all
7,774 eligible training components, 10,852 validation rows across 1,646 components, 4,297 excluded
rows, 108 queries (28 atomic and 80 pair conjunction), and 446,758 sparse query-state rows. Every
query retained at least ten eligible verified validation rows.

All persisted hashes matched the manifest:

- manifest: `57911ecee4654f17cc456a537f4295de155a3cf3269a3dfc289c95ee630f4a46`;
- pairs: `cc7690d9bc979028dd3af0e292b715f6c7c391daac269be0fc5a7b7c17a86aa9`;
- exclusions: `f4b561f3a8173c75b7619d27b222924118afddfb3391f21e20346a48a242bf45`;
- queries: `a440e26a32468a9e613aaa2034b476b453a8bae988c08128df3f14f251c4552c`;
- query states: `06afff85e1d4b3d96f7c62eeadd3ba50892b26cae558aec64a5bcba41c37ce8b`.

The five artifacts use 31,212,396 bytes in total. Freeze this exact version and these hashes before
TF-IDF or neural feature extraction. E02b remains validation-only because the earlier eligibility
audit contaminated the old test artifacts for confirmatory use.

## 2026-08-23 15:26:20 BST — E02b TF-IDF/SVD baseline accepted

**Status:** passed independent persisted-artifact read-back. This run fit only train-derived
TF-IDF and SVD state. It did not fit an alignment probe, calculate a validation ranking, or select
a representation pair.

The clean worktree at Git commit `e75ffa2d00f327b84fb009779d52400deed0667e` loaded the accepted
E02b input version and completed the frozen 6-mer TF-IDF/SVD node in 124.60 seconds. The output
version is `2026-08-23T14.21.58.754Z`. It contains 30,821 unique 512-dimensional sequence vectors
for 30,852 source rows, all 4,096 A/C/G/T 6-mers, and all 512 persisted SVD components.

Independent read-back found finite L2-normalized vectors, exact per-row embedding hashes,
contiguous vocabulary and component indices, and exact table hashes:

- manifest: `2c6d5c0023d884951580df7e5202fd356e5c15ca8c298a0628abb4462a219cbb`;
- features: `d783f66cb234bd228d9de304884b23849d75d9b5eef2a7510d61b0cc83486a41`;
- vocabulary: `39b75ff9154f34f2265971d9cb623dcb5215f8d693b3a914607a7bfb531be6e6`;
- SVD state: `dcdddf96c4aed4d3706afd19d902e61b885f1b6b6896cbd0b7a7176d5b016b25`.

The complete feature product uses 93,002,965 bytes and zero GPU-hours. Freeze this exact baseline
before the neural feature runs.

## 2026-08-23 15:39:31 BST — E02b complete benchmark compute authorization

**Status:** approved before any E02b neural feature extraction, probe fitting, validation ranking,
or candidate selection. The accepted E02b input and TF-IDF/SVD artifacts are the only E02b
artifacts available at this boundary.

**Approval:** the user instructed the agent in chat to use `g6-big` or allocate compute, complete
the benchmarks, save all data, and make material project progress. Use durable approval reference
`chat-2026-08-23-e02b-complete-benchmark`. The existing `g6-big` alias resolves to the preserved
on-demand `g6.2xlarge` L4 host in `us-east-1`. Use the observed Linux on-demand price of $0.9776
per instance-hour. The price excludes storage and data transfer.

**Compute boundary:** authorize at most 15 additional instance-hours, which is at most $14.664 in
instance charges at the observed price. The exact command limits sum to 14.25 hours:

| Stage | Candidate | Command limit | Maximum instance charge |
| --- | --- | ---: | ---: |
| DNA features | Carbon-500M | 0.75 h | $0.7332 |
| DNA features | GENERanno prokaryote 500M | 7.00 h | $6.8432 |
| DNA features | GENERator-v2 prokaryote 1.2B | 1.50 h | $1.4664 |
| Text features | BGE-base-en-v1.5 | 0.50 h | $0.4888 |
| Text features | GTE-ModernBERT-base | 0.75 h | $0.7332 |
| Text features | Qwen3-Embedding-0.6B | 0.75 h | $0.7332 |
| Alignment probes | Complete 4 by 3 by 3 factorial | 3.00 h | $2.9328 |

Each command must use the approval-gated runner and its external deadline. Do not reset a command
deadline, silently retry, change a candidate recipe, change a stage limit, or use a partial
artifact. Preserve every failed version and record it as a technical failure. Perform independent
persisted read-back before accepting each feature or alignment artifact. Freeze exact versions,
content hashes, physical bytes, and measured GPU-hours before a later stage can load them. The
earlier test-contamination record remains in force. E02b is validation-only.

## 2026-08-23 16:15:19 BST — E02b Carbon-500M DNA features accepted

**Status:** passed independent persisted-artifact read-back. The stage extracted frozen DNA
features only. It did not fit an alignment probe, calculate a validation ranking, or select a
candidate.

The clean `g6.2xlarge` worktree at Git commit
`a6aa0e9a6a5f9fadf4fcbc5f7f604530aca0a6bc` loaded the exact accepted E02b input and invariance
artifacts. The command completed in 1,150.54 seconds under its 0.75-hour limit. The measured
command cost was $0.312436 at $0.9776 per instance-hour. The feature node used 1,091.46 seconds,
processed 235,586,356 unique base pairs at 215,844 bp/s, and used 1,286,269,952 peak allocated
device bytes.

Version `2026-08-23T14.54.08.128Z` contains 30,821 unique L2-normalized 1,024-dimensional vectors
for 30,852 source rows and 30,831 exact circular-window coverage rows. Independent read-back bound
the accepted input, numerical-smoke, invariance, recipe, Transformers 5.12.1 runtime, clean Git
state, and exact compute authorization. It reconstructed every circular window and matched every
per-row and table hash. The persisted artifacts use 162,999,200 bytes.

Freeze these identities:

- manifest: `3115b841c90921cca59c330c6956bbe1062b2a7ce4fc975f043174c5638a753c`;
- features: `bb86fd296a3ba2ed2fdafcb7e21b89c918a11b2a4c2ec9ba8727af0c0713be9a`;
- coverage: `33cef5d12055f9845743ab4e82f35edde74cd05a85272b3496115562b743ee21`;
- extraction GPU-hours: `0.30318436455498965`.

No validation ranking was computed. Carbon-500M remains one candidate in the frozen factorial; this
feature acceptance is not a model-selection result.

## 2026-08-23 16:18:56 BST — E02b BGE-base text features accepted

**Status:** passed independent persisted-artifact read-back. This stage encoded frozen document
and query-role text only. It did not read query states, fit an alignment probe, calculate a
validation ranking, or select a candidate.

The clean `g6.2xlarge` worktree at Git commit
`1f7effc08119aad90d0e2829581b2c12ec99dfc3` loaded the accepted E02b pairs, query texts, and input
manifest. The command completed in 80.97 seconds under its 0.5-hour limit and cost $0.021987 at the
observed instance price. The feature node used 38.17 seconds, encoded 810.90 texts/s, and used
833,486,848 peak allocated device bytes.

Version `2026-08-23T15.16.22.538Z` contains all 30,844 unique paired descriptions and all 108
unique frozen query texts as finite L2-normalized 768-dimensional vectors. The maximum observed
prompted input was 258 tokens, below the frozen 512-token stop boundary. Independent read-back
matched the exact role-specific text hashes, per-row vector hashes, model recipe, Transformers
5.12.1 runtime, input hashes, clean Git state, and compute authorization.

Freeze these identities:

- manifest: `9855a2b15e851d77cd9019d9c591bb3c927d009c5642de6f829d2d2052a84741`;
- features: `5a69bb6647ac977778c0a3d6253bbbe94a8b9db2a2790ea3eac6590e6899f14a`;
- extraction GPU-hours: `0.010602783840149642`;
- persisted bytes: `92,373,563`.

No validation ranking was computed. BGE-base remains one text candidate in the frozen factorial.

## 2026-08-23 16:22:56 BST — E02b GTE-ModernBERT text features accepted

**Status:** passed independent persisted-artifact read-back. This stage encoded frozen document
and query-role text only. It did not read query states, fit an alignment probe, calculate a
validation ranking, or select a candidate.

The clean `g6.2xlarge` worktree at Git commit
`569d35492df41e03428b96da6001b2db6d30eb63` loaded the accepted E02b pairs, query texts, and input
manifest. The command completed in 127.19 seconds under its 0.75-hour limit and cost $0.034539 at
the observed instance price. The feature node used 84.38 seconds, encoded 366.80 texts/s, and used
881,805,312 peak allocated device bytes.

Version `2026-08-23T15.19.38.226Z` contains all 30,844 unique paired descriptions and all 108
unique frozen query texts as finite L2-normalized 768-dimensional vectors. The maximum observed
input was 179 tokens, below the frozen 8,192-token stop boundary. Independent read-back matched
the exact role-specific text hashes, per-row vector hashes, model recipe, Transformers 5.12.1
runtime, input hashes, clean Git state, and compute authorization.

Freeze these identities:

- manifest: `a94551c87aec31085c97c7fcbf362722b196cc510c3dcd3fef7fb9249ee1edd9`;
- features: `90139f7e41c1cf30b929c7ceebcca3496466a6b21620897734c62826166a7443`;
- extraction GPU-hours: `0.02344021446382006`;
- persisted bytes: `92,409,461`.

No validation ranking was computed. GTE-ModernBERT remains one text candidate in the frozen
factorial.

## 2026-08-23 16:29:32 BST — E02b Qwen3-Embedding text features accepted

**Status:** passed independent persisted-artifact read-back. This stage encoded frozen document
and query-role text only. It did not read query states, fit an alignment probe, calculate a
validation ranking, or select a candidate.

The clean `g6.2xlarge` worktree at Git commit
`f8a6d00cc1755dc41042bf56d713be119cf62a98` loaded the accepted E02b pairs, query texts, and input
manifest. The command completed in 262.82 seconds under its 0.75-hour limit and cost $0.071371 at
the observed instance price. The feature node used 207.81 seconds, encoded 148.95 texts/s, and used
2,067,610,112 peak allocated device bytes.

Version `2026-08-23T15.23.41.974Z` contains all 30,844 unique paired descriptions and all 108
unique frozen query texts as finite L2-normalized 1,024-dimensional vectors. The maximum observed
prompted input was 189 tokens, below the frozen 32,768-token stop boundary. Independent read-back
matched the exact role-specific text hashes, per-row vector hashes, model recipe, Transformers
5.12.1 runtime, input hashes, clean Git state, and compute authorization.

Freeze these identities:

- manifest: `17d2d025a560f34355ca244cfd629178ec0c6c5767159bebf6d44116dcda7235`;
- features: `c5cbf04e7d034ea4dbe0ad874adc728e6d8ae8c75ee7cbac278cbb7c44218d8d`;
- extraction GPU-hours: `0.05772405528359943`;
- persisted bytes: `118,110,567`.

No validation ranking was computed. All three frozen text candidate feature products are now
accepted. Alignment remains blocked until the two remaining neural DNA candidates pass the same
independent read-back and are frozen.

## 2026-08-23 17:21:49 BST — E02b GENERator-v2 DNA features accepted

**Status:** passed independent persisted-artifact read-back. The stage extracted frozen DNA
features only. It did not fit an alignment probe, calculate a validation ranking, or select a
candidate.

The clean `g6.2xlarge` worktree at Git commit
`ccc9b855521c48ce423eee916c3bcdeecacd79be` loaded the exact accepted E02b input and invariance
artifacts with Transformers 4.49.0. The command completed in 2,919.79 seconds under its 1.5-hour
limit. The measured command cost was $0.792887. The feature node used 2,814.01 seconds, processed
235,586,356 unique base pairs at 83,719 bp/s, and used 2,887,000,576 peak allocated device bytes.

Version `2026-08-23T15.30.18.164Z` contains 30,821 unique L2-normalized 2,048-dimensional vectors
for 30,852 source rows and 30,821 exact circular-window coverage rows. Independent read-back bound
the accepted input, numerical-smoke, invariance, recipe, runtime, clean Git state, and exact
compute authorization. It reconstructed every circular window and matched every per-row and table
hash. The persisted artifacts use 313,980,591 bytes.

Freeze these identities:

- manifest: `b1d96a4afbcb60f0e970529d1c0d5813c237d3ed7acce5f201336b9f655b1074`;
- features: `72642a470e4f150316170dae620795f7b246b46b6cbcad0b59571930296bf7a1`;
- coverage: `0c2370e7f188cbb116b1383518fecc73786f8eb50b4c798a3e12f851e41e2efd`;
- extraction GPU-hours: `0.7816696715965453`.

No validation ranking was computed. GENERator-v2 remains one candidate in the frozen factorial.

## 2026-08-24 01:35:22 BST — E02b GENERanno partial artifact rejected

**Status:** failed technical run. Do not accept or reuse version
`2026-08-23T16.23.06.763Z`. The alignment factorial did not start, no validation ranking was
computed, and no candidate was selected.

The exact approved GENERanno command started at 2026-08-23 16:23:05 UTC on the existing
`g6.2xlarge`. It used clean Git commit
`3afc292f85222f22d477bbac40be37c55d7dac56`, Transformers 4.49.0, the frozen E02b input,
accepted invariance version `2026-08-23T00.53.22.430Z`, approval reference
`chat-2026-08-23-e02b-complete-benchmark`, and the unchanged 7-hour command cap.

**Observed failure evidence:**

- systemd removed remote login session `50226` at 2026-08-23 21:20:37 UTC after 4 hours 57
  minutes and recorded 4 hours 57 minutes of consumed CPU time;
- no kernel out-of-memory event appeared in the inspected 21:10–21:30 UTC system log interval;
- the command log contains pipeline startup but no completion, timeout, Python exception, or
  post-run cost record;
- S3 contains a 208,302,597-byte feature object for version
  `2026-08-23T16.23.06.763Z`, but contains no matching coverage or manifest object;
- the partial table has 30,821 unique GENERanno sequence hashes, dimension 1,280, and content hash
  `7b6f6cbf5d0fa495599a50b3a3cbfc75e1727ccfa656145922ad69fd3773f0ed`;
- independent read-back failed with `FileNotFoundError` for the versioned coverage product.

The feature object exists, and the next two required artifacts do not. The available evidence does
not establish the exact sequence between feature upload and session removal. The cause of session
removal is unknown. This is an orchestration failure, not evidence about retrieval quality.
Coverage, per-row window accounting, runtime manifest, extraction GPU-hours, and final output
hashes are unavailable, so the partial feature table cannot pass the frozen acceptance contract.

**Derived cost:** the approximately 17,852-second failed session cost approximately $4.847810 at
$0.9776 per instance-hour, before storage and transfer. Completed E02b neural feature commands and
this failed command total approximately $6.081030 in measured-command instance charges. Complete
host charges are higher because the host remained running between commands.

**Stopping decision:** preserve the partial S3 object and both run/read-back logs as failed-run
provenance. Do not retry, reconstruct the missing products, change parameters, or start alignment
without new user authorization. The next decision is whether to approve one exact GENERanno retry
with orchestration that is independent of a terminal or login session.

## 2026-08-24 12:00:33 BST — E02b host provenance corrected and completion authorized

**Status:** authorization recorded before retry results. No encoder or alignment command was
running when this entry was written.

AWS read-back identified SSH host `g6-big`, hostname `ip-172-31-90-236`, as on-demand instance
`i-0cda00ffb3cacfc12`, type `g6.4xlarge`, in `us-east-1b`. The 2026-08-23 E02b manifests recorded
the same host as `g6.2xlarge` at `$0.9776` per hour. AWS's current price record gives `$1.3232` per
on-demand Linux `g6.4xlarge` instance-hour. This is a provenance and derived-cost error. It does
not change any model output or artifact hash. Preserve the immutable manifests and attach the
correction in configuration.

At the corrected rate, the rejected 17,852-second GENERanno command cost approximately
`$6.561602`, not `$4.847810`. Scaling the previously reported completed-plus-failed command time
to the corrected rate gives approximately `$8.230789`, not `$6.081030`. Complete host charges are
higher because the instance remained running outside the measured commands.

The partial GENERanno version contains a complete-looking 30,821-row feature table but no coverage
or manifest. The pipeline has no persisted extraction checkpoint. Do not reuse or complete it.
Run the full stage under a new version with the same input, accepted invariance artifact, model and
tokenizer revisions, precision, pooling, windowing, seed, and seven-hour deadline. Use a detached
system service so SSH session removal cannot terminate the run.

The user authorized the GENERanno retry and the full frozen alignment factorial with approval
reference `chat-2026-08-24-e02b-finish-benchmark`. The actual host is `g6.4xlarge` at `$1.3232`
per hour. The GENERanno cap is seven hours and `$9.2624`; the alignment cap is three hours and
`$3.9696`. The combined additional cap is ten hours and `$13.2320` before storage and transfer.
No other paid stage is authorized.

## 2026-08-24 17:31:00 BST — E02b GENERanno DNA features accepted after detached retry

**Status:** passed independent persisted-artifact read-back. The retry extracted the frozen DNA
features only. It did not fit an alignment probe, calculate a validation ranking, or select a
candidate.

The detached systemd command used clean Git commit
`6534e2eda05776218e4f61979f6b3d729496c957`, Transformers 4.49.0, bfloat16, the pinned
`GenerTeam/GENERanno-prokaryote-0.5b-base` revision
`d02db0f24f2c62fa1efde760217cdf75771b0228`, accepted input version
`2026-08-23T14.16.15.778Z`, and accepted invariance version
`2026-08-23T00.53.22.430Z`. It ran on the NVIDIA L4 in on-demand `g6.4xlarge` instance
`i-0cda00ffb3cacfc12` in `us-east-1b`. The service started at 2026-08-24 11:03:44 UTC and
completed successfully at 16:01:00 UTC. The wrapper measured 17,835.84 seconds and a command
charge of `$6.555663` at `$1.3232` per instance-hour, below the seven-hour and `$9.2624` limits.
The feature node measured 17,754.93 seconds, or 4.931926 GPU-hours.

Version `2026-08-24T11.03.45.874Z` contains 30,821 unique L2-normalized 1,280-dimensional vectors
for 30,852 source rows and 42,700 circular-window coverage rows. Independent read-back verified
every source identity, complete base coverage, model and tokenizer revisions, runtime recipe,
accepted input and invariance identities, clean Git state, compute authorization, and physical
object size. The three persisted products use 209,933,751 bytes.

Freeze these identities:

- manifest: `cdbedeeee110900d602d399968a1be0b0d614f65007dc07d689fa4e492a3d13b`;
- features: `8bd9b216632bbc2d225001ff225910e65a76228f72f8adbcf1d8129bed1d5c37`;
- coverage: `6d721dea4a3ac83b9866ff4ead7d6fe3ef5fad601dd8107ce50b48889e6f5eb6`;
- extraction GPU-hours: `4.931926208106387`;
- persisted bytes: `209,933,751`.

The rejected version `2026-08-23T16.23.06.763Z` remains preserved and rejected. It was not used
to create or validate the accepted version. All seven frozen feature products are now accepted.
The authorized 4-DNA by 3-text by 3-seed alignment factorial can start after this registry change
passes tests and is present in a clean remote checkout.

## 2026-08-24 18:05:00 BST — E02b alignment accepted and validation pair selected

**Status:** complete validation-only selection. All 36 planned DNA-by-text-by-seed configurations
completed and passed independent persisted-artifact read-back. No test row was read. Gate 2 did
not start.

The detached alignment command used clean Git commit
`410fa42c280716dde5461535d7a1baef109bec57` on the NVIDIA L4 in the corrected on-demand
`g6.4xlarge` host. It used the four accepted DNA candidates, three accepted text candidates, and
seeds 13, 42, and 20260818. The command started at 2026-08-24 16:34:47 UTC and completed at
16:52:52 UTC. The wrapper measured 1,085.08 seconds and `$0.398828` at `$1.3232` per instance-hour,
below the three-hour and `$3.9696` limits. The alignment node measured 1,053.36 seconds.

Version `2026-08-24T16.34.48.358Z` contains all nine planned outputs. Independent read-back
recomputed the frozen feature bindings, whitening state, 36 training histories and checkpoints,
194,400 validation rankings, 15,552 query-metric rows, and 72,000 whole-component bootstrap rows.
It verified the complete factorial and reapplied the selection rule. The outputs use 245,589,153
bytes.

The selected pair is 6-mer TF-IDF/SVD DNA plus Qwen3-Embedding-0.6B text. Its mean validation
query-macro `utility@10` is `0.153086` with a whole-component 95% interval of
`[0.076227, 0.188279]`. Seed utilities are `0.155556`, `0.158333`, and `0.145370`. The selected
mean contains verified, contradicted, and unknown fractions `0.425617`, `0.272531`, and `0.301852`
at K=10. Atomic-query utility is `0.602381`; pair-conjunction utility is `-0.004167`. The positive
combined result therefore does not establish successful conjunction retrieval.

The Carbon-500M plus BGE-base incumbent mean is `-0.041049` with interval
`[-0.088580, 0.008642]`. The selected pair improves the frozen primary metric by `0.194136`, which
exceeds the `0.01` retention guard. It is not a practical tie under the preregistered rule. The
runner-up is GENERanno plus Qwen at `0.141049` with interval `[0.057701, 0.157716]`.

Paired identity retrieval also favors the selected pair. Across seeds, its sequence-to-description
R@1/R@10 are approximately `0.1279`/`0.3711`, and description-to-sequence R@1/R@10 are
approximately `0.1434`/`0.3926`. The incumbent values are approximately `0.0810`/`0.2815` and
`0.0813`/`0.2825`, respectively. These are secondary metrics and did not control selection.

Freeze these identities:

- selection report: `a675a3a3fac1b87827749764caeea07a395debf86c0ee886998417fd9a5b8d25`;
- whitening state: `1415be70f5eb8a4be5ad6c42f62dc53d4ef569c7c4889178cf9ea3556c12d8be`;
- probe checkpoints: `d8c81804272afe3c9af971b90cf39abc81cfc8c9db151c6fa5e9c38bd6e255ea`;
- training history: `fbd52828e593215fdce38b88fc41408b63fccb8a0c756c7a6deba51f7c7f2fa3`;
- paired metrics: `92cd682a145459d5d35db025770b387fc0c9db976a86b37a9bbf2599d866a427`;
- query rankings: `cd8e488f078d284a792f5d0a05a2fc7b1b47f5cdc3ee5b3cb31ed86deb947dac`;
- query metrics: `c0f0b0346e1585c82bee31018beaa34b43b3f66a506e0810c46118367bb17bc0`;
- query summaries: `b924889bc91205379b8c609f68dbbc07b84d9b240e01feabf1c3bfbcca3f2f69`;
- bootstrap draws: `0c8f65a9b12c0ad1dc032dc9249b39f7c814fa3c6b051afd595f22b66da2f8a7`.

The retry and alignment together used 5.2558 wrapper instance-hours and `$6.954491` of the
additional ten-hour and `$13.2320` cap. Adding the corrected previously recorded commands gives an
approximate E02b measured-command total of `$15.185280`. Complete host charges are higher because
the shared host also ran outside the timed commands.

The primary hypothesis that a neural prokaryote-specific or larger DNA encoder would win was not
supported by this validation selection. TF-IDF/SVD had the highest descriptive mean across text
encoders (`0.072634`), followed by GENERanno (`0.056893`), Carbon-500M (`0.001235`), and
GENERATOR-v2 (`-0.017284`). Qwen had the highest descriptive mean across DNA encoders (`0.102315`);
GTE and BGE were `-0.006481` and `-0.010725`. These factorial means are descriptive, not causal.

Next, freeze this pair for the Gate 2 experiment definition. Full-population extraction is a
separate post-selection action and requires a resolved cost estimate and approval. Do not use the
contaminated current test split for a confirmatory claim.

## 2026-08-25 — E03/E04 Gate 2 completed

The frozen comparison used 108 controlled queries, the 20,000-row training panel, TF-IDF/SVD DNA,
Qwen3 text, two objectives, and three seeds. Artifact `2026-08-25T10.52.39.447Z` passed independent
read-back for all seven table hashes, 6 checkpoints, 1,800 updates, 32,400 rankings, 2,592 query
metric rows, and 12,000 bootstrap rows. No test row was read.

Verified-set supervision improved pair-query utility@10 from `0.30875` to `0.48792`. The paired
difference was `0.17917`, with whole-component interval `[0.10375, 0.25294]`, so the frozen decision
rule passed. All six W&B runs completed. The accepted clean run used commit `fe6fbaec`, 208.7
wrapper seconds, and `$0.07672`; total measured cost including a 1.15-second pre-data wrapper
failure was `$0.07714`. The failed wrapper log remains on `g6-big`.

## 2026-08-26 — E05 unseen-composition comparison preregistered

E05 asks whether the Gate 2 benefit survives when conjunctions are genuinely absent from
training. Both objectives use the frozen 20,000-row panel, TF-IDF/SVD DNA, Qwen3 text, 28 atomic
queries, 300 updates, and seeds 13, 42, and 20260818. All 80 evaluation queries are the existing
`atoms_seen_conjunction_unseen` pairs; every constituent atom is trained, no pair-query vector or
pair label is trained, and the 10,852-row validation gallery shares zero similarity components
with training. No test row may be read.

The primary endpoint is pair-query utility@10. Verified-set supervision passes only if its mean
improvement over paired identity is at least `0.01` and the lower bound of the paired 2,000-draw
whole-component 95% bootstrap interval is above zero. The user authorized the clean detached run
on the existing `g6.4xlarge` L4 under reference
`chat-2026-08-26-e05-unseen-composition`, capped at 0.5 instance-hours and `$0.6616`. W&B group
`e05-unseen-composition-v0.1` is required; exact outputs and failures go to the E05 S3 report.

## 2026-08-26 12:46:47 BST — E05 first attempt rejected after provenance check

Unit `vec2vec-e05-unseen-composition-20260826.service` ran clean commit `2453827` for 196 seconds
on the L4, approximately `$0.07204`. All six model fits and W&B uploads completed, but the command
exited before saving any S3 result because W&B had created expected untracked files and the runner
incorrectly checked Git cleanliness after training. Runs `dtnlwtsd`, `sibfa8ox`, `suswbgsy`,
`l8s9belh`, `lqhk2isz`, and `xbxcxrtl` are rejected evidence; no E05 S3 object exists.

The retry will not change data, queries, objectives, seeds, optimizer, metric, or decision rule.
The fix records Git state before W&B starts and ignores the local W&B cache. The retry is capped at
0.44 hours, leaving the combined service ceiling below the originally authorized 0.5 hours and
`$0.6616`.

## 2026-08-26 12:52:42 BST — E05 unseen-composition comparison accepted

The unchanged retry at clean commit `10485e8` trained only the 28 atomic queries and evaluated all
80 held-out conjunctions on the component-disjoint validation gallery. Paired-identity utility@10
was `-0.12250`; verified-set utility was `0.17417`. The improvement was `0.29667`, with paired
whole-component interval `[0.22499, 0.34292]`, so the frozen rule passed. No test row was read. All
six W&B runs completed: `8ekn65i7`, `5pwsx6rv`, `h6clzct3`, `s6cyojzj`, `i6jwppli`, and
`t92ya6us`.

S3 version `2026-08-26T11.49.24.525Z` passed independent report, summary, input, feature, query
partition, seed, runtime, Git, tracking, metric, interval, and hash read-back. Report SHA-256 is
`182fb0dd75a1bd3159bc24b488e4921ff7dac1c6292352a408c8ff6d2d44082d`; summary SHA-256 is
`91afb8ba2185a70dbbb609a3030b406d9b2195382f502e5ca19c85b3098a42d4`. The accepted wrapper used
202.92 seconds and `$0.07458`; combined measured cost including the rejected attempt was
approximately `$0.14663`.

## 2026-08-26 — E06 population-scale composition protocol frozen; compute pending

E06 asks whether the E05 supervision effect survives replacing the selected 20,000-row training
panel with all 88,474 uppercase-ACGT training rows. It keeps the 10,852-row component-disjoint
validation gallery, 28 atomic training queries, 80 unseen pair-conjunction queries, two objectives,
three seeds, 300 updates, metric, bootstrap, and decision rule unchanged. TF-IDF/SVD is refit on
the full eligible training population and Qwen3 text features are regenerated for the expanded
panel. This changes data scale, not the encoder recipes or supervision comparison.

The deterministic pre-outcome panel contains 99,326 rows. Its pairs SHA-256 is
`90d9157d8695c788ec162a48a6a3cb2f9ae33be725345ff0e955840d0ecb21de`; the exclusion, query, and
query-state hashes remain `f4b561f3a8173c75b7619d27b222924118afddfb3391f21e20346a48a242bf45`,
`a440e26a32468a9e613aaa2034b476b453a8bae988c08128df3f14f251c4552c`, and
`06afff85e1d4b3d96f7c62eeadd3ba50892b26cae558aec64a5bcba41c37ce8b`. No model outcome was read.

The primary endpoint remains validation pair-query macro utility@10. Verified-set supervision
passes only if its mean improvement over paired identity is at least `0.01` and the lower bound of
the paired 2,000-draw whole-component 95% bootstrap interval is above zero. The historical test
split remains contaminated, and S3 has no newer Addgene raw snapshot, so E06 is an exploratory
scale-robustness result rather than confirmation. The user authorized a clean detached
`g6.4xlarge` L4 run under reference `chat-2026-08-26-e06-population-scale-auto-under-20`, with a
combined 0.75 instance-hour / `$0.9924` ceiling at the current `$1.3232` hourly price. Feature
generation is limited to 0.50 hours and comparison to the remaining 0.25 hours.

## 2026-08-26 14:02 BST — E06 population-scale features accepted after technical retry

The input build completed in 111 seconds at clean commit `7a27b45`, producing version
`2026-08-26T12.24.14.212Z`. Independent read-back verified 88,474 training rows, 10,852 validation
rows, 108 queries, the four preregistered table hashes, and input-manifest SHA-256
`880cea9088d64720032fdd3b6ef70aa8d99e006908168f539fd432924e8c9362`.

The first concurrent TF-IDF and Qwen commands completed their feature calculations but failed
before returning any Kedro output because detached systemd ran as root and Git rejected the
Ubuntu-owned worktree during final provenance capture. No E06 feature or feature-manifest object
was written. The preserved units ran from 12:27:39 to 12:42:10 UTC. The unchanged retry ran both
units as Ubuntu from 12:43:28; Qwen completed at 12:56:47 and TF-IDF at 12:57:58 UTC.

Version `2026-08-26T12.43.28.764Z` passed independent persisted read-back. It contains 99,188
unique normalized 512-dimensional TF-IDF/SVD vectors and 99,396 normalized 1,024-dimensional Qwen
vectors: 99,288 documents plus 108 queries. Every source identity, per-row embedding hash,
dimension, finite value, normalization, recipe, model revision, input binding, Git state, and table
hash matched. Freeze these identities:

- DNA features `4376e4e0cec03dcfa6665239436f396818648aab5ef2d7c9bfd518ad537e6fe0`;
- vocabulary `9b85d6a85329aa19f42b7880aa2e8b2f178e83118c9f702f148a395e7f3bf117`;
- SVD state `8bc96e926545cfa95283270c0a60e142e44dace20fe0fd399ec432a218e21aee`;
- Qwen features `9c131e45a457163f141e840056faf383a2ee0a7a84fc9a967e361d41aa5c2fce`;
- DNA manifest `cca81b327b40b1aede4ae1027d0a75cfdf35bd4f3400861eeb9d4519de47d0e5`;
- text manifest `2e5567275d2f2c664730438dfa2643a7fb0bec5710785ab3ecba77b98dafa66d`.

Input build, rejected feature attempt, and accepted retry used approximately 0.5144 wall-clock
instance-hours and `$0.68071`. The comparison is capped at 0.20 hours / `$0.26464`, keeping the
combined hard ceiling at approximately `$0.94535`, below the original `$0.9924` authorization.
No model outcome has been read.

## 2026-08-26 14:14 BST — E06 population-scale comparison accepted

The clean detached comparison at commit `962e12e` used all 88,474 eligible training rows, the
unchanged 10,852-row validation gallery, 28 atomic training queries, and 80 unseen conjunctions.
Paired-identity utility@10 was `-0.10167`; verified-set utility was `0.17333`. The improvement was
`0.27500`, with paired whole-component interval `[0.18707, 0.33878]`, so the frozen rule passed.
No test row was read. Independent W&B read-back found all six runs finished with the frozen E06
group: `z3iklyeb`, `z12cfncc`, `2nw6hs66`, `mekvk889`, `jip3org4`, and `032oihq4`.

S3 version `2026-08-26T13.06.51.803Z` passed independent report, summary, population, objective,
seed, tracking, runtime, Git, metric, interval, and hash read-back. Report SHA-256 is
`800e23597f209197835b9648c4663cc3d35e686d0ac59e04c50bf1838e015230`; summary SHA-256 is
`135a3b10cbd70fd991331caa430e8fdbe6d2636b23d1fd8e30f9404d1acde764`. The comparison wrapper
used 426.26 seconds and approximately `$0.15667`. The complete scoped E06 work, including input
construction, the rejected feature attempt, accepted feature retry, and comparison, used
approximately `$0.83739`, below the authorized `$0.9924` ceiling.

E05 and E06 agree: verified-set supervision improves retrieval for unseen conjunctions under the
frozen validation protocol, and the effect is not an artifact of the 20,000-row training sample.
This remains exploratory because the historical test split is contaminated and no new raw-data
snapshot exists for a clean confirmatory holdout.

## 2026-08-27 — Final model fit accepted

Final-model-v1 collapsed the historical splits after model selection and fit the accepted
TF-IDF/SVD + Qwen3 + verified-set recipe to all 110,267 A/C/G/T plasmids. The clean L4 run at
commit `7856556` finished in 1,376.90 seconds before persistence, cost approximately `$0.50609`,
and logged to W&B run `m4eeei4w`. Final loss was `0.42155`; this is an optimization diagnostic,
not an evaluation result.

The validated 255.6 MB bundle and 110,267-row index are under
`hf://buckets/McClain/plasmidclip-train-ckpts/models/vec2vec-final-v1/78565560b8473b9d1145cc9818084af63dfe0702`.
Manifest SHA-256 is `284a1315ae1c39b2624f09f78ef3ec0f18e8f40fd8f0c0e11d96d274b61c877e`;
model and index SHA-256 values are `7b489c29d7dd765ac18910ac0aebb364ca81dbbadc7607ba54aa35697526f1f6`
and `0d4d5450cf191e7e2aeafbf59c052a548cde3b0a8b4c43bf14c5292fc5ea8764`. Independent
read-back reproduced sampled index vectors within `2.54e-7`. No evaluation was performed.

## 2026-08-27 — E07 additive retrieval audit preregistered

E05/E06 trained only on 28 atomic queries but evaluated the Qwen encoding of each unseen
conjunction; they did not evaluate the original hypothesis that `q_A + q_B` retrieves the
intersection. E07 closes that gap without tuning. It reproduces both accepted E06 objectives on
88,474 training rows for seeds 13, 42, and 20260818, and audits additivity on the verified-set
fits. It compares direct conjunction text with the sum of the two projected atomic vectors on the
same 80 unseen conjunctions and 10,852-row component-disjoint validation gallery. The primary
metric is pair-query utility@10; the paired whole-component bootstrap reports
`atomic_sum - direct_text`. Mean Jensen-Shannon divergence of their full-gallery distributions is
secondary. No test row may be read.

This is an audit of the accepted cosine-normalized baseline, not the unnormalized maximum-entropy
model proposed in the original study plan. The result is exploratory and selects no
hyperparameters. The run is capped at 0.25 `g6.4xlarge` instance-hours and `$0.33080` under
approval reference `chat-2026-08-27-additive-audit-auto-under-20`.

## 2026-08-27 — E07 additive retrieval audit accepted

The clean L4 run at commit `5331b73` reproduced the accepted E06 direct-text results and all
checkpoint, training-history, and whitening hashes. Atomic-sum utility@10 was `0.19333` versus
`0.17333` for direct conjunction text, a difference of `0.02000` with paired whole-component
interval `[-0.01792, 0.06043]`. Atomic-sum utility itself remained positive, interval
`[0.11291, 0.21876]`. Mean full-gallery Jensen-Shannon divergence between the two induced
distributions was `0.14578`.

Thus vector addition retrieves valid unseen conjunctions under the cosine baseline, but there is
no clear evidence that it outperforms direct conjunction encoding. This supports continuing to the
original unnormalized maximum-entropy formulation; it does not establish the proposed natural-
parameter algebra. S3 version `2026-08-27T15.33.47.015Z` passed read-back. Report and summary hashes
are `a9a1ca17eab82fca3c4774326013034552dbf34583239fddcb19c5017515e275` and
`54fa9a4e1ace084ac3ae69068a5f4234ec77e80f072a5d37edd0646ca0ddabdb`. All six W&B runs finished.
The measured stage used 414.99 seconds and `$0.15253`; the wrapper used 436.65 seconds and
`$0.16049`.

## 2026-08-27 — E08 natural-parameter model preregistered

E08 implements the original maximum-entropy formulation: unnormalized 512-dimensional linear
projections, fixed temperature `0.07`, exact normalization over every known training candidate,
unknown candidates excluded, and scores `log μ(x) + q·z/τ`. It compares uniform-plasmid and
uniform-v2-component base measures over three frozen seeds, with 300 updates and no tuning. The
primary endpoint is atomic-sum utility@10 on the component-disjoint validation gallery; the paired
2,000-draw v2-component bootstrap compares the two base measures. Direct conjunction text and
direct-versus-sum distribution divergence are secondary. No historical test row may be read.

Only four of the 28 frozen atoms have both verified and contradicted training candidates: low/high
copy class and 30/37 °C growth temperature. Treating the other 24 atoms' unknown candidates as
negative would violate the study contract, so E08 freezes these four atoms and their four unseen
cross-facet conjunctions. This is an honest but underpowered controlled test, not a general
retrieval claim. The run is capped at one `g6.4xlarge` hour and `$1.3232` under standing approval
reference `chat-2026-08-27-e08-natural-parameters-auto-under-20`.

## 2026-08-27 — E08 first execution rejected for nonfinite secondary diagnostic

The clean run at commit `0dbdf93` completed all six frozen fits and W&B runs, but the probability-
space Jensen–Shannon calculation underflowed for extreme logits and persisted `NaN`. S3 version
`2026-08-27T16.08.12.789Z` is rejected and retained. The wrapper used 427.82 seconds and `$0.15725`.
The technical retry may change only the log-space JSD calculation; it must reproduce the first
run's checkpoint, history, summary, bootstrap, and whitening hashes exactly.

## 2026-08-27 — E08 completed as an optimization failure

The unchanged retry at commit `c6a1b45` reproduced every frozen model and primary-result hash.
Uniform-plasmid atomic-sum utility@10 was `-0.43333` (interval `[-0.50000, -0.32500]`);
uniform-v2-component was `-0.50833` (interval `[-0.63333, -0.37479]`). The component-minus-
plasmid difference was `-0.07500`, interval `[-0.24188, 0.07500]`, so neither base measure was
selected. Atomic sum was not distinguishable from direct text under either measure. Stable mean
Jensen–Shannon divergences were `0.39462` and `0.46549`.

This does not reject the natural-parameter hypothesis: optimization diverged in all six fits.
Loss rose from roughly `11.6–13.1` after the first update to `317–1,134`, while query norms grew to
means of `15.7–34.0`. The next experiment must stabilize scale/optimization on validation before
another algebra comparison. Accepted evidence version `2026-08-27T16.17.31.598Z` has report
SHA-256 `178eb98d922296c89839c536f84773453c60fa0768eb1205d53815a0736da4c7`; all six W&B runs
finished. The accepted wrapper used 428.57 seconds and `$0.15752`; including the rejected
diagnostic attempt, E08 used `$0.31477`.

## 2026-08-27 — HF annotation audit and E09 stability calibration preregistered

The private HF table `full158k-structured-v1/full158k_structured.parquet` contains the same
115,120 plasmids used by this project. Every row has positive feature calls (median 37; 248,432
distinct strings), but the table has no feature-caller version, coverage record, or explicit
absence evidence. It therefore does not justify turning an uncalled feature into a negative. E09
keeps the four E08 atoms with reviewed closed-world contradictions and excludes unknowns.

E09 varies only the learning rate over `[1e-6, 3e-6, 1e-5, 3e-5, 1e-4]`, holding the E08 model,
two base measures, three seeds, 300 updates, temperature, weight decay, features, and data fixed. A
rate is eligible only if all six fits finish with final loss at most 99% of initial loss, no loss
above 110% of initial loss, and query/sequence norms no larger than 5. The eligible rate with the
lowest mean final training loss is selected; validation retrieval is not consulted during
selection and is evaluated once afterward. The `g6.4xlarge` ceiling is 1.25 hours / `$1.654` under
reference `chat-2026-08-27-e09-natural-parameter-stability-auto-under-20`.

## 2026-08-27 — E09 natural-parameter optimization stabilized

All six fits were stable at `1e-6`, `3e-6`, `1e-5`, and `3e-5`; all six `1e-4` fits failed the
frozen stability rule (worst loss `31.99`, query norm `10.47`, sequence norm `6.75`). Training-only
selection chose `3e-5`, whose mean final loss was `9.95743` versus mean initial loss `12.40728`.
The six selected fits were then evaluated once on validation. All 36 W&B runs finished under group
`e09-natural-parameter-stability-v0.1`.

Uniform-plasmid atomic-sum utility@10 was `0.37500` (interval `[-0.10000, 0.72500]`) versus direct
text `-0.23333`; their paired difference was `0.60833`, interval `[0.10833, 0.82500]`.
Uniform-component atomic sum was `0.22500` (interval `[-0.02500, 0.55000]`) versus direct text
`0.10000`; difference `0.12500`, interval `[-0.08333, 0.40833]`. The component-minus-plasmid
atomic-sum difference was `-0.15000`, interval `[-0.52500, 0.37500]`, so no base measure was
selected. This establishes that E08's failure was optimization-induced, but four conjunctions are
too few for a general algebra claim. Accepted S3 version `2026-08-27T17.30.52.908Z` has report
SHA-256 `f1e844788ed5bc6b8f063e991cee2f50094f0c5c9bb0bef8680e60952ea86bab`; wrapper time was
522.16 seconds and cost `$0.19192`.

## 2026-08-27 — E10 weak-annotation scale experiment preregistered

**Question:** Does the stable natural-parameter model retrieve many annotation-defined features
and unseen two-feature conjunctions when an uncalled feature is treated as a noisy weak negative?

**Weak-label assumption:** a positive means the pinned annotation list contains the normalized
feature name; a weak negative means the same pipeline did not report it. This is not evidence of
biological absence. The experiment is exploratory and cannot support a confirmatory biological
claim.

**Frozen data:** join the accepted E06 component-disjoint panel (88,474 train; 10,852 validation)
to the private HF bucket object `full158k-structured-v1/full158k_structured.parquet`. The object has
Xet hash `6d247cfad610042bdac978b402d6a44f20d72716a30d03870cd0346d3b7f250a`, file SHA-256
`eaf4ef6885aded6e984f974c71f1c32ffb08b74cf1cf96aa69af8d6f3993f855`, and 115,120 rows. The
historical test split will not be read.

**Frozen vocabulary and queries:** normalize only spelling, case, Unicode, and punctuation; do not
merge biological aliases. Select 64 atoms by fixed train/validation support and component rules,
removing near-duplicate call sets above Jaccard 0.95. Select 128 supported conjunctions with train
Jaccard at most 0.80, maximum degree six, and at least one conjunction per atom. The resulting query
table SHA-256 is `b93a73db58b9a149a8458e5cd36bcd03f70997aee12e013c7d27908274275770`.
There are 551,749 positive training pairs, 1,230,048 deterministically sampled weak negatives, and
81,681 positive validation query pairs. Held-out conjunction support ranges from 22 to 1,416
validation plasmids (median 97.5).

**Frozen model and evaluation:** reuse the accepted E06 TF-IDF/SVD DNA and Qwen3 text features,
the E09 learning rate `3e-5`, uniform-v2-component base mass, 512-dimensional unnormalized heads,
300 updates, and seeds 13, 42, and 20260818. Compare direct conjunction text with the sum of its
two atomic natural parameters at K=10; report atomic retrieval, paired query-bootstrap intervals,
all training stability diagnostics, and all three W&B runs. No hyperparameter is selected from E10
retrieval outcomes. The g6.4xlarge L4 ceiling is 6 hours at $1.3232/hour, at most $7.9392.

## 2026-08-27 — E10 weak-label scaling supports additive retrieval

All three frozen fits were stable: initial losses `9.55–9.61` fell to `7.415–7.419`, maximum query
norms were `3.31–3.35`, and maximum sampled sequence norms were `2.89–3.05`. Atomic retrieval
utility@10 was `0.81563` (90.78% weak-label precision@10). Across 128 unseen conjunctions,
atomic-sum utility@10 was `0.41771` (70.89% precision) versus `0.18229` (59.11% precision) for
direct conjunction text. The paired difference was `0.23542`; the 2,000-draw query-bootstrap
interval was `[0.19010, 0.27760]`. This is strong exploratory evidence that addition scales beyond
four atoms under the weak closed-world annotation assumption, not evidence of biological absence
or final-holdout generalization.

Accepted S3 version `2026-08-27T18.46.12.234Z` contains 192 query rows, three checkpoints, 3,840
metric rows, and report SHA-256 `3b84fd0a20a377165a15f470e17365400756c5790526cef09169cb5d8b04f6b8`.
All W&B runs finished: `nncvbyr2`, `tp266gz5`, and `59xmy72h`. The measured stage took 345.59
seconds and `$0.12702`; the systemd wrapper ran about 394 seconds and `$0.14482`. No test row was
read. The L4 is idle and no E10 process remains.

## 2026-08-28 — E11 direct atomic-classifier ceiling preregistered

E11 asks whether E10 is limited by its shared Qwen-to-natural-parameter projection. It reuses the
exact E10 64 atoms, 128 conjunctions, 88,474 training rows, 10,852 validation rows, query hash
`b93a73db58b9a149a8458e5cd36bcd03f70997aee12e013c7d27908274275770`, and TF-IDF/SVD DNA
features. One deterministic zero-initialized linear classifier is fit per atom with class-balanced
binary cross-entropy over all 551,749 positive and 5,110,587 unreported weak-negative pairs. No
test row is read and no retrieval result selects a hyperparameter.

The frozen primary AND score is the sum of empirical-prior-corrected log probabilities. Raw-logit
sum and calibrated minimum are diagnostics. Primary utility@10 is paired by query against accepted
E10 atomic addition and bootstrapped over the 128 queries with 2,000 draws. Training uses 400
full-batch AdamW updates, learning rate `0.01`, weight decay `1e-4`, float32, and W&B group
`e11-atomic-classifier-v0.1`. The `g6.4xlarge` ceiling is one hour / `$1.3232` under reference
`chat-2026-08-28-e11-atomic-classifier-auto-under-20`.

## 2026-08-28 — E11 finds large known-atom classifier and calibration headroom

The deterministic fit was stable: balanced training loss fell from `0.69315` to `0.06386`, with
maximum weight norm `4.41`. Atomic utility@10 reached `0.93125`. The preregistered calibrated
log-probability sum reached conjunction utility@10 `0.82188`, interval `[0.75781, 0.88281]`, versus
E10 atomic addition `0.41771`. The paired improvement was `0.40417`, interval
`[0.32030, 0.49324]`. Calibrated minimum was a post-hoc diagnostic at `0.83125`; it is not selected.
Raw-logit sum was only `0.34219`, showing that per-atom prior calibration, not addition alone,
accounts for much of the gain. The direct heads are a known-vocabulary ceiling and do not yet
support unseen natural-language atoms.

Accepted S3 version `2026-08-28T10.19.38.951Z` passed independent checkpoint, metric, bootstrap,
and W&B read-back; report SHA-256 is
`8ea38fa831993052fa5f71b6dd66bb107d5092febc8172b25985dda81626e4c4`. Run `ktlo1bk4` finished on
an NVIDIA L4 in float32. Measured stage time/cost was 116.59 seconds / `$0.04285`; the systemd
wrapper was 156 seconds / `$0.05734`, with 1.8 GB peak memory and no swap. It ran from base commit
`dcd205d83db70ef6f1e156900105a6e42e9ea0bd` plus recorded dirty diff
`4ca30b167c06643c9a490cb6653aa289ef0f1bc840cabc18683bdb36bf1f6810`. No test row was read and
no E11 process remains.

## 2026-08-28 — Compositional-search measurement contract

Compositional search will report separate axes at each K. `strict_adherence` is the fraction of
retrieved plasmids satisfying every required atom. `mean_clause_adherence` gives partial credit for
individual required atoms and exposes one-sided retrieval. `useful_component_fraction` is the
number of distinct sequence-similarity components among strictly adherent hits divided by K, so a
top-K list of near-duplicates does not look fully useful. Partial-only and zero-clause fractions,
strict-component diversity, and first-strict rank are diagnostics. The historical signed utility
`2 * strict_adherence - 1` remains for continuity only.

No combined scalar is selected yet: adherence and non-redundant utility must both be shown. The
current contract covers positive AND queries. When explicit exclusions are added, forbidden-clause
violation will be a separate axis rather than being averaged into positive adherence.
