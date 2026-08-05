# vec2vec

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

The PlasmidCLIP data pipeline, rebuilt on Kedro with an S3 data catalog.

It turns the raw Addgene release into a paired
**(plasmid DNA sequence, natural-language description)** dataset with
leakage-aware splits, constraint-based relevance labels, and the audits needed
to trust both. Modelling is not part of this repository yet.

## Pipelines

| Pipeline | What it does | In `__default__` |
| --- | --- | --- |
| `processing` | Streams the raw Addgene JSON into canonical records; normalizes pLannotate and plasmidkit into one annotation table and per-plasmid feature lists. | yes |
| `dataset` | Joins descriptions to records, assigns leakage-aware splits, and attaches the constraints each description surfaces. | yes |
| `audit` | Builds the structured-query curriculum and measures how many provable hard negatives it yields. | yes |
| `constraint_semantics` | Profiles raw constraint values, grouped-split concentration, and the pLannotate-only annotation view for the E00 feasibility gate. | no |
| `facet_audit_sample` | Draws the frozen, component-aware E00 metadata-review sample. It does not accept labels. | no |
| `descriptions` | Generates descriptions through OpenRouter, merges the partitions, and quality-checks them. | **no — paid** |
| `import_descriptions` | Adopts the already-published descriptions instead of regenerating them. | no |

```bash
kedro run                          # processing -> dataset -> audit
kedro run --pipeline processing
kedro run --pipeline descriptions  # costs money; see below
kedro run --pipelines constraint_semantics
kedro run --pipelines facet_audit_sample
kedro viz                          # requires the `viz` extra
```

## Data flow

```text
addgene_plasmids.json ─┐
                       ├─> addgene_records ──────────────┐
plannotate.parquet ─┐  │                                  │
plasmidkit.parquet ─┴──┴─> addgene_annotations ──> addgene_annotation_features
                                                          │
                                    plasmid_descriptions ─┤
                                                          v
                                                  plasmid_pairs
                                                          v
                                                  retrieval_dataset ──> hard_negative_*
```

Storage lives entirely in the catalog. `conf/base/globals.yml` holds two roots —
`raw` (upstream drops, only ever read) and `lake` (this project's output) — and
every dataset is built from them, so moving to another bucket is a two-line
change. Outputs are laid out by Kedro data layer:

```text
s3://<bucket>/kedro/
├── 02_intermediate/   addgene_records, addgene_annotations, addgene_annotation_features
├── 03_primary/        description_partitions, plasmid_descriptions, plasmid_pairs
├── 04_feature/        retrieval_dataset  (versioned)
└── 08_reporting/      processing report, description QC, dataset audit, hard-negative audit
```

## Current state

Built from the Addgene release on 2026-08-04, with descriptions imported from
the published dataset rather than regenerated:

| Artifact | Rows | Size |
| --- | --- | --- |
| `addgene_records` | 115,120 plasmids | 63 MB |
| `addgene_annotations` | 5,936,251 annotations | 53 MB |
| `addgene_annotation_features` | 141,089 sequences | 6 MB |
| `plasmid_descriptions` | 158,331 imported, 115,120 matched | 15 MB |
| `retrieval_dataset` | 115,120 pairs | 93 MB |

- **Splits:** 92,097 / 11,515 / 11,508 over 14,157 leakage components, with
  **zero** components straddling a grouped split.
- **Constraints:** the median description surfaces 6 functional constraint
  groups; 216 of 115,120 rows surface none.
- **Descriptions:** 393 chars mean, ~3 sentences, 22 duplicate groups, and 6
  invented replication origins across the whole set (0.005%).
- **Supervision:** 276,168 structured queries over the train split; median 5
  provably-contradicting hard negatives per query, rising 1 → 6 → 12 with query
  order as alternative positives fall 24 → 12 → 6.

The 115,120 figure independently matches both the upstream project's cleaned
index and the `rows` field in its phase-2 cache manifest, which is the check
that the port reproduces the original.

## Setup

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

AWS credentials resolve through the normal boto3 chain — environment variables,
`~/.aws/credentials`, `AWS_PROFILE`, or an instance/SSO role. The catalog names
no credentials file, so there is nothing to configure and nothing secret to
commit by accident. Description generation additionally needs
`OPENROUTER_API_KEY` in the environment, and the description import needs
`HF_TOKEN`.

## Descriptions: import first, generate second

Roughly 158k descriptions already exist in the published
`UCL-CSSB/plasmidclip-addgene-158k` dataset. Adopting them costs nothing and
keeps their provenance columns intact:

```bash
HF_TOKEN=... kedro run --pipeline import_descriptions
```

Only run the paid generation pipeline for plasmids the import did not cover.

The published repository is private, so `HF_TOKEN` must be set. Note that a
Hugging Face private-storage overage on the *owning namespace* returns
`403 Forbidden` on every private repo in it, not just the one over quota —
worth checking first if a download suddenly stops working.

### Description generation costs money

`kedro run --pipeline descriptions` makes one LLM call per plasmid, so it is
deliberately excluded from the default pipeline. Set
`descriptions.cost_cap_usd` before running it, and use `descriptions.limit` to
price a pilot first:

```bash
kedro run --pipeline descriptions --params descriptions.limit=200,descriptions.cost_cap_usd=1
```

The step is interruptible. Each batch of `descriptions.batch_size` plasmids is
generated at the moment its partition is written, and a re-run reads back the
partitions that already exist and skips those plasmids — so a crash, a cost cap
or a Ctrl-C costs at most one batch. The prompt is frozen in
`vec2vec.lib.prompts`; every row records the prompt hash and the hash of the
metadata payload that produced it.

## Two Kedro details worth knowing before you edit config

**Transcoded dataset names are load-bearing.** `addgene_records@full` /
`@metadata` and `retrieval_dataset@full` / `@audit` are the same file read
through different `load_args.columns` projections. The `@` suffix is not
cosmetic: Kedro resolves the dependency graph on the part *before* it, which is
what guarantees the node writing `addgene_records` runs before every node
reading a narrowed view of it. Renaming them to two independent names silently
removes that ordering guarantee.

The one place two independent names *are* correct is
`description_partitions` / `description_partitions_completed`, because
generation both writes and reads that prefix — a single transcoded name would
make one node its own upstream, which Kedro rejects as a cycle. They share a
YAML anchor so their paths cannot drift apart.

**Kedro replaces a top-level parameter key across environments, it does not
deep-merge it.** Overriding `addgene.chunk_size` in `conf/test/parameters.yml`
deletes `addgene.include_partial` and `addgene.limit`. Each environment must
restate the whole block. This silently broke the test environment once; nodes
now index `params[...]` directly rather than defaulting with `.get()`, so a
dropped key fails loudly instead of quietly running a different config.

## Leakage-aware splits

73% of Addgene plasmids share a backbone with ten or more near-siblings, and
some rows share a byte-identical sequence. `split_grouped` unions rows into
components by *both* family key and exact-sequence hash and assigns whole
components together, so no family and no duplicate sequence can straddle a
split. `split_random` is kept as an explicit baseline: the gap between the two
is how much of a retrieval score is scaffold memorization.

## Relevance is many-to-many

A generated description states requirements that several plasmids can
legitimately satisfy, so scoring only the paired sequence as correct mislabels
genuine matches as errors. `vec2vec.lib.relevance` extracts only the metadata
values that appear *literally* in a description and stores them on every row as
`surfaced_constraints_json` and `structured_constraints_json`.

The rule that makes this usable is three-way, not two-way: a candidate whose
metadata records a different value **contradicts** a constraint, but one that
records *nothing* for that field is **unknown** and must never be counted as a
negative. `partition_candidates_by_field` returns those three buckets.

The `audit` pipeline then checks that this supervision is actually usable:
it builds the nested query curriculum for every eligible row and counts the
same-backbone peers whose recorded metadata *provably* contradicts each query.

## Layout

```text
src/vec2vec/
├── lib/          pure logic: no Kedro, no I/O, directly unit-tested
├── datasets/     custom Kedro datasets for sources too large to materialize
└── pipelines/    node wiring; nodes compose lib functions and nothing else
conf/
├── base/         catalog, parameters, globals
└── test/         the same catalog pointed at a temporary local directory
tests/
├── lib/          unit tests for the logic
└── pipelines/    real Kedro sessions over a small fixture release
studies/           questions, experiment specifications, interpretation notebooks, and reports
```

The first active study is
[`set_valued_compositional_embeddings`](studies/set_valued_compositional_embeddings/). Its
benchmark-feasibility gate must pass before model training begins.

```bash
pytest              # all tests are offline
ruff check . && ruff format --check .
```

`conf/test` runs the real pipelines against a local fixture, so composition
breakage fails in CI rather than on S3. To use it by hand:

```bash
VEC2VEC_TEST_ROOT=/tmp/vec2vec kedro run --env test
```

## Differences from the original pipeline

This is a port of the `plasmid-clip` data pipeline, not a transcription. The
substantive changes:

- **Annotations are their own table.** pLannotate and plasmidkit rows were
  embedded verbatim inside each record's metadata blob, which made the record
  file ~5 GB and forced consumers to branch on per-source coordinate names.
  They now share one schema in `addgene_annotations`, with per-plasmid feature
  lists in `addgene_annotation_features`. Interval values are preserved exactly
  as published — the `source` column says which convention applies.
- **Splits are computed once.** The original assigned grouped splits by family,
  then ran a separate union-find repair pass to fix exact-sequence duplicates
  that had straddled a split. Doing the union first makes the repair
  unnecessary and the invariant true by construction.
- **The clean-up pass is gone.** It existed to remove fragment sequences that
  had been ingested as whole plasmids and to re-verify them against the raw
  JSON. Fragments are now excluded at ingestion, so there is nothing to undo.
- **`split_random` means what it says.** The original overwrote it with a split
  grouped by exact sequence; it is now an ordinary row-level random split.
- **Storage is the catalog.** The bespoke `storage.py` / `layout.py` /
  `settings.py` backend abstraction and the hand-rolled checksum manifests are
  replaced by catalog entries, fsspec, and Kedro's dataset versioning.
- **Parquet throughout.** Records and descriptions were JSONL written through
  hand-managed fsspec handles, several of which silently only worked on local
  paths despite taking "URI" arguments.

Left for the modelling phase, deliberately: in-batch relevance masks,
embedding-based retrieval metrics, and Hugging Face publication. PLSDB
ingestion and the constraint-judging API were written and then removed once it
was clear nothing consumed them — both are in this branch's history if the
encoder bake-off or a relevance evaluator brings them back.
