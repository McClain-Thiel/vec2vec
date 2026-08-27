# vec2vec

Kedro pipeline for building the paired Addgene dataset and the two feature tables used by the
selected vec2vec model: 6-mer TF-IDF/SVD DNA and Qwen3-Embedding-0.6B text.

## Current finding

On the frozen validation benchmark, 6-mer TF-IDF/SVD DNA plus Qwen3-Embedding-0.6B text ranked
first among 12 encoder pairs: utility@10 `0.15309`, whole-component interval
`[0.07623, 0.18828]`. Neural DNA encoders did not beat the TF-IDF/SVD baseline.

With that encoder pair fixed, verified-set supervision improved pair-query utility@10 from
`0.30875` to `0.48792`. The paired difference was `0.17917`, with interval
`[0.10375, 0.25294]`. This is validation evidence, not a confirmatory test. Pair queries were seen
during training, so it does not establish unseen composition.

E05 withheld every conjunction from training while retaining its 28 constituent atomic queries.
On 80 unseen conjunctions, verified-set supervision improved utility@10 from `-0.12250` to
`0.17417`; the difference was `0.29667`, with interval `[0.22499, 0.34292]`. This is validation-set
evidence for compositional generalization under the frozen linear-probe protocol, not final test
performance.

E06 expanded training from 20,000 to all 88,474 eligible rows and refit the selected features.
Utility@10 improved from `-0.10167` to `0.17333`; the difference was `0.27500`, with interval
`[0.18707, 0.33878]`. The E05 effect therefore survived the population-scale robustness check.
The historical test split remains contaminated, so this is still exploratory validation evidence.

The deployable final fit uses all 110,267 eligible annotated plasmids and is stored in the HF
bucket with W&B run `m4eeei4w`. It performs no evaluation and adds no scientific claim.

The complete retained evidence is four tables:

- [`results/encoders.csv`](results/encoders.csv): all 12 encoder pairs.
- [`results/supervision.csv`](results/supervision.csv): paired versus set supervision.
- [`results/composition.csv`](results/composition.csv): E05 and population-scale E06 unseen pairs.
- [`results/artifacts.csv`](results/artifacts.csv): exact versions, hashes, locations, and failures.

Regenerate and verify them from the accepted S3 reports:

```bash
python scripts/summarize_results.py
python scripts/summarize_results.py --check
```

Recompute either finding from the frozen feature artifacts with the same script. The command
requires the approval reference, host, time limit, and price because it trains on a GPU:

```bash
python scripts/summarize_results.py --reproduce alignment \
  --approval-reference <approval> --region <region> --instance-type <type> \
  --instance-hour-limit <hours> --observed-instance-price-usd-per-hour <price>
```

Use `--reproduce supervision`, `composition`, or `scale` for the three supervision comparisons.
All stages require the exact Python 3.11.14 environment and NVIDIA L4 recorded in
`result_reproduction`; install it with `uv sync --extra modeling`.

## Data pipeline

| Pipeline | Output |
| --- | --- |
| `processing` | Canonical Addgene records and annotations |
| `import_descriptions` | Published PlasmidCLIP descriptions |
| `descriptions` | Newly generated descriptions; paid OpenRouter calls |
| `dataset` | Paired retrieval dataset and grouped split |
| `audit` | Structured-query and hard-negative summary |
| `modeling_data` | Complete tagged DAG from raw data through selected feature tables |

```bash
uv venv --python 3.11.14
uv sync --extra dev

kedro run --pipeline processing
HF_TOKEN=... kedro run --pipeline import_descriptions
kedro run --pipeline dataset
kedro run --pipeline audit
```

The costly stages stay outside `__default__`. Run one stage of the same DAG with tags, for example
`kedro run --pipeline modeling_data --tags tfidf`. The `similarity-graph` tag requires minimap2 and
the `similarity-graph` extra. The `qwen` tag requires the `modeling` extra, a GPU, and an explicit
approved `modeling_features.compute_authorization` runtime block.

Storage roots are in `conf/base/globals.yml`; all other locations are in the Kedro catalog.
`OPENROUTER_API_KEY` is read from the environment or the ignored root `.env` file. Description
generation is excluded from the default pipeline because it makes paid API calls.

## Scope

The repository keeps data construction, the append-only
[`EXPERIMENT_LOG.md`](studies/set_valued_compositional_embeddings/EXPERIMENT_LOG.md), and the
accepted result summary. Superseded experiment code and detailed reports remain available through
commits
[`434e760`](https://github.com/McClain-Thiel/vec2vec/commit/434e760e2885287b73351d0df0e7ddb69b36be00)
and [`1b4cd8e`](https://github.com/McClain-Thiel/vec2vec/commit/1b4cd8e743c5c223afcd2ab5256ee6f6123d846d).
