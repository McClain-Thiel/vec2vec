# Encoder Prior and Candidate Review

## Conclusion first

- **Use Carbon-500M as the incumbent, not as an assumed winner.** It beat the other PlasmidCLIP
  encoders on the actual frozen retrieval task even though Mistral-DNA led the earlier geometry
  benchmark.
- **Test three stronger DNA alternatives in Gate 1:** Carbon-3B, GENERanno prokaryote 500M, and
  GENERator-v2 prokaryote 1.2B.
- **Keep BGE-base as the text incumbent.** Cross it, Qwen3-Embedding-0.6B, and
  GTE-ModernBERT-base with every DNA representation in the bounded panel.
- **Defer Evo 2 7B.** It is biologically relevant but costs much more to load and evaluate. Run it
  only if the bounded panel fails to beat Carbon-500M and a larger compute budget is approved.
- **Do not select from published leaderboards.** The validation-only vec2vec retrieval bake-off
  chooses the encoder. External results define candidates and risks only.

## Scope and evidence classes

This review was frozen on 2026-08-18. It uses:

- **Observed project evidence:** files and tracked results in PlasmidCLIP at commit
  [`77c4590`](https://github.com/McClain-Thiel/plasmid-clip/tree/77c4590730edb8807efbdb552d598866ef55460a).
- **Observed provider evidence:** official repositories, papers, and model cards.
- **Derived decisions:** the Gate 1 candidate panel and exclusion reasons below.
- **Unknown:** which encoder performs best on the vec2vec query benchmark.

Provider benchmark claims are not independent evidence for plasmid retrieval. Model pretraining
corpora can also contain Addgene or related plasmid sequences. The available model cards do not
support a complete contamination audit.

## What PlasmidCLIP already measured

PlasmidCLIP ran two distinct encoder comparisons. The first used 100 curated Addgene plasmids and
measured circular-rotation similarity, reverse-complement similarity, annotation enrichment, and
confounds. Mistral-DNA-bacteria led that geometry screen. Carbon-500M ranked second. Evo 1 did not
run because its `flash_attn` requirement did not match the host software stack.

The second comparison held the paired retrieval recipe and BGE-base text tower fixed. It measured
leakage-safe sequence-to-description retrieval. That comparison reversed the geometry ranking:

| Frozen DNA tower | R@1 | R@10 | Median rank |
| --- | ---: | ---: | ---: |
| PlasmidGPT-GRPO | 3.2% | 16.1% | 109 |
| Mistral-DNA-bacteria | 6.0% | 24.5% | 57 |
| Carbon-500M | **8.1%** | **28.8%** | **41** |

Train-only whitening with no component removal and a 512-dimensional projection improved the
Carbon result to R@1 8.8%, R@10 30.5%, and median rank 37. Carbon's raw representation had an
effective rank of 475/1,024. Mistral-DNA had an effective rank of about 6/768. This collapse
explains why strong invariance did not produce strong retrieval.

The BGE-base text tower had an effective rank of 472/768. Text whitening improved R@10 from 30.5%
to 32.0%, but median rank remained 37. The text representation was not the main bottleneck in that
experiment.

The corrected full PlasmidCLIP model later reached test R@10 50.24% and median rank 10.3 in a
2,999-candidate pool-matched evaluation. It reached full-pool R@10 28.56% and median rank 52. These
paired-description metrics are not comparable to the vec2vec set-retrieval benchmark. See the
[append-only PlasmidCLIP experiment log](https://github.com/McClain-Thiel/plasmid-clip/blob/77c4590730edb8807efbdb552d598866ef55460a/experiments/LOG.md)
and the [original encoder W&B run](https://wandb.ai/mcclain/plasmidclip-encoder-bakeoff/runs/gpb4yxq8).

PlasmidCLIP also found that Carbon inference in IEEE float16 produced invalid values. Bfloat16 and
float32 worked. Gate 1 therefore rejects float16.

## DNA panel for Gate 1

| Role | Model and frozen revision | Relevant observed facts | Decision |
| --- | --- | --- | --- |
| Incumbent | `HuggingFaceBio/Carbon-500M@106e36ff51b5dfbfe0b078ad18ad37a6956c5714` | 500M; 8,192 6-mer tokens; about 49 kbp; prior plasmid retrieval winner | Run |
| Scale test | `HuggingFaceBio/Carbon-3B@95c3c68fc77fdf70b1582031bacf9d7753f72cf2` | 3B; 32,768 6-mer tokens; about 197 kbp; same family and tokenizer | Run |
| Prokaryote test | `GenerTeam/GENERanno-prokaryote-0.5b-base@d02db0f24f2c62fa1efde760217cdf75771b0228` | 500M; single-base tokens; 8,192 bp; trained on 715 billion bp of prokaryotic DNA | Run |
| Long prokaryote test | `GenerTeam/GENERator-v2-prokaryote-1.2b-base@8b2f768b0d293953518ff91d34600f9322ef1f94` | 1.2B; 16,384 6-mer tokens; 98,304 bp; prokaryote-specific release | Run |
| High-cost reserve | `arcinstitute/evo2_7b@bda0089f92582d5baabf0f22d9fc85f3588f6b58` | 7B; single-base; long-context checkpoint; microbial, phage, and plasmid training data | Defer |

The comparison also includes a train-fitted 6-mer TF-IDF plus truncated-SVD baseline. This
network-free baseline tests whether a neural model adds useful retrieval information.

The Carbon repository describes Carbon-3B as the flagship and Carbon-500M as a draft model for
speculative decoding. Carbon pretraining is eukaryote-focused, with a minority prokaryote share.
This makes Carbon-3B a useful scale control but not a guaranteed plasmid winner. See the
[Carbon repository](https://github.com/huggingface/carbon),
[Carbon-500M card](https://huggingface.co/HuggingFaceBio/Carbon-500M), and
[Carbon-3B card](https://huggingface.co/HuggingFaceBio/Carbon-3B).

GENERanno provides the strongest bounded prokaryote-specific contrast. Its 8,192-bp nominal
context means that at least 33.7526% of this study's plasmids require multiple windows. Model tags
can increase that fraction. See the
[GENERanno model card](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-base) and
[official repository](https://github.com/GenerTeam/GENERanno).

GENERator-v2 has a nominal 98,304-bp context, which is 618 bp shorter than the observed maximum.
Its 6-mer tokenizer requires input lengths that are multiples of six. Gate 1 must measure the
tag-safe context and preserve all bases with circular windows. It cannot apply the model card's
optional left truncation. See the
[prokaryote model card](https://huggingface.co/GenerTeam/GENERator-v2-prokaryote-1.2b-base) and
[official repository](https://github.com/GenerTeam/GENERator).

Evo 2 is a relevant reserve because its OpenGenome2 training data spans bacteria, archaea, phage,
and other domains, and its long checkpoint supports contexts up to 1 million bases. Its 7B size
and specialized runtime make it a separate compute decision. See the
[Evo 2 Nature paper](https://www.nature.com/articles/s41586-026-10176-5) and
[official checkpoint](https://huggingface.co/arcinstitute/evo2_7b).

## Text panel for Gate 1

| Role | Model and frozen revision | Context and output | Decision |
| --- | --- | --- | --- |
| Incumbent | `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | 768-dimensional; prior text tower | Run |
| Modern compact test | `Alibaba-NLP/gte-modernbert-base@e7f32e3c00f91d699e8c43b53106206bcc72bb22` | 8,192 tokens; Apache-2.0 | Run |
| Modern instruction test | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 32,768 tokens; 1,024-dimensional; Apache-2.0 | Run |

BGE-base remains the incumbent because PlasmidCLIP measured a healthy representation on the same
description source. External text benchmarks do not establish performance on terse plasmid
constraints. Qwen3 and GTE provide two current, permissively licensed contrasts with different
model sizes and embedding recipes. See the
[BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5),
[GTE-ModernBERT model card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base), and
[Qwen3-Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## Models not selected for the bounded panel

| Model | Reason |
| --- | --- |
| PlasmidGPT variants | PlasmidCLIP already found lower target-retrieval performance than Carbon-500M. |
| Mistral-DNA-bacteria | It led invariance but collapsed in target retrieval. It remains a historical control, not a new candidate. |
| PlasmidCLIP fine-tuned checkpoint | Its old training split can overlap `split_grouped_v2` validation and test rows. Treat it as contaminated until a row-level audit proves otherwise. |
| `InstaDeepAI/NTv3_100M_post_131kb@8dd4c9d1a5187804c7edec785b118228a2eb1e37` | Current weights use a non-commercial license. The long-context post-training targets animal and plant functional tracks, not plasmids. |
| `CladeTeam/CENO-300M-131k@9788ebafdfe201a01d532301e9c93c108dad5a52` | The July 2026 release is too new for the primary gate. Keep this Apache-2.0 checkpoint on the watchlist. |
| `macwiatrak/bacformer-large-masked-complete-genomes@84f85c37a05e55559142b842c1569e265183b554` | It consumes ordered protein embeddings and gene calls, not raw plasmid DNA. It belongs in a later annotation-aware comparison. |
| HyenaDNA and Caduceus | Their main pretraining and reported tasks are human-genome focused. They do not displace the prokaryote-specific candidates. |
| Carbon-8B and GENERator-v2 3B | They add cost before the smaller same-family candidates establish value. |

The NTv3 license and task scope come from the
[official NTv3 model card](https://huggingface.co/InstaDeepAI/NTv3_100M_post_131kb).
CENO's release state comes from the [official CENO repository](https://github.com/CladeTeam/CENO).
Bacformer's input contract comes from the
[official Bacformer repository](https://github.com/macwiatrak/Bacformer).

## Decision rule

No model in this report is selected as the final encoder. The
[E02 fixed-representation bake-off](../experiments/E02_fixed_representation_bakeoff.md) makes that
decision with a full DNA-by-text factorial on `split_grouped_v2` training data and validation
queries only. The test split remains unread until the later confirmatory experiment.
