# Constraint Accuracy Benchmark v0.1

## Conclusion

The fixed rule-derived validation sample passes its planned hand and strong-model checks. All 30
hand-checked applications passed. All 240 model-judged applications were valid, semantically
supported, and in scope. This is sufficient to use the exact-rule claims as noisy training
supervision without reviewing every claim.

## Fixed inputs

| Item | Identity |
| --- | --- |
| Validation sample | `2026-08-06T08.44.42.865Z` |
| Judge packets | `2026-08-06T08.58.54.091Z` |
| Prompt | `constraint-benchmark-judge-v1` |
| Prompt hash | `491cd43a849cb74d624f0c00c4ab1b6b740d6d3107f1f43c7108f728140019c4` |
| Judge implementation | Git commit `400a47d` |
| Model | `openai/gpt-5.6-sol` |
| Provider | `OpenAI` |
| Complete decisions | `2026-08-06T09.01.02.445Z` |

The paid run used a clean worktree, high reasoning, strict structured output, zero retries, no
temperature parameter, and a USD 7.50 cost cap. It did not use generated descriptions, browsing, or
plasmidkit fallback.

## Results

| Facet | Rows | Passes | Pass fraction | 95% Wilson interval |
| --- | ---: | ---: | ---: | --- |
| Copy class | 47 | 47 | 1.000 | 0.9244–1.0000 |
| Expression context | 54 | 54 | 1.000 | 0.9336–1.0000 |
| Growth temperature | 61 | 61 | 1.000 | 0.9408–1.0000 |
| Use category | 30 | 30 | 1.000 | 0.8865–1.0000 |
| Bacterial selection | 48 | 48 | 1.000 | 0.9259–1.0000 |
| **Overall** | **240** | **240** | **1.000** | **0.9842–1.0000** |

A pass means the response was schema-valid, semantic support was `supported`, and benchmark scope
was `in_scope`. There were no invalid, uncertain, out-of-scope, unsupported, or manual-review rows.
All upstream responses identify model `openai/gpt-5.6-sol` and provider `OpenAI`.

The complete run took 1,073.7 seconds and cost USD 4.670470. The earlier five-application smoke cost
USD 0.117088. Total paid cost for this benchmark stage was USD 4.787558.

## Interpretation

The result supports mapping consistency, not biological function. The primary evidence is the exact
Addgene metadata value, and the relations preserve this limitation. pLannotate features are
supplementary. An absent or differently named pLannotate feature does not negate an exact metadata
field.

The model sometimes wrote `field: value` in `evidence_used` instead of only a field name. Every base
field still resolves to supplied evidence, so this is a formatting variation rather than unsupported
evidence.

## Limits

- GPT-5.6 Sol is the accuracy reference. It is not an independent human gold set.
- The judge sees the proposed mapping. This is useful for semantic and scope validation but less
  sensitive to upstream Addgene metadata errors.
- The sample is stratified by facet. The interval describes this fixed sample and model reference.
- Rare reviewed bacterial mappings were covered by the earlier targeted gate rather than this
  representative sample.

## Decision

Accept rule contract
`aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2` for noisy training
supervision. Keep unknown and disabled values unlabeled. Proceed to frozen query and candidate-gallery
construction; do not add claim-by-claim review.
