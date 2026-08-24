# Constraint Rule Validation v0.1

## Status

Accepted for noisy training supervision. This decision does not create biological ground truth or
claim-by-claim benchmark labels.

## Accepted contract

- Evidence version: `e00-constraint-evidence-v0.1`.
- Rule contract SHA-256: `aab672e2a0d64cd1b6c90daf90c6429367bc3861612781b6de4cfc45f47dbfa2`.
- Copy class: configured exact `included` mappings.
- Growth temperature: exact 30 and 37 degree mappings only.
- Bacterial selection: configured exact `included` mappings and the five reviewed exact mappings.
- Intended use: exact controlled expression and use-category mappings.

The stored growth value `23`, unknown copy values, excluded bacterial syntax, and free-text intended
use remain unlabeled.

## Evidence

1. The 30-application hand check passed all selected applications. Selection identity SHA-256 is
   `ba823d9de942f1ebd2c73592b1830eb8e063e2f062d1149dbff1c3a3db99cbb8`.
2. The fixed 240-application strong-model benchmark returned 240 valid responses, 240 `supported`
   semantic decisions, and 240 `in_scope` decisions.
3. The model-reference pass fraction is 1.0. Its 95% Wilson interval on this fixed sample is
   0.984246 to 1.0.
4. The earlier targeted 16-packet gate covered the rare reviewed bacterial mappings that do not
   occur in the representative 240-application sample.
5. The complete run used `openai/gpt-5.6-sol` from provider `OpenAI`, cost USD 4.670470, and produced
   output version `2026-08-06T09.01.02.445Z`.

## Limits

- The accuracy reference is a strong model, not an independent laboratory measurement or a large
  human gold set.
- The model sees the proposed mapping, so the benchmark detects semantic inconsistency and scope
  errors more directly than upstream Addgene metadata errors.
- The validation sample is stratified by facet and is not a simple population sample.
- The rules support limited claims such as `reported_as`, `tagged_for`, and
  `reported_selection_includes`. They do not establish biological performance.

## Decision

Use the accepted contract to produce training supervision. Do not review each generated claim. If a
later error clusters by exact mapping, create a new rule version, regenerate the evidence, and keep
this result as research history.
