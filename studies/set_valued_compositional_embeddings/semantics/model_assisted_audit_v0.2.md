# Model-Assisted Facet Audit Protocol v0.2

## Status

Validator v0.1 failed before generation because the pinned Anthropic transport rejected JSON Schema
keywords. Validator v0.2 used a portable transport schema but stalled on Amazon Bedrock and was
stopped before a result artifact was written. Validator v0.3 found that Azure structured outputs are
disabled for this workspace. Validator v0.4 generated a response but failed the unchanged local
Pydantic contract. Validator v0.5 passed a one-row GPT-5.6 Sol diagnostic. Validator v0.6 keeps the
same request and raises only the run cost cap. This protocol changes who performs the primary
knowledge check. It does not rewrite the frozen sample or the version 1 facet rules.

The complete Sol v0.6 gate later produced 30 valid responses. The gate stopped further scaling
because intended-use judgments mixed semantic meaning with version 1 scope and growth-temperature
judgments remained rule-inconsistent. See `../reports/04_gpt56_sol_validator.md`.

## Purpose

Use a stronger, independent language model for the primary synthetic-biology knowledge check. Use
Luna as a separate first opinion. Send disagreement, uncertainty, invalid output, and source
conflict to a named human for resolution.

Human status alone is not evidence of greater subject knowledge. Model status alone is also not
evidence of correctness. The source record and the narrow facet definition remain the evidence.

## Fixed 30-row validation gate

- Packet version: `2026-08-05T09.44.22.296Z`.
- Luna decision version: `2026-08-05T09.54.34.956Z`.
- Validator model: `openai/gpt-5.6-sol`.
- Provider: `OpenAI`, pinned with fallbacks disabled.
- Reasoning: high effort.
- Temperature: not sent.
- Seed: 17.
- Maximum output tokens: 3,000.
- Provider-side structured output: required with the full JSON Schema.
- Response contract: the same complete schema is also applied locally with Pydantic.
- Reported-cost cap: USD 0.75.
- Maximum requests: 30.
- Transport retries: zero. A failed request becomes a preserved result row.

The validator receives the same packet content and Pydantic response contract as Luna. It does not
receive Luna's verdict, reason, suggestions, or generated plasmid description. It cannot browse the
Addgene URL.

Local Pydantic validation enforces required fields, extra-field rejection, verdict values, list
bounds, exact packet identities, reason length, snake case, and conditional suggestions. Invalid
text is retained and is not repaired or converted into a decision.

## Decision roles

- **Luna first opinion:** a preserved proposal from the completed pilot.
- **Sol validator:** the primary stronger-model knowledge-check proposal.
- **Manual adjudicator:** resolves model disagreement, `uncertain`, invalid output, or a source
  conflict. The adjudicator can use source records and external references and must record a reason.
- **Accepted evidence:** a later E00 product. Neither model writes it directly.

Agreement between the two models is supporting evidence about repeatability across model families.
It is not an accuracy estimate. A shared error remains possible.

## Thirty-row gate

Run the validator once on the exact frozen packets. Preserve every response and cost. Before any
918-row run, inspect:

1. structural validity and packet identity;
2. exact verdict agreement with Luna;
3. DAP, Trimethoprim, temperature, and intended-use failure cases;
4. consistency across rows with the same classified value;
5. suggestions that exceed the classified-value scope.

Do not scale the paid validator to all 918 rows if it repeats an obvious domain error, has invalid
responses, or produces unexplained rule inconsistency. This stopping condition was met. Write a new
protocol version for any prompt, schema, evidence, provider, or model change.

## Review export

Create one model-free CSV from the fixed 918-row sample. It contains exact source evidence and blank
human-decision fields. It excludes generated descriptions and all model conclusions. Use it only
for manual conflict resolution or a separate blinded audit.

Human decisions use the serialized `HumanAuditDecision` schema. A non-supported decision requires
a short reason and a timezone-aware timestamp. A changed decision retains the previous decision
identity.

## Interpretation limit

This procedure validates mappings against available Addgene metadata. It does not prove biological
function, experimental success, or completeness. Model consensus can support a proposed mapping,
but it cannot turn absent source evidence into negative evidence.
