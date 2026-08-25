# E03/E04: paired identity versus verified-set supervision

**Question.** Does verified multi-positive supervision improve retrieval of verified answer sets
over paired-identity supervision? Both objectives use the accepted TF-IDF/SVD DNA and Qwen3 text
features, the same 512-dimensional linear heads, initial weights, sampled batches, optimizer, and
three seeds. Each update contains all 108 controlled queries and one uniformly sampled verified
training sequence per query. E03 treats only the sampled diagonal pairs as positive. E04 treats
every known verified query-sequence relation in that same batch as positive. Other in-batch pairs,
including unknown relations, act as contrastive negatives; this is a shared limitation. Training
uses the frozen 20,000-row panel. Evaluation uses the frozen validation gallery. Test rows and test
metrics remain unread. Pair queries are seen during training, so this is not a composition test.

**Protocol and decision.** Run 300 final-update training steps for seeds 13, 42, and 20260818 with
AdamW, learning rate 0.001, weight decay 0.01, temperature 0.07, and no checkpoint selection. The
primary estimate is the difference in pair-query macro utility@10, averaged over seeds, with a
paired 2,000-draw whole-component bootstrap interval. The result supports set supervision only if
E04 improves E03 by at least 0.01 and the interval lower bound is above zero. Atomic and combined
verified, contradicted, unknown, and utility fractions are secondary. Each seed-objective unit logs
to W&B and all checkpoints and evaluation tables persist through Kedro. The paid run must have an
explicit wall-clock and cost cap in configuration before launch. The approved cap is 0.5 hours on
the existing on-demand `g6.4xlarge` in `us-east-1` at $1.3232 per hour: $0.6616 maximum.
