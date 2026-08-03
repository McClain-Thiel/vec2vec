"""Generate and quality-check natural-language plasmid descriptions.

This is the only paid step in the project: one LLM call per plasmid. It is
therefore written to be interruptible. Work is split into batches, each batch is
generated lazily at the moment its partition is written, and a re-run reads back
the partitions that already exist and skips those plasmids. A crash, a cost cap
or a Ctrl-C costs you at most one batch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from vec2vec.lib import openrouter, prompts, qc
from vec2vec.lib.text import as_list

logger = logging.getLogger(__name__)

DESCRIPTION_COLUMNS = (
    "sequence_id",
    "description",
    "generation_model",
    "prompt_version",
    "prompt_hash",
    "input_hash",
    "cost_usd",
)


@dataclass
class _Budget:
    """A shared spend counter across lazily generated batches."""

    cap_usd: float | None
    spent_usd: float = 0.0
    exhausted: bool = field(default=False)

    def charge(self, amount: float) -> None:
        self.spent_usd += amount
        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:
            self.exhausted = True


def _prompt_rows(metadata: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach annotation feature lists to record metadata for prompting."""
    merged = metadata.merge(features, on="sequence_id", how="left")
    merged["annotation_features"] = merged["annotation_features"].map(as_list)
    return merged


def _completed_ids(partitions: dict[str, Callable[[], pd.DataFrame]]) -> set[str]:
    """Sequence ids already present in previously written partitions."""
    done: set[str] = set()
    for load in partitions.values():
        done.update(load()["sequence_id"].astype(str))
    return done


def _describe_batch(
    rows: list[dict[str, Any]],
    *,
    budget: _Budget,
    settings: dict[str, Any],
    api_key: str,
) -> pd.DataFrame:
    """Generate one batch of descriptions concurrently and return its partition."""
    model = str(settings["model"])
    concurrency = int(settings.get("concurrency", 6))
    generated: list[dict[str, Any]] = []
    failures = 0

    def describe(
        client: httpx.Client, row: dict[str, Any]
    ) -> tuple[dict[str, Any], openrouter.Completion | None, str | None]:
        try:
            completion = openrouter.complete(
                client,
                prompts.build_messages(row),
                model=model,
                api_key=api_key,
                max_tokens=int(settings.get("max_tokens", 256)),
                temperature=float(settings.get("temperature", 0.2)),
                provider=settings.get("provider"),
            )
        except Exception as error:  # noqa: BLE001 - one plasmid must not sink the batch
            return row, None, f"{type(error).__name__}: {error}"
        return row, completion, None

    with httpx.Client() as client, ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = pool.map(lambda row: describe(client, row), rows)
        for row, completion, error in results:
            budget.charge(completion.cost_usd if completion else 0.0)
            if completion is None or not completion.text:
                failures += 1
                logger.warning("Description failed for %s: %s", row["sequence_id"], error)
                continue
            generated.append(
                {
                    "sequence_id": row["sequence_id"],
                    "description": completion.text,
                    "generation_model": model,
                    "prompt_version": prompts.PROMPT_VERSION,
                    "prompt_hash": prompts.prompt_hash(),
                    "input_hash": prompts.input_hash(row),
                    "cost_usd": round(completion.cost_usd, 6),
                }
            )

    logger.info(
        "Batch complete: %s written, %s failed, $%.3f spent in total",
        len(generated),
        failures,
        budget.spent_usd,
    )
    return pd.DataFrame(generated, columns=list(DESCRIPTION_COLUMNS))


def generate_descriptions(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    completed: dict[str, Callable[[], pd.DataFrame]],
    params: dict[str, Any],
    credentials: dict[str, Any],
) -> dict[str, Callable[[], pd.DataFrame]]:
    """Return one lazy partition per batch of plasmids still needing a description.

    Each returned callable performs its own LLM calls when the catalog saves it,
    so partitions land on storage as the run progresses. Once the configured
    ``cost_cap_usd`` is reached, remaining batches return empty and the run stops
    spending.
    """
    api_key = credentials.get("api_key")
    if not api_key:
        raise ValueError("OpenRouter credentials are missing an 'api_key' entry")

    rows = _prompt_rows(metadata, features)
    done = _completed_ids(completed)
    pending = rows.loc[~rows["sequence_id"].isin(done)]
    limit = params.get("limit")
    if limit is not None:
        pending = pending.head(int(limit))

    batch_size = int(params.get("batch_size", 2_000))
    budget = _Budget(cap_usd=params.get("cost_cap_usd"))
    records = pending.to_dict("records")
    logger.info(
        "Descriptions: %s already generated, %s pending, %s batches of %s",
        f"{len(done):,}",
        f"{len(records):,}",
        -(-len(records) // batch_size),
        batch_size,
    )

    def make_partition(batch: list[dict[str, Any]]) -> Callable[[], pd.DataFrame]:
        def build() -> pd.DataFrame:
            if budget.exhausted:
                logger.warning("Cost cap of $%.2f reached; skipping batch", budget.cap_usd)
                return pd.DataFrame(columns=list(DESCRIPTION_COLUMNS))
            return _describe_batch(batch, budget=budget, settings=params, api_key=api_key)

        return build

    # Offset partition names past the existing ones so a resume never overwrites.
    offset = len(completed)
    return {
        f"part-{offset + index:05d}": make_partition(records[start : start + batch_size])
        for index, start in enumerate(range(0, len(records), batch_size))
    }


def merge_descriptions(partitions: dict[str, Callable[[], pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate description partitions into one table, keeping the first of any duplicate."""
    if not partitions:
        raise ValueError("no description partitions found; run the generation node first")
    # A batch skipped by the cost cap, or one where every call failed, leaves an
    # empty partition behind; those carry no dtypes worth concatenating.
    frames = [frame for _, load in sorted(partitions.items()) if not (frame := load()).empty]
    if not frames:
        raise ValueError("every description partition is empty; no descriptions were generated")
    merged = pd.concat(frames, ignore_index=True)
    deduplicated = merged.drop_duplicates(subset="sequence_id", keep="first").reset_index(drop=True)
    if len(deduplicated) < len(merged):
        logger.warning("Dropped %s duplicate descriptions", len(merged) - len(deduplicated))
    logger.info("Merged %s descriptions", f"{len(deduplicated):,}")
    return deduplicated


def check_descriptions(
    descriptions: pd.DataFrame,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Profile the generated descriptions and flag unsupported specificity.

    Descriptions with no current source record are reported and skipped rather
    than treated as an error: a published description set is generally built
    from an older snapshot and legitimately covers plasmids this run excludes.
    Pairing is where an unmatched description actually matters, and
    :func:`vec2vec.pipelines.dataset.nodes.assemble_pairs` inner-joins there.

    Returns:
        The QC report, and the full table of flagged records for review.
    """
    rows = _prompt_rows(metadata, features)
    joined = descriptions.merge(rows, on="sequence_id", how="inner", suffixes=("", "_record"))
    orphaned = len(descriptions) - len(joined)
    if orphaned:
        logger.warning("Skipping %s descriptions with no matching source record", f"{orphaned:,}")
    if joined.empty:
        raise ValueError("no description matches a source record; check the inputs")

    texts = joined["description"].astype(str).tolist()
    records = (
        joined.drop(columns=["description"])
        .rename(columns={"description_record": "description"})
        .to_dict("records")
    )
    flagged = qc.unsupported_entity_flags(
        list(zip(joined["sequence_id"].astype(str), texts, records, strict=True))
    )

    report = {
        "described": len(texts),
        "orphaned_descriptions": orphaned,
        "length": qc.length_stats(texts),
        "field_coverage": qc.field_coverage(list(zip(texts, records, strict=True))),
        "duplicates": qc.duplicate_stats(texts),
        "unsupported_specificity": {
            "n_flagged": len(flagged),
            "frac_flagged": round(len(flagged) / max(len(texts), 1), 4),
            "n_with_invented_origin": sum(1 for flag in flagged if flag["invented_origins"]),
            "examples": flagged[: int(params.get("report_examples", 10))],
        },
    }
    logger.info(
        "QC: %s descriptions, %s flagged (%s with an invented origin)",
        f"{len(texts):,}",
        f"{len(flagged):,}",
        report["unsupported_specificity"]["n_with_invented_origin"],
    )
    return report, pd.DataFrame(flagged)
