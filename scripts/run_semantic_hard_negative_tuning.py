#!/usr/bin/env python3
"""Tune semantic query prompts and hard-negative adaptation for E15."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_description_initialized_tuning import (
    _checkpoint,
    _evaluate,
    _fit_shared_calibration,
    _scores,
    _select_variant,
)
from run_text_conditioned_classifier import _folds, _prepare_features, _sample_positions
from run_weak_annotation_classifier import _composition_summary, _hardware, _interval
from run_weak_annotation_experiment import (
    PROJECT_ROOT,
    _array_sha256,
    _catalog_and_params,
    _file_sha256,
    _git_state,
    _load_annotations,
    _verify_frame_hash,
)

from vec2vec.lib import alignment_probe, weak_annotations
from vec2vec.lib.serialization import dataframe_content_sha256, json_content_sha256

OUTPUT_DATASETS = (
    "e15_semantic_hard_negative_checkpoint",
    "e15_semantic_hard_negative_atomic_metrics",
    "e15_semantic_hard_negative_compositional_metrics",
    "e15_semantic_hard_negative_report",
)

_ALIASES = {
    "ampicillin": "an ampicillin-resistance selectable marker",
    "aph 3 iia": "APH(3')-IIa, a kanamycin and neomycin resistance enzyme",
    "blatem": "TEM beta-lactamase, an ampicillin-resistance enzyme",
    "ble": "a bleomycin or zeocin resistance selectable marker",
    "blemx6": "the bleMX6 bleomycin or zeocin resistance marker",
    "bleor": "a bleomycin or zeocin resistance selectable marker",
    "egfp": "enhanced green fluorescent protein, a fluorescent reporter",
    "gmr": "a gentamicin-resistance selectable marker",
    "kanmx": "the kanMX geneticin or kanamycin resistance marker",
    "kanr": "a kanamycin-resistance selectable marker",
    "kanr2 gene": "a kanamycin-resistance gene",
    "mcherry": "mCherry red fluorescent protein, a fluorescent reporter",
    "neo": "a neomycin or geneticin resistance selectable marker",
    "neor kanr": "a neomycin and kanamycin resistance selectable marker",
    "neor kanr 3": "a neomycin and kanamycin resistance selectable marker",
    "puro": "a puromycin-resistance selectable marker",
    "puror": "a puromycin-resistance selectable marker",
    "rbs": "a ribosome-binding site controlling translation initiation",
    "tracrrna": "a CRISPR trans-activating guide RNA component",
    "wpre": "the woodchuck hepatitis virus post-transcriptional regulatory element",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "validate"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--approval-reference")
    parser.add_argument("--region")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-hour-limit", type=float)
    parser.add_argument("--price-usd-per-hour", type=float)
    args = parser.parse_args()

    catalog, context_params = _catalog_and_params()
    e10_params = dict(context_params["weak_annotation_experiment"])
    params = dict(context_params["semantic_hard_negative_tuning_experiment"])
    if args.stage == "validate":
        if not args.version:
            parser.error("--version is required for validation")
        _validate_outputs(catalog, params, args.version)
        return
    if args.annotations is None:
        parser.error("--annotations is required for the run")

    pairs = catalog.load("e06_pairs", version=str(e10_params["inputs"]["panel_version"]))
    _verify_frame_hash(
        pairs,
        expected=str(e10_params["inputs"]["pairs_sha256"]),
        sort_columns=["panel_role", "sequence_id"],
        name="E06 pairs",
    )
    annotations = _load_annotations(args.annotations, e10_params)
    benchmark = weak_annotations.build_weak_annotation_benchmark(pairs, annotations, e10_params)
    query_hash = dataframe_content_sha256(
        benchmark.queries, sort_columns=["query_kind", "query_id"]
    )
    if query_hash != str(params["inputs"]["queries_sha256"]):
        raise RuntimeError("E15 query table differs from the frozen E10-E14 table")

    authorization = _validated_authorization(args, params)
    started = time.perf_counter()
    deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0
    checkpoint, atomic, composition, result = _run(
        catalog,
        context_params,
        e10_params,
        params,
        pairs,
        benchmark,
        authorization,
        deadline=deadline,
    )
    elapsed = time.perf_counter() - started
    versions = {name: catalog.get(name).resolve_save_version() for name in OUTPUT_DATASETS}
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"E15 output save versions differ: {versions}")
    output_hashes = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(
            atomic, sort_columns=["variant", "query_id", "k"]
        ),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition, sort_columns=["variant", "query_id", "k"]
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
        "resolved_configuration": params,
        "inputs": {**params["inputs"], "annotation_file_sha256": _file_sha256(args.annotations)},
        "population": {
            "training_rows": int(e10_params["inputs"]["training_rows"]),
            "validation_rows": int(e10_params["inputs"]["validation_rows"]),
            "atomic_queries": int(params["inputs"]["atomic_queries"]),
            "conjunction_queries": int(params["inputs"]["pair_queries"]),
            "test_rows_read": False,
        },
        **result,
        "output_hashes": output_hashes,
        "execution": {
            **authorization,
            "maximum_cost_usd": (
                authorization["instance_hour_limit"]
                * authorization["observed_instance_price_usd_per_hour"]
            ),
            "elapsed_seconds": elapsed,
            "cost_usd": elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"],
            "hardware": _hardware(),
            "git": _git_state(),
            "source_sha256": {
                "script": _file_sha256(Path(__file__)),
                "alignment_probe_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/alignment_probe.py"
                ),
                "parameters": _file_sha256(PROJECT_ROOT / "conf/base/parameters_modeling_data.yml"),
            },
            "artifact_versions": versions,
        },
        "known_limitations": [
            "E15 tunes on the reused validation split and is not confirmatory.",
            "Uncalled annotations are noisy weak negatives, not verified biological absences.",
            "Semantic definitions are deterministic templates and a small curated alias map.",
            "The selected all-atom fit is persisted but not evaluated as a deployable model.",
            "No test row is read.",
        ],
    }
    catalog.save("e15_semantic_hard_negative_checkpoint", checkpoint)
    catalog.save("e15_semantic_hard_negative_atomic_metrics", atomic)
    catalog.save("e15_semantic_hard_negative_compositional_metrics", composition)
    catalog.save("e15_semantic_hard_negative_report", report)
    _validate_outputs(catalog, params, next(iter(versions.values())))


def _run(
    catalog: Any,
    context_params: dict[str, Any],
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    authorization: dict[str, Any],
    *,
    deadline: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import wandb

    atomic_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("atomic")
    ].reset_index(drop=True)
    pair_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("pair_conjunction")
    ].reset_index(drop=True)
    prompts = _semantic_prompts(atomic_queries)
    features = _prepare_features(
        catalog,
        context_params,
        e10_params,
        params,
        pairs,
        atomic_queries,
        deadline=deadline,
        query_texts=prompts["text"].astype(str).tolist(),
    )
    prompt_count = int(params["semantic_prompts"]["prompts_per_atom"])
    query = features["query"].reshape(len(atomic_queries), prompt_count, -1).mean(axis=1)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    features["query"] = query
    features["semantic_query_embeddings_sha256"] = _array_sha256(query)
    train = pairs.loc[pairs["panel_role"].eq("alignment_train")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    validation = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    probe = params["description_pretraining"]
    run = wandb.init(
        project=str(params["tracking"]["project"]),
        entity=params["tracking"].get("entity"),
        group=str(params["tracking"]["group"]),
        name="e15-semantic-hard-negative-tuning",
        tags=list(params["tracking"]["tags"]),
        config={
            "protocol_version": params["protocol_version"],
            "semantic_prompts": params["semantic_prompts"],
            "description_pretraining": params["description_pretraining"],
            "adaptation": params["adaptation"],
            "hard_negatives": params["hard_negatives"],
            "variants": params["variants"],
            "authorization": authorization,
        },
        reinit="finish_previous",
    )
    try:
        pretrained, pretrain_history = alignment_probe.train_alignment_probe(
            features["dna_train"].astype(np.float32),
            features["text_train"].astype(np.float32),
            train["sequence_sha256"].astype(str).to_numpy(),
            train["description_sha256"].astype(str).to_numpy(),
            seed=int(probe["seed"]),
            projection_dimension=int(probe["projection_dimension"]),
            epochs=int(probe["epochs"]),
            batch_size=int(probe["batch_size"]),
            learning_rate=float(probe["learning_rate"]),
            weight_decay=float(probe["weight_decay"]),
            initial_temperature=float(probe["initial_temperature"]),
            maximum_logit_scale=float(probe["maximum_logit_scale"]),
            device=str(params["device"]),
            deadline_monotonic=deadline,
        )
        for row in pretrain_history.itertuples(index=False):
            run.log(
                {
                    "pretrain/loss": float(row.mean_loss),
                    "pretrain/logit_scale": float(row.logit_scale),
                }
            )
        pools = _hard_negative_pools(pretrained, features, benchmark, params)
        folds = _folds(atomic_queries["query_id"].astype(str).tolist(), params)
        sample = _sample_positions(
            features["train_ids"],
            count=int(params["calibration"]["training_rows"]),
            salt=str(params["calibration"]["sample_salt"]),
        )
        atomic_tables, composition_tables, variant_reports = [], [], {}
        for variant, negatives_per_query in params["variants"].items():
            oof_logits = np.empty(
                (len(features["dna_validation"]), len(atomic_queries)), dtype=np.float32
            )
            fold_reports = []
            for fold in range(int(params["outer_folds"])):
                train_atoms = np.flatnonzero(folds != fold)
                held_atoms = np.flatnonzero(folds == fold)
                state, history = _adapt(
                    pretrained,
                    features,
                    benchmark,
                    pools,
                    train_atoms,
                    params,
                    negatives_per_query=negatives_per_query,
                    deadline=deadline,
                )
                for row in history.iloc[::20].itertuples(index=False):
                    run.log(
                        {
                            f"adaptation/{variant}/fold_{fold}_loss": float(row.loss),
                            f"adaptation/{variant}/fold_{fold}_logit_scale": float(row.logit_scale),
                        }
                    )
                raw_train = _scores(
                    features["dna_train"][sample], features["query"][train_atoms], state, params
                )
                slope, intercept, calibration_loss = _fit_shared_calibration(
                    raw_train, benchmark.train_verified[train_atoms][:, sample], params
                )
                raw_validation = _scores(
                    features["dna_validation"], features["query"][held_atoms], state, params
                )
                oof_logits[:, held_atoms] = (slope * raw_validation + intercept).T
                fold_reports.append(
                    {
                        "fold": fold,
                        "training_atoms": len(train_atoms),
                        "held_atoms": len(held_atoms),
                        "negatives_per_query": negatives_per_query,
                        "final_adaptation_loss": float(history.iloc[-1]["loss"]),
                        "calibration_loss": calibration_loss,
                    }
                )
            atomic, composition = _evaluate(
                oof_logits, atomic_queries, pair_queries, validation, benchmark, params
            )
            atomic = atomic.assign(variant=variant)
            composition = composition.assign(variant=variant)
            atomic_tables.append(atomic)
            composition_tables.append(composition)
            summary = _composition_summary(composition, params)
            atomic_at_10 = float(
                atomic.loc[atomic["k"].eq(int(params["primary_k"])), "utility"].mean()
            )
            variant_reports[variant] = {
                "negatives_per_query": negatives_per_query,
                "atomic_utility_at_10": atomic_at_10,
                "compositional_summary": summary,
                "folds": fold_reports,
            }
            run.summary[f"validation/{variant}/atomic_utility_at_10"] = atomic_at_10
            for metric, values in summary["primary_k"].items():
                run.summary[f"validation/{variant}/{metric}_at_10"] = values["mean"]
        atomic = pd.concat(atomic_tables, ignore_index=True).sort_values(
            ["variant", "query_id", "k"], kind="stable", ignore_index=True
        )
        composition = pd.concat(composition_tables, ignore_index=True).sort_values(
            ["variant", "query_id", "k"], kind="stable", ignore_index=True
        )
        selected = _select_variant(variant_reports, params)
        comparison = _compare_to_e14(catalog, composition, selected, params)
        selected_state, selected_history = _adapt(
            pretrained,
            features,
            benchmark,
            pools,
            np.arange(len(atomic_queries)),
            params,
            negatives_per_query=params["variants"][selected],
            deadline=deadline,
        )
        raw_train = _scores(
            features["dna_train"][sample], features["query"], selected_state, params
        )
        slope, intercept, calibration_loss = _fit_shared_calibration(
            raw_train, benchmark.train_verified[:, sample], params
        )
        checkpoint = _checkpoint(
            selected_state, features, selected, slope=slope, intercept=intercept
        )
        checkpoint.loc[0, "model_id"] = "semantic_hard_negative_selected_all_atom_fit"
        run.summary["selection/variant"] = selected
        run.summary["selection/minus_e14"] = comparison["selected_minus_e14"]
        run_id, run_url = str(run.id), str(run.url)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return (
        checkpoint,
        atomic,
        composition,
        {
            "semantic_prompts": {
                "rows": len(prompts),
                "sha256": dataframe_content_sha256(prompts, sort_columns=["query_id", "prompt_id"]),
            },
            "feature_preparation": {
                key: value for key, value in features.items() if key.endswith("sha256")
            },
            "description_pretraining": {
                "initial_loss": float(pretrain_history.iloc[0]["mean_loss"]),
                "final_loss": float(pretrain_history.iloc[-1]["mean_loss"]),
                "epochs": len(pretrain_history),
            },
            "hard_negative_pool_sha256": _array_sha256(pools),
            "variants": variant_reports,
            "selection": {
                "variant": selected,
                "rule": str(params["selection"]["rule"]),
                "all_atom_final_adaptation_loss": float(selected_history.iloc[-1]["loss"]),
                "all_atom_calibration_loss": calibration_loss,
                "validation_selected": True,
            },
            "comparison": comparison,
            "tracking": {"run_id": run_id, "url": run_url, "status": "complete"},
            "decision": {
                "status": "exploratory_tuning_result",
                "atom_targets_held_out_during_evaluation": True,
                "confirmatory_claim": False,
                "test_rows_read": False,
            },
        },
    )


def _semantic_prompts(atomic_queries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query in atomic_queries.itertuples(index=False):
        key = json.loads(query.annotation_keys_json)[0]
        name = str(query.canonical_query_text).removeprefix("plasmid annotated with ")
        definition = _ALIASES.get(key, _generic_definition(key))
        texts = (
            str(query.canonical_query_text),
            f"Find plasmids containing the genetic feature {name}.",
            f"A DNA construct with {definition}.",
            f"Retrieve plasmids annotated with {key.replace('_', ' ')}.",
        )
        rows.extend(
            {"query_id": str(query.query_id), "prompt_id": position, "text": text}
            for position, text in enumerate(texts)
        )
    return pd.DataFrame(rows)


def _generic_definition(key: str) -> str:
    if "promoter" in key:
        return f"the {key}, a transcriptional promoter"
    if "terminator" in key or "poly a" in key or " pa " in f" {key} ":
        return f"the {key}, a transcription termination or RNA-processing element"
    if " ori" in f" {key}" or key.endswith("ori"):
        return f"the {key}, a plasmid origin of replication"
    if "intron" in key:
        return f"the {key}, an intronic sequence"
    if key.startswith("attb") or key == "loxp":
        return f"the {key}, a site-specific recombination sequence"
    if "ltr" in key or "itr" in key:
        return f"the {key}, a viral terminal repeat sequence"
    if "grna" in key:
        return f"the {key}, a CRISPR guide RNA component"
    if "9xhis" in key:
        return "a nine-histidine affinity purification tag"
    return f"the annotated genetic element {key}"


def _hard_negative_pools(
    state: dict[str, Any],
    features: dict[str, Any],
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    params: dict[str, Any],
) -> np.ndarray:
    scores = _scores(features["dna_train"], features["query"], state, params)
    size = int(params["hard_negatives"]["pool_rows_per_atom"])
    pools = []
    for row, labels in zip(scores, benchmark.train_verified, strict=True):
        candidates = np.flatnonzero(~labels)
        order = np.argsort(-row[candidates], kind="stable")
        if len(order) < size:
            raise RuntimeError("E15 hard-negative pool exceeds available weak negatives")
        pools.append(candidates[order[:size]])
    return np.asarray(pools, dtype=np.int64)


def _adapt(
    pretrained: dict[str, Any],
    features: dict[str, Any],
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    pools: np.ndarray,
    atoms: np.ndarray,
    params: dict[str, Any],
    *,
    negatives_per_query: int | None,
    deadline: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    adaptation = params["adaptation"]
    if negatives_per_query is None:
        return alignment_probe.train_controlled_query_probe(
            features["dna_train"].astype(np.float32),
            features["query"][atoms].astype(np.float32),
            benchmark.train_verified[atoms],
            objective="verified_set",
            seed=int(adaptation["seed"]),
            projection_dimension=int(params["description_pretraining"]["projection_dimension"]),
            updates=int(adaptation["updates"]),
            learning_rate=float(adaptation["learning_rate"]),
            weight_decay=float(adaptation["weight_decay"]),
            initial_temperature=float(params["description_pretraining"]["initial_temperature"]),
            maximum_logit_scale=float(params["description_pretraining"]["maximum_logit_scale"]),
            device=str(params["device"]),
            initial_state=pretrained,
            deadline_monotonic=deadline,
        )
    return alignment_probe.train_hard_negative_query_probe(
        features["dna_train"].astype(np.float32),
        features["query"][atoms].astype(np.float32),
        benchmark.train_verified[atoms],
        pools[atoms],
        seed=int(adaptation["seed"]),
        updates=int(adaptation["updates"]),
        negatives_per_query=int(negatives_per_query),
        learning_rate=float(adaptation["learning_rate"]),
        weight_decay=float(adaptation["weight_decay"]),
        maximum_logit_scale=float(params["description_pretraining"]["maximum_logit_scale"]),
        device=str(params["device"]),
        initial_state=pretrained,
        deadline_monotonic=deadline,
    )


def _compare_to_e14(
    catalog: Any, composition: pd.DataFrame, selected: str, params: dict[str, Any]
) -> dict[str, Any]:
    e14 = catalog.load(
        "e14_description_initialized_compositional_metrics",
        version=str(params["inputs"]["e14_version"]),
    )
    _verify_frame_hash(
        e14,
        expected=str(params["inputs"]["e14_compositional_metrics_sha256"]),
        sort_columns=["variant", "query_id", "k"],
        name="E14 compositional metrics",
    )
    k = int(params["primary_k"])
    current = composition.loc[
        composition["variant"].eq(selected) & composition["k"].eq(k)
    ].set_index("query_id")["signed_strict_utility"]
    reference = e14.loc[
        e14["variant"].eq(str(params["inputs"]["e14_variant"])) & e14["k"].eq(k)
    ].set_index("query_id")["signed_strict_utility"]
    joined = pd.concat([current.rename("selected"), reference.rename("e14")], axis=1).dropna()
    if len(joined) != int(params["inputs"]["pair_queries"]):
        raise RuntimeError("E15 selected and E14 query metrics do not align")
    difference = (joined["selected"] - joined["e14"]).to_numpy()
    generator = np.random.default_rng(int(params["bootstrap_seed"]))
    positions = generator.integers(
        0, len(joined), size=(int(params["bootstrap_draws"]), len(joined))
    )
    return {
        "selected_variant": selected,
        "selected_signed_strict_utility_at_10": float(joined["selected"].mean()),
        "e14_signed_strict_utility_at_10": float(joined["e14"].mean()),
        "selected_minus_e14": float(difference.mean()),
        "selected_minus_e14_query_bootstrap_95_interval": _interval(
            difference[positions].mean(axis=1)
        ),
    }


def _validate_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    checkpoint = catalog.load("e15_semantic_hard_negative_checkpoint", version=version)
    atomic = catalog.load("e15_semantic_hard_negative_atomic_metrics", version=version)
    composition = catalog.load("e15_semantic_hard_negative_compositional_metrics", version=version)
    report = catalog.load("e15_semantic_hard_negative_report", version=version)
    observed = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(
            atomic, sort_columns=["variant", "query_id", "k"]
        ),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition, sort_columns=["variant", "query_id", "k"]
        ),
    }
    if observed != report["output_hashes"]:
        raise RuntimeError("E15 persisted output hashes differ")
    variants = set(params["variants"])
    if len(atomic) != len(variants) * int(params["inputs"]["atomic_queries"]) * len(
        params["cutoffs"]
    ) or len(composition) != len(variants) * int(params["inputs"]["pair_queries"]) * len(
        params["cutoffs"]
    ):
        raise RuntimeError("E15 persisted metric row counts differ")
    if (
        set(atomic["variant"].astype(str)) != variants
        or set(composition["variant"].astype(str)) != variants
    ):
        raise RuntimeError("E15 variant coverage is incomplete")
    selected = str(report["selection"]["variant"])
    if selected not in variants or selected != str(checkpoint.iloc[0]["selected_variant"]):
        raise RuntimeError("E15 selection and checkpoint differ")
    for variant in variants:
        summary = _composition_summary(composition.loc[composition["variant"].eq(variant)], params)
        if summary != report["variants"][variant]["compositional_summary"]:
            raise RuntimeError(f"E15 {variant} summary differs from recalculation")
    if report["tracking"].get("status") != "complete" or report["population"]["test_rows_read"]:
        raise RuntimeError("E15 tracking or test-read state is invalid")
    print(
        json.dumps(
            {
                "status": "e15_semantic_hard_negative_outputs_validated",
                "version": version,
                "report_sha256": json_content_sha256(report),
                "output_hashes": observed,
                "selection": report["selection"],
                "comparison": report["comparison"],
                "tracking": report["tracking"],
            },
            sort_keys=True,
        )
    )


def _validated_authorization(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "approval_reference": args.approval_reference,
        "region": args.region,
        "instance_type": args.instance_type,
        "instance_hour_limit": args.instance_hour_limit,
        "observed_instance_price_usd_per_hour": args.price_usd_per_hour,
    }
    if observed != params["compute_authorization"]:
        raise ValueError(f"compute authorization differs from frozen E15 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E15 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


if __name__ == "__main__":
    main()
