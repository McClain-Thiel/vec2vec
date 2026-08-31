#!/usr/bin/env python3
"""Run E16 broad-vocabulary text-to-classifier distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_description_initialized_tuning import _evaluate, _select_variant
from run_semantic_hard_negative_tuning import _semantic_prompts
from run_text_conditioned_classifier import _prepare_features
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
    "e16_broad_vocabulary",
    "e16_broad_vocabulary_checkpoint",
    "e16_broad_vocabulary_atomic_metrics",
    "e16_broad_vocabulary_compositional_metrics",
    "e16_broad_vocabulary_report",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run", "validate"), required=True)
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
    params = dict(context_params["broad_vocabulary_distillation_experiment"])
    if args.stage == "validate":
        if not args.version:
            parser.error("--version is required for validation")
        _validate_outputs(catalog, params, args.version)
        return
    if args.annotations is None:
        parser.error("--annotations is required for audit and run stages")

    pairs = catalog.load("e06_pairs", version=str(e10_params["inputs"]["panel_version"]))
    _verify_frame_hash(
        pairs,
        expected=str(e10_params["inputs"]["pairs_sha256"]),
        sort_columns=["panel_role", "sequence_id"],
        name="E06 pairs",
    )
    annotations = _load_annotations(args.annotations, e10_params)
    benchmark = weak_annotations.build_weak_annotation_benchmark(pairs, annotations, e10_params)
    if dataframe_content_sha256(benchmark.queries, sort_columns=["query_kind", "query_id"]) != str(
        params["inputs"]["queries_sha256"]
    ):
        raise RuntimeError("E16 target queries differ from the frozen E10-E15 table")
    vocabulary, train_labels, audit = _build_vocabulary(
        pairs, annotations, benchmark, e10_params, params
    )
    expected_audit = params["vocabulary"]["expected_audit"]
    observed_audit = {key: audit[key] for key in expected_audit}
    if observed_audit != expected_audit:
        raise RuntimeError(f"E16 vocabulary audit differs: {observed_audit}")
    if args.stage == "audit":
        print(json.dumps(audit, sort_keys=True))
        return

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
        vocabulary,
        train_labels,
        authorization,
        deadline=deadline,
    )
    elapsed = time.perf_counter() - started
    versions = {name: catalog.get(name).resolve_save_version() for name in OUTPUT_DATASETS}
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"E16 output save versions differ: {versions}")
    output_hashes = {
        "vocabulary_sha256": dataframe_content_sha256(
            vocabulary, sort_columns=["concept_role", "query_id"]
        ),
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
            "student_training_concepts": int(vocabulary["concept_role"].eq("student_train").sum()),
            "held_target_concepts": int(vocabulary["concept_role"].eq("held_target").sum()),
            "conjunction_queries": int(params["inputs"]["pair_queries"]),
            "test_rows_read": False,
        },
        "vocabulary_audit": audit,
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
            "E16 tunes on the reused validation targets and is not confirmatory.",
            "Teacher classifiers treat unreported annotations as weak negatives.",
            "Positive-set Jaccard removes label-near-duplicate target leakage but not every alias.",
            "The frozen Qwen encoder and fixed DNA features are not fine-tuned.",
            "No test row is read.",
        ],
    }
    catalog.save("e16_broad_vocabulary", vocabulary)
    catalog.save("e16_broad_vocabulary_checkpoint", checkpoint)
    catalog.save("e16_broad_vocabulary_atomic_metrics", atomic)
    catalog.save("e16_broad_vocabulary_compositional_metrics", composition)
    catalog.save("e16_broad_vocabulary_report", report)
    _validate_outputs(catalog, params, next(iter(versions.values())))


def _build_vocabulary(
    pairs: pd.DataFrame,
    annotations: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    e10_params: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    train, _, calls = weak_annotations._validated_calls(pairs, annotations, e10_params)
    train_calls = calls.loc[calls["panel_role"].eq("alignment_train")]
    validation_calls = calls.loc[calls["panel_role"].eq("validation_gallery")]
    support = weak_annotations._annotation_support(train_calls, validation_calls)
    held_queries = benchmark.queries.loc[benchmark.queries["query_kind"].eq("atomic")].copy()
    held_keys = [json.loads(value)[0] for value in held_queries["annotation_keys_json"]]
    vocabulary = params["vocabulary"]
    eligible = support.loc[
        support["train_positive_rows"].between(
            int(vocabulary["minimum_train_rows"]), int(vocabulary["maximum_train_rows"])
        )
        & support["train_positive_components"].ge(int(vocabulary["minimum_train_components"]))
        & support["validation_positive_rows"].ge(int(vocabulary["minimum_validation_rows"]))
        & support["validation_positive_components"].ge(
            int(vocabulary["minimum_validation_components"])
        )
    ].sort_values(["train_positive_rows", "annotation_key"], ascending=[False, True], kind="stable")
    candidate_keys = eligible["annotation_key"].astype(str).tolist()
    sets = weak_annotations._call_sets(train_calls, sorted(set(candidate_keys) | set(held_keys)))
    maximum_jaccard = float(vocabulary["maximum_target_jaccard"])
    selected, excluded_near_target = [], []
    for key in candidate_keys:
        if key in held_keys:
            continue
        if any(_jaccard(sets[key], sets[held]) >= maximum_jaccard for held in held_keys):
            excluded_near_target.append(key)
            continue
        selected.append(key)
        if len(selected) == int(vocabulary["student_training_concepts"]):
            break
    if len(selected) != int(vocabulary["student_training_concepts"]):
        raise RuntimeError(
            f"E16 found {len(selected)} leakage-controlled training concepts, not "
            f"{vocabulary['student_training_concepts']}"
        )
    display = weak_annotations._display_names(train_calls, [*selected, *held_keys])
    support_by_key = support.set_index("annotation_key")
    training_rows = []
    for key in selected:
        values = support_by_key.loc[key]
        training_rows.append(
            {
                "query_id": hashlib.sha256(f"e16|student-train|{key}".encode()).hexdigest(),
                "concept_role": "student_train",
                "annotation_keys_json": json.dumps([key], separators=(",", ":")),
                "canonical_query_text": f"plasmid annotated with {display[key]}",
                "train_positive_rows": int(values["train_positive_rows"]),
                "train_positive_components": int(values["train_positive_components"]),
                "validation_positive_rows": int(values["validation_positive_rows"]),
                "validation_positive_components": int(values["validation_positive_components"]),
            }
        )
    held = held_queries.loc[
        :,
        [
            "query_id",
            "annotation_keys_json",
            "canonical_query_text",
            "train_positive_rows",
            "train_positive_components",
            "validation_positive_rows",
            "validation_positive_components",
        ],
    ].assign(concept_role="held_target")
    vocabulary_frame = pd.concat(
        [pd.DataFrame(training_rows), held], ignore_index=True
    ).sort_values(["concept_role", "query_id"], kind="stable", ignore_index=True)
    train_ids = train["sequence_id"].astype(str).tolist()
    positions = {value: position for position, value in enumerate(train_ids)}
    labels = np.zeros((len(vocabulary_frame), len(train)), dtype=bool)
    for row_position, row in enumerate(vocabulary_frame.itertuples(index=False)):
        key = json.loads(row.annotation_keys_json)[0]
        labels[row_position, [positions[value] for value in sets[key]]] = True
    held_positions = np.flatnonzero(vocabulary_frame["concept_role"].eq("held_target"))
    if not np.array_equal(labels[held_positions], benchmark.train_verified[: len(held_keys)]):
        raise RuntimeError("E16 held-target weak labels differ from E10-E15")
    audit = {
        "eligible_concepts": len(eligible),
        "student_training_concepts": len(selected),
        "held_target_concepts": len(held_keys),
        "excluded_exact_targets": len(set(candidate_keys) & set(held_keys)),
        "excluded_near_target_concepts": len(excluded_near_target),
        "maximum_target_jaccard": maximum_jaccard,
        "vocabulary_sha256": dataframe_content_sha256(
            vocabulary_frame, sort_columns=["concept_role", "query_id"]
        ),
        "training_labels_sha256": _array_sha256(labels),
    }
    return vocabulary_frame, labels, audit


def _run(
    catalog: Any,
    context_params: dict[str, Any],
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    vocabulary: pd.DataFrame,
    train_labels: np.ndarray,
    authorization: dict[str, Any],
    *,
    deadline: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import wandb

    prompts = _semantic_prompts(vocabulary)
    features = _prepare_features(
        catalog,
        context_params,
        e10_params,
        params,
        pairs,
        vocabulary,
        deadline=deadline,
        query_texts=prompts["text"].astype(str).tolist(),
    )
    prompt_count = int(params["semantic_prompts"]["prompts_per_concept"])
    text = features["query"].reshape(len(vocabulary), prompt_count, -1).mean(axis=1)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    train_positions = np.flatnonzero(vocabulary["concept_role"].eq("student_train"))
    held_positions = np.flatnonzero(vocabulary["concept_role"].eq("held_target"))
    atomic_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("atomic")
    ].reset_index(drop=True)
    pair_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("pair_conjunction")
    ].reset_index(drop=True)
    validation = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    run = wandb.init(
        project=str(params["tracking"]["project"]),
        entity=params["tracking"].get("entity"),
        group=str(params["tracking"]["group"]),
        name="e16-broad-vocabulary-distillation",
        tags=list(params["tracking"]["tags"]),
        config={
            "protocol_version": params["protocol_version"],
            "vocabulary": params["vocabulary"],
            "teacher": params["teacher"],
            "students": params["students"],
            "authorization": authorization,
        },
        reinit="finish_previous",
    )
    try:
        teacher_params = params["teacher"]
        teacher, teacher_history = alignment_probe.train_atomic_logistic_probe(
            features["dna_train"].astype(np.float32),
            train_labels,
            updates=int(teacher_params["updates"]),
            learning_rate=float(teacher_params["learning_rate"]),
            weight_decay=float(teacher_params["weight_decay"]),
            device=str(params["device"]),
            deadline_monotonic=deadline,
        )
        for row in teacher_history.iloc[::10].itertuples(index=False):
            run.log(
                {
                    "teacher/loss": float(row.loss),
                    "teacher/weight_norm_max": float(row.weight_norm_max),
                }
            )
        teacher_targets = np.column_stack(
            [teacher["weight"], teacher["bias"], teacher["log_prior_odds"]]
        ).astype(np.float32)
        generated, student_states, histories = _fit_students(
            text[train_positions],
            text[held_positions],
            teacher_targets[train_positions],
            params,
            deadline=deadline,
        )
        for name, history in histories.items():
            for row in history.iloc[::100].itertuples(index=False):
                run.log({f"student/{name}/loss": float(row.loss)})
        generated["teacher_oracle"] = teacher_targets[held_positions]
        atomic_tables, composition_tables, reports = [], [], {}
        for variant, targets in generated.items():
            state = _targets_to_classifier(targets, features["dna_train"].shape[1])
            _, calibrated = alignment_probe.atomic_logistic_scores(
                features["dna_validation"].astype(np.float32), state
            )
            atomic, composition = _evaluate(
                calibrated,
                atomic_queries,
                pair_queries,
                validation,
                benchmark,
                params,
            )
            atomic = atomic.assign(variant=variant)
            composition = composition.assign(variant=variant)
            atomic_tables.append(atomic)
            composition_tables.append(composition)
            summary = _composition_summary(composition, params)
            atomic_at_10 = float(
                atomic.loc[atomic["k"].eq(int(params["primary_k"])), "utility"].mean()
            )
            reports[variant] = {
                "atomic_utility_at_10": atomic_at_10,
                "compositional_summary": summary,
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
        candidates = {name: reports[name] for name in params["students"]}
        selected = _select_variant(candidates, params)
        comparison = _compare_to_e15(catalog, composition, selected, params)
        checkpoint = _checkpoint(
            selected,
            student_states[selected],
            generated[selected],
            features,
            vocabulary,
        )
        run.summary["selection/variant"] = selected
        run.summary["selection/minus_e15"] = comparison["selected_minus_e15"]
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
            "feature_preparation": {
                **{key: value for key, value in features.items() if key.endswith("sha256")},
                "semantic_text_sha256": _array_sha256(text),
                "semantic_prompts_sha256": dataframe_content_sha256(
                    prompts, sort_columns=["query_id", "prompt_id"]
                ),
            },
            "teacher": {
                "initial_loss": float(teacher_history.iloc[0]["loss"]),
                "final_loss": float(teacher_history.iloc[-1]["loss"]),
                "weight_sha256": _array_sha256(teacher["weight"]),
                "bias_sha256": _array_sha256(teacher["bias"]),
                "log_prior_odds_sha256": _array_sha256(teacher["log_prior_odds"]),
            },
            "student_histories": {
                name: {
                    "initial_loss": float(history.iloc[0]["loss"]),
                    "final_loss": float(history.iloc[-1]["loss"]),
                }
                for name, history in histories.items()
            },
            "variants": reports,
            "selection": {
                "variant": selected,
                "rule": str(params["selection"]["rule"]),
                "validation_selected": True,
            },
            "comparison": comparison,
            "tracking": {"run_id": run_id, "url": run_url, "status": "complete"},
            "decision": {
                "status": "exploratory_tuning_result",
                "held_target_labels_absent_from_student_training": True,
                "confirmatory_claim": False,
                "test_rows_read": False,
            },
        },
    )


def _fit_students(
    training_text: np.ndarray,
    held_text: np.ndarray,
    training_targets: np.ndarray,
    params: dict[str, Any],
    *,
    deadline: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    generated, states, histories = {}, {}, {}
    target_mean = training_targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = training_targets.std(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(target_scale, np.float32(1e-6))
    standardized = (training_targets - target_mean) / target_scale
    similarity = held_text @ training_text.T
    nearest = np.argmax(similarity, axis=1)
    generated["nearest_teacher"] = training_targets[nearest]
    states["nearest_teacher"] = {"nearest_training_positions": nearest.astype(np.int64)}
    for name, config in params["students"].items():
        kind = str(config["kind"])
        if kind == "nearest_teacher":
            continue
        if kind == "ridge":
            mapping, intercept = weak_annotations.fit_ridge_map(
                training_text, standardized, alpha=float(config["alpha"])
            )
            generated[name] = (held_text @ mapping + intercept) * target_scale + target_mean
            states[name] = {
                "mapping": mapping.astype(np.float32),
                "intercept": intercept.astype(np.float32),
                "target_mean": target_mean,
                "target_scale": target_scale,
            }
            continue
        if kind != "mlp":
            raise ValueError(f"unknown E16 student kind: {kind}")
        state, history = alignment_probe.train_text_conditioned_head_generator(
            training_text.astype(np.float32),
            training_targets.astype(np.float32),
            seed=int(config["seed"]),
            hidden_dimension=int(config["hidden_dimension"]),
            updates=int(config["updates"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            device=str(params["device"]),
            deadline_monotonic=deadline,
        )
        generated[name] = alignment_probe.predict_text_conditioned_heads(held_text, state)
        states[name], histories[name] = state, history
    return generated, states, histories


def _targets_to_classifier(targets: np.ndarray, dna_dimension: int) -> dict[str, np.ndarray]:
    values = np.asarray(targets, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dna_dimension + 2:
        raise ValueError("E16 generated classifier target shape differs")
    return {
        "weight": values[:, :dna_dimension],
        "bias": values[:, dna_dimension],
        "log_prior_odds": values[:, dna_dimension + 1],
    }


def _compare_to_e15(
    catalog: Any, composition: pd.DataFrame, selected: str, params: dict[str, Any]
) -> dict[str, Any]:
    e15 = catalog.load(
        "e15_semantic_hard_negative_compositional_metrics",
        version=str(params["inputs"]["e15_version"]),
    )
    _verify_frame_hash(
        e15,
        expected=str(params["inputs"]["e15_compositional_metrics_sha256"]),
        sort_columns=["variant", "query_id", "k"],
        name="E15 compositional metrics",
    )
    k = int(params["primary_k"])
    current = composition.loc[
        composition["variant"].eq(selected) & composition["k"].eq(k)
    ].set_index("query_id")["signed_strict_utility"]
    reference = e15.loc[
        e15["variant"].eq(str(params["inputs"]["e15_variant"])) & e15["k"].eq(k)
    ].set_index("query_id")["signed_strict_utility"]
    joined = pd.concat([current.rename("selected"), reference.rename("e15")], axis=1).dropna()
    if len(joined) != int(params["inputs"]["pair_queries"]):
        raise RuntimeError("E16 selected and E15 query metrics do not align")
    difference = (joined["selected"] - joined["e15"]).to_numpy()
    generator = np.random.default_rng(int(params["bootstrap_seed"]))
    positions = generator.integers(
        0, len(joined), size=(int(params["bootstrap_draws"]), len(joined))
    )
    return {
        "selected_variant": selected,
        "selected_signed_strict_utility_at_10": float(joined["selected"].mean()),
        "e15_signed_strict_utility_at_10": float(joined["e15"].mean()),
        "selected_minus_e15": float(difference.mean()),
        "selected_minus_e15_query_bootstrap_95_interval": _interval(
            difference[positions].mean(axis=1)
        ),
    }


def _checkpoint(
    selected: str,
    student_state: dict[str, Any],
    generated_targets: np.ndarray,
    features: dict[str, Any],
    vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    arrays = {
        "generated_targets": generated_targets,
        "dna_whitening_mean": features["dna_whitening_mean"],
        "dna_whitening_matrix": features["dna_whitening_matrix"],
        "text_whitening_mean": features["text_whitening_mean"],
        "text_whitening_matrix": features["text_whitening_matrix"],
        **student_state,
    }
    row: dict[str, Any] = {
        "model_id": "broad_vocabulary_selected_head_generator",
        "selected_variant": selected,
        "held_query_ids_json": json.dumps(
            vocabulary.loc[vocabulary["concept_role"].eq("held_target"), "query_id"].tolist(),
            separators=(",", ":"),
        ),
        "text_encoder_recipe_json": json.dumps(
            features["text_encoder_recipe"], sort_keys=True, separators=(",", ":")
        ),
    }
    for name, value in arrays.items():
        values = np.atleast_1d(np.asarray(value))
        if values.dtype.kind not in "iufb":
            raise ValueError(f"E16 checkpoint array {name} is not numeric")
        row[name] = values.reshape(-1).tolist()
        row[f"{name}_rows"] = values.shape[0]
        row[f"{name}_columns"] = values.shape[1] if values.ndim == 2 else 1
        row[f"{name}_sha256"] = _array_sha256(values)
    return pd.DataFrame([row])


def _validate_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    vocabulary = catalog.load("e16_broad_vocabulary", version=version)
    checkpoint = catalog.load("e16_broad_vocabulary_checkpoint", version=version)
    atomic = catalog.load("e16_broad_vocabulary_atomic_metrics", version=version)
    composition = catalog.load("e16_broad_vocabulary_compositional_metrics", version=version)
    report = catalog.load("e16_broad_vocabulary_report", version=version)
    observed = {
        "vocabulary_sha256": dataframe_content_sha256(
            vocabulary, sort_columns=["concept_role", "query_id"]
        ),
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(
            atomic, sort_columns=["variant", "query_id", "k"]
        ),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition, sort_columns=["variant", "query_id", "k"]
        ),
    }
    if observed != report["output_hashes"]:
        raise RuntimeError("E16 persisted output hashes differ")
    variants = {*params["students"], "teacher_oracle"}
    if len(atomic) != len(variants) * int(params["inputs"]["atomic_queries"]) * len(
        params["cutoffs"]
    ) or len(composition) != len(variants) * int(params["inputs"]["pair_queries"]) * len(
        params["cutoffs"]
    ):
        raise RuntimeError("E16 persisted metric row counts differ")
    if (
        set(atomic["variant"].astype(str)) != variants
        or set(composition["variant"].astype(str)) != variants
    ):
        raise RuntimeError("E16 variant coverage is incomplete")
    selected = str(report["selection"]["variant"])
    if selected not in params["students"] or selected != str(
        checkpoint.iloc[0]["selected_variant"]
    ):
        raise RuntimeError("E16 selection and checkpoint differ")
    for variant in variants:
        summary = _composition_summary(composition.loc[composition["variant"].eq(variant)], params)
        if summary != report["variants"][variant]["compositional_summary"]:
            raise RuntimeError(f"E16 {variant} summary differs from recalculation")
    if report["tracking"].get("status") != "complete" or report["population"]["test_rows_read"]:
        raise RuntimeError("E16 tracking or test-read state is invalid")
    print(
        json.dumps(
            {
                "status": "e16_broad_vocabulary_outputs_validated",
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
        raise ValueError(f"compute authorization differs from frozen E16 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E16 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


if __name__ == "__main__":
    main()
