"""Deterministic exact-rule labels and a compact validation benchmark.

Training labels are noisy supervision, not benchmark truth. The functions in
this module apply only enabled exact mappings, preserve each raw value, and
leave every other value unlabeled.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from vec2vec.lib.serialization import stable_json, to_jsonable
from vec2vec.lib.text import as_list, exact_metadata_key, sha256_text

_SOURCE_FIELDS = ("plasmid_copy", "growth_temp", "bacterial_resistance", "vector_types")
_REQUIRED_RETRIEVAL_COLUMNS = {
    "sequence_id",
    "sequence_sha256",
    "addgene_id",
    "url",
    "source_description",
    "leakage_component",
    "split_grouped",
    *_SOURCE_FIELDS,
}
_REQUIRED_ANNOTATION_COLUMNS = {
    "sequence_id",
    "source",
    "feature",
    "feature_type",
    "start",
    "end",
    "strand",
    "confidence",
}
_ALLOWED_SECTIONS = {
    "copy_class": {"included"},
    "growth_temperature": {"included", "reviewed_mappings"},
    "bacterial_selection": {"included", "reviewed_mappings"},
    "intended_use": {"expression_included", "use_included"},
}


@dataclass(frozen=True)
class ExactMapping:
    """One configured whole-value mapping."""

    source_field: str
    raw_value: str
    canonical_values: tuple[str, ...]
    rule_id: str
    facet: str
    relation: str
    section: str
    mapping_note: str | None


def _validate_inputs(
    retrieval: pd.DataFrame,
    plannotate: pd.DataFrame,
    params: Mapping[str, Any],
) -> tuple[str, str]:
    missing = _REQUIRED_RETRIEVAL_COLUMNS.difference(retrieval.columns)
    if missing:
        raise ValueError(f"retrieval dataset is missing columns: {sorted(missing)}")
    if retrieval.empty:
        raise ValueError("retrieval dataset is empty")
    if retrieval["sequence_id"].isna().any() or retrieval["sequence_id"].duplicated().any():
        raise ValueError("retrieval dataset needs unique, non-missing sequence_id values")
    identity = ["sequence_sha256", "leakage_component", "split_grouped"]
    if retrieval[identity].isna().any().any():
        raise ValueError("retrieval dataset contains missing identity values")

    training_split = str(params["training_split"])
    benchmark_split = str(params["benchmark_split"])
    if training_split == benchmark_split:
        raise ValueError("training_split and benchmark_split must differ")
    observed_splits = set(retrieval["split_grouped"].astype(str))
    allowed_splits = {training_split, benchmark_split}
    if not observed_splits <= allowed_splits:
        raise ValueError(
            "constraint evidence loaded unexpected splits: "
            f"{sorted(observed_splits - allowed_splits)}"
        )
    if not allowed_splits <= observed_splits:
        raise ValueError(f"constraint evidence needs both splits: {sorted(allowed_splits)}")

    missing_annotations = _REQUIRED_ANNOTATION_COLUMNS.difference(plannotate.columns)
    if missing_annotations:
        raise ValueError(f"pLannotate data is missing columns: {sorted(missing_annotations)}")
    sources = set(plannotate["source"].dropna().astype(str))
    if sources - {"plannotate"}:
        raise ValueError(f"annotation input contains non-pLannotate sources: {sorted(sources)}")
    return training_split, benchmark_split


def _canonical_values(
    specification: Sequence[str] | Mapping[str, Any],
    *,
    label: str,
) -> tuple[tuple[str, ...], str | None]:
    if isinstance(specification, Mapping):
        if set(specification) != {"canonical_values", "interpretation"}:
            raise ValueError(f"{label} must contain canonical_values and interpretation")
        values = specification["canonical_values"]
        note = str(specification["interpretation"]).strip()
        if not note:
            raise ValueError(f"{label} needs an interpretation")
    else:
        values = specification
        note = None
    canonical = tuple(str(value) for value in values)
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError(f"{label} contains invalid canonical values")
    return canonical, note


def _mapping_contract(
    facet_params: Mapping[str, Any], params: Mapping[str, Any]
) -> tuple[dict[tuple[str, str], ExactMapping], str]:
    enabled = params["enabled_sections"]
    if set(enabled) != set(_ALLOWED_SECTIONS):
        raise ValueError("enabled_sections must name all supported rule groups")
    for rule_name, sections in enabled.items():
        unknown = set(sections) - _ALLOWED_SECTIONS[rule_name]
        if unknown:
            raise ValueError(f"{rule_name} has unsupported enabled sections: {sorted(unknown)}")

    descriptors = {
        ("copy_class", "included"): (
            "plasmid_copy",
            facet_params["copy_class"]["facet"],
            facet_params["copy_class"]["relation"],
        ),
        ("growth_temperature", "included"): (
            "growth_temp",
            facet_params["growth_temperature"]["facet"],
            facet_params["growth_temperature"]["relation"],
        ),
        ("growth_temperature", "reviewed_mappings"): (
            "growth_temp",
            facet_params["growth_temperature"]["facet"],
            facet_params["growth_temperature"]["relation"],
        ),
        ("bacterial_selection", "included"): (
            "bacterial_resistance",
            facet_params["bacterial_selection"]["facet"],
            facet_params["bacterial_selection"]["relation"],
        ),
        ("bacterial_selection", "reviewed_mappings"): (
            "bacterial_resistance",
            facet_params["bacterial_selection"]["facet"],
            facet_params["bacterial_selection"]["relation"],
        ),
        ("intended_use", "expression_included"): (
            "vector_types",
            facet_params["intended_use"]["expression_facet"],
            facet_params["intended_use"]["expression_relation"],
        ),
        ("intended_use", "use_included"): (
            "vector_types",
            facet_params["intended_use"]["use_facet"],
            facet_params["intended_use"]["use_relation"],
        ),
    }

    mappings: dict[tuple[str, str], ExactMapping] = {}
    contract_records: list[dict[str, Any]] = []
    for rule_name in sorted(enabled):
        rule = facet_params[rule_name]
        for section in sorted(enabled[rule_name]):
            source_field, facet, relation = descriptors[(rule_name, section)]
            for raw_value, specification in sorted(rule[section].items()):
                canonical, note = _canonical_values(
                    specification,
                    label=f"{rule_name}.{section}.{raw_value}",
                )
                key = exact_metadata_key(raw_value)
                if key is None or (source_field, key) in mappings:
                    raise ValueError(
                        f"duplicate or empty exact mapping for {source_field}: {raw_value!r}"
                    )
                mapping = ExactMapping(
                    source_field=source_field,
                    raw_value=str(raw_value),
                    canonical_values=canonical,
                    rule_id=str(rule["rule_id"]),
                    facet=str(facet),
                    relation=str(relation),
                    section=str(section),
                    mapping_note=note,
                )
                mappings[(source_field, key)] = mapping
                contract_records.append(mapping.__dict__)
    return mappings, sha256_text(stable_json(contract_records))


def _source_values(row: Any, field: str) -> list[Any]:
    if field == "vector_types":
        return as_list(getattr(row, field))
    return [getattr(row, field)]


def _build_applications(
    retrieval: pd.DataFrame,
    mappings: Mapping[tuple[str, str], ExactMapping],
    params: Mapping[str, Any],
    contract_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    field_counts = {
        field: {"source_units": 0, "mapped_units": 0, "unlabeled_units": 0, "null_units": 0}
        for field in _SOURCE_FIELDS
    }
    unlabeled: dict[str, Counter[str]] = {field: Counter() for field in _SOURCE_FIELDS}
    evidence_version = str(params["evidence_version"])
    sampling_key = str(params["sampling_key"])

    ordered = retrieval.sort_values("sequence_id", kind="stable")
    for row in ordered.itertuples(index=False):
        for field in _SOURCE_FIELDS:
            values = _source_values(row, field)
            if not values:
                field_counts[field]["null_units"] += 1
                continue
            for raw_value in values:
                field_counts[field]["source_units"] += 1
                key = exact_metadata_key(raw_value)
                if key is None:
                    field_counts[field]["null_units"] += 1
                    continue
                mapping = mappings.get((field, key))
                if mapping is None:
                    field_counts[field]["unlabeled_units"] += 1
                    unlabeled[field][stable_json(raw_value)] += 1
                    continue

                field_counts[field]["mapped_units"] += 1
                source_value_json = stable_json(raw_value)
                application_id = sha256_text(
                    f"{evidence_version}|{row.sequence_id}|{field}|{source_value_json}|"
                    f"{mapping.rule_id}|{mapping.facet}"
                )
                records.append(
                    {
                        "evidence_version": evidence_version,
                        "rule_contract_sha256": contract_hash,
                        "mapping_application_id": application_id,
                        "sequence_id": str(row.sequence_id),
                        "sequence_sha256": str(row.sequence_sha256),
                        "addgene_id": int(row.addgene_id),
                        "url": str(row.url),
                        "source_description": to_jsonable(row.source_description),
                        "leakage_component": str(row.leakage_component),
                        "split_grouped": str(row.split_grouped),
                        "rule_id": mapping.rule_id,
                        "facet": mapping.facet,
                        "relation": mapping.relation,
                        "source_field": field,
                        "source_value_json": source_value_json,
                        "canonical_values_json": stable_json(sorted(mapping.canonical_values)),
                        "mapping_section": mapping.section,
                        "mapping_note": mapping.mapping_note,
                        "selection_hash": sha256_text(
                            f"{sampling_key}|{mapping.facet}|{row.leakage_component}|"
                            f"{application_id}"
                        ),
                    }
                )

    if not records:
        raise ValueError("enabled exact mappings produced no constraint applications")
    applications = pd.DataFrame.from_records(records)
    if applications["mapping_application_id"].duplicated().any():
        raise ValueError("mapping_application_id values are not unique")

    for field, counts in field_counts.items():
        denominator = counts["mapped_units"] + counts["unlabeled_units"]
        counts["mapping_coverage"] = (
            round(counts["mapped_units"] / denominator, 6) if denominator else None
        )
        ordered_unlabeled = sorted(unlabeled[field].items(), key=lambda item: (-item[1], item[0]))
        counts["unlabeled_distinct_values"] = len(ordered_unlabeled)
        counts["unlabeled_values_sha256"] = sha256_text(stable_json(ordered_unlabeled))
        counts["top_unlabeled_values"] = [
            {"source_value_json": value, "count": count} for value, count in ordered_unlabeled[:20]
        ]
        counts["top_unlabeled_values_limit"] = 20
    return applications, field_counts


def _training_claims(applications: pd.DataFrame, training_split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keep = [
        "evidence_version",
        "rule_contract_sha256",
        "mapping_application_id",
        "sequence_id",
        "sequence_sha256",
        "addgene_id",
        "url",
        "leakage_component",
        "split_grouped",
        "rule_id",
        "facet",
        "relation",
        "source_field",
        "source_value_json",
        "mapping_section",
        "mapping_note",
    ]
    for record in applications.loc[applications["split_grouped"].eq(training_split)].to_dict(
        "records"
    ):
        for canonical_value in json.loads(record["canonical_values_json"]):
            row = {column: record[column] for column in keep}
            row.update(
                {
                    "canonical_value": str(canonical_value),
                    "evidence_id": sha256_text(
                        f"{record['mapping_application_id']}|{canonical_value}"
                    ),
                    "label_source": "deterministic_exact_rule",
                    "training_label_created": True,
                    "benchmark_label_created": False,
                }
            )
            rows.append(row)
    claims = pd.DataFrame.from_records(rows)
    if claims.empty:
        raise ValueError("training split produced no constraint claims")
    if claims["evidence_id"].duplicated().any():
        raise ValueError("evidence_id values are not unique")
    return claims.sort_values(
        ["sequence_id", "facet", "canonical_value", "evidence_id"], kind="stable"
    ).reset_index(drop=True)


def _benchmark_sample(
    applications: pd.DataFrame,
    benchmark_split: str,
    params: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    benchmark = params["benchmark"]
    target = int(benchmark["target_applications"])
    minimum = int(benchmark["minimum_per_facet"])
    if target <= 0 or minimum < 0:
        raise ValueError("benchmark target must be positive and minimum_per_facet non-negative")

    candidates = applications.loc[applications["split_grouped"].eq(benchmark_split)].copy()
    candidates = candidates.sort_values("selection_hash", kind="stable")
    candidates = candidates.drop_duplicates(["facet", "leakage_component"], keep="first")
    if candidates.empty:
        raise ValueError("benchmark split produced no component-distinct applications")

    minimum_total = sum(
        min(minimum, int(count)) for count in candidates["facet"].value_counts().values
    )
    if target < minimum_total:
        raise ValueError(
            f"benchmark target {target} is smaller than the facet minimum total {minimum_total}"
        )

    selected_ids: set[str] = set()
    for facet in sorted(candidates["facet"].unique()):
        facet_rows = candidates.loc[candidates["facet"].eq(facet)]
        selected_ids.update(
            facet_rows.head(min(minimum, len(facet_rows)))["mapping_application_id"]
        )
    remaining = candidates.loc[~candidates["mapping_application_id"].isin(selected_ids)]
    room = max(0, min(target, len(candidates)) - len(selected_ids))
    selected_ids.update(remaining.head(room)["mapping_application_id"])

    sample = candidates.loc[candidates["mapping_application_id"].isin(selected_ids)].copy()
    sample = sample.sort_values("selection_hash", kind="stable").head(target).reset_index(drop=True)
    sample.insert(0, "benchmark_index", range(1, len(sample) + 1))
    sample.insert(1, "benchmark_sample_version", str(params["benchmark_sample_version"]))
    sample["judge_status"] = "not_run"
    sample["benchmark_label_created"] = False
    allocation = {
        str(facet): int(count)
        for facet, count in sample["facet"].value_counts().sort_index().items()
    }
    return sample, allocation


def _attach_plannotate(sample: pd.DataFrame, plannotate: pd.DataFrame) -> pd.DataFrame:
    selected = set(sample["sequence_id"])
    evidence = plannotate.loc[plannotate["sequence_id"].astype(str).isin(selected)].copy()
    evidence = evidence.sort_values(
        ["sequence_id", "start", "end", "feature", "feature_type"], kind="stable"
    )
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for record in evidence.to_dict("records"):
        sequence_id = str(record.pop("sequence_id"))
        record.pop("source", None)
        by_sequence.setdefault(sequence_id, []).append(to_jsonable(record))

    result = sample.copy()
    records = result["sequence_id"].map(lambda value: by_sequence.get(str(value), []))
    result["plannotate_features_json"] = records.map(stable_json)
    result["plannotate_feature_count"] = records.map(len)
    result["plannotate_evidence_state"] = result["plannotate_feature_count"].map(
        lambda count: "present" if count else "missing"
    )
    return result


def _population_hash(retrieval: pd.DataFrame) -> str:
    columns = ["sequence_id", "sequence_sha256", "leakage_component", "split_grouped"]
    records = retrieval[columns].astype(str).sort_values("sequence_id", kind="stable")
    return sha256_text(
        "\n".join("|".join(row) for row in records.itertuples(index=False, name=None))
    )


def build_constraint_evidence(
    retrieval: pd.DataFrame,
    plannotate: pd.DataFrame,
    params: Mapping[str, Any],
    facet_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build all enabled training claims and a deterministic validation sample."""
    training_split, benchmark_split = _validate_inputs(retrieval, plannotate, params)
    mappings, contract_hash = _mapping_contract(facet_params, params)
    applications, field_counts = _build_applications(retrieval, mappings, params, contract_hash)
    training = _training_claims(applications, training_split)
    benchmark, allocation = _benchmark_sample(applications, benchmark_split, params)
    benchmark = _attach_plannotate(benchmark, plannotate)

    manifest = {
        "evidence_version": str(params["evidence_version"]),
        "benchmark_sample_version": str(params["benchmark_sample_version"]),
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "input_population_sha256": _population_hash(retrieval),
        "rule_contract_sha256": contract_hash,
        "training_split": training_split,
        "benchmark_split": benchmark_split,
        "input_rows_by_split": {
            str(split): int(count)
            for split, count in retrieval["split_grouped"].value_counts().sort_index().items()
        },
        "enabled_exact_mapping_count": len(mappings),
        "mapping_applications_by_split": {
            str(split): int(count)
            for split, count in applications["split_grouped"].value_counts().sort_index().items()
        },
        "mapping_applications_by_facet": {
            str(facet): int(count)
            for facet, count in applications["facet"].value_counts().sort_index().items()
        },
        "training_claims_created": int(len(training)),
        "benchmark_target_applications": int(params["benchmark"]["target_applications"]),
        "benchmark_sample_applications": int(len(benchmark)),
        "benchmark_allocation_by_facet": allocation,
        "benchmark_unique_components": int(benchmark["leakage_component"].nunique()),
        "benchmark_sequences_with_plannotate": int(
            benchmark.loc[benchmark["plannotate_feature_count"].gt(0), "sequence_id"].nunique()
        ),
        "benchmark_sequences_without_plannotate": int(
            benchmark.loc[benchmark["plannotate_feature_count"].eq(0), "sequence_id"].nunique()
        ),
        "source_field_coverage": field_counts,
        "label_policy": "deterministic_exact_rules_for_training",
        "benchmark_labels_created": False,
        "test_rows_loaded": 0,
        "test_metadata_used": False,
        "generated_descriptions_used": False,
        "annotation_source": "plannotate",
        "plasmidkit_fallback_used": False,
    }
    return training, benchmark, manifest
