"""Deterministic sampling for the E00 manual facet audit.

The functions in this module preserve raw metadata and proposed mappings. They do not accept a
mapping, create benchmark evidence, or inspect model output.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from vec2vec.lib.serialization import stable_json, to_jsonable
from vec2vec.lib.text import as_list, exact_metadata_key, sha256_text

_REQUIRED_COLUMNS = {
    "sequence_id",
    "sequence_sha256",
    "addgene_id",
    "url",
    "description",
    "source_description",
    "leakage_component",
    "split_grouped",
    "plasmid_copy",
    "growth_temp",
    "bacterial_resistance",
    "vector_types",
}


def _require_columns(frame: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"retrieval dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("retrieval dataset is empty")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("retrieval dataset needs unique, non-missing sequence_id values")
    identity = ["sequence_sha256", "leakage_component", "split_grouped"]
    if frame[identity].isna().any().any():
        raise ValueError("retrieval dataset contains missing sampling identity values")


def _load_mapping(values: Mapping[str, Sequence[str]], label: str) -> dict[str, tuple[str, ...]]:
    """Build a punctuation-preserving lookup and reject normalized-key collisions."""
    result: dict[str, tuple[str, ...]] = {}
    for raw_value, canonical_values in values.items():
        key = exact_metadata_key(raw_value)
        canonical = tuple(str(value) for value in canonical_values)
        if key is None or not canonical or len(set(canonical)) != len(canonical):
            raise ValueError(f"{label} contains an invalid mapping for {raw_value!r}")
        if key in result:
            raise ValueError(f"{label} contains a duplicate exact key: {raw_value!r}")
        result[key] = canonical
    return result


def _load_reviewed_mappings(
    values: Mapping[str, Mapping[str, Any]], label: str
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Load exact mappings that need a reviewer-readable interpretation."""
    mappings: dict[str, tuple[str, ...]] = {}
    notes: dict[str, str] = {}
    required_fields = {"canonical_values", "interpretation"}
    for raw_value, specification in values.items():
        if set(specification) != required_fields:
            raise ValueError(
                f"{label} mapping for {raw_value!r} must contain only "
                "canonical_values and interpretation"
            )
        loaded = _load_mapping(
            {raw_value: specification["canonical_values"]},
            f"{label}.{raw_value}",
        )
        key, canonical_values = next(iter(loaded.items()))
        interpretation = str(specification["interpretation"]).strip()
        if not interpretation:
            raise ValueError(f"{label} mapping for {raw_value!r} needs an interpretation")
        if key in mappings:
            raise ValueError(f"{label} contains a duplicate exact key: {raw_value!r}")
        mappings[key] = canonical_values
        notes[key] = interpretation
    return mappings, notes


def _load_exclusions(values: Mapping[str, str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_value, reason in values.items():
        key = exact_metadata_key(raw_value)
        if key is None or not str(reason).strip():
            raise ValueError(f"{label} contains an invalid exclusion for {raw_value!r}")
        if key in result:
            raise ValueError(f"{label} contains a duplicate exact key: {raw_value!r}")
        result[key] = str(reason)
    return result


def _load_missing(values: Sequence[str], label: str) -> set[str]:
    result: set[str] = set()
    for raw_value in values:
        key = exact_metadata_key(raw_value)
        if key is None or key in result:
            raise ValueError(f"{label} contains an invalid value: {raw_value!r}")
        result.add(key)
    return result


def _check_disjoint(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"{label} contains overlapping keys: {sorted(overlap)}")


def _population_identity(frame: pd.DataFrame) -> str:
    """Hash identity, split, and component fields without reading test metadata."""
    digest = hashlib.sha256()
    columns = ["sequence_id", "sequence_sha256", "leakage_component", "split_grouped"]
    ordered = frame.loc[:, columns].astype(str).sort_values("sequence_id", kind="stable")
    for row in ordered.itertuples(index=False, name=None):
        digest.update("|".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _candidate(
    row_index: int,
    row: pd.Series,
    *,
    sampling_key: str,
    audit_version: str,
    rule_id: str,
    facet: str,
    relation: str | None,
    stratum: str,
    source_field: str,
    source_value: Any,
    classified_values: Sequence[str],
    canonical_values: Sequence[str],
    mapping_status: str,
    proposed_evidence_state: str,
    exclusion_reason: str | None = None,
    mapping_note: str | None = None,
) -> dict[str, Any]:
    component = str(row["leakage_component"])
    sequence_id = str(row["sequence_id"])
    selection_hash = sha256_text(f"{sampling_key}|{facet}|{stratum}|{component}|{sequence_id}")
    claims = [
        {"facet": facet, "relation": relation, "canonical_value": value}
        for value in sorted(canonical_values)
    ]
    source_value_json = stable_json(source_value)
    return {
        "row_index": row_index,
        "audit_version": audit_version,
        "rule_id": rule_id,
        "facet": facet,
        "relation": relation,
        "stratum": stratum,
        "source_field": source_field,
        "source_value_json": source_value_json,
        "classified_source_values_json": stable_json(sorted(classified_values)),
        "canonical_values": tuple(sorted(canonical_values)),
        "canonical_values_json": stable_json(sorted(canonical_values)),
        "proposed_claims_json": stable_json(claims),
        "mapping_status": mapping_status,
        "proposed_evidence_state": proposed_evidence_state,
        "mapping_note": mapping_note,
        "exclusion_reason": exclusion_reason,
        "leakage_component": component,
        "sequence_id": sequence_id,
        "selection_hash": selection_hash,
        "audit_row_id": sha256_text(
            f"{audit_version}|{facet}|{stratum}|{sequence_id}|{source_value_json}"
        ),
    }


def _keep_component_candidate(
    candidates: dict[str, dict[str, dict[str, Any]]], record: dict[str, Any]
) -> None:
    """Keep the lowest-hash row for one component in one stratum."""
    by_component = candidates.setdefault(record["stratum"], {})
    previous = by_component.get(record["leakage_component"])
    if previous is None or record["selection_hash"] < previous["selection_hash"]:
        by_component[record["leakage_component"]] = record


def _sample_candidates(
    records: Sequence[dict[str, Any]],
    *,
    target: int | None,
    minimum_per_canonical: int,
    expected_canonical_values: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select candidates by the frozen hash order and canonical-value minimum."""
    ordered = sorted(records, key=lambda record: record["selection_hash"])
    available_by_value: Counter[str] = Counter({value: 0 for value in expected_canonical_values})
    available_by_value.update(value for record in ordered for value in record["canonical_values"])
    selected: dict[str, dict[str, Any]] = {}

    if minimum_per_canonical:
        for value in sorted(available_by_value):
            for record in ordered:
                if value not in record["canonical_values"]:
                    continue
                selected.setdefault(record["leakage_component"], record)
                selected_count = sum(
                    value in item["canonical_values"] for item in selected.values()
                )
                if selected_count >= minimum_per_canonical:
                    break

    if target is not None and len(selected) > target:
        raise ValueError("canonical minimums require more rows than the stratum target")

    limit = len(ordered) if target is None else target
    for record in ordered:
        if len(selected) >= limit:
            break
        selected.setdefault(record["leakage_component"], record)

    sampled = sorted(selected.values(), key=lambda record: record["selection_hash"])
    selected_by_value = {
        value: sum(value in record["canonical_values"] for record in sampled)
        for value in sorted(available_by_value)
    }
    available = len(ordered)
    target_met = len(sampled) == available if target is None else len(sampled) >= target
    minimum_met = all(
        available_by_value[value] >= minimum_per_canonical
        and selected_by_value[value] >= minimum_per_canonical
        for value in available_by_value
    )
    if not available_by_value:
        minimum_met = minimum_per_canonical == 0
    summary = {
        "available_components": available,
        "selected_components": len(sampled),
        "target_components": target,
        "sample_all": target is None,
        "target_met": target_met,
        "minimum_per_canonical": minimum_per_canonical,
        "minimum_met": minimum_met,
        "available_components_by_canonical": dict(available_by_value),
        "selected_components_by_canonical": selected_by_value,
    }
    return sampled, summary


def _classifiers(params: Mapping[str, Any]) -> dict[str, Any]:
    copy = params["copy_class"]
    growth = params["growth_temperature"]
    resistance = params["bacterial_selection"]
    intended = params["intended_use"]

    copy_included = _load_mapping(copy["included"], "copy_class.included")
    copy_missing = _load_missing(copy["missing_exact"], "copy_class.missing_exact")
    growth_included = _load_mapping(growth["included"], "growth_temperature.included")
    growth_reviewed, growth_notes = _load_reviewed_mappings(
        growth["reviewed_mappings"], "growth_temperature.reviewed_mappings"
    )
    growth_held_out = _load_exclusions(growth["held_out"], "growth_temperature.held_out")
    growth_missing = _load_missing(growth["missing_exact"], "growth_temperature.missing_exact")
    resistance_included = _load_mapping(resistance["included"], "bacterial_selection.included")
    resistance_reviewed, resistance_notes = _load_reviewed_mappings(
        resistance["reviewed_mappings"], "bacterial_selection.reviewed_mappings"
    )
    resistance_excluded = _load_exclusions(resistance["excluded"], "bacterial_selection.excluded")
    resistance_missing = _load_missing(
        resistance["missing_exact"], "bacterial_selection.missing_exact"
    )
    expression = _load_mapping(intended["expression_included"], "intended_use.expression_included")
    use = _load_mapping(intended["use_included"], "intended_use.use_included")
    intended_excluded = _load_exclusions(intended["excluded_exact"], "intended_use.excluded_exact")
    intended_missing = _load_missing(intended["missing_exact"], "intended_use.missing_exact")

    _check_disjoint(growth_included, growth_reviewed, "growth temperature mappings")
    growth_included = growth_included | growth_reviewed
    _check_disjoint(growth_included, growth_held_out, "growth temperature")
    _check_disjoint(resistance_included, resistance_reviewed, "bacterial selection mappings")
    resistance_included = resistance_included | resistance_reviewed
    _check_disjoint(resistance_included, resistance_excluded, "bacterial selection")
    _check_disjoint(expression, use, "intended use controlled mappings")
    _check_disjoint(expression | use, intended_excluded, "intended use")
    if set(copy_included) & copy_missing:
        raise ValueError("copy class contains included missing-value keys")
    if (set(growth_included) | set(growth_held_out)) & growth_missing:
        raise ValueError("growth temperature contains included missing-value keys")
    if (set(resistance_included) | set(resistance_excluded)) & resistance_missing:
        raise ValueError("bacterial selection contains included missing-value keys")
    if (set(expression) | set(use) | set(intended_excluded)) & intended_missing:
        raise ValueError("intended use contains included missing-value keys")
    return {
        "copy": copy_included,
        "copy_missing": copy_missing,
        "growth": growth_included,
        "growth_notes": growth_notes,
        "growth_held_out": growth_held_out,
        "growth_missing": growth_missing,
        "resistance": resistance_included,
        "resistance_notes": resistance_notes,
        "resistance_excluded": resistance_excluded,
        "resistance_missing": resistance_missing,
        "expression": expression,
        "use": use,
        "intended_excluded": intended_excluded,
        "intended_missing": intended_missing,
    }


def _vocabulary_classification(
    field: str,
    key: str | None,
    params: Mapping[str, Any],
    maps: Mapping[str, Any],
) -> tuple[str, str, str | None, tuple[str, ...], str | None]:
    """Return rule, facet, relation, canonical values, and exclusion reason."""
    if key is None:
        rule_name = {
            "plasmid_copy": "copy_class",
            "growth_temp": "growth_temperature",
            "bacterial_resistance": "bacterial_selection",
            "vector_types": "intended_use",
        }[field]
        rule = params[rule_name]
        facet = rule.get("facet", "addgene_intended_use_source")
        return str(rule["rule_id"]), str(facet), None, (), "Source field is missing."

    if field == "plasmid_copy":
        if key in maps["copy_missing"]:
            rule = params["copy_class"]
            return rule["rule_id"], rule["facet"], None, (), "Configured missing value."
        if key not in maps["copy"]:
            raise ValueError(f"unclassified plasmid_copy value: {key!r}")
        rule = params["copy_class"]
        return rule["rule_id"], rule["facet"], rule["relation"], maps["copy"][key], None
    if field == "growth_temp":
        rule = params["growth_temperature"]
        if key in maps["growth_missing"]:
            return rule["rule_id"], rule["facet"], None, (), "Configured missing value."
        if key in maps["growth"]:
            return rule["rule_id"], rule["facet"], rule["relation"], maps["growth"][key], None
        if key in maps["growth_held_out"]:
            return rule["rule_id"], rule["facet"], None, (), maps["growth_held_out"][key]
        raise ValueError(f"unclassified growth_temp value: {key!r}")
    if field == "bacterial_resistance":
        rule = params["bacterial_selection"]
        if key in maps["resistance_missing"]:
            return rule["rule_id"], rule["facet"], None, (), "Configured missing value."
        if key in maps["resistance"]:
            return (
                rule["rule_id"],
                rule["facet"],
                rule["relation"],
                maps["resistance"][key],
                None,
            )
        if key in maps["resistance_excluded"]:
            return (
                rule["rule_id"],
                rule["facet"],
                None,
                (),
                maps["resistance_excluded"][key],
            )
        raise ValueError(f"unclassified bacterial_resistance value: {key!r}")
    if field == "vector_types":
        rule = params["intended_use"]
        if key in maps["intended_missing"]:
            return (
                rule["rule_id"],
                "addgene_intended_use_source",
                None,
                (),
                "Configured missing value.",
            )
        if key in maps["expression"]:
            return (
                rule["rule_id"],
                rule["expression_facet"],
                rule["expression_relation"],
                maps["expression"][key],
                None,
            )
        if key in maps["use"]:
            return (
                rule["rule_id"],
                rule["use_facet"],
                rule["use_relation"],
                maps["use"][key],
                None,
            )
        reason = maps["intended_excluded"].get(key, str(rule["default_exclusion_reason"]))
        return rule["rule_id"], "addgene_intended_use_source", None, (), reason
    raise ValueError(f"unsupported audit field: {field!r}")


def _build_vocabulary(
    frame: pd.DataFrame,
    *,
    params: Mapping[str, Any],
    maps: Mapping[str, Any],
) -> pd.DataFrame:
    """Classify configured and observed audit-population values."""
    accumulators: dict[tuple[str, str | None], dict[str, Any]] = {}
    configured: dict[tuple[str, str], str] = {}
    configured_sections = [
        ("plasmid_copy", params["copy_class"]["included"]),
        ("plasmid_copy", params["copy_class"]["missing_exact"]),
        ("growth_temp", params["growth_temperature"]["included"]),
        ("growth_temp", params["growth_temperature"]["reviewed_mappings"]),
        ("growth_temp", params["growth_temperature"]["held_out"]),
        ("growth_temp", params["growth_temperature"]["missing_exact"]),
        ("bacterial_resistance", params["bacterial_selection"]["included"]),
        ("bacterial_resistance", params["bacterial_selection"]["reviewed_mappings"]),
        ("bacterial_resistance", params["bacterial_selection"]["excluded"]),
        ("bacterial_resistance", params["bacterial_selection"]["missing_exact"]),
        ("vector_types", params["intended_use"]["expression_included"]),
        ("vector_types", params["intended_use"]["use_included"]),
        ("vector_types", params["intended_use"]["excluded_exact"]),
        ("vector_types", params["intended_use"]["missing_exact"]),
    ]
    for field, values in configured_sections:
        for raw_value in values:
            key = exact_metadata_key(raw_value)
            if key is None:
                raise ValueError(f"configured {field} value is empty: {raw_value!r}")
            configured[(field, key)] = str(raw_value)
            accumulators.setdefault(
                (field, key),
                {"raw_variants": Counter(), "rows": 0, "components": set()},
            )

    list_fields = {"vector_types"}
    for row in frame.itertuples(index=False):
        row_values = row._asdict()
        component = str(row_values["leakage_component"])
        for field in ("plasmid_copy", "growth_temp", "bacterial_resistance", "vector_types"):
            raw_cell = row_values[field]
            values = as_list(raw_cell) if field in list_fields else [raw_cell]
            values = [value for value in values if exact_metadata_key(value) is not None] or [None]
            seen_keys: set[str | None] = set()
            for value in values:
                key = exact_metadata_key(value)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                lookup = (field, key)
                entry = accumulators.setdefault(
                    lookup,
                    {
                        "raw_variants": Counter(),
                        "rows": 0,
                        "components": set(),
                    },
                )
                entry["raw_variants"][stable_json(value)] += 1
                entry["rows"] += 1
                entry["components"].add(component)

    records: list[dict[str, Any]] = []
    for (field, key), entry in sorted(
        accumulators.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        rule_id, facet, relation, canonical, reason = _vocabulary_classification(
            field, key, params, maps
        )
        configured_missing = (
            (field == "plasmid_copy" and key in maps["copy_missing"])
            or (field == "growth_temp" and key in maps["growth_missing"])
            or (field == "bacterial_resistance" and key in maps["resistance_missing"])
            or (field == "vector_types" and key in maps["intended_missing"])
        )
        if key is None or configured_missing:
            status = "missing"
        elif field == "growth_temp" and key in maps["growth_held_out"]:
            status = "held_out"
        else:
            status = "proposed_include" if canonical else "proposed_exclude"
        records.append(
            {
                "rule_id": rule_id,
                "facet": facet,
                "relation": relation,
                "source_field": field,
                "exact_key": key,
                "configured_source_value_json": (
                    stable_json(configured[(field, key)])
                    if key is not None and (field, key) in configured
                    else None
                ),
                "raw_variants_json": stable_json(
                    [
                        {"raw_value_json": raw, "rows": count}
                        for raw, count in sorted(
                            entry["raw_variants"].items(), key=lambda item: (-item[1], item[0])
                        )
                    ]
                ),
                "mapping_status": status,
                "canonical_values_json": stable_json(sorted(canonical)),
                "mapping_note": (
                    maps["growth_notes"].get(key)
                    if field == "growth_temp"
                    else maps["resistance_notes"].get(key)
                    if field == "bacterial_resistance"
                    else None
                ),
                "exclusion_reason": reason,
                "observed_in_audit_population": entry["rows"] > 0,
                "audit_population_row_support": entry["rows"],
                "audit_population_component_support": len(entry["components"]),
            }
        )
    return pd.DataFrame.from_records(records)


def build_facet_audit_sample(
    frame: pd.DataFrame, params: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the frozen component-aware review sample and its manifest.

    Returns:
        The row-review sample, observed-value classification, and a sampling manifest. All mappings
        and evidence states remain proposals.
    """
    _require_columns(frame)
    audit_version = str(params["audit_version"])
    sampling_key = str(params["sampling_key"])
    eligible_splits = tuple(str(value) for value in params["eligible_splits"])
    if not audit_version or not sampling_key:
        raise ValueError("audit_version and sampling_key cannot be empty")
    if not eligible_splits or len(set(eligible_splits)) != len(eligible_splits):
        raise ValueError("eligible_splits must be non-empty and unique")
    if "test" in eligible_splits:
        raise ValueError("test cannot be an eligible facet-audit split")
    observed_splits = set(frame["split_grouped"].astype(str))
    missing_splits = set(eligible_splits).difference(observed_splits)
    if missing_splits:
        raise ValueError(f"eligible splits are absent from the input: {sorted(missing_splits)}")

    maps = _classifiers(params)
    eligible_set = set(eligible_splits)
    eligible = frame.loc[frame["split_grouped"].astype(str).isin(eligible_set)].reset_index(
        drop=True
    )
    vocabulary = _build_vocabulary(eligible, params=params, maps=maps)
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    sample_specs: dict[str, tuple[int | None, int, tuple[str, ...]]] = {}

    copy = params["copy_class"]
    growth = params["growth_temperature"]
    resistance = params["bacterial_selection"]
    intended = params["intended_use"]
    for canonical in sorted(value[0] for value in maps["copy"].values()):
        sample_specs[f"copy_class:{canonical}"] = (
            int(copy["target_per_value"]),
            0,
            (canonical,),
        )
    sample_specs["copy_class:missing"] = (int(copy["missing_target"]), 0, ())
    for canonical in sorted(
        value[0] for key, value in maps["growth"].items() if key not in maps["growth_notes"]
    ):
        sample_specs[f"growth_temperature:{canonical}"] = (
            int(growth["target_per_value"]),
            0,
            (canonical,),
        )
    sample_specs["growth_temperature:reviewed_exact_mapping"] = (
        None,
        0,
        tuple(
            sorted(
                {
                    value
                    for key, values in maps["growth"].items()
                    if key in maps["growth_notes"]
                    for value in values
                }
            )
        ),
    )
    sample_specs["growth_temperature:held_out"] = (
        None if growth["sample_all_held_out"] else int(growth["target_per_value"]),
        0,
        (),
    )
    sample_specs["growth_temperature:missing"] = (
        None if growth["sample_all_missing"] else int(growth["target_per_value"]),
        0,
        (),
    )
    resistance_single_values = tuple(
        sorted(
            {
                values[0]
                for key, values in maps["resistance"].items()
                if key not in maps["resistance_notes"] and len(values) == 1
            }
        )
    )
    resistance_combination_values = tuple(
        sorted(
            {
                value
                for key, values in maps["resistance"].items()
                if key not in maps["resistance_notes"] and len(values) > 1
                for value in values
            }
        )
    )
    sample_specs["bacterial_selection:proposed_include_single"] = (
        int(resistance["included_single_target"]),
        int(resistance["minimum_per_canonical"]),
        resistance_single_values,
    )
    sample_specs["bacterial_selection:proposed_include_combination"] = (
        int(resistance["included_combination_target"]),
        int(resistance["minimum_per_canonical"]),
        resistance_combination_values,
    )
    sample_specs["bacterial_selection:reviewed_exact_mapping"] = (
        None,
        0,
        tuple(
            sorted(
                {
                    value
                    for key, values in maps["resistance"].items()
                    if key in maps["resistance_notes"]
                    for value in values
                }
            )
        ),
    )
    sample_specs["bacterial_selection:proposed_exclude"] = (
        int(resistance["excluded_target"]),
        0,
        (),
    )
    sample_specs["bacterial_selection:missing"] = (
        None if resistance["sample_all_missing"] else int(resistance["excluded_target"]),
        0,
        (),
    )
    sample_specs["intended_use:expression_context"] = (
        int(intended["expression_target"]),
        int(intended["minimum_per_canonical"]),
        tuple(sorted({value for values in maps["expression"].values() for value in values})),
    )
    sample_specs["intended_use:use_category"] = (
        int(intended["use_target"]),
        int(intended["minimum_per_canonical"]),
        tuple(sorted({value for values in maps["use"].values() for value in values})),
    )
    sample_specs["intended_use:proposed_exclude"] = (
        int(intended["excluded_target"]),
        0,
        (),
    )
    sample_specs["intended_use:missing"] = (int(intended["missing_target"]), 0, ())

    for row_index, row in eligible.iterrows():
        copy_key = exact_metadata_key(row["plasmid_copy"])
        if copy_key is None or copy_key in maps["copy_missing"]:
            record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=copy["rule_id"],
                facet=copy["facet"],
                relation=None,
                stratum="copy_class:missing",
                source_field="plasmid_copy",
                source_value=row["plasmid_copy"],
                classified_values=(),
                canonical_values=(),
                mapping_status="missing",
                proposed_evidence_state="unknown",
                exclusion_reason="Source field is missing.",
            )
        else:
            canonical = maps["copy"].get(copy_key)
            if canonical is None:
                raise ValueError(f"unclassified plasmid_copy value: {copy_key!r}")
            record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=copy["rule_id"],
                facet=copy["facet"],
                relation=copy["relation"],
                stratum=f"copy_class:{canonical[0]}",
                source_field="plasmid_copy",
                source_value=row["plasmid_copy"],
                classified_values=(str(row["plasmid_copy"]),),
                canonical_values=canonical,
                mapping_status="proposed_include",
                proposed_evidence_state="verified",
            )
        _keep_component_candidate(candidates, record)

        growth_key = exact_metadata_key(row["growth_temp"])
        if growth_key is None or growth_key in maps["growth_missing"]:
            growth_record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=growth["rule_id"],
                facet=growth["facet"],
                relation=None,
                stratum="growth_temperature:missing",
                source_field="growth_temp",
                source_value=row["growth_temp"],
                classified_values=(),
                canonical_values=(),
                mapping_status="missing",
                proposed_evidence_state="unknown",
                exclusion_reason="Source field is missing.",
            )
        elif growth_key in maps["growth"]:
            canonical = maps["growth"][growth_key]
            growth_record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=growth["rule_id"],
                facet=growth["facet"],
                relation=growth["relation"],
                stratum=(
                    "growth_temperature:reviewed_exact_mapping"
                    if growth_key in maps["growth_notes"]
                    else f"growth_temperature:{canonical[0]}"
                ),
                source_field="growth_temp",
                source_value=row["growth_temp"],
                classified_values=(str(row["growth_temp"]),),
                canonical_values=canonical,
                mapping_status="proposed_include",
                proposed_evidence_state="verified",
                mapping_note=maps["growth_notes"].get(growth_key),
            )
        elif growth_key in maps["growth_held_out"]:
            growth_record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=growth["rule_id"],
                facet=growth["facet"],
                relation=None,
                stratum="growth_temperature:held_out",
                source_field="growth_temp",
                source_value=row["growth_temp"],
                classified_values=(str(row["growth_temp"]),),
                canonical_values=(),
                mapping_status="held_out",
                proposed_evidence_state="unknown",
                exclusion_reason=maps["growth_held_out"][growth_key],
            )
        else:
            raise ValueError(f"unclassified growth_temp value: {growth_key!r}")
        _keep_component_candidate(candidates, growth_record)

        resistance_key = exact_metadata_key(row["bacterial_resistance"])
        if resistance_key is None or resistance_key in maps["resistance_missing"]:
            resistance_record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=resistance["rule_id"],
                facet=resistance["facet"],
                relation=None,
                stratum="bacterial_selection:missing",
                source_field="bacterial_resistance",
                source_value=row["bacterial_resistance"],
                classified_values=(),
                canonical_values=(),
                mapping_status="missing",
                proposed_evidence_state="unknown",
                exclusion_reason="Source field is missing.",
            )
        elif resistance_key in maps["resistance"]:
            canonical = maps["resistance"][resistance_key]
            resistance_record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=resistance["rule_id"],
                facet=resistance["facet"],
                relation=resistance["relation"],
                stratum=(
                    "bacterial_selection:reviewed_exact_mapping"
                    if resistance_key in maps["resistance_notes"]
                    else "bacterial_selection:proposed_include_single"
                    if len(canonical) == 1
                    else "bacterial_selection:proposed_include_combination"
                ),
                source_field="bacterial_resistance",
                source_value=row["bacterial_resistance"],
                classified_values=(str(row["bacterial_resistance"]),),
                canonical_values=canonical,
                mapping_status="proposed_include",
                proposed_evidence_state="verified",
                mapping_note=maps["resistance_notes"].get(resistance_key),
            )
        elif resistance_key in maps["resistance_excluded"]:
            resistance_record = _candidate(
                row_index,
                row,
                sampling_key=sampling_key,
                audit_version=audit_version,
                rule_id=resistance["rule_id"],
                facet=resistance["facet"],
                relation=None,
                stratum="bacterial_selection:proposed_exclude",
                source_field="bacterial_resistance",
                source_value=row["bacterial_resistance"],
                classified_values=(str(row["bacterial_resistance"]),),
                canonical_values=(),
                mapping_status="proposed_exclude",
                proposed_evidence_state="unknown",
                exclusion_reason=maps["resistance_excluded"][resistance_key],
            )
        else:
            raise ValueError(f"unclassified bacterial_resistance value: {resistance_key!r}")
        _keep_component_candidate(candidates, resistance_record)

        raw_vector_values = [
            value
            for value in as_list(row["vector_types"])
            if exact_metadata_key(value) is not None
            and exact_metadata_key(value) not in maps["intended_missing"]
        ]
        if not raw_vector_values:
            _keep_component_candidate(
                candidates,
                _candidate(
                    row_index,
                    row,
                    sampling_key=sampling_key,
                    audit_version=audit_version,
                    rule_id=intended["rule_id"],
                    facet="addgene_intended_use_source",
                    relation=None,
                    stratum="intended_use:missing",
                    source_field="vector_types",
                    source_value=row["vector_types"],
                    classified_values=(),
                    canonical_values=(),
                    mapping_status="missing",
                    proposed_evidence_state="unknown",
                    exclusion_reason="Source field is missing.",
                ),
            )
            continue

        expression_values: list[str] = []
        expression_canonical: set[str] = set()
        use_values: list[str] = []
        use_canonical: set[str] = set()
        excluded_values: list[str] = []
        exclusion_reasons: dict[str, str] = {}
        for value in raw_vector_values:
            key = exact_metadata_key(value)
            if key in maps["expression"]:
                expression_values.append(str(value))
                expression_canonical.update(maps["expression"][key])
            elif key in maps["use"]:
                use_values.append(str(value))
                use_canonical.update(maps["use"][key])
            else:
                excluded_values.append(str(value))
                exclusion_reasons[str(value)] = maps["intended_excluded"].get(
                    key, str(intended["default_exclusion_reason"])
                )
        if expression_values:
            _keep_component_candidate(
                candidates,
                _candidate(
                    row_index,
                    row,
                    sampling_key=sampling_key,
                    audit_version=audit_version,
                    rule_id=intended["rule_id"],
                    facet=intended["expression_facet"],
                    relation=intended["expression_relation"],
                    stratum="intended_use:expression_context",
                    source_field="vector_types",
                    source_value=row["vector_types"],
                    classified_values=expression_values,
                    canonical_values=sorted(expression_canonical),
                    mapping_status="proposed_include",
                    proposed_evidence_state="verified",
                ),
            )
        if use_values:
            _keep_component_candidate(
                candidates,
                _candidate(
                    row_index,
                    row,
                    sampling_key=sampling_key,
                    audit_version=audit_version,
                    rule_id=intended["rule_id"],
                    facet=intended["use_facet"],
                    relation=intended["use_relation"],
                    stratum="intended_use:use_category",
                    source_field="vector_types",
                    source_value=row["vector_types"],
                    classified_values=use_values,
                    canonical_values=sorted(use_canonical),
                    mapping_status="proposed_include",
                    proposed_evidence_state="verified",
                ),
            )
        if excluded_values:
            _keep_component_candidate(
                candidates,
                _candidate(
                    row_index,
                    row,
                    sampling_key=sampling_key,
                    audit_version=audit_version,
                    rule_id=intended["rule_id"],
                    facet="addgene_intended_use_source",
                    relation=None,
                    stratum="intended_use:proposed_exclude",
                    source_field="vector_types",
                    source_value=row["vector_types"],
                    classified_values=excluded_values,
                    canonical_values=(),
                    mapping_status="proposed_exclude",
                    proposed_evidence_state="unknown",
                    exclusion_reason=stable_json(exclusion_reasons),
                ),
            )

    selected: list[dict[str, Any]] = []
    stratum_manifest: dict[str, Any] = {}
    for stratum, (target, minimum, expected_values) in sample_specs.items():
        records = list(candidates.get(stratum, {}).values())
        sampled, summary = _sample_candidates(
            records,
            target=target,
            minimum_per_canonical=minimum,
            expected_canonical_values=expected_values,
        )
        for rank, record in enumerate(sampled, start=1):
            record["selection_rank"] = rank
        selected.extend(sampled)
        stratum_manifest[stratum] = summary
    if not selected:
        raise RuntimeError("facet audit produced no review rows")

    output_records: list[dict[str, Any]] = []
    modulus = int(params["second_review_modulus"])
    if modulus < 2:
        raise ValueError("second_review_modulus must be at least 2")
    for record in selected:
        source = eligible.iloc[record.pop("row_index")]
        canonical_values = record.pop("canonical_values")
        record.update(
            {
                "input_retrieval_version": str(params["input_retrieval_version"]),
                "addgene_id": int(source["addgene_id"]),
                "url": str(source["url"]),
                "split_grouped": str(source["split_grouped"]),
                "generated_description": to_jsonable(source["description"]),
                "source_description": to_jsonable(source["source_description"]),
                "second_review_sample": int(record["selection_hash"][:16], 16) % modulus == 0,
                "canonical_value_count": len(canonical_values),
            }
        )
        output_records.append(record)

    sample = (
        pd.DataFrame.from_records(output_records)
        .sort_values(["stratum", "selection_rank", "selection_hash"], kind="stable")
        .reset_index(drop=True)
    )
    if sample["audit_row_id"].duplicated().any():
        raise RuntimeError("facet audit produced duplicate audit_row_id values")
    if set(sample["split_grouped"]) - eligible_set:
        raise RuntimeError("facet audit sample contains an ineligible split")

    manifest = {
        "audit_version": audit_version,
        "sampling_key": sampling_key,
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "input_population_sha256": _population_identity(frame),
        "input_rows": len(frame),
        "eligible_splits": list(eligible_splits),
        "eligible_rows": len(eligible),
        "test_rows_excluded": int(frame["split_grouped"].astype(str).eq("test").sum()),
        "eligible_components": int(eligible["leakage_component"].nunique()),
        "sample_rows": len(sample),
        "sample_components": int(sample["leakage_component"].nunique()),
        "vocabulary_rows": len(vocabulary),
        "second_review_modulus": modulus,
        "strata": stratum_manifest,
        "accepted_labels_created": False,
        "test_metadata_used_for_sampling": False,
    }
    return sample, vocabulary, manifest
