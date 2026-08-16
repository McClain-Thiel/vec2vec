"""Sparse verified and contradicted states for the frozen constraint contract.

Unknown states are represented by absence. A contradicted row can be created
only by an explicit configured conflict rule.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from vec2vec.lib.constraint_rules import (
    SOURCE_FIELDS,
    ExactMapping,
    build_mapping_contract,
    source_values,
)
from vec2vec.lib.serialization import stable_json
from vec2vec.lib.text import exact_metadata_key, sha256_text

VERIFIED = "verified"
CONTRADICTED = "contradicted"

ConstraintKey = tuple[str, str, str, str]

_REQUIRED_RETRIEVAL_COLUMNS = {
    "sequence_id",
    "sequence_sha256",
    "leakage_component",
    "split_grouped",
    *SOURCE_FIELDS,
}


def _constraint_key(mapping: ExactMapping, canonical_value: str) -> ConstraintKey:
    return mapping.facet, mapping.relation, canonical_value, mapping.rule_id


def _constraint_id(key: ConstraintKey) -> str:
    facet, relation, canonical_value, rule_id = key
    identity = {
        "canonical_value": canonical_value,
        "facet": facet,
        "relation": relation,
        "rule_id": rule_id,
    }
    return sha256_text(stable_json(identity))


def _rule_version(rule_id: str) -> str:
    version = rule_id.rpartition(".")[2]
    if not version.startswith("v") or len(version) == 1:
        raise ValueError(f"rule_id must end in a version such as '.v0_1': {rule_id!r}")
    return version


def _validate_retrieval(retrieval: pd.DataFrame, allowed_splits: Sequence[str]) -> None:
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
    if not allowed_splits or len(set(allowed_splits)) != len(allowed_splits):
        raise ValueError("allowed_splits must be non-empty and unique")
    unexpected = set(retrieval["split_grouped"].astype(str)).difference(allowed_splits)
    if unexpected:
        raise ValueError(f"retrieval dataset contains unexpected splits: {sorted(unexpected)}")


def _constraint_keys(
    mappings: Mapping[tuple[str, str], ExactMapping],
) -> set[ConstraintKey]:
    return {
        _constraint_key(mapping, canonical_value)
        for mapping in mappings.values()
        for canonical_value in mapping.canonical_values
    }


def _load_conflicts(
    specifications: Sequence[Mapping[str, Any]],
    constraint_keys: set[ConstraintKey],
) -> tuple[
    dict[ConstraintKey, dict[ConstraintKey, str]],
    dict[ConstraintKey, tuple[str, str]],
]:
    conflicts: dict[ConstraintKey, dict[ConstraintKey, str]] = defaultdict(dict)
    membership: dict[ConstraintKey, tuple[str, str]] = {}
    required = {"name", "conflict_rule_id", "facet", "relation", "rule_id", "values"}
    names: set[str] = set()
    conflict_rule_ids: set[str] = set()

    for specification in specifications:
        if set(specification) != required:
            raise ValueError(f"conflict group must contain exactly: {sorted(required)}")
        name = str(specification["name"])
        conflict_rule_id = str(specification["conflict_rule_id"])
        _rule_version(conflict_rule_id)
        if not name or name in names:
            raise ValueError(f"conflict group name is empty or duplicated: {name!r}")
        if conflict_rule_id in conflict_rule_ids:
            raise ValueError(f"conflict_rule_id is duplicated: {conflict_rule_id!r}")
        names.add(name)
        conflict_rule_ids.add(conflict_rule_id)

        values = tuple(str(value) for value in specification["values"])
        if len(values) < 2 or len(values) != len(set(values)):
            raise ValueError(f"conflict group {name!r} needs at least two unique values")
        keys = tuple(
            (
                str(specification["facet"]),
                str(specification["relation"]),
                value,
                str(specification["rule_id"]),
            )
            for value in values
        )
        missing = set(keys).difference(constraint_keys)
        if missing:
            raise ValueError(
                f"conflict group {name!r} names unmapped constraints: {sorted(missing)}"
            )
        overlap = set(keys).intersection(membership)
        if overlap:
            raise ValueError(f"constraints occur in multiple conflict groups: {sorted(overlap)}")

        for key in keys:
            membership[key] = (name, conflict_rule_id)
            for other in keys:
                if other != key:
                    conflicts[key][other] = conflict_rule_id
    return dict(conflicts), membership


def _base_vocabulary(
    constraint_keys: set[ConstraintKey],
    membership: Mapping[ConstraintKey, tuple[str, str]],
    state_version: str,
    contract_hash: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key in sorted(constraint_keys):
        facet, relation, canonical_value, rule_id = key
        conflict_group, conflict_rule_id = membership.get(key, (None, None))
        records.append(
            {
                "state_version": state_version,
                "rule_contract_sha256": contract_hash,
                "constraint_id": _constraint_id(key),
                "facet": facet,
                "relation": relation,
                "canonical_value": canonical_value,
                "rule_id": rule_id,
                "rule_version": _rule_version(rule_id),
                "conflict_group": conflict_group,
                "conflict_rule_id": conflict_rule_id,
                "has_reviewed_conflict_rule": conflict_group is not None,
            }
        )
    vocabulary = pd.DataFrame.from_records(records)
    if vocabulary.empty or vocabulary["constraint_id"].duplicated().any():
        raise RuntimeError("constraint vocabulary is empty or has duplicate identifiers")
    return vocabulary


def _add_evidence(
    evidence: dict[tuple[str, str, str], dict[str, str]],
    *,
    sequence_id: str,
    constraint_id: str,
    state: str,
    record: Mapping[str, Any],
) -> None:
    key = (sequence_id, constraint_id, state)
    serialized = stable_json(record)
    evidence[key][serialized] = serialized


def _state_rows(
    retrieval: pd.DataFrame,
    mappings: Mapping[tuple[str, str], ExactMapping],
    conflicts: Mapping[ConstraintKey, Mapping[ConstraintKey, str]],
    vocabulary: pd.DataFrame,
    state_version: str,
    contract_hash: str,
) -> pd.DataFrame:
    identity_to_id = {
        (row.facet, row.relation, row.canonical_value, row.rule_id): row.constraint_id
        for row in vocabulary.itertuples(index=False)
    }
    constraint_details = vocabulary.set_index("constraint_id").to_dict("index")
    evidence: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    row_identity: dict[str, dict[str, str]] = {}

    ordered = retrieval.sort_values("sequence_id", kind="stable")
    for row in ordered.itertuples(index=False):
        sequence_id = str(row.sequence_id)
        row_identity[sequence_id] = {
            "sequence_sha256": str(row.sequence_sha256),
            "leakage_component": str(row.leakage_component),
            "split_grouped": str(row.split_grouped),
        }
        for field in SOURCE_FIELDS:
            for raw_value in source_values(row, field):
                raw_key = exact_metadata_key(raw_value)
                if raw_key is None:
                    continue
                mapping = mappings.get((field, raw_key))
                if mapping is None:
                    continue
                source_value_json = stable_json(raw_value)
                for canonical_value in mapping.canonical_values:
                    observed_key = _constraint_key(mapping, canonical_value)
                    observed_id = identity_to_id[observed_key]
                    common = {
                        "mapping_section": mapping.section,
                        "mapping_note": mapping.mapping_note,
                        "observed_constraint_id": observed_id,
                        "source_field": field,
                        "source_rule_id": mapping.rule_id,
                        "source_value_json": source_value_json,
                    }
                    _add_evidence(
                        evidence,
                        sequence_id=sequence_id,
                        constraint_id=observed_id,
                        state=VERIFIED,
                        record={
                            **common,
                            "evidence_rule_id": mapping.rule_id,
                            "evidence_type": "exact_metadata_mapping",
                        },
                    )
                    for contradicted_key, conflict_rule_id in conflicts.get(
                        observed_key, {}
                    ).items():
                        _add_evidence(
                            evidence,
                            sequence_id=sequence_id,
                            constraint_id=identity_to_id[contradicted_key],
                            state=CONTRADICTED,
                            record={
                                **common,
                                "evidence_rule_id": conflict_rule_id,
                                "evidence_type": "reviewed_conflict_rule",
                            },
                        )

    records: list[dict[str, Any]] = []
    for (sequence_id, constraint_id, state), serialized_evidence in sorted(evidence.items()):
        details = constraint_details[constraint_id]
        sources = [serialized_evidence[key] for key in sorted(serialized_evidence)]
        records.append(
            {
                "state_id": sha256_text(f"{state_version}|{sequence_id}|{constraint_id}|{state}"),
                "state_version": state_version,
                "rule_contract_sha256": contract_hash,
                "sequence_id": sequence_id,
                **row_identity[sequence_id],
                "constraint_id": constraint_id,
                "facet": details["facet"],
                "relation": details["relation"],
                "canonical_value": details["canonical_value"],
                "rule_id": details["rule_id"],
                "state": state,
                "evidence_count": len(sources),
                "evidence_json": f"[{','.join(sources)}]",
            }
        )
    states = pd.DataFrame.from_records(records)
    if states.empty:
        raise ValueError("enabled mappings produced no constraint states")
    if states["state_id"].duplicated().any():
        raise RuntimeError("constraint state identifiers are not unique")
    pair_states = states.groupby(["sequence_id", "constraint_id"], sort=False)["state"].nunique()
    conflicts_observed = pair_states[pair_states > 1]
    if not conflicts_observed.empty:
        examples = [list(index) for index in conflicts_observed.index[:5]]
        raise ValueError(
            f"plasmid-constraint pairs have both verified and contradicted evidence: {examples}"
        )
    return states.sort_values(
        ["sequence_id", "facet", "canonical_value", "state"], kind="stable"
    ).reset_index(drop=True)


def _attach_support(
    vocabulary: pd.DataFrame,
    states: pd.DataFrame,
    allowed_splits: Sequence[str],
    training_split: str,
) -> pd.DataFrame:
    if training_split not in allowed_splits:
        raise ValueError("training_split must occur in allowed_splits")
    result = vocabulary.copy()
    for split in (*allowed_splits, "total"):
        population = states if split == "total" else states.loc[states["split_grouped"].eq(split)]
        for state in (VERIFIED, CONTRADICTED):
            selected = population.loc[population["state"].eq(state)]
            rows = selected.groupby("constraint_id")["sequence_id"].nunique()
            components = selected.groupby("constraint_id")["leakage_component"].nunique()
            prefix = f"{split}_{state}"
            result[f"{prefix}_row_support"] = (
                result["constraint_id"].map(rows).fillna(0).astype("int64")
            )
            result[f"{prefix}_component_support"] = (
                result["constraint_id"].map(components).fillna(0).astype("int64")
            )
    result["train_row_support"] = result[f"{training_split}_{VERIFIED}_row_support"]
    result["train_component_support"] = result[f"{training_split}_{VERIFIED}_component_support"]
    return result.sort_values(
        ["facet", "canonical_value", "constraint_id"], kind="stable"
    ).reset_index(drop=True)


def retrieval_population_sha256(retrieval: pd.DataFrame) -> str:
    """Return the content identity of the retrieval rows used by this table."""
    columns = ["sequence_id", "sequence_sha256", "leakage_component", "split_grouped"]
    records = retrieval[columns].astype(str).sort_values("sequence_id", kind="stable")
    return sha256_text(
        "\n".join("|".join(row) for row in records.itertuples(index=False, name=None))
    )


def retrieval_state_input_sha256(retrieval: pd.DataFrame) -> str:
    """Return the content identity of identities and constraint source fields."""
    columns = [
        "sequence_id",
        "sequence_sha256",
        "leakage_component",
        "split_grouped",
        *SOURCE_FIELDS,
    ]
    ordered = retrieval[columns].sort_values("sequence_id", kind="stable")
    digest = hashlib.sha256()
    for record in ordered.to_dict("records"):
        digest.update(stable_json(record).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_constraint_state_tables(
    retrieval: pd.DataFrame,
    state_params: Mapping[str, Any],
    evidence_params: Mapping[str, Any],
    facet_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the stable vocabulary and sparse three-state evidence product."""
    state_version = str(state_params["state_version"])
    allowed_splits = tuple(str(split) for split in state_params["allowed_splits"])
    training_split = str(state_params["training_split"])
    _validate_retrieval(retrieval, allowed_splits)
    population_hash = retrieval_population_sha256(retrieval)
    expected_population_hash = str(state_params["expected_input_population_sha256"])
    if population_hash != expected_population_hash:
        raise ValueError(
            "retrieval population changed: "
            f"expected {expected_population_hash}, observed {population_hash}"
        )
    state_input_hash = retrieval_state_input_sha256(retrieval)
    expected_state_input_hash = str(state_params["expected_state_input_sha256"])
    if state_input_hash != expected_state_input_hash:
        raise ValueError(
            "constraint source data changed: "
            f"expected {expected_state_input_hash}, observed {state_input_hash}"
        )

    mappings, contract_hash = build_mapping_contract(
        facet_params, evidence_params["enabled_sections"]
    )
    expected_contract = str(state_params["expected_rule_contract_sha256"])
    if contract_hash != expected_contract:
        raise ValueError(
            "accepted rule contract changed: "
            f"expected {expected_contract}, observed {contract_hash}"
        )
    keys = _constraint_keys(mappings)
    conflicts, membership = _load_conflicts(state_params["conflict_groups"], keys)
    vocabulary = _base_vocabulary(keys, membership, state_version, contract_hash)
    states = _state_rows(
        retrieval,
        mappings,
        conflicts,
        vocabulary,
        state_version,
        contract_hash,
    )
    vocabulary = _attach_support(vocabulary, states, allowed_splits, training_split)

    state_counts = states.groupby(["split_grouped", "state"], sort=True).size()
    facet_counts = states.groupby(["facet", "state"], sort=True).size()
    manifest = {
        "state_version": state_version,
        "input_retrieval_version": str(state_params["input_retrieval_version"]),
        "input_population_sha256": population_hash,
        "state_input_sha256": state_input_hash,
        "rule_contract_sha256": contract_hash,
        "input_rows_by_split": {
            str(split): int(count)
            for split, count in retrieval["split_grouped"].value_counts().sort_index().items()
        },
        "constraint_count": int(len(vocabulary)),
        "constraints_with_reviewed_conflicts": int(vocabulary["has_reviewed_conflict_rule"].sum()),
        "conflict_group_count": int(len(state_params["conflict_groups"])),
        "state_rows": int(len(states)),
        "state_rows_by_split_and_state": {
            f"{split}|{state}": int(count) for (split, state), count in state_counts.items()
        },
        "state_rows_by_facet_and_state": {
            f"{facet}|{state}": int(count) for (facet, state), count in facet_counts.items()
        },
        "sequences_with_verified_evidence": int(
            states.loc[states["state"].eq(VERIFIED), "sequence_id"].nunique()
        ),
        "sequences_with_contradicted_evidence": int(
            states.loc[states["state"].eq(CONTRADICTED), "sequence_id"].nunique()
        ),
        "unknown_policy": "absence_from_sparse_table",
        "pair_state_conflicts": 0,
        "stable_content_identifiers": True,
        "test_metadata_used_for_rule_selection": False,
        "test_states_derived_with_frozen_rules": "test" in set(allowed_splits),
        "generated_descriptions_used": False,
        "annotation_evidence_used": False,
    }
    return vocabulary, states, manifest
