"""Validated exact metadata mappings shared by constraint data products.

The mappings support narrow claims about recorded Addgene fields. They do not
establish biological function, and values outside enabled exact mappings remain
unknown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vec2vec.lib.serialization import stable_json
from vec2vec.lib.text import as_list, exact_metadata_key, sha256_text

SOURCE_FIELDS = ("plasmid_copy", "growth_temp", "bacterial_resistance", "vector_types")

_ALLOWED_SECTIONS = {
    "copy_class": {"included"},
    "growth_temperature": {"included", "reviewed_mappings"},
    "bacterial_selection": {"included", "reviewed_mappings"},
    "intended_use": {"expression_included", "use_included"},
}


@dataclass(frozen=True)
class ExactMapping:
    """One configured whole-value metadata mapping."""

    source_field: str
    raw_value: str
    canonical_values: tuple[str, ...]
    rule_id: str
    facet: str
    relation: str
    section: str
    mapping_note: str | None


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


def build_mapping_contract(
    facet_params: Mapping[str, Any],
    enabled_sections: Mapping[str, Sequence[str]],
) -> tuple[dict[tuple[str, str], ExactMapping], str]:
    """Load enabled mappings and return their content identity."""
    if set(enabled_sections) != set(_ALLOWED_SECTIONS):
        raise ValueError("enabled_sections must name all supported rule groups")
    for rule_name, sections in enabled_sections.items():
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
    for rule_name in sorted(enabled_sections):
        rule = facet_params[rule_name]
        for section in sorted(enabled_sections[rule_name]):
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


def source_values(record: Any, field: str) -> list[Any]:
    """Return source units using the field's stored cardinality."""
    if field == "vector_types":
        return as_list(getattr(record, field))
    return [getattr(record, field)]
