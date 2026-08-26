"""Composition checks for the single selected modeling-data DAG."""

from vec2vec.pipeline_registry import register_pipelines


def test_modeling_data_is_one_tagged_dependency_graph() -> None:
    pipelines = register_pipelines()
    modeling = pipelines["modeling_data"]
    names = {node.name for node in modeling.nodes}

    assert {
        "process_addgene_records",
        "import_published_descriptions",
        "add_splits_and_constraints",
        "build_constraint_states",
        "build_similarity_graph",
        "build_similarity_split",
        "build_query_benchmark",
        "build_model_panels",
        "fit_selected_dna_features",
        "extract_selected_text_features",
    } <= names
    assert {node.name for node in modeling.only_nodes_with_tags("tfidf").nodes} == {
        "fit_selected_dna_features"
    }
    assert {node.name for node in modeling.only_nodes_with_tags("qwen").nodes} == {
        "extract_selected_text_features"
    }


def test_paid_and_large_compute_are_not_in_default() -> None:
    default_names = {node.name for node in register_pipelines()["__default__"].nodes}
    assert "build_similarity_graph" not in default_names
    assert "extract_selected_text_features" not in default_names
