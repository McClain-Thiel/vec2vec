"""Tests for strict graph-supervisor failure classification."""

from scripts import supervise_similarity_data


def test_classifies_only_success_and_two_recorded_technical_stops():
    assert (
        supervise_similarity_data.classify_graph_log("Pipeline execution completed successfully")
        == "success"
    )
    assert (
        supervise_similarity_data.classify_graph_log(
            "RuntimeError: global graph reached its fixed wall-time limit"
        )
        == "retryable_technical_stop"
    )
    assert (
        supervise_similarity_data.classify_graph_log(
            "free disk 3999 is below calibration minimum 4000"
        )
        == "retryable_technical_stop"
    )


def test_classifies_success_wrapped_across_lines_by_the_rich_renderer():
    # Real Kedro console output, redirected to a file: Rich hard-wraps this
    # exact phrase mid-word against its column width, splitting the message
    # across two physical lines with a right-aligned "runner.py:119" gutter.
    wrapped = (
        "INFO     Pipeline execution completed          runner.py:119\n"
        "                             successfully in 396.9 sec.\n"
    )
    assert supervise_similarity_data.classify_graph_log(wrapped) == "success"


def test_does_not_retry_scientific_or_validation_failures():
    failures = (
        "candidate query saturated at adaptive cap",
        "shard checkpoint identity mismatch",
        "query reached its fixed task timeout",
        "graph manifest differs from accepted content hash",
        "unexpected parser failure",
        "",
    )
    assert all(
        supervise_similarity_data.classify_graph_log(value) == "non_retryable_failure"
        for value in failures
    )
