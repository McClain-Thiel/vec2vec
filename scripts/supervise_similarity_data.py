"""Rearm known technical graph stops, then run the complete data finalizer."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

RETRIEVAL_VERSION = "2026-08-04T09.02.10.007Z"
POPULATION_SHA256 = "7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5"
GRAPH_SESSION = "vec2vec-graph-autoresume-20260817"
BACKUP_PREFIX = (
    "s3://plasmidclip/research-backups/vec2vec/e00/global_similarity_graph_v0.1/"
    "2026-08-17T08-48-41Z"
)
REQUIRED_FREE_BYTES = 60_000_000_000
STABLE_SAMPLES = 10
SAMPLE_SECONDS = 60
SYNC_SECONDS = 300
MAXIMUM_TECHNICAL_RETRIES = 4


def classify_graph_log(text: str) -> str:
    """Classify one completed graph log without weakening failure policy."""
    if "Pipeline execution completed successfully" in text:
        return "success"
    technical_markers = (
        "global graph reached its fixed wall-time limit",
        "is below calibration minimum",
    )
    if any(marker in text for marker in technical_markers):
        return "retryable_technical_stop"
    return "non_retryable_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--current-session", default=GRAPH_SESSION)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    log_root = workspace / "data/09_scratch/similarity_graph_calibration"
    current_log = log_root / "global_similarity_graph_autoresume_20260817.log"
    supervisor_log = log_root / "similarity_data_supervisor_20260817.log"
    checkpoint_root = (
        log_root
        / POPULATION_SHA256
        / "queries/global_similarity_graph_v0.1/checkpoints/exact-cap10000"
    )

    _record(supervisor_log, f"waiting for active session {args.current_session}")
    while _tmux_session_exists(args.current_session):
        time.sleep(10)

    attempt = 0
    while True:
        if not current_log.exists():
            raise RuntimeError(f"graph log is missing: {current_log}")
        classification = classify_graph_log(current_log.read_text(encoding="utf-8"))
        archived_log = log_root / f"global_similarity_graph_attempt_{attempt:02d}_20260817.log"
        if not archived_log.exists():
            shutil.copy2(current_log, archived_log)
        _upload_log(archived_log, attempt)
        _record(supervisor_log, f"attempt={attempt} classification={classification}")
        if classification == "success":
            _record(supervisor_log, "starting full data finalizer")
            with (log_root / "finalize_similarity_data_20260817.log").open(
                "a", encoding="utf-8"
            ) as handle:
                subprocess.run(
                    ["./.venv/bin/python", "scripts/finalize_similarity_data.py"],
                    cwd=workspace,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            _record(supervisor_log, "full data finalizer completed successfully")
            return
        if classification != "retryable_technical_stop":
            raise RuntimeError("graph failed for a non-retryable reason; inspect the archived log")
        if attempt >= MAXIMUM_TECHNICAL_RETRIES:
            raise RuntimeError("graph crossed the fixed maximum technical retry count")

        _sync_checkpoints(checkpoint_root)
        _wait_for_stable_disk(workspace, supervisor_log)
        attempt += 1
        _record(supervisor_log, f"starting technical retry attempt={attempt}")
        with current_log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [
                    "./.venv/bin/kedro",
                    "run",
                    "--pipeline",
                    "similarity_graph",
                    "--load-versions",
                    f"retrieval_dataset@split_audit:{RETRIEVAL_VERSION}",
                ],
                cwd=workspace,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_sync = time.monotonic() + SYNC_SECONDS
            while process.poll() is None:
                if time.monotonic() >= next_sync:
                    _sync_checkpoints(checkpoint_root)
                    next_sync = time.monotonic() + SYNC_SECONDS
                time.sleep(10)
        _sync_checkpoints(checkpoint_root)


def _wait_for_stable_disk(workspace: Path, log_path: Path) -> None:
    consecutive = 0
    while consecutive < STABLE_SAMPLES:
        free = shutil.disk_usage(workspace).free
        consecutive = consecutive + 1 if free >= REQUIRED_FREE_BYTES else 0
        _record(
            log_path,
            f"free_bytes={free} required={REQUIRED_FREE_BYTES} "
            f"consecutive={consecutive}/{STABLE_SAMPLES}",
        )
        if consecutive < STABLE_SAMPLES:
            time.sleep(SAMPLE_SECONDS)


def _sync_checkpoints(checkpoint_root: Path) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(checkpoint_root),
            f"{BACKUP_PREFIX}/in-progress-checkpoints/exact-cap10000/",
            "--sse",
            "AES256",
            "--only-show-errors",
        ],
        check=True,
    )


def _upload_log(path: Path, attempt: int) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(path),
            f"{BACKUP_PREFIX}/run-logs/graph-attempt-{attempt:02d}.log",
            "--sse",
            "AES256",
            "--checksum-algorithm",
            "SHA256",
            "--no-progress",
        ],
        check=True,
    )


def _tmux_session_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _record(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


if __name__ == "__main__":
    main()
