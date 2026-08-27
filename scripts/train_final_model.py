"""Fit and validate the selected final vec2vec model."""

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import wandb

from vec2vec.lib import final_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--approval-reference")
    parser.add_argument("--region")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-hour-limit", type=float)
    parser.add_argument("--observed-instance-price-usd-per-hour", type=float)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _authorization(args):
    values = {
        "approval_reference": str(args.approval_reference or "").strip(),
        "region": str(args.region or "").strip(),
        "instance_type": str(args.instance_type or "").strip(),
        "instance_hour_limit": float(args.instance_hour_limit or math.nan),
        "observed_instance_price_usd_per_hour": float(
            args.observed_instance_price_usd_per_hour or math.nan
        ),
    }
    for name in ("approval_reference", "region", "instance_type"):
        if not values[name]:
            raise ValueError(f"--{name.replace('_', '-')} is required")
    for name in ("instance_hour_limit", "observed_instance_price_usd_per_hour"):
        if not math.isfinite(values[name]) or values[name] <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    maximum_cost = values["instance_hour_limit"] * values["observed_instance_price_usd_per_hour"]
    if maximum_cost >= 20.0:
        raise ValueError(f"final fit maximum cost must remain below $20, observed ${maximum_cost}")
    return {**values, "maximum_cost_usd": maximum_cost}


def _git_state():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("final fit must start from a clean Git checkout")
    return commit


def _runtime(device):
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("final fit requires CUDA")
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name(0),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "torch", "transformers", "wandb")
        },
    }


def main():
    args = _arguments()
    if args.validate_only:
        manifest = final_model.validate_bundle(args.output_dir)
        print(json.dumps({"status": "valid", "manifest": manifest}, sort_keys=True))
        return
    if args.source is None or args.states is None or args.queries is None:
        raise ValueError("--source, --states, and --queries are required for training")
    authorization = _authorization(args)
    git_commit = _git_state()
    runtime = _runtime(args.device)
    started = time.perf_counter()
    deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0
    training, queries, verified_mask, audit = final_model.load_final_inputs(
        args.source, args.states, args.queries
    )
    run = wandb.init(
        project="vec2vec",
        name="final-model-v1-seed-20260818",
        group="final-model-v1",
        tags=["final-fit", "verified-set", "population-scale"],
        config={
            **final_model.FINAL_RECIPE,
            "authorization": authorization,
            "input_audit": audit,
            "git_commit": git_commit,
        },
    )
    try:
        model, index, history, timings = final_model.fit_final_model(
            training,
            queries,
            verified_mask,
            device=args.device,
            deadline_monotonic=deadline,
        )
        elapsed = time.perf_counter() - started
        observed_cost = elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"]
        manifest = final_model.save_bundle(
            args.output_dir,
            model,
            index,
            training,
            queries,
            history,
            {
                "protocol_version": final_model.FINAL_RECIPE["protocol_version"],
                "recipe": final_model.FINAL_RECIPE,
                "input_audit": audit,
                "input_locations": {
                    "plasmids": (
                        "hf://buckets/McClain/plasmidclip-train-ckpts/"
                        "datasets/full158k-structured-v1/full158k_structured.parquet"
                    ),
                    "constraint_states": (
                        "s3://plasmidclip/kedro/05_model_input/e00/"
                        "plasmid_constraint_state.parquet/2026-08-06T13.27.47.937Z/"
                        "plasmid_constraint_state.parquet"
                    ),
                    "queries": (
                        "s3://plasmidclip/kedro/05_model_input/e06/queries.parquet/"
                        "2026-08-26T12.24.14.212Z/queries.parquet"
                    ),
                },
                "historical_splits_collapsed_for_final_fit": True,
                "evaluation_performed": False,
                "git_commit": git_commit,
                "git_dirty": False,
                "runtime": runtime,
                "timings_seconds": timings,
                "elapsed_seconds_before_persistence": elapsed,
                "observed_cost_usd_before_persistence": observed_cost,
                "authorization": authorization,
                "wandb": {"run_id": run.id, "url": run.url},
            },
        )
        for row in history.itertuples(index=False):
            run.log(
                {
                    "train/loss": float(row.loss),
                    "train/logit_scale": float(row.logit_scale),
                    "train/true_positive_pairs": int(row.true_positive_pairs),
                    "train/unique_candidate_rows": int(row.unique_candidate_rows),
                },
                step=int(row.update),
            )
        run.summary.update(
            {
                "final/training_rows": audit["eligible_rows"],
                "final/atomic_queries": audit["atomic_queries"],
                "final/final_loss": float(history.iloc[-1]["loss"]),
                "final/index_rows": int(len(index)),
                "final/observed_cost_usd_before_persistence": observed_cost,
                **{f"timing/{key}": value for key, value in timings.items()},
            }
        )
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    print(
        json.dumps(
            {
                "status": "final_model_complete",
                "output_dir": str(args.output_dir),
                "wandb": manifest["wandb"],
                "input_audit": audit,
                "files": manifest["files"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
