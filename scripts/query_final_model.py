"""Retrieve plasmids from the accepted final vec2vec bundle."""

import argparse
import json
from pathlib import Path

from vec2vec.lib import final_model


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = _arguments()
    vectors, token_counts = final_model.encode_queries(
        args.bundle,
        args.query,
        device=args.device,
    )
    results = final_model.retrieve(args.bundle, vectors, top_k=args.top_k)
    records = results.to_dict(orient="records")
    for record in records:
        query_index = int(record.pop("query_index"))
        record["query"] = args.query[query_index]
        record["query_tokens"] = token_counts[query_index]
        print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
