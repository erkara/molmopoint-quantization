"""Download and SHA256-validate PixMo Points Eval images."""

from __future__ import annotations

import argparse
from collections import Counter

from molmo_quant.pixmo_dataset import load_pixmo_dataset, prepare_image_cache, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="allenai/pixmo-points-eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-arrow")
    parser.add_argument("--cache-dir", default="data/pixmo-points-eval/images")
    parser.add_argument("--manifest", default="data/pixmo-points-eval/manifest.json")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_pixmo_dataset(args.dataset, args.split, args.dataset_arrow)
    records = prepare_image_cache(
        dataset,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    write_manifest(args.manifest, records)
    counts = Counter(record.status for record in records)
    print(f"[prepare] Dataset rows: {len(dataset)}")
    print(f"[prepare] Manifest rows: {len(records)}; statuses: {dict(counts)}")
    print(f"[prepare] Wrote {args.manifest}")
    return 0 if counts.get("ready", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
