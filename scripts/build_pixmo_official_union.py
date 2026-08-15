#!/usr/bin/env python3
"""Build the largest SHA-verified public PixMo-Points evaluation snapshot available."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset, load_from_disk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--previous-manifest", required=True)
    parser.add_argument("--official-download", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    raw = Dataset.from_parquet(args.parquet)
    raw = raw.add_column("source_index", list(range(len(raw))))

    previous = json.loads(Path(args.previous_manifest).read_text(encoding="utf-8"))
    image_by_url = {
        row["image_url"]: row["image_path"]
        for row in previous
        if row.get("status") == "ready" and _valid_file(row)
    }
    downloaded = load_from_disk(args.official_download)
    image_by_url.update(
        {row["image_url"]: row["image"] for row in downloaded if Path(row["image"]).is_file()}
    )

    available = raw.filter(lambda url: url in image_by_url, input_columns=["image_url"])
    available = available.add_column(
        "image", [image_by_url[url] for url in available["image_url"]]
    )
    available.save_to_disk(str(output))
    source_indices = available["source_index"]
    print(
        f"Saved {len(available)}/{len(raw)} rows to {output}; "
        f"source-index range={min(source_indices)}..{max(source_indices)}"
    )
    return 0


def _valid_file(row: dict) -> bool:
    path = Path(row["image_path"])
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == row["image_sha256"]


if __name__ == "__main__":
    raise SystemExit(main())
