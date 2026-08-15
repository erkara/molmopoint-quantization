"""Download and validate the public PixMo pointing evaluation images."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    index: int
    status: str
    image_path: str | None
    image_sha256: str
    image_url: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pixmo_dataset(
    dataset_id: str = "allenai/pixmo-points-eval",
    split: str = "test",
    dataset_arrow: str | Path | None = None,
):
    """Load from the Hub, with a deterministic fallback to its Arrow cache."""
    from datasets import Dataset, load_dataset

    if dataset_arrow is not None:
        return Dataset.from_file(str(Path(dataset_arrow).expanduser().resolve()))

    cached = find_cached_arrow(dataset_id, split)
    if cached is not None:
        return Dataset.from_file(str(cached))

    try:
        return load_dataset(dataset_id, split=split)
    except Exception as hub_error:
        cached = find_cached_arrow(dataset_id, split)
        if cached is None:
            raise RuntimeError(
                f"Could not load {dataset_id}:{split} from the Hub and no cached Arrow file exists"
            ) from hub_error
        return Dataset.from_file(str(cached))


def find_cached_arrow(dataset_id: str, split: str) -> Path | None:
    from datasets import config

    dataset_cache_name = dataset_id.replace("/", "___")
    candidates = sorted(
        Path(config.HF_DATASETS_CACHE).glob(
            f"{dataset_cache_name}/**/{dataset_id.split('/')[-1]}-{split}.arrow"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def prepare_image_cache(
    dataset,
    *,
    cache_dir: str | Path,
    max_examples: int | None = None,
    workers: int = 8,
    timeout_seconds: float = 30.0,
    retries: int = 2,
) -> list[ImageRecord]:
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), max_examples) if max_examples is not None else len(dataset)
    inputs = [
        (index, str(dataset[index]["image_url"]), str(dataset[index]["image_sha256"]))
        for index in range(count)
    ]

    records: list[ImageRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                fetch_verified_image,
                index=index,
                url=url,
                expected_sha256=sha256,
                cache_dir=cache_root,
                timeout_seconds=timeout_seconds,
                retries=retries,
            ): index
            for index, url, sha256 in inputs
        }
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda record: record.index)


def fetch_verified_image(
    *,
    index: int,
    url: str,
    expected_sha256: str,
    cache_dir: str | Path,
    timeout_seconds: float = 30.0,
    retries: int = 2,
) -> ImageRecord:
    cache_root = Path(cache_dir)
    destination = cache_root / expected_sha256[:2] / f"{expected_sha256}.img"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            _validate_image_file(destination, expected_sha256)
            return ImageRecord(index, "ready", str(destination), expected_sha256, url)
        except Exception:
            destination.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Molmo-Quantization-Eval/0.1"},
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
                )
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(payload)
            _validate_image_file(temporary, expected_sha256)
            temporary.replace(destination)
            return ImageRecord(index, "ready", str(destination), expected_sha256, url)
        except ValueError as error:
            # Changed bytes cannot become valid by retrying the same URL.
            last_error = error
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if 400 <= error.code < 500:
                break
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

    return ImageRecord(
        index=index,
        status="unavailable",
        image_path=None,
        image_sha256=expected_sha256,
        image_url=url,
        error=f"{type(last_error).__name__}: {last_error}",
    )


def write_manifest(path: str | Path, records: Iterable[ImageRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in records]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_manifest(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(record["index"]): record for record in payload}


def _validate_image_file(path: Path, expected_sha256: str) -> None:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Cached image SHA256 mismatch for {path}")
    with Image.open(path) as image:
        image.verify()
