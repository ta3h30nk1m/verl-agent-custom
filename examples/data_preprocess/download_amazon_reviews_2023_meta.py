#!/usr/bin/env python3
"""
Download Amazon Reviews 2023 item metadata files by category.

This downloads only the metadata JSONL.GZ files, not review files. The files are
large, so the default category list is intentionally small and shopping-task
oriented.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/amazon_reviews_2023_meta"
BASE_META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"

CATEGORIES = (
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
)

DEFAULT_CATEGORIES = (
    "Musical_Instruments",
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
)


def meta_url(category: str) -> str:
    return f"{BASE_META_URL}/meta_{category}.jsonl.gz"


def output_path(output_dir: Path, category: str) -> Path:
    return output_dir / f"meta_{category}.jsonl.gz"


def human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def validate_categories(categories: list[str]) -> None:
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        raise ValueError(
            "Unknown category/categories: "
            + ", ".join(unknown)
            + "\nUse --list-categories to see valid names."
        )


def first_json_record(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"First JSONL record is not an object in {path}")
                return record
    raise ValueError(f"No JSONL records found in {path}")


def verify_metadata_file(path: Path) -> dict[str, Any]:
    record = first_json_record(path)
    required = {"title", "parent_asin"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{path} does not look like Amazon Reviews 2023 metadata; missing fields: {missing}")
    return {
        "parent_asin": record.get("parent_asin"),
        "title": record.get("title"),
        "main_category": record.get("main_category"),
        "has_details": isinstance(record.get("details"), dict),
    }


def download_one(
    *,
    category: str,
    url: str,
    path: Path,
    overwrite: bool,
    timeout: int,
    retries: int,
    chunk_size: int,
    verify: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        result: dict[str, Any] = {
            "category": category,
            "url": url,
            "path": str(path),
            "status": "exists",
            "bytes": path.stat().st_size,
        }
        if verify:
            result["sample"] = verify_metadata_file(path)
        return result

    if dry_run:
        return {
            "category": category,
            "url": url,
            "path": str(path),
            "status": "dry_run",
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        bytes_written = 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "verl-agent-custom-data-prep/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
                total_header = response.headers.get("Content-Length")
                total_bytes = int(total_header) if total_header and total_header.isdigit() else None
                print(f"downloading category={category} size={human_bytes(total_bytes)} output={path}")
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
            tmp_path.replace(path)
            result = {
                "category": category,
                "url": url,
                "path": str(path),
                "status": "downloaded",
                "bytes": bytes_written,
            }
            if verify:
                result["sample"] = verify_metadata_file(path)
            return result
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt <= retries:
                wait_seconds = min(30, 2 ** (attempt - 1))
                print(f"retrying category={category} attempt={attempt}/{retries + 1} wait={wait_seconds}s error={exc}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to download {category} from {url}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 item metadata JSONL.GZ files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--category",
        dest="category_list",
        action="append",
        choices=list(CATEGORIES),
        help="Category to download. Can be repeated. Defaults to a small shopping-oriented set.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORIES),
        help="Categories to download. Alternative to repeated --category.",
    )
    parser.add_argument("--all-categories", action="store_true", help="Download all metadata categories.")
    parser.add_argument("--list-categories", action="store_true", help="Print valid category names and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload even when the output file exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without network access.")
    parser.add_argument("--no-verify", action="store_true", help="Skip gzip/JSON metadata verification after download.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    return parser.parse_args()


def choose_categories(args: argparse.Namespace) -> list[str]:
    if args.all_categories:
        return list(CATEGORIES)
    chosen: list[str] = []
    if args.category_list:
        chosen.extend(args.category_list)
    if args.categories:
        chosen.extend(args.categories)
    if not chosen:
        chosen = list(DEFAULT_CATEGORIES)
    deduped = list(dict.fromkeys(chosen))
    validate_categories(deduped)
    return deduped


def main() -> None:
    args = parse_args()
    if args.list_categories:
        print("\n".join(CATEGORIES))
        return

    categories = choose_categories(args)
    manifest = {
        "source": "Amazon Reviews 2023",
        "base_meta_url": BASE_META_URL,
        "output_dir": str(args.output_dir),
        "categories": categories,
        "files": [],
    }

    for category in categories:
        result = download_one(
            category=category,
            url=meta_url(category),
            path=output_path(args.output_dir, category),
            overwrite=args.overwrite,
            timeout=args.timeout,
            retries=args.retries,
            chunk_size=args.chunk_size_mb * 1024 * 1024,
            verify=not args.no_verify and not args.dry_run,
            dry_run=args.dry_run,
        )
        manifest["files"].append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "metadata_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
