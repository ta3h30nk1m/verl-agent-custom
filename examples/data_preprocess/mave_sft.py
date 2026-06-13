#!/usr/bin/env python3
"""
Build SFT datasets from MAVE-style product attribute extraction data.

Expected positive input is the full MAVE JSONL after joining labels with cleaned
Amazon product metadata:
  {"id": ..., "category": ..., "paragraphs": [...], "attributes": [...]}

Negative input is optional but strongly recommended. It may be either the full
MAVE negative JSONL, or a label-only JSONL as long as each row contains a product
id and an attribute name. Null targets are generated only from this negative
input, not from unlabeled attributes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data/mave_raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/mave_sft"
DEFAULT_TASKS = (
    "single_attribute_qa",
    "evidence_grounded_extraction",
    "multi_attribute_card_completion",
    "product_customer_qa",
    "faceted_search_filtering",
)
MAVE_LABEL_URLS = {
    "positive": "https://media.githubusercontent.com/media/google-research-datasets/MAVE/main/labels/mave_positives_labels.jsonl",
    "negative": "https://media.githubusercontent.com/media/google-research-datasets/MAVE/main/labels/mave_negatives_labels.jsonl",
}
MAVE_LABEL_FILENAMES = {
    "positive": "mave_positives_labels.jsonl",
    "negative": "mave_negatives_labels.jsonl",
}
DEFAULT_AMAZON23_META_DIR = REPO_ROOT / "data/amazon_reviews_2023_meta"
DEFAULT_AMAZON23_FULL_POSITIVE = DEFAULT_RAW_DIR / "mave_positives_full_amazon23.jsonl"


@dataclass(frozen=True)
class Paragraph:
    pid: int
    source: str
    text: str


@dataclass(frozen=True)
class Evidence:
    value: str
    pid: int
    begin: int | None
    end: int | None


@dataclass(frozen=True)
class Attribute:
    key: str
    evidences: tuple[Evidence, ...]

    @property
    def canonical_value(self) -> str:
        counts = Counter(evidence.value for evidence in self.evidences if evidence.value)
        if counts:
            return counts.most_common(1)[0][0]
        return ""


@dataclass(frozen=True)
class MAVEProduct:
    product_id: str
    category: str
    paragraphs: tuple[Paragraph, ...]
    attributes: tuple[Attribute, ...]


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}") from exc
            if isinstance(item, dict):
                yield item


def download_file(url: str, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        print(f"using_existing={path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    print(f"downloading={url} output={path}")
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to download {url}. If this environment has no network access, "
            "download the MAVE label files manually or clone the MAVE repo with git-lfs."
        ) from exc
    tmp_path.replace(path)


def download_mave_labels(raw_dir: Path, *, overwrite: bool = False) -> dict[str, Path]:
    labels_dir = raw_dir / "labels"
    paths = {
        kind: labels_dir / filename
        for kind, filename in MAVE_LABEL_FILENAMES.items()
    }
    for kind, path in paths.items():
        download_file(MAVE_LABEL_URLS[kind], path, overwrite=overwrite)
    return paths


def default_positive_candidates(raw_dir: Path) -> list[Path]:
    return [
        raw_dir / "reproduce/mave_positives_full.jsonl",
        raw_dir / "reproduce/mave_positives_full.jsonl.gz",
        raw_dir / "mave_positives_full.jsonl",
        raw_dir / "mave_positives_full.jsonl.gz",
        raw_dir / "full_mave_positives.jsonl",
        raw_dir / "full_mave_positives.jsonl.gz",
        raw_dir / "mave_positives.jsonl",
        raw_dir / "mave_positives.jsonl.gz",
    ]


def find_default_positive_jsonl(raw_dir: Path) -> Path | None:
    for path in default_positive_candidates(raw_dir):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def has_full_product_profiles(path: Path, *, max_records: int = 20) -> bool:
    checked = 0
    for record in iter_jsonl(path):
        checked += 1
        if normalize_paragraphs(record) and any(attr.evidences for attr in normalize_attributes(record)):
            return True
        if checked >= max_records:
            break
    return False


def default_label_path(raw_dir: Path, kind: str) -> Path:
    return raw_dir / "labels" / MAVE_LABEL_FILENAMES[kind]


def find_default_label_jsonl(raw_dir: Path, kind: str) -> Path | None:
    path = default_label_path(raw_dir, kind)
    if path.exists() and path.stat().st_size > 0:
        return path
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists() and gz_path.stat().st_size > 0:
        return gz_path
    return None


def amazon23_meta_files(args: argparse.Namespace) -> list[Path]:
    files = []
    if args.amazon23_meta_jsonl:
        files.extend(args.amazon23_meta_jsonl)
    if args.amazon23_meta_dir:
        files.extend(sorted(args.amazon23_meta_dir.glob("meta_*.jsonl.gz")))
        files.extend(sorted(args.amazon23_meta_dir.glob("meta_*.jsonl")))
    deduped = list(dict.fromkeys(files))
    missing = [path for path in deduped if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Amazon 2023 metadata file(s): " + ", ".join(str(path) for path in missing[:10]))
    return deduped


def is_amazon23_join_requested(args: argparse.Namespace) -> bool:
    return bool(args.amazon23_meta_jsonl or args.amazon23_meta_dir)


def ensure_input_files(args: argparse.Namespace) -> None:
    downloaded_labels = {}
    if args.download_mave_labels or args.download_only:
        downloaded_labels = download_mave_labels(args.raw_dir, overwrite=args.overwrite_downloads)
        if args.negative_jsonl is None:
            args.negative_jsonl = downloaded_labels["negative"]

    if args.positive_label_jsonl is None:
        args.positive_label_jsonl = find_default_label_jsonl(args.raw_dir, "positive")
    if args.negative_jsonl is None:
        args.negative_jsonl = find_default_label_jsonl(args.raw_dir, "negative")

    if is_amazon23_join_requested(args):
        positive_jsonl_is_full = (
            args.positive_jsonl is not None
            and args.positive_jsonl.exists()
            and has_full_product_profiles(args.positive_jsonl)
        )
        if not positive_jsonl_is_full and args.positive_label_jsonl is None and args.positive_jsonl is not None and args.positive_jsonl.exists():
            args.positive_label_jsonl = args.positive_jsonl
        if not positive_jsonl_is_full:
            if args.positive_label_jsonl is None:
                if not downloaded_labels:
                    downloaded_labels = download_mave_labels(args.raw_dir, overwrite=args.overwrite_downloads)
                args.positive_label_jsonl = downloaded_labels["positive"]
                if args.negative_jsonl is None:
                    args.negative_jsonl = downloaded_labels["negative"]
            args.positive_jsonl = build_amazon23_full_positive_jsonl(args)

    if args.positive_jsonl is None:
        args.positive_jsonl = find_default_positive_jsonl(args.raw_dir)

    if args.download_only:
        if not downloaded_labels:
            downloaded_labels = download_mave_labels(args.raw_dir, overwrite=args.overwrite_downloads)
        print(
            json.dumps(
                {
                    "downloaded_labels": {key: str(path) for key, path in downloaded_labels.items()},
                    "note": (
                        "These official files are label-only. To build SFT examples with product profiles, "
                        "join them with cleaned Amazon metadata and pass the resulting full positive JSONL "
                        "as --positive-jsonl."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.positive_jsonl is None:
        raise RuntimeError(
            "Missing --positive-jsonl. The official MAVE GitHub download provides label files, "
            "but this SFT builder needs a full positive JSONL containing product paragraphs. "
            "Create the full MAVE file by joining MAVE labels with cleaned Amazon metadata, "
            "then rerun with --positive-jsonl /path/to/full_mave_positives.jsonl. "
            "Use --download-mave-labels --download-only to fetch the official labels first."
        )

    if not args.positive_jsonl.exists():
        raise FileNotFoundError(f"--positive-jsonl does not exist: {args.positive_jsonl}")
    if args.negative_jsonl is not None and not args.negative_jsonl.exists():
        raise FileNotFoundError(f"--negative-jsonl does not exist: {args.negative_jsonl}")
    if not has_full_product_profiles(args.positive_jsonl):
        raise RuntimeError(
            f"{args.positive_jsonl} does not look like the full MAVE positive JSONL because it "
            "does not contain product paragraphs with evidence-bearing attributes. The official "
            "mave_positives_labels.jsonl file is label-only; it is useful for the MAVE join step, "
            "but it is not enough to create these SFT prompts."
        )


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def normalize_paragraphs(record: dict[str, Any]) -> tuple[Paragraph, ...]:
    raw_paragraphs = record.get("paragraphs") or record.get("sources") or []
    paragraphs: list[Paragraph] = []
    if not isinstance(raw_paragraphs, list):
        return tuple()

    for idx, raw in enumerate(raw_paragraphs):
        if not isinstance(raw, dict):
            continue
        text = clean_space(raw.get("text"))
        if not text:
            continue
        source = clean_space(raw.get("source") or raw.get("name") or raw.get("type") or f"source_{idx}")
        raw_pid = raw.get("pid", raw.get("id", idx))
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            pid = idx
        paragraphs.append(Paragraph(pid=pid, source=source, text=text))
    return tuple(paragraphs)


def normalize_evidence(raw: dict[str, Any]) -> Evidence | None:
    value = clean_space(raw.get("value") or raw.get("answer") or raw.get("text"))
    if not value:
        return None
    try:
        pid = int(raw.get("pid", raw.get("source_id", raw.get("paragraph_id", 0))))
    except (TypeError, ValueError):
        pid = 0

    def maybe_int(key: str) -> int | None:
        try:
            value = raw.get(key)
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    return Evidence(value=value, pid=pid, begin=maybe_int("begin"), end=maybe_int("end"))


def normalize_attributes(record: dict[str, Any]) -> tuple[Attribute, ...]:
    raw_attributes = record.get("attributes") or record.get("labels") or []
    attrs: list[Attribute] = []

    if isinstance(raw_attributes, dict):
        raw_attributes = [{"key": key, "evidences": value} for key, value in raw_attributes.items()]
    if not isinstance(raw_attributes, list):
        return tuple()

    for raw_attr in raw_attributes:
        if not isinstance(raw_attr, dict):
            continue
        key = clean_space(raw_attr.get("key") or raw_attr.get("attribute") or raw_attr.get("name"))
        if not key:
            continue

        raw_evidences = raw_attr.get("evidences")
        if raw_evidences is None and any(field in raw_attr for field in ("value", "answer", "text")):
            raw_evidences = [raw_attr]
        if isinstance(raw_evidences, dict):
            raw_evidences = [raw_evidences]
        if not isinstance(raw_evidences, list):
            raw_evidences = []

        evidences = tuple(
            evidence
            for evidence in (normalize_evidence(raw_evidence) for raw_evidence in raw_evidences if isinstance(raw_evidence, dict))
            if evidence is not None
        )
        attrs.append(Attribute(key=key, evidences=evidences))

    return tuple(attrs)


def compact_detail_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_space(value)
    if isinstance(value, (int, float, bool)):
        return clean_space(value)
    if isinstance(value, list):
        return clean_space(", ".join(compact_detail_value(item) for item in value if compact_detail_value(item)))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = compact_detail_value(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return clean_space("; ".join(parts))
    return clean_space(value)


def append_amazon23_paragraph(paragraphs: list[dict[str, Any]], source: str, text: Any, *, max_chars: int) -> None:
    text = compact_detail_value(text)
    if not text:
        return
    if max_chars > 0:
        text = text[:max_chars].rstrip()
    paragraphs.append({"pid": len(paragraphs), "source": source, "text": text})


def amazon23_paragraphs(record: dict[str, Any], *, max_detail_paragraphs: int, max_paragraph_chars: int) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    append_amazon23_paragraph(paragraphs, "title", record.get("title"), max_chars=max_paragraph_chars)
    append_amazon23_paragraph(paragraphs, "brand", record.get("store"), max_chars=max_paragraph_chars)
    append_amazon23_paragraph(paragraphs, "description", record.get("description"), max_chars=max_paragraph_chars)
    append_amazon23_paragraph(paragraphs, "features", record.get("features"), max_chars=max_paragraph_chars)
    append_amazon23_paragraph(paragraphs, "categories", record.get("categories"), max_chars=max_paragraph_chars)

    details = record.get("details")
    if isinstance(details, dict):
        used = 0
        for key, value in details.items():
            text = compact_detail_value(value)
            if not text:
                continue
            append_amazon23_paragraph(paragraphs, f"detail:{clean_space(key)}", f"{key}: {text}", max_chars=max_paragraph_chars)
            used += 1
            if max_detail_paragraphs > 0 and used >= max_detail_paragraphs:
                break
    return paragraphs


def find_value_span(value: str, paragraphs: list[dict[str, Any]]) -> dict[str, Any] | None:
    needle = clean_space(value)
    if not needle:
        return None
    needle_lower = needle.lower()
    for paragraph in paragraphs:
        text = paragraph["text"]
        begin = text.lower().find(needle_lower)
        if begin >= 0:
            end = begin + len(needle)
            return {
                "value": text[begin:end],
                "pid": paragraph["pid"],
                "begin": begin,
                "end": end,
            }
    return None


def reground_attributes_for_amazon23(
    label_record: dict[str, Any],
    paragraphs: list[dict[str, Any]],
    *,
    allow_ungrounded: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    stats = Counter()
    grounded_attrs = []
    for attr in normalize_attributes(label_record):
        seen_values = []
        for evidence in attr.evidences:
            value = clean_space(evidence.value)
            if value and value not in seen_values:
                seen_values.append(value)
        if not seen_values:
            stats["attributes_without_label_value"] += 1
            continue

        evidences = []
        for value in seen_values:
            grounded = find_value_span(value, paragraphs)
            if grounded is not None:
                evidences.append(grounded)
            elif allow_ungrounded:
                evidences.append({"value": value, "pid": 0, "begin": None, "end": None})
            else:
                stats["values_not_found_in_amazon23_text"] += 1

        if not evidences:
            stats["attributes_dropped_ungrounded"] += 1
            continue
        grounded_attrs.append({"key": attr.key, "evidences": evidences})
        stats["attributes_grounded"] += 1
    return grounded_attrs, stats


def load_positive_label_map(path: Path) -> tuple[dict[str, dict[str, Any]], Counter]:
    labels = {}
    stats = Counter()
    for record in iter_jsonl(path):
        stats["positive_label_records_seen"] += 1
        product_id = product_id_from_record(record)
        if not product_id:
            stats["positive_label_records_without_id"] += 1
            continue
        attrs = normalize_attributes(record)
        if not any(attr.evidences for attr in attrs):
            stats["positive_label_records_without_evidence"] += 1
            continue
        labels[product_id] = {
            "id": product_id,
            "category": clean_space(first_present(record, ("category", "category_name", "product_category"))),
            "attributes": record.get("attributes") or record.get("labels") or [],
        }
    stats["positive_label_records_loaded"] = len(labels)
    return labels, stats


def open_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def build_amazon23_full_positive_jsonl(args: argparse.Namespace) -> Path:
    output_path = args.amazon23_full_output
    if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite_amazon23_full:
        if has_full_product_profiles(output_path):
            print(f"using_existing_amazon23_full_positive={output_path}")
            return output_path

    if args.positive_label_jsonl is None:
        raise RuntimeError("Missing --positive-label-jsonl for Amazon 2023 join.")
    meta_files = amazon23_meta_files(args)
    if not meta_files:
        raise RuntimeError("No Amazon 2023 metadata files found. Pass --amazon23-meta-dir or --amazon23-meta-jsonl.")

    labels_by_id, stats = load_positive_label_map(args.positive_label_jsonl)
    matched_ids = set()
    skipped_seen_ids = set()
    print(
        json.dumps(
            {
                "amazon23_join": "start",
                "positive_label_jsonl": str(args.positive_label_jsonl),
                "num_labels": len(labels_by_id),
                "num_meta_files": len(meta_files),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )

    with open_jsonl_writer(output_path) as f:
        for meta_file in meta_files:
            file_stats = Counter()
            for meta_idx, meta_record in enumerate(iter_jsonl(meta_file), start=1):
                file_stats["amazon23_meta_records_seen"] += 1
                parent_asin = clean_space(meta_record.get("parent_asin"))
                if not parent_asin or parent_asin not in labels_by_id:
                    continue
                if parent_asin in matched_ids:
                    file_stats["amazon23_duplicate_matches_skipped"] += 1
                    continue

                label_record = labels_by_id[parent_asin]
                paragraphs = amazon23_paragraphs(
                    meta_record,
                    max_detail_paragraphs=args.amazon23_max_detail_paragraphs,
                    max_paragraph_chars=args.amazon23_max_paragraph_chars,
                )
                if not paragraphs:
                    file_stats["amazon23_matches_without_paragraphs"] += 1
                    skipped_seen_ids.add(parent_asin)
                    continue
                attrs, grounding_stats = reground_attributes_for_amazon23(
                    label_record,
                    paragraphs,
                    allow_ungrounded=args.amazon23_allow_ungrounded,
                )
                file_stats.update(grounding_stats)
                if len(attrs) < args.amazon23_min_grounded_attributes:
                    file_stats["amazon23_matches_below_min_grounded_attrs"] += 1
                    skipped_seen_ids.add(parent_asin)
                    continue

                category = clean_space(label_record.get("category")) or clean_space(meta_record.get("main_category"))
                full_record = {
                    "id": parent_asin,
                    "category": category,
                    "paragraphs": paragraphs,
                    "attributes": attrs,
                    "metadata": {
                        "source": "amazon_reviews_2023",
                        "meta_file": str(meta_file),
                        "main_category": meta_record.get("main_category"),
                        "parent_asin": parent_asin,
                    },
                }
                f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
                matched_ids.add(parent_asin)
                file_stats["amazon23_full_records_written"] += 1
                if args.amazon23_join_progress_every and len(matched_ids) % args.amazon23_join_progress_every == 0:
                    print(f"amazon23_join_written={len(matched_ids)} current_file={meta_file.name}")

                if args.max_products is not None and len(matched_ids) >= args.max_products:
                    break

            stats.update(file_stats)
            print(
                json.dumps(
                    {
                        "amazon23_join_file": str(meta_file),
                        "file_stats": dict(file_stats),
                        "total_written": len(matched_ids),
                    },
                    ensure_ascii=False,
                )
            )
            if args.max_products is not None and len(matched_ids) >= args.max_products:
                break

    metadata = {
        "positive_label_jsonl": str(args.positive_label_jsonl),
        "amazon23_meta_files": [str(path) for path in meta_files],
        "amazon23_full_output": str(output_path),
        "stats": {
            **dict(stats),
            "positive_label_records_matched": len(matched_ids),
            "positive_label_records_seen_but_skipped": len(skipped_seen_ids),
            "positive_label_records_unmatched": max(0, len(labels_by_id) - len(matched_ids) - len(skipped_seen_ids)),
            "amazon23_allow_ungrounded": args.amazon23_allow_ungrounded,
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"amazon23_join": "done", **metadata["stats"]}, indent=2, ensure_ascii=False))
    return output_path


def normalize_product(record: dict[str, Any]) -> MAVEProduct | None:
    product_id = clean_space(first_present(record, ("id", "product_id", "asin", "asin_id")))
    category = clean_space(first_present(record, ("category", "category_name", "product_category")))
    paragraphs = normalize_paragraphs(record)
    attributes = tuple(attr for attr in normalize_attributes(record) if attr.evidences and attr.canonical_value)
    if not product_id or not category or not paragraphs or not attributes:
        return None
    return MAVEProduct(product_id=product_id, category=category, paragraphs=paragraphs, attributes=attributes)


def negative_attribute_keys(record: dict[str, Any]) -> tuple[str, ...]:
    attrs = normalize_attributes(record)
    keys = [attr.key for attr in attrs if attr.key and not attr.evidences]

    top_level_key = clean_space(first_present(record, ("key", "attribute", "attribute_key", "attribute_name", "name")))
    if top_level_key:
        keys.append(top_level_key)

    return tuple(dict.fromkeys(keys))


def product_id_from_record(record: dict[str, Any]) -> str:
    return clean_space(first_present(record, ("id", "product_id", "asin", "asin_id")))


def source_by_pid(product: MAVEProduct) -> dict[int, Paragraph]:
    return {paragraph.pid: paragraph for paragraph in product.paragraphs}


def render_product_profile(product: MAVEProduct) -> str:
    lines = []
    for paragraph in product.paragraphs:
        lines.append(f"[{paragraph.source}] {paragraph.text}")
    return "\n".join(lines)


def render_sources(product: MAVEProduct) -> str:
    lines = []
    for paragraph in product.paragraphs:
        lines.append(f"[{paragraph.pid}:{paragraph.source}] {paragraph.text}")
    return "\n".join(lines)


def evidence_payload(product: MAVEProduct, attr: Attribute, *, include_source_text: bool = True) -> list[dict[str, Any]]:
    paragraphs = source_by_pid(product)
    evidence_items: list[dict[str, Any]] = []
    for evidence in attr.evidences:
        paragraph = paragraphs.get(evidence.pid)
        if paragraph is None:
            continue
        payload: dict[str, Any] = {
            "source_id": evidence.pid,
            "source": paragraph.source,
            "text": evidence.value,
            "begin": evidence.begin,
            "end": evidence.end,
        }
        if include_source_text:
            payload["source_text"] = paragraph.text
        evidence_items.append(payload)
    return evidence_items


def make_row(
    *,
    prompt: str,
    response_obj: Any,
    task_type: str,
    product: MAVEProduct,
    split: str,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = json_dumps(response_obj)
    row_id = stable_int(f"{product.product_id}|{task_type}|{prompt}|{response}|{seed}")
    row = {
        "id": f"mave-{row_id:08x}",
        "data_source": "mave",
        "task_type": task_type,
        "split": split,
        "product_id": product.product_id,
        "category": product.category,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }
    if metadata:
        row.update(metadata)
    return row


def single_attribute_qa_rows(
    product: MAVEProduct,
    *,
    negative_keys: tuple[str, ...],
    split: str,
    rng: random.Random,
    seed: int,
    negatives_per_positive: float,
) -> list[dict[str, Any]]:
    rows = []
    profile = render_product_profile(product)
    for attr in product.attributes:
        prompt = (
            f"Category: {product.category}\n\n"
            f"Product profile:\n{profile}\n\n"
            f"Question:\nWhat is the value of the attribute \"{attr.key}\" for this product?"
        )
        response = {
            "attribute": attr.key,
            "value": attr.canonical_value,
            "answerable": True,
        }
        rows.append(
            make_row(
                prompt=prompt,
                response_obj=response,
                task_type="single_attribute_qa",
                product=product,
                split=split,
                seed=seed,
                metadata={"attribute": attr.key, "answerable": True},
            )
        )

    if negative_keys and negatives_per_positive > 0:
        sample_size = min(len(negative_keys), round(len(product.attributes) * negatives_per_positive))
        for key in rng.sample(list(negative_keys), sample_size):
            prompt = (
                f"Category: {product.category}\n\n"
                f"Product profile:\n{profile}\n\n"
                f"Question:\nWhat is the value of the attribute \"{key}\" for this product?"
            )
            response = {
                "attribute": key,
                "value": None,
                "answerable": False,
            }
            rows.append(
                make_row(
                    prompt=prompt,
                    response_obj=response,
                    task_type="single_attribute_qa",
                    product=product,
                    split=split,
                    seed=seed,
                    metadata={"attribute": key, "answerable": False},
                )
            )
    return rows


def evidence_grounded_rows(product: MAVEProduct, *, split: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    sources = render_sources(product)
    for attr in product.attributes:
        prompt = (
            f"Extract the value of \"{attr.key}\" from the product profile.\n"
            "Return the value and supporting evidence.\n\n"
            f"Category: {product.category}\n\n"
            f"Sources:\n{sources}"
        )
        response = {
            "attribute": attr.key,
            "value": attr.canonical_value,
            "evidence": evidence_payload(product, attr, include_source_text=False),
        }
        rows.append(
            make_row(
                prompt=prompt,
                response_obj=response,
                task_type="evidence_grounded_extraction",
                product=product,
                split=split,
                seed=seed,
                metadata={"attribute": attr.key, "answerable": True},
            )
        )
    return rows


def multi_attribute_card_rows(
    product: MAVEProduct,
    *,
    negative_keys: tuple[str, ...],
    split: str,
    rng: random.Random,
    seed: int,
    min_attributes: int,
    max_attributes: int,
    negatives_per_card: int,
    cards_per_product: int,
) -> list[dict[str, Any]]:
    if not product.attributes:
        return []

    rows = []
    profile = render_product_profile(product)
    positive_attrs = list(product.attributes)
    attr_by_key = {attr.key: attr for attr in positive_attrs}
    card_count = max(1, cards_per_product)

    for card_idx in range(card_count):
        max_pos = min(max_attributes, len(positive_attrs))
        min_pos = min(min_attributes, max_pos)
        if card_idx == 0:
            chosen_positive = positive_attrs[:max_pos]
        else:
            chosen_positive = rng.sample(positive_attrs, rng.randint(min_pos, max_pos))

        chosen_negative = []
        if negative_keys and negatives_per_card > 0:
            chosen_negative = rng.sample(list(negative_keys), min(len(negative_keys), negatives_per_card))

        target_keys = [attr.key for attr in chosen_positive] + chosen_negative
        rng.shuffle(target_keys)

        prompt = (
            "You are given a product profile and a list of target attributes.\n"
            "Fill in the product attribute table. Use null if the value is not stated.\n\n"
            f"Category: {product.category}\n\n"
            f"Product profile:\n{profile}\n\n"
            "Target attributes:\n"
            + "\n".join(f"- {key}" for key in target_keys)
        )
        response = {
            key: attr_by_key[key].canonical_value if key in attr_by_key else None
            for key in target_keys
        }
        rows.append(
            make_row(
                prompt=prompt,
                response_obj=response,
                task_type="multi_attribute_card_completion",
                product=product,
                split=split,
                seed=seed,
                metadata={"target_attributes": target_keys},
            )
        )
    return rows


def product_customer_qa_rows(product: MAVEProduct, *, split: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    profile = render_product_profile(product)
    for attr in product.attributes:
        prompt = (
            f"Product:\n{profile}\n\n"
            f"User question:\nWhat is the {attr.key.lower()} of this product?"
        )
        evidence_items = evidence_payload(product, attr, include_source_text=True)
        response = {
            "answer": attr.canonical_value,
            "attribute": attr.key,
            "evidence": evidence_items[0]["source_text"] if evidence_items else None,
        }
        rows.append(
            make_row(
                prompt=prompt,
                response_obj=response,
                task_type="product_customer_qa",
                product=product,
                split=split,
                seed=seed,
                metadata={"attribute": attr.key, "answerable": True},
            )
        )
    return rows


def faceted_positive_row(
    product: MAVEProduct,
    attrs: list[Attribute],
    *,
    split: str,
    seed: int,
) -> dict[str, Any]:
    profile = render_product_profile(product)
    constraints = {attr.key: attr.canonical_value for attr in attrs}
    prompt = (
        "User wants a product with:\n"
        + "\n".join(f"- {key}: {value}" for key, value in constraints.items())
        + "\n\n"
        f"Product profile:\n{profile}\n\n"
        "Does this product satisfy the user's constraints?"
    )
    response = {
        "match": True,
        "matched_attributes": constraints,
        "missing_or_mismatched_attributes": {},
    }
    return make_row(
        prompt=prompt,
        response_obj=response,
        task_type="faceted_search_filtering",
        product=product,
        split=split,
        seed=seed,
        metadata={"target_attributes": list(constraints), "match": True},
    )


def faceted_negative_row(
    product: MAVEProduct,
    constraints: dict[str, str | None],
    missing_or_mismatched: dict[str, Any],
    *,
    split: str,
    seed: int,
) -> dict[str, Any]:
    profile = render_product_profile(product)
    prompt = (
        "User wants a product with:\n"
        + "\n".join(f"- {key}: {value}" for key, value in constraints.items())
        + "\n\n"
        f"Product profile:\n{profile}\n\n"
        "Does this product satisfy the user's constraints?"
    )
    positives = {attr.key: attr.canonical_value for attr in product.attributes}
    matched = {
        key: value
        for key, value in constraints.items()
        if key in positives and positives[key] == value
    }
    response = {
        "match": False,
        "matched_attributes": matched,
        "missing_or_mismatched_attributes": missing_or_mismatched,
    }
    return make_row(
        prompt=prompt,
        response_obj=response,
        task_type="faceted_search_filtering",
        product=product,
        split=split,
        seed=seed,
        metadata={"target_attributes": list(constraints), "match": False},
    )


def faceted_search_rows(
    product: MAVEProduct,
    *,
    negative_keys: tuple[str, ...],
    value_pool_by_category_attr: dict[tuple[str, str], set[str]],
    value_pool_by_attr: dict[str, set[str]],
    split: str,
    rng: random.Random,
    seed: int,
    min_constraints: int,
    max_constraints: int,
    positive_cards_per_product: int,
    negative_cards_per_product: int,
) -> list[dict[str, Any]]:
    rows = []
    positive_attrs = list(product.attributes)
    if not positive_attrs:
        return rows

    for card_idx in range(max(1, positive_cards_per_product)):
        max_pos = min(max_constraints, len(positive_attrs))
        min_pos = min(min_constraints, max_pos)
        chosen = positive_attrs[:max_pos] if card_idx == 0 else rng.sample(positive_attrs, rng.randint(min_pos, max_pos))
        rows.append(faceted_positive_row(product, chosen, split=split, seed=seed))

    for _ in range(max(0, negative_cards_per_product)):
        mode = "missing_attribute"
        candidate_attrs = [
            attr
            for attr in positive_attrs
            if len(value_pool_by_category_attr.get((product.category, attr.key), set()) - {attr.canonical_value}) > 0
        ]
        if candidate_attrs and (not negative_keys or rng.random() < 0.5):
            mode = "mismatched_value"

        if mode == "mismatched_value":
            mismatch_attr = rng.choice(candidate_attrs)
            other_values = sorted(value_pool_by_category_attr[(product.category, mismatch_attr.key)] - {mismatch_attr.canonical_value})
            wrong_value = rng.choice(other_values)
            constraints = {mismatch_attr.key: wrong_value}
            if len(positive_attrs) > 1 and rng.random() < 0.7:
                extra_attr = rng.choice([attr for attr in positive_attrs if attr.key != mismatch_attr.key])
                constraints[extra_attr.key] = extra_attr.canonical_value
            missing_or_mismatched = {
                mismatch_attr.key: {
                    "expected": wrong_value,
                    "observed": mismatch_attr.canonical_value,
                }
            }
        else:
            if not negative_keys:
                continue
            missing_key = rng.choice(list(negative_keys))
            desired_values = sorted(
                value_pool_by_category_attr.get((product.category, missing_key), set())
                or value_pool_by_attr.get(missing_key, set())
            )
            desired_value = rng.choice(desired_values) if desired_values else "a stated value"
            constraints = {missing_key: desired_value}
            if positive_attrs and rng.random() < 0.7:
                extra_attr = rng.choice(positive_attrs)
                constraints[extra_attr.key] = extra_attr.canonical_value
            missing_or_mismatched = {
                missing_key: {
                    "expected": desired_value,
                    "observed": None,
                    "reason": "attribute not stated in the product profile",
                }
            }

        rows.append(
            faceted_negative_row(
                product,
                constraints,
                missing_or_mismatched,
                split=split,
                seed=seed,
            )
        )
    return rows


def split_for_product(product_id: str, *, val_ratio: float, test_ratio: float, seed: int) -> str:
    bucket = stable_int(f"{seed}|{product_id}") / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def load_negative_map(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    negative_map: dict[str, list[str]] = defaultdict(list)
    for record in iter_jsonl(path):
        product_id = product_id_from_record(record)
        if not product_id:
            continue
        for key in negative_attribute_keys(record):
            if key not in negative_map[product_id]:
                negative_map[product_id].append(key)
    return {product_id: tuple(keys) for product_id, keys in negative_map.items()}


def build_value_pool(products: list[MAVEProduct]) -> dict[tuple[str, str], set[str]]:
    pool: dict[tuple[str, str], set[str]] = defaultdict(set)
    for product in products:
        for attr in product.attributes:
            pool[(product.category, attr.key)].add(attr.canonical_value)
    return pool


def build_value_pool_by_attr(products: list[MAVEProduct]) -> dict[str, set[str]]:
    pool: dict[str, set[str]] = defaultdict(set)
    for product in products:
        for attr in product.attributes:
            pool[attr.key].add(attr.canonical_value)
    return pool


def load_products(path: Path, *, max_products: int | None) -> tuple[list[MAVEProduct], Counter]:
    products = []
    stats = Counter()
    for record in iter_jsonl(path):
        stats["records_seen"] += 1
        product = normalize_product(record)
        if product is None:
            stats["records_skipped"] += 1
            continue
        products.append(product)
        stats["products_loaded"] += 1
        stats["positive_product_attribute_pairs"] += len(product.attributes)
        if max_products is not None and len(products) >= max_products:
            break
    return products, stats


def maybe_sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples_per_task: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples_per_task is None:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)

    sampled = []
    for task_type, task_rows in grouped.items():
        if len(task_rows) <= max_examples_per_task:
            sampled.extend(task_rows)
            continue
        rng = random.Random(seed + stable_int(task_type))
        sampled.extend(rng.sample(task_rows, max_examples_per_task))
    return sampled


def build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    products, load_stats = load_products(args.positive_jsonl, max_products=args.max_products)
    negative_map = load_negative_map(args.negative_jsonl)
    value_pool = build_value_pool(products)
    value_pool_by_attr = build_value_pool_by_attr(products)

    rows: list[dict[str, Any]] = []
    tasks = set(args.tasks)
    stats = Counter(load_stats)
    stats["products_with_negative_labels"] = sum(1 for product in products if product.product_id in negative_map)
    stats["negative_product_attribute_pairs_loaded"] = sum(len(keys) for keys in negative_map.values())

    for product_idx, product in enumerate(products):
        product_seed = args.seed + stable_int(product.product_id)
        rng = random.Random(product_seed)
        split = split_for_product(product.product_id, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
        negative_keys = tuple(key for key in negative_map.get(product.product_id, ()) if key not in {attr.key for attr in product.attributes})

        if "single_attribute_qa" in tasks:
            rows.extend(
                single_attribute_qa_rows(
                    product,
                    negative_keys=negative_keys,
                    split=split,
                    rng=rng,
                    seed=args.seed,
                    negatives_per_positive=args.single_negative_ratio,
                )
            )
        if "evidence_grounded_extraction" in tasks:
            rows.extend(evidence_grounded_rows(product, split=split, seed=args.seed))
        if "multi_attribute_card_completion" in tasks:
            rows.extend(
                multi_attribute_card_rows(
                    product,
                    negative_keys=negative_keys,
                    split=split,
                    rng=rng,
                    seed=args.seed,
                    min_attributes=args.card_min_attributes,
                    max_attributes=args.card_max_attributes,
                    negatives_per_card=args.card_negatives,
                    cards_per_product=args.cards_per_product,
                )
            )
        if "product_customer_qa" in tasks:
            rows.extend(product_customer_qa_rows(product, split=split, seed=args.seed))
        if "faceted_search_filtering" in tasks:
            rows.extend(
                faceted_search_rows(
                    product,
                    negative_keys=negative_keys,
                    value_pool_by_category_attr=value_pool,
                    value_pool_by_attr=value_pool_by_attr,
                    split=split,
                    rng=rng,
                    seed=args.seed,
                    min_constraints=args.faceted_min_constraints,
                    max_constraints=args.faceted_max_constraints,
                    positive_cards_per_product=args.faceted_positive_per_product,
                    negative_cards_per_product=args.faceted_negative_per_product,
                )
            )

        if args.progress_every and (product_idx + 1) % args.progress_every == 0:
            print(f"processed_products={product_idx + 1} generated_rows={len(rows)}")

    rows = maybe_sample_rows(rows, max_examples_per_task=args.max_examples_per_task, seed=args.seed)
    random.Random(args.seed).shuffle(rows)

    split_counts = Counter(row["split"] for row in rows)
    task_counts = Counter(row["task_type"] for row in rows)
    stats.update(
        {
            "num_examples": len(rows),
            "num_train": split_counts["train"],
            "num_val": split_counts["val"],
            "num_test": split_counts["test"],
        }
    )
    return rows, {
        "stats": dict(stats),
        "task_counts": dict(task_counts),
        "split_counts": dict(split_counts),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        return False
    return True


def split_group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [row for row in rows if row["split"] == "train"],
        "val": [row for row in rows if row["split"] == "val"],
        "test": [row for row in rows if row["split"] == "test"],
        "all": rows,
    }


def write_group_outputs(
    output_dir: Path,
    grouped: dict[str, list[dict[str, Any]]],
    *,
    write_parquet: bool,
    allow_missing_parquet: bool,
) -> dict[str, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in grouped.items():
        write_jsonl(output_dir / f"{name}.jsonl", split_rows)

    if not write_parquet:
        return {name: False for name in grouped}

    parquet_written = {}
    for name, split_rows in grouped.items():
        parquet_written[name] = maybe_write_parquet(output_dir / f"{name}.parquet", split_rows)
        if split_rows and not parquet_written[name] and not allow_missing_parquet:
            raise RuntimeError(
                "Failed to write parquet. Install pandas and pyarrow, or rerun with --no-write-parquet."
            )
    return parquet_written


def write_task_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_root = args.output_dir / args.task_output_dir_name
    task_metadata: dict[str, Any] = {
        "root": str(task_root),
        "tasks": {},
    }
    for task_type in args.tasks:
        task_rows = [row for row in rows if row["task_type"] == task_type]
        grouped = split_group(task_rows)
        parquet_written = write_group_outputs(
            task_root / task_type,
            grouped,
            write_parquet=args.write_parquet,
            allow_missing_parquet=args.allow_missing_parquet,
        )
        task_metadata["tasks"][task_type] = {
            "output_dir": str(task_root / task_type),
            "num_examples": len(task_rows),
            "num_train": len(grouped["train"]),
            "num_val": len(grouped["val"]),
            "num_test": len(grouped["test"]),
            "parquet_written": parquet_written,
            "train_file": str(task_root / task_type / "train.parquet") if parquet_written.get("train") else None,
            "val_file": str(task_root / task_type / "val.parquet") if parquet_written.get("val") else None,
        }

    (task_root / "metadata.json").write_text(
        json.dumps(task_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return task_metadata


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = split_group(rows)
    parquet_written = write_group_outputs(
        args.output_dir,
        grouped,
        write_parquet=args.write_parquet,
        allow_missing_parquet=args.allow_missing_parquet,
    )
    task_outputs = write_task_outputs(args, rows) if args.write_task_datasets else None

    metadata = {
        **metadata,
        "positive_jsonl": str(args.positive_jsonl),
        "positive_label_jsonl": str(args.positive_label_jsonl) if args.positive_label_jsonl else None,
        "negative_jsonl": str(args.negative_jsonl) if args.negative_jsonl else None,
        "amazon23_meta_dir": str(args.amazon23_meta_dir) if args.amazon23_meta_dir else None,
        "amazon23_meta_jsonl": [str(path) for path in args.amazon23_meta_jsonl] if args.amazon23_meta_jsonl else None,
        "amazon23_full_output": str(args.amazon23_full_output) if is_amazon23_join_requested(args) else None,
        "output_dir": str(args.output_dir),
        "tasks": args.tasks,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "parquet_written": parquet_written,
        "task_outputs": task_outputs,
        "format": {
            "prompt": "SFT user prompt string",
            "response": "JSON string target",
            "messages": "chat messages for data.multiturn.messages_key=messages",
            "task_type": "one of the five MAVE-derived task families",
        },
        "train_command_hint": (
            "Use data.train_files=<output_dir>/train.parquet "
            "data.val_files=<output_dir>/val.parquet data.prompt_key=prompt data.response_key=response. "
            f"For a single task, use <output_dir>/{args.task_output_dir_name}/<task_type>/train.parquet."
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build five SFT task datasets from MAVE full JSONL files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory used for downloaded MAVE labels and default full-file lookup.",
    )
    parser.add_argument(
        "--positive-jsonl",
        type=Path,
        default=None,
        help="Full MAVE positive JSONL with paragraphs and positive attribute evidences.",
    )
    parser.add_argument(
        "--positive-label-jsonl",
        type=Path,
        default=None,
        help=(
            "MAVE positive label-only JSONL. Used with --amazon23-meta-dir/--amazon23-meta-jsonl "
            "to build a full positive JSONL from Amazon Reviews 2023 metadata."
        ),
    )
    parser.add_argument(
        "--negative-jsonl",
        type=Path,
        default=None,
        help="Optional MAVE negative JSONL or label JSONL. Null examples are created only from this file.",
    )
    parser.add_argument(
        "--download-mave-labels",
        action="store_true",
        help="Download official MAVE positive/negative label JSONL files into --raw-dir/labels.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download official label files and exit without building SFT data.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Redownload MAVE label files even if they already exist.",
    )
    parser.add_argument(
        "--amazon23-meta-dir",
        type=Path,
        default=None,
        help="Directory containing Amazon Reviews 2023 meta_*.jsonl.gz files to join with MAVE labels.",
    )
    parser.add_argument(
        "--amazon23-meta-jsonl",
        type=Path,
        action="append",
        default=None,
        help="Specific Amazon Reviews 2023 metadata JSONL/JSONL.GZ file. Can be repeated.",
    )
    parser.add_argument(
        "--amazon23-full-output",
        type=Path,
        default=DEFAULT_AMAZON23_FULL_POSITIVE,
        help="Where to write/read the Amazon-2023-joined full MAVE positive JSONL.",
    )
    parser.add_argument(
        "--overwrite-amazon23-full",
        action="store_true",
        help="Rebuild --amazon23-full-output even if it already exists.",
    )
    parser.add_argument(
        "--amazon23-allow-ungrounded",
        action="store_true",
        help="Keep attribute values that cannot be found in 2023 metadata text. Not recommended for evidence tasks.",
    )
    parser.add_argument("--amazon23-min-grounded-attributes", type=int, default=1)
    parser.add_argument("--amazon23-max-detail-paragraphs", type=int, default=80)
    parser.add_argument("--amazon23-max-paragraph-chars", type=int, default=2000)
    parser.add_argument("--amazon23-join-progress-every", type=int, default=50000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="+", choices=list(DEFAULT_TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-examples-per-task", type=int, default=None)
    parser.add_argument("--single-negative-ratio", type=float, default=1.0)
    parser.add_argument("--card-min-attributes", type=int, default=2)
    parser.add_argument("--card-max-attributes", type=int, default=6)
    parser.add_argument("--card-negatives", type=int, default=1)
    parser.add_argument("--cards-per-product", type=int, default=1)
    parser.add_argument("--faceted-min-constraints", type=int, default=1)
    parser.add_argument("--faceted-max-constraints", type=int, default=3)
    parser.add_argument("--faceted-positive-per-product", type=int, default=1)
    parser.add_argument("--faceted-negative-per-product", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-task-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write one train/val/test/all dataset directory per task under <output-dir>/by_task.",
    )
    parser.add_argument("--task-output-dir-name", default="by_task")
    parser.add_argument("--allow-missing-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to less than 1.")
    ensure_input_files(args)
    if args.download_only:
        return
    rows, metadata = build_examples(args)
    write_outputs(args, rows, metadata)


if __name__ == "__main__":
    main()
