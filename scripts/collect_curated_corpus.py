#!/usr/bin/env python3
"""
Collect pre-shift curated corpora for Unbox retrospective experiments.

The script uses OpenAlex search with hand-curated query seeds per paradigm shift,
then applies strict year guards to prevent leakage.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
from pathlib import Path
import time
from typing import Any

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_SLEEP_SECONDS = 0.4
MAX_RETRIES = 4

CORPUS_CONFIGS: dict[str, dict[str, object]] = {
    "transformer": {
        "start_year": 2014,
        "end_year": 2016,
        "target_year": 2017,
        "queries": [
            "sequence to sequence learning neural machine translation",
            "recurrent neural network sequence modeling",
            "long short term memory machine translation",
            "attention mechanism encoder decoder",
            "gated recurrent unit language modeling",
            "bidirectional lstm tagging",
            "neural machine translation rare words",
            "sequence prediction recurrent neural networks",
        ],
        "topic_keywords": [
            "sequence",
            "translation",
            "recurrent",
            "lstm",
            "gru",
            "encoder",
            "decoder",
            "attention",
            "language model",
        ],
    },
    "diffusion": {
        "start_year": 2017,
        "end_year": 2019,
        "target_year": 2020,
        "queries": [
            "generative adversarial network image synthesis",
            "normalizing flow density estimation",
            "variational autoencoder image generation",
            "autoregressive image modeling",
            "energy based model generative",
            "denoising score matching deep learning",
            "generative modeling likelihood based",
            "implicit generative models training",
        ],
        "topic_keywords": [
            "generative",
            "adversarial",
            "gan",
            "flow",
            "variational",
            "autoencoder",
            "autoregressive",
            "density",
            "score matching",
            "energy based",
            "image synthesis",
        ],
    },
    "icl": {
        "start_year": 2017,
        "end_year": 2019,
        "target_year": 2020,
        "queries": [
            "fine tuning pretrained language models downstream tasks",
            "language model pretraining fine tuning",
            "BERT fine tuning task adaptation",
            "pretrained representations fine tuning NLP",
            "transfer learning fine tuning text classification",
            "weight updating adaptation pretrained models",
            "supervised fine tuning language understanding",
            "multi task fine tuning pretrained transformers",
            "few shot learning meta learning",
            "task specific fine tuning pretrained neural networks",
        ],
        "topic_keywords": [
            "fine tuning",
            "fine-tuning",
            "pretrain",
            "pretraining",
            "downstream",
            "task adaptation",
            "transfer learning",
            "BERT",
            "language model",
            "weight update",
            "task specific",
        ],
    },
    "vit": {
        "start_year": 2017,
        "end_year": 2019,
        "target_year": 2020,
        "queries": [
            "convolutional neural networks image recognition",
            "object detection convolutional architecture",
            "image classification deep residual networks",
            "vision representation learning convolution",
            "semantic segmentation convolutional network",
            "self attention in vision models",
            "non local neural networks computer vision",
            "image patches deep learning",
        ],
        "topic_keywords": [
            "vision",
            "image",
            "convolution",
            "cnn",
            "residual",
            "object detection",
            "segmentation",
            "attention",
            "non local",
        ],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect curated pre-shift corpora from OpenAlex"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for category datasets",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=80,
        help="Number of papers to keep per category",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=300,
        help="Maximum fetched candidates per query",
    )
    parser.add_argument(
        "--email",
        default="",
        help="Optional contact email for OpenAlex polite pool",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CORPUS_CONFIGS.keys()),
        choices=list(CORPUS_CONFIGS.keys()),
        help="Subset of categories to collect",
    )
    return parser.parse_args()


def _request_json(url: str, params: dict[str, str]) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 429:
                wait_seconds = min(30, 2 * attempt)
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} retries: {last_error}")


def _reconstruct_abstract(abstract_index: dict[str, list[int]] | None) -> str:
    if not abstract_index:
        return ""
    position_to_token: dict[int, str] = {}
    for token, positions in abstract_index.items():
        for position in positions:
            position_to_token[position] = token
    if not position_to_token:
        return ""
    ordered_positions = sorted(position_to_token)
    return " ".join(position_to_token[p] for p in ordered_positions)


def _venue_name(work: dict[str, Any]) -> str:
    primary_location = work.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            name = source.get("display_name")
            if isinstance(name, str):
                return name
    host_venue = work.get("host_venue")
    if isinstance(host_venue, dict):
        name = host_venue.get("display_name")
        if isinstance(name, str):
            return name
    return ""


def _author_list(work: dict[str, Any]) -> list[dict[str, str]]:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return []
    result: list[dict[str, str]] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        author_id = author.get("id")
        display_name = author.get("display_name")
        if isinstance(display_name, str):
            result.append(
                {
                    "authorId": str(author_id) if author_id else "",
                    "name": display_name,
                }
            )
    return result


def _paper_text_for_scoring(work: dict[str, Any], abstract: str) -> str:
    title = work.get("display_name")
    title_text = title if isinstance(title, str) else ""
    venue = _venue_name(work)
    return f"{title_text} {abstract} {venue}".lower()


def _relevance_score(
    work: dict[str, Any],
    abstract: str,
    topic_keywords: Iterable[str],
) -> float:
    text = _paper_text_for_scoring(work, abstract)
    keyword_hits = 0
    for keyword in topic_keywords:
        key = keyword.lower()
        if key in text:
            keyword_hits += 1
    citation_count_raw = work.get("cited_by_count", 0)
    citation_count = citation_count_raw if isinstance(citation_count_raw, int) else 0
    citation_bonus = math.log10(citation_count + 1)
    abstract_bonus = 1.0 if abstract else -0.5
    return (2.0 * keyword_hits) + citation_bonus + abstract_bonus


def _fetch_query_results(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
    email: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cursor = "*"
    per_page = 200

    while len(results) < max_results:
        params: dict[str, str] = {
            "search": query,
            "filter": f"from_publication_date:{start_year}-01-01,"
            + f"to_publication_date:{end_year}-12-31,"
            + "type:article|proceedings-article,language:en",
            "per-page": str(per_page),
            "cursor": cursor,
            "sort": "cited_by_count:desc",
        }
        if email:
            params["mailto"] = email

        payload = _request_json(OPENALEX_WORKS_URL, params)
        page_results = payload.get("results")
        if not isinstance(page_results, list) or not page_results:
            break

        for work in page_results:
            if isinstance(work, dict):
                results.append(work)
                if len(results) >= max_results:
                    break

        meta = payload.get("meta")
        next_cursor = ""
        if isinstance(meta, dict):
            cursor_value = meta.get("next_cursor")
            if isinstance(cursor_value, str):
                next_cursor = cursor_value
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(REQUEST_SLEEP_SECONDS)

    return results


def _normalize_paper(work: dict[str, Any], category: str) -> dict[str, Any] | None:
    year = work.get("publication_year")
    if not isinstance(year, int):
        return None
    title = work.get("display_name")
    if not isinstance(title, str) or not title.strip():
        return None
    abstract = _reconstruct_abstract(
        work.get("abstract_inverted_index")
        if isinstance(work.get("abstract_inverted_index"), dict)
        else None
    )

    return {
        "paperId": str(work.get("id", "")),
        "title": title.strip(),
        "authors": _author_list(work),
        "year": year,
        "abstract": abstract,
        "venue": _venue_name(work),
        "citationCount": int(work.get("cited_by_count", 0) or 0),
        "source": "openalex",
        "category": category,
    }


def _config_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"Invalid integer config value for {key}: {value}")


def _config_str_list(config: dict[str, object], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Invalid list config value for {key}: {value}")
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            output.append(item)
    if not output:
        raise ValueError(f"Empty list config value for {key}")
    return output


def collect_category(
    category: str,
    config: dict[str, object],
    target_count: int,
    max_per_query: int,
    email: str,
) -> list[dict[str, Any]]:
    candidates: dict[str, tuple[float, dict[str, Any]]] = {}
    queries = _config_str_list(config, "queries")
    topic_keywords = _config_str_list(config, "topic_keywords")
    start_year = _config_int(config, "start_year")
    end_year = _config_int(config, "end_year")
    target_year = _config_int(config, "target_year")

    for query in queries:
        works = _fetch_query_results(
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_results=max_per_query,
            email=email,
        )

        for work in works:
            normalized = _normalize_paper(work, category)
            if not normalized:
                continue
            year = normalized["year"]
            if not isinstance(year, int):
                continue
            if year > end_year or year >= target_year:
                continue

            paper_id = str(normalized["paperId"])
            if not paper_id:
                continue

            score = _relevance_score(work, str(normalized["abstract"]), topic_keywords)
            existing = candidates.get(paper_id)
            if not existing or score > existing[0]:
                candidates[paper_id] = (score, normalized)

    ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
    return [paper for _, paper in ranked[:target_count]]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _category_stats(
    rows: list[dict[str, Any]], config: dict[str, object]
) -> dict[str, Any]:
    years = [row["year"] for row in rows if isinstance(row.get("year"), int)]
    abstract_count = sum(1 for row in rows if str(row.get("abstract", "")).strip())
    leakage_count = sum(
        1
        for row in rows
        if isinstance(row.get("year"), int)
        and int(row["year"]) > _config_int(config, "end_year")
    )
    return {
        "paper_count": len(rows),
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "abstract_ratio": round((abstract_count / len(rows)), 3) if rows else 0.0,
        "leakage_count": leakage_count,
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    report: dict[str, Any] = {
        "generated_at": int(time.time()),
        "target_per_category": args.target_per_category,
        "max_per_query": args.max_per_query,
        "categories": {},
    }

    for category in args.categories:
        config = CORPUS_CONFIGS[category]
        rows = collect_category(
            category=category,
            config=config,
            target_count=args.target_per_category,
            max_per_query=args.max_per_query,
            email=args.email,
        )
        output_path = output_dir / category / "papers.jsonl"
        _write_jsonl(output_path, rows)
        stats = _category_stats(rows, config)
        report["categories"][category] = {
            "config": {
                "start_year": _config_int(config, "start_year"),
                "end_year": _config_int(config, "end_year"),
                "target_year": _config_int(config, "target_year"),
            },
            "output": str(output_path),
            "stats": stats,
        }
        print(
            f"[{category}] papers={stats['paper_count']} "
            f"years={stats['min_year']}-{stats['max_year']} "
            f"abstract_ratio={stats['abstract_ratio']} "
            f"leakage={stats['leakage_count']}"
        )

    report_path = output_dir / "corpus_collection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
