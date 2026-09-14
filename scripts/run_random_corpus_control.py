#!/usr/bin/env python3
"""Experiment 3: Random Corpus Redundancy Control.

Collects 60 random AI papers from OpenAlex (no topic filter beyond "AI/ML",
years 2015-2019), runs the clean prompt extraction with GPT-4o on 15 papers,
and compares redundancy rate with the focused corpus (~90%).

If redundancy is also ~90% -> LLM homogeneity (the model produces similar
assumptions regardless of input). If lower -> genuine field consensus in
the focused corpus.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

embed_texts = getattr(importlib.import_module("src.semantic_match"), "embed_texts")
cosine_similarity = getattr(importlib.import_module("src.semantic_match"), "cosine_similarity")

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "random_corpus_control"
MODEL = "gpt-4o"
MAX_RETRIES = 3
RETRY_DELAY = 5

CLEAN_PROMPT_TEMPLATE = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

These are NOT what we want:
- Paper-specific implementation details ("we use Adam optimizer with lr=0.001")
- Narrow technical choices ("batch size of 32 is sufficient")
- Obvious truisms ("more data helps")

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories:
- architectural: Structural requirements
- training: Learning procedure requirements
- data: Data requirements
- theoretical: Theoretical constraints
- evaluation: Evaluation paradigm assumptions

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how foundational they are to the entire field (not just this paper).

Format:
{{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "architectural|training|data|theoretical|evaluation"}}]}}

Paper text:
{paper_text}"""


def fetch_random_papers(target: int = 60) -> list[dict]:
    """Fetch random AI/ML papers from OpenAlex (2015-2019)."""
    requests = importlib.import_module("requests")

    # OpenAlex concept IDs for AI and ML
    # C154945302 = Artificial intelligence, C119857082 = Machine learning
    papers = []
    page = 1
    per_page = 50
    # Use random seed for reproducibility but different sample each time
    seed = random.randint(1000, 9999)

    print(f"Fetching random AI/ML papers from OpenAlex (seed={seed})...")

    while len(papers) < target and page <= 5:
        url = (
            f"https://api.openalex.org/works?"
            f"filter=concepts.id:C154945302|C119857082,"
            f"publication_year:2015-2019,"
            f"has_abstract:true,"
            f"type:article"
            f"&per_page={per_page}"
            f"&page={page}"
            f"&seed={seed}"
            f"&sample={per_page}"
            f"&mailto=research@example.com"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    print(f"  Failed to fetch page {page}: {exc}")
                    data = {"results": []}
                    break
                print(f"  Retry {attempt}: {exc}")
                time.sleep(RETRY_DELAY)

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            abstract_inv = work.get("abstract_inverted_index")
            if not abstract_inv:
                continue

            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in abstract_inv.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(w for _, w in word_positions)

            if len(abstract) < 100:
                continue

            title = work.get("title", "")
            if not title:
                continue

            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append({
                        "authorId": author.get("id", ""),
                        "name": author["display_name"],
                    })

            papers.append({
                "paperId": work.get("id", ""),
                "title": title,
                "authors": authors,
                "year": work.get("publication_year", 0),
                "abstract": abstract,
                "venue": work.get("primary_location", {}).get("source", {}).get("display_name", "") if work.get("primary_location") and work["primary_location"].get("source") else "",
                "citationCount": work.get("cited_by_count", 0),
                "source": "openalex_random",
            })

        print(f"  Page {page}: fetched {len(results)} works, total papers: {len(papers)}")
        page += 1
        time.sleep(1.0)

    papers = papers[:target]
    print(f"  Collected {len(papers)} random papers")
    return papers


def call_gpt4o(prompt: str, api_key: str) -> list[dict]:
    openai = importlib.import_module("openai")
    client = openai.OpenAI(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                seed=42,
            )
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Retry {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(RETRY_DELAY * attempt)

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                if isinstance(parsed, dict) and "assumptions" in parsed:
                    return parsed["assumptions"]
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON: {text[:300]}")


def compute_pairwise_similarity(assumptions: list[str]) -> dict:
    """Compute average pairwise similarity among all assumptions."""
    if len(assumptions) < 2:
        return {"avg_pairwise_similarity": 0.0, "max_pairwise_similarity": 0.0}

    embeddings = embed_texts(assumptions)
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))

    return {
        "avg_pairwise_similarity": round(sum(sims) / len(sims), 4),
        "max_pairwise_similarity": round(max(sims), 4),
        "min_pairwise_similarity": round(min(sims), 4),
        "num_pairs": len(sims),
    }


def compute_semantic_redundancy(assumptions: list[str], threshold: float = 0.85) -> dict:
    """Compute semantic redundancy: how many assumptions are near-duplicates."""
    if len(assumptions) < 2:
        return {"semantic_unique": len(assumptions), "semantic_redundancy_rate": 0.0}

    embeddings = embed_texts(assumptions)
    # Greedy deduplication: keep assumption if no kept assumption has sim > threshold
    kept_indices = [0]
    for i in range(1, len(embeddings)):
        is_dup = False
        for j in kept_indices:
            if cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)

    return {
        "total": len(assumptions),
        "semantic_unique": len(kept_indices),
        "semantic_redundancy_rate": round(1.0 - len(kept_indices) / len(assumptions), 4),
    }


def main():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY required")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch random papers
    papers = fetch_random_papers(60)

    # Save collected papers
    papers_path = OUTPUT_DIR / "random_papers.jsonl"
    with open(papers_path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    print(f"Saved papers to {papers_path}")

    # Step 2: Run extraction on first 15 papers
    subset = papers[:15]
    print(f"\nRunning extraction on {len(subset)} random papers...")

    all_assumptions = []
    for i, paper in enumerate(subset):
        title = str(paper.get("title", ""))
        abstract = str(paper.get("abstract", ""))
        paper_text = f"Title: {title}\nAbstract: {abstract}"
        prompt = CLEAN_PROMPT_TEMPLATE.format(paper_text=paper_text)

        print(f"  [{i+1}/{len(subset)}] {title[:60]}...", flush=True)
        try:
            raw = call_gpt4o(prompt, openai_key)
            for item in raw:
                if isinstance(item, dict):
                    item["source_paper"] = title
                    all_assumptions.append(item)
                else:
                    all_assumptions.append({
                        "assumption": str(item),
                        "confidence": 0.5,
                        "category": "unknown",
                        "source_paper": title,
                    })
        except Exception as exc:
            print(f"    ERROR: {exc}")
        time.sleep(1.5)

    # Step 3: Compute redundancy
    print(f"\nComputing redundancy metrics...")

    # Exact dedup
    seen = set()
    exact_unique = []
    for a in all_assumptions:
        key = a["assumption"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            exact_unique.append(a)

    raw_count = len(all_assumptions)
    exact_unique_count = len(exact_unique)
    exact_redundancy = round(1.0 - exact_unique_count / max(1, raw_count), 4)

    assumption_texts = [a["assumption"] for a in exact_unique]

    # Semantic redundancy
    sem_redundancy = compute_semantic_redundancy(assumption_texts, threshold=0.85)

    # Pairwise similarity
    pairwise = compute_pairwise_similarity(assumption_texts)

    # Step 4: Compare with focused corpus
    # Load transformer focused corpus results for comparison
    focused_path = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt" / "transformer.json"
    focused_comparison = {}
    if focused_path.exists():
        focused_data = json.loads(focused_path.read_text(encoding="utf-8"))
        focused_raw = focused_data.get("num_raw_assumptions", 0)
        focused_unique = focused_data.get("num_unique_assumptions", 0)
        focused_redundancy = round(1.0 - focused_unique / max(1, focused_raw), 4)
        focused_comparison = {
            "focused_raw_count": focused_raw,
            "focused_exact_unique": focused_unique,
            "focused_exact_redundancy": focused_redundancy,
        }

    # Step 5: Category diversity analysis
    categories = {}
    for a in exact_unique:
        cat = a.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    # Save results
    result = {
        "experiment": "random_corpus_control",
        "model": MODEL,
        "num_papers_collected": len(papers),
        "num_papers_extracted": len(subset),
        "paper_topics": [p.get("title", "")[:80] for p in subset],
        "raw_assumption_count": raw_count,
        "exact_unique_count": exact_unique_count,
        "exact_redundancy_rate": exact_redundancy,
        "semantic_redundancy": sem_redundancy,
        "pairwise_similarity": pairwise,
        "category_distribution": categories,
        "focused_corpus_comparison": focused_comparison,
        "interpretation": "",
        "assumptions": exact_unique,
        "ran_at": datetime.now(UTC).isoformat(),
    }

    # Generate interpretation
    if focused_comparison:
        if abs(exact_redundancy - focused_comparison.get("focused_exact_redundancy", 0)) < 0.1:
            result["interpretation"] = (
                f"Random corpus redundancy ({exact_redundancy:.1%}) is similar to focused corpus "
                f"({focused_comparison.get('focused_exact_redundancy', 0):.1%}). "
                f"This suggests LLM homogeneity: the model produces similar assumption patterns "
                f"regardless of input specificity."
            )
        else:
            result["interpretation"] = (
                f"Random corpus redundancy ({exact_redundancy:.1%}) differs from focused corpus "
                f"({focused_comparison.get('focused_exact_redundancy', 0):.1%}). "
                f"Lower redundancy in random corpus suggests the focused corpus captures "
                f"genuine field consensus rather than LLM homogeneity."
            )

    result_path = OUTPUT_DIR / "results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("RANDOM CORPUS CONTROL RESULTS")
    print(f"{'='*60}")
    print(f"  Papers collected: {len(papers)}")
    print(f"  Papers extracted: {len(subset)}")
    print(f"  Raw assumptions: {raw_count}")
    print(f"  Exact unique: {exact_unique_count}")
    print(f"  Exact redundancy: {exact_redundancy:.1%}")
    print(f"  Semantic unique (0.85 threshold): {sem_redundancy['semantic_unique']}")
    print(f"  Semantic redundancy: {sem_redundancy['semantic_redundancy_rate']:.1%}")
    print(f"  Avg pairwise similarity: {pairwise['avg_pairwise_similarity']:.4f}")
    if focused_comparison:
        print(f"\n  --- Comparison with focused corpus ---")
        print(f"  Focused raw: {focused_comparison.get('focused_raw_count')}")
        print(f"  Focused unique: {focused_comparison.get('focused_exact_unique')}")
        print(f"  Focused redundancy: {focused_comparison.get('focused_exact_redundancy', 0):.1%}")
    print(f"\n  Category distribution: {categories}")
    print(f"\n  Interpretation: {result['interpretation']}")
    print(f"\n  Saved: {result_path}")


if __name__ == "__main__":
    main()
