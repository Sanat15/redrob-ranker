#!/usr/bin/env python3
"""
Redrob Hackathon — Main ranking pipeline
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints respected:
- CPU only (no GPU)
- No external API calls during ranking
- < 5 minutes on 16GB RAM for 100K candidates
- Produces validate_submission.py-passing CSV
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

from scorer import score_candidate


def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz file, streaming."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    encoding = "utf-8"

    count = 0
    with opener(p, mode, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
                if count % 10000 == 0:
                    print(f"  Loaded {count:,} candidates...", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {count}: {e}", file=sys.stderr)


def rank_candidates(candidates_path: str, top_n: int = 100) -> list[dict]:
    """
    Score all candidates and return top N sorted by score descending.
    Uses a streaming approach — never loads all 100K into memory as scored objects.
    """
    # We keep only the top N during streaming using a simple list + min-heap
    # For 100K candidates at ~1KB each unscored, this is fine.
    # Scores are cheap (pure Python math), so we can store all scored results.

    print("Scoring candidates...", file=sys.stderr)
    t0 = time.time()

    scored = []
    for i, candidate in enumerate(load_candidates(candidates_path)):
        result = score_candidate(candidate)
        scored.append({
            "candidate_id": candidate["candidate_id"],
            "score": result["score"],
            "reasoning": result["reasoning"],
            "is_honeypot": result["is_honeypot"],
        })

    elapsed = time.time() - t0
    print(f"Scored {len(scored):,} candidates in {elapsed:.1f}s", file=sys.stderr)

    # Sort by score descending; tie-break by candidate_id ascending (as per spec)
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    top = scored[:top_n]

    # Log honeypot rate in top 100 for self-check
    honeypots_in_top = sum(1 for c in top if c["is_honeypot"])
    print(f"Honeypots in top {top_n}: {honeypots_in_top} ({honeypots_in_top}%)", file=sys.stderr)
    if honeypots_in_top > 10:
        print("WARNING: honeypot rate > 10% — submission will be disqualified at Stage 3!", file=sys.stderr)

    return top


def write_submission(ranked: list[dict], output_path: str) -> None:
    """Write the submission CSV in the format required by submission_spec."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        # Assign ranks 1–100; ensure scores are non-increasing
        prev_score = None
        for i, entry in enumerate(ranked):
            rank = i + 1
            score = entry["score"]

            # Enforce non-increasing score (spec requirement)
            if prev_score is not None and score > prev_score:
                score = prev_score
            prev_score = score

            writer.writerow([
                entry["candidate_id"],
                rank,
                f"{score:.6f}",
                entry["reasoning"],
            ])

    print(f"Submission written to: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Redrob hackathon candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path (e.g. team_xxx.csv)")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to include (default: 100)")
    args = parser.parse_args()

    print(f"Loading candidates from: {args.candidates}", file=sys.stderr)
    t_start = time.time()

    ranked = rank_candidates(args.candidates, top_n=args.top)
    write_submission(ranked, args.out)

    total = time.time() - t_start
    print(f"Done in {total:.1f}s total", file=sys.stderr)
    print(f"Top 5 preview:", file=sys.stderr)
    for entry in ranked[:5]:
        print(f"  #{ranked.index(entry)+1} {entry['candidate_id']} | score={entry['score']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
