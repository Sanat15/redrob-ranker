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
import heapq
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


def rank_candidates(candidates_path, top_n=100):
    heap = []
    print("Scoring candidates...", file=sys.stderr)
    t0 = time.time()

    for candidate in load_candidates(candidates_path):
        result = score_candidate(candidate)
        score = result["score"]

        # NEW (Day 2 → Day 3 handoff item from person_a_day2_progress.md §3):
        # saved_by_recruiters_30d is used ONLY as a sort tiebreaker, never as a
        # score multiplier. As a multiplier it tied 10 candidates at exactly
        # 1.0000 (1.10 github bonus × 1.08 saved bonus on a 0.90 base, capped
        # at 1.0), which collapsed top-10 ordering to alphabetical-by-id —
        # the worst possible outcome given NDCG@10 is 50% of the score.
        saved_30d = candidate.get("redrob_signals", {}).get("saved_by_recruiters_30d", 0)

        entry = {
            "candidate_id": candidate["candidate_id"],
            "score": score,
            "reasoning": result["reasoning"],
            "is_honeypot": result["is_honeypot"],
            "saved_30d": saved_30d,
        }
        if len(heap) < top_n:
            heapq.heappush(heap, (score, candidate["candidate_id"], entry))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, candidate["candidate_id"], entry))

    elapsed = time.time() - t0
    print(f"Scored in {elapsed:.1f}s", file=sys.stderr)

    # Sort: score descending -> candidate_id ascending (per spec tie-break rule).
    top = sorted(heap, key=lambda x: (-x[0], x[1]))
    return [x[2] for x in top]


def write_submission(ranked: list[dict], output_path: str) -> None:
    """Write the submission CSV in the format required by submission_spec."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        # Assign ranks 1-100; ensure scores are non-increasing
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
    print("Top 5 preview:", file=sys.stderr)
    for i, entry in enumerate(ranked[:5], 1):
        print(f"  #{i} {entry['candidate_id']} | score={entry['score']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
