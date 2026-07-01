#!/usr/bin/env python3
"""
Benchmarking script for ranker & scorer on full 100K candidate dataset
Measures execution time and throughput for the Redrob AI candidate ranking pipeline (v3)

Scorer v3 improvements:
  - Concept-level semantic matching: production-ML vocabulary now matches IR equivalents
  - Production-evidence bonus cap increased 0.07 → 0.12
  - Education weight reduced 0.10 → 0.05
  - Offline-online correlation uses co-occurrence heuristic
"""
import json
import time
import sys
from pathlib import Path
from scorer import score_candidate
from rank import load_candidates, rank_candidates


def benchmark_full_pipeline(candidates_path, sample_scoring=1000):
    """Benchmark the full ranking pipeline on actual data."""
    print("\n" + "=" * 70)
    print("🚀 FULL PIPELINE BENCHMARK (100K Candidates)")
    print("=" * 70)

    candidates_path = Path(candidates_path)
    if not candidates_path.exists():
        print(f"❌ File not found: {candidates_path}")
        sys.exit(1)

    print(f"\n📁 Input: {candidates_path.name}")
    print(f"   Size: {candidates_path.stat().st_size / (1024**2):.1f} MB")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1: Sample scoring performance (measure per-candidate time)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n📊 PHASE 1: Single Scorer Performance ({sample_scoring} samples)")
    print("─" * 70)

    score_times = []
    sample_count = 0

    t0 = time.perf_counter()

    with open(candidates_path) as f:
        for i, line in enumerate(f):
            if sample_count >= sample_scoring:
                break

            line = line.strip()
            if not line:
                continue

            candidate = json.loads(line)

            t_start = time.perf_counter()
            result = score_candidate(candidate)
            t_end = time.perf_counter()

            score_times.append((t_end - t_start) * 1000)  # ms
            sample_count += 1

            if (i + 1) % 200 == 0:
                avg = sum(score_times) / len(score_times)
                print(f"  Sampled {sample_count:4d} candidates | "
                      f"Last: {score_times[-1]:6.2f}ms | "
                      f"Avg: {avg:6.2f}ms | "
                      f"Throughput: {1000/avg:6.1f}/sec")

    t_sample_end = time.perf_counter()
    sample_time = t_sample_end - t0

    avg_score_time = sum(score_times) / len(score_times) if score_times else 0
    estimated_100k_time = (100000 * avg_score_time) / 1000

    print("─" * 70)
    print(f"  Min time:              {min(score_times):7.2f}ms")
    print(f"  Max time:              {max(score_times):7.2f}ms")
    print(f"  Avg time:              {avg_score_time:7.2f}ms")
    print(f"  Throughput:            {1000/avg_score_time:6.1f} candidates/sec")
    print(f"  Actual sample time:    {sample_time:7.2f}s")
    print(f"  Estimated 100K time:   {estimated_100k_time:7.1f}s ({estimated_100k_time/60:.1f}m)")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2: Full ranking on complete 100K dataset
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n📊 PHASE 2: Full Ranking Pipeline (All 100K candidates)")
    print("─" * 70)

    t_start = time.perf_counter()

    ranked = rank_candidates(str(candidates_path), top_n=100)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    throughput = 100000 / total_time if total_time > 0 else 0
    per_candidate = (total_time / 100000) * 1000

    print(f"  Total time:            {total_time:7.2f}s ({total_time/60:.2f}m)")
    print(f"  Per candidate:         {per_candidate:7.2f}ms")
    print(f"  Throughput:            {throughput:6.1f} candidates/sec")

    print(f"\n  Top 10 ranked candidates:")
    for i, entry in enumerate(ranked[:10], 1):
        print(f"    #{i:2d} {entry['candidate_id']} | score={entry['score']:.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Scorer per-candidate:  {avg_score_time:7.2f}ms")
    print(f"  Full pipeline (100K):  {total_time:7.2f}s")
    print(f"  Overall throughput:    {throughput:6.1f} candidates/sec")
    print(f"  Within 5-min target:   {'✅ YES' if total_time <= 300 else '❌ NO'}")
    print("=" * 70 + "\n")


def benchmark_rank_only(candidates_path):
    """Benchmark just the ranking pipeline without detailed scoring breakdown."""
    print("\n" + "=" * 70)
    print("⚡ QUICK RANKING ONLY (No per-candidate timing)")
    print("=" * 70)

    print(f"\n📁 Input: {Path(candidates_path).name}")

    t_start = time.perf_counter()
    ranked = rank_candidates(candidates_path, top_n=100)
    t_end = time.perf_counter()

    total_time = t_end - t_start
    throughput = 100000 / total_time if total_time > 0 else 0

    print(f"\n  Total time:   {total_time:7.2f}s ({total_time/60:.2f}m)")
    print(f"  Throughput:   {throughput:6.1f} candidates/sec")
    print("\n  Top 5 results:")
    for i, entry in enumerate(ranked[:5], 1):
        print(f"    #{i} {entry['candidate_id']} | score={entry['score']:.6f}")

    print("=" * 70 + "\n")
    return total_time, throughput


def main():
    candidates_path = Path(
        "/home/tushar/redrob-ranker/[PUB] India_runs_data_and_ai_challenge (2)/"
        "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.json"
    )

    if not candidates_path.exists():
        print(f"❌ Candidates file not found: {candidates_path}")
        sys.exit(1)

    try:
        # Run full benchmark with sampling
        benchmark_full_pipeline(str(candidates_path), sample_scoring=1000)

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
