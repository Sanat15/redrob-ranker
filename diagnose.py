"""
Diagnostic script — Person A Day 2
Run: python diagnose.py --candidates ./candidates.jsonl.gz
"""

import argparse
import gzip
import json
import sys
from scorer import score_candidate, is_honeypot

def load_candidates(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    args = parser.parse_args()

    print("Scoring all candidates...", file=sys.stderr)
    scored = []
    for c in load_candidates(args.candidates):
        r = score_candidate(c)
        scored.append({
            "candidate_id": c["candidate_id"],
            "score": r["score"],
            "title": c["profile"]["current_title"],
            "company": c["profile"]["current_company"],
            "country": c["profile"]["country"],
            "city": c["profile"]["location"],
            "yoe": c["profile"]["years_of_experience"],
            "open_to_work": c["redrob_signals"]["open_to_work_flag"],
            "notice": c["redrob_signals"]["notice_period_days"],
            "response_rate": c["redrob_signals"]["recruiter_response_rate"],
            "last_active": c["redrob_signals"]["last_active_date"],
            "salary_max": c["redrob_signals"]["expected_salary_range_inr_lpa"]["max"],
            "saved_30d": c["redrob_signals"]["saved_by_recruiters_30d"],
            "github": c["redrob_signals"]["github_activity_score"],
            "components": r["components"],
            "multiplier": r["multiplier"],
            "is_honeypot": r["is_honeypot"],
            "reasoning": r["reasoning"],
        })

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    print("\n" + "="*80)
    print("RANKS 1–50 FULL AUDIT")
    print("="*80)
    for i, c in enumerate(scored[:50], 1):
        flag = ""
        if c["country"] != "India":           flag += " [NON-INDIA]"
        if not c["open_to_work"]:             flag += " [CLOSED]"
        if c["yoe"] < 4:                      flag += " [LOW-YOE]"
        if c["notice"] > 90:                  flag += " [LONG-NOTICE]"
        if c["salary_max"] > 80:              flag += " [HIGH-SALARY]"
        if c["is_honeypot"]:                  flag += " [HONEYPOT!]"

        print(f"\nRank {i:3d} | {c['score']:.4f} | {c['candidate_id']}")
        print(f"  {c['title']} @ {c['company']}")
        print(f"  {c['city']}, {c['country']} | {c['yoe']}yr | notice:{c['notice']}d | "
              f"resp:{c['response_rate']:.0%} | salary_max:{c['salary_max']}LPA | "
              f"saved_30d:{c['saved_30d']} | github:{c['github']}{flag}")
        comp = c['components']
        print(f"  tt:{comp.get('title',0):.2f} cd:{comp.get('career_domain',0):.2f} "
              f"cf:{comp.get('company_fit',0):.2f} sk:{comp.get('skills',0):.2f} "
              f"ex:{comp.get('experience',0):.2f} lo:{comp.get('location',0):.2f} "
              f"ed:{comp.get('education',0):.2f} pe:{comp.get('production_evidence',0):.2f} "
              f"bm:{c['multiplier']:.2f}")
        print(f"  Reasoning: {c['reasoning'][:120]}")

    print("\n" + "="*80)
    print("PROBLEM CHECKS")
    print("="*80)

    top100 = scored[:100]

    # Check 1: Non-tech titles in top 100
    non_tech_keywords = ["marketing","sales","hr ","customer support","operations manager",
                         "accountant","content","finance"]
    non_tech_found = [c for c in top100
                      if any(k in c["title"].lower() for k in non_tech_keywords)]
    print(f"\n[1] Non-tech titles in top 100: {len(non_tech_found)}")
    for c in non_tech_found:
        print(f"    Rank {top100.index(c)+1}: {c['title']} @ {c['company']} | score:{c['score']:.4f}")

    # Check 2: Non-India without relocation
    non_india = [c for c in top100 if c["country"] != "India"]
    print(f"\n[2] Non-India candidates in top 100: {len(non_india)}")
    for c in non_india:
        print(f"    Rank {top100.index(c)+1}: {c['country']} | {c['title']} | score:{c['score']:.4f}")

    # Check 3: Honeypots
    honeypots = [c for c in top100 if c["is_honeypot"]]
    print(f"\n[3] Honeypots in top 100: {len(honeypots)} (limit: 10)")

    # Check 4: Low YoE in top 20
    low_yoe = [c for c in top100[:20] if c["yoe"] < 4]
    print(f"\n[4] Candidates with < 4 YoE in top 20: {len(low_yoe)}")
    for c in low_yoe:
        print(f"    Rank {top100.index(c)+1}: {c['yoe']}yr | {c['title']} | score:{c['score']:.4f}")

    # Check 5: High salary expectations
    high_sal = [c for c in top100 if c["salary_max"] > 80]
    print(f"\n[5] Candidates expecting > 80 LPA in top 100: {len(high_sal)}")
    for c in high_sal[:5]:
        print(f"    Rank {top100.index(c)+1}: max {c['salary_max']}LPA | {c['title']} | score:{c['score']:.4f}")

    # Check 6: saved_by_recruiters_30d distribution
    saved_vals = sorted([c["saved_30d"] for c in top100], reverse=True)
    print(f"\n[6] saved_by_recruiters_30d in top 100:")
    print(f"    max:{saved_vals[0]} min:{saved_vals[-1]} "
          f"avg:{sum(saved_vals)/len(saved_vals):.1f}")
    print(f"    Top 5 saved values: {saved_vals[:5]}")

    # Check 7: Reasoning variation — sample 10 random rows
    import random
    random.seed(42)
    sample10 = random.sample(top100, 10)
    print(f"\n[7] Reasoning variation check (10 random from top 100):")
    reasonings = [c["reasoning"] for c in sample10]
    unique_starts = len(set(r[:40] for r in reasonings))
    print(f"    Unique reasoning starts (first 40 chars): {unique_starts}/10")
    if unique_starts < 8:
        print("    WARNING: Too many similar reasonings — Stage 4 will flag this")
    for r in reasonings[:3]:
        print(f"    Sample: {r[:100]}")

    print("\n" + "="*80)
    print("SCORE DISTRIBUTION")
    print("="*80)
    milestones = [1, 5, 10, 20, 50, 100]
    for m in milestones:
        print(f"  Score at rank {m:3d}: {scored[m-1]['score']:.4f}")

if __name__ == "__main__":
    main()