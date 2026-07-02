# redrob-ranker

Rule-based AI candidate ranking system built for the **Redrob Intelligent Candidate
Discovery Challenge** (Hack2Skill — "The Data & AI Challenge").

Given a job description and 100,000 candidate profiles, this pipeline outputs the
top 100 best-fit candidates as a ranked CSV — the way a great recruiter would
shortlist them, not by keyword-matching skill lists.

**Live sandbox:** https://redrob-ranker-yzva3g7w7d9mkg52simzn4.streamlit.app/

---

## The problem

> "The right answer is not 'find candidates whose skills section contains the most
> AI keywords.' That's a trap we've explicitly built into the dataset."
> — from the job description

The dataset is adversarial by design:

- **Non-tech candidates carry ML keywords.** Marketing Managers and Accountants have
  "Embeddings" and "PyTorch" in their skills array with `proficiency: expert` —
  but 0 endorsements and 0 months of usage.
- **Career descriptions are recycled templates.** The same paragraph about shipping
  ranking models appears verbatim under "Civil Engineer," "Marketing Manager," and
  "Accountant." Scanning descriptions for keywords adds noise, not signal.
- **Behavioral fit matters as much as technical fit.** A perfect-on-paper candidate
  who hasn't logged in for 6 months and ghosts 95% of recruiter messages isn't,
  for hiring purposes, actually available.

We built a transparent, rule-based scorer instead of an ML model, because there is
no labeled ground truth to train one on — and every rule below is something we can
defend line-by-line.

---

## Scoring formula

```
base = (
    0.12 × title_score        +
    0.15 × career_domain_score +
    0.08 × company_fit_score  +
    0.30 × skills_score        +
    0.15 × experience_score    +
    0.10 × location_score      +
    0.02 × education_score
) + production_evidence_bonus (up to +0.12)

final_score = base
    × behavioral_multiplier         (0.25×–1.02×)
    × high_throughput_bonus         (1.0×–1.15×, gated — see below)
    × hard_gate_multiplier          (1.0×, or a heavy penalty — see below)
    × integrity_reliability_factor  (0.80×–1.00×)
```

`title_career_score` from earlier versions is now split into three
independently-weighted components (`title`, `career_domain`, `company_fit`)
so that a strong title alone can't carry a weak career history, and vice
versa.

### Skills trust formula (the anti-keyword-stuffing mechanism)

```
trust = proficiency_weight × min(1, endorsements / 25) × min(1, duration_months / 18)

proficiency_weight:  expert=1.0   advanced=0.75   intermediate=0.4   beginner=0.15
```

An `expert` skill with `duration_months = 0` scores **0.0 trust, not 1.0**. Someone
who lists twenty "expert" skills they've never actually used gets credit for none
of them.

### Behavioral multiplier (multiplicative, 0.25×–1.10×)

Applied to the whole base score, not added to it — an unavailable great candidate
should score worse than an available good one.

| Signal | Effect |
|---|---|
| `open_to_work_flag = false` | ×0.5 |
| Inactive 180+ days | ×0.6 |
| Inactive 90–180 days | ×0.8 |
| Recruiter response rate < 10% | ×0.7 |
| Recruiter response rate < 25% | ×0.85 |
| Notice period > 150 days | ×0.70 |
| Notice period > 90 days | ×0.85 |
| GitHub activity ≥ 60 | ×1.10 |
| Interview completion rate < 0.4 | ×0.85 |
| Interview completion rate < 0.6 | ×0.92 |
| Floor | max(0.25, multiplier) |

### Hard gates (`hard_gate_multiplier`, v4) — JD-explicit disqualifiers

Component weighting alone wasn't strong enough to keep these out of a top-100
slot — a full 100K audit found a pure-Accountant profile reaching 0.36 and a
pure-consulting-career profile reaching 0.45 on the *other* 80–90% of the
weighted sum, both uncomfortably close to the ~0.53 rank-100 cutoff, because
career descriptions in this dataset are recycled across unrelated titles
(see "The problem" above). These are explicit multiplicative gates applied
on top of the base score, not just a zeroed component:

| Disqualifier | Gate | JD language |
|---|---|---|
| Non-tech current title (Marketing, Sales, HR, Ops, Accountant, …) | ×0.15 | "we will not move forward" |
| Entire career at a pure consulting firm (TCS, Wipro, Infosys, Accenture, Cognizant, Capgemini, …) | ×0.35 | "we've had bad fit experiences in both directions" |
| Outside India, not willing to relocate, and prefers onsite/hybrid work | ×0.55 | "case-by-case" (deliberately the mildest gate — see rationale in `scorer.py`) |
| Career dominated by computer vision/speech/robotics with **zero** retrieval/ranking/recommendation evidence anywhere | ×0.80 | "primary expertise [CV/speech/robotics] without significant NLP/IR exposure" |

None of these zero the score outright (nothing about the profile is
fabricated/impossible, unlike honeypots) — they push the candidate far
enough down that no combination of other strengths recovers a top-100 slot.
The CV/speech gate matches on **career text** (title + description across
the whole career), not the skills list, so one stray "OpenCV" skill entry
doesn't trigger it — only a career with zero retrieval/ranking/recsys
evidence anywhere does.

### Other v4 fixes

- **Title-chaser job-hopping** (`_career_progression_bonus`): a title-level
  climb (Engineer → Senior → Staff → …) earned via a string of sub-18-month
  stints no longer earns the progression bonus — the JD explicitly calls
  this pattern out ("optimizing for titles by switching companies every 1.5
  years"). Genuine progression (staying and growing into a role) still does.
- **`high_throughput_bonus` gated**: previously applied unconditionally, so
  mentioning "Redis" or "1M QPS" in any job — including an unrelated one —
  bought a score multiplier. Now gated behind the same title/skills
  relevance bar as the production-evidence bonus.
- **NLP as a scored skill capability**: "NLP" / "Natural Language
  Processing" skill entries previously earned zero skill-level credit
  anywhere in `skills_score()` (only the bare word affected `title_score`'s
  strong-title-terms check). Added to the `llm` capability group.
- **Location tiering refined to the JD's actual text**: "Candidates in
  Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply" — Pune/Noida (the
  JD's HQ cities) score 1.0, the four explicitly-named cities score 0.95,
  other India tech hubs (Bangalore, Chennai, Kolkata) score 0.90.

### Honeypot detection → score 0.01 (run before any other scoring)

1. Any skill with `proficiency = expert` and `duration_months = 0`
2. ≥ 3 skills that are `expert` + 0 endorsements + < 6 months duration
3. Sum of career-history months exceeds `years_of_experience × 12 + 18` by a wide margin

### Soft adjustments

- **Salary fit:** base score ×0.93 if expected salary max > ₹85 LPA (above likely
  Series A budget); ×0.95 if max < ₹8 LPA (possible data error or very junior).
- **`saved_by_recruiters_30d`:** used only as a **sort tiebreaker** on equal scores,
  not as a score multiplier. (Tested as a multiplier first — it tied 10 candidates
  at exactly 1.0000 and broke top-10 differentiation, which directly hurts NDCG@10.
  See `person_a_day2_progress.md` for the full writeup.)

---

## Repository structure

| File | Purpose |
|---|---|
| `scorer.py` | Core scoring engine — five components + behavioral multiplier + honeypot detection + reasoning generator |
| `rank.py` | CLI pipeline — streams candidates, keeps a 100-entry min-heap, writes the submission CSV |
| `diagnose.py` | Full-100K diagnostic — score distribution, problem checks (honeypot rate, non-tech titles in top 100, etc.) |
| `test_scorer.py` | 89 unit tests across 18 test classes, incl. v3 semantic matching, production-evidence, and v4 hard-gate tests |
| `app.py` | Streamlit sandbox — upload up to 100 candidate profiles, see ranked output + score breakdown live |
| `submission_metadata.yaml` | Filled submission metadata (team, approach, design decisions, technical specs) |
| `Dockerfile` | Reproducible container for the Stage 3 evaluation run |
| `requirements.txt` | `streamlit`, `pandas` (sandbox only — the ranking pipeline itself is pure stdlib) |

---

## Usage

```bash
git clone https://github.com/Sanat15/redrob-ranker.git
cd redrob-ranker
pip install -r requirements.txt

# Run the ranker on the full dataset
python rank.py --candidates candidates.jsonl --out team_yaks.csv

# Validate the output against the organizer's spec
python validate_submission.py team_yaks.csv
```

Also works transparently on a gzipped dataset (`candidates.jsonl.gz`) — `load_candidates()`
branches on file extension, no flag needed.

**Run the test suite:**
```bash
python -m pytest test_scorer.py -v
```

**Run the local diagnostic** (score distribution + sanity checks on the full 100K):
```bash
python diagnose.py --candidates candidates.jsonl
```

**Run the sandbox locally:**
```bash
streamlit run app.py
```

---

## Performance

| Constraint | Budget | Measured |
|---|---|---|
| Wall-clock time (100K candidates) | ≤ 5 minutes | **~70–90 seconds** (full local run, incl. v4 gates) |
| RAM | ≤ 16 GB | Only top-100 entries ever held in memory (heapq) |
| GPU | None used | CPU only |
| External API calls during ranking | None | None — pure stdlib Python |

`rank.py` uses a min-heap (`heapq`) rather than a full sort: O(N log K) instead of
O(N log N), with only `K=100` entries in memory at any time regardless of dataset size.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Rule-based, not an ML model | No labeled training data exists; rules are transparent and defensible at the Stage 5 interview |
| No career-description scanning | Descriptions are recycled templates across unrelated candidates — scanning them adds noise, not signal |
| Skills trust formula over presence check | Defeats keyword stuffing — endorsed, long-duration skills are what "real expertise" looks like |
| Behavioral multiplier is multiplicative | Severely penalizes unavailability rather than averaging it away |
| `saved_by_recruiters_30d` as tiebreaker only | Using it as a score multiplier collapsed 10 candidates to an identical 1.0000, destroying NDCG@10 differentiation |
| `heapq` min-heap | O(N log K) vs. O(N log N); constant ~100-entry memory footprint |
| Education weighted at only 2% | The JD states no educational requirement; production experience matters far more |
| Production-evidence bonus cap at 0.12 | Verified production ranking/retrieval ownership is the strongest signal for the role |
| Concept-level semantic matching | Production-ML vocabulary (XGBoost, discovery feed, optimization target) now matches IR equivalents |
| Hard gates as separate multipliers, not zeroed components (v4) | A full-100K audit found non-tech-title and consulting-only profiles reaching 0.36–0.45 on the *other* 80–90% of component weight — close enough to the ~0.53 rank-100 cutoff to be a real risk, not a theoretical one |
| Location gate is the mildest of the three (v4) | JD says "case-by-case" for outside-India candidates, not "we will not move forward" (reserved for title/consulting); our own manual review calibration (`rank.md`) also placed one such candidate mid-tier rather than deep-tail |
| `high_throughput_bonus` gated behind title/skills relevance (v4) | Previously unconditional — mentioning "Redis" in an unrelated job shouldn't buy a score bump |

---

## Results snapshot

Full 100K run (scorer v4), top 10 of 100:

| Rank | Score | ID |
|---|---|---|
| 1 | 0.8633 | CAND_0018499 |
| 2 | 0.8268 | CAND_0081846 |
| 3 | 0.7894 | CAND_0008425 |
| 4 | 0.7846 | CAND_0077337 |
| 5 | 0.7310 | CAND_0079387 |
| 6 | 0.7244 | CAND_0041669 |
| 7 | 0.7185 | CAND_0041610 |
| 8 | 0.7159 | CAND_0066376 |
| 9 | 0.7096 | CAND_0088025 |
| 10 | 0.7085 | CAND_0064326 |

**v3 improvements:** Semantic matching expanded to recognize production-ML vocabulary
(XGBoost/LightGBM, discovery feed, optimization target, offline-online correlation,
drift detection, retraining cadence) as equivalent to IR jargon; education weight
reduced from 10% → 5%; production-evidence bonus cap increased 7% → 12%.

**v4 improvements (this pass):** a full 100K-candidate audit (`audit_full.py`) found
three JD-explicit disqualifiers reaching uncomfortably-close-to-top-100 scores on
component weight alone — non-tech titles (best case 0.36 at rank 894), pure-consulting
careers (0.45 at rank 232), and outside-India/non-relocating/onsite candidates (4 of
them *inside* the actual top 100, at ranks 71–88). Added explicit multiplicative hard
gates for all three, plus a fourth for the previously-unimplemented CV/speech/robotics-
without-NLP/IR exclusion; fixed a job-hopping "title-chaser" loophole in the career-
progression bonus; closed a loophole where `high_throughput_bonus` applied even to
unrelated roles; added NLP as a scored skill capability (previously zero-credit); and
refined location tiering to match the JD's actual named cities. Net effect: the 4
logistically-non-viable candidates are gone from the top 100, replaced by 4 verified
India-based candidates; best-case score for each disqualifier trap dropped by 30–130x
its former rank; 0 honeypots and 0 non-tech titles remain in the top 2,000 (previously
0 non-tech titles in top 500 but 76 within top 2,000 — now 0).

Sanity checks on the top 100 (see `audit_full.py`): 0 honeypots (nearest honeypot rank
84,820 of 100,000), 0 non-tech titles, 0 pure-consulting careers, 0 logistically-
impossible locations, spread 0.53–0.86.

Full methodology and the rest of the top 100 are in the progress logs and the
submission PDF deck.

---

## Team

| Member | Role |
|---|---|
| **Sanat** ([@Sanat15](https://github.com/Sanat15)) | Infrastructure, pipeline, sandbox, submission mechanics |
| **Tushar** ([@IITI-tushar](https://github.com/IITI-tushar))| Scoring architecture & algorithm design |

## License

MIT — see `LICENSE`.
