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
final_score = (
    0.35 × title_career_score  +
    0.30 × skills_score        +
    0.15 × experience_score    +
    0.10 × location_score      +
    0.10 × education_score
) × behavioral_multiplier
```

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
| Inactive 90–180 / 180+ days | ×0.8 / ×0.6 |
| Recruiter response rate < 10% / < 25% | ×0.7 / ×0.85 |
| Notice period > 150 / > 90 days | ×0.70 / ×0.85 |
| GitHub activity ≥ 60 / ≥ 30 | ×1.10 / ×1.05 |
| Interview completion rate < 0.4 / < 0.6 | ×0.85 / ×0.92 |
| Floor | max(0.25, multiplier) |

### Hard disqualifiers → score 0.0

- Non-tech current title (Marketing, Sales, HR, Ops, Accountant, …)
- Entire career at a pure consulting firm (TCS, Wipro, Infosys, Accenture, Cognizant, Capgemini, …)
- Outside India, not willing to relocate, and prefers onsite/hybrid work

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
| `test_scorer.py` | 59 unit tests across 7 test classes, incl. a regression anchor on the known rank-1 candidate |
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
| Wall-clock time (100K candidates) | ≤ 5 minutes | **~10–13 seconds** |
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
| Consulting-firm penalty ×0.4 | Directly encodes the JD's explicit disqualifier for consulting-only careers |
| Education weighted at only 10% | The JD states no educational requirement; production experience matters far more |

---

## Results snapshot

Full 100K diagnostic (scorer v2.1), top 10 of 100:

| Rank | Score | YoE | Title | Company | City |
|---|---|---|---|---|---|
| 1 | 0.9866 | 5.4 | NLP Engineer | Glance | Chandigarh |
| 2 | 0.9658 | 6.9 | AI Engineer | Microsoft | Trivandrum |
| 3 | 0.9630 | 7.8 | Senior AI Engineer | Netflix | Vizag |
| 4 | 0.9597 | 7.2 | Senior ML Engineer | Zomato | Noida |
| 5 | 0.9486 | 6.5 | Recommendation Systems Eng | Amazon | Pune |
| 6 | 0.9479 | 8.6 | Staff ML Engineer | Yellow.ai | Jaipur |
| 7 | 0.9383 | 8.9 | Senior NLP Engineer | Salesforce | Coimbatore |
| 8 | 0.9363 | 7.0 | Staff ML Engineer | Paytm | Kochi |
| 9 | 0.9351 | 8.0 | Recommendation Systems Eng | CRED | Noida |
| 10 | 0.9310 | 4.2 | Search Engineer | Verloop.io | Mumbai |

Sanity checks on the top 100: 0 honeypots, 0 non-tech titles, 0 candidates with
salary expectations > ₹80 LPA, 6 non-India candidates (all at strong product
companies, ranked 49+, consistent with the JD's "outside India: case-by-case").

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
