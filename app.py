#!/usr/bin/env python3
"""
Redrob Hackathon — Streamlit Sandbox
Senior AI Engineer Candidate Ranker

Deploy to Streamlit Cloud for the required sandbox submission link.
Usage: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd

from scorer import score_candidate

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🔍",
    layout="wide",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Scoring Formula")
    st.markdown("""
    ```
    score = (
      0.35 × title/career  +
      0.30 × skills        +
      0.15 × experience    +
      0.10 × location      +
      0.05 × education     +
      production_evidence_bonus (up to +0.12)
    ) × behavioral_multiplier
    ```
    **v3 updates:** Education 0.10 → 0.05 (JD has no education requirement);
    production-evidence bonus cap increased 0.07 → 0.12; improved semantic
    matching for concept-level ranking/retrieval vocabulary (production ML
    team phrasing now matches IR jargon equivalently).

    **Skills trust formula** (defeats keyword stuffing):
    ```
    trust = proficiency_weight
          × min(1, endorsements / 25)
          × min(1, duration_months / 18)
    ```
    Expert + 0 months = **0.0 trust**, not 1.0.

    ---
    **Behavioral multiplier** (0.25× – 1.1×):

    | Signal | Effect |
    |--------|--------|
    | Not open to work | ×0.5 |
    | Inactive 180+ days | ×0.6 |
    | Response rate < 10% | ×0.7 |
    | Notice > 150 days | ×0.7 |
    | GitHub score ≥ 60 | ×1.1 |

    ---
    **Hard disqualifiers** → score 0:
    - Non-tech title (Marketing, HR, Sales…)
    - Pure consulting career (TCS/Wipro/Infosys…)
    - Outside India + not relocating + onsite/hybrid

    **Honeypots** → score 0.01:
    - Expert skill + 0 months duration
    - ≥3 expert skills: 0 endorsements + < 6 months
    - Career months >> stated YoE
    """)

# ─── Main ───────────────────────────────────────────────────────────────────
st.title("🔍 Redrob Intelligent Candidate Ranker")
st.caption("Senior AI Engineer — Founding Team @ Redrob AI · Rule-based trust-weighted scoring pipeline")

tab_rank, tab_how = st.tabs(["📊 Upload & Rank", "📖 How It Works"])

# ════════════════════════════════════════════════════════════
# TAB 1: Upload & Rank
# ════════════════════════════════════════════════════════════
with tab_rank:

    st.markdown(
        "Upload a **JSON** or **JSONL** file of up to **100 candidate profiles**. "
        "Use `sample_candidates.json` from the hackathon bundle for a quick demo."
    )

    uploaded = st.file_uploader(
        "Choose a JSON or JSONL file",
        type=["json", "jsonl"],
        help="JSON: list of candidate objects. JSONL: one candidate per line (first 100 lines used).",
    )

    if uploaded:
        # ── Load ────────────────────────────────────────────
        try:
            if uploaded.name.endswith(".jsonl"):
                lines = uploaded.read().decode("utf-8").splitlines()
                candidates = [json.loads(l) for l in lines if l.strip()]
            else:
                raw = json.loads(uploaded.read().decode("utf-8"))
                candidates = [raw] if isinstance(raw, dict) else raw
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            st.error(f"❌ Could not parse file: {e}")
            st.stop()

        if len(candidates) > 100:
            st.warning(f"⚠️ {len(candidates)} profiles uploaded — scoring first 100 only.")
            candidates = candidates[:100]

        st.success(f"✅ Loaded **{len(candidates)}** candidate profile(s).")

        # ── Score ───────────────────────────────────────────
        errors = []
        rows = []

        with st.spinner(f"Scoring {len(candidates)} candidates..."):
            for c in candidates:
                cid = c.get("candidate_id", "?")
                try:
                    r = score_candidate(c)
                    p = c.get("profile", {})
                    sig = c.get("redrob_signals", {})
                    rows.append({
                        "candidate_id":  cid,
                        "name":          p.get("name", "—"),
                        "title":         p.get("current_title", "—"),
                        "company":       p.get("current_company", "—"),
                        "location":      p.get("location", "—"),
                        "country":       p.get("country", "—"),
                        "yoe":           p.get("years_of_experience", 0),
                        "score":         r["score"],
                        "tc":            r["components"].get("title_career", 0),
                        "sk":            r["components"].get("skills", 0),
                        "ex":            r["components"].get("experience", 0),
                        "lo":            r["components"].get("location", 0),
                        "ed":            r["components"].get("education", 0),
                        "bm":            r["multiplier"],
                        "honeypot":      r["is_honeypot"],
                        "open_to_work":  sig.get("open_to_work_flag", False),
                        "response_rate": sig.get("recruiter_response_rate", 0),
                        "notice_days":   sig.get("notice_period_days", "—"),
                        "reasoning":     r["reasoning"],
                    })
                except Exception as e:
                    errors.append(f"{cid}: {e}")

        if errors:
            with st.expander(f"⚠️ {len(errors)} scoring error(s)"):
                for err in errors:
                    st.text(err)

        if not rows:
            st.error("No candidates could be scored. Check your JSON format.")
            st.stop()

        # ── Sort & rank ─────────────────────────────────────
        rows.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        for i, row in enumerate(rows, 1):
            row["rank"] = i

        df = pd.DataFrame(rows)

        # ── KPI row ─────────────────────────────────────────
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Candidates Scored", len(df))
        c2.metric("Honeypots Flagged", int(df["honeypot"].sum()))
        c3.metric("Top Score", f"{df['score'].max():.4f}")
        c4.metric("Open to Work", int(df["open_to_work"].sum()))
        c5.metric("India-Based", int((df["country"].str.lower() == "india").sum()))

        # ── Ranked table ────────────────────────────────────
        st.subheader("Ranked Results")
        display = (
            df[["rank", "candidate_id", "name", "title", "company", "location", "yoe", "score", "honeypot"]]
            .rename(columns={
                "rank": "Rank", "candidate_id": "ID", "name": "Name",
                "title": "Title", "company": "Company", "location": "Location",
                "yoe": "YoE", "score": "Score", "honeypot": "🚫 Honeypot",
            })
        )
        st.dataframe(display, use_container_width=True)

        # ── Score breakdown ──────────────────────────────────
        st.subheader("Score Breakdown by Component")
        breakdown = (
            df[["rank", "candidate_id", "title", "tc", "sk", "ex", "lo", "ed", "bm", "score"]]
            .rename(columns={
                "rank": "Rank", "candidate_id": "ID", "title": "Title",
                "tc": "Title/Career (35%)", "sk": "Skills (30%)",
                "ex": "Exp (15%)", "lo": "Location (10%)", "ed": "Edu (5%)",
                "bm": "Behavioral ×", "score": "Final Score",
            })
        )
        st.dataframe(breakdown, use_container_width=True)

        # ── Top 10 cards ────────────────────────────────────
        st.subheader("Top 10 — Detailed View")
        top10 = df[df["rank"] <= min(10, len(df))].iterrows()

        for _, row in top10:
            hp_badge = " 🚫 HONEYPOT" if row["honeypot"] else ""
            with st.expander(
                f"#{int(row['rank'])}  ·  {row['candidate_id']}  ·  "
                f"{row['title']} @ {row['company']}  ·  **{row['score']:.4f}**{hp_badge}"
            ):
                left, right = st.columns([1, 2])
                with left:
                    st.markdown(f"**Experience:** {row['yoe']} years")
                    st.markdown(f"**Location:** {row['location']}, {row['country']}")
                    st.markdown(f"**Open to Work:** {'✅ Yes' if row['open_to_work'] else '❌ No'}")
                    st.markdown(f"**Recruiter Response:** {row['response_rate']:.0%}")
                    st.markdown(f"**Notice Period:** {row['notice_days']} days")
                    st.markdown(f"**Behavioral Multiplier:** {row['bm']:.3f}×")
                with right:
                    st.markdown(f"**Reasoning:** {row['reasoning']}")
                    st.markdown("**Component Scores:**")
                    comp_df = pd.DataFrame([{
                        "Component": "Title/Career (35%)",
                        "Score": f"{row['tc']:.3f}",
                        "Weighted": f"{row['tc'] * 0.35:.3f}",
                    }, {
                        "Component": "Skills (30%)",
                        "Score": f"{row['sk']:.3f}",
                        "Weighted": f"{row['sk'] * 0.30:.3f}",
                    }, {
                        "Component": "Experience (15%)",
                        "Score": f"{row['ex']:.3f}",
                        "Weighted": f"{row['ex'] * 0.15:.3f}",
                    }, {
                        "Component": "Location (10%)",
                        "Score": f"{row['lo']:.3f}",
                        "Weighted": f"{row['lo'] * 0.10:.3f}",
                    }, {
                        "Component": "Education (5%)",
                        "Score": f"{row['ed']:.3f}",
                        "Weighted": f"{row['ed'] * 0.05:.3f}",
                    }])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── Download ─────────────────────────────────────────
        st.markdown("---")
        csv_out = (
            df[["candidate_id", "rank", "score", "reasoning"]]
            .sort_values("rank")
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            label="⬇️ Download Submission CSV",
            data=csv_out,
            file_name="submission_sample.csv",
            mime="text/csv",
            help="Downloads the ranked results in submission format.",
        )

    else:
        st.info(
            "👆 Upload a JSON file to rank candidates. "
            "Try `sample_candidates.json` from the hackathon bundle."
        )
        with st.expander("📋 Expected JSON format"):
            st.code(
                """[
  {
    "candidate_id": "CAND_0000031",
    "profile": {
      "name": "Ayesha Rajan",
      "current_title": "Recommendation Systems Engineer",
      "current_company": "Swiggy",
      "years_of_experience": 6,
      "location": "Hyderabad",
      "country": "India"
    },
    "career_history": [
      {
        "company": "Swiggy",
        "title": "Recommendation Systems Engineer",
        "duration_months": 36,
        "is_current": true
      }
    ],
    "skills": [
      { "name": "Pinecone",   "proficiency": "expert", "endorsements": 34, "duration_months": 88 },
      { "name": "Embeddings", "proficiency": "expert", "endorsements": 48, "duration_months": 60 }
    ],
    "education": [{ "tier": "tier_1", "field_of_study": "Computer Science" }],
    "redrob_signals": {
      "open_to_work_flag": true,
      "recruiter_response_rate": 0.91,
      "notice_period_days": 60,
      "last_active_date": "2026-06-20",
      "github_activity_score": 72
    }
  }
]""",
                language="json",
            )


# ════════════════════════════════════════════════════════════
# TAB 2: How It Works
# ════════════════════════════════════════════════════════════
with tab_how:
    st.header("Architecture & Design Decisions")

    st.markdown("""
    ### The Problem with Keyword Matching
    The dataset contains a deliberate trap: non-tech candidates (Marketing Managers, Accountants)
    have ML keywords in their skills arrays. A system that checks for keyword *presence* would
    rank them alongside real engineers.

    We discovered a second trap: **career history descriptions are recycled templates**.
    The same "Trained and shipped multiple ranking models using XGBoost..." description
    appears verbatim under "Civil Engineer" and "Marketing Manager" profiles.
    Scanning career descriptions for ML keywords would add noise, not signal.

    ---
    ### Our Solution: Trust-Weighted Rule-Based Scoring

    #### Component 1 — Title / Career (35%)
    The highest-weight component. Current title keywords + company type across career history.

    | Title Pattern | Base Score |
    |--------------|-----------|
    | ML / Search / Ranking / Retrieval / Recommendation Engineer | 0.85 |
    | Data Scientist, LLM roles | 0.70 |
    | Generic Software Engineer | 0.50 |
    | Data Engineer | 0.40 |
    | Non-tech (Marketing, HR, Sales, Ops…) | **0.0 — disqualified** |

    Company modifier: entire career at TCS/Wipro/Infosys/Accenture/Cognizant → **×0.4 penalty**
    (directly encoding the JD's explicit consulting-firm disqualifier).

    ---
    #### Component 2 — Skills Trust Formula (30%)
    ```
    trust = proficiency_weight × min(1, endorsements / 25) × min(1, duration_months / 18)

    proficiency weights: expert=1.0, advanced=0.75, intermediate=0.4, beginner=0.15
    ```

    An **expert** skill with **0 months** of usage = **0.0 trust** (not 1.0).
    This is the core defence against keyword stuffing.

    Core skills (must-haves from JD): embeddings, sentence-transformers, vector databases
    (Pinecone/Weaviate/Qdrant/FAISS…), information retrieval, ranking, BM25, hybrid search,
    Python, A/B testing, NDCG, MRR.

    ---
    #### Component 3 — Experience (15%)
    Sweet spot is 5–9 years per the JD. Below 2 years returns 0.0.

    #### Component 4 — Location (10%)
    Pune/Noida preferred; Hyderabad/Mumbai/Delhi NCR welcome. Outside India = heavy penalty
    unless candidate is willing to relocate.

    #### Component 5 — Education (5%)
    Institution tier + CS field bonus. Weak signal — the JD has no educational
    requirements — deliberately capped low so it can't outweigh real production evidence.

    #### Production Evidence Bonus (up to +0.12)
    Qualified candidates (`title/career ≥ 0.60` and `skills ≥ 0.35`) receive a
    category-based bonus for concrete evidence of relevant systems, evaluation,
    ownership, and production operations. Categories count once and are weighted
    by job recency (current 1.0, second 0.7, older 0.4).

    Each category recognizes both IR vocabulary (BM25, NDCG, "learning to rank")
    **and** production-ranking vocabulary (XGBoost/LightGBM discovery-feed
    ranking, "optimization target", offline-online metric correlation, drift
    detection/retraining cadence) — so a candidate who describes the same
    ranking-ownership work without IR jargon isn't penalized for word choice.

    ---
    #### Behavioral Multiplier (multiplicative, 0.25× – 1.1×)
    Applied to the entire base score, not added to it. This ensures availability penalties
    are severe. A perfect-on-paper candidate who hasn't logged in for 6 months and has a
    5% recruiter response rate is, for hiring purposes, not actually available.

    ---
    #### Honeypot Detection
    Profiles with impossible signals are scored 0.01 and excluded from top 100:
    - `expert` proficiency + `duration_months = 0` (logically impossible)
    - ≥3 expert skills with 0 endorsements and < 6 months usage
    - Sum of career history months far exceeds stated years of experience

    ---
    #### Why No ML Model?
    No labelled training data available. A model needs ground truth to train on.
    Our rule-based system encodes what a great recruiter knows — it is transparent,
    debuggable, and every design decision can be defended. An ML model trained on
    no data would be arbitrary.
    """)
