"""
Redrob Hackathon — Candidate Scorer v4
JD: Senior AI Engineer — Founding Team @ Redrob AI

Changes from v1:
  - Expanded CORE_SKILLS (recommendation systems, haystack, bi/cross-encoder, etc.)
  - Expanded STRONG_PRODUCT_COS (fictional dataset companies + Indian AI cos)
  - behavioral_multiplier: added github_activity_score bonus + interview_completion_rate penalty
  - _build_reasoning: now emits specific numbers (YoE, skill endorsements/duration,
    response rate, notice period) — required to pass Stage 4 manual review
  - score_candidate: final score capped at 1.0

Changes from v2 (production-evidence audit):
  - EVIDENCE_CATEGORIES expanded with concept-level terms so production-ranking
    ownership described in non-IR vocabulary (XGBoost/LightGBM discovery-feed
    ranking, "optimization target", plain-English "connect users to relevant
    matches") maps into the SAME four existing categories — no new categories,
    no company/candidate-specific terms.
  - Fixed a real matching bug: "A/B testing" (gerund form, the most common
    phrasing in the dataset) was not matched by the "a/b test" term due to a
    strict word-boundary suffix check.
  - Added a co-occurrence heuristic for "offline-online correlation" (offline
    + online + correlate/predict in the same job text) instead of requiring
    one exact template sentence — rewards the concept, not a keyword.
  - Rebalanced weights: education 0.10 → 0.05 (JD has no education
    requirement — see education_score docstring), and moved that 0.05 into
    PRODUCTION_EVIDENCE_MAX_BONUS (0.07 → 0.12), so verified production
    ranking-system ownership counts for more than pedigree.

Changes from v3 (full-100K audit of JD-explicit disqualifiers):
  - hard_gate_multiplier(): three JD-explicit "we will not move forward" /
    "case-by-case" profiles were only losing weight in ONE small component
    (title 12%, company_fit 8%, location 10%) — the rest of the weighted sum
    was untouched, and a full 100K audit found a pure-Accountant profile
    reaching 0.36 and a pure-consulting-career profile reaching 0.45, both
    uncomfortably close to the ~0.53 rank-100 cutoff. Non-tech titles and
    pure-consulting careers now get an explicit, heavier multiplicative
    gate (×0.15 / ×0.35); logistically-impossible locations (outside India,
    unwilling to relocate, needs onsite/hybrid) get a milder ×0.55 gate,
    since the JD's own language for that case is "case-by-case", not "will
    not move forward" — this also matches our own manual-review calibration
    (rank.md), which placed such a candidate mid-tier, not deep-tail.
  - _cv_speech_dominant_multiplier(): implements the JD's explicit "primary
    expertise is computer vision, speech, or robotics without significant
    NLP/IR exposure" exclusion, which had no implementation at all before
    this pass. Matches on career TEXT (not the skills list, to avoid
    penalizing a stray skill entry), and only fires when there is ZERO
    retrieval/ranking/recommendation evidence anywhere in the career.
  - _career_progression_bonus(): no longer rewards title-level climbs
    (Engineer → Senior → Staff → …) earned via a string of sub-18-month
    job-hops — the JD explicitly calls this "title-chaser" pattern out.
  - high_throughput_bonus() is now gated behind the same title/skills
    relevance bar as production_evidence_score(), so mentioning "Redis" or
    "1M QPS" in an unrelated (non-ML) job can no longer buy a score bump.
  - CAPABILITY_GROUPS["llm"] gained "nlp" / "natural language processing" —
    previously these skill entries earned zero skill-level credit anywhere.
  - Location tiering refined to the JD's actual text ("Candidates in
    Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply"): Pune/Noida (JD
    HQ cities) score 1.0, those four named cities score 0.95, other India
    tech hubs (Bangalore, Chennai, Kolkata — not named either way) score
    0.90. Previously all of these scored an identical 1.0.

  Net effect on the full 100K dataset (see audit_full.py): the 4 candidates
  formerly in the top 100 who were outside India, unwilling to relocate,
  and needed onsite/hybrid work are gone, replaced by 4 verified India-based
  candidates; honeypot rate, non-tech-title, and pure-consulting best-case
  ranks all moved 30-130x further from the top-100 cutoff. Top-10 order is
  otherwise materially unchanged (rank 1 unchanged; largest top-100 mover
  outside the 4 removed candidates was ~10 positions).
"""

import math
import re
from functools import lru_cache
from datetime import date, datetime

# ─────────────────────────────────────────────
# JD-derived constants
# ─────────────────────────────────────────────

# Update this if you run the scorer on a different date.
# Using a fixed date keeps scoring deterministic and reproducible.
REFERENCE_DATE = date(2026, 6, 27)

# ── Core skills ──────────────────────────────────────────────────────────────
# These are the "absolutely need" skills from the JD.
# Scoring: ~5 core skills at full trust = core_score 1.0
CORE_SKILLS = {
    # Embedding & retrieval systems
    "embeddings", "embedding",
    "sentence transformers", "sentence-transformers", "sentence transformer",
    "vector search", "semantic search", "dense retrieval",
    "dense passage retrieval", "dpr",

    # Vector databases / hybrid search infra
    "vector database", "vector db", "vector store",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "annoy", "chromadb", "pgvector",

    # Ranking / IR — specific algorithms
    "information retrieval", "ranking", "learning to rank",
    "learning-to-rank", "ltr", "bm25", "hybrid search",
    "reranking", "re-ranking",

    # Ranking / IR — concept-level terms (same expertise, different vocabulary)
    # A candidate listing "Ranking Systems" has identical relevance to one listing
    # "Learning to Rank" — both describe building retrieval/ranking systems.
    "ranking systems", "search systems", "search infrastructure",
    "search backend", "search and discovery", "search & discovery",
    "information retrieval systems",
    "content matching", "candidate matching", "matching systems", "matching layer",
    "vector representations", "text encoders",
    "indexing algorithms", "indexing",
    "dense ranking", "sparse ranking",

    # Retrieval architectures
    "bi-encoder", "bi encoder", "cross-encoder", "cross encoder",
    "colbert", "haystack",

    # Recommendation systems (closely aligned with role, treated as core)
    "recommendation systems", "recommendation system",
    "recommender systems", "recommender system",

    # Python (hard requirement from JD)
    "python",

    # Evaluation (hard requirement: "hands-on evaluation framework experience")
    "a/b testing", "ab testing", "experimentation",
    "ndcg", "mrr", "map", "offline evaluation",
    "ranking evaluation", "retrieval evaluation",
}

# ── Niche / nice-to-have skills ──────────────────────────────────────────────
# Boost, not required. Contribute at 50% weight vs core skills.
NICHE_SKILLS = {
    # LLM fine-tuning (explicitly in JD nice-to-have)
    "llm fine-tuning", "fine-tuning llms", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "rlhf", "instruction tuning", "dpo",

    # NLP / transformers
    "nlp", "natural language processing",
    "transformers", "hugging face transformers", "hugging face", "huggingface",
    "bert", "roberta", "t5",

    # RAG
    "rag", "retrieval augmented generation", "retrieval-augmented generation",

    # Traditional ML (learning-to-rank is in JD nice-to-have)
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "machine learning", "applied ml", "feature engineering",
    "scikit-learn", "sklearn",

    # MLOps / infra
    "mlops", "mlflow", "weights & biases", "wandb", "kubeflow", "bentoml",
    "kafka", "spark", "flink", "distributed systems",
    "kubernetes", "docker", "aws", "gcp", "azure",

    # Search infra / adjacent
    "solr", "lucene", "vespa", "typesense",
    "data drift", "model monitoring",
}

# Aliases to normalise before matching (raw_name → canonical name)
SKILL_NORMALIZE = {
    "sentence transformers": "sentence transformers",
    "hugging face transformers": "hugging face transformers",
    "hugging face": "hugging face",
    "information retrieval": "information retrieval",
    "fine-tuning llms": "fine-tuning",
    "llm fine-tuning": "fine-tuning",
    "fine tuning": "fine-tuning",
    "ab testing": "a/b testing",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "dense passage retrieval": "dpr",
    "recommendation system": "recommendation systems",
    "recommender system": "recommendation systems",
    "recommender systems": "recommendation systems",
    "bi encoder": "bi-encoder",
    "cross encoder": "cross-encoder",
    "learning-to-rank": "learning to rank",
    "ltr": "learning to rank",
    "sklearn": "scikit-learn",
    "wandb": "weights & biases",
    "huggingface": "hugging face",
}

# ── Production evidence ──────────────────────────────────────────────────────
# Career evidence that a qualified candidate has built and operated the systems
# this JD requires. Matching is category-based so repeated keywords do not help.
#
# v2 note: the original term lists recognised explicit IR/vendor vocabulary
# (BM25, Pinecone, "learning to rank") well but under-matched engineers who
# describe the same ranking-ownership work in production terminology instead
# (XGBoost/LightGBM discovery-feed ranking, "optimization target", "offline-
# online correlation", plain-English descriptions of relevance/matching).
# The additions below map that vocabulary into the SAME four categories —
# no new categories, no company- or candidate-specific terms — so a candidate
# who says "connect users to relevant matches" gets the same relevant_systems
# credit as one who says "semantic search", and a candidate who says "offline
# metrics correlated with online engagement" gets the same evaluation credit
# as one who says "NDCG". This generalizes to any of the 100K candidates who
# use this phrasing, not just the ones reviewed manually.
EVIDENCE_CATEGORIES = {
    "relevant_systems": {
        "ranking", "retrieval", "recommendation", "hybrid retrieval",
        "semantic search", "candidate matching", "vector search",
        "re-ranking", "reranking", "search",
        # Concept-level additions: same system, production-team vocabulary.
        "relevance", "matching layer", "personalization", "discovery feed",
        "behavioral signal", "behavioral-signal",
        # Implicit production vocabulary: engineers describe infrastructure without saying "retrieval"
        "surface relevant", "relevant content", "match", "matching infrastructure",
        "query understanding", "understand query", "query intent",
        "information retrieval", "information need",
        "ranking calibration", "calibrate ranking",
        "index", "indexing",
    },
    "evaluation": {
        "offline evaluation", "online evaluation", "ndcg", "mrr",
        "a/b test", "ab test", "calibration", "evaluation framework",
        # "A/B testing" (the gerund form) is the single most common phrasing
        # in the dataset for this concept and was previously unmatched by
        # "a/b test" due to the word-boundary check requiring an exact suffix.
        "a/b testing", "ab testing",
        # Concept-level additions: defining what a ranking model optimizes for,
        # and the click/engagement metrics that feed that definition.
        "optimization target", "click-through", "click through",
        # Implicit evaluation: engineers test ranking quality without saying "NDCG"
        "ranking quality", "relevance quality", "ranking accuracy",
        "offline metric", "online metric",
    },
    "ownership": {
        "owned", "designed", "built", "led", "shipped", "deployed",
        "architected", "end-to-end", "end to end",
        # Implicit ownership: production engineers use these terms
        "responsible", "maintain", "operated", "managed",
    },
    "operational": {
        "latency", "throughput", "qps", "monitoring", "feature store",
        "production", "index refresh", "p99", "sla",
        # Concept-level additions: the operational discipline around a live
        # ranking model (drift, retraining, feature health) described in
        # production-ML vocabulary rather than generic "monitoring".
        "drift detection", "retraining cadence", "feature monitoring",
        "embedding drift",
        # Implicit operations: scale and serving language
        "serving", "query volume", "scale", "infrastructure",
        "caching", "cache layer", "database",
    },
}

PRODUCTION_EVIDENCE_MAX_BONUS = 0.12
PRODUCTION_EVIDENCE_TC_GATE = 0.60
PRODUCTION_EVIDENCE_SK_GATE = 0.35

# ── Company lists ─────────────────────────────────────────────────────────────

# Pure consulting firms — entire career at these = heavy penalty (JD explicit)
BIG_CONSULTING = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcltech", "tech mahindra", "mphasis", "hexaware",
    "mindtree", "l&t infotech", "ltimindtree", "persistent systems",
    "niit technologies", "mastech",
}

# Product companies that signal strong background for this role
# Includes real companies AND fictional dataset companies
# (The fictional ones are used as placeholders in the synthetic dataset;
#  a "Hooli ML Engineer" should be rewarded the same as a real product co.)
STRONG_PRODUCT_COS = {
    # ── Indian unicorns / consumer product companies ──
    "swiggy", "zomato", "flipkart", "meesho", "razorpay", "cred",
    "paytm", "phonepe", "ola", "rapido", "zepto", "blinkit",
    "dunzo", "slice", "groww", "zerodha", "upstox", "freshworks",
    "postman", "hasura", "setu", "atlassian",
    "sharechat", "dailyhunt", "inmobi", "glance",
    "lenskart", "nykaa", "myntra", "bigbasket", "cars24",
    "spinny", "delhivery", "shadowfax", "shiprocket",
    "jupiter", "fi", "fi money",
    "practo", "1mg", "pharmeasy", "healthkart",
    "byju", "unacademy", "vedantu",
    "cleartrip", "makemytrip", "yatra",
    "urban company", "browserstack",

    # ── Indian AI-first companies ──
    "sarvam", "sarvam.ai", "krutrim",
    "haptik", "yellow.ai", "observe.ai",
    "murf", "murf.ai",
    "mad street den",
    "uniphore", "vernacular.ai",
    "senseforth", "active.ai", "conversica",

    # ── Global FAANG / major tech ──
    "google", "microsoft", "amazon", "meta", "apple",
    "netflix", "uber", "airbnb", "stripe",
    "openai", "anthropic", "deepmind", "cohere",
    "databricks", "snowflake", "confluent",
    "spotify", "twitter", "linkedin", "salesforce",

    # ── Fictional dataset companies ──
    # These are placeholder company names in the synthetic 100K dataset.
    # Real ML engineers in the dataset have these as employers.
    "hooli", "pied piper", "piedpiper",
    "globex", "globex inc",
    "initech",
    "acme", "acme corp",
    "wayne enterprises",
    "stark industries",
    "dunder mifflin",
}

# Non-tech titles — hard disqualifier (title_career_score returns 0.0)
NON_TECH_TITLES = {
    # Marketing / sales
    "marketing manager", "marketing executive", "marketing specialist",
    "marketing director", "marketing head",
    "sales manager", "sales executive", "sales representative",
    "sales director", "account manager", "business development",
    # HR
    "hr manager", "hr executive", "hr business", "human resources",
    "recruiter", "talent acquisition",
    # Support / ops
    "customer support", "customer service", "customer success",
    "operations manager", "operations executive", "operations head",
    # Finance
    "finance manager", "accountant", "accounts manager",
    "ca ", "chartered accountant",
    # Content / design
    "content writer", "content manager", "content strategist",
    "graphic designer", "visual designer",
    # Engineering (non-software)
    "civil engineer", "mechanical engineer", "electrical engineer",
    "structural engineer", "chemical engineer",
    # Other clear non-tech
    "project manager",   # weak but not hard-zero; see below
    "business analyst",  # weak but not hard-zero; see below
}

# These two get a low-but-nonzero title score (not full disqualifier)
# A PM with strong ML skills could still appear in the long tail
WEAK_TITLE_OVERRIDE = {"project manager", "business analyst"}

# JD text, verbatim: "Location: Pune/Noida-preferred but flexible... Candidates
# in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply." Bangalore/Chennai/
# Kolkata are not called out either way — still full India-based candidates,
# just not the cities the JD explicitly names. Kept as a gentle tiering
# (location is only 10% of the weighted sum) rather than a penalty.
HQ_LOCATIONS = {"pune", "noida"}
JD_WELCOME_LOCATIONS = {
    "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
}
OTHER_INDIA_TECH_HUBS = {"bangalore", "bengaluru", "chennai", "kolkata"}

EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.2,
    "unknown": 0.3,
}

CS_FIELDS = {
    "computer science", "computer engineering", "information technology",
    "software engineering", "electronics", "electrical engineering",
    "mathematics", "statistics", "data science", "artificial intelligence",
    "machine learning", "information systems",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.strip().lower()


_MATCH_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_match_text(text: str) -> str:
    return _MATCH_NORMALIZE_RE.sub(" ", str(text).lower()).strip()


@lru_cache(maxsize=512)
def _normalize_match_phrase(phrase: str) -> str:
    return _normalize_match_text(phrase)


def _phrase_present_norm(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_match_phrase(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _days_since(date_str: str) -> int:
    try:
        d = date.fromisoformat(date_str)
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999


def _skill_trust(skill: dict) -> float:
    """
    Compute trust-weighted score for a single skill entry.

    trust = proficiency_weight × min(1, endorsements/25) × min(1, duration_months/18)

    An "expert" skill with 0 months of usage = 0.0.
    This is the core anti-keyword-stuffing mechanism.
    """
    proficiency_map = {
        "expert":       1.0,
        "advanced":     0.75,
        "intermediate": 0.4,
        "beginner":     0.15,
    }
    pw = proficiency_map.get(_norm(skill.get("proficiency", "beginner")), 0.15)
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)
    endorsement_factor = min(1.0, endorsements / 25.0)
    duration_factor = min(1.0, duration / 18.0)
    return pw * endorsement_factor * duration_factor


# ─────────────────────────────────────────────
# Honeypot Detection
# ─────────────────────────────────────────────

# def is_honeypot(candidate: dict) -> bool:
#     """
#     Returns True if the profile has impossible or fabricated signals.
#     Honeypots get score = 0.01 to stay out of top 100.

#     Three detection patterns (from dataset analysis):
#       1. Any skill with proficiency=expert AND duration_months=0
#       2. 3+ expert skills with 0 endorsements AND < 6 months duration
#       3. Career timeline months >> stated YoE (impossible overlap)
#     """
#     skills = candidate.get("skills", [])
#     profile = candidate.get("profile", {})
#     career = candidate.get("career_history", [])

#     # Pattern 1: expert skill with zero usage — physically impossible
#     for s in skills:
#         if _norm(s.get("proficiency", "")) == "expert" and s.get("duration_months", 1) == 0:
#             return True

#     # Pattern 2: multiple unverified expert skills
#     suspicious = [
#         s for s in skills
#         if _norm(s.get("proficiency", "")) == "expert"
#         and s.get("endorsements", 0) == 0
#         and s.get("duration_months", 0) < 6
#     ]
#     if len(suspicious) >= 3:
#         return True

#     # Pattern 3: career timeline longer than stated experience allows
#     career_months = sum(j.get("duration_months", 0) for j in career)
#     yoe_months = profile.get("years_of_experience", 0) * 12
#     if career_months > yoe_months * 1.4 + 24:
#         return True

#     return False
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def evaluate_integrity(candidate: dict) -> dict:
    """
    Evaluate profile integrity.

    Returns:
    {
        "hard_fail": bool,
        "reliability": float,
        "reasons": list[str]
    }
    """

    hard_fail = False
    reliability = 1.0
    reasons = []

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])

    yoe_months = max(0.0, float(profile.get("years_of_experience", 0) or 0) * 12)
    # Use REFERENCE_DATE for deterministic scoring, not datetime.now()
    reference_datetime = datetime.combine(REFERENCE_DATE, datetime.min.time())

    ############################################################
    # Employment timeline
    ############################################################

    intervals = []
    current_jobs = 0

    for job in career:

        start = job.get("start_date")
        end = job.get("end_date")

        if job.get("is_current"):
            current_jobs += 1

        if not start:
            continue

        try:
            s = datetime.strptime(start, "%Y-%m-%d")

            if end:
                e = datetime.strptime(end, "%Y-%m-%d")
            else:
                e = reference_datetime

            if s > e:
                hard_fail = True
                reasons.append("reversed_employment_dates")
                continue

            intervals.append((s, e))

        except (TypeError, ValueError):
            reliability -= 0.03
            reasons.append("invalid_employment_date")
            continue

    ############################################################
    # Multiple current jobs
    ############################################################

    if current_jobs > 2:
        hard_fail = True
        reasons.append("multiple_current_jobs")
    elif current_jobs == 2:
        reliability -= 0.08
        reasons.append("multiple_current_jobs")

    ############################################################
    # Concurrent jobs
    ############################################################

    events = []

    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))

    events.sort()

    active = 0

    for _, delta in events:
        active += delta

        if active >= 3:
            hard_fail = True
            reasons.append("excessive_concurrent_jobs")
            break

    ############################################################
    # Worked months
    ############################################################

    intervals.sort(key=lambda x: x[0])

    merged = []

    for interval in intervals:

        if not merged:
            merged.append(interval)
            continue

        ps, pe = merged[-1]
        cs, ce = interval

        if cs <= pe:
            merged[-1] = (ps, max(pe, ce))
        else:
            merged.append(interval)

    worked_days = sum((e - s).days for s, e in merged)
    worked_months = worked_days / 30.44

    diff = worked_months - yoe_months

    # Stated YoE is rounded, so allow a small absolute/proportional margin.
    # The released honeypot example (3 YoE but 61 worked months) must fail.
    # Use proportional rule: worked_months > yoe_months * 1.25 + 6
    allowed_months = yoe_months * 1.25 + 6
    if yoe_months > 0 and worked_months > allowed_months:
        hard_fail = True
        reasons.append("career_months_exceed_yoe")
    elif diff > 9 and not hard_fail:
        reliability -= 0.08
        reasons.append("career_months_high")

    ############################################################
    # Education chronology
    ############################################################

    def degree_level(name):

        d = str(name).lower()

        if any(x in d for x in ["ph.d", "phd", "doctor"]):
            return 3

        if any(x in d for x in [
            "m.tech","m.e.","m.s.","m.sc","master","mba"
        ]):
            return 2

        if any(x in d for x in [
            "b.tech","b.e.","b.s.","b.sc","bachelor","b.a."
        ]):
            return 1

        return 0

    parsed = []

    for edu in education:

        try:
            start = int(edu.get("start_year"))
            end = int(edu.get("end_year"))
        except:
            continue

        if end < start:
            hard_fail = True
            reasons.append("education_end_before_start")
            continue

        parsed.append({
            "level": degree_level(edu.get("degree")),
            "start": start,
            "end": end
        })

    for higher in parsed:
        if higher["level"] <= 1:
            continue
        for lower in parsed:
            if lower["level"] <= 0 or higher["level"] <= lower["level"]:
                continue

            # A higher degree completed before the prerequisite degree began is
            # impossible. Overlaps are suspicious but can represent dual or
            # integrated programmes, so they are only a soft signal.
            if higher["end"] < lower["start"]:
                hard_fail = True
                reasons.append(
                    "masters_before_bachelors"
                    if higher["level"] == 2
                    else "phd_before_previous_degree"
                )
            elif higher["start"] < lower["end"] - 1:
                reliability -= 0.08
                reasons.append("overlapping_degrees")

    ############################################################
    # Skill integrity
    ############################################################

    zero_duration = 0
    fake_experts = 0
    absurd_duration = 0

    for s in skills:

        prof = _norm(s.get("proficiency", ""))

        dur = s.get("duration_months", 0)

        endors = s.get("endorsements", 0)

        if prof == "expert":

            if dur == 0:
                zero_duration += 1

            if dur < 6 and endors == 0:
                fake_experts += 1

        if yoe_months > 0 and dur > yoe_months + 36:
            absurd_duration += 1

    if zero_duration >= 2:
        hard_fail = True
        reasons.append("multiple_zero_duration_experts")

    elif zero_duration == 1:
        reliability -= 0.05

    if fake_experts >= 3:
        hard_fail = True
        reasons.append("multiple_unverified_experts")

    if absurd_duration >= 4:
        hard_fail = True
        reasons.append("impossible_skill_durations")

    elif absurd_duration > 0:
        reliability -= 0.10

    ############################################################
    # Current company consistency
    ############################################################

    current_company = str(
        profile.get("current_company", "")
    ).lower()

    if current_company:

        found = False

        for job in career:

            if job.get("is_current"):

                company = str(job.get("company", "")).lower()

                if current_company in company or company in current_company:
                    found = True
                    break

        if not found and current_jobs > 0:
            reliability -= 0.10
            reasons.append("current_company_mismatch")

    ############################################################
    # Finalize
    ############################################################

    reliability = max(0.25 if hard_fail else 0.0, min(1.0, reliability))

    return {
        "hard_fail": hard_fail,
        "reliability": round(reliability, 2),
        "reasons": sorted(set(reasons))
    }


def is_honeypot(candidate: dict) -> bool:
    """Backward-compatible public API used by tests, diagnostics and the app."""
    return evaluate_integrity(candidate)["hard_fail"]

# ─────────────────────────────────────────────
# Component Scorers
# ─────────────────────────────────────────────
#   NEW  [title_career_score()  TILL 660

# ============================================================
# SIGNAL GROUPS
# ============================================================

SIGNALS = {

    "retrieval": {
        "retrieval", "search", "ranking",
        "learning to rank", "ltr",
        "hybrid retrieval", "dense retrieval",
        "semantic search", "vector search",
        "bm25", "faiss",
        "elasticsearch", "opensearch",
        "qdrant", "milvus", "weaviate",
        "pinecone", "pgvector",
        "ndcg", "mrr", "recall@k",
        "rerank", "reranking",
        "search relevance",
        # Implicit retrieval vocabulary (production-team phrasing)
        "matching layer", "match infrastructure",
        "surface relevant", "relevant content",
        "query understanding", "understand query",
        "information retrieval", "information need",
        "ranking calibration", "calibrate",
        "index refresh", "indexing",
        "query", "serving queries",
    },

    "recommendation": {
        "recommendation",
        "recommendation systems",
        "recommendation system",
        "collaborative filtering",
        "matrix factorization",
        "personalization",
        "discovery feed",
        "candidate matching",
        # Implicit recommendation vocabulary
        "suggest", "discovery", "match user",
    },

    "llm": {
        "llm", "rag",
        "gpt", "llama", "mistral",
        "bert",
        "transformer",
        "transformers",
        "sentence transformer",
        "sentence-transformer",
        "langchain",
        "llamaindex",
        "haystack",
        "fine tuning",
        "fine-tuning",
        "lora",
        "qlora",
        "peft"
    },

    "operations": {
        "production",
        "latency",
        "p95",
        "throughput",
        "qps",
        "deployment",
        "monitoring",
        "drift",
        "mlflow",
        "kubeflow",
        "feature store",
        "offline",
        "online",
        "ab test",
        "a/b",
        # Implicit operations vocabulary (scale & serving)
        "serving", "query volume", "scale",
        "infrastructure", "caching", "cache layer",
        "database", "maintained", "operated",
        "high availability",
    },

    "ownership": {
        "led",
        "owned",
        "designed",
        "architected",
        "implemented",
        "built",
        "built from scratch",
        "migrated",
        "rolled out",
        "launched",
        "shipped",
        "mentored",
        # Implicit ownership
        "responsible", "managed", "team",
    },

    "impact": {
        "%",
        "improved",
        "reduced",
        "decreased",
        "increased",
        "million",
        "10m",
        "30m",
        "35m",
        "50m",
        "100m",
        # Implicit scale language
        "billions", "thousands", "large", "massive",
    }
}

# ============================================================
# HIT RATIO
# ============================================================

def _phrase_present(text: str, phrase: str) -> bool:
    """Match a complete normalized phrase without regex backtracking."""
    return _phrase_present_norm(_normalize_match_text(text), phrase)


def _signal_hits(text: str, signals: set[str]) -> int:
    normalized_text = _normalize_match_text(text)
    return sum(1 for signal in signals if _phrase_present_norm(normalized_text, signal))


def _signal_hits_norm(normalized_text: str, signals: set[str]) -> int:
    return sum(1 for signal in signals if _phrase_present_norm(normalized_text, signal))


def _signal_score(text: str, signals: set[str], saturation: int) -> float:
    return min(1.0, _signal_hits(text, signals) / max(1, saturation))


def _signal_score_norm(normalized_text: str, signals: set[str], saturation: int) -> float:
    return min(1.0, _signal_hits_norm(normalized_text, signals) / max(1, saturation))


# ============================================================
# SCORE ONE JOB
# ============================================================

def _job_score(job):

    text = _normalize_match_text(
        f"{job.get('title','')} "
        f"{job.get('description','')}"
    )

    retrieval = _signal_score_norm(text, SIGNALS["retrieval"], 2)
    recommendation = _signal_score_norm(text, SIGNALS["recommendation"], 2)
    llm = _signal_score_norm(text, SIGNALS["llm"], 3)
    operations = _signal_score_norm(text, SIGNALS["operations"], 3)
    ownership = _signal_score_norm(text, SIGNALS["ownership"], 2)
    impact = _signal_score_norm(text, SIGNALS["impact"], 2)

    score = (
        0.35 * retrieval +
        0.15 * recommendation +
        0.15 * llm +
        0.15 * ownership +
        0.10 * operations +
        0.10 * impact
    )

    return min(score, 1.0)


# ============================================================
# TITLE PROGRESSION
# ============================================================

LEVELS = [
    "engineer",
    "senior",
    "lead",
    "staff",
    "principal",
    "architect",
    "director"
]


def _career_progression_bonus(career):

    if len(career) < 2:
        return 0.0

    previous = None
    improvements = 0
    improvement_durations = []

    for job in reversed(career):

        title = job.get("title", "").lower()

        level = 0

        for i, keyword in enumerate(LEVELS):
            if keyword in title:
                level = i

        if previous is not None and level > previous:
            improvements += 1
            improvement_durations.append(job.get("duration_months", 0) or 0)

        previous = level

    if improvements == 0:
        return 0.0

    # JD explicit disqualifier: "title-chasers" who climb Senior -> Staff ->
    # Principal by switching companies every ~1.5 years. Title progression
    # earned by staying and growing into a role is a positive signal; the
    # same progression earned via a string of sub-18-month stints is exactly
    # the anti-pattern the JD calls out, so it should not also earn a bonus.
    avg_duration = sum(improvement_durations) / len(improvement_durations)
    if improvements >= 2 and avg_duration < 18:
        return 0.0

    return min(improvements * 0.015, 0.06)


# ============================================================
# TITLE SCORE — Current title relevance only
# ============================================================

def title_score(candidate):
    """Score based on current title only (isolated from company/career history)."""
    profile = candidate.get("profile", {})
    current_title = _normalize_match_text(profile.get("current_title", ""))

    # Hard disqualifier: non-tech current title
    for bad_title in NON_TECH_TITLES:
        if _phrase_present_norm(current_title, bad_title):
            if bad_title in WEAK_TITLE_OVERRIDE:
                return 0.15
            return 0.0

    strong_title_terms = {
        "recommendation", "ranking", "retrieval", "search", "nlp",
        "ml engineer", "machine learning", "applied ml", "applied ai",
        "ai engineer", "information retrieval", "vector", "embedding",
    }
    moderate_title_terms = {
        "data scientist", "data science", "research engineer", "llm",
        "generative", "foundation model",
    }
    generic_title_terms = {
        "software engineer", "sde", "backend", "platform engineer",
        "senior engineer",
    }
    # Production-focused titles that imply systems/infrastructure work
    production_title_terms = {
        "production systems", "production engineer", "infrastructure engineer",
        "systems engineer", "platform", "scale",
    }

    if any(term in current_title for term in strong_title_terms):
        return 0.90
    elif any(term in current_title for term in moderate_title_terms):
        return 0.70
    elif any(term in current_title for term in production_title_terms):
        return 0.65  # Production-focused titles get modest boost
    elif any(term in current_title for term in generic_title_terms):
        return 0.50
    else:
        return 0.30


# ============================================================
# CAREER EVIDENCE SCORE — Job descriptions and domain relevance
# ============================================================

def career_evidence_score(candidate):
    """Score based on job descriptions and career history relevance."""
    career = candidate.get("career_history", [])

    if not career:
        return 0.0

    total = 0.0
    total_weight = 0.0
    seen_descriptions = set()

    for idx, job in enumerate(career):
        description = " ".join(str(job.get("description", "")).lower().split())
        job_for_scoring = job
        weight_multiplier = 1.0

        if description and description in seen_descriptions:
            # Duplicate description: reduce weight to avoid double-counting keywords,
            # but keep the description for title-based relevance
            weight_multiplier = 0.4
        elif description:
            seen_descriptions.add(description)

        relevance = _job_score(job_for_scoring)

        duration = max(job.get("duration_months", 12), 1)
        duration_weight = math.log1p(duration)

        recency = 1.0 if job.get("is_current") else max(0.35, 0.85 ** idx)
        weight = duration_weight * recency * weight_multiplier

        total += relevance * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    score = total / total_weight
    score += _career_progression_bonus(career)
    return max(0.0, min(score, 1.0))


# ============================================================
# COMPANY FIT SCORE — Company quality with limited bonuses
# ============================================================

def company_fit_score(candidate):
    """
    Company quality score with reduced bonuses.
    - Consulting penalty: hard stop at 0.45 (per JD)
    - Product company bonus: capped at +0.10 (not +0.25)
    - Unknown companies: neutral 0.80
    """
    career = candidate.get("career_history", [])

    if not career:
        return 0.80

    companies = [str(job.get("company", "")).lower() for job in career]
    consulting_count = sum(
        any(x in c for x in BIG_CONSULTING) for c in companies
    )
    product_count = sum(
        any(x in c for x in STRONG_PRODUCT_COS) for c in companies
    )

    n = len(companies)

    # Pure consulting career — JD explicit disqualifier
    if consulting_count == n:
        return 0.45

    # Mix of consulting + other, no product experience
    if consulting_count > 0 and product_count == 0:
        return 0.75

    # Has product company experience — limited bonus
    if product_count > 0:
        ratio = product_count / n
        return min(0.90, 0.80 + 0.10 * ratio)

    # Unknown companies
    return 0.80

# # def title_career_score(candidate: dict) -> float:
#     """
#     Score based on current title + full career history.
#     Rewards: ML/AI/Search/Ranking engineers at product companies.
#     Penalizes: non-tech titles, pure consulting careers.
#     """
#     profile = candidate.get("profile", {})
#     career = candidate.get("career_history", [])

#     current_title = _norm(profile.get("current_title", ""))
#     current_industry = _norm(profile.get("current_industry", ""))

#     # Hard disqualifier: non-tech current title
#     # (excluding weak-title-override roles that get low-but-nonzero score)
#     for bad_title in NON_TECH_TITLES:
#         if bad_title in current_title:
#             # Project Manager / Business Analyst — not hard zero, just weak
#             if bad_title in WEAK_TITLE_OVERRIDE:
#                 base = 0.15
#                 break
#             return 0.0
#     else:
#         # Determine base score from title keywords
#         strong_ml_keywords = {
#             "recommendation", "ranking", "retrieval", "search",
#             "nlp", "ml engineer", "machine learning", "applied ml",
#             "applied ai", "ai engineer", "information retrieval",
#             "vector", "embedding",
#         }
#         moderate_ml_keywords = {
#             "data scientist", "data science", "research engineer",
#             "llm", "generative", "foundation model",
#         }
#         generic_swe_keywords = {
#             "software engineer", "sde", "backend", "fullstack",
#             "full-stack", "platform engineer", "senior engineer",
#         }
#         data_eng_keywords = {
#             "data engineer", "analytics engineer", "etl",
#         }

#         if any(k in current_title for k in strong_ml_keywords):
#             base = 0.85
#         elif any(k in current_title for k in moderate_ml_keywords):
#             base = 0.70
#         elif any(k in current_title for k in generic_swe_keywords):
#             base = 0.50
#         elif any(k in current_title for k in data_eng_keywords):
#             base = 0.40
#         else:
#             base = 0.25   # Other tech titles (DevOps, QA, Mobile, etc.)

#     # ── Company quality modifier ──────────────────────────────────────────────
#     all_companies = [_norm(j.get("company", "")) for j in career]
#     consulting_count = sum(
#         1 for co in all_companies
#         if any(bc in co for bc in BIG_CONSULTING)
#     )
#     product_count = sum(
#         1 for co in all_companies
#         if any(pc in co for pc in STRONG_PRODUCT_COS)
#     )

#     total_jobs = len(all_companies)
#     if total_jobs == 0:
#         company_modifier = 0.8
#     elif consulting_count == total_jobs:
#         # Entire career at consulting — JD explicitly disqualifies this
#         company_modifier = 0.4
#     elif consulting_count > 0 and product_count == 0:
#         # Mix of consulting + generic companies, no strong product cos
#         company_modifier = 0.7
#     elif product_count > 0:
#         # At least some product company experience
#         product_ratio = product_count / total_jobs
#         company_modifier = 0.9 + 0.1 * product_ratio
#     else:
#         company_modifier = 0.85   # Unknown companies — neutral assumption

#     # Industry signal (minor)
#     # if any(k in current_industry for k in ("software", "ai", "ml", "saas", "technology")):
#     #     industry_modifier = 1.05
#     # elif "it services" in current_industry:
#     #     industry_modifier = 0.9
#     # else:
#     #     industry_modifier = 1.0

#     score = base * company_modifier #* industry_modifier
#     return min(1.0, max(0.0, score))
# # 

# def skills_score(candidate: dict) -> float:
#     """
#     Score based on skill match with JD requirements.
#     Uses trust-weighted scoring (proficiency × endorsements × duration).
#     Keyword stuffers (expert with 0 usage) contribute 0.0.
#     """
#     skills = candidate.get("skills", [])
#     if not skills:
#         return 0.0

#     core_coverage = 0.0
#     niche_bonus = 0.0
#     core_matched = set()
#     niche_matched = set()

#     for skill_entry in skills:
#         raw_name = _norm(skill_entry.get("name", ""))
#         name = SKILL_NORMALIZE.get(raw_name, raw_name)
#         trust = _skill_trust(skill_entry)

#         if name in CORE_SKILLS or raw_name in CORE_SKILLS:
#             if name not in core_matched:
#                 core_coverage += trust
#                 core_matched.add(name)
#         elif name in NICHE_SKILLS or raw_name in NICHE_SKILLS:
#             if name not in niche_matched:
#                 niche_bonus += trust * 0.5
#                 niche_matched.add(name)

#     # ~5 genuine core skills at full trust → core_score = 1.0
#     core_score = min(1.0, core_coverage / 5.0)
#     # ~4 niche skills at full trust → niche_score = 0.5 max
#     niche_score = min(0.5, niche_bonus / 4.0)

#     # Platform assessment bonus (verified scores from Redrob)
#     assessment_bonus = 0.0
#     assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
#     for skill_name, score_val in assessments.items():
#         norm_name = _norm(skill_name)
#         if norm_name in CORE_SKILLS or norm_name in NICHE_SKILLS:
#             assessment_bonus += (score_val / 100.0) * 0.05
#     assessment_bonus = min(0.1, assessment_bonus)

#     return min(1.0, core_score * 0.7 + niche_score * 0.3 + assessment_bonus)



# Canonical capability taxonomy shared by structured-skill and career support
# matching. Vendor breadth is deliberately capped within a capability.
CAPABILITY_GROUPS = {
    "python": {
        "cap": 0.18,
        "skills": {"python"},
        "evidence": {"python"},
    },
    "retrieval": {
        "cap": 0.24,
        "skills": {
            "retrieval", "information retrieval", "information retrieval systems",
            "search", "search infrastructure", "search backend",
            "search and discovery", "search & discovery", "ranking",
            "ranking systems", "learning to rank", "ltr", "semantic search",
            "vector search", "embeddings", "embedding", "bm25",
            "content matching", "candidate matching", "matching systems",
            "vector representations", "text encoders", "indexing algorithms",
            "indexing", "dense retrieval", "hybrid search", "hybrid retrieval",
            "dense passage retrieval", "dpr", "sentence transformers",
            "sentence-transformers", "sentence transformer",
            # Implicit retrieval terminology
            "matching", "matching layer", "matching infrastructure",
            "relevance", "relevant", "index", "indexing",
        },
        "evidence": {
            "retrieval", "information retrieval", "search infrastructure",
            "search backend", "search and discovery", "ranking layer",
            "ranking model", "ranking models", "ranking pipeline", "ranking system",
            "learning to rank", "semantic search", "vector search", "embedding based",
            "embedding-based", "hybrid retrieval", "hybrid search", "dense retrieval",
            "bm25", "reranking", "re-ranking", "retrieval pipeline", "retrieval system",
            "retrieval-augmented generation", "rag",
            # Implicit retrieval evidence
            "matching layer", "surface relevant", "relevant content",
            "query understanding", "query intent", "ranking calibration",
            "index refresh", "serving queries", "query volume",
        },
    },
    "vector_infra": {
        "cap": 0.15,
        "skills": {
            "faiss", "pinecone", "qdrant", "milvus", "weaviate", "pgvector",
            "elasticsearch", "opensearch", "vector database", "vector db",
            "vector store", "annoy", "chromadb",
        },
        "evidence": {
            "faiss", "pinecone", "qdrant", "milvus", "weaviate", "pgvector",
            "elasticsearch", "opensearch", "vector database", "vector store",
            "vector database", "index refresh", "vector index",
        },
    },
    "recommendation": {
        "cap": 0.12,
        "skills": {
            "recommendation systems", "recommendation system",
            "recommender systems", "recommender system", "recommendation engine",
            "collaborative filtering", "matrix factorization",
        },
        "evidence": {
            "recommendation system", "recommender system", "recommendation engine",
            "collaborative filtering", "matrix factorization",
            "personalization", "discovery feed", "behavioral signal",
        },
    },
    "evaluation": {
        "cap": 0.14,
        "skills": {
            "a/b testing", "ab testing", "evaluation", "ndcg", "mrr", "map",
            "offline evaluation", "ranking evaluation", "retrieval evaluation",
            "experimentation", "experimentation framework",
        },
        "evidence": {
            "a/b testing", "ab testing", "a/b testing", "offline evaluation",
            "online evaluation", "evaluation framework", "ndcg", "mrr", "map",
            "recall@", "precision@", "relevance labels", "relevance label",
            "golden dataset", "calibration",
            "offline online correlation", "online offline correlation",
            "optimization target", "a/b test", "ab test",
        },
    },
    "operations": {
        "cap": 0.10,
        "skills": {
            "mlops", "mlflow", "kubeflow", "bentoml", "docker", "kubernetes",
            "model monitoring", "data drift", "drift detection", "monitoring",
        },
        "evidence": {
            "latency", "throughput", "qps", "p95", "p99", "monitoring", "drift",
            "index refresh", "feature store", "rollback", "autoscaling",
            "production deployment", "built and operated", "deployment",
            "production", "observability",
        },
    },
    "llm": {
        "cap": 0.07,
        "skills": {
            "rag", "haystack", "llm", "prompt engineering", "lora", "qlora",
            "peft", "fine-tuning", "fine tuning", "transformers",
            "hugging face transformers", "hugging face", "langchain", "llamaindex",
            # NLP itself was previously unrepresented in any capability group
            # (it only affected title_score's strong-title-terms check), so a
            # candidate's "NLP" / "Natural Language Processing" skill entries
            # earned zero skill-level credit no matter how well-endorsed.
            "nlp", "natural language processing",
        },
        "evidence": {
            "retrieval augmented generation", "rag", "fine tuned", "fine tuning",
            "lora", "qlora", "peft", "llm", "transformers",
            "nlp", "natural language processing",
        },
    },
}


def _capability_supported(text: str, phrases: set[str]) -> bool:
    return any(_phrase_present(text, phrase) for phrase in phrases)


def _capability_supported_norm(normalized_text: str, phrases: set[str]) -> bool:
    return any(_phrase_present_norm(normalized_text, phrase) for phrase in phrases)


def skills_score(candidate: dict) -> float:
    """Score capabilities once, with independent career/assessment support."""
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    recent_jobs = sorted(
        career,
        key=lambda job: (
            bool(job.get("is_current")),
            job.get("end_date") or "",
            job.get("start_date") or "",
        ),
        reverse=True,
    )[:3]
    recent_text = _normalize_match_text(" ".join(
        f"{job.get('title', '')} {job.get('description', '')}"
        for job in recent_jobs
    ))
    summary_text = _normalize_match_text(
        f"{profile.get('headline', '')} {profile.get('current_title', '')} "
        f"{profile.get('summary', '')}"
    )
    assessments = {
        SKILL_NORMALIZE.get(_norm(name), _norm(name)): float(score)
        for name, score in candidate.get("redrob_signals", {})
        .get("skill_assessment_scores", {}).items()
    }

    supported = {
        group: _capability_supported_norm(recent_text, cfg["evidence"])
        for group, cfg in CAPABILITY_GROUPS.items()
    }
    grouped = {group: [] for group in CAPABILITY_GROUPS}

    for skill in skills:
        raw = _norm(skill.get("name", ""))
        name = SKILL_NORMALIZE.get(raw, raw)
        group = next(
            (
                group_name
                for group_name, cfg in CAPABILITY_GROUPS.items()
                if name in cfg["skills"] or raw in cfg["skills"]
            ),
            None,
        )
        if group is None:
            continue

        assessment = max(
            assessments.get(name, -1),
            assessments.get(raw, -1),
        )
        if assessment >= 80:
            confidence = 0.95
        elif assessment >= 60:
            confidence = 0.80
        elif _phrase_present_norm(recent_text, name) or _phrase_present_norm(recent_text, raw):
            confidence = 1.0
        elif supported[group]:
            confidence = 0.85
        elif _phrase_present_norm(summary_text, name) or _phrase_present_norm(summary_text, raw):
            confidence = 0.70
        else:
            confidence = 0.50

        grouped[group].append(_skill_trust(skill) * confidence)

    total = 0.0
    for group, values in grouped.items():
        if not values:
            continue
        values.sort(reverse=True)
        strength = values[0] + sum(min(0.04, value * 0.10) for value in values[1:])
        total += CAPABILITY_GROUPS[group]["cap"] * min(1.0, strength)

    assessment_bonus = min(
        0.03,
        sum(
            (score / 100.0) * 0.01
            for skill, score in assessments.items()
            if any(skill in cfg["skills"] for cfg in CAPABILITY_GROUPS.values())
        ),
    )
    return min(1.0, total + assessment_bonus)

def _contains_evidence_term(text: str, term: str) -> bool:
    """Match a complete term, avoiding false positives such as search/research."""
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _has_offline_online_correlation(text: str) -> bool:
    """
    Detect the "offline-online correlation" concept by co-occurrence rather
    than an exact phrase.

    This concept — validating that an offline metric actually predicts online
    behavior — is one of the clearest signals of real evaluation rigor, but
    candidates phrase it many different ways ("offline-online correlation",
    "offline metrics that actually correlated with online engagement",
    "online/offline metric correlation"). Rewarding co-occurrence of
    offline + online + a correlation/prediction verb generalizes across all
    of these phrasings instead of hardcoding one template sentence.
    """
    lowered = _normalize_match_text(text)
    has_both_sides = "offline" in lowered and "online" in lowered
    has_correlation_language = (
        "correlat" in lowered or "predict" in lowered
    )
    return has_both_sides and has_correlation_language


PRODUCTION_EVIDENCE_GROUPS = {
    "retrieval": {
        "hybrid retrieval", "hybrid search", "dense retrieval",
        "retrieval pipeline", "retrieval system", "vector search",
        "semantic search", "embedding based", "embedding-based", "bm25",
        "faiss", "qdrant", "weaviate", "milvus", "pgvector",
        "elasticsearch", "opensearch", "retrieval augmented generation",
        "rag",
        # Implicit retrieval evidence
        "matching layer", "match infrastructure",
        "surface relevant", "relevant content", "relevance",
        "query understanding", "query intent",
        "index refresh", "indexing", "serving queries",
    },
    "ranking": {
        "learning to rank", "ltr", "ranking layer", "ranking model",
        "ranking pipeline", "ranking system", "reranking", "re-ranking",
        "reranker", "xgboost", "lightgbm", "gradient boosted",
        # Implicit ranking evidence
        "ranking calibration", "calibrate ranking", "ranking quality",
    },
    "recommendation": {
        "recommendation system", "recommender system",
        "collaborative filtering", "matrix factorization", "personalization",
        "discovery feed",
    },
    "evaluation": {
        "ndcg", "mrr", "map", "recall k", "precision k",
        "evaluation framework", "golden dataset", "relevance label",
        "a b test", "ab test", "a b testing", "ab testing", "calibration",
        "offline online correlation", "online offline correlation",
        "optimization target", "click through", "click through rate",
        # Implicit evaluation evidence
        "ranking quality", "relevance quality", "ranking accuracy",
        "offline metric", "online metric",
    },
    "operations": {
        "latency", "p95", "p99", "throughput", "qps", "drift",
        "monitoring", "observability", "index refresh", "rollback",
        "feature store", "autoscaling", "caching", "built and operated",
        "production deployment",
        # Implicit operations evidence
        "serving", "query volume", "scale", "infrastructure",
        "cache layer", "database", "maintained", "operated",
        "high availability", "billions", "millions",
    },
    "ownership": {
        "owned", "led", "end to end", "from scratch", "designed",
        "architected", "rolled out", "deployed", "shipped",
        # Implicit ownership
        "responsible", "managed", "team", "built",
    },
}

# JD explicit disqualifier: "People whose primary expertise is computer
# vision, speech, or robotics without significant NLP/IR exposure. We
# respect your work but you'd be re-learning fundamentals here."
CV_SPEECH_ROBOTICS_TERMS = {
    "computer vision", "image classification", "object detection",
    "yolo", "opencv", "cnn", "convolutional neural network",
    "gan", "gans", "generative adversarial",
    "diffusion model", "diffusion models",
    "speech recognition", "asr", "tts", "text to speech", "text-to-speech",
    "robotics", "autonomous vehicle", "autonomous driving", "slam", "lidar",
}


def _cv_speech_dominant_multiplier(candidate: dict) -> float:
    """
    Soft penalty for the CV/speech/robotics-without-NLP/IR profile the JD
    explicitly says it will not move forward on.

    Matched against career TEXT (title + description across the whole
    career), not the skills list — a single stray "OpenCV" skill entry
    shouldn't trigger this; a career genuinely built around CV/speech with
    zero retrieval/ranking/recommendation evidence anywhere should. This
    mirrors the category-matching style already used by
    production_evidence_score() rather than a raw keyword-frequency count.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 1.0

    full_text = _normalize_match_text(" ".join(
        f"{job.get('title', '')} {job.get('description', '')}" for job in career
    ))

    cv_hits = _signal_hits_norm(full_text, CV_SPEECH_ROBOTICS_TERMS)
    relevant_hits = (
        _signal_hits_norm(full_text, PRODUCTION_EVIDENCE_GROUPS["retrieval"])
        + _signal_hits_norm(full_text, PRODUCTION_EVIDENCE_GROUPS["ranking"])
        + _signal_hits_norm(full_text, PRODUCTION_EVIDENCE_GROUPS["recommendation"])
    )

    if cv_hits >= 2 and relevant_hits == 0:
        return 0.80
    return 1.0


PRODUCTION_EVIDENCE_SCALE_PATTERNS = (
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:m|million|b|billion)\+?\s*"
        r"(?:users|queries|documents|profiles|items|records)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:k|thousand)\+?\s*"
        r"(?:users|queries|documents|profiles|items|records)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:\.\d+)?\s*qps\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:improved|reduced|decreased|increased)\b.{0,80}\b\d+(?:\.\d+)?%",
        flags=re.IGNORECASE,
    ),
)


# def production_evidence_score(candidate: dict) -> float:
#     """
#     Return a 0–1 score for concrete production evidence in career history.

#     Each evidence category counts once, using the highest recency weight of a
#     matching job: current=1.0, second job=0.7, older jobs=0.4. Both job title
#     and description are considered.
#     """
#     career = candidate.get("career_history", [])
#     category_weights = {category: 0.0 for category in EVIDENCE_CATEGORIES}

#     for index, job in enumerate(career):
#         if job.get("is_current", False):
#             recency_weight = 1.0
#         elif index == 1:
#             recency_weight = 0.7
#         else:
#             recency_weight = 0.4

#         text = f"{job.get('title', '')} {job.get('description', '')}"
#         for category, terms in EVIDENCE_CATEGORIES.items():
#             matched = any(_contains_evidence_term(text, term) for term in terms)
#             if category == "evaluation" and not matched:
#                 matched = _has_offline_online_correlation(text)
#             if matched:
#                 category_weights[category] = max(
#                     category_weights[category], recency_weight
#                 )

#     return sum(category_weights.values()) / len(EVIDENCE_CATEGORIES)



def production_evidence_score(candidate: dict) -> float:
    """
    Contextual, depth-aware production evidence.

    Repeated descriptions count once. Operations, ownership and scale only
    count when the same job contains retrieval/ranking/recommendation evidence.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    dimensions = {
        "retrieval": 0.0,
        "ranking": 0.0,
        "evaluation": 0.0,
        "operations": 0.0,
        "ownership": 0.0,
        "scale": 0.0,
    }
    weights = {
        "retrieval": 1.35,
        "ranking": 1.25,
        "evaluation": 1.20,
        "operations": 0.95,
        "ownership": 1.10,
        "scale": 1.15,
    }
    ordered_jobs = sorted(
        career,
        key=lambda job: (
            bool(job.get("is_current")),
            job.get("end_date") or "",
            job.get("start_date") or "",
        ),
        reverse=True,
    )
    seen_descriptions = set()

    for index, job in enumerate(ordered_jobs):
        recency = 1.0 if job.get("is_current") else (0.75 if index == 1 else 0.55 if index == 2 else 0.35)
        description = _normalize_match_text(job.get("description", ""))
        if description and description in seen_descriptions:
            description = ""
        elif description:
            seen_descriptions.add(description)
        text = _normalize_match_text(f"{job.get('title', '')} {description}")

        hit_counts = {
            name: _signal_hits_norm(text, group_patterns)
            for name, group_patterns in PRODUCTION_EVIDENCE_GROUPS.items()
        }
        evaluation_hits = hit_counts["evaluation"] + (1 if _has_offline_online_correlation(text) else 0)
        relevant_hits = (
            hit_counts["retrieval"]
            + hit_counts["ranking"]
            + hit_counts["recommendation"]
        )

        if hit_counts["retrieval"]:
            depth = min(1.0, 0.55 + 0.15 * (hit_counts["retrieval"] - 1))
            dimensions["retrieval"] = max(dimensions["retrieval"], recency * depth)
        if hit_counts["ranking"]:
            depth = min(1.0, 0.55 + 0.15 * (hit_counts["ranking"] - 1))
            dimensions["ranking"] = max(dimensions["ranking"], recency * depth)
        if evaluation_hits and relevant_hits:
            depth = min(1.0, 0.55 + 0.15 * (evaluation_hits - 1))
            dimensions["evaluation"] = max(dimensions["evaluation"], recency * depth)
        if hit_counts["operations"] and relevant_hits:
            depth = min(1.0, 0.45 + 0.15 * (hit_counts["operations"] - 1))
            dimensions["operations"] = max(dimensions["operations"], recency * depth)
        if hit_counts["ownership"] and relevant_hits:
            depth = min(1.0, 0.45 + 0.15 * (hit_counts["ownership"] - 1))
            dimensions["ownership"] = max(dimensions["ownership"], recency * depth)

        scale_hits = sum(bool(pattern.search(text)) for pattern in PRODUCTION_EVIDENCE_SCALE_PATTERNS)
        if scale_hits and relevant_hits:
            depth = min(1.0, 0.55 + 0.20 * (scale_hits - 1))
            dimensions["scale"] = max(dimensions["scale"], recency * depth)

    possible = sum(weights.values())
    return sum(dimensions[name] * weights[name] for name in dimensions) / possible

def experience_score(candidate: dict) -> float:
    """Score based on years of experience — sweet spot 5–9 per JD."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe < 2:
        return 0.0
    elif yoe < 4:
        return 0.3 + (yoe - 2) / 2.0 * 0.3     # ramps 0.30 → 0.60
    elif yoe <= 9:
        return 0.7 + (yoe - 4) / 5.0 * 0.25    # ramps 0.70 → 0.95 (sweet spot)
    elif yoe <= 12:
        return 0.95
    else:
        return 0.85   # Slight dip — may be overqualified / overpriced


def location_score(candidate: dict) -> float:
    """Score based on India location + city + relocation willingness."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = _norm(profile.get("country", ""))
    location = _norm(profile.get("location", ""))
    willing = signals.get("willing_to_relocate", False)
    work_mode = _norm(signals.get("preferred_work_mode", ""))

    if country == "india":
        if any(city in location for city in HQ_LOCATIONS):
            return 1.0
        elif any(city in location for city in JD_WELCOME_LOCATIONS):
            return 0.95
        elif any(city in location for city in OTHER_INDIA_TECH_HUBS):
            return 0.90
        elif willing:
            return 0.85
        else:
            return 0.70
    else:
        if willing:
            return 0.50
        elif work_mode in ("remote", "flexible"):
            return 0.35
        else:
            return 0.10


def education_score(candidate: dict) -> float:
    """Weak signal per JD (JD has no education requirement) — 10% weight only."""
    education = candidate.get("education", [])
    if not education:
        return 0.4   # Unknown → neutral baseline

    best_tier_score = 0.0
    cs_field_bonus = 0.0

    for edu in education:
        tier = edu.get("tier", "unknown")
        tier_score = EDUCATION_TIER_SCORES.get(tier, 0.4)
        best_tier_score = max(best_tier_score, tier_score)

        field = _norm(edu.get("field_of_study", ""))
        if any(cf in field for cf in CS_FIELDS):
            cs_field_bonus = 0.1

    return min(1.0, best_tier_score + cs_field_bonus)


def high_throughput_bonus(candidate: dict) -> float:
    """
    Bonus for candidates with explicit large-scale systems experience: handling 1M+ RPS,
    custom data structures, lock-free implementations, microsecond latencies, distributed caching.
    JD emphasizes systems-level optimization for inference workloads. Returns 1.0–1.15 multiplier.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 1.0

    scale_patterns = {
        r"\b\d+[mk]?\s*(?:rps|requests?\s*per\s*second|qps)",
        r"\b(?:lock.?free|lock-free|atomic|cas|compare.?and.?swap)",
        r"\bredis\b",
        r"\b(?:microsecond|sub.?millisecond|us latency|μs)",
        r"\bcustom\s+(?:data\s+)?struct",
        r"\b(?:memcached|distributed cache)",
        r"\b\d+(?:[.,]\d+)?\s*[km](?:\+|plus)?\s*requests",
    }

    hit_count = 0

    for job in career:
        text = _normalize_match_text(f"{job.get('title', '')} {job.get('description', '')}")
        for pattern in scale_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hit_count += 1
                if hit_count >= 3:
                    return 1.15

    if hit_count == 2:
        return 1.10
    elif hit_count == 1:
        return 1.05

    return 1.0


def hard_gate_multiplier(candidate: dict, tt: float, cf: float) -> float:
    """
    Multiplicative penalty for the three JD-explicit disqualifiers whose
    component weight alone (12% for title, 8% for company_fit, 10% for
    location) is not strong enough to keep a candidate who is strong on
    every OTHER axis out of a top-100 slot:

      - Non-tech current title (Marketing/HR/Accountant/...). title_score()
        already zeroes the title component, but that is only 12% of the
        weighted sum — the remaining 88% (skills/career/experience/location/
        education) is untouched, and career descriptions in this dataset
        are known to be recycled across unrelated titles (see README), so a
        "Marketing Manager" can still post a respectable score on paper.
      - An entire career at a pure consulting firm. The JD says explicitly:
        "we've had bad fit experiences in both directions."
      - Based outside India, unwilling to relocate, AND requiring onsite or
        hybrid work — logistically hard to take the role regardless of fit.

    These are not honeypots (nothing is fabricated/impossible about the
    profile), so this does not zero the score the way evaluate_integrity()
    does — it pushes them down the ranking so that no combination of other
    strengths can pull them into a top-100 slot on their own.

    The location case is deliberately the mildest of the three (0.55, vs.
    0.15/0.35 for the other two): the JD's own language for it is "case-by-
    case", not "we will not move forward" (that phrase is reserved for the
    title/consulting-style disqualifiers), and our own manual-review
    calibration (rank.md) placed at least one such candidate competitively
    mid-tier rather than in the deep tail. location_score() already scores
    this combination at 0.10 within its own 10%-weighted term, so this gate
    is a top-up on top of an already-low contribution, not the primary
    penalty mechanism.
    """
    gate = 1.0

    # title_score() returns exactly 0.0 only for the hard-disqualifier
    # bucket; WEAK_TITLE_OVERRIDE roles (PM/BA) return 0.15 and are
    # deliberately left out of this gate.
    if tt == 0.0:
        gate *= 0.15

    # company_fit_score() returns exactly 0.45 only when consulting_count
    # equals the total number of jobs (pure-consulting career).
    if cf <= 0.50:
        gate *= 0.35

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    if (
        _norm(profile.get("country", "")) != "india"
        and not signals.get("willing_to_relocate", True)
        and _norm(signals.get("preferred_work_mode", "")) in ("onsite", "hybrid")
    ):
        gate *= 0.55

    return gate


def behavioral_multiplier(candidate: dict) -> float:
    """
    Availability / engagement multiplier applied to base score.
    Returns 0.25–1.10 (above 1.0 possible only with strong github signal).

    Rationale: a perfect-on-paper candidate who is unavailable is,
    for hiring purposes, not actually a candidate.
    """
    signals = candidate.get("redrob_signals", {})
    multiplier = 1.0

    # ── Availability signals ──────────────────────────────────────────────────

    # Not open to work is the strongest single negative signal
    if not signals.get("open_to_work_flag", True):
        multiplier *= 0.5

    # Recency of platform activity — penalize only if we have a valid, old date
    # Missing data is neutral (1.0), not a penalty
    last_active = signals.get("last_active_date")
    if last_active:
        days_active = _days_since(last_active)
        # _days_since returns 9999 for invalid dates; only penalize if valid and old
        if days_active < 9000 and days_active > 180:
            multiplier *= 0.6
        elif days_active < 9000 and days_active > 90:
            multiplier *= 0.8

    # ── Engagement signals ────────────────────────────────────────────────────

    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.10:
        multiplier *= 0.7
    elif response_rate < 0.25:
        multiplier *= 0.85

    # Notice period (JD prefers sub-30; can buy out up to 30 days)
    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        pass                      # 1.00 — ideal
    elif notice <= 60:
        multiplier *= 0.99        # minor friction
    elif notice <= 90:
        multiplier *= 0.97        # explicitly still in scope in the JD
    elif notice <= 120:
        multiplier *= 0.94
    elif notice <= 150:
        multiplier *= 0.90        # significant
    else:
        multiplier *= 0.82


    # ── Quality signals ───────────────────────────────────────────────────────

    # GitHub activity — positive signal for open-source contributions (JD nice-to-have)
    # Tiers lowered from 1.10/1.05 to 1.05/1.03: the previous values were collapsing
    # top-4 candidates at 1.0000 (high base × 1.10 > 1.0 cap), destroying NDCG@10
    # differentiation. These values keep github as a meaningful signal without hitting the cap.
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 60:
        multiplier *= 1.02
    elif github_score >= 30:
        multiplier *= 1.01
    # -1 means no GitHub linked → no bonus, no penalty

    # Interview reliability — do they actually show up?
    icr = signals.get("interview_completion_rate", 1.0)
    if icr < 0.4:
        multiplier *= 0.85
    elif icr < 0.6:
        multiplier *= 0.92

    # ── Floor ─────────────────────────────────────────────────────────────────
    return max(0.25, multiplier)


# ─────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────

WEIGHTS = {
    "title":        0.12,
    "career_domain": 0.15,
    "company_fit":  0.08,
    "skills":       0.30,
    "experience":   0.15,
    "location":     0.10,
    # Reduced from 0.05 to 0.02. The JD has no education requirement.
    # Education should mainly help detect contradictions, not materially rank.
    "education":    0.02,
}


def _get_top_core_skills(candidate: dict, n: int = 3) -> list[dict]:
    """Return up to n core skills sorted by trust score descending."""
    skills = candidate.get("skills", [])
    scored = []
    seen = set()
    for s in skills:
        raw = _norm(s.get("name", ""))
        canonical = SKILL_NORMALIZE.get(raw, raw)
        if (canonical in CORE_SKILLS or raw in CORE_SKILLS) and canonical not in seen:
            trust = _skill_trust(s)
            if trust > 0:
                scored.append((trust, s))
                seen.add(canonical)
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]]


def score_candidate(candidate: dict) -> dict:
    """
    Score a single candidate against the Senior AI Engineer JD.

    Returns:
        score:       float 0.0–∞ (no hard 1.0 cap to allow deterministic tie-breaking)
        components:  dict of individual component scores
        multiplier:  float (behavioral multiplier applied)
        is_honeypot: bool
        reasoning:   str (1–2 sentences, specific to this candidate)
    """
    integrity = evaluate_integrity(candidate)

    if integrity["hard_fail"]:
        return {
            "score": 0.01,
            "components": {},
            "multiplier": 0.0,
            "is_honeypot": True,
            "reasoning": "Integrity check failed: " + ", ".join(integrity["reasons"]),
        }

    tt = title_score(candidate)
    ce = career_evidence_score(candidate)
    cf = company_fit_score(candidate)
    sk = skills_score(candidate)
    ex = experience_score(candidate)
    lo = location_score(candidate)
    ed = education_score(candidate)

    base = (
        WEIGHTS["title"]        * tt +
        WEIGHTS["career_domain"] * ce +
        WEIGHTS["company_fit"]  * cf +
        WEIGHTS["skills"]       * sk +
        WEIGHTS["experience"]   * ex +
        WEIGHTS["location"]     * lo +
        WEIGHTS["education"]    * ed
    )

    production_evidence = 0.0
    if tt >= PRODUCTION_EVIDENCE_TC_GATE and sk >= PRODUCTION_EVIDENCE_SK_GATE:
        production_evidence = production_evidence_score(candidate)

    base += (
        PRODUCTION_EVIDENCE_MAX_BONUS *
        production_evidence
    )

    bm = behavioral_multiplier(candidate)

    # Systems-scale bonus for high-throughput engineering — gated behind the
    # same title/skills relevance bar as production_evidence, so mentioning
    # "Redis" or "1M QPS" in an unrelated (e.g. non-ML) job can't buy a score
    # bump on its own.
    throughput_bonus = 1.0
    if tt >= PRODUCTION_EVIDENCE_TC_GATE and sk >= PRODUCTION_EVIDENCE_SK_GATE:
        throughput_bonus = high_throughput_bonus(candidate)

    # JD-explicit disqualifiers that the weighted sum alone doesn't suppress
    # hard enough (see hard_gate_multiplier docstring), plus the CV/speech/
    # robotics-without-NLP/IR profile the JD also explicitly excludes.
    gate = hard_gate_multiplier(candidate, tt, cf)
    gate *= _cv_speech_dominant_multiplier(candidate)

    final = base * bm * throughput_bonus * gate

    # Integrity reliability adjustment (soft, not hard cap)
    final *= (0.80 + 0.20 * integrity["reliability"])

    # **CRITICAL: Deterministic tie-breaking via micro-modifiers**
    # Remove hard 1.0 cap to avoid massive score collapse among top candidates.
    # Add micro-modifiers from redrob_signals to ensure every candidate has unique score.
    signals = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github_score = signals.get("github_activity_score", -1)
    saved_30d = signals.get("saved_by_recruiters_30d", 0)

    tiebreaker = (
        response_rate * 0.0001 +
        max(0, github_score) * 0.00001 +
        saved_30d * 0.000001
    )
    final += tiebreaker

    # Soft ceiling at 1.0, but allow above for tie-breaking precision
    # (rank.py will normalize if needed for output)
    final = max(0.0, final)

    reasoning = _build_reasoning(candidate, tt, sk, ex, lo, bm, final)

    return {
        "score": round(final, 6),
        "components": {
            "title": round(tt, 3),
            "career_domain": round(ce, 3),
            "company_fit": round(cf, 3),
            "skills": round(sk, 3),
            "experience": round(ex, 3),
            "location": round(lo, 3),
            "education": round(ed, 3),
            "production_evidence": round(production_evidence, 3),
        },
        "multiplier": round(bm, 3),
        "is_honeypot": False,
        "integrity": integrity,
        "integrity_reliability": round(integrity["reliability"], 2),
        "integrity_reasons": integrity["reasons"],
        "reasoning": reasoning,
    }


def _build_reasoning(candidate, tt, sk, ex, lo, bm, final) -> str:
    """
    Generate specific, honest, rank-consistent reasoning.

    Must include at least 2 of: YoE, company name, specific skill name,
    location, notice period, response rate.
    Tone must match rank (high score = enthusiastic, low score = honest about gaps).
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = _norm(profile.get("country", ""))
    notice = signals.get("notice_period_days", 0)
    open_to = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    days_active = _days_since(signals.get("last_active_date", "2020-01-01"))
    willing = signals.get("willing_to_relocate", False)

    # ── Build skill summary ────────────────────────────────────────────────────
    top_skills = _get_top_core_skills(candidate, n=2)
    skill_parts = []
    for s in top_skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        endorse = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        skill_parts.append(f"{prof} {name} ({endorse} endorse, {dur}mo)")
    skill_str = " and ".join(skill_parts) if skill_parts else None

    # ── Identify concerns ──────────────────────────────────────────────────────
    concerns = []
    if not open_to:
        concerns.append("not marked open to work")
    if days_active > 180:
        concerns.append(f"inactive for {days_active // 30} months")
    elif days_active > 90:
        concerns.append(f"low recent activity ({days_active} days)")
    if country != "india" and not willing:
        concerns.append(f"based outside India ({profile.get('country', '')}), not relocating")
    if notice > 90:
        concerns.append(f"{notice}-day notice exceeds JD preference")
    if response_rate < 0.25:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    # ── Compose reasoning by score tier ───────────────────────────────────────

    if final >= 0.60:
        # Strong candidate — lead with specifics, mention concern as footnote
        parts = [f"{yoe:.0f}yr {title} at {company}"]
        if skill_str:
            parts.append(skill_str)
        parts.append(f"{location}-based")
        parts.append(f"{response_rate:.0%} recruiter response rate")
        parts.append(f"{notice}-day notice")
        main = "; ".join(parts) + "."
        if concerns:
            main += f" Note: {concerns[0]}."
        return main

    elif final >= 0.30:
        # Moderate — balanced strengths and concerns
        strength_parts = [f"{yoe:.0f}yr {title} at {company}"]
        if skill_str:
            strength_parts.append(skill_str)
        strength_str = "; ".join(strength_parts)

        if concerns:
            concern_str = "; ".join(concerns[:2])
            return f"{strength_str}. Concerns: {concern_str}."
        else:
            return f"{strength_str}; {location}-based; {response_rate:.0%} response rate."

    else:
        # Weak candidate — honest, in long tail
        if tt == 0.0:
            role_note = f"current role ({title}) is non-technical"
        elif skill_str:
            role_note = f"partial skill match: {skill_str}"
        else:
            role_note = "limited evidence of required ML/retrieval skills"

        concern_str = "; ".join(concerns[:2]) if concerns else "below score threshold"
        return (
            f"{yoe:.0f}yr {title} at {company}; {role_note}. "
            f"Ranked in long tail; concerns: {concern_str}."
        )


# ─────────────────────────────────────────────
# Quick test runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        candidates = data if isinstance(data, list) else [data]
    else:
        print("Usage: python scorer.py sample_candidates.json")
        sys.exit(0)

    results = []
    for c in candidates:
        r = score_candidate(c)
        results.append((r["score"], c["candidate_id"], c["profile"]["current_title"], r))

    results.sort(reverse=True)
    print(f"\n{'Rank':>4} {'Score':>7}  {'ID':<15} {'Title':<45} {'Honeypot'}")
    print("─" * 90)
    for i, (score, cid, title, r) in enumerate(results, 1):
        hp = "🚨" if r["is_honeypot"] else ""
        comp = r.get("components", {})
        print(f"{i:>4} {score:>7.4f}  {cid:<15} {title[:44]:<45} {hp}")
        if comp:
            print(f"       tc={comp.get('title_career',0):.2f}  "
                  f"sk={comp.get('skills',0):.2f}  "
                  f"ex={comp.get('experience',0):.2f}  "
                  f"lo={comp.get('location',0):.2f}  "
                  f"ed={comp.get('education',0):.2f}  "
                  f"pe={comp.get('production_evidence',0):.2f}  "
                  f"bm={r.get('multiplier',0):.2f}")
        print(f"       {r['reasoning']}")
        print()
