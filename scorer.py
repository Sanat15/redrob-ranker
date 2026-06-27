"""
Redrob Hackathon — Candidate Scorer
JD: Senior AI Engineer — Founding Team @ Redrob AI

This module implements the full scoring pipeline.
Import and use score_candidate(candidate: dict) -> float
"""

import math
from datetime import date, datetime

# ─────────────────────────────────────────────
# JD-derived constants
# ─────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 23)   # today; update if needed

# Core skills: these are the "absolutely need" skills from the JD
CORE_SKILLS = {
    # Embedding retrieval systems
    "embeddings", "sentence transformers", "sentence-transformers",
    "vector search", "semantic search", "dense retrieval",
    # Vector databases / hybrid search infra
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "annoy", "chromadb",
    # Ranking / IR
    "information retrieval", "ranking", "learning to rank",
    "learning-to-rank", "ltr", "bm25", "hybrid search",
    "reranking", "re-ranking",
    # Python (hard requirement)
    "python",
    # Evaluation
    "a/b testing", "ab testing", "experimentation",
    "ndcg", "mrr", "map", "offline evaluation",
}

# Nice-to-have skills (boost, not required)
NICHE_SKILLS = {
    # LLM fine-tuning
    "llm fine-tuning", "fine-tuning llms", "fine-tuning",
    "lora", "qlora", "peft", "rlhf",
    # NLP / transformers
    "nlp", "natural language processing",
    "transformers", "hugging face transformers", "hugging face",
    "bert", "rag", "retrieval augmented generation",
    # ML general
    "xgboost", "lightgbm", "gradient boosting",
    "machine learning", "applied ml", "mlops", "mlflow",
    "feature engineering",
    # Infra
    "kafka", "spark", "flink", "distributed systems",
    "kubernetes", "docker", "aws", "gcp",
    # Monitoring / eval
    "weights & biases", "wandb", "data drift",
}

# Exact skill names as they appear in data (case-insensitive match below)
SKILL_NORMALIZE = {
    "sentence transformers": "sentence transformers",
    "hugging face transformers": "hugging face transformers",
    "information retrieval": "information retrieval",
    "fine-tuning llms": "fine-tuning",
    "llm fine-tuning": "fine-tuning",
}

# Pure consulting firms (entire career at these = heavy penalty)
BIG_CONSULTING = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcltech", "tech mahindra", "mphasis", "hexaware",
}

# Product companies that signal strong background for this role
STRONG_PRODUCT_COS = {
    "swiggy", "zomato", "flipkart", "meesho", "razorpay", "cred",
    "paytm", "phonepe", "ola", "rapido", "zepto", "blinkit",
    "dunzo", "slice", "groww", "zerodha", "upstox", "freshworks",
    "postman", "hasura", "setu", "atlassian", "microsoft", "google",
    "amazon", "meta", "apple", "netflix", "uber", "airbnb", "stripe",
    "mad street den", "sharechat", "dailyhunt", "inmobi",
    "cleartrip", "makemytrip", "yatra", "healthkart", "pharmeasy",
    "practo", "1mg", "byju", "unacademy", "vedantu",
}

# Non-tech roles that disqualify regardless of skills listed
NON_TECH_TITLES = {
    "marketing manager", "marketing executive", "marketing specialist",
    "sales manager", "sales executive", "sales representative",
    "hr manager", "hr executive", "human resources",
    "customer support", "customer service", "customer success",
    "operations manager", "operations executive",
    "project manager",   # edge case — only if no technical substance
    "business analyst",  # lower score, not disqualified
    "content writer", "content manager",
    "finance manager", "accountant",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "chennai", "kolkata",
}

EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.3,
    "unknown": 0.4,
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


def _days_since(date_str: str) -> int:
    try:
        d = date.fromisoformat(date_str)
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999


def _skill_trust(skill: dict) -> float:
    """Compute trust-weighted skill score for a single skill entry."""
    proficiency_map = {
        "expert": 1.0,
        "advanced": 0.75,
        "intermediate": 0.4,
        "beginner": 0.15,
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

def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if the candidate profile has impossible signals.
    Honeypots get score = 0.01 to stay out of top 100.
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # Pattern 1: expert skill with 0 duration months
    expert_zero_duration = [
        s for s in skills
        if _norm(s.get("proficiency", "")) == "expert"
        and s.get("duration_months", 1) == 0
    ]
    if expert_zero_duration:
        return True

    # Pattern 2: multiple expert skills with both 0 endorsements and < 6 months duration
    suspicious_expert = [
        s for s in skills
        if _norm(s.get("proficiency", "")) == "expert"
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) < 6
    ]
    if len(suspicious_expert) >= 3:
        return True

    # Pattern 3: career history total months far exceeds stated YoE
    career_months = sum(j.get("duration_months", 0) for j in career)
    yoe_months = profile.get("years_of_experience", 0) * 12
    if career_months > yoe_months * 1.4 + 24:
        return True

    return False


# ─────────────────────────────────────────────
# Component Scorers
# ─────────────────────────────────────────────

def title_career_score(candidate: dict) -> float:
    """
    Score based on current title + full career history.
    Rewards: ML/AI/Search engineers at product companies.
    Penalizes: non-tech titles, pure consulting careers.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = _norm(profile.get("current_title", ""))
    current_industry = _norm(profile.get("current_industry", ""))

    # Hard disqualifier: non-tech current title
    for bad_title in NON_TECH_TITLES:
        if bad_title in current_title:
            return 0.0

    # Determine base score from title keywords
    strong_ml_title_keywords = {
        "recommendation", "ranking", "retrieval", "search",
        "nlp", "ml engineer", "machine learning", "applied ml",
        "applied ai", "ai engineer", "information retrieval",
        "vector", "embedding",
    }
    moderate_ml_keywords = {
        "data scientist", "data science", "research engineer",
        "llm", "generative", "foundation model",
    }
    generic_swe_keywords = {
        "software engineer", "sde", "backend", "fullstack",
        "full-stack", "platform engineer",
    }
    data_eng_keywords = {
        "data engineer", "analytics engineer", "etl",
    }

    if any(k in current_title for k in strong_ml_title_keywords):
        base = 0.85
    elif any(k in current_title for k in moderate_ml_keywords):
        base = 0.70
    elif any(k in current_title for k in generic_swe_keywords):
        base = 0.50
    elif any(k in current_title for k in data_eng_keywords):
        base = 0.40
    else:
        base = 0.25  # Other tech titles (DevOps, QA, etc.)

    # Company quality signal from career history
    all_companies = [_norm(j.get("company", "")) for j in career]
    consulting_count = sum(1 for co in all_companies if any(bc in co for bc in BIG_CONSULTING))
    product_count = sum(1 for co in all_companies if any(pc in co for pc in STRONG_PRODUCT_COS))

    total_jobs = len(all_companies)
    if total_jobs == 0:
        company_modifier = 0.8
    elif consulting_count == total_jobs:
        # Entire career at consulting — heavy penalty per JD
        company_modifier = 0.4
    elif consulting_count > 0 and product_count == 0:
        # Mostly consulting, some non-consulting but not strong product cos
        company_modifier = 0.7
    elif product_count > 0:
        # At least some product company experience
        product_ratio = product_count / total_jobs
        company_modifier = 0.9 + 0.1 * product_ratio
    else:
        company_modifier = 0.85  # Unknown companies, assume neutral

    # Industry modifier
    if "software" in current_industry or "ai" in current_industry or "ml" in current_industry:
        industry_modifier = 1.05
    elif "it services" in current_industry:
        industry_modifier = 0.9
    else:
        industry_modifier = 1.0

    score = base * company_modifier * industry_modifier
    return min(1.0, max(0.0, score))


def skills_score(candidate: dict) -> float:
    """
    Score based on skill match with JD requirements.
    Uses trust-weighted scoring (proficiency × endorsements × duration).
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    core_coverage = 0.0
    niche_bonus = 0.0
    core_matched = set()
    niche_matched = set()

    for skill_entry in skills:
        raw_name = _norm(skill_entry.get("name", ""))
        # Normalize aliases
        name = SKILL_NORMALIZE.get(raw_name, raw_name)
        trust = _skill_trust(skill_entry)

        if name in CORE_SKILLS or raw_name in CORE_SKILLS:
            if name not in core_matched:
                core_coverage += trust
                core_matched.add(name)
        elif name in NICHE_SKILLS or raw_name in NICHE_SKILLS:
            if name not in niche_matched:
                niche_bonus += trust * 0.5
                niche_matched.add(name)

    # Normalize: ~5 core skills at full trust = 1.0
    core_score = min(1.0, core_coverage / 5.0)
    # Normalize: ~4 niche skills at full trust = 0.5 bonus max
    niche_score = min(0.5, niche_bonus / 4.0)

    # Also reward assessment scores if available
    assessment_bonus = 0.0
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for skill_name, score_val in assessments.items():
        norm_name = _norm(skill_name)
        if norm_name in CORE_SKILLS or norm_name in NICHE_SKILLS:
            assessment_bonus += (score_val / 100.0) * 0.05
    assessment_bonus = min(0.1, assessment_bonus)

    return min(1.0, core_score * 0.7 + niche_score * 0.3 + assessment_bonus)


def experience_score(candidate: dict) -> float:
    """Score based on years of experience — sweet spot is 5–9 per JD."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe < 2:
        return 0.0
    elif yoe < 4:
        return 0.3 + (yoe - 2) / 2.0 * 0.3   # 0.3 → 0.6
    elif yoe <= 9:
        return 0.7 + (yoe - 4) / 5.0 * 0.25   # 0.7 → 0.95
    elif yoe <= 12:
        return 0.95
    else:
        return 0.85   # Slight dip — may be overqualified/overpriced


def location_score(candidate: dict) -> float:
    """Score based on India location + city + relocation willingness."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = _norm(profile.get("country", ""))
    location = _norm(profile.get("location", ""))
    willing = signals.get("willing_to_relocate", False)
    work_mode = _norm(signals.get("preferred_work_mode", ""))

    if country == "india":
        # Check if they're in a preferred city
        in_preferred = any(city in location for city in PREFERRED_LOCATIONS)
        if in_preferred:
            return 1.0
        elif willing:
            return 0.90
        else:
            return 0.70
    else:
        # Outside India
        if willing:
            return 0.50
        elif work_mode in ("remote", "flexible"):
            return 0.35
        else:
            return 0.10


def education_score(candidate: dict) -> float:
    """Weak signal per JD — only contributes 10% to final score."""
    education = candidate.get("education", [])
    if not education:
        return 0.4   # Unknown baseline

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


def behavioral_multiplier(candidate: dict) -> float:
    """
    Availability/engagement multiplier based on Redrob behavioral signals.
    Returns a value between 0.25 and 1.0 to multiply the base score.
    """
    signals = candidate.get("redrob_signals", {})

    multiplier = 1.0

    # Not open to work is a major availability signal
    if not signals.get("open_to_work_flag", True):
        multiplier *= 0.5

    # Recency of platform activity
    days_active = _days_since(signals.get("last_active_date", "2020-01-01"))
    if days_active > 180:
        multiplier *= 0.6
    elif days_active > 90:
        multiplier *= 0.8

    # Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.10:
        multiplier *= 0.7
    elif response_rate < 0.25:
        multiplier *= 0.85

    # Notice period (JD loves sub-30; buyout up to 30 days available)
    notice = signals.get("notice_period_days", 60)
    if notice > 150:
        multiplier *= 0.70
    elif notice > 90:
        multiplier *= 0.85

    # Floor: even unavailable candidates may be worth showing
    return max(0.25, multiplier)


# ─────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────

WEIGHTS = {
    "title_career": 0.35,
    "skills":       0.30,
    "experience":   0.15,
    "location":     0.10,
    "education":    0.10,
}


def score_candidate(candidate: dict) -> dict:
    """
    Score a single candidate against the Senior AI Engineer JD.
    
    Returns a dict with:
        score: float (0.0–1.0, final composite after behavioral multiplier)
        components: dict of individual component scores
        multiplier: float (behavioral multiplier applied)
        is_honeypot: bool
        reasoning: str (1–2 sentence justification)
    """
    if is_honeypot(candidate):
        return {
            "score": 0.01,
            "components": {},
            "multiplier": 0.0,
            "is_honeypot": True,
            "reasoning": "Profile contains impossible signals (expert skills with no usage history) — likely honeypot.",
        }

    tc = title_career_score(candidate)
    sk = skills_score(candidate)
    ex = experience_score(candidate)
    lo = location_score(candidate)
    ed = education_score(candidate)

    base = (
        WEIGHTS["title_career"] * tc +
        WEIGHTS["skills"]       * sk +
        WEIGHTS["experience"]   * ex +
        WEIGHTS["location"]     * lo +
        WEIGHTS["education"]    * ed
    )

    bm = behavioral_multiplier(candidate)
    final = base * bm

    reasoning = _build_reasoning(candidate, tc, sk, ex, lo, bm, final)

    return {
        "score": round(final, 6),
        "components": {
            "title_career": round(tc, 3),
            "skills": round(sk, 3),
            "experience": round(ex, 3),
            "location": round(lo, 3),
            "education": round(ed, 3),
        },
        "multiplier": round(bm, 3),
        "is_honeypot": False,
        "reasoning": reasoning,
    }


def _build_reasoning(candidate, tc, sk, ex, lo, bm, final) -> str:
    """Generate specific, honest, rank-consistent reasoning."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "")
    notice = signals.get("notice_period_days", "?")
    open_to = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0)
    days_active = _days_since(signals.get("last_active_date", "2020-01-01"))

    # Build strength and concern notes
    strengths = []
    concerns = []

    # Title / career
    if tc >= 0.8:
        strengths.append(f"{title} at {company} directly matches retrieval/ranking focus of the JD")
    elif tc >= 0.55:
        strengths.append(f"{yoe:.0f} yrs as {title} shows relevant technical background")
    elif tc < 0.35:
        concerns.append(f"current role ({title}) is not a strong fit for AI engineering")

    # Skills
    if sk >= 0.7:
        strengths.append("strong core ML/search skill set with evidence of use")
    elif sk >= 0.4:
        strengths.append("partial skill match — has some key skills but gaps exist")
    else:
        concerns.append("limited evidence of required retrieval/ranking skills")

    # Location
    if country != "India" and not signals.get("willing_to_relocate", False):
        concerns.append(f"based outside India ({country}) and not open to relocation")
    elif location:
        strengths.append(f"{location}-based")

    # Behavioral
    if not open_to:
        concerns.append("not marked open to work")
    if days_active > 180:
        concerns.append(f"inactive for {days_active // 30} months")
    if notice > 90:
        concerns.append(f"{notice}-day notice period exceeds JD preference")

    # Compose
    strength_str = "; ".join(strengths[:2]) if strengths else "some relevant background"
    concern_str = "; ".join(concerns[:2]) if concerns else None

    if concern_str:
        return f"{strength_str}. Concerns: {concern_str}."
    else:
        return f"{strength_str}. Strong behavioral engagement signals."


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        if isinstance(data, list):
            candidates = data
        else:
            candidates = [data]
    else:
        # Inline micro-test
        print("Usage: python scorer.py sample_candidates.json")
        sys.exit(0)

    results = []
    for c in candidates:
        r = score_candidate(c)
        results.append((r["score"], c["candidate_id"], c["profile"]["current_title"], r))

    results.sort(reverse=True)
    for i, (score, cid, title, r) in enumerate(results, 1):
        hp = " [HONEYPOT]" if r["is_honeypot"] else ""
        print(f"Rank {i:3d} | {score:.4f} | {cid} | {title}{hp}")
        print(f"         Components: {r['components']} | Behavioral: {r['multiplier']}")
        print(f"         {r['reasoning']}")
        print()
