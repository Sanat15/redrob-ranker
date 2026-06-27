"""
Redrob Hackathon — Candidate Scorer v2
JD: Senior AI Engineer — Founding Team @ Redrob AI

Changes from v1:
  - Expanded CORE_SKILLS (recommendation systems, haystack, bi/cross-encoder, etc.)
  - Expanded STRONG_PRODUCT_COS (fictional dataset companies + Indian AI cos)
  - behavioral_multiplier: added github_activity_score bonus + interview_completion_rate penalty
  - _build_reasoning: now emits specific numbers (YoE, skill endorsements/duration,
    response rate, notice period) — required to pass Stage 4 manual review
  - score_candidate: final score capped at 1.0
"""

import math
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

    # Ranking / IR
    "information retrieval", "ranking", "learning to rank",
    "learning-to-rank", "ltr", "bm25", "hybrid search",
    "reranking", "re-ranking",

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

def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if the profile has impossible or fabricated signals.
    Honeypots get score = 0.01 to stay out of top 100.

    Three detection patterns (from dataset analysis):
      1. Any skill with proficiency=expert AND duration_months=0
      2. 3+ expert skills with 0 endorsements AND < 6 months duration
      3. Career timeline months >> stated YoE (impossible overlap)
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # Pattern 1: expert skill with zero usage — physically impossible
    for s in skills:
        if _norm(s.get("proficiency", "")) == "expert" and s.get("duration_months", 1) == 0:
            return True

    # Pattern 2: multiple unverified expert skills
    suspicious = [
        s for s in skills
        if _norm(s.get("proficiency", "")) == "expert"
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) < 6
    ]
    if len(suspicious) >= 3:
        return True

    # Pattern 3: career timeline longer than stated experience allows
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
    Rewards: ML/AI/Search/Ranking engineers at product companies.
    Penalizes: non-tech titles, pure consulting careers.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = _norm(profile.get("current_title", ""))
    current_industry = _norm(profile.get("current_industry", ""))

    # Hard disqualifier: non-tech current title
    # (excluding weak-title-override roles that get low-but-nonzero score)
    for bad_title in NON_TECH_TITLES:
        if bad_title in current_title:
            # Project Manager / Business Analyst — not hard zero, just weak
            if bad_title in WEAK_TITLE_OVERRIDE:
                base = 0.15
                break
            return 0.0
    else:
        # Determine base score from title keywords
        strong_ml_keywords = {
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
            "full-stack", "platform engineer", "senior engineer",
        }
        data_eng_keywords = {
            "data engineer", "analytics engineer", "etl",
        }

        if any(k in current_title for k in strong_ml_keywords):
            base = 0.85
        elif any(k in current_title for k in moderate_ml_keywords):
            base = 0.70
        elif any(k in current_title for k in generic_swe_keywords):
            base = 0.50
        elif any(k in current_title for k in data_eng_keywords):
            base = 0.40
        else:
            base = 0.25   # Other tech titles (DevOps, QA, Mobile, etc.)

    # ── Company quality modifier ──────────────────────────────────────────────
    all_companies = [_norm(j.get("company", "")) for j in career]
    consulting_count = sum(
        1 for co in all_companies
        if any(bc in co for bc in BIG_CONSULTING)
    )
    product_count = sum(
        1 for co in all_companies
        if any(pc in co for pc in STRONG_PRODUCT_COS)
    )

    total_jobs = len(all_companies)
    if total_jobs == 0:
        company_modifier = 0.8
    elif consulting_count == total_jobs:
        # Entire career at consulting — JD explicitly disqualifies this
        company_modifier = 0.4
    elif consulting_count > 0 and product_count == 0:
        # Mix of consulting + generic companies, no strong product cos
        company_modifier = 0.7
    elif product_count > 0:
        # At least some product company experience
        product_ratio = product_count / total_jobs
        company_modifier = 0.9 + 0.1 * product_ratio
    else:
        company_modifier = 0.85   # Unknown companies — neutral assumption

    # Industry signal (minor)
    if any(k in current_industry for k in ("software", "ai", "ml", "saas", "technology")):
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
    Keyword stuffers (expert with 0 usage) contribute 0.0.
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

    # ~5 genuine core skills at full trust → core_score = 1.0
    core_score = min(1.0, core_coverage / 5.0)
    # ~4 niche skills at full trust → niche_score = 0.5 max
    niche_score = min(0.5, niche_bonus / 4.0)

    # Platform assessment bonus (verified scores from Redrob)
    assessment_bonus = 0.0
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for skill_name, score_val in assessments.items():
        norm_name = _norm(skill_name)
        if norm_name in CORE_SKILLS or norm_name in NICHE_SKILLS:
            assessment_bonus += (score_val / 100.0) * 0.05
    assessment_bonus = min(0.1, assessment_bonus)

    return min(1.0, core_score * 0.7 + niche_score * 0.3 + assessment_bonus)


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
        in_preferred = any(city in location for city in PREFERRED_LOCATIONS)
        if in_preferred:
            return 1.0
        elif willing:
            return 0.90
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

    # Recency of platform activity
    days_active = _days_since(signals.get("last_active_date", "2020-01-01"))
    if days_active > 180:
        multiplier *= 0.6
    elif days_active > 90:
        multiplier *= 0.8

    # ── Engagement signals ────────────────────────────────────────────────────

    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.10:
        multiplier *= 0.7
    elif response_rate < 0.25:
        multiplier *= 0.85

    # Notice period (JD prefers sub-30; can buy out up to 30 days)
    notice = signals.get("notice_period_days", 60)
    if notice > 150:
        multiplier *= 0.70
    elif notice > 90:
        multiplier *= 0.85

    # ── Quality signals ───────────────────────────────────────────────────────

    # GitHub activity — positive signal for open-source contributions (JD nice-to-have)
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 60:
        multiplier *= 1.10
    elif github_score >= 30:
        multiplier *= 1.05
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
    "title_career": 0.35,
    "skills":       0.30,
    "experience":   0.15,
    "location":     0.10,
    "education":    0.10,
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
        score:       float 0.0–1.0 (final composite after behavioral multiplier)
        components:  dict of individual component scores
        multiplier:  float (behavioral multiplier applied)
        is_honeypot: bool
        reasoning:   str (1–2 sentences, specific to this candidate)
    """
    if is_honeypot(candidate):
        return {
            "score": 0.01,
            "components": {},
            "multiplier": 0.0,
            "is_honeypot": True,
            "reasoning": (
                "Profile contains impossible signals "
                "(expert skills with no usage history) — likely honeypot candidate."
            ),
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

    # Soft salary fit — Redrob Series A budget is approximately 25-70 LPA.
    salary = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
    salary_max = salary.get("max", 40)
    if salary_max > 85:
        base *= 0.93   # above likely budget ceiling
    elif salary_max < 8:
        base *= 0.95   # suspiciously low — possible data error or very junior

    bm = behavioral_multiplier(candidate)
    final = min(1.0, base * bm)   # cap at 1.0

    reasoning = _build_reasoning(candidate, tc, sk, ex, lo, bm, final)

    return {
        "score": round(final, 6),
        "components": {
            "title_career": round(tc, 3),
            "skills":       round(sk, 3),
            "experience":   round(ex, 3),
            "location":     round(lo, 3),
            "education":    round(ed, 3),
        },
        "multiplier": round(bm, 3),
        "is_honeypot": False,
        "reasoning": reasoning,
    }


def _build_reasoning(candidate, tc, sk, ex, lo, bm, final) -> str:
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
        if tc == 0.0:
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
                  f"bm={r.get('multiplier',0):.2f}")
        print(f"       {r['reasoning']}")
        print()
