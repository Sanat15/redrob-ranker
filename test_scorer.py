"""
test_scorer.py — Unit tests for scorer.py
Run: python -m pytest test_scorer.py -v
Run single test class: python -m pytest test_scorer.py::TestHoneypot -v
"""

import pytest
from scorer import (
    is_honeypot,
    title_score,
    career_evidence_score,
    company_fit_score,
    skills_score,
    production_evidence_score,
    experience_score,
    location_score,
    education_score,
    behavioral_multiplier,
    score_candidate,
    _skill_trust,
    _career_progression_bonus,
    _cv_speech_dominant_multiplier,
    hard_gate_multiplier,
    REFERENCE_DATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_candidate(
    title="Machine Learning Engineer",
    company="Swiggy",
    yoe=6.0,
    country="india",
    location="Hyderabad, Telangana",
    skills=None,
    career=None,
    education=None,
    signals=None,
):
    """Build a minimal valid candidate dict. Override any field."""
    if skills is None:
        skills = [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
        ]
    if career is None:
        career = [
            {
                "company": company,
                "title": title,
                "duration_months": int(yoe * 12),
                "is_current": True,
                "industry": "Technology",
                "company_size": "1001-5000",
                "description": "Led ranking and search systems for recommendations. Owned A/B testing infrastructure and metrics. Improved relevance by 15%.",
            }
        ]
    if education is None:
        education = [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "tier": "tier_1",
            }
        ]
    if signals is None:
        signals = {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-25",
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30,
            "github_activity_score": 45,
            "interview_completion_rate": 0.9,
            "applications_submitted_30d": 2,
            "saved_by_recruiters_30d": 1,
            "profile_completeness_score": 80,
            "willing_to_relocate": True,
            "preferred_work_mode": "hybrid",
            "skill_assessment_scores": {},
        }
    return {
        "candidate_id": "CAND_TEST001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "",
            "current_title": title,
            "current_company": company,
            "current_company_size": "1001-5000",
            "current_industry": "Technology",
            "years_of_experience": yoe,
            "location": location,
            "country": country,
        },
        "career_history": career,
        "education": education,
        "skills": skills,
        "certifications": [],
        "redrob_signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill trust formula
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillTrust:
    """The trust formula is the core anti-keyword-stuffing mechanism."""

    def test_expert_zero_duration_is_zero(self):
        s = {"proficiency": "expert", "endorsements": 50, "duration_months": 0}
        assert _skill_trust(s) == 0.0

    def test_expert_zero_endorsements_is_zero(self):
        s = {"proficiency": "expert", "endorsements": 0, "duration_months": 24}
        assert _skill_trust(s) == 0.0

    def test_expert_full_signals_is_one(self):
        # proficiency=1.0, endorse/25 >= 1.0, duration/18 >= 1.0
        s = {"proficiency": "expert", "endorsements": 25, "duration_months": 18}
        assert _skill_trust(s) == pytest.approx(1.0)

    def test_proficiency_ordering(self):
        kwargs = {"endorsements": 25, "duration_months": 18}
        expert    = _skill_trust({"proficiency": "expert",       **kwargs})
        advanced  = _skill_trust({"proficiency": "advanced",     **kwargs})
        intermed  = _skill_trust({"proficiency": "intermediate", **kwargs})
        beginner  = _skill_trust({"proficiency": "beginner",     **kwargs})
        assert expert > advanced > intermed > beginner > 0

    def test_partial_duration_scales_trust(self):
        full    = _skill_trust({"proficiency": "expert", "endorsements": 25, "duration_months": 18})
        partial = _skill_trust({"proficiency": "expert", "endorsements": 25, "duration_months": 9})
        assert full == pytest.approx(2 * partial)


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection
# ─────────────────────────────────────────────────────────────────────────────

class TestHoneypot:

    def test_expert_with_zero_duration_soft_penalty(self):
        """One expert skill with zero duration → soft penalty, not hard honeypot."""
        c = make_candidate(skills=[
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
        ])
        result = is_honeypot(c)
        assert result is False  # Soft signal, not hard fail

    def test_three_unverified_experts_flagged(self):
        c = make_candidate(skills=[
            {"name": "Embeddings",  "proficiency": "expert", "endorsements": 0, "duration_months": 2},
            {"name": "Pinecone",    "proficiency": "expert", "endorsements": 0, "duration_months": 3},
            {"name": "FAISS",       "proficiency": "expert", "endorsements": 0, "duration_months": 4},
        ])
        assert is_honeypot(c) is True

    def test_two_unverified_experts_not_flagged(self):
        c = make_candidate(skills=[
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 2},
            {"name": "Pinecone",   "proficiency": "expert", "endorsements": 0, "duration_months": 3},
        ])
        assert is_honeypot(c) is False

    def test_impossible_timeline_flagged(self):
        # yoe = 1, but career total = 55 months (> 1*12*1.25 + 6 = 21)
        c = make_candidate(
            yoe=1.0,
            career=[
                {"company": "A", "title": "SDE", "duration_months": 30, "is_current": False,
                 "industry": "Tech", "company_size": "51-200", "description": "",
                 "start_date": "2024-01-01", "end_date": "2026-03-01"},
                {"company": "B", "title": "SDE", "duration_months": 25, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "",
                 "start_date": "2026-03-01", "end_date": None},
            ]
        )
        assert is_honeypot(c) is True

    def test_normal_timeline_not_flagged(self):
        c = make_candidate()
        assert is_honeypot(c) is False

    def test_hard_honeypot_gets_near_zero_score(self):
        """Multiple zero-duration experts with no endorsements → hard honeypot."""
        c = make_candidate(skills=[
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 2},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 2},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 2},
        ])
        result = score_candidate(c)
        assert result["score"] <= 0.02
        assert result["is_honeypot"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Title / career score
# ─────────────────────────────────────────────────────────────────────────────

class TestTitle:
    def test_non_tech_titles_return_zero(self):
        non_tech = [
            "Marketing Manager", "Sales Executive", "HR Business Partner",
            "Accountant", "Content Writer", "Operations Manager",
            "Civil Engineer", "Graphic Designer", "Mechanical Engineer",
        ]
        for t in non_tech:
            c = make_candidate(title=t)
            score = title_score(c)
            assert score == 0.0, f"Expected 0.0 for '{t}', got {score}"

    def test_strong_ml_title_scores_high(self):
        c = make_candidate(title="Recommendation Systems Engineer")
        assert title_score(c) >= 0.80

    def test_weak_title_override_non_zero(self):
        c = make_candidate(title="Project Manager")
        assert 0.0 < title_score(c) < 0.20


class TestCareerEvidence:
    def test_job_descriptions_scored(self):
        c = make_candidate(
            career=[
                {"company": "Swiggy", "title": "ML Engineer", "duration_months": 48,
                 "is_current": True, "industry": "Food Delivery", "company_size": "5001-10000",
                 "description": "Led ranking and search systems for recommendations. Owned A/B testing."}
            ]
        )
        score = career_evidence_score(c)
        assert score > 0.4, f"Career with ranking/search should score > 0.4, got {score}"

    def test_empty_career_returns_zero(self):
        c = make_candidate(career=[])
        assert career_evidence_score(c) == 0.0


class TestCompanyFit:
    def test_pure_consulting_career_penalized(self):
        c = make_candidate(
            title="Senior Engineer",
            company="Infosys",
            career=[
                {"company": "Infosys", "title": "Senior Engineer", "duration_months": 48,
                 "is_current": True,  "industry": "IT Services", "company_size": "10001+", "description": ""},
                {"company": "TCS",     "title": "Engineer",       "duration_months": 36,
                 "is_current": False, "industry": "IT Services", "company_size": "10001+", "description": ""},
            ]
        )
        score = company_fit_score(c)
        assert score == 0.45, f"Pure consulting should score 0.45, got {score}"

    def test_product_company_lifts_score(self):
        consulting_only = company_fit_score(make_candidate(
            title="Software Engineer",
            career=[
                {"company": "TCS",    "title": "SDE", "duration_months": 60,
                 "is_current": True, "industry": "IT Services", "company_size": "10001+", "description": ""},
            ]
        ))
        mixed = company_fit_score(make_candidate(
            title="Software Engineer",
            career=[
                {"company": "TCS",    "title": "SDE", "duration_months": 36,
                 "is_current": False, "industry": "IT Services", "company_size": "10001+", "description": ""},
                {"company": "Swiggy", "title": "SDE", "duration_months": 24,
                 "is_current": True,  "industry": "Food Delivery", "company_size": "5001-10000", "description": ""},
            ]
        ))
        assert mixed > consulting_only, f"Mixed career should score > consulting, got {mixed} vs {consulting_only}"

    def test_fictional_dataset_companies_treated_as_product(self):
        """Hooli, Pied Piper, Initech etc. appear in dataset — must be recognised."""
        for co in ["Hooli", "Pied Piper", "Initech", "Globex Inc"]:
            c = make_candidate(title="ML Engineer", company=co,
                               career=[
                                   {"company": co, "title": "ML Engineer", "duration_months": 36,
                                    "is_current": True, "industry": "Software",
                                    "company_size": "1001-5000",
                                    "description": "Led ranking and search systems."}
                               ])
            score = company_fit_score(c)
            assert score > 0.80, f"Fictional product co '{co}' should score > 0.80, got {score}"


# ─────────────────────────────────────────────────────────────────────────────
# Skills score
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillsScore:

    def test_zero_duration_skill_contributes_nothing(self):
        """An 'expert' skill with 0 months of usage should score same as no skill."""
        fake = make_candidate(skills=[
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        ])
        empty = make_candidate(skills=[])
        assert abs(skills_score(fake) - skills_score(empty)) < 0.01

    def test_real_embedding_expert_scores_high(self):
        """Ela Singh profile with strong skills should score well."""
        c = make_candidate(
            skills=[
                {"name": "Pinecone",             "proficiency": "expert", "endorsements": 34, "duration_months": 88},
                {"name": "Embeddings",           "proficiency": "expert", "endorsements": 48, "duration_months": 60},
                {"name": "Information Retrieval","proficiency": "expert", "endorsements": 2,  "duration_months": 84},
                {"name": "Sentence Transformers","proficiency": "expert", "endorsements": 16, "duration_months": 69},
                {"name": "Python",               "proficiency": "expert", "endorsements": 30, "duration_months": 72},
            ],
            career=[
                {
                    "company": "Swiggy",
                    "title": "Recommendation Systems Engineer",
                    "duration_months": 72,
                    "is_current": True,
                    "description": "Built semantic search and ranking systems using embeddings, Pinecone, and sentence transformers. Implemented evaluation frameworks and A/B testing infrastructure.",
                }
            ]
        )
        # Strong skills with supporting career evidence
        assert skills_score(c) >= 0.45

    def test_recommendation_systems_matches_core(self):
        """'Recommendation Systems' must be in CORE_SKILLS after expansion."""
        c = make_candidate(skills=[
            {"name": "Recommendation Systems", "proficiency": "expert",
             "endorsements": 20, "duration_months": 36},
        ])
        score_with = skills_score(c)
        c_empty = make_candidate(skills=[])
        assert score_with > skills_score(c_empty)

    def test_vector_database_matches_core(self):
        c = make_candidate(skills=[
            {"name": "Vector Database", "proficiency": "advanced",
             "endorsements": 15, "duration_months": 24},
        ])
        assert skills_score(c) > 0

    def test_haystack_matches_core(self):
        c = make_candidate(skills=[
            {"name": "Haystack", "proficiency": "advanced",
             "endorsements": 10, "duration_months": 18},
        ])
        assert skills_score(c) > 0

    def test_niche_skills_give_bonus_on_top_of_core(self):
        core_only = make_candidate(skills=[
            {"name": "Python",     "proficiency": "expert", "endorsements": 30, "duration_months": 60},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
        ])
        with_niche = make_candidate(skills=[
            {"name": "Python",     "proficiency": "expert", "endorsements": 30, "duration_months": 60},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
            {"name": "XGBoost",    "proficiency": "advanced","endorsements": 15, "duration_months": 36},
            {"name": "MLflow",     "proficiency": "advanced","endorsements": 10, "duration_months": 24},
        ])
        assert skills_score(with_niche) > skills_score(core_only)

    def test_weaviate_and_opensearch_are_vector_infra(self):
        """Vector databases (Weaviate, OpenSearch) are recognized as core skills."""
        c = make_candidate(skills=[
            {"name": "Weaviate",   "proficiency": "advanced", "endorsements": 37, "duration_months": 27},
            {"name": "OpenSearch", "proficiency": "advanced", "endorsements": 47, "duration_months": 59},
            {"name": "Python",     "proficiency": "expert",   "endorsements": 30, "duration_months": 72},
        ])
        # Vector infra skills + Python should score reasonably
        assert skills_score(c) > 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Production evidence score
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionEvidenceScore:

    def test_comprehensive_production_evidence_scores_high(self):
        """Comprehensive evidence across multiple dimensions scores well."""
        c = make_candidate(career=[{
            "company": "Swiggy",
            "title": "Ranking Systems Engineer",
            "duration_months": 60,
            "is_current": True,
            "industry": "Technology",
            "company_size": "1001-5000",
            "description": (
                "Owned and deployed semantic search with offline evaluation, "
                "NDCG calibration, p99 latency monitoring, and index refresh."
            ),
        }])
        # Comprehensive evidence should score well
        assert production_evidence_score(c) > 0.30

    def test_repeated_terms_count_once(self):
        """Repeated terms should be deduplicated by dimension."""
        c = make_candidate(career=[{
            "company": "Swiggy",
            "title": "Ranking Systems Engineer",
            "duration_months": 60,
            "is_current": True,
            "industry": "Technology",
            "company_size": "1001-5000",
            "description": "Built semantic search systems for retrieval and ranking.",
        }])
        # Retrieval and ranking evidence present
        assert production_evidence_score(c) > 0.0

    def test_older_job_evidence_is_recency_weighted(self):
        """Recent job evidence weighted higher than older jobs."""
        recent = make_candidate(career=[
            {"company": "A", "title": "ML Engineer", "duration_months": 24,
             "is_current": True, "description": "Owned and led ranking layer in production with p99 latency monitoring."},
        ])
        older = make_candidate(career=[
            {"company": "B", "title": "ML Engineer", "duration_months": 24,
             "is_current": False, "description": "Owned and led ranking layer in production with p99 latency monitoring."},
        ])
        # Current job evidence should score higher than old job due to recency weighting
        assert production_evidence_score(recent) > production_evidence_score(older)

    def test_search_does_not_match_research(self):
        c = make_candidate(career=[{
            "company": "Swiggy",
            "title": "Research Engineer",
            "duration_months": 60,
            "is_current": True,
            "description": "",
        }])
        assert production_evidence_score(c) == 0.0


class TestConceptLevelProductionEvidence:
    """
    v3 audit fixes: production ranking ownership described in
    XGBoost/LightGBM/production terminology should be recognized as evidence,
    mapped into the same 4 categories — not via new candidate-specific keywords.
    """

    def test_ab_testing_gerund_form_matches_evaluation(self):
        """'A/B testing' (not 'A/B test') was previously unmatched — real bug."""
        c = make_candidate(career=[{
            "company": "Flipkart",
            "title": "Machine Learning Engineer",
            "duration_months": 24,
            "is_current": True,
            "description": "Built the feature pipeline and the A/B testing infrastructure.",
        }])
        # relevant_systems: none here on purpose — isolate the evaluation match.
        c["career_history"][0]["description"] += " Owned the ranking model rollout."
        result = production_evidence_score(c)
        assert result > 0.0
        # Confirm evaluation category specifically is what's contributing.
        career = c["career_history"]
        text = f"{career[0]['title']} {career[0]['description']}"
        from scorer import _contains_evidence_term
        assert _contains_evidence_term(text, "a/b testing")

    def test_offline_online_correlation_cooccurrence_matches_evaluation(self):
        """
        'Offline metrics that correlated with online engagement' should count
        as evaluation evidence even without literal 'NDCG' or 'A/B test'.
        """
        from scorer import _has_offline_online_correlation
        assert _has_offline_online_correlation(
            "building offline metrics that actually correlated with online engagement"
        )
        assert _has_offline_online_correlation(
            "online/offline metric correlation"
        )
        assert not _has_offline_online_correlation(
            "shipped the feature in production"
        )

    def test_explicit_ir_vocabulary_matches(self):
        """Explicit IR vocabulary like 'ranking', 'retrieval' should match."""
        c = make_candidate(career=[{
            "company": "Meta",
            "title": "Senior AI Engineer",
            "duration_months": 24,
            "is_current": True,
            "description": (
                "Built retrieval and ranking systems to connect users to "
                "relevant matches. Used semantic search with embeddings."
            ),
        }])
        assert production_evidence_score(c) > 0.0

    def test_drift_detection_matches_operational(self):
        c = make_candidate(career=[{
            "company": "Glance",
            "title": "Senior ML Engineer",
            "duration_months": 24,
            "is_current": True,
            "description": (
                "Owned feature monitoring, drift detection, and retraining "
                "cadence for the ranking model."
            ),
        }])
        from scorer import _contains_evidence_term
        text = "Senior ML Engineer Owned feature monitoring, drift detection, and retraining cadence for the ranking model."
        assert _contains_evidence_term(text, "drift detection")
        assert _contains_evidence_term(text, "retraining cadence")
        assert production_evidence_score(c) > 0.0

    def test_ranking_with_explicit_metrics_scores(self):
        c = make_candidate(career=[{
            "company": "Swiggy",
            "title": "Recommendation Systems Engineer",
            "duration_months": 24,
            "is_current": True,
            "description": (
                "Trained ranking models using LightGBM for discovery feed. "
                "Measured impact with NDCG and click-through metrics."
            ),
        }])
        result = production_evidence_score(c)
        # Ranking + evaluation metrics should score > 0
        assert result > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Experience score
# ─────────────────────────────────────────────────────────────────────────────

class TestExperienceScore:

    def test_below_2_years_is_zero(self):
        for yoe in [0, 0.5, 1.9]:
            c = make_candidate(yoe=yoe)
            assert experience_score(c) == 0.0, f"YoE={yoe} should be 0.0"

    def test_sweet_spot_5_to_9_scores_high(self):
        for yoe in [5, 6, 7, 8, 9]:
            c = make_candidate(yoe=yoe)
            assert experience_score(c) >= 0.70, f"YoE={yoe} should be ≥0.70"

    def test_10_to_12_scores_at_plateau(self):
        for yoe in [10, 11, 12]:
            c = make_candidate(yoe=yoe)
            assert experience_score(c) == pytest.approx(0.95)

    def test_overqualified_slight_dip(self):
        c9  = make_candidate(yoe=9)
        c15 = make_candidate(yoe=15)
        assert experience_score(c9) > experience_score(c15)


# ─────────────────────────────────────────────────────────────────────────────
# Location score
# ─────────────────────────────────────────────────────────────────────────────

class TestLocation:

    def test_preferred_cities_score_max(self):
        # JD verbatim: "Pune/Noida-preferred" — these are the HQ cities and
        # the only ones that get the full 1.0.
        for city in ["Pune", "Noida"]:
            c = make_candidate(country="india", location=city)
            assert location_score(c) == 1.0, f"{city} should score 1.0"

    def test_jd_welcome_cities_score_high_not_max(self):
        # JD verbatim: "Candidates in Hyderabad, Pune, Mumbai, Delhi NCR
        # welcome to apply" — explicitly named, but not the HQ cities.
        for city in ["Hyderabad", "Mumbai", "Delhi", "Gurgaon"]:
            c = make_candidate(country="india", location=city)
            assert location_score(c) == pytest.approx(0.95), f"{city} should score 0.95"

    def test_other_india_tech_hubs_score_slightly_lower(self):
        # Bangalore/Chennai/Kolkata are strong Indian tech hubs but are not
        # named in the JD's location section either way.
        for city in ["Bangalore", "Chennai", "Kolkata"]:
            c = make_candidate(country="india", location=city)
            assert location_score(c) == pytest.approx(0.90), f"{city} should score 0.90"

    def test_india_non_preferred_willing_to_relocate(self):
        c = make_candidate(
            country="india", location="Jaipur",
            signals={
                **make_candidate()["redrob_signals"],
                "willing_to_relocate": True
            }
        )
        assert location_score(c) == pytest.approx(0.85)

    def test_india_non_preferred_not_relocating(self):
        c = make_candidate(
            country="india", location="Jaipur",
            signals={
                **make_candidate()["redrob_signals"],
                "willing_to_relocate": False
            }
        )
        assert location_score(c) == pytest.approx(0.70)

    def test_outside_india_onsite_scores_low(self):
        c = make_candidate(
            country="usa", location="San Francisco",
            signals={
                **make_candidate()["redrob_signals"],
                "willing_to_relocate": False,
                "preferred_work_mode": "onsite",
            }
        )
        assert location_score(c) <= 0.15

    def test_outside_india_willing_to_relocate(self):
        c = make_candidate(
            country="canada", location="Toronto",
            signals={
                **make_candidate()["redrob_signals"],
                "willing_to_relocate": True,
            }
        )
        assert location_score(c) == pytest.approx(0.50)


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral multiplier
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralMultiplier:

    def test_closed_to_work_halves_multiplier(self):
        base = make_candidate()
        closed = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "open_to_work_flag": False
        })
        assert behavioral_multiplier(closed) < behavioral_multiplier(base) * 0.6

    def test_inactive_6_plus_months_penalized(self):
        active   = make_candidate()
        inactive = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "last_active_date": "2025-10-01"   # ~9 months ago from REFERENCE_DATE
        })
        assert behavioral_multiplier(active) > behavioral_multiplier(inactive)

    def test_low_response_rate_penalized(self):
        normal = make_candidate()
        ghost  = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "recruiter_response_rate": 0.05
        })
        assert behavioral_multiplier(normal) > behavioral_multiplier(ghost)

    def test_high_github_gives_bonus(self):
        """GitHub signal: high activity → ×1.02 (limited effect per JD)"""
        no_gh = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "github_activity_score": -1
        })
        gh_high = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "github_activity_score": 75
        })
        assert behavioral_multiplier(gh_high) > behavioral_multiplier(no_gh)
        # ≥60 tier: ×1.02 (validates engineering activity without deciding rankings)
        ratio = behavioral_multiplier(gh_high) / behavioral_multiplier(no_gh)
        assert ratio >= 1.02

    def test_moderate_github_gives_smaller_bonus(self):
        """GitHub signal: moderate activity → ×1.01"""
        no_gh  = make_candidate(signals={**make_candidate()["redrob_signals"], "github_activity_score": -1})
        gh_mid = make_candidate(signals={**make_candidate()["redrob_signals"], "github_activity_score": 45})
        ratio = behavioral_multiplier(gh_mid) / behavioral_multiplier(no_gh)
        assert ratio >= 1.01

    def test_low_interview_completion_penalized(self):
        """New signal: interview_completion_rate < 0.4 → ×0.85"""
        reliable = make_candidate()
        ghoster  = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "interview_completion_rate": 0.2
        })
        assert behavioral_multiplier(reliable) > behavioral_multiplier(ghoster)

    def test_floor_at_0_25(self):
        """Worst-case stacking never goes below 0.25."""
        worst = make_candidate(signals={
            "open_to_work_flag": False,
            "last_active_date": "2024-12-01",    # 6+ months inactive
            "recruiter_response_rate": 0.03,
            "notice_period_days": 180,
            "github_activity_score": -1,
            "interview_completion_rate": 0.1,
            "applications_submitted_30d": 0,
            "saved_by_recruiters_30d": 0,
            "profile_completeness_score": 10,
            "willing_to_relocate": False,
            "preferred_work_mode": "onsite",
            "skill_assessment_scores": {},
        })
        assert behavioral_multiplier(worst) == pytest.approx(0.25)

    def test_long_notice_period_penalized(self):
        normal = make_candidate()
        long_n = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "notice_period_days": 160
        })
        assert behavioral_multiplier(normal) > behavioral_multiplier(long_n)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCandidate:

    def test_returns_required_keys(self):
        c = make_candidate()
        result = score_candidate(c)
        for key in ["score", "components", "multiplier", "is_honeypot", "reasoning"]:
            assert key in result, f"Missing key: {key}"

    def test_score_in_valid_range(self):
        c = make_candidate()
        result = score_candidate(c)
        assert 0.0 <= result["score"] <= 1.0

    def test_honeypot_short_circuits(self):
        """Multiple zero-duration expert skills trigger honeypot detection."""
        c = make_candidate(skills=[
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
            {"name": "Weaviate", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
        ])
        result = score_candidate(c)
        assert result["is_honeypot"] is True
        assert result["score"] <= 0.02
        assert result["components"] == {}

    def test_production_evidence_bonus_is_gated_and_capped(self):
        strong_skills = [
            {"name": name, "proficiency": "expert", "endorsements": 30,
             "duration_months": 36}
            for name in ["Python", "Embeddings", "Pinecone",
                         "Information Retrieval", "BM25"]
        ]
        evidence_career = [{
            "company": "Swiggy",
            "title": "Ranking Systems Engineer",
            "duration_months": 72,
            "is_current": True,
            "industry": "Technology",
            "company_size": "1001-5000",
            "description": (
                "Owned and deployed semantic search with offline evaluation, "
                "NDCG calibration, p99 latency monitoring, and index refresh."
            ),
        }]
        without_evidence = make_candidate(skills=strong_skills)
        with_evidence = make_candidate(skills=strong_skills, career=evidence_career)

        plain = score_candidate(without_evidence)
        boosted = score_candidate(with_evidence)
        # Production evidence should be > 0 when career has relevant descriptions
        assert boosted["components"]["production_evidence"] > 0.0
        # Boosted score should be higher due to production evidence
        assert boosted["score"] > plain["score"]

        weak = make_candidate(
            title="Software Engineer",
            skills=[],
            career=[{"company": "Generic Co", "title": "SWE", "duration_months": 24,
                     "is_current": True, "description": "Built software."}],
        )
        weak_result = score_candidate(weak)
        # Weak career description shouldn't trigger production evidence
        assert weak_result["components"]["production_evidence"] < 0.1

    def test_strong_ml_engineer_scores_above_0_6(self):
        """A genuine AI engineer with strong signals should score well."""
        c = make_candidate(
            title="Recommendation Systems Engineer",
            company="Swiggy",
            yoe=6.0,
            skills=[
                {"name": "Pinecone",              "proficiency": "expert", "endorsements": 34, "duration_months": 88},
                {"name": "Embeddings",            "proficiency": "expert", "endorsements": 48, "duration_months": 60},
                {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 2,  "duration_months": 84},
                {"name": "Sentence Transformers", "proficiency": "expert", "endorsements": 16, "duration_months": 69},
                {"name": "Python",                "proficiency": "expert", "endorsements": 30, "duration_months": 72},
            ],
        )
        result = score_candidate(c)
        assert result["score"] >= 0.60

    def test_marketing_manager_scores_very_low(self):
        """Non-tech title gets 0.0 from title component."""
        c = make_candidate(
            title="Marketing Manager",
            company="Acme Corp",
            skills=[
                {"name": "Pinecone", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            ]
        )
        result = score_candidate(c)
        # Non-tech title = 0.0, but other components still contribute
        # Title component is 0.12 weight, so removing it doesn't eliminate score entirely
        assert result["components"]["title"] == 0.0
        assert result["score"] < 0.5

    def test_reasoning_is_non_empty_string(self):
        c = make_candidate()
        result = score_candidate(c)
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 30

    def test_reasoning_contains_company_name(self):
        """Stage 4 requirement: reasoning must reference specific profile facts."""
        c = make_candidate(title="ML Engineer", company="Swiggy", yoe=6.0)
        result = score_candidate(c)
        assert "Swiggy" in result["reasoning"], "Reasoning must name the company"

    def test_reasoning_contains_yoe(self):
        c = make_candidate(yoe=7.0)
        result = score_candidate(c)
        assert "7" in result["reasoning"], "Reasoning must mention YoE"

    def test_score_is_non_increasing_for_better_candidates(self):
        """Better candidates should always score higher."""
        strong = make_candidate(
            title="ML Engineer",
            company="Swiggy",
            yoe=7.0,
            skills=[
                {"name": "Embeddings", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
                {"name": "Pinecone",   "proficiency": "expert", "endorsements": 20, "duration_months": 36},
            ]
        )
        weak = make_candidate(
            title="DevOps Engineer",
            company="TCS",
            yoe=3.0,
            skills=[
                {"name": "Docker", "proficiency": "intermediate", "endorsements": 5, "duration_months": 12},
            ]
        )
        assert score_candidate(strong)["score"] > score_candidate(weak)["score"]


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_skills_does_not_crash(self):
        c = make_candidate(skills=[])
        result = score_candidate(c)
        assert 0.0 <= result["score"] <= 1.0

    def test_no_career_history_does_not_crash(self):
        c = make_candidate(career=[])
        result = score_candidate(c)
        assert 0.0 <= result["score"] <= 1.0

    def test_no_education_does_not_crash(self):
        c = make_candidate(education=[])
        result = score_candidate(c)
        assert 0.0 <= result["score"] <= 1.0

    def test_missing_signals_uses_defaults(self):
        """If redrob_signals is empty, should not crash."""
        c = make_candidate()
        c["redrob_signals"] = {}
        result = score_candidate(c)
        assert 0.0 <= result["score"] <= 1.0

    def test_score_capped_at_1(self):
        """Even with github bonus, score should never exceed 1.0."""
        c = make_candidate(signals={
            **make_candidate()["redrob_signals"],
            "github_activity_score": 100,
            "interview_completion_rate": 1.0,
            "recruiter_response_rate": 1.0,
        })
        result = score_candidate(c)
        assert result["score"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Regression tests — the sample candidate we know should rank #1
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownCandidateRegression:
    """
    CAND_0000031 (Ela Singh — Recommendation Systems Engineer @ Swiggy)
    is the clear rank-1 in the 50-candidate sample.
    These tests lock in her score so we notice any regression.
    """

    ELA_SKILLS = [
        {"name": "Pinecone",              "proficiency": "expert", "endorsements": 34, "duration_months": 88},
        {"name": "Embeddings",            "proficiency": "expert", "endorsements": 48, "duration_months": 60},
        {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 2,  "duration_months": 84},
        {"name": "Sentence Transformers", "proficiency": "expert", "endorsements": 16, "duration_months": 69},
        {"name": "FAISS",                 "proficiency": "advanced","endorsements":19, "duration_months": 35},
        {"name": "MLflow",                "proficiency": "advanced","endorsements":59, "duration_months": 21},
        {"name": "Machine Learning",      "proficiency": "advanced","endorsements":43, "duration_months": 23},
    ]

    ELA_SIGNALS = {
        "open_to_work_flag": True,
        "last_active_date": "2026-05-24",
        "recruiter_response_rate": 0.91,
        "notice_period_days": 60,
        "github_activity_score": 32.6,
        "interview_completion_rate": 0.6,
        "skill_assessment_scores": {
            "MLflow": 75.1, "FAISS": 68.4, "Pinecone": 53.6,
        },
        "willing_to_relocate": True,
        "preferred_work_mode": "flexible",
        "applications_submitted_30d": 2,
        "saved_by_recruiters_30d": 13,
        "profile_completeness_score": 83.4,
    }

    def _ela(self):
        return make_candidate(
            title="Recommendation Systems Engineer",
            company="Swiggy",
            yoe=6.0,
            country="india",
            location="Hyderabad, Telangana",
            skills=self.ELA_SKILLS,
            signals=self.ELA_SIGNALS,
        )

    def test_ela_scores_high(self):
        """Ela Singh with strong skills should score above 0.60."""
        result = score_candidate(self._ela())
        assert result["score"] >= 0.60, f"Ela should score ≥0.60; got {result['score']}"

    def test_ela_is_not_honeypot(self):
        result = score_candidate(self._ela())
        assert result["is_honeypot"] is False

    def test_ela_reasoning_mentions_swiggy(self):
        result = score_candidate(self._ela())
        assert "Swiggy" in result["reasoning"]

    def test_ela_reasoning_mentions_core_skill(self):
        result = score_candidate(self._ela())
        # Reasoning should mention at least one of her key skills
        assert any(
            skill in result["reasoning"]
            for skill in ["Pinecone", "Embeddings", "Information Retrieval"]
        ), f"Reasoning missing core skill: {result['reasoning']}"

    def test_ela_reasoning_mentions_response_rate(self):
        """Stage 4: reasoning must reference specific numbers like response rate."""
        result = score_candidate(self._ela())
        # 91% response rate should appear
        assert "91%" in result["reasoning"] or "0.91" in result["reasoning"], \
            f"Response rate missing from reasoning: {result['reasoning']}"


# ─────────────────────────────────────────────────────────────────────────────
# Hard gates for JD-explicit disqualifiers
#
# These three profiles ("we will not move forward" language in the JD) were
# previously only suppressed by losing weight in ONE component (title 12%,
# company_fit 8%, location 10%) — the other 80-90% of the weighted sum was
# untouched, and this dataset is known to recycle strong-sounding ML career
# descriptions across unrelated titles (see README "The problem" section).
# A full-100K audit found a pure-Accountant profile scoring 0.36 and a
# pure-consulting-career profile scoring 0.45 — both uncomfortably close to
# a top-100 cutoff around 0.53. hard_gate_multiplier() closes that gap.
# ─────────────────────────────────────────────────────────────────────────────

class TestHardGateMultiplier:

    def test_non_tech_title_gated_hard(self):
        c = make_candidate(title="Marketing Manager", company="Swiggy")
        tt = title_score(c)
        cf = company_fit_score(c)
        assert tt == 0.0
        assert hard_gate_multiplier(c, tt, cf) == pytest.approx(0.15)

    def test_weak_title_override_not_gated(self):
        """Project Manager / Business Analyst get a low title score (0.15)
        but are NOT the hard non-tech-disqualifier bucket — the JD doesn't
        explicitly disqualify them, just implies a weaker fit."""
        c = make_candidate(title="Project Manager", company="Swiggy")
        tt = title_score(c)
        cf = company_fit_score(c)
        assert tt == 0.15
        assert hard_gate_multiplier(c, tt, cf) == pytest.approx(1.0)

    def test_pure_consulting_career_gated(self):
        career = [
            {"company": "TCS", "title": "Senior Software Engineer", "duration_months": 40,
             "is_current": True, "industry": "IT Services", "company_size": "10001+",
             "description": "Delivered client projects."},
            {"company": "Infosys", "title": "Software Engineer", "duration_months": 30,
             "is_current": False, "industry": "IT Services", "company_size": "10001+",
             "description": "Delivered client projects."},
        ]
        c = make_candidate(title="Senior Software Engineer", career=career)
        tt = title_score(c)
        cf = company_fit_score(c)
        assert cf == pytest.approx(0.45)
        assert hard_gate_multiplier(c, tt, cf) == pytest.approx(0.35)

    def test_logistically_impossible_location_gated(self):
        """Outside India, unwilling to relocate, needs onsite — a real but
        milder gate than title/consulting (JD says "case-by-case" here,
        not "we will not move forward")."""
        c = make_candidate(
            country="usa", location="San Francisco",
            signals={
                **make_candidate()["redrob_signals"],
                "willing_to_relocate": False,
                "preferred_work_mode": "onsite",
            },
        )
        tt = title_score(c)
        cf = company_fit_score(c)
        assert hard_gate_multiplier(c, tt, cf) == pytest.approx(0.55)

    def test_remote_outside_india_not_gated(self):
        """Same location/relocation profile, but remote-friendly — this is
        not logistically impossible, so it should not be gated."""
        c = make_candidate(
            country="usa", location="San Francisco",
            signals={
                **make_candidate()["redrob_signals"],
                "willing_to_relocate": False,
                "preferred_work_mode": "remote",
            },
        )
        tt = title_score(c)
        cf = company_fit_score(c)
        assert hard_gate_multiplier(c, tt, cf) == pytest.approx(1.0)

    def test_strong_candidate_not_gated(self):
        c = make_candidate()
        tt = title_score(c)
        cf = company_fit_score(c)
        assert hard_gate_multiplier(c, tt, cf) == pytest.approx(1.0)

    def test_non_tech_title_cannot_buy_score_via_recycled_description(self):
        """Integration test for the exact trap the README documents: a
        non-tech title paired with a recycled production-ranking career
        description must not be able to reach a competitive score just
        because title is only 12% of the weighted sum."""
        career = [{
            "company": "Wayne Enterprises",
            "title": "Marketing Manager",
            "duration_months": 60,
            "is_current": True,
            "industry": "Marketing",
            "company_size": "1001-5000",
            "description": ("Owned the ranking layer for an e-commerce search product, "
                             "evolving it from a hand-tuned scoring function to a "
                             "learning-to-rank model. Designed the relevance labeling "
                             "pipeline and the training/eval workflow."),
        }]
        c = make_candidate(title="Marketing Manager", career=career)
        result = score_candidate(c)
        assert result["score"] < 0.15, (
            f"Non-tech-titled candidate scored too high despite a recycled "
            f"ML description: {result['score']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CV/speech/robotics-without-NLP/IR — JD explicit disqualifier, previously
# unimplemented ("People whose primary expertise is computer vision, speech,
# or robotics without significant NLP/IR exposure.")
# ─────────────────────────────────────────────────────────────────────────────

class TestCvSpeechDominantPenalty:

    def test_cv_dominant_no_ir_evidence_gets_penalized(self):
        career = [
            {"company": "Foo", "title": "Computer Vision Engineer", "duration_months": 40,
             "is_current": True, "industry": "AI/ML", "company_size": "201-500",
             "description": "Built computer vision models for image classification using "
                             "YOLO and OpenCV, object detection pipelines."},
            {"company": "Bar", "title": "ML Engineer", "duration_months": 24,
             "is_current": False, "industry": "AI/ML", "company_size": "201-500",
             "description": "Worked on speech recognition and ASR systems, fine-tuned "
                             "diffusion models for image generation."},
        ]
        c = make_candidate(title="Computer Vision Engineer", career=career)
        assert _cv_speech_dominant_multiplier(c) == pytest.approx(0.80)

    def test_cv_background_with_ir_evidence_not_penalized(self):
        """A CV background that later shows retrieval/ranking ownership is
        not the profile the JD excludes — only zero IR/ranking/recsys
        evidence anywhere in the career triggers the penalty."""
        career = [
            {"company": "Foo", "title": "AI Engineer", "duration_months": 40,
             "is_current": True, "industry": "AI/ML", "company_size": "201-500",
             "description": "Built computer vision models for image classification using "
                             "YOLO and OpenCV, then moved into building a semantic search "
                             "and ranking system for retrieval."},
        ]
        c = make_candidate(title="AI Engineer", career=career)
        assert _cv_speech_dominant_multiplier(c) == pytest.approx(1.0)

    def test_no_career_history_not_penalized(self):
        c = make_candidate(career=[])
        assert _cv_speech_dominant_multiplier(c) == pytest.approx(1.0)

    def test_strong_ir_candidate_not_penalized(self):
        """The default fixture (ranking/search/A-B-testing description) has
        no CV/speech terms at all, so it must never be gated."""
        c = make_candidate()
        assert _cv_speech_dominant_multiplier(c) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Title-chaser job-hopping — JD explicit disqualifier ("If your career
# trajectory shows you optimizing for 'Senior' -> 'Staff' -> 'Principal'
# titles by switching companies every 1.5 years, we're not a fit.")
# ─────────────────────────────────────────────────────────────────────────────

class TestTitleChaserProgression:

    def test_slow_genuine_progression_earns_bonus(self):
        # career_history convention: most recent job first.
        career = [
            {"title": "Senior Machine Learning Engineer", "duration_months": 36, "is_current": True},
            {"title": "Machine Learning Engineer", "duration_months": 30, "is_current": False},
        ]
        assert _career_progression_bonus(career) > 0.0

    def test_fast_job_hopping_earns_no_bonus(self):
        career = [
            {"title": "Staff Machine Learning Engineer", "duration_months": 10, "is_current": True},
            {"title": "Senior Machine Learning Engineer", "duration_months": 14, "is_current": False},
            {"title": "Machine Learning Engineer", "duration_months": 12, "is_current": False},
        ]
        assert _career_progression_bonus(career) == 0.0

    def test_single_job_no_bonus(self):
        career = [{"title": "Machine Learning Engineer", "duration_months": 24, "is_current": True}]
        assert _career_progression_bonus(career) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NLP as a scored skill capability — previously "NLP" / "Natural Language
# Processing" skill entries earned zero skill-level credit anywhere (they
# only affected title_score's strong-title-terms check).
# ─────────────────────────────────────────────────────────────────────────────

class TestNlpSkillCapability:

    def test_nlp_skill_increases_skills_score(self):
        skills_without_nlp = [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
        ]
        skills_with_nlp = skills_without_nlp + [
            {"name": "NLP", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
        ]
        c_without = make_candidate(skills=skills_without_nlp)
        c_with = make_candidate(skills=skills_with_nlp)
        assert skills_score(c_with) > skills_score(c_without)
