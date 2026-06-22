"""
feature_extractor.py — Phase 1: Offline Pre-computation
Reads candidates.jsonl, computes all features, saves features.parquet
Run once (no time limit). Produces features.parquet for the ranker.

Usage:
    python src/feature_extractor.py --candidates ./data/candidates.jsonl --out ./features.parquet
"""

import argparse
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─── JD configuration ──────────────────────────────────────────────────────────

JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Pune Noida Series A
5 to 9 years total experience 4 to 5 years applied ML AI product companies
production embeddings sentence-transformers BGE E5 OpenAI embeddings
production vector database hybrid search Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch
strong Python code quality production ML
ranking evaluation frameworks NDCG MRR MAP information retrieval
shipped end-to-end ranking search recommendation system at scale
retrieval augmented generation RAG LLM fine-tuning LoRA
NLP natural language processing A/B testing
"""

HARD_SKILLS = [
    "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5",
    "vector database", "vector db", "vector store",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "hybrid search", "retrieval", "ranking", "llm", "large language model",
    "fine-tuning", "fine tuning", "finetuning", "lora",
    "ndcg", "mrr", "map", "a/b testing", "ab testing",
    "rag", "retrieval augmented generation",
    "nlp", "natural language processing", "information retrieval",
    "python", "production ml", "recommendation system", "recommendation",
    "transformers", "bert", "gpt", "openai", "huggingface",
    "semantic search", "reranking", "re-ranking", "bi-encoder", "cross-encoder",
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "mindtree", "hexaware", "l&t infotech", "ltimindtree",
]

PRODUCT_COMPANY_SIGNALS = [
    "startup", "product", "saas", "series a", "series b", "series c",
    "seed funded", "vc-backed", "fintech", "edtech", "healthtech",
    "e-commerce", "ecommerce", "platform",
]

PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
    "gurugram", "gurgaon", "ncr",
]

NON_TECH_TITLES = [
    "marketing manager", "accountant", "hr manager", "human resources",
    "operations manager", "customer support", "sales executive", "content writer",
    "graphic designer", "project manager", "civil engineer", "mechanical engineer",
    "business analyst", "recruiter",
]

AI_CORE_SKILLS = {
    "embeddings", "sentence-transformers", "bge", "e5", "pinecone", "weaviate",
    "qdrant", "milvus", "faiss", "elasticsearch", "hybrid search", "retrieval",
    "ranking", "llm", "fine-tuning", "lora", "ndcg", "mrr", "map", "rag", "nlp",
    "python", "recommendation", "transformers", "bert", "gpt", "huggingface",
    "semantic search", "reranking", "information retrieval", "vector database",
    "production ml",
}

# Reference date for recency calculations
REFERENCE_DATE = date(2026, 6, 22)


# ─── Honeypot detection ─────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Returns (is_honeypot, list_of_reasons)."""
    reasons = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Rule 1: Career duration impossibility
    total_months = sum(r.get("duration_months", 0) for r in career)
    yoe = profile.get("years_of_experience", 0)
    if total_months > (yoe * 12) + 18:
        reasons.append(f"impossible_tenure: {total_months}mo > {yoe}yr*12+18")

    # Rule 2: Expert skill with zero duration
    for skill in skills:
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 1) == 0:
            reasons.append(f"expert_zero_duration: {skill['name']}")

    # Rule 3: Non-technical title with many AI keyword skills (keyword stuffer)
    title = profile.get("current_title", "").lower()
    is_non_tech = any(t in title for t in NON_TECH_TITLES)
    if is_non_tech:
        skill_names_lower = {s["name"].lower() for s in skills}
        ai_matches = sum(1 for s in AI_CORE_SKILLS if any(s in sn for sn in skill_names_lower))
        if ai_matches > 5:
            reasons.append(f"non_tech_title_with_ai_skills: {title}, {ai_matches} AI skills")

    # Rule 4: Implausibly young YoE for claimed skills count
    if yoe < 1 and len(skills) > 15:
        reasons.append(f"too_many_skills_for_yoe: {len(skills)} skills, {yoe} yrs")

    # Rule 5: All skills expert (suspiciously perfect)
    if len(skills) >= 8:
        expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
        if expert_count == len(skills):
            reasons.append(f"all_skills_expert: {expert_count}/{len(skills)}")

    return (len(reasons) > 0), reasons


# ─── Individual feature functions ───────────────────────────────────────────────

def compute_skill_match_score(candidate: dict, jd_vector, vectorizer) -> float:
    """TF-IDF cosine similarity between candidate's text and JD."""
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # Build candidate text corpus (weighted)
    skill_names = " ".join(s["name"] for s in skills)
    # Weight proficiency: expert x3, advanced x2, intermediate x1
    weighted_skills = []
    for s in skills:
        prof = s.get("proficiency", "beginner")
        name = s["name"]
        multiplier = {"expert": 3, "advanced": 2, "intermediate": 1}.get(prof, 1)
        weighted_skills.extend([name] * multiplier)
    weighted_skill_text = " ".join(weighted_skills)

    career_desc = " ".join(r.get("description", "") for r in career)
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")

    candidate_text = f"{headline} {summary} {weighted_skill_text} {career_desc}"

    try:
        candidate_vec = vectorizer.transform([candidate_text])
        score = cosine_similarity(jd_vector, candidate_vec)[0][0]
        return float(score)
    except Exception:
        return 0.0


def count_hard_skill_matches(candidate: dict) -> int:
    """Count how many JD hard skills the candidate has."""
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    all_text = " ".join([
        " ".join(s["name"].lower() for s in skills),
        profile.get("summary", "").lower(),
        profile.get("headline", "").lower(),
        " ".join(r.get("description", "").lower() for r in career),
    ])

    count = 0
    for hs in HARD_SKILLS:
        if hs.lower() in all_text:
            count += 1
    return count


def compute_career_quality(candidate: dict) -> float:
    """
    Score based on:
    - % of career at non-consulting product companies
    - Years in applied ML/AI roles
    - Penalize job-hopping (avg tenure < 18 months)
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    total_months = sum(r.get("duration_months", 0) for r in career)
    if total_months == 0:
        return 0.0

    product_months = 0
    ai_months = 0
    consulting_months = 0

    for role in career:
        company_lower = role.get("company", "").lower()
        industry_lower = role.get("industry", "").lower()
        title_lower = role.get("title", "").lower()
        desc_lower = role.get("description", "").lower()
        dur = role.get("duration_months", 0)

        # Consulting check
        is_consulting = any(c in company_lower for c in CONSULTING_FIRMS)
        if is_consulting:
            consulting_months += dur
        else:
            product_months += dur

        # AI/ML role check
        ai_title_signals = [
            "machine learning", "ml engineer", "ai engineer", "data scientist",
            "nlp engineer", "research scientist", "applied scientist",
            "ranking engineer", "search engineer", "recommendation",
            "deep learning", "llm", "language model",
        ]
        if any(sig in title_lower for sig in ai_title_signals):
            ai_months += dur
        elif any(sig in desc_lower for sig in ["embedding", "vector", "nlp", "ranking", "retrieval", "llm", "transformer"]):
            ai_months += min(dur, dur // 2)  # partial credit for AI work in non-AI title

    product_ratio = product_months / total_months
    ai_ratio = min(ai_months / total_months, 1.0)

    # Penalize job-hopping (< 18 months avg tenure)
    avg_tenure = total_months / len(career)
    job_hop_penalty = 0.0
    if avg_tenure < 12:
        job_hop_penalty = 0.3
    elif avg_tenure < 18:
        job_hop_penalty = 0.15

    score = (product_ratio * 0.5) + (ai_ratio * 0.5) - job_hop_penalty
    return max(0.0, min(1.0, score))


def compute_experience_fit(candidate: dict) -> float:
    """Score based on years_of_experience. Ideal: 5–9 yrs."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.75
    elif 9 < yoe <= 12:
        return 0.75
    elif 3 <= yoe < 4:
        return 0.5
    elif 12 < yoe <= 15:
        return 0.5
    else:
        return 0.1


def compute_location_fit(candidate: dict) -> float:
    """Score based on location and relocation willingness."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0
    elif country == "india" and willing_to_relocate:
        return 0.7
    elif country == "india":
        return 0.4
    elif willing_to_relocate:
        return 0.3
    else:
        return 0.0


def compute_behavioral_score(candidate: dict) -> float:
    """
    Behavioral sub-score from redrob_signals.
    behavioral = (recruiter_response_rate × 0.35)
               + (recency_score           × 0.30)
               + (open_to_work_flag       × 0.20)
               + (interview_completion    × 0.15)
    """
    signals = candidate.get("redrob_signals", {})

    # Recruiter response rate (0-1)
    rrr = signals.get("recruiter_response_rate", 0.0)

    # Recency: days since last_active
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (REFERENCE_DATE - last_active).days
        recency_score = max(0.0, 1.0 - (days_inactive / 365))
    except Exception:
        recency_score = 0.5

    # Open to work
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    # Interview completion rate (0-1)
    icr = signals.get("interview_completion_rate", 0.5)

    behavioral = (rrr * 0.35) + (recency_score * 0.30) + (open_to_work * 0.20) + (icr * 0.15)
    return min(1.0, behavioral)


def compute_availability_score(candidate: dict) -> float:
    """Score based on notice_period_days."""
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 90)

    if notice <= 15:
        return 1.0
    elif notice <= 30:
        return 0.9
    elif notice <= 45:
        return 0.7
    elif notice <= 60:
        return 0.5
    elif notice <= 90:
        return 0.3
    else:
        return 0.1


def compute_disqualifier_penalty(candidate: dict) -> float:
    """Returns a penalty [0, 1] for disqualifying signals."""
    penalty = 0.0
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "").lower()

    # Pure consulting career
    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(c in r.get("company", "").lower() for c in CONSULTING_FIRMS)
    )
    if total_months > 0 and consulting_months / total_months > 0.8:
        penalty += 0.4

    # AI experience only very recent (<12 months)
    ai_months_recent = 0
    for role in career:
        desc = role.get("description", "").lower()
        title_role = role.get("title", "").lower()
        dur = role.get("duration_months", 0)
        if any(k in desc or k in title_role for k in ["llm", "langchain", "chatgpt", "openai", "rag"]):
            ai_months_recent += dur
    if ai_months_recent > 0 and ai_months_recent < 12:
        penalty += 0.2

    # Pure research / academic (no production deployment)
    research_signals = ["phd", "research scientist", "research engineer", "professor", "postdoc"]
    if any(sig in title for sig in research_signals):
        # Check if any production work mentioned
        prod_signals = ["deployed", "production", "serving", "api", "latency", "throughput", "scale"]
        all_desc = " ".join(r.get("description", "").lower() for r in career)
        if not any(sig in all_desc for sig in prod_signals):
            penalty += 0.25

    # No production code in last 18 months (architect mode)
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive > 547:  # ~18 months
            penalty += 0.2
    except Exception:
        pass

    # Extreme job hopper (avg tenure < 12 months)
    if career:
        avg_tenure = total_months / len(career)
        if avg_tenure < 12:
            penalty += 0.15

    return min(0.9, penalty)


def compute_github_bonus(candidate: dict) -> float:
    """Bonus for active GitHub presence."""
    signals = candidate.get("redrob_signals", {})
    score = signals.get("github_activity_score", -1)
    if score < 0:
        return 0.0
    return min(1.0, score / 100.0) * 0.5  # max 0.5 bonus contribution


def compute_assessment_score(candidate: dict) -> float:
    """Average skill assessment scores on platform."""
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.5  # neutral
    avg = sum(assessments.values()) / len(assessments)
    return min(1.0, avg / 100.0)


def compute_education_bonus(candidate: dict) -> float:
    """Bonus for tier_1 / tier_2 institutions, CS/ML fields."""
    education = candidate.get("education", [])
    bonus = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        if tier == "tier_1":
            bonus += 0.2
        elif tier == "tier_2":
            bonus += 0.1
        if any(f in field for f in ["computer science", "machine learning", "artificial intelligence", "data science"]):
            bonus += 0.1
        if degree in ["m.tech", "m.s.", "ms", "phd", "ph.d", "m.e."]:
            bonus += 0.05
    return min(0.4, bonus)


# ─── Main extraction loop ────────────────────────────────────────────────────────

def extract_features(candidates_path: str, out_path: str):
    print(f"[feature_extractor] Loading candidates from: {candidates_path}")

    # First pass: collect all candidate texts for TF-IDF fitting
    candidates = []
    texts_for_tfidf = [JD_TEXT]

    print("[feature_extractor] Reading JSONL (pass 1 — collecting texts)...")
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                candidates.append(c)
                profile = c.get("profile", {})
                skills = c.get("skills", [])
                career = c.get("career_history", [])
                text = " ".join([
                    profile.get("headline", ""),
                    profile.get("summary", ""),
                    " ".join(s["name"] for s in skills),
                    " ".join(r.get("description", "") for r in career),
                ])
                texts_for_tfidf.append(text)
            except json.JSONDecodeError:
                continue

    print(f"[feature_extractor] Loaded {len(candidates):,} candidates")

    # Fit TF-IDF on all texts
    print("[feature_extractor] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
    )
    vectorizer.fit(texts_for_tfidf)
    jd_vector = vectorizer.transform([JD_TEXT])

    # Second pass: extract features
    print("[feature_extractor] Extracting features per candidate...")
    rows = []
    for c in tqdm(candidates, desc="Extracting"):
        cid = c.get("candidate_id", "")
        hp_flag, hp_reasons = is_honeypot(c)

        row = {
            "candidate_id": cid,
            "is_honeypot": hp_flag,
            "honeypot_reasons": "|".join(hp_reasons),
            # Core scores
            "tfidf_similarity": compute_skill_match_score(c, jd_vector, vectorizer),
            "hard_skill_count": count_hard_skill_matches(c),
            "career_quality": compute_career_quality(c),
            "experience_fit": compute_experience_fit(c),
            "location_fit": compute_location_fit(c),
            "behavioral_score": compute_behavioral_score(c),
            "availability_score": compute_availability_score(c),
            "disqualifier_penalty": compute_disqualifier_penalty(c),
            "github_bonus": compute_github_bonus(c),
            "assessment_score": compute_assessment_score(c),
            "education_bonus": compute_education_bonus(c),
            # Raw signals for reasoning
            "years_of_experience": c.get("profile", {}).get("years_of_experience", 0),
            "current_title": c.get("profile", {}).get("current_title", ""),
            "current_company": c.get("profile", {}).get("current_company", ""),
            "location": c.get("profile", {}).get("location", ""),
            "country": c.get("profile", {}).get("country", ""),
            "notice_period_days": c.get("redrob_signals", {}).get("notice_period_days", 90),
            "recruiter_response_rate": c.get("redrob_signals", {}).get("recruiter_response_rate", 0),
            "open_to_work_flag": c.get("redrob_signals", {}).get("open_to_work_flag", False),
            "last_active_date": c.get("redrob_signals", {}).get("last_active_date", ""),
            "willing_to_relocate": c.get("redrob_signals", {}).get("willing_to_relocate", False),
            "github_activity_score": c.get("redrob_signals", {}).get("github_activity_score", -1),
            "interview_completion_rate": c.get("redrob_signals", {}).get("interview_completion_rate", 0),
            "profile_completeness_score": c.get("redrob_signals", {}).get("profile_completeness_score", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[feature_extractor] Saved features for {len(df):,} candidates -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from candidates.jsonl")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./features.parquet", help="Output parquet path")
    args = parser.parse_args()
    extract_features(args.candidates, args.out)
