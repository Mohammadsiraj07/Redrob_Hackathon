"""
ranker.py — Phase 2: Fast Ranking (< 5 min on CPU)
Reads features.parquet, computes final scores, outputs submission.csv

Usage:
    python src/ranker.py --features ./features.parquet --out ./submission.csv
"""

import argparse
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Scoring weights (from REDROB_PLAN.md) ──────────────────────────────────────

WEIGHTS = {
    "skill_match":    0.40,
    "career_quality": 0.25,
    "behavioral":     0.20,
    "availability":   0.10,
    "location_fit":   0.05,
}

HONEYPOT_CAP = 0.05  # Honeypots never score above this

# ─── Skill match composite ───────────────────────────────────────────────────────

def compute_skill_match_composite(df: pd.DataFrame) -> pd.Series:
    """
    Blend TF-IDF cosine similarity + hard skill count (normalized).
    TF-IDF: 0.6 weight, hard skill count: 0.4 weight.
    """
    tfidf = df["tfidf_similarity"].clip(0, 1)

    # Normalize hard skill count (max observed in dataset)
    max_hard = df["hard_skill_count"].max()
    if max_hard == 0:
        max_hard = 1
    hard_norm = (df["hard_skill_count"] / max_hard).clip(0, 1)

    # Assessment score bonus
    assessment = df["assessment_score"].clip(0, 1)

    # GitHub bonus (normalized 0-0.5)
    github_norm = df["github_bonus"].clip(0, 0.5) * 2  # scale to 0-1

    skill_match = (
        tfidf       * 0.45
        + hard_norm * 0.35
        + assessment * 0.10
        + github_norm * 0.10
    ).clip(0, 1)

    return skill_match


def compute_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute final composite scores for all candidates."""
    df = df.copy()

    # Build skill_match composite
    df["skill_match"] = compute_skill_match_composite(df)

    # Final score formula
    df["raw_score"] = (
        df["skill_match"]    * WEIGHTS["skill_match"]
        + df["career_quality"] * WEIGHTS["career_quality"]
        + df["behavioral_score"] * WEIGHTS["behavioral"]
        + df["availability_score"] * WEIGHTS["availability"]
        + df["location_fit"]  * WEIGHTS["location_fit"]
        - df["disqualifier_penalty"] * 0.3  # penalty applied proportionally
        + df["education_bonus"] * 0.05  # small education bonus
    ).clip(0, 1)

    # Apply honeypot cap
    df["final_score"] = df["raw_score"]
    honeypot_mask = df["is_honeypot"] == True
    df.loc[honeypot_mask, "final_score"] = df.loc[honeypot_mask, "final_score"].clip(0, HONEYPOT_CAP)

    return df


def generate_reasoning(row: pd.Series) -> str:
    """Generate a highly natural, candidate-specific reasoning string."""
    title = str(row.get("current_title", "AI Engineer")).strip()
    yoe = row.get("years_of_experience", 0)
    company = str(row.get("current_company", "")).strip()
    location = str(row.get("location", "")).strip()
    notice = int(row.get("notice_period_days", 90))
    rrr = row.get("recruiter_response_rate", 0)
    hard_skills = int(row.get("hard_skill_count", 0))
    open_to_work = bool(row.get("open_to_work_flag", False))
    github = row.get("github_activity_score", -1)
    assessment = row.get("assessment_score", 0)
    willing_to_relocate = bool(row.get("willing_to_relocate", False))
    career_q = row.get("career_quality", 0)
    matched_skills = str(row.get("matched_skills_list", "")).strip()
    top_edu = str(row.get("top_education", "")).strip()

    # Clean strings
    if not title or title == "nan" or title == "AI Engineer":
        title_phrase = "as an AI/ML Engineer"
    else:
        title_phrase = f"as a {title}"

    if not company or company == "nan":
        company_phrase = ""
    else:
        company_phrase = f"at {company}"

    # Proper display names for ML acronyms (override .title() behaviour)
    SKILL_DISPLAY_NAMES = {
        "bge": "BGE", "e5": "E5", "rag": "RAG", "nlp": "NLP", "llm": "LLM",
        "faiss": "FAISS", "ndcg": "NDCG", "mrr": "MRR", "map": "MAP",
        "lora": "LoRA", "bert": "BERT", "gpt": "GPT", "openai": "OpenAI",
        "sentence-transformers": "Sentence-Transformers",
        "sentence transformers": "Sentence-Transformers",
        "huggingface": "HuggingFace", "a/b testing": "A/B Testing",
        "ab testing": "A/B Testing", "vector database": "Vector Database",
        "vector db": "Vector DB", "hybrid search": "Hybrid Search",
        "semantic search": "Semantic Search", "retrieval": "Retrieval",
        "ranking": "Ranking", "reranking": "Reranking",
        "fine-tuning": "Fine-Tuning", "fine tuning": "Fine-Tuning",
        "finetuning": "Fine-Tuning", "information retrieval": "Information Retrieval",
        "recommendation system": "Recommendation System",
        "natural language processing": "Natural Language Processing",
        "retrieval augmented generation": "Retrieval Augmented Generation",
        "large language model": "Large Language Model", "production ml": "Production ML",
        "bi-encoder": "Bi-Encoder", "cross-encoder": "Cross-Encoder",
    }

    # Skills phrase
    if matched_skills and matched_skills != "nan":
        raw_skills = [s.strip() for s in matched_skills.split(",") if s.strip()][:3]
        clean_skills = [SKILL_DISPLAY_NAMES.get(s.lower(), s.title()) for s in raw_skills]
        if len(clean_skills) > 1:
            skills_list = ', '.join(clean_skills[:-1]) + ' and ' + clean_skills[-1]
            skills_phrase = f"including {skills_list}"
        elif len(clean_skills) == 1:
            skills_list = clean_skills[0]
            skills_phrase = f"including {skills_list}"
        else:
            skills_list = "core AI concepts"
            skills_phrase = "across core AI concepts"
    else:
        skills_list = "foundational ML and Python systems"
        skills_phrase = "across foundational ML and Python systems"

    # Pedigree phrase
    if career_q > 0.8 and top_edu and top_edu != "nan":
        pedigree_phrase = f"a solid product company pedigree with studies at {top_edu}"
    elif career_q > 0.8:
        pedigree_phrase = "a strong product-oriented engineering pedigree"
    elif top_edu and top_edu != "nan":
        pedigree_phrase = f"an educational background from {top_edu}"
    else:
        pedigree_phrase = "a solid technical foundation"

    # Availability phrase
    if notice <= 15:
        if open_to_work:
            avail_phrase = "immediately available and actively seeking roles"
        else:
            avail_phrase = "immediately available for new opportunities"
    elif notice <= 30:
        if open_to_work:
            avail_phrase = "actively looking with a 30-day notice period"
        else:
            avail_phrase = "on a short 30-day notice period"
    else:
        if open_to_work:
            avail_phrase = f"actively looking with a {notice}-day notice period"
        else:
            avail_phrase = f"on a {notice}-day notice period"

    # Active phrase
    active_details = []
    if github >= 60:
        active_details.append(f"highly active on GitHub ({int(github)} score)")
    if rrr > 0.7:
        active_details.append(f"highly responsive to outreach ({rrr:.0%})")
    if assessment > 0.7:
        active_details.append(f"scored top tier on technical assessments ({assessment:.0%})")
    
    if active_details:
        if len(active_details) > 1:
            active_phrase = ", ".join(active_details[:-1]) + ", and " + active_details[-1]
        else:
            active_phrase = active_details[0]
    else:
        active_phrase = ""

    # Location phrase
    is_local = "noida" in location.lower() or "pune" in location.lower()
    if is_local:
        location_phrase = f"based locally in {location}"
    elif willing_to_relocate:
        location_phrase = f"based in {location} but willing to relocate"
    elif location:
        location_phrase = f"based in {location}"
    else:
        location_phrase = ""

    # Hashing for template selection
    cid = str(row.get("candidate_id", ""))
    h = sum(ord(char) for char in cid) % 4

    if h == 0:
        # Style 0: Strong recommendation focus
        sentence = f"Strong hiring recommendation for our founding team. Candidate brings {yoe:.1f} years of applied ML experience, highlighted by their role {company_phrase} {title_phrase}. Matches {hard_skills} core JD skills ({skills_phrase}), backed by {pedigree_phrase}."
        if location_phrase:
            sentence += f" They are {location_phrase}."
        if avail_phrase:
            sentence += f" Currently {avail_phrase}."
        if active_phrase:
            sentence += f" Additionally, they are {active_phrase}."
    elif h == 1:
        # Style 1: Technical alignment focus
        sentence = f"Highly aligned on our technical requirements, matching {hard_skills} core skills ({skills_phrase}). Spent {yoe:.1f} years {title_phrase} {company_phrase} and displays {pedigree_phrase}."
        if avail_phrase:
            sentence += f" They are currently {avail_phrase}."
        if location_phrase:
            sentence += f" Location-wise, they are {location_phrase}."
        if active_phrase:
            sentence += f" On the platform, they are {active_phrase}."
    elif h == 2:
        # Style 2: Career progression focus
        sentence = f"An impressive {yoe:.1f}-year engineering career, specializing in AI/ML {title_phrase} {company_phrase}. Demonstrates {pedigree_phrase} and matches {hard_skills} JD skills ({skills_phrase})."
        if location_phrase:
            sentence += f" They are {location_phrase}."
        if avail_phrase:
            sentence += f" Currently {avail_phrase}."
        if active_phrase:
            sentence += f" Recruiter signal: candidate is {active_phrase}."
    else:
        # Style 3: Founding team focus
        sentence = f"Excellent founding-engineer profile with {yoe:.1f} years in applied ML, recently working {company_phrase} {title_phrase}. Well-matched on {hard_skills} core JD skills, particularly {skills_list}, combined with {pedigree_phrase}."
        if avail_phrase:
            sentence += f" They are {avail_phrase}."
        if location_phrase:
            sentence += f" Currently {location_phrase}."
        if active_phrase:
            sentence += f" Recruiter signals show they are {active_phrase}."

    # Clean double spaces/periods
    sentence = sentence.replace("..", ".").replace(" .", ".").replace(" ,", ",").replace("  ", " ").strip()
    return sentence


def rank_candidates(features_path: str, out_path: str, top_n: int = 100):
    print(f"[ranker] Loading features from: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"[ranker] Loaded {len(df):,} candidates")

    # Compute scores
    print("[ranker] Computing final scores...")
    df = compute_final_scores(df)

    # Add a micro-tiebreaker that uses additional signals to eliminate ties
    df["tiebreaker"] = (
        df["github_activity_score"].clip(0, 100) / 100 * 0.0005
        + df["profile_completeness_score"].clip(0, 100) / 100 * 0.0003
        + df["recruiter_response_rate"] * 0.0002
    )
    # Round scores BEFORE sorting. We round to 6 decimal places to eliminate ties.
    df["score_rounded"] = (df["final_score"] + df["tiebreaker"]).clip(0, 1).round(6)

    # Sort: score desc, then candidate_id asc (validator tie-break rule)
    df_sorted = df.sort_values(
        by=["score_rounded", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Print stats on honeypots found
    total_honeypots = df["is_honeypot"].sum()
    print(f"[ranker] Honeypots detected: {total_honeypots:,} / {len(df):,} ({total_honeypots/len(df)*100:.1f}%)")

    # Take top 100
    top100 = df_sorted.head(top_n).copy()

    # Verify no honeypots in top 100
    hp_in_top = top100["is_honeypot"].sum()
    print(f"[ranker] Honeypots in top {top_n}: {hp_in_top}")
    if hp_in_top > 0:
        print(f"  [WARNING] {hp_in_top} honeypots in top {top_n}! Review manually.")
        print(top100[top100["is_honeypot"] == True][["candidate_id", "current_title", "score_rounded", "honeypot_reasons"]])

    # Generate reasoning column
    print("[ranker] Generating reasoning strings...")
    top100["reasoning"] = top100.apply(generate_reasoning, axis=1)

    # Assign ranks 1-100
    top100["rank"] = range(1, top_n + 1)

    # Use pre-rounded score column
    top100["score"] = top100["score_rounded"]

    # Select output columns
    submission = top100[["candidate_id", "rank", "score", "reasoning"]].copy()

    # Save
    submission.to_csv(out_path, index=False)
    print(f"[ranker] Saved {len(submission)} ranked candidates -> {out_path}")

    # Print top 10 for manual review
    print("\n--- TOP 10 CANDIDATES ---------------------------------------------------")
    for _, row in top100.head(10).iterrows():
        print(f"  #{int(row['rank']):>2} | {row['candidate_id']} | {row['score']:.4f} | {row['current_title']} | {row['reasoning'][:80]}...")
    print("------------------------------------------------------------------------\n")

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates from features.parquet")
    parser.add_argument("--features", default="./features.parquet", help="Input parquet path")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of top candidates to output")
    args = parser.parse_args()
    rank_candidates(args.features, args.out, args.top)
