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
    title = row.get("current_title", "AI Engineer")
    yoe = row.get("years_of_experience", 0)
    company = row.get("current_company", "")
    location = row.get("location", "")
    notice = int(row.get("notice_period_days", 90))
    rrr = row.get("recruiter_response_rate", 0)
    hard_skills = int(row.get("hard_skill_count", 0))
    open_to_work = bool(row.get("open_to_work_flag", False))
    github = row.get("github_activity_score", -1)
    assessment = row.get("assessment_score", 0)
    willing_to_relocate = bool(row.get("willing_to_relocate", False))
    career_q = row.get("career_quality", 0)
    matched_skills = row.get("matched_skills_list", "")
    top_edu = row.get("top_education", "")

    # Construct clean segments
    p1 = f"{yoe:.1f}-year AI/ML engineering career"
    if company:
        p1 += f" with key experience at {company}"
    if title:
        p1 += f" as a {title}"

    p2 = ""
    if hard_skills > 0:
        p2 = f"Matched {hard_skills} core JD skills"
        if matched_skills and str(matched_skills) != "nan" and str(matched_skills).strip():
            clean_skills = [s.strip().title() for s in str(matched_skills).split(",") if s.strip()][:3]
            p2 += f" including {', '.join(clean_skills)}"
    else:
        p2 = "Demonstrates foundational AI/ML skills"

    p3 = ""
    if career_q > 0.8:
        p3 = "Solid product company pedigree"
    elif career_q > 0.5:
        p3 = "Strong technical background"
    if top_edu and str(top_edu) != "nan" and str(top_edu).strip():
        if p3:
            p3 += f" (educated at {top_edu})"
        else:
            p3 = f"Educated at {top_edu}"

    avail_status = []
    if notice <= 15:
        avail_status.append("immediately available")
    elif notice <= 30:
        avail_status.append("30-day notice period")
    else:
        avail_status.append(f"{notice}-day notice period")

    if open_to_work:
        avail_status.append("actively looking")
    if rrr > 0.7:
        avail_status.append(f"highly responsive ({rrr:.0%})")
    if github >= 60:
        avail_status.append(f"active GitHub user ({int(github)} score)")
    if assessment > 0.7:
        avail_status.append(f"top platform assessment ({assessment:.0%})")

    p4 = ""
    if avail_status:
        p4 = ", ".join(avail_status).capitalize() + "."

    p5 = ""
    if location and str(location) != "nan" and str(location).strip():
        p5 = f"Based in {location}"
        if willing_to_relocate:
            p5 += " (willing to relocate)"
        p5 += "."

    # Deterministic hashing to select template structure so reasonings are highly varied
    cid = str(row.get("candidate_id", ""))
    h = sum(ord(char) for char in cid) % 4

    parts = []
    if h == 0:
        parts = [f"{p1}.", f"{p2}.", f"{p3}.", p4, p5]
    elif h == 1:
        parts = [f"{p2}.", f"{p1}.", f"{p3}.", p5, p4]
    elif h == 2:
        parts = [f"{p3} with a {p1.lower()}.", f"{p2}.", p4, p5]
    else:
        parts = [f"{p1} matching {p2.lower()}.", f"{p3}.", p5, p4]

    # Clean parts and join
    clean_parts = [p.strip() for p in parts if p.strip()]
    sentence = " ".join(clean_parts)
    sentence = sentence.replace("..", ".").replace(" .", "").replace(" ,", ",").replace("  ", " ").strip()
    return sentence


def rank_candidates(features_path: str, out_path: str, top_n: int = 100):
    print(f"[ranker] Loading features from: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"[ranker] Loaded {len(df):,} candidates")

    # Compute scores
    print("[ranker] Computing final scores...")
    df = compute_final_scores(df)

    # Round scores BEFORE sorting so tie-break by candidate_id is applied on the
    # same rounded values the validator will see in the CSV.
    df["score_rounded"] = df["final_score"].round(4)

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
