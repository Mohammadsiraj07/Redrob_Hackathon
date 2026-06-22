import pandas as pd
import numpy as np
from pathlib import Path
import json

def run_tests():
    print("=" * 60)
    # 1. Check file existence
    sub_path = Path("submission.csv")
    feat_path = Path("features.parquet")
    
    print(f"Checking files...")
    assert sub_path.exists(), "submission.csv does not exist!"
    assert feat_path.exists(), "features.parquet does not exist!"
    print("✓ Files exist.")
    
    # 2. Load data
    sub_df = pd.read_csv(sub_path)
    feat_df = pd.read_parquet(feat_path)
    
    # 3. Assert schema and row counts
    print("\nAsserting submission schema & row count...")
    assert len(sub_df) == 100, f"Expected exactly 100 rows, got {len(sub_df)}"
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    assert list(sub_df.columns) == expected_cols, f"Columns mismatch. Got {list(sub_df.columns)}"
    print(f"✓ submission.csv has exactly 100 rows and correct columns: {expected_cols}")
    
    # 4. Assert rank sequencing
    print("\nAsserting rank sequence...")
    assert list(sub_df["rank"]) == list(range(1, 101)), "Ranks must be 1 to 100 sequentially!"
    print("✓ Ranks are sequentially numbered 1 to 100.")
    
    # 5. Assert monotonicity of scores
    print("\nAsserting scores are non-increasing...")
    scores = sub_df["score"].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Score ordering violation at index {i}: {scores[i]} < {scores[i+1]}"
    print("✓ Scores are monotonically non-increasing.")
    
    # 6. Assert tie-breaking rule (Lexicographical Candidate ID sorting for equal scores)
    print("\nAsserting tie-breaking sorting...")
    for score in sub_df["score"].unique():
        tied_cands = sub_df[sub_df["score"] == score]
        if len(tied_cands) > 1:
            ids = tied_cands["candidate_id"].tolist()
            sorted_ids = sorted(ids)
            assert ids == sorted_ids, f"Tie-break violation for score {score}: {ids} is not sorted ascending!"
    print("✓ Equal scores are correctly sorted by candidate_id in ascending order.")

    # 7. Check Honeypots in top 100
    print("\nChecking for honeypots in top 100...")
    honeypot_ids = set(feat_df[feat_df["is_honeypot"] == True]["candidate_id"])
    top100_ids = set(sub_df["candidate_id"])
    honeypots_in_top100 = top100_ids.intersection(honeypot_ids)
    
    assert len(honeypots_in_top100) == 0, f"Disqualification Warning! Found {len(honeypots_in_top100)} honeypots in top 100: {honeypots_in_top100}"
    print(f"✓ Zero honeypots found in top 100 (Honeypot rate: 0.0%).")
    
    # 7.5. Check for overqualification in top 10
    print("\nChecking for overqualified candidates in top 10...")
    top10_ids = sub_df.head(10)["candidate_id"].tolist()
    top10_feats = feat_df[feat_df["candidate_id"].isin(top10_ids)]
    overqualified_top10 = top10_feats[top10_feats["years_of_experience"] > 12]
    if len(overqualified_top10) > 0:
        print(f"  [WARNING] Found {len(overqualified_top10)} overqualified candidates (YoE > 12) in top 10:")
        for _, row in overqualified_top10.iterrows():
            print(f"    - {row['candidate_id']} has {row['years_of_experience']} YoE")
        assert False, "Disqualification Warning! Overqualified candidate in top 10."
    print("✓ No overqualified candidates (YoE > 12) in top 10.")
    
    # 8. Print global Honeypot detection stats
    print("\nGlobal Honeypot statistics:")
    total_cands = len(feat_df)
    total_hp = feat_df["is_honeypot"].sum()
    print(f"  Total Candidates: {total_cands:,}")
    print(f"  Total Honeypots Flagged: {total_hp:,} ({total_hp/total_cands*100:.2f}%)")
    
    # Analyze honeypot reasons
    reasons_count = {}
    for r_str in feat_df[feat_df["is_honeypot"] == True]["honeypot_reasons"]:
        for reason in r_str.split("|"):
            reason_type = reason.split(":")[0]
            reasons_count[reason_type] = reasons_count.get(reason_type, 0) + 1
    for r_type, count in reasons_count.items():
        print(f"    - {r_type}: {count:,} candidates caught")

    # 9. Audit top 10 candidates details
    print("\n" + "="*30 + " TOP 10 CANDIDATES DEEP AUDIT " + "="*30)
    top10_merged = sub_df.head(10).merge(feat_df, on="candidate_id")
    for _, c in top10_merged.iterrows():
        print(f"Rank #{int(c['rank'])}: {c['candidate_id']} | Score: {c['score']:.4f}")
        print(f"  Title: {c['current_title']} | Company: {c['current_company']} | Location: {c['location']} (Relocate: {c['willing_to_relocate']})")
        print(f"  YoE: {c['years_of_experience']:.1f} yrs | Notice Period: {int(c['notice_period_days'])} days | Response Rate: {c['recruiter_response_rate']:.1%}")
        print(f"  Hard JD-matched skills count: {int(c['hard_skill_count'])} | GitHub activity: {c['github_activity_score']}")
        print(f"  Assessment score: {c['assessment_score']:.1f}% | Career Quality Index: {c['career_quality']:.2f}")
        print(f"  Reasoning: {c['reasoning'][:120]}...")
        print("-" * 90)

    print("\n✓ ALL TESTS PASSED SUCCESSFULLY! Submission is 100% compliant.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
