"""
rank.py — Single reproduce command entry point.
Runs the full pipeline: feature extraction (if needed) then ranking.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    
    (Optional: skip pre-computation if features.parquet already exists)
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --features ./features.parquet
"""
import argparse
import time
from pathlib import Path
from src.feature_extractor import extract_features
from src.ranker import rank_candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end candidate ranking pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--features", default="./features.parquet", help="Features parquet path (will be created if missing)")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip feature extraction if features.parquet exists")
    args = parser.parse_args()

    features_path = Path(args.features)

    # Phase 1: Feature Extraction
    if not (args.skip_precompute and features_path.exists()):
        print("=" * 60)
        print("PHASE 1: Feature Extraction")
        print("=" * 60)
        t0 = time.time()
        extract_features(args.candidates, str(features_path))
        print(f"Phase 1 complete in {time.time() - t0:.1f}s\n")
    else:
        print(f"[rank.py] Skipping pre-computation, using existing {features_path}")

    # Phase 2: Ranking (must complete < 5 min)
    print("=" * 60)
    print("PHASE 2: Ranking")
    print("=" * 60)
    t1 = time.time()
    rank_candidates(str(features_path), args.out)
    elapsed = time.time() - t1
    print(f"Phase 2 complete in {elapsed:.1f}s")
    if elapsed > 300:
        print(f"[WARNING] Phase 2 took {elapsed:.1f}s > 5 min limit!")
    else:
        print(f"[OK] Phase 2 within 5-min limit ({elapsed:.1f}s)")

    print(f"\n[OK] Done. Submission saved to: {args.out}")
