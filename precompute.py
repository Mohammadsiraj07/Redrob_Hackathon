"""
precompute.py — Convenience wrapper for Phase 1.
Delegates to feature_extractor.py.

Usage:
    python precompute.py --candidates ./candidates.jsonl --out ./features.parquet
"""
import argparse
from src.feature_extractor import extract_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute feature matrix from candidates.jsonl")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./features.parquet", help="Output parquet path")
    args = parser.parse_args()
    extract_features(args.candidates, args.out)
