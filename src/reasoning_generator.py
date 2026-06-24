"""
reasoning_generator.py -- LLM-powered reasoning for top-100 candidates.
Reads submission.csv + features.parquet + candidates.jsonl, calls an LLM API
to replace template reasoning strings with natural recruiter-quality arguments.

Run ONCE after ranking, before final submission:
    python src/reasoning_generator.py `
        --submission  ./submission.csv `
        --features    ./features.parquet `
        --candidates  "./data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" `
        --out         ./submission.csv `
        --api-key     YOUR_GROQ_API_KEY

Supported providers (set --api-base):
  Groq  (free, fast): https://api.groq.com/openai/v1   model: llama-3.1-70b-versatile
  OpenAI:             https://api.openai.com/v1          model: gpt-4o-mini
  Any OpenAI-compat:  set --api-base and --model accordingly

The script patches only the reasoning column in submission.csv.
All other columns (candidate_id, rank, score) are untouched.
Template fallback is used automatically if any API call fails.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

# --- Default configuration ---------------------------------------------------

DEFAULT_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL    = "llama-3.3-70b-versatile"
FALLBACK_MODEL   = "llama-3.1-8b-instant"
MAX_TOKENS       = 200      # ~80 words + safety margin
TEMPERATURE      = 0.4      # low for consistent, factual output
RATE_LIMIT_SLEEP = 1.5      # seconds between calls (Groq free tier: 30 RPM)
MAX_RETRIES      = 3
RETRY_SLEEP      = 5

# --- JD context (injected into every prompt) ---------------------------------

JD_CONTEXT = """
Role: Senior AI Engineer -- Founding Team at Redrob AI (Pune / Noida, Series A)
Key requirements:
- 5-9 years total, 4-5 in applied ML/AI at product companies (NOT consulting/IT services)
- Production embeddings: sentence-transformers, BGE, E5, OpenAI embeddings
- Vector DBs: Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch
- Ranking metrics: NDCG, MRR, MAP -- must have shipped ranking/search/recommendation at scale
- RAG, LLM fine-tuning (LoRA), retrieval augmented generation
- Founding team: hands-on, product-builder mindset, immediately contributing
- Preferred locations: Pune, Noida (or willing to relocate)
""".strip()


# --- Helpers -----------------------------------------------------------------

def github_label(score):
    if score < 0:
        return "N/A"
    if score >= 80:
        return f"{int(score)}/100 (highly active)"
    if score >= 50:
        return f"{int(score)}/100 (active)"
    return f"{int(score)}/100 (low)"


# --- Prompt builder ----------------------------------------------------------

def build_prompt(row: pd.Series, candidate: dict) -> str:
    profile   = candidate.get("profile", {})
    career    = candidate.get("career_history", [])
    skills    = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals   = candidate.get("redrob_signals", {})

    # Career (most recent 3 roles, 200 chars each)
    career_lines = []
    for r in career[:3]:
        desc = r.get("description", "")[:200].replace("\n", " ")
        career_lines.append(
            f"  - {r.get('title','')} at {r.get('company','')} "
            f"({r.get('duration_months',0)} months): {desc}"
        )
    career_str = "\n".join(career_lines) if career_lines else "  - No career data"

    top_skills = ", ".join(s["name"] for s in skills[:12]) or "None listed"

    edu_parts = [
        f"{e.get('degree','')} from {e.get('institution','')} ({e.get('tier','?')} tier)"
        for e in education[:2]
    ]
    edu_str = "; ".join(edu_parts) or "Not specified"

    github_score = signals.get("github_activity_score", -1)
    rrr    = signals.get("recruiter_response_rate", 0)
    notice = int(signals.get("notice_period_days", 90))
    otw    = signals.get("open_to_work_flag", False)

    prompt = f"""You are a senior technical recruiter at Redrob AI. Write a concise, direct hiring recommendation for the candidate below.

{JD_CONTEXT}

CANDIDATE:
Current role: {profile.get('current_title','')} at {profile.get('current_company','')}
Location: {profile.get('location','')} | YoE: {row.get('years_of_experience',0):.1f} years
Notice: {notice} days | Open to work: {otw} | GitHub: {github_label(github_score)} | Response rate: {rrr:.0%}

Headline: {profile.get('headline','')}
Summary: {profile.get('summary','')[:400]}

Recent career:
{career_str}

Top skills: {top_skills}
Education: {edu_str}
JD skill matches: {int(row.get('hard_skill_count', 0))} of 30 core skills
Platform assessment: {row.get('assessment_score', 0):.0%}

INSTRUCTIONS:
- Write exactly 2-3 sentences (60-90 words)
- Make a specific HIRING ARGUMENT -- reference their actual companies, projects, metrics, and named technologies
- Explain what makes their background uniquely valuable for THIS founding role
- Do NOT start with "This candidate" or "The candidate"
- Do NOT end with "Prioritizing this candidate..." or any variation
- Do NOT use the words "crucial", "prioritize", "prioritise", or "prioritizing"
- Do NOT use the phrase "skills like" -- name the skills directly
- Do NOT use bullet points or numbered lists
- Do NOT repeat the candidate ID
- End with a specific technical insight about their fit, not a generic recommendation
- Sound like an experienced recruiter who knows the tech stack, not a bot summarising a form

Write only the reasoning, nothing else:"""

    return prompt


# --- Post-processing cleanup -------------------------------------------------

BANNED_PHRASES = [
    "prioritizing this candidate",
    "prioritising this candidate",
    "this makes them a strong candidate",
    "this makes them an ideal candidate",
    "this makes them a top candidate",
    "skills like ",
]

def clean_llm_output(text: str) -> str:
    """Remove repetitive LLM closing patterns and banned phrases."""
    if not text:
        return text

    # Split into sentences and filter out any containing banned phrases
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    clean_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(b in s_lower for b in BANNED_PHRASES):
            continue
        # Also skip sentences that are just generic filler
        if s_lower.startswith("overall") or s_lower.startswith("in summary"):
            continue
        clean_sentences.append(s)

    if clean_sentences:
        result = '. '.join(clean_sentences)
        if not result.endswith('.'):
            result += '.'
        # Fix broken decimals and URLs: "7. 2" → "7.2", "Verloop. io" → "Verloop.io"
        result = re.sub(r'(\d+)\.\s+(\d)', r'\1.\2', result)
        result = re.sub(r'(\w+)\.\s+(io|ai|com|org|in|edu)\b', r'\1.\2', result)
        return result
    return text  # fallback: return original if everything got filtered


# --- API call ----------------------------------------------------------------

def call_llm(prompt: str, api_base: str, api_key: str, model: str, fallback_model: str = FALLBACK_MODEL):
    url     = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Try with primary model first
    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip accidental prefixes
            for prefix in ["Reasoning:", "Answer:", "Response:", "Recommendation:"]:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            return text
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = RETRY_SLEEP * attempt
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after) + 1.0
                    except ValueError:
                        pass
                else:
                    reset_tokens = resp.headers.get("x-ratelimit-reset-tokens")
                    if reset_tokens:
                        try:
                            wait = float(reset_tokens.rstrip('s')) + 1.0
                        except ValueError:
                            pass
                print(f"\n    [rate-limit 429] waiting {wait:.1f}s...", end="", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    [HTTP {resp.status_code}] {e}")
                break
        except Exception as e:
            print(f"\n    [error attempt {attempt}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)

    # Fallback to secondary model if primary model fails
    if fallback_model and model != fallback_model:
        print(f"\n    [fallback-model] attempting with {fallback_model}...", end="", flush=True)
        payload = {
            "model":       fallback_model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            for prefix in ["Reasoning:", "Answer:", "Response:", "Recommendation:"]:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            return text
        except Exception as e:
            print(f" failed: {e}", end="", flush=True)

    return None


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LLM reasoning for top-100 candidates")
    parser.add_argument("--submission",  required=True)
    parser.add_argument("--features",    required=True)
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--api-key",     required=True)
    parser.add_argument("--api-base",    default=DEFAULT_API_BASE)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--dry-run",     action="store_true", help="Print prompts, no API calls")
    parser.add_argument("--start-rank",  type=int, default=1,   help="Resume from rank N")
    parser.add_argument("--end-rank",    type=int, default=100, help="Stop after rank N")
    args = parser.parse_args()

    # Load submission
    print(f"[reasoning_gen] Loading {args.submission}")
    sub_df = pd.read_csv(args.submission)
    assert len(sub_df) == 100, f"Expected 100 rows, got {len(sub_df)}"

    # Load features
    print(f"[reasoning_gen] Loading {args.features}")
    feat_df = pd.read_parquet(args.features).set_index("candidate_id")

    # Load top-100 full profiles
    print(f"[reasoning_gen] Loading candidate profiles...")
    top_ids = set(sub_df["candidate_id"].tolist())
    candidate_map = {}
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in top_ids:
                candidate_map[c["candidate_id"]] = c
            if len(candidate_map) == len(top_ids):
                break
    print(f"[reasoning_gen] Found {len(candidate_map)}/100 profiles")

    if not args.dry_run:
        print(f"\n[reasoning_gen] Using API: {args.api_base} | Model: {args.model}")
        print(f"[reasoning_gen] Processing ranks {args.start_rank}-{args.end_rank}\n")

    updated = 0
    failed  = 0

    for i, row in sub_df.iterrows():
        rank = int(row["rank"])
        if rank < args.start_rank or rank > args.end_rank:
            continue

        cid       = row["candidate_id"]
        candidate = candidate_map.get(cid, {})
        feat_row  = feat_df.loc[cid] if cid in feat_df.index else pd.Series()
        merged    = pd.concat([row, feat_row])
        prompt    = build_prompt(merged, candidate)

        print(f"  Rank #{rank:>3} | {cid} | score={row['score']:.4f} ...", end=" ", flush=True)

        if args.dry_run:
            print("DRY-RUN")
            print(f"\n{'='*70}\n{prompt}\n{'='*70}\n")
            continue

        llm_text = call_llm(prompt, args.api_base, args.api_key, args.model)

        if llm_text and len(llm_text.split()) >= 20:
            llm_text = clean_llm_output(llm_text)
            sub_df.at[i, "reasoning"] = llm_text
            updated += 1
            print(f"OK ({len(llm_text.split())} words)")
        else:
            failed += 1
            print("FALLBACK (template kept)")

        time.sleep(RATE_LIMIT_SLEEP)

    if not args.dry_run:
        # Validate schema before saving
        assert len(sub_df) == 100
        assert list(sub_df.columns) == ["candidate_id", "rank", "score", "reasoning"]

        sub_df.to_csv(args.out, index=False)
        print(f"\n[reasoning_gen] Done: {updated} updated, {failed} fallbacks")
        print(f"[reasoning_gen] Saved -> {args.out}")

        print("\n--- TOP 5 PREVIEW ---")
        for _, r in sub_df.head(5).iterrows():
            print(f"\nRank #{int(r['rank'])} | {r['candidate_id']} | score={r['score']:.4f}")
            print(f"  {r['reasoning']}")


if __name__ == "__main__":
    main()
