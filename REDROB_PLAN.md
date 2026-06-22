# Redrob AI Hackathon — Track 1: Intelligent Candidate Ranking
## Master Plan

---

## 1. Problem Statement

Redrob has a pool of **100,000 synthetic candidate profiles**. A single Job Description (JD) is provided for a **Senior AI Engineer (Founding Team)** role at Redrob AI (Pune/Noida, Series A).

The task: **rank the top 100 best-fit candidates** from that pool, output as a CSV with scores and reasoning.

The challenge is NOT to find candidates who mention "RAG" or "Pinecone" in their skills list. The JD explicitly says keyword matching is the trap. The winning system reasons about:
- What the role *actually* needs vs. what the JD literally says
- Career trajectory and product company experience (not consulting farms)
- Behavioral availability signals (response rate, last active, notice period)
- Honeypot detection (impossible profiles designed to fool keyword matchers)

---

## 2. What Makes This Hard

| Challenge | Why it matters |
|---|---|
| 100K candidates, CPU only, 5-min limit | Cannot call any LLM per candidate at inference time |
| Honeypot trap (~80 profiles) | >10% honeypots in top 100 = disqualification |
| Keyword stuffers | High AI keyword density but wrong career (e.g., Marketing Manager) |
| Behavioral twins | Two identical-on-paper candidates — behavioral signals break the tie |
| Hidden ground truth | No live leaderboard; score revealed only after deadline |
| 3 submissions max | Cannot trial-and-error your way to a good score |

---

## 3. JD Decoded — What the Role Actually Needs

**Hard requirements (disqualifiers if missing):**
- 5–9 years total experience (4–5 in applied ML/AI at product companies)
- Production embeddings experience (sentence-transformers, BGE, E5, OpenAI embeddings)
- Production vector DB / hybrid search (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch)
- Strong Python, code quality
- Experience designing ranking evaluation frameworks (NDCG, MRR, MAP)

**Strong positives:**
- Shipped end-to-end ranking/search/recommendation system at scale
- Product company background (not pure TCS/Infosys/Wipro/Accenture etc.)
- Located in or willing to relocate to Pune / Noida
- Notice period ≤ 30 days (or buyable)

**Explicit disqualifiers (down-weight heavily, don't rank in top 100):**
- Pure consulting career (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
- AI experience = only recent LangChain tutorials (<12 months)
- Pure research / academic (no production deployment)
- Computer vision / speech / robotics without NLP/IR
- No production code in last 18 months ("architect" mode)
- Title-chasers (job-hop every 1.5 years)

---

## 4. Architecture — Two-Phase Pipeline

### Phase 1: Offline Pre-computation (no time limit)
Run once. Produces a feature matrix saved to disk.

```
candidates.jsonl (100K)
        ↓
  feature_extractor.py
        ↓
  features.parquet  ← saved to disk
```

**Features extracted per candidate:**

| Feature Group | Signals |
|---|---|
| Skill match | TF-IDF overlap between JD keywords and candidate skills + career descriptions |
| Career quality | % of career at product companies (not consulting), years in applied ML roles |
| Experience fit | years_of_experience in [4, 12] range; penalty outside |
| Location | Pune / Noida / Hyderabad / Mumbai / Delhi NCR = bonus; willing_to_relocate |
| Behavioral | recruiter_response_rate, days since last_active, open_to_work_flag, interview_completion_rate, notice_period_days |
| Availability | notice_period_days ≤ 30 = strong bonus |
| Honeypot flags | Impossible tenure, expert on 10+ skills with 0 months, title mismatch |

### Phase 2: Fast Ranking (must complete in <5 min on CPU)

```
features.parquet
        ↓
  ranker.py  (pure numpy/pandas math, no API calls)
        ↓
  top 100 selected
        ↓
  reasoning generator (offline LLM call via Redrob AI 70B — only 100 calls)
        ↓
  submission.csv
```

---

## 5. Scoring Formula

```
final_score = (skill_match    × 0.40)
            + (career_quality × 0.25)
            + (behavioral     × 0.20)
            + (availability   × 0.10)
            + (location_fit   × 0.05)
            - (honeypot_penalty)
            - (disqualifier_penalty)
```

**Behavioral sub-score:**
```
behavioral = (recruiter_response_rate × 0.35)
           + (recency_score           × 0.30)   # days since last_active, normalized
           + (open_to_work_flag       × 0.20)   # binary
           + (interview_completion    × 0.15)
```

**Honeypot detection (hard filter before scoring):**
- `years_of_experience` vs sum of `duration_months` across career history → flag if impossible
- Skill with `proficiency = expert` but `duration_months = 0` → flag
- Current title clearly non-technical (Marketing Manager, Accountant, HR) with AI keyword skills → flag
- Flagged candidates capped at score 0.05 (still ranked, but never in top 100)

---

## 6. Tech Stack

| Component | Tool | Why |
|---|---|---|
| Data loading | `polars` or `pandas` | Fast JSONL reading |
| Skill matching | `sklearn TfidfVectorizer` | CPU-fast, no GPU needed |
| Scoring | `numpy` | Pure math, microseconds per candidate |
| Reasoning (top 100 only) | Redrob AI 70B via API | Offline, 100 calls max |
| Sandbox demo | `Streamlit` on HuggingFace Spaces | Required by submission spec |
| Output | `.csv` (UTF-8) | Submission format |

---

## 7. Key JD Keywords for Skill Matching

```python
HARD_SKILLS = [
    "embeddings", "sentence-transformers", "BGE", "E5", "vector database",
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "FAISS", "Elasticsearch",
    "hybrid search", "retrieval", "ranking", "LLM", "fine-tuning", "LoRA",
    "NDCG", "MRR", "MAP", "A/B testing", "RAG", "NLP", "information retrieval",
    "Python", "production ML", "recommendation system"
]

CONSULTING_FIRMS = [
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant",
    "Capgemini", "HCL", "Tech Mahindra"
]

PREFERRED_LOCATIONS = [
    "Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Bengaluru"
]
```

---

## 8. Honeypot Detection Rules

```python
def is_honeypot(candidate):
    # Rule 1: Career duration impossibility
    total_months = sum(r['duration_months'] for r in candidate['career_history'])
    if total_months > (candidate['profile']['years_of_experience'] * 12) + 18:
        return True

    # Rule 2: Expert skill with zero duration
    for skill in candidate['skills']:
        if skill['proficiency'] == 'expert' and skill.get('duration_months', 1) == 0:
            return True

    # Rule 3: Non-technical title with AI keyword skills (keyword stuffer)
    non_tech_titles = ['marketing manager', 'accountant', 'hr manager',
                       'operations manager', 'customer support']
    title = candidate['profile']['current_title'].lower()
    if any(t in title for t in non_tech_titles):
        ai_skills = count_ai_skill_matches(candidate)
        if ai_skills > 5:  # keyword stuffer trap
            return True

    return False
```

---

## 9. Submission Requirements Checklist

| Item | Status |
|---|---|
| `submission.csv` — exactly 100 rows, ranks 1–100, non-increasing scores | Build |
| GitHub repo — clean code, README, single reproduce command | Build |
| `submission_metadata.yaml` — filled from template | Fill |
| Sandbox link — HuggingFace Spaces / Streamlit demo | Deploy |
| AI tools declaration — declare Redrob AI 70B used for reasoning | Declare |

**Single reproduce command (target):**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Pre-computation (allowed to exceed 5 min):
```bash
python precompute.py --candidates ./candidates.jsonl --out ./features.parquet
```

---

## 10. Evaluation Metrics (How Judges Score You)

```
Final composite = 0.50 × NDCG@10
               + 0.30 × NDCG@50
               + 0.15 × MAP
               + 0.05 × P@10
```

**Implication:** NDCG@10 is half your score. Your **top 10 candidates must be near-perfect**. Spend disproportionate effort on getting ranks 1–10 right. A great rank-1 candidate who is wrong hurts you 5× more than a wrong rank-50 candidate.

---

## 11. Build Order

```
Day 1 (Today)
  ├── Load sample_candidates.json, manually read 10 profiles
  ├── Write feature_extractor.py
  ├── Write honeypot detector
  └── Test scoring on 50-sample file

Day 2
  ├── Run precompute.py on full 100K candidates.jsonl
  ├── Write ranker.py
  ├── Inspect top 100 manually — do they make sense?
  └── Fix scoring weights based on manual review

Day 3
  ├── Generate reasoning column via Redrob AI 70B (top 100 only)
  ├── Validate with validate_submission.py
  └── First submission

Day 4–5 (buffer)
  ├── Deploy Streamlit sandbox to HuggingFace Spaces
  ├── Write GitHub README with reproduce instructions
  ├── Fill submission_metadata.yaml
  └── Second/third submission if needed
```

---

## 12. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pre-computation takes >1 hour on 100K | Medium | Use polars (faster than pandas), batch processing |
| Honeypot rate >10% in top 100 | Low if rules are explicit | Manual review of top 100 before submitting |
| Skill match too keyword-heavy | Medium | Supplement with career_history description text |
| Top 10 wrong (kills NDCG@10) | Medium | Manual audit top 10 against JD before submitting |
| Sandbox fails at Stage 3 | Low | Test locally before deploying |

---

## 13. What Winning Looks Like

A submission that wins:
1. Has ranks 1–10 that a human recruiter would look at and say "yes, obviously"
2. Has **zero** honeypots in top 100
3. Has reasoning strings that are specific to each candidate (not templated)
4. Runs reproducibly in under 5 minutes on CPU
5. Has a working sandbox and clean GitHub repo

A submission that loses:
- Ranks a Marketing Manager #1 because their skills section says "RAG, Pinecone, LLM"
- Has 12 honeypots in top 100 → disqualified
- Calls an LLM API per candidate → fails Stage 3 compute reproduction
- Identical reasoning strings → penalized at Stage 4
