# Redrob AI Hackathon — Track 1: Data & AI Challenge
## Intelligent Candidate Ranking System (Senior AI Engineer - Founding Team)
*Author: Mohammad Siraj (ML Engineer / Individual Participant)*

---

## Slide 1: Title & Overview
### AI-Powered Candidate Ranking: Thinking Like an Expert Recruiter
* **The Goal**: Rank the top 100 candidates from 100,000 synthetic profiles for the **Senior AI Engineer (Founding Team)** role at Redrob AI (Pune/Noida, Series A).
* **The Paradigm**: Move beyond keyword matching to multi-dimensional candidate evaluation.
* **Core Philosophy**: A great recruiter values career progression, company pedigree, behavioral signs, and active engagement over skill list density.
* **Key Achievement**: 100% compliant ranking pipeline executing in **< 1 second on CPU** during inference, with **0% Honeypot rate** in the shortlist.

---

## Slide 2: The Trap of Keyword Matching & Honeypots
### Why Naive Searching Fails
* **The Trap**: Synthetic profiles contain "keyword stuffers" (e.g. Marketing Managers with "RAG, Milvus, Pinecone, LLMs" in their skills list) and intentional honeypots designed to fool keyword searchers.
* **The Consequence**: Naive TF-IDF or vector search of skills puts non-tech profiles or fake candidates in the top 10.
* **Our Honeypot Shield (5-Rule Filter)**:
  1. *Tenure Check*: Flags duration-experience impossibility (Total tenure > YoE * 12 + 18 months).
  2. *Expert Zero-Duration*: Flags expert proficiency claimed for 0 months duration.
  3. *Title-Skill Discrepancy*: Flags non-technical current titles (HR, Marketing, CS) claiming >5 core AI skills.
  4. *Too-Many-Skills Mismatch*: Flags very low experience (<1 YoE) claiming >15 skills.
  5. *Suspicious Perfection*: Flags profiles claiming "Expert" for every single skill (8+ skills).
* **Result**: Caps honeypot scores at `0.05`, automatically eliminating them from the top 100.

---

## Slide 3: Two-Phase Ranking Architecture
### Scalable & Compliant Design
* **Phase 1: Feature Extraction (Offline, No Time Limit)**
  * Reads full `candidates.jsonl` (487MB, 100K profiles).
  * Computes TF-IDF cosine similarity, hard skill matches, career quality index, availability, location, and behavioral metrics.
  * Outputs a structured `features.parquet` file (4MB) to disk.
  * Prevents high-memory overhead and runs in ~4 minutes.
* **Phase 2: Ultra-Fast Ranking (At Inference, <5 min CPU Limit)**
  * Loads `features.parquet` instantly.
  * Applies scoring weights, honeypot caps, and tie-breaking sorting.
  * Outputs final `submission.csv` in **0.2 seconds**.
  * Zero network dependencies, zero LLM API call latency.

---

## Slide 4: Multi-Signal Scoring System
### The Scoring Formula

$$Score = (SkillMatch \times 0.40) + (CareerQuality \times 0.25) + (Behavioral \times 0.20) + (Availability \times 0.10) + (Location \times 0.05) - (Disqualifiers) + (EducationBonus)$$

1. **Skill Match Composite (40%)**:
   * Blends global TF-IDF similarity (45%), hard-skill keyword matches (35%), online assessments (10%), and GitHub contributions (10%).
   * Weight multiplier applied to skill proficiencies (Expert = 3x, Advanced = 2x, Intermediate = 1x).
2. **Career Quality Index (25%)**:
   * Computes the ratio of career history spent in product-based companies vs. outsourcing/consulting firms.
   * Measures time spent in actual applied ML/AI roles and penalizes job-hopping (average tenure < 18 months).
3. **Behavioral Engagement (20%)**:
   * Evaluates recruiter response rate (35%), recency of platform activity (30%), open-to-work status (20%), and interview completion rate (15%).
4. **Availability & Location (15%)**:
   * Notice period score: immediate/15d availability receives a premium over 90d notice.
   * Location Fit: Candidates already in Pune/Noida or willing to relocate within India are prioritized.

---

## Slide 5: The Validation & Quality Audit
### Getting the Top 10 Exactly Right
* **Judges Metric**: $0.50 \times NDCG@10 + 0.30 \times NDCG@50 + 0.15 \times MAP + 0.05 \times P@10$.
* **NDCG@10 represents 50% of the grade**, making the top 10 ranks critical.
* **Tie-Breaker Rule**: Handled by pre-rounding scores to 4 decimal places and sorting lexicographically by `candidate_id` ascending for equivalent scores.
* **Validator Results**: Full pipeline validation passed successfully.
* **Top Rank Profile Breakdown**:
  * **Rank 1 (CAND_0018499)**: Senior ML Engineer with 7.2 years of experience at Zomato (Noida). Active on GitHub (94 activity score), immediate availability, active looking, and 20 JD-matched skills.
  * **Rank 2 (CAND_0042506)**: Search Engineer with 4.2 years of experience at verloop.io. Matched 20 core JD skills, immediately available, actively looking, active GitHub user (64 score), top platform assessment (74%). Based in Mumbai, Maharashtra.

---

## Slide 6: Interactive Recruiter Sandbox
### Real-Time Tweakability & Dashboard
* **Streamlit App**: Pastes custom JDs, dynamically processes candidate lists, and recalculates matches in real-time.
* **Metrics Dashboard**: Displays total candidates, honeypots flagged, average scores, and distribution of scores.
* **Deep-Dive Tab Panels**:
  * *Career History Timeline*: Shows descriptions, tenures, and titles for any selected candidate.
  * *Skills Radar*: Color-coded badges highlighting expert, advanced, and intermediate skills.
  * *Behavioral Panel*: Real-time progress bars for assessment scores, recruiter response rate, and activity indicators.
* **HuggingFace Ready**: Designed to run cleanly on HuggingFace Spaces.

---

## Slide 7: Why This Solution Wins
### Engineering Highlights
1. **Zero LLM API Latency**: Performs complex evaluation locally using Scikit-Learn and Pandas. Ready for strict offline compute environments.
2. **Robust Honeypot Detection**: 5 layered rules prevent adversarial candidate gaming.
3. **Unique Reasoning**: Generates highly granular, descriptive summary strings for every single shortlisted candidate rather than standard templates.
4. **Clean Codebase**: Adheres to modern software design patterns with standalone wrappers, unit checks, and documentation.
