import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import src.feature_extractor as fe
import src.ranker as rk

# Set page config
st.set_page_config(
    page_title="Redrob AI - Candidate Ranker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark-mode accent style */
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #F3F4F6;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .badge-primary {
        background-color: #3B82F6;
        color: white;
    }
    .badge-secondary {
        background-color: #6B7280;
        color: white;
    }
    .badge-success {
        background-color: #10B981;
        color: white;
    }
    .badge-warning {
        background-color: #F59E0B;
        color: white;
    }
    .badge-danger {
        background-color: #EF4444;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Intro
st.markdown('<div class="main-header">Redrob AI Recruiter Sandbox</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered candidate ranking system matching talents like an expert recruiter. Fits the Senior AI Engineer (Founding Team) role.</div>', unsafe_allow_html=True)

# ─── Sidebar Configuration ──────────────────────────────────────────────────
st.sidebar.image("https://raw.githubusercontent.com/redrob-co/assets/main/logo.png", width=150)
st.sidebar.title("Configuration")

# 1. Job Description text area
st.sidebar.subheader("Job Description")
jd_text = st.sidebar.text_area(
    "Edit Job Description",
    value=fe.JD_TEXT.strip(),
    height=200
)

# 2. Weights sliders
st.sidebar.subheader("Scoring Weights")
w_skill = st.sidebar.slider("Skill Match (Cosine + Keywords)", 0.0, 1.0, rk.WEIGHTS["skill_match"], 0.05)
w_career = st.sidebar.slider("Career Quality (Product vs Consulting)", 0.0, 1.0, rk.WEIGHTS["career_quality"], 0.05)
w_behavioral = st.sidebar.slider("Behavioral (Active + Response Rate)", 0.0, 1.0, rk.WEIGHTS["behavioral"], 0.05)
w_avail = st.sidebar.slider("Availability (Notice Period)", 0.0, 1.0, rk.WEIGHTS["availability"], 0.05)
w_loc = st.sidebar.slider("Location Fit (Pune/Noida/Relocate)", 0.0, 1.0, rk.WEIGHTS["location_fit"], 0.05)

# Normalize weights if they don't sum to 1
total_w = w_skill + w_career + w_behavioral + w_avail + w_loc
if total_w > 0:
    weights = {
        "skill_match": w_skill / total_w,
        "career_quality": w_career / total_w,
        "behavioral": w_behavioral / total_w,
        "availability": w_avail / total_w,
        "location_fit": w_loc / total_w
    }
else:
    weights = rk.WEIGHTS

if total_w != 1.0 and total_w > 0:
    st.sidebar.warning(f"Weights normalized to sum to 1.0 (original sum: {total_w:.2f})")

# 3. Honeypot config
st.sidebar.subheader("Honeypot Filter")
enable_honeypots = st.sidebar.checkbox("Cap Honeypot Scores", value=True)
honeypot_cap = st.sidebar.number_input("Honeypot Cap Score", min_value=0.0, max_value=1.0, value=rk.HONEYPOT_CAP, step=0.01)

# Helper to process uploaded file or local file
def process_data(file_content, jd_input_text):
    # Parse inputs (can be list of dicts from json or jsonl)
    candidates = []
    
    # Try parsing as JSON array
    try:
        candidates = json.loads(file_content)
    except json.JSONDecodeError:
        # Try JSON Lines
        candidates = []
        for line in file_content.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not candidates:
        return None

    # TF-IDF Setup
    texts_for_tfidf = [jd_input_text]
    for c in candidates:
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

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=1
    )
    vectorizer.fit(texts_for_tfidf)
    jd_vector = vectorizer.transform([jd_input_text])

    # Extract features
    rows = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        hp_flag, hp_reasons = fe.is_honeypot(c)
        
        # Skill text matching
        skills = c.get("skills", [])
        weighted_skills = []
        for s in skills:
            prof = s.get("proficiency", "beginner")
            name = s["name"]
            multiplier = {"expert": 3, "advanced": 2, "intermediate": 1}.get(prof, 1)
            weighted_skills.extend([name] * multiplier)
        weighted_skill_text = " ".join(weighted_skills)
        profile = c.get("profile", {})
        career = c.get("career_history", [])
        candidate_text = f"{profile.get('headline', '')} {profile.get('summary', '')} {weighted_skill_text} {' '.join(r.get('description', '') for r in career)}"
        
        try:
            cand_vec = vectorizer.transform([candidate_text])
            tfidf_sim = float(cosine_similarity(jd_vector, cand_vec)[0][0])
        except Exception:
            tfidf_sim = 0.0

        row = {
            "candidate_id": cid,
            "is_honeypot": hp_flag,
            "honeypot_reasons": "|".join(hp_reasons),
            "tfidf_similarity": tfidf_sim,
            "hard_skill_count": fe.count_hard_skill_matches(c),
            "career_quality": fe.compute_career_quality(c),
            "experience_fit": fe.compute_experience_fit(c),
            "location_fit": fe.compute_location_fit(c),
            "behavioral_score": fe.compute_behavioral_score(c),
            "availability_score": fe.compute_availability_score(c),
            "disqualifier_penalty": fe.compute_disqualifier_penalty(c),
            "github_bonus": fe.compute_github_bonus(c),
            "assessment_score": fe.compute_assessment_score(c),
            "education_bonus": fe.compute_education_bonus(c),
            # Raw fields for profile display
            "years_of_experience": profile.get("years_of_experience", 0),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "notice_period_days": c.get("redrob_signals", {}).get("notice_period_days", 90),
            "recruiter_response_rate": c.get("redrob_signals", {}).get("recruiter_response_rate", 0),
            "open_to_work_flag": c.get("redrob_signals", {}).get("open_to_work_flag", False),
            "last_active_date": c.get("redrob_signals", {}).get("last_active_date", ""),
            "willing_to_relocate": c.get("redrob_signals", {}).get("willing_to_relocate", False),
            "github_activity_score": c.get("redrob_signals", {}).get("github_activity_score", -1),
            "interview_completion_rate": c.get("redrob_signals", {}).get("interview_completion_rate", 0),
            "candidate_raw": c  # Store raw candidate dict for profiling
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute scores
    max_hard = df["hard_skill_count"].max()
    if max_hard == 0:
        max_hard = 1
    hard_norm = (df["hard_skill_count"] / max_hard).clip(0, 1)

    df["skill_match"] = (
        df["tfidf_similarity"] * 0.45
        + hard_norm * 0.35
        + df["assessment_score"].clip(0, 1) * 0.10
        + df["github_bonus"].clip(0, 0.5) * 2 * 0.10
    ).clip(0, 1)

    df["raw_score"] = (
        df["skill_match"] * weights["skill_match"]
        + df["career_quality"] * weights["career_quality"]
        + df["behavioral_score"] * weights["behavioral"]
        + df["availability_score"] * weights["availability"]
        + df["location_fit"] * weights["location_fit"]
        - df["disqualifier_penalty"] * 0.3
        + df["education_bonus"] * 0.05
    ).clip(0, 1)

    df["final_score"] = df["raw_score"]
    if enable_honeypots:
        hp_mask = df["is_honeypot"] == True
        df.loc[hp_mask, "final_score"] = df.loc[hp_mask, "final_score"].clip(0, honeypot_cap)

    # Round and sort
    df["score_rounded"] = df["final_score"].round(4)
    df_sorted = df.sort_values(
        by=["score_rounded", "candidate_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # Reasoning
    df_sorted["reasoning"] = df_sorted.apply(rk.generate_reasoning, axis=1)
    df_sorted["rank"] = range(1, len(df_sorted) + 1)
    
    return df_sorted


# ─── File Upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload candidates file (JSON or JSONL)", type=["json", "jsonl"])

candidates_df = None

# If no file uploaded, allow using the sample
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    with st.spinner("Processing candidates..."):
        candidates_df = process_data(content, jd_text)
else:
    st.info("Upload a candidate profile file or click the button below to load the 50 sample candidates.")
    if st.button("Load 50 Sample Candidates"):
        sample_path = Path("sample_candidates.jsonl")
        if not sample_path.exists():
            sample_path = Path("data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json")
        
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
            with st.spinner("Processing sample candidates..."):
                candidates_df = process_data(content, jd_text)
        else:
            st.error(f"Sample candidates file not found at {sample_path}")

# ─── Dashboard Render ───────────────────────────────────────────────────────
if candidates_df is not None:
    st.success(f"Successfully processed {len(candidates_df)} candidates!")

    # 1. Metric Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Total Evaluated</div>
            <div class="metric-value">{len(candidates_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        num_hp = candidates_df["is_honeypot"].sum()
        pct_hp = (num_hp / len(candidates_df)) * 100
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Honeypots Flagged</div>
            <div class="metric-value">{num_hp} <span style="font-size:1.1rem; color:#EF4444;">({pct_hp:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_score = candidates_df["score_rounded"].mean()
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value">{avg_score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        top_score = candidates_df["score_rounded"].max()
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Top Candidate Score</div>
            <div class="metric-value">{top_score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Layout: Table on left, Details on right
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Rankings & Shortlist")
        
        # Filter options
        filters_col1, filters_col2 = st.columns(2)
        with filters_col1:
            loc_search = st.text_input("Filter by Location keyword", "")
        with filters_col2:
            exclude_hp_display = st.checkbox("Hide honeypots in list", value=False)
            
        display_df = candidates_df.copy()
        if loc_search:
            display_df = display_df[display_df["location"].str.contains(loc_search, case=False, na=False)]
        if exclude_hp_display:
            display_df = display_df[display_df["is_honeypot"] == False]

        # Selectable table
        st.write(f"Showing {len(display_df)} candidates after filters:")
        
        grid_data = display_df[["rank", "candidate_id", "score_rounded", "current_title", "current_company", "location", "is_honeypot"]].copy()
        grid_data.columns = ["Rank", "Candidate ID", "Score", "Title", "Company", "Location", "Honeypot?"]
        
        # Display as a dataframe with highlighting
        st.dataframe(
            grid_data,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.4f"),
                "Honeypot?": st.column_config.CheckboxColumn()
            },
            hide_index=True
        )

        # Dropdown selection for deep dive
        st.subheader("Select Candidate for Deep-Dive Analysis")
        selected_cid = st.selectbox(
            "Choose Candidate ID",
            options=display_df["candidate_id"].tolist(),
            format_func=lambda x: f"Rank {display_df[display_df['candidate_id']==x]['rank'].values[0]} | {x} | {display_df[display_df['candidate_id']==x]['current_title'].values[0]} ({display_df[display_df['candidate_id']==x]['score_rounded'].values[0]})"
        )

    with right_col:
        if selected_cid:
            c_row = candidates_df[candidates_df["candidate_id"] == selected_cid].iloc[0]
            raw_c = c_row["candidate_raw"]
            
            st.subheader(f"Profile Analyzer: {selected_cid}")
            
            # Anonymized Identity Card
            name = raw_c.get("profile", {}).get("anonymized_name", "Anonymized Candidate")
            title = c_row["current_title"]
            company = c_row["current_company"]
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #8B5CF6;">
                <h3 style="margin-top:0;">{name}</h3>
                <strong>{title}</strong> at {company or 'Unknown Company'}<br/>
                <span style="color:#9CA3AF;">{c_row['location']}, {c_row['country']} | {c_row['years_of_experience']:.1f} YoE</span>
            </div>
            """, unsafe_allow_html=True)

            # Score details breakdown
            st.write("#### Score Component Analysis")
            
            # Form composite scores
            skills_pct = float(c_row["skill_match"])
            career_pct = float(c_row["career_quality"])
            behavioral_pct = float(c_row["behavioral_score"])
            availability_pct = float(c_row["availability_score"])
            location_pct = float(c_row["location_fit"])
            disqual_val = float(c_row["disqualifier_penalty"])
            edu_val = float(c_row["education_bonus"])
            
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.metric("Final Score", f"{c_row['score_rounded']:.4f}")
                st.write(f"**Skill Match:** {skills_pct:.1%}")
                st.write(f"**Career Quality:** {career_pct:.1%}")
                st.write(f"**Behavioral Fit:** {behavioral_pct:.1%}")
            with comp_col2:
                if c_row["is_honeypot"]:
                    st.error("⚠️ HONEYPOT DETECTED")
                    st.write(f"**Reasons:** `{c_row['honeypot_reasons']}`")
                else:
                    st.success("✅ Legit Profile")
                st.write(f"**Availability:** {availability_pct:.1%}")
                st.write(f"**Location Fit:** {location_pct:.1%}")
                st.write(f"**Disqualifier Penalty:** -{disqual_val*30:.1%}")
                st.write(f"**Education Bonus:** +{edu_val*5:.1%}")

            # Reasoning
            st.write("#### Recruiter Summary (Reasoning String)")
            st.info(c_row["reasoning"])

            # Tabs for Work, Skills, Signals
            tab_work, tab_skills, tab_signals = st.tabs(["💼 Career History", "🛠️ Skills & Education", "📊 Platform Signals"])
            
            with tab_work:
                st.write("**Career Summary:**")
                st.markdown(f"*{raw_c.get('profile', {}).get('summary', 'No summary provided.')}*")
                st.write("---")
                st.write("**Experience Timeline:**")
                for exp in raw_c.get("career_history", []):
                    st.markdown(f"""
                    **{exp.get('title')}** @ {exp.get('company')}  
                    *{exp.get('start_date')} to {exp.get('end_date') or 'Present'} ({exp.get('duration_months')} months) | Industry: {exp.get('industry')}*  
                    {exp.get('description')}  
                    ---
                    """)
                    
            with tab_skills:
                st.write("**Skills:**")
                # Badges for skills
                skills_html = ""
                for sk in raw_c.get("skills", []):
                    prof = sk.get("proficiency", "beginner")
                    badge_class = "badge-secondary"
                    if prof == "expert":
                        badge_class = "badge-danger"
                    elif prof == "advanced":
                        badge_class = "badge-warning"
                    elif prof == "intermediate":
                        badge_class = "badge-primary"
                    
                    skills_html += f'<span class="badge {badge_class}">{sk.get("name")} ({prof})</span>'
                st.markdown(skills_html, unsafe_allow_html=True)
                
                st.write("\n**Education:**")
                for edu in raw_c.get("education", []):
                    st.markdown(f"""
                    - **{edu.get('degree')} in {edu.get('field_of_study')}**  
                      {edu.get('institution')} ({edu.get('start_year')} - {edu.get('end_year')}) | Grade: {edu.get('grade')} | **{edu.get('tier')}**
                    """)
                    
            with tab_signals:
                sigs = raw_c.get("redrob_signals", {})
                st.write(f"**Notice Period:** {sigs.get('notice_period_days', 'N/A')} days")
                st.write(f"**Recruiter Response Rate:** {sigs.get('recruiter_response_rate', 0.0):.1%}")
                st.write(f"**Interview Completion Rate:** {sigs.get('interview_completion_rate', 0.0):.1%}")
                st.write(f"**GitHub Activity Score:** {sigs.get('github_activity_score', -1)}")
                st.write(f"**Open To Work:** {sigs.get('open_to_work_flag', False)}")
                st.write(f"**Last Active Date:** {sigs.get('last_active_date', 'N/A')}")
                
                assessments = sigs.get("skill_assessment_scores", {})
                if assessments:
                    st.write("**Skill Assessment Scores:**")
                    for k, v in assessments.items():
                        st.progress(v / 100.0, text=f"{k}: {v}/100")
