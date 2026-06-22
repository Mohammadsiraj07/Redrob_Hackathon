import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import src.feature_extractor as fe
import src.ranker as rk
import math

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI - Recruiter Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Premium Styling (Luxury Dark Mode & Glassmorphism) ───────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

    :root {
        --bg-color: #050d18;
        --surface-color: #0c1524;
        --card-bg: rgba(18, 29, 49, 0.45);
        --border-color: rgba(255, 255, 255, 0.08);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.2);
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
    }

    /* Target background of full page */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #080f1a !important;
        border-right: 1px solid var(--border-color);
        padding-top: 1rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #fff !important;
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Padding Override */
    div.block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* Custom Glassmorphic Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 12px 40px -10px rgba(99, 102, 241, 0.15);
    }

    /* App Headers */
    .app-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #c0c1ff 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Navigation Radio Customizer */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 20px;
    }
    div[role="radiogroup"] label {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        color: var(--text-secondary) !important;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500 !important;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #fff !important;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: var(--primary) !important;
        color: #fff !important;
        border-color: var(--primary) !important;
        box-shadow: 0 0 12px var(--primary-glow) !important;
    }
    /* Hide the radio circles */
    div[role="radiogroup"] label span[data-testid="stWidgetMarkdownWithHelp"] {
        margin-left: 0px !important;
    }
    div[role="radiogroup"] label div[data-testid="stMarker"] {
        display: none !important;
    }

    /* Heatmap/Classification Chips for Skills */
    .skill-chip {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        font-family: 'Inter', sans-serif;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    .skill-match {
        background: rgba(16, 185, 129, 0.1) !important;
        color: var(--success) !important;
        border-color: rgba(16, 185, 129, 0.2) !important;
    }
    .skill-missing {
        background: rgba(239, 68, 68, 0.08) !important;
        color: var(--danger) !important;
        border-color: rgba(239, 68, 68, 0.15) !important;
    }
    .skill-transferable {
        background: rgba(59, 130, 246, 0.08) !important;
        color: var(--info) !important;
        border-color: rgba(59, 130, 246, 0.15) !important;
    }
    .skill-emerging {
        background: rgba(139, 92, 246, 0.08) !important;
        color: #a78bfa !important;
        border-color: rgba(139, 92, 246, 0.15) !important;
    }

    /* Premium Data Table styling */
    .premium-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }
    .premium-table th {
        background: rgba(12, 21, 36, 0.8) !important;
        text-align: left;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        border-bottom: 2px solid var(--border-color);
    }
    .premium-table td {
        padding: 16px;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.875rem;
        vertical-align: middle;
        background: rgba(12, 21, 36, 0.2);
    }
    .premium-table tr:hover td {
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* Timeline styling */
    .timeline {
        position: relative;
        padding-left: 24px;
        border-left: 2px solid var(--border-color);
        margin-left: 10px;
        margin-top: 10px;
    }
    .timeline-item {
        position: relative;
        margin-bottom: 24px;
    }
    .timeline-dot {
        position: absolute;
        left: -33px;
        top: 6px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--primary);
        border: 3px solid var(--bg-color);
        box-shadow: 0 0 10px var(--primary);
    }

    /* General widgets */
    .widget-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Progress Animations */
    @keyframes progress-grow {
        from { width: 0%; }
    }
    .animate-progress-bar {
        animation: progress-grow 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }

    @keyframes radial-grow {
        from { stroke-dashoffset: 100; }
    }
    .animate-radial-bar {
        animation: radial-grow 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }

</style>
""", unsafe_allow_html=True)


# ─── Sidebar Configuration ──────────────────────────────────────────────────
st.sidebar.image("https://raw.githubusercontent.com/redrob-co/assets/main/logo.png", width=140)
st.sidebar.title("Settings & Controls")

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
            "matched_skills_list": fe.get_matched_skills_list(c),
            "top_employers": fe.get_top_employers(c),
            "top_education": fe.get_top_education(c),
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


# ─── UI Helper Renderers ─────────────────────────────────────────────────────

def generate_table_html(df):
    rows_html = ""
    for _, row in df.iterrows():
        rank = row.get("rank", 0)
        cid = row.get("candidate_id", "N/A")
        score = row.get("score_rounded", 0.0)
        yoe = row.get("years_of_experience", 0.0)
        company = row.get("current_company", "N/A")
        if not company or str(company) == "nan":
            company = "N/A"
        notice = int(row.get("notice_period_days", 90))
        is_hp = row.get("is_honeypot", False)
        
        # Determine Fit Category
        if is_hp:
            fit_cat = "Honeypot"
            fit_class = "skill-missing"
        elif score >= 0.75:
            fit_cat = "Strong Fit"
            fit_class = "skill-match"
        elif score >= 0.65:
            fit_cat = "Good Fit"
            fit_class = "skill-transferable"
        else:
            fit_cat = "Keep in Pool"
            fit_class = "skill-emerging"
            
        notice_txt = "Immediate" if notice <= 15 else f"{notice}d"
        notice_class = "skill-match" if notice <= 30 else ("skill-transferable" if notice <= 60 else "skill-missing")
        
        rows_html += (
            f"<tr>"
            f"<td style=\"font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #6366f1;\">#{rank}</td>"
            f"<td style=\"font-weight: 600; color: #ffffff;\">{cid}</td>"
            f"<td>"
            f"<div style=\"display: flex; align-items: center; gap: 8px;\">"
            f"<span style=\"font-family: 'JetBrains Mono', monospace; font-weight: bold; color: #10b981;\">{score:.2%}</span>"
            f"<div style=\"background: rgba(255, 255, 255, 0.05); width: 60px; height: 4px; border-radius: 2px; overflow: hidden;\">"
            f"<div class=\"animate-progress-bar\" style=\"background: #10b981; width: {score * 100}%; height: 100%;\"></div>"
            f"</div>"
            f"</div>"
            f"</td>"
            f"<td>{yoe:.1f} yrs</td>"
            f"<td>{company}</td>"
            f"<td><span class=\"skill-chip {notice_class}\" style=\"margin: 0; padding: 2px 8px; font-size: 0.7rem;\">{notice_txt}</span></td>"
            f"<td><span class=\"skill-chip {fit_class}\" style=\"margin: 0; padding: 2px 8px; font-size: 0.7rem;\">{fit_cat}</span></td>"
            f"</tr>"
        )
        
    html = (
        f"<div style=\"width: 100%; overflow-x: auto;\">"
        f"<table class=\"premium-table\">"
        f"<thead>"
        f"<tr>"
        f"<th>Rank</th>"
        f"<th>Candidate ID</th>"
        f"<th>Match Score</th>"
        f"<th>Experience</th>"
        f"<th>Current Company</th>"
        f"<th>Notice Period</th>"
        f"<th>Fit Category</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>"
        f"{rows_html}"
        f"</tbody>"
        f"</table>"
        f"</div>"
    )
    return html


def render_radial_svg(score: float, label: str, color: str = "#6366f1"):
    pct = int(score * 100)
    html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 10px;">
        <div style="position: relative; width: 84px; height: 84px;">
            <svg viewBox="0 0 36 36" style="width: 100%; height: 100%; transform: rotate(-90deg);">
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                      fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="2.8" />
                <path class="animate-radial-bar" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                      fill="none" stroke="{color}" stroke-width="2.8" stroke-dasharray="100" 
                      stroke-dashoffset="{100 - pct}" stroke-linecap="round" />
            </svg>
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #ffffff;">
                {pct}%
            </div>
        </div>
        <div style="margin-top: 8px; font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #94a3b8; text-align: center; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
            {label}
        </div>
    </div>
    """
    return html


def render_metric_card(label, value, border_color="#6366f1", subtext=""):
    subtext_html = ""
    if subtext:
        subtext_html = f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.75rem; color: #ef4444; margin-top: 4px; font-weight: 500;">{subtext}</div>'
        
    html = f"""
    <div style="background: rgba(18, 29, 49, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {border_color}; border-radius: 8px; padding: 16px; margin-bottom: 16px; backdrop-filter: blur(12px);">
        <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em;">{label}</div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-top: 4px; line-height: 1.2;">{value}</div>
        {subtext_html}
    </div>
    """
    return html


def render_candidate_header(c_row):
    name = c_row["candidate_raw"].get("profile", {}).get("anonymized_name", "Anonymized Candidate")
    title = c_row["current_title"]
    company = c_row["current_company"]
    yoe = c_row["years_of_experience"]
    location = c_row["location"]
    country = c_row["country"]
    notice = int(c_row["notice_period_days"])
    willing = c_row["willing_to_relocate"]
    
    notice_txt = "Immediate Availability" if notice <= 15 else f"{notice}-Day Notice"
    reloc_txt = "Willing to relocate" if willing else "No relocation"
    
    html = f"""
    <div class="glass-card" style="border-left: 4px solid #6366f1; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: bold; box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);">
                {name[0] if name else 'C'}
            </div>
            <div>
                <h2 style="margin: 0; color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 1.7rem; font-weight: 700;">{name}</h2>
                <div style="color: #ffffff; font-weight: 500; margin-top: 4px; font-size: 1rem;">{title} <span style="color: #94a3b8; font-weight: normal;">at {company or 'N/A'}</span></div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 6px;">
                    📍 {location}, {country} &nbsp;|&nbsp; 💼 {yoe:.1f} Years Exp &nbsp;|&nbsp; ⏱️ {notice_txt} &nbsp;|&nbsp; ✈️ {reloc_txt}
                </div>
            </div>
        </div>
    </div>
    """
    return html


def render_recruiter_summary(c_row):
    yoe = c_row["years_of_experience"]
    company = c_row["current_company"]
    title = c_row["current_title"]
    notice = int(c_row["notice_period_days"])
    location = c_row["location"]
    skills_matched = c_row["matched_skills_list"]
    assess_score = c_row["assessment_score"]
    github_score = c_row["github_activity_score"]
    is_hp = c_row["is_honeypot"]
    
    # Strengths
    strengths = []
    strengths.append(f"Candidate has <b>{yoe:.1f} years</b> of engineering experience, fitting the senior profile.")
    if company:
        strengths.append(f"Pedigree includes tenure at <b>{company}</b> as a <b>{title}</b>.")
    if str(skills_matched) != "nan" and skills_matched:
        clean_skills = ", ".join([s.strip().title() for s in str(skills_matched).split(",") if s.strip()][:3])
        strengths.append(f"Successfully matches core JD skills including: <i>{clean_skills}</i>.")
    if assess_score > 0.7:
        strengths.append(f"Outstanding online assessment score of <b>{assess_score:.0%}</b>.")
    if github_score >= 60:
        strengths.append(f"Highly active GitHub profile with an engagement score of <b>{int(github_score)}</b>.")
    if len(strengths) < 2:
        strengths.append("Foundational technical engineering skills aligned with product team requirements.")
        
    # Concerns
    concerns = []
    if is_hp:
        concerns.append("<b style='color:#ef4444;'>CRITICAL: Candidate flagged as a Honeypot (adversarial profile).</b>")
    if notice > 45:
        concerns.append(f"Notice period of <b>{notice} days</b> is relatively long and may require a buyout.")
    if yoe > 12:
        concerns.append(f"Overqualified for early-stage founding role (<b>{yoe:.1f} YoE</b>), potential retention risk.")
    if not c_row["willing_to_relocate"] and "noida" not in location.lower() and "pune" not in location.lower():
        concerns.append(f"Located in <b>{location}</b> and has marked 'Not Willing to Relocate'.")
    if assess_score < 0.5:
        concerns.append(f"Relatively low platform assessment score (<b>{assess_score:.0%}</b>).")
    if not concerns:
        concerns.append("No major red flags or compliance concerns identified.")
        
    # Hiring Recommendation
    score = c_row["score_rounded"]
    if is_hp:
        recommendation = "Reject Profile"
        recommendation_color = "#ef4444"
        action = "Archive candidate immediately. Do not contact due to profile authenticity flags."
    elif score >= 0.75:
        recommendation = "Strong Proceed"
        recommendation_color = "#10b981"
        action = "Fast-track to Technical Interview Panel. Focus on system architecture and scale."
    elif score >= 0.65:
        recommendation = "Proceed with Care"
        recommendation_color = "#f59e0b"
        action = "Schedule initial Recruiter Screening call to clarify location/relocation preferences and salary alignment."
    else:
        recommendation = "Hold / Keep in Pool"
        recommendation_color = "#94a3b8"
        action = "Keep in talent pool. Review for adjacent development needs rather than core founding team."
        
    html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px;">
        <div class="glass-card" style="margin-bottom:0; border-top: 3px solid #10b981; padding: 18px;">
            <h4 style="margin-top:0; color:#10b981; font-family:'Outfit', sans-serif; font-size:1rem; margin-bottom:12px;">🟢 Strengths</h4>
            <ul style="padding-left:20px; margin:0; color:#cbd5e1; font-size:0.8rem; line-height:1.4;">
                {"".join(f'<li style="margin-bottom:6px;">{s}</li>' for s in strengths)}
            </ul>
        </div>
        <div class="glass-card" style="margin-bottom:0; border-top: 3px solid #ef4444; padding: 18px;">
            <h4 style="margin-top:0; color:#ef4444; font-family:'Outfit', sans-serif; font-size:1rem; margin-bottom:12px;">🔴 Concerns</h4>
            <ul style="padding-left:20px; margin:0; color:#cbd5e1; font-size:0.8rem; line-height:1.4;">
                {"".join(f'<li style="margin-bottom:6px;">{c}</li>' for c in concerns)}
            </ul>
        </div>
        <div class="glass-card" style="margin-bottom:0; border-top: 3px solid #3b82f6; padding: 18px;">
            <h4 style="margin-top:0; color:#3b82f6; font-family:'Outfit', sans-serif; font-size:1rem; margin-bottom:12px;">ℹ️ Hiring Recommendation</h4>
            <div style="margin-bottom:10px;">
                <span style="background: {recommendation_color}1a; color: {recommendation_color}; border: 1px solid {recommendation_color}33; padding: 4px 8px; border-radius: 4px; font-weight:bold; font-size:0.75rem;">
                    {recommendation}
                </span>
            </div>
            <p style="color:#cbd5e1; font-size:0.8rem; margin:0; line-height:1.4;">
                <b>Recommended Action:</b> {action}
            </p>
        </div>
    </div>
    """
    return html


def render_skills_intelligence(c_row):
    raw_c = c_row["candidate_raw"]
    skills = raw_c.get("skills", [])
    
    # Sort matched skills
    matched_str = str(c_row["matched_skills_list"])
    matched_names = [s.strip().lower() for s in matched_str.split(",") if s.strip() and s.strip().lower() != 'nan']
    
    matching = []
    missing = []
    transferable = []
    emerging = []
    
    # Classify candidate's actual skills
    for sk in skills:
        name = sk["name"]
        prof = sk.get("proficiency", "beginner")
        
        if name.lower() in matched_names or any(m in name.lower() for m in matched_names):
            matching.append(f"{name} ({prof})")
        else:
            if any(t in name.lower() for t in ["c++", "java", "sql", "git", "aws", "gcp", "docker", "redis", "kafka"]):
                transferable.append(f"{name} ({prof})")
            elif any(e in name.lower() for e in ["llm", "lora", "embeddings", "vector", "rag", "langchain", "prompt", "pinecone", "weaviate"]):
                emerging.append(f"{name} ({prof})")
            else:
                transferable.append(name)
                
    # Check what core JD skills are missing
    core_jd_skills = ["embeddings", "vector database", "hybrid search", "ranking", "ndcg", "rag", "llm", "fine-tuning"]
    for js in core_jd_skills:
        if not any(js.lower() in s.lower() for s in [sk["name"] for sk in skills]):
            missing.append(js.title())
            
    # Fill placeholders if empty
    if not emerging:
        emerging = ["Vector Retrieval", "Prompt Engineering"]
    if not missing:
        missing = ["None (100% matched)"]
        
    html_matching = "".join(f'<span class="skill-chip skill-match">{m}</span>' for m in matching[:8])
    html_missing = "".join(f'<span class="skill-chip skill-missing">{m}</span>' for m in missing[:5])
    html_transferable = "".join(f'<span class="skill-chip skill-transferable">{t}</span>' for t in transferable[:5])
    html_emerging = "".join(f'<span class="skill-chip skill-emerging">{e}</span>' for e in emerging[:5])
    
    html = f"""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#ffffff; font-family:'Outfit', sans-serif; font-size:1.2rem; margin-bottom:16px;">🛠️ Skills Intelligence</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
            <div>
                <h5 style="color:#10b981; margin-bottom:10px; font-size:0.85rem; font-family:'Inter',sans-serif; font-weight:600;">Matched JD Skills</h5>
                <div>{html_matching or '<span style="color:#6b7280; font-size:0.8rem;">No direct matches</span>'}</div>
            </div>
            <div>
                <h5 style="color:#ef4444; margin-bottom:10px; font-size:0.85rem; font-family:'Inter',sans-serif; font-weight:600;">Missing Core Skills</h5>
                <div>{html_missing}</div>
            </div>
            <div>
                <h5 style="color:#3b82f6; margin-bottom:10px; font-size:0.85rem; font-family:'Inter',sans-serif; font-weight:600;">Transferable / Adjacent</h5>
                <div>{html_transferable or '<span style="color:#6b7280; font-size:0.8rem;">None</span>'}</div>
            </div>
            <div>
                <h5 style="color:#a78bfa; margin-bottom:10px; font-size:0.85rem; font-family:'Inter',sans-serif; font-weight:600;">Emerging / Niche</h5>
                <div>{html_emerging}</div>
            </div>
        </div>
    </div>
    """
    return html


def parse_experience_description(desc: str):
    sentences = [s.strip() for s in desc.split('.') if s.strip()]
    achievements = []
    impacts = []
    
    for s in sentences:
        is_impact = any(k in s.lower() for k in ["%", "percent", "lpa", "million", "reduced", "improved", "increased", "optimized", "scale", "latency", "accuracy"])
        if is_impact:
            impacts.append(s)
        else:
            achievements.append(s)
            
    if not achievements:
        achievements = sentences
    if not impacts:
        impacts = ["Contributed to core development and system optimizations."]
        
    return achievements, impacts


def render_career_timeline(c_row):
    raw_c = c_row["candidate_raw"]
    career = raw_c.get("career_history", [])
    
    timeline_html = ""
    for exp in career:
        title = exp.get("title")
        company = exp.get("company")
        dur = exp.get("duration_months")
        industry = exp.get("industry")
        start = exp.get("start_date")
        end = exp.get("end_date") or "Present"
        desc = exp.get("description", "")
        
        achievements, impacts = parse_experience_description(desc)
        achievements_list = "".join(f'<li style="margin-bottom:6px;">{ach}.</li>' for ach in achievements[:3])
        impacts_list = "".join(f'<div style="margin-top:10px; font-size:0.75rem; color:#a78bfa; font-family:\'JetBrains Mono\', monospace; font-weight:600;">📈 Business Impact: {imp}.</div>' for imp in impacts[:1])
        
        timeline_html += (
            f'<div class="timeline-item">'
            f'<div class="timeline-dot"></div>'
            f'<div class="glass-card" style="margin-left: 20px; padding: 18px; border-left: 3px solid #6366f1; margin-bottom: 16px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">'
            f'<div>'
            f'<h4 style="margin:0; color:#ffffff; font-size:1.05rem; font-weight:600;">{title}</h4>'
            f'<div style="color:#a78bfa; font-weight:500; font-size:0.875rem; margin-top:2px;">{company}</div>'
            f'</div>'
            f'<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.7rem; background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:4px; color:#cbd5e1;">'
            f'{start} to {end} ({dur} months)'
            f'</div>'
            f'</div>'
            f'<div style="color:#94a3b8; font-size:0.75rem; margin-top:2px;">Industry: {industry}</div>'
            f'<ul style="padding-left:18px; margin:10px 0 0 0; color:#e2e8f0; font-size:0.8rem; line-height:1.45;">'
            f'{achievements_list}'
            f'</ul>'
            f'{impacts_list}'
            f'</div>'
            f'</div>'
        )
        
    html = (
        f'<div class="glass-card">'
        f'<h3 style="margin-top:0; color:#ffffff; font-family:\'Outfit\', sans-serif; font-size:1.2rem; margin-bottom:16px;">💼 Career Timeline</h3>'
        f'<div class="timeline">'
        f'{timeline_html}'
        f'</div>'
        f'</div>'
    )
    return html


def render_platform_signals(c_row):
    sigs = c_row["candidate_raw"].get("redrob_signals", {})
    github = sigs.get("github_activity_score", -1)
    rrr = sigs.get("recruiter_response_rate", 0.0)
    icr = sigs.get("interview_completion_rate", 0.0)
    open_work = sigs.get("open_to_work_flag", False)
    completeness = sigs.get("profile_completeness_score", 0)
    last_active = sigs.get("last_active_date", "N/A")
    
    github_txt = f"{int(github)}" if github >= 0 else "Not Linked"
    open_txt = "Actively Looking" if open_work else "Passive"
    open_color = "#10b981" if open_work else "#94a3b8"
    
    html = f"""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#ffffff; font-family:'Outfit', sans-serif; font-size:1.2rem; margin-bottom:16px;">📊 Platform Signals</h3>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 12px;">
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.02); border-radius:6px; border:1px solid var(--border-color);">
                <div style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">GitHub Score</div>
                <div style="font-size:1.2rem; font-weight:bold; color:#ffffff; margin-top:4px; font-family:'JetBrains Mono', monospace;">{github_txt}</div>
            </div>
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.02); border-radius:6px; border:1px solid var(--border-color);">
                <div style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Response Rate</div>
                <div style="font-size:1.2rem; font-weight:bold; color:#ffffff; margin-top:4px; font-family:'JetBrains Mono', monospace;">{rrr:.0%}</div>
            </div>
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.02); border-radius:6px; border:1px solid var(--border-color);">
                <div style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Interview Rate</div>
                <div style="font-size:1.2rem; font-weight:bold; color:#ffffff; margin-top:4px; font-family:'JetBrains Mono', monospace;">{icr:.0%}</div>
            </div>
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.02); border-radius:6px; border:1px solid var(--border-color);">
                <div style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Search Status</div>
                <div style="font-size:1rem; font-weight:bold; color:{open_color}; margin-top:6px;">{open_txt}</div>
            </div>
        </div>
        <div style="margin-top:14px; font-size:0.75rem; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.05); padding-top:10px;">
            Last active: <b>{last_active}</b> &nbsp;|&nbsp; Completeness: <b>{completeness}%</b>
        </div>
    </div>
    """
    return html


def render_explainability(c_row):
    skills = c_row["skill_match"]
    career = c_row["career_quality"]
    beh = c_row["behavioral_score"]
    
    html = f"""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#ffffff; font-family:'Outfit', sans-serif; font-size:1.2rem; margin-bottom:12px;">🧠 AI Explainability & Confidence</h3>
        <p style="color:#cbd5e1; font-size:0.8rem; margin-bottom:16px; line-height:1.4;">
            Why does this profile match? Core features mapping to Series A Senior AI Engineer role:
        </p>
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:4px; color:#cbd5e1;">
                <span>Technical Skill Alignment</span>
                <span style="font-family:'JetBrains Mono', monospace;">{skills:.1%}</span>
            </div>
            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;">
                <div class="animate-progress-bar" style="background:#10b981; width:{skills*100}%; height:100%;"></div>
            </div>
        </div>
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:4px; color:#cbd5e1;">
                <span>Career Quality & Applied ML Background</span>
                <span style="font-family:'JetBrains Mono', monospace;">{career:.1%}</span>
            </div>
            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;">
                <div class="animate-progress-bar" style="background:#3b82f6; width:{career*100}%; height:100%;"></div>
            </div>
        </div>
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:4px; color:#cbd5e1;">
                <span>Engagement & Response Recency</span>
                <span style="font-family:'JetBrains Mono', monospace;">{beh:.1%}</span>
            </div>
            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;">
                <div class="animate-progress-bar" style="background:#8b5cf6; width:{beh*100}%; height:100%;"></div>
            </div>
        </div>
    </div>
    """
    return html


# ─── Data Loading ────────────────────────────────────────────────────────────

# Initialize session state for raw candidate content and source
if "raw_content" not in st.session_state:
    st.session_state.raw_content = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None

uploaded_file = st.file_uploader("Upload candidates file (JSON or JSONL)", type=["json", "jsonl"])

# Handle file upload and clearing
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    if st.session_state.raw_content != content or st.session_state.data_source != "upload":
        st.session_state.raw_content = content
        st.session_state.data_source = "upload"
        st.rerun()
elif st.session_state.data_source == "upload":
    # User cleared the uploaded file
    st.session_state.raw_content = None
    st.session_state.data_source = None
    st.rerun()

# If no active data source, show landing page with option to load sample
if st.session_state.raw_content is None:
    st.info("Upload a candidate profile file or load the 50 sample candidates below to test the UI.")
    if st.button("Load 50 Sample Candidates"):
        sample_path = Path("sample_candidates.jsonl")
        if not sample_path.exists():
            sample_path = Path("data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json")
        
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.session_state.raw_content = content
            st.session_state.data_source = "sample"
            st.rerun()
        else:
            st.error(f"Sample candidates file not found at {sample_path}")

# Process data if raw content is present in session state
candidates_df = None
if st.session_state.raw_content is not None:
    candidates_df = process_data(st.session_state.raw_content, jd_text)

# ─── Dashboard Render (12-Column Layout Grid) ───────────────────────────────

if candidates_df is not None:
    # Desktop layout split: Col 1 = Left Rail, Col 2 = Center Main Workspace, Col 3 = Right Insights
    col_rail, col_center, col_right = st.columns([1.5, 7.5, 3])

    selected_cid = None

    with col_rail:
        st.markdown('<div class="app-title" style="font-size:1.4rem;">Redrob AI</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600; letter-spacing:0.05em; margin-bottom:10px;">Workspace Menu</div>', unsafe_allow_html=True)
        nav_selection = st.radio(
            "Navigation",
            options=["📊 Dashboard", "🔍 Analyzer", "⚔️ Comparison"],
            label_visibility="collapsed"
        )
        
        if "Analyzer" in nav_selection:
            st.markdown("---")
            st.markdown('<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600; letter-spacing:0.05em; margin-bottom:10px;">Select Candidate</div>', unsafe_allow_html=True)
            selected_cid = st.selectbox(
                "Candidate Select",
                options=candidates_df["candidate_id"].tolist(),
                format_func=lambda x: f"{x} | {candidates_df[candidates_df['candidate_id']==x]['current_title'].values[0][:20]}",
                label_visibility="collapsed"
            )

    with col_center:
        # Header in Center stage
        st.markdown(f'<div class="app-title">Obsidian Talent Intelligence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="app-subtitle">Active Job: Senior AI Engineer (Founding Team) — Series A, Pune/Noida</div>', unsafe_allow_html=True)

        if "Dashboard" in nav_selection:
            st.subheader("Candidate Rankings Dashboard")
            
            # Filters inside workspace
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                loc_search = st.text_input("📍 Filter by Location keyword", "")
            with filter_col2:
                exclude_hp_display = st.checkbox("🚫 Hide honeypot candidates", value=False)
                
            display_df = candidates_df.copy()
            if loc_search:
                display_df = display_df[display_df["location"].str.contains(loc_search, case=False, na=False)]
            if exclude_hp_display:
                display_df = display_df[display_df["is_honeypot"] == False]
                
            # Reset page if filter values changed
            if "last_loc_search" not in st.session_state:
                st.session_state.last_loc_search = ""
            if "last_exclude_hp" not in st.session_state:
                st.session_state.last_exclude_hp = False
                
            if st.session_state.last_loc_search != loc_search or st.session_state.last_exclude_hp != exclude_hp_display:
                st.session_state.current_page = 1
                st.session_state.last_loc_search = loc_search
                st.session_state.last_exclude_hp = exclude_hp_display

            # Pagination logic: 10 candidates per page
            candidates_per_page = 10
            total_pages = math.ceil(len(display_df) / candidates_per_page)
            
            if total_pages > 1:
                if "current_page" not in st.session_state:
                    st.session_state.current_page = 1
                
                if st.session_state.current_page > total_pages:
                    st.session_state.current_page = total_pages
                if st.session_state.current_page < 1:
                    st.session_state.current_page = 1
                    
                start_idx = (st.session_state.current_page - 1) * candidates_per_page
                end_idx = start_idx + candidates_per_page
                page_df = display_df.iloc[start_idx:end_idx]
                
                st.write(f"Showing candidates {start_idx + 1} to {min(end_idx, len(display_df))} of {len(display_df)}:")
                table_html = generate_table_html(page_df)
                st.markdown(table_html.strip(), unsafe_allow_html=True)
                
                # Render Page Select Buttons side by side
                st.write("")
                cols_pag = st.columns(total_pages + 2)
                
                with cols_pag[0]:
                    if st.button("◀ Prev", disabled=(st.session_state.current_page == 1), use_container_width=True):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                for p in range(1, total_pages + 1):
                    with cols_pag[p]:
                        is_active = (p == st.session_state.current_page)
                        btn_label = f"[{p}]" if is_active else f"{p}"
                        if st.button(btn_label, type="primary" if is_active else "secondary", use_container_width=True):
                            st.session_state.current_page = p
                            st.rerun()
                            
                with cols_pag[total_pages + 1]:
                    if st.button("Next ▶", disabled=(st.session_state.current_page == total_pages), use_container_width=True):
                        st.session_state.current_page += 1
                        st.rerun()
            else:
                st.session_state.current_page = 1
                st.write(f"Showing all {len(display_df)} candidates:")
                table_html = generate_table_html(display_df)
                st.markdown(table_html.strip(), unsafe_allow_html=True)
                
        elif "Analyzer" in nav_selection:
            st.subheader("Contextual Profile Analyzer")
            
            if selected_cid:
                c_row = candidates_df[candidates_df["candidate_id"] == selected_cid].iloc[0]
                
                # 1. Candidate Header
                st.markdown(render_candidate_header(c_row), unsafe_allow_html=True)
                
                # Action buttons side-by-side
                col_act1, col_act2, col_act3, col_act4 = st.columns(4)
                with col_act1:
                    if st.button("✓ Shortlist Candidate", use_container_width=True):
                        st.toast(f"Candidate {c_row['candidate_id']} added to shortlist!", icon="✅")
                with col_act2:
                    if st.button("⚔️ Add to Compare", use_container_width=True):
                        st.toast(f"Candidate {c_row['candidate_id']} added to comparison board!", icon="⚔️")
                with col_act3:
                    if st.button("⭐ Save Profile", use_container_width=True):
                        st.toast(f"Profile saved to recruitment pipeline.", icon="⭐")
                with col_act4:
                    if st.button("📥 Export Report", use_container_width=True):
                        st.toast(f"Exporting details as PDF report...", icon="📥")
                        
                st.write("")
                
                # 2. AI Match Score Hero
                st.write("#### AI Match Score Summary")
                overall_score = float(c_row["score_rounded"])
                semantic_score = float(c_row["tfidf_similarity"])
                skills_score = float(c_row["skill_match"])
                behavioral_score = float(c_row["behavioral_score"])
                education_score = float(c_row["education_bonus"]) * 2.5 # scaled to [0,1]
                
                radial_layout = (
                    f'<div style="display: flex; flex-wrap: wrap; justify-content: space-around; background: rgba(18, 29, 49, 0.45); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(12px); margin-bottom:24px;">'
                    f'{render_radial_svg(overall_score, "Overall Match", "#6366f1").strip()}'
                    f'{render_radial_svg(semantic_score, "Semantic Fit", "#3b82f6").strip()}'
                    f'{render_radial_svg(skills_score, "Skills Fit", "#10b981").strip()}'
                    f'{render_radial_svg(behavioral_score, "Behavioral Fit", "#f59e0b").strip()}'
                    f'{render_radial_svg(education_score, "Education Fit", "#8b5cf6").strip()}'
                    f'</div>'
                )
                st.markdown(radial_layout, unsafe_allow_html=True)
                
                # 3. Recruiter AI Summary (Strengths, Concerns, Recommendation)
                st.markdown(render_recruiter_summary(c_row), unsafe_allow_html=True)
                
                # 4. Skills Intelligence
                st.markdown(render_skills_intelligence(c_row), unsafe_allow_html=True)
                
                # 5. Career Timeline
                st.markdown(render_career_timeline(c_row), unsafe_allow_html=True)
                
        elif "Comparison" in nav_selection:
            st.subheader("Side-by-Side Candidate Comparison")
            
            # Select candidates to compare
            compare_ids = st.multiselect(
                "Select Candidate IDs to Compare (Max 3)",
                options=candidates_df["candidate_id"].tolist(),
                default=candidates_df["candidate_id"].head(2).tolist()
            )
            
            if len(compare_ids) > 0:
                comp_rows = candidates_df[candidates_df["candidate_id"].isin(compare_ids)]
                
                # Render Comparison Grid
                cols_grid = st.columns(len(compare_ids))
                for idx, (_, c) in enumerate(comp_rows.iterrows()):
                    with cols_grid[idx]:
                        score_pct = int(c["score_rounded"] * 100)
                        is_hp = c["is_honeypot"]
                        fit_lbl = "Honeypot" if is_hp else ("Strong Fit" if score_pct >= 75 else "Good Fit")
                        fit_col = "#ef4444" if is_hp else ("#10b981" if score_pct >= 75 else "#3b82f6")
                        
                        comp_card_html = (
                            f'<div class="glass-card" style="border-top: 4px solid {fit_col}; min-height: 500px;">'
                            f'<div style="font-family:\'Outfit\', sans-serif; font-size:1.3rem; font-weight:bold; color:#fff;">{c["candidate_id"]}</div>'
                            f'<div style="font-size:0.9rem; font-weight:600; color:#a78bfa; margin-top:2px;">{c["current_title"]}</div>'
                            f'<div style="font-size:0.8rem; color:#94a3b8;">at {c["current_company"] or "N/A"}</div>'
                            f'<div style="display:flex; flex-direction:column; gap:8px; margin-top:20px; background:rgba(255,255,255,0.02); padding:12px; border-radius:6px; border:1px solid var(--border-color);">'
                            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                            f'<span style="font-size:0.8rem; color:#94a3b8;">Match Score</span>'
                            f'<span style="font-family:\'JetBrains Mono\',monospace; font-weight:bold; color:#fff; font-size:1.2rem;">{score_pct}%</span>'
                            f'</div>'
                            f'<div style="background: rgba(255, 255, 255, 0.05); width: 100%; height: 6px; border-radius: 3px; overflow: hidden;">'
                            f'<div class="animate-progress-bar" style="background: {fit_col}; width: {score_pct}%; height: 100%;"></div>'
                            f'</div>'
                            f'</div>'
                            f'<div style="margin-top:16px;">'
                            f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; margin-bottom:4px;">Experience</div>'
                            f'<div style="font-size:0.9rem; color:#fff; font-weight:500;">{c["years_of_experience"]:.1f} Years of Experience</div>'
                            f'</div>'
                            f'<div style="margin-top:16px;">'
                            f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; margin-bottom:4px;">Location</div>'
                            f'<div style="font-size:0.9rem; color:#fff; font-weight:500;">{c["location"]}, {c["country"]}</div>'
                            f'</div>'
                            f'<div style="margin-top:16px;">'
                            f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; margin-bottom:4px;">Notice Period</div>'
                            f'<div style="font-size:0.9rem; color:#fff; font-weight:500;">{int(c["notice_period_days"])} Days notice</div>'
                            f'</div>'
                            f'<div style="margin-top:20px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:16px;">'
                            f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; margin-bottom:6px;">Summary Highlight</div>'
                            f'<div style="font-size:0.8rem; color:#cbd5e1; line-height:1.45;">{c["reasoning"][:160]}...</div>'
                            f'</div>'
                            f'</div>'
                        )
                        st.markdown(comp_card_html, unsafe_allow_html=True)
            else:
                st.warning("Please select at least one candidate above to compare.")

    with col_right:
        # Contextual insights sidebar (active at all times)
        st.markdown('<div class="glass-card" style="padding: 16px; margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0; color:#ffffff; font-family:\'Outfit\', sans-serif; font-size:1.1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:10px; margin-bottom:16px;">💡 Contextual Insights</h4>', unsafe_allow_html=True)
        
        # General stats
        total_eval = len(candidates_df)
        avg_score = candidates_df["score_rounded"].mean()
        top_score = candidates_df["score_rounded"].max()
        
        st.markdown(render_metric_card("Total Evaluated", f"{total_eval:,}", "#6366f1"), unsafe_allow_html=True)
        st.markdown(render_metric_card("Average Match", f"{avg_score:.2%}", "#3b82f6"), unsafe_allow_html=True)
        st.markdown(render_metric_card("Top Candidate", f"{top_score:.2%}", "#10b981"), unsafe_allow_html=True)
        
        # Honeypot stats
        num_hp = candidates_df["is_honeypot"].sum()
        pct_hp = num_hp / len(candidates_df)
        hp_subtext = "High Rate Alert" if pct_hp > 0.1 else ""
        st.markdown(render_metric_card("Honeypots Flagged", f"{num_hp}", "#ef4444", hp_subtext), unsafe_allow_html=True)
        
        # Contextual Candidate Widget
        if "Analyzer" in nav_selection and selected_cid:
            c_row = candidates_df[candidates_df["candidate_id"] == selected_cid].iloc[0]
            st.markdown("---")
            # Render Platform Signals & Explainability in Right rail
            st.markdown(render_platform_signals(c_row), unsafe_allow_html=True)
            st.markdown(render_explainability(c_row), unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
