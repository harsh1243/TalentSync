import streamlit as st
import pandas as pd
import joblib
import importlib
from pathlib import Path
from html import escape
import rank_engine_full as nb

st.set_page_config(page_title="Job ↔ Resume Matcher", layout="wide")

st.markdown("""
<style>
/* ---------- MAIN APP BACKGROUND ---------- */
html, body, .stApp, .main, .block-container {
    background-color: #e6f7ff !important; /* light blue */
    color: #014f86 !important; /* dark blue text */
}

/* ---------- TEXT COLORS ---------- */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #014f86 !important; /* dark blue text */
}

/* ---------- TEXT AREA + INPUT ---------- */
textarea, input, .stTextArea textarea, .stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #0a74d2 !important; /* medium blue border */
    border-radius: 8px !important;
    outline: none !important;
    box-shadow: none !important;
}

textarea:focus, input:focus, .stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #0a74d2 !important;
    box-shadow: 0 0 6px rgba(10, 116, 210, 0.3) !important; /* blue shadow */
}

/* ---------- BUTTONS ---------- */
div.stButton > button {
    background-color: #065da6 !important;  /* darker blue */
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 700 !important; /* Made it bolder */
    font-size: 1rem !important; /* Added font size */
    transition: all 0.2s ease-in-out;
}

/* --- FIX FOR ALL BUTTON TEXT --- */
/* This forces text *inside* all buttons to be white, overriding the global rule */
div.stButton > button span,
div.stButton > button p,
div.stButton > button div {
    color: #ffffff !important;
}
/* --- END FIX --- */


div.stButton > button:hover {
    background-color: #0a74d2 !important; /* medium blue (old default) */
    transform: scale(1.03);
}
div.stButton > button:has(span:contains('')) {
    background-color: #0a74d2 !important; /* medium blue (old default) */
    color: #fff !important;
}

/* ---------- SLIDER ---------- */
.stSlider > div > div > div {
    background: #0a74d2 !imporant; /* medium blue */
}

/* ---------- SCORE BADGE ---------- */
.score-badge {
    background: #ccebff !important; /* light blue badge bg */
    color: #014f86 !important; /* dark blue badge text */
}

/* ---------- SIDEBAR (if any) ---------- */
[data-testid="stSidebar"] {
    background-color: #e6f7ff !important; /* light blue */
    color: #014f86 !important; /* dark blue text */
}

/* Remove Streamlit theme inheritance */
[data-testid="stAppViewContainer"] {
    background-color: #e6f7ff !important;
}
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "home"
if "mode" not in st.session_state:
    st.session_state.mode = "resume_to_jobs"

@st.cache_resource
def load_models_and_scaler():
    base = Path(".")
    models = {}
    mapping = {
        "Ridge": "best_ridge.pkl",
        "SVR": "best_svr.pkl",
        "XGBoost": "best_xgb.pkl",
        "Ordinal": "best_ordinal.pkl"
    }
    for label, fname in mapping.items():
        p = base / "models" / fname
        if p.exists():
            try:
                models[label] = joblib.load(str(p))
            except Exception:
                pass
    scaler = None
    if (base / "models" / "scaler.pkl").exists():
        try:
            scaler = joblib.load(str(base / "models" / "scaler.pkl"))
        except Exception:
            scaler = None
    return models, scaler

models, scaler = load_models_and_scaler()

def pretty_display(df, text_col='job_description', score_col='score', top_k=None):
    if top_k is not None:
        df = df.head(top_k)
    badge_css = """
    <style>
    .score-badge {
        display:inline-block;
        min-width:36px;
        height:36px;
        line-height:36px;
        border-radius:18px;
        background:#e6f4ea;
        color:#0b6e4f;
        text-align:center;
        font-weight:700;
        font-size:0.95rem;
    }
    .result-row {
        padding: 8px 0;
    }
    .result-text {
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .divider {
        border-top: 1px solid #cce4f7;
        margin: 6px 0;
    }
    </style>
    """
    st.markdown(badge_css, unsafe_allow_html=True)
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        score = row.get(score_col, '')
        text_html = escape(text)
        col_left, col_right = st.columns([11,1])
        with col_left:
            st.markdown(f'<div class="result-row"><div class="result-text">{text_html}</div></div>', unsafe_allow_html=True)
        with col_right:
            st.markdown(f'<div style="display:flex;justify-content:flex-end;"><div class="score-badge">{int(score)}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def _render_home_header():
    """Render the title and subtitle for the home page."""
    st.markdown("<h1 style='text-align: center; color: #014f86;'>TalentSync</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: #014f86; margin-bottom: 30px;'>Please select your role to begin</h3>",
        unsafe_allow_html=True
    )


def _render_home_card_styles():
    """Inject CSS styles for the role selection cards."""
    card_style = """
    <style>
    /* Target the native st.container() within the columns */
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #cce4f7;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
        box-shadow: 0 8px 24px rgba(10,116,210,0.15);
    }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)


def _render_recruiter_card():
    """Render the recruiter role selection card."""
    with st.container():
        st.markdown("### 🏢 Recruiter")
        st.markdown("Find the best resumes for a job description.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("I'm a Recruiter", key="recruiter_btn", use_container_width=True):
            st.session_state.mode = "job_to_resumes"
            st.session_state.page = "match"
            st.rerun()


def _render_jobseeker_card():
    """Render the job seeker role selection card."""
    with st.container():
        st.markdown("### 👤 Job Seeker")
        st.markdown("Find the best job listings for your resume.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("I'm a Job Seeker", key="jobseeker_btn", use_container_width=True):
            st.session_state.mode = "resume_to_jobs"
            st.session_state.page = "match"
            st.rerun()


def _render_home_role_cards():
    """Render the two role selection cards side by side."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        card_col1, card_col2 = st.columns(2)
        with card_col1:
            _render_recruiter_card()
        with card_col2:
            _render_jobseeker_card()


def show_home():
    _render_home_header()
    _render_home_card_styles()
    _render_home_role_cards()


def show_match_page():
    mode = st.session_state.get("mode", "resume_to_jobs")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    if mode == "resume_to_jobs":
        st.markdown("<h2 style='color:#014f86;'>Job Seeker Mode</h2>", unsafe_allow_html=True)
        st.markdown("Paste your resume below and we'll find the best matching jobs.")
        resume_text = st.text_area("Your Resume", height=300, placeholder="Paste your resume text here...")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

        if st.button("Find Matching Jobs"):
            if not resume_text.strip():
                st.warning("Please enter your resume text.")
            else:
                with st.spinner("Ranking jobs..."):
                    results = nb.rank_jobs_for_resume(resume_text, models=models, scaler=scaler, top_k=top_k)
                if results is not None and not results.empty:
                    st.success(f"Top {min(top_k, len(results))} matching jobs:")
                    pretty_display(results, text_col='job_description', score_col='score', top_k=top_k)
                else:
                    st.info("No results found.")

    else:
        st.markdown("<h2 style='color:#014f86;'>Recruiter Mode</h2>", unsafe_allow_html=True)
        st.markdown("Paste a job description below and we'll find the best matching resumes.")
        job_text = st.text_area("Job Description", height=300, placeholder="Paste the job description here...")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

        if st.button("Find Matching Resumes"):
            if not job_text.strip():
                st.warning("Please enter a job description.")
            else:
                with st.spinner("Ranking resumes..."):
                    results = nb.rank_resumes_for_job(job_text, models=models, scaler=scaler, top_k=top_k)
                if results is not None and not results.empty:
                    st.success(f"Top {min(top_k, len(results))} matching resumes:")
                    pretty_display(results, text_col='resume', score_col='score', top_k=top_k)
                else:
                    st.info("No results found.")


# --- Main router ---
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "match":
    show_match_page()