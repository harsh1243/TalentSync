import streamlit as st
import pandas as pd
import joblib
import importlib
from pathlib import Path
from html import escape
import rank_engine_full as nb

st.set_page_config(page_title="Job ↔ Resume Matcher", layout="wide")

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
        p = base / fname
        if p.exists():
            try:
                models[label] = joblib.load(str(p))
            except Exception:
                pass
    scaler = None
    if (base / "scaler.pkl").exists():
        try:
            scaler = joblib.load(str(base / "scaler.pkl"))
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
      color:#0b3b18;
      text-align:center;
      font-weight:700;
      box-shadow: 0 1px 0 rgba(0,0,0,0.15) inset;
    }
    .result-row { padding: 10px 0; }
    .result-text { font-size:16px; color: #e6eef8; }
    .divider { border-bottom:1px solid rgba(255,255,255,0.06); margin:8px 0; }
    </style>
    """
    st.markdown(badge_css, unsafe_allow_html=True)
    for i, row in df.reset_index(drop=True).iterrows():
        text = str(row.get(text_col, ''))
        score = row.get(score_col, '')
        text_html = escape(text)
        col_left, col_right = st.columns([11,1])
        with col_left:
            st.markdown(f'<div class="result-row"><div class="result-text">{text_html}</div></div>', unsafe_allow_html=True)
        with col_right:
            st.markdown(f'<div style="display:flex;justify-content:flex-end;"><div class="score-badge">{int(score)}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def show_home():
    st.title("Are you a job seeker or recruiter?")
    st.write("")
    c1, c2, _ = st.columns([1,1,4])
    with c1:
        if st.button("I'm a job seeker", key="jobseeker", use_container_width=True):
            st.session_state.mode = "resume_to_jobs"
            st.session_state.page = "match"
    with c2:
        if st.button("I'm a recruiter", key="recruiter", use_container_width=True):
            st.session_state.mode = "job_to_resumes"
            st.session_state.page = "match"

def show_match_page():
    st.header("Job Seeker" if st.session_state.mode == "resume_to_jobs" else "Recruiter")
    st.write("Paste your text below and choose a model. Results show Top-K items with score on the right.")
    model_names = list(models.keys()) if models else ["Ridge","SVR","XGBoost","Ordinal"]
    model_choice = st.selectbox("Model", model_names)
    if st.session_state.mode == "resume_to_jobs":
        anchor_text = st.text_area("Enter your resume", height=260)
    else:
        anchor_text = st.text_area("Enter the job description", height=260)
    top_k = st.slider("Top K", 1, 50, 5)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Rank", key="rank"):
            if not anchor_text or not anchor_text.strip():
                st.warning("Please enter text before ranking.")
            else:
                importlib.reload(nb)
                if model_choice in models:
                    model = models[model_choice]
                else:
                    if models:
                        first_label = list(models.keys())[0]
                        model = models[first_label]
                        st.info(f"Model '{model_choice}' not found; using '{first_label}' instead.")
                    else:
                        st.error("No model files found. Put your .pkl models (best_*.pkl) and scaler.pkl in this folder.")
                        st.stop()
                job_df = pd.read_csv("demo_jobs_50.csv")
                resume_df = pd.read_csv("demo_resumes_50.csv")
                try:
                    with st.spinner("Preparing features and scoring..."):
                        results = nb.rank_using_model(
                            anchor_text=anchor_text,
                            mode=st.session_state.mode,
                            job_df=job_df,
                            resume_df=resume_df,
                            model=model,
                            scaler=scaler,
                            top_k=top_k
                        )
                    st.success("Ranking complete!")
                    text_col = 'job_description' if st.session_state.mode == "resume_to_jobs" else 'resume'
                    pretty_display(results, text_col=text_col, score_col='score', top_k=top_k)
                except Exception as e:
                    st.error("Error during ranking:")
                    st.exception(e)
    with col2:
        if st.button("Back", key="back"):
            st.session_state.page = "home"

if st.session_state.page == "home":
    show_home()
else:
    show_match_page()
