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
    .result-text { font-size:16px; color: #014f86; } /* Changed color for better visibility on light bg */
    .divider { border-bottom:1px solid rgba(10, 116, 210, 0.2); margin:8px 0; } /* Lightened blue divider */
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

    st.markdown("<h1 style='text-align: center; color: #014f86;'>TalentSync</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #014f86; margin-bottom: 30px;'>Please select your role to begin</h3>", unsafe_allow_html=True)


    card_style = """
    <style>
    /* Target the native st.container() within the columns */
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
        background-color: #ffffff;
        border: 2px solid #0a74d2; /* medium blue border */
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 300px; /* Set a consistent height */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Pushes button to bottom */
    }
    
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(10, 116, 210, 0.2); /* blue shadow */
    }

    /* Corrected selectors for p and button */
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] p {
        font-size: 1rem;
        color: #014f86 !important; /* Force color */
        flex-grow: 1; /* Allows text to take up space */
        padding-top: 1rem;
        text-align: center; /* Ensure text is centered */
    }

    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] .stButton button {
        width: 100%; /* Make button fill card width */
        font-size: 1.1rem;
        padding: 0.75rem 1rem !important;
        /* The global fix above should handle color, but we can be specific */
    }
    
    /* --- THIS IS THE HYPER-SPECIFIC FIX --- 
    It targets *only* text elements inside the home page buttons
    */
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] .stButton button span,
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] .stButton button p,
    [data-testid="stHorizontalBlock"] > div > [data-testid="stVerticalBlock"] > [data-testid="stContainer"] .stButton button div {
        color: #ffffff !important;
    }
    /* --- END NEW FIX --- */
    
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
 
        with st.container():
           
           
            st.header("👨‍💼 I am a Job Seeker")
            st.markdown("<p>Find the perfect job by matching your resume to thousands of listings.</p>", unsafe_allow_html=True)
            
            # The button's click handler has the original logic
            if st.button("Find a Job", key="jobseeker", use_container_width=True):
                st.session_state.mode = "resume_to_jobs"
                st.session_state.page = "match"
                st.rerun() # Use rerun for a cleaner page transition
            
          

    with col2:
       
        with st.container():
            
            
           
            st.header("📈 I am a Recruiter")
            st.markdown("<p>Discover top talent by matching your job description to qualified candidates.</p>", unsafe_allow_html=True)
            
            # The button's click handler has the original logic
            if st.button("Find Talent", key="recruiter", use_container_width=True):
                st.session_state.mode = "job_to_resumes"
                st.session_state.page = "match"
                st.rerun() # Use rerun for a cleaner page transition
            
          

def show_match_page():
    st.header("Job Seeker" if st.session_state.mode == "resume_to_jobs" else "Recruiter")
    st.write("Paste your text below and choose a model. Results show Top-K items with score on the right.")
    model_names = ["Ridge", "SVR", "XGBoost", "Ordinal"]

  
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = None

    cols = st.columns(len(model_names), gap="medium")

    for i, name in enumerate(model_names):
      
        label = f" {name}" if st.session_state.model_choice == name else name 

        if cols[i].button(label, use_container_width=True):
            st.session_state.model_choice = name

   
    model_choice = st.session_state.model_choice

    
    if model_choice:
        st.success(f"Selected model: {model_choice}")
    else:
       
        st.info("Please select a model from the options above.")

    if st.session_state.mode == "resume_to_jobs":
        anchor_text = st.text_area("Enter your resume", height=260, placeholder="Paste your full resume text here...")
    else:
        anchor_text = st.text_area("Enter the job description", height=260, placeholder="Paste the full job description text here...")
    top_k = st.slider("Top K", 1, 50, 5)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Rank", key="rank"):
            if not anchor_text or not anchor_text.strip():
                st.warning("Please enter text before ranking.")
            elif not model_choice:
                st.warning("Please select a model before ranking.") # Added check
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
                
               
                try:
                    job_df = pd.read_csv("demo_jobs_50.csv")
                    resume_df = pd.read_csv("demo_resumes_50.csv")
                except FileNotFoundError as e:
                    st.error(f"Demo file not found: {e.filename}. Please make sure 'demo_jobs_50.csv' and 'demo_resumes_50.csv' are in the same folder.")
                    st.stop()
                
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
            st.rerun() # Added rerun for consistency

# Main app router
if st.session_state.page == "home":
    show_home()
else:
    show_match_page()
