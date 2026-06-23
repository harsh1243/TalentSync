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
    background-color: #e6f7ff !important;
    color: #014f86 !important;
}

/* ---------- TEXT COLORS ---------- */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #014f86 !important;
}

/* ---------- TEXT AREA + INPUT ---------- */
textarea, input, .stTextArea textarea, .stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #0a74d2 !important;
    border-radius: 8px !important;
    outline: none !important;
    box-shadow: none !important;
}

textarea:focus, input:focus, .stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #0a74d2 !important;
    box-shadow: 0 0 6px rgba(10, 116, 210, 0.3) !important;
}

/* ---------- BUTTONS ---------- */
div.stButton > button {
    background-color: #065da6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease-in-out;
}

div.stButton > button span,
div.stButton > button p,
div.stButton > button div {
    color: #ffffff !important;
}

div.stButton > button:hover {
    background-color: #0a74d2 !important;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)


def pretty_display(results):
    """Display match results in a formatted way."""
    if results is None or len(results) == 0:
        st.warning("No results to display.")
        return

    for i, row in enumerate(results):
        with st.expander(f"Match #{i+1}: {row.get('title', 'N/A')} — Score: {row.get('score', 0):.2f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Title:** {escape(str(row.get('title', 'N/A')))}")
                st.markdown(f"**Company:** {escape(str(row.get('company', 'N/A')))}")
                st.markdown(f"**Location:** {escape(str(row.get('location', 'N/A')))}")
            with col2:
                st.markdown(f"**Score:** {row.get('score', 0):.4f}")
                st.markdown(f"**Category:** {escape(str(row.get('category', 'N/A')))}")
            st.markdown("**Description:**")
            st.write(row.get('description', 'N/A'))


def _fetch_match_data(resume_text, job_category, top_n):
    """Fetch and return match results for the given resume and filters."""
    try:
        results = nb.rank(resume_text, category=job_category, top_n=top_n)
        return results, None
    except Exception as e:
        return None, str(e)


def _render_match_header():
    """Render the header section of the match page."""
    st.title("Job ↔ Resume Matcher")
    st.markdown("Paste your resume below and find the best matching job postings.")


def _render_resume_input():
    """Render the resume text input area and return the entered text."""
    return st.text_area(
        "Paste your resume here:",
        height=300,
        placeholder="Enter your resume text..."
    )


def _render_filter_controls(categories):
    """Render filter controls and return selected values."""
    col1, col2 = st.columns(2)
    with col1:
        job_category = st.selectbox("Filter by category (optional):", ["All"] + categories)
    with col2:
        top_n = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    return job_category if job_category != "All" else None, top_n


def _render_match_results(resume_text, job_category, top_n):
    """Fetch and render match results."""
    if not resume_text.strip():
        st.warning("Please enter your resume text before searching.")
        return

    with st.spinner("Finding best matches..."):
        results, error = _fetch_match_data(resume_text, job_category, top_n)

    if error:
        st.error(f"An error occurred: {error}")
    elif results is not None:
        st.success(f"Found {len(results)} match(es)!")
        pretty_display(results)
    else:
        st.warning("No matches found.")


def show_match_page():
    """Orchestrate the match page rendering."""
    _render_match_header()

    try:
        categories = nb.get_categories()
    except Exception:
        categories = []

    resume_text = _render_resume_input()
    job_category, top_n = _render_filter_controls(categories)

    if st.button("Find Matches"):
        _render_match_results(resume_text, job_category, top_n)


def main():
    show_match_page()


if __name__ == "__main__":
    main()