# TalentSync

A CV↔job matching system that predicts how well a résumé fits a job description, plus the LinkedIn job-market analysis that motivated its feature design.

Instead of scoring a résumé against a job with a single cosine similarity, TalentSync engineers **10 similarity and structural features** per résumé–job pair and feeds them to a **supervised regression model** trained on human-assigned relevance scores. The result is a 1–5 relevance score used to rank candidates or postings.

---

## What it does

The Streamlit app works in two directions:

| Role | Input | Output |
|---|---|---|
| **Job Seeker** | Paste your résumé | Top-K job postings ranked by fit |
| **Recruiter** | Paste a job description | Top-K candidate résumés ranked by fit |

You pick one of four trained models at runtime (Ridge, SVR, XGBoost, Ordinal), set Top-K with a slider (1–50), and get results with a 1–5 score badge on each row.

## How the matching works

```
résumé + job text
   ↓  clean_text            lowercase, strip punctuation, collapse whitespace
   ↓  tokenize              NLTK word_tokenize, drop English stopwords, alphabetic only
   ↓  lemmatize             WordNetLemmatizer
   ↓  feature extraction    10 features per pair (below)
   ↓  StandardScaler        scaler.pkl
   ↓  model.predict         Ridge | SVR | XGBoost | Ordinal
   ↓  clip + round          → integer score 1–5, sorted descending
```

### The 10 features

**Similarity (3)**
- `tfidf_cosine` — TF-IDF cosine similarity
- `jaccard` — Jaccard overlap on lemmatized token sets
- `bert_sim` — Sentence-BERT cosine using `all-MiniLM-L6-v2` (captures meaning where wording differs)

**Skill coverage (3)**
- `overlap_count` — shared tokens
- `percent_required_present` — fraction of the job's terms the résumé covers
- `missing_skill_count` — the gap

**Structural (4)**
- `resume_length`, `job_length`, `overlap_ratio`, `token_count_diff`

The first three answer *"how similar is this text?"*; the rest answer *"how much of what the job asks for is actually here?"* — two questions a lone similarity score conflates.

### The models

All four are trained in `ml_implementation.ipynb` with `GridSearchCV` + `KFold`, evaluated on MAE and a confusion matrix, and the best of each is pickled into `models/`:

| Model | Library | Why it's here |
|---|---|---|
| Ridge | scikit-learn | Linear baseline, regularized |
| SVR | scikit-learn | Non-linear kernel |
| XGBoost | xgboost | Gradient boosting, handles feature interactions |
| Ordinal | `mord.LogisticIT` | Treats 1–5 as **ordered** — 1 vs 5 is a bigger error than 4 vs 5 |

The Ordinal model is the most principled fit: `match_score` is an ordinal rating, not an unbounded continuous value.

---

## Quickstart

```bash
git clone https://github.com/harsh1243/TalentSync.git
cd TalentSync

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

No training required — the trained models are committed to `models/`.

**Two things to know:**

1. **Run from the repo root.** `app.py` loads `models/` and `dataset/` by relative path, so launching from elsewhere fails to find them.
2. **First run needs internet.** `rank_engine_full.py` calls `nltk.download()` for `punkt`, `stopwords`, and `wordnet` at import, and `sentence-transformers` fetches `all-MiniLM-L6-v2` (~90 MB) on first use. Both cache locally afterward.

---

## Repo structure

```
TalentSync/
├── app.py                        Streamlit UI — role selection, model picker, ranked results
├── rank_engine_full.py           Preprocessing + feature extraction + ranking (imported by app.py)
│
├── text_preprocessing.ipynb      Walkthrough: cleaning → tokenizing → lemmatizing
├── convert_features.ipynb        Walkthrough: deriving the 10 features
├── ml_implementation.ipynb       Training + tuning the 4 models, MAE comparison
│
├── dataset/
│   ├── resume_job_dataset.csv    Training data — job_description, resume, match_score
│   ├── demo_jobs_50.csv          50 postings the app ranks against
│   └── demo_resumes_50.csv       50 résumés the app ranks against
├── models/
│   ├── best_ridge.pkl  best_svr.pkl  best_xgb.pkl  best_ordinal.pkl
│   └── scaler.pkl                Fitted StandardScaler — required for correct predictions
│
└── market-analysis/              LinkedIn job-market analysis (see below)
```

The three notebooks mirror the pipeline in `rank_engine_full.py` step by step — read them in the order listed to follow how a raw text pair becomes a score.

---

## Market analysis

`market-analysis/` analyses **1.3M+ LinkedIn job postings** to understand what employers actually ask for. This is the exploratory groundwork behind the matcher's feature design: the skill-overlap features exist because the association-rule mining showed skills arrive in predictable bundles, not independently.

| Notebook | What it does |
|---|---|
| `data_clearning.ipynb` | Cleans and normalizes raw postings; NLTK stopwords + lemmatization |
| `eda.ipynb` | Top job titles, companies, locations, job types; posting-time and day-of-month patterns |
| `clustering.ipynb` | TF-IDF over a curated skill vocabulary → TruncatedSVD → K-Means, then centroid inspection to label each role cluster |
| `assosication_rules.ipynb` | FP-Growth via `mlxtend` to mine skill co-occurrence rules (which skills appear together) |
| `harsh.pbix` | Power BI dashboard over the analysis output |

**Dataset:** [1.3M LinkedIn Jobs and Skills 2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) by *asaniczka* (Kaggle). Not committed here — download it separately to re-run these notebooks.

> This folder was merged in from a separate repository with its full commit history preserved. Because those commits recorded paths at the repo root, `git log -- market-analysis/` shows only the merge commit; use `git log --graph` or `git log --follow market-analysis/<file>` to see the original history.

---

## Known gaps

- **`requirements.txt` covers the app, not the notebooks.** The `market-analysis/` notebooks additionally need `mlxtend`, `matplotlib`, `seaborn`, and `scipy`. `spacy` is listed but currently unused.
- **The app ranks against fixed 50-row demo files**, not a live posting index — it demonstrates the ranking model rather than serving as a production search backend.
- **Skill matching is token-overlap based.** "JS" and "JavaScript" are different tokens, so `bert_sim` carries most of the semantic load.
