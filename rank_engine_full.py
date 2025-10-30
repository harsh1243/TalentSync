import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import pandas as pd
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sentence_transformers import SentenceTransformer, util

nltk.download("punkt")
nltk.download("punkt_tab")   
nltk.download("stopwords")
nltk.download("wordnet")
def clean_text(df):
    #extracting job descriptions and resumes from dataframe
    job_desc=df["job_description"]
    resume=df["resume"]
    
    #converting all text to lowercase 
    job_desc_lower=[t.lower() for t in job_desc]
    resume_lower=[t.lower() for t in resume]

    #removing punctuation by replacing it with spaces
    pattern = r'[{}]'.format(re.escape(string.punctuation))
    job_desc_clean = [re.sub(pattern, ' ', t) for t in job_desc_lower]
    resume_clean   = [re.sub(pattern, ' ', t) for t in resume_lower]

    #replacing  multiple spaces/tabs/newlines with a single space and trim ends
    pattern=r'\s+'
    job_desc_clean = [re.sub(pattern, ' ', t).strip() for t in job_desc_clean]
    resume_clean   = [re.sub(pattern, ' ', t).strip() for t in resume_clean]

    df["job_description"]=job_desc_clean
    df["resume"]=resume_clean
    return df 


def tokenize_and_remove_stopwords(df):

    job_desc=df["job_description"]
    resume=df["resume"]
  
    tokens_job_desc = [word_tokenize(t) for t in job_desc]
    tokens_resume   = [word_tokenize(t) for t in resume]

    # load English stopwords
    stop_words = set(stopwords.words("english"))

    # filter out stopwords + keep only alphabetic tokens
    filtered_job_desc = [[w for w in tokens if w.isalpha() and w not in stop_words]
                         for tokens in tokens_job_desc]
    filtered_resume   = [[w for w in tokens if w.isalpha() and w not in stop_words]
                         for tokens in tokens_resume]

    return filtered_job_desc, filtered_resume

def lemmatize_tokens(job_tokens, resume_tokens):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
   
    lemmatized_job_desc = [[lemmatizer.lemmatize(w) for w in tokens]
                           for tokens in job_tokens]
    lemmatized_resume   = [[lemmatizer.lemmatize(w) for w in tokens]
                           for tokens in resume_tokens]
    
    return lemmatized_job_desc, lemmatized_resume


def compute_tfidf_cosine(resume_texts, job_texts):
    vectorizer = TfidfVectorizer()
    combined   = resume_texts + job_texts
    tfidf_matrix = vectorizer.fit_transform(combined)
    n = len(resume_texts)

    resume_tfidf = tfidf_matrix[:n]
    job_tfidf    = tfidf_matrix[n:]

    tfidf_cosine = [
        float(cosine_similarity(resume_tfidf[i], job_tfidf[i])[0][0])
        for i in range(n)
    ]

    return tfidf_cosine


def compute_jaccard(resume_sets, job_sets):
    def jaccard_sim(a, b):
        return len(a & b) / len(a | b) if len(a | b) > 0 else 0

    jaccard = [
        jaccard_sim(a, b)
        for a, b in zip(resume_sets, job_sets)
    ]

    return jaccard


def compute_sentencebert_cosine(resume_texts, job_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    emb_r = model.encode(resume_texts, convert_to_tensor=True)
    emb_l = model.encode(job_texts, convert_to_tensor=True)

    pairwise_similarities = [
        util.cos_sim(emb_r[i], emb_l[i]).item()
        for i in range(min(len(resume_texts), len(job_texts)))
    ]

    return pairwise_similarities


def compute_skill_features(l3, l4):
    def skill_features(resume_tokens, job_tokens):
        rset = set(resume_tokens or [])
        jset = set(job_tokens or [])

        overlap = rset & jset
        overlap_count = len(overlap)

        job_count = len(jset)
        percent_required_present = overlap_count / job_count if job_count > 0 else 0.0

        missing_skill_count = job_count - overlap_count

        return overlap_count, percent_required_present, missing_skill_count

    return [skill_features(r, j) for r, j in zip(l3, l4)]


def compute_structural_features(l3, l4):
    resume_length    = [len(r) for r in l3]
    job_length       = [len(j) for j in l4]
    overlap_ratio    = [
        len(set(r) & set(j)) / len(set(r) | set(j)) if len(set(r) | set(j)) > 0 else 0
        for r, j in zip(l3, l4)
    ]
    token_count_diff = [abs(len(r) - len(j)) for r, j in zip(l3, l4)]
    return resume_length, job_length, overlap_ratio, token_count_diff


def preprocessing(df_pairs):
    df_clean = clean_text(df_pairs.copy())
    job_tokens, resume_tokens = tokenize_and_remove_stopwords(df_clean)
    lemmatized_job, lemmatized_resume = lemmatize_tokens(job_tokens, resume_tokens)
    return lemmatized_resume, lemmatized_job


def create_pair_dataframe(anchor_text, mode, job_df, resume_df):
  
    if mode == "job_to_resumes":
        # find resume column
        if "resume_text" in resume_df.columns:
            other_texts = resume_df["resume_text"].tolist()
        elif "resume" in resume_df.columns:
            other_texts = resume_df["resume"].tolist()
        else:
            raise KeyError("resume_df must contain 'resume_text' or 'resume' column.")
        df_pairs = pd.DataFrame({
            "job_description": [anchor_text] * len(other_texts),
            "resume": other_texts
        })
    else: 
       
        if "job_text" in job_df.columns:
            other_texts = job_df["job_text"].tolist()
        elif "job_description" in job_df.columns:
            other_texts = job_df["job_description"].tolist()
        else:
            raise KeyError("job_df must contain 'job_text' or 'job_description' column.")
        df_pairs = pd.DataFrame({
            "job_description": other_texts,
            "resume": [anchor_text] * len(other_texts)
        })

    return df_pairs.reset_index(drop=True)



def create_features_dataframe(resume_textss,job_textss,resume_text_f,job_text_f) :
    features_df=pd.DataFrame()
    
    features_df["tfidf_cosine"] = compute_tfidf_cosine(resume_textss, job_textss)
    features_df["jaccard"] = compute_jaccard([set(r) for r in resume_text_f], [set(j) for j in job_text_f])
    features_df["bert_sim"] = compute_sentencebert_cosine(resume_textss, job_textss)


    overlap_count, percent_required_present, missing_skill_count = zip(*compute_skill_features(resume_text_f,job_text_f))
    features_df["overlap_count"]= overlap_count
    features_df["percent_required_present"]= percent_required_present
    features_df["missing_skill_count"]= missing_skill_count

    resume_length, job_length, overlap_ratio, token_count_diff = compute_structural_features(resume_text_f,job_text_f)
    features_df["resume_length"]= resume_length
    features_df["job_length"]= job_length
    features_df["overlap_ratio"]= overlap_ratio
    features_df["token_count_diff"]=token_count_diff
    return features_df
    



def rank_using_model(anchor_text, mode, job_df, resume_df, model, scaler=None, top_k=5):
    import re
    
    df_final = create_pair_dataframe(anchor_text, mode, job_df, resume_df)
    resume_text_f, job_text_f = preprocessing(df_final)
    resume_textss = [" ".join(r) for r in resume_text_f]
    job_textss    = [" ".join(j) for j in job_text_f]
    features_df = create_features_dataframe(resume_textss, job_textss, resume_text_f, job_text_f)

    if scaler is not None:
        X = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
    else:
        X = features_df.copy()

    raw = model.predict(X)
    int_scores = np.clip(np.round(raw), 1, 5).astype(int)
    df_final["score"] = int_scores
    return df_final.sort_values("score", ascending=False).reset_index(drop=True)
