import pandas as pd
import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sqlite3
import matplotlib.pyplot as plt
import re

# NLTK resources (include both tagger names to avoid env mismatch)
nltk.download('punkt', quiet=True)
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except:
    pass
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Say your query…")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except Exception:
            return ""

def perform_nlp_tasks(text):
    tokens = word_tokenize(text) if text else []
    tags = pos_tag(tokens) if tokens else []
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text) if text else {"neg":0,"neu":1,"pos":0,"compound":0}
    return tokens, tags, sentiment

# --- Robust, priority-based extractor ---
def extract_search_params(text):
    s = text.strip()
    s_l = s.lower()
    sp = {}

    # 1) Department queries (explicit "department" keyword)
    if "department" in s_l or "dept" in s_l:
        m = re.search(r'\b(?:dept|department)\s*(?:of|in|for|:)?\s*([A-Za-z &/+\-]+)\b', s, re.I)
        if not m:
            m = re.search(r'\bdetails?\s+(?:of|in|from|for|at)\s+([A-Za-z &/+\-]+)\s+department\b', s, re.I)
        if m:
            dept = re.sub(r'\bdepartment\b', '', m.group(1), flags=re.I).strip()
            if dept:
                sp["department"] = dept
                return sp  # priority satisfied

    # 2) Job title queries (role/title/position keywords)
    if any(k in s_l for k in [" job title", "title", "position", "role", "as a ", "as an ", "with role", "with title"]):
        mt = re.search(r'\b(?:job\s*title|title|position|role)\s*(?:of|is|=|:)?\s*([A-Za-z /&\-]+)\b', s, re.I)
        if not mt:
            mt = re.search(r'\b(?:as|for)\s+an?\s+([A-Za-z /&\-]+)\b', s, re.I)
        if not mt:
            mt = re.search(r'\bdetails?\s+(?:of|for)\s+([A-Za-z /&\-]+)\b(?:\s+(?:role|position|title))?\b', s, re.I)
        if mt:
            jt = mt.group(1).strip()
            # Avoid picking up words like 'details' as title
            if jt and jt.lower() not in {"details", "information"}:
                sp["job_title"] = jt
                return sp  # priority satisfied

    # 3) Otherwise treat as a person-name query: “…details of Robert Patel”
    m = re.search(r'(?:details?|info(?:rmation)?|profile|records?)\s+of\s+([A-Za-z][A-Za-z\'\-]*(?:\s+[A-Za-z][A-Za-z\'\-]*){0,3})', s, re.I)
    if m:
        sp["employee_name"] = m.group(1).strip()
        return sp

    # 3b) Fallback: longest proper-noun span
    tokens = word_tokenize(s)
    tags = pos_tag(tokens)
    best, cur = [], []
    for w, t in tags:
        if t in {"NNP", "NNPS"}:
            cur.append(w)
        else:
            if len(cur) > len(best): best = cur
            cur = []
    if len(cur) > len(best): best = cur
    name = " ".join(best).strip()
    if name:
        sp["employee_name"] = name

    return sp

def search_employees(db_path, sp):
    if not any(sp.get(k) for k in ("employee_name","department","job_title")):
        st.warning("Please specify a name, department, or job title.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    q = "SELECT * FROM employees WHERE 1=1"
    p = []
    if sp.get("employee_name"):
        q += " AND [Full Name] LIKE ? COLLATE NOCASE"
        p.append(f"%{sp['employee_name']}%")
    if sp.get("department"):
        q += " AND Department LIKE ? COLLATE NOCASE"
        p.append(f"%{sp['department']}%")
    if sp.get("job_title"):
        q += " AND [Job Title] LIKE ? COLLATE NOCASE"
        p.append(f"%{sp['job_title']}%")
    df = pd.read_sql(q, conn, params=p)
    conn.close()
    return df

def plot_gender_distribution_pie(df):
    if df.empty or 'Gender' not in df.columns:
        st.info("No gender data to plot.")
        return
    counts = df['Gender'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def plot_employee_distribution_bar(df, category):
    if df.empty or category not in df.columns:
        st.info(f"No {category} data to plot.")
        return
    counts = df[category].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_title(f'Employee Distribution by {category}')
    ax.set_xlabel(category)
    ax.set_ylabel('Count')
    st.pyplot(fig)

st.title("Employee Data Search")
st.header("Search")
option = st.radio("Select search method:", ("Speech Recognition","Manual Input"))
db_path = '/Users/rutvijjadhav/Downloads/employee_data.db'

if option == "Speech Recognition":
    if st.button("Start Recording"):
        query_text = speech_to_text()
        st.write("You said:"); st.success(query_text if query_text else "(empty)")
        tokens, tags, sentiment = perform_nlp_tasks(query_text)
        sp = extract_search_params(query_text)
        st.caption(f"Extracted filters → {sp}")  # helpful debug
        df = search_employees(db_path, sp)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### NLP Tasks Results")
            st.write("Tokens:", tokens)
            st.write("POS Tags:", tags)
            st.write("Sentiment Analysis:", sentiment)
        st.divider()
        with col2:
            if not df.empty:
                st.write("### Search Results")
                st.write(df)
                st.write("### Gender Distribution (Pie Chart)")
                plot_gender_distribution_pie(df)
                st.write("### Employee Distribution by Department (Bar Chart)")
                plot_employee_distribution_bar(df, 'Department')
                st.write("### Employee Distribution by Job Title (Bar Chart)")
                plot_employee_distribution_bar(df, 'Job Title')
            else:
                st.warning("No matching results.")
else:
    sp = {}
    sp['employee_name'] = st.text_input("Employee Name:")
    sp['department']   = st.text_input("Department:")
    sp['job_title']    = st.text_input("Job Title:")
    if st.button("Search"):
        st.caption(f"Extracted filters → { {k:v for k,v in sp.items() if v} }")
        tokens, tags, sentiment = ([], [], {"neg":0,"neu":1,"pos":0,"compound":0})
        df = search_employees(db_path, sp)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### NLP Tasks Results")
            st.write("Tokens: N/A (Manual)")
            st.write("POS Tags: N/A (Manual)")
            st.write("Sentiment Analysis:", sentiment)
        st.divider()
        with col2:
            if not df.empty:
                st.write("### Search Results")
                st.write(df)
                st.write("### Gender Distribution (Pie Chart)")
                plot_gender_distribution_pie(df)
                st.write("### Employee Distribution by Department (Bar Chart)")
                plot_employee_distribution_bar(df, 'Department')
                st.write("### Employee Distribution by Job Title (Bar Chart)")
                plot_employee_distribution_bar(df, 'Job Title')
            else:
                st.warning("No matching results.")
