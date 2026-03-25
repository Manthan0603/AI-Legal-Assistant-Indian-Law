# =================================================
# ⚖ AI LEGAL ASSISTANT — SINGLE FILE APP
# =================================================

# ---------------- IMPORTS ----------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import faiss
import ollama
from sentence_transformers import SentenceTransformer


# ================= CASE TYPE CLASSIFIER =================


def get_case_subtype(text):
    text = str(text).lower()

    crime_map = {
        "Murder": ["murder", "302 ipc", "killed", "homicide"],
        "Rape": ["rape", "376 ipc", "sexual assault"],
        "Theft": ["theft", "stolen", "379 ipc", "robbery", "snatching"],
        "Fraud": ["fraud", "cheating", "420 ipc", "scam"],
        "Assault": ["assault", "hurt", "323 ipc", "325 ipc", "attack"],
        "Kidnapping": ["kidnap", "abduction", "363 ipc", "364 ipc"],
        "Domestic Violence": ["domestic violence", "498a", "dowry harassment"],
        "Drugs": ["ndps", "drug", "narcotic"],
        "Tax": ["tax", "gst", "income tax", "evasion"],
        "Service": ["service matter", "promotion", "employment"],
        "Property": ["property dispute", "land dispute", "possession"],
    }

    for crime, keywords in crime_map.items():
        if any(word in text for word in keywords):
            return crime

    return "Other"


# ---------------- PAGE CONFIG ----------------


st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)


# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():
    import os
    import json
    import pandas as pd

    # Correct path (MATCH YOUR FOLDER + FILE NAME EXACTLY)
    dataset_path = r"IndicLegalQA Dataset/IndicLegalQA Dataset_10K.json"

    try:
        if not os.path.exists(dataset_path):
            st.error(f"Dataset not found at: {os.path.abspath(dataset_path)}")
            return pd.DataFrame()

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle nested JSON
        if isinstance(data, dict):
            key = list(data.keys())[0]
            df = pd.DataFrame(data[key])
        else:
            df = pd.DataFrame(data)

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


# Load once
df = load_data()
global index

# ================= CASE TYPE COLUMN =================
if "case_subtype" not in df.columns:
    combined_text = (
            df["case_name"].fillna("") + " " +
            df["question"].fillna("") + " " +
            df["answer"].fillna("")
    )

    df["case_subtype"] = combined_text.apply(get_case_subtype)

# ===== FIX DUPLICATE DATE COLUMN =====
if "judgement_date" in df.columns and "judgment_date" in df.columns:
    df["judgment_date"] = df["judgment_date"].fillna(df["judgement_date"])
    df.drop(columns=["judgement_date"], inplace=True)

if df.empty:
    st.stop()


# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ---------------- BUILD FAISS INDEX ----------------
# ---------------- BUILD / LOAD FAISS INDEX ----------------
import os

@st.cache_resource
def build_index(df):

    index_file = "faiss_index.bin"

    # ================= LOAD EXISTING INDEX =================
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        return index

    # ================= CREATE NEW INDEX =================
    questions = df["question"].astype(str).tolist()

    embeddings = model.encode(questions, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ================= SAVE INDEX =================
    faiss.write_index(index, index_file)

    return index

# =================================================
# 🔍 SEARCH FUNCTIONS
# =================================================

def ai_search_top_cases(user_question, top_k=3):
    user_embedding = model.encode([user_question])
    user_embedding = np.array(user_embedding).astype("float32")

    distances, indices = index.search(user_embedding, top_k)

    return df.iloc[indices[0]]


def hybrid_search_legal_answer(user_question):
    # Keyword search first
    keyword_match = df[
        df["question"].str.contains(user_question, case=False, na=False)
    ]

    if len(keyword_match) > 0:
        return keyword_match.iloc[0]

    # Else AI Search
    user_embedding = model.encode([user_question])
    user_embedding = np.array(user_embedding).astype("float32")

    distances, indices = index.search(user_embedding, 1)

    return df.iloc[indices[0][0]]


# =================================================
# 🤖 LLM CONNECTION
# =================================================

def ask_mistral(prompt):
    # ================= CLEAN LLM OUTPUT =================
    def clean_response(text):

        stop_phrases = [
            "Remember:",
            "Note:",
            "Rules:",
            "Code",
            "assistant",
            "AI system",
            "This response",
            "In conclusion"
        ]

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            if any(phrase.lower() in line.lower() for phrase in stop_phrases):
                break
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    # ================= PRIMARY (CPU - STABLE) =================
    try:
        response = ollama.chat(
            model="phi",   # CPU-first
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as cpu_error:
        print("CPU model failed, trying GPU...")

    # ================= SECONDARY (GPU - BEST EFFORT) =================
    try:
        response = ollama.chat(
            model="mistral",   # GPU attempt
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as gpu_error:
        print("GPU also failed.")

    # ================= FINAL FALLBACK =================
    return "⚠️ AI temporarily unavailable. Showing best matched case."

# =================================================
# ⚖ MAIN CHATBOT FUNCTION
# =================================================

def legal_chatbot(user_question):
    # =============================
    # STEP 1 — Get Case Result
    # =============================
    case_result = hybrid_search_legal_answer(user_question)

    if case_result is None:
        return "No relevant legal case found in dataset."

    # =============================
    # STEP 2 — Clean Missing Values
    # =============================
    case_name = case_result.get("case_name", "Unknown Case")

    judgment_date = case_result.get("judgment_date", "Not Available")
    if str(judgment_date) == "nan":
        judgment_date = "Date Not Available"

    case_answer = case_result.get("answer", "No summary available.")



    # =============================
    # STEP 3 — Build Context
    # =============================
    context = f"""
Case Name: {case_name}
Judgment Date: {judgment_date}
Case Summary: {case_answer}
"""

    # =============================
    # STEP 4 — STRONG PROMPT CONTROL
    # =============================
    prompt = f"""
You are an legal expert specializing in Indian case law and Indian Supreme Court Legal Case Assistant.

STRICT RULES:
- You MUST answer using ONLY the given case.
- You MUST NOT give general law definition.
- You MUST NOT use outside knowledge.
- If case info is limited → say "Based on available case data..."

Using the case information below, generate a structured legal explanation.
---------------------
LEGAL CASE DATA:
{context}
---------------------

USER QUESTION:
{user_question}

Provide the answer in the following structured format:

1. 🔍 Crime Summary:
- What crime was committed?

2. 📜 Facts of the Case:
- Key events and background.

3. ⚖️ Legal Issues:
- What legal questions were considered?

4. 🏛️ Court Judgment:
- What did the court decide?

5. 🧠 Reasoning:
- Why did the court decide this?

6. 📌 Final Outcome:
- Final verdict and punishment (if any).

IMPORTANT:
- Use simple language.
- Be clear and structured.
- Do NOT add unrelated information.
"""
    # =============================
    # STEP 5 — Ask LLM
    # =============================
    final_answer = ask_mistral(prompt)

    # ✅ CLEAN RESPONSE
    final_answer = clean_response(final_answer)
    
    # =============================
    # STEP 6 — Add Case Header (UI Improvement)
    # =============================
    formatted_answer = f"""
📌 Case: {case_name}  
📅 Date: {judgment_date}

🧾 Legal Explanation:
{final_answer}
"""

    return formatted_answer
index = build_index(df)


# =================================================
# 🎨 STREAMLIT UI
# =================================================

# ================= SIDEBAR =================

st.sidebar.title("⚖ Legal AI Pro")

page = st.sidebar.radio(
    "Navigation",
    ["Chat Assistant", "Analytics", "Dataset"]
)

# ================= NEW CHAT BUTTON =================

if st.sidebar.button("➕ New Chat"):
    # clear only current chat
    st.session_state.messages = []
    st.session_state.similar_cases = None

    st.rerun()

st.sidebar.divider()

# ================= SEARCH HISTORY =================

st.sidebar.subheader("🕘 Search History")

# Initialize history if not exists
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize similar cases
if "similar_cases" not in st.session_state:
    st.session_state.similar_cases = None

# Show last 5 searches
for i, item in enumerate(st.session_state.history[::-1][:5]):

    if st.sidebar.button(
        f"🔎 {item['query']}",
        key=f"history_{i}"
    ):
# Clear current chat
        st.session_state.messages = []

        # Load old conversation instead of searching again
        st.session_state.messages.append(("user", item["query"]))
        st.session_state.messages.append(("bot", item["response"], item["cases"]))

        st.rerun()

# =================================================
# 💬 CHAT PAGE
# =================================================
if page == "Chat Assistant":

    st.title("⚖️ AI Legal Chat Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.chat_input("Ask Indian Legal Question...")

    if query:

        # show user message
        st.session_state.messages.append(("user", query))

        # generate AI response
        response = legal_chatbot(query)
        top_cases = ai_search_top_cases(query, 3)

        # show bot response
        st.session_state.messages.append(("bot", response, top_cases))

        # save full chat to sidebar history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "query": query,
            "response": response,
            "cases": top_cases
        })
    # Show Chat
    for message_idx, message in enumerate(st.session_state.messages):

        if message[0] == "user":

            st.chat_message("user").write(message[1])

        else:

            st.chat_message("assistant").write(message[1])

            # show similar cases for THIS answer
            if len(message) > 2:

                cases = message[2]

                st.markdown("### 📂 Similar Cases")

                # ✅ Add unique key (IMPORTANT)
                view_mode = st.toggle("Show Detailed View", key=f"similar_toggle_main_{message_idx}", value=False)

                for _, row in cases.iterrows():

                    case_name = row.get("case_name", "Unknown Case")
                    judgment_date = row.get("judgment_date", None)
                    summary = row.get("answer", "")

                    if judgment_date is None or str(judgment_date) == "nan":
                        judgment_date = "Date Not Available"

                    if view_mode:
                        # ================= DETAILED VIEW =================
                        st.markdown(f"""
                        <div style="
                            background-color:#111827;
                            padding:15px;
                            border-radius:12px;
                            margin-bottom:12px;
                            border-left:4px solid #22c55e;
                        ">

                        <b>📌 {case_name}</b><br>
                        🗓 {judgment_date}<br><br>

                        📄 <b>Case Summary:</b><br>
                        {summary}
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        # ================= COMPACT VIEW =================
                        st.markdown(f"""
                        <div style="
                            background-color:#111827;
                            padding:12px;
                            border-radius:10px;
                            margin-bottom:10px;
                            border-left:4px solid #ef4444;
                        ">

                        <b>📌 {case_name}</b><br>
                        🗓 {judgment_date}

                        </div>
                        """, unsafe_allow_html=True)


# ================= ANALYTICS PAGE =================
elif page == "Analytics":

    st.title("📊 Dataset Crime Analytics")

    if df is None or df.empty:
        st.error("Dataset not loaded")
        st.stop()

    # ---------- UNIQUE CASE TABLE ----------
    unique_cases = (
        df[["case_name", "judgment_date", "case_subtype"]]
        .drop_duplicates(subset=["case_name"])
        .reset_index(drop=True)
    )

    # ---------- METRICS ----------
    col1, col2 = st.columns(2)
    # col1.metric("Total Rows", len(df))
    #  col2.metric("Unique Cases", len(unique_cases))

    st.divider()

    # ================= YEAR CHART =================
    st.subheader("📅 Unique Cases By Judgment Year")

    # Create Year Column
    unique_cases["Judgment Year"] = (
        unique_cases["judgment_date"]
        .astype(str)
        .str.extract(r'(\d{4})')[0]
        .fillna("Unknown")
    )

    # Count Unique Cases
    year_counts = unique_cases["Judgment Year"].value_counts().sort_index()

    # Convert To DataFrame + Rename Column
    year_df = year_counts.reset_index()
    year_df.columns = ["Judgment Year", "Case Counts"]

    # Show Chart
    st.bar_chart(year_df.set_index("Judgment Year"))

# ================= DATASET PAGE =================
elif page == "Dataset":

    st.title("📂 Dataset Explorer")

    # ================= UNIQUE CASE TABLE =================

    unique_cases = df[["case_name", "judgment_date"]].drop_duplicates()

    # Clean ordinal suffix (1st, 2nd, 3rd, 4th)
    unique_cases["clean_date"] = unique_cases["judgment_date"].str.replace(
        r'(\d+)(st|nd|rd|th)', r'\1', regex=True
    )

    # Convert to datetime for sorting
    unique_cases["sort_date"] = pd.to_datetime(
        unique_cases["clean_date"],
        errors="coerce"
    )

    # Extract year and month for filters
    unique_cases["Year"] = unique_cases["sort_date"].dt.year
    unique_cases["Month"] = unique_cases["sort_date"].dt.month_name()

    # ================= FILTER UI =================

    st.subheader("📅 Filter Cases")

    col1, col2 = st.columns(2)

    selected_year = col1.selectbox(
        "Select Year",
        ["All"] + sorted(unique_cases["Year"].dropna().unique().tolist())
    )

    selected_month = col2.selectbox(
        "Select Month",
        ["All"] + sorted(unique_cases["Month"].dropna().unique().tolist())
    )

    # ================= APPLY FILTER =================

    filtered_cases = unique_cases.copy()

    if selected_year != "All":
        filtered_cases = filtered_cases[filtered_cases["Year"] == selected_year]

    if selected_month != "All":
        filtered_cases = filtered_cases[filtered_cases["Month"] == selected_month]

    # ================= SORT CASES =================

    filtered_cases = filtered_cases.sort_values(by="sort_date")

    # ================= REMOVE HELPER COLUMNS =================

    display_cases = filtered_cases.drop(
        columns=["clean_date", "sort_date", "Year", "Month"]
    )

    # ================= RESET INDEX =================

    display_cases = display_cases.reset_index(drop=True)

    # ================= RENAME COLUMNS =================

    display_cases.rename(columns={
        "case_name": "Case Name",
        "judgment_date": "Judgment Date"
    }, inplace=True)

    # ================= CREATE SERIAL NUMBER =================

    display_cases.index += 1
    display_cases.index.name = "Sr No"

    # ================= DISPLAY TABLE =================

    st.subheader("📋 Filtered Cases")

    st.dataframe(
        display_cases,
        use_container_width=True,
        height=450
    )

    # ================= SEARCH + SELECT =================
    st.write(f"Showing {len(display_cases)} cases")
    st.subheader("🔎 Search & Open Case")

    # Reset index for serial numbering
    display_df = display_cases.reset_index(drop=True)
    display_df.index += 1
    display_df.index.name = "Sr No"

    # Create display label
    display_df["Display Name"] = (
            display_df.index.astype(str) + ". " + display_df["Case Name"]
    )

    # Search box
    search_query = st.text_input("Search Case Name")

    filtered_df = display_df

    if search_query:
        filtered_df = display_df[
            display_df["Case Name"]
            .str.contains(search_query, case=False, na=False)
        ]

    # Selectbox with Sr No + Case Name
    selected_display = st.selectbox(
        "Select Case",
        filtered_df["Display Name"]
    )

    if selected_display:

        # Extract actual case name
        selected_case = selected_display.split(". ", 1)[1]

        case_data = df[df["case_name"] == selected_case]

        st.markdown("### ⚖️ Case Details")

        st.write("**Case Name:**", selected_case)
        st.write("**Judgment Date:**", case_data.iloc[0]["judgment_date"])

        st.markdown("---")

        for _, row in case_data.iterrows():
            st.markdown(f"**Q:** {row['question']}")
            st.markdown(f"**A:** {row['answer']}")
            st.markdown("---")
    # # ================= CASE DETAILS =================
    # case_data = df[df["case_name"] == selected_case]
    #
    # st.divider()
    # st.subheader("📄 Case Details")
    #
    # st.write("### Case Name")
    # st.info(selected_case)
    #
    # st.write("### Judgment Date")
    # st.info(case_data.iloc[0]["judgment_date"])
    #
    # st.divider()
    #
    # # ================= Q&A =================
    # st.subheader("❓ Questions & Answers")
    #
    # for _, row in case_data.iterrows():
    #     st.markdown(f"**Q:** {row['question']}")
    #     st.markdown(f"**A:** {row['answer']}")
    #     st.markdown("---")
