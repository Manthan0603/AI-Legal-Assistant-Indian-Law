import streamlit as st
import pandas as pd
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_resource
def load_data():
    with open("IndicLegalQA Dataset/IndicLegalQA Dataset_10K.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------
# CREATE EMBEDDINGS + FAISS INDEX
# -------------------------------
@st.cache_resource
def create_faiss_index(df):

    questions = df["question"].tolist()

    embeddings = model.encode(questions)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings

index, question_embeddings = create_faiss_index(df)

# -------------------------------
# OLLAMA MISTRAL CALL
# -------------------------------
def ask_mistral(prompt):

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# -------------------------------
# AI SEARCH
# -------------------------------
def ai_search_top_cases(query, top_k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    return df.iloc[indices[0]]

# -------------------------------
# MAIN CHATBOT
# -------------------------------
def legal_chatbot(user_question):

    main_case = ai_search_top_cases(user_question, 1).iloc[0]
    top_cases = ai_search_top_cases(user_question, 3)

    context = f"""
    Case Name: {main_case['case_name']}
    Judgment Date: {main_case['judgment_date']}
    Case Summary: {main_case['answer']}
    """

    prompt = f"""
    You are an Indian legal assistant.

    Use the case below to answer clearly.

    {context}

    Question: {user_question}
    """

    answer = ask_mistral(prompt)

    similar_text = "\n\n📚 Similar Cases:\n"

    for i, row in enumerate(top_cases.itertuples(), 1):
        date_value = row.judgment_date if pd.notna(row.judgment_date) else "Date Not Available"
        similar_text += f"{i}. {row.case_name} ({date_value})\n"

    return answer + similar_text

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("⚖ AI Legal Assistant")

question = st.text_input("Ask Legal Question")

if st.button("Ask"):

    if question.strip() != "":
        with st.spinner("Thinking..."):
            result = legal_chatbot(question)

        st.write(result)
