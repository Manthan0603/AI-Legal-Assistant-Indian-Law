# ⚖️ AI Legal Assistant – Indian Case Law

An intelligent AI-powered legal assistant designed to simplify and explain Indian case law using advanced NLP, semantic search, and LLM-based reasoning.

---

## 🚀 Overview

The **AI Legal Assistant** allows users to:

* Search legal cases using natural language
* Understand complex judgments in simple terms
* Explore similar cases
* Get structured legal explanations

This system combines **semantic search (FAISS)** with **LLM (Mistral via Ollama)** to deliver accurate and explainable legal insights.

---

## ✨ Key Features

### 🔍 Hybrid Legal Search Engine

* Exact case name matching
* Keyword-based classification
* Semantic search using FAISS

### 🧠 AI-Powered Explanation

* Converts complex legal judgments into:

  * What Happened
  * Background
  * Legal Issue
  * Court Decision
  * Reason
  * Final Outcome

### ⚖️ Case Classification System

* Automatically categorizes cases:

  * Criminal (Murder, Fraud, Theft, etc.)
  * Constitutional Law
  * Property & Civil Disputes
  * Corporate, Taxation, Labour Law

### 📊 Similar Case Retrieval

* Finds related cases based on:

  * Legal category
  * Context similarity

### 💬 Smart Chat UI

* Suggested queries
* Chat history
* Stop generation feature
* Confidence score visualization

### 📄 Case Brief Export

* Download structured legal summaries as PDF-ready HTML

---

## 🏗️ Tech Stack

### Frontend

* Streamlit (Interactive UI)

### Backend / AI

* Python
* FAISS (Vector Search)
* Sentence Transformers (`all-MiniLM-L6-v2`)
* Ollama (Mistral LLM)

### Data Processing

* Pandas
* NumPy

---

## 📦 Dependencies

```
streamlit
pandas
numpy
faiss-cpu
sentence-transformers
torch
scikit-learn
scipy
ollama
altair
```

---

## 📂 Project Structure

```
AI-Legal-Assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── faiss_index.bin         # Vector index (auto-generated)
├── IndicLegalQA Dataset/   # Dataset folder
│
├── notebooks/
│   └── Legal_Search.ipynb
│
└── assets/
    └── UI images / icons
```

---

## ⚙️ How It Works

### 1. Data Loading

* Loads legal dataset (IndicLegalQA)
* Cleans and normalizes case names

### 2. Case Classification

* Uses regex-based taxonomy for legal categories

### 3. Embedding Generation

* Converts questions into vectors using SentenceTransformer

### 4. FAISS Indexing

* Stores embeddings for fast similarity search

### 5. Hybrid Search Pipeline

* Case name match
* Crime/category match
* Semantic similarity (FAISS)

### 6. AI Response Generation

* Uses Mistral via Ollama
* Structured prompt engineering for clear output

---

## ▶️ How to Run Locally

### 1. Clone Repository

```
git clone https://github.com/Manthan0603/AI-Legal-Assistant-Indian-Law.git
cd AI-Legal-Assistant-Indian-Law
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Start Ollama (Important)

```
ollama run mistral
```

### 4. Run App

```
streamlit run app.py
```

---

## 📊 Dataset

* Dataset: **IndicLegalQA (10K cases)**
* Contains:

  * Case name
  * Question
  * Answer
  * Judgment details

---

## 🎯 Use Cases

* Law students (case understanding)
* Legal researchers
* Quick legal insights for general users
* AI-based legal analytics systems

---

## ⚠️ Limitations

* Depends on dataset quality
* Not a substitute for professional legal advice
* Requires local LLM setup (Ollama)

---

## 🔮 Future Enhancements

* Multi-language support (Hindi, Gujarati)
* Voice-based legal assistant
* Cloud deployment (Streamlit Cloud / AWS)
* Legal citation generator
* Real-time case updates

---

## 👨‍💻 Author

**Manthan Mangaroliya**
 AI/ML Enthusiast

---

## 📜 License

This project is for educational and research purposes.
