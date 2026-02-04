⚖️ AI Legal Assistant for Indian Case Law

📌 Project Overview
AI Legal Assistant is an intelligent legal chatbot designed to answer Indian legal case-related questions using AI-based semantic search and Large Language Models (LLMs).
The system retrieves relevant legal cases from the IndicLegalQA dataset and generates easy-to-understand legal explanations for users.


🚀 Features

🔍 Semantic Legal Case Search using FAISS + Sentence Transformers
🤖 AI Explanation Generation using Mistral (via Ollama)
📊 Legal Case Similarity Recommendation
🌐 Streamlit Web Interface
🇮🇳 Indian Legal Dataset Support


🧠 Technologies Used
Category	           Technology
Programming	         Python
Frontend	           Streamlit
AI Embeddings	       Sentence Transformers
Vector Search	       FAISS
LLM	                 Ollama (Mistral Model)
Data Processing	     Pandas, NumPy
ML Utilities	       Scikit-Learn


📂 Project Structure
AI-Legal-Assistant-Indian-Law/
│
├── IndicLegalQA Dataset/
├── Stage1_Legal_Search.ipynb
├── app.py
├── requirements.txt
└── README.md


📊 Dataset
IndicLegalQA Dataset
Indian Legal Question Answer Dataset used for training and retrieval.


⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/AI-Legal-Assistant-Indian-Law.git
cd AI-Legal-Assistant-Indian-Law

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install Ollama (If Not Installed)
Download from:
👉 https://ollama.com

Run model:
ollama run mistral

4️⃣ Run Streamlit App
streamlit run app.py


💻 Application Usage
Enter legal query
System retrieves similar legal case
AI generates simplified explanation
Similar cases are suggested


🖥️ Sample Queries
Corruption cases in India
Army promotion dispute Supreme Court case
BCCI related legal cases
CBI investigation cases


🔮 Future Improvements
Chat History Support
Case PDF Linking
Multi-language Support (Hindi / Gujarati)
Online Deployment
Voice Input Support


👨‍💻 Author
Manthan Mangaroliya
B.Tech Information Technology


📜 License
This project is for educational and research purposes.
