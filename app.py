# =================================================
# ⚖ AI LEGAL ASSISTANT — SINGLE FILE APP
# =================================================


# =================================================
# IMPORT REQUIRED LIBRARIES
# =================================================
# Streamlit → UI framework
# Pandas/Numpy → Data handling
# Matplotlib → Visualization
# FAISS → Vector similarity search
# Groq → LLM interaction
# SentenceTransformer → Text embeddings


import streamlit as st
import pandas as pd
import numpy as np

import json
import faiss
import time
from groq import Groq

import re
import altair as alt
from sentence_transformers import SentenceTransformer
import datetime

# =================================================
# USAGE TRACKING & SESSION STATE
# =================================================
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
    st.session_state.date = datetime.date.today()

# Reset daily
if st.session_state.date != datetime.date.today():
    st.session_state.request_count = 0
    st.session_state.date = datetime.date.today()

# Daily limit
DAILY_LIMIT = 10

# Initialize Groq Client
# NOTE: Your API key is stored in .streamlit/secrets.toml
# Do NOT place the actual 'gsk_...' key as the index in st.secrets[...]
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =================================================
# FUNCTION: get_case_subtype
# PURPOSE:
# Classifies a legal case into a predefined category
# based on keyword matching from text.
# INPUT: raw text (case name + question + answer)
# OUTPUT: case category (e.g., Murder, Fraud, etc.)
# =================================================


def get_case_subtype(text, case_name=""):
    text = str(text).lower()
    case_name = str(case_name).lower()
    
    # Combined text for classification
    full_text = case_name + " " + text

    # Priority-based taxonomy using regex patterns
    taxonomy = {
        "Murder/Homicide": [r"murder", r"302 ipc", r"homicide", r"killing", r"life imprisonment", r"culpable homicide", r"conviction under section 302", r"kill"],
        "Sexual Offenses/Rape": [r"rape", r"376 ipc", r"sexual assault", r"pocso", r"posco", r"modesty of woman", r"sexual intercourse"],
        "Fraud/Cheating": [r"fraud", r"cheating", r"420 ipc", r"scam", r"forgery", r"misrepresentation", r"falsification", r"pecuniary", r"deceit"],
        "Theft/Robbery": [r"theft", r"robbery", r"379 ipc", r"snatching", r"stolen", r"burglary", r"dacoity", r"loot"],
        "Assault/Violence": [r"assault", r"323 ipc", r"325 ipc", r"attack", r"hurt", r"physical violence", r"grievous hurt", r"weapon", r"injury"],
        "Kidnapping": [r"kidnap", r"abduction", r"363 ipc", r"364 ipc", r"hostage"],
        "Domestic Violence/Dowry": [r"domestic violence", r"498a", r"dowry", r"harassment", r"cruelty by husband", r"marital dispute", r"matrimonial home", r"husband and wife"],
        "Drugs/NDPS": [r"ndps", r"narcotic", r"drugs", r"contraband", r"possession of drugs", r"psychotropic", r"heroin", r"ganja", r"opium"],
        "Constitutional - Public Interest (PIL)": [r"public interest litigation", r"pil", r"social justice", r"pro bono publico", r"human rights", r"environmental protection under article"],
        "Constitutional - Affirmative Action": [r"scheduled tribe", r"scheduled caste", r"reservation", r"st/sc", r"backward class", r"mandal commission", r"article 15", r"article 16"],
        "Constitutional - Writs & Remedies": [r"writ of mandamus", r"certiorari", r"quo warranto", r"prohibition", r"article 226", r"article 32"],
        "Constitutional - Fundamental Rights": [r"fundamental right", r"article 14", r"article 19", r"article 21", r"habeas corpus", r"right to life", r"right to equality", r"freedom of speech", r"discrimination", r"untouchability"],
        "Constitutional - Administrative": [r"ultra vires", r"governance", r"legislature", r"parliament", r"doctrine of pith and substance", r"colorable legislation", r"separation of powers", r"rule of law", r"state action"],
        "Constitutional - General": [r"constitution", r"basic structure", r"amendment of constitution", r"article \d+"],
        "Company/Corporate Law": [r"company", r"corporate", r"companies act", r"sebi", r"shareholder", r"board of directors", r"winding up", r"merger"],
        "Banking/Finance": [r"banking", r"loan", r"mortgage", r"guarantee", r"cheque", r"ni act", r"138 ni act", r"debt recovery", r"insolvency", r"bankruptcy", r"drp", r"npa", r"financial", r"banker", r"recovery of dues"],
        "Taxation": [r"income tax", r"excise", r"customs", r"gst", r"vat", r"tax evasion", r"levy of duty", r"tariff", r"it act", r"direct tax", r"commodity", r"sales tax", r"revenue", r"commissioner of income tax"],
        "Property/Land": [r"property dispute", r"land dispute", r"possession", r"tenancy", r"eviction", r"partition", r"encroachment", r"title deed", r"sale deed", r"specific performance", r"land acquisition", r"easement", r"building", r"premises", r"rent", r"lease", r"tenant", r"landlord", r"transfer of property", r"rent control"],
        "Arbitration": [r"arbitration", r"arbitrator", r"arbitral", r"award", r"section 34", r"conciliation", r"arbitration act", r"arbitrament"],
        "Intellectual Property": [r"trademark", r"copyright", r"patent", r"infringement", r"passing off", r"ipr", r"patent act"],
        "Environmental": [r"environment", r"pollution", r"green tribunal", r"ngt", r"forest", r"ecology", r"wildlife", r"pollution control board"],
        "Education/Medical": [r"university", r"college", r"examination", r"upsc", r"mci", r"medical council", r"admission", r"student", r"degree", r"affiliation", r"medical college", r"educational", r"school", r"teacher", r"academic"],
        "Motor Vehicle/MACT": [r"motor vehicle", r"mact", r"accident claim", r"rash and negligent", r"279 ipc", r"304a ipc", r"traffic", r"driving license", r"motor vehicles act"],
        "Family/Matrimonial": [r"marriage", r"divorce", r"alimony", r"custody", r"widow", r"adoption", r"maintenance", r"hindu adoption", r"family dispute", r"succession", r"inheritance", r"nullity of marriage", r"separation", r"guardianship"],
        "Service/Employment": [r"promotion", r"appointment", r"termination", r"dismissal", r"suspension", r"seniority", r"pension", r"family pension", r"salary", r"wages", r"employment", r"crpf", r"army", r"dgms", r"service matter", r"back wages", r"retiral benefits", r"misconduct", r"disciplinary", r"departmental inquiry", r"civil servant"],
        "Labour/Industrial Law": [r"labour", r"industrial dispute", r"workman", r"trade union", r"gratuity", r"provident fund", r"epf", r"esi", r"minimum wages", r"industrial disputes act"],
        "Consumer/Insurance": [r"consumer protection", r"insurance claim", r"deficiency in service", r"consumer forum", r"policy claim", r"compensation", r"consumer dispute"],
        "Contempt of Court": [r"contempt", r"scandalize", r"denigrate", r"administration of justice", r"disobedience", r"contemptuous"],
        "Media/Telecom": [r"broadcast", r"telecom", r"cable", r"sports act", r"television", r"retransmitted", r"signalling", r"spectrum", r"internet"],
    }

    # First pass: Specific matches with word boundaries applied to FULL TEXT
    for category, patterns in taxonomy.items():
        for pattern in patterns:
            if re.search(r'\b' + pattern + r'\b', full_text):
                return category

    # Second pass: Generic matches
    if any(re.search(r'\b' + word + r'\b', full_text) for word in [r"criminal", r"offense", r"ipc", r"conviction", r"acquittal", r"bail", r"accused", r"police", r"crpc", r"magistrate", r"remand"]):
        return "General Criminal"
    if any(re.search(r'\b' + word + r'\b', full_text) for word in [r"civil", r"suit", r"decree", r"petition", r"cpc", r"contempt", r"litigation", r"plaintiff", r"respondent", r"appellant", r"injunction", r"stay order"]):
        return "General Civil"

    return "Other"


# =================================================
# STREAMLIT PAGE CONFIGURATION
# Sets title, icon, and layout of app
# =================================================


st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="classical symbol of justice .png",
    layout="wide"
)


# =================================================
# CUSTOM UI STYLING
# Applies background, fonts, buttons, and theme
# =================================================


@st.cache_data
def get_base64_of_bin_file(bin_file):
    import base64
    import os
    try:
        if not os.path.exists(bin_file):
            return ""
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

def load_custom_css():
    img_b64 = get_base64_of_bin_file("classical symbol of justice .png")
    
    if img_b64:
        bg_style = f"""
        .stApp {{
            background-image: linear-gradient(rgba(2, 6, 23, 0.10), rgba(15, 23, 42, 1)), url("data:image/png;base64,{img_b64}") !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-position: center, right 2vw top 25% !important;
            background-size: cover, auto 85vh !important;
        }}
        """
    else:
        bg_style = """
        .stApp {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
        }
        """

    st.markdown(f"<style>{bg_style}</style>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0;
    }

    /* Universal Heading Styles */
    .unified-h1 {
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 4.8rem !important; 
        font-weight: 800 !important;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #3b82f6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0 0 1.5rem 0 !important;
        line-height: 1.1 !important;
        letter-spacing: -1.5px !important;
    }
    .unified-h1 i {
        -webkit-text-fill-color: #3b82f6;
        filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.6));
        font-size: 0.85em;
    }
    
    .unified-h3 {
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 2.2rem !important; 
        font-weight: 800 !important;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #3b82f6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 1.5rem 0 1.5rem 0 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.5px !important;
    }
    .unified-h3 i {
        -webkit-text-fill-color: #3b82f6;
        font-size: 0.85em;
    }

    /* Main App Background is handled dynamically above */

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }

    /* Input Box (Chat Input) */
    /* Ensure the chat bottom container is transparent */
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Input Box (Chat Input) - Premium Glass Theme */
    .stChatInputContainer {
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
        padding-left: 10px !important;
    }
    
    .stChatInputContainer:focus-within {
        background: rgba(30, 41, 59, 0.6) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.25) !important;
    }

    /* Modern Circular Send Button */
    [data-testid="stChatInput"] button {
        background: rgba(59, 130, 246, 0.2) !important;
        border-radius: 50% !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        width: 38px !important;
        height: 38px !important;
        margin: auto 4px auto auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        background: rgba(59, 130, 246, 0.4) !important;
        transform: scale(1.05);
    }
    
    [data-testid="stChatInput"] button svg {
        fill: #60a5fa !important;
    }

    .stChatInputContainer textarea {
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        padding-left: 5px !important;
    }
    
    .stChatInputContainer textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Remove double-box from Chat Input's inner text area */
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] > div > div > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Primary Buttons (New Chat) */
    button[kind="primary"] {
        border-radius: 12px !important;
        border: none !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        text-align: center !important;
        justify-content: center !important;
        display: flex !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Secondary Buttons (History items) */
    button[kind="secondary"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        background: rgba(30, 41, 59, 0.4) !important;
        color: #e2e8f0 !important;
        font-weight: 400 !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 0.75rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    button[kind="secondary"] p {
        text-align: left !important;
        width: 100%;
        margin-left: 5px;
    }
    button[kind="secondary"]:hover {
        background: rgba(59, 130, 246, 0.1) !important;
        color: #ffffff !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }

    /* Force left-align for Sidebar History Buttons */
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] .stBaseButton-secondary {
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 10px !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"] div,
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"] div,
    [data-testid="stSidebar"] button[kind="secondary"] p,
    [data-testid="stSidebar"] button[kind="secondary"] span {
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        width: auto !important;
        margin: 0 !important;
    }

    /* Remove the solid black background from the bottom chat input bar */
    .stChatFloatingInputContainer,
    .stChatInputContainer,
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Radio Labels hide default text */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }
    
    /* Suggestion Chips Style */
    div.stButton > button[kind="secondary"] {
        white-space: nowrap !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 0.8rem !important;
    }
    
    .suggestion-title {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 15px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.5px;
    }
    
    /* Make Selectboxes and Text Inputs Translucent to reveal background */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: rgba(2, 6, 23, 0.20) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Make DataFrames and Charts Transparent by lowering canvas opacity so the image bleeds through */
    div[data-testid="stDataFrame"],
    div[data-testid="stVegaLiteChart"], 
    div[data-testid="stArrowVegaLiteChart"],
    div[data-testid="stChart"] {
        opacity: 0.65 !important;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Entire Radio Group spacing */
    div[role="radiogroup"] {
        gap: 0.5rem !important;
    }

    /* Individual Radio Item (Pill Style) */
    div[role="radiogroup"] > label {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        margin: 0 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    /* Hover State */
    div[role="radiogroup"] > label:hover {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }

    /* Selected State using :has selector */
    div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.5) 100%) !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    }

    /* Hide the actual circle dots */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Text styling */
    div[role="radiogroup"] > label p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    
    div[role="radiogroup"] > label:has(input:checked) p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    /* Secondary buttons (like expanders) */
    .st-emotion-cache-1629p8f { 
        /* Targeting common expander/toggle colors */
        color: #e2e8f0;
    }

    /* Chat Messages Wrapper */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 1.5rem !important;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        background-color: transparent !important;
    }
    
    /* User Message distinct style */
    [data-testid="stChatMessage"]:has(div.st-emotion-cache-1c7y2kd) {
         /* A fallback targeting if needed, but we'll stick to Streamlit's structural child order or just rely on the content */
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    
    /* Ensure Chat Input Inner Area Remains Single-Box */
    div[data-testid="stChatInput"] div[data-baseweb="base-input"],
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    div.stChatInputContainer div[data-baseweb="base-input"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Mobile Responsiveness Fixes */
    @media (max-width: 768px) {
        .unified-h1 {
            font-size: 2.2rem !important;
            line-height: 1.2 !important;
        }

        .home-subtitle {
            font-size: 1rem !important;
        }

        .stApp {
            background-position: center, right -40px top 10% !important;
            background-size: cover, auto 60vh !important;
        }
        
        .home-hero {
            padding: 1rem !important;
        }

        .home-card-container {
            padding: 10px !important;
        }

        .stChatInputContainer {
            width: 95% !important;
            margin: auto !important;
        }

        .home-feature-card {
            margin-bottom: 20px !important;
        }
        
        /* Force Stack Columns on Mobile for Streamlit */
        [data-testid="column"] {
            min-width: 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()


# =================================================
# FUNCTION: load_data
# PURPOSE:
# Loads legal dataset from JSON file
# Handles missing file and formatting issues
# RETURNS: Pandas DataFrame
# =================================================


@st.cache_data
def load_data():
    import os
    import json
    import pandas as pd
    
    # [CACHE BUST: Taxonomy Update 3 - Extreme normalizer & order fix]

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

        # =================================================
        # AUTO-CLASSIFY ON LOAD (Ensures labels update)
        # =================================================
        if "case_name" in df.columns:
            # Normalize case names to prevent duplicate entries from spelling variations like "vs" vs "vs."
            df["case_name"] = df["case_name"].str.replace(r"\s+v(s)?\.?\s+", " vs. ", case=False, regex=True).str.strip()
            
            # EXTREME SCRUBBER: Map structural differences & appendages ("& Ors.", "Etc.", brackets) to a single definitive instance
            clean = df["case_name"].str.lower()
            clean = clean.str.replace(r"(\s+(&|and)\s+(ors\.?|others|anr\.?|another|etc\.?).*?$)", "", case=False, regex=True)
            clean = clean.str.replace(r"\(.*?\)", "", regex=True)
            clean = clean.str.replace(r"^the\s+", "", case=False, regex=True)
            clean = clean.str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
            
            # Map all row variations that share the SAME clean name base to exactly the very FIRST string seen
            name_map = df.groupby(clean)["case_name"].first().to_dict()
            df["case_name"] = clean.map(name_map).fillna(df["case_name"])
            
            # Group all Q&A pairs for the same case to ensure uniform classification
            df['full_text'] = df['question'].fillna("") + " " + df['answer'].fillna("")
            
            case_texts = df.groupby('case_name')['full_text'].apply(lambda x: ' '.join(x)).reset_index()
            
            case_texts['case_subtype'] = case_texts.apply(
                lambda row: get_case_subtype(row['full_text'], row['case_name']), axis=1
            )
            
            subtype_map = dict(zip(case_texts['case_name'], case_texts['case_subtype']))
            df['case_subtype'] = df['case_name'].map(subtype_map)
            df.drop(columns=['full_text'], inplace=True)

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


# Load once
df = load_data()
global index


# =================================================
# DATA PRE-PROCESSING
# =================================================

# ===== FIX DUPLICATE DATE COLUMN =====


if "judgement_date" in df.columns and "judgment_date" in df.columns:
    df["judgment_date"] = df["judgment_date"].fillna(df["judgement_date"])
    df.drop(columns=["judgement_date"], inplace=True)

if df.empty:
    st.stop()


# =================================================
# LOAD NLP EMBEDDING MODEL
# Used for converting text into vectors
# =================================================


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# =================================================
# FUNCTION: build_index
# PURPOSE:
# Creates or loads FAISS index for fast similarity search
# Stores embeddings of all questions
# =================================================


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
# FUNCTION: ai_search_top_cases
# PURPOSE:
# Finds top K similar legal cases using FAISS
# =================================================


def get_similar_cases(case_row, top_k=3):
    """Finds similar cases by prioritizing the same legal category (subtype)."""
    if case_row is None or case_row.empty:
        return pd.DataFrame()

    subtype = case_row.get("case_subtype", "Other")
    main_name = case_row.get("case_name", "")

    # Priority 1: Same category, random sample
    category_matches = df[df["case_subtype"] == subtype]
    category_matches = category_matches[category_matches["case_name"] != main_name]
    category_matches = category_matches.drop_duplicates(subset=["case_name"])

    if len(category_matches) >= top_k:
        return category_matches.sample(top_k)
    
    # Fallback to general variety if category is small
    return category_matches.head(top_k)


# =================================================
# FUNCTION: hybrid_search_legal_answer
# PURPOSE:
# Combines multiple search techniques:
# 1. Exact match
# 2. Partial match
# 3. Keyword search
# 4. AI vector search (FAISS)
# =================================================


def extract_case_name(query):
    """Extracts a 'Party vs. Party' style case name from a natural language query."""
    match = re.search(r'([\w\s]+vs?\.?[\s]+[\w\s]+)', query, re.IGNORECASE)
    return match.group(1).strip() if match else None


def classify_query(query):
    """Categorizes the user's question into Case Search, Crime Scan, or General Legal Inquiry."""
    query = query.lower()
    if any(word in query for word in ["vs", " v ", " v. "]):
        return "case_name"
    elif any(word in query for word in ["murder", "rape", "fraud", "theft", "crime", "kidnap", "stolen", "ipc", "crpc"]):
        return "crime"
    else:
        return "general"


def hybrid_search_legal_answer(user_question):
    query_lower = user_question.lower().strip()
    query_type = classify_query(user_question)

    # =============================
    # ROUTE 1: DIRECT CASE SEARCH
    # =============================
    if query_type == "case_name":
        # Look for the exact match in our verified normalized column
        for _, row in df.iterrows():
            if query_lower == str(row.get("case_name", "")).lower():
                return row, 1.0  # Exact match = 100% confidence
        
        # If no exact match but user intended a case name, try a safe contains match
        extracted = extract_case_name(user_question)
        if extracted:
            match = df[df["case_name"].str.contains(extracted, case=False, na=False)]
            if not match.empty:
                return match.iloc[0], 0.9  # Partial match = 90% confidence

    # =============================
    # ROUTE 2: CRIME/SUBTYPE FILTER
    # =============================
    elif query_type == "crime":
        filtered = df[df["case_subtype"].str.lower().str.contains(query_lower, na=False)]
        if not filtered.empty:
            return filtered.sample(1).iloc[0], 0.8  # Random sample from category = 80% confidence

    # =============================
    # ROUTE 3: SEMANTIC FAISS (Fallback)
    # =============================
    user_embedding = model.encode([user_question])
    user_embedding = np.array(user_embedding).astype("float32")

    distances, indices = index.search(user_embedding, 1)
    distance = distances[0][0]
    
    # Confidence = max(0, 1 - distance)
    # Since L2 distances in MiniLM usually sit between 0.3 and 1.5
    confidence = max(0.0, min(1.0, 1.2 - distance)) # Normalized around our 1.2 threshold

    # 🚨 STRICT RELEVANCE: Reject anything with L2 distance > 1.2
    if distance > 1.2:
        return None, 0.0

    return df.iloc[indices[0][0]], float(confidence)


# =================================================
# FUNCTION: ask_llm
# PURPOSE:
# Sends prompt to Llama-3-70B via Groq API
# Replaces old local Ollama integration
# =================================================


def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800
        )

        content = response.choices[0].message.content

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return content, input_tokens, output_tokens, total_tokens

    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0

# =================================================
# FUNCTION: typing_effect
# PURPOSE:
# Simulates a typing effect for a better AI experience
# =================================================
def typing_effect(text):
    placeholder = st.empty()
    displayed_text = ""

    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(0.003) # High-speed professional legal typing

    placeholder.markdown(displayed_text)
    return displayed_text


# =================================================
# FUNCTION: build_chatbot_context
# PURPOSE:
# Prepares structured prompt for LLM
# Extracts case info and formats response
# =================================================


def build_chatbot_context(user_question, case_result=None):

    # =============================
    # STEP 1 — Get Case Result (If not provided)
    # =============================
    if case_result is None:
        case_result, confidence = hybrid_search_legal_answer(user_question)

    if case_result is None or (isinstance(case_result, pd.Series) and case_result.empty):
        return None, None, None, []

    # =============================
    # STEP 2 — Extract data
    # =============================
    case_name = case_result.get("case_name", "Unknown Case")

    judgment_date = case_result.get("judgment_date", "Not Available")

    case_type = case_result.get("case_subtype", "Other")

    case_answer = case_result.get("answer", "No summary available.")


    # 🔥 REMOVE BAD INSTRUCTIONS FROM DATA
    stop_words = [
        "Rules:",
        "SEO Analyst",
        "review team",
        "feedback",
        "optimize",
        "assistant's response",
        "Question:",
        "First, we need to"
    ]

    for word in stop_words:
        if word.lower() in case_answer.lower():
            case_answer = case_answer.split(word)[0]

    case_answer = case_answer.strip()

    # =============================
    # STEP 3 — Build Context
    # =============================
    context = f"""
Case Name: {case_name}
Case Category: {case_type}
Judgment Date: {judgment_date}
Case Details: - Case Name: {case_name} - Category: {case_type} - Judgment Date: {judgment_date}  Full Case Explanation: {case_answer}
"""

    # =============================
    # STEP 4 — STRONG PROMPT CONTROL
    # =============================
   prompt = f"""
You are a highly experienced Indian Legal Analyst.

IMPORTANT:
- Use ONLY the given case data
- Do NOT hallucinate
- But provide detailed legal reasoning

CASE DATA:
{context}

USER QUESTION:
{user_question}

INSTRUCTIONS:

1. Explain in a structured but PROFESSIONAL legal manner
2. Avoid generic statements
3. Expand reasoning using legal interpretation
4. If data is limited, logically infer but stay grounded

FORMAT:

🔹 What Happened:
- Explain incident clearly with context

🔹 Background:
- Explain possible cause, legal situation

🔹 Legal Issue:
- Explain core legal questions deeply

🔹 Court Decision:
- Clearly state judgment

🔹 Reasoning:
- Explain WHY court decided (IMPORTANT — detailed)

🔹 Final Outcome:
- Explain result clearly

IMPORTANT:
- Do NOT repeat lines
- Avoid vague statements like "court found guilty"
- Add logical explanation
"""
    # =============================
    # STEP 4b — QUERY INTENT DETECTION
    # =============================
    query_lower_intent = user_question.lower()
    if any(word in query_lower_intent for word in ["legal issue", "what is the issue", "problem"]):
        prompt += "\n[FOCUS INSTRUCTION]: Focus ONLY on the Legal Issue section. Expand it with at least 4 bullet points."
    elif any(word in query_lower_intent for word in ["court decision", "what did the court", "judge decide", "ruling"]):
        prompt += "\n[FOCUS INSTRUCTION]: Focus ONLY on the Court Decision and Reason sections. Expand them with at least 4 bullet points each."
    elif any(word in query_lower_intent for word in ["final outcome", "verdict", "punishment", "result", "sentence"]):
        prompt += "\n[FOCUS INSTRUCTION]: Focus ONLY on the Final Outcome section. Expand it with all available details."

    # =============================
    # STEP 5 — Add Case Header
    # =============================
    # Smart truncation at exactly the last full word before the 350 character limit
    limit = 350
    if len(case_answer) > limit:
        short_summary = case_answer[:limit].rsplit(' ', 1)[0] + "..."
    else:
        short_summary = case_answer

    header = f"""
:material/label: **Case:** {case_name}  
:material/event: **Date:** {judgment_date}  
:material/category: **Type:** {case_type}  

:material/psychology: **Quick Summary:**  
{short_summary}

---
"""

    return prompt, header, case_result, []
    

# =================================================
# FUNCTION: generate_brief_html
# PURPOSE:
# Creates a high-fidelity styled HTML document for export
# Translates internal icons and markdown to document-friendly formats
# =================================================
def generate_brief_html(p_name, judgment_date, case_type, user_question, header_content, assistant_response):
    import re
    
    # Mapping Material icons to Emojis for PDF compatibility
    icon_map = {
        ":material/label:": "📝",
        ":material/event:": "📅",
        ":material/category:": "📁",
        ":material/psychology:": "🧠",
        ":material/receipt_long:": "📜",
        ":material/menu_book:": "📖",
        ":material/gavel:": "⚖️",
        ":material/account_balance:": "🏛️"
    }
    
    clean_header = header_content
    # Remove the horizontal rule from header if present for cleaner PDF
    clean_header = clean_header.replace("---", "")
    
    clean_ai = assistant_response
    
    for m, e in icon_map.items():
        clean_header = clean_header.replace(m, e)
        clean_ai = clean_ai.replace(m, e)

    # Simple Markdown to HTML
    def simple_md(text):
        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Bullet points
        text = text.replace("\n- ", "<br>• ")
        # Line breaks
        text = text.replace("\n", "<br>")
        return text

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
                color: #1e293b; 
                line-height: 1.6; 
                padding: 40px; 
                background-color: #ffffff; 
                max-width: 900px;
                margin: auto;
            }}
            .header-info {{ 
                color: #1e3a8a; 
                border-bottom: 3px solid #3b82f6; 
                padding-bottom: 15px; 
                margin-bottom: 25px; 
            }}
            .case-title {{ 
                font-size: 28px; 
                font-weight: 800; 
                margin-bottom: 8px; 
                line-height: 1.2;
            }}
            .subtitle {{ 
                font-size: 13px; 
                color: #64748b; 
                text-transform: uppercase; 
                letter-spacing: 1.5px; 
                font-weight: 600;
            }}
            .card {{ 
                background-color: #f1f5f9; 
                border-radius: 16px; 
                padding: 24px; 
                border: 1px solid #e2e8f0; 
                margin-bottom: 30px; 
                font-size: 15px;
            }}
            .section-header {{ 
                display: flex; 
                align-items: center; 
                border-left: 5px solid #3b82f6; 
                background-color: #eff6ff; 
                padding: 12px 20px; 
                border-radius: 0 10px 10px 0; 
                margin: 40px 0 20px 0; 
                font-weight: 700; 
                color: #1e40af; 
                font-size: 18px;
            }}
            .content {{ 
                padding: 0 10px; 
                font-size: 16px;
            }}
            .italic-quote {{ 
                font-style: italic; 
                color: #475569; 
                background-color: #f8fafc;
                padding: 15px 25px;
                border-radius: 8px;
                margin: 10px 0;
            }}
            b {{ color: #0f172a; }}
        </style>
    </head>
    <body>
        <div class="header-info">
            <div class="case-title">{p_name}</div>
            <div class="subtitle">JUDGMENT DATE: {judgment_date} | CATEGORY: {case_type}</div>
        </div>

        <div class="card">
            {simple_md(clean_header)}
        </div>

        <div class="section-header">User Question</div>
        <div class="italic-quote">
            "{user_question}"
        </div>

        <div class="section-header">AI Reply on Case</div>
        <div class="content">
            {simple_md(clean_ai)}
        </div>
        
        <div style="margin-top: 50px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #f1f5f9; padding-top: 20px;">
            Generated by AI Legal Assistant for Indian Case Law
        </div>
    </body>
    </html>
    """
    return html


index = build_index(df)


# =================================================
# SIDEBAR UI
# Handles navigation, new chat, and history
# =================================================


# ================= SIDEBAR =================

st.sidebar.markdown("<h2 style='text-align: center; color: #f8fafc; margin-bottom: 25px;'><i class='fa-solid fa-scale-balanced'></i> Legal AI Pro</h2>", unsafe_allow_html=True)

# Navigation labels with their corresponding Material Icons
nav_map = {
    "Home": "home",
    "Chat Assistant": "chat",
    "Analytics": "analytics",
    "Dataset": "database",
    "About": "info"
}

page = st.sidebar.radio(
    "Navigation",
    options=list(nav_map.keys()),
    format_func=lambda x: f":material/{nav_map[x]}: {x}",
    index=0,
    label_visibility="collapsed"
)

# ================= NEW CHAT BUTTON =================
st.sidebar.write("") # Spacing
if st.sidebar.button("New Chat", type="primary", use_container_width=True, icon=":material/add:"):
    # clear only current chat
    st.session_state.messages = []
    st.session_state.similar_cases = None
    st.session_state.show_suggestions = True # Restore suggestions for new chat

    st.rerun()

st.sidebar.divider()

# ================= SEARCH HISTORY =================

st.sidebar.markdown("<h3 style='color: #f8fafc;'><i class='fa-solid fa-clock-rotate-left'></i> Search History</h3>", unsafe_allow_html=True)

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

    # Clean display text for sidebar 
    q = item['query']
    display_q = q[:24] + "..." if len(q) > 24 else q

    if st.sidebar.button(
            display_q,
            key=f"history_{i}",
            type="secondary",
            use_container_width=True,
            icon=":material/history:"
    ):
        # Clear current chat
        st.session_state.messages = []

        # Load old conversation instead of searching again
        st.session_state.messages.append(("user", item["query"]))
        # Store confidence as 5th element: ("bot", response, cases, pdf, confidence)
        st.session_state.messages.append((
            "bot", 
            item["response"], 
            item["cases"], 
            None, 
            item.get("confidence", 1.0)
        ))

        st.rerun()


# =================================================
# CHAT INTERFACE
# Handles user interaction and AI response display
# =================================================


def render_confidence_ui(confidence):
    """Render a unique, high-fidelity confidence insight card."""
    if confidence is None:
        return
        
    try:
        conf = float(confidence)
        pct = int(conf * 100)
    except (ValueError, TypeError):
        return
    
    # Theme configuration based on score
    if pct > 75:
        color, label, desc = "#10b981", "PRECISE MATCH", "This case aligns exceptionally well with your facts and legal context."
        glow = "rgba(16, 185, 129, 0.15)"
    elif pct > 40:
        color, label, desc = "#f59e0b", "PARTIAL MATCH", "The case contains relevant legal precedents but facts may vary."
        glow = "rgba(245, 158, 11, 0.15)"
    else:
        color, label, desc = "#ef4444", "BROAD MATCH", "Query matched on broad legal themes; direct relevance is limited."
        glow = "rgba(239, 68, 68, 0.15)"

    # SVG Dash Offset Calculation (Perimeter ≈ 251.2)
    dash_offset = 251.2 * (1 - conf)

    # HTML Template (Flattened to avoid markdown code-block detection)
    html_template = """<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 20px; padding: 24px; margin: 20px 0; backdrop-filter: blur(15px); box-shadow: 0 10px 40px rgba(0,0,0,0.3), inset 0 0 20px {{GLOW}}; display: flex; align-items: center; gap: 30px; max-width: 600px;">
<div style="position: relative; width: 100px; height: 100px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
<svg width="100" height="100" viewBox="0 0 100 100">
<circle cx="50" cy="50" r="40" stroke="rgba(255,255,255,0.05)" stroke-width="10" fill="transparent" />
<circle cx="50" cy="50" r="40" stroke="{{COLOR}}" stroke-width="10" fill="transparent" stroke-dasharray="251.2" stroke-dashoffset="{{OFFSET}}" stroke-linecap="round" style="transition: stroke-dashoffset 1.5s ease-out; filter: drop-shadow(0 0 5px {{COLOR}});" transform="rotate(-90 50 50)" />
</svg>
<div style="position: absolute; text-align: center;">
<div style="font-size: 1.6rem; font-weight: 900; color: white; line-height: 1;">{{PCT}}<span style="font-size: 0.8rem;">%</span></div>
</div>
</div>
<div style="flex-grow: 1;">
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
<div style="width: 8px; height: 8px; border-radius: 50%; background: {{COLOR}}; box-shadow: 0 0 10px {{COLOR}};"></div>
<span style="color: {{COLOR}}; font-weight: 800; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;">{{LABEL}}</span>
</div>
<div style="color: white; font-size: 1.3rem; font-weight: 700; margin-bottom: 4px;">Search Retrieval Insight</div>
<div style="color: #94a3b8; font-size: 1rem; line-height: 1.5;">{{DESC}}</div>
</div>
</div>"""
    
    # Safe replacement
    rendered_html = html_template.replace("{{COLOR}}", color)\
                                 .replace("{{GLOW}}", glow)\
                                 .replace("{{OFFSET}}", str(dash_offset))\
                                 .replace("{{PCT}}", str(pct))\
                                 .replace("{{LABEL}}", label)\
                                 .replace("{{DESC}}", desc)
                                 
    st.markdown(rendered_html, unsafe_allow_html=True)
    st.divider()

@st.fragment
def render_similar_cases_ui(cases, msg_idx, is_live=False):
    st.markdown("<h4 style='color: #94a3b8; margin-top: 25px; letter-spacing: 1px; font-weight: 700; text-transform: uppercase; font-size: 0.85rem;'><i class='fa-solid fa-folder-open'></i> Similar Case Matches</h4>", unsafe_allow_html=True)

    for i, (_, row) in enumerate(cases.iterrows()):
        case_name = row.get("case_name", "Unknown Case")
        date = row.get("judgment_date", "Date N/A")
        summary = row.get("answer", "No summary available.")
        
        # Toggle for details
        clean_key = "".join(filter(str.isalnum, case_name))[:15]
        show_details = st.toggle(f"Show Details of: {case_name}", key=f"tg_{msg_idx}_{i}_{clean_key}")
        
        # Box styling (reverting to vertical premium boxes)
        border_color = "#10b981" if show_details else "#3b82f6"
        icon_color = "#10b981" if show_details else "#60a5fa"
        
        if show_details:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.4); padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.05); border-left: 5px solid {border_color}; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                <div style="color: {icon_color}; font-weight: 700; font-size: 1.1rem; margin-bottom: 5px;"><i class="fa-solid fa-paperclip"></i> {case_name}</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 15px;"><i class="fa-solid fa-calendar"></i> {date}</div>
                <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color: #f8fafc; font-weight: 600; font-size: 0.9rem; margin-bottom: 8px;"><i class="fa-solid fa-file-lines"></i> Case Summary:</div>
                    <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">{summary}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.3); padding: 15px; border-radius: 10px; margin-bottom: 12px; border: 1px solid rgba(255, 255, 255, 0.05); border-left: 5px solid {border_color};">
                <div style="color: {icon_color}; font-weight: 600; font-size: 1rem;"><i class="fa-solid fa-paperclip"></i> {case_name}</div>
                <div style="color: #64748b; font-size: 0.8rem;"><i class="fa-solid fa-calendar-days"></i> {date}</div>
            </div>
            """, unsafe_allow_html=True)

# =================================================
# PAGE RENDERING LOGIC
# =================================================

if page == "Home":
    st.markdown("""
    <style>
    @keyframes fadeInSlideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.4)); }
        50% { filter: drop-shadow(0 0 25px rgba(16, 185, 129, 0.6)); }
        100% { filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.4)); }
    }

    .home-hero {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        text-align: left; 
        padding: 2rem 0 3rem 0;
        animation: fadeInSlideUp 0.8s ease-out forwards;
        width: 100%;
    }
    

    
    .pulse-icon-hero {
        animation: pulseGlow 4s infinite ease-in-out;
        color: #3b82f6; /* Base color before gradient glow */
        font-size: 4rem;
        -webkit-text-fill-color: #3b82f6; /* overrides the text gradient */
    }

    .home-subtitle {
        font-size: 1.45rem; 
        color: #cbd5e1; 
        max-width: 750px; 
        margin: 0;
        line-height: 1.6;
        font-weight: 300;
    }
    
    .brand-highlight {
        color: #f8fafc;
        font-weight: 600;
        border-bottom: 2px solid #10b981;
    }

    .home-card-container {
        padding: 20px 0 40px 0;
    }

    .home-feature-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.6) 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        height: 100%;
        backdrop-filter: blur(16px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        opacity: 0;
        animation: fadeInSlideUp 0.8s ease-out forwards;
    }
    
    /* Staggered animation delays */
    .delay-1 { animation-delay: 0.2s; }
    .delay-2 { animation-delay: 0.4s; }
    .delay-3 { animation-delay: 0.6s; }

    .home-feature-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.6), inset 0 0 20px rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.4);
    }

    .icon-wrapper {
        width: 70px;
        height: 70px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .home-feature-card:hover .icon-wrapper {
        transform: scale(1.1) rotate(5deg);
    }

    .icon-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .icon-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .icon-purple { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }

    .home-feature-card h3 {
        color: #f8fafc !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 1rem;
        width: 100%;
        text-align: center;
        justify-content: center;
    }

    .home-feature-card p {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0;
    }
    </style>
    
    <div class='home-hero'>
        <h1 class='unified-h1'>
            <i class='fa-solid fa-scale-balanced pulse-icon-hero'></i> Legal AI Pro
        </h1>
        <p class='home-subtitle'>
            Empowering you to explore, understand, and analyze <span class='brand-highlight'>Indian Case Law</span> instantly with the Intelligence of AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Features Grid
    st.markdown("<div class='home-card-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='home-feature-card delay-1'>
                <div class='icon-wrapper icon-blue'><i class='fa-solid fa-robot'></i></div>
                <h3 class='unified-h3'>Intelligent Chat</h3>
                <p>An interactive AI assistant that understands complex legal queries and distills relevant Indian cases into plain language.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='home-feature-card delay-2'>
                <div class='icon-wrapper icon-green'><i class='fa-solid fa-chart-pie'></i></div>
                <h3 class='unified-h3'>Deep Analytics</h3>
                <p>Unlock profound insights into crime distributions, historical trends, and case velocity across thousands of judgments.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='home-feature-card delay-3'>
                <div class='icon-wrapper icon-purple'><i class='fa-solid fa-magnifying-glass-chart'></i></div>
                <h3 class='unified-h3'>Dataset Explorer</h3>
                <p>Navigate the comprehensive IndicLegalQA repository with powerful multi-dimensional filtering and semantic sorting.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.info("💡 **Ready to explore?** Use the **Navigation** sidebar on the left to start searching for legal cases, chatting with the AI, or exploring the core dataset!")

elif page == "Chat Assistant":
    st.markdown("<h1 class='unified-h1'><i class='fa-solid fa-scale-balanced'></i> AI Legal Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Ask anything about Indian law — get clear, AI-powered answers instantly.</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_suggestions" not in st.session_state:
        st.session_state.show_suggestions = True
    if len(st.session_state.messages) > 0:
        st.session_state.show_suggestions = False

    # 1. Display Chat History
    for msg_idx, message in enumerate(st.session_state.messages):
        if message[0] == "user":
            with st.chat_message("user", avatar="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiByeD0iOCIgZmlsbD0iIzNiODJmNiIvPjxwYXRoIGQ9Ik0xMiAxMkMxNC4yMSAxMiAxNiAxMC4yMSAxNiA4QzE2IDUuNzkgMTQuMjEgNCAxMiA0QzkuNzkgNCA4IDUuNzkgOCA4QzggMTAuMjEgOS43OSAxMiAxMiAxMlpNMTIgMTRDOS4zMyAxNCA0IDE1LjM0IDQgMThWMjBIMjBWMThDMjAgMTUuMzQgMTQuNjcgMTQgMTIgMTRaIiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg=="):
                st.markdown(message[1])
        else:
            with st.chat_message("assistant", avatar="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiByeD0iOCIgZmlsbD0iIzFlMjkzYiIgc3Ryb2tlPSIjMTBiOTgxIiBzdHJva2Utd2lkdGg9IjEuNSIvPjxwYXRoIGQ9Ik0xMiA0TDMgOUwzIDExSDIxTDIxIDlMMTIgNFpNMyAxMkg2VjE5SDNWMTJaTTggMTJIMTFWMTlIOFYxMlpNMTMgMTJIMTZWMTlIMTNWMTJaTTE4IDEySDIxVjE5SDE4VjEyWk0zIDIxSDIxVjIzSDNWMjFaIiBmaWxsPSIjMTBiOTgxIi8+PC9zdmc+"):
                st.markdown(message[1])
                # 1. Confidence UI
                if len(message) > 4: 
                    render_confidence_ui(message[4])
                
                # 2. Download Button
                if len(message) > 3 and message[3] is not None:
                    pdf_payload = message[3]
                    p_data = pdf_payload["html"].encode("utf-8") if isinstance(pdf_payload, dict) else pdf_payload.encode("utf-8")
                    p_name = pdf_payload.get("name", "case") if isinstance(pdf_payload, dict) else "case"
                    st.download_button(label=":material/picture_as_pdf: Download Case Brief (PDF)", data=p_data, file_name=f"{p_name[:30]}.html", mime="text/html", key=f"pdf_h_{msg_idx}", type="primary")

            if len(message) > 2 and message[2] is not None and len(message[2]) > 0:
                render_similar_cases_ui(message[2], msg_idx)

    # 2. Suggestions & Input UI
    active_q = None
    
    # Check input FIRST to update flags before rendering
    user_input = st.chat_input("Ask your Legal query here...")
    
    if user_input:
        st.session_state.show_suggestions = False
        active_q = user_input
    elif "active_suggestion" in st.session_state and st.session_state.active_suggestion:
        active_q = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        st.session_state.show_suggestions = False

    # Render suggestions ONLY if flag is still true
    if st.session_state.show_suggestions:
        st.markdown("<div class='suggestion-title'><i class='fa-solid fa-wand-magic-sparkles'></i> Suggested Queries</div>", unsafe_allow_html=True)
        suggestions = [
            "What is the legal issue in a homicide case?", 
            "Explain a property dispute case", 
            "What is financial fraud?", 
            "What is income tax dispute?", 
            "What happens if promotion is denied?"
        ]
        # Custom ratios to give more space to longer text
        cols = st.columns([1.4, 1.2, 0.9, 0.9, 1.4])
        for idx, (col, sug_text) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(sug_text, key=f"sugB_{idx}", type="secondary", use_container_width=True):
                    st.session_state.active_suggestion = sug_text
                    st.session_state.show_suggestions = False
                    st.rerun()

    # 3. Process Query
    if active_q:
        if st.session_state.request_count >= DAILY_LIMIT:
            st.error("⚠️ Daily limit reached. Try again tomorrow.")
            st.stop()

        msg_idx = len(st.session_state.messages)
        st.session_state.messages.append(("user", active_q))
        with st.chat_message("user", avatar="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiByeD0iOCIgZmlsbD0iIzNiODJmNiIvPjxwYXRoIGQ9Ik0xMiAxMkMxNC4yMSAxMiAxNiAxMC4yMSAxNiA4QzE2IDUuNzkgMTQuMjEgNCAxMiA0QzkuNzkgNCA4IDUuNzkgOCA4QzggMTAuMjEgOS43OSAxMiAxMiAxMlpNMTIgMTRDOS4zMyAxNCA0IDE1LjM0IDQgMThWMjBIMjBWMThDMjAgMTUuMzQgMTQuNjcgMTQgMTIgMTRaIiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg=="):
            st.markdown(active_q)

        case_res, conf = hybrid_search_legal_answer(active_q)
        prompt, header, _, _ = build_chatbot_context(active_q, case_res)
        top_c = get_similar_cases(case_res, 3)

        with st.chat_message("assistant", avatar="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiByeD0iOCIgZmlsbD0iIzFlMjkzYiIgc3Ryb2tlPSIjMTBiOTgxIiBzdHJva2Utd2lkdGg9IjEuNSIvPjxwYXRoIGQ9Ik0xMiA0TDMgOUwzIDExSDIxTDIxIDlMMTIgNFpNMyAxMkg2VjE5SDNWMTJaTTggMTJIMTFWMTlIOFYxMlpNMTMgMTJIMTZWMTlIMTNWMTJaTTE4IDEySDIxVjE5SDE4VjEyWk0zIDIxSDIxVjIzSDNWMjFaIiBmaWxsPSIjMTBiOTgxIi8+PC9zdmc+"):
            if not prompt:
                f_resp = "No relevant case found."
                p_pay = None
                st.markdown(f_resp)
            else:
                p_name = case_res.get("case_name", "Case")
                st.markdown(header)
                with st.spinner("Analyzing legal case..."):
                    response, in_tok, out_tok, total_tok = ask_llm(prompt)
                
                typing_effect(response)

                # Increment usage
                st.session_state.request_count += 1

                # Show usage
                st.caption(f"🧠 Used: {st.session_state.request_count}/{DAILY_LIMIT}")
                st.caption(f"📊 Tokens: {total_tok} (Input: {in_tok}, Output: {out_tok})")
                
                # Progress Bar
                st.progress(st.session_state.request_count / DAILY_LIMIT)
                render_confidence_ui(conf)
                f_resp = header + "\n" + response
                
                # HTML PDF Gen
                pdf_h = generate_brief_html(
                    p_name=p_name,
                    judgment_date=case_res.get("judgment_date", "N/A"),
                    case_type=case_res.get("case_subtype", "N/A"),
                    user_question=active_q,
                    header_content=header,
                    assistant_response=response
                )
                st.download_button(label=":material/picture_as_pdf: Download Case Brief (PDF)", data=pdf_h.encode("utf-8"), file_name=f"{p_name[:30]}.html", mime="text/html", key=f"pdf_l_{msg_idx}", type="primary")
                p_pay = {"html": pdf_h, "name": p_name}

        if len(top_c) > 0: render_similar_cases_ui(top_c, msg_idx + 1, is_live=True)
        st.session_state.messages.append(("bot", f_resp, top_c, p_pay, conf))
        if "history" not in st.session_state: st.session_state.history = []
        st.session_state.history.append({"query": active_q, "response": f_resp, "cases": top_c, "confidence": conf})


# =================================================
# ANALYTICS PAGE
# Shows insights like cases per year
# =================================================


elif page == "Analytics":

    st.markdown("<h1 class='unified-h1'><i class='fa-solid fa-chart-simple'></i> Analytics</h1>", unsafe_allow_html=True)

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
    col1, col2, col3 = st.columns(3)
    col1.metric(":material/folder: Total Unique Cases", len(df["case_name"].unique()))
    col2.metric(":material/category: Total Categories", df["case_subtype"].nunique())
    col3.metric(":material/description: Total QA Records", len(df))

    st.divider()

    # ================= YEAR CHART =================
    st.markdown("<h3 class='unified-h3'><i class='fa-solid fa-calendar-days'></i> Unique Cases By Judgment Year</h3>", unsafe_allow_html=True)

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
    st.bar_chart(year_df.set_index("Judgment Year"), height=350)
    
    st.divider()

    # ================= CASE TYPE PIE CHART =================
    st.markdown("<h3 class='unified-h3'><i class='fa-solid fa-chart-pie'></i> Case Distribution By Year</h3>", unsafe_allow_html=True)
    
    # Get available years
    available_years = sorted(unique_cases[unique_cases["Judgment Year"] != "Unknown"]["Judgment Year"].unique().tolist(), reverse=True)
    year_options = ["All Years"] + available_years
    
    # Selectbox for year
    selected_year = st.selectbox("Select Judgment Year to view distribution:", year_options)
    
    # Filter dataset
    if selected_year == "All Years":
        pie_data = unique_cases
    else:
        pie_data = unique_cases[unique_cases["Judgment Year"] == selected_year]
        
    if pie_data.empty:
        st.warning(f"No case data available for {selected_year}.")
    else:
        # Count Unique Cases by Type
        type_counts = pie_data["case_subtype"].value_counts().reset_index()
        type_counts.columns = ["Case Type", "Case Counts"]
    
        # Create Interactive Donut Chart

        
        # Base pie chart
        pie_chart = alt.Chart(type_counts).mark_arc(innerRadius=90, stroke="#020617", strokeWidth=1).encode(
            theta=alt.Theta(field="Case Counts", type="quantitative"),
            color=alt.Color(
                field="Case Type", 
                type="nominal", 
                sort=alt.EncodingSortField(field="Case Counts", order="descending"),
                scale=alt.Scale(scheme='category20'), 
                legend=alt.Legend(
                    title="Legal Domain", 
                    labelLimit=800,
                    symbolLimit=50,
                    columns=2,          # Split into two columns to save vertical space
                    labelFontSize=13,   # Increase readability
                    titleFontSize=14,
                    padding=20          # Add breathing room
                )
            ),
            order=alt.Order(field="Case Counts", sort="descending"),
            tooltip=['Case Type', 'Case Counts']
        ).properties(height=600)
        
        st.altair_chart(pie_chart, use_container_width=True)


# =================================================
# DATASET EXPLORER
# Allows filtering, searching, and viewing cases
# =================================================


elif page == "Dataset":

    st.markdown("<h1 class='unified-h1'><i class='fa-solid fa-folder-open'></i> Dataset</h1>", unsafe_allow_html=True)

    # ================= UNIQUE CASE TABLE =================

    unique_cases = df[["case_name", "judgment_date", "case_subtype"]].drop_duplicates(subset=["case_name"])

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

    st.markdown("<h3 class='unified-h3'><i class='fa-solid fa-calendar-day'></i> Filter Cases</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Cascading Logic:
    # 1. Select Year -> updates Month options
    # 2. Select Month -> updates Case Type options

    # Year Options
    year_options = sorted(unique_cases["Year"].dropna().unique().tolist())
    year_options = [int(y) for y in year_options] # Display as integers
    selected_year = col1.selectbox("Select Year", ["All"] + year_options)

    # Filter data for Month options
    month_data = unique_cases.copy()
    if selected_year != "All":
        month_data = month_data[month_data["Year"] == selected_year]
    
    month_options = sorted(month_data["Month"].dropna().unique().tolist())
    selected_month = col2.selectbox("Select Month", ["All"] + month_options)

    # Filter data for Case Type options
    type_data = month_data.copy()
    if selected_month != "All":
        type_data = type_data[type_data["Month"] == selected_month]
    
    type_options = sorted(type_data["case_subtype"].dropna().unique().tolist())
    selected_type = col3.selectbox("Select Case Type", ["All"] + type_options)

    # ================= APPLY FILTER =================

    filtered_cases = unique_cases.copy()

    if selected_year != "All":
        filtered_cases = filtered_cases[filtered_cases["Year"] == selected_year]

    if selected_month != "All":
        filtered_cases = filtered_cases[filtered_cases["Month"] == selected_month]

    if selected_type != "All":
        filtered_cases = filtered_cases[filtered_cases["case_subtype"] == selected_type]

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
        "judgment_date": "Judgment Date",
        "case_subtype": "Case Type"
    }, inplace=True)

    # ================= CREATE SERIAL NUMBER =================

    display_cases.index += 1
    display_cases.index.name = "Sr No"

    # ================= DISPLAY TABLE =================

    st.markdown("<h3 class='unified-h3'><i class='fa-solid fa-list'></i> Filtered Cases</h3>", unsafe_allow_html=True)

    st.dataframe(
        display_cases,
        use_container_width=True,
        height=450
    )

    # ================= SEARCH + SELECT =================
    st.write(f"Showing {len(display_cases)} cases")
    st.markdown("<h3 class='unified-h3'><i class='fa-solid fa-magnifying-glass'></i> Search & Open Case</h3>", unsafe_allow_html=True)

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

        st.markdown("<h3 class='unified-h3'><i class='fa-solid fa-scale-balanced'></i> Case Details</h3>", unsafe_allow_html=True)

        st.write("**Case Name:**", selected_case)
        st.write("**Judgment Date:**", case_data.iloc[0]["judgment_date"])

        st.markdown("---")

        for _, row in case_data.iterrows():
            st.markdown(f"**Q:** {row['question']}")
            st.markdown(f"**A:** {row['answer']}")
            st.markdown("---")

elif page == "About":
    st.markdown("""
<style>
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.95) translateY(20px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.about-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.75) 100%);
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    animation: popIn 0.6s ease-out forwards;
    opacity: 0;
}

.about-delay-1 { animation-delay: 0.1s; }
.about-delay-2 { animation-delay: 0.2s; }
.about-delay-3 { animation-delay: 0.3s; }
.about-delay-4 { animation-delay: 0.4s; }

.about-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 45px rgba(0,0,0,0.6), inset 0 0 20px rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.3);
}

.about-card-title {
    margin-top: 0 !important;
    display: flex;
    align-items: center;
    gap: 15px;
    color: #f8fafc !important;
    font-size: 1.6rem !important;
    font-weight: 700;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.about-card-title i {
    color: #34d399;
}

.title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    font-size: 4.8rem !important; 
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #3b82f6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 1.5rem 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -1.5px !important;
    animation: popIn 0.8s ease-out forwards;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.3rem;
    margin-bottom: 3.5rem;
    font-weight: 300;
    text-align: center;
    animation: popIn 0.6s ease-out forwards;
}

.about-card p, .about-card li {
    color: #cbd5e1 !important;
    line-height: 1.8;
    font-size: 1.05rem;
}

.about-card ul {
    padding-left: 1.5rem;
}

.about-card li {
    margin-bottom: 0.75rem;
}

.tag {
    background: rgba(15, 23, 42, 0.8);
    padding: 8px 18px;
    border-radius: 12px;
    margin: 6px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    font-size: 0.95rem;
    font-weight: 500;
    color: #60a5fa;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

.tag:hover {
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateY(-2px);
    color: #93c5fd;
    box-shadow: 0 6px 15px rgba(59, 130, 246, 0.2);
}
</style>

<h1 class="unified-h1" style="animation: popIn 0.8s ease-out forwards;"><i class='fa-solid fa-scale-balanced'></i> AI Legal Assistant</h1>
<p class="home-subtitle" style="margin-bottom: 3.5rem; max-width: 100%; text-align: left; animation: popIn 0.6s ease-out forwards;">An advanced intelligent information system for Indian Case Law analysis.</p>
""", unsafe_allow_html=True)

    # Section 1: About the Project
    st.markdown("""
<div class="about-card about-delay-1">
    <div class="about-card-title"><i class='fa-solid fa-briefcase'></i> About the Project</div>
    <p>
        <strong>AI Legal Assistant</strong> is a revolutionary system designed to democratize and simplify access to Indian legal precedents. 
        The project actively bridges the gap between highly complex legal jargon and everyday individuals by transforming lengthy, 
        dense court judgments into structured, perfectly formatted, and easy-to-understand explanations.
    </p>
    <p>
        The platform leverages localized Mistral Large Language Models, Natural Language Processing (NLP), and FAISS-based semantic vector searches 
        to yield extreme accuracy. Users can chat using simple, everyday English, and the application instantly retrieves 
        the most relevant case from a sprawling legal dataset to contextualize the answer.
    </p>
</div>
""", unsafe_allow_html=True)

    # Section 2: Core Concept
    st.markdown("""
<div class="about-card about-delay-2">
    <div class="about-card-title"><i class='fa-solid fa-brain'></i> Core Concept</div>
    <p>
        Legal documents consistently pose challenges precisely because they are heavily technical. This system resolves that pain-point using a specialized pipeline:
    </p>
    <ul>
        <li><strong>Extraction:</strong> Rapid keyword and semantic vector analysis of over 10,000 Indic legal cases.</li>
        <li><strong>Processing:</strong> Injecting case content into specifically prompt-engineered Mistral templates.</li>
        <li><strong>Simplification:</strong> Yielding cleanly bulleted overviews completely free of disjointed legal jargon.</li>
    </ul>
    <p style='margin-top: 1.5rem; margin-bottom: 0.5rem;'><b>Strictly Structured Outputs:</b></p>
    <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
        <span class="tag"><i class="fa-solid fa-receipt"></i> What Happened</span>
        <span class="tag"><i class="fa-solid fa-book-open"></i> Background</span>
        <span class="tag"><i class="fa-solid fa-gavel"></i> Legal Issue</span>
        <span class="tag"><i class="fa-solid fa-scale-unbalanced"></i> Court Decision</span>
        <span class="tag"><i class="fa-solid fa-lightbulb"></i> Reason</span>
        <span class="tag"><i class="fa-solid fa-flag-checkered"></i> Outcome</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Section 3 & 4
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-card about-delay-3" style="height: 96%;">
            <div class="about-card-title"><i class='fa-solid fa-bullseye'></i> Core Capabilities</div>
            <ul>
                <li><strong>Hybrid Vector Search:</strong> Finds matches even if you don't use strict legal terminology.</li>
                <li><strong>Structured Generation:</strong> Enforces distinct summary points for supreme clarity.</li>
                <li><strong>Conversational UI:</strong> Feels like texting a specialized legal mentor.</li>
                <li><strong>Data Analytics:</strong> Visualize trends in historical courts and judgment years.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card about-delay-4" style="height: 96%;">
            <div class="about-card-title"><i class='fa-solid fa-triangle-exclamation'></i> Scope & Limitations</div>
            <ul>
                <li><strong>Static Knowledge:</strong> The dataset is locked and does not pull live cases from the web.</li>
                <li><strong>Educational Purpose:</strong> This tool is to learn concepts, not to substitute a real lawyer.</li>
                <li><strong>No Legal Advice:</strong> Extrapolations made by the AI should NOT be used in a real courts.</li>
                <li><strong>Token Limits:</strong> Extremely long cases may have their endings truncated.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    @keyframes floatingBadge {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }
    
    .unique-footer-container {
        display: flex;
        justify-content: center;
        width: 100%;
        padding-top: 25px;
        padding-bottom: 50px;
        animation: popIn 0.8s ease-out forwards, floatingBadge 6s ease-in-out infinite;
        opacity: 0;
        animation-delay: 0.5s, 1.3s;
    }
    
    .footer-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 50px;
        padding: 6px 24px 6px 6px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4), inset 0 0 20px rgba(52, 211, 153, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: default;
    }
    
    .footer-badge:hover {
        background: rgba(15, 23, 42, 0.7);
        border-color: rgba(52, 211, 153, 0.5);
        box-shadow: 0 15px 35px rgba(0,0,0,0.5), inset 0 0 25px rgba(52, 211, 153, 0.15);
        transform: scale(1.03);
    }
    
    .footer-badge-icon {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        border-radius: 50%;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.2rem;
        margin-right: 18px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .footer-badge-icon::after {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: linear-gradient(transparent, rgba(255,255,255,0.3), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; top: -100%; }
        100% { left: 100%; top: 100%; }
    }
    
    .footer-badge-text {
        display: flex;
        flex-direction: column;
        text-align: left;
    }
    
    .footer-badge-main {
        color: #f8fafc;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer-badge-sub {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 400;
        margin-top: 2px;
    }
    </style>
    
    <div class="unique-footer-container">
        <div class="footer-badge">
            <div class="footer-badge-icon"><i class="fa-solid fa-shield-halved"></i></div>
            <div class="footer-badge-text">
                <span class="footer-badge-main">AI Legal Research Engine</span>
                <span class="footer-badge-sub">Strictly for educational exploration. Does not constitute certified legal counsel.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
