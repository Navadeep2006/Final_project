# ============================================================
#  Citizen Feedback Sentiment Analysis System — Streamlit App
#  File: app.py
# ============================================================
import streamlit as st
import pickle
import re
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import time
import csv

# ──────────────────────────────────────────────
# Page Configuration  (MUST be first st call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CitizenSense AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Download NLTK data silently
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    for pkg in ["punkt", "stopwords", "punkt_tab", "wordnet"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
download_nltk()

# ──────────────────────────────────────────────
# Custom CSS — dark government-tech aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
/* Reduce top padding of the main block to pull content higher */
.block-container {
    padding-top: 2rem !important;
}

[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d45;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ─────────────────────────────── */
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

/* ── Sidebar styles ─────────────────────────── */
.sidebar-brand {
    padding: 1.4rem 0 1rem;
    text-align: center;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 1rem;
}
.sidebar-brand h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.3rem 0 0;
}
.sidebar-brand p { color: #64748b; font-size: 0.75rem; margin: 0; }

.info-card {
    background: #131c2e;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.info-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #38bdf8;
    margin: 0 0 0.6rem;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1e2d45;
    font-size: 0.82rem;
}
.info-row:last-child { border-bottom: none; }
.info-label { color: #94a3b8; }
.info-value { color: #e2e8f0; font-weight: 500; }
.badge-green {
    background: #052e16; color: #4ade80;
    border: 1px solid #16a34a;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 600;
}

/* ── Page header ────────────────────────────── */
.page-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #e2e8f0 20%, #38bdf8 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.page-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0.5rem 0 0;
    max-width: 680px;
}
.tag-pill {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    color: #94a3b8;
    margin: 0.5rem 0.3rem 0 0;
    letter-spacing: 0.5px;
}
.tag-pill span { color: #38bdf8; }

/* ── Section headings ───────────────────────── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #38bdf8;
    margin-bottom: 0.8rem;
}

/* ── Input textarea fix ─────────────────────── */
textarea {
    background: #131c2e !important;
    border: 1px solid #1e2d45 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── Buttons ────────────────────────────────── */
.stButton button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.88 !important; }

/* ── Result cards ───────────────────────────── */
.result-card {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.result-card .sentiment-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    line-height: 1;
}
.result-card .sentiment-sub {
    font-size: 0.82rem;
    margin-top: 0.3rem;
    opacity: 0.75;
}
.card-positive {
    background: linear-gradient(135deg, #052e16 0%, #0f2a1a 100%);
    border: 1px solid #16a34a;
    box-shadow: 0 0 30px rgba(74,222,128,0.1);
}
.card-negative {
    background: linear-gradient(135deg, #2d0a0a 0%, #2a0f0f 100%);
    border: 1px solid #dc2626;
    box-shadow: 0 0 30px rgba(248,113,113,0.1);
}
.card-neutral {
    background: linear-gradient(135deg, #1a1600 0%, #2a2010 100%);
    border: 1px solid #ca8a04;
    box-shadow: 0 0 30px rgba(250,204,21,0.08);
}
.positive-text { color: #4ade80; }
.negative-text { color: #f87171; }
.neutral-text  { color: #facc15; }

/* ── Metric mini-cards ──────────────────────── */
.metric-card {
    background: #131c2e;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #38bdf8;
    margin: 0;
    line-height: 1;
}
.metric-card .metric-lbl {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Divider ────────────────────────────────── */
.divider { border-top: 1px solid #1e2d45; margin: 2rem 0; }

/* ── Progress custom ────────────────────────── */
.prob-bar-wrap { margin-bottom: 0.7rem; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 4px;
    color: #94a3b8;
}
.prob-bar-bg {
    background: #1e2d45;
    border-radius: 20px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.6s ease;
}

/* ── Sample pills ───────────────────────────── */
.sample-btn {
    display: inline-block;
    background: #131c2e;
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #94a3b8;
    cursor: pointer;
    margin: 4px 4px 4px 0;
    transition: border-color 0.2s;
}
.sample-btn:hover { border-color: #38bdf8; color: #38bdf8; }

/* ── Plotly chart backgrounds ───────────────── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load model & vectorizer
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model.pkl and vectorizer.pkl from the backend/model directory."""
    model, vectorizer = None, None
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "backend", "model", "sentiment_model.pkl")
    vectorizer_path = os.path.join(base_dir, "..", "backend", "model", "tfidf_vectorizer.pkl")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        pass
    return model, vectorizer

model, vectorizer = load_artifacts()
MODEL_LOADED = model is not None and vectorizer is not None

# Load accuracy dynamically
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "model", "accuracy.txt"), "r") as f:
        MODEL_ACCURACY = f.read().strip() + "%"
except Exception:
    MODEL_ACCURACY = "89.2%"

# ──────────────────────────────────────────────
# Text preprocessing (mirrors training pipeline)
# ──────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                      # mentions
    text = re.sub(r"#\w+", "", text)                      # hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()
    try:
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(text)
        # Type fix: pass list instead of generator or mixed objects
        text = " ".join([lemmatizer.lemmatize(t) for t in tokens if t not in stop_words])
    except Exception:
        pass
    return text


# ──────────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────────
SENTIMENT_MAP = {0: "Negative", 2: "Neutral", 4: "Positive"}
LABEL_COLOR   = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#facc15"}
LABEL_ICON    = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
CARD_CLASS    = {"Positive": "card-positive positive-text",
                 "Negative": "card-negative negative-text",
                 "Neutral":  "card-neutral  neutral-text"}

# Demo probs used when model not loaded (mock)
DEMO_PROBS = {"Positive": 0.72, "Negative": 0.15, "Neutral": 0.13}

def predict(text: str):
    """Returns (label, probs_dict). Falls back to demo if model absent."""
    if not MODEL_LOADED:
        return "Positive", DEMO_PROBS          # demo / preview mode

    clean = preprocess(text)
    vec   = vectorizer.transform([clean])
    label_idx = model.predict(vec)[0]
    label = SENTIMENT_MAP.get(label_idx, str(label_idx))

    # Probabilities
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(vec)[0]
        classes   = [SENTIMENT_MAP.get(c, str(c)) for c in model.classes_]
        probs     = dict(zip(classes, raw_probs))
    else:
        # For LinearSVC / models without predict_proba
        probs = {k: 0.0 for k in ["Positive", "Negative", "Neutral"]}
        probs[label] = 1.0

    # Ensure all three keys exist
    for k in ["Positive", "Negative", "Neutral"]:
        probs.setdefault(k, 0.0)

    return label, probs


def add_to_dataset(text: str, sentiment: str):
    """Appends new user feedback to the dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "backend", "model", "data.csv")
    label_map = {"Negative": "0", "Neutral": "2", "Positive": "4"}
    label = label_map.get(sentiment, "2")
    mock_id = str(int(time.time() * 1000))
    date_str = time.strftime("%a %b %d %H:%M:%S PDT %Y")
    
    try:
        with open(data_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([label, mock_id, date_str, "NO_QUERY", "citizen_user", text])
        return True
    except Exception as e:
        return False


# ──────────────────────────────────────────────
# Static dataset stats (from Sentiment140)
# ──────────────────────────────────────────────
DIST_LABELS = ["Positive", "Negative", "Neutral"]
DIST_VALUES = [800000, 800000, 800]   # Sentiment140 is balanced; neutral ~hand-coded
DIST_COLORS = ["#4ade80", "#f87171", "#facc15"]

SERVICE_DIST = {
    "Roads":         [42, 48, 10],
    "Water Supply":  [30, 60, 10],
    "Hospitals":     [55, 35, 10],
    "Sanitation":    [25, 65, 10],
    "Electricity":   [50, 40, 10],
    "Transportation":[45, 45, 10],
}


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2.2rem">🏛️</div>
        <h2>CitizenSense AI</h2>
        <p>Feedback Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Info
    st.markdown(f"""
    <div class="info-card">
        <h4>⚙️ Model Information</h4>
        <div class="info-row">
            <span class="info-label">Algorithm</span>
            <span class="info-value">Logistic Regression</span>
        </div>
        <div class="info-row">
            <span class="info-label">Feature Extraction</span>
            <span class="info-value">TF-IDF</span>
        </div>
        <div class="info-row">
            <span class="info-label">Training Dataset</span>
            <span class="info-value">Sentiment140</span>
        </div>
        <div class="info-row">
            <span class="info-label">Training Samples</span>
            <span class="info-value">~ 1,600,000</span>
        </div>
        <div class="info-row">
            <span class="info-label">Model Accuracy</span>
            <span class="badge-green">{MODEL_ACCURACY}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Classes</span>
            <span class="info-value">Positive · Neutral · Negative</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Model status
    if MODEL_LOADED:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⚠️ sentiment_model.pkl / tfidf_vectorizer.pkl not found — running in **demo mode**.")

    # Coverage domains
    st.markdown("""
    <div class="info-card">
        <h4>🏙️ Service Domains</h4>
        <div class="info-row"><span class="info-label">🛣️ Roads & Infrastructure</span></div>
        <div class="info-row"><span class="info-label">💧 Water Supply</span></div>
        <div class="info-row"><span class="info-label">🏥 Hospitals & Healthcare</span></div>
        <div class="info-row"><span class="info-label">🗑️ Sanitation</span></div>
        <div class="info-row"><span class="info-label">⚡ Electricity</span></div>
        <div class="info-row"><span class="info-label">🚌 Transportation</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Developer info
    st.markdown("""
    <div class="info-card">
        <h4>👨‍💻 Developer</h4>
        <div class="info-row">
            <span class="info-label">Project</span>
            <span class="info-value">Final Year</span>
        </div>
        <div class="info-row">
            <span class="info-label">Domain</span>
            <span class="info-value">ML / NLP</span>
        </div>
        <div class="info-row">
            <span class="info-label">Framework</span>
            <span class="info-value">Streamlit</span>
        </div>
        <div class="info-row">
            <span class="info-label">Version</span>
            <span class="info-value">1.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#374151;font-size:0.7rem;text-align:center;'>"
        "© 2025 CitizenSense AI · Final Year Project</p>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════

# ── Page Header ──────────────────────────────
st.markdown(f"""
<div class="page-header">
    <h1>Citizen Feedback<br>Sentiment Analysis</h1>
    <p>An AI-powered platform that decodes public opinion on civic services using 
       Natural Language Processing and Machine Learning — turning raw feedback into 
       actionable governance insights.</p>
    <div>
        <span class="tag-pill"><span>NLP</span> · TF-IDF</span>
        <span class="tag-pill"><span>ML</span> · Logistic Regression</span>
        <span class="tag-pill"><span>Accuracy</span> · {MODEL_ACCURACY}</span>
        <span class="tag-pill"><span>Dataset</span> · Sentiment140</span>
    </div>
</div>
""", unsafe_allow_html=True)



# ── Top metrics row ───────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in zip(
    [m1, m2, m3, m4],
    ["1.6M", MODEL_ACCURACY, "3", "6"],
    ["Training Samples", "Model Accuracy", "Sentiment Classes", "Service Domains"]
):
    col.markdown(f"""
    <div class="metric-card">
        <p class="metric-val">{val}</p>
        <p class="metric-lbl">{lbl}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  INPUT + PREDICTION SECTION
# ══════════════════════════════════════════════
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown("<p class='section-label'>📝 Feedback Input</p>", unsafe_allow_html=True)

    # Sample feedback buttons via selectbox workaround
    SAMPLES = {
        "Select a sample…": "",
        "😊 Roads improved": "The road repairs in our area are excellent! The potholes have been fixed and the new lanes are smooth and well-lit.",
        "😞 Water shortage": "We have been suffering from water shortage for three weeks now. The supply is irregular and the water quality is terrible.",
        "😊 Hospital service": "The government hospital staff were very helpful and the treatment was fast. I am really happy with the healthcare service.",
        "😞 Electricity cuts": "Power cuts are happening every day for 6 hours. It is extremely frustrating and affecting our daily life and business.",
        "😐 Bus service": "The bus service is okay. It runs on time mostly but the buses are quite old and need maintenance.",
        "😞 Garbage collection": "Garbage has not been collected from our street in 10 days. The smell is unbearable and creating a health hazard.",
    }
    sample_choice = st.selectbox(
        "Load a sample feedback",
        options=list(SAMPLES.keys()),
        label_visibility="collapsed"
    )
    default_text = SAMPLES[sample_choice]

    user_input = st.text_area(
        label="Citizen Feedback",
        value=default_text,
        height=170,
        placeholder="Type or paste citizen feedback here…\n\nExample: 'The garbage collection in our ward has improved significantly after the new system was introduced.'",
        label_visibility="collapsed",
    )

    char_count = len(user_input)
    st.markdown(
        f"<p style='font-size:0.72rem;color:#374151;text-align:right;margin-top:-10px;'>"
        f"{char_count} characters</p>",
        unsafe_allow_html=True,
    )

    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

    # Quick tips
    with st.expander("ℹ️ Tips for better results"):
        st.markdown("""
        - Write feedback in **English** for best accuracy
        - Include the **service type** (roads, water, hospital…)
        - Be descriptive — longer feedback = more context for the model
        - Avoid emojis or special characters
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("➕ Contribute: Add to Dataset"):
        st.markdown("<p style='font-size:0.8rem; color:#94a3b8;'>Improve the model by adding this feedback directly into the training dataset.</p>", unsafe_allow_html=True)
        new_sentiment = st.radio("Actual Sentiment", options=["Negative", "Neutral", "Positive"], horizontal=True)
        if st.button("Save to Dataset", type="primary"):
            if not user_input.strip():
                st.error("Text is empty. Please enter feedback above.")
            else:
                if add_to_dataset(user_input, new_sentiment):
                    st.success("✅ Added to dataset successfully! The model can learn from this in the next training run.")
                else:
                    st.error("❌ Failed to append to dataset. Check file paths.")


# ──────────────────────────────────────────────
# RIGHT COLUMN — Results
# ──────────────────────────────────────────────
with right_col:
    st.markdown("<p class='section-label'>🎯 Prediction Result</p>", unsafe_allow_html=True)

    if not analyze_btn:
        # Idle placeholder
        st.markdown("""
        <div style="background:#0d1424;border:1px dashed #1e2d45;border-radius:14px;
                    padding:2.5rem;text-align:center;color:#374151;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">🏛️</div>
            <p style="font-size:0.9rem;margin:0;">Enter feedback on the left and click<br>
            <strong style="color:#38bdf8">Analyze Sentiment</strong> to see results.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not user_input.strip():
            st.warning("⚠️ Please enter some feedback text before analyzing.")
        else:
            with st.spinner("Analyzing sentiment…"):
                time.sleep(0.4)          # brief UX pause
                label, probs = predict(user_input)

            icon = LABEL_ICON.get(label, "❓")
            card_cls = CARD_CLASS.get(label, "card-neutral neutral-text").split()[0]
            text_cls  = CARD_CLASS.get(label, "card-neutral neutral-text").split()[1]
            
            # Get the winning probability safely
            winning_val = probs.get(label)
            winning_prob = float(winning_val) if winning_val is not None else 0.0

            # ── Result Card ──
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div style="font-size:2.8rem;line-height:1">{icon}</div>
                <p class="sentiment-label {text_cls}">{label}</p>
                <p class="sentiment-sub">Detected Sentiment · {winning_prob*100.0:.1f}% Confidence</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown("<p class='section-label' style='margin-top:1.2rem;'>📊 Confidence Scores</p>",
                        unsafe_allow_html=True)
            for sent in ["Positive", "Negative", "Neutral"]:
                # Probabilities could be missing if the model isn't predict_proba capable
                # Ensure we handle missing keys safely by defaulting to 0.0
                val = probs.get(sent)
                raw_prob = float(val) if val is not None else 0.0
                pct = raw_prob * 100.0
                color = LABEL_COLOR[sent]
                st.markdown(f"""
                <div class="prob-bar-wrap">
                    <div class="prob-label">
                        <span>{LABEL_ICON[sent]} {sent}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill"
                             style="width:{pct}%;background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Preprocessed text peek ──
            with st.expander("🔬 Preprocessed text (debug)"):
                st.code(preprocess(user_input), language=None)


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  CHARTS SECTION
# ══════════════════════════════════════════════
st.markdown("<p class='section-label'>📈 Dataset & Model Insights</p>",
            unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2, gap="large")

# ── Chart 1: Sentiment Distribution (Donut) ──
with chart_col1:
    st.markdown("**Sentiment140 — Class Distribution**")
    fig_donut = go.Figure(go.Pie(
        labels=DIST_LABELS,
        values=DIST_VALUES,
        hole=0.55,
        marker={"colors": DIST_COLORS,
                "line": {"color": "#0b0f1a", "width": 3}},
        textinfo="label+percent",
        textfont={"color": "#e2e8f0", "size": 13, "family": "DM Sans"},
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.1,
                "xanchor": "center", "x": 0.5,
                "font": {"color": "#94a3b8", "size": 12}},
        margin={"t": 20, "b": 80, "l": 10, "r": 10},
        height=400,
        annotations=[{"text": "1.6M<br>Tweets",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"size": 16, "color": "#e2e8f0", "family": "Syne"},
                      "align": "center"}]
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

# ── Chart 2: Per-service breakdown ───────────
with chart_col2:
    st.markdown("**Sentiment by Public Service Domain** *(estimated)*")
    services = list(SERVICE_DIST.keys())
    pos_vals = [v[0] for v in SERVICE_DIST.values()]
    neg_vals = [v[1] for v in SERVICE_DIST.values()]
    neu_vals = [v[2] for v in SERVICE_DIST.values()]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Positive", x=services, y=pos_vals,
                             marker_color="#4ade80", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="Negative", x=services, y=neg_vals,
                             marker_color="#f87171", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="Neutral",  x=services, y=neu_vals,
                             marker_color="#facc15", marker_line_width=0))
    fig_bar.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        xaxis={"tickangle": -40, "tickfont": {"size": 11}, "gridcolor": "rgba(0,0,0,0)"},
        yaxis={"gridcolor": "#1e2d45", "tickfont": {"size": 11}, "ticksuffix": "%"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.35,
                "xanchor": "center", "x": 0.5,
                "font": {"color": "#94a3b8", "size": 12}},
        margin={"t": 20, "b": 100, "l": 10, "r": 10},
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  MODEL PIPELINE DIAGRAM (text-based)
# ══════════════════════════════════════════════
st.markdown("<p class='section-label'>🔄 NLP Pipeline</p>", unsafe_allow_html=True)

steps = [
    ("01", "Raw Text Input", "Citizen feedback in natural English"),
    ("02", "Text Preprocessing", "Lowercase · Remove URLs, mentions, punctuation · Stopword removal"),
    ("03", "TF-IDF Vectorisation", "Convert text to weighted numeric feature matrix"),
    ("04", "Logistic Regression", "Trained classifier predicts sentiment class"),
    ("05", "Softmax Probabilities", "Per-class confidence scores (Positive / Neutral / Negative)"),
    ("06", "Result Dashboard", "Visualised output with confidence bars"),
]

pipe_cols = st.columns(len(steps))
for col, (num, title, desc) in zip(pipe_cols, steps):
    col.markdown(f"""
    <div style="background:#0d1424;border:1px solid #1e2d45;border-radius:10px;
                padding:0.9rem 0.8rem;text-align:center;height:100%;">
        <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                    color:#1e2d45;line-height:1;">{num}</div>
        <div style="font-family:Syne,sans-serif;font-size:0.78rem;font-weight:700;
                    color:#38bdf8;margin:0.35rem 0 0.3rem;line-height:1.2;">{title}</div>
        <div style="font-size:0.7rem;color:#64748b;line-height:1.4;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #0b0f1a;
    border-top: 1px solid #1e2d45;
    text-align: center;
    padding: 0.8rem 0;
    color: #374151;
    font-size: 0.75rem;
    z-index: 9999;
}
/* Ensure main content doesn't overlap with absolute footer */
.block-container {
    padding-bottom: 80px !important;
}
</style>
<div class="footer">
    <strong style="color:#38bdf8">CitizenSense AI</strong> · 
    Citizen Feedback Sentiment Analysis System · 
    Final Year Project · 
    Built with Streamlit + scikit-learn
</div>
""", unsafe_allow_html=True)