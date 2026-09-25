# ==============================================================================
# Application: app.py
# Description: Production-grade interactive Streamlit web application for Local
# Brand Sentiment Analysis. Features an executive UI design with real-time
# multi-engine sentiment prediction, interactive Plotly visualizations,
# confidence score gauges, keyword highlighting, EDA charts, and dataset explorer.
# ==============================================================================

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Determine absolute path to the directory containing this script
ROOT_DIR: Path = Path(__file__).resolve().parent

# Append the internal 'src' directory to Python's system path for module imports
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import text preprocessing and lexicon functions from sentiment_utils
from sentiment_utils import clean_text, lexicon_sentiment, extract_aspect, is_noise, POSITIVE_WORDS, NEGATIVE_WORDS

# Import the inference predictor class from predict module
from predict import SentimentPredictor


# ------------------------------------------------------------------------------
# 1. STREAMLIT APPLICATION CONFIGURATION & PREMIUM STYLING
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="BrandSense AI • Local Brand Sentiment Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for an executive dark-glass aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top decoration bar & general background polish */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }

    /* Executive Hero Header */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.12) 0%, rgba(99, 102, 241, 0.08) 50%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F8FAFC 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #CBD5E1;
        margin-right: 8px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
    }

    /* Executive Metric Card Container Styling */
    .metric-card-pro {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        color: #F8FAFC;
        box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }
    .metric-card-pro:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 28px -6px rgba(56, 189, 248, 0.15);
    }
    .metric-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .metric-card-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-card-icon {
        font-size: 1.25rem;
        opacity: 0.9;
    }
    .metric-card-value {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .metric-card-subtext {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }

    /* Sentiment Result Highlight Cards */
    .sentiment-result-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 10px;
    }
    .badge-positive {
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.8) 0%, rgba(4, 120, 87, 0.9) 100%);
        color: #A7F3D0;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #10B981;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }
    .badge-neutral {
        background: linear-gradient(135deg, rgba(55, 65, 81, 0.8) 0%, rgba(75, 85, 99, 0.9) 100%);
        color: #E2E8F0;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #64748B;
        box-shadow: 0 4px 14px rgba(100, 116, 139, 0.25);
    }
    .badge-negative {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.8) 0%, rgba(190, 18, 60, 0.9) 100%);
        color: #FECDD3;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #F43F5E;
        box-shadow: 0 4px 14px rgba(244, 63, 94, 0.25);
    }

    /* Keyword highlight styling */
    .token-pos {
        background: rgba(16, 185, 129, 0.22);
        color: #34D399;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .token-neg {
        background: rgba(244, 63, 94, 0.22);
        color: #FB7185;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* Probability Micro Bar */
    .prob-bar-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94A3B8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.08) !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0B0F17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS FOR OPTIMAL PERFORMANCE
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Sentiment Inference Pipeline...")
def get_predictor() -> Optional[SentimentPredictor]:
    """Instantiate and cache the SentimentPredictor model singleton."""
    try:
        return SentimentPredictor()
    except Exception as exc:
        st.error(f"Failed to load trained model artifacts: {exc}")
        return None


@st.cache_data(show_spinner="Loading Survey Results Data...")
def load_survey_results() -> Optional[pd.DataFrame]:
    """Read and cache preprocessed survey sentiment results from disk."""
    results_path: Path = ROOT_DIR / 'outputs' / 'sentiment_results.csv'
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


@st.cache_data(show_spinner="Loading Model Performance Metrics...")
def load_metrics_data() -> Optional[Dict[str, Any]]:
    """Read and cache training performance metrics JSON from disk."""
    metrics_path: Path = ROOT_DIR / 'outputs' / 'model_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------------------
# 3. SENTIMENT KEYWORD HIGHLIGHTING ENGINE
# ------------------------------------------------------------------------------

def highlight_sentiment_tokens(text: str) -> str:
    """Scan tokens and render color-coded spans for positive and negative words."""
    cleaned = clean_text(text)
    tokens: List[str] = cleaned.split()
    formatted_words: List[str] = []

    for word in tokens:
        if word in POSITIVE_WORDS:
            formatted_words.append(f"<span class='token-pos'>✓ {word}</span>")
        elif word in NEGATIVE_WORDS:
            formatted_words.append(f"<span class='token-neg'>✕ {word}</span>")
        else:
            formatted_words.append(word)

    return " ".join(formatted_words) if formatted_words else "<em>No text tokens found</em>"


# ------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & SYSTEM STATUS
# ------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
        <span style="font-size: 1.8rem;">⚡</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #F8FAFC;">BrandSense AI</div>
            <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 600;">SENTIMENT INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; margin-bottom: 16px; font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
        Production NLP pipeline analyzing consumer perception, brand reputation, and satisfaction across Instagram & Twitter empirical survey data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Engine Health")
    model_exists = (ROOT_DIR / 'models' / 'sentiment_model.joblib').exists()
    results_exist = (ROOT_DIR / 'outputs' / 'sentiment_results.csv').exists()

    st.markdown(f"""
    <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.88rem; background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #CBD5E1;">ML Logistic Model</span>
            <span style="color: {'#34D399' if model_exists else '#F43F5E'}; font-weight: 700;">{'● ONLINE' if model_exists else '○ OFFLINE'}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #CBD5E1;">Survey Dataset</span>
            <span style="color: {'#34D399' if results_exist else '#F43F5E'}; font-weight: 700;">{'● LOADED' if results_exist else '○ MISSING'}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #CBD5E1;">Hybrid Consensus</span>
            <span style="color: #34D399; font-weight: 700;">● ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; text-align: center;">
        Scikit-Learn • TF-IDF • Streamlit • Python 3.13<br>
        © 2026 BrandSense Intelligence
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 5. MAIN DASHBOARD HERO & EXECUTIVE KPI METRICS
# ------------------------------------------------------------------------------

# Render Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Local Brand Sentiment Intelligence</div>
    <div class="hero-subtitle">
        Empirical Social Media (Instagram & Twitter) NLP Platform with Hybrid Machine Learning & Domain Lexicon Consensus.
    </div>
    <div>
        <span class="badge-pill"><span class="pulse-dot"></span> Real-Time Inference</span>
        <span class="badge-pill">🛡️ Balanced Class Weights</span>
        <span class="badge-pill">🎯 TF-IDF Sublinear N-Grams</span>
        <span class="badge-pill">📊 131 Empirical Survey Responses</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch cached dataset and metrics
survey_df = load_survey_results()
metrics_dict = load_metrics_data()
predictor = get_predictor()

# Compute summary KPI numbers if survey dataset is loaded
if survey_df is not None:
    total_responses: int = len(survey_df)
    sentiment_counts = survey_df['sentiment'].value_counts()
    pos_count: int = int(sentiment_counts.get('Positive', 0))
    neu_count: int = int(sentiment_counts.get('Neutral', 0))
    neg_count: int = int(sentiment_counts.get('Negative', 0))

    pos_pct: float = (pos_count / total_responses) * 100 if total_responses else 0.0
    neu_pct: float = (neu_count / total_responses) * 100 if total_responses else 0.0
    neg_pct: float = (neg_count / total_responses) * 100 if total_responses else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-card-header">
                <span class="metric-card-label">Analyzed Corpus</span>
                <span class="metric-card-icon">📑</span>
            </div>
            <div class="metric-card-value" style="color: #38BDF8;">{total_responses}</div>
            <div class="metric-card-subtext">Total Verified Consumer Comments</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-card-header">
                <span class="metric-card-label">Positive Sentiment</span>
                <span class="metric-card-icon">💚</span>
            </div>
            <div class="metric-card-value" style="color: #34D399;">{pos_count} <span style="font-size: 0.95rem; color: #94A3B8; font-weight: 500;">({pos_pct:.1f}%)</span></div>
            <div class="metric-card-subtext">Delighted & Satisfied Feedback</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-card-header">
                <span class="metric-card-label">Neutral / Objective</span>
                <span class="metric-card-icon">⚪</span>
            </div>
            <div class="metric-card-value" style="color: #CBD5E1;">{neu_count} <span style="font-size: 0.95rem; color: #94A3B8; font-weight: 500;">({neu_pct:.1f}%)</span></div>
            <div class="metric-card-subtext">Factual Inquiries & Attributes</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-card-header">
                <span class="metric-card-label">Negative Grievance</span>
                <span class="metric-card-icon">🔴</span>
            </div>
            <div class="metric-card-value" style="color: #FB7185;">{neg_count} <span style="font-size: 0.95rem; color: #94A3B8; font-weight: 500;">({neg_pct:.1f}%)</span></div>
            <div class="metric-card-subtext">Complaints, Delays & Critiques</div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 6. TABBED EXECUTIVE DASHBOARD
# ------------------------------------------------------------------------------

tab_predict, tab_overview, tab_wordclouds, tab_charts, tab_metrics, tab_data = st.tabs([
    "⚡ Real-Time Predictor",
    "📈 Sentiment Distribution",
    "☁️ Word Cloud Insights",
    "📊 Consumer Survey Analytics",
    "🏆 Model Performance",
    "📑 Dataset Explorer"
])


# ==============================================================================
# TAB 1: REAL-TIME INTERACTIVE SENTIMENT PREDICTOR
# ==============================================================================
with tab_predict:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="margin-bottom: 4px; font-weight: 700;">Live Consumer Comment Analyzer</h3>
        <p style="color: #94A3B8; font-size: 0.9rem;">
            Test any customer feedback or review. The prediction leverages balanced Logistic Regression calibrated with domain negation lexicon consensus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "user_comment_text" not in st.session_state:
        st.session_state["user_comment_text"] = "The product quality is absolutely amazing and customer support responded super fast!"

    if "run_analysis_flag" not in st.session_state:
        st.session_state["run_analysis_flag"] = True

    def select_preset_sample(sample_string: str) -> None:
        st.session_state["user_comment_text"] = sample_string
        st.session_state["run_analysis_flag"] = True

    st.markdown("<span style='font-size: 0.85rem; font-weight: 600; color: #94A3B8;'>Quick Example Presets:</span>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)

    p_col1.button(
        label="🌟 Positive Review (Praise)",
        on_click=select_preset_sample,
        args=("The product quality is absolutely amazing and customer support responded super fast!",),
        use_container_width=True
    )
    p_col2.button(
        label="💬 Neutral Inquiry (Pricing/Catalog)",
        on_click=select_preset_sample,
        args=("I checked their social media page to see the latest collection and pricing details.",),
        use_container_width=True
    )
    p_col3.button(
        label="⚠️ Negative Grievance (Delivery Delay)",
        on_click=select_preset_sample,
        args=("Terrible experience, very delayed delivery and completely rude customer service.",),
        use_container_width=True
    )

    user_comment: str = st.text_area(
        label="Enter social media comment or review:",
        value=st.session_state["user_comment_text"],
        placeholder="Type or paste customer text...",
        height=95,
        label_visibility="collapsed"
    )
    st.session_state["user_comment_text"] = user_comment

    analyze_button_clicked: bool = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_button_clicked or st.session_state.get("run_analysis_flag", False):
        st.session_state["run_analysis_flag"] = False

        if not user_comment.strip():
            st.warning("Please provide a valid comment to analyze.")
        elif predictor is None:
            st.error("Prediction engine is currently unavailable. Please run src/train_model.py first.")
        else:
            result: Dict[str, Any] = predictor.predict(user_comment)
            predicted_sentiment: str = str(result['sentiment'])
            confidence_val: float = float(result['confidence'])
            probs: Dict[str, float] = result['probabilities']

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1, 1.4])

            with res_col1:
                st.markdown("""
                <div class="sentiment-result-card">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600; margin-bottom: 8px;">
                        Primary Classification
                    </div>
                """, unsafe_allow_html=True)

                if predicted_sentiment == 'Positive':
                    st.markdown("<div class='badge-positive'>🟢 POSITIVE SENTIMENT</div>", unsafe_allow_html=True)
                elif predicted_sentiment == 'Negative':
                    st.markdown("<div class='badge-negative'>🔴 NEGATIVE SENTIMENT</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='badge-neutral'>⚪ NEUTRAL SENTIMENT</div>", unsafe_allow_html=True)

                detected_aspect = result.get('primary_aspect', extract_aspect(user_comment))

                st.markdown(f"""
                    <div style="margin-top: 12px; margin-bottom: 8px;">
                        <span class="badge-pill" style="border-color: rgba(56, 189, 248, 0.3);">📌 Aspect: <strong style="color: #38BDF8;">{detected_aspect}</strong></span>
                    </div>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem;">
                        <span style="color: #94A3B8;">Model Confidence</span>
                        <span style="font-weight: 700; color: #38BDF8;">{confidence_val * 100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(confidence_val))

            with res_col2:
                st.markdown("""
                <div class="sentiment-result-card">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600; margin-bottom: 8px;">
                        Posterior Probability Distribution
                    </div>
                """, unsafe_allow_html=True)

                prob_cols = st.columns(3)
                with prob_cols[0]:
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div style="color: #34D399; font-weight: 800; font-size: 1.3rem;">{probs.get('Positive', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">POSITIVE</div>
                    </div>
                    """, unsafe_allow_html=True)
                with prob_cols[1]:
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div style="color: #CBD5E1; font-weight: 800; font-size: 1.3rem;">{probs.get('Neutral', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">NEUTRAL</div>
                    </div>
                    """, unsafe_allow_html=True)
                with prob_cols[2]:
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div style="color: #FB7185; font-weight: 800; font-size: 1.3rem;">{probs.get('Negative', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">NEGATIVE</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # Keyword Token Extraction
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 🔎 Sentiment Keyword Token Extraction")
            highlighted_html: str = highlight_sentiment_tokens(user_comment)
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); padding: 14px 18px; border-radius: 10px; font-size: 1rem; line-height: 1.8;">
                {highlighted_html}
            </div>
            """, unsafe_allow_html=True)

            # Rationale Accordion
            with st.expander("ℹ️ Technical Consensus Rationale"):
                st.markdown(f"""
                - **ML Logistic Classifier Output:** `{result['ml_sentiment']}`
                - **Lexicon Polarity Output:** `{result['lexicon_sentiment']}` (Score: `{result['lexicon_score']}`)
                - **Consensus Rationale:** {result['explanation']}
                """)


# ==============================================================================
# TAB 2: SENTIMENT DISTRIBUTION OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("Sentiment Distribution in Empirical Survey")
    if survey_df is not None:
        s_counts = survey_df['sentiment'].value_counts().reset_index()
        s_counts.columns = ['Sentiment', 'Count']

        color_map = {
            'Positive': '#10B981',
            'Neutral': '#64748B',
            'Negative': '#F43F5E'
        }

        c_left, c_right = st.columns([1.4, 1])

        with c_left:
            fig = px.pie(
                s_counts,
                values='Count',
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=color_map,
                hole=0.55,
                title="Overall Sentiment Share"
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#0F172A', width=2))
            )
            fig.update_layout(
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', family='Plus Jakarta Sans'),
                margin=dict(t=40, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); padding: 20px; border-radius: 12px; margin-top: 20px;">
                <h4 style="margin-bottom: 12px; color: #F8FAFC;">Distribution Breakdown</h4>
            """, unsafe_allow_html=True)
            for _, row in s_counts.iterrows():
                pct = (row['Count'] / len(survey_df)) * 100
                color = color_map.get(row['Sentiment'], '#CBD5E1')
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 600; color: {color};">● {row['Sentiment']}</span>
                    <span style="color: #F8FAFC; font-weight: 700;">{row['Count']} <span style="color: #94A3B8; font-size: 0.85rem;">({pct:.1f}%)</span></span>
                </div>
                """, unsafe_allow_html=True)

            st.caption("Neutral responses dominate because many respondents noted objective product attributes (such as 'price' or 'comments') rather than strong emotional reviews.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Aspect Distribution Chart
        if 'primary_aspect' in survey_df.columns:
            st.markdown("---")
            st.markdown("#### 🎯 Business Domain Drivers & Key Focus Areas")
            st.markdown("Primary operational aspects identified in customer opinion responses:")

            asp_counts = survey_df[~survey_df.get('is_noise', False)]['primary_aspect'].value_counts().reset_index()
            asp_counts.columns = ['Aspect', 'Mentions']

            fig_asp = px.bar(
                asp_counts,
                x='Mentions',
                y='Aspect',
                orientation='h',
                color='Aspect',
                title="Consumer Opinion Frequency by Domain Aspect",
                color_discrete_sequence=['#38BDF8', '#818CF8', '#34D399', '#FBBF24', '#F43F5E', '#94A3B8']
            )
            fig_asp.update_layout(
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', family='Plus Jakarta Sans'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.06)', title="Number of Customer Comments"),
                yaxis=dict(autorange="reversed", title=""),
                margin=dict(t=40, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_asp, use_container_width=True)
    else:
        st.info("Run `python run_project.py` to generate sentiment results.")


# ==============================================================================
# TAB 3: WORD CLOUDS
# ==============================================================================
with tab_wordclouds:
    st.subheader("Customer Opinion & Suggestion Term Clouds")
    st.markdown("High-frequency terms extracted and visualized using Natural Language Processing normalization.")

    wc1_path = ROOT_DIR / 'outputs' / 'wordcloud_opinions.png'
    wc2_path = ROOT_DIR / 'outputs' / 'wordcloud_suggestions.png'

    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        if wc1_path.exists():
            st.markdown("""
            <div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; background: rgba(15,23,42,0.6); padding: 12px;">
                <h5 style="margin-bottom: 8px;">💭 Customer Brand Opinions</h5>
            """, unsafe_allow_html=True)
            st.image(str(wc1_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with wc_col2:
        if wc2_path.exists():
            st.markdown("""
            <div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; background: rgba(15,23,42,0.6); padding: 12px;">
                <h5 style="margin-bottom: 8px;">💡 Improvement Suggestions</h5>
            """, unsafe_allow_html=True)
            st.image(str(wc2_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 4: CATEGORICAL SURVEY CHARTS
# ==============================================================================
with tab_charts:
    st.subheader("Survey Question Analytics & Customer Behavior")
    st.markdown("Exploratory Data Analysis generated from multi-choice customer responses.")

    chart_files = sorted(list((ROOT_DIR / 'outputs').glob('chart_*.png')))
    if chart_files:
        for i in range(0, len(chart_files), 2):
            g_cols = st.columns(2)
            with g_cols[0]:
                st.markdown(f"""
                <div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; background: rgba(15,23,42,0.6); padding: 10px; margin-bottom: 16px;">
                """, unsafe_allow_html=True)
                st.image(str(chart_files[i]), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            if i + 1 < len(chart_files):
                with g_cols[1]:
                    st.markdown(f"""
                    <div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; background: rgba(15,23,42,0.6); padding: 10px; margin-bottom: 16px;">
                    """, unsafe_allow_html=True)
                    st.image(str(chart_files[i + 1]), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No survey charts found. Run `python src/analyze_data.py` to generate them.")


# ==============================================================================
# TAB 5: MODEL PERFORMANCE & TECHNICAL METRICS
# ==============================================================================
with tab_metrics:
    st.subheader("Machine Learning Model Performance & Metrics")
    if metrics_dict is not None:
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-card-label">Training Corpus</div>
                <div class="metric-card-value" style="color: #38BDF8;">{metrics_dict.get('total_samples_trained', 'N/A')}</div>
                <div class="metric-card-subtext">Total Verified Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c2:
            acc = metrics_dict.get('accuracy')
            acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-card-label">Validation Accuracy</div>
                <div class="metric-card-value" style="color: #34D399;">{acc_str}</div>
                <div class="metric-card-subtext">Hold-Out Validation Performance</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c3:
            st.markdown("""
            <div class="metric-card-pro">
                <div class="metric-card-label">Classification Engine</div>
                <div class="metric-card-value" style="color: #818CF8; font-size: 1.6rem; margin-top: 6px;">Balanced LR</div>
                <div class="metric-card-subtext">Logistic Regression with L2 Regularization</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Classification Report")
        report_file = ROOT_DIR / 'outputs' / 'classification_report.csv'
        if report_file.exists():
            rep_df = pd.read_csv(report_file, index_col=0)
            st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

        st.markdown("#### Confusion Matrix")
        cm_file = ROOT_DIR / 'outputs' / 'confusion_matrix.csv'
        if cm_file.exists():
            cm_df = pd.read_csv(cm_file, index_col=0)
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("No model metrics file found. Run `python src/train_model.py` to generate metrics.")


# ==============================================================================
# TAB 6: RAW SURVEY DATASET EXPLORER
# ==============================================================================
with tab_data:
    st.subheader("Survey Dataset Explorer")
    if survey_df is not None:
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
        with f_col1:
            sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
        with f_col2:
            aspect_options = ["All"]
            if 'primary_aspect' in survey_df.columns:
                aspect_options += sorted([str(a) for a in survey_df['primary_aspect'].dropna().unique() if a])
            aspect_filter = st.selectbox("Filter by Aspect:", aspect_options)
        with f_col3:
            search_query = st.text_input("Search in comment text:", "", placeholder="Type keywords like 'delivery', 'quality', 'price'...")

        hide_noise = st.checkbox("🔍 Filter out non-informative / placeholder answers (e.g. 'idk', '-', 'none')", value=False)

        filtered_df = survey_df.copy()
        if hide_noise and 'is_noise' in filtered_df.columns:
            filtered_df = filtered_df[~filtered_df['is_noise']]
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
        if aspect_filter != "All" and 'primary_aspect' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['primary_aspect'] == aspect_filter]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df['text'].str.contains(search_query, case=False, na=False)]

        st.markdown(f"""
        <div style="display: flex; gap: 18px; background: rgba(15, 23, 42, 0.6); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px; font-size: 0.88rem;">
            <span>Filtered Records: <strong style="color: #38BDF8;">{len(filtered_df)}</strong> of {len(survey_df)}</span>
            <span>Sentiment Filter: <strong style="color: #F8FAFC;">{sentiment_filter}</strong></span>
            <span>Aspect Filter: <strong style="color: #F8FAFC;">{aspect_filter}</strong></span>
        </div>
        """, unsafe_allow_html=True)

        cols_to_show = ['text', 'clean_text', 'sentiment']
        rename_dict = {
            'text': 'Raw Comment',
            'clean_text': 'Sanitized & Normalized Text',
            'sentiment': 'Calibrated Sentiment'
        }
        if 'primary_aspect' in filtered_df.columns:
            cols_to_show.append('primary_aspect')
            rename_dict['primary_aspect'] = 'Business Aspect'
        if 'lexicon_score' in filtered_df.columns:
            cols_to_show.append('lexicon_score')
            rename_dict['lexicon_score'] = 'Polarity Score'

        st.dataframe(filtered_df[cols_to_show].rename(columns=rename_dict), use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_sentiment_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No dataset loaded.")
