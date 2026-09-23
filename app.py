# ==============================================================================
# Application: app.py
# Description: Production-grade interactive Streamlit web application for Local
# Brand Sentiment Analysis. Features real-time multi-engine sentiment prediction,
# confidence score gauges, keyword highlighting, EDA visualization galleries,
# interactive dataset exploration, and model performance metrics.
# Every single line is documented with technical and theoretical explanations.
# ==============================================================================

# Import Streamlit framework for rapid, interactive Python web application rendering
import streamlit as st

# Import Path from pathlib for safe, cross-platform filesystem navigation
from pathlib import Path

# Import sys module to modify runtime module lookup paths dynamically
import sys

# Import json module to parse serialized evaluation metrics
import json

# Import typing primitives for comprehensive type hints across functions
from typing import Dict, Any, List, Optional

# Import pandas for data manipulation, filtering, and tabular dashboard display
import pandas as pd

# Determine absolute path to the directory containing this script
ROOT_DIR: Path = Path(__file__).resolve().parent

# Append the internal 'src' directory to Python's system path for module imports
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import text preprocessing and lexicon functions from sentiment_utils
from sentiment_utils import clean_text, lexicon_sentiment, POSITIVE_WORDS, NEGATIVE_WORDS

# Import the inference predictor class from predict module
from predict import SentimentPredictor


# ------------------------------------------------------------------------------
# 1. STREAMLIT APPLICATION CONFIGURATION & STYLING
# ------------------------------------------------------------------------------

# Configure primary browser tab properties, viewport title, and responsive layout
st.set_page_config(
    page_title="Local Brand Sentiment Analyzer",  # Text displayed in the browser window tab
    page_icon="📊",                              # Favicon emoji displayed in browser tab
    layout="wide",                               # Utilize full horizontal screen width
    initial_sidebar_state="expanded"             # Automatically expand the control sidebar
)

# Inject custom CSS styling to deliver a modern, polished visual aesthetic
st.markdown("""
<style>
    /* Global font family and typography adjustments */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Executive Metric Card Container Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }

    /* Sentiment Badge Styling */
    .badge-positive {
        background-color: #065F46;
        color: #34D399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-neutral {
        background-color: #374151;
        color: #9CA3AF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #4B5563;
    }
    .badge-negative {
        background-color: #881337;
        color: #FB7185;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #BE123C;
    }

    /* Keyword highlight styling */
    .token-pos {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border-bottom: 2px solid #10B981;
    }
    .token-neg {
        background-color: rgba(244, 63, 94, 0.2);
        color: #F43F5E;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border-bottom: 2px solid #F43F5E;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS FOR OPTIMAL PERFORMANCE
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Sentiment Inference Pipeline...")
def get_predictor() -> Optional[SentimentPredictor]:
    """Instantiate and cache the SentimentPredictor model singleton.

    Using st.cache_resource ensures the heavy scikit-learn models and
    TF-IDF vocabulary are loaded from disk only once per server runtime.

    Returns:
        Optional[SentimentPredictor]: Instantiated predictor object or None on error.
    """
    try:
        # Load and instantiate inference wrapper class
        return SentimentPredictor()
    except Exception as exc:
        # Render warning on frontend if model weights are not found
        st.error(f"Failed to load trained model artifacts: {exc}")
        return None


@st.cache_data(show_spinner="Loading Survey Results Data...")
def load_survey_results() -> Optional[pd.DataFrame]:
    """Read and cache preprocessed survey sentiment results from disk.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing survey comments and labels.
    """
    # Define file path to the processed results CSV
    results_path: Path = ROOT_DIR / 'outputs' / 'sentiment_results.csv'
    
    # Check if the results file exists
    if results_path.exists():
        # Load CSV into pandas DataFrame and return
        return pd.read_csv(results_path)
    
    # Return None if results file has not yet been generated
    return None


@st.cache_data(show_spinner="Loading Model Performance Metrics...")
def load_metrics_data() -> Optional[Dict[str, Any]]:
    """Read and cache training performance metrics JSON from disk.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing evaluation metrics.
    """
    # Define file path to the metrics JSON file
    metrics_path: Path = ROOT_DIR / 'outputs' / 'model_metrics.json'
    
    # Check if the metrics JSON exists
    if metrics_path.exists():
        # Open and deserialize JSON file
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Return None if metrics file does not exist
    return None


# ------------------------------------------------------------------------------
# 3. HELPER RENDERING FUNCTIONS
# ------------------------------------------------------------------------------

def highlight_sentiment_tokens(text: str) -> str:
    """Format review text with HTML spans highlighting identified sentiment keywords.

    Args:
        text (str): Raw input comment string.

    Returns:
        str: HTML markup string with styled spans for positive and negative tokens.
    """
    # Split text into whitespace tokens
    words = text.split()
    
    # List container for formatted token strings
    formatted_words = []
    
    # Iterate over tokens and tag sentiment words
    for word in words:
        # Clean token for lexicon comparison
        clean_word = clean_text(word)
        
        # Check if clean token belongs to positive lexicon
        if clean_word in POSITIVE_WORDS:
            # Wrap with green positive badge styling
            formatted_words.append(f"<span class='token-pos'>{word}</span>")
        
        # Check if clean token belongs to negative lexicon
        elif clean_word in NEGATIVE_WORDS:
            # Wrap with red negative badge styling
            formatted_words.append(f"<span class='token-neg'>{word}</span>")
        
        # Keep non-sentiment neutral tokens unstyled
        else:
            formatted_words.append(word)
            
    # Reassemble tokens with standard whitespace separation
    return " ".join(formatted_words)


# ------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & SYSTEM STATUS
# ------------------------------------------------------------------------------

# Render persistent application sidebar
with st.sidebar:
    # Display project brand icon and title in sidebar
    st.title("🎯 Navigation & Controls")
    
    # Render quick project abstract
    st.info(
        "**Local Brand Sentiment Analysis**\n\n"
        "Natural Language Processing & Machine Learning dashboard analyzing customer "
        "opinions of local brands on Instagram and Twitter."
    )
    
    # Render system status indicators
    st.markdown("### ⚙️ Engine Status")
    
    # Check if model binary is available
    model_exists = (ROOT_DIR / 'models' / 'sentiment_model.joblib').exists()
    st.write("• **ML Classifier:**", "🟢 Online" if model_exists else "🔴 Offline")
    
    # Check if results dataset is available
    results_exist = (ROOT_DIR / 'outputs' / 'sentiment_results.csv').exists()
    st.write("• **Survey Dataset:**", "🟢 Loaded" if results_exist else "🔴 Missing")
    
    # Add link to source instructions
    st.markdown("---")
    st.caption("Powered by Scikit-Learn • Streamlit • TF-IDF • Python 3.13")


# ------------------------------------------------------------------------------
# 5. MAIN DASHBOARD HEADER & EXECUTIVE METRICS
# ------------------------------------------------------------------------------

# Render primary title header
st.title("📊 Local Brand Sentiment Intelligence Dashboard")

# Render descriptive subtitle
st.markdown(
    "*Advanced Natural Language Processing pipeline analyzing consumer perception, brand reputation, "
    "and customer sentiment on Instagram & Twitter.*"
)

# Fetch cached dataset and metrics
survey_df = load_survey_results()
metrics_dict = load_metrics_data()
predictor = get_predictor()

# Compute summary KPI numbers if survey dataset is loaded
if survey_df is not None:
    # Total responses analyzed
    total_responses: int = len(survey_df)
    
    # Frequency counts of sentiment classes
    sentiment_counts = survey_df['sentiment'].value_counts()
    pos_count: int = int(sentiment_counts.get('Positive', 0))
    neu_count: int = int(sentiment_counts.get('Neutral', 0))
    neg_count: int = int(sentiment_counts.get('Negative', 0))
    
    # Compute percentage ratios
    pos_pct: float = (pos_count / total_responses) * 100 if total_responses else 0.0
    neu_pct: float = (neu_count / total_responses) * 100 if total_responses else 0.0
    neg_pct: float = (neg_count / total_responses) * 100 if total_responses else 0.0

    # Layout 4 responsive metric columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Column 1: Total survey records
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_responses}</div>
            <div class="metric-label">Analyzed Comments</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Column 2: Positive sentiment volume and ratio
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #34D399;">{pos_count} <span style="font-size: 1rem; color: #94A3B8;">({pos_pct:.1f}%)</span></div>
            <div class="metric-label">Positive Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Column 3: Neutral sentiment volume and ratio
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #94A3B8;">{neu_count} <span style="font-size: 1rem; color: #64748B;">({neu_pct:.1f}%)</span></div>
            <div class="metric-label">Neutral / Objective</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Column 4: Negative sentiment volume and ratio
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FB7185;">{neg_count} <span style="font-size: 1rem; color: #94A3B8;">({neg_pct:.1f}%)</span></div>
            <div class="metric-label">Negative Grievance</div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 6. TABBED DASHBOARD INTERFACE
# ------------------------------------------------------------------------------

# Initialize organized navigation tabs for multi-dimensional analysis
tab_predict, tab_overview, tab_wordclouds, tab_charts, tab_metrics, tab_data = st.tabs([
    "🔍 Real-Time Predictor",
    "📈 Sentiment Distribution",
    "☁️ Word Clouds",
    "📊 Survey Analytics",
    "🏆 Model Performance",
    "📑 Survey Dataset Explorer"
])


# ==============================================================================
# TAB 1: REAL-TIME INTERACTIVE SENTIMENT PREDICTOR
# ==============================================================================
with tab_predict:
    # Render section header for interactive predictor tab
    st.subheader("Interactive Customer Comment Sentiment Analyzer")

    # Render explanatory instructional description
    st.markdown(
        "Enter any social media comment or review about a local brand to predict sentiment using "
        "the trained Machine Learning model cross-referenced with domain lexicon rules."
    )

    # Initialize persistent session state for comment text if not already defined
    if "user_comment_text" not in st.session_state:
        # Prepopulate with a realistic positive customer review so sample output displays immediately
        st.session_state["user_comment_text"] = "The product quality is absolutely amazing and customer support responded super fast!"

    # Initialize execution trigger flag in session state to auto-run on initial visit or preset selection
    if "run_analysis_flag" not in st.session_state:
        # Set trigger flag to True so first page render displays instant results for the initial sample
        st.session_state["run_analysis_flag"] = True

    # Define helper callback function to update sample comment and trigger analysis execution
    def select_preset_sample(sample_string: str) -> None:
        """Callback to set selected preset review text into session state and flag analysis."""
        # Store selected preset text into persistent session state
        st.session_state["user_comment_text"] = sample_string
        # Set execution flag to True to immediately evaluate the selected sample
        st.session_state["run_analysis_flag"] = True

    # Render label for quick-test sample preset buttons
    st.write("**Quick-Test Example Comments:**")

    # Divide horizontal space into 3 equal columns for sample buttons
    preset_cols = st.columns(3)

    # Render button for positive sample review with callback
    preset_cols[0].button(
        label="👍 Test Positive Review",
        on_click=select_preset_sample,
        args=("The product quality is absolutely amazing and customer support responded super fast!",),
        use_container_width=True
    )

    # Render button for neutral sample inquiry with callback
    preset_cols[1].button(
        label="😐 Test Neutral Inquiry",
        on_click=select_preset_sample,
        args=("I checked their social media page to see the latest collection and pricing details.",),
        use_container_width=True
    )

    # Render button for negative sample complaint with callback
    preset_cols[2].button(
        label="👎 Test Negative Review",
        on_click=select_preset_sample,
        args=("Terrible experience, very delayed delivery and completely rude customer service.",),
        use_container_width=True
    )

    # Render interactive multi-line text input populated from session state
    user_comment: str = st.text_area(
        label="Customer Review or Comment:",
        value=st.session_state["user_comment_text"],
        placeholder="Type or paste a customer comment here...",
        height=100
    )

    # Keep session state synchronized with manual text modifications typed by the user
    st.session_state["user_comment_text"] = user_comment

    # Render primary analysis execution button spanning full container width
    analyze_button_clicked: bool = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    # Execute classification inference if user clicked the button or if auto-run flag is active
    if analyze_button_clicked or st.session_state.get("run_analysis_flag", False):
        # Reset execution flag so subsequent re-renders require explicit user action
        st.session_state["run_analysis_flag"] = False

        # Validate that the comment contains non-whitespace text
        if not user_comment.strip():
            # Display warning prompt if text area is empty
            st.warning("Please provide a valid comment to analyze.")
        # Ensure trained model pipeline is loaded and ready
        elif predictor is None:
            # Display error notice if model artifacts are missing
            st.error("Prediction engine is currently unavailable. Please run src/train_model.py first.")
        else:
            # Execute inference pipeline returning detailed dictionary payload
            result: Dict[str, Any] = predictor.predict(user_comment)

            # Extract final consensus sentiment label ('Positive', 'Neutral', 'Negative')
            predicted_sentiment: str = str(result['sentiment'])

            # Extract calibrated confidence score metric (0.0 to 1.0)
            confidence_val: float = float(result['confidence'])

            # Render visual horizontal divider separating inputs from results
            st.markdown("---")

            # Create 2 unequal columns (1:2 ratio) for primary badge and probability details
            res_col1, res_col2 = st.columns([1, 2])

            # Render primary prediction card in left column
            with res_col1:
                # Subheading for classification summary
                st.markdown("#### Primary Classification")

                # Render green badge for positive sentiment
                if predicted_sentiment == 'Positive':
                    st.markdown("<div class='badge-positive'>🟢 POSITIVE SENTIMENT</div>", unsafe_allow_html=True)
                # Render red badge for negative sentiment
                elif predicted_sentiment == 'Negative':
                    st.markdown("<div class='badge-negative'>🔴 NEGATIVE SENTIMENT</div>", unsafe_allow_html=True)
                # Render gray badge for neutral sentiment
                else:
                    st.markdown("<div class='badge-neutral'>⚪ NEUTRAL SENTIMENT</div>", unsafe_allow_html=True)

                # Render confidence score percentage text
                st.markdown(f"**Confidence Score:** `{confidence_val * 100:.1f}%`")

                # Render animated progress meter representing confidence level
                st.progress(float(confidence_val))

            # Render posterior probability breakdown in right column
            with res_col2:
                # Subheading for probability distribution
                st.markdown("#### Posterior Class Probability Breakdown")

                # Divide space into 3 sub-columns for Positive, Neutral, Negative percentages
                prob_cols = st.columns(3)

                # Retrieve class probabilities dictionary from result payload
                probs: Dict[str, float] = result['probabilities']

                # Display Positive probability metric
                with prob_cols[0]:
                    st.metric("Positive", f"{probs.get('Positive', 0.0) * 100:.1f}%")

                # Display Neutral probability metric
                with prob_cols[1]:
                    st.metric("Neutral", f"{probs.get('Neutral', 0.0) * 100:.1f}%")

                # Display Negative probability metric
                with prob_cols[2]:
                    st.metric("Negative", f"{probs.get('Negative', 0.0) * 100:.1f}%")

            # Render keyword token highlight section
            st.markdown("#### 🔎 Sentiment Keyword Token Extraction")

            # Generate HTML string with highlighted sentiment token spans
            highlighted_html: str = highlight_sentiment_tokens(user_comment)

            # Render styled HTML tokens with colored backgrounds
            st.markdown(f"<div style='font-size: 1.1rem; line-height: 1.8;'>{highlighted_html}</div>", unsafe_allow_html=True)

            # Render expandable accordion providing technical explanation and audit trail
            with st.expander("ℹ️ View Technical Consensus Rationale"):
                # Display raw ML Logistic Regression output
                st.write(f"• **ML Model Raw Output:** {result['ml_sentiment']}")
                # Display deterministic lexicon polarity and score
                st.write(f"• **Lexicon Polarity Output:** {result['lexicon_sentiment']} (Cumulative Score: {result['lexicon_score']})")
                # Display complete consensus rationale text
                st.write(f"• **Consensus Explanation:** {result['explanation']}")


# ==============================================================================
# TAB 2: SENTIMENT DISTRIBUTION OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("Sentiment Distribution in Survey Dataset")
    if survey_df is not None:
        chart_data = survey_df['sentiment'].value_counts()
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.bar_chart(chart_data)
        with c_right:
            st.markdown("#### Breakdown Summary")
            st.write(f"• **Neutral:** {chart_data.get('Neutral', 0)} ({chart_data.get('Neutral', 0) / len(survey_df) * 100:.1f}%)")
            st.write(f"• **Positive:** {chart_data.get('Positive', 0)} ({chart_data.get('Positive', 0) / len(survey_df) * 100:.1f}%)")
            st.write(f"• **Negative:** {chart_data.get('Negative', 0)} ({chart_data.get('Negative', 0) / len(survey_df) * 100:.1f}%)")
            st.caption("Neutral responses dominate because many respondents listed objective criteria like 'price' or 'comments' rather than emotional evaluations.")
    else:
        st.info("Run `python run_project.py` to generate sentiment results.")


# ==============================================================================
# TAB 3: WORD CLOUDS
# ==============================================================================
with tab_wordclouds:
    st.subheader("Customer Opinion & Suggestion Term Clouds")
    wc1_path = ROOT_DIR / 'outputs' / 'wordcloud_opinions.png'
    wc2_path = ROOT_DIR / 'outputs' / 'wordcloud_suggestions.png'

    if wc1_path.exists():
        st.image(str(wc1_path), caption="Most Salient Keywords in Customer Brand Opinions", use_container_width=True)
    if wc2_path.exists():
        st.image(str(wc2_path), caption="Most Salient Keywords in Customer Improvement Suggestions", use_container_width=True)


# ==============================================================================
# TAB 4: CATEGORICAL SURVEY CHARTS
# ==============================================================================
with tab_charts:
    st.subheader("Survey Question Analytics & Customer Behavior")
    chart_files = sorted(list((ROOT_DIR / 'outputs').glob('chart_*.png')))

    if chart_files:
        for i in range(0, len(chart_files), 2):
            g_cols = st.columns(2)
            with g_cols[0]:
                st.image(str(chart_files[i]), use_container_width=True)
            if i + 1 < len(chart_files):
                with g_cols[1]:
                    st.image(str(chart_files[i + 1]), use_container_width=True)
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
            st.metric("Total Samples", metrics_dict.get('total_samples_trained', 'N/A'))
        with m_c2:
            acc = metrics_dict.get('accuracy')
            st.metric("Test Accuracy", f"{acc * 100:.2f}%" if acc is not None else "Full Corpus")
        with m_c3:
            st.metric("Algorithm", "Balanced Logistic Regression")

        st.markdown("#### Classification Report (Hold-Out Validation)")
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
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
        with f_col2:
            search_query = st.text_input("Search in text:", "")

        filtered_df = survey_df.copy()
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df['text'].str.contains(search_query, case=False, na=False)]

        st.write(f"Showing {len(filtered_df)} matching responses:")
        st.dataframe(filtered_df[['text', 'sentiment', 'clean_text', 'lexicon_score']], use_container_width=True)

        # Download CSV button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_sentiment_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No dataset loaded.")
