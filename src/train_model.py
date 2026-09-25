# ==============================================================================
# Module: train_model.py
# Description: Production-grade Machine Learning training and evaluation pipeline
# for sentiment classification of local brand customer reviews.
# Utilizes TF-IDF n-gram feature extraction, balanced Logistic Regression,
# cross-validated evaluation metrics, and standardized model serialization.
# Every single line is documented with technical and theoretical explanations.
# ==============================================================================

# Import Path for robust, cross-platform filesystem directory and file management
from pathlib import Path

# Import sys module to dynamically configure Python runtime search paths
import sys

# Import json module to serialize comprehensive model performance metrics into standard JSON
import json

# Import typing primitives for strict type checking and self-documenting code
from typing import Dict, Any, Tuple, Optional

# Import pandas for high-performance tabular data manipulation and Excel I/O
import pandas as pd

# Import numpy for numerical operations and array manipulation
import numpy as np

# Import joblib for efficient disk serialization of scikit-learn models and sparse matrices
import joblib

# Import train_test_split to partition datasets into independent training and validation subsets
from sklearn.model_selection import train_test_split

# Import TfidfVectorizer to convert raw text documents into normalized TF-IDF feature matrices
from sklearn.feature_extraction.text import TfidfVectorizer

# Import LogisticRegression classifier for convex, multi-class probabilistic sentiment modeling
from sklearn.linear_model import LogisticRegression

# Import evaluation metrics for comprehensive assessment of classification performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Determine the absolute project root directory (two levels up from this script: src/ -> root)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Inject the 'src' directory into sys.path to enable clean absolute module imports
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import text preprocessing functions and dynamic column detection from sentiment_utils
from sentiment_utils import clean_text, lexicon_sentiment, load_text_column, extract_aspect, is_noise

# Define standard filesystem paths for data input, model binaries, and output reports
DATA_FILE_PATH: Path = ROOT_DIR / 'data' / 'survey_responses.xlsx'

# Path to the directory where visual charts, metrics, and processed data CSVs are stored
OUTPUTS_DIR_PATH: Path = ROOT_DIR / 'outputs'

# Path to the directory where trained estimator and vectorizer artifacts are persisted
MODELS_DIR_PATH: Path = ROOT_DIR / 'models'


# ------------------------------------------------------------------------------
# 1. DATA INGESTION AND PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------

def load_and_preprocess_data(data_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load survey data, extract opinion text, apply text cleaning, and generate weak labels.

    Args:
        data_path (Path): Path to the Excel spreadsheet containing survey responses.

    Returns:
        Tuple[pd.DataFrame, str]: Preprocessed DataFrame containing clean text and labels,
                                  along with the name of the extracted question column.
    """
    # Verify that the survey dataset exists on disk before attempting to read it
    if not data_path.exists():
        # Raise an explicit FileNotFoundError with path context if the file is absent
        raise FileNotFoundError(f"Survey dataset not found at expected path: {data_path}")

    # Read the Excel file into a pandas DataFrame using openpyxl engine
    raw_df: pd.DataFrame = pd.read_excel(data_path)

    # Automatically identify the open-ended customer opinion column from survey headers
    target_column_name: str = load_text_column(raw_df)

    # Create an isolated DataFrame copy containing only the identified text column
    processed_df: pd.DataFrame = raw_df[[target_column_name]].copy()

    # Rename the selected column to a canonical, standardized name: 'text'
    processed_df.columns = ['text']

    # Impute missing/NaN values with empty strings and cast all elements to string type
    processed_df['text'] = processed_df['text'].fillna('').astype(str)

    # Filter out entries where the raw text is blank or composed entirely of whitespace
    processed_df = processed_df[processed_df['text'].str.strip().ne('')]

    # Apply text normalization (lowercasing, punctuation, regex cleaning) to create 'clean_text'
    processed_df['clean_text'] = processed_df['text'].apply(clean_text)

    # Eliminate records whose text became empty after removing non-alphanumeric noise
    processed_df = processed_df[processed_df['clean_text'].str.strip().ne('')]

    # Compute rule-based weak sentiment labels and numeric polarity scores using the lexicon engine
    sentiment_tuples = processed_df['clean_text'].apply(lambda comment: lexicon_sentiment(comment))

    # Unpack the sentiment tuples into distinct 'sentiment' and 'lexicon_score' columns
    processed_df['sentiment'] = [st[0] for st in sentiment_tuples]

    # Assign integer lexicon polarity score for analytical auditing and distribution checks
    processed_df['lexicon_score'] = [st[1] for st in sentiment_tuples]

    # Assign domain aspect tag and noise indicator
    processed_df['primary_aspect'] = processed_df['clean_text'].apply(extract_aspect)
    processed_df['is_noise'] = processed_df['clean_text'].apply(is_noise)

    # Inspect class frequency distribution across the generated sentiment categories
    class_frequencies: pd.Series = processed_df['sentiment'].value_counts()

    # If any minority class has fewer than 2 examples, drop sparse classes to allow training
    if class_frequencies.min() < 2 and 'Neutral' in class_frequencies.index:
        # Filter out neutral records if extreme scarcity prevents valid model convergence
        processed_df = processed_df[processed_df['sentiment'] != 'Neutral']

    # Reset DataFrame integer index to maintain a clean sequential index from 0 to N-1
    processed_df = processed_df.reset_index(drop=True)

    # Return the clean processed DataFrame along with the detected question column name
    return processed_df, target_column_name


# ------------------------------------------------------------------------------
# 2. FEATURE EXTRACTION (TF-IDF VECTORIZATION)
# ------------------------------------------------------------------------------

def extract_features(corpus: pd.Series) -> Tuple[TfidfVectorizer, Any]:
    """Transform text corpus into a Term Frequency-Inverse Document Frequency (TF-IDF) sparse matrix.

    Configured with unigrams and bigrams (1,2) to capture phrase context (e.g., 'not good'),
    sublinear TF scaling to dampen the effect of very high frequency words, and L2 normalization.

    Args:
        corpus (pd.Series): Series containing cleaned text documents.

    Returns:
        Tuple[TfidfVectorizer, Any]: Fitted TfidfVectorizer instance and the resulting sparse feature matrix X.
    """
    # Instantiate TfidfVectorizer with academic and industrial best-practice hyperparameters
    vectorizer: TfidfVectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # Extract both individual words (unigrams) and 2-word pairs (bigrams)
        max_features=3000,          # Constrain vocabulary to top 3,000 informative features to prevent overfitting
        sublinear_tf=True,          # Apply sublinear scaling (1 + log(tf)) to prevent high-frequency term domination
        min_df=1,                   # Include terms appearing in at least 1 document for small survey corpora
        norm='l2'                   # Apply Euclidean L2 normalization across sample vectors for unit length
    )

    # Learn document vocabulary and compute inverse document frequency (IDF) weights, returning sparse matrix X
    feature_matrix = vectorizer.fit_transform(corpus)

    # Return both the fitted vectorizer object (for future inference) and the numerical feature matrix
    return vectorizer, feature_matrix


# ------------------------------------------------------------------------------
# 3. MODEL TRAINING & PERFORMANCE EVALUATION
# ------------------------------------------------------------------------------

def train_and_evaluate(
    feature_matrix: Any,
    labels: pd.Series,
    target_column_name: str
) -> Tuple[LogisticRegression, Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Train a balanced Logistic Regression classifier and compute rigorous evaluation metrics.

    Uses stratified splitting when data volume allows, balancing class weights inversely
    proportional to class frequencies to handle imbalanced customer sentiment distributions.

    Args:
        feature_matrix: Scipy sparse matrix of TF-IDF feature representations.
        labels: Series of categorical sentiment target labels ('Positive', 'Neutral', 'Negative').
        target_column_name: Original survey question header for auditing and provenance.

    Returns:
        Tuple containing the trained LogisticRegression model, metrics dictionary,
        classification report DataFrame, and confusion matrix DataFrame.
    """
    # Initialize the performance metrics dictionary with foundational metadata
    metrics_summary: Dict[str, Any] = {
        'total_samples_trained': int(len(labels)),
        'source_question_column': target_column_name,
        'class_distribution': labels.value_counts().to_dict()
    }

    # Canonical list of distinct sentiment classes expected in the classification task
    target_labels: List[str] = ['Positive', 'Neutral', 'Negative']

    # Initialize container variables for reports to be exported as structured DataFrames
    report_dataframe: Optional[pd.DataFrame] = None
    confusion_dataframe: Optional[pd.DataFrame] = None

    # Verify if the dataset has sufficient class diversity and row volume to perform a split
    has_sufficient_classes: bool = labels.nunique() >= 2
    has_sufficient_samples: bool = len(labels) >= 10

    if has_sufficient_classes and has_sufficient_samples:
        # Determine whether stratified sampling is mathematically viable (min 2 instances per class)
        can_stratify: bool = labels.value_counts().min() >= 2
        
        # Apply stratification if viable so train and test sets maintain identical class ratios
        stratification_target = labels if can_stratify else None

        # Partition feature matrix and labels into 80% training set and 20% held-out test set
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            labels,
            test_size=0.20,
            random_state=42,           # Fixed seed for reproducibility across training runs
            stratify=stratification_target
        )

        # Instantiate Logistic Regression with balanced class weights to compensate for class skew
        model: LogisticRegression = LogisticRegression(
            max_iter=1000,              # Allow generous iterations for L-BFGS optimization to reach convergence
            class_weight='balanced',    # Weight samples inversely proportional to class frequencies: n / (k * n_c)
            solver='lbfgs',             # Quasi-Newton optimization algorithm well-suited for small-to-medium datasets
            random_state=42             # Fixed random seed ensuring deterministic solver convergence
        )

        # Optimize model parameters (weights and intercept) using the training partition
        model.fit(X_train, y_train)

        # Generate discrete class predictions on the unseen held-out test partition
        y_predictions = model.predict(X_test)

        # Compute standard overall classification accuracy: correct predictions / total predictions
        accuracy_val: float = float(accuracy_score(y_test, y_predictions))
        metrics_summary['accuracy'] = accuracy_val

        # Compute precision, recall, and F1-score for each class as a structured dictionary
        class_report_dict = classification_report(
            y_test,
            y_predictions,
            zero_division=0,
            output_dict=True
        )
        metrics_summary['classification_report'] = class_report_dict

        # Convert classification report dictionary into a clean pandas DataFrame for CSV export
        report_dataframe = pd.DataFrame(class_report_dict).transpose()

        # Compute confusion matrix mapping ground-truth test labels against predicted labels
        active_classes: List[str] = [cls for cls in target_labels if cls in labels.unique()]
        conf_matrix_array = confusion_matrix(y_test, y_predictions, labels=active_classes)
        metrics_summary['confusion_matrix'] = conf_matrix_array.tolist()

        # Format confusion matrix as a labeled DataFrame with descriptive index and columns
        confusion_dataframe = pd.DataFrame(
            conf_matrix_array,
            index=[f"Actual_{cls}" for cls in active_classes],
            columns=[f"Pred_{cls}" for cls in active_classes]
        )
    else:
        # Fallback branch for micro-datasets: fit model on all available samples without holdout
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        )
        # Train model across the entirety of available labeled data
        model.fit(feature_matrix, labels)

        # Record explanatory metadata noting the absence of a split due to dataset size
        metrics_summary['accuracy'] = None
        metrics_summary['note'] = 'Dataset did not support a reliable held-out test split; fitted on all samples.'

    # Return the trained model, metrics dictionary, and structured evaluation DataFrames
    return model, metrics_summary, report_dataframe, confusion_dataframe


# ------------------------------------------------------------------------------
# 4. ARTIFACT PERSISTENCE & REPORT EXPORTATION
# ------------------------------------------------------------------------------

def save_artifacts(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    dataset: pd.DataFrame,
    metrics: Dict[str, Any],
    report_df: Optional[pd.DataFrame],
    confusion_df: Optional[pd.DataFrame]
) -> None:
    """Serialize model binaries, preprocessed datasets, and performance reports to disk.

    Args:
        model: Trained LogisticRegression classifier instance.
        vectorizer: Fitted TfidfVectorizer instance.
        dataset: Preprocessed DataFrame containing clean text and sentiment labels.
        metrics: Dictionary of evaluated metrics and training metadata.
        report_df: Optional DataFrame of precision, recall, and F1 metrics.
        confusion_df: Optional DataFrame representing the confusion matrix.
    """
    # Create the output directories if they do not already exist
    OUTPUTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # Persist the trained Logistic Regression model binary to disk using joblib compression
    joblib.dump(model, MODELS_DIR_PATH / 'sentiment_model.joblib')

    # Persist the fitted TF-IDF Vectorizer vocabulary and IDF weights to disk
    joblib.dump(vectorizer, MODELS_DIR_PATH / 'tfidf_vectorizer.joblib')

    # Export the preprocessed dataset with clean text and sentiment labels as CSV
    dataset.to_csv(OUTPUTS_DIR_PATH / 'sentiment_results.csv', index=False)

    # If classification report was computed, save it as a structured CSV report
    if report_df is not None:
        report_df.to_csv(OUTPUTS_DIR_PATH / 'classification_report.csv')

    # If confusion matrix was computed, save it as a structured CSV report
    if confusion_df is not None:
        confusion_df.to_csv(OUTPUTS_DIR_PATH / 'confusion_matrix.csv')

    # Save human-readable formatted metrics summary to a text file for quick inspection
    with open(OUTPUTS_DIR_PATH / 'model_metrics.txt', 'w', encoding='utf-8') as f:
        # Write clean formatted JSON representation with 4-space indentation
        f.write(json.dumps(metrics, indent=4))

    # Also persist standard machine-readable JSON metrics file for API/dashboard consumption
    with open(OUTPUTS_DIR_PATH / 'model_metrics.json', 'w', encoding='utf-8') as f:
        # Serialize entire metrics dictionary to standard JSON format
        json.dump(metrics, f, indent=4)


# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION ENTRYPOINT
# ------------------------------------------------------------------------------

def main() -> None:
    """Orchestrate end-to-end dataset loading, training, evaluation, and serialization."""
    # Print clear pipeline initialization banner to standard output
    print("=" * 70)
    print("  LOCAL BRAND SENTIMENT ANALYSIS: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Ingest and preprocess survey responses
    print("\n[Step 1/4] Ingesting survey responses and executing text normalization...")
    processed_df, source_col = load_and_preprocess_data(DATA_FILE_PATH)
    print(f"  -> Extracted survey question: '{source_col}'")
    print(f"  -> Successfully processed records: {len(processed_df)}")
    print(f"  -> Sentiment distribution:\n{processed_df['sentiment'].value_counts().to_string(index=True)}")

    # Step 2: Extract TF-IDF n-gram feature representations
    print("\n[Step 2/4] Vectorizing text corpus with TF-IDF (Unigrams + Bigrams)...")
    vectorizer, X_features = extract_features(processed_df['clean_text'])
    print(f"  -> Vocabulary feature dimension: {X_features.shape[1]} unique n-grams")

    # Step 3: Train Logistic Regression and calculate evaluation metrics
    print("\n[Step 3/4] Fitting balanced Logistic Regression & computing evaluation metrics...")
    model, metrics, report_df, confusion_df = train_and_evaluate(
        feature_matrix=X_features,
        labels=processed_df['sentiment'],
        target_column_name=source_col
    )
    if metrics.get('accuracy') is not None:
        print(f"  -> Model Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    else:
        print("  -> Model trained on full corpus (stratified holdout not applicable)")

    # Step 4: Persist all models, vectorizers, and evaluation files to disk
    print("\n[Step 4/4] Serializing model binaries and evaluation reports to disk...")
    save_artifacts(model, vectorizer, processed_df, metrics, report_df, confusion_df)
    print(f"  -> Model saved to: {MODELS_DIR_PATH / 'sentiment_model.joblib'}")
    print(f"  -> Vectorizer saved to: {MODELS_DIR_PATH / 'tfidf_vectorizer.joblib'}")
    print(f"  -> Processed dataset saved to: {OUTPUTS_DIR_PATH / 'sentiment_results.csv'}")
    print(f"  -> Metrics saved to: {OUTPUTS_DIR_PATH / 'model_metrics.json'}")

    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


# Standard Python boilerplate to ensure script executes only when invoked directly
if __name__ == '__main__':
    # Execute the primary training orchestration pipeline
    main()
