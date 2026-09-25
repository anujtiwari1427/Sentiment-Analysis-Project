# ==============================================================================
# Module: predict.py
# Description: Production-grade inference engine and interactive CLI for local
# brand customer sentiment prediction. Delivers machine learning classifications,
# posterior class probabilities (confidence scores), lexicon validation, and an
# intelligent hybrid ensemble consensus.
# Every single line is documented with technical and theoretical explanations.
# ==============================================================================

# Import Path for robust, cross-platform filesystem path navigation
from pathlib import Path

# Import sys module to modify runtime module search path and exit cleanly
import sys

# Import typing primitives for comprehensive static type annotations
from typing import Dict, Any, Tuple, Optional, List

# Import joblib to deserialize trained model estimators and TF-IDF vectorizers
import joblib

# Determine the absolute project root directory (two folder levels above: src/ -> root)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Inject the 'src' directory into sys.path to enable clean local imports
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import text normalization, aspect extraction, and lexicon sentiment functions from sentiment_utils
from sentiment_utils import clean_text, lexicon_sentiment, extract_aspect

# Define file paths to serialized model binary and vectorizer artifacts
MODEL_FILE_PATH: Path = ROOT_DIR / 'models' / 'sentiment_model.joblib'
VECTORIZER_FILE_PATH: Path = ROOT_DIR / 'models' / 'tfidf_vectorizer.joblib'


# ------------------------------------------------------------------------------
# 1. PRODUCTION INFERENCE ENGINE CLASS
# ------------------------------------------------------------------------------

class SentimentPredictor:
    """Production inference wrapper for customer opinion sentiment analysis.

    Encapsulates artifact loading, input sanitization, TF-IDF vectorization,
    calibrated probability estimation, lexicon auditing, and hybrid consensus.
    """

    def __init__(self, model_path: Path = MODEL_FILE_PATH, vectorizer_path: Path = VECTORIZER_FILE_PATH) -> None:
        """Initialize and deserialize the trained model and vectorizer artifacts.

        Args:
            model_path (Path): Filesystem path to the serialized scikit-learn classifier.
            vectorizer_path (Path): Filesystem path to the serialized TfidfVectorizer.
        """
        # Validate that the trained classifier artifact exists on disk before attempting to load
        if not model_path.exists():
            # Raise an explicit FileNotFoundError instructing the developer to train the model first
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run train_model.py first.")

        # Validate that the TF-IDF vectorizer artifact exists on disk
        if not vectorizer_path.exists():
            # Raise an explicit FileNotFoundError if the vectorizer file is missing
            raise FileNotFoundError(f"Vectorizer not found at '{vectorizer_path}'. Please run train_model.py first.")

        # Deserialize the trained Logistic Regression estimator into active memory
        self.model = joblib.load(model_path)

        # Deserialize the fitted TF-IDF vectorizer into active memory
        self.vectorizer = joblib.load(vectorizer_path)

        # Cache the list of class label names recognized by the trained classifier
        self.classes: List[str] = list(self.model.classes_)

    def predict(self, raw_comment: str) -> Dict[str, Any]:
        """Perform full sentiment inference on a raw customer comment string.

        Integrates machine learning TF-IDF classification with domain lexicon rules
        to generate robust predictions even for vocabulary unseen in survey training.

        Args:
            raw_comment (str): Unprocessed customer review or social media feedback.

        Returns:
            Dict[str, Any]: Comprehensive inference payload including ML prediction,
                            confidence score, class probabilities, lexicon cross-check,
                            and hybrid consensus sentiment.
        """
        # Handle empty, None, or whitespace-only strings gracefully
        if not raw_comment or not raw_comment.strip():
            # Return a default neutral payload if the provided text carries no content
            return {
                'raw_text': raw_comment,
                'clean_text': '',
                'sentiment': 'Neutral',
                'confidence': 0.0,
                'probabilities': {cls_name: 0.0 for cls_name in self.classes},
                'ml_sentiment': 'Neutral',
                'lexicon_sentiment': 'Neutral',
                'lexicon_score': 0,
                'consensus_sentiment': 'Neutral',
                'explanation': 'Input text was empty or contained only whitespace.'
            }

        # Apply standardized cleaning pipeline (lowercasing, punctuation stripping, regex cleanup)
        normalized_text: str = clean_text(raw_comment)

        # Transform normalized string into a sparse TF-IDF feature vector using cached vectorizer
        feature_vector = self.vectorizer.transform([normalized_text])

        # Compute deterministic class label prediction using the trained Logistic Regression model
        ml_predicted_class: str = str(self.model.predict(feature_vector)[0])

        # Compute posterior probabilities across all target classes using logistic sigmoid/softmax
        probability_distribution = self.model.predict_proba(feature_vector)[0]

        # Determine class index mapping for positional probability lookups
        class_indices: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Evaluate the comment using the deterministic lexicon engine for verification
        lex_label, lex_score = lexicon_sentiment(raw_comment)

        # Initialize prior belief weights for each class in the target distribution
        prior_weights: List[float] = [0.0] * len(self.classes)

        # Assign domain priors based on the presence and intensity of sentiment keywords
        if lex_score >= 1:
            # Positive evidence: assign high prior to Positive proportional to score magnitude
            for cls_name in self.classes:
                if cls_name == 'Positive':
                    prior_weights[class_indices[cls_name]] = 0.80 + min(0.15, 0.05 * lex_score)
                elif cls_name == 'Neutral':
                    prior_weights[class_indices[cls_name]] = 0.15
                else:
                    prior_weights[class_indices[cls_name]] = 0.05
        elif lex_score <= -1:
            # Negative evidence: assign high prior to Negative proportional to complaint magnitude
            for cls_name in self.classes:
                if cls_name == 'Negative':
                    prior_weights[class_indices[cls_name]] = 0.80 + min(0.15, 0.05 * abs(lex_score))
                elif cls_name == 'Neutral':
                    prior_weights[class_indices[cls_name]] = 0.15
                else:
                    prior_weights[class_indices[cls_name]] = 0.05
        else:
            # Neutral evidence: text contains no emotional polarity keywords (pure informational inquiry)
            for cls_name in self.classes:
                if cls_name == 'Neutral':
                    prior_weights[class_indices[cls_name]] = 0.80
                elif cls_name == 'Positive':
                    prior_weights[class_indices[cls_name]] = 0.10
                else:
                    prior_weights[class_indices[cls_name]] = 0.10

        # Calculate the sum of unnormalized prior weights
        sum_priors: float = sum(prior_weights)

        # Normalize prior distribution vector so sum equals unit 1.0
        normalized_priors: List[float] = [weight / sum_priors for weight in prior_weights]

        # Blend ML posterior distribution with domain prior (35% ML weight, 65% Lexicon prior)
        blended_probabilities: List[float] = [
            (0.35 * float(ml_prob)) + (0.65 * normalized_priors[idx])
            for idx, ml_prob in enumerate(probability_distribution)
        ]

        # Calculate sum of blended probabilities
        sum_blended: float = sum(blended_probabilities)

        # Re-normalize blended probability distribution to sum exactly to 1.0
        final_probabilities: List[float] = [bp / sum_blended for bp in blended_probabilities]

        # Map calibrated probabilities to their respective class labels
        calibrated_probs_dict: Dict[str, float] = {
            class_name: round(float(final_probabilities[class_indices[class_name]]), 4)
            for class_name in self.classes
        }

        # Identify winning class index with the maximum calibrated posterior probability
        winning_class_index: int = blended_probabilities.index(max(blended_probabilities))

        # Retrieve winning class name from cached class list
        consensus: str = str(self.classes[winning_class_index])

        # Record maximum probability as the calibrated confidence metric
        confidence_metric: float = round(float(final_probabilities[winning_class_index]), 4)

        # Formulate informative analytical rationale explaining classification drivers
        if consensus == 'Positive':
            rationale = (
                f"Classified as Positive ({confidence_metric * 100:.1f}% confidence) based on affirmative "
                f"sentiment tokens and favorable review patterns (Lexicon Score: {lex_score})."
            )
        elif consensus == 'Negative':
            rationale = (
                f"Classified as Negative ({confidence_metric * 100:.1f}% confidence) based on customer "
                f"grievance tokens and negative review patterns (Lexicon Score: {lex_score})."
            )
        else:
            rationale = (
                f"Classified as Neutral ({confidence_metric * 100:.1f}% confidence) based on objective, "
                f"informational inquiry language devoid of emotional sentiment tokens (Lexicon Score: {lex_score})."
            )

        # Assemble and return comprehensive analytical prediction payload
        return {
            'raw_text': raw_comment,
            'clean_text': normalized_text,
            'sentiment': consensus,
            'confidence': confidence_metric,
            'probabilities': calibrated_probs_dict,
            'ml_sentiment': ml_predicted_class,
            'lexicon_sentiment': lex_label,
            'lexicon_score': lex_score,
            'consensus_sentiment': consensus,
            'primary_aspect': extract_aspect(normalized_text),
            'explanation': rationale
        }


# ------------------------------------------------------------------------------
# 2. CONVENIENCE FUNCTION FOR DIRECT API USE
# ------------------------------------------------------------------------------

# Global singleton predictor instance to prevent redundant disk deserialization on repetitive calls
_PREDICTOR_INSTANCE: Optional[SentimentPredictor] = None

def predict_sentiment(text: str) -> str:
    """Convenience function returning the predicted sentiment class for a text string.

    Args:
        text (str): Input customer review or comment.

    Returns:
        str: Predicted consensus sentiment label ('Positive', 'Neutral', or 'Negative').
    """
    # Reference the module-level singleton predictor variable
    global _PREDICTOR_INSTANCE

    # Instantiate the predictor once if it has not yet been initialized
    if _PREDICTOR_INSTANCE is None:
        # Load models from disk into singleton instance
        _PREDICTOR_INSTANCE = SentimentPredictor()

    # Execute inference and extract the predicted sentiment category string
    result = _PREDICTOR_INSTANCE.predict(text)

    # Return the categorical consensus sentiment prediction label
    return result['sentiment']


# ------------------------------------------------------------------------------
# 3. INTERACTIVE COMMAND-LINE INTERFACE (CLI)
# ------------------------------------------------------------------------------

def interactive_cli() -> None:
    """Run an interactive console loop allowing users to test sentiment predictions live."""
    # Print welcome header banner to terminal
    print("=" * 70)
    print("  LOCAL BRAND SENTIMENT ANALYZER: INTERACTIVE PREDICTION CONSOLE")
    print("=" * 70)
    print("Type a customer comment and press Enter to evaluate sentiment.")
    print("Type 'exit' or 'quit' at any prompt to terminate the session.\n")

    # Initialize the sentiment predictor instance
    try:
        # Instantiate prediction engine
        predictor = SentimentPredictor()
    except Exception as exc:
        # Print error details if artifacts are missing or unreadable
        print(f"[ERROR] Failed to load model artifacts: {exc}")
        print("Please ensure you have executed: python src/train_model.py")
        return

    # Enter continuous user input evaluation loop
    while True:
        try:
            # Prompt user for input text from terminal
            user_input: str = input("\nEnter customer comment: ").strip()

            # Check if user entered termination command
            if user_input.lower() in {'exit', 'quit', 'q'}:
                # Print exit message and break out of loop
                print("\nExiting sentiment console. Goodbye!")
                break

            # Skip empty inputs and prompt again
            if not user_input:
                # Inform user that input was empty
                print("Please enter a non-empty comment.")
                continue

            # Compute detailed prediction payload
            analysis: Dict[str, Any] = predictor.predict(user_input)

            # Display formatted analytical results to terminal
            print("\n" + "-" * 55)
            print(f"  Final Sentiment     : {analysis['sentiment'].upper()}")
            print(f"  Confidence Score    : {analysis['confidence'] * 100:.1f}%")
            print(f"  ML Class Prediction : {analysis['ml_sentiment']}")
            print(f"  Class Probabilities : {analysis['probabilities']}")
            print(f"  Lexicon Rule Check  : {analysis['lexicon_sentiment']} (Score: {analysis['lexicon_score']})")
            print(f"  Analysis Rationale  : {analysis['explanation']}")
            print("-" * 55)

        # Handle keyboard interrupt gracefully (Ctrl+C)
        except KeyboardInterrupt:
            # Terminate loop cleanly on user interrupt
            print("\nSession interrupted. Exiting.")
            break


# Standard script execution check
if __name__ == '__main__':
    # Launch interactive command-line interface
    interactive_cli()
