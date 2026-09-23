# ==============================================================================
# Module: sentiment_utils.py
# Description: Production-grade NLP text preprocessing and lexicon sentiment engine.
# Provides robust text cleaning, domain-specific sentiment scoring with negation
# handling, and automatic column detection for survey datasets.
# Every single line is documented with technical and theoretical explanations.
# ==============================================================================

# Import standard library regular expressions module for pattern matching and text substitutions
import re

# Import string module providing access to pre-defined ASCII punctuation constants
import string

# Import Path class from pathlib for cross-platform, object-oriented filesystem paths
from pathlib import Path

# Import typing primitives for comprehensive static type hinting and code clarity
from typing import Tuple, Optional, List, Set, Any

# Import pandas for tabular data structure manipulation and type annotations
import pandas as pd


# ------------------------------------------------------------------------------
# 1. DOMAIN-SPECIFIC SENTIMENT LEXICONS
# ------------------------------------------------------------------------------

# Define positive sentiment keywords tailored specifically to social media local brand reviews
# Using a Python set provides O(1) average-time complexity membership lookups during text scanning
POSITIVE_WORDS: Set[str] = {
    # General positive evaluation terms
    'good', 'great', 'excellent', 'amazing', 'awesome', 'best', 'love', 'loved',
    'like', 'liked', 'happy', 'pleased', 'delighted', 'glad', 'satisfying',
    'satisfied', 'satisfaction', 'impressive', 'impressed', 'superb', 'wonderful',
    
    # Customer service & responsiveness terms
    'helpful', 'fast', 'quick', 'rapid', 'speedy', 'responsive', 'response',
    'supportive', 'kind', 'friendly', 'polite', 'professional', 'attentive',
    
    # Product quality, authenticity & craftsmanship terms
    'quality', 'high', 'durable', 'premium', 'genuine', 'authentic', 'honest',
    'trustworthy', 'trust', 'reliable', 'dependable', 'safe', 'clean', 'transparent',
    
    # Value, pricing, & recommendation terms
    'affordable', 'reasonable', 'worth', 'worthy', 'useful', 'nice', 'better',
    'improve', 'improvement', 'success', 'successful', 'recommend', 'recommended',
    'seamless', 'perfect', 'fabulous', 'stellar', 'flawless', 'exceptional'
}

# Define negative sentiment keywords capturing customer grievances, distrust, and poor service
# Pre-allocating as a hash set ensures constant-time O(1) lookup during sentiment classification
NEGATIVE_WORDS: Set[str] = {
    # Severe dissatisfaction and affective negative terms
    'bad', 'poor', 'terrible', 'worst', 'hate', 'hated', 'dislike', 'unhappy',
    'horrible', 'awful', 'dreadful', 'useless', 'disgusting', 'pathetic',
    
    # Deceptive behavior, distrust & authenticity issues common to local brand reviews
    'fake', 'scam', 'fraud', 'cheat', 'cheated', 'dishonest', 'misleading',
    'counterfeit', 'untrustworthy', 'suspicious', 'shady', 'bogus',
    
    # Delivery, delay, and responsiveness failure terms
    'slow', 'late', 'delayed', 'delay', 'unresponsive', 'rude', 'careless',
    'ignored', 'ignoring', 'ignore', 'neglect', 'neglected', 'unreachable',
    
    # Price, value, and physical product failure terms
    'expensive', 'overpriced', 'costly', 'low', 'defect', 'defective', 'damaged',
    'broken', 'faulty', 'flawed', 'inferior', 'trash', 'garbage', 'waste',
    
    # Service grievance and frustration terms
    'issue', 'issues', 'problem', 'problems', 'complaint', 'complaints',
    'frustrating', 'frustrated', 'disappointed', 'disappointment',
    'unsatisfied', 'unsatisfactory', 'annoying', 'annoyed', 'regret'
}

# Define grammatical negation words that invert the polar meaning of subsequent sentiment words
# Stored as a set for efficient O(1) contextual checking in n-gram token windows
NEGATION_WORDS: Set[str] = {
    'not', 'no', 'never', 'neither', 'nor', 'hardly', 'scarcely', 'barely',
    'dont', 'doesnt', 'didnt', 'isnt', 'wasnt', 'arent', 'werent',
    'wont', 'wouldnt', 'cant', 'cannot', 'couldnt', 'shouldnt', 'havent', 'hasnt', 'hadnt'
}


# ------------------------------------------------------------------------------
# 2. TEXT NORMALIZATION & PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """Standardize, clean, and normalize raw text for NLP pipelines.

    Performs case folding, URL removal, contraction normalization,
    punctuation stripping, digit removal, and whitespace collapse.

    Args:
        text (Optional[str]): Raw input string or potentially None/NaN value.

    Returns:
        str: Fully cleaned, lowercased, and sanitized text string.
    """
    # Guard against None, NaN, or non-string inputs by coercing to empty string or str
    if text is None:
        # Assign empty string immediately if the provided input is None
        return ""
    
    # Coerce input to string type to safely handle numeric or timestamp objects from Excel
    text = str(text)
    
    # Remove hyperlinks (http://, https://, and www. addresses) using compiled regex patterns
    # Matching non-whitespace characters (\S+) following the URL protocol or www prefix
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Convert all characters to lowercase to eliminate vocabulary duplication due to casing
    text = text.lower()
    
    # Normalize standard apostrophes and single quotation marks to ensure contraction consistency
    # E.g., transforming "don't" into "dont", allowing straightforward matching with negation sets
    text = text.replace("'", "").replace("’", "").replace("`", "")
    
    # Strip all standard ASCII punctuation symbols using str.maketrans mapping table
    # This replaces each punctuation character with None (deleting it from the resulting string)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Replace all numerical digits (\d+) with a single space to focus solely on semantic words
    text = re.sub(r'\d+', ' ', text)
    
    # Collapse multiple contiguous whitespace characters (\s+) into a single clean space
    # and strip leading and trailing whitespace from the final normalized text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return the clean normalized string ready for tokenization or vectorization
    return text


# ------------------------------------------------------------------------------
# 3. RULE-BASED LEXICON SENTIMENT SCORING WITH CONTEXTUAL NEGATION
# ------------------------------------------------------------------------------

def lexicon_sentiment(text: str) -> Tuple[str, int]:
    """Compute deterministic sentiment polarity using an augmented lexicon engine.

    Iterates through normalized word tokens, assigns polarity scores (+1/-1),
    and inspects prior token contexts (up to 2 tokens back) for negations.

    Args:
        text (str): Input customer review or comment.

    Returns:
        Tuple[str, int]: A tuple containing the sentiment label ('Positive',
                         'Neutral', or 'Negative') and the aggregate integer score.
    """
    # Sanitize and normalize input text through the established preprocessing pipeline
    cleaned: str = clean_text(text)
    
    # Split the cleaned text on spaces into a sequential list of individual word tokens
    tokens: List[str] = cleaned.split()
    
    # Initialize the cumulative sentiment polarity score accumulator to zero
    score: int = 0
    
    # Iterate through tokens with their respective positional index to enable look-behind analysis
    for i, token in enumerate(tokens):
        # Determine base polarity: +1 if token is in POSITIVE_WORDS, -1 if in NEGATIVE_WORDS, else 0
        if token in POSITIVE_WORDS:
            # Assign positive unit score for words expressing satisfaction or brand trust
            token_val: int = 1
        elif token in NEGATIVE_WORDS:
            # Assign negative unit score for words expressing dissatisfaction or brand distrust
            token_val: int = -1
        else:
            # Assign zero score for neutral or domain-agnostic vocabulary tokens
            token_val: int = 0
        
        # Check for contextual negation if the current token carries non-zero sentiment polarity
        if token_val != 0:
            # Flag indicating whether a preceding negation was detected in the context window
            is_negated: bool = False
            
            # Inspect immediate preceding token (1 token look-behind, e.g., "not good")
            if i > 0 and tokens[i - 1] in NEGATION_WORDS:
                # Set negation flag to true when immediate predecessor is a negation word
                is_negated = True
            
            # Inspect secondary preceding token (2 tokens look-behind, e.g., "not very good")
            elif i > 1 and tokens[i - 2] in NEGATION_WORDS:
                # Set negation flag to true when modifier separates negation from sentiment word
                is_negated = True
            
            # Invert the polarity value if negation was detected in the contextual window
            if is_negated:
                # Multiply token polarity by -1 (inverting positive to negative and vice-versa)
                token_val = -token_val
            
            # Add the evaluated token polarity score to the running total score
            score += token_val
    
    # Map final cumulative integer polarity score to categorical sentiment labels
    if score > 0:
        # Scores strictly greater than zero designate net positive customer sentiment
        label: str = 'Positive'
    elif score < 0:
        # Scores strictly less than zero designate net negative customer sentiment
        label: str = 'Negative'
    else:
        # A score of exactly zero designates neutral or balanced customer sentiment
        label: str = 'Neutral'
    
    # Return a 2-element tuple consisting of the discrete label and integer polarity score
    return label, score


# ------------------------------------------------------------------------------
# 4. INTELLIGENT COLUMN DISCOVERY FOR SURVEY EXCEL DATASETS
# ------------------------------------------------------------------------------

def load_text_column(df: pd.DataFrame) -> str:
    """Identify the primary open-ended brand opinion column in the survey dataset.

    Inspects column header substrings dynamically to locate question prompts
    regarding customer opinions or reasons for brand perception.

    Args:
        df (pd.DataFrame): Input survey dataset loaded as a pandas DataFrame.

    Returns:
        str: Selected column name containing the primary open-ended text.
    """
    # Search for column headers explicitly containing the substring 'main reason' (case-insensitive)
    candidates: List[Any] = [c for c in df.columns if 'main reason' in str(c).lower()]
    
    # If a match is identified, return the first matching column name
    if candidates:
        # Return detected primary question column
        return str(candidates[0])
    
    # Fallback search for column headers containing both 'reason' and 'opinion' or 'positive'
    candidates = [
        c for c in df.columns 
        if 'reason' in str(c).lower() and ('opinion' in str(c).lower() or 'positive' in str(c).lower())
    ]
    
    # If secondary candidate match is found, return that column header
    if candidates:
        # Return fallback matching question column
        return str(candidates[0])
    
    # Default fallback: pick the penultimate column, standard for survey opinion prompts
    return str(df.columns[-2])


def load_suggestion_column(df: pd.DataFrame) -> str:
    """Identify the suggestion or feedback column in the survey dataset.

    Args:
        df (pd.DataFrame): Input survey dataset loaded as a pandas DataFrame.

    Returns:
        str: Selected column name containing customer suggestions.
    """
    # Search for column headers explicitly containing 'suggestion' or 'feedback'
    candidates: List[Any] = [
        c for c in df.columns 
        if 'suggestion' in str(c).lower() or 'feedback' in str(c).lower() or 'improve' in str(c).lower()
    ]
    
    # If candidate column exists, return the first match
    if candidates:
        # Return identified suggestion column name
        return str(candidates[0])
    
    # Default fallback: return the very last column in the survey sheet
    return str(df.columns[-1])
