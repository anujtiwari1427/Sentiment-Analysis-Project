# ==============================================================================
# Module: sentiment_utils.py
# Description: Production-grade NLP text preprocessing, aspect extraction,
# and calibrated lexicon sentiment engine. Provides advanced text cleaning,
# informal social media typo resolution, domain aspect detection, noise tagging,
# and contextual lookbehind negation handling.
# ==============================================================================

import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ------------------------------------------------------------------------------
# 1. DOMAIN-SPECIFIC SENTIMENT LEXICONS
# ------------------------------------------------------------------------------

POSITIVE_WORDS: Set[str] = {
    # Direct sentiment indicators
    'positive', 'good', 'great', 'excellent', 'amazing', 'awesome', 'best', 'love', 'loved',
    'like', 'liked', 'happy', 'pleased', 'delighted', 'glad', 'satisfying', 'satisfied',
    'satisfaction', 'impressive', 'impressed', 'superb', 'wonderful',

    # Customer service & responsiveness
    'helpful', 'fast', 'quick', 'rapid', 'speedy', 'responsive', 'responsiveness', 'response',
    'supportive', 'kind', 'friendly', 'polite', 'professional', 'attentive', 'prompt',
    'empathy', 'empathetic', 'care', 'caring',

    # Product quality, authenticity & craftsmanship
    'quality', 'high', 'durable', 'premium', 'genuine', 'authentic', 'authenticity', 'honest',
    'honesty', 'trustworthy', 'trustworthiness', 'trust', 'reliable', 'dependable', 'safe',
    'clean', 'transparent', 'transparency', 'consistent', 'consistency',

    # Value, pricing & recommendations
    'affordable', 'reasonable', 'worth', 'worthy', 'useful', 'nice', 'better', 'well',
    'improve', 'improvement', 'success', 'successful', 'recommend', 'recommended',
    'recommendation', 'seamless', 'perfect', 'fabulous', 'stellar', 'flawless', 'exceptional'
}

NEGATIVE_WORDS: Set[str] = {
    # Direct sentiment indicators
    'negative', 'bad', 'poor', 'terrible', 'worst', 'hate', 'hated', 'dislike', 'unhappy',
    'horrible', 'awful', 'dreadful', 'useless', 'disgusting', 'pathetic',

    # Deceptive behavior, distrust & authenticity issues
    'fake', 'scam', 'fraud', 'cheat', 'cheated', 'dishonest', 'misleading', 'counterfeit',
    'untrustworthy', 'suspicious', 'shady', 'bogus', 'unethical',

    # Delivery, delay & responsiveness failures
    'slow', 'late', 'delayed', 'delay', 'unresponsive', 'rude', 'careless', 'ignored',
    'ignoring', 'ignore', 'neglect', 'neglected', 'unreachable', 'arrogant', 'arrogance',
    'delete', 'deleting',

    # Price, value, and physical product failure
    'expensive', 'overpriced', 'costly', 'low', 'defect', 'defective', 'damaged', 'broken',
    'faulty', 'flawed', 'inferior', 'trash', 'garbage', 'waste',

    # Service grievance and frustration
    'issue', 'issues', 'problem', 'problems', 'complaint', 'complaints', 'frustrating',
    'frustrated', 'disappointed', 'disappointment', 'unsatisfied', 'unsatisfactory',
    'annoying', 'annoyed', 'regret', 'destroy', 'destroys', 'shatter', 'shatters'
}

NEGATION_WORDS: Set[str] = {
    'not', 'no', 'never', 'neither', 'nor', 'hardly', 'scarcely', 'barely',
    'dont', 'doesnt', 'didnt', 'isnt', 'wasnt', 'arent', 'werent',
    'wont', 'wouldnt', 'cant', 'cannot', 'couldnt', 'shouldnt', 'havent', 'hasnt', 'hadnt'
}

# Known non-informative or placeholder survey responses
NOISE_TERMS: Set[str] = {
    'idk', 'no idea', 'hm', 'hh', 'to view', 'nothing else', 'none', 'comments',
    'g', 'cash flow', 'experience', 'company strategy', 'yes', 'no', 'ohhh yesss',
    'yesssss', 'research about produ', 'presentation', 'online shopping', 'to view',
    'to create public awareness'
}


# ------------------------------------------------------------------------------
# 2. ADVANCED TEXT NORMALIZATION & PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """Standardize, normalize, and clean raw customer comments for NLP pipelines.

    Replaces punctuation delimiters with spaces to prevent token collision (e.g.
    'Quality,etc' -> 'quality etc'), normalizes social media abbreviations and typos,
    removes URLs, digits, and unicode encoding artifacts.

    Args:
        text (Optional[str]): Raw input review or comment.

    Returns:
        str: Fully cleaned, normalized, lowercased string.
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Normalize unicode quotation marks and dashes
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\ufffd", " ")

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Lowercase
    text = text.lower()

    # 4. Replace separators and delimiters with spaces so words do not collide
    # e.g., 'price, quality,etc' -> 'price quality etc', 'positive/negative' -> 'positive negative'
    text = re.sub(r'[/\\_\-,\;\:\!\?\(\)\[\]\{\}<>|*&^%$#@+=.]+', ' ', text)

    # 5. Remove quotation marks for consistent contraction matching
    text = text.replace("'", "").replace('"', "").replace("`", "")

    # 6. Normalize common informal social media typos & abbreviations using word boundaries
    typo_mappings = [
        (r'\bcoustomer\b|\bcustomer\b', 'customer'),
        (r'\bcoustomers\b|\bcustomers\b', 'customers'),
        (r'\bbcz\b|\bbcuz\b|\bbcoz\b', 'because'),
        (r'\bprodu9\b', 'product'),
        (r'\btrustworthyness\b', 'trustworthiness'),
        (r'\bdms\b', 'direct messages'),
        (r'\bdm\b', 'direct message'),
        (r'\bapprox\b', 'approximately'),
        (r'\bv\.\b|\bv\b', 'very'),
        (r'\bpls\b|\bplz\b', 'please'),
        (r'\bidk\b', 'i do not know'),
        (r'\bdont\b', 'dont'),
        (r'\bcant\b', 'cant'),
    ]
    for pattern, replacement in typo_mappings:
        text = re.sub(pattern, replacement, text)

    # 7. Strip digits and non-alphanumeric noise (keep clean ascii letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 8. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_noise(text: str) -> bool:
    """Identify whether a raw comment is non-informative or a placeholder.

    Args:
        text (str): Cleaned or raw comment string.

    Returns:
        bool: True if comment contains little to no actionable sentiment information.
    """
    cleaned = clean_text(text)
    if not cleaned or len(cleaned) <= 2:
        return True
    if cleaned in NOISE_TERMS:
        return True
    return False


# ------------------------------------------------------------------------------
# 3. DOMAIN ASPECT & THEME EXTRACTION
# ------------------------------------------------------------------------------

def extract_aspect(text: str) -> str:
    """Identify the primary business domain aspect addressed in a customer comment.

    Args:
        text (str): Raw or cleaned comment text.

    Returns:
        str: Categorical domain aspect (e.g. 'Customer Service', 'Product Quality').
    """
    cleaned = clean_text(text)
    tokens: Set[str] = set(cleaned.split())

    service_tokens: Set[str] = {
        'service', 'customer', 'response', 'respond', 'responds', 'dms', 'reply',
        'replies', 'support', 'complaint', 'complaints', 'empathy', 'rude', 'polite',
        'helpful', 'prompt', 'communication', 'message', 'messages'
    }
    quality_tokens: Set[str] = {
        'quality', 'product', 'products', 'durable', 'durability', 'authentic',
        'authenticity', 'genuine', 'cloth', 'defect', 'defective', 'finish', 'craftsmanship'
    }
    delivery_tokens: Set[str] = {
        'delivery', 'shipping', 'delayed', 'delay', 'slow', 'fast', 'speed', 'tracking',
        'dispatch', 'order', 'orders', 'scam'
    }
    pricing_tokens: Set[str] = {
        'price', 'pricing', 'cost', 'expensive', 'affordable', 'rate', 'rates',
        'cheap', 'overpriced', 'value', 'money', 'offers', 'discount'
    }
    social_tokens: Set[str] = {
        'reviews', 'review', 'influencer', 'influencers', 'reputation', 'trust',
        'advertising', 'advertise', 'branding', 'feedback', 'posts', 'recommendations'
    }

    aspect_scores: Dict[str, int] = {
        'Customer Service': len(tokens & service_tokens),
        'Product Quality': len(tokens & quality_tokens),
        'Delivery & Reliability': len(tokens & delivery_tokens),
        'Pricing & Value': len(tokens & pricing_tokens),
        'Social Proof & Reviews': len(tokens & social_tokens)
    }

    top_aspect = max(aspect_scores, key=aspect_scores.get)
    if aspect_scores[top_aspect] > 0:
        return top_aspect
    return 'General Brand Perception'


# ------------------------------------------------------------------------------
# 4. RULE-BASED LEXICON SENTIMENT SCORING WITH CONTEXTUAL NEGATION
# ------------------------------------------------------------------------------

def lexicon_sentiment(text: str) -> Tuple[str, int]:
    """Compute calibrated sentiment polarity using an augmented lexicon engine.

    Includes compound phrase detection (e.g. 'not same', 'fake review'),
    lookbehind negation windows, and domain-tuned polarity scores.

    Args:
        text (str): Input customer review or comment.

    Returns:
        Tuple[str, int]: (sentiment_label, aggregate_polarity_score)
    """
    cleaned: str = clean_text(text)

    # Empty or noise comments default to Neutral
    if is_noise(cleaned):
        return 'Neutral', 0

    tokens: List[str] = cleaned.split()
    score: int = 0

    # Compound negative phrase detection
    if 'not same' in cleaned or 'not as advertised' in cleaned:
        score -= 2
    if 'fake review' in cleaned or 'fake followers' in cleaned:
        score -= 2

    # Token-level scoring with negation window
    for i, token in enumerate(tokens):
        token_val = 0
        if token in POSITIVE_WORDS:
            token_val = 1
        elif token in NEGATIVE_WORDS:
            token_val = -1

        if token_val != 0:
            is_negated = False
            # Check 1 token look-behind
            if i > 0 and tokens[i - 1] in NEGATION_WORDS:
                is_negated = True
            # Check 2 tokens look-behind (e.g. 'not very good')
            elif i > 1 and tokens[i - 2] in NEGATION_WORDS:
                is_negated = True

            if is_negated:
                token_val = -token_val

            score += token_val

    # Final label resolution
    if score > 0:
        label = 'Positive'
    elif score < 0:
        label = 'Negative'
    else:
        label = 'Neutral'

    return label, score


# ------------------------------------------------------------------------------
# 5. INTELLIGENT COLUMN DISCOVERY FOR SURVEY EXCEL DATASETS
# ------------------------------------------------------------------------------

def load_text_column(df: pd.DataFrame) -> str:
    """Identify the primary open-ended brand opinion column in the survey dataset."""
    candidates: List[Any] = [c for c in df.columns if 'main reason' in str(c).lower()]
    if candidates:
        return str(candidates[0])

    candidates = [
        c for c in df.columns
        if 'reason' in str(c).lower() and ('opinion' in str(c).lower() or 'positive' in str(c).lower())
    ]
    if candidates:
        return str(candidates[0])

    return str(df.columns[-2])


def load_suggestion_column(df: pd.DataFrame) -> str:
    """Identify the suggestion or feedback column in the survey dataset."""
    candidates: List[Any] = [
        c for c in df.columns
        if 'suggestion' in str(c).lower() or 'feedback' in str(c).lower() or 'improve' in str(c).lower()
    ]
    if candidates:
        return str(candidates[0])

    return str(df.columns[-1])
