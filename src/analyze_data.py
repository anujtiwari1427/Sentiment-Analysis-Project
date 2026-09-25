# ==============================================================================
# Module: analyze_data.py
# Description: Production-grade Exploratory Data Analysis (EDA) engine.
# Generates comprehensive data dictionaries, high-resolution aesthetic word clouds,
# sentiment frequency distributions, and categorical survey question charts.
# Every single line is documented with technical and theoretical explanations.
# ==============================================================================

# Import Path from pathlib for safe, cross-platform filesystem path manipulation
from pathlib import Path

# Import sys module to dynamically append repository paths for internal module imports
import sys

# Import typing primitives for strict type safety and clarity
from typing import List, Optional

# Import pandas for high-performance tabular data structures and Excel data extraction
import pandas as pd

# Import matplotlib.pyplot for publication-quality statistical charts and figure generation
import matplotlib.pyplot as plt

# Import WordCloud library to generate frequency-weighted visual term clouds
from wordcloud import WordCloud, STOPWORDS

# Determine the absolute project root directory (two directory levels up: src/ -> root)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Inject the 'src' directory into Python's module resolution path list
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import text preprocessing functions and dynamic survey column detection routines
from sentiment_utils import clean_text, load_text_column, load_suggestion_column

# Define standard filesystem location of the source Excel survey dataset
DATA_FILE_PATH: Path = ROOT_DIR / 'data' / 'survey_responses.xlsx'

# Define standard filesystem directory path where generated analysis charts will be saved
OUTPUTS_DIR_PATH: Path = ROOT_DIR / 'outputs'

# Ensure the outputs directory exists on disk, creating parent folders if necessary
OUTPUTS_DIR_PATH.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 1. DATA DICTIONARY GENERATION
# ------------------------------------------------------------------------------

def generate_data_dictionary(df: pd.DataFrame, output_dir: Path) -> Path:
    """Analyze DataFrame schema and generate a structured data dictionary CSV.

    Records column names, assigned data types, total non-null counts,
    and missing value statistics for full analytical reproducibility.

    Args:
        df (pd.DataFrame): Raw survey responses DataFrame.
        output_dir (Path): Destination folder for the generated CSV.

    Returns:
        Path: Destination path of the exported data dictionary file.
    """
    # Construct a metadata summary DataFrame capturing structural schema information
    data_dict_df: pd.DataFrame = pd.DataFrame({
        # Capture the original full question text or column header name
        'column_name': [str(col) for col in df.columns],
        
        # Record the internal pandas data type (e.g., object, int64, float64)
        'data_type': [str(dtype) for dtype in df.dtypes],
        
        # Count the absolute number of non-missing (valid) responses per question
        'valid_count': [int(df[col].notna().sum()) for col in df.columns],
        
        # Count the absolute number of missing (NaN/empty) entries per question
        'missing_count': [int(df[col].isna().sum()) for col in df.columns],
        
        # Compute the percentage of missing values relative to the entire dataset size
        'missing_percentage': [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
        
        # Compute count of distinct unique values per question column
        'unique_values_count': [int(df[col].nunique()) for col in df.columns]
    })

    # Define target CSV file path for the data dictionary artifact
    dict_csv_path: Path = output_dir / 'data_dictionary.csv'

    # Export DataFrame to CSV format without writing the default numerical row index
    data_dict_df.to_csv(dict_csv_path, index=False)

    # Return the path to the written data dictionary file
    return dict_csv_path


# ------------------------------------------------------------------------------
# 2. HIGH-RESOLUTION WORD CLOUD GENERATION
# ------------------------------------------------------------------------------

def generate_wordcloud(
    text_corpus: str,
    output_image_path: Path,
    chart_title: str,
    colormap_name: str = 'viridis'
) -> Optional[Path]:
    """Generate and export a publication-ready WordCloud from a normalized text corpus.

    Args:
        text_corpus (str): Aggregated string of cleaned text documents.
        output_image_path (Path): Target file path for the output PNG image.
        chart_title (str): Title displayed above the generated word cloud.
        colormap_name (str): Matplotlib color palette for visual styling.

    Returns:
        Optional[Path]: Output image path if text is non-empty, else None.
    """
    # Guard against empty or whitespace-only text inputs to prevent generation errors
    if not text_corpus or not text_corpus.strip():
        # Return None immediately if no informative text is available to visualize
        return None

    # Define custom domain-specific stopwords to eliminate filler survey terminology
    custom_stopwords = set(STOPWORDS)
    
    # Add domain noise words that do not add distinct sentiment signal
    custom_stopwords.update({
        'brand', 'brands', 'local', 'social', 'media', 'product', 'products', 'make', 'makes',
        'etc', 'idk', 'hm', 'hh', 'none', 'nothing', 'else', 'comments', 'view'
    })

    # Instantiate the WordCloud generator with high resolution and crisp font rendering
    wordcloud_generator = WordCloud(
        width=1600,                         # Width in pixels for crisp ultra-high definition rendering
        height=800,                         # Height in pixels preserving standard 2:1 widescreen aspect ratio
        background_color='#0E1117',         # Elegant deep dark theme background matching modern dashboards
        colormap=colormap_name,             # Color gradient applied to word frequencies
        collocations=False,                 # Avoid duplicate bigram pairing to focus on core semantic terms
        stopwords=custom_stopwords,         # Filter out non-informative generic words
        max_words=100,                      # Restrict to top 100 most salient terms for visual clarity
        random_state=42                     # Deterministic layout reproducibility across runs
    )

    # Compute word frequencies and layout coordinates from the aggregated corpus
    generated_cloud = wordcloud_generator.generate(text_corpus)

    # Initialize a new matplotlib figure with 16x8 inch canvas dimensions
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#0E1117')

    # Render word cloud image array with bilinear interpolation for smooth antialiased edges
    ax.imshow(generated_cloud, interpolation='bilinear')

    # Disable axis tick lines, numbers, and border spines for an unobstructed visual
    ax.axis('off')

    # Apply a modern, clean figure title with white text and padding
    ax.set_title(chart_title, fontsize=20, color='white', fontweight='bold', pad=20)

    # Adjust subplot margins tightly to eliminate wasteful whitespace padding
    plt.tight_layout()

    # Save rendered graphic to disk at 200 DPI for publication and dashboard display
    plt.savefig(output_image_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())

    # Close the figure to immediately release system memory and prevent GUI leaks
    plt.close(fig)

    # Return the path to the newly written word cloud image
    return output_image_path


# ------------------------------------------------------------------------------
# 3. CATEGORICAL SURVEY QUESTION VISUALIZATION
# ------------------------------------------------------------------------------

def generate_categorical_charts(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Generate polished horizontal bar charts for categorical survey questions.

    Args:
        df (pd.DataFrame): Raw survey responses DataFrame.
        output_dir (Path): Destination folder for generated chart images.

    Returns:
        List[Path]: List of file paths for all generated PNG charts.
    """
    # Initialize an empty accumulator list to track paths of created chart images
    generated_chart_paths: List[Path] = []

    # Modern, cohesive color palette for styling categorical response bars
    bar_colors: List[str] = ['#4F46E5', '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#8B5CF6']

    # Iterate over representative multiple-choice questions (columns index 4 through 10)
    for index, column_name in enumerate(df.columns[4:10]):
        # Drop missing values and cast categorical responses to string
        clean_series: pd.Series = df[column_name].dropna().astype(str)

        # Inspect if column is a valid categorical question (between 2 and 15 unique answers)
        if 1 < clean_series.nunique() <= 15 and len(clean_series) > 0:
            # Extract top 10 most frequent response categories
            value_counts_series: pd.Series = clean_series.value_counts().head(10)

            # Sort ascending so the most popular response displays at the top of the horizontal bar chart
            sorted_counts: pd.Series = value_counts_series.sort_values(ascending=True)

            # Initialize figure and axes with 10x5 inch dimensions and clean styling
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FAFAFA')
            
            # Set background color of the plot area itself
            ax.set_facecolor('#FFFFFF')

            # Render horizontal bars with a modern aesthetic color
            bars = ax.barh(
                y=sorted_counts.index,
                width=sorted_counts.values,
                color=bar_colors[index % len(bar_colors)],
                edgecolor='none',
                height=0.65
            )

            # Add numeric frequency labels directly to the end of each bar for readability
            for bar in bars:
                # Retrieve bar width which corresponds to the response count value
                width_val = bar.get_width()
                
                # Render text label slightly offset from the bar edge
                ax.text(
                    width_val + (max(sorted_counts.values) * 0.01),  # X position slightly to the right of bar
                    bar.get_y() + bar.get_height() / 2,              # Centered vertically within the bar
                    f" {int(width_val)}",                            # Formatted integer response count
                    va='center',                                      # Vertical alignment centered
                    ha='left',                                        # Horizontal alignment to the left
                    fontsize=10,                                      # Font size for metric text
                    fontweight='bold',                                # Bold weight for visual emphasis
                    color='#334155'                                   # Slate gray tone
                )

            # Format question header into a concise title (truncated at 65 characters if necessary)
            chart_title_str: str = str(column_name)
            if len(chart_title_str) > 65:
                # Append ellipsis to indicate truncation of long survey questions
                chart_title_str = chart_title_str[:62] + '...'

            # Set chart title with styling
            ax.set_title(chart_title_str, fontsize=12, fontweight='bold', color='#1E293B', pad=15)

            # Set X-axis label
            ax.set_xlabel('Number of Survey Respondents', fontsize=10, color='#475569', labelpad=10)

            # Format axis ticks with clean font styling
            ax.tick_params(axis='y', labelsize=9, colors='#334155')
            ax.tick_params(axis='x', labelsize=9, colors='#64748B')

            # Remove unsightly top and right chart spines for a minimalist, modern aesthetic
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CBD5E1')
            ax.spines['bottom'].set_color('#CBD5E1')

            # Add subtle vertical grid lines to guide the eye across response count bars
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='#94A3B8')
            
            # Ensure grid lines are drawn behind bar patches
            ax.set_axisbelow(True)

            # Adjust layout tightly to prevent label clipping
            plt.tight_layout()

            # Construct target path for saving the chart image
            chart_target_path: Path = output_dir / f"chart_{index + 1}.png"

            # Export plot graphic at 180 DPI resolution
            plt.savefig(chart_target_path, dpi=180, bbox_inches='tight')

            # Close figure canvas to free memory
            plt.close(fig)

            # Record created chart path
            generated_chart_paths.append(chart_target_path)

    # Return the complete list of successfully generated chart image paths
    return generated_chart_paths


# ------------------------------------------------------------------------------
# 4. MAIN EXPLORATORY DATA ANALYSIS PIPELINE ENTRYPOINT
# ------------------------------------------------------------------------------

def main() -> None:
    """Execute complete end-to-end Exploratory Data Analysis workflow."""
    # Print pipeline banner
    print("=" * 70)
    print("  LOCAL BRAND SENTIMENT ANALYSIS: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # Verify presence of raw survey file
    if not DATA_FILE_PATH.exists():
        # Raise error if survey file cannot be found
        raise FileNotFoundError(f"Survey data not found at: {DATA_FILE_PATH}")

    # Load survey responses spreadsheet
    print("\n[Step 1/4] Reading survey responses Excel spreadsheet...")
    survey_df: pd.DataFrame = pd.read_excel(DATA_FILE_PATH)
    print(f"  -> Survey shape: {survey_df.shape[0]} responses across {survey_df.shape[1]} columns")

    # Generate schema and data dictionary
    print("\n[Step 2/4] Generating schema summary and data dictionary...")
    dict_path = generate_data_dictionary(survey_df, OUTPUTS_DIR_PATH)
    print(f"  -> Data dictionary saved to: {dict_path}")

    # Extract open-ended opinion text and suggestion text
    print("\n[Step 3/4] Building high-definition Word Clouds from customer texts...")
    opinion_col_name: str = load_text_column(survey_df)
    suggestion_col_name: str = load_suggestion_column(survey_df)

    # Clean and aggregate opinion text documents into a unified string corpus
    opinion_corpus: str = ' '.join(survey_df[opinion_col_name].dropna().astype(str).map(clean_text))
    wc_opinions_path: Path = OUTPUTS_DIR_PATH / 'wordcloud_opinions.png'
    generate_wordcloud(
        text_corpus=opinion_corpus,
        output_image_path=wc_opinions_path,
        chart_title='Customer Brand Opinion Keywords',
        colormap_name='plasma'
    )
    print(f"  -> Opinion Word Cloud saved to: {wc_opinions_path}")

    # Clean and aggregate suggestion text documents into a unified string corpus
    suggestion_corpus: str = ' '.join(survey_df[suggestion_col_name].dropna().astype(str).map(clean_text))
    wc_suggestions_path: Path = OUTPUTS_DIR_PATH / 'wordcloud_suggestions.png'
    generate_wordcloud(
        text_corpus=suggestion_corpus,
        output_image_path=wc_suggestions_path,
        chart_title='Customer Improvement Suggestions Keywords',
        colormap_name='cividis'
    )
    print(f"  -> Suggestion Word Cloud saved to: {wc_suggestions_path}")

    # Generate categorical question distribution charts
    print("\n[Step 4/4] Plotting categorical survey question response charts...")
    charts = generate_categorical_charts(survey_df, OUTPUTS_DIR_PATH)
    print(f"  -> Successfully generated {len(charts)} categorical response charts.")

    print("\n" + "=" * 70)
    print("  EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


# Standard Python execution boilerplate
if __name__ == '__main__':
    # Execute primary EDA analysis routine
    main()
