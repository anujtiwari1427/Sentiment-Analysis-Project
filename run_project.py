# ==============================================================================
# Script: run_project.py
# Description: Master automated pipeline orchestrator for the Local Brand Sentiment
# Analysis system. Sequentially executes exploratory data analysis, chart generation,
# feature extraction, balanced model training, and performance metric export.
# Every single line is documented with technical and theoretical explanations.
# ==============================================================================

# Import time module to benchmark elapsed execution durations for pipeline stages
import time

# Import sys module to access standard runtime parameters and Python interpreter binary
import sys

# Import subprocess module to spawn independent Python child processes safely
import subprocess

# Import Path from pathlib for robust, cross-platform filesystem directory handling
from pathlib import Path

# Resolve the absolute canonical filesystem directory path containing this script
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Define absolute path to the Exploratory Data Analysis execution script
EDA_SCRIPT_PATH: Path = PROJECT_ROOT / 'src' / 'analyze_data.py'

# Define absolute path to the Model Training & Serialization execution script
TRAINING_SCRIPT_PATH: Path = PROJECT_ROOT / 'src' / 'train_model.py'


# ------------------------------------------------------------------------------
# 1. PROCESS RUNNER WRAPPER WITH BENCHMARKING
# ------------------------------------------------------------------------------

def execute_stage(stage_number: int, stage_name: str, script_path: Path) -> float:
    """Execute a single pipeline python script in an isolated subprocess.

    Args:
        stage_number (int): Sequential pipeline stage index.
        stage_name (str): Descriptive title of the stage for console feedback.
        script_path (Path): Filesystem path to the Python script to be executed.

    Returns:
        float: Total elapsed execution duration in seconds.
    """
    # Print clear banner marking the beginning of the pipeline execution stage
    print("\n" + "=" * 75)
    print(f"  [STAGE {stage_number}] STARTING: {stage_name.upper()}")
    print(f"  Script: {script_path.relative_to(PROJECT_ROOT)}")
    print("=" * 75 + "\n")

    # Record start time using high-precision monotonic clock to avoid system clock shifts
    start_time: float = time.perf_counter()

    try:
        # Spawn isolated child process using current Python interpreter: sys.executable
        # check=True raises a CalledProcessError if the child exits with a non-zero status
        subprocess.check_call([sys.executable, str(script_path)])

    # Catch child execution errors and print contextual failure diagnostic
    except subprocess.CalledProcessError as process_error:
        # Print failure notice with stage context
        print(f"\n[FATAL ERROR] Stage {stage_number} ({stage_name}) failed with exit code {process_error.returncode}!")
        # Re-raise exception to halt the downstream pipeline execution immediately
        raise process_error

    # Record finish time upon successful process termination
    end_time: float = time.perf_counter()

    # Calculate net elapsed time in seconds
    elapsed_seconds: float = end_time - start_time

    # Print success confirmation along with execution time benchmark
    print(f"\n>> Stage {stage_number} completed successfully in {elapsed_seconds:.2f} seconds.")

    # Return elapsed duration for aggregate reporting
    return elapsed_seconds


# ------------------------------------------------------------------------------
# 2. MASTER ORCHESTRATION PIPELINE
# ------------------------------------------------------------------------------

def main() -> None:
    """Coordinate the sequential execution of all pipeline stages and report totals."""
    # Print master pipeline header
    print("#" * 75)
    print("  LOCAL BRAND SENTIMENT ANALYSIS: END-TO-END PIPELINE ORCHESTRATOR")
    print("#" * 75)

    # Benchmark total pipeline wall-clock time
    pipeline_start_wall: float = time.perf_counter()

    # Track stage benchmarks in a list of tuples: (stage_title, duration_in_seconds)
    benchmarks = []

    # Execute Stage 1: Exploratory Data Analysis & WordCloud Generation
    t1 = execute_stage(1, "Exploratory Data Analysis & Visualizations", EDA_SCRIPT_PATH)
    benchmarks.append(("EDA & Visualizations", t1))

    # Execute Stage 2: Feature Engineering, Model Training & Metric Serialization
    t2 = execute_stage(2, "Feature Extraction & Model Training", TRAINING_SCRIPT_PATH)
    benchmarks.append(("Model Training & Metrics", t2))

    # Benchmark total completion wall-clock time
    total_pipeline_time = time.perf_counter() - pipeline_start_wall

    # Print final summary report
    print("\n" + "#" * 75)
    print("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("#" * 75)
    print("\nExecution Timing Breakdown:")
    for name, duration in benchmarks:
        print(f"  - {name:<35}: {duration:.2f} s")
    print(f"  - {'Total Pipeline Elapsed Time':<35}: {total_pipeline_time:.2f} s")
    print("\nArtifact Locations:")
    print(f"  - Visual Outputs & Metrics : {PROJECT_ROOT / 'outputs'}")
    print(f"  - Serialized Model Binaries: {PROJECT_ROOT / 'models'}")
    print("\nTo launch the interactive dashboard, run:")
    print("  streamlit run app.py\n")


# Standard Python execution boilerplate
if __name__ == '__main__':
    # Run the master pipeline entrypoint
    main()
