"""Refactored WordSim dataset processing using the new modular architecture.

This script demonstrates how to use the new routing extraction pipeline
for WordSim-style datasets (simple word lists).
"""

import os

# Add parent directory to path for imports
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.pipeline import extract_wordsim_routes


def load_wordsim_words() -> list[str]:
    """Load words from the WordSim dataset."""
    df_similarity = pd.read_csv("similarity/wordsim353crowd.csv")
    cols = [df_similarity[col] for col in ["Word 1", "Word 2"]]
    words = pd.concat(cols).drop_duplicates().reset_index(drop=True).tolist()
    return words


def main():
    """Main function demonstrating WordSim processing with the new architecture."""

    # Load the WordSim words
    print("Loading WordSim words...")
    words = load_wordsim_words()
    print(f"Found {len(words)} unique words")

    # Extract routes using the new pipeline (normalized)
    print("Extracting normalized routes...")
    df_normalized = extract_wordsim_routes(
        model_name="google/switch-base-8",
        word_list=words,
        normalize=True,
        cache_file="data/switch_base_8_wordsim_normalized.parquet.gzip",
    )

    # Extract routes using the new pipeline (raw)
    print("Extracting raw routes...")
    df_raw = extract_wordsim_routes(
        model_name="google/switch-base-8",
        word_list=words,
        normalize=False,
        cache_file="data/switch_base_8_wordsim_raw.parquet.gzip",
    )

    print("Processing complete!")
    print(f"Normalized data shape: {df_normalized.shape}")
    print(f"Raw data shape: {df_raw.shape}")

    # Show sample of the data
    print("\nSample of normalized data:")
    print(df_normalized.head())

    # Show available columns
    print(f"\nColumns: {df_normalized.columns.tolist()}")

    # Show some statistics
    print(f"\nUnique words processed: {df_normalized['original_word'].nunique()}")
    print(f"Total tokens: {len(df_normalized)}")
    print(
        f"Average tokens per word: {len(df_normalized) / df_normalized['original_word'].nunique():.2f}"
    )


if __name__ == "__main__":
    main()
