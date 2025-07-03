"""Refactored UD dataset processing using the new modular architecture.

This script demonstrates how to use the new routing extraction pipeline
for Universal Dependencies datasets.
"""

import os

# Add parent directory to path for imports
import sys

from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.adapters import collapse_by_token_id, filter_common_tokens
from routing.pipeline import extract_ud_routes


def main():
    """Main function demonstrating UD processing with the new architecture."""

    # Load the UD dataset
    print("Loading Universal Dependencies dataset...")
    data = load_dataset("universal_dependencies", "en_ewt")

    # Extract routes using the new pipeline (normalized)
    print("Extracting normalized routes...")
    df_normalized = extract_ud_routes(
        model_name="google/switch-base-8",
        dataset=data["train"],
        normalize=True,
        cache_file="data/switch_base_8_ud_train_routes_normalized.parquet.gzip",
    )

    # Extract routes using the new pipeline (raw)
    print("Extracting raw routes...")
    df_raw = extract_ud_routes(
        model_name="google/switch-base-8",
        dataset=data["train"],
        normalize=False,
        cache_file="data/switch_base_8_ud_train_routes_raw.parquet.gzip",
    )

    # Post-processing: collapse by token ID and filter common tokens
    print("Post-processing normalized data...")
    df_normalized_collapsed = collapse_by_token_id(df_normalized)
    df_normalized_filtered = filter_common_tokens(df_normalized_collapsed)

    print("Post-processing raw data...")
    df_raw_collapsed = collapse_by_token_id(df_raw)
    df_raw_filtered = filter_common_tokens(df_raw_collapsed)

    # Save processed data
    df_normalized_filtered.to_parquet(
        "data/switch_base_8_ud_train_collapsed_normalized.parquet.gzip",
        compression="gzip",
    )
    df_raw_filtered.to_parquet(
        "data/switch_base_8_ud_train_collapsed_raw.parquet.gzip", compression="gzip"
    )

    print("Processing complete!")
    print(f"Normalized data shape: {df_normalized.shape}")
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Collapsed normalized shape: {df_normalized_filtered.shape}")
    print(f"Collapsed raw shape: {df_raw_filtered.shape}")


if __name__ == "__main__":
    main()
