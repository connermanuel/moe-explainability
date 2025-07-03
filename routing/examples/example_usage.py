"""Example usage of the refactored routing extraction package.

This script demonstrates various ways to use the new modular architecture
for extracting routing probabilities from Switch Transformer models.
"""

import os

# Add parent directory to path for imports
import sys

import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new modular components
from routing import (
    RouterConfig,
    RoutingPipeline,
    extract_ud_routes,
    extract_wordsim_routes,
    quick_route_analysis,
)


def example_1_simple_word_analysis():
    """Example 1: Simple word-level routing analysis."""
    print("=== Example 1: Simple Word Analysis ===")

    # Quick analysis of a single text
    result = quick_route_analysis("The quick brown fox jumps over the lazy dog.")

    print(f"Text: {result['text']}")
    print(f"Number of tokens: {result['num_tokens']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Route matrix shape: {result['route_matrix'].shape}")
    print(f"Dominant experts: {result['route_stats']['dominant_experts_per_token']}")
    print()


def example_2_word_list_processing():
    """Example 2: Process a list of words."""
    print("=== Example 2: Word List Processing ===")

    words = ["hello", "world", "neural", "networks", "transformer", "attention"]

    df = extract_wordsim_routes(
        word_list=words, normalize=True, cache_file="data/example_words.parquet.gzip"
    )

    print(f"Processed {len(words)} words into {len(df)} token rows")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df[["original_word", "token_text", "token_id"]].head())
    print()


def example_3_ud_dataset():
    """Example 3: Process UD dataset (small subset)."""
    print("=== Example 3: UD Dataset Processing ===")

    # Load a small subset of UD data
    data = load_dataset("universal_dependencies", "en_ewt", split="train[:10]")

    df = extract_ud_routes(
        dataset=data, normalize=True, cache_file="data/example_ud_subset.parquet.gzip"
    )

    print(f"Processed {len(data)} sentences into {len(df)} token rows")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df[["sentence_id", "token_text", "sent_token", "upos", "xpos"]].head())
    print()


def example_4_custom_configuration():
    """Example 4: Using custom configuration and pipeline."""
    print("=== Example 4: Custom Configuration ===")

    # Create custom configuration
    config = RouterConfig.for_switch_base_8(
        normalize=False,  # Use raw logits instead of log probabilities
        device="cuda",
    )

    # Create pipeline with custom config
    pipeline = RoutingPipeline("google/switch-base-8", config)

    # Process some custom data
    custom_data = [
        {"text": "Machine learning is fascinating."},
        {"text": "Deep neural networks learn complex patterns."},
        {"text": "Transformers revolutionized natural language processing."},
    ]

    df = pipeline.extract_routes_flexible(
        dataset=custom_data, text_field="text", add_special_tokens=True
    )

    print(f"Processed {len(custom_data)} texts with custom config")
    print(f"Using raw logits (not normalized): {not config.normalize}")
    print(f"Sample route vector shape: {df['route_vector'].iloc[0].shape}")
    print()


def example_5_comparing_normalizations():
    """Example 5: Compare normalized vs raw routing probabilities."""
    print("=== Example 5: Comparing Normalizations ===")

    text = "Attention mechanisms are powerful."

    # Get normalized routes
    result_norm = quick_route_analysis(text, normalize=True)

    # Get raw routes
    config_raw = RouterConfig.for_switch_base_8(normalize=False)
    pipeline_raw = RoutingPipeline("google/switch-base-8", config_raw)
    result_raw = pipeline_raw.extract_single_text(text)

    print(f"Text: {text}")
    print(
        f"Normalized route range: [{result_norm['route_matrix'].min():.3f}, {result_norm['route_matrix'].max():.3f}]"
    )
    print(
        f"Raw route range: [{result_raw['route_matrix'].min():.3f}, {result_raw['route_matrix'].max():.3f}]"
    )

    # Show first token's routing probabilities for each layer
    print("\nFirst token routing (normalized vs raw):")
    norm_routes = result_norm["route_matrix"][0]
    raw_routes = result_raw["route_matrix"][0]

    # Assuming 6 layers with 8 experts each
    for layer_idx in range(6):
        start_idx = layer_idx * 8
        end_idx = start_idx + 8
        norm_layer = norm_routes[start_idx:end_idx]
        raw_layer = raw_routes[start_idx:end_idx]
        print(
            f"  Layer {layer_idx + 1}: norm_max={norm_layer.max():.3f}, raw_max={raw_layer.max():.3f}"
        )
    print()


def main():
    """Run all examples."""
    print("Router Extraction Package Examples")
    print("=" * 50)

    try:
        example_1_simple_word_analysis()
        example_2_word_list_processing()
        example_3_ud_dataset()
        example_4_custom_configuration()
        example_5_comparing_normalizations()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("- A CUDA-capable GPU or CPU for inference")
        print("- The transformers library installed")
        print("- The datasets library installed")
        print("- Internet connection to download models/datasets")


if __name__ == "__main__":
    main()
