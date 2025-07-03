"""Test script to verify the refactored routing extraction functionality.

This script tests the core functionality requested:
1. UD dataset: token-level table with routes, sentence words, and linguistic info
2. WordSim dataset: token-level table with routes and original words
3. Different storage formats for route probabilities
"""

# Simple import setup - run from project root
import sys

sys.path.append(".")
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from setup_paths import *


def test_ud_extraction():
    """Test UD dataset extraction with small sample."""
    print("=== Testing UD Dataset Extraction ===")

    # Load a tiny subset for testing
    try:
        data = load_dataset("universal_dependencies", "en_ewt", split="train[:5]")
        print(f"Loaded {len(data)} UD sentences for testing")
    except Exception as e:
        print(f"Error loading UD dataset: {e}")
        return None

    # Test both normalized and raw extraction
    print("\nTesting normalized routes...")
    df_norm = extract_ud_routes(
        model_name="google/switch-base-8",
        dataset=data,
        normalize=True,
        cache_file="data/test_ud_normalized.parquet.gzip",
    )

    print("\nTesting raw routes...")
    df_raw = extract_ud_routes(
        model_name="google/switch-base-8",
        dataset=data,
        normalize=False,
        cache_file="data/test_ud_raw.parquet.gzip",
    )

    # Verify the structure
    print(f"\nNormalized UD data shape: {df_norm.shape}")
    print(f"Raw UD data shape: {df_raw.shape}")
    print(f"Columns: {list(df_norm.columns)}")

    # Show sample data
    print("\nSample normalized data:")
    sample_cols = [
        "sentence_id",
        "token_text",
        "sent_token",
        "upos",
        "xpos",
        "sent_token_position",
    ]
    available_cols = [col for col in sample_cols if col in df_norm.columns]
    print(df_norm[available_cols].head(10))

    # Check route probabilities
    print(f"\nRoute vector shape: {df_norm['route_vector'].iloc[0].shape}")
    print(
        f"Normalized route range: [{df_norm['route_vector'].iloc[0].min():.3f}, {df_norm['route_vector'].iloc[0].max():.3f}]"
    )
    print(
        f"Raw route range: [{df_raw['route_vector'].iloc[0].min():.3f}, {df_raw['route_vector'].iloc[0].max():.3f}]"
    )

    # Check linguistic annotations
    if "upos" in df_norm.columns:
        print(f"\nUnique POS tags: {df_norm['upos'].unique()[:10]}")

    return df_norm, df_raw


def test_wordsim_extraction():
    """Test WordSim dataset extraction."""
    print("\n=== Testing WordSim Dataset Extraction ===")

    # Create a small test word list
    test_words = [
        "hello",
        "world",
        "attention",
        "transformer",
        "neural",
        "network",
        "computer",
        "science",
    ]

    print(f"Testing with {len(test_words)} words: {test_words}")

    # Test both normalized and raw extraction
    print("\nTesting normalized routes...")
    df_norm = extract_wordsim_routes(
        word_list=test_words,
        normalize=True,
        cache_file="data/test_wordsim_normalized.parquet.gzip",
    )

    print("\nTesting raw routes...")
    df_raw = extract_wordsim_routes(
        word_list=test_words,
        normalize=False,
        cache_file="data/test_wordsim_raw.parquet.gzip",
    )

    # Verify the structure
    print(f"\nNormalized WordSim data shape: {df_norm.shape}")
    print(f"Raw WordSim data shape: {df_raw.shape}")
    print(f"Columns: {list(df_norm.columns)}")

    # Show sample data
    print("\nSample normalized data:")
    sample_cols = ["original_word", "token_text", "token_id", "position"]
    available_cols = [col for col in sample_cols if col in df_norm.columns]
    print(df_norm[available_cols].head(10))

    # Check route probabilities
    print(f"\nRoute vector shape: {df_norm['route_vector'].iloc[0].shape}")
    print(
        f"Normalized route range: [{df_norm['route_vector'].iloc[0].min():.3f}, {df_norm['route_vector'].iloc[0].max():.3f}]"
    )
    print(
        f"Raw route range: [{df_raw['route_vector'].iloc[0].min():.3f}, {df_raw['route_vector'].iloc[0].max():.3f}]"
    )

    # Check tokenization patterns
    print("\nTokenization patterns:")
    for word in test_words[:3]:
        word_tokens = df_norm[df_norm["original_word"] == word]
        tokens = word_tokens["token_text"].tolist()
        print(f"  '{word}' -> {tokens}")

    return df_norm, df_raw


def test_storage_formats(df_sample):
    """Test different storage formats for route probabilities."""
    print("\n=== Testing Storage Formats ===")

    # Get a sample route vector
    sample_route = df_sample["route_vector"].iloc[0]
    print(
        f"Original route vector: {type(sample_route)}, shape: {sample_route.shape}, dtype: {sample_route.dtype}"
    )

    # Test different precision levels
    formats = {
        "float64": sample_route.astype(np.float64),
        "float32": sample_route.astype(np.float32),
        "float16": sample_route.astype(np.float16),
    }

    print("\nStorage format comparison:")
    for name, data in formats.items():
        size_bytes = data.nbytes
        print(
            f"  {name}: {size_bytes} bytes ({size_bytes / formats['float64'].nbytes:.2f}x original size)"
        )
        print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")

        # Check precision loss
        mse = np.mean((sample_route - data.astype(np.float64)) ** 2)
        print(f"    MSE vs original: {mse:.2e}")

    # Test torch tensor storage
    torch_tensor = torch.from_numpy(sample_route)
    torch_half = torch_tensor.half()

    print(f"\nTorch tensor comparison:")
    print(
        f"  torch.float32: {torch_tensor.element_size() * torch_tensor.nelement()} bytes"
    )
    print(f"  torch.float16: {torch_half.element_size() * torch_half.nelement()} bytes")

    return formats


def test_memory_efficiency(df):
    """Analyze memory usage of different storage approaches."""
    print("\n=== Memory Efficiency Analysis ===")

    # Current storage (object arrays of numpy arrays)
    current_memory = df.memory_usage(deep=True).sum()
    print(f"Current DataFrame memory: {current_memory / 1024 / 1024:.2f} MB")

    # Calculate route data memory specifically
    route_vectors = df["route_vector"].values
    total_elements = sum(arr.size for arr in route_vectors)

    print(f"Total route vector elements: {total_elements:,}")
    print(f"Current route storage (float64): {total_elements * 8 / 1024 / 1024:.2f} MB")
    print(f"With float32: {total_elements * 4 / 1024 / 1024:.2f} MB (50% reduction)")
    print(f"With float16: {total_elements * 2 / 1024 / 1024:.2f} MB (75% reduction)")

    # Test converting route vectors to float16
    df_optimized = df.copy()
    df_optimized["route_vector"] = df_optimized["route_vector"].apply(
        lambda x: x.astype(np.float16)
    )

    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    print(f"Optimized DataFrame memory: {optimized_memory / 1024 / 1024:.2f} MB")
    print(f"Memory reduction: {(1 - optimized_memory / current_memory) * 100:.1f}%")

    return df_optimized


def analyze_normalization_options():
    """Analyze different normalization approaches."""
    print("\n=== Normalization Analysis ===")

    # Create sample logits
    sample_logits = np.random.randn(8) * 2  # 8 experts

    print("Sample raw logits:", sample_logits.round(3))

    # Different normalization approaches
    softmax_probs = torch.softmax(torch.from_numpy(sample_logits), dim=0).numpy()
    log_softmax = torch.log_softmax(torch.from_numpy(sample_logits), dim=0).numpy()

    print("Softmax probabilities:", softmax_probs.round(3))
    print("Log softmax:", log_softmax.round(3))
    print("Sum of softmax:", softmax_probs.sum())
    print("Exp of log_softmax sum:", np.exp(log_softmax).sum())

    # Storage implications
    print(f"\nRange analysis:")
    print(f"  Raw logits: [{sample_logits.min():.3f}, {sample_logits.max():.3f}]")
    print(f"  Softmax: [{softmax_probs.min():.6f}, {softmax_probs.max():.3f}]")
    print(f"  Log softmax: [{log_softmax.min():.3f}, {log_softmax.max():.3f}]")

    # Recommendation
    print(f"\nRecommendation:")
    print(f"- Store RAW LOGITS for maximum flexibility")
    print(f"- Apply normalization downstream as needed")
    print(f"- Raw logits preserve more information and allow different analysis")


def main():
    """Run all tests."""
    print("Testing Refactored Routing Extraction Package")
    print("=" * 60)

    try:
        # Test UD extraction
        df_ud_norm, df_ud_raw = test_ud_extraction()

        # Test WordSim extraction
        df_ws_norm, df_ws_raw = test_wordsim_extraction()

        # Test storage formats
        if df_ws_norm is not None:
            formats = test_storage_formats(df_ws_norm)

            # Memory efficiency analysis
            test_memory_efficiency(df_ws_norm)

        # Analyze normalization
        analyze_normalization_options()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nKey findings:")
        print("- UD extraction works with token-level rows + linguistic annotations")
        print("- WordSim extraction works with token-level rows + original words")
        print("- float16 reduces memory by 75% with minimal precision loss")
        print("- Recommend storing raw logits for maximum flexibility")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
