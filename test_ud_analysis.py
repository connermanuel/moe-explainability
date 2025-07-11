"""Test script for UD analysis pipeline."""

from moe_explainability.analysis.ud_analysis import analyze_ud_dataset, quick_ud_analysis, analyze_by_pos_tags


def test_quick_analysis():
    """Test quick analysis with a single sentence."""
    print("=== Testing Quick UD Analysis ===")
    
    text = "The quick brown fox jumps over the lazy dog."
    df = quick_ud_analysis(text)
    
    print(f"Input text: {text}")
    print(f"Generated DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nSample data:")
    print(df[["token_text", "word_text", "upos", "lemma"]].head())
    print()


def test_small_dataset():
    """Test with a small UD dataset sample."""
    print("=== Testing Small UD Dataset ===")
    
    # Process a very small sample
    df = analyze_ud_dataset(
        dataset_split="train[:3]",  # Just 3 sentences
        cache_file=None,  # No caching for testing
        show_progress=True
    )
    
    print(f"Processed DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nSample data:")
    print(df[["sentence_id", "token_text", "word_text", "upos", "deprel"]].head(10))
    
    print("\nPOS tag analysis:")
    pos_stats = analyze_by_pos_tags(df)
    print(pos_stats.head())
    
    print()


def test_alignment_quality():
    """Test alignment quality analysis."""
    print("=== Testing Alignment Quality ===")
    
    # Process a small sample
    df = analyze_ud_dataset(
        dataset_split="train[:5]",
        cache_file=None,
        show_progress=False
    )
    
    from moe_explainability.datasets.ud_processing import analyze_alignment_quality
    
    quality = analyze_alignment_quality(df)
    print("Alignment Quality Metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    
    print()


if __name__ == "__main__":
    try:
        test_quick_analysis()
        test_small_dataset()
        test_alignment_quality()
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()