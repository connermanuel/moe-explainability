"""High-level UD dataset analysis interface."""

from typing import Optional
import pandas as pd
from datasets import load_dataset

from ..models.configs import ModelConfig, SWITCH_BASE_8
from ..models.loading import load_model
from ..routing.extraction import extract_tokens
from ..routing.utils import cache_to_file
from ..datasets.ud_processing import process_ud_dataset, analyze_alignment_quality


def analyze_ud_dataset(
    dataset_split: str = "train[:100]",
    model_config: ModelConfig = SWITCH_BASE_8,
    cache_file: Optional[str] = None,
    dataset_name: str = "universal_dependencies",
    dataset_config: str = "en_ewt",
    show_progress: bool = True
) -> pd.DataFrame:
    """High-level function to analyze UD dataset with routing information.
    
    Args:
        dataset_split: Dataset split to use (e.g., "train[:100]")
        model_config: Model configuration
        cache_file: Optional cache file path
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (language)
        show_progress: Whether to show progress bars
        
    Returns:
        DataFrame with enriched token information
    """
    # Define the processing function
    def _process():
        print(f"Loading model: {model_config.name}")
        model, tokenizer = load_model(model_config)
        
        print(f"Loading dataset: {dataset_name}/{dataset_config} split={dataset_split}")
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        
        # Create extraction function
        def extract_tokens_fn(text):
            return extract_tokens(model, tokenizer, text, model_config)
        
        print("Processing dataset...")
        df = process_ud_dataset(dataset, extract_tokens_fn, show_progress)
        
        # Print alignment quality
        if show_progress:
            quality = analyze_alignment_quality(df)
            print(f"\n=== Alignment Quality ===")
            print(f"Total tokens: {quality['total_tokens']}")
            print(f"Aligned tokens: {quality['aligned_tokens']} ({quality['alignment_rate']:.1%})")
            print(f"Special tokens: {quality['special_tokens']} ({quality['special_token_rate']:.1%})")
            print(f"Unique sentences: {quality['unique_sentences']}")
            print(f"Avg tokens per sentence: {quality['avg_tokens_per_sentence']:.1f}")
            print(f"Linguistic features available: {quality['linguistic_features_available']}")
        
        return df
    
    # Use caching if specified
    if cache_file:
        @cache_to_file
        def _cached_process():
            return _process()
        
        return _cached_process(filename=cache_file)
    else:
        return _process()


def quick_ud_analysis(
    text: str,
    model_config: ModelConfig = SWITCH_BASE_8
) -> pd.DataFrame:
    """Quick analysis of a single text with mock UD structure.
    
    Args:
        text: Text to analyze
        model_config: Model configuration
        
    Returns:
        DataFrame with routing information (without real linguistic features)
    """
    # Load model
    model, tokenizer = load_model(model_config)
    
    # Extract tokens
    tokens = extract_tokens(model, tokenizer, text, model_config)
    
    # Create mock UD data
    mock_sentence_data = {
        "text": text,
        "tokens": text.split(),  # Simple word splitting
        "upos": ["NOUN"] * len(text.split()),  # Mock POS tags
        "lemmas": [word.lower() for word in text.split()],  # Mock lemmas
    }
    
    # Process as if it were a UD sentence
    from ..datasets.ud_processing import process_ud_sentence
    
    def extract_tokens_fn(text):
        return extract_tokens(model, tokenizer, text, model_config)
    
    return process_ud_sentence(mock_sentence_data, 0, extract_tokens_fn)


def analyze_by_pos_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze routing patterns by POS tags.
    
    Args:
        df: Processed UD dataset DataFrame
        
    Returns:
        DataFrame with routing statistics by POS tag
    """
    from ..datasets.ud_processing import filter_aligned_tokens
    
    # Filter to only aligned tokens
    aligned_df = filter_aligned_tokens(df)
    
    # Group by POS tag and compute statistics
    pos_stats = aligned_df.groupby("upos").agg({
        "token_id": "count",
        "route_vector": lambda x: x.apply(lambda arr: arr.mean()).mean(),  # Mean routing activation
        "word_text": lambda x: x.nunique(),  # Unique words
    }).rename(columns={
        "token_id": "token_count",
        "route_vector": "mean_routing_activation",
        "word_text": "unique_words"
    }).reset_index()
    
    # Sort by frequency
    pos_stats = pos_stats.sort_values("token_count", ascending=False)
    
    return pos_stats


def analyze_expert_specialization(df: pd.DataFrame, layer: int = 1) -> pd.DataFrame:
    """Analyze expert specialization patterns for a specific layer.
    
    Args:
        df: Processed UD dataset DataFrame
        layer: Layer number to analyze
        
    Returns:
        DataFrame with expert specialization by linguistic features
    """
    from ..datasets.ud_processing import filter_aligned_tokens
    
    # Filter to only aligned tokens
    aligned_df = filter_aligned_tokens(df)
    
    # Get logits for the specified layer
    layer_col = f"layer_{layer}_logits"
    if layer_col not in aligned_df.columns:
        raise ValueError(f"Layer {layer} not found in data")
    
    # For each POS tag, compute mean expert activations
    pos_expert_stats = []
    
    for pos_tag in aligned_df["upos"].dropna().unique():
        pos_tokens = aligned_df[aligned_df["upos"] == pos_tag]
        
        if len(pos_tokens) == 0:
            continue
        
        # Mean expert logits for this POS tag
        mean_logits = pos_tokens[layer_col].apply(lambda x: x).mean()
        
        pos_expert_stats.append({
            "pos_tag": pos_tag,
            "token_count": len(pos_tokens),
            "mean_expert_logits": mean_logits,
            "dominant_expert": mean_logits.argmax(),
            "max_activation": mean_logits.max(),
        })
    
    return pd.DataFrame(pos_expert_stats).sort_values("token_count", ascending=False)