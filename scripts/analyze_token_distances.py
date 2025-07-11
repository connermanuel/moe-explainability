# %%
"""Analyze routing distances between different types of token pairs."""

import random
from collections import defaultdict
from typing import List, Tuple, Literal

import numpy as np
from datasets import load_dataset

from moe_explainability.analysis.distances import token_routing_distance
from moe_explainability.datasets.configs import UD_ENGLISH_EWT
from moe_explainability.datasets.processing import extract_and_align_routing_tokens
from moe_explainability.models.configs import SWITCH_BASE_8
from moe_explainability.models.loading import create_extraction_fn
from moe_explainability.routing.extraction import AnnotatedToken
from moe_explainability.routing.utils import cache_annotated_tokens


# %%
def classify_token_pair(
    token1: AnnotatedToken,
    token2: AnnotatedToken,
) -> Literal["same_word", "same_sentence", "different_sentence"]:
    """Classify the relationship between two tokens."""
    # Check if same sentence
    if token1.sentence_id != token2.sentence_id:
        return "different_sentence"
    
    # Same sentence - check if same word
    if (token1.word_index >= 0 and token2.word_index >= 0 and 
        token1.word_index == token2.word_index):
        return "same_word"
    
    # Same sentence, different words
    return "same_sentence"


def compute_pairwise_distances(
    tokens: List[AnnotatedToken],
    method: Literal["layer_mean", "concatenated"] = "layer_mean",
    normalize: bool = True,
    max_pairs: int = 10000,
) -> List[Tuple[str, float]]:
    """Compute pairwise distances between tokens with classifications."""
    results = []
    
    # Only consider aligned tokens for meaningful comparison
    aligned_tokens = [t for t in tokens if t.word_index >= 0]
    
    if len(aligned_tokens) < 2:
        return results
    
    # Compute distances for a sample of pairs
    random.seed(42)  # For reproducibility
    
    pairs_computed = 0
    for i in range(len(aligned_tokens)):
        for j in range(i + 1, len(aligned_tokens)):
            if pairs_computed >= max_pairs:
                break
            
            token1 = aligned_tokens[i]
            token2 = aligned_tokens[j]
            
            try:
                classification = classify_token_pair(token1, token2)
                distance = token_routing_distance(token1.token, token2.token, method, normalize)
                results.append((classification, distance))
                pairs_computed += 1
            except Exception:
                # Skip pairs that can't be computed
                continue
        
        if pairs_computed >= max_pairs:
            break
    
    return results


def analyze_distance_patterns(
    distance_results: List[Tuple[str, float]]
) -> dict:
    """Analyze patterns in distance results."""
    groups = defaultdict(list)
    for classification, distance in distance_results:
        groups[classification].append(distance)
    
    analysis = {}
    for classification, distances in groups.items():
        if distances:
            analysis[classification] = {
                "count": len(distances),
                "mean": np.mean(distances),
                "std": np.std(distances),
                "median": np.median(distances),
                "min": np.min(distances),
                "max": np.max(distances),
            }
    
    return analysis


# %%
@cache_annotated_tokens
def load_and_process_ud_data(dataset_split: str = "train[:100]"):
    """Load and process UD data with caching."""
    print("Loading UD dataset...")
    ud_data = load_dataset("universal_dependencies", "en_ewt", split=dataset_split)
    print(f"Loaded {len(ud_data)} sentences")
    
    # Create extraction function
    print("Loading model...")
    extract_tokens_fn = create_extraction_fn(SWITCH_BASE_8)
    
    # Process using new architecture
    print("Processing with new architecture...")
    annotated_tokens = extract_and_align_routing_tokens(
        data=list(ud_data),
        config=UD_ENGLISH_EWT,
        extract_tokens_fn=extract_tokens_fn,
    )
    
    return annotated_tokens


# %%
print("=== Token Distance Analysis ===")

# Load and process data (with caching)
annotated_tokens = load_and_process_ud_data(
    dataset_split="train[:50]",
    filename="data/cached_ud_tokens_50.pkl"
)

print(f"Processed {len(annotated_tokens)} tokens")

# %%
# Show some examples of each token pair type
print("\n=== Token Pair Examples ===")
aligned_tokens = [t for t in annotated_tokens if t.word_index >= 0]

examples = {"same_word": [], "same_sentence": [], "different_sentence": []}

for i in range(min(100, len(aligned_tokens))):
    for j in range(i + 1, min(100, len(aligned_tokens))):
        if len(examples["same_word"]) < 3 and len(examples["same_sentence"]) < 3 and len(examples["different_sentence"]) < 3:
            token1 = aligned_tokens[i]
            token2 = aligned_tokens[j]
            classification = classify_token_pair(token1, token2)
            
            if len(examples[classification]) < 3:
                examples[classification].append((token1, token2))

for classification, token_pairs in examples.items():
    print(f"\n{classification.upper()}:")
    for i, (t1, t2) in enumerate(token_pairs):
        print(f"  {i+1}. '{t1.token_text}' (word: '{t1.word_text}', sent: {t1.sentence_id}) "
              f"<-> '{t2.token_text}' (word: '{t2.word_text}', sent: {t2.sentence_id})")

# %%
# Compute pairwise distances
print("\n=== Computing Pairwise Distances ===")
distance_results = compute_pairwise_distances(
    annotated_tokens,
    method="layer_mean",
    normalize=True,
    max_pairs=5000
)

print(f"Computed {len(distance_results)} pairwise distances")

# %%
# Analyze distance patterns
print("\n=== Distance Analysis ===")
analysis = analyze_distance_patterns(distance_results)

for classification, stats in analysis.items():
    print(f"\n{classification.upper()}:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

# %%
# Compare methods
print("\n=== Comparing Distance Methods ===")
methods = ["layer_mean", "concatenated"]
method_results = {}

for method in methods:
    print(f"Computing distances with {method} method...")
    distances = compute_pairwise_distances(
        annotated_tokens,
        method=method,
        normalize=True,
        max_pairs=1000
    )
    method_results[method] = analyze_distance_patterns(distances)

# Compare results
for classification in ["same_word", "same_sentence", "different_sentence"]:
    print(f"\n{classification.upper()} - Method Comparison:")
    for method in methods:
        if classification in method_results[method]:
            stats = method_results[method][classification]
            print(f"  {method}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

# %%
print("\n=== Analysis Complete ===")