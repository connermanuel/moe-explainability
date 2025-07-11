# %%
"""Experiment: Correlate routing distances with human word similarity judgments."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from typing import Dict, List

from moe_explainability.analysis.distances import token_group_distance, group_tokens_by_text
from moe_explainability.datasets.configs import WORDSIM_353
from moe_explainability.datasets.processing import extract_and_align_routing_tokens
from moe_explainability.models.configs import SWITCH_BASE_8
from moe_explainability.models.loading import create_extraction_fn
from moe_explainability.routing.extraction import AnnotatedToken
from moe_explainability.routing.utils import cache_annotated_tokens


# %%
@cache_annotated_tokens
def load_and_process_wordsim_data():
    """Load and process WordSim data with caching."""
    print("Loading WordSim data...")
    data = WORDSIM_353.load_data()
    print(f"Loaded {len(data)} unique words")
    
    # Create extraction function
    print("Loading model...")
    extract_tokens_fn = create_extraction_fn(SWITCH_BASE_8)
    
    # Process using new architecture
    print("Processing words with routing extraction...")
    annotated_tokens = extract_and_align_routing_tokens(
        data=data,
        config=WORDSIM_353,
        extract_tokens_fn=extract_tokens_fn,
    )
    
    return annotated_tokens


def load_wordsim_pairs() -> pd.DataFrame:
    """Load the original WordSim word pairs and similarity scores."""
    df = pd.read_csv("data/wordsim353crowd.csv")
    df["Word 1"] = df["Word 1"].str.lower()
    df["Word 2"] = df["Word 2"].str.lower()
    return df


def compute_routing_similarities(
    annotated_tokens: List[AnnotatedToken],
    word_pairs: pd.DataFrame,
    aggregation_method: str = "mean",
    distance_method: str = "layer_mean",
) -> List[float]:
    """Compute routing distances for word pairs.
    
    Args:
        annotated_tokens: Processed tokens for all words
        word_pairs: DataFrame with word pairs and human similarities
        aggregation_method: How to aggregate tokens within each word
        distance_method: How to compute distance between words
        
    Returns:
        List of routing distances corresponding to word pairs
    """
    # Group tokens by word
    word_groups = group_tokens_by_text(annotated_tokens)
    
    # Convert to just the Token objects for distance computation
    word_token_groups = {}
    for word, tokens in word_groups.items():
        word_token_groups[word] = [t.token for t in tokens]
    
    routing_distances = []
    
    for _, row in word_pairs.iterrows():
        word1 = row["Word 1"]
        word2 = row["Word 2"]
        
        if word1 in word_token_groups and word2 in word_token_groups:
            try:
                distance = token_group_distance(
                    word_token_groups[word1],
                    word_token_groups[word2],
                    aggregation_method=aggregation_method,
                    distance_method=distance_method,
                    normalize=True,
                )
                routing_distances.append(distance)
            except Exception as e:
                print(f"Error computing distance for {word1}-{word2}: {e}")
                routing_distances.append(np.nan)
        else:
            print(f"Missing tokens for {word1}-{word2}")
            routing_distances.append(np.nan)
    
    return routing_distances


# %%
print("=== WordSim Routing Correlation Experiment ===")

# Load and process data (with caching)
annotated_tokens = load_and_process_wordsim_data(
    filename="data/cached_wordsim_tokens.pkl"
)

print(f"Processed {len(annotated_tokens)} tokens")

# %%
# Load word pairs and human similarity scores
word_pairs = load_wordsim_pairs()
print(f"Loaded {len(word_pairs)} word pairs")

# Show some examples
print("\nExample word pairs:")
print(word_pairs.head())

# %%
# Analyze token patterns
print("\n=== Token Analysis ===")
word_groups = group_tokens_by_text(annotated_tokens)

# Show token count distribution
token_counts = [len(tokens) for tokens in word_groups.values()]
print(f"Words with tokens: {len(word_groups)}")
print(f"Avg tokens per word: {np.mean(token_counts):.2f}")
print(f"Token count range: {min(token_counts)}-{max(token_counts)}")

# Show some examples
print("\nExample tokenization:")
for i, (word, tokens) in enumerate(list(word_groups.items())[:5]):
    token_texts = [t.token.text for t in tokens]
    print(f"  {word}: {token_texts}")

# %%
# Compute routing distances for different methods
print("\n=== Computing Routing Distances ===")

methods = [
    ("first", "layer_mean"),
    ("mean", "layer_mean"),
    ("first", "concatenated"),
    ("mean", "concatenated"),
]

results = {}

for agg_method, dist_method in methods:
    print(f"Computing distances with {agg_method} aggregation, {dist_method} distance...")
    
    distances = compute_routing_similarities(
        annotated_tokens,
        word_pairs,
        aggregation_method=agg_method,
        distance_method=dist_method,
    )
    
    results[f"{agg_method}_{dist_method}"] = distances

# %%
# Analyze correlations
print("\n=== Correlation Analysis ===")

human_scores = word_pairs["Human (Mean)"].values

for method_name, distances in results.items():
    # Remove NaN values for correlation
    valid_mask = ~np.isnan(distances)
    valid_human = human_scores[valid_mask]
    valid_distances = np.array(distances)[valid_mask]
    
    if len(valid_distances) > 0:
        # Note: We expect negative correlation (high similarity = low distance)
        spearman_corr, spearman_p = spearmanr(valid_human, valid_distances)
        pearson_corr, pearson_p = pearsonr(valid_human, valid_distances)
        
        print(f"\n{method_name.upper()}:")
        print(f"  Valid pairs: {len(valid_distances)}/{len(distances)}")
        print(f"  Mean distance: {np.mean(valid_distances):.4f}")
        print(f"  Spearman r: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"  Pearson r: {pearson_corr:.4f} (p={pearson_p:.4f})")
    else:
        print(f"\n{method_name.upper()}: No valid distances computed")

# %%
# Create visualization
print("\n=== Creating Visualizations ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (method_name, distances) in enumerate(results.items()):
    ax = axes[i]
    
    # Remove NaN values
    valid_mask = ~np.isnan(distances)
    valid_human = human_scores[valid_mask]
    valid_distances = np.array(distances)[valid_mask]
    
    if len(valid_distances) > 0:
        ax.scatter(valid_human, valid_distances, alpha=0.6, s=20)
        
        # Add trend line
        z = np.polyfit(valid_human, valid_distances, 1)
        p = np.poly1d(z)
        ax.plot(valid_human, p(valid_human), "r--", alpha=0.8)
        
        # Calculate correlation for title
        corr, _ = spearmanr(valid_human, valid_distances)
        ax.set_title(f"{method_name}\nSpearman r = {corr:.3f}")
        
        ax.set_xlabel("Human Similarity Score")
        ax.set_ylabel("Routing Distance")
    else:
        ax.set_title(f"{method_name}\nNo valid data")
        ax.set_xlabel("Human Similarity Score")
        ax.set_ylabel("Routing Distance")

plt.tight_layout()
plt.savefig("experiments/wordsim_routing_correlation.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Analyze specific examples
print("\n=== Example Analysis ===")

# Find best and worst correlations
best_method = max(results.keys(), key=lambda k: 
    spearmanr(human_scores[~np.isnan(results[k])], 
              np.array(results[k])[~np.isnan(results[k])])[0] if np.sum(~np.isnan(results[k])) > 0 else -1)

print(f"Best method: {best_method}")

# Show some specific examples
distances = results[best_method]
valid_mask = ~np.isnan(distances)

examples_df = word_pairs.copy()
examples_df["routing_distance"] = distances
examples_df = examples_df[valid_mask]

# Sort by human similarity
examples_df = examples_df.sort_values("Human (Mean)")

print("\nMost similar word pairs (highest human scores):")
print(examples_df.tail(10)[["Word 1", "Word 2", "Human (Mean)", "routing_distance"]])

print("\nLeast similar word pairs (lowest human scores):")
print(examples_df.head(10)[["Word 1", "Word 2", "Human (Mean)", "routing_distance"]])

# %%
print("\n=== Experiment Complete ===")