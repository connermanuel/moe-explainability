# %% Imports
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm

from scripts.get_routes.get_routes import (
    collapse_df_by_input_id,
    extract_token_routes_df,
    filter_df,
    load_model_and_tokenizer,
)

# %% Load model and data
model_name = "google/switch-base-8"
tokenizer, model = load_model_and_tokenizer(model_name)
data = load_dataset("universal_dependencies", "en_ewt")

# %% Load token routes
df = extract_token_routes_df(
    model,
    tokenizer,
    data_subset=data["train"],
    filename="data/switch_base_8_ud_train_token_routes_normalized.parquet.gzip",
)


# %% Prepare data for clustering and run elbow method
def evaluate_clusters(X: np.ndarray, n_clusters_range: range):
    """Evaluate different numbers of clusters using inertia (sum of squared distances to centroids)."""
    inertias = []

    for k in n_clusters_range:
        print(f"Evaluating k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    return inertias


X = np.stack(df["expert_1"].values)

n_clusters_range = range(1, 40)
inertias = evaluate_clusters(X, n_clusters_range)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# %% Robustness Analysis
# Load and prepare type-level data for robustness analysis


def analyze_cluster_robustness(
    X: np.ndarray, k: int, n_runs: int = 10, batch_size: int = 1000
) -> float:
    """
    Analyze how robust k-means clustering is across different random seeds using F1 score.
    Processes matrices in batches to be memory efficient.

    Args:
        X: Input data matrix
        k: Number of clusters
        n_runs: Number of different random seeds to try
        batch_size: Size of batches for processing pairwise comparisons

    Returns:
        float: Average F1 score (0-1) where 1 means perfectly consistent clusters
    """
    # Store cluster assignments for each run
    all_labels = []

    # Run clustering multiple times with different seeds
    for seed in tqdm(range(n_runs), desc="Running clustering"):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = kmeans.fit_predict(X)
        all_labels.append(labels)

    # For each pair of runs, compute F1 score
    f1_scores = []
    for labels1, labels2 in tqdm(
        combinations(all_labels, 2), total=len(all_labels) * (len(all_labels) - 1) / 2
    ):
        n_points = len(labels1)
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            batch_size_actual = batch_end - i

            # Create batch matrices
            same_cluster1_batch = np.zeros((batch_size_actual, n_points))
            same_cluster2_batch = np.zeros((batch_size_actual, n_points))

            # Fill batch matrices
            for j in range(batch_size_actual):
                same_cluster1_batch[j] = labels1 == labels1[i + j]
                same_cluster2_batch[j] = labels2 == labels2[i + j]

            # Update counts
            true_positives += np.sum(
                (same_cluster1_batch == 1) & (same_cluster2_batch == 1)
            )
            false_positives += np.sum(
                (same_cluster1_batch == 0) & (same_cluster2_batch == 1)
            )
            false_negatives += np.sum(
                (same_cluster1_batch == 1) & (same_cluster2_batch == 0)
            )

        # Compute precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores.append(f1)

    return np.mean(f1_scores)


# %% Test robustness for different k values on type-level data
# Expert 3 robustness
df_types = collapse_df_by_input_id(df)
X_types = np.stack(df_types["expert_1"].values)
k_values = range(4, 21, 2)
robustness_scores = []

for k in k_values:
    score = analyze_cluster_robustness(
        X_types, k=k, n_runs=5
    )  # Fewer runs to save time
    robustness_scores.append(score)
    print(f"k={k}: {score:.3f}")

# Plot robustness vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, robustness_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Robustness score")
plt.title("Cluster Robustness vs Number of Clusters")
plt.grid(True)
plt.show()

# %% Test robustness for different k values on token-level data
X_tokens = np.stack(filter_df(df)["expert_1"].values)
print(f"Number of tokens: {len(X_tokens):,}")


# Test different batch sizes for k=8 to find optimal
def test_batch_sizes(
    X: np.ndarray,
    batch_sizes: list[int],
    k: int = 8,
    n_runs: int = 3,
    memory_growth_threshold: float = 100,  # MB per second
    max_memory: float = 32000,  # 32GB in MB
):
    """
    Test different batch sizes and measure time and peak memory usage.

    Args:
        X: Input data matrix
        batch_sizes: List of batch sizes to test
        k: Number of clusters
        n_runs: Number of clustering runs
        memory_growth_threshold: Maximum acceptable memory growth rate in MB/s
        max_memory: Maximum acceptable total memory usage in MB
    """
    import os
    import threading
    import time
    from queue import Queue

    import psutil

    process = psutil.Process(os.getpid())
    results = []

    class MemoryMonitor:
        def __init__(self, stop_event, peak_memory, growth_rate):
            self.stop_event = stop_event
            self.peak_memory = peak_memory
            self.growth_rate = growth_rate
            self.measurements = []
            self.start_time = time.time()

        def run(self):
            while not self.stop_event.is_set():
                current_mem = process.memory_info().rss / 1024 / 1024  # MB
                current_time = time.time() - self.start_time
                self.measurements.append((current_time, current_mem))

                # Calculate growth rate if we have enough measurements
                if len(self.measurements) >= 3:
                    recent_measurements = self.measurements[-3:]
                    time_diff = recent_measurements[-1][0] - recent_measurements[0][0]
                    mem_diff = recent_measurements[-1][1] - recent_measurements[0][1]
                    if time_diff > 0:  # Avoid division by zero
                        current_growth_rate = mem_diff / time_diff
                        self.growth_rate.put(current_growth_rate)

                        # Check if memory usage is too high
                        if current_mem > max_memory:
                            print(
                                f"\nWARNING: Memory usage too high ({current_mem:.0f}MB > {max_memory}MB)"
                            )
                            self.stop_event.set()
                            return

                        # Check if growth rate is too high
                        if current_growth_rate > memory_growth_threshold:
                            print(
                                f"\nWARNING: Memory growing too fast ({current_growth_rate:.0f}MB/s > {memory_growth_threshold}MB/s)"
                            )
                            self.stop_event.set()
                            return

                self.peak_memory.put(current_mem)
                time.sleep(0.1)  # Check every 100ms

    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size:,}")

        # Setup memory monitoring
        stop_event = threading.Event()
        peak_memory = Queue()
        growth_rate = Queue()
        monitor = MemoryMonitor(stop_event, peak_memory, growth_rate)
        monitor_thread = threading.Thread(target=monitor.run)

        # Start monitoring
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        monitor_thread.start()

        try:
            # Time the computation
            start_time = time.time()
            _ = analyze_cluster_robustness(X, k=k, n_runs=n_runs, batch_size=batch_size)
            elapsed = time.time() - start_time

            # Stop monitoring and get peak memory
            stop_event.set()
            monitor_thread.join()

            # Get the peak memory from the queue
            peak_mem = (
                max(list(peak_memory.queue)) if not peak_memory.empty() else mem_before
            )
            mem_increase = peak_mem - mem_before

            results.append(
                {
                    "batch_size": batch_size,
                    "time": elapsed,
                    "peak_memory_increase_mb": mem_increase,
                    "completed": True,
                }
            )
            print(f"Time: {elapsed:.1f}s, Peak Memory increase: {mem_increase:.1f}MB")

        except Exception as e:
            print(f"Error during computation: {str(e)}")
            stop_event.set()
            monitor_thread.join()

            # Add result with failure indication
            results.append({"batch_size": batch_size, "completed": False})

            # If we hit memory issues, skip larger batch sizes
            if isinstance(e, MemoryError) or stop_event.is_set():
                print("Stopping tests as memory usage is growing too quickly")
                break

    # Filter out incomplete results for plotting
    completed_results = [r for r in results if r.get("completed", False)]
    if not completed_results:
        print("No batch sizes completed successfully")
        return results

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Time plot
    ax1.plot(
        [r["batch_size"] for r in completed_results],
        [r["time"] for r in completed_results],
        marker="o",
    )
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Processing Time vs Batch Size")
    ax1.grid(True)

    # Memory plot
    ax2.plot(
        [r["batch_size"] for r in completed_results],
        [r["peak_memory_increase_mb"] for r in completed_results],
        marker="o",
    )
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Peak Memory Increase (MB)")
    ax2.set_title("Peak Memory Usage vs Batch Size")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return results


# Test a range of batch sizes
batch_sizes = [500, 1000, 2000, 5000, 10000]  # Added larger sizes to test memory limits
results = test_batch_sizes(
    X_tokens,
    batch_sizes,
    n_runs=2,
    memory_growth_threshold=100,  # Stop if memory grows faster than 100MB/s
    max_memory=8000,  # Stop if total memory exceeds 8GB
)

# Use the results to choose optimal batch size for full analysis
# We'll pick the largest batch size that keeps peak memory increase under 1GB
memory_threshold = 1000  # MB
valid_results = [r for r in results if r["peak_memory_increase_mb"] < memory_threshold]
if valid_results:
    optimal_batch_size = max(valid_results, key=lambda x: x["batch_size"])["batch_size"]
else:
    optimal_batch_size = min(results, key=lambda x: x["peak_memory_increase_mb"])[
        "batch_size"
    ]

print(f"\nChosen optimal batch size: {optimal_batch_size:,}")

# Now run the full analysis with the optimal batch size
k_values = range(4, 21, 2)
robustness_scores = []

for k in k_values:
    score = analyze_cluster_robustness(
        X_tokens, k=k, n_runs=5, batch_size=optimal_batch_size
    )
    robustness_scores.append(score)
    print(f"k={k}: {score:.3f}")

# Plot robustness vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, robustness_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Robustness score")
plt.title("Cluster Robustness vs Number of Clusters (Token Level)")
plt.grid(True)
plt.show()


# %% Analyze cluster patterns with respect to linguistic features
def analyze_cluster_linguistics(
    df: pd.DataFrame, expert_layer: int = 1, k: int = 8, random_state: int = 42
):
    """
    Analyze the relationship between clusters and linguistic features.

    Args:
        df: DataFrame with token information
        expert_layer: Which expert layer to use for clustering
        k: Number of clusters
        random_state: Random seed for KMeans

    Returns:
        DataFrame with cluster assignments
    """
    print(
        f"Analyzing linguistic patterns for k={k} clusters at expert layer {expert_layer}"
    )

    df = df.copy()

    X = np.stack(df[f"expert_{expert_layer}"].values)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X).astype(str)

    # Analyze POS distributions in each cluster
    pos_cols = ["upos", "xpos"]
    for pos_col in pos_cols:
        if pos_col in df.columns:
            print(f"\n=== {pos_col.upper()} Distribution Across Clusters ===")

            # Create raw contingency table
            contingency = pd.crosstab(df["cluster"], df[pos_col])

            # Calculate expected counts based on overall distribution
            overall_dist = df[pos_col].value_counts(normalize=True)
            cluster_sizes = contingency.sum(axis=1)
            expected_counts = pd.DataFrame(
                {pos: cluster_sizes * prob for pos, prob in overall_dist.items()},
                index=contingency.index,
            )

            # Calculate z-scores
            z_scores = (contingency - expected_counts) / np.sqrt(expected_counts)

            # Display top 3 overrepresented POS tags for each cluster
            for cluster in sorted(df["cluster"].unique()):
                top_pos = z_scores.loc[cluster].sort_values(ascending=False).head(3)
                print(f"\nCluster {cluster} most overrepresented POS tags:")
                for pos, z_score in top_pos.items():
                    print(f"  {pos}: z-score = {z_score:.2f}")

            # Plot heatmap of z-scores
            plt.figure(figsize=(12, 8))
            sns.heatmap(z_scores, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
            plt.title(f"Cluster vs {pos_col.upper()} Z-Scores")
            plt.tight_layout()
            plt.show()

    # Analyze expert routing preferences
    print("\n=== Expert Routing Preferences ===")
    expert_col = f"expert_{expert_layer}"
    # For each cluster, find which expert gets the highest probability on average
    cluster_expert_prefs = {}
    for cluster in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster]
        # Extract the index of the max value for each expert vector
        expert_vectors = np.stack(cluster_df[expert_col].values)
        max_indices = np.argmax(expert_vectors, axis=1)
        # Count occurrences of each expert index
        expert_counts = Counter(max_indices)
        # Get distribution over experts
        total = sum(expert_counts.values())
        expert_dist = {
            i: expert_counts.get(i, 0) / total for i in range(8)
        }  # Assuming 8 experts
        cluster_expert_prefs[cluster] = expert_dist

    # Plot expert distribution heatmap
    layer_num = expert_col.split("_")[1]
    expert_dist_matrix = pd.DataFrame.from_dict(cluster_expert_prefs, orient="index")
    plt.figure(figsize=(12, 8))
    sns.heatmap(expert_dist_matrix, cmap="viridis", annot=True, fmt=".2%")
    plt.title(f"Expert Distribution by Cluster (Layer {layer_num})")
    plt.xlabel("Expert Index")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

    # Analyze relationships between clusters and sentence position
    # if "decoded_token_position" in df.columns:
    #     print("\n=== Token Position Analysis ===")

    #     # Group by sentence_id to get sentence lengths
    #     sent_lengths = df.groupby("sentence_id")["decoded_token_position"].max() + 1
    #     sent_length_map = sent_lengths.to_dict()

    #     # Add relative position column
    #     df["relative_position"] = df.apply(
    #         lambda x: x["decoded_token_position"] / sent_length_map[x["sentence_id"]],
    #         axis=1,
    #     )

    #     # Compute position statistics
    #     position_stats = df.groupby("cluster").agg(
    #         {
    #             "decoded_token_position": ["mean", "std", "median"],
    #             "relative_position": ["mean", "std", "median"],
    #         }
    #     )
    #     print("\nPosition Statistics by Cluster:")
    #     print(position_stats)

    #     # Create subplots for absolute and relative positions
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    #     # Absolute position distribution
    #     sns.boxplot(x="cluster", y="decoded_token_position", data=df, ax=ax1)
    #     ax1.set_title("Absolute Position Distribution by Cluster")
    #     ax1.set_xlabel("Cluster")
    #     ax1.set_ylabel("Position in Sequence")

    #     # Relative position distribution
    #     sns.boxplot(x="cluster", y="relative_position", data=df, ax=ax2)
    #     ax2.set_title("Relative Position Distribution by Cluster")
    #     ax2.set_xlabel("Cluster")
    #     ax2.set_ylabel("Relative Position (0-1)")

    #     plt.tight_layout()
    #     plt.show()

    return df


# %% Execute the analysis for different k values
k_values = [6]
for k in k_values:
    print(f"\n{'=' * 20} Analyzing k={k} {'=' * 20}")
    df_with_clusters = analyze_cluster_linguistics(filter_df(df), expert_layer=1, k=k)


# %% Find representative examples for each cluster
def find_representative_examples(df_with_clusters, expert_layer=1, num_examples=20):
    """
    Calculate alignment scores for each example with its cluster's directionality
    and add them to the DataFrame.

    Args:
        df_with_clusters: DataFrame with cluster assignments
        expert_layer: Which expert layer to analyze
        num_examples: Number of top examples to print for each cluster

    Returns:
        DataFrame with added alignment_score column
    """
    from sklearn.metrics.pairwise import cosine_similarity

    expert_col = f"expert_{expert_layer}"
    df_with_scores = df_with_clusters.copy()
    df_with_scores["alignment_score"] = 0.0
    all_expert_vectors = np.stack(df_with_clusters[expert_col].values)
    global_mean = np.mean(all_expert_vectors, axis=0)

    for cluster in sorted(df_with_clusters["cluster"].unique()):
        print(f"\n=== Processing alignment scores for Cluster {cluster} ===")
        cluster_df = df_with_clusters[df_with_clusters["cluster"] == cluster]
        cluster_vectors = np.stack(cluster_df[expert_col].values)
        cluster_mean = np.mean(cluster_vectors, axis=0)
        directionality = cluster_mean - global_mean
        directionality_norm = directionality / np.linalg.norm(directionality)
        alignments = cosine_similarity(
            cluster_vectors, directionality_norm.reshape(1, -1)
        )
        cluster_indices = df_with_clusters[df_with_clusters["cluster"] == cluster].index
        df_with_scores.loc[cluster_indices, "alignment_score"] = alignments.flatten()
        top_indices = np.argsort(alignments.flatten())[-num_examples:][::-1]
        top_example_indices = cluster_df.iloc[top_indices].index
        print(f"\nTop {num_examples} examples for Cluster {cluster}:")
        for i, idx in enumerate(top_example_indices):
            example = df_with_scores.loc[idx]
            print(
                f"  {i + 1}. Token: {example.get('decoded_token', '')}"
                f"     POS: {example.get('upos', '')} / {example.get('xpos', '')}"
            )

    return df_with_scores


# Add alignment scores to the DataFrame
df_with_alignments = find_representative_examples(df_with_clusters, expert_layer=1)


# %% Analyze XPOS-UPOS relationships
def analyze_xpos_upos_relationships(df: pd.DataFrame):
    print("\n=== Analyzing XPOS-UPOS relationships ===")

    # Create contingency table and normalize by UPOS (rows)
    pos_contingency = pd.crosstab(
        filter_df(df)["upos"], filter_df(df)["xpos"], normalize="index"
    )

    # For each XPOS tag, find its primary UPOS association
    xpos_primary_upos = {}
    for xpos in pos_contingency.columns:
        primary_upos = pos_contingency[xpos].idxmax()
        primary_upos_value = pos_contingency[xpos].max()
        xpos_primary_upos[xpos] = (primary_upos, primary_upos_value)

    # Sort UPOS (rows) numerically
    sorted_rows = sorted(pos_contingency.index)

    # Sort XPOS (columns) by their primary UPOS association
    # Within each UPOS group, sort by the strength of the association
    sorted_cols = sorted(
        pos_contingency.columns,
        key=lambda x: (
            sorted_rows.index(xpos_primary_upos[x][0]),
            -xpos_primary_upos[x][1],
        ),
    )

    # Reorder the contingency table
    pos_contingency = pos_contingency.loc[sorted_rows, sorted_cols]

    # Create heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        pos_contingency,
        cmap="viridis",
        annot=True,
        fmt=".2%",
        cbar_kws={"label": "Proportion within UPOS"},
    )
    plt.title("XPOS Distribution within each UPOS Category")
    plt.xlabel("XPOS (Penn Treebank Tags)")
    plt.ylabel("UPOS (Universal POS Tags)")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\nNumber of unique tags:")
    print(f"UPOS tags: {len(pos_contingency.index)}")
    print(f"XPOS tags: {len(pos_contingency.columns)}")


analyze_xpos_upos_relationships(df)

# %%

# Idea: construct a bipartite network of tokens and experts
# Draw an edge token -> expert if token routes to expert
# Can we learn a latent representation of tokens as well as experts
# Optimize a noise contrastive loss - tokens and experts that are connected should be similar
# Map the experts in space, tokens with highest affinity should be closest to that expert
# Good thing - you can map all the experts in the same space
# Is layer 1 expert 3 similar to layer 5 expert 1?
