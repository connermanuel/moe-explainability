# %% Imports
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scripts.get_routes.get_routes import (
    cache_to_file,
    collapse_df_by_input_id,
    filter_df,
)


# %% Clustering
def cluster_df(
    df: pd.DataFrame,
    data_col: str,
    n_clusters: int,
    random_state: int = 42,
) -> pd.Series:
    """Performs k-means clustering on the vectors in `data_col` of the DataFrame `df`.

    Returns a pandas series with the cluster assignments."""
    X = np.stack(df[data_col].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return kmeans.fit_predict(X).astype(str)


# %% Basic analysis and plotting
@cache_to_file
def reduce_routes(
    df: pd.DataFrame,
    route_col: str,
    method: str = "tsne",
    dimensions: int = 2,
) -> pd.DataFrame:
    """Reduces the dimensionality of route vectors using t-SNE or PCA."""
    # Dimensionality reduction
    if method.lower() == "tsne":
        if dimensions == 2:
            from tsnecuda import TSNE
        else:
            from sklearn.manifold import TSNE

            print("Using sklearn.manifold.TSNE for 3D t-SNE. This will be slow.")

        reducer = TSNE(n_components=dimensions, random_seed=42)
    elif method.lower() == "pca":
        reducer = PCA(n_components=dimensions)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    print(
        f"Reducing {df.shape[0]} routes to {dimensions} dimensions using {method.upper()}"
    )
    routes_reduced = reducer.fit_transform(np.stack(df[route_col].values))
    return pd.DataFrame(routes_reduced, columns=[f"dim_{i}" for i in range(dimensions)])


def plot_routes_by_category(
    df,
    route_col="route_vector",
    cat_col="xpos",
    method="pca",
    dimensions=2,
):
    """Plots route vectors clustered by a given category."""
    # Get reduced dimensions from cache or compute them
    routes_reduced_df = reduce_routes(df, route_col, method, dimensions)
    routes_reduced = routes_reduced_df.values

    if dimensions == 2:
        # Create 2D seaborn plot
        plot_df = pd.DataFrame(
            {
                "x": routes_reduced[:, 0],
                "y": routes_reduced[:, 1],
                "category": df[cat_col].astype(str),
            }
        )
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x="x", y="y", hue="category", alpha=0.6)
        plt.title(f"Route Vectors Clustered by {cat_col} ({method.upper()})")
        plt.show()

    elif dimensions == 3:
        # Create 2D or 3D plotly plot
        import plotly.express as px

        plot_df = pd.DataFrame(
            {
                "x": routes_reduced[:, 0],
                "y": routes_reduced[:, 1],
                "z": routes_reduced[:, 2],
                "category": df[cat_col].astype(str),
            }
        )
        fig = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="z",
            color="category",
            opacity=0.6,
            title=f"Route Vectors Clustered by {cat_col} ({method.upper()})",
        )
        fig.update_traces(marker_size=3)
        fig.show()


# %% Statistical analysis / transitions
def cluster_and_analyze(
    df: pd.DataFrame,
    data_col: str,
    list_cats: list[str],
    n_clusters: int,
    plot: bool = True,
    display_dims: int | None = 3,
) -> pd.Series:
    """
    Performs k-means clustering on the vectors in `data_col` of the DataFrame `df`,
    then analyzes the distribution of values in `list_cats` within each cluster.
    Finally, plots the clusters using `plot_routes_by_category`.

    Returns a pandas series with the cluster assignments.
    """

    df = df.copy()
    df["cluster"] = cluster_df(df, data_col, n_clusters, random_state=42)

    for cluster_id in range(n_clusters):
        print(f"\n--- Cluster {cluster_id} ---")
        df_clusters = df[df["cluster"] == str(cluster_id)]

        for cat_col in list_cats:
            most_common = Counter(df_clusters[cat_col]).most_common(5)
            most_common_props = {k: v / len(df_clusters) for k, v in most_common}
            print(f"\nTop 5 most common '{cat_col}' values:")
            for k, v in most_common_props.items():
                print(f"{k}: {v:.2%}")

    if plot:
        plot_routes_by_category(
            df,
            route_col=data_col,
            cat_col="cluster",
            method="pca",
            dimensions=display_dims,
        )

    return df["cluster"]


def test_cluster_uniformity(df, data_col, cat_col, k, random_state=42):
    """
    Performs k-means clustering and then tests if the distribution of cat_col
    is uniform across the clusters using a Chi-Square test.

    Returns:
        p-value from the Chi-Square test.
    """

    # Cluster
    X = np.stack(df[data_col].values)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X).astype(str)

    for cluster_id in range(k):
        print(f"\n--- Cluster {cluster_id} ---")
        cluster_df = df[df["cluster"] == str(cluster_id)]

        most_common = Counter(cluster_df[cat_col]).most_common(5)
        most_common_props = {k: v / len(cluster_df) for k, v in most_common}
        print(f"\nTop 5 most common '{cat_col}' values:")
        for k, v in most_common_props.items():
            print(f"{k}: {v:.2%}")

    # Create contingency table
    contingency_table = df.groupby(["cluster", cat_col]).size().unstack(fill_value=0)

    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return p


def cluster_and_analyze_transitions(df: pd.DataFrame, n_clusters: int = 8):
    """
    Clusters the DataFrame based on expert router probabilities and analyzes transitions
    between cluster assignments across different expert layers.

    Args:
        df (pd.DataFrame): The input DataFrame containing expert router probabilities.
        n_clusters (int): The number of clusters to use for k-means clustering.

    Returns:
        pd.DataFrame: The DataFrame with added cluster assignment columns for each expert layer.
    """

    expert_layers = [1, 3, 5, 7, 9, 11]
    cluster_cols = {}

    # Cluster each expert layer
    for layer in expert_layers:
        data_col = f"expert_{layer}"
        cluster_col = f"cluster_{layer}"
        clusters = cluster_df(df, data_col, n_clusters)
        df[cluster_col] = clusters
        cluster_cols[layer] = cluster_col

    # Calculate and display transition matrices
    for i in range(len(expert_layers) - 1):
        layer1 = expert_layers[i]
        layer2 = expert_layers[i + 1]
        cluster_col1 = cluster_cols[layer1]
        cluster_col2 = cluster_cols[layer2]

        transition_counts = (
            df.groupby([cluster_col1, cluster_col2]).size().unstack(fill_value=0)
        )
        transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix, annot=True, cmap="mako", fmt=".2f")
        plt.title(f"Transition Matrix: Layer {layer1} -> Layer {layer2}")
        plt.xlabel(f"Cluster Layer {layer2}")
        plt.ylabel(f"Cluster Layer {layer1}")
        plt.show()

    return df


if __name__ == "__main__":
    df = pd.read_parquet(
        "data/switch_base_8_ud_train_token_routes_normalized.parquet.gzip"
    )
    df_types = collapse_df_by_input_id(df)
    df_types_filtered = filter_df(df_types)

    df_unnorm = pd.read_parquet(
        "data/switch_base_8_ud_train_token_routes_raw.parquet.gzip"
    )
    df_types_unnorm = collapse_df_by_input_id(df_unnorm)
    df_types_unnorm_filtered = filter_df(df_types_unnorm)

    # Example analyses
    cluster_and_analyze(
        df=df_types_filtered,
        data_col="expert_1",
        list_cats=["xpos"],
        n_clusters=8,
        display_dims=3,
    )
    df_multiple_clusterings = df_unnorm.copy()
    for i in range(4, 17):
        col_name = f"clustering_{i}"
        df_multiple_clusterings[col_name] = cluster_and_analyze(
            df=df_types_unnorm_filtered,
            data_col="expert_1",
            list_cats=[],
            n_clusters=i,
            plot=False,
        )
    cluster_and_analyze_transitions(df_unnorm, n_clusters=8)
