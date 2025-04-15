# %% Imports
import functools
import os
import typing as tp
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer, SwitchTransformersEncoderModel


# %% Utility Functions
def cache_to_file(func):
    """Caches the output of a function to a parquet file."""

    @functools.wraps(func)
    def wrapper(*args, filename: str | None = None, compress: bool = True, **kwargs):
        if filename and os.path.exists(filename):
            print(f"Loading {filename}")
            return pd.read_parquet(filename)
        print(f"Computing {func.__name__}")
        df = func(*args, **kwargs)
        if filename:
            print(f"Saving {filename}")
            df.to_parquet(filename, compression="gzip" if compress else None)
        return df

    return wrapper


# %% Model and Data Loading
def load_model_and_tokenizer(
    model_name: str,
) -> tuple[AutoTokenizer, SwitchTransformersEncoderModel]:
    """Loads the model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SwitchTransformersEncoderModel.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


def get_token_indices(text: str, tokens: list[str]) -> list[int]:
    target_text = text.replace(" ", "").lower()
    indices = []
    current_pos = 0
    token_idx = 0

    while current_pos < len(target_text) and token_idx < len(tokens):
        token = tokens[token_idx]
        token_clean = token.replace(" ", "").lower()

        text_remainder = target_text[current_pos:]
        if token_clean and text_remainder.startswith(token_clean):
            indices.append(token_idx)
            current_pos += len(token_clean)

        token_idx += 1
    return indices


def map_decoded_tokens_to_sent_tokens(
    decoded_tokens: list[str], sent_tokens: list[str]
) -> list[int]:
    """Maps decoded tokens to sentence token indices.

    Returns a list L where L[i] is the index of the sentence token that the i-th decoded token belongs to."""
    mapping = []
    sent_token_idxs = sum(
        [[i] * len(sent_tokens[i]) for i in range(len(sent_tokens))], []
    )

    total_len = 0
    for decoded_token in decoded_tokens:
        total_len += len(decoded_token)
        try:
            mapped = sent_token_idxs[total_len - 1]
        except IndexError:
            mapped = -1
        mapping.append(mapped)

    return mapping


def iter_sentence(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    datum: dict[str],
    device: str = "cuda",
    normalize: bool = True,
) -> tp.Iterator[dict[str, str]]:
    """Iterates through the tokens of a sentence and returns expert probabilities."""
    # Tokenize the input sentences
    input_ids = tokenizer(datum["text"], return_tensors="pt").input_ids.to(device)

    # Call the encoder model on the entire batch
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_router_logits=True,
        )
        router_probs = outputs.router_probs

    # Clean up the tokenizer tokens and the sentence tokens to align with the text
    text = datum["text"]
    input_ids = input_ids[0].tolist()
    tokenizer_tokens = [tokenizer.decode(input_id) for input_id in input_ids]
    tokenizer_token_idxs = get_token_indices(text, tokenizer_tokens)
    input_ids = [input_ids[idx] for idx in tokenizer_token_idxs] + [
        tokenizer.eos_token_id
    ]
    tokenizer_tokens = [tokenizer_tokens[idx] for idx in tokenizer_token_idxs] + [
        "</s>"
    ]

    sent_token_idxs = get_token_indices(text, datum["tokens"])
    for feature in datum.keys():
        if isinstance(datum[feature], list):
            datum[feature] = [datum[feature][idx] for idx in sent_token_idxs]

    # Do the mapping with the cleaned tokens
    mapping = map_decoded_tokens_to_sent_tokens(tokenizer_tokens, datum["tokens"])
    assert len(mapping) == len(tokenizer_tokens)

    for decoded_token_id, sent_token_id in enumerate(mapping):
        input_id = input_ids[decoded_token_id]
        if input_id == tokenizer.eos_token_id:
            sent_token_id = -1
            sent_token = "<eos>"
            upos = -1
            xpos = None
        else:
            sent_token = datum["tokens"][sent_token_id]
            xpos = datum["xpos"][sent_token_id]
            upos = datum["upos"][sent_token_id]

        row = {
            "decoded_token_position": decoded_token_id,
            "sent_token_position": sent_token_id,
            "input_id": input_id,
            "decoded_token": tokenizer_tokens[decoded_token_id],
            "sent_token": sent_token,
            "xpos": xpos,
            "upos": upos,
        }
        for layer_num in range(1, len(router_probs), 2):
            probs = router_probs[layer_num][0][0][decoded_token_id]
            if normalize:
                probs = F.log_softmax(probs, dim=0)
            probs = probs.cpu().numpy()
            row[f"expert_{layer_num}"] = probs
        yield row


def create_route_vector(row) -> np.ndarray:
    all_probs = [row[f"expert_{i}"] for i in range(1, 12, 2)]
    return np.concatenate(all_probs)


@cache_to_file
def extract_token_routes_df(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    data_subset: tp.Any,
    normalize: bool = True,
) -> pd.DataFrame:
    """Converts a subset of the dataset to a DataFrame with expert probabilities."""

    rows = []
    for sent_id, datum in tqdm(enumerate(data_subset), total=len(data_subset)):
        for entry in iter_sentence(model, tokenizer, datum, normalize=normalize):
            entry = {"sentence_id": sent_id, **entry}
            rows.append(entry)

    df = pd.DataFrame(rows)

    df["route_vector"] = df.apply(create_route_vector, axis=1)

    return df


def get_unaligned_tokens(df: pd.DataFrame) -> pd.DataFrame:
    mask_1 = df.apply(lambda x: x["decoded_token"] not in x["sent_token"], axis=1)
    mask_2 = df.apply(lambda x: x["sent_token"] not in x["decoded_token"], axis=1)
    mask_3 = df["sent_token_position"] != -1
    return df[mask_1 & mask_2 & mask_3]


# %% Data Preprocessing
def collapse_df_by_input_id(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses the DataFrame by input_id, aggregating features."""

    def mode(x):
        try:
            return x.value_counts().idxmax()
        except ValueError:
            return None

    agg_dict = {
        "decoded_token": "first",
        "xpos": mode,
        "upos": mode,
        "route_vector": "mean",
        "sentence_id": "count",
    }
    for i in range(1, 12, 2):
        agg_dict[f"expert_{i}"] = "mean"

    df_collapsed = df.groupby("input_id").agg(agg_dict).reset_index()
    df_collapsed = df_collapsed.rename(columns={"sentence_id": "count"})
    return df_collapsed


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filters out specific input_ids from the DataFrame."""
    return df[~df["input_id"].isin([1, 3])]


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


# %% Main Execution
if __name__ == "__main__":
    # Load model and data
    model_name = "google/switch-base-8"
    tokenizer, model = load_model_and_tokenizer(model_name)
    data = load_dataset("universal_dependencies", "en_ewt")

    # Create and preprocess data
    df = extract_token_routes_df(
        model,
        tokenizer,
        data_subset=data["train"],
        filename="data/switch_base_8_ud_train_token_routes_normalized.parquet.gzip",
    )
    df_types = collapse_df_by_input_id(df)
    df_types_filtered = filter_df(df_types)

    df_unnorm = extract_token_routes_df(
        model,
        tokenizer,
        data_subset=data["train"],
        normalize=False,
        filename="data/switch_base_8_ud_train_token_routes_raw.parquet.gzip",
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

    # %% Cluster by different numbers of clusters
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
    # %% Transitions
    cluster_and_analyze_transitions(df_unnorm, n_clusters=8)


# Qualitative analysis - look at some specific examples in some clusters
# e.g. though we found a cluster where DT is the determinant tag, we figure out that those DTs are in the first position
# The DS probably also has a dependency parse - what is the dependency arc for each token? Incoming - POS - outgoing
# Randomly sample frome each cluster and look at (a) any possible systematicity and (b)
# Can we get a mixture of homogeneity / skewedness and see how that changes over layers
# We could also get the transition matrix over layers e.g. expert 1 -> 3

# Why are we doing this?
# The hope - we want to see the affinities of each cluster for each expert
# e.g. what if we have two clusters that are very different (maybe two completely diff parts of speech?)
# but both map to the same cluster?

# How do we verify robustness?
# e.g. take random pairs from the data. Answer the question - should this pair be in the same cluster
# or in different clusters?
# How many
