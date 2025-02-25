# %%
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from transformers import AutoTokenizer, SwitchTransformersEncoderModel


# %% Load the model and dataset
def load_model_and_tokenizer(model_name, device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SwitchTransformersEncoderModel.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


model_name = "google/switch-base-8"
model, tokenizer = load_model_and_tokenizer(model_name)
data = load_dataset("universal_dependencies", "en_ewt")


# %% Get token probs for all tokens in all layers


def cache_to_file(func):
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


def map_input_ids_to_tokens(
    decoded_tokens: list[str], sent_tokens: list[str]
) -> list[int]:
    """Example:
    decoded_tokens = ['Al', '-', 'Z', 'a', 'man']
    sent_tokens = ['Al', '-', 'Zaman']
    output = [0, 1, 2, 2, 2]
    """
    mapping = []
    token_idx = 0
    token = sent_tokens[token_idx]
    token_accum = ""

    for input_id in decoded_tokens:
        token_accum += input_id
        mapping.append(token_idx)
        if token_accum == token:
            token_idx += 1
            if token_idx < len(sent_tokens):
                token = sent_tokens[token_idx]
            token_accum = ""

    return mapping


def iter_sentence(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    datum: dict[str],
    device: str = "cuda",
    normalize: bool = True,
) -> tp.Iterator[dict[str, str]]:
    """Iterate through the tokens of an entry from UD and return the expert number for each token in each layer."""
    # Tokenize the input sentences
    input_ids = tokenizer(
        datum["text"], return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)

    # Call the encoder model on the entire batch
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, output_hidden_states=True, output_router_logits=True
        )
        router_probs = outputs.router_probs

    # Combine the two lists and iterate every element from each list
    tokenizer_tokens = [tokenizer.decode(input_id) for input_id in input_ids[0]]
    mapping = map_input_ids_to_tokens(tokenizer_tokens, datum["tokens"])
    assert len(mapping) == len(tokenizer_tokens)

    for decoded_token_id, sent_token_id in enumerate(mapping):
        input_id = input_ids[0, decoded_token_id].item()
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
        for layer_num in range(len(router_probs)):
            if layer_num % 2 == 1:
                probs = router_probs[layer_num][0][0][decoded_token_id]
                if normalize:
                    probs = F.softmax(probs, dim=0)
                probs = probs.cpu().numpy()
                row[f"expert_{layer_num}"] = probs
        yield row


@cache_to_file
def subset_to_df(data_subset: tp.Any, normalize: bool = True) -> pd.DataFrame:
    rows = []
    for sent_id, datum in tqdm(enumerate(data_subset), total=len(data_subset)):
        for entry in iter_sentence(model, tokenizer, datum, normalize=normalize):
            entry = {"sentence_id": sent_id, **entry}
            rows.append(entry)

    df = pd.DataFrame(rows)

    def create_route_vector(row) -> np.ndarray:
        all_probs = [row[f"expert_{i}"] for i in range(1, 12, 2)]
        return np.concatenate(all_probs)

    df["route_vector"] = df.apply(create_route_vector, axis=1)

    return df


df = subset_to_df(
    data_subset=data["train"],
    normalize=False,
    filename="ud_train_route_logits.parquet.gzip",
)
df.head()


# %% Plotting function
@cache_to_file
def reduce_routes(
    df: pd.DataFrame,
    route_col: str,
    method: str = "tsne",
    dimensions: int = 2,
) -> pd.DataFrame:
    """Reduce the route vectors to 2D or 3D using t-SNE or PCA.
    Returns a DataFrame with the reduced dimensions."""
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


def analyze_clusters(routes_reduced, categories, filename: str = None) -> pd.DataFrame:
    # Calculate silhouette scores
    @cache_to_file
    def get_sample_scores_df(routes_reduced, categories):
        sil_samples = silhouette_samples(routes_reduced, categories)

        # Get per-cluster scores
        cluster_scores = {}
        unique_categories = np.unique(categories)
        for category in tqdm(unique_categories):
            mask = categories == category
            cluster_scores[category] = np.mean(sil_samples[mask])

        # Convert to dataframe for nice display
        scores_df = pd.DataFrame(
            {
                "Category": cluster_scores.keys(),
                "Silhouette Score": cluster_scores.values(),
            }
        ).sort_values("Silhouette Score", ascending=False)
        return scores_df

    # Print overall and per-cluster scores
    print(
        f"Overall Silhouette Score: {silhouette_score(routes_reduced, categories):.3f}"
    )

    scores_df = get_sample_scores_df(routes_reduced, categories, filename=filename)
    print("\nPer-category Silhouette Scores:")
    print(scores_df)

    return scores_df


def plot_routes_by_category(
    df,
    route_col="route_vector",
    cat_col="xpos",
    method="pca",
    dimensions=2,
):
    # Get reduced dimensions from cache or compute them
    # cache_file_routes = f"routes_reduced/{method}_{dimensions}d.parquet.gzip"
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


# %% Notes


# 3D tensor: (tokens, layers, experts)
# How many times did you see his particular token get routed through i'th layer's k'th expert?
# Instead of taking the counts, you can consider the activation probabilities
# Use types, not tokens -> collapse by type and for now take in the most common pos tag
# Probably want to do k-means cluster agnostic to POS tag, then ask - is there homogeneity in terms of syntatic tags?
# Benefits:
# - do tensor decomposition
# - low dimensional representations of every token
# - maybe we can start with atomic subwords
# - for switch-base-8, there arent a lot of layers and experts so dimensionality reduction is not needed
# - We can do clustering on 2D slices of this tensor (e.g. choose one layer or choose one expert)


def collapse_df_by_input_id(df: pd.DataFrame) -> pd.DataFrame:
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


df_types = collapse_df_by_input_id(df)
df_types.head()


# %% Cluster and analyze by POS
def cluster_and_analyze(
    df: pd.DataFrame,
    data_col: str,
    list_cats: list[str],
    k: int,
    display_dims: int = 3,
):
    """
    Performs k-means clustering on the vectors in `data_col` of the DataFrame `df`,
    then analyzes the distribution of values in `list_cats` within each cluster.
    Finally, plots the clusters using `plot_routes_by_category`.
    """

    df = df.copy()
    X = np.stack(df[data_col].values)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X).astype(str)

    for cluster_id in range(k):
        print(f"\n--- Cluster {cluster_id} ---")
        cluster_df = df[df["cluster"] == str(cluster_id)]

        for cat_col in list_cats:
            most_common = Counter(cluster_df[cat_col]).most_common(5)
            most_common_props = {k: v / len(cluster_df) for k, v in most_common}
            print(f"\nTop 5 most common '{cat_col}' values:")
            for k, v in most_common_props.items():
                print(f"{k}: {v:.2%}")

    plot_routes_by_category(
        df, route_col=data_col, cat_col="cluster", method="pca", dimensions=display_dims
    )


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["input_id"].isin([1, 3])]


cluster_and_analyze(
    df=filter_df(df_types), data_col="expert_1", list_cats=["xpos"], k=8, display_dims=3
)

# %%

df_unnorm = subset_to_df(
    data_subset=data["train"],
    normalize=False,
    filename="ud_train_route_logits.parquet.gzip",
)
df_types_unnorm = collapse_df_by_input_id(df_unnorm)
df_types_unnorm.head()

cluster_and_analyze(
    df=filter_df(df_types_unnorm),
    data_col="expert_1",
    list_cats=["xpos"],
    k=8,
    display_dims=3,
)

# %%
# Null hypothesis - every cluster should be uniform over tags / no systematicity
# or at least proportional to the frequency of tags in the dataset
# If there is systematicity, divergence from uniform should be high
# You can rank clusters by divergence from uniformity in a few ways
# - KL divergence
# - Chi-square test
# - Kolmogorov-Smirnov test
# - Pick clusters which are extremeley non-uniform
# You could also bootstrap from the dataset = e.g. sample with replacement and recompute the divergence
# Problem: clusters from different runs of k-means are not the same and don't map one-to-one

# How does position of token inside a word (e.g. subword token vs atomic token) affect routing?
# some TF-IDF statistic on the tokens