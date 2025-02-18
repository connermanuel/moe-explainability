# %%
import functools
import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
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
                row[f"expert_id_{layer_num}"] = router_probs[layer_num][1][
                    0, decoded_token_id
                ].item()
        yield row


@cache_to_file
def subset_to_df(data_subset: tp.Any) -> pd.DataFrame:
    rows = []
    for sent_id, datum in tqdm(enumerate(data_subset), total=len(data_subset)):
        for entry in iter_sentence(model, tokenizer, datum):
            entry = {"sentence_id": sent_id, **entry}
            rows.append(entry)

    df = pd.DataFrame(rows)

    def integers_to_one_hot(arr, n):
        result = np.zeros(len(arr) * n)
        result[np.arange(len(arr)) * n + arr] = 1
        return result

    def create_route_vector(row):
        experts = np.array([row[f"expert_id_{i}"] for i in range(1, 12, 2)])
        return integers_to_one_hot(experts, 8)

    df["route_vector"] = df.apply(create_route_vector, axis=1)

    return df


df = subset_to_df(data_subset=data["train"], filename="ud_train_routes.parquet.gzip")
df.head()


# %% Plotting function
@cache_to_file
def reduce_routes(
    df: pd.DataFrame,
    route_col: str,
    method: str = "tsne",
    dimensions: int = 2,
) -> pd.DataFrame:
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
    method="tsne",
    dimensions=2,
):
    # Get reduced dimensions from cache or compute them
    cache_file_routes = f"routes_reduced/{method}_{dimensions}d.parquet.gzip"
    routes_reduced_df = reduce_routes(
        df, route_col, method, dimensions, filename=cache_file_routes
    )
    routes_reduced = routes_reduced_df.values
    print("Reduction complete.")

    # Add cluster analysis
    print("\nAnalyzing clusters...")

    # cache_file_scores = f"silhouette_scores/{method}_{dimensions}d.parquet.gzip"
    # analyze_clusters(routes_reduced, df[cat_col].values, filename=cache_file_scores)

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


# %%
plot_routes_by_category(df, cat_col="upos", method="pca", dimensions=3)
# %%

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
