# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

# %%
df_types = pd.read_parquet("data/switch_base_8_wordsim_logprobs.parquet.gzip")
# Get router probabilities instead of log probs
df_types_probs = df_types.copy()
cols = [
    "route_vector",
    "expert_1",
    "expert_3",
    "expert_5",
    "expert_7",
    "expert_9",
    "expert_11",
]
for col in cols:
    df_types_probs[col] = df_types[col].apply(lambda x: np.exp(x))

df_similarity = pd.read_csv("similarity/wordsim353crowd.csv")
df_similarity = df_similarity.rename(
    columns={"Word 1": "word1", "Word 2": "word2", "Human (Mean)": "sim_score"}
)


# %%
def get_similarity_scores(
    df_types: pd.DataFrame,
    df_similarity: pd.DataFrame,
    col_name: str,
    similarity_measure: str = "cosine",
) -> pd.DataFrame:
    """Get the similarity scores from df_types using the similarity measure.

    Construct the pairs of words from df_similarity using the representations in df_types,
    then compute the similarity scores using the similarity measure.

    e.g if the first row of df_similarity is:
    word1, word2, sim_score
    apple, orange, 0.8

    and col_name is "route_vector", then the first row of the result should be:
    word1, word2, route_vector1, route_vector2, sim_score
    apple, orange, route_vector1, route_vector2, sim_score

    where route_vector1 and route_vector2 are the route vectors of apple and orange,
    respectively, from df_types, and sim_score is the similarity measure applied to
    those route vectors.
    """

    df_similarity = df_similarity[["word1", "word2"]].copy()

    def similarity_func(x, y):
        if similarity_measure == "cosine":
            return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
        elif similarity_measure == "jensenshannon":
            return 1 - jensenshannon(x, y).item()

    rows = []
    for word1, word2 in zip(df_similarity["word1"], df_similarity["word2"]):
        try:
            route_vector1 = df_types[df_types["text"] == word1][col_name].values[0]
            route_vector2 = df_types[df_types["text"] == word2][col_name].values[0]
        except IndexError:
            continue

        sim_score = similarity_func(route_vector1, route_vector2)
        rows.append({"word1": word1, "word2": word2, "sim_score": sim_score})

    return pd.DataFrame(rows)


# %%
def evaluate_similarity_scores(
    df_types: pd.DataFrame,
    df_similarity: pd.DataFrame,
    col_name: str,
    similarity_measure: str = "cosine",
) -> float:
    """Evaluate the similarity scores from df_types using the similarity measure."""
    route_scores = get_similarity_scores(
        df_types, df_similarity, col_name, similarity_measure
    )
    route_scores["ref_score"] = route_scores.apply(
        lambda row: df_similarity.loc[
            (df_similarity["word1"] == row["word1"])
            & (df_similarity["word2"] == row["word2"])
        ]["sim_score"].values[0],
        axis=1,
    )

    corr = route_scores["sim_score"].corr(route_scores["ref_score"])
    plt.scatter(route_scores["sim_score"], route_scores["ref_score"])
    plt.xlabel("Route Score")
    plt.ylabel("Reference Score")
    plt.title(f"{col_name} Route Score vs Reference Score (corr: {corr:.2f})")
    plt.show()

    return corr


# %%
evaluate_similarity_scores(
    df_types=df_types_probs,
    df_similarity=df_similarity,
    col_name="route_vector",
    similarity_measure="jensenshannon",
)

# %%
# Layer 1
evaluate_similarity_scores(
    df_types=df_types_probs,
    df_similarity=df_similarity,
    col_name="expert_1",
    similarity_measure="jensenshannon",
)

# %% Expert 11 probs
evaluate_similarity_scores(
    df_types=df_types_probs,
    df_similarity=df_similarity,
    col_name="expert_11",
    similarity_measure="jensenshannon",
)

# %%
