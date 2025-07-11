# %%
"""Example script showing how to use general functions for UD dataset processing."""

import pandas as pd
from datasets import load_dataset

from moe_explainability.datasets.alignment import align_tokens
from moe_explainability.datasets.processing import (
    add_post_processing,
    process_structured_data,
)
from moe_explainability.models.configs import SWITCH_BASE_8
from moe_explainability.models.loading import load_model
from moe_explainability.routing.extraction import extract_tokens


# %%
def add_word_alignment(df):
    """Add word-level alignment to the DataFrame (UD-specific post-processing)."""
    results = []

    for item_id, group in df.groupby("item_id"):
        # Get UD tokens for this sentence
        ud_tokens = group.iloc[0]["tokens"]
        if not ud_tokens:
            # No UD tokens, add rows as-is
            for _, row in group.iterrows():
                row_dict = row.to_dict()
                row_dict.update(
                    {
                        "word_index": -1,
                        "word_text": "<no_ud_tokens>",
                        "aligned_upos": None,
                        "aligned_lemma": None,
                        "aligned_deprel": None,
                    }
                )
                results.append(row_dict)
            continue

        # Get tokenizer tokens
        tokenizer_tokens = group["token_text"].tolist()
        text = group.iloc[0]["text"]

        # Align tokens
        alignment = align_tokens(text, tokenizer_tokens, ud_tokens)

        # Add alignment info to each token
        for _, row in group.iterrows():
            token_pos = row["position"]
            ud_token_idx = alignment.tokenizer_to_sentence.get(token_pos, -1)

            row_dict = row.to_dict()

            if ud_token_idx >= 0 and ud_token_idx < len(ud_tokens):
                # Successfully aligned
                upos_list = row["upos"] if row["upos"] else []
                lemmas_list = row["lemmas"] if row["lemmas"] else []
                deprel_list = row["deprel"] if row["deprel"] else []

                row_dict.update(
                    {
                        "word_index": ud_token_idx,
                        "word_text": ud_tokens[ud_token_idx],
                        "aligned_upos": upos_list[ud_token_idx]
                        if ud_token_idx < len(upos_list)
                        else None,
                        "aligned_lemma": lemmas_list[ud_token_idx]
                        if ud_token_idx < len(lemmas_list)
                        else None,
                        "aligned_deprel": deprel_list[ud_token_idx]
                        if ud_token_idx < len(deprel_list)
                        else None,
                    }
                )
            else:
                # Special token or unaligned
                row_dict.update(
                    {
                        "word_index": -1,
                        "word_text": "<special>",
                        "aligned_upos": None,
                        "aligned_lemma": None,
                        "aligned_deprel": None,
                    }
                )

            results.append(row_dict)

    return pd.DataFrame(results)


# %%
print("=== UD Processing with General Functions ===")

# Load UD dataset (small subset for demo)
print("Loading UD dataset...")
ud_data = load_dataset("universal_dependencies", "en_ewt", split="train[:5]")
print(f"Loaded {len(ud_data)} sentences")

# Load model
print("Loading model...")
model, tokenizer = load_model(SWITCH_BASE_8)


# %%
# Create extraction function
def extract_fn(text):
    """Extract routing tokens from text."""
    return extract_tokens(model, tokenizer, text, SWITCH_BASE_8)


# %%
# Use the GENERAL process_structured_data function
print("Processing with general structured data function...")
df = process_structured_data(
    data=list(ud_data),
    text_field="text",
    extract_fn=extract_fn,
    extra_fields=["tokens", "upos", "lemmas", "deprel", "head", "feats"],
)

print(f"Initial DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# %%
# Add word-level alignment using the general post-processing function
print("Adding word-level alignment...")
df_aligned = add_post_processing(df, add_word_alignment)

print(f"Final DataFrame shape: {df_aligned.shape}")
print(f"Final columns: {df_aligned.columns.tolist()}")

# %%
# Show sample data
print("\n=== Sample Data ===")
sample_cols = [
    "item_id",
    "token_text",
    "position",
    "word_text",
    "aligned_upos",
    "aligned_lemma",
]
available_cols = [col for col in sample_cols if col in df_aligned.columns]
print(df_aligned[available_cols].head(10))

# %%
# Show alignment quality
print("\n=== Alignment Quality ===")
total_tokens = len(df_aligned)
aligned_tokens = len(df_aligned[df_aligned["word_index"] >= 0])
print(f"Total tokens: {total_tokens}")
print(f"Aligned tokens: {aligned_tokens}")
print(f"Alignment rate: {aligned_tokens / total_tokens:.2%}")

# %%
# Show POS distribution
print("\n=== POS Distribution ===")
pos_counts = df_aligned["aligned_upos"].value_counts().head(10)
print(pos_counts)
