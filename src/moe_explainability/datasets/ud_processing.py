"""UD dataset processing with token-to-word alignment and linguistic features."""

from typing import Any, Callable, Dict, List

import pandas as pd
from datasets import Dataset

from ..routing.extraction import Token
from .alignment import align_tokens


def process_ud_sentence(
    sentence_data: Dict[str, Any],
    sentence_id: int,
    extract_tokens_fn: Callable[[str], List[Token]],
) -> pd.DataFrame:
    """Process a single UD sentence with routing extraction and linguistic alignment.

    Args:
        sentence_data: UD sentence data with text, tokens, linguistic features
        sentence_id: Unique sentence identifier
        extract_tokens_fn: Function to extract routing tokens from text

    Returns:
        DataFrame with one row per tokenizer token, enriched with linguistic info
    """
    # Extract routing information
    text = sentence_data["text"]
    routing_tokens = extract_tokens_fn(text)

    # Get UD tokens and linguistic features
    ud_tokens = sentence_data["tokens"]

    # Align tokenizer tokens with UD tokens
    tokenizer_token_texts = [t.text for t in routing_tokens]
    alignment = align_tokens(
        text=text, tokenizer_tokens=tokenizer_token_texts, sentence_tokens=ud_tokens
    )

    # Create enriched token data
    rows = []
    for i, token in enumerate(routing_tokens):
        # Get aligned UD token index
        ud_token_idx = alignment.tokenizer_to_sentence.get(i, -1)

        # Base token data
        row = {
            "token_id": token.id,
            "token_text": token.text,
            "token_position": token.position,
            "route_vector": token.get_route_vector(),
            # Sentence-level info
            "sentence_id": sentence_id,
            "sentence_text": text,
            "sentence_length": len(ud_tokens),
        }

        # Add layer-specific logits
        for layer, logits in token.layer_logits.items():
            row[f"layer_{layer}_logits"] = logits

        # Add word-level and linguistic information
        if ud_token_idx >= 0 and ud_token_idx < len(ud_tokens):
            # Successfully aligned to a UD token
            row.update(
                {
                    "word_index": ud_token_idx,
                    "word_text": ud_tokens[ud_token_idx],
                    "word_position": ud_token_idx,
                    "upos": sentence_data.get("upos", [None] * len(ud_tokens))[
                        ud_token_idx
                    ],
                    "xpos": sentence_data.get("xpos", [None] * len(ud_tokens))[
                        ud_token_idx
                    ],
                    "lemma": sentence_data.get("lemmas", [None] * len(ud_tokens))[
                        ud_token_idx
                    ],
                    "head": sentence_data.get("head", [None] * len(ud_tokens))[
                        ud_token_idx
                    ],
                    "deprel": sentence_data.get("deprel", [None] * len(ud_tokens))[
                        ud_token_idx
                    ],
                    "feats": sentence_data.get("feats", [None] * len(ud_tokens))[
                        ud_token_idx
                    ],
                }
            )
        else:
            # Special token or unaligned token
            row.update(
                {
                    "word_index": -1,
                    "word_text": "<special>",
                    "word_position": -1,
                    "upos": None,
                    "xpos": None,
                    "lemma": None,
                    "head": None,
                    "deprel": None,
                    "feats": None,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def process_ud_dataset(
    dataset: Dataset,
    extract_tokens_fn: Callable[[str], List[Token]],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Process entire UD dataset with routing and linguistic information.

    Args:
        dataset: UD dataset from HuggingFace
        extract_tokens_fn: Function to extract routing tokens from text
        show_progress: Whether to show progress bar

    Returns:
        Combined DataFrame with all sentences and tokens
    """
    from tqdm import tqdm

    all_dfs = []

    iterator = enumerate(dataset)
    if show_progress:
        iterator = tqdm(iterator, total=len(dataset), desc="Processing UD sentences")

    for sentence_id, sentence_data in iterator:
        try:
            df = process_ud_sentence(sentence_data, sentence_id, extract_tokens_fn)
            all_dfs.append(df)
        except Exception as e:
            if show_progress:
                print(f"Error processing sentence {sentence_id}: {e}")
            continue

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def analyze_alignment_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the quality of token-to-word alignment in the processed dataset.

    Args:
        df: Processed UD dataset DataFrame

    Returns:
        Dictionary with alignment quality metrics
    """
    total_tokens = len(df)
    aligned_tokens = len(df[df["word_index"] >= 0])
    special_tokens = len(df[df["word_index"] == -1])

    return {
        "total_tokens": total_tokens,
        "aligned_tokens": aligned_tokens,
        "special_tokens": special_tokens,
        "alignment_rate": aligned_tokens / total_tokens if total_tokens > 0 else 0,
        "special_token_rate": special_tokens / total_tokens if total_tokens > 0 else 0,
        "unique_sentences": df["sentence_id"].nunique(),
        "avg_tokens_per_sentence": df.groupby("sentence_id").size().mean(),
        "linguistic_features_available": {
            "upos": df["upos"].notna().sum(),
            "lemma": df["lemma"].notna().sum(),
            "deprel": df["deprel"].notna().sum(),
        },
    }


def filter_aligned_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include tokens that were successfully aligned to UD tokens.

    Args:
        df: Processed UD dataset DataFrame

    Returns:
        Filtered DataFrame with only aligned tokens
    """
    return df[df["word_index"] >= 0].copy()


def get_tokens_by_pos(df: pd.DataFrame, pos_tag: str) -> pd.DataFrame:
    """Get all tokens with a specific POS tag.

    Args:
        df: Processed UD dataset DataFrame
        pos_tag: POS tag to filter by (e.g., "NOUN", "VERB")

    Returns:
        Filtered DataFrame with only tokens of the specified POS tag
    """
    return df[df["upos"] == pos_tag].copy()


def get_sentence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics per sentence.

    Args:
        df: Processed UD dataset DataFrame

    Returns:
        DataFrame with one row per sentence and summary statistics
    """
    return (
        df.groupby("sentence_id")
        .agg(
            {
                "sentence_text": "first",
                "sentence_length": "first",
                "token_id": "count",  # Number of tokenizer tokens
                "word_index": lambda x: (x >= 0).sum(),  # Number of aligned tokens
                "upos": lambda x: x.notna().sum(),  # Number of tokens with POS tags
            }
        )
        .rename(
            columns={
                "token_id": "num_tokenizer_tokens",
                "word_index": "num_aligned_tokens",
                "upos": "num_pos_tagged_tokens",
            }
        )
        .reset_index()
    )
