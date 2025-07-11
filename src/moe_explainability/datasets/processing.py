"""Dataset processing utilities."""

from typing import Any, Callable

import pandas as pd

from moe_explainability.routing.extraction import Token


def tokens_to_dataframe(
    tokens: list[Token], extra_data: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Convert Token objects to analysis-ready DataFrame.

    Args:
        tokens: list of Token objects
        extra_data: Optional extra data to add to each row

    Returns:
        DataFrame with token and routing information
    """
    rows = []

    for token in tokens:
        row = {
            "token_id": token.id,
            "token_text": token.text,
            "position": token.position,
            "route_vector": token.get_route_vector(),
        }

        # Add layer-specific logits
        for layer, logits in token.layer_logits.items():
            row[f"layer_{layer}_logits"] = logits

        # Add extra data if provided
        if extra_data:
            row.update(extra_data)

        rows.append(row)

    return pd.DataFrame(rows)


def process_texts(
    texts: list[str],
    extract_fn: Callable[[str], list[Token]],
    extra_data_fn: Callable[[str, int], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Process multiple texts with a generic extraction function.

    Args:
        texts: list of texts to process
        extract_fn: Function that takes text and returns Token objects
        extra_data_fn: Optional function to generate extra data per text

    Returns:
        Combined DataFrame with all tokens
    """
    all_dfs = []

    for i, text in enumerate(texts):
        tokens = extract_fn(text)

        # Get extra data if function provided
        extra_data = extra_data_fn(text, i) if extra_data_fn else {"text_id": i}

        df = tokens_to_dataframe(tokens, extra_data)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def filter_special_tokens(
    df: pd.DataFrame, exclude_token_ids: list[int] | None = None
) -> pd.DataFrame:
    """Filter out special tokens from DataFrame.

    Args:
        df: DataFrame with token data
        exclude_token_ids: Token IDs to exclude (default: common special tokens)

    Returns:
        Filtered DataFrame
    """
    if exclude_token_ids is None:
        exclude_token_ids = [0, 1, 2, 3]  # Common special token IDs

    return df[~df["token_id"].isin(exclude_token_ids)]


def aggregate_by_token_id(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DataFrame by token_id (useful for analyzing token patterns).

    Args:
        df: DataFrame with token data

    Returns:
        Aggregated DataFrame with mean routing vectors per token_id
    """
    # Define aggregation strategies
    agg_dict = {
        "token_text": "first",
        "route_vector": lambda x: x.apply(lambda arr: arr).mean(),  # Mean of arrays
        "count": "size",
    }

    # Add aggregation for layer-specific logits
    layer_cols = [
        col
        for col in df.columns
        if col.startswith("layer_") and col.endswith("_logits")
    ]
    for col in layer_cols:
        agg_dict[col] = lambda x: x.apply(lambda arr: arr).mean()

    result = df.groupby("token_id").agg(agg_dict).reset_index()
    result = result.rename(columns={"count": "occurrence_count"})

    return result


def process_structured_data(
    data: list[dict[str, Any]], 
    text_field: str,
    extract_fn: Callable[[str], list[Token]],
    extra_fields: list[str] | None = None
) -> pd.DataFrame:
    """Process structured data (like UD) using general functions.
    
    Args:
        data: List of dictionaries with structured data
        text_field: Key containing the text to process
        extract_fn: Function to extract tokens from text
        extra_fields: Additional fields to include from each data item
        
    Returns:
        DataFrame with routing and structured data
    """
    texts = [item[text_field] for item in data]
    
    def extra_data_fn(text: str, text_id: int) -> dict[str, Any]:
        """Extract extra data from structured item."""
        item = data[text_id]
        extra_data = {"item_id": text_id, "text": text}
        
        if extra_fields:
            for field in extra_fields:
                if field in item:
                    extra_data[field] = item[field]
        
        return extra_data
    
    return process_texts(texts, extract_fn, extra_data_fn)


def add_post_processing(df: pd.DataFrame, post_process_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Apply post-processing function to DataFrame.
    
    Args:
        df: Input DataFrame
        post_process_fn: Function to apply post-processing
        
    Returns:
        Post-processed DataFrame
    """
    return post_process_fn(df)
