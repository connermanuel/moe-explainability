"""Dataset processing utilities."""

from typing import Any, Callable

import pandas as pd

from moe_explainability.routing.extraction import Token, AnnotatedToken
from moe_explainability.datasets.configs import DatasetConfig
from moe_explainability.datasets.alignment import align_tokens


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


def extract_and_align_routing_tokens(
    data: list[dict[str, Any]],
    config: DatasetConfig,
    extract_tokens_fn: Callable[[str], list[Token]],
) -> list[AnnotatedToken]:
    """Extract routing tokens and align them with linguistic data.

    Args:
        data: List of dictionaries with structured data
        config: Dataset configuration specifying field mappings
        extract_tokens_fn: Function to extract tokens from text

    Returns:
        List of AnnotatedToken objects with routing and linguistic data aligned
    """
    all_annotated_tokens = []
    
    for sentence_id, item in enumerate(data):
        # Step 1: Extract tokens with routing information
        text = item[config.text_field]
        tokens = extract_tokens_fn(text)
        
        # Step 2: Align tokens with words using existing alignment logic
        tokenizer_token_texts = [t.text for t in tokens]
        sentence_tokens = item[config.words_field]
        alignment = align_tokens(text, tokenizer_token_texts, sentence_tokens)
        
        # Step 3: Create annotated tokens for this sentence
        annotated_tokens = create_annotated_tokens(
            tokens=tokens,
            alignment=alignment,
            word_data={field: item[field] for field in config.word_fields},
            sentence_data={field: item[field] for field in config.sentence_fields},
            sentence_id=sentence_id,
            sentence_text=text,
        )
        all_annotated_tokens.extend(annotated_tokens)
    
    return all_annotated_tokens


def create_annotated_tokens(
    tokens: list[Token],
    alignment,
    word_data: dict[str, Any],
    sentence_data: dict[str, Any],
    sentence_id: int,
    sentence_text: str,
) -> list[AnnotatedToken]:
    """Create AnnotatedToken objects combining token, word, and sentence data.
    
    Args:
        tokens: List of Token objects with routing information
        alignment: TokenAlignment object from align_tokens
        word_data: Dictionary of word-level fields
        sentence_data: Dictionary of sentence-level fields
        sentence_id: Unique sentence identifier
        sentence_text: Original sentence text
        
    Returns:
        List of AnnotatedToken objects with aligned data
    """
    annotated_tokens = []
    
    for i, token in enumerate(tokens):
        # Get aligned word index
        word_idx = alignment.tokenizer_to_sentence.get(i, -1)
        
        # Prepare word-level fields
        word_fields = {}
        word_text = "<special>"
        word_position = -1
        
        if word_idx >= 0 and word_idx < len(word_data.get("tokens", [])):
            # Successfully aligned to a word
            word_text = word_data["tokens"][word_idx] if "tokens" in word_data else "<special>"
            word_position = word_idx
            
            # Add other word-level fields
            for field, values in word_data.items():
                if field != "tokens" and word_idx < len(values):
                    word_fields[field] = values[word_idx]
        
        # Create AnnotatedToken
        annotated_token = AnnotatedToken(
            token=token,
            word_index=word_idx,
            word_text=word_text,
            word_position=word_position,
            word_fields=word_fields,
            sentence_id=sentence_id,
            sentence_text=sentence_text,
            sentence_length=len(word_data.get("tokens", [])),
            sentence_fields=sentence_data.copy(),
        )
        
        annotated_tokens.append(annotated_token)
    
    return annotated_tokens


def annotated_tokens_to_dataframe(tokens: list[AnnotatedToken]) -> pd.DataFrame:
    """Convert AnnotatedToken objects to DataFrame for analysis.
    
    Args:
        tokens: List of AnnotatedToken objects
        
    Returns:
        DataFrame with one row per token, compatible with existing analysis code
    """
    rows = []
    
    for token in tokens:
        # Base token and sentence data
        row = {
            "token_id": token.token.id,
            "token_text": token.token.text,
            "token_position": token.token.position,
            "route_vector": token.token.get_route_vector(),
            "sentence_id": token.sentence_id,
            "sentence_text": token.sentence_text,
            "sentence_length": token.sentence_length,
        }
        
        # Add layer-specific logits
        for layer, logits in token.token.layer_logits.items():
            row[f"layer_{layer}_logits"] = logits
        
        # Add word alignment info
        row.update({
            "word_index": token.word_index,
            "word_text": token.word_text,
            "word_position": token.word_position,
        })
        
        # Add word-level fields
        for field, value in token.word_fields.items():
            row[field] = value
        
        # Add sentence-level fields
        for field, value in token.sentence_fields.items():
            row[f"sentence_{field}"] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)
