"""Dataset adapters for different types of routing analysis.

This module provides adapters that handle dataset-specific processing
while using the core routing extraction functionality.
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from .alignment import TokenAligner
from .core import RouterExtractor, SequenceRoutes, TokenRoute


@dataclass
class ProcessedToken:
    """A token with both routing and dataset-specific information."""

    # Core routing information
    token_route: TokenRoute

    # Dataset-specific fields (flexible)
    extra_fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "token_id": self.token_route.token_id,
            "token_text": self.token_route.token_text,
            "position": self.token_route.position,
            "route_vector": self.token_route.get_route_vector(),
        }

        # Add layer-specific probabilities
        for layer, probs in self.token_route.layer_probs.items():
            result[f"expert_{layer}"] = probs

        # Add hidden states if available
        if self.token_route.layer_hidden_states:
            for layer, hidden in self.token_route.layer_hidden_states.items():
                result[f"hidden_state_{layer}"] = hidden

        # Add extra fields
        result.update(self.extra_fields)

        return result


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    def __init__(self, router_extractor: RouterExtractor):
        self.router_extractor = router_extractor

    @abstractmethod
    def process_item(self, item: Dict[str, Any], item_id: int) -> List[ProcessedToken]:
        """Process a single dataset item and return processed tokens."""
        pass

    def process_dataset(
        self,
        dataset: tp.Union[Dataset, List[Dict[str, Any]]],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Process an entire dataset and return a DataFrame."""
        all_tokens = []

        iterator = enumerate(dataset)
        if show_progress:
            iterator = tqdm(iterator, total=len(dataset), desc="Processing dataset")

        for item_id, item in iterator:
            processed_tokens = self.process_item(item, item_id)
            all_tokens.extend(processed_tokens)

        # Convert to DataFrame
        if not all_tokens:
            return pd.DataFrame()

        rows = [token.to_dict() for token in all_tokens]
        return pd.DataFrame(rows)


class WordSimAdapter(DatasetAdapter):
    """Adapter for WordSim-style datasets (simple word lists)."""

    def process_item(self, item: Dict[str, Any], item_id: int) -> List[ProcessedToken]:
        """Process a single word item.

        Expected item format:
        {
            "text": "word"  # or other field containing the word
        }
        """
        # Extract the word text
        if "text" in item:
            word = item["text"]
        elif "word" in item:
            word = item["word"]
        else:
            # Try to find any string field
            word = next((v for v in item.values() if isinstance(v, str)), None)
            if word is None:
                raise ValueError(f"Could not find text field in item: {item}")

        # Extract routing information
        sequence_routes = self.router_extractor.extract_routes(
            word,
            add_special_tokens=False,  # For single words, usually don't want special tokens
        )

        # Convert to ProcessedTokens
        processed_tokens = []
        for token_route in sequence_routes.token_routes:
            extra_fields = {
                "item_id": item_id,
                "original_word": word,
            }
            # Add any other fields from the original item
            for key, value in item.items():
                if key not in ["text", "word"] and not key.startswith("_"):
                    extra_fields[f"item_{key}"] = value

            processed_tokens.append(
                ProcessedToken(
                    token_route=token_route,
                    extra_fields=extra_fields,
                )
            )

        return processed_tokens


class UDAdapter(DatasetAdapter):
    """Adapter for Universal Dependencies dataset with linguistic annotations."""

    def __init__(self, router_extractor: RouterExtractor):
        super().__init__(router_extractor)
        self.token_aligner = TokenAligner()

    def process_item(self, item: Dict[str, Any], item_id: int) -> List[ProcessedToken]:
        """Process a single UD sentence item.

        Expected item format:
        {
            "text": "The sentence text",
            "tokens": ["The", "sentence", "text"],
            "upos": ["DET", "NOUN", "NOUN"],
            "xpos": ["DT", "NN", "NN"],
            # ... other linguistic features
        }
        """
        text = item["text"]
        sent_tokens = item.get("tokens", [])

        # Extract routing information
        sequence_routes = self.router_extractor.extract_routes(text)

        # Align tokenizer tokens with sentence tokens
        tokenizer_tokens = [tr.token_text for tr in sequence_routes.token_routes]
        alignment = self.token_aligner.align_tokens(
            text=text,
            tokenizer_tokens=tokenizer_tokens,
            sentence_tokens=sent_tokens,
        )

        # Create ProcessedTokens with linguistic annotations
        processed_tokens = []
        for idx, token_route in enumerate(sequence_routes.token_routes):
            sent_token_idx = alignment.tokenizer_to_sentence.get(idx, -1)

            extra_fields = {
                "sentence_id": item_id,
                "decoded_token_position": idx,
                "sent_token_position": sent_token_idx,
            }

            # Add linguistic annotations if available
            if sent_token_idx >= 0 and sent_token_idx < len(sent_tokens):
                extra_fields["sent_token"] = sent_tokens[sent_token_idx]

                # Add all linguistic features
                for feature_name, feature_values in item.items():
                    if (
                        isinstance(feature_values, list)
                        and len(feature_values) == len(sent_tokens)
                        and feature_name not in ["text", "tokens"]
                    ):
                        try:
                            extra_fields[feature_name] = feature_values[sent_token_idx]
                        except IndexError:
                            extra_fields[feature_name] = None
            else:
                # Special tokens (like EOS)
                extra_fields["sent_token"] = "<special>"
                for feature_name, feature_values in item.items():
                    if isinstance(feature_values, list) and feature_name not in [
                        "text",
                        "tokens",
                    ]:
                        extra_fields[feature_name] = None

            processed_tokens.append(
                ProcessedToken(
                    token_route=token_route,
                    extra_fields=extra_fields,
                )
            )

        return processed_tokens


class FlexibleAdapter(DatasetAdapter):
    """Flexible adapter that can handle various dataset formats."""

    def __init__(
        self,
        router_extractor: RouterExtractor,
        text_field: str = "text",
        add_special_tokens: bool = True,
        extra_fields: Optional[List[str]] = None,
    ):
        super().__init__(router_extractor)
        self.text_field = text_field
        self.add_special_tokens = add_special_tokens
        self.extra_fields = extra_fields or []

    def process_item(self, item: Dict[str, Any], item_id: int) -> List[ProcessedToken]:
        """Process a single item with flexible field mapping."""
        text = item[self.text_field]

        # Extract routing information
        sequence_routes = self.router_extractor.extract_routes(
            text, add_special_tokens=self.add_special_tokens
        )

        # Create ProcessedTokens
        processed_tokens = []
        for token_route in sequence_routes.token_routes:
            extra_fields = {"item_id": item_id}

            # Add requested extra fields
            for field_name in self.extra_fields:
                if field_name in item:
                    extra_fields[field_name] = item[field_name]

            processed_tokens.append(
                ProcessedToken(
                    token_route=token_route,
                    extra_fields=extra_fields,
                )
            )

        return processed_tokens


# Utility functions for common post-processing
def collapse_by_token_id(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse DataFrame by token_id, aggregating features."""

    def safe_mode(series):
        """Get mode, handling cases where no mode exists."""
        try:
            return series.value_counts().idxmax()
        except ValueError:
            return None

    # Define aggregation strategies
    agg_dict = {
        "token_text": "first",
        "route_vector": "mean",
        "count": "size",  # This will count occurrences
    }

    # Add aggregation for expert columns
    expert_cols = [col for col in df.columns if col.startswith("expert_")]
    for col in expert_cols:
        agg_dict[col] = "mean"

    # Add aggregation for categorical columns (use mode)
    categorical_cols = ["upos", "xpos", "sent_token"]
    for col in categorical_cols:
        if col in df.columns:
            agg_dict[col] = safe_mode

    # Add aggregation for other numeric columns
    numeric_cols = [col for col in df.columns if col.startswith("hidden_state_")]
    for col in numeric_cols:
        agg_dict[col] = "mean"

    result = df.groupby("token_id").agg(agg_dict).reset_index()
    result = result.rename(columns={"count": "occurrence_count"})

    return result


def filter_common_tokens(
    df: pd.DataFrame, exclude_token_ids: List[int] = None
) -> pd.DataFrame:
    """Filter out common tokens like BOS, EOS, PAD."""
    if exclude_token_ids is None:
        exclude_token_ids = [0, 1, 2, 3]  # Common special token IDs

    return df[~df["token_id"].isin(exclude_token_ids)]
