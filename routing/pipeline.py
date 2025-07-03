"""Main pipeline for routing extraction across different datasets.

This module provides high-level functions that orchestrate the routing
extraction process using the modular components (core, adapters, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset

from .adapters import DatasetAdapter, FlexibleAdapter, UDAdapter, WordSimAdapter
from .core import RouterConfig, RouterExtractor, load_model_and_tokenizer
from .utils import cache_to_file, get_device_info


class RoutingPipeline:
    """High-level pipeline for routing extraction."""

    def __init__(
        self,
        model_name: str,
        config: Optional[RouterConfig] = None,
        device: Optional[str] = None,
    ):
        """Initialize the routing pipeline.

        Args:
            model_name: Name of the model to load
            config: Router configuration (will use defaults if None)
            device: Device to use (will auto-detect if None)
        """
        self.model_name = model_name

        # Set up device
        if device is None:
            device_info = get_device_info()
            device = device_info["default_device"]

        # Set up configuration
        if config is None:
            config = RouterConfig.for_switch_base_8(device=device)
        else:
            config.device = device

        self.config = config

        # Load model and tokenizer
        print(f"Loading model {model_name} on {device}")
        tokenizer, model = load_model_and_tokenizer(model_name, device)

        # Create router extractor
        self.router_extractor = RouterExtractor(model, tokenizer, config)

    def extract_routes_wordsim(
        self,
        dataset: Union[Dataset, List[Dict[str, Any]]],
        cache_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extract routes for WordSim-style datasets."""
        adapter = WordSimAdapter(self.router_extractor)

        if cache_file:

            @cache_to_file
            def _extract():
                return adapter.process_dataset(dataset)

            return _extract(filename=cache_file)
        else:
            return adapter.process_dataset(dataset)

    def extract_routes_ud(
        self,
        dataset: Union[Dataset, List[Dict[str, Any]]],
        cache_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extract routes for Universal Dependencies datasets."""
        adapter = UDAdapter(self.router_extractor)

        if cache_file:

            @cache_to_file
            def _extract():
                return adapter.process_dataset(dataset)

            return _extract(filename=cache_file)
        else:
            return adapter.process_dataset(dataset)

    def extract_routes_flexible(
        self,
        dataset: Union[Dataset, List[Dict[str, Any]]],
        text_field: str = "text",
        extra_fields: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        cache_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extract routes with flexible configuration."""
        adapter = FlexibleAdapter(
            self.router_extractor,
            text_field=text_field,
            extra_fields=extra_fields or [],
            add_special_tokens=add_special_tokens,
        )

        if cache_file:

            @cache_to_file
            def _extract():
                return adapter.process_dataset(dataset)

            return _extract(filename=cache_file)
        else:
            return adapter.process_dataset(dataset)

    def extract_single_text(self, text: str) -> Dict[str, Any]:
        """Extract routes for a single text (useful for debugging/exploration)."""
        sequence_routes = self.router_extractor.extract_routes(text)

        return {
            "text": text,
            "num_tokens": len(sequence_routes),
            "tokens": [tr.token_text for tr in sequence_routes.token_routes],
            "route_matrix": sequence_routes.get_route_matrix(),
            "sequence_routes": sequence_routes,  # Full object for advanced use
        }


# Convenience functions for common use cases
def extract_wordsim_routes(
    model_name: str = "google/switch-base-8",
    dataset: Optional[Union[Dataset, List[Dict[str, Any]]]] = None,
    word_list: Optional[List[str]] = None,
    normalize: bool = True,
    cache_file: Optional[str] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Extract routing probabilities for a list of words.

    Args:
        model_name: Name of the model to use
        dataset: Dataset with word information (alternative to word_list)
        word_list: List of words to process (alternative to dataset)
        normalize: Whether to normalize routing probabilities
        cache_file: Optional cache file path
        device: Device to use for computation

    Returns:
        DataFrame with routing information for each word
    """
    # Prepare dataset
    if dataset is None and word_list is not None:
        dataset = [{"text": word} for word in word_list]
    elif dataset is None:
        raise ValueError("Must provide either dataset or word_list")

    # Set up configuration
    config = RouterConfig.for_switch_base_8(
        normalize=normalize, device=device or "cuda"
    )

    # Create pipeline and extract routes
    pipeline = RoutingPipeline(model_name, config, device)
    return pipeline.extract_routes_wordsim(dataset, cache_file)


def extract_ud_routes(
    model_name: str = "google/switch-base-8",
    dataset: Union[Dataset, List[Dict[str, Any]]] = None,
    dataset_split: str = "train",
    normalize: bool = True,
    cache_file: Optional[str] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Extract routing probabilities for Universal Dependencies dataset.

    Args:
        model_name: Name of the model to use
        dataset: UD dataset (if None, will load from HuggingFace)
        dataset_split: Which split to use if loading from HuggingFace
        normalize: Whether to normalize routing probabilities
        cache_file: Optional cache file path
        device: Device to use for computation

    Returns:
        DataFrame with routing information for each token
    """
    # Load dataset if not provided
    if dataset is None:
        from datasets import load_dataset

        dataset = load_dataset("universal_dependencies", "en_ewt")[dataset_split]

    # Set up configuration
    config = RouterConfig.for_switch_base_8(
        normalize=normalize, device=device or "cuda"
    )

    # Create pipeline and extract routes
    pipeline = RoutingPipeline(model_name, config, device)
    return pipeline.extract_routes_ud(dataset, cache_file)


def quick_route_analysis(
    text: str,
    model_name: str = "google/switch-base-8",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Quick routing analysis for a single text (useful for exploration).

    Args:
        text: Text to analyze
        model_name: Name of the model to use
        normalize: Whether to normalize routing probabilities

    Returns:
        Dictionary with routing analysis results
    """
    config = RouterConfig.for_switch_base_8(normalize=normalize)
    pipeline = RoutingPipeline(model_name, config)

    result = pipeline.extract_single_text(text)

    # Add some basic analysis
    route_matrix = result["route_matrix"]
    result["route_stats"] = {
        "mean_entropy": -route_matrix.sum(axis=1).mean(),  # Approximate entropy
        "max_probability_per_token": route_matrix.max(axis=1).tolist(),
        "dominant_experts_per_token": route_matrix.argmax(axis=1).tolist(),
    }

    return result
