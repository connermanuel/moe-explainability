"""Modular routing extraction package for Switch Transformer models."""

# Core functionality
from .core import RouterConfig, RouterExtractor, load_model_and_tokenizer

# High-level pipeline
from .pipeline import (
    RoutingPipeline,
    extract_ud_routes,
    extract_wordsim_routes,
    quick_route_analysis,
)

# Utilities
from .utils import cache_to_file, get_device_info

__version__ = "0.1.0"
