"""Utility functions for routing analysis."""

import functools
import os
from typing import Any, Callable, Optional

import pandas as pd


def cache_to_file(func):
    """Decorator to cache DataFrame-returning functions to parquet files."""

    @functools.wraps(func)
    def wrapper(*args, filename: Optional[str] = None, compress: bool = True, **kwargs):
        if filename and os.path.exists(filename):
            print(f"Loading cached data from {filename}")
            return pd.read_parquet(filename)

        print(f"Computing {func.__name__}...")
        result = func(*args, **kwargs)

        if filename:
            print(f"Caching result to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            result.to_parquet(filename, compression="gzip" if compress else None)

        return result

    return wrapper


def get_device_info() -> dict:
    """Get information about available compute devices."""
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "default_device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Create a summary of DataFrame structure."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "null_counts": df.isnull().sum().to_dict(),
    }
