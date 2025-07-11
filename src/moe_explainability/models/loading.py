"""Model loading utilities."""

from typing import Tuple, Callable

import torch
from transformers import AutoTokenizer, PreTrainedModel

from .configs import ModelConfig


def load_model_and_tokenizer(
    config: ModelConfig,
) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """Load model and tokenizer from configuration."""
    tokenizer = AutoTokenizer.from_pretrained(config.name)

    # Load model based on type
    if "switch" in config.name.lower():
        from transformers import SwitchTransformersEncoderModel

        model = SwitchTransformersEncoderModel.from_pretrained(config.name)
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(config.name)

    # Move to device
    model.to(config.device)
    model.eval()  # Set to evaluation mode

    return model, tokenizer


def get_device_info() -> dict:
    """Get information about available compute devices."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "default_device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def create_extraction_fn(model_config: ModelConfig) -> Callable[[str], list]:
    """Create a function that extracts tokens from text using the specified model.
    
    Args:
        model_config: Configuration for the model to load
        
    Returns:
        Function that takes text and returns list of Token objects
    """
    from ..routing.extraction import extract_tokens
    
    model, tokenizer = load_model_and_tokenizer(model_config)

    def extract_fn(text: str):
        """Extract tokens from text."""
        return extract_tokens(model, tokenizer, text, model_config)

    return extract_fn
