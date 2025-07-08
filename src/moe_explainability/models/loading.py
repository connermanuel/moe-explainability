"""Model loading utilities."""

from typing import Tuple
import torch
from transformers import AutoTokenizer, PreTrainedModel

from .configs import ModelConfig


def load_model(config: ModelConfig) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """Load model and tokenizer from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
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
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "default_device": "cuda" if torch.cuda.is_available() else "cpu",
    }