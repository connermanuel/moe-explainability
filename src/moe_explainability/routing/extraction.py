"""Core routing extraction functions."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer

from ..models.configs import ModelConfig


@dataclass
class Token:
    """A token with routing information."""
    
    id: int
    text: str
    position: int
    layer_logits: Dict[int, np.ndarray]  # {layer: raw_logits}
    
    def get_route_vector(self) -> np.ndarray:
        """Concatenate all layer logits into a single vector."""
        return np.concatenate([
            self.layer_logits[layer] 
            for layer in sorted(self.layer_logits.keys())
        ])
    
    def get_normalized_logits(self, layer: int, method: str = "softmax") -> np.ndarray:
        """Get normalized probabilities for a specific layer.
        
        Args:
            layer: Layer number
            method: Normalization method ("softmax" or "log_softmax")
        """
        if layer not in self.layer_logits:
            raise ValueError(f"Layer {layer} not available")
        
        logits = torch.from_numpy(self.layer_logits[layer])
        if method == "softmax":
            return F.softmax(logits, dim=0).numpy()
        elif method == "log_softmax":
            return F.log_softmax(logits, dim=0).numpy()
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def extract_tokens(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer, 
    text: str,
    config: ModelConfig,
    add_special_tokens: bool = True
) -> List[Token]:
    """Extract routing information for a single text.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        text: Input text
        config: Model configuration
        add_special_tokens: Whether to add special tokens (BOS/EOS)
    
    Returns:
        List of Token objects with routing information
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = inputs.input_ids.to(config.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_router_logits=True,
        )
    
    # Extract tokens
    tokens = []
    sequence_length = input_ids.shape[1]
    
    for token_pos in range(sequence_length):
        token_id = input_ids[0, token_pos].item()
        token_text = tokenizer.decode(token_id)
        
        layer_logits = {}
        for layer_num in config.layers:
            if layer_num < len(outputs.router_probs):
                router_logits = outputs.router_probs[layer_num][0][0][token_pos]
                layer_logits[layer_num] = router_logits.cpu().numpy()
        
        tokens.append(Token(
            id=token_id,
            text=token_text,
            position=token_pos,
            layer_logits=layer_logits,
        ))
    
    return tokens


def extract_tokens_batch(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str], 
    config: ModelConfig,
    add_special_tokens: bool = True
) -> List[List[Token]]:
    """Extract routing information for multiple texts.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        texts: List of input texts
        config: Model configuration
        add_special_tokens: Whether to add special tokens
    
    Returns:
        List of lists of Token objects
    """
    return [
        extract_tokens(model, tokenizer, text, config, add_special_tokens)
        for text in texts
    ]