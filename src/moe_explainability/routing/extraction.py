"""Core routing extraction functions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel

from moe_explainability.models.configs import ModelConfig


@dataclass
class Token:
    """A token with routing information."""

    id: int
    text: str
    position: int
    layer_logits: Dict[int, np.ndarray]  # {layer: raw_logits}

    def get_route_vector(self) -> np.ndarray:
        """Concatenate all layer logits into a single vector."""
        return np.concatenate(
            [self.layer_logits[layer] for layer in sorted(self.layer_logits.keys())]
        )


@dataclass
class AnnotatedToken:
    """A token with routing information and aligned linguistic data."""

    token: Token

    # Alignment info
    word_index: int = -1
    word_text: str = "<special>"
    word_position: int = -1

    # Word-level fields (flexible dictionary)
    word_fields: Dict[str, Any] = field(default_factory=dict)

    # Sentence-level context
    sentence_id: int = 0
    sentence_text: str = ""
    sentence_length: int = 0
    sentence_fields: Dict[str, Any] = field(default_factory=dict)


def extract_tokens(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    text: str,
    config: ModelConfig,
    add_special_tokens: bool = True,
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

        tokens.append(
            Token(
                id=token_id,
                text=token_text,
                position=token_pos,
                layer_logits=layer_logits,
            )
        )

    return tokens


def extract_tokens_batch(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    config: ModelConfig,
    add_special_tokens: bool = True,
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
