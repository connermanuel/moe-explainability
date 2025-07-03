"""Core routing extraction functionality for Switch Transformer models.

This module provides clean abstractions for extracting routing probabilities
from Switch Transformer models, independent of dataset-specific processing.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel


@dataclass
class RouterConfig:
    """Configuration for router extraction."""

    # Model layers to extract routing from (e.g., [1, 3, 5, 7, 9, 11] for switch-base-8)
    router_layers: List[int]

    # Whether to normalize routing probabilities with log_softmax
    normalize: bool = True

    # Device to run inference on
    device: str = "cuda"

    # Whether to extract hidden states as well
    extract_hidden_states: bool = False

    @classmethod
    def for_switch_base_8(
        cls, normalize: bool = True, device: str = "cuda"
    ) -> RouterConfig:
        """Create config for switch-base-8 model."""
        return cls(
            router_layers=[1, 3, 5, 7, 9, 11],
            normalize=normalize,
            device=device,
        )


@dataclass
class TokenRoute:
    """Single token routing information."""

    # Token information
    token_id: int
    token_text: str
    position: int

    # Routing probabilities per layer {layer_num: np.ndarray}
    layer_probs: Dict[int, np.ndarray]

    # Optional hidden states per layer {layer_num: np.ndarray}
    layer_hidden_states: Optional[Dict[int, np.ndarray]] = None

    def get_route_vector(self) -> np.ndarray:
        """Concatenate all layer probabilities into a single vector."""
        return np.concatenate(
            [self.layer_probs[layer] for layer in sorted(self.layer_probs.keys())]
        )


@dataclass
class SequenceRoutes:
    """Routing information for a complete sequence."""

    # Original text
    text: str

    # List of TokenRoute objects
    token_routes: List[TokenRoute]

    # Configuration used
    config: RouterConfig

    def __len__(self) -> int:
        return len(self.token_routes)

    def __iter__(self):
        return iter(self.token_routes)

    def get_route_matrix(self) -> np.ndarray:
        """Get routing probabilities as a matrix [n_tokens, n_features]."""
        return np.stack([token.get_route_vector() for token in self.token_routes])


class RouterExtractor:
    """Core class for extracting routing probabilities from Switch Transformer models."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        config: RouterConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.to(config.device)

    def extract_routes(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> SequenceRoutes:
        """Extract routing probabilities for a single text sequence.

        Args:
            text: Input text to process
            add_special_tokens: Whether to add special tokens (BOS/EOS)

        Returns:
            SequenceRoutes object containing all routing information
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        input_ids = inputs.input_ids.to(self.config.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=self.config.extract_hidden_states,
                output_router_logits=True,
            )

        # Extract routing information
        token_routes = []
        sequence_length = input_ids.shape[1]

        for token_pos in range(sequence_length):
            token_id = input_ids[0, token_pos].item()
            token_text = self.tokenizer.decode(token_id)

            # Extract routing probabilities for each configured layer
            layer_probs = {}
            layer_hidden_states = {} if self.config.extract_hidden_states else None

            for layer_num in self.config.router_layers:
                if layer_num < len(outputs.router_probs):
                    # Extract routing logits for this token and layer
                    router_logits = outputs.router_probs[layer_num][0][0][token_pos]

                    # Apply normalization if requested
                    if self.config.normalize:
                        probs = F.log_softmax(router_logits, dim=0)
                    else:
                        probs = router_logits

                    layer_probs[layer_num] = probs.cpu().numpy()

                # Extract hidden states if requested
                if (
                    self.config.extract_hidden_states
                    and hasattr(outputs, "hidden_states")
                    and outputs.hidden_states is not None
                    and layer_num < len(outputs.hidden_states)
                ):
                    hidden_state = outputs.hidden_states[layer_num][0][token_pos]
                    layer_hidden_states[layer_num] = hidden_state.cpu().numpy()

            token_routes.append(
                TokenRoute(
                    token_id=token_id,
                    token_text=token_text,
                    position=token_pos,
                    layer_probs=layer_probs,
                    layer_hidden_states=layer_hidden_states,
                )
            )

        return SequenceRoutes(
            text=text,
            token_routes=token_routes,
            config=self.config,
        )

    def extract_routes_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[SequenceRoutes]:
        """Extract routing probabilities for multiple texts.

        Note: This processes each text individually to handle variable lengths.
        For better batching, consider padding/truncation strategies.
        """
        return [
            self.extract_routes(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, PreTrainedModel]:
    """Load model and tokenizer with proper device handling."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Import the specific model class for better type hints
    if "switch" in model_name.lower():
        from transformers import SwitchTransformersEncoderModel

        model = SwitchTransformersEncoderModel.from_pretrained(model_name)
    else:
        # Fallback for other model types
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name)

    model.to(device)
    return tokenizer, model
