"""Distance functions for routing analysis."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal, List, Dict
from scipy.spatial.distance import jensenshannon

from ..routing.extraction import Token, AnnotatedToken


def token_routing_distance(
    token1: Token,
    token2: Token,
    method: Literal["layer_mean", "concatenated"] = "layer_mean",
    normalize: bool = True,
) -> float:
    """Compute routing distance between two tokens.
    
    Args:
        token1: First token
        token2: Second token
        method: Distance computation method
            - "layer_mean": Compute JS distance per layer, then average
            - "concatenated": Concatenate all layers, then compute JS distance
        normalize: Whether to normalize logits to probabilities using softmax
        
    Returns:
        Routing distance between tokens
    """
    # Get common layers
    layers1 = set(token1.layer_logits.keys())
    layers2 = set(token2.layer_logits.keys())
    common_layers = sorted(layers1.intersection(layers2))
    
    if not common_layers:
        raise ValueError("No common layers found between tokens")
    
    if method == "layer_mean":
        # Compute distance per layer, then average
        distances = []
        for layer in common_layers:
            logits1 = token1.layer_logits[layer]
            logits2 = token2.layer_logits[layer]
            
            if normalize:
                logits1 = F.softmax(torch.from_numpy(logits1), dim=-1).numpy()
                logits2 = F.softmax(torch.from_numpy(logits2), dim=-1).numpy()
            
            distance = jensenshannon(logits1, logits2)
            distances.append(distance)
        
        return np.mean(distances)
    
    elif method == "concatenated":
        # Concatenate all layers, then compute distance
        logits1_list = []
        logits2_list = []
        
        for layer in common_layers:
            logits1_list.append(token1.layer_logits[layer])
            logits2_list.append(token2.layer_logits[layer])
        
        concat_logits1 = np.concatenate(logits1_list)
        concat_logits2 = np.concatenate(logits2_list)
        
        if normalize:
            concat_logits1 = F.softmax(torch.from_numpy(concat_logits1), dim=-1).numpy()
            concat_logits2 = F.softmax(torch.from_numpy(concat_logits2), dim=-1).numpy()
        
        return jensenshannon(concat_logits1, concat_logits2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def group_tokens_by_text(tokens: List[AnnotatedToken]) -> Dict[str, List[AnnotatedToken]]:
    """Group tokens by their original text.
    
    Args:
        tokens: List of AnnotatedToken objects
        
    Returns:
        Dictionary mapping text to list of tokens from that text
    """
    groups = {}
    for token in tokens:
        text = token.sentence_text
        if text not in groups:
            groups[text] = []
        groups[text].append(token)
    return groups


def aggregate_token_group(
    tokens: List[Token],
    method: Literal["first", "mean"] = "mean",
    normalize: bool = True,
) -> Token:
    """Aggregate a group of tokens into a single representative token.
    
    Args:
        tokens: List of tokens to aggregate
        method: Aggregation method
            - "first": Use the first token
            - "mean": Take mean of normalized logits across all tokens
        normalize: Whether to normalize logits before aggregation (for mean method)
        
    Returns:
        Single Token representing the group
    """
    if not tokens:
        raise ValueError("Cannot aggregate empty token list")
    
    if method == "first":
        return tokens[0]
    
    elif method == "mean":
        # Get common layers across all tokens
        all_layers = [set(token.layer_logits.keys()) for token in tokens]
        common_layers = sorted(set.intersection(*all_layers))
        
        if not common_layers:
            raise ValueError("No common layers found across tokens")
        
        # Aggregate logits per layer
        aggregated_logits = {}
        for layer in common_layers:
            layer_logits = []
            for token in tokens:
                logits = token.layer_logits[layer]
                if normalize:
                    logits = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
                layer_logits.append(logits)
            
            # Mean across tokens
            aggregated_logits[layer] = np.mean(layer_logits, axis=0)
        
        # Create representative token
        return Token(
            id=-1,  # Placeholder ID for aggregated token
            text=f"<aggregated_{len(tokens)}_tokens>",
            position=-1,
            layer_logits=aggregated_logits,
        )
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def token_group_distance(
    group1: List[Token],
    group2: List[Token],
    aggregation_method: Literal["first", "mean"] = "mean",
    distance_method: Literal["layer_mean", "concatenated"] = "layer_mean",
    normalize: bool = True,
) -> float:
    """Compute distance between two groups of tokens.
    
    Args:
        group1: First group of tokens
        group2: Second group of tokens
        aggregation_method: How to aggregate tokens within each group
        distance_method: How to compute distance between aggregated tokens
        normalize: Whether to normalize logits
        
    Returns:
        Distance between the two token groups
    """
    # Aggregate each group
    agg_token1 = aggregate_token_group(group1, aggregation_method, normalize)
    agg_token2 = aggregate_token_group(group2, aggregation_method, normalize)
    
    # Compute distance between aggregated tokens
    # Note: We set normalize=False since we already normalized during aggregation if needed
    return token_routing_distance(
        agg_token1, agg_token2, 
        method=distance_method, 
        normalize=False if aggregation_method == "mean" and normalize else normalize
    )