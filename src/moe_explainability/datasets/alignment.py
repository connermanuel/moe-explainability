"""Token alignment utilities."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TokenAlignment:
    """Result of token alignment between tokenizer and sentence tokens."""
    
    tokenizer_to_sentence: Dict[int, int]
    sentence_to_tokenizer: Dict[int, List[int]]
    unaligned_tokenizer: List[int]
    unaligned_sentence: List[int]


def align_tokens(
    text: str,
    tokenizer_tokens: List[str],
    sentence_tokens: List[str],
) -> TokenAlignment:
    """Align tokenizer tokens with sentence tokens.
    
    Args:
        text: Original text
        tokenizer_tokens: Tokens as produced by the tokenizer
        sentence_tokens: Original sentence tokens (e.g., from UD dataset)
        
    Returns:
        TokenAlignment object with mapping information
    """
    # Clean tokens for alignment
    clean_tokenizer_tokens = [token.replace(" ", "").lower() for token in tokenizer_tokens]
    clean_sentence_tokens = [token.replace(" ", "").lower() for token in sentence_tokens]
    
    # Get indices of tokens that actually appear in the text
    tokenizer_indices = _get_token_indices(text, clean_tokenizer_tokens)
    sentence_indices = _get_token_indices(text, clean_sentence_tokens)
    
    # Filter tokens to only those that appear in text
    filtered_tokenizer_tokens = [clean_tokenizer_tokens[i] for i in tokenizer_indices]
    filtered_sentence_tokens = [clean_sentence_tokens[i] for i in sentence_indices]
    
    # Create mapping between filtered tokens
    mapping = _map_tokens_to_sentence_tokens(filtered_tokenizer_tokens, filtered_sentence_tokens)
    
    # Convert back to original token indices
    tokenizer_to_sentence = {}
    sentence_to_tokenizer = {}
    
    for filtered_tok_idx, sent_idx in enumerate(mapping):
        if sent_idx >= 0:
            original_tok_idx = tokenizer_indices[filtered_tok_idx]
            original_sent_idx = sentence_indices[sent_idx]
            
            tokenizer_to_sentence[original_tok_idx] = original_sent_idx
            
            if original_sent_idx not in sentence_to_tokenizer:
                sentence_to_tokenizer[original_sent_idx] = []
            sentence_to_tokenizer[original_sent_idx].append(original_tok_idx)
    
    # Handle special tokens
    for tok_idx, token in enumerate(tokenizer_tokens):
        if tok_idx not in tokenizer_to_sentence:
            if token in ["</s>", "<s>", "<eos>", "<pad>", "<unk>"]:
                tokenizer_to_sentence[tok_idx] = -1  # Special token marker
    
    # Find unaligned tokens
    unaligned_tokenizer = [
        i for i in range(len(tokenizer_tokens)) if i not in tokenizer_to_sentence
    ]
    unaligned_sentence = [
        i for i in range(len(sentence_tokens)) if i not in sentence_to_tokenizer
    ]
    
    return TokenAlignment(
        tokenizer_to_sentence=tokenizer_to_sentence,
        sentence_to_tokenizer=sentence_to_tokenizer,
        unaligned_tokenizer=unaligned_tokenizer,
        unaligned_sentence=unaligned_sentence,
    )


def _get_token_indices(text: str, tokens: List[str]) -> List[int]:
    """Get indices of tokens that appear in the text."""
    target_text = text.replace(" ", "").lower()
    indices = []
    current_pos = 0
    
    for token_idx, token in enumerate(tokens):
        if not token:  # Skip empty tokens
            continue
        
        text_remainder = target_text[current_pos:]
        if text_remainder.startswith(token):
            indices.append(token_idx)
            current_pos += len(token)
        
        # Stop if we've consumed all the text
        if current_pos >= len(target_text):
            break
    
    return indices


def _map_tokens_to_sentence_tokens(
    tokenizer_tokens: List[str], sentence_tokens: List[str]
) -> List[int]:
    """Map tokenizer tokens to sentence token indices."""
    # Create character-level mapping of sentence tokens
    sent_token_char_map = []
    for sent_idx, sent_token in enumerate(sentence_tokens):
        sent_token_char_map.extend([sent_idx] * len(sent_token))
    
    # Map each tokenizer token
    mapping = []
    total_chars_consumed = 0
    
    for tokenizer_token in tokenizer_tokens:
        token_len = len(tokenizer_token)
        
        if total_chars_consumed + token_len <= len(sent_token_char_map):
            # Find the most common sentence token index for this character range
            char_indices = sent_token_char_map[
                total_chars_consumed : total_chars_consumed + token_len
            ]
            
            if char_indices:
                # Use the most common sentence token index
                most_common_idx = max(set(char_indices), key=char_indices.count)
                mapping.append(most_common_idx)
            else:
                mapping.append(-1)
            
            total_chars_consumed += token_len
        else:
            mapping.append(-1)
    
    return mapping