"""Token alignment utilities for matching tokenizer tokens with sentence tokens.

This module handles the complex task of aligning tokens produced by a tokenizer
with the original word tokens in a sentence, which is particularly important
for linguistic analysis with datasets like Universal Dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TokenAlignment:
    """Result of token alignment between tokenizer and sentence tokens."""

    # Mapping from tokenizer token index to sentence token index
    tokenizer_to_sentence: Dict[int, int]

    # Mapping from sentence token index to list of tokenizer token indices
    sentence_to_tokenizer: Dict[int, List[int]]

    # Tokens that couldn't be aligned
    unaligned_tokenizer: List[int]
    unaligned_sentence: List[int]


class TokenAligner:
    """Handles alignment between tokenizer tokens and sentence tokens."""

    def __init__(self):
        pass

    def align_tokens(
        self,
        text: str,
        tokenizer_tokens: List[str],
        sentence_tokens: List[str],
    ) -> TokenAlignment:
        """Align tokenizer tokens with sentence tokens.

        This method attempts to match tokens by reconstructing the text
        and finding overlaps. It handles cases where:
        - Tokenizer splits words differently than sentence tokenization
        - Tokenizer adds special tokens
        - There are slight formatting differences

        Args:
            text: Original text
            tokenizer_tokens: Tokens as produced by the tokenizer
            sentence_tokens: Original sentence tokens (e.g., from UD dataset)

        Returns:
            TokenAlignment object with mapping information
        """
        # Clean tokens for alignment
        clean_tokenizer_tokens = self._clean_tokens(tokenizer_tokens)
        clean_sentence_tokens = self._clean_tokens(sentence_tokens)

        # Get indices of tokens that actually appear in the text
        tokenizer_indices = self._get_token_indices(text, clean_tokenizer_tokens)
        sentence_indices = self._get_token_indices(text, clean_sentence_tokens)

        # Filter tokens to only those that appear in text
        filtered_tokenizer_tokens = [
            clean_tokenizer_tokens[i] for i in tokenizer_indices
        ]
        filtered_sentence_tokens = [clean_sentence_tokens[i] for i in sentence_indices]

        # Create mapping between filtered tokens
        mapping = self._map_tokens_to_sentence_tokens(
            filtered_tokenizer_tokens, filtered_sentence_tokens
        )

        # Convert back to original token indices
        tokenizer_to_sentence = {}
        sentence_to_tokenizer = {}

        for filtered_tok_idx, sent_idx in enumerate(mapping):
            if sent_idx >= 0:
                # Map filtered index back to original tokenizer index
                original_tok_idx = tokenizer_indices[filtered_tok_idx]
                original_sent_idx = sentence_indices[sent_idx]

                tokenizer_to_sentence[original_tok_idx] = original_sent_idx

                if original_sent_idx not in sentence_to_tokenizer:
                    sentence_to_tokenizer[original_sent_idx] = []
                sentence_to_tokenizer[original_sent_idx].append(original_tok_idx)

        # Handle special tokens (like EOS) that weren't in the filtered tokens
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

    def _clean_tokens(self, tokens: List[str]) -> List[str]:
        """Clean tokens by removing spaces and converting to lowercase."""
        return [token.replace(" ", "").lower() for token in tokens]

    def _get_token_indices(self, text: str, tokens: List[str]) -> List[int]:
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
        self, tokenizer_tokens: List[str], sentence_tokens: List[str]
    ) -> List[int]:
        """Map tokenizer tokens to sentence token indices.

        Returns a list where each element i contains the sentence token index
        that tokenizer token i belongs to, or -1 if no mapping found.
        """
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


# Utility functions for analyzing alignment quality
def analyze_alignment_quality(
    alignment: TokenAlignment,
    tokenizer_tokens: List[str],
    sentence_tokens: List[str],
) -> Dict[str, float]:
    """Analyze the quality of token alignment."""
    total_tokenizer = len(tokenizer_tokens)
    total_sentence = len(sentence_tokens)

    aligned_tokenizer = len(alignment.tokenizer_to_sentence)
    aligned_sentence = len(alignment.sentence_to_tokenizer)

    # Calculate alignment rates
    tokenizer_alignment_rate = (
        aligned_tokenizer / total_tokenizer if total_tokenizer > 0 else 0
    )
    sentence_alignment_rate = (
        aligned_sentence / total_sentence if total_sentence > 0 else 0
    )

    return {
        "tokenizer_alignment_rate": tokenizer_alignment_rate,
        "sentence_alignment_rate": sentence_alignment_rate,
        "unaligned_tokenizer_count": len(alignment.unaligned_tokenizer),
        "unaligned_sentence_count": len(alignment.unaligned_sentence),
        "total_tokenizer_tokens": total_tokenizer,
        "total_sentence_tokens": total_sentence,
    }


def get_misaligned_examples(
    alignment: TokenAlignment,
    tokenizer_tokens: List[str],
    sentence_tokens: List[str],
    max_examples: int = 10,
) -> Dict[str, List[str]]:
    """Get examples of tokens that couldn't be aligned."""
    misaligned_tokenizer = [
        tokenizer_tokens[i] for i in alignment.unaligned_tokenizer[:max_examples]
    ]
    misaligned_sentence = [
        sentence_tokens[i] for i in alignment.unaligned_sentence[:max_examples]
    ]

    return {
        "unaligned_tokenizer_examples": misaligned_tokenizer,
        "unaligned_sentence_examples": misaligned_sentence,
    }
