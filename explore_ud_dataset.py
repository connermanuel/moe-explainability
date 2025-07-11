"""Explore UD dataset structure to understand the data."""

import pandas as pd
from datasets import load_dataset


def explore_ud_dataset():
    """Explore the structure of the UD dataset."""
    print("=== Universal Dependencies Dataset Exploration ===\n")

    # Load a small sample
    data = load_dataset("universal_dependencies", "en_ewt", split="train[:5]")

    print(f"Dataset size: {len(data)}")
    print(f"Dataset features: {data.features}")
    print()

    # Look at the first few examples
    for i, example in enumerate(data):
        print(f"=== Example {i + 1} ===")
        print(f"Text: {example['text']}")
        print(f"Tokens: {example['tokens']}")
        print(f"UPOS: {example['upos']}")
        print(f"XPOS: {example['xpos']}")
        print(f"Lemmas: {example['lemmas']}")
        print(f"Heads: {example['head']}")
        print(f"Deprel: {example['deprel']}")
        print(f"Features: {example['feats']}")
        print()

        if i >= 2:  # Just show first 3 examples
            break

    # Analyze the structure
    print("=== Analysis ===")
    example = data[0]
    print(f"Number of tokens in first sentence: {len(example['tokens'])}")
    print(f"Text length: {len(example['text'])}")
    print(f"Token alignment challenge: tokenizer vs sentence tokens")
    print(f"Available linguistic features: {list(example.keys())}")
    print()


if __name__ == "__main__":
    explore_ud_dataset()
