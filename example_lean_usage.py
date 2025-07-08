"""Example usage of the new lean architecture."""

from moe_explainability.datasets.processing import process_texts, tokens_to_dataframe
from moe_explainability.models.configs import SWITCH_BASE_8, ModelConfig
from moe_explainability.models.loading import load_model
from moe_explainability.routing.extraction import extract_tokens


def simple_example():
    """Simple example showing the lean functional approach."""
    print("=== Lean Architecture Example ===")

    config = SWITCH_BASE_8
    model, tokenizer = load_model(config)

    # Extract routing for a single text
    text = "The quick brown fox jumps over the lazy dog."
    tokens = extract_tokens(model, tokenizer, text, config)

    print(f"Text: {text}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Tokens: {[t.text for t in tokens]}")
    print(f"First token route vector shape: {tokens[0].get_route_vector().shape}")

    # Convert to DataFrame for analysis
    df = tokens_to_dataframe(tokens)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print()


def batch_example():
    """Example showing batch processing."""
    print("=== Batch Processing Example ===")

    config = SWITCH_BASE_8
    model, tokenizer = load_model(config)

    # Create a simple extraction function
    def extract_fn(text):
        return extract_tokens(model, tokenizer, text, config)

    # Process multiple texts
    texts = [
        "Hello world!",
        "Machine learning is fascinating.",
        "Neural networks learn patterns.",
    ]

    # Use the generic processing function
    df = process_texts(texts, extract_fn)

    print(f"Processed {len(texts)} texts")
    print(f"Total tokens: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df[["text_id", "token_text", "position"]].head())
    print()


def normalization_example():
    """Example showing downstream normalization."""
    print("=== Normalization Example ===")

    config = SWITCH_BASE_8
    model, tokenizer = load_model(config)

    # Extract raw logits
    tokens = extract_tokens(model, tokenizer, "attention", config)
    token = tokens[0]  # First token

    print(f"Token: {token.text}")
    print(f"Raw logits for layer 1: {token.layer_logits[1][:5]}...")  # First 5

    # Apply different normalizations downstream
    softmax_probs = token.get_normalized_logits(1, "softmax")
    log_probs = token.get_normalized_logits(1, "log_softmax")

    print(f"Softmax probabilities: {softmax_probs[:5]}...")
    print(f"Log probabilities: {log_probs[:5]}...")
    print()


if __name__ == "__main__":
    try:
        simple_example()
        batch_example()
        normalization_example()
        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the dependencies installed and GPU/CPU available.")
