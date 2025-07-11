# %%
"""Example script showing how to use the new architecture for UD dataset processing."""

from datasets import load_dataset

from moe_explainability.datasets.configs import UD_ENGLISH_EWT
from moe_explainability.datasets.processing import (
    annotated_tokens_to_dataframe,
    extract_and_align_routing_tokens,
)
from moe_explainability.models.configs import SWITCH_BASE_8
from moe_explainability.models.loading import create_extraction_fn

# %%
print("=== UD Processing with New Architecture ===")

# Load UD dataset (small subset for demo)
print("Loading UD dataset...")
ud_data = load_dataset("universal_dependencies", "en_ewt", split="train[:5]")
print(f"Loaded {len(ud_data)} sentences")

# %%
# Create extraction function (loads model automatically)
print("Loading model...")
extract_tokens_fn = create_extraction_fn(SWITCH_BASE_8)

# %%
# Process using new architecture
print("Processing with new architecture...")
annotated_tokens = extract_and_align_routing_tokens(
    data=list(ud_data),
    config=UD_ENGLISH_EWT,
    extract_tokens_fn=extract_tokens_fn,
)

print(f"Processed {len(annotated_tokens)} tokens")

# %%
# Convert to DataFrame for analysis
print("Converting to DataFrame...")
df = annotated_tokens_to_dataframe(annotated_tokens)

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# %%
# Show sample data
print("\n=== Sample Data ===")
sample_cols = [
    "sentence_id",
    "token_text",
    "word_text",
    "upos",
    "lemmas",
    "deprel",
]
available_cols = [col for col in sample_cols if col in df.columns]
print(df[available_cols].head(10))

# %%
# Show alignment quality
print("\n=== Alignment Quality ===")
total_tokens = len(df)
aligned_tokens = len(df[df["word_index"] >= 0])
print(f"Total tokens: {total_tokens}")
print(f"Aligned tokens: {aligned_tokens}")
print(f"Alignment rate: {aligned_tokens / total_tokens:.2%}")

# %%
# Show POS distribution
print("\n=== POS Distribution ===")
pos_counts = df["upos"].value_counts().head(10)
print(pos_counts)

# %%
# Example: Work with AnnotatedToken objects directly
print("\n=== Working with AnnotatedToken objects ===")
# Find all NOUN tokens
noun_tokens = [
    token for token in annotated_tokens if token.word_fields.get("upos") == "NOUN"
]
print(f"Found {len(noun_tokens)} NOUN tokens")

# Show routing patterns for first few NOUNs
for i, token in enumerate(noun_tokens[:3]):
    print(
        f"NOUN {i + 1}: '{token.word_text}' -> route_vector shape: {token.token.get_route_vector().shape}"
    )
