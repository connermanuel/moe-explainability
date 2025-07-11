# UD Dataset Analysis Plan

## Goal
Create a table where each row is a **token** from the UD dataset, containing:
- **Routing information**: Expert probabilities for each layer
- **Word-level information**: Which word the token came from
- **Sentence-level information**: Sentence index, position within sentence
- **Linguistic information**: POS tags, dependencies, lemmas, etc.

## Understanding the Challenge

### UD Dataset Structure
```python
{
    "text": "The quick brown fox jumps over the lazy dog.",
    "tokens": ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."],
    "upos": ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN", "PUNCT"],
    "xpos": ["DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN", "."],
    "lemmas": ["the", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog", "."],
    "head": [4, 4, 4, 5, 0, 9, 9, 9, 5, 5],
    "deprel": ["det", "amod", "amod", "nsubj", "root", "case", "det", "amod", "nmod", "punct"],
    "feats": ["Definite=Def|PronType=Art", "Degree=Pos", ...]
}
```

### Token Alignment Challenge
- **Tokenizer tokens**: `["▁The", "▁quick", "▁brown", "▁fox", "▁jumps", ...]`
- **UD tokens**: `["The", "quick", "brown", "fox", "jumps", ...]`
- **Need to align**: Tokenizer tokens → UD tokens to get linguistic features

## Implementation Plan

### Phase 1: Core UD Processing Functions
**Location**: `src/moe_explainability/datasets/ud_processing.py`

```python
def process_ud_sentence(sentence_data, extract_tokens_fn):
    """Process a single UD sentence with routing extraction."""
    # 1. Extract routing tokens from text
    # 2. Align tokenizer tokens with UD tokens
    # 3. Create enriched token data
    # 4. Return DataFrame with all information

def process_ud_dataset(dataset, model, tokenizer, config):
    """Process entire UD dataset."""
    # Use process_ud_sentence for each sentence
    # Combine results into single DataFrame
```

### Phase 2: Token-to-Word Alignment
**Location**: Use existing `src/moe_explainability/datasets/alignment.py`

```python
# For each sentence:
alignment = align_tokens(
    text=sentence_data["text"],
    tokenizer_tokens=[t.text for t in routing_tokens],
    sentence_tokens=sentence_data["tokens"]
)
```

### Phase 3: DataFrame Creation
**Target Output Schema**:
```python
{
    # Token-level info
    "token_id": int,
    "token_text": str,
    "token_position": int,
    "route_vector": np.ndarray,
    "layer_1_logits": np.ndarray,
    "layer_3_logits": np.ndarray,
    # ... for all layers
    
    # Sentence-level info
    "sentence_id": int,
    "sentence_text": str,
    "sentence_length": int,
    
    # Word-level info (from alignment)
    "word_index": int,           # Index in UD tokens
    "word_text": str,            # Original UD token
    "word_position": int,        # Position in sentence
    
    # Linguistic features (from UD)
    "upos": str,                 # Universal POS tag
    "xpos": str,                 # Language-specific POS tag
    "lemma": str,                # Lemma
    "head": int,                 # Dependency head
    "deprel": str,               # Dependency relation
    "feats": str,                # Morphological features
}
```

### Phase 4: High-Level Interface
**Location**: `src/moe_explainability/analysis/ud_analysis.py`

```python
def analyze_ud_dataset(
    dataset_split="train[:100]",  # Small sample for testing
    model_config=SWITCH_BASE_8,
    cache_file=None
):
    """High-level function to analyze UD dataset."""
    # 1. Load model and dataset
    # 2. Process with enriched token information
    # 3. Return analysis-ready DataFrame
```

## Implementation Steps

### Step 1: Explore UD Dataset
- [x] Create exploration script
- [ ] Run and understand structure
- [ ] Document findings

### Step 2: Implement Core Functions
- [ ] Create `ud_processing.py` with sentence processing
- [ ] Test alignment with sample sentences
- [ ] Verify linguistic feature extraction

### Step 3: Integration
- [ ] Create high-level analysis function
- [ ] Test with small dataset sample
- [ ] Optimize for larger datasets

### Step 4: Analysis & Validation
- [ ] Verify token-to-word alignment quality
- [ ] Check linguistic feature accuracy
- [ ] Create example analysis notebook

## Usage Example
```python
from moe_explainability.analysis.ud_analysis import analyze_ud_dataset

# Process UD dataset with routing information
df = analyze_ud_dataset(
    dataset_split="train[:100]",
    cache_file="data/ud_tokens_enriched.parquet"
)

# Now you can analyze:
# - Routing patterns by POS tag
# - Expert specialization by linguistic features
# - Token-level vs word-level routing differences
```

## Success Criteria
- [ ] Each row represents one tokenizer token
- [ ] Routing information available for each token
- [ ] Linguistic features correctly aligned
- [ ] Scalable to full UD dataset
- [ ] Clean, functional implementation