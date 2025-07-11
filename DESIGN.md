# MoE Explainability Dataset Processing Architecture

## Overview

This document outlines the design for a flexible, modular architecture for processing datasets with MoE (Mixture of Experts) routing analysis. The system handles multiple data granularities (token, word, sentence) and supports various datasets and model architectures.

## Core Design Principles

1. **Granularity Independence**: Token-level and word-level granularities will not necessarily align
2. **Modular Abstraction**: "Alignment" and "Extraction" are pluggable, interchangeable components
3. **Configuration-Driven**: Dataset schemas are explicitly defined through configuration
4. **Extensibility**: Easy to add new datasets, models, and processing strategies

## Basic Architecture

### Data Granularities

The system processes three distinct data granularities:

- **Token-level**: Individual tokenizer tokens with routing information (1 route vector per token)
- **Word-level**: Linguistic units with semantic/syntactic features (1 POS tag per word)  
- **Sentence-level**: Context and metadata (1 sentence ID per sentence)

### Core Components

```
DatasetConfig → Processing Pipeline → Enriched DataFrame
     ↓                    ↓
Text + Metadata → Token Extraction → Token-Word Alignment → Final Output
```

## Implementation Design

### 1. DatasetConfig Class

```python
@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    
    name: str                    # Dataset identifier
    text_field: str             # Field containing text to process
    word_fields: List[str]      # Fields with word-level data
    words_field: str            # Field containing actual words for alignment
    sentence_fields: List[str]  # Fields with sentence-level data
    
    # Optional extensions
    alignment_strategy: str = "character_based"
    nested_structure: Optional[Dict[str, str]] = None
```

**Example UD Configuration:**
```python
UD_ENGLISH_EWT = DatasetConfig(
    name="ud_english_ewt",
    text_field="text",
    word_fields=["tokens", "upos", "xpos", "lemmas", "head", "deprel", "feats"],
    words_field="tokens",
    sentence_fields=[]  # No sentence-level fields in UD
)
```

### 2. Functional Processing Pipeline

```python
def extract_and_align_routing_tokens(
    data: List[Dict[str, Any]],
    config: DatasetConfig,
    extract_tokens_fn: Callable[[str], List[Token]],
) -> pd.DataFrame:
    """Generic structured data processing pipeline."""
    
    all_dfs = []
    
    for sentence_id, item in enumerate(data):
        # Step 1: Extract tokens with routing information
        text = item[config.text_field]
        tokens = extract_tokens_fn(text)
        
        # Step 2: Align tokens with words using existing alignment logic
        word_data = {field: item[field] for field in config.word_fields}
        alignment = align_tokens(text, [t.text for t in tokens], item[config.words_field])
        
        # Step 3: Create enriched DataFrame
        df = create_enriched_dataframe(
            tokens, alignment, word_data, 
            sentence_data=item, sentence_id=sentence_id
        )
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)
```

### 3. Lean Functional Approach

**Key Decision**: Keep the design lean and functional, avoiding unnecessary classes:
- Use existing `extract_tokens()` function from `routing/extraction.py`
- Use existing `align_tokens()` function from `datasets/alignment.py`
- Add helper functions as needed, but avoid class hierarchies
- Reserve classes only for dataclasses and core data structures

### 4. Output DataFrame Schema

Each row represents a **token** with:
- Token-level: `token_id`, `token_text`, `position`, `route_vector`, `layer_X_logits`
- Word-level: `word_index`, `word_text`, `upos`, `lemma`, `deprel`, etc.
- Sentence-level: `sentence_id`, `sentence_text`, `sentence_length`

**Key insight**: Multiple tokens can map to the same word, some tokens may not align to any word (special tokens).

## Current Implementation (Phase 1)

### Immediate Goals
1. ✅ Create `DatasetConfig` class in `datasets/configs.py`
2. Refactor `extract_and_align_routing_tokens()` to use config-driven approach
3. Create helper function for enriched DataFrame creation
4. Create UD dataset configuration
5. Update existing UD processing to use new architecture

### File Changes
- `src/moe_explainability/datasets/configs.py` → Add `DatasetConfig`
- `src/moe_explainability/datasets/processing.py` → Refactor with config approach
- `src/moe_explainability/datasets/ud_processing.py` → Use generic processing
- `src/moe_explainability/analysis/ud_analysis.py` → Update to new interface

## Future Extensions

### 1. Multiple Alignment Strategies
- **Character-based**: Current approach using character-level matching
- **Heuristic**: Simple string matching with fuzzy logic
- **Transformer-based**: Use alignment models for complex tokenization mismatches
- **Approximate**: Handle cases where perfect alignment isn't possible

### 2. Multi-Model Support
```python
def extract_switch_tokens(model, tokenizer, text, config):
    """Extract tokens from Switch Transformer models."""
    
def extract_glam_tokens(model, tokenizer, text, config):
    """Extract tokens from GLaM models."""
```

### 3. Hierarchical Dataset Support
For datasets with nested structure (documents → paragraphs → sentences):
```python
@dataclass
class DatasetConfig:
    nested_structure: Optional[Dict[str, str]] = None
    # {"document_id": "doc_field", "paragraph_id": "para_field"}
```

### 4. Performance Optimizations
- **Batch Processing**: Process multiple texts simultaneously
- **Streaming**: Handle large datasets without loading everything into memory
- **Caching**: Smart caching strategies for different model/dataset combinations

### 5. Advanced Alignment Features
- **Confidence Scores**: Measure alignment quality
- **Multiple Alignments**: Handle ambiguous token-word mappings
- **Cross-Linguistic**: Support for different tokenization strategies per language

## Potential Challenges & Solutions

### Challenge 1: Token Alignment Complexity
**Problem**: Different tokenizers have vastly different strategies
**Solution**: Pluggable alignment strategies + approximate alignment fallbacks

### Challenge 2: Dataset Schema Variability  
**Problem**: Nested structures, optional fields, different versions
**Solution**: Flexible configuration + schema validation + graceful degradation

### Challenge 3: Model-Specific Routing
**Problem**: Different MoE architectures expose routing differently
**Solution**: Abstract TokenExtractor interface + model-specific implementations

### Challenge 4: Memory & Performance
**Problem**: Large datasets with many linguistic features
**Solution**: Streaming processing + efficient caching + batch operations

## Migration Strategy

1. **Phase 1**: Implement basic architecture with UD dataset
2. **Phase 2**: Add second dataset to validate generalization
3. **Phase 3**: Add second model type to validate extraction abstraction
4. **Phase 4**: Implement advanced alignment strategies
5. **Phase 5**: Add performance optimizations and streaming support

This architecture provides a solid foundation that can evolve with research needs while maintaining clean separation of concerns.

## Development Principles & Best Practices

Based on implementation experience, here are key principles for maintaining and extending this codebase:

### Design Philosophy

#### **1. Lean and Functional Approach**
- **Minimize classes**: Reserve classes only for dataclasses and core data structures
- **Prefer functions**: Use functions over class hierarchies for operations
- **Avoid wrapper functions**: Don't create one-line functions that just call another function
- **Composition over inheritance**: `AnnotatedToken` contains `Token` rather than extending it

#### **2. Explicit Configuration**
- **Schema-driven**: Make data granularities explicit through `DatasetConfig`
- **Universal formats**: All data loaders return `List[Dict[str, Any]]` for interchangeability
- **Pluggable components**: Use function parameters rather than inheritance for flexibility

#### **3. Separation of Concerns**
- **Generic vs. specific**: Keep bespoke experiment logic in scripts, not packages
- **Modular abstraction**: Separate extraction, alignment, and analysis
- **Data structure choice**: Use appropriate structures (`AnnotatedToken` vs `DataFrame`) based on use case

### Implementation Best Practices

#### **Code Quality**
```python
# Good: Descriptive function names
extract_and_align_routing_tokens()

# Bad: Generic names  
process_structured_data()

# Good: Minimal imports
import torch.nn.functional as F

# Bad: Unnecessary casting
logits_tensor = torch.from_numpy(logits)
```

#### **Error Handling**
- **Graceful degradation**: Analysis functions should skip failures, not crash
- **Informative messages**: Print specific errors for debugging
- **Edge case handling**: Handle missing tokens, alignment failures, etc.

#### **Performance & Caching**
- **Cache expensive operations**: Routing extraction is slow, cache to pickle/parquet
- **Sample for development**: Use small datasets (`train[:5]`) for testing
- **Batch operations**: Process multiple items together when possible

### Development Workflow

#### **File Management**
1. **Always read before editing**: Check existing file structure first
2. **Clean imports**: Remove unused imports caught by linters
3. **Validate configurations**: Check DatasetConfig validation in `__post_init__`

#### **Script Structure**
- **Use code cells**: Structure scripts with `# %%` for interactive development
- **Reproducibility**: Set random seeds for sampling operations
- **Progress tracking**: Use `tqdm` and progress messages for long operations

#### **Data Handling**
```python
# Good: Handle missing data gracefully
if word1 in word_groups and word2 in word_groups:
    try:
        distance = compute_distance(...)
    except Exception:
        distance = np.nan

# Good: Universal data format
def load_ud_data(...) -> List[Dict[str, Any]]:
    return [dict(item) for item in dataset]
```

### Extension Guidelines

#### **Adding New Datasets**
1. Create loader function returning `List[Dict[str, Any]]`
2. Define `DatasetConfig` with appropriate field mappings
3. Test with small sample first
4. Add caching for expensive processing

#### **Adding New Analysis**
1. Keep experiment-specific logic in `/experiments` scripts
2. Extract reusable functions to appropriate modules
3. Use existing distance/alignment functions when possible
4. Include visualization and statistical analysis

#### **Adding New Models**
1. Extend `ModelConfig` with new model parameters
2. Update `load_model_and_tokenizer` for new model types
3. Ensure routing extraction works with new architectures
4. Test token alignment quality

### Common Pitfalls

#### **Data Structure Mismatches**
- **Problem**: Mixing `Token` and `AnnotatedToken` in function signatures
- **Solution**: Be explicit about which type functions expect

#### **Alignment Failures**
- **Problem**: Perfect alignment isn't always possible
- **Solution**: Design for graceful degradation, filter to aligned tokens when needed

#### **Memory Issues**
- **Problem**: Large datasets with many linguistic features
- **Solution**: Use streaming, sampling, and caching strategically

#### **Import Errors**
- **Problem**: Circular imports between modules
- **Solution**: Import functions locally when needed, avoid cross-module dependencies

### Testing Strategy

#### **Validation Approach**
1. **Start small**: Test with `train[:5]` samples
2. **Check alignment**: Verify token-word alignment quality
3. **Validate distances**: Ensure distance functions return reasonable values
4. **Compare methods**: Test multiple parameter combinations

#### **Debugging Tools**
- **Print intermediate results**: Show token counts, alignment rates, etc.
- **Visualize data**: Use scatter plots and distributions to understand patterns
- **Cache for iteration**: Save processed data to avoid recomputation during debugging

This approach has proven effective for building extensible, maintainable research code that balances flexibility with performance.