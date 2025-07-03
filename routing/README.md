# MOE Routing Extraction Package

A clean, modular package for extracting routing probabilities from Switch Transformer models, designed for research use.

## 📁 Project Structure

```
routing/                         # Main routing extraction package
├── README.md                    # This file
├── __init__.py                  # Package imports
│
├── core.py                      # Core routing extraction (RouterExtractor, RouterConfig)
├── adapters.py                  # Dataset adapters (UDAdapter, WordSimAdapter, etc.)
├── alignment.py                 # Token alignment utilities for UD dataset
├── pipeline.py                  # High-level orchestration (RoutingPipeline)
├── utils.py                     # Common utilities (caching, device info)
│
├── examples/
│   ├── example_usage.py         # Comprehensive usage examples
│   └── test_functionality.py   # Tests and verification
│
├── process_ud.py                # UD dataset processing script
├── process_wordsim.py           # WordSim dataset processing script
│
└── data/                        # Generated data files (cached results)
```

## 🚀 Quick Start

### 1. Basic Usage (Interactive)

```python
# Add to Python path (if needed)
import sys
sys.path.append('.')  # From project root
from setup_paths import *  # Handles all imports

# Or import directly
from routing import extract_ud_routes, extract_wordsim_routes, quick_route_analysis

# Quick analysis of a single text
result = quick_route_analysis("The attention mechanism is powerful.")
print(f"Tokens: {result['tokens']}")
print(f"Dominant experts: {result['route_stats']['dominant_experts_per_token']}")

# Extract routes for a word list
df_words = extract_wordsim_routes(
    word_list=["hello", "world", "attention"],
    normalize=False,  # Store raw logits (recommended)
    cache_file="data/my_words.parquet.gzip"
)

# Extract routes for UD dataset
df_ud = extract_ud_routes(
    normalize=False,  # Store raw logits (recommended)
    cache_file="data/ud_routes.parquet.gzip"
)
```

### 2. Run Processing Scripts

```bash
# Process UD dataset
python routing/process_ud.py

# Process WordSim dataset  
python routing/process_wordsim.py

# Run examples and tests
python routing/examples/example_usage.py
python routing/examples/test_functionality.py
```

### 3. Run with Python Module

```bash
# From project root
python -m routing.examples.example_usage
python -m routing.process_ud
```

## 📊 Data Output

### UD Dataset Output
Each row = 1 token with:
- **Token info**: `token_id`, `token_text`, `position`
- **Routing**: `route_vector` (48-dim), `expert_1`, `expert_3`, ..., `expert_11`
- **Sentence mapping**: `sentence_id`, `sent_token`, `sent_token_position`
- **Linguistic annotations**: `upos`, `xpos`, `lemmas`, `feats`, `head`, `deprel`

### WordSim Dataset Output
Each row = 1 token with:
- **Token info**: `token_id`, `token_text`, `position`
- **Routing**: `route_vector` (48-dim), `expert_1`, `expert_3`, ..., `expert_11`
- **Word mapping**: `original_word`, `item_id`

## 🎯 Research-Optimized Features

### Memory Efficiency
```python
# Optimize storage for large datasets
df['route_vector'] = df['route_vector'].apply(lambda x: x.astype(np.float16))
# 75% memory reduction with minimal precision loss
```

### Flexible Configuration
```python
from get_routes import RouterConfig, RoutingPipeline

# Custom configuration
config = RouterConfig(
    router_layers=[1, 3, 5, 7, 9, 11],  # Which layers to extract
    normalize=False,                     # Store raw logits (recommended)
    device="cuda",                       # GPU acceleration
    extract_hidden_states=True           # Also extract hidden states
)

# Use with pipeline
pipeline = RoutingPipeline("google/switch-base-8", config)
```

### Downstream Normalization
```python
import torch

# Apply different normalizations as needed
raw_logits = torch.from_numpy(df['route_vector'].values)
softmax_probs = torch.softmax(raw_logits, dim=-1)
log_probs = torch.log_softmax(raw_logits, dim=-1)
```

## 🔬 For Research Use

### Easy Experimentation
- **Modular design**: Swap components without rewriting everything
- **Clean interfaces**: Easy to extend for new models/datasets
- **Configurable**: Test different normalization strategies

### Common Research Patterns
```python
# Compare different expert layers
for layer in [1, 3, 5, 7, 9, 11]:
    layer_probs = df[f'expert_{layer}'].values
    # Analyze routing patterns...

# Token-level analysis with linguistic features
ud_df = extract_ud_routes(normalize=False)
for pos_tag in ['NOUN', 'VERB', 'ADJ']:
    pos_tokens = ud_df[ud_df['upos'] == pos_tag]
    # Analyze routing by POS tag...

# Word-level analysis
word_df = extract_wordsim_routes(word_list=my_words, normalize=False)
# Analyze semantic similarity via routing...
```

### Adding New Datasets
```python
from get_routes import FlexibleAdapter, RouterExtractor

# Create adapter for your dataset
adapter = FlexibleAdapter(
    extractor, 
    text_field="content",
    extra_fields=["category", "source"]
)
df = adapter.process_dataset(my_custom_data)
```

## 💡 Design Philosophy

This package follows research-friendly design principles:

1. **Modular but not over-engineered** - Easy to understand and modify
2. **Extensible** - Add new models/datasets by following clear patterns  
3. **Cacheable** - Avoid recomputing expensive operations
4. **Type-safe** - Comprehensive type hints for Python 3.10+
5. **Memory-conscious** - Efficient storage options for large datasets

## 🛠️ Troubleshooting

### Import Issues
If you get `ModuleNotFoundError`, make sure to add the parent directory to your Python path:
```python
import sys
sys.path.append('/path/to/your/project/scripts')
from get_routes import extract_ud_routes
```

### Memory Issues
For large datasets:
```python
# Use float16 for route vectors
df['route_vector'] = df['route_vector'].apply(lambda x: x.astype(np.float16))

# Process in chunks
for chunk in dataset.iter(batch_size=100):
    df_chunk = extract_ud_routes(dataset=chunk)
    # Process chunk...
```

### GPU Issues
```python
# Force CPU usage
config = RouterConfig.for_switch_base_8(device="cpu")
```

## 📈 Performance Tips

- **Use raw logits** (`normalize=False`) for storage efficiency
- **Cache results** with parquet compression for repeated analysis  
- **Use float16** for route vectors to reduce memory by 75%
- **Process incrementally** for very large datasets
- **Reuse RouterExtractor** instances when processing multiple batches

Perfect for research - clean, extensible, and not over-engineered! 🔬 