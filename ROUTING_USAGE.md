# MOE Routing Extraction - Quick Usage Guide

## 📁 Clean Project Structure

```
moe-explainability/
├── routing/                     # 🎯 Main routing package
│   ├── core.py                  # Core extraction logic  
│   ├── adapters.py              # Dataset adapters
│   ├── pipeline.py              # High-level functions
│   ├── process_ud.py            # UD dataset script
│   ├── process_wordsim.py       # WordSim script
│   └── examples/                # Examples & tests
│
├── setup_paths.py               # 🔧 Simple import helper
├── scripts/                     # 📊 Other analysis scripts
└── data/                        # 💾 Generated data
```

## 🚀 Recommended Usage

### **Option 1: Quick & Easy (Recommended for research)**
```python
# From project root, add this to any script:
import sys
sys.path.append('.')
from setup_paths import *

# Now you can use everything directly:
result = quick_route_analysis("Hello world!")
df_words = extract_wordsim_routes(word_list=["hello", "world"])
df_ud = extract_ud_routes(normalize=False)  # Raw logits recommended
```

### **Option 2: Direct Import**
```python
# From project root
from routing import extract_ud_routes, extract_wordsim_routes, quick_route_analysis

# Use normally
df = extract_ud_routes(cache_file="data/my_routes.parquet.gzip")
```

### **Option 3: Run Scripts Directly**
```bash
# From project root
python routing/process_ud.py           # Process UD dataset
python routing/process_wordsim.py      # Process WordSim dataset
python routing/examples/example_usage.py  # See examples
```

## 🎯 Key Functions

```python
# Quick analysis
result = quick_route_analysis("Text to analyze")
# Returns: tokens, route_matrix, route_stats

# UD dataset (token-level with linguistic annotations) 
df_ud = extract_ud_routes(
    normalize=False,  # Store raw logits (recommended)
    cache_file="data/ud_routes.parquet.gzip"
)

# WordSim dataset (token-level with original words)
df_words = extract_wordsim_routes(
    word_list=["attention", "transformer"], 
    normalize=False,
    cache_file="data/word_routes.parquet.gzip"
)
```

## 💡 Research Tips

### Memory Optimization
```python
# Reduce memory by 75% with minimal precision loss
df['route_vector'] = df['route_vector'].apply(lambda x: x.astype(np.float16))
```

### Flexible Configuration
```python
from routing import RouterConfig, RoutingPipeline

config = RouterConfig.for_switch_base_8(
    normalize=False,      # Raw logits for flexibility
    device="cuda"         # Use GPU
)
pipeline = RoutingPipeline("google/switch-base-8", config)
```

### Downstream Analysis
```python
import torch

# Apply different normalizations as needed
raw_logits = torch.from_numpy(df['route_vector'].values)
softmax_probs = torch.softmax(raw_logits, dim=-1)
log_probs = torch.log_softmax(raw_logits, dim=-1)
```

## ✅ Verification

Test that everything works:
```bash
# From project root
python -c "from setup_paths import *; print('All good!')"
python routing/examples/test_functionality.py
```

**Perfect for research - clean, flexible, and not over-engineered!** 🔬 