# EmbeddingRank: Information-Theoretic Dimension Allocation for Multivariate Sequence Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EmbeddingRank** is a framework for analyzing multivariate time series data and automatically determining optimal embedding dimension allocation based on information-theoretic principles. Instead of using equal allocation or manual tuning, EmbeddingRank uses principled measures of information content, sequential complexity, and cross-attribute dependencies to allocate embedding dimensions proportionally to each attribute's complexity.

## 🚀 Key Features

- **Information-Theoretic Foundation**: Uses entropy measures and mutual information for principled dimension allocation
- **Sequential Complexity Analysis**: Detects temporal patterns including k-gram analysis and medium-range dependencies
- **Mixed Data Support**: Handles both discrete and continuous attributes seamlessly
- **Adaptive Budget Recommendations**: Provides minimum, recommended, and optimal embedding budgets
- **GPU Acceleration**: Full PyTorch implementation with CUDA support

## 🎯 Problem Solved

When working with multivariate sequence embeddings (e.g., for transformers), you face a fundamental tradeoff:

- **Summation**: `embed_a + embed_b + embed_c` → Information loss through dilution
- **Concatenation**: `[embed_a; embed_b; embed_c]` → Dimensionality explosion
- **Manual Allocation**: Time-consuming and suboptimal

**EmbeddingRank** automatically determines how much embedding space each attribute deserves based on its intrinsic complexity.

## 📊 How It Works

EmbeddingRank performs a 6-step analysis:

1. **Information Content Analysis**: Measures entropy/differential entropy for each attribute
2. **Sequential Complexity Analysis**: Detects temporal patterns and dependencies  
3. **Cross-Attribute Dependencies**: Computes mutual information between attributes
4. **Unified Complexity Scoring**: Combines metrics with learned weights
5. **Budget Recommendations**: Provides information-theoretic bounds
6. **Dimension Allocation**: Distributes dimensions proportionally to complexity

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/embeddingrank.git
cd embeddingrank

# Install dependencies
pip install torch numpy

# Or use conda
conda install pytorch numpy -c pytorch
```

**Requirements:**
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.19+

## 🚀 Quick Start

```python
import torch
from embeddingrank import analyze_embedding_requirements

# Your multivariate time series data [samples, attributes]
# Mix of discrete and continuous attributes
data = torch.stack([
    torch.randint(1, 13, (10000,)).float(),    # Discrete: pitch (12 semitones)
    torch.randint(3, 8, (10000,)).float(),     # Discrete: octave (5 octaves)
    torch.randint(4, 64, (10000,)).float(),    # Discrete: duration (note lengths)
    torch.randint(0, 108, (10000,)).float(),   # Discrete: delta (time intervals)
    torch.rand(10000) * 6.75,                  # Continuous: timing (seconds)
    torch.randn(10000) * 20                    # Continuous: semitones (pitch change)
], dim=1)

# Define your schema
attribute_names = ['Pitch', 'Octave', 'Duration', 'Delta', 'Timing', 'Semitones']
attribute_types = ['discrete', 'discrete', 'discrete', 'discrete', 'continuous', 'continuous']

# Analyze and get allocations
allocations, budgets = analyze_embedding_requirements(
    data=data,
    attribute_names=attribute_names,
    attribute_types=attribute_types,
    total_embedding_dim=None  # Auto-recommend budget
)

print(f"Recommended total dimensions: {budgets['recommended']}")
print(f"Per-attribute allocation: {allocations.tolist()}")
```

## 📈 Example Output: Musical Sequence Analysis

Real analysis output from 951,333 musical events across 403 Mozart pieces:

```
=== EMBEDDING ALLOCATION ANALYSIS ===
Device: cuda
Dataset: 951,333 samples, 6 attributes
Attribute types: ['discrete', 'discrete', 'discrete', 'discrete', 'continuous', 'continuous']
Analysis depth: 8-grams for discrete, medium-range patterns

STEP 1: INFORMATION CONTENT ANALYSIS
Measuring theoretical minimum dimensions for each attribute type.
  Pitch (discrete):
    Vocabulary size: 12
    Entropy: 3.460 bits
    Theoretical min dimensions: 4
    → INTERPRETATION: Low entropy - skewed distribution
  Octave (discrete):
    Vocabulary size: 8
    Entropy: 1.983 bits
    Theoretical min dimensions: 2
    → INTERPRETATION: Low entropy - skewed distribution
  Duration (discrete):
    Vocabulary size: 12
    Entropy: 2.997 bits
    Theoretical min dimensions: 3
    → INTERPRETATION: Low entropy - skewed distribution
  Delta (discrete):
    Vocabulary size: 108
    Entropy: 2.484 bits
    Theoretical min dimensions: 3
    → INTERPRETATION: Low entropy - skewed distribution
  Timing (continuous):
    Value range: [0.000, 6.750]
    Differential entropy: 1.000 nats
    Effective dimensions: 1
    → INTERPRETATION: Low variability - concentrated distribution
  Semitones (continuous):
    Value range: [-65.000, 69.000]
    Differential entropy: 3.686 nats
    Effective dimensions: 4
    → INTERPRETATION: High variability - complex distribution

STEP 2: SEQUENTIAL COMPLEXITY ANALYSIS
Measuring temporal/sequential patterns including medium-range dependencies.
  Pitch (discrete):
    8-gram complexity: 1.092
    Medium-range pattern complexity: 4.296
    Final sequential complexity: 10.000
    → INTERPRETATION: Highly random transitions, complex patterns
  Delta (discrete):
    8-gram complexity: 1.000
    Medium-range pattern complexity: 8.000
    Final sequential complexity: 19.000
    → INTERPRETATION: Highly random transitions, complex patterns

STEP 3: CROSS-ATTRIBUTE DEPENDENCY ANALYSIS
Measuring information sharing between attributes.
  Delta dependencies:
    I(Delta,Timing): 1.7969
      → Moderate dependency - some correlation
    Total dependency score: 2.4384
    → INTERPRETATION: Moderately connected, some shared structure

STEP 5: EMBEDDING BUDGET RECOMMENDATIONS
  Minimum viable budget: 17 dimensions
    → Information-theoretic floor, expect information loss below this
  Recommended budget: 45 dimensions
    → Enhanced complexity-adjusted for medium-range patterns
  Optimal budget: 43 dimensions
    → Diminishing returns threshold, maximum practical benefit

STEP 6: DIMENSION ALLOCATION
  Final allocations:
    Pitch (discrete): 7 dims (15.6%)
    Octave (discrete): 6 dims (13.3%)
    Duration (discrete): 7 dims (15.6%)
    Delta (discrete): 10 dims (22.2%)
    Timing (continuous): 7 dims (15.6%)
    Semitones (continuous): 8 dims (17.8%)

=== SUMMARY ===
Most complex attribute: Delta (gets 10 dims)
Least complex attribute: Octave (gets 6 dims)
Recommendation: MODERATE allocation differences justified
```

## 🔬 Advanced Usage

### Musical Data Analysis

```python
# Real-world example with musical sequences
from embeddingrank import analyze_embedding_requirements

# Musical event data: [samples, 6_attributes]
# Attributes: pitch, octave, duration, delta_time, timing, semitone_changes
musical_data = load_musical_events()  # Shape: [951333, 6]

allocations, budgets = analyze_embedding_requirements(
    data=musical_data,
    attribute_names=['Pitch', 'Octave', 'Duration', 'Delta', 'Timing', 'Semitones'],
    attribute_types=['discrete', 'discrete', 'discrete', 'discrete', 'continuous', 'continuous'],
    k_gram_size=8,  # Analyze 8-note musical phrases
    device=torch.device('cuda')
)

# Result: Delta gets 10 dims (22.2%) due to high temporal complexity
# Octave gets only 6 dims (13.3%) due to limited range and predictability
```

### Custom Analysis Parameters

```python
# Fine-tune the analysis for your domain
allocations, budgets = analyze_embedding_requirements(
    data=data,
    attribute_names=names,
    attribute_types=types,
    k_gram_size=12,  # Deeper sequential analysis
    total_embedding_dim=64,  # Fixed budget constraint
    device=torch.device('cuda')
)
```

### Integration with Transformers

```python
import torch.nn as nn

# Example with musical data: 4 discrete + 2 continuous attributes
vocab_sizes = [12, 8, 12, 108, None, None]  # None for continuous
attribute_types = ['discrete', 'discrete', 'discrete', 'discrete', 'continuous', 'continuous']

# Get optimal dimension allocations
allocations, _ = analyze_embedding_requirements(data, names, attribute_types)
# Result: [7, 6, 7, 10, 7, 8] for musical data

# Create embedding layers with allocated dimensions
embeddings = nn.ModuleList([
    nn.Embedding(vocab_sizes[i], allocations[i]) if attribute_types[i] == 'discrete'
    else nn.Linear(1, allocations[i]) 
    for i in range(len(attribute_types))
])

def embed_multivariate(inputs):
    """
    inputs: [batch_size, seq_len, num_attributes]
    returns: [batch_size, seq_len, total_embedding_dim]
    """
    embedded = []
    for i, attr_type in enumerate(attribute_types):
        x = inputs[:, :, i]  # [batch_size, seq_len]
        
        if attr_type == 'discrete':
            embedded.append(embeddings[i](x.long()))
        else:
            embedded.append(embeddings[i](x.unsqueeze(-1)))
    
    return torch.cat(embedded, dim=-1)  # Optimal concatenation

# Usage
inputs = torch.randn(32, 100, 6)  # [batch, sequence, attributes]
embedded = embed_multivariate(inputs)  # [32, 100, 45] with optimal allocation
```

## 📚 API Reference

### Core Functions

#### `analyze_embedding_requirements()`
Main entry point for embedding analysis.

**Parameters:**
- `data` (torch.Tensor): Input tensor [samples, attributes]
- `attribute_names` (List[str]): Human-readable attribute names
- `attribute_types` (List[str]): 'discrete' or 'continuous' for each attribute
- `total_embedding_dim` (int, optional): Fixed budget, None for auto-recommendation
- `k_gram_size` (int, default=8): Sequential pattern analysis depth
- `device` (torch.device, optional): Computing device

**Returns:**
- `allocations` (torch.Tensor): Dimension allocation per attribute
- `budget_recommendations` (Dict[str, int]): Budget analysis with 'minimum', 'recommended', 'optimal'

### Complexity Analysis Functions

#### `compute_enhanced_higher_order_complexity()`
Enhanced k-gram analysis for sequential patterns beyond Markov chains.

#### `detect_medium_range_patterns()`
Pattern repetition analysis with gap tolerance for medium-range dependencies.

#### `compute_mixed_mutual_information()` 
Mutual information between mixed discrete/continuous variables using appropriate estimators.

## 🧪 Research Background

This implementation is based on information-theoretic principles for optimal representation learning:

- **Information Content**: Uses Shannon entropy for discrete and differential entropy for continuous variables
- **Sequential Complexity**: Extends beyond Markov chains with enhanced k-gram analysis and medium-range pattern detection
- **Cross-Dependencies**: Mutual information quantifies attribute relationships for coordination requirements
- **Principled Allocation**: Combines metrics with theoretically motivated weights (35% information, 40% sequential, 25% dependencies)

### Key Improvements Over Basic Methods

| Method | Domain | Approach | Sequential Patterns | Mixed Types | Budget Estimation |
|--------|---------|----------|-------------------|-------------|------------------|
| **EmbeddingRank** | Time Series | Information Theory | ✅ Enhanced k-gram + Medium-range | ✅ | ✅ Automatic |
| Equal Allocation | General | Manual | ❌ | ✅ | ❌ Manual |
| Cardinality-based | Categorical | Heuristic | ❌ | ❌ | ❌ Manual |
| Learned Embeddings | General | End-to-end | ❌ | ✅ | ❌ Manual |

## 🔬 Experimental Validation

Run the comprehensive test suite:

```python
from embeddingranktest import run_validation_suite

# Tests entropy calculation, mutual information, transition matrices,
# sequential complexity, dimension allocation, and known musical patterns
run_validation_suite()
```

Run the synthetic demonstration:

```python
from embeddingrank import demo_with_synthetic_data

# Generates mixed discrete/continuous data with known patterns
allocations, budgets = demo_with_synthetic_data()
```

## ⚡ Performance

- **Scalability**: Handles millions of samples efficiently with adaptive sampling strategies
- **Memory**: O(V²) where V is maximum vocabulary size, optimized with vocabulary capping
- **Time Complexity**: O(L×N + N²×K) for L samples, N attributes, K k-gram size
- **GPU Support**: Full CUDA acceleration for all tensor operations

### Real-World Performance
- Mozart corpus (951k events): ~30 seconds analysis on RTX 4000
- Memory usage: ~2GB for million-sample datasets
- Automatic efficiency adaptations for high-cardinality variables

## 🤝 Contributing

We welcome contributions! Priority areas:

- **Kernel-based MI estimation** for better continuous variable analysis
- **Hierarchical pattern detection** for very long sequences
- **Domain-specific benchmarks** and evaluation metrics
- **Distributed processing** for massive datasets

### Development Setup

```bash
git clone https://github.com/yourusername/embeddingrank.git
cd embeddingrank

# Install in development mode
pip install -e .

# Run validation tests
python embeddingranktest.py

# Format code (following your preferences)
# Alphabetical method ordering, type hints, minimal blank lines
```

## 📄 Citation

If you use EmbeddingRank in your research, please cite:

```bibtex
@software{embeddingrank2025,
  title={EmbeddingRank: Information-Theoretic Dimension Allocation for Multivariate Sequence Embeddings},
  author={Maurice Calvert},
  year={2025},
  url={https://github.com/yourusername/embeddingrank}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Information theory foundations inspired by Shannon's work on entropy
- Sequential analysis techniques from time series and music informatics literature  
- Mutual information estimation methods from machine learning research

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/embeddingrank/issues)
- **Email**: maurice@calvert.ch

---

**Principled embedding allocation for the ML research community**
