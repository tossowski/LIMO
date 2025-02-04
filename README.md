# LIMO: Less Is More for Mathematical Reasoning 🚀 

<div align="center">

[![Conference](https://img.shields.io/badge/ARXIV-2024-blue)](https://arxiv.org/abs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/GAIR-NLP/LIMO.svg?style=social&label=Star&maxAge=2592000)](https://github.com/GAIR-NLP/LIMO)
[![Twitter Follow](https://img.shields.io/twitter/follow/GAIR_NLP?style=social)](https://twitter.com/GAIR_NLP)

*An efficient approach for mathematical reasoning with minimal but high-quality training data*
</div>

## 📌 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)  
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Zoo](#model-zoo)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

LIMO challenges the conventional wisdom in mathematical reasoning by demonstrating that models can achieve superior performance with significantly less but higher quality training data. Our approach:

- 🎯 Achieves SOTA with only 817 carefully curated training samples
- 🌟 Shows strong generalization across diverse problem types
- 🔬 Provides comprehensive evaluation over 12 benchmarks
- 📚 Releases high-quality datasets and evaluation tools

## Key Results

| Model | AIME | MATH | Training Samples |
|-------|------|------|-----------------|
| LIMO (Ours) | **57.1%** | **94.8%** | 817 |
| Previous SOTA | 6.5% | 59.2% | 100k+ |

<details>
<summary>Click to see more detailed results</summary>

[Detailed performance breakdown across different benchmarks]

</details>

## Getting Started

### Installation
```bash
# Install from PyPI
pip install limo-math

# Install from source
git clone https://github.com/GAIR-NLP/LIMO.git
cd LIMO
pip install -e .
```

```python
from limo import LIMOModel

model = LIMOModel.from_pretrained("GAIR-NLP/LIMO")
result = model.solve("Find all values of x that satisfy x^2 + 3x + 2 = 0")
print(result)
```
