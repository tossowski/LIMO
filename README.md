# LIMO: Less is More for Reasoning 🚀

<div align="center">

[![Conference](https://img.shields.io/badge/ARXIV-2024-blue)](https://arxiv.org/abs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/GAIR-NLP/LIMO.svg?style=social&label=Star&maxAge=2592000)](https://github.com/GAIR-NLP/LIMO)

*Less Training Data, Better Mathematical Reasoning*

</div>

## Overview

LIMO is a groundbreaking approach that challenges the conventional wisdom in mathematical reasoning. We demonstrate that complex reasoning capabilities can emerge from minimal but high-quality training data, achieving state-of-the-art performance while using just 1% of the training data required by previous approaches.

| 🔥 **Key Results** | Performance |
|-------------------|-------------|
| AIME Test         | **57.1%**   |
| MATH Test         | **94.8%**   |
| Training Samples  | Only 817    |

## Key Features

- 🎯 **Unprecedented Efficiency**: Achieves SOTA with just 817 training samples
- 🌟 **Strong Generalization**: Outperforms models trained on 100x more data
- 🔬 **Rigorous Methodology**: Comprehensive evaluation across 12 benchmarks
- 📦 **Complete Package**: Includes models, data, and evaluation pipelines

## Main Results

### In-Domain Performance


### Out-of-Domain Generalization

LIMO consistently outperforms existing approaches across various out-of-distribution scenarios, demonstrating robust generalization capabilities.

## Quick Start

```bash
# Install LIMO
pip install limo-math

# Basic usage
from limo import LIMOModel

model = LIMOModel.from_pretrained("GAIR-NLP/LIMO")
result = model.solve("Solve for x: x^2 + 3x + 2 = 0")



# Clone repository
git clone https://github.com/GAIR-NLP/LIMO.git
cd LIMO

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python evaluate.py --model limo-base --benchmark aime


@article{limo2024,
  title={LIMO: Less is More for Reasoning},
  author={},
  journal={arXiv preprint arXiv:},
  year={2024}
}
