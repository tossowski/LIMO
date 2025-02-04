# LIMO: Less is More for Reasoning

<div align="center">

[![Conference](https://img.shields.io/badge/ARXIV-2024-blue)](https://arxiv.org/abs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🌟 Highlights

- 🚀 **Breakthrough in Mathematical Reasoning**: Achieves 57.1% accuracy on AIME and 94.8% on MATH with only 817 training samples
- 💡 **Challenge to Conventional Wisdom**: Demonstrates that complex reasoning abilities can emerge from minimal but high-quality training data
- 🌍 **Strong Generalization**: Outperforms models trained on 100x more data across diverse out-of-distribution scenarios
- 🔍 **Novel Hypothesis**: Introduces the LIMO Hypothesis explaining how sophisticated reasoning emerges through minimal but precise demonstrations

## 📊 Main Results

![Main Results](assets/main_results.png)

LIMO significantly outperforms existing approaches while using just 1% of the training data:

| Model | AIME | MATH | Training Samples |
|-------|------|------|-----------------|
| LIMO (Ours) | 57.1% | 94.8% | 817 |
| Previous SOTA | 6.5% | 59.2% | 100k+ |

## 🛠️ Installation

```bash
git clone https://github.com/GAIR-NLP/LIMO.git
cd LIMO
pip install -r requirements.txt
