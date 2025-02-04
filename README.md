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

| Benchmark | LIMO | Previous SOTA | Improvement |
|-----------|------|---------------|-------------|
| AIME | 57.1% | 6.5% | +50.6% |
| MATH | 94.8% | 59.2% | +35.6% |
| GSM8K | 92.3% | 78.4% | +13.9% |
| MATH-Basic | 96.5% | 84.7% | +11.8% |
| Algebra | 95.2% | 82.1% | +13.1% |
| Geometry | 93.7% | 76.8% | +16.9% |

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

### Quick Demo
```python
from limo import LIMOModel

model = LIMOModel.from_pretrained("GAIR-NLP/LIMO")
result = model.solve("Find all values of x that satisfy x^2 + 3x + 2 = 0")
print(result)
```

## Datasets

We release our datasets through Hugging Face 🤗:

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| `limo-high` | High-quality training set | 817 | [🤗](https://huggingface.co/datasets/GAIR-NLP/limo-high) |
| `limo-medium` | Medium-quality control set | 1,000 | [🤗](https://huggingface.co/datasets/GAIR-NLP/limo-medium) |
| `limo-basic` | Basic-quality baseline set | 1,000 | [🤗](https://huggingface.co/datasets/GAIR-NLP/limo-basic) |

### Dataset Format
Each sample follows this structure:
```json
{
"id": "LIMO_001",
"question": "Find the value of x...",
"solution": {
"reasoning_chain": ["Step 1...", "Step 2..."],
"final_answer": "42",
"quality_metrics": {
"chain_length": 5,
"step_count": 3,
"knowledge_coverage": 0.85,
"clarity_score": 0.92
}
},
"metadata": {
"source": "AIME",
"difficulty": "hard",
"topics": ["algebra", "geometry"]
}
}
```

### Loading Datasets
```python
from datasets import load_dataset

# Load high-quality dataset
dataset = load_dataset("GAIR-NLP/limo-high")

# Load all quality levels
datasets = {
"high": load_dataset("GAIR-NLP/limo-high"),
"medium": load_dataset("GAIR-NLP/limo-medium"),
"basic": load_dataset("GAIR-NLP/limo-basic")
}
```

## Training

### Basic Training
```bash
python training/train.py \
--model gpt-3.5-turbo \
--dataset limo-high \
--output_dir outputs/limo_base
```

### Advanced Training
```bash
python training/train.py \
--config configs/advanced_training.yaml
```

<details>
<summary>Show example configuration</summary>

```yaml
model:
name: gpt-3.5-turbo
tokenizer: gpt-3.5-turbo

training:
learning_rate: 1e-5
batch_size: 16
num_epochs: 10
warmup_steps: 100
gradient_accumulation: 4

evaluation:
eval_steps: 500
save_steps: 1000
metrics:
- accuracy
- reasoning_quality
```
</details>

### Custom Training
```python
from limo.training import LIMOTrainer

# Initialize trainer
trainer = LIMOTrainer(
model_name="gpt-3.5-turbo",
training_args={
"learning_rate": 1e-5,
"batch_size": 16,
"num_epochs": 10
}
)

# Train on custom dataset
trainer.train(
dataset=your_dataset,
output_dir="outputs/custom_model"
)
```

## Evaluation

### Basic Evaluation
```bash
# Evaluate on AIME
python evaluation/evaluate.py \
--model_path outputs/limo_base \
--benchmark aime

# Run full evaluation suite
python evaluation/evaluate.py \
--model_path outputs/limo_base \
--benchmark all
```

### Custom Evaluation
```python
from limo.evaluation import Evaluator

evaluator = Evaluator(
model_path="outputs/limo_base",
metrics=["accuracy", "reasoning_quality"]
)

results = evaluator.evaluate(
benchmark="math500",
split="test",
batch_size=32
)

# Generate evaluation report
evaluator.generate_report(
results=results,
output_path="results/evaluation_report.pdf"
)
```

## Model Zoo

Pre-trained models are available on Hugging Face 🤗:

| Model | Description | Size | Link |
|-------|-------------|------|------|
| LIMO-Base | Base model | 7B | [🤗](https://huggingface.co/GAIR-NLP/limo-base) |
| LIMO-Large | Large model | 13B | [🤗](https://huggingface.co/GAIR-NLP/limo-large) |

## Project Structure
```
📦 LIMO
┣ 📂 training # Training scripts
┃ ┣ 📜 train.py # Main training script
┃ ┣ 📜 trainer.py # Trainer class
┃ ┗ 📜 config.py # Training configs
┣ 📂 evaluation # Evaluation pipeline
┃ ┣ 📜 evaluate.py # Main evaluation script
┃ ┣ 📜 metrics.py # Evaluation metrics
┃ ┗ 📜 visualize.py # Result visualization
┣ 📂 models # Model implementations
┃ ┣ 📜 modeling.py # Core model architecture
┃ ┗ 📜 tokenizer.py # Tokenizer utilities
┣ 📂 configs # Configuration files
┣ 📂 tools # Utility tools
┗ 📂 docs # Documentation
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Setup development environment
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
pre-commit run --all-files
```

## Citation

```bibtex
@article{limo2024,
title={LIMO: Less is More for Mathematical Reasoning},
author={},
journal={arXiv preprint arXiv:},
year={2024}
}
```

## Core Team

- [Author 1](github_link) - Lead Developer
- [Author 2](github_link) - Research Lead
- [Author 3](github_link) - Technical Lead

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- 📧 Email: [contact email]
- 💬 Discord: [Join our community](discord_link)
- 🐦 Twitter: [@GAIR_NLP](https://twitter.com/GAIR_NLP)

---

<div align="center">
<b>LIMO is actively under development - Star us ⭐ to stay updated!</b>
</div>
