# Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection

This is the official repository for **"Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection"**, submitted to and under review at The Fourteenth International Conference on Learning Representations (ICLR 2026).

## Overview

This repository implements a novel approach to enhance the trustworthiness of fine-tuned Large Language Models (LLMs) through regularized subset selection. The method leverages Data Attribution and Determinantal Point Processes (DPP) to identify and utilize the most influential training examples for improving model behavior on trustworthiness tasks.

## Requirements

- **Python**: 3.9+
- **Dependencies**: See `requirements.txt` for complete list

### Key Dependencies
- `torch==2.7.1`
- `transformers==4.52.4`
- `kronfluence==1.0.1`
- `accelerate==1.7.0`
- `datasets==3.6.0`

## Installation

1. Clone the repository:
```bash
git clone https://anonymous.4open.science/r/tracing-llm-trust-7806
cd tracing-llm-trust
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with:
HUB_CACHE=/path/to/your/cache
HUB_TOKEN=your_huggingface_token
```

## Usage

The pipeline consists of five main steps:

### Step 1: Compute Influence Factors

Compute EKFAC factors and influence scores for all datapoints in the Dahoas/Static-HH dataset using Kronfluence (Grosse et al. 2023).

```bash
python3 compute_factors_and_scores.py
```

**Configuration**: Edit the script to specify:
- `MODEL_NAMES`: List of HuggingFace model names
- `TASK_NAME`: Task type ("bias", "ethics", "truth")
- `MODEL_FAMILY`: Model family ("pythia" or "qwen")

This populates `ekfac_experiments/influence_results/` with EKFAC factors and scores.

### Step 2: Generate Training Representations

Extract model representations for DPP computation.

```bash
python3 representations/train_reps.py
```

**Configuration**: Set `MODEL_NAME` and other parameters in the script.

### Step 3: Compute DPP Indices

Generate Determinantal Point Process indices for regularized subset selection.

```bash
python3 dpp/greedy_dpp.py
```

**Prerequisites**: Run Step 2 first to populate the `representations/` folder.

### Step 4: Synthesize Gradients

Generate gradients for non-embedding layers in the final transformer blocks of Pythia or Qwen2.5 models.

```bash
python3 make_gradients.py
```

**Configuration**: Edit the script to specify:
- `MODEL_NAMES`: Target models
- `TASK_NAME`: Evaluation task
- `BOTTOM_K`: Number of influential points to use
- `SCORES_TYPE`: Scoring method ("pure_ekfac", "dpp", "random")

### Step 5: Parameter Update via Natural Gradient Ascent

Update model parameters using natural gradient ascent with the synthesized gradients.

```bash
python3 gradient_ascent_and_save.py
```

**Configuration**: Set:
- `MODEL_NAME`: Target model
- `LEARNING_RATE`: Learning rate for gradient ascent
- `BOTTOM_K`: Number of points used for gradients
- `SCORES_TYPE`: Scoring method used

### Step 6: Evaluation

Run evaluations on updated models using the evaluation scripts.

```bash
# Anthropic evaluation loss (Perplexity)
python3 evaluation_scripts/anthropic_eval.py

# Task-specific evaluations (Log-Odds)
python3 evaluation_scripts/bias_loss.py
python3 evaluation_scripts/ethics_loss.py
python3 evaluation_scripts/truth_loss.py
```

## Project Structure

```
tracing-llm-trust/
├── compute_factors_and_scores.py    # Step 1: Compute influence factors
├── make_gradients.py                # Step 4: Synthesize gradients
├── gradient_ascent_and_save.py      # Step 5: Parameter updates
├── requirements.txt                 # Dependencies
├── data/                           # Datasets used by Utils
│   ├── ethics/                     # Ethics evaluation data
│   └── truthfulness/              # Truthfulness evaluation data
├── dpp/                           # Determinantal Point Process
│   └── greedy_dpp.py             # DPP index computation
├── ekfac_experiments/             # Influence analysis results
│   └── influence_results/        # EKFAC factors and scores
├── evaluation_scripts/            # Model evaluation
│   ├── anthropic_eval.py         # Anthropic evaluation
│   ├── bias_loss.py             # Bias evaluation
│   ├── ethics_loss.py           # Ethics evaluation
│   └── truth_loss.py            # Truthfulness evaluation
├── representations/              # Model representations
│   └── train_reps.py            # Representation extraction
├── tasks/                        # Task definitions used by Kronfluence
│   ├── backward_task.py         # Backward task implementation
│   ├── bias_task.py            # Bias task
│   ├── ethics_task.py          # Ethics task
│   └── truth_task.py           # Truthfulness task
└── utils/                       # Utility functions
    └── performance_functions/   # Performance evaluation utilities
```

## Key Features

- **Influence-based Selection**: Uses Kronfluence to identify influential training examples
- **DPP Regularization**: Implements Determinantal Point Processes for diverse subset selection
- **Multi-task Support**: Supports bias, ethics, and truthfulness evaluation tasks
- **Model Flexibility**: Compatible with Pythia and Qwen2.5 model families
- **Comprehensive Evaluation**: Includes Anthropic and task-specific evaluation metrics
