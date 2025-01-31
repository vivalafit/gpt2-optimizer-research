# GPT-2 Optimizer Research

## Project Overview
This repository focuses on fine-tuning and optimizing the GPT-2 model for ISO-standard requirements generation. The project explores model evaluation, perplexity analysis, and optimization techniques.
Currently its only basic work - optimisation will come in next commits.

## Directory Structure
- `src/train_model.py` - Script for fine-tuning the GPT-2 model.
- `src/evaluate_model.py` - Script for evaluating model performance.
- `train_iso.csv`, `val_iso.csv`, `test_iso.csv` - Datasets used for training, validation, and testing.

## How to Run
1. Create a virtual environment:
   ```bash
   python -m venv test-environment
   source test-environment/bin/activate  # Or `Scripts\activate` on Windows