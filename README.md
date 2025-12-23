# Medical VQA: Multi-Model Comparison with Question Type Awareness
A PyTorch-based implementation for Medical Visual Question Answering (VQA), featuring three model architectures and question-type-aware evaluation (Yes/No vs. Open-ended questions). The framework supports comprehensive performance analysis with metrics like Accuracy, Exact Match, Macro F1, and BLEU score.

# Table of Contents
* Overview
* Key Features
* Model Architectures
* Dataset Preparation
* Installation
* Training & Evaluation
* Output Structure

# Overview
This project compares three VQA models tailored for medical imaging scenarios, with a focus on question type awareness—a critical distinction in medical VQA where answers often fall into binary (Yes/No) or descriptive (Open-ended) categories. The framework includes end-to-end training, validation, and evaluation pipelines, with detailed metrics for different question types.

# Key Features
Three Model Architectures: Baseline and improved models for fair comparison.
Question Type Classification: Automatically distinguishes Yes/No and Open-ended questions.
## Multi-Metric Evaluation:
Overall: Accuracy, Loss
Yes/No Questions: Accuracy
Open-ended Questions: Exact Match, Macro F1, BLEU-4
Reproducibility: Fixed random seeds and detailed configuration.
Visualization: Training curves, question-type performance comparison plots.
Modular Design: Easy to extend with new models, datasets, or metrics.

# Model Architectures
Baseline1: Resnet50 + Unidirectional LSTM
Baseline2: Resnet50 + TestCNN
Baseline1: Resnet50 + Bidirectional LSTM

# Dataset Preparation
We use a medical VQA dataset structured as follows (customize paths in config.py):
<img width="1442" height="378" alt="屏幕截图 2025-12-23 153214" src="https://github.com/user-attachments/assets/76ea7e03-0219-4bd3-bbd4-1afcb6771361" />
## Dataset preview
<img width="1228" height="518" alt="Requencies" src="https://github.com/user-attachments/assets/e68fecbc-2497-4628-b2d6-21761d9da313" />
<img width="1283" height="326" alt="屏幕截图 2025-12-19 143533" src="https://github.com/user-attachments/assets/d360867c-c9b8-4ea7-bec1-2a90cb7b8e9d" />
<img width="1223" height="733" alt="屏幕截图 2025-12-19 142357" src="https://github.com/user-attachments/assets/df2bba75-8d96-4b3e-a7c0-c96498ebf1b6" />




# Installation
Prerequisites
* Python 3.8+
* PyTorch 1.10+
* TorchVision 0.11+
* Other dependencies: pillow, numpy, nltk, tqdm, matplotlib, argparse, pickle-mixin

# Training & Evaluation
Modify hyperparameters via command-line arguments：
* python main.py --epochs 50 --batch_size 32 --lr 5e-5 --save_dir ./custom_results

# Output Structure
Results are saved in a timestamped directory (e.g., 3models_comparison_results/exp_20240520_143000_seed_42/)
## JSON Results Format
3models_results.json includes:
Training/validation loss/accuracy for each epoch.
Detailed metrics (per question type) for the best epoch of each model.
Configuration parameters (seed, batch size, learning rate, etc.).
