# Group-MAX-A-Cross-Modal-Fusion-Medical-VQA-Model-Based-on-ResNet-SE-and-Bidirectional-LSTM


ResNet-SE + BiLSTM Cross-Modal Fusion Medical VQA Model

A deep learning project for Medical Visual Question Answering (Med-VQA) that fuses visual features from pathology images and textual features from clinical questions to provide accurate answers.

Project Overview

Medical Visual Question Answering (Med-VQA) enables interactive information extraction from medical images via natural language queries, addressing critical needs in clinical diagnosis, medical training, and telemedicine. This project proposes a cross-modal fusion model combining ResNet-SE (image encoder with attention mechanism) and Bidirectional LSTM (BiLSTM) (text encoder) to handle the unique challenges of Med-VQA:

- High-resolution medical images with subtle pathological features

- Domain-specific terminology and complex anatomical relationships in clinical questions

- Data imbalance and scarcity of annotated medical datasets

We evaluate the proposed model against two baseline architectures on the PathVQA dataset, focusing on both closed-ended (Yes/No) and open-ended (What/Where) medical questions.

Dataset

PathVQA Dataset

- Source: He et al., 2020 (large-scale pathology-focused Med-VQA dataset)

- Scale: 32,799 question-answer pairs + 4,998 pathology images

- Question Distribution:

Question Type

Count (Percentage)

Task Type

Yes/No 16,334 (49.8%)

Closed-ended

What 13,402 (40.9%)

Open-ended

Where 1,268 (4.0%)

Open-ended

How

1,014 (3.0%)

Open-ended

Others

781 (2.3%)

Open-ended

- Data Split: Training (60%) → 19,755 pairs; Validation (20%) → 6,279 pairs; Test (20%) → 6,761 pairs

- Key Feature: Includes both closed-ended (binary diagnosis) and open-ended (identification) questions, with long-tailed answer distribution (mimicking real clinical scenarios)

Data Preprocessing

Image Preprocessing

1. Resizing: Training (random resized crop to 224x224, scale [0.8,1.0]); Validation (fixed 224x224)

2. Augmentation: Random horizontal flip (p=0.5), ±15° rotation, color jitter (brightness/contrast/saturation/hue adjustments)

3. Normalization: Tensor conversion + ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

Text Preprocessing

1. Lowercasing (case insensitivity)

2. Tokenization (whitespace splitting)

3. Padding/Truncation (fixed length = 50 tokens)

4. Vocabulary Mapping: Pre-trained dictionary (4,631 unique tokens) + GloVe 300D embeddings

5. Special Tokens: Padding (<PAD>), Unknown (<UNK>)

Answer Preprocessing

- Mapping to integer labels (single-answer samples only)

- Shared answer-to-label dictionary across training/validation/test sets

Model Architecture

Three models are implemented for comparative analysis, each consisting of Image Encoder + Text Encoder + Multimodal Fusion Layer:

Component

Baseline 1 (ResNet50 + UniLSTM)

Baseline 2 (ResNet-SE + TextCNN)

Proposed Model (ResNet-SE + BiLSTM)

Image Encoder

ResNet50 (ImageNet pre-trained)

ResNet-SE (ResNet50 + SE Attention)

ResNet-SE (same as Baseline 2)

Text Encoder

UniLSTM (512 hidden units, 300D output)

TextCNN (filters: 2/3/4-gram, 128 filters each)

BiLSTM (2 layers, 384 hidden units/direction, 300D output)

Fusion Layer

Concatenation (2048 + 300 = 2348D) → FCN

Same as Baseline 1

Same as Baselines

Output Layer

Softmax (answer label prediction)

Same as Baseline 1

Same as Baselines

Key Design Choices

- ResNet-SE: Residual connections solve gradient vanishing; SE attention enhances diagnostic features and suppresses noise.

- BiLSTM: Captures bidirectional semantic dependencies in medical questions (e.g., "GMS-stained" + "microorganism" relationships).

- Cross-Modal Alignment: Fuses focused visual features with comprehensive text semantics for accurate QA.

Experimental Setup

Evaluation Metrics

Task Type

Metrics

Closed-ended

Accuracy (primary)

Open-ended

Exact Match (EM) Accuracy, Macro-F1, BLEU

Generalization

Validation Accuracy

Reproducibility Guarantees

- Fixed random seeds (PyTorch, NumPy)

- Unified data preprocessing pipeline

- Identical training config: Optimizer (Adam), Learning Rate (1e-4), Batch Size (32), Epochs (50)

- Publicly available dataset and pre-trained weights

Preliminary Results

Closed-ended Questions (Yes/No)

Model

Best Validation Accuracy

Baseline 1 (ResNet50 + UniLSTM)

32.07%

Baseline 2 (ResNet-SE + TextCNN)

67.95%

Proposed Model (ResNet-SE + BiLSTM)

67.98%

Open-ended Questions

Model

EM Accuracy (%)

Token-level F1 (%)

BLEU (%)


Key Insights

1. SE Attention is Critical: Drives 35%+ accuracy improvement (Baseline 1 → Baseline 2/Proposed Model).

2. Text Encoder Impact: BiLSTM outperforms TextCNN slightly on open-ended tasks (better long-distance semantic understanding).

3. Closed-ended Dominance: All models perform better on Yes/No questions (clinical priority for binary diagnosis).

Installation

Prerequisites

- Python 3.8+

- PyTorch 1.10+

- TorchVision 0.11+

- Other dependencies:

pip install numpy pandas matplotlib scikit-learn nltk pillow

Dataset Download

1. Download PathVQA dataset from official source (or contact authors for access).

2. Organize dataset into the following structure:

data/
├── train/
│   ├── images/       # Training images (JPEG/PNG)
│   └── qa_pairs.csv  # Training question-answer pairs
├── val/
│   ├── images/       # Validation images
│   └── qa_pairs.csv
└── test/
    ├── images/       # Test images
    └── qa_pairs.csv

Model Weights

- Pre-trained ResNet-SE weights: Auto-downloaded via TorchVision (ImageNet initialization).

- GloVe 300D embeddings: Download from GloVe official site and place in embeddings/ folder.

Usage

Training

# Train Proposed Model (ResNet-SE + BiLSTM)
python train.py --model proposed --epochs 50 --batch_size 32 --lr 1e-4

# Train Baseline 1
python train.py --model baseline1 --epochs 50 --batch_size 32 --lr 1e-4

# Train Baseline 2
python train.py --model baseline2 --epochs 50 --batch_size 32 --lr 1e-4

Inference

# Run inference on test set
python infer.py --model proposed --weights_path ./checkpoints/proposed_best.pth --output ./results/proposed_test.csv

Visualization

# Plot training/validation accuracy curves
python visualize.py --log_path ./logs/proposed_train.log --save_path ./plots/accuracy_curve.png

Project Structure

Med-VQA-ResNet-SE-BiLSTM/
├── data/               # Dataset (train/val/test)
├── embeddings/         # GloVe embeddings
├── checkpoints/        # Trained model weights
├── logs/               # Training logs
├── plots/              # Visualization results
├── results/            # Inference outputs
├── src/
│   ├── data_prep.py    # Data preprocessing functions
│   ├── models.py       # Model definitions (ResNet-SE, BiLSTM, etc.)
│   ├── train_utils.py  # Training/validation utilities
│   └── metrics.py      # Evaluation metrics (EM, F1, BLEU)
├── train.py            # Training script
├── infer.py            # Inference script
├── visualize.py        # Visualization script
└── README.md           # Project documentation

Team

Name

Matrix No.

Role

HUANG BANGNING

23105394

Model Design & Training

WU QI

24083398

Data Preprocessing & Evaluation

License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Acknowledgments

- PathVQA dataset authors: He et al., 2020 (UCSD AI4H Lab)

- GloVe embeddings: Stanford NLP Group

- Pre-trained ResNet weights: TorchVision

Future Work

1. Improve open-ended question performance (address long-tailed answer distribution).

2. Explore advanced fusion strategies (e.g., attention-based fusion instead of concatenation).

3. Extend to multi-modal data (e.g., combine pathology images with electronic health records).

4. Deploy as a web-based tool for clinical use.

For questions or issues, please open an issue in the GitHub repository or contact the team members.
