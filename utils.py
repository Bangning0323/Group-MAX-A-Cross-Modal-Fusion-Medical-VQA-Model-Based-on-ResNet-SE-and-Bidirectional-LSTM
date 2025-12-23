import os
import random
import numpy as np
import torch
import pickle
import json
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def set_seed(seed=42):
    """Fix all random seeds to ensure the reproducibility of the experiment"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_official_vocab_and_embedding(dict_path, embedding_path):
    """Load the vocabulary and pre-trained embedding matrix"""
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary not found: {dict_path}")
    with open(dict_path, 'rb') as f:
        official_dict = pickle.load(f)

    word2idx = {}
    if isinstance(official_dict, list):
        for sub_dict in official_dict:
            if isinstance(sub_dict, dict):
                for word, idx in sub_dict.items():
                    if word not in word2idx:
                        word2idx[word] = idx
    elif isinstance(official_dict, dict):
        word2idx = official_dict

    embedding_matrix = np.load(embedding_path)
    vocab_size = len(word2idx)
    embed_vocab_size, embed_dim = embedding_matrix.shape

    if vocab_size != embed_vocab_size:
        adjusted_embedding = np.zeros((vocab_size, embed_dim))
        for word, idx in word2idx.items():
            if idx < embed_vocab_size:
                adjusted_embedding[idx] = embedding_matrix[idx]
            else:
                adjusted_embedding[idx] = np.random.normal(0, 0.01, embed_dim)
        embedding_matrix = adjusted_embedding


    for special_token in ['<PAD>', '<UNK>']:
        if special_token not in word2idx:
            new_idx = len(word2idx)
            word2idx[special_token] = new_idx
            special_embedding = np.random.normal(0, 0.01, embed_dim)
            embedding_matrix = np.vstack([embedding_matrix, special_embedding])

    return word2idx, embedding_matrix


def compute_open_metrics(pred_answers, true_answers):
    """Calculating evaluation metrics for open-ended questions: Exact Match, Macro F1, BLEU"""
    # 1. Exact Match
    exact_match = sum(
        1 for pred, true in zip(pred_answers, true_answers) if pred.strip().lower() == true.strip().lower()) / len(
        pred_answers)

    # 2. Macro-averaged F1
    def tokenize(text):
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        return set(nltk.word_tokenize(text)) if text else set()

    f1_scores = []
    for pred, true in zip(pred_answers, true_answers):
        pred_tokens = tokenize(pred)
        true_tokens = tokenize(true)
        if len(pred_tokens) == 0 and len(true_tokens) == 0:
            f1_scores.append(1.0)
        elif len(pred_tokens) == 0 or len(true_tokens) == 0:
            f1_scores.append(0.0)
        else:
            intersection = len(pred_tokens & true_tokens)
            precision = intersection / len(pred_tokens)
            recall = intersection / len(true_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
    macro_f1 = np.mean(f1_scores)

    # 3. BLEU Score（4-gram）
    def bleu_single(pred, true):
        pred_tokens = nltk.word_tokenize(pred.lower().strip())
        true_tokens = [nltk.word_tokenize(true.lower().strip())]
        return sentence_bleu(true_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    bleu_scores = [bleu_single(pred, true) for pred, true in zip(pred_answers, true_answers)]
    bleu_score = np.mean(bleu_scores)

    return exact_match * 100, macro_f1 * 100, bleu_score * 100


def count_model_params(model, model_name):
    """Number of model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} - Total params: {total:,}, Trainable params: {trainable:,}")
    return total, trainable


def save_results(results, save_path):
    with open(os.path.join(save_path, '3models_results.json'), 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)


def print_summary_report(results):
    model_keys = ['baseline1', 'baseline2', 'proposed']
    model_names = [
        'Baseline1 (ResNet50+UniLSTM)',
        'Baseline2 (ResNet50+TextCNN)',
        'Proposed (ResNet-SE+BiLSTM)'
    ]
    best_metrics = results['best_metrics']

    print("\n" + "=" * 80)
    print("Result Summary Report")
    print("=" * 80)
    for key, name in zip(model_keys, model_names):
        metrics = best_metrics[key]
        detailed = metrics['best_detailed']
        print(f"\n{metrics['name']}:")
        print(f"  Best validation accuracy: {metrics['best_val_acc']:.2f}% (Epoch {metrics['best_epoch']})")
        print(f"  Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"  Yes/No question - ACC: {detailed['yesno']['acc']:.2f}% (Samples: {detailed['yesno']['samples']})")
        print(f"  Open-ended question - Exact Match: {detailed['open']['exact_match']:.2f}%")
        print(f"                          Macro F1: {detailed['open']['macro_f1']:.2f}%")
        print(
            f"                              BLEU: {detailed['open']['bleu']:.2f}% (Samples: {detailed['open']['samples']})")

    improvement1 = best_metrics['proposed']['best_val_acc'] - best_metrics['baseline1']['best_val_acc']
    improvement2 = best_metrics['proposed']['best_val_acc'] - best_metrics['baseline2']['best_val_acc']
    print(f"\nPerformance Improvement：")
    print(f"  Proposed vs Baseline1: +{improvement1:.2f}%")
    print(f"  Proposed vs Baseline2: +{improvement2:.2f}%")
    print("=" * 80)