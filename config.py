import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Three-Models Comparison for Medical VQA (Question Type Aware)')
    parser.add_argument('--data_dir', type=str, default='./pvqa', help='Dataset root directory')
    parser.add_argument('--dict_path', type=str, default='./pvqa/pvqa_dictionary.pkl',
                        help='Vocabulary dictionary path')
    parser.add_argument('--embedding_path', type=str, default='./pvqa/glove_pvqa_300d.npy',
                        help='Pretrained embedding path')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum question length')
    parser.add_argument('--save_dir', type=str, default='./3models_comparison_results', help='Results save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4, help='DataLoader workers')
    return parser.parse_args()

FIXED_CONFIG = {
    'image_size': 224,
    'embed_dim': 300,
    'text_proj_dim': 300,
    'se_reduction': 16,
    'cnn_num_filters': 128,
    'cnn_filter_sizes': [2, 3, 4],
    'dropout_rate': 0.3,
    'lstm_hidden_dim': 384,
    'lstm_num_layers': 2
}