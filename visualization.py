import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_question_type_comparison(results, save_path):

    model_names = ['Baseline1', 'Baseline2', 'Proposed']
    model_keys = ['baseline1', 'baseline2', 'proposed']
    colors = ['#FF7F0E', '#2CA02C', '#1F77B4']

    yesno_accs = [results[key]['best_detailed']['yesno']['acc'] for key in model_keys]
    open_em = [results[key]['best_detailed']['open']['exact_match'] for key in model_keys]
    open_f1 = [results[key]['best_detailed']['open']['macro_f1'] for key in model_keys]
    open_bleu = [results[key]['best_detailed']['open']['bleu'] for key in model_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subgraph 1: Accuracy of Yes/No Questions
    axes[0].bar(model_names, yesno_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Yes/No Questions Performance', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    # 添加数值标签
    for i, acc in enumerate(yesno_accs):
        axes[0].text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Subgraph 2: Multi-index Comparison of Open-ended Questions
    x = np.arange(len(model_names))
    width = 0.25
    axes[1].bar(x - width, open_em, width, label='Exact Match', color=colors[0], alpha=0.8, edgecolor='black')
    axes[1].bar(x, open_f1, width, label='Macro F1', color=colors[1], alpha=0.8, edgecolor='black')
    axes[1].bar(x + width, open_bleu, width, label='BLEU', color=colors[2], alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Score (%)', fontsize=12)
    axes[1].set_title('Open-ended Questions Performance', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, (em, f1, bleu) in enumerate(zip(open_em, open_f1, open_bleu)):
        axes[1].text(i - width, em + 1, f'{em:.1f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i, f1 + 1, f'{f1:.1f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width, bleu + 1, f'{bleu:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{save_path}/question_type_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/question_type_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_3models_comparison(results, save_path):
    """Loss/Accuracy Comparison Curve"""
    epochs = list(range(1, len(results['baseline1']['train_loss']) + 1))
    plt.figure(figsize=(14, 6))

    colors = ['#FF7F0E', '#2CA02C', '#1F77B4']
    model_names = ['Baseline1 (ResNet50+UniLSTM)', 'Baseline2 (ResNet50+TextCNN)', 'Proposed (ResNet-SE+BiLSTM)']
    model_keys = ['baseline1', 'baseline2', 'proposed']

    # Subgraph 1: Loss Curve
    plt.subplot(1, 2, 1)
    for i, (key, name) in enumerate(zip(model_keys, model_names)):
        plt.plot(epochs, results[key]['train_loss'], label=f'{name} (Train)',
                 color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, results[key]['val_loss'], label=f'{name} (Val)',
                 color=colors[i], linestyle='--', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs[::2])

    # Subgraph 2: Accuracy Curve
    plt.subplot(1, 2, 2)
    for i, (key, name) in enumerate(zip(model_keys, model_names)):
        plt.plot(epochs, results[key]['train_acc'], label=f'{name} (Train)',
                 color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, results[key]['val_acc'], label=f'{name} (Val)',
                 color=colors[i], linestyle='--', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training & Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs[::2])

    plt.tight_layout()
    plt.savefig(f"{save_path}/3models_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/3models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()