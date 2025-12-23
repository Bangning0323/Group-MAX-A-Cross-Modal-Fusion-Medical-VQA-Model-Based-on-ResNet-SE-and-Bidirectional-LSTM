import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from config import get_args, FIXED_CONFIG
from utils import set_seed, load_official_vocab_and_embedding, count_model_params, save_results, print_summary_report
from dataset import MedicalVQADataset, get_image_transforms
from models import ModelBaseline1, ModelBaseline2, ModelProposed
from train_val import train_one_epoch, validate
from visualization import plot_3models_comparison, plot_question_type_comparison

def main():
    args = get_args()
    set_seed(args.seed)
    print(f"Random seed fixed to: {args.seed}")

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_save_dir = os.path.join(args.save_dir, f'exp_{timestamp}_seed_{args.seed}')
    os.makedirs(exp_save_dir, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 1. Load the vocabulary and embedding matrix
    print("\n[Step 1/5] Loading vocabulary and embedding matrix...")
    word2idx, embedding_matrix = load_official_vocab_and_embedding(
        dict_path=args.dict_path,
        embedding_path=args.embedding_path
    )
    print(f"Vocab size: {len(word2idx)}, Embedding dimension: {embedding_matrix.shape[1]}")

    # 2. Dataset path configuration and verification
    print("\n[Step 2/5] Initializing datasets...")
    train_qa_path = os.path.join(args.data_dir, 'qas', 'train_vqa.pkl')
    val_qa_path = os.path.join(args.data_dir, 'qas', 'val_vqa.pkl')
    train_img_dir = os.path.join(args.data_dir, 'images', 'train')
    val_img_dir = os.path.join(args.data_dir, 'images', 'val')

    for path in [train_qa_path, val_qa_path, train_img_dir, val_img_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

    # 3. Build datasets and data loaders
    train_transform = get_image_transforms(image_size=FIXED_CONFIG['image_size'], is_train=True)
    val_transform = get_image_transforms(image_size=FIXED_CONFIG['image_size'], is_train=False)

    train_dataset = MedicalVQADataset(
        qa_path=train_qa_path,
        img_dir=train_img_dir,
        word2idx=word2idx,
        max_length=args.max_length,
        transform=train_transform
    )
    val_dataset = MedicalVQADataset(
        qa_path=val_qa_path,
        img_dir=val_img_dir,
        word2idx=word2idx,
        answer_to_label=train_dataset.answer_to_label,
        max_length=args.max_length,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Dataset statistical information
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Number of classes (answers): {len(train_dataset.answer_to_label)}")

    # 4. Initialize the model, optimizer, and loss function
    print("\n[Step 3/5] Initializing models...")
    num_classes = len(train_dataset.answer_to_label)

    # Model initialization
    model_baseline1 = ModelBaseline1(embedding_matrix, num_classes).to(device)
    model_baseline2 = ModelBaseline2(embedding_matrix, num_classes).to(device)
    model_proposed = ModelProposed(embedding_matrix, num_classes).to(device)

    # Optimizer
    optimizer_b1 = optim.Adam(model_baseline1.parameters(), lr=args.lr)
    optimizer_b2 = optim.Adam(model_baseline2.parameters(), lr=args.lr)
    optimizer_p = optim.Adam(model_proposed.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler_b1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_b1, T_max=args.epochs)
    scheduler_b2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_b2, T_max=args.epochs)
    scheduler_p = optim.lr_scheduler.CosineAnnealingLR(optimizer_p, T_max=args.epochs)

    # Loss function (shared by the three groups of models)
    criterion = nn.CrossEntropyLoss()

    count_model_params(model_baseline1, "Baseline1")
    count_model_params(model_baseline2, "Baseline2")
    count_model_params(model_proposed, "Proposed Model")

    # 5. Initialize the result storage
    results = {
        'baseline1': {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'detailed_metrics': []
        },
        'baseline2': {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'detailed_metrics': []
        },
        'proposed': {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'detailed_metrics': []
        },
        'config': vars(args),
        'fixed_config': FIXED_CONFIG
    }

    # 6. Start training
    print("\n[Step 4/5] Starting training...")
    for epoch in range(args.epochs):
        print(f"\n{'=' * 50} Epoch {epoch + 1:2d}/{args.epochs} {'=' * 50}")

        # Train three groups of models
        print("\n--- Training Baseline1 (ResNet50+UniLSTM) ---")
        train_loss_b1, train_acc_b1 = train_one_epoch(model_baseline1, train_loader, criterion, optimizer_b1, device)
        print("\n--- Training Baseline2 (ResNet50+TextCNN) ---")
        train_loss_b2, train_acc_b2 = train_one_epoch(model_baseline2, train_loader, criterion, optimizer_b2, device)
        print("\n--- Training Proposed Model (ResNet-SE+BiLSTM) ---")
        train_loss_p, train_acc_p = train_one_epoch(model_proposed, train_loader, criterion, optimizer_p, device)

        # Validation
        print("\n--- Validating Baseline1 ---")
        val_loss_b1, val_acc_b1, detailed_b1 = validate(model_baseline1, val_loader, criterion, device,
                                                        train_dataset.label_to_answer)
        print("\n--- Validating Baseline2 ---")
        val_loss_b2, val_acc_b2, detailed_b2 = validate(model_baseline2, val_loader, criterion, device,
                                                        train_dataset.label_to_answer)
        print("\n--- Validating Proposed Model ---")
        val_loss_p, val_acc_p, detailed_p = validate(model_proposed, val_loader, criterion, device,
                                                     train_dataset.label_to_answer)

        # Renew
        scheduler_b1.step()
        scheduler_b2.step()
        scheduler_p.step()

        # Save result
        results['baseline1']['train_loss'].append(train_loss_b1)
        results['baseline1']['train_acc'].append(train_acc_b1)
        results['baseline1']['val_loss'].append(val_loss_b1)
        results['baseline1']['val_acc'].append(val_acc_b1)
        results['baseline1']['detailed_metrics'].append(detailed_b1)

        results['baseline2']['train_loss'].append(train_loss_b2)
        results['baseline2']['train_acc'].append(train_acc_b2)
        results['baseline2']['val_loss'].append(val_loss_b2)
        results['baseline2']['val_acc'].append(val_acc_b2)
        results['baseline2']['detailed_metrics'].append(detailed_b2)

        results['proposed']['train_loss'].append(train_loss_p)
        results['proposed']['train_acc'].append(train_acc_p)
        results['proposed']['val_loss'].append(val_loss_p)
        results['proposed']['val_acc'].append(val_acc_p)
        results['proposed']['detailed_metrics'].append(detailed_p)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f"Baseline1 - Train Loss: {train_loss_b1:.4f}, Train Acc: {train_acc_b1:.2f}%, Val Loss: {val_loss_b1:.4f}, Val Acc: {val_acc_b1:.2f}%")
        print(f"  Yes/No Acc: {detailed_b1['yesno']['acc']:.2f}%, Open EM: {detailed_b1['open']['exact_match']:.2f}%")
        print(
            f"Baseline2 - Train Loss: {train_loss_b2:.4f}, Train Acc: {train_acc_b2:.2f}%, Val Loss: {val_loss_b2:.4f}, Val Acc: {val_acc_b2:.2f}%")
        print(f"  Yes/No Acc: {detailed_b2['yesno']['acc']:.2f}%, Open EM: {detailed_b2['open']['exact_match']:.2f}%")
        print(
            f"Proposed  - Train Loss: {train_loss_p:.4f}, Train Acc: {train_acc_p:.2f}%, Val Loss: {val_loss_p:.4f}, Val Acc: {val_acc_p:.2f}%")
        print(f"  Yes/No Acc: {detailed_p['yesno']['acc']:.2f}%, Open EM: {detailed_p['open']['exact_match']:.2f}%")

    model_keys = ['baseline1', 'baseline2', 'proposed']
    model_names = [
        'Baseline1 (ResNet50+UniLSTM)',
        'Baseline2 (ResNet50+TextCNN)',
        'Proposed (ResNet-SE+BiLSTM)'
    ]
    best_metrics = {}
    for key, name in zip(model_keys, model_names):
        val_accs = results[key]['val_acc']
        best_val_acc_idx = val_accs.index(max(val_accs))
        best_metrics[key] = {
            'name': name,
            'best_val_acc': round(val_accs[best_val_acc_idx], 2),
            'best_val_loss': round(results[key]['val_loss'][best_val_acc_idx], 4),
            'best_epoch': best_val_acc_idx + 1,
            'best_detailed': results[key]['detailed_metrics'][best_val_acc_idx]
        }
    results['best_metrics'] = best_metrics

    print("\n[Step 5/5] Saving results and plots...")
    save_results(results, exp_save_dir)
    plot_3models_comparison(results, exp_save_dir)
    plot_question_type_comparison(results, exp_save_dir)

    best_val_acc_p = max(results['proposed']['val_acc'])
    best_epoch_p = results['proposed']['val_acc'].index(best_val_acc_p) + 1
    torch.save({
        'epoch': best_epoch_p,
        'model_state_dict': model_proposed.state_dict(),
        'optimizer_state_dict': optimizer_p.state_dict(),
        'best_val_acc': best_val_acc_p,
        'best_detailed_metrics': results['proposed']['detailed_metrics'][best_epoch_p - 1],
        'word2idx': word2idx,
        'answer_to_label': train_dataset.answer_to_label,
        'label_to_answer': train_dataset.label_to_answer,
        'embedding_matrix': embedding_matrix
    }, f"{exp_save_dir}/best_proposed_model.pth")

    print_summary_report(results)

    print(f"\nTraining completed!")
    print(f"Results saved to: {exp_save_dir}")
    print(f"Best proposed model saved to: {exp_save_dir}/best_proposed_model.pth")

if __name__ == '__main__':
    main()