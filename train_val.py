import torch
from tqdm import tqdm
from utils import compute_open_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, questions, labels, _ in progress_bar:  # 忽略问题类型
        images, questions, labels = images.to(device), questions.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / (progress_bar.n + 1)
        avg_acc = 100.0 * correct / total
        progress_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Acc': f"{avg_acc:.2f}%"})
    final_acc = 100.0 * correct / total
    return total_loss / len(dataloader), final_acc

def evaluate_by_question_type(model, dataloader, criterion, device, label_to_answer):
    """Evaluate model performance by question type"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0


    yesno_correct = 0
    yesno_samples = 0
    open_pred_answers = []
    open_true_answers = []

    with torch.no_grad():
        for images, questions, labels, q_types in tqdm(dataloader, desc="Evaluating by question type"):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Prediction result
            _, preds = torch.max(outputs, 1)
            batch_correct = (preds == labels).cpu().numpy()
            total_correct += batch_correct.sum()
            total_samples += len(labels)

            # Statistics by question types
            for idx, q_type in enumerate(q_types):
                if q_type == 'yesno':
                    yesno_samples += 1
                    yesno_correct += batch_correct[idx]
                else:  # open
                    pred_answer = label_to_answer[preds[idx].item()]
                    true_answer = label_to_answer[labels[idx].item()]
                    open_pred_answers.append(pred_answer)
                    open_true_answers.append(true_answer)

    total_acc = (total_correct / total_samples) * 100
    total_loss_avg = total_loss / len(dataloader)

    # Calculate the Yes/No
    yesno_acc = (yesno_correct / yesno_samples) * 100 if yesno_samples > 0 else 0.0

    # Calculate open
    open_exact_match = open_macro_f1 = open_bleu = 0.0
    if len(open_pred_answers) > 0:
        open_exact_match, open_macro_f1, open_bleu = compute_open_metrics(open_pred_answers, open_true_answers)

    return {
        'total_loss': total_loss_avg,
        'total_acc': total_acc,
        'yesno': {'acc': yesno_acc, 'samples': yesno_samples},
        'open': {
            'exact_match': open_exact_match,
            'macro_f1': open_macro_f1,
            'bleu': open_bleu,
            'samples': len(open_pred_answers)
        }
    }

def validate(model, dataloader, criterion, device, label_to_answer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, labels, _ in tqdm(dataloader, desc="Validating", leave=False):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    total_acc = 100.0 * correct / total
    total_loss_avg = total_loss / len(dataloader)

    detailed_metrics = evaluate_by_question_type(model, dataloader, criterion, device, label_to_answer)
    return total_loss_avg, total_acc, detailed_metrics