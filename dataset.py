import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
from question_classifier import QuestionTypeClassifier


class MedicalVQADataset(Dataset):
    def __init__(self, qa_path, img_dir, word2idx, answer_to_label=None, max_length=50, transform=None):
        with open(qa_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.question_classifier = QuestionTypeClassifier()
        self.data = []
        self.question_types = []

        # Filter samples with multiple answers or no answers
        for item in raw_data:
            img_id = item['img_id']
            question = item['sent']
            label_dict = item['label']
            if len(label_dict) != 1:
                continue  #Filter samples with multiple answers or no answers
            answer = list(label_dict.keys())[0]

            q_type = self.question_classifier.classify(question)
            self.data.append({'image_id': img_id, 'question': question, 'answer': answer})
            self.question_types.append(q_type)

        self.img_dir = img_dir
        self.word2idx = word2idx
        self.max_length = max_length
        self.transform = transform

        if answer_to_label is not None:
            self.answer_to_label = answer_to_label
            self.label_to_answer = {v: k for k, v in self.answer_to_label.items()}
            original_len = len(self.data)
            valid_indices = [i for i, item in enumerate(self.data) if item['answer'] in self.answer_to_label]
            self.data = [self.data[i] for i in valid_indices]
            self.question_types = [self.question_types[i] for i in valid_indices]
            filtered_count = original_len - len(self.data)
            if filtered_count > 0:
                print(f"Filtered {filtered_count} samples (unknown answers)")
        else:
            all_answers = [item['answer'] for item in self.data]
            unique_answers = sorted(set(all_answers))
            self.answer_to_label = {ans: idx for idx, ans in enumerate(unique_answers)}
            self.label_to_answer = {v: k for k, v in self.answer_to_label.items()}

        yesno_count = self.question_types.count('yesno')
        open_count = self.question_types.count('open')
        print(f"Question type distribution - Yes/No: {yesno_count}, Open: {open_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = item['image_id']
        question = item['question'].lower().split()
        answer = item['answer']
        q_type = self.question_types[idx]

        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
        img_path = None
        for ext in extensions:
            candidate = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_id}")

        # Image preprocessing
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Text preprocessing
        indices = []
        for token in question[:self.max_length]:
            indices.append(self.word2idx.get(token, self.word2idx.get('<UNK>', 0)))
        while len(indices) < self.max_length:
            indices.append(self.word2idx.get('<PAD>', 0))
        question_tensor = torch.tensor(indices, dtype=torch.long)

        label = self.answer_to_label[answer]
        return image, question_tensor, label, q_type


# Image preprocessing
def get_image_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])