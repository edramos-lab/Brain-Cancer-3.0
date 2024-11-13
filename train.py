import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import wandb
import argparse

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, hgg_dir, lgg_dir, transform=None):
        self.hgg_dir = hgg_dir
        self.lgg_dir = lgg_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        for img_name in os.listdir(self.hgg_dir):
            img_path = os.path.join(self.hgg_dir, img_name)
            self.images.append(img_path)
            self.labels.append(1)
        for img_name in os.listdir(self.lgg_dir):
            img_path = os.path.join(self.lgg_dir, img_name)
            self.images.append(img_path)
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CustomTestDataset(Dataset):
    def __init__(self, hgg_test_dir, lgg_test_dir, num_images_per_class=200, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        hgg_images = os.listdir(hgg_test_dir)[:num_images_per_class]
        for img in hgg_images:
            self.images.append(os.path.join(hgg_test_dir, img))
            self.labels.append(1)
        lgg_images = os.listdir(lgg_test_dir)[:num_images_per_class]
        for img in lgg_images:
            self.images.append(os.path.join(lgg_test_dir, img))
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def prepare_datasets(dataset_dir, dataset_ratio):
    hgg_dir = os.path.join(dataset_dir, 'training/HGG_training')
    lgg_dir = os.path.join(dataset_dir, 'training/LGG_training')
    hgg_test_dir = os.path.join(dataset_dir, 'testing/HGG_test')
    lgg_test_dir = os.path.join(dataset_dir, 'testing/LGG_test')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(hgg_dir, lgg_dir, transform=train_transform)
    test_dataset = CustomTestDataset(hgg_test_dir, lgg_test_dir, num_images_per_class=200, transform=test_transform)

    num_train_samples = int(len(train_dataset) * dataset_ratio)
    min_test_samples = 20
    num_test_samples = max(int(len(test_dataset) * dataset_ratio), min_test_samples)

    train_subset, _ = random_split(train_dataset, [num_train_samples, len(train_dataset) - num_train_samples])
    test_subset, _ = random_split(test_dataset, [num_test_samples, len(test_dataset) - num_test_samples])

    return train_subset, test_subset

def train_and_evaluate(args):
    train_subset, test_subset = prepare_datasets(args.dataset_dir, args.dataset_ratio)

    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch, shuffle=False)

    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subset)):
        wandb.init(project=args.project_name)

        train_fold_subset = Subset(train_subset, train_idx)
        val_fold_subset = Subset(train_subset, val_idx)

        train_loader = DataLoader(train_fold_subset, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_fold_subset, batch_size=args.batch, shuffle=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            all_labels = []
            all_outputs = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

            train_accuracy = accuracy_score(all_labels, all_outputs)
            train_precision = precision_score(all_labels, all_outputs, average='binary')
            train_recall = recall_score(all_labels, all_outputs, average='binary')
            train_f1 = f1_score(all_labels, all_outputs, average='binary')

            wandb.log({
                "Model": args.model,
                "Batch Size": args.batch,
                "Initial Learning Rate": args.lr,
                "Dataset Ratio": args.dataset_ratio,
                "Fold": fold + 1,
                "Epoch": epoch + 1,
                "Loss": running_loss / len(train_loader),
                "Accuracy": train_accuracy,
                "Precision": train_precision,
                "Recall": train_recall,
                "F1 Score": train_f1,
                "Final Learning Rate": optimizer.param_groups[0]['lr']
            })

            model.eval()
            val_loss = 0.0
            all_val_labels = []
            all_val_outputs = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
                    val_loss += loss.item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_outputs.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

            val_accuracy = accuracy_score(all_val_labels, all_val_outputs)
            val_precision = precision_score(all_val_labels, all_val_outputs, average='binary')
            val_recall = recall_score(all_val_labels, all_val_outputs, average='binary')
            val_f1 = f1_score(all_val_labels, all_val_outputs, average='binary')

            wandb.log({
                "Fold": fold + 1,
                "Validation Loss": val_loss / len(val_loader),
                "Validation Accuracy": val_accuracy,
                "Validation Precision": val_precision,
                "Validation Recall": val_recall,
                "Validation F1 Score": val_f1,
            })

            scheduler.step(val_loss)

            model.eval()
            test_loss = 0.0
            all_test_labels = []
            all_test_outputs = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
                    test_loss += loss.item()
                    all_test_labels.extend(labels.cpu().numpy())
                    all_test_outputs.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

            test_accuracy = accuracy_score(all_test_labels, all_test_outputs)
            test_precision = precision_score(all_test_labels, all_test_outputs, average='binary')
            test_recall = recall_score(all_test_labels, all_test_outputs, average='binary')
            test_f1 = f1_score(all_test_labels, all_test_outputs, average='binary')
            test_cm = confusion_matrix(all_test_labels, all_test_outputs)

            wandb.log({
                "Fold": fold + 1,
                "Test Loss": test_loss / len(test_loader),
                "Test Accuracy": test_accuracy,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1 Score": test_f1,
            })

            plt.figure(figsize=(6, 6))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Low Grade', 'High Grade'], yticklabels=['Low Grade', 'High Grade'])
            plt.title(f'Test Confusion Matrix for Fold {fold + 1}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            wandb.log({"Test Confusion Matrix": wandb.Image(plt)})
            plt.tight_layout()
            plt.show()

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on the BRATS dataset.')
    parser.add_argument('--project-name', type=str, required=True, help='WandB project name')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='Model name')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset_ratio', type=float, default=0.3, help='Ratio of the dataset to use for training and testing')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of k-folds for cross-validation')

    args = parser.parse_args()
    train_and_evaluate(args)
