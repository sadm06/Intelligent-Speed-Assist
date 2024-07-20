import csv
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights

MODELS_PATH = 'models/classification/'
DATASET_PATH = 'datasets/GTSRB/'
TRAIN_LABELS_PATH = 'datasets/GTSRB/Train.csv'
TEST_LABELS_PATH = 'datasets/GTSRB/Test.csv'

NUM_CLASSES = 43
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 100
PATIENCE = 10


class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                                   p=0.3),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.2)
        ])
        self.data_dict = self.data_frame.to_dict('list')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_dict['Path'][idx])
        image = Image.open(img_path).convert("RGB")
        roi_x1, roi_y1, roi_x2, roi_y2 = self.data_dict['Roi.X1'][idx], self.data_dict['Roi.Y1'][idx], \
            self.data_dict['Roi.X2'][idx], self.data_dict['Roi.Y2'][idx]
        image = image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
        image = ImageOps.autocontrast(image)
        image = self.preprocess_transform(image)
        class_id = self.data_dict['ClassId'][idx]

        if self.transform:
            image = self.transform(image)

        return image, class_id


class ResNetTrafficSignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetTrafficSignClassifier, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class CustomCNNTrafficSignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNNTrafficSignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs=NUM_EPOCHS, patience=10,
                model_path=MODELS_PATH):
    best_accuracy = 0.0
    epochs_since_improvement = 0
    epochs_trained = 0

    results_path = os.path.join(model_path, 'results.csv')
    header = ['epoch', '\t\ttrain_loss', '\t\tval_loss', '\t\tval_accuracy', '\t\tprecision', '\t\trecall',
              '\t\tf1_score']
    with open(results_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    start_time = time.time()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, class_id in train_loader:
            images, class_id = images.to(device), class_id.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, class_id)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epochs_trained += 1
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")

        val_loss, val_accuracy, precision, recall, f1_score = validate_model(model, val_loader, criterion, device)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(model_path, 'best.pth'))
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        results = [f'{epoch + 1}', f'\t\t\t{train_loss:.4f}', f'\t\t\t{val_loss:.4f}', f'\t\t\t{val_accuracy:.4f}',
                   f'\t\t\t\t{precision:.4f}', f'\t\t\t{recall:.4f}', f'\t\t{f1_score:.4f}']
        with open(results_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)

        scheduler.step(val_loss)

        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    end_time = time.time()
    elapsed_time = int((end_time - start_time) / 60)
    return elapsed_time, epochs_trained


def validate_model(model, validation_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_pred = []
    all_labels = []

    with torch.no_grad():
        for images, class_id in validation_loader:
            images, class_id = images.to(device), class_id.to(device)
            outputs = model(images)
            loss = criterion(outputs, class_id)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == class_id).sum().item()
            all_pred.extend(predicted.cpu().numpy())
            all_labels.extend(class_id.cpu().numpy())

    avg_loss = total_loss / len(validation_loader)
    accuracy = total_correct / len(validation_loader.dataset)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_pred, average='macro')
    print(
        f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
        f'F1 Score: {f1_score:.4f}')
    model.train()
    return avg_loss, accuracy, precision, recall, f1_score


def save_training_info(info, model_path):
    info_path = os.path.join(model_path, 'train_info.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f'{key}: {value}\n')


def create_training_dir(base_path):
    existing_dirs = [d for d in os.listdir(base_path) if d.startswith('train')]
    if not existing_dirs:
        new_dir_number = 1
    else:
        existing_numbers = [int(d.split('train')[-1]) for d in existing_dirs if d.split('train')[-1].isdigit()]
        new_dir_number = max(existing_numbers) + 1 if existing_numbers else 1
    new_training_path = os.path.join(base_path, f'train{new_dir_number}')
    os.makedirs(new_training_path, exist_ok=True)
    return new_training_path


def plot_class_distribution(csv_file, title):
    df = pd.read_csv(csv_file)
    class_counts = df['ClassId'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    if not os.path.exists(os.path.join(DATASET_PATH, title + '.png')):
        plt.savefig(os.path.join(DATASET_PATH, title + '.png'))
    plt.show()


transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    # plot_class_distribution(TRAIN_LABELS_PATH, 'Training Dataset Class Distribution')
    # plot_class_distribution(TEST_LABELS_PATH, 'Testing Dataset Class Distribution')
    training_dataset = TrafficSignDataset(csv_file=TRAIN_LABELS_PATH, root_dir=DATASET_PATH)
    testing_dataset = TrafficSignDataset(csv_file=TEST_LABELS_PATH, root_dir=DATASET_PATH)
    combined_dataset = ConcatDataset([training_dataset, testing_dataset])
    print(f"Training dataset size: {len(combined_dataset)}")

    print("Splitting dataset...")
    class_counts = [0] * NUM_CLASSES
    for _, class_id in combined_dataset:
        class_counts[class_id] += 1

    train_indices, val_indices = train_test_split(range(len(combined_dataset)), test_size=0.2,
                                                  stratify=[item[1] for item in combined_dataset])

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)

    model = CustomCNNTrafficSignClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    training_path = create_training_dir(MODELS_PATH)
    print("Training model…")
    elapsed_time, epochs_trained = train_model(model, criterion, optimizer, train_loader, val_loader, device,
                                               epochs=NUM_EPOCHS, patience=PATIENCE, model_path=training_path)
    print("Training complete.")
    torch.save(model.state_dict(), os.path.join(training_path, 'last.pth'))

    training_info = {
        'task': 'classification',
        'model': model.__class__.__name__,
        'epochs': epochs_trained,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'patience': PATIENCE,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'minutes_trained': elapsed_time
    }
    save_training_info(training_info, training_path)


if __name__ == '__main__':
    main()
