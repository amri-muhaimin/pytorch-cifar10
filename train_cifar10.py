import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------
# 1. Konfigurasi umum
# --------------------------
BATCH_SIZE = 64          # kalau OOM, turunkan jadi 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 2          # bisa 0 kalau error di Windows
MODEL_SAVE_PATH = "cifar10_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# 2. Definisi model CNN sederhana
# --------------------------
class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 64 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 128 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                # 128 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------
# 3. Data augmentation & DataLoader
# --------------------------
def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    # Transform untuk train: augmentasi + normalisasi
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    # Transform untuk test/validasi: hanya normalisasi
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    data_dir = "./data"

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    return train_loader, test_loader


# --------------------------
# 4. Fungsi train & evaluasi
# --------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  [batch {batch_idx+1}/{len(dataloader)}] "
                  f"loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    elapsed = time.time() - start_time
    print(f"Train Epoch {epoch}: "
          f"loss={epoch_loss:.4f}, acc={epoch_acc*100:.2f}%, "
          f"time={elapsed:.1f}s")

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, epoch, split="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    elapsed = time.time() - start_time
    print(f"{split} Epoch {epoch}: "
          f"loss={epoch_loss:.4f}, acc={epoch_acc*100:.2f}%, "
          f"time={elapsed:.1f}s")

    return epoch_loss, epoch_acc


# --------------------------
# 5. Main training loop
# --------------------------
def main():
    os.makedirs("checkpoints", exist_ok=True)

    print("Device:", DEVICE)

    train_loader, test_loader = get_dataloaders()

    model = SimpleCIFAR10CNN(num_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        val_loss, val_acc = evaluate(
            model, test_loader, criterion, epoch, split="Val"
        )

        # Simpan model terbaik berdasarkan akurasi val
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("checkpoints", MODEL_SAVE_PATH)
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} "
                  f"(acc={best_acc*100:.2f}%)")

    print(f"\nTraining selesai. Best Val Acc = {best_acc*100:.2f}%.")


if __name__ == "__main__":
    main()
