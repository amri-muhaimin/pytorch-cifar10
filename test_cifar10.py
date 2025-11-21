import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------
# 1. Konfigurasi
# --------------------------
BATCH_SIZE = 64
NUM_WORKERS = 2
MODEL_PATH = os.path.join("checkpoints", "cifar10_cnn.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# 2. Definisi model (harus sama persis dengan saat training)
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
# 3. DataLoader untuk CIFAR-10 test
# --------------------------
def get_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,   # data sudah di-download saat training
        transform=transform_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    return test_loader, test_dataset.classes


# --------------------------
# 4. Evaluasi model di test set
# --------------------------
def evaluate(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

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

    test_loss = running_loss / total
    test_acc = correct / total

    return test_loss, test_acc


# --------------------------
# 5. Tes beberapa sampel
# --------------------------
def show_some_predictions(model, dataloader, classes, num_samples=5):
    model.eval()
    shown = 0

    print("\nContoh prediksi:")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.size(0)):
                if shown >= num_samples:
                    return

                true_label = classes[targets[i].item()]
                pred_label = classes[predicted[i].item()]
                print(f"Sample {shown+1}: True = {true_label:10s} | Pred = {pred_label:10s}")
                shown += 1


# --------------------------
# 6. Main
# --------------------------
def main():
    print("Device:", DEVICE)
    print("Loading model from:", MODEL_PATH)

    # Load model
    model = SimpleCIFAR10CNN(num_classes=10).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Data
    test_loader, classes = get_test_loader()

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"\nTest Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc*100:.2f}%")

    # Show some predictions
    show_some_predictions(model, test_loader, classes, num_samples=10)


if __name__ == "__main__":
    main()
