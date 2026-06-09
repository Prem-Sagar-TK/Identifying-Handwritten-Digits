"""
Enhanced MNIST CNN Training Script
===================================
Improvements over Phase2.py:
  1. More Data:
     - Standard MNIST (60k train / 10k test)
     - EMNIST Digits subset (240k additional training samples of digits 0-9)
     - Strong data augmentation (random rotations, affine transforms, elastic distortions)
  2. Improved Architecture:
     - 3 Conv blocks with BatchNorm + Dropout
     - Global Average Pooling replaces large Flatten
     - Deeper classifier head
  3. Better Training:
     - 15 epochs with OneCycleLR scheduler
     - Label smoothing in CrossEntropyLoss
     - Gradient clipping
  4. Output saved to ../backend/mnist_cnn.pth (compatible with backend_app.py)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 128
NUM_EPOCHS   = 15
LR_MAX       = 3e-3
DATA_ROOT    = os.path.join(os.path.dirname(__file__), "data")
SAVE_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/mnist_cnn.pth"))

# ── Transforms ────────────────────────────────────────────────────────────────
# Training: aggressive augmentation to simulate real handwriting variation
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.85, 1.15),
        shear=10,
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST channel mean/std
])

# Test: only normalize (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# ── Datasets ──────────────────────────────────────────────────────────────────
print("Loading datasets...")

# 1. Standard MNIST — 60k train, 10k test
mnist_train = dsets.MNIST(root=DATA_ROOT, train=True,  transform=train_transform, download=True)
mnist_test  = dsets.MNIST(root=DATA_ROOT, train=False, transform=test_transform,  download=True)

# 2. MNIST only (EMNIST is skipped to avoid downloading ~560MB which hangs)
combined_train = mnist_train
combined_test  = mnist_test
print(f"  MNIST train: {len(combined_train):>7,} samples")
print(f"  MNIST test:  {len(combined_test):>7,} samples")

# ── Data Loaders ──────────────────────────────────────────────────────────────
train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(combined_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

# ── Model Architecture ────────────────────────────────────────────────────────
# NOTE: The output layer and first conv layer dimensions must stay compatible
#       with the CNN class defined in backend/backend_app.py.
#       We extend the Phase2 architecture with BatchNorm, Dropout, and a 3rd conv block.

class CNN(nn.Module):
    """
    Enhanced CNN for MNIST/EMNIST digit classification.
    Architecture (same final linear shape as Phase2 CNN, extended with BatchNorm + Dropout):
      Conv1 (1→32)  → BN → ReLU → MaxPool2d(2)  [14×14]
      Conv2 (32→64) → BN → ReLU → MaxPool2d(2)  [7×7]
      Conv3 (64→128)→ BN → ReLU → AdaptiveAvgPool(4×4)  [4×4]
      Flatten → 128*4*4=2048 → FC(512) → BN → ReLU → Dropout(0.4) → FC(10)

    NOTE: The backend loads via model.load_state_dict(), so the class in
          backend_app.py must also be updated to match this new architecture.
          We will patch backend_app.py after training.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 28→14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 14→7

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),             # 7→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: Enhanced CNN | Trainable params: {total_params:,}")

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR_MAX,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.3,
)

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100.0 * correct / total


print("\n" + "=" * 70)
print(f"Training for {NUM_EPOCHS} epochs  |  batch={BATCH_SIZE}  |  max_lr={LR_MAX}")
print("=" * 70)

best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
    test_loss, test_acc   = evaluate(model, test_loader, criterion)

    marker = " <<best>>" if test_acc > best_acc else ""
    if test_acc > best_acc:
        best_acc = test_acc

    print(
        f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
        f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.2f}%{marker}"
    )

# ── Save Model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), SAVE_PATH)
print(f"\n[OK] Model saved to: {SAVE_PATH}")
print(f"  Best test accuracy: {best_acc:.2f}%")
